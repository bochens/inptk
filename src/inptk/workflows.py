"""Native-observation concentration workflows with named output curves."""

from __future__ import annotations

import json
from typing import Literal, TypeVar

import numpy as np
import pandas as pd

from .alignment import align_observations
from .experiment import AnalysisResult, Experiment, SampleMetadata
from .methods import (
    curve_specifications,
    resolve_curves,
    validate_combination_method,
    validate_temperature_ranges,
)
from .processing import (
    differential_spectrum,
    frozen_fraction,
    prepare_fraction_analysis,
)
from .resampling import resample_spectrum
from .tables import UNITS, CumulativeSpectrumTable, CurveSpectrumTable, FrozenFractionTable
from .water_blank import estimate_point, sample_rows

SpectrumT = TypeVar("SpectrumT", bound=CumulativeSpectrumTable)


def convert_concentration(spectrum: SpectrumT, samples: dict[str, SampleMetadata], *, basis: str):
    """Convert suspension concentration once, preserving the table's identities."""
    if basis not in UNITS:
        raise ValueError(f"Unknown concentration basis {basis!r}")
    data = spectrum.to_dataframe()
    if not data.basis.eq("suspension").all():
        raise ValueError(
            "Conversion requires suspension concentrations; this result was already converted"
        )
    for sample_id, rows in data.groupby("sample_id", sort=False):
        factor = samples[str(sample_id)].factor_for(basis)
        for name in ("concentration", "lower_error", "upper_error"):
            if name in data:
                data.loc[rows.index, name] *= factor
    data["basis"], data["unit"] = basis, UNITS[basis]
    return type(spectrum)(
        data, history=spectrum.history + [{"operation": "convert_concentration", "basis": basis}]
    )


def _curve_columns(data: pd.DataFrame) -> list[str]:
    if "curve_id" in data:
        return ["sample_id", "curve_id"]
    columns = ["run_id", "sample_id", "cycle_id"]
    if "measurement_id" in data:
        columns.append("measurement_id")
    return columns


def subtract_blanks(
    spectrum: SpectrumT, blank_by_target: dict[str, CumulativeSpectrumTable]
) -> SpectrumT:
    """Subtract explicit, temperature-matched sample/filter blank spectra.

    Map a named curve ID, or an input spectrum's sample ID, to one
    already aligned blank curve. This is separate from raw assay-blank fitting.
    Opposite error widths use an approximate independent-error propagation.
    """
    data = spectrum.to_dataframe()
    target_column = "curve_id" if isinstance(spectrum, CurveSpectrumTable) else "sample_id"
    if not data.basis.eq("suspension").all():
        raise ValueError("Blank correction requires suspension concentrations")
    extra = set(blank_by_target) - set(data[target_column])
    if extra:
        raise ValueError(f"Blank mapping names unknown targets: {sorted(extra)}")
    prior_methods = []
    method = "approximate_independent_filter_blank_error_propagation"
    for target, blank in blank_by_target.items():
        background = blank.to_dataframe()
        if not background.basis.eq("suspension").all():
            raise ValueError("Blank correction requires suspension concentrations")
        if len(background[_curve_columns(background)].drop_duplicates()) != 1:
            raise ValueError("Each mapped filter blank must contain one explicitly selected curve")
        if background.temperature_C.duplicated().any():
            raise ValueError(
                "Filter blank must have one value per temperature; "
                "sample its curve explicitly before subtraction"
            )
        rows = data[(data[target_column] == target) & np.isfinite(data.concentration)]
        if rows.empty:
            continue
        aligned = rows[["temperature_C"]].merge(
            background, on="temperature_C", how="left", validate="many_to_one", indicator=True
        )
        if not aligned["_merge"].eq("both").all():
            raise ValueError(
                f"Filter blank for {target!r} does not cover all requested temperatures"
            )
        data.loc[rows.index, "concentration"] = (
            rows.concentration.to_numpy() - aligned.concentration.to_numpy()
        )
        for column, other in (("lower_error", "upper_error"), ("upper_error", "lower_error")):
            if column in rows and other in aligned:
                data.loc[rows.index, column] = np.hypot(
                    rows[column].to_numpy(), aligned[other].to_numpy()
                )
            elif column in rows:
                raise ValueError("Blank and sample must both supply uncertainty to propagate it")
        columns = _curve_columns(rows) + ["temperature_C"]
        if "point_id" in rows:
            columns.append("point_id")
        provenance = rows[columns].copy()
        provenance["uncertainty_method"] = rows.get("uncertainty_method")
        prior_methods.extend(provenance.to_dict("records"))
        data.loc[rows.index, "uncertainty_method"] = method
        data.loc[rows.index, "correction_state"] = "blank_corrected"
        data.loc[rows.index, "is_extrapolated"] = (
            rows.is_extrapolated.to_numpy() | aligned.is_extrapolated.to_numpy()
        )
    return type(spectrum)(
        data,
        history=spectrum.history
        + [
            {
                "operation": "subtract_filter_blank",
                "targets": list(blank_by_target),
                "uncertainty_method": method,
                "source_uncertainty_methods": prior_methods,
                "uncertainty_assumption": (
                    "Independent errors; approximate propagation with opposite "
                    "blank error direction for subtraction"
                ),
                "blanks": {
                    key: value.to_dataframe().to_dict("records")
                    for key, value in blank_by_target.items()
                },
            }
        ],
    )


def _state_key(point) -> tuple:
    columns = ["measurement_id", "run_id", "cycle_id", "n_frozen", "n_total"]
    return tuple(
        tuple(sorted(frame[columns].itertuples(index=False, name=None)))
        for frame in (point.samples, point.blanks)
    )


def _sources(point) -> list[dict]:
    records = []
    for role, frame in (("sample", point.samples), ("blank", point.blanks)):
        for row in frame.to_dict("records"):
            item: dict[str, object] = {
                key: str(row[key])
                for key in ("measurement_id", "run_id", "cycle_id", "observation_id")
            }
            item.update(
                role=role,
                observed_temperature_C=float(row["temperature_C"]),
                alignment="exact" if row["temperature_C"] == point.temperature_C else "latest",
            )
            if "time_s" in row and pd.notna(row["time_s"]):
                item["time_s"] = float(row["time_s"])
            records.append(item)
    return records


def estimate_concentration(
    fractions: FrozenFractionTable,
    *,
    experiment: Experiment,
    method: Literal["mle", "average"] = "mle",
    temperature_ranges_C=None,
    curves=None,
    z: float = 1.96,
    water_blank_correction: bool = True,
) -> CurveSpectrumTable:
    """Estimate named curves from original states with separate run backgrounds.

    With no curves supplied, each sample/run/cycle remains separate. Explicit
    curves can span runs but select only one cycle from each run. Observations
    align only where needed, using latest warmer states at observed targets.
    MLE fits complete freezing histories with monotone sample and blank curves;
    Average estimates each target separately. Original count rows stay unchanged.
    """
    method = validate_combination_method(method)
    if not np.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    fractions, experiment = prepare_fraction_analysis(
        fractions, experiment, water_blank_correction=water_blank_correction
    )
    frame = fractions.to_dataframe()
    source = sample_rows(frame, experiment)
    if source.empty:
        raise ValueError("No sample observations are available for concentration calculation")
    blank_ids = {key for ids in experiment.water_blank_map.values() for key in ids}
    ranges = validate_temperature_ranges(
        temperature_ranges_C, measurement_ids=set(experiment.measurements) - blank_ids
    )
    groups = resolve_curves(curves, experiment, frame)
    records, notices, group_alignment, cache = [], [], {}, {}
    joint_fits = {}
    for curve_id, group in groups.items():
        members = group["members"]
        ids = sorted(member["measurement_id"] for member in members)
        supports = {}
        for member in members:
            selected = source[
                (source.measurement_id == member["measurement_id"])
                & (source.cycle_id == member["cycle_id"])
            ]
            supports[member["measurement_id"]] = (
                float(selected.temperature_C.min()),
                float(selected.temperature_C.max()),
            )
        points = align_observations(
            frame, members, water_blank_map=experiment.water_blank_map, temperature_ranges_C=ranges
        )
        if method == "mle":
            from dataclasses import replace

            from .curve_fit import fit_curve

            estimates, joint_fits[curve_id] = fit_curve(points, experiment, z=z)
            # A fitted temperature curve has one state per native temperature.
            # Original image order and every count row stay in the experiment.
            by_temperature = {point.temperature_C: point for point in points}
            points = [
                replace(by_temperature[temperature], point_order=index, point_id=f"point:{index}")
                for index, temperature in enumerate(sorted(by_temperature, reverse=True))
            ]
        empty_count = 0
        group_alignment[curve_id] = sorted({point.alignment for point in points})
        for point in points:
            contributors = sorted(point.samples.measurement_id.astype(str).tolist())
            available = sorted(
                key for key, (cold, warm) in supports.items() if cold <= point.temperature_C <= warm
            )
            record = {
                "sample_id": group["sample_id"],
                "curve_id": curve_id,
                "point_id": point.point_id,
                "point_order": point.point_order,
                "temperature_C": point.temperature_C,
                "alignment": point.alignment,
                "concentration": np.nan,
                "lower_error": np.nan,
                "upper_error": np.nan,
                "unit": "INP_per_mL_suspension",
                "basis": "suspension",
                "qc_flag": 1,
                "source_measurement_ids": json.dumps(ids),
                "available_measurement_ids": json.dumps(available),
                "contributing_measurement_ids": json.dumps(contributors),
                "contributor_count": len(contributors),
                "source_measurement_id": contributors[0] if len(contributors) == 1 else "",
                "selection_status": "no_eligible_measurements"
                if not contributors
                else "single"
                if len(contributors) == 1
                else "combined",
                "dilution_fold": experiment.measurements[contributors[0]].dilution
                if len(contributors) == 1
                else np.nan,
                "water_blank_ids": json.dumps(
                    sorted(point.blanks.measurement_id.astype(str).tolist())
                ),
                "source_observations": json.dumps(_sources(point)),
                "uncertainty_method": "joint_curve_profile_likelihood"
                if method == "mle"
                else "bonferroni_marginal_profile_bounds"
                if method == "average" and len(contributors) > 1
                else "joint_sample_water_blank_profile_likelihood"
                if experiment.water_blank_map
                else "binomial_Poisson_profile_likelihood",
                "correction_state": "water_blank_corrected"
                if experiment.water_blank_map
                else "uncorrected",
            }
            if contributors:
                if method == "mle":
                    estimate, lower, upper = estimates[point.temperature_C]
                    finite = bool(np.isfinite([estimate, lower, upper]).all())
                    fit = (estimate, lower, upper, finite)
                else:
                    key = _state_key(point)
                    if key not in cache:
                        cache[key] = estimate_point(
                            point.samples,
                            point.blanks,
                            experiment,
                            confidence_drop=z**2 / 2,
                            method=method,
                        )
                    fit = cache[key]
                record.update(
                    concentration=fit[0],
                    lower_error=fit[1],
                    upper_error=fit[2],
                    qc_flag=0 if fit[3] else 1,
                )
            else:
                empty_count += 1
            record["at_zero_boundary"] = record["concentration"] == 0
            records.append(record)
        if empty_count:
            notices.append(f"Curve {curve_id!r}: {empty_count} points have no eligible input")
    settings = {
        "operation": "estimate_concentration",
        "estimation_method": method,
        "curves": curve_specifications(groups),
        "curve_sources": groups,
        "temperature_ranges_C": ranges,
        "range_boundaries": "inclusive; source and target",
        "alignment": group_alignment,
        "alignment_rule": "latest observation at or warmer than target, only where needed",
        "z": float(z),
        "confidence_drop": float(z**2 / 2),
        "water_blank_correction": water_blank_correction,
        "water_blank_correction_applied": bool(experiment.water_blank_map),
        "water_blank_model": "volume_scaled",
        "background_groups": "separate by run and selected cycle",
        "water_blank_map": dict(experiment.water_blank_map),
        "joint_curve_fits": joint_fits,
        "uncertainty_assumption": (
            "Independent physical droplet sets, repeated observations never pooled; "
            "profile bounds and Bonferroni-adjusted average bounds have approximate coverage"
        ),
        "warnings": notices,
    }
    return CurveSpectrumTable(
        pd.DataFrame.from_records(records), history=fractions.history + [settings]
    )


def _validate_decrease_policy(decrease_policy: str) -> None:
    if decrease_policy not in ("stop_at_decrease", "skip_decreases"):
        raise ValueError("decrease_policy must be 'stop_at_decrease' or 'skip_decreases'")


def _final_candidates(
    spectrum: SpectrumT,
    *,
    decrease_policy: Literal["stop_at_decrease", "skip_decreases"],
) -> SpectrumT:
    """Mark native point selection in observation order without changing values."""
    _validate_decrease_policy(decrease_policy)
    if not isinstance(spectrum, CumulativeSpectrumTable):
        raise TypeError("spectrum must be a cumulative spectrum table")
    data = spectrum.to_dataframe()
    data["used_in_final"] = False
    data["final_selection_status"] = "nonfinite"
    data["segment_id"] = ""
    keys = _curve_columns(data)
    reports, notices = [], []
    for identity, original_rows in data.groupby(keys, sort=False):
        rows = (
            original_rows.sort_values("point_order", kind="stable")
            if "point_order" in original_rows
            else original_rows
        )
        previous = None
        previous_point = None
        stopped = False
        segment = -1
        contiguous = False
        excluded = []
        retained = 0
        for index, row in rows.iterrows():
            concentration = float(row.concentration) if pd.notna(row.concentration) else np.nan
            if not np.isfinite(concentration):
                status = "nonfinite"
            elif stopped and decrease_policy == "stop_at_decrease":
                status = "after_decrease"
            elif (
                previous is not None
                and concentration < previous
                and not np.isclose(concentration, previous, rtol=1e-9, atol=0)
            ):
                status = "decrease"
                stopped = True
            else:
                status = "kept"
            data.at[index, "final_selection_status"] = status
            if status == "kept":
                if not contiguous:
                    segment += 1
                data.at[index, "segment_id"] = str(segment)
                data.at[index, "used_in_final"] = True
                previous, previous_point = concentration, row.get("point_id")
                retained += 1
                contiguous = True
            else:
                contiguous = False
                excluded.append(
                    {
                        "point_id": row.get("point_id"),
                        "temperature_C": float(row.temperature_C),
                        "concentration": concentration,
                        "reason": status,
                        "reference_point_id": previous_point,
                        "reference_concentration": previous,
                    }
                )
        report: dict[str, object] = dict(zip(keys, identity, strict=True))
        report.update(retained_count=retained, excluded_count=len(excluded), excluded=excluded)
        reports.append(report)
        if excluded:
            notices.append(
                f"{dict(zip(keys, identity, strict=True))}: final selection excluded "
                f"{len(excluded)} of {len(rows)} points using {decrease_policy}"
            )
        if retained == 0:
            notices.append(
                f"{dict(zip(keys, identity, strict=True))}: no finite concentration points remain"
            )
    return type(spectrum)(
        data,
        history=spectrum.history
        + [
            {
                "operation": "finalize_spectrum",
                "decrease_policy": decrease_policy,
                "comparison_order": "point_order, or supplied observation order",
                "numerical_relative_tolerance": 1e-9,
                "numerical_absolute_tolerance": 0.0,
                "groups": reports,
                "warnings": notices,
            }
        ],
    )


def finalize_spectrum(
    spectrum: SpectrumT,
    *,
    decrease_policy: Literal["stop_at_decrease", "skip_decreases"] = "stop_at_decrease",
) -> SpectrumT:
    """Keep nondecreasing concentration in observation order, allowing fitting roundoff.

    No temperature sorting, rounding, value adjustment or uncertainty reduction
    is performed. Segment IDs retain gaps for safe later resampling.
    """
    return _final_candidates(spectrum, decrease_policy=decrease_policy).select(used_in_final=True)


def analyze_concentration(
    experiment: Experiment,
    *,
    method: Literal["mle", "average"] = "mle",
    temperature_ranges_C=None,
    curves=None,
    output_basis: str = "suspension",
    z: float = 1.96,
    differential: bool = False,
    blank_by_curve: dict[str, CumulativeSpectrumTable] | None = None,
    water_blank_correction: bool = True,
    decrease_policy: Literal["stop_at_decrease", "skip_decreases"] = "stop_at_decrease",
    output_step_C: float | None = None,
    output_method: Literal["sample", "interpolate"] = "sample",
) -> AnalysisResult:
    """Use original observations, align only when needed, and optionally grid the result."""
    from .water_blank import analysis_experiment

    _validate_decrease_policy(decrease_policy)
    method = validate_combination_method(method)
    if output_basis not in UNITS:
        raise ValueError(f"Unknown output_basis {output_basis!r}")
    if output_method not in ("sample", "interpolate"):
        raise ValueError("output_method must be 'sample' or 'interpolate'")
    if output_step_C is not None and (not np.isfinite(output_step_C) or output_step_C <= 0):
        raise ValueError("output_step_C must be finite and positive")
    source = analysis_experiment(experiment, water_blank_correction=water_blank_correction)
    fractions = frozen_fraction(source)
    observed_fractions = fractions if source is experiment else frozen_fraction(experiment)
    if differential:
        requested_curves = resolve_curves(curves, source, fractions.to_dataframe())
        if any(len(group["members"]) != 1 for group in requested_curves.values()):
            raise ValueError(
                "Differential output currently requires individual curves. "
                "Request one input per curve, or use differential_spectrum separately."
            )
        if output_basis != "suspension" or blank_by_curve:
            raise ValueError(
                "Differential output currently requires suspension basis without "
                "an additional sample/filter blank spectrum"
            )
    combined = estimate_concentration(
        fractions,
        experiment=experiment,
        method=method,
        temperature_ranges_C=temperature_ranges_C,
        curves=curves,
        z=z,
        water_blank_correction=water_blank_correction,
    )
    analysis_fractions = fractions
    if differential and curves is not None:
        # Differential fits need only selected input/cycle rows and their blanks.
        # The archived observations remain complete.
        resolved = combined.history[-1]["curve_sources"]
        keys = ["measurement_id", "run_id", "cycle_id"]
        members = {
            tuple(str(member[key]) for key in keys)
            for group in resolved.values()
            for member in group["members"]
        }
        blanks = (
            {
                (blank_id, run, cycle)
                for measurement, run, cycle in members
                for blank_id in experiment.water_blank_map.get(measurement, [])
            }
            if water_blank_correction
            else set()
        )
        requested = members | blanks
        frame = fractions.to_dataframe()
        selected = [
            tuple(row) in requested for row in frame[keys].itertuples(index=False, name=None)
        ]
        analysis_fractions = FrozenFractionTable(
            frame.loc[selected],
            history=fractions.history
            + [
                {
                    "operation": "select_curve_inputs",
                    "curve_ids": list(resolved),
                    "members": [dict(zip(keys, member, strict=True)) for member in sorted(members)],
                    "water_blank_context": [
                        dict(zip(keys, blank, strict=True)) for blank in sorted(blanks)
                    ],
                }
            ],
        )
    candidates = subtract_blanks(combined, blank_by_curve) if blank_by_curve else combined
    if output_basis != "suspension":
        candidates = convert_concentration(candidates, experiment.samples, basis=output_basis)
    final_candidates = _final_candidates(candidates, decrease_policy=decrease_policy)
    final = final_candidates.select(used_in_final=True)
    sampled = (
        resample_spectrum(final, step_C=output_step_C, method=output_method)
        if output_step_C is not None
        else None
    )
    differential_result = (
        differential_spectrum(
            analysis_fractions,
            experiment=experiment,
            temperature_ranges_C=temperature_ranges_C,
            method=method,
            z=z,
            water_blank_correction=water_blank_correction,
        )
        if differential
        else None
    )
    settings = {
        "estimation_method": method,
        "temperature_ranges_C": combined.history[-1]["temperature_ranges_C"],
        "curves": combined.history[-1]["curves"],
        "observation_processing": "native; latest warmer alignment only where required",
        "output_basis": output_basis,
        "output_step_C": output_step_C,
        "output_method": output_method if sampled is not None else None,
        "z": z,
        "differential": differential,
        "water_blank_correction": water_blank_correction,
        "water_blank_correction_applied": water_blank_correction
        and bool(experiment.water_blank_map),
        "water_blank_model": "volume_scaled",
        "decrease_policy": decrease_policy,
    }
    history = final.history + (sampled.history[-1:] if sampled is not None else [])
    warnings = list(
        dict.fromkeys(
            final.warnings
            + (sampled.warnings if sampled is not None else [])
            + (differential_result.warnings if differential_result is not None else [])
        )
    )
    from .results import assemble_curves

    return AnalysisResult(
        experiment=experiment,
        frozen_fraction=observed_fractions,
        curves=assemble_curves(
            final,
            final_candidates,
            combined.history[-1]["curve_sources"],
            experiment=experiment,
            resampled=sampled,
            differential=differential_result,
        ),
        settings=settings,
        history=history,
        warnings=warnings,
    )
