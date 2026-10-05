"""Native-observation concentration workflows with named output curves."""

from __future__ import annotations

from typing import Literal, TypeVar, cast

import numpy as np
import pandas as pd

from .estimation import estimate_concentration
from .experiment import AnalysisResult, Experiment, SampleMetadata
from .methods import (
    resolve_curves,
)
from .processing import (
    _individual_spectra,
    differentiate_spectrum,
    frozen_fraction,
)
from .settings import DEFAULTS, AnalysisSettings, validate_decrease_policy
from .tables import UNITS, CumulativeSpectrumTable, CurveSpectrumTable

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


def _final_candidates(
    spectrum: SpectrumT,
    *,
    decrease_policy: Literal["stop_at_decrease", "skip_decreases"],
) -> SpectrumT:
    """Trim reporting intervals; apply decrease selection only to combined curves."""
    validate_decrease_policy(decrease_policy)
    if not isinstance(spectrum, CumulativeSpectrumTable):
        raise TypeError("spectrum must be a cumulative spectrum table")
    data = spectrum.to_dataframe()
    used = np.zeros(len(data), dtype=bool)
    statuses = np.full(len(data), "nonfinite", dtype=object)
    segments = np.full(len(data), "", dtype=object)
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
            index = cast(int, index)  # ScientificTable stores rows with a RangeIndex.
            concentration = float(row.concentration) if pd.notna(row.concentration) else np.nan
            reporting = row.get("reporting_status", "within_freezing_interval")
            if reporting != "within_freezing_interval":
                status = reporting
            elif row.get("curve_kind") == "individual":
                status = "kept"
            elif not np.isfinite(concentration):
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
            statuses[index] = status
            if status == "kept":
                if not contiguous:
                    segment += 1
                segments[index] = str(segment)
                used[index] = True
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
        scientific_exclusions = [
            item for item in excluded
            if item["reason"] in {"nonfinite", "decrease", "after_decrease"}
        ]
        if scientific_exclusions:
            notices.append(
                f"{dict(zip(keys, identity, strict=True))}: final selection excluded "
                f"{len(scientific_exclusions)} of {len(rows)} points using {decrease_policy}"
            )
        if retained == 0:
            notices.append(
                f"{dict(zip(keys, identity, strict=True))}: no finite concentration points remain"
            )
    data["used_in_final"] = used
    data["final_selection_status"] = statuses
    data["segment_id"] = segments
    return type(spectrum)(
        data,
        history=spectrum.history
        + [
            {
                "operation": "finalize_spectrum",
                "reporting_rule": "observed_sample_freezing_interval"
                if "reporting_status" in data else "no_count_limits_provided",
                "decrease_policy": decrease_policy,
                "decrease_policy_scope": "combined_curves_only",
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
    decrease_policy: Literal["stop_at_decrease", "skip_decreases"] = DEFAULTS.decrease_policy,
) -> SpectrumT:
    """Keep individual points in their freezing interval; select combined points.

    Estimates outside the observed sample freezing interval are omitted when
    the estimation step has recorded those limits. Input observations still
    constrain the fit. No temperature sorting, rounding, value adjustment or
    uncertainty reduction is performed. Individual curves keep every calculated
    point within their original interval, including flagged nonfinite values.
    Decrease selection applies only to combined curves. Segment IDs retain gaps
    for plotting.
    """
    return _final_candidates(spectrum, decrease_policy=decrease_policy).select(used_in_final=True)


def analyze_concentration(
    experiment: Experiment,
    *,
    method: Literal["mle", "average"] = DEFAULTS.method,
    fit_step_C: float | None = DEFAULTS.fit_step_C,
    temperature_ranges_C=None,
    temperature_step_C: float | None = DEFAULTS.temperature_step_C,
    temperature_start_C: float | None = DEFAULTS.temperature_start_C,
    temperature_end_C: float | None = DEFAULTS.temperature_end_C,
    temperature_method: Literal["latest", "max", "window"] = DEFAULTS.temperature_method,
    temperature_window_C: float | None = DEFAULTS.temperature_window_C,
    curves=None,
    output_basis: str = DEFAULTS.output_basis,
    z: float = DEFAULTS.z,
    differential: bool = DEFAULTS.differential,
    blank_by_curve: dict[str, CumulativeSpectrumTable] | None = None,
    water_blank_correction: bool = DEFAULTS.water_blank_correction,
    water_blank_after_first_freeze: bool = DEFAULTS.water_blank_after_first_freeze,
    water_blank_temperature_range_C=None,
    decrease_policy: Literal["stop_at_decrease", "skip_decreases"] = DEFAULTS.decrease_policy,
) -> AnalysisResult:
    """Run the public calculation steps using native or initially gridded counts."""
    from .water_blank import analysis_experiment

    AnalysisSettings(
        method=method,
        fit_step_C=fit_step_C,
        temperature_step_C=temperature_step_C,
        temperature_start_C=temperature_start_C,
        temperature_end_C=temperature_end_C,
        temperature_method=temperature_method,
        temperature_window_C=temperature_window_C,
        z=z,
        water_blank_correction=water_blank_correction,
        water_blank_after_first_freeze=water_blank_after_first_freeze,
        water_blank_temperature_range_C=water_blank_temperature_range_C,
        output_basis=output_basis,
        differential=differential,
        decrease_policy=decrease_policy,
    )
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
        fit_step_C=fit_step_C,
        temperature_ranges_C=temperature_ranges_C,
        temperature_step_C=temperature_step_C,
        temperature_start_C=temperature_start_C,
        temperature_end_C=temperature_end_C,
        temperature_method=temperature_method,
        temperature_window_C=temperature_window_C,
        curves=curves,
        z=z,
        water_blank_correction=water_blank_correction,
        water_blank_after_first_freeze=water_blank_after_first_freeze,
        water_blank_temperature_range_C=water_blank_temperature_range_C,
    )
    candidates = subtract_blanks(combined, blank_by_curve) if blank_by_curve else combined
    if output_basis != "suspension":
        candidates = convert_concentration(candidates, experiment.samples, basis=output_basis)
    final_candidates = _final_candidates(candidates, decrease_policy=decrease_policy)
    final = final_candidates.select(used_in_final=True)
    differential_result = None
    if differential:
        # Reuse the estimates already calculated above. Differencing cannot
        # change the fit, select another count state, or repeat optimization.
        differential_result = {
            name: differentiate_spectrum(
                _individual_spectra(
                    combined.select(curve_id=name),
                    fractions,
                    source,
                )
            )
            for name in combined.history[-1]["curve_sources"]
        }
    settings = {
        "estimation_method": method,
        "fit_step_C": fit_step_C,
        "temperature_ranges_C": combined.history[-1]["temperature_ranges_C"],
        "resolved_temperature_ranges_C": combined.history[-1]["resolved_temperature_ranges_C"],
        "temperature_step_C": temperature_step_C,
        "temperature_start_C": temperature_start_C,
        "temperature_end_C": temperature_end_C,
        "temperature_method": temperature_method,
        "temperature_window_C": temperature_window_C,
        "curves": combined.history[-1]["curves"],
        "observation_processing": (
            f"selected {temperature_step_C:g} C grid; {temperature_method}; before blank correction"
            if temperature_step_C is not None
            else "native; latest warmer alignment only where required"
        ),
        "output_basis": output_basis,
        "z": z,
        "differential": differential,
        "water_blank_correction": water_blank_correction,
        "water_blank_correction_applied": water_blank_correction
        and bool(experiment.water_blank_map),
        "water_blank_model": "volume_scaled",
        "water_blank_after_first_freeze": water_blank_after_first_freeze,
        "water_blank_temperature_range_C": combined.history[-1]["water_blank_temperature_range_C"],
        "water_blank_controls": combined.history[-1]["water_blank_controls"],
        "decrease_policy": decrease_policy,
        "reporting_rule": "observed_sample_freezing_interval",
        "reporting_intervals_C": combined.history[-1]["reporting_intervals_C"],
    }
    history = final.history
    warnings = list(
        dict.fromkeys(
            final.warnings
            + [
                warning
                for table in (differential_result or {}).values()
                for warning in table.warnings
            ]
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
            differential=differential_result,
        ),
        settings=settings,
        history=history,
        warnings=warnings,
    )
