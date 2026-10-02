"""One concentration workflow with explicit measurement temperature ranges."""

from __future__ import annotations

import json
from typing import Literal, cast

import numpy as np
import pandas as pd

from ._engine.water_blank_math import average_concentration, fit_concentration
from .experiment import AnalysisResult, Experiment, SampleMetadata
from .methods import validate_combination_method, validate_temperature_ranges
from .processing import (
    cumulative_spectrum,
    differential_spectrum,
    frozen_fraction,
    prepare_fraction_analysis,
)
from .tables import UNITS, CumulativeSpectrumTable, FrozenFractionTable


def convert_concentration(
    spectrum: CumulativeSpectrumTable, samples: dict[str, SampleMetadata], *, basis: str
):
    """Convert suspension concentration once; return a new cumulative spectrum."""
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
    data["basis"] = basis
    data["unit"] = UNITS[basis]
    return CumulativeSpectrumTable(
        data, history=spectrum.history + [{"operation": "convert_concentration", "basis": basis}]
    )


def subtract_blanks(
    spectrum: CumulativeSpectrumTable, blank_by_sample: dict[str, CumulativeSpectrumTable]
) -> CumulativeSpectrumTable:
    """Match blank curves by target sample, run, cycle and temperature.

    Error widths use an independent-error root-sum-of-squares approximation.
    Subtraction pairs the sample's lower error with the blank's upper error,
    and vice versa. This is not an exact confidence interval for the difference.
    """
    data = spectrum.to_dataframe()
    source_uncertainty_methods = []
    propagated_method = "approximate_independent_filter_blank_error_propagation"
    if not data.basis.eq("suspension").all():
        raise ValueError("Blank correction requires suspension concentrations")
    extra = set(blank_by_sample) - set(data.sample_id)
    if extra:
        raise ValueError(f"Blank mapping names unknown target samples: {sorted(extra)}")
    for sample_id, blank in blank_by_sample.items():
        background = blank.to_dataframe()
        if not background.basis.eq("suspension").all():
            raise ValueError("Blank correction requires suspension concentrations")
        if background.sample_id.nunique() != 1:
            raise ValueError("Each mapped blank must contain exactly one blank sample")
        keys = ["run_id", "cycle_id", "temperature_C"]
        if background.duplicated(keys).any():
            raise ValueError("Blank curves contain duplicate run/cycle/temperature rows")
        rows = data[(data.sample_id == sample_id) & np.isfinite(data.concentration)]
        if rows.empty:
            continue
        aligned = rows[keys].merge(
            background, on=keys, how="left", validate="one_to_one", indicator=True
        )
        if not aligned["_merge"].eq("both").all():
            raise ValueError(
                f"Blank for {sample_id!r} does not cover all run/cycle/temperature rows"
            )
        data.loc[rows.index, "concentration"] = (
            rows.concentration.to_numpy() - aligned.concentration.to_numpy()
        )
        for column, blank_column in (
            ("lower_error", "upper_error"),
            ("upper_error", "lower_error"),
        ):
            if column in rows and blank_column in aligned:
                data.loc[rows.index, column] = np.sqrt(
                    rows[column].to_numpy() ** 2 + aligned[blank_column].to_numpy() ** 2
                )
            elif column in rows:
                raise ValueError("Blank and sample must both supply uncertainty to propagate it")
        data.loc[rows.index, "is_extrapolated"] = (
            rows.is_extrapolated.to_numpy() | aligned.is_extrapolated.to_numpy()
        )
        provenance_columns = ["sample_id", "run_id", "cycle_id", "temperature_C"]
        if "measurement_id" in rows:
            provenance_columns.append("measurement_id")
        prior_methods = rows[provenance_columns].copy()
        prior_methods["uncertainty_method"] = rows.get("uncertainty_method")
        source_uncertainty_methods.extend(prior_methods.to_dict("records"))
        data.loc[rows.index, "uncertainty_method"] = propagated_method
        data.loc[rows.index, "correction_state"] = "blank_corrected"
    return CumulativeSpectrumTable(
        data,
        history=spectrum.history
        + [
            {
                "operation": "subtract_filter_blank",
                "targets": list(blank_by_sample),
                "uncertainty_method": propagated_method,
                "source_uncertainty_methods": source_uncertainty_methods,
                "uncertainty_assumption": (
                    "independent errors; approximate root-sum-of-squares with opposite "
                    "blank error direction for subtraction"
                ),
                "blanks": {
                    key: value.to_dataframe().to_dict("records")
                    for key, value in blank_by_sample.items()
                },
            }
        ],
    )


def combine_dilutions(
    fractions: FrozenFractionTable,
    *,
    experiment: Experiment,
    method: Literal["mle", "average"] = "mle",
    temperature_ranges_C=None,
    z: float = 1.96,
    water_blank_correction: bool = True,
) -> CumulativeSpectrumTable:
    """Combine eligible measurements by joint MLE or arithmetic mean at each temperature.

    Ranges use exact measurement names and inclusive min_C/max_C boundaries.
    An omitted boundary or measurement places no additional restriction on its
    observed temperature support. One measurement uses the same count likelihood
    as several measurements. Average uses separate measurement estimates and
    conservative bounds that allow dependent errors from a shared blank.
    Sample, run and cycle groups are always calculated separately.
    """
    from .water_blank import fit_raw_rows, sample_rows

    method = validate_combination_method(method)
    estimate = average_concentration if method == "average" else fit_concentration
    if not np.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    fractions, experiment = prepare_fraction_analysis(
        fractions,
        experiment,
        water_blank_correction=water_blank_correction,
    )
    all_rows = fractions.to_dataframe()
    source = sample_rows(all_rows, experiment)
    blank_ids = {name for names in experiment.water_blank_map.values() for name in names}
    ranges = validate_temperature_ranges(
        temperature_ranges_C, measurement_ids=set(experiment.measurements) - blank_ids
    )
    if source.empty:
        raise ValueError("No sample observations are available for concentration calculation")
    records, notices = [], []
    for identity, group in source.groupby(["run_id", "sample_id", "cycle_id"], sort=False):
        run_id, sample_id, cycle_id = map(str, identity)
        group_ids = sorted(group.measurement_id.unique())
        expected = {
            key
            for key, value in experiment.measurements.items()
            if key not in blank_ids and (value.sample_id, value.run_id) == (sample_id, run_id)
        }
        missing = expected - set(group_ids)
        if missing:
            notices.append(
                f"{sample_id}, run {run_id}, cycle {cycle_id}: "
                f"absent measurements {sorted(missing)}"
            )
        empty_count = 0
        for temperature, rows in group.groupby("temperature_C", sort=False):
            temperature = float(cast(float, temperature))
            eligible = []
            for row in rows.itertuples():
                limits = ranges.get(str(row.measurement_id), {})
                minimum, maximum = limits.get("min_C"), limits.get("max_C")
                eligible.append(
                    (minimum is None or temperature >= minimum)
                    and (maximum is None or temperature <= maximum)
                )
            selected = rows.loc[eligible]
            ids = sorted(selected.measurement_id.astype(str).tolist())
            record = {
                "run_id": run_id,
                "sample_id": sample_id,
                "cycle_id": cycle_id,
                "temperature_C": temperature,
                "concentration": np.nan,
                "lower_error": np.nan,
                "upper_error": np.nan,
                "unit": "INP_per_mL_suspension",
                "basis": "suspension",
                "qc_flag": 1,
                "source_measurement_ids": json.dumps(group_ids),
                "available_measurement_ids": json.dumps(sorted(rows.measurement_id.astype(str))),
                "contributing_measurement_ids": json.dumps(ids),
                "contributor_count": len(ids),
                "source_measurement_id": ids[0] if len(ids) == 1 else "",
                "selection_status": "no_eligible_measurements"
                if not ids
                else "single"
                if len(ids) == 1
                else "combined",
                "dilution_fold": experiment.measurements[ids[0]].dilution
                if len(ids) == 1
                else np.nan,
                "uncertainty_method": "bonferroni_marginal_profile_bounds"
                if method == "average"
                else "joint_sample_water_blank_profile_likelihood"
                if experiment.water_blank_map
                else "binomial_Poisson_profile_likelihood",
                "correction_state": "water_blank_corrected"
                if experiment.water_blank_map
                else "uncorrected",
            }
            if not ids:
                empty_count += 1
            else:
                if experiment.water_blank_map:
                    fit = fit_raw_rows(
                        selected, all_rows, experiment, confidence_drop=z**2 / 2, method=method
                    )
                    record["water_blank_ids"] = json.dumps(
                        sorted(experiment.water_blank_map[ids[0]])
                    )
                else:
                    metadata = [
                        experiment.measurements[str(key)] for key in selected.measurement_id
                    ]
                    fit = estimate(
                        selected.n_frozen.to_numpy(),
                        selected.n_total.to_numpy(),
                        [item.dilution for item in metadata],
                        [item.droplet_volume_uL for item in metadata],
                        confidence_drop=z**2 / 2,
                    )
                record.update(
                    concentration=fit[0],
                    lower_error=fit[1],
                    upper_error=fit[2],
                    qc_flag=0 if fit[3] else 1,
                )
            records.append(record)
        if empty_count:
            notices.append(
                f"{sample_id}, run {run_id}, cycle {cycle_id}: "
                f"{empty_count} temperatures have no eligible measurements; gaps are retained"
            )
    settings = {
        "operation": "combine_dilutions",
        "estimation_method": method,
        "temperature_ranges_C": ranges,
        "range_boundaries": "inclusive",
        "z": float(z),
        "confidence_drop": float(z**2 / 2),
        "water_blank_correction": water_blank_correction,
        "water_blank_model": "volume_scaled",
        "water_blank_correction_applied": bool(experiment.water_blank_map),
        "water_blank_map": dict(experiment.water_blank_map),
        "uncertainty_assumption": (
            "pointwise profile bounds; physical droplet sets independent; "
            "shared blank observations retained. Average combines Bonferroni-adjusted "
            "marginal endpoints without assuming independent errors; coverage is approximate."
        ),
        "warnings": notices,
    }
    return CumulativeSpectrumTable(
        pd.DataFrame.from_records(records), history=fractions.history + [settings]
    )


def _validate_decrease_policy(decrease_policy: str) -> None:
    if decrease_policy not in ("stop_at_decrease", "skip_decreases"):
        raise ValueError("decrease_policy must be 'stop_at_decrease' or 'skip_decreases'")


def _final_candidates(
    spectrum: CumulativeSpectrumTable,
    *,
    decrease_policy: Literal["stop_at_decrease", "skip_decreases"],
) -> CumulativeSpectrumTable:
    """Mark final point selection while preserving every input value and error."""
    _validate_decrease_policy(decrease_policy)
    # Optimizer roundoff must not create a physical decrease when contributors
    # change. A relative-only comparison preserves real tiny signals dropping to zero.
    numerical_relative_tolerance = 1e-9
    if not isinstance(spectrum, CumulativeSpectrumTable):
        raise TypeError("spectrum must be a CumulativeSpectrumTable")
    data = spectrum.to_dataframe()
    data["used_in_final"] = False
    data["final_selection_status"] = "nonfinite"
    keys = ["run_id", "sample_id", "cycle_id"]
    if "measurement_id" in data:
        keys.append("measurement_id")
    groups, notices = [], []
    for identity, rows in data.groupby(keys, sort=False):
        group: dict[str, object] = dict(zip(keys, identity))
        previous_temperature = None
        previous_concentration = None
        first_decrease_temperature = None
        excluded = []
        retained_count = 0
        for index, row in rows.sort_values("temperature_C", ascending=False).iterrows():
            temperature = float(row.temperature_C)
            concentration = float(row.concentration) if pd.notna(row.concentration) else np.nan
            if not np.isfinite(concentration):
                status = "nonfinite"
            elif first_decrease_temperature is not None and decrease_policy == "stop_at_decrease":
                status = "colder_than_decrease"
            elif (
                previous_concentration is not None
                and concentration < previous_concentration
                and not np.isclose(
                    concentration, previous_concentration,
                    rtol=numerical_relative_tolerance, atol=0.0,
                )
            ):
                status = "decrease"
                if first_decrease_temperature is None:
                    first_decrease_temperature = temperature
            else:
                status = "kept"
            data.at[index, "final_selection_status"] = status
            if status == "kept":
                data.at[index, "used_in_final"] = True
                retained_count += 1
                previous_temperature = temperature
                previous_concentration = concentration
            else:
                excluded.append(
                    {
                        "temperature_C": temperature,
                        "concentration": concentration,
                        "reason": status,
                        "reference_temperature_C": previous_temperature,
                        "reference_concentration": previous_concentration,
                    }
                )
        group.update(
            first_decrease_temperature_C=first_decrease_temperature,
            retained_count=retained_count,
            excluded_count=len(excluded),
            excluded=excluded,
        )
        groups.append(group)
        description = ", ".join(f"{key}={value}" for key, value in zip(keys, identity))
        if excluded:
            notices.append(
                f"{description}: final selection excluded {len(excluded)} of {len(rows)} "
                f"points using {decrease_policy}"
            )
        if retained_count == 0:
            notices.append(
                f"{description}: no finite concentration points remain in the final result"
            )
    return CumulativeSpectrumTable(
        data,
        history=spectrum.history
        + [
            {
                "operation": "finalize_spectrum",
                "decrease_policy": decrease_policy,
                "comparison_order": "warm_to_cold",
                "numerical_relative_tolerance": numerical_relative_tolerance,
                "numerical_absolute_tolerance": 0.0,
                "groups": groups,
                "warnings": notices,
            }
        ],
    )


def finalize_spectrum(
    spectrum: CumulativeSpectrumTable,
    *,
    decrease_policy: Literal["stop_at_decrease", "skip_decreases"] = "stop_at_decrease",
) -> CumulativeSpectrumTable:
    """Select cumulative values nondecreasing within numerical fitting precision.

    Apply this after blank subtraction and concentration-unit conversion.
    ``stop_at_decrease`` excludes the first decrease and every colder point.
    ``skip_decreases`` excludes points below the last retained value, allowing
    later recovery. Each sample, run, cycle and optional measurement is handled
    separately. Equal values remain; nonfinite values are excluded and do not
    set the comparison baseline. A fixed relative tolerance of 1e-9 treats
    optimizer roundoff as equality; zero absolute tolerance preserves real tiny
    signals dropping to zero. Concentrations and errors are never modified.
    Returned history records the numerical tolerance and every excluded point.
    """
    return _final_candidates(spectrum, decrease_policy=decrease_policy).select(used_in_final=True)


def analyze_concentration(
    experiment: Experiment,
    *,
    method: Literal["mle", "average"] = "mle",
    temperature_ranges_C=None,
    output_basis: str = "suspension",
    step_C: float = 0.5,
    temperature_method: Literal["max", "latest", "window_max_count"] = "latest",
    temperature_tolerance_C: float | None = None,
    z: float = 1.96,
    differential: bool = False,
    blank_by_sample: dict[str, CumulativeSpectrumTable] | None = None,
    water_blank_correction: bool = True,
    decrease_policy: Literal["stop_at_decrease", "skip_decreases"] = "stop_at_decrease",
) -> AnalysisResult:
    """Run the separately callable processing steps and retain all their results."""
    from .water_blank import analysis_experiment

    _validate_decrease_policy(decrease_policy)
    method = validate_combination_method(method)
    analysis_source = analysis_experiment(
        experiment,
        water_blank_correction=water_blank_correction,
    )
    if temperature_method == "window_max_count":
        raise ValueError(
            "Concentration estimation does not support window_max_count: its synthetic "
            "warm zero rows are not raw measurements; use latest or max"
        )
    if output_basis not in UNITS:
        raise ValueError(f"Unknown output_basis {output_basis!r}")
    if not np.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    source = analysis_source.counts.to_dataframe()
    if source.empty:
        raise ValueError("Cannot analyze an experiment with no observations")
    fractions = frozen_fraction(
        analysis_source,
        step_C=step_C,
        temperature_method=temperature_method,
        temperature_tolerance_C=temperature_tolerance_C,
    )
    per_dilution = cumulative_spectrum(
        fractions,
        experiment=experiment,
        temperature_ranges_C=temperature_ranges_C,
        z=z,
        water_blank_correction=water_blank_correction,
    )
    combined = combine_dilutions(
        fractions,
        method=method,
        experiment=experiment,
        temperature_ranges_C=temperature_ranges_C,
        z=z,
        water_blank_correction=water_blank_correction,
    )
    candidates = subtract_blanks(combined, blank_by_sample) if blank_by_sample else combined
    if output_basis != "suspension":
        candidates = convert_concentration(candidates, experiment.samples, basis=output_basis)
    final_candidates = _final_candidates(candidates, decrease_policy=decrease_policy)
    final = final_candidates.select(used_in_final=True)
    differential_result = (
        differential_spectrum(
            fractions,
            experiment=experiment,
            temperature_ranges_C=temperature_ranges_C,
            water_blank_correction=water_blank_correction,
        )
        if differential
        else None
    )
    tolerance = temperature_tolerance_C
    if tolerance is None:
        tolerance = {"latest": 0.0, "max": 0.05, "window_max_count": 0.01}[temperature_method]
    settings = {
        "estimation_method": method,
        "temperature_ranges_C": combined.history[-1]["temperature_ranges_C"],
        "output_basis": output_basis,
        "step_C": step_C,
        "temperature_method": temperature_method,
        "temperature_tolerance_C": tolerance,
        "z": z,
        "differential": differential,
        "water_blank_correction": water_blank_correction,
        "water_blank_model": "volume_scaled",
        "water_blank_correction_applied": water_blank_correction
        and bool(experiment.water_blank_map),
        "decrease_policy": decrease_policy,
    }
    return AnalysisResult(
        experiment,
        fractions,
        per_dilution,
        combined,
        final,
        differential_result,
        final_candidates=final_candidates,
        settings=settings,
        history=final.history,
        warnings=final.warnings,
    )
