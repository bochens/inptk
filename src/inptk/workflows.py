"""Complete concentration workflows using the retained OLAF/UFOLAF methods."""

from __future__ import annotations

import json
from itertools import pairwise
from typing import Literal, cast

import numpy as np
import pandas as pd

from . import _engine as engine
from .experiment import AnalysisResult, Experiment, SampleMetadata
from .methods import (
    MLE,
    DilutionMethod,
    ManualStitch,
    Stitch,
    method_name,
    method_options,
    resolve_method,
)
from .processing import (
    cumulative_spectrum,
    differential_spectrum,
    frozen_fraction,
    prepare_fraction_analysis,
)
from .processing import (
    fraction_inputs as _fraction_inputs,
)
from .processing import (
    public_spectrum as _public_spectrum,
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
        rows = data[data.sample_id == sample_id]
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
        data.loc[rows.index, "correction_state"] = "blank_corrected"
    return CumulativeSpectrumTable(
        data,
        history=spectrum.history
        + [
            {
                "operation": "subtract_filter_blank",
                "targets": list(blank_by_sample),
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


def _combine_with_water_blank(fractions, experiment, method, z, dilution_order, settings):
    from ._engine.transforms import _stitch_cumulative_group
    from .water_blank import corrected_frame, mle_group, sample_rows, validate_raw_mle

    source = sample_rows(fractions.to_dataframe(), experiment)
    individual = None
    if isinstance(method, MLE):
        validate_raw_mle(method)
    else:
        individual = corrected_frame(fractions, experiment, z=z)
    frames, notices = [], []
    for (run_id, sample_id, cycle_id), group in source.groupby(
        ["run_id", "sample_id", "cycle_id"], sort=False
    ):
        identity = {"run_id": str(run_id), "sample_id": str(sample_id), "cycle_id": str(cycle_id)}
        measurement_ids = group.measurement_id.astype(str).unique().tolist()
        expected = {
            key
            for key in experiment.water_blank_map
            if experiment.measurements[key].sample_id == sample_id
            and experiment.measurements[key].run_id == run_id
        }
        if expected - set(measurement_ids):
            notices.append(
                f"{identity}: absent measurements {sorted(expected - set(measurement_ids))}"
            )
        if isinstance(method, MLE):
            frame = mle_group(
                group,
                fractions.to_dataframe(),
                experiment,
                method,
                confidence_drop=method_options(method, z=z)["confidence_drop"],
            )
            uncertainty_method = "joint_sample_water_blank_profile_likelihood"
        else:
            selected = individual.loc[
                (individual.run_id == run_id)
                & (individual.sample_id == sample_id)
                & (individual.cycle_id == cycle_id)
            ].copy()
            if isinstance(method, ManualStitch):
                frame, messages = _manual_stitch(selected, method, dilution_order)
                notices.extend(f"{identity}: {message}" for message in messages)
            elif len(measurement_ids) == 1:
                frame = selected.copy()
                frame["source_measurement_id"] = frame.measurement_id
                if method != Stitch():
                    notices.append(
                        f"{identity}: only one measurement; automatic settings not applied"
                    )
            else:
                internal = selected.rename(
                    columns={
                        "concentration": "value",
                        "unit": "value_unit",
                        "lower_error": "lower_ci",
                        "upper_error": "upper_ci",
                    }
                ).assign(source_sample_id=selected.measurement_id)
                frame = _stitch_cumulative_group(
                    internal,
                    str(sample_id),
                    min_unfrozen=method.min_unfrozen,
                ).rename(
                    columns={
                        "value": "concentration",
                        "value_unit": "unit",
                        "lower_ci": "lower_error",
                        "upper_ci": "upper_error",
                    }
                )
            uncertainty_method = "joint_sample_water_blank_profile_likelihood"
        frame = frame.drop(
            columns=[
                "measurement_id",
                "n_frozen",
                "n_total",
                "fraction_frozen",
                "time_s",
                "error_components",
            ],
            errors="ignore",
        ).assign(**identity)
        frame["source_measurement_ids"] = json.dumps(measurement_ids)
        frame["uncertainty_method"] = uncertainty_method
        frame["correction_state"] = "water_blank_corrected"
        frame["qc_flag"] = frame.qc_flag.astype(int) | np.where(frame.concentration < 0, 2, 0)
        frames.append(frame)
    settings.update(
        water_blank_map=dict(experiment.water_blank_map),
        water_blank_matching="exact_run_cycle_temperature",
        warnings=notices,
    )
    return CumulativeSpectrumTable(
        pd.concat(frames, ignore_index=True), history=fractions.history + [settings]
    )


def _validate_method_inputs(method, experiment, source) -> list[float]:
    """Reject unknown measurement settings and ambiguous manual selections up front."""
    from .water_blank import sample_rows

    source = sample_rows(source, experiment)
    dilutions = sorted(
        {float(experiment.measurements[key].dilution) for key in source.measurement_id.unique()}
    )
    if isinstance(method, MLE):
        for name in (
            "temperature_eligibility_C",
            "likelihood_weights",
            "action_counts",
        ):
            mapping = getattr(method, name)
            if mapping is not None:
                available = set(source.measurement_id)
                unknown = set(mapping) - available
                if unknown:
                    raise ValueError(
                        f"{name} names unknown measurements: {sorted(unknown)}; "
                        f"available measurement names: {sorted(available)}"
                    )
    if isinstance(method, ManualStitch):
        if len(method.switch_temperatures_C) != len(dilutions) - 1:
            raise ValueError(
                f"Manual stitching with {len(dilutions)} dilutions requires "
                f"{len(dilutions) - 1} switch temperatures"
            )
        for identity, rows in source.groupby(["run_id", "sample_id", "cycle_id"], sort=False):
            present = sorted(
                float(experiment.measurements[key].dilution) for key in rows.measurement_id.unique()
            )
            if any(np.isclose(a, b, rtol=0, atol=1e-12) for a, b in pairwise(present)):
                raise ValueError(
                    f"Manual stitching has multiple measurements at the same dilution in {identity}"
                )
            if present != dilutions:
                raise ValueError(
                    "Manual stitching requires the same dilution factors "
                    "in every sample/run/cycle; "
                    f"{identity} has {present}, expected {dilutions}. "
                    "Analyze different dilution sets separately."
                )
    return dilutions


def _manual_stitch(data, method: ManualStitch, dilution_order: list[float]):
    """Pick the requested curve on the union grid, preserving unavailable points."""
    records = []
    selected_dilutions = set()
    for temperature in sorted(data.temperature_C.unique(), reverse=True):
        position = sum(temperature <= switch for switch in method.switch_temperatures_C)
        dilution = dilution_order[position]
        at_temperature = data[data.temperature_C == temperature]
        candidates = at_temperature[
            np.isclose(at_temperature.dilution_fold, dilution, rtol=0, atol=1e-12)
        ]
        if len(candidates) > 1:
            raise ValueError(
                f"Manual stitching has multiple measurements for dilution {dilution:g}"
            )
        if candidates.empty:
            row = at_temperature.iloc[0].to_dict()
            row.update(
                concentration=np.nan,
                lower_error=np.nan,
                upper_error=np.nan,
                source_measurement_id="",
                selection_status="temperature_unavailable",
                qc_flag=1,
            )
        else:
            row = candidates.iloc[0].to_dict()
            row["source_measurement_id"] = str(row["measurement_id"])
            row["selection_status"] = "selected"
            if not np.isfinite(row["concentration"]):
                row.update(
                    concentration=np.nan,
                    lower_error=np.nan,
                    upper_error=np.nan,
                    selection_status="nonfinite_concentration",
                    qc_flag=1,
                )
            else:
                selected_dilutions.add(dilution)
        row.pop("measurement_id", None)
        row["dilution_fold"] = dilution
        records.append(row)
    frame = pd.DataFrame.from_records(records)
    notices = []
    unavailable = frame[frame.selection_status != "selected"]
    if not unavailable.empty:
        notices.append(
            f"Manual stitching left {len(unavailable)} temperatures missing: "
            f"{', '.join(sorted(unavailable.selection_status.unique()))}; no fallback was used"
        )
    unused = set(dilution_order) - selected_dilutions
    if unused:
        notices.append(
            f"No finite points were used for dilution factors {sorted(unused)} "
            "under the manual switches"
        )
    return frame, notices


def combine_dilutions(
    fractions: FrozenFractionTable,
    *,
    experiment: Experiment,
    method: str | DilutionMethod = "stitch",
    z: float = 1.96,
    water_blank_correction: bool = True,
) -> CumulativeSpectrumTable:
    """Combine dilution measurements separately for each sample, run and cycle.

    The fractions carry observed counts; experiment supplies droplet volume and
    dilution metadata. Select Stitch, ManualStitch, or MLE settings. Warnings and
    effective method settings are retained on the returned spectrum.
    """
    chosen = resolve_method(method)
    if not np.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    fractions, experiment = prepare_fraction_analysis(
        fractions, experiment, water_blank_correction=water_blank_correction
    )
    source = fractions.to_dataframe()
    dilution_order = _validate_method_inputs(chosen, experiment, source)
    options = method_options(chosen, z=z)
    settings = {
        "operation": "combine_dilutions",
        "water_blank_correction": water_blank_correction,
        "water_blank_correction_applied": bool(experiment.water_blank_map),
        "dilution_method": method_name(chosen),
        "method_options": options,
        "z": z,
    }
    if isinstance(chosen, ManualStitch):
        settings["manual_dilution_order"] = dilution_order
    if experiment.water_blank_map:
        return _combine_with_water_blank(fractions, experiment, chosen, z, dilution_order, settings)
    if isinstance(chosen, ManualStitch):
        individual = cumulative_spectrum(fractions, experiment=experiment, z=z).to_dataframe()
    frames, notices = [], []
    for (run_id, sample_id, cycle_id), group in source.groupby(
        ["run_id", "sample_id", "cycle_id"], sort=False
    ):
        run_id, sample_id, cycle_id = str(run_id), str(sample_id), str(cycle_id)
        identity = {"run_id": run_id, "sample_id": sample_id, "cycle_id": cycle_id}
        measurement_ids = list(group.measurement_id.unique())
        expected = {
            key
            for key, value in experiment.measurements.items()
            if (value.sample_id, value.run_id) == (sample_id, run_id)
        }
        missing = expected - set(measurement_ids)
        if missing:
            notices.append(
                f"{sample_id}, run {run_id}, cycle {cycle_id}: "
                f"absent measurements {sorted(missing)}"
            )
        if isinstance(chosen, ManualStitch):
            selected = individual[
                (individual.run_id == run_id)
                & (individual.sample_id == sample_id)
                & (individual.cycle_id == cycle_id)
            ]
            frame, manual_notices = _manual_stitch(selected, chosen, dilution_order)
            notices.extend(
                f"{sample_id}, run {run_id}, cycle {cycle_id}: {notice}"
                for notice in manual_notices
            )
            uncertainty_method = "OLAF_Agresti_Coull_error_width"
        else:
            inputs = _fraction_inputs(group, experiment)
            if isinstance(chosen, Stitch):
                if len(inputs) == 1:
                    if chosen != Stitch():
                        notices.append(
                            f"{sample_id}, run {run_id}, cycle {cycle_id}: only one measurement; "
                            "automatic stitching settings are not applied"
                        )
                    combined = engine.cumulative_spec(inputs[0], z=z)
                    uncertainty_method = "OLAF_Agresti_Coull_error_width"
                else:
                    combined = engine.cumulative_spec_stitch(
                        inputs,
                        sample_group_by={key: sample_id for key in measurement_ids},
                        z=z,
                        min_unfrozen=chosen.min_unfrozen,
                    )
                    uncertainty_method = "OLAF_Agresti_Coull_error_width"
            else:
                combined = engine.cumulative_spec_mle(
                    inputs,
                    sample_group_by={key: sample_id for key in measurement_ids},
                    enforce_monotone=False,
                    **options,
                )
                uncertainty_method = "binomial_Poisson_profile_likelihood"
            frame = _public_spectrum(
                cast(engine.CumulativeNucleusSpectrumTable, combined), **identity
            )
            if isinstance(chosen, Stitch):
                sources = {
                    float(experiment.measurements[key].dilution): str(key)
                    for key in measurement_ids
                }
                frame["source_measurement_id"] = (
                    frame.dilution_fold.map(sources)
                    .fillna("")
                    .where(np.isfinite(frame.concentration), "")
                )
                frame["selection_status"] = np.where(
                    frame.source_measurement_id.ne(""), "selected", "nonfinite_concentration"
                )
        frame["source_measurement_ids"] = json.dumps(measurement_ids)
        frame["uncertainty_method"] = uncertainty_method
        frames.append(frame)
    settings["warnings"] = notices
    return CumulativeSpectrumTable(
        pd.concat(frames, ignore_index=True), history=fractions.history + [settings]
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
            elif previous_concentration is not None and concentration < previous_concentration:
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
    """Select nondecreasing cumulative values in cooling order without changing them.

    Apply this after blank subtraction and concentration-unit conversion.
    ``stop_at_decrease`` excludes the first decrease and every colder point.
    ``skip_decreases`` excludes points below the last retained value, allowing
    later recovery. Each sample, run, cycle and optional measurement is handled
    separately. Equal values remain; nonfinite values are excluded and do not
    set the comparison baseline. Returned history records every excluded point.
    """
    return _final_candidates(spectrum, decrease_policy=decrease_policy).select(used_in_final=True)


def analyze_concentration(
    experiment: Experiment,
    *,
    dilution_method: str | DilutionMethod = "stitch",
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
    analysis_source = analysis_experiment(experiment, water_blank_correction=water_blank_correction)
    if analysis_source.water_blank_map and temperature_method == "window_max_count":
        raise ValueError(
            "Raw water-blank correction does not support window_max_count: its synthetic "
            "warm zero rows are not raw measurements; use latest or max"
        )
    method = resolve_method(dilution_method)
    if output_basis not in UNITS:
        raise ValueError(f"Unknown output_basis {output_basis!r}")
    if not np.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    source = analysis_source.counts.to_dataframe()
    if source.empty:
        raise ValueError("Cannot analyze an experiment with no observations")
    dilution_order = _validate_method_inputs(method, analysis_source, source)
    fractions = frozen_fraction(
        analysis_source,
        step_C=step_C,
        temperature_method=temperature_method,
        temperature_tolerance_C=temperature_tolerance_C,
    )
    per_dilution = cumulative_spectrum(
        fractions, experiment=experiment, z=z, water_blank_correction=water_blank_correction
    )
    combined = combine_dilutions(
        fractions,
        experiment=experiment,
        method=method,
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
            fractions, experiment=experiment, water_blank_correction=water_blank_correction
        )
        if differential
        else None
    )
    tolerance = temperature_tolerance_C
    if tolerance is None:
        tolerance = {"latest": 0.0, "max": 0.05, "window_max_count": 0.01}[temperature_method]
    settings = {
        "dilution_method": method_name(method),
        "method_options": method_options(method, z=z),
        "output_basis": output_basis,
        "step_C": step_C,
        "temperature_method": temperature_method,
        "temperature_tolerance_C": tolerance,
        "z": z,
        "differential": differential,
        "water_blank_correction": water_blank_correction,
        "water_blank_correction_applied": water_blank_correction
        and bool(experiment.water_blank_map),
        "decrease_policy": decrease_policy,
    }
    if isinstance(method, ManualStitch):
        settings["manual_dilution_order"] = dilution_order
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
