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
    validate_fraction_context,
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
            ("lower_error", "upper_error"), ("upper_error", "lower_error")
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


def _validate_method_inputs(method, experiment, source) -> list[float]:
    """Reject unknown measurement settings and ambiguous manual selections up front."""
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
    enforce_monotone: bool = False,
) -> CumulativeSpectrumTable:
    """Combine dilution measurements separately for each sample, run and cycle.

    The fractions carry observed counts; experiment supplies droplet volume and
    dilution metadata. Select Stitch, ManualStitch, or MLE settings. Warnings and
    effective method settings are retained on the returned spectrum.
    """
    chosen = resolve_method(method)
    if isinstance(chosen, MLE) and enforce_monotone:
        raise ValueError(
            "MLE requires enforce_monotone=False: cumulative counts at adjacent "
            "temperatures describe the same droplets, not independent observations"
        )
    if isinstance(chosen, ManualStitch) and enforce_monotone:
        raise ValueError("Manual stitching selects curves directly; enforce_monotone must be False")
    if not np.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    validate_fraction_context(fractions, experiment)
    source = fractions.to_dataframe()
    dilution_order = _validate_method_inputs(chosen, experiment, source)
    options = method_options(chosen, z=z)
    settings = {
        "operation": "combine_dilutions",
        "dilution_method": method_name(chosen),
        "method_options": options,
        "z": z,
        "enforce_monotone": enforce_monotone,
    }
    if isinstance(chosen, ManualStitch):
        settings["manual_dilution_order"] = dilution_order
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
                    if chosen != Stitch() or enforce_monotone:
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
                        enforce_monotone=enforce_monotone,
                        z=z,
                        min_unfrozen=chosen.min_unfrozen,
                        overlap_points=chosen.overlap_points,
                    )
                    uncertainty_method = "OLAF_stitched_error_width"
            else:
                combined = engine.cumulative_spec_mle(
                    inputs,
                    sample_group_by={key: sample_id for key in measurement_ids},
                    enforce_monotone=enforce_monotone,
                    **options,
                )
                uncertainty_method = "binomial_Poisson_profile_likelihood"
            frame = _public_spectrum(
                cast(engine.CumulativeNucleusSpectrumTable, combined), **identity
            )
        frame["source_measurement_ids"] = json.dumps(measurement_ids)
        frame["uncertainty_method"] = uncertainty_method
        frames.append(frame)
    settings["warnings"] = notices
    return CumulativeSpectrumTable(
        pd.concat(frames, ignore_index=True), history=fractions.history + [settings]
    )


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
    enforce_monotone: bool = False,
) -> AnalysisResult:
    """Run the separately callable processing steps and retain all their results."""
    method = resolve_method(dilution_method)
    if isinstance(method, MLE) and enforce_monotone:
        raise ValueError(
            "MLE requires enforce_monotone=False: cumulative counts at adjacent "
            "temperatures describe the same droplets, not independent observations"
        )
    if output_basis not in UNITS:
        raise ValueError(f"Unknown output_basis {output_basis!r}")
    if not np.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    if isinstance(method, ManualStitch) and enforce_monotone:
        raise ValueError("Manual stitching selects curves directly; enforce_monotone must be False")
    source = experiment.counts.to_dataframe()
    if source.empty:
        raise ValueError("Cannot analyze an experiment with no observations")
    dilution_order = _validate_method_inputs(method, experiment, source)
    fractions = frozen_fraction(
        experiment,
        step_C=step_C,
        temperature_method=temperature_method,
        temperature_tolerance_C=temperature_tolerance_C,
    )
    per_dilution = cumulative_spectrum(fractions, experiment=experiment, z=z)
    combined = combine_dilutions(
        fractions, experiment=experiment, method=method, z=z, enforce_monotone=enforce_monotone
    )
    final = subtract_blanks(combined, blank_by_sample) if blank_by_sample else combined
    if output_basis != "suspension":
        final = convert_concentration(final, experiment.samples, basis=output_basis)
    differential_result = (
        differential_spectrum(fractions, experiment=experiment) if differential else None
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
        "enforce_monotone": enforce_monotone,
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
        settings=settings,
        history=final.history,
        warnings=final.warnings,
    )
