"""Complete concentration workflows using the retained OLAF/UFOLAF methods."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Literal, cast

import numpy as np
import pandas as pd

from . import _engine as engine
from .experiment import AnalysisResult, Experiment, SampleMetadata
from .tables import UNITS, CumulativeSpectrumTable, DifferentialSpectrumTable, FrozenFractionTable


def _engine_metadata(experiment, measurement_id):
    measurement = experiment.measurements[measurement_id]
    sample = experiment.samples[measurement.sample_id]
    values = asdict(sample)
    values.update(
        sample_id=measurement_id,
        sample_name=sample.sample_id,
        sample_long_name=sample.sample_id,
        well_volume_uL=measurement.droplet_volume_uL,
        dilution=measurement.dilution,
    )
    return engine.SampleMetadata(**values)


def _tag(frame, *, run_id, sample_id, cycle_id, measurement_id=None):
    frame = frame.copy()
    frame["run_id"] = run_id
    frame["sample_id"] = sample_id
    frame["cycle_id"] = cycle_id
    if measurement_id is not None:
        frame["measurement_id"] = measurement_id
    return frame


def _public_spectrum(table, **identity):
    return _tag(
        table.to_dataframe().rename(
            columns={
                "value": "concentration",
                "value_unit": "unit",
                "lower_ci": "lower_error",
                "upper_ci": "upper_error",
            }
        ),
        **identity,
    )


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


def _subtract_blanks(spectrum, blank_by_sample):
    """Match blank curves explicitly by target sample, run, cycle and temperature."""
    data = spectrum.to_dataframe()
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
        for column in ("lower_error", "upper_error"):
            if column in rows and column in aligned:
                data.loc[rows.index, column] = np.sqrt(
                    rows[column].to_numpy() ** 2 + aligned[column].to_numpy() ** 2
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
                    "independent sample and blank errors; retained root-sum-of-squares method"
                ),
                "blanks": {
                    key: value.to_dataframe().to_dict("records")
                    for key, value in blank_by_sample.items()
                },
            }
        ],
    )


def analyze_concentration(
    experiment: Experiment,
    *,
    dilution_method: str = "stitch",
    output_basis: str = "suspension",
    step_C: float = 0.5,
    temperature_method: Literal["max", "latest", "olaf"] = "max",
    temperature_tolerance_C: float | None = None,
    z: float = 1.96,
    differential: bool = False,
    blank_by_sample: dict[str, CumulativeSpectrumTable] | None = None,
    enforce_monotone: bool = False,
) -> AnalysisResult:
    """Produce one combined spectrum per original sample, run and cycle.

    Dilution methods are retained: 'stitch' selects portions of per-dilution
    curves; 'mle' fits the count observations from the dilutions jointly.
    Different runs and repeated cycles are always kept separate.
    """
    if dilution_method not in ("stitch", "mle"):
        raise ValueError("dilution_method must be stitch or mle")
    if output_basis not in UNITS:
        raise ValueError(f"Unknown output_basis {output_basis!r}")
    if not np.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    if not np.isfinite(step_C) or step_C <= 0:
        raise ValueError("step_C must be finite and positive")
    tolerance = temperature_tolerance_C
    if tolerance is None:
        tolerance = 0.01 if temperature_method == "olaf" else 0.05
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("temperature_tolerance_C must be finite and nonnegative")
    settings = {
        "dilution_method": dilution_method,
        "output_basis": output_basis,
        "step_C": step_C,
        "temperature_method": temperature_method,
        "temperature_tolerance_C": tolerance,
        "z": z,
        "differential": differential,
        "enforce_monotone": enforce_monotone,
    }
    source = experiment.counts.to_dataframe()
    if source.empty:
        raise ValueError("Cannot analyze an experiment with no observations")
    fraction_frames, per_frames, combined_frames, differential_frames = [], [], [], []
    notices = []
    for (run_id, sample_id, cycle_id), group in source.groupby(
        ["run_id", "sample_id", "cycle_id"], sort=False
    ):
        run_id, sample_id, cycle_id = str(run_id), str(sample_id), str(cycle_id)
        fraction_inputs = []
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
        for measurement_id, rows in group.groupby("measurement_id", sort=False):
            measurement_id = str(measurement_id)
            frame = rows.copy()
            frame["sample_id"] = measurement_id
            frame["cycle"] = cycle_id
            metadata = _engine_metadata(experiment, measurement_id)
            counts = engine.CountsTable.from_dataframe(frame, metadata=metadata)
            fraction = cast(
                engine.TemperatureFrozenFractionTable,
                engine.fraction_frozen(
                    counts,
                    step_C=step_C,
                    method=temperature_method,
                    temperature_tolerance_C=tolerance,
                    cooling_only=True,
                ),
            )
            fraction_inputs.append(fraction)
            fraction_frames.append(
                _tag(fraction.to_dataframe(), **identity, measurement_id=measurement_id)
            )
            per_frames.append(
                _public_spectrum(
                    engine.cumulative_spec(fraction, z=z), **identity, measurement_id=measurement_id
                )
            )
            if differential:
                differential_frames.append(
                    _public_spectrum(
                        engine.differential_spec(fraction),
                        **identity,
                        measurement_id=measurement_id,
                    )
                )
        if len(fraction_inputs) == 1 and dilution_method == "stitch":
            engine_combined = engine.cumulative_spec(fraction_inputs[0], z=z)
            uncertainty_method = "OLAF_Agresti_Coull_error_width"
        elif dilution_method == "stitch":
            engine_combined = engine.cumulative_spec_stitch(
                fraction_inputs,
                sample_group_by={key: sample_id for key in measurement_ids},
                enforce_monotone=enforce_monotone,
                z=z,
            )
            uncertainty_method = "OLAF_stitched_error_width"
        else:
            engine_combined = engine.cumulative_spec_mle(
                fraction_inputs,
                sample_group_by={key: sample_id for key in measurement_ids},
                enforce_monotone=enforce_monotone,
                confidence_drop=z**2 / 2,
            )
            uncertainty_method = "binomial_Poisson_profile_likelihood"
        frame = _public_spectrum(engine_combined, **identity)
        frame["source_measurement_ids"] = json.dumps(measurement_ids)
        frame["uncertainty_method"] = uncertainty_method
        combined_frames.append(frame)
    history = [{"operation": "analyze_concentration", **settings}]
    fractions = FrozenFractionTable(pd.concat(fraction_frames, ignore_index=True), history=history)
    per_frame = pd.concat(per_frames, ignore_index=True)
    per_frame["uncertainty_method"] = "OLAF_Agresti_Coull_error_width"
    per = CumulativeSpectrumTable(per_frame, history=history)
    combined = CumulativeSpectrumTable(
        pd.concat(combined_frames, ignore_index=True), history=history
    )
    final = _subtract_blanks(combined, blank_by_sample) if blank_by_sample else combined
    if output_basis != "suspension":
        final = convert_concentration(final, experiment.samples, basis=output_basis)
    differential_result = (
        DifferentialSpectrumTable(
            pd.concat(differential_frames, ignore_index=True), history=history
        )
        if differential
        else None
    )
    return AnalysisResult(
        experiment,
        fractions,
        per,
        combined,
        final,
        differential_result,
        settings=settings,
        history=final.history,
        warnings=notices,
    )
