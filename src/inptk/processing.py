"""Individual processing steps, with identities and preceding history preserved."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict
from typing import Literal, cast

import numpy as np
import pandas as pd

from . import _engine as engine
from .experiment import Experiment
from .tables import (
    CountsTable,
    CumulativeSpectrumTable,
    DifferentialSpectrumTable,
    FrozenFractionTable,
)


def engine_metadata(experiment: Experiment, measurement_id: str) -> engine.SampleMetadata:
    """Attach original-sample information to one retained calculation input."""
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


def public_spectrum(
    table: engine.CumulativeNucleusSpectrumTable | engine.DifferentialNucleusSpectrumTable,
    **identity: str,
) -> pd.DataFrame:
    """Convert retained spectrum column names and restore public identities."""
    frame = table.to_dataframe().rename(
        columns={
            "value": "concentration",
            "value_unit": "unit",
            "lower_ci": "lower_error",
            "upper_ci": "upper_error",
        }
    )
    return frame.assign(**identity)


def validate_fraction_context(
    fractions: FrozenFractionTable, experiment: Experiment
) -> pd.DataFrame:
    """Validate measurement ownership and return a copy of fraction rows."""
    if not isinstance(fractions, FrozenFractionTable):
        raise TypeError("fractions must be a FrozenFractionTable")
    if not isinstance(experiment, Experiment):
        raise TypeError("experiment must be an Experiment")
    frame = fractions.to_dataframe()
    if frame.empty:
        raise ValueError("Cannot calculate a spectrum from an empty frozen-fraction table")
    identities = frame[["measurement_id", "sample_id", "run_id"]].drop_duplicates()
    for row in identities.itertuples(index=False):
        measurement_id = str(row.measurement_id)
        if measurement_id not in experiment.measurements:
            raise ValueError(f"Missing measurement metadata for {measurement_id!r}")
        measurement = experiment.measurements[measurement_id]
        if (row.sample_id, row.run_id) != (measurement.sample_id, measurement.run_id):
            raise ValueError(
                f"Frozen-fraction identities disagree with metadata for {measurement_id!r}"
            )
        if measurement.sample_id not in experiment.samples:
            raise ValueError(f"Missing sample metadata for {measurement.sample_id!r}")
    return frame


def prepare_fraction_analysis(
    fractions: FrozenFractionTable, experiment: Experiment, *, water_blank_correction: bool
) -> tuple[FrozenFractionTable, Experiment]:
    """Validate identities and remove blank context only from a disabled analysis view."""
    from .water_blank import analysis_experiment, sample_rows

    validate_fraction_context(fractions, experiment)
    view = analysis_experiment(experiment, water_blank_correction=water_blank_correction)
    frame = fractions.to_dataframe()
    if view.water_blank_map and (
        any(
            step.get("operation") == "frozen_fraction"
            and step.get("temperature_method") == "window_max_count"
            for step in fractions.history
        )
        or (
            "temperature_bin_method" in frame
            and frame.temperature_bin_method.astype(str)
            .str.contains("window_max_count", regex=False)
            .any()
        )
    ):
        raise ValueError(
            "Raw water-blank correction does not support window_max_count: its synthetic "
            "warm zero rows are not raw measurements; use latest or max"
        )
    if view is not experiment:
        rows = sample_rows(fractions.to_dataframe(), experiment)
        if rows.empty:
            raise ValueError("No sample observations remain after excluding water-blank sets")
        fractions = FrozenFractionTable(rows, history=fractions.history)
    return fractions, view


def fraction_inputs(
    rows: pd.DataFrame, experiment: Experiment
) -> list[engine.TemperatureFrozenFractionTable]:
    """Build separate dilution inputs for exactly one sample, run and cycle."""
    if rows.empty:
        raise ValueError("Cannot build calculation inputs from empty frozen-fraction rows")
    if len(rows[["run_id", "sample_id", "cycle_id"]].drop_duplicates()) != 1:
        raise ValueError("Calculation inputs must contain exactly one sample, run and cycle")
    inputs = []
    for measurement_id, group in rows.groupby("measurement_id", sort=False):
        measurement_id = str(measurement_id)
        frame = group.assign(sample_id=measurement_id)
        # from_dataframe retains width, reduction method, interval bounds and
        # observation counts; reconstructing only the counts would lose these.
        inputs.append(
            engine.TemperatureFrozenFractionTable.from_dataframe(
                frame, metadata=engine_metadata(experiment, measurement_id)
            )
        )
    return inputs


def frozen_fraction(
    source: Experiment | CountsTable,
    *,
    step_C: float = 0.5,
    temperature_method: Literal["max", "latest", "window_max_count"] = "latest",
    temperature_tolerance_C: float | None = None,
) -> FrozenFractionTable:
    """Evaluate frozen fractions at temperature thresholds using counts alone.

    Dilution and droplet-volume metadata are unnecessary for this step. Every
    measurement and freezing cycle is reduced separately, using its cooling phase.

    ``max`` selects the highest frozen fraction among observations at or warmer
    than each target (allowing the tolerance). ``latest`` selects the last such
    observation. ``window_max_count`` selects the highest frozen count within
    the temperature tolerance window; if empty, it uses the highest count on
    the warmer side. The selected frozen and total counts remain paired.

    ``window_max_count`` also retains a first-freeze row rounded to 0.1 C and
    four warmer zero rows. It adapts original OLAF's table-construction rule.
    Default selection is ``latest`` with zero tolerance. ``max`` defaults to
    0.05 C tolerance; ``window_max_count`` defaults to 0.01 C.
    """
    counts = source.counts if isinstance(source, Experiment) else source
    if not isinstance(counts, CountsTable):
        raise TypeError("source must be an Experiment or CountsTable")
    if temperature_method not in ("max", "latest", "window_max_count"):
        raise ValueError("temperature_method must be max, latest, or window_max_count")
    if not np.isfinite(step_C) or step_C <= 0:
        raise ValueError("step_C must be finite and positive")
    tolerance = temperature_tolerance_C
    if tolerance is None:
        tolerance = {"latest": 0.0, "max": 0.05, "window_max_count": 0.01}[temperature_method]
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("temperature_tolerance_C must be finite and nonnegative")
    source_frame = counts.to_dataframe()
    if source_frame.empty:
        raise ValueError("Cannot calculate frozen fractions from an empty counts table")
    frames = []
    for (run_id, sample_id, cycle_id, measurement_id), rows in source_frame.groupby(
        ["run_id", "sample_id", "cycle_id", "measurement_id"], sort=False
    ):
        identity = {
            "run_id": str(run_id),
            "sample_id": str(sample_id),
            "cycle_id": str(cycle_id),
            "measurement_id": str(measurement_id),
        }
        raw = engine.CountsTable.from_dataframe(
            rows.assign(sample_id=str(measurement_id), cycle=str(cycle_id))
        )
        reduced = cast(
            engine.TemperatureFrozenFractionTable,
            engine.fraction_frozen(
                raw,
                step_C=step_C,
                method=temperature_method,
                temperature_tolerance_C=tolerance,
                cooling_only=True,
            ),
        )
        frames.append(reduced.to_dataframe().assign(**identity))
    return FrozenFractionTable(
        pd.concat(frames, ignore_index=True),
        history=counts.history
        + [
            {
                "operation": "frozen_fraction",
                "step_C": float(step_C),
                "temperature_method": temperature_method,
                "temperature_tolerance_C": float(tolerance),
                "cooling_only": True,
            }
        ],
    )


def _grouped_fractions(
    frame: pd.DataFrame, experiment: Experiment
) -> Iterator[tuple[dict[str, str], engine.TemperatureFrozenFractionTable]]:
    for (run_id, sample_id, cycle_id), rows in frame.groupby(
        ["run_id", "sample_id", "cycle_id"], sort=False
    ):
        for fraction in fraction_inputs(rows, experiment):
            yield (
                {
                    "run_id": str(run_id),
                    "sample_id": str(sample_id),
                    "cycle_id": str(cycle_id),
                    "measurement_id": str(fraction.sample_id[0]),
                },
                fraction,
            )


def cumulative_spectrum(
    fractions: FrozenFractionTable,
    *,
    experiment: Experiment,
    z: float = 1.96,
    water_blank_correction: bool = True,
) -> CumulativeSpectrumTable:
    """Calculate concentration for each measurement, preserving every cycle."""
    if not np.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    fractions, experiment = prepare_fraction_analysis(
        fractions, experiment, water_blank_correction=water_blank_correction
    )
    frame = fractions.to_dataframe()
    if experiment.water_blank_map:
        from .water_blank import cumulative_with_water_blank

        return cumulative_with_water_blank(fractions, experiment, z=z)
    frames = []
    for identity, fraction in _grouped_fractions(frame, experiment):
        calculated = cast(
            engine.CumulativeNucleusSpectrumTable, engine.cumulative_spec(fraction, z=z)
        )
        frames.append(public_spectrum(calculated, **identity))
    result = pd.concat(frames, ignore_index=True)
    result["uncertainty_method"] = "OLAF_Agresti_Coull_error_width"
    return CumulativeSpectrumTable(
        result,
        history=fractions.history
        + [
            {
                "operation": "cumulative_spectrum",
                "z": float(z),
                "water_blank_correction": water_blank_correction,
                "water_blank_correction_applied": False,
            }
        ],
    )


def differential_spectrum(
    fractions: FrozenFractionTable,
    *,
    experiment: Experiment,
    water_blank_correction: bool = True,
) -> DifferentialSpectrumTable:
    """Calculate adjacent concentration changes per degree for each droplet set.

    Each interval uses its actual temperature width and both endpoint frozen
    fractions, so corrected totals may change. The first state has no preceding
    observed interval and supplies no output row. Negative changes are retained
    with quality flag 2; nonfinite values carry flag 1. This arithmetic does not
    establish that arbitrary droplet loss is a valid background correction.
    """
    fractions, experiment = prepare_fraction_analysis(
        fractions, experiment, water_blank_correction=water_blank_correction
    )
    frame = fractions.to_dataframe()
    if experiment.water_blank_map:
        cumulative = cumulative_spectrum(
            fractions, experiment=experiment, water_blank_correction=water_blank_correction
        )
        intervals = []
        for _, group in cumulative.to_dataframe().groupby(
            ["run_id", "sample_id", "cycle_id", "measurement_id"], sort=False
        ):
            group = group.sort_values("temperature_C", ascending=False)
            temperatures = group.temperature_C.to_numpy(dtype=float)
            out = group.iloc[:-1][["run_id", "sample_id", "cycle_id", "measurement_id"]].copy()
            out["temperature_C"] = temperatures[:-1]
            out["temperature_bin_left_C"] = temperatures[1:]
            out["temperature_bin_right_C"] = temperatures[:-1]
            with np.errstate(invalid="ignore"):
                out["concentration"] = np.diff(group.concentration) / -np.diff(temperatures)
            out["unit"] = "INP_per_mL_suspension_per_C"
            out["basis"] = "suspension"
            out["qc_flag"] = np.where(np.isfinite(out.concentration), 0, 1) | np.where(
                out.concentration < 0, 2, 0
            )
            intervals.append(out)
        return DifferentialSpectrumTable(
            pd.concat(intervals, ignore_index=True),
            history=cumulative.history
            + [
                {
                    "operation": "differential_spectrum",
                    "water_blank_correction": water_blank_correction,
                }
            ],
        )
    frames = []
    for identity, fraction in _grouped_fractions(frame, experiment):
        calculated = cast(
            engine.DifferentialNucleusSpectrumTable, engine.differential_spec(fraction)
        )
        frames.append(public_spectrum(calculated, **identity))
    return DifferentialSpectrumTable(
        pd.concat(frames, ignore_index=True),
        history=fractions.history
        + [
            {
                "operation": "differential_spectrum",
                "water_blank_correction": water_blank_correction,
            }
        ],
    )
