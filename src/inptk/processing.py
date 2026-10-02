"""Individual processing steps, with identities and preceding history preserved."""

from __future__ import annotations

from typing import Literal, cast

import numpy as np
import pandas as pd

from . import _engine as engine
from ._engine.water_blank_math import fit_concentration
from .experiment import Experiment
from .methods import validate_temperature_ranges
from .tables import (
    CountsTable,
    CumulativeSpectrumTable,
    DifferentialSpectrumTable,
    FrozenFractionTable,
)


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
    fractions: FrozenFractionTable,
    experiment: Experiment,
    *,
    water_blank_correction: bool,
) -> tuple[FrozenFractionTable, Experiment]:
    """Validate identities and remove blank context only from a disabled analysis view."""
    from .water_blank import analysis_experiment, sample_rows

    validate_fraction_context(fractions, experiment)
    view = analysis_experiment(
        experiment,
        water_blank_correction=water_blank_correction,
    )
    frame = fractions.to_dataframe()
    if any(
        step.get("operation") == "frozen_fraction"
        and step.get("temperature_method") == "window_max_count"
        for step in fractions.history
    ) or (
        "temperature_bin_method" in frame
        and frame.temperature_bin_method.astype(str)
        .str.contains("window_max_count", regex=False)
        .any()
    ):
        raise ValueError(
            "Concentration estimation does not support window_max_count: its synthetic "
            "warm zero rows are not raw measurements; use latest or max"
        )
    if view is not experiment:
        rows = sample_rows(fractions.to_dataframe(), experiment)
        if rows.empty:
            raise ValueError("No sample observations remain after excluding water-blank sets")
        fractions = FrozenFractionTable(rows, history=fractions.history)
    return fractions, view


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


def cumulative_spectrum(
    fractions: FrozenFractionTable,
    *,
    experiment: Experiment,
    temperature_ranges_C=None,
    z: float = 1.96,
    water_blank_correction: bool = True,
) -> CumulativeSpectrumTable:
    """Calculate each measurement with the same count model used for combination.

    Outside-range rows retain their observed counts but have no concentration
    estimate. Matching blank observations are required only for eligible rows.
    """
    import json

    from .water_blank import fit_raw_rows, pooled_blank, sample_rows

    if not np.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    fractions, experiment = prepare_fraction_analysis(
        fractions,
        experiment,
        water_blank_correction=water_blank_correction,
    )
    frame = fractions.to_dataframe()
    source = sample_rows(frame, experiment)
    blank_ids = {key for ids in experiment.water_blank_map.values() for key in ids}
    ranges = validate_temperature_ranges(
        temperature_ranges_C, measurement_ids=set(experiment.measurements) - blank_ids
    )
    if source.empty:
        raise ValueError("No sample observations are available for concentration calculation")
    records = []
    for position in range(len(source)):
        selected = source.iloc[[position]]
        row = selected.iloc[0].to_dict()
        measurement = str(row["measurement_id"])
        metadata = experiment.measurements[measurement]
        limits = ranges.get(measurement, {})
        minimum, maximum = limits.get("min_C"), limits.get("max_C")
        temperature = float(row["temperature_C"])
        eligible = (minimum is None or temperature >= minimum) and (
            maximum is None or temperature <= maximum
        )
        row.update(
            concentration=np.nan,
            lower_error=np.nan,
            upper_error=np.nan,
            unit="INP_per_mL_suspension",
            basis="suspension",
            qc_flag=1,
            dilution_fold=metadata.dilution,
            selection_status="selected" if eligible else "outside_temperature_range",
            uncertainty_method="joint_sample_water_blank_profile_likelihood"
            if experiment.water_blank_map
            else "binomial_Poisson_profile_likelihood",
            correction_state="water_blank_corrected"
            if experiment.water_blank_map
            else "uncorrected",
        )
        if experiment.water_blank_map:
            row["water_blank_ids"] = json.dumps(sorted(experiment.water_blank_map[measurement]))
        if eligible:
            if experiment.water_blank_map:
                fit = fit_raw_rows(selected, frame, experiment, confidence_drop=z**2 / 2)
                blank = (
                    pooled_blank(selected, frame, experiment.water_blank_map[measurement])
                    .iloc[0]
                    .to_dict()
                )
                row.update(blank_n_frozen=blank["n_frozen"], blank_n_total=blank["n_total"])
            else:
                fit = fit_concentration(
                    row["n_frozen"],
                    row["n_total"],
                    metadata.dilution,
                    metadata.droplet_volume_uL,
                    confidence_drop=z**2 / 2,
                )
            row.update(
                concentration=fit[0],
                lower_error=fit[1],
                upper_error=fit[2],
                qc_flag=0 if fit[3] else 1,
            )
        row["at_zero_boundary"] = row["concentration"] == 0
        records.append(row)
    return CumulativeSpectrumTable(
        pd.DataFrame.from_records(records),
        history=fractions.history
        + [
            {
                "operation": "cumulative_spectrum",
                "estimation_method": "mle",
                "z": float(z),
                "temperature_ranges_C": ranges,
                "range_boundaries": "inclusive",
                "water_blank_correction": water_blank_correction,
                "water_blank_model": "volume_scaled",
                "water_blank_correction_applied": bool(experiment.water_blank_map),
                "water_blank_map": dict(experiment.water_blank_map),
                "uncertainty_assumption": (
                    "pointwise count likelihood; sample and blank observations at each "
                    "temperature retain their own totals and volumes"
                ),
            }
        ],
    )


def differential_spectrum(
    fractions: FrozenFractionTable,
    *,
    experiment: Experiment,
    temperature_ranges_C=None,
    water_blank_correction: bool = True,
) -> DifferentialSpectrumTable:
    """Calculate adjacent concentration changes per degree for each droplet set.

    Each interval uses its actual width and both endpoint concentrations.
    Excluded or unavailable endpoints produce a flagged missing interval;
    intervals never bridge a missing point. Cycles remain separate.
    """
    cumulative = cumulative_spectrum(
        fractions,
        experiment=experiment,
        temperature_ranges_C=temperature_ranges_C,
        water_blank_correction=water_blank_correction,
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
        history=cumulative.history + [{"operation": "differential_spectrum"}],
    )
