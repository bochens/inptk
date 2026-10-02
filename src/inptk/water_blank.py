"""Raw sample/blank calculations with explicit physical blank assignments.

Counts in this path have not been background-subtracted. The caller selects
actual sample and blank observations; blank observations are never fabricated.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from ._engine.water_blank_math import average_concentration, fit_concentration
from .experiment import Experiment
from .tables import CountsTable


def sample_rows(frame: pd.DataFrame, experiment: Experiment) -> pd.DataFrame:
    """Exclude physical water blanks from the sample analysis groups."""
    return frame.loc[
        ~frame.measurement_id.isin(
            {key for ids in experiment.water_blank_map.values() for key in ids}
        )
    ].copy()


def analysis_experiment(
    experiment: Experiment,
    *,
    water_blank_correction: bool,
) -> Experiment:
    """Preserve raw observations or return a sample-only view when correction is off.

    Raw blank correction assumes the full assay-blank background scales with
    droplet volume. The caller pairs the intended assay material, geometry,
    preparation and cooling conditions; these are not inferred from names.
    """
    if not isinstance(water_blank_correction, bool):
        raise TypeError("water_blank_correction must be a bool")
    if not isinstance(experiment, Experiment):
        raise TypeError("experiment must be an Experiment")
    if water_blank_correction or not experiment.water_blank_map:
        return experiment
    measurements = {
        key: value
        for key, value in experiment.measurements.items()
        if key in experiment.water_blank_map
    }
    sample_ids = {value.sample_id for value in measurements.values()}
    return Experiment(
        counts=CountsTable(
            sample_rows(experiment.counts.to_dataframe(), experiment),
            history=experiment.counts.history
            + [
                {
                    "operation": "water_blank_correction",
                    "enabled": False,
                    "water_blank_model": "volume_scaled",
                    "water_blank_map": dict(experiment.water_blank_map),
                }
            ],
        ),
        samples={key: value for key, value in experiment.samples.items() if key in sample_ids},
        measurements=measurements,
        source=dict(experiment.source),
    )


def estimate_point(
    samples: pd.DataFrame,
    blanks: pd.DataFrame,
    experiment: Experiment,
    *,
    confidence_drop: float,
    method: str = "mle",
) -> tuple[float, float, float, bool]:
    """Estimate one concentration from explicitly selected physical observations.

    The caller handles temperature alignment and retains each row's measured
    temperature and observation identity. Selected temperatures need not be
    identical. This function uses the actual counts, dilution and well volume;
    it never interpolates counts or combines repeated cycles of a droplet set.

    With an active water-blank map, supply every assigned blank exactly once
    for each selected run/cycle. Runs have independent background concentrations,
    while their samples share one original-sample concentration for MLE. Average
    fits each sample with its own run's blanks before taking the arithmetic mean.
    Without a map, blanks must be empty and no background is fitted.
    """
    if not isinstance(experiment, Experiment):
        raise TypeError("experiment must be an Experiment")
    if method not in ("mle", "average"):
        raise ValueError("method must be 'mle' or 'average'")
    if not isinstance(samples, pd.DataFrame) or not isinstance(blanks, pd.DataFrame):
        raise TypeError("samples and blanks must be pandas DataFrames")
    if samples.empty:
        raise ValueError("Concentration fitting requires sample observations")
    required = {
        "measurement_id", "sample_id", "run_id", "cycle_id",
        "temperature_C", "n_frozen", "n_total",
    }
    for role, rows in (("Sample", samples), ("Blank", blanks)):
        if rows.empty:
            continue
        missing = required - set(rows)
        if missing:
            raise ValueError(f"{role} observations are missing columns: {sorted(missing)}")
        for name in ("measurement_id", "sample_id", "run_id", "cycle_id"):
            if any(not isinstance(value, str) or not value.strip() for value in rows[name]):
                raise ValueError(f"{role} {name} must contain nonempty string identifiers")
        if rows.measurement_id.duplicated().any():
            raise ValueError(
                f"{role} observations require one row per physical measurement; "
                "repeated temperatures or cycles cannot be counted as independent droplets"
            )
        if (rows.groupby("run_id").cycle_id.nunique() > 1).any():
            raise ValueError(f"{role} observations require one selected cycle per run")
        temperatures = pd.to_numeric(rows.temperature_C, errors="raise")
        if not np.isfinite(temperatures).all():
            raise ValueError(f"{role} observation temperatures must be finite")
        for row in rows.itertuples(index=False):
            if row.measurement_id not in experiment.measurements:
                raise ValueError(f"Unknown measurement {row.measurement_id!r}")
            metadata = experiment.measurements[str(row.measurement_id)]
            if (row.sample_id, row.run_id) != (metadata.sample_id, metadata.run_id):
                raise ValueError(
                    f"Observation identities disagree with metadata for {row.measurement_id!r}"
                )
    if samples.sample_id.nunique() != 1:
        raise ValueError("Concentration fitting requires one parent sample")

    sample_metadata = [experiment.measurements[key] for key in samples.measurement_id]
    estimate = average_concentration if method == "average" else fit_concentration
    common: dict[str, Any] = {
        "n_frozen": samples.n_frozen.to_numpy(),
        "n_total": samples.n_total.to_numpy(),
        "dilution": [item.dilution for item in sample_metadata],
        "well_volume_uL": [item.droplet_volume_uL for item in sample_metadata],
        "confidence_drop": confidence_drop,
    }
    if not experiment.water_blank_map:
        if not blanks.empty:
            raise ValueError("Blank observations require experiment.water_blank_map assignments")
        return estimate(**common)

    expected: set[tuple[str, str]] = set()
    for row in samples.itertuples(index=False):
        if row.measurement_id not in experiment.water_blank_map:
            raise ValueError(
                f"Measurement {row.measurement_id!r} is not an assigned sample measurement"
            )
        expected.update(
            (blank_id, str(row.cycle_id))
            for blank_id in experiment.water_blank_map[str(row.measurement_id)]
        )
    supplied = set() if blanks.empty else set(zip(blanks.measurement_id, blanks.cycle_id))
    if supplied != expected:
        missing_pairs, extra_pairs = sorted(expected - supplied), sorted(supplied - expected)
        raise ValueError(
            "Selected blank observations must exactly match assigned measurement/cycle pairs; "
            f"missing={missing_pairs}, extra={extra_pairs}"
        )
    blank_metadata = [experiment.measurements[key] for key in blanks.measurement_id]

    def groups(rows: pd.DataFrame) -> list[str]:
        return [json.dumps([row.run_id, row.cycle_id]) for row in rows.itertuples(index=False)]

    return estimate(
        **common,
        blank_frozen=blanks.n_frozen.to_numpy(),
        blank_total=blanks.n_total.to_numpy(),
        blank_volume_uL=[item.droplet_volume_uL for item in blank_metadata],
        sample_blank_group=groups(samples),
        blank_group=groups(blanks),
    )
