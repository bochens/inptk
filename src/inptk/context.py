"""Validate analysis inputs without changing the original experiment."""

from __future__ import annotations

import pandas as pd

from .experiment import Experiment
from .tables import FrozenFractionTable


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
            "warm zero rows are not raw measurements; use original observed counts"
        )
    if view is not experiment:
        rows = sample_rows(fractions.to_dataframe(), experiment)
        if rows.empty:
            raise ValueError("No sample observations remain after excluding water-blank sets")
        fractions = FrozenFractionTable(rows, history=fractions.history)
    return fractions, view
