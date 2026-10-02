"""Raw sample/blank calculations with explicit physical blank assignments.

Counts in this path have not been background-subtracted. Pairing is exact by
run, cycle and selected temperature; blank observations are never fabricated.
"""

from __future__ import annotations

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


def _paired_blank(rows: pd.DataFrame, all_rows: pd.DataFrame, blank_id: str) -> pd.DataFrame:
    keys = ["run_id", "cycle_id", "temperature_C"]
    blank = all_rows.loc[all_rows.measurement_id == blank_id].set_index(keys)
    requested = pd.MultiIndex.from_frame(rows[keys])
    if not requested.isin(blank.index).all():
        missing = requested[~requested.isin(blank.index)].tolist()
        raise ValueError(
            f"Water blank {blank_id!r} lacks matching run/cycle/temperature observations: "
            f"{missing[:5]}; no blank extrapolation is performed"
        )
    return blank.loc[requested].reset_index()


def pooled_blank(rows: pd.DataFrame, all_rows: pd.DataFrame, blank_ids: list[str]) -> pd.DataFrame:
    """Summarize observed blank counts; fitting retains separate volumes and terms."""
    pooled = rows[["run_id", "cycle_id", "temperature_C"]].reset_index(drop=True).copy()
    pooled["n_frozen"] = 0.0
    pooled["n_total"] = 0.0
    for blank_id in blank_ids:
        blank = _paired_blank(rows, all_rows, blank_id)
        pooled["n_frozen"] += blank.n_frozen.to_numpy()
        pooled["n_total"] += blank.n_total.to_numpy()
    return pooled


def fit_raw_rows(rows, all_rows, experiment, *, confidence_drop, method="mle"):
    """Fit selected sample rows and each physical blank once, using their own volumes."""
    if rows.empty:
        raise ValueError("Joint blank fitting requires sample observations")
    groups = {tuple(sorted(experiment.water_blank_map[str(key)])) for key in rows.measurement_id}
    if len(groups) != 1:
        raise ValueError("Joint water-blank fitting requires one shared blank group")
    if len(rows[["run_id", "sample_id", "cycle_id", "temperature_C"]].drop_duplicates()) != 1:
        raise ValueError("Joint fitting requires one sample/run/cycle/temperature")
    blank_ids = list(next(iter(groups)))
    blanks = [_paired_blank(rows.iloc[:1], all_rows, key).iloc[0] for key in blank_ids]
    metadata = [experiment.measurements[str(key)] for key in rows.measurement_id]
    estimate = average_concentration if method == "average" else fit_concentration
    return estimate(
        rows.n_frozen.to_numpy(),
        rows.n_total.to_numpy(),
        [item.dilution for item in metadata],
        [item.droplet_volume_uL for item in metadata],
        blank_frozen=[float(row["n_frozen"]) for row in blanks],
        blank_total=[float(row["n_total"]) for row in blanks],
        blank_volume_uL=[experiment.measurements[key].droplet_volume_uL for key in blank_ids],
        confidence_drop=confidence_drop,
    )
