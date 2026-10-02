"""Raw sample/blank calculations with explicit physical blank assignments.

Counts in this path have not been background-subtracted. Pairing is exact by
run, cycle and selected temperature; blank observations are never fabricated.
"""

from __future__ import annotations

import json
from typing import cast

import numpy as np
import pandas as pd

from ._engine.water_blank_math import joint_water_blank_mle
from .experiment import Experiment
from .methods import MLE
from .tables import CountsTable, CumulativeSpectrumTable, FrozenFractionTable


def sample_rows(frame: pd.DataFrame, experiment: Experiment) -> pd.DataFrame:
    """Exclude physical water blanks from the sample analysis groups."""
    return frame.loc[
        ~frame.measurement_id.isin(
            {key for ids in experiment.water_blank_map.values() for key in ids}
        )
    ].copy()


def analysis_experiment(experiment: Experiment, *, water_blank_correction: bool) -> Experiment:
    """Return a sample-only view when correction is disabled, retaining the original."""
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


def fit_raw_rows(rows, all_rows, experiment, *, confidence_drop):
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
    return joint_water_blank_mle(
        rows.n_frozen.to_numpy(),
        rows.n_total.to_numpy(),
        [item.dilution for item in metadata],
        [item.droplet_volume_uL for item in metadata],
        [float(row["n_frozen"]) for row in blanks],
        [float(row["n_total"]) for row in blanks],
        blank_volume_uL=[experiment.measurements[key].droplet_volume_uL for key in blank_ids],
        confidence_drop=confidence_drop,
    )


def corrected_frame(
    fractions: FrozenFractionTable, experiment: Experiment, *, z: float
) -> pd.DataFrame:
    """Fit each raw dilution and its blank together, including blank uncertainty."""
    source = fractions.to_dataframe()
    frames = []
    for measurement_id, rows in sample_rows(source, experiment).groupby(
        "measurement_id", sort=False
    ):
        measurement_id = str(measurement_id)
        metadata = experiment.measurements[measurement_id]
        blank_ids = sorted(experiment.water_blank_map[measurement_id])
        blank = pooled_blank(rows, source, blank_ids)
        fits = [
            fit_raw_rows(rows.iloc[[index]], source, experiment, confidence_drop=z**2 / 2)
            for index in range(len(rows))
        ]
        out = rows.copy()
        out["concentration"] = [fit[0] for fit in fits]
        out["lower_error"] = [fit[1] for fit in fits]
        out["upper_error"] = [fit[2] for fit in fits]
        out["qc_flag"] = [0 if fit[3] else 1 for fit in fits]
        out["dilution_fold"] = metadata.dilution
        out["water_blank_ids"] = json.dumps(blank_ids)
        out["blank_n_frozen"] = blank.n_frozen.to_numpy()
        out["blank_n_total"] = blank.n_total.to_numpy()
        out["unit"] = "INP_per_mL_suspension"
        out["basis"] = "suspension"
        out["correction_state"] = "water_blank_corrected"
        out["uncertainty_method"] = "joint_sample_water_blank_profile_likelihood"
        out["at_zero_boundary"] = out.concentration.eq(0)
        frames.append(out)
    if not frames:
        raise ValueError("Raw water-blank input contains no sample measurement observations")
    return pd.concat(frames, ignore_index=True)


def cumulative_with_water_blank(
    fractions: FrozenFractionTable, experiment: Experiment, *, z: float
) -> CumulativeSpectrumTable:
    frame = corrected_frame(fractions, experiment, z=z)
    return CumulativeSpectrumTable(
        frame,
        history=fractions.history
        + [
            {
                "operation": "cumulative_spectrum_with_water_blank",
                "water_blank_map": dict(experiment.water_blank_map),
                "water_blank_correction": True,
                "z": float(z),
                "matching": "exact_run_cycle_temperature",
                "uncertainty_assumption": (
                    "independent raw droplets; common Poisson water background per liquid volume; "
                    "each supplied volume enters separately; pointwise profile-likelihood intervals"
                ),
            }
        ],
    )


def validate_raw_mle(method: MLE) -> None:
    if method.mask_mode == "rebase_counts":
        raise ValueError(
            "Raw water-blank MLE requires unmodified counts; rebase_counts is unsupported"
        )
    if any(
        getattr(method, name) is not None
        for name in (
            "likelihood_weights",
            "action_counts",
            "action_weight_lambda",
            "action_weight_half_life",
        )
    ):
        raise ValueError(
            "Raw water-blank MLE does not support weighted or action-reweighted counts"
        )


def mle_group(
    group: pd.DataFrame,
    all_fractions: pd.DataFrame,
    experiment: Experiment,
    method: MLE,
    *,
    confidence_drop: float,
) -> pd.DataFrame:
    validate_raw_mle(method)
    groups = {
        tuple(sorted(experiment.water_blank_map[str(key)])) for key in group.measurement_id.unique()
    }
    if len(groups) != 1:
        raise ValueError("Raw water-blank MLE requires a shared blank group per sample/run/cycle")
    blank_ids = list(next(iter(groups)))
    records = []
    limits = method.temperature_eligibility_C or {}
    for temperature, rows in group.groupby("temperature_C", sort=False):
        eligible = rows.loc[
            [
                float(cast(float, temperature)) <= limits.get(str(key), np.inf)
                for key in rows.measurement_id
            ]
        ]
        row = rows.iloc[0].to_dict()
        row.pop("measurement_id", None)
        # Frozen counts belong to individual measurements, never to this fitted curve.
        for column in ("n_frozen", "n_total", "fraction_frozen", "time_s"):
            row.pop(column, None)
        row.update(
            concentration=np.nan,
            lower_error=np.nan,
            upper_error=np.nan,
            unit="INP_per_mL_suspension",
            basis="suspension",
            qc_flag=1,
            water_blank_ids=json.dumps(blank_ids),
            correction_state="water_blank_corrected",
            contributing_measurement_ids=json.dumps(eligible.measurement_id.astype(str).tolist()),
        )
        if not eligible.empty:
            blank = pooled_blank(eligible.iloc[:1], all_fractions, blank_ids).iloc[0]
            value, lower, upper, finite = fit_raw_rows(
                eligible, all_fractions, experiment, confidence_drop=confidence_drop
            )
            row.update(
                concentration=value,
                lower_error=lower,
                upper_error=upper,
                qc_flag=0 if finite else 1,
                blank_n_frozen=float(blank["n_frozen"]),
                blank_n_total=float(blank["n_total"]),
            )
        records.append(row)
    return pd.DataFrame.from_records(records)
