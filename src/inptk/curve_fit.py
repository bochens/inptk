"""Prepare original count trajectories for a joint monotone freezing-curve fit."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ._engine.curve_likelihood import CurveLikelihood, FreezingSeries
from .resampling import _grid


def _trajectory(rows, *, name):
    gridded = "fit_temperature_C" in rows
    rows = rows.drop_duplicates(
        ["observation_id", "fit_temperature_C"] if gridded else ["observation_id"]
    )
    if gridded:
        rows = rows.sort_values("fit_temperature_C", ascending=False, kind="stable").copy()
        rows["temperature_C"] = rows.fit_temperature_C
    elif "time_s" in rows:
        rows = rows.sort_values("time_s", kind="stable")
    rows = rows.reset_index(drop=True)
    if rows.n_total.nunique() != 1 or (
        gridded and not rows.source_total_is_fixed.all()
    ):
        raise ValueError(
            f"Joint MLE requires a fixed set of wells: {name!r} has changing total counts. "
            "Supply raw counts or select a valid temperature range; corrected totals "
            "cannot be treated as independent freezing events."
        )
    if rows.n_frozen.diff().lt(0).any() or (
        gridded and not rows.source_frozen_is_cumulative.all()
    ):
        raise ValueError(
            f"Joint MLE requires cumulative first-freezing counts: {name!r} has a decrease. "
            "Review the raw freezing observations or select a valid temperature range."
        )
    # Use the established latest-warmer rule for holds and temperature reversals.
    # This is an analysis view only; the Experiment keeps every original row.
    if gridded:
        selected = rows
    else:
        observed = rows.temperature_C.to_numpy(dtype=float)
        targets = np.unique(observed)[::-1]
        order = np.argsort(-observed, kind="stable")
        last = np.searchsorted(-observed[order], -targets, side="right") - 1
        positions = np.maximum.accumulate(order)[last]
        selected = rows.iloc[positions].copy()
        selected["temperature_C"] = targets
    # Zero-event runs telescope exactly in the event likelihood. Keep both
    # edges of every observed increase and the observation-period endpoints.
    changed = np.flatnonzero(selected.n_frozen.diff().fillna(0).gt(0))
    keep = np.unique(np.r_[0, len(selected) - 1, changed, changed - 1])
    return selected.iloc[keep], len(rows)


def fit_curve(points, experiment, *, z, fit_step_C=None, output_step_C=None):
    """Fit original streams, then evaluate estimates and bounds at requested spacing."""
    if output_step_C is not None and (
        isinstance(output_step_C, bool) or not np.isfinite(output_step_C) or output_step_C <= 0
    ):
        raise ValueError("output_step_C must be finite and positive")
    sample_pieces = [point.samples for point in points if not point.samples.empty]
    if not sample_pieces:
        return {}, {"physical_droplets": 0, "observation_count": 0}
    blank_pieces = [point.blanks for point in points if not point.blanks.empty]
    streams, identities = [], []
    observation_count = 0
    targets = sorted({p.temperature_C for p in points if not p.samples.empty}, reverse=True)
    for role, pieces in (("sample", sample_pieces), ("blank", blank_pieces)):
        if not pieces:
            continue
        frame = pd.concat(pieces, ignore_index=True)
        for (measurement, run, cycle), rows in frame.groupby(
            ["measurement_id", "run_id", "cycle_id"], sort=False
        ):
            selected, count = _trajectory(rows, name=measurement)
            observation_count += count
            metadata = experiment.measurements[measurement]
            volume = metadata.droplet_volume_uL / 1000
            background = json.dumps([run, cycle]) if experiment.water_blank_map else ""
            streams.append(FreezingSeries(
                temperatures=selected.temperature_C.to_numpy(dtype=float),
                frozen=selected.n_frozen.to_numpy(dtype=int),
                total=int(selected.n_total.iloc[0]),
                sample_exposure=volume / metadata.dilution if role == "sample" else 0.0,
                blank_exposure=volume if background else 0.0,
                background=background,
            ))
            identities.append({
                "measurement_id": measurement, "run_id": run, "cycle_id": cycle,
                "role": role, "n_total": int(selected.n_total.iloc[0]),
                "observation_ids": selected.observation_id.tolist(),
                "fit_temperatures_C": selected.temperature_C.tolist(),
            })
    # The engine includes both original targets and each stream's own observed
    # transitions, so blank intervals are not snapped onto sample temperatures.
    model = CurveLikelihood(streams, targets, fit_step_C=fit_step_C)
    output_temperatures = (
        targets if output_step_C is None else _grid(pd.Series(targets), output_step_C)
    )
    estimates = {
        temperature: model.estimate(temperature, z**2 / 2) for temperature in output_temperatures
    }
    details = {
        "physical_droplets": model.physical_droplets,
        "observation_count": observation_count,
        "likelihood": "first_freezing_intervals_and_unfrozen_survivors",
        "monotonicity": "nonnegative_sample_and_background_increments_during_cooling",
        "temperature_view": (
            f"selected grid: {sample_pieces[0].temperature_selection.iloc[0]}"
            if "fit_temperature_C" in sample_pieces[0]
            else "latest_warmer; original rows remain in experiment"
        ),
        "temperature_order": "warm_to_cold",
        "uncertainty": "pointwise_profile_bounds_from_joint_curve_likelihood",
        "confidence_drop": z**2 / 2,
        "uncertainty_coverage": "nominal approximate; not a simultaneous confidence band",
        "unbounded_tail": "no finite point estimate reported",
        "fit_step_C": fit_step_C,
        "output_step_C": output_step_C,
        "curve_shape": ("native_temperature_steps" if fit_step_C is None
                        else "piecewise_linear_cumulative_concentration"),
        "sources": identities,
    }
    return estimates, details
