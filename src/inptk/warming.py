"""Optionally leave out warming observations before any analysis."""

from __future__ import annotations

from numbers import Real

import numpy as np

from .experiment import Experiment
from .tables import CountsTable

DEFAULT_WARMING_HYSTERESIS_C = 0.2
_GROUP_KEYS = ["measurement_id", "run_id", "cycle_id"]


def validate_warming_settings(drop_warming, warming_hysteresis_C):
    if not isinstance(drop_warming, bool):
        raise TypeError("drop_warming must be a bool (True or False)")
    if (
        isinstance(warming_hysteresis_C, bool)
        or not isinstance(warming_hysteresis_C, Real)
        or not np.isfinite(warming_hysteresis_C)
        or warming_hysteresis_C < 0
    ):
        raise ValueError("warming_hysteresis_C must be finite and nonnegative")


def warming_settings(drop_warming, warming_hysteresis_C):
    """Settings entries recorded only when warming rows are left out."""
    if not drop_warming:
        return {}
    return {"drop_warming": True, "warming_hysteresis_C": float(warming_hysteresis_C)}


def drop_warming(
    source: Experiment | CountsTable,
    *,
    warming_hysteresis_C: float = DEFAULT_WARMING_HYSTERESIS_C,
) -> Experiment | CountsTable:
    """Leave out observations made while a cycle warms back up.

    Each measurement, run and cycle is read in acquisition order (time_s, then
    input order). Once the cycle has gone below 0 C, an observation more than
    warming_hysteresis_C warmer than the coldest temperature reached so far in
    that cycle is left out. Nothing is left out before the first sub-zero
    observation, so drift before cooling is kept. Frozen counts are cumulative:
    the next kept observation of the cycle receives the largest frozen count
    of the observations left out before it. Freezes after the last kept
    observation cannot be carried and produce a warning.

    Returns the same kind of input with the remaining rows in their original
    order. The left-out rows are recorded in history with the reason "warming".
    """
    validate_warming_settings(True, warming_hysteresis_C)
    counts = source.counts if isinstance(source, Experiment) else source
    if not isinstance(counts, CountsTable):
        raise TypeError("source must be an Experiment or CountsTable")
    hysteresis = float(warming_hysteresis_C)
    frame = counts.to_dataframe()
    keep = np.ones(len(frame), dtype=bool)
    n_frozen = frame.n_frozen.to_numpy(copy=True)
    dropped, omitted, carried, warnings = [], [], [], []
    for (measurement, run, cycle), rows in frame.groupby(_GROUP_KEYS, sort=False):
        if "time_s" in rows:
            rows = rows.sort_values("time_s", kind="stable")
        coldest = None
        pending = None
        last_kept = None
        for index, row in zip(rows.index, rows.itertuples(index=False)):
            temperature = float(row.temperature_C)
            if coldest is not None and coldest < 0 and temperature > coldest + hysteresis:
                keep[index] = False
                pending = max(pending or 0, int(row.n_frozen))
                record = {
                    "measurement_id": measurement, "sample_id": str(row.sample_id),
                    "run_id": run, "cycle_id": cycle,
                    "observation_id": str(row.observation_id),
                    "temperature_C": temperature,
                    "n_frozen": int(row.n_frozen), "n_total": int(row.n_total),
                }
                if "time_s" in rows:
                    record["time_s"] = float(row.time_s)
                omitted.append({**record, "reason": "warming"})
                continue
            coldest = temperature if coldest is None else min(coldest, temperature)
            if pending is not None and pending > n_frozen[index]:
                carried.append({
                    "measurement_id": measurement, "run_id": run, "cycle_id": cycle,
                    "observation_id": str(row.observation_id),
                    "n_frozen_observed": int(n_frozen[index]), "n_frozen_used": pending,
                })
                n_frozen[index] = pending
            pending = None
            last_kept = index
        count = int((~keep[rows.index]).sum())
        if not count:
            continue
        dropped.append({
            "measurement_id": measurement, "sample_id": str(rows.sample_id.iloc[0]),
            "run_id": run, "cycle_id": cycle, "rows_dropped": count,
        })
        if pending is not None and pending > n_frozen[last_kept]:
            warnings.append(
                f"{measurement}: {pending - int(n_frozen[last_kept])} freeze event(s) in run "
                f"{run} cycle {cycle} fall on warming rows after its last kept row "
                "and are not counted"
            )
    if dropped:
        warnings.insert(0, (
            f"drop_warming left out {sum(item['rows_dropped'] for item in dropped)} warming "
            f"rows (hysteresis {hysteresis:g} C): "
            + "; ".join(
                f"{item['measurement_id']} run {item['run_id']} cycle {item['cycle_id']}: "
                f"{item['rows_dropped']}" for item in dropped
            )
        ))
    frame["n_frozen"] = n_frozen
    table = type(counts)(
        frame.loc[keep],
        history=counts.history + [{
            "operation": "drop_warming",
            "warming_hysteresis_C": hysteresis,
            "rule": "after a cycle goes below 0 C, leave out rows warmer than its "
            "coldest temperature so far plus warming_hysteresis_C",
            "observation_order": "time_s_then_input_order",
            "rows_dropped": dropped,
            "omitted_observations": omitted,
            "carried_frozen_counts": carried,
            "warnings": warnings,
        }],
    )
    if not isinstance(source, Experiment):
        return table
    return Experiment(
        table,
        source.samples,
        source.measurements,
        source=source.source,
        water_blank_map=source.water_blank_map,
    )
