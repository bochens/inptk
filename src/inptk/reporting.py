"""Reporting limits from original sample freezing events, independent of fitting."""

from __future__ import annotations

import numpy as np


def freezing_intervals(frame, groups, ranges):
    """Find the observed freezing-temperature extent for each requested curve.

    A new maximum frozen count marks an event; an initially positive count marks
    the first observed event. Repeated count states and recovery after a falling
    count do not create new events. Blank inputs never determine these limits.
    Individual limits always use the original events. Combined limits intersect
    each original event interval with the selected calculation range. A cutoff
    never removes an event and thereby invents a different freezing interval.
    """
    events = {}
    keys = ["measurement_id", "run_id", "cycle_id"]
    for identity, rows in frame.groupby(keys, sort=False):
        if "time_s" in rows:
            rows = rows.sort_values("time_s", kind="stable")
        events[identity] = _event_temperatures(rows)
    intervals = {}
    for name, group in groups.items():
        bounds = []
        for member in group["members"]:
            identity = tuple(member[column] for column in keys)
            temperatures = events[identity]
            if temperatures.empty:
                continue
            cold, warm = float(temperatures.min()), float(temperatures.max())
            if len(group["members"]) > 1:
                limits = ranges.get(member["measurement_id"], {})
                if limits.get("min_C") is not None:
                    cold = max(cold, limits["min_C"])
                if limits.get("max_C") is not None:
                    warm = min(warm, limits["max_C"])
            if cold <= warm:
                bounds.append((cold, warm))
        intervals[name] = {
            "min_C": min(bound[0] for bound in bounds) if bounds else None,
            "max_C": max(bound[1] for bound in bounds) if bounds else None,
        }
    return intervals


def mark_reporting_intervals(frame, intervals):
    """Keep diagnostic rows; concentration and errors outside event bounds are NaN."""
    data = frame.copy()
    data["reporting_min_C"] = data.curve_id.map(lambda name: intervals[name]["min_C"]).astype(float)
    data["reporting_max_C"] = data.curve_id.map(lambda name: intervals[name]["max_C"]).astype(float)
    data["reporting_status"] = np.select(
        [
            data.reporting_min_C.isna(),
            data.temperature_C.gt(data.reporting_max_C),
            data.temperature_C.lt(data.reporting_min_C),
        ],
        ["no_freezing_events", "before_first_freeze", "after_last_freeze"],
        default="within_freezing_interval",
    )
    outside = data.reporting_status.ne("within_freezing_interval")
    data.loc[outside, ["concentration", "lower_error", "upper_error"]] = np.nan
    data.loc[outside, "at_zero_boundary"] = False
    data.loc[outside, "qc_flag"] = data.loc[outside, "qc_flag"].astype(int) | 1
    return data


def reportable_spectrum(spectrum):
    """Select reportable estimates; manually supplied spectra have no count limits."""
    if "reporting_status" not in spectrum.columns:
        return spectrum
    return spectrum.select(reporting_status="within_freezing_interval")


def _event_temperatures(rows):
    """Temperatures where the frozen count reaches a new maximum (freezing events)."""
    previous = rows.n_frozen.cummax().shift(fill_value=0)
    return rows.loc[rows.n_frozen.gt(previous), "temperature_C"]


def sample_range_details(rows, temperatures, limits):
    """Report an input's full and selected spans on the available calculation points.

    The full span is the calculation points inside the original first-to-last
    freezing interval. The selected span is the part of it inside the limits.
    """
    events = _event_temperatures(rows)
    values = np.asarray(temperatures, dtype=float)
    useful = values[(values >= events.min()) & (values <= events.max())]
    full = {"min_C": float(useful.min()), "max_C": float(useful.max())} if useful.size else None
    selected = useful[(useful >= (-np.inf if limits.get("min_C") is None else limits["min_C"]))
                      & (useful <= (np.inf if limits.get("max_C") is None else limits["max_C"]))]
    extent = ({"min_C": float(selected.min()), "max_C": float(selected.max())}
              if selected.size else None)
    return {"full_range_C": full, "selected_range_C": extent}
