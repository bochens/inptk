"""Reporting limits from original sample freezing events, independent of fitting."""

from __future__ import annotations

import numpy as np


def freezing_intervals(frame, groups, ranges):
    """Find the observed freezing-temperature extent for each requested curve.

    A new maximum frozen count marks an event; an initially positive count marks
    the first observed event. Repeated count states and recovery after a falling
    count do not create new events. Blank inputs never determine these limits.
    Individual limits always use the original events. Combined limits use only
    events inside the selected input ranges.
    """
    events = {}
    keys = ["measurement_id", "run_id", "cycle_id"]
    for identity, rows in frame.groupby(keys, sort=False):
        if "time_s" in rows:
            rows = rows.sort_values("time_s", kind="stable")
        previous = rows.n_frozen.cummax().shift(fill_value=0)
        temperatures = rows.loc[rows.n_frozen.gt(previous), "temperature_C"]
        events[identity] = temperatures
    intervals = {}
    for name, group in groups.items():
        bounds = []
        for member in group["members"]:
            identity = tuple(member[column] for column in keys)
            temperatures = events[identity]
            if len(group["members"]) > 1:
                limits = ranges.get(member["measurement_id"], {})
                if limits.get("min_C") is not None:
                    temperatures = temperatures[temperatures.ge(limits["min_C"])]
                if limits.get("max_C") is not None:
                    temperatures = temperatures[temperatures.le(limits["max_C"])]
            if not temperatures.empty:
                bounds.append((float(temperatures.min()), float(temperatures.max())))
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
