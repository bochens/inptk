"""Select observed count pairs on a temperature grid, before blank correction."""

from __future__ import annotations

from decimal import Decimal
from functools import cached_property
from numbers import Real

import numpy as np
import pandas as pd


def validate_temperature_selection(step_C, method, window_C, start_C=None, end_C=None):
    """A window is its full width; no colder-observation tolerance is implicit."""
    if not isinstance(method, str) or method not in ("latest", "max", "window"):
        raise ValueError("temperature_method must be 'latest', 'max', or 'window'")
    for name, value in (("temperature_step_C", step_C), ("temperature_window_C", window_C)):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, Real)
            or not np.isfinite(value) or value <= 0
        ):
            raise ValueError(f"{name} must be finite and positive")
    if step_C is None and (method != "latest" or window_C is not None):
        raise ValueError("A temperature selection rule requires temperature_step_C")
    if method == "window" and window_C is None:
        raise ValueError("window requires temperature_window_C (full window width)")
    if method != "window" and window_C is not None:
        raise ValueError("temperature_window_C applies only to window")
    for name, value in (("temperature_start_C", start_C), ("temperature_end_C", end_C)):
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if step_C is None:
                raise ValueError(f"{name} requires temperature_step_C")
    if start_C is not None and end_C is not None and start_C < end_C:
        raise ValueError("temperature_start_C must be warmer than or equal to temperature_end_C")


def temperature_grid(temperatures, *, step_C, start_C=None, end_C=None):
    """Build a cooling grid anchored at the requested or observed warm endpoint.

    Both endpoints are included without rounding their temperatures. The final
    interval is shorter when the span is not an exact multiple of the spacing.
    Decimal arithmetic avoids moving a boundary through floating-point drift.
    """
    warm = Decimal(str(float(max(temperatures) if start_C is None else start_C)))
    cold = Decimal(str(float(min(temperatures) if end_C is None else end_C)))
    if warm < cold:
        raise ValueError("Grid start must be warmer than or equal to grid end")
    step = Decimal(str(float(step_C)))
    size = int((warm - cold) // step)
    targets = [float(warm - index * step) for index in range(size + 1)]
    if targets[-1] != float(cold):
        targets.append(float(cold))
    return targets


class CountSelector:
    """Index a stream once, selecting actual rows without interpolating counts."""

    def __init__(self, rows):
        self.rows = rows.copy()
        self.rows["source_total_is_fixed"] = rows.n_total.nunique() <= 1
        self.rows["source_frozen_is_cumulative"] = not rows.n_frozen.diff().lt(0).any()
        self.temperatures = rows.temperature_C.to_numpy()
        self.frozen = rows.n_frozen.to_numpy()
        self.fractions = (rows.n_frozen / rows.n_total).to_numpy()

    def select(self, target, *, method="latest", window_C=None, gridded=True):
        if method == "window":
            selected = (self.temperatures >= target - window_C / 2) & (
                self.temperatures <= target + window_C / 2
            )
        else:
            selected = self.temperatures >= target
        positions = np.flatnonzero(selected)
        if not positions.size:
            return None
        if method in ("max", "window"):
            scores = self.fractions[positions] if method == "max" else self.frozen[positions]
            positions = positions[scores == scores.max()]
        chosen = self.rows.iloc[positions[-1]].copy()  # latest observation wins ties
        if gridded:
            chosen["fit_temperature_C"] = target
            chosen["temperature_selection"] = method
        return chosen

    @cached_property
    def acquisitions(self):
        """Index exact images/timestamps only when native alignment needs them."""
        indices = {}
        for name in ("picture_id", "time_s"):
            if name not in self.rows:
                continue
            lookup = {}
            for position, (temperature, value) in enumerate(zip(self.temperatures, self.rows[name])):
                if pd.isna(value) or value == "":
                    continue
                key = (temperature, value)
                lookup[key] = -1 if key in lookup else position
            indices[name] = lookup
        return indices

    def matching_acquisition(self, sample):
        for name, lookup in self.acquisitions.items():
            value = sample.get(name)
            if pd.isna(value) or value == "":
                continue
            position = lookup.get((sample["temperature_C"], value), -1)
            if position >= 0:
                return position
        return None


def grid_points(
    frame, members, *, water_blank_map, ranges, step_C, method, window_C, start_C=None, end_C=None,
    sample_freezing_intervals_C=None, retain_full_range_constraints=False,
):
    """Select each sample and blank independently; never pool repeated counts.

    Source rows retain measured temperatures and observation IDs. A separate
    fit_temperature_C records where the selected state will enter the fit.
    Empty sample windows create gaps; missing required blank states are errors.
    """
    from .alignment import AlignedPoint, _average_sample_eligible, _in_range, _ordered_rows
    from .reporting import resolve_sample_range

    keys = [(str(m["measurement_id"]), str(m["run_id"]), str(m["cycle_id"])) for m in members]
    blank_keys = {
        (str(blank), run, cycle)
        for measurement, run, cycle in keys for blank in water_blank_map.get(measurement, [])
    }
    streams = {}
    for key in dict.fromkeys([*keys, *sorted(blank_keys)]):
        measurement, run, cycle = key
        rows = _ordered_rows(frame.loc[
            frame.measurement_id.eq(measurement) & frame.run_id.eq(run) & frame.cycle_id.eq(cycle)
        ])
        limits = ranges.get(measurement, {}) if key not in blank_keys else {}
        streams[key] = (rows, CountSelector(rows), limits,
                        (rows.temperature_C.min(), rows.temperature_C.max()))

    temperatures = pd.concat([streams[key][0].temperature_C for key in keys])
    targets = temperature_grid(temperatures, step_C=step_C, start_C=start_C, end_C=end_C)
    selections, resolved, range_details = {}, {}, {}
    for key, (rows, selector, limits, (cold, warm)) in streams.items():
        # Select from the complete observed stream once. Limits only select
        # target states; they cannot change which acquisition supplies a state.
        choices = {t: selector.select(t, method=method, window_C=window_C)
                   for t in targets if cold <= t <= warm}
        if key not in blank_keys:
            requested = limits
            limits, details = resolve_sample_range(
                rows, [t for t, row in choices.items() if row is not None], limits
            )
            if not retain_full_range_constraints:
                limits = requested
            range_details[key[0]] = {
                "run_id": key[1], "cycle_id": key[2], **details, "calculation_limits_C": limits,
            }
        if any(value is not None for value in limits.values()):
            # Validate the retained observation span without changing selection.
            # Source rows just outside a target boundary may supply its counts.
            source_ids = [row["observation_id"] for t, row in choices.items()
                          if row is not None and _in_range(t, limits)]
            covered = rows.temperature_C.between(
                -np.inf if limits.get("min_C") is None else limits["min_C"],
                np.inf if limits.get("max_C") is None else limits["max_C"],
            ) | rows.observation_id.isin(source_ids)
            observed = rows.loc[covered]
            fixed = observed.n_total.nunique() <= 1
            cumulative = not observed.n_frozen.diff().lt(0).any()
            for t, row in choices.items():
                if row is not None and _in_range(t, limits):
                    row["source_total_is_fixed"] = fixed
                    row["source_frozen_is_cumulative"] = cumulative
        selections[key], resolved[key] = choices, limits

    def select(key, target):
        return selections[key].get(target) if _in_range(target, resolved[key]) else None

    empty = frame.iloc[:0].copy()
    points = []
    for order, target in enumerate(targets):
        samples, needed = [], set()
        for key in keys:
            row = select(key, target)
            if row is not None and _average_sample_eligible(
                row, target, sample_freezing_intervals_C
            ):
                samples.append(row)
                needed.update((str(b), key[1], key[2]) for b in water_blank_map.get(key[0], []))
        blanks = []
        for key in sorted(needed):
            row = select(key, target)
            if row is None:
                raise ValueError(
                    f"Blank {key[0]!r} has no eligible {method} observation at "
                    f"{target:g} C in run {key[1]!r}, cycle {key[2]!r}; "
                    "no blank state is invented or carried across an empty window"
                )
            blanks.append(row)
        points.append(AlignedPoint(
            point_id=f"point:{order}", temperature_C=target,
            samples=pd.DataFrame(samples).reset_index(drop=True) if samples else empty.copy(),
            blanks=pd.DataFrame(blanks).reset_index(drop=True) if blanks else empty.copy(),
            alignment=method, point_order=order, range_details=range_details,
        ))
    return points
