"""Select observed count pairs on a temperature grid, before blank correction."""

from __future__ import annotations

from decimal import Decimal
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


def grid_points(
    frame, members, *, water_blank_map, ranges, step_C, method, window_C, start_C=None, end_C=None,
):
    """Select each sample and blank independently; never pool repeated counts.

    Source rows retain measured temperatures and observation IDs. A separate
    fit_temperature_C records where the selected state will enter the fit.
    Empty sample windows create gaps; missing required blank states are errors.
    """
    from .alignment import AlignedPoint, _in_range, _ordered_rows

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
        eligible = rows.loc[
            rows.temperature_C.map(lambda t, bounds=limits: _in_range(t, bounds))
        ].copy()
        # Preserve these checks across selection: max must not conceal invalid
        # raw trajectories from the whole-curve likelihood.
        eligible["source_total_is_fixed"] = eligible.n_total.nunique() <= 1
        eligible["source_frozen_is_cumulative"] = not eligible.n_frozen.diff().lt(0).any()
        streams[key] = (rows, eligible, limits)

    def select(key, target):
        rows, eligible, limits = streams[key]
        if rows.empty or not rows.temperature_C.min() <= target <= rows.temperature_C.max():
            return None
        if not _in_range(target, limits):
            return None
        if method == "window":
            candidates = eligible.loc[
                eligible.temperature_C.between(target - window_C / 2, target + window_C / 2)
            ]
        else:
            candidates = eligible.loc[eligible.temperature_C.ge(target)]
        if candidates.empty:
            return None
        if method == "max":
            fraction = candidates.n_frozen / candidates.n_total
            candidates = candidates.loc[fraction.eq(fraction.max())]
        elif method == "window":
            candidates = candidates.loc[candidates.n_frozen.eq(candidates.n_frozen.max())]
        chosen = candidates.iloc[-1].copy()  # latest observation wins ties
        chosen["fit_temperature_C"] = target
        chosen["temperature_selection"] = method
        return chosen

    temperatures = pd.concat([streams[key][0].temperature_C for key in keys])
    empty = frame.iloc[:0].copy()
    points = []
    targets = temperature_grid(temperatures, step_C=step_C, start_C=start_C, end_C=end_C)
    for order, target in enumerate(targets):
        samples, needed = [], set()
        for key in keys:
            row = select(key, target)
            if row is not None:
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
            alignment=method, point_order=order,
        ))
    return points
