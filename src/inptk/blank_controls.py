"""One assigned water control per run/cycle, with explicit observation limits."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .alignment import _ordered_rows, align_observations
from .methods import validate_water_blank_range
from .temperature_selection import CountSelector


@dataclass(frozen=True)
class BlankState:
    """Which real controls supply a point, or which backgrounds are fixed at zero."""

    expected_pairs: tuple[tuple[str, str], ...] = ()
    zero_groups: tuple[str, ...] = ()
    unavailable: bool = False


class BlankControls:
    """Resolve controls once from original observations, independently of sample ranges.

    All assigned wells in a run/cycle share one background. The optional onset
    fixes that background at zero before any assigned blank first freezes. A
    manual range filters observations only: latest selection can use a retained
    warmer observation at a colder target; it cannot invent a warmer observation.
    Whole-curve MLE receives retained trajectories, never those carried states.
    """

    def __init__(self, experiment, members, *, raw_counts,
                 after_first_freeze=False, temperature_range_C=None):
        self.experiment = experiment
        self.after_first_freeze = after_first_freeze
        self.window_C = None
        self.limits = validate_water_blank_range(temperature_range_C)
        self.active = bool(experiment.water_blank_map) and (
            after_first_freeze or any(v is not None for v in self.limits.values())
        )
        self.groups, self.streams, self.supports = {}, {}, {}
        if not self.active:
            return
        assigned = {}
        for member in members:
            key = (member["run_id"], member["cycle_id"])
            assigned.setdefault(key, set()).update(
                experiment.water_blank_map.get(member["measurement_id"], [])
            )
        raw = raw_counts
        for (run, cycle), names in assigned.items():
            if not names:
                continue
            first = []
            originals = {}
            for name in sorted(names):
                rows = _ordered_rows(raw.loc[
                    raw.measurement_id.eq(name) & raw.run_id.eq(run) & raw.cycle_id.eq(cycle)
                ])
                if rows.empty:
                    raise ValueError(
                        f"Blank {name!r} has no observations in run {run!r}, cycle {cycle!r}"
                    )
                originals[name] = rows
                self.supports[(name, run, cycle)] = (
                    float(rows.temperature_C.min()), float(rows.temperature_C.max())
                )
                frozen = rows.loc[rows.n_frozen.gt(0)]
                if not frozen.empty:
                    first.append(float(frozen.temperature_C.iloc[0]))
            onset = max(first) if first else None
            limits = dict(self.limits)
            if after_first_freeze and onset is not None:
                warm = limits.get("max_C")
                limits["max_C"] = onset if warm is None else min(warm, onset)
            zero = after_first_freeze and onset is None
            group = {
                "run_id": run, "cycle_id": cycle, "measurement_ids": sorted(names),
                "first_freeze_temperature_C": onset,
                "effective_temperature_range_C": limits,
                "fixed_zero_throughout": zero,
            }
            self.groups[(run, cycle)] = group
            for name, rows in originals.items():
                cold, warm = limits.get("min_C"), limits.get("max_C")
                eligible = rows.loc[rows.temperature_C.between(
                    -np.inf if cold is None else cold, np.inf if warm is None else warm
                )]
                self.streams[(name, run, cycle)] = CountSelector(eligible)
            if not zero and all(
                self.streams[(name, run, cycle)].rows.empty for name in names
            ):
                raise ValueError(
                    f"Water-blank range retains no observations in run {run!r}, cycle {cycle!r}"
                )

    @property
    def details(self):
        return list(self.groups.values())

    def is_zero(self, key, temperature):
        group = self.groups[key]
        return self.after_first_freeze and (
            group["fixed_zero_throughout"] or temperature > group["first_freeze_temperature_C"]
        )

    def native_coverage(self, member, temperatures):
        """Check native latest-warmer availability without building aligned tables."""
        key = (member["run_id"], member["cycle_id"])
        values = np.asarray(temperatures)
        group = self.groups[key]
        if group["fixed_zero_throughout"]:
            return np.ones(len(values), dtype=bool)
        available = np.zeros(len(values), dtype=bool)
        for name in group["measurement_ids"]:
            selector = self.streams[(name, *key)]
            if selector.rows.empty:
                continue
            cold, warm = self.supports[(name, *key)]
            available |= (values >= cold) & (values <= min(warm, selector.temperatures.max()))
        if self.after_first_freeze:
            available |= values > group["first_freeze_temperature_C"]
        return available

    def align(self, frame, members, **settings):
        if not self.active:
            return align_observations(
                frame, members, water_blank_map=self.experiment.water_blank_map, **settings
            )
        points = align_observations(frame, members, water_blank_map={}, **settings)
        gridded = settings.get("temperature_step_C") is not None
        method = settings.get("temperature_method", "latest") if gridded else "latest"
        window = settings.get("temperature_window_C")
        self.window_C = window
        return [self._attach(point, method=method, window=window, gridded=gridded)
                for point in points]

    def _attach(self, point, *, method, window, gridded):
        blanks, zeros, unavailable = [], [], False
        samples_by_group = {}
        for sample in point.sample_records:
            samples_by_group.setdefault((sample["run_id"], sample["cycle_id"]), sample)
        for key, sample in samples_by_group.items():
            if key not in self.groups:
                continue
            if self.is_zero(key, point.temperature_C):
                zeros.append(json.dumps(list(key)))
                continue
            selected = []
            for name in self.groups[key]["measurement_ids"]:
                selector = self.streams[(name, *key)]
                cold, warm = self.supports[(name, *key)]
                if not cold <= point.temperature_C <= warm:
                    continue
                match = None if gridded else selector.matching_acquisition(sample)
                row = (selector.rows.iloc[match].copy() if match is not None else
                       selector.select(point.temperature_C, method=method,
                                       window_C=window, gridded=gridded))
                if row is not None:
                    selected.append(row)
            if not selected:
                unavailable = True
            blanks.extend(selected)
        frame = (pd.DataFrame(blanks).reset_index(drop=True) if blanks
                 else point.blanks.iloc[:0].copy())
        state = BlankState(
            expected_pairs=tuple(sorted((r.measurement_id, r.cycle_id)
                                        for r in frame.itertuples(index=False))),
            zero_groups=tuple(sorted(zeros)), unavailable=unavailable,
        )
        return replace(point, blanks=frame, blank_state=state)

    def fit_rows(self, points):
        """Retained real control histories, selected on the analysis grid when present."""
        if not points:
            return []
        gridded = any("fit_temperature_C" in p.samples for p in points)
        needed = {
            (sample["run_id"], sample["cycle_id"])
            for point in points for sample in point.sample_records
            if not self.is_zero((sample["run_id"], sample["cycle_id"]), point.temperature_C)
        }
        rows = []
        for (name, run, cycle), selector in self.streams.items():
            if (run, cycle) not in needed or selector.rows.empty:
                continue
            if not gridded:
                rows.append(selector.rows)
                continue
            selected = []
            cold, warm = selector.temperatures.min(), selector.temperatures.max()
            for point in points:
                if not cold <= point.temperature_C <= warm:
                    continue
                row = selector.select(
                    point.temperature_C, method=point.alignment,
                    window_C=self.window_C,
                )
                if row is not None:
                    selected.append(row)
            if selected:
                rows.append(pd.DataFrame(selected).reset_index(drop=True))
        present = {(row.run_id.iloc[0], row.cycle_id.iloc[0]) for row in rows}
        for key in needed:
            if key not in present:
                raise ValueError(
                    f"Water-blank range has no eligible grid observation in run {key[0]!r}, "
                    f"cycle {key[1]!r}; adjust the blank range or temperature grid"
                )
        return rows
