"""Choose nonoverlapping intervals using the actual Average calculation states."""

from __future__ import annotations

import numpy as np

from .reporting import freezing_intervals
from .water_blank import estimate_point


class RangePlanner:
    """Prefer the longest earlier interval; shorten it only to enable a handoff.

    Every trial selects the same aligned count states as analysis. Limits choose
    calculation targets; they never change the source acquisition at a target.
    """

    def __init__(self, frame, members, experiment, bases, grid, z, estimates, blank_controls,
                 calculation_rows):
        self.frame = frame
        self.members = members
        self.experiment = experiment
        self.grid = grid
        self.z = z
        self.estimates = estimates
        self.blank_controls = blank_controls
        self.calculation_rows = calculation_rows
        self.trials = {}
        self.outside = float(frame.temperature_C.max()) + 1
        self.freezing_intervals = freezing_intervals(
            frame, {m["measurement_id"]: {"members": [m]} for m in members}, {}
        )
        self.bases = {}
        for name, blocks in bases.items():
            bounds = self.freezing_intervals[name]
            self.bases[name] = []
            if bounds["min_C"] is None:
                continue
            for block in blocks:
                cold = max(block["min_C"], bounds["min_C"])
                warm = min(block["max_C"], bounds["max_C"])
                if cold <= warm:
                    temperatures = calculation_rows[name].temperature_C
                    available = temperatures[temperatures.between(cold, warm)]
                    if not available.empty:
                        self.bases[name].append({"min_C": float(available.min()),
                                                 "max_C": float(available.max())})

    def points(self, ranges):
        disabled = {"min_C": self.outside, "max_C": self.outside}
        return self.blank_controls.align(
            self.frame, self.members,
            temperature_ranges_C={m["measurement_id"]: ranges.get(m["measurement_id"], disabled)
                                  for m in self.members},
            sample_freezing_intervals_C=self.freezing_intervals,
            **self.grid,
        )

    def profile(self, ranges):
        records = []
        for point in self.points(ranges):
            if point.samples.empty:
                continue
            key = point.count_key()
            if key not in self.estimates:
                self.estimates[key] = estimate_point(
                    point.samples, point.blanks, self.experiment, method="average",
                    z=self.z,
                    blank_state=point.blank_state,
                )
            records.append((float(point.temperature_C), self.estimates[key][0]))
        return records

    def interval(self, name, previous=None, cold_limit=None):
        for base in self.bases[name]:
            selected = self._interval(name, base, previous, cold_limit)
            if selected is not None:
                return selected
            # A block containing actual calculation points must not be replaced
            # by a colder block for the first input. Blocks between grid targets
            # have no calculation points and cannot determine a grid range.
            if previous is None and self.profile({name: base}):
                break
        return None

    def _interval(self, name, base, previous, cold_limit):
        cold, warm = base["min_C"], base["max_C"]
        if cold_limit is not None:
            cold = max(cold, cold_limit)
        if previous is not None:
            # Inclusive ranges must not meet at the same temperature. Use an
            # actual calculation temperature below the previous cold boundary.
            rows = self.calculation_rows[name]
            candidates = rows.loc[
                rows.temperature_C.lt(previous["limits"]["min_C"]), "temperature_C"
            ]
            if candidates.empty:
                return None
            warm = min(warm, float(candidates.max()))
        reference = previous["profile"][-1][1] if previous is not None else 0.0
        bounds = self.freezing_intervals[name]
        warm_reason = ("first_freezing_event" if warm == bounds["max_C"]
                       else "count_or_coverage_limit")
        cold_reason = ("last_freezing_event" if cold == bounds["min_C"]
                       else "count_or_coverage_limit")
        while cold <= warm:
            key = (name, cold, warm)
            if key not in self.trials:
                self.trials[key] = self.profile(
                    {name: {"min_C": cold, "max_C": warm}}
                )
            profile = self.trials[key]
            if not profile:
                return None
            if profile[0][1] < reference or not np.isfinite(profile[0][1]):
                # Advance the warm limit until the new input can continue the
                # previous one, recalculating the selected counts each time.
                colder = [t for t, _ in profile if t < profile[0][0]]
                if not colder:
                    return None
                warm = min(warm, max(colder))
                warm_reason = ("negative_corrected_concentration" if profile[0][1] < 0
                               else "below_previous_concentration")
                continue
            bad = next(
                (i for i, (_, value) in enumerate(profile)
                 if not np.isfinite(value) or value < 0
                 or (i and value < profile[i - 1][1])),
                None,
            )
            if bad is None:
                return {"limits": {"min_C": cold, "max_C": warm},
                        "profile": profile, "warm_reason": warm_reason,
                        "cold_reason": cold_reason}
            # An interval cannot distinguish repeated observations at one T.
            # Exclude the entire failing temperature and everything colder.
            warmer = [t for t, _ in profile[:bad] if t > profile[bad][0]]
            if not warmer:
                return None
            cold = max(cold, min(warmer))
            cold_reason = ("negative_corrected_concentration" if profile[bad][1] < 0
                           else "concentration_decrease")
        return None

    def choose(self, names):
        chosen = {}
        for index, name in enumerate(names):
            previous_name = names[index - 1] if index else None
            previous = chosen.get(previous_name)
            selected = self.interval(name, previous)
            if selected is None and previous is not None:
                # Retry the handoff with progressively shorter preceding ranges.
                # Retain its warm end and its handoff from the input before it.
                before = chosen.get(names[index - 2]) if index > 1 else None
                for temperature in sorted({t for t, _ in previous["profile"]
                                           if t > previous["limits"]["min_C"]}):
                    shortened = self.interval(previous_name, before, cold_limit=temperature)
                    if shortened is None or shortened["profile"][0][0] != previous["profile"][0][0]:
                        continue
                    selected = self.interval(name, shortened)
                    if selected is not None:
                        shortened["cold_reason"] = "shortened_for_monotone_handoff"
                        chosen[previous_name] = shortened
                        break
            if selected is None:
                break
            chosen[name] = selected
        # Native temperature reversals may revisit an earlier input. Check the
        # actual combined sequence as well as each proposed interval.
        while chosen:
            ranges = {name: item["limits"] for name, item in chosen.items()}
            profile = self.profile(ranges)
            bad = next(
                (i for i, (_, value) in enumerate(profile)
                 if not np.isfinite(value) or value < 0
                 or (i and value < profile[i - 1][1])),
                None,
            )
            if bad is None:
                break
            warmer = [t for t, _ in profile[:bad] if t > profile[bad][0]]
            if not warmer:
                chosen = {}
                break
            cutoff = min(warmer)
            for name in list(chosen):
                limits = chosen[name]["limits"]
                if limits["max_C"] < cutoff:
                    del chosen[name]
                elif limits["min_C"] < cutoff:
                    chosen[name]["limits"] = dict(limits, min_C=cutoff)
                    chosen[name]["cold_reason"] = (
                        "negative_corrected_concentration" if profile[bad][1] < 0
                        else "concentration_decrease"
                    )
        return chosen
