"""Match observed droplet states without interpolating or pooling their counts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .blank_controls import BlankState


@dataclass(frozen=True)
class AlignedPoint:
    """One calculation point and the actual sample and blank rows supplying it."""

    point_id: str
    temperature_C: float
    samples: pd.DataFrame
    blanks: pd.DataFrame
    alignment: str
    point_order: int
    blank_state: BlankState | None = None
    range_details: dict | None = None

    @cached_property
    def sample_records(self):
        return _records(self.samples)

    @cached_property
    def blank_records(self):
        return _records(self.blanks)

    def count_key(self):
        """Physical identities and counts; temperature does not change a point fit."""
        keys = ("measurement_id", "run_id", "cycle_id", "n_frozen", "n_total")
        counts = tuple(tuple(sorted(tuple(row[key] for key in keys) for row in records))
                       for records in (self.sample_records, self.blank_records))
        return counts if self.blank_state is None else (*counts, self.blank_state)


def _records(frame):
    # Small aligned tables dominate native-row workflows. Convert their values
    # once, without pandas building a Series for every column of every point.
    columns = tuple(frame.columns)
    return [dict(zip(columns, row)) for row in frame.to_numpy()]


def _ordered_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if "time_s" in rows:
        rows = rows.sort_values("time_s", kind="stable")
    return rows.reset_index(drop=True).copy()


def _synchronized(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    """Recognize matching sequences, never matching generated row-number IDs."""
    if len(left) != len(right) or left.empty:
        return False
    if not np.array_equal(left.temperature_C.to_numpy(), right.temperature_C.to_numpy()):
        return False
    if (
        "time_s" in left
        and "time_s" in right
        and np.array_equal(left.time_s.to_numpy(), right.time_s.to_numpy())
    ):
        return True
    return not left.temperature_C.duplicated().any()


def _in_support(rows: pd.DataFrame, temperature: float) -> bool:
    return bool(
        not rows.empty and rows.temperature_C.min() <= temperature <= rows.temperature_C.max()
    )


def _in_range(temperature: float, limits: Mapping) -> bool:
    minimum, maximum = limits.get("min_C"), limits.get("max_C")
    return (minimum is None or temperature >= minimum) and (
        maximum is None or temperature <= maximum
    )


def _average_sample_eligible(row, temperature, intervals):
    """Average uses positive sample counts inside that input's freezing interval."""
    if intervals is None:
        return True
    limits = intervals[str(row["measurement_id"])]
    return (row["n_frozen"] > 0 and limits["min_C"] is not None
            and _in_range(temperature, limits))


def _latest_position(rows: pd.DataFrame, temperature: float, limits: Mapping) -> int | None:
    if not _in_support(rows, temperature) or not _in_range(temperature, limits):
        return None
    positions = np.flatnonzero(rows.temperature_C.ge(temperature).to_numpy())
    return int(positions[-1]) if positions.size else None


def _matching_acquisition(sample: pd.Series, blank: pd.DataFrame) -> int | None:
    """Match a real image or timestamp at the observed temperature when unambiguous."""
    same_temperature = blank.temperature_C.eq(sample.temperature_C)
    for name in ("picture_id", "time_s"):
        value = sample.get(name)
        if name not in blank or pd.isna(value) or value == "":
            continue
        matches = np.flatnonzero((same_temperature & blank[name].eq(value)).to_numpy())
        if matches.size == 1:
            return int(matches[0])
    return None


def align_observations(
    frame: pd.DataFrame,
    members: Sequence[Mapping[str, object]],
    *,
    water_blank_map: Mapping[str, Sequence[str]],
    temperature_ranges_C: Mapping[str, Mapping] | None,
    temperature_step_C: float | None = None,
    temperature_start_C: float | None = None,
    temperature_end_C: float | None = None,
    temperature_method: str = "latest",
    temperature_window_C: float | None = None,
    sample_freezing_intervals_C: Mapping[str, Mapping] | None = None,
    retain_full_range_constraints: bool = False,
) -> list[AlignedPoint]:
    """Keep native sequences, or explicitly select counts on a regular grid.

    Sample members identify one physical set and one cycle. An aligned target is
    drawn from original sample temperatures unless temperature_step_C is supplied.
    Native rows keep their actual temperature, counts, time and observation ID.
    Each physical blank enters a point once. Matching sample/blank acquisitions
    move together for synchronized runs. No source sequence is cooling-trimmed.
    An explicit grid selects samples and blanks independently by the same rule.
    Limits choose targets after count selection. Joint MLE can retain outside
    reporting constraints when a limit selects an input's full useful edge.
    """
    from .temperature_selection import grid_points, validate_temperature_selection

    validate_temperature_selection(
        temperature_step_C, temperature_method, temperature_window_C,
        temperature_start_C, temperature_end_C,
    )
    required = {
        "measurement_id",
        "run_id",
        "cycle_id",
        "temperature_C",
        "observation_id",
        "n_total",
        "n_frozen",
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Observation alignment is missing columns: {sorted(missing)}")
    if not members:
        raise ValueError("Observation alignment requires at least one sample member")
    ranges = temperature_ranges_C or {}
    streams: dict[tuple[str, str, str], pd.DataFrame] = {}
    run_members: dict[tuple[str, str], list[tuple[str, str, str]]] = {}
    for member in members:
        key = (str(member["measurement_id"]), str(member["run_id"]), str(member["cycle_id"]))
        if key in streams:
            raise ValueError(f"Repeated physical sample member: {key}")
        measurement, run, cycle = key
        rows = frame.loc[
            frame.measurement_id.eq(measurement) & frame.run_id.eq(run) & frame.cycle_id.eq(cycle)
        ]
        if rows.empty:
            raise ValueError(f"No observations for sample member {key}")
        streams[key] = _ordered_rows(rows)
        run_members.setdefault((run, cycle), []).append(key)
    if any(
        len({cycle for run_id, cycle in run_members if run_id == run}) > 1 for run, _ in run_members
    ):
        raise ValueError("One calculation group cannot combine cycles from the same run")

    if temperature_step_C is not None:
        return grid_points(
            frame, members, water_blank_map=water_blank_map, ranges=ranges,
            step_C=temperature_step_C, start_C=temperature_start_C, end_C=temperature_end_C,
            method=temperature_method, window_C=temperature_window_C,
            sample_freezing_intervals_C=sample_freezing_intervals_C,
            retain_full_range_constraints=retain_full_range_constraints,
        )

    blank_streams: dict[tuple[str, str, str], pd.DataFrame] = {}
    for measurement, run, cycle in streams:
        for blank_id in water_blank_map.get(measurement, []):
            key = (str(blank_id), run, cycle)
            if key not in blank_streams:
                blank_streams[key] = _ordered_rows(
                    frame.loc[
                        frame.measurement_id.eq(blank_id)
                        & frame.run_id.eq(run)
                        & frame.cycle_id.eq(cycle)
                    ]
                )
    synchronized_runs = {
        identity: all(_synchronized(streams[keys[0]], streams[key]) for key in keys[1:])
        for identity, keys in run_members.items()
    }
    synchronized_blanks = {
        (sample_key, blank_key): _synchronized(streams[sample_key], blank)
        for sample_key in streams
        for blank_key, blank in blank_streams.items()
        if sample_key[1:] == blank_key[1:]
        and blank_key[0] in water_blank_map.get(sample_key[0], [])
    }
    blank_support = {
        key: (float(rows.temperature_C.min()), float(rows.temperature_C.max()))
        for key, rows in blank_streams.items()
    }
    native = len(run_members) == 1 and next(iter(synchronized_runs.values()))
    reference = next(iter(streams.values()))
    targets = (
        reference.temperature_C.to_numpy(dtype=float)
        if native
        else np.unique(
            np.concatenate([rows.temperature_C.to_numpy(dtype=float) for rows in streams.values()])
        )[::-1]
    )
    from .reporting import resolve_sample_range

    range_details, resolved = {}, {}
    values = np.asarray(targets, dtype=float)
    for key, rows in streams.items():
        support = values[(values >= rows.temperature_C.min()) & (values <= rows.temperature_C.max())]
        requested = ranges.get(key[0], {})
        limits, details = resolve_sample_range(rows, support, requested)
        if not retain_full_range_constraints:
            limits = requested
        range_details[key[0]] = {
            "run_id": key[1], "cycle_id": key[2], **details, "calculation_limits_C": limits,
        }
        resolved[key[0]] = limits
    ranges = resolved
    empty = frame.iloc[:0].copy()
    points = []
    for point_order, target in enumerate(targets):
        temperature = float(target)
        selected: dict[tuple[str, str, str], tuple[pd.Series, int]] = {}
        for identity, keys in run_members.items():
            if synchronized_runs[identity]:
                positions = []
                for key in keys:
                    limits = ranges.get(key[0], {})
                    position = (
                        point_order
                        if native
                        else _latest_position(streams[key], temperature, limits)
                    )
                    if position is not None and _in_range(temperature, limits):
                        positions.append(position)
                if not positions:
                    continue
                # A synchronized run contributes one acquisition state. A range
                # may exclude a member at that state; do not mix in an older state.
                position = max(positions)
                for key in keys:
                    row = streams[key].iloc[position]
                    limits = ranges.get(key[0], {})
                    if (_in_range(temperature, limits)
                            and _average_sample_eligible(row, temperature, sample_freezing_intervals_C)):
                        selected[key] = (row, position)
            else:
                for key in keys:
                    position = _latest_position(streams[key], temperature, ranges.get(key[0], {}))
                    if position is not None:
                        row = streams[key].iloc[position]
                        if _average_sample_eligible(row, temperature, sample_freezing_intervals_C):
                            selected[key] = (row, position)

        blank_positions: dict[tuple[str, str, str], list[int]] = {}
        alignment = "native" if native else "latest"
        for key, (sample, position) in selected.items():
            measurement, run, cycle = key
            for blank_id in water_blank_map.get(measurement, []):
                blank_key = (str(blank_id), run, cycle)
                blank = blank_streams[blank_key]
                minimum, maximum = blank_support[blank_key]
                if not minimum <= temperature <= maximum:
                    raise ValueError(
                        f"Blank {blank_id!r} lacks observed temperature coverage for "
                        f"measurement {measurement!r}, run {run!r}, cycle {cycle!r}, "
                        f"temperature {temperature:g} C; no blank extrapolation is performed"
                    )
                match = None
                if synchronized_blanks[(key, blank_key)]:
                    match = position
                else:
                    match = _matching_acquisition(sample, blank)
                if match is None:
                    match = _latest_position(blank, temperature, {})
                    alignment = "latest"
                if match is None:
                    raise ValueError(
                        f"Blank {blank_id!r} has no observation at or warmer than "
                        f"{temperature:g} C in run {run!r}, cycle {cycle!r}"
                    )
                blank_positions.setdefault(blank_key, []).append(match)
        selected_blanks = {}
        for blank_key, positions in sorted(blank_positions.items()):
            blank = blank_streams[blank_key]
            if len(set(positions)) == 1:
                match = positions[0]
            else:
                # Asynchronous sample states can request different acquisitions
                # from one physical blank. Resolve at the common target with the
                # same latest-warmer rule; never duplicate that blank's droplets
                # or let member order determine the background observation.
                latest = _latest_position(blank, temperature, {})
                if latest is None:
                    raise ValueError(f"Blank {blank_key!r} has no state at {temperature:g} C")
                match = latest
                alignment = "latest"
            selected_blanks[blank_key] = (blank.iloc[match], match)
        if len(selected) == 1:
            key, (_, position) = next(iter(selected.items()))
            samples = streams[key].iloc[position:position + 1]
        else:
            samples = pd.DataFrame([row for row, _ in selected.values()]) if selected else empty
        if len(selected_blanks) == 1:
            key, (_, position) = next(iter(selected_blanks.items()))
            blanks = blank_streams[key].iloc[position:position + 1]
        else:
            blanks = (pd.DataFrame([row for row, _ in selected_blanks.values()])
                      if selected_blanks else empty)
        points.append(
            AlignedPoint(
                point_id=f"point:{point_order}",
                temperature_C=temperature,
                samples=samples.reset_index(drop=True),
                blanks=blanks.reset_index(drop=True),
                alignment=alignment,
                point_order=point_order,
                range_details=range_details,
            )
        )
    return points
