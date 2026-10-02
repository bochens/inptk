"""Validate concentration combination choices and measured temperature ranges."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING, Literal, TypedDict, cast
from urllib.parse import quote

if TYPE_CHECKING:
    import pandas as pd

    from .experiment import Experiment


def validate_combination_method(value) -> Literal["mle", "average"]:
    """Accept the two explicit ways to combine eligible dilution measurements."""
    if not isinstance(value, str):
        raise TypeError("method must be a string: 'mle' or 'average'")
    if value not in ("mle", "average"):
        raise ValueError("method must be 'mle' or 'average'")
    return cast(Literal["mle", "average"], value)


def _temperature_bound(value, *, measurement_id: str, boundary: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"{boundary} for measurement {measurement_id!r} must be a finite number or None"
        )
    try:
        number = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{boundary} for measurement {measurement_id!r} must be finite") from error
    if not math.isfinite(number):
        raise ValueError(f"{boundary} for measurement {measurement_id!r} must be finite")
    return number


def validate_temperature_ranges(
    value, *, measurement_ids: set[str]
) -> dict[str, dict[str, float | None]]:
    """Copy and normalize optional, inclusive ranges keyed by exact measurement names.

    A missing or None boundary is unlimited. An omitted measurement uses its full
    observed temperature support. Callers supply sample measurement IDs only, so
    mapped water blanks cannot receive sample eligibility ranges. Returned values
    are plain dictionaries containing Python floats or None, suitable for JSON.
    """
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("temperature_ranges_C must map measurement names to range objects")
    normalized = {}
    for measurement_id, bounds in value.items():
        if not isinstance(measurement_id, str) or not measurement_id.strip():
            raise ValueError("temperature_ranges_C keys must be non-empty measurement names")
        if measurement_id not in measurement_ids:
            raise ValueError(
                f"Unknown measurement in temperature_ranges_C: {measurement_id!r}; "
                f"available measurement names: {sorted(measurement_ids)}"
            )
        if not isinstance(bounds, Mapping):
            raise TypeError(f"Temperature range for {measurement_id!r} must be a range object")
        unknown = set(bounds) - {"min_C", "max_C"}
        if unknown:
            raise ValueError(
                f"Unknown temperature range keys for {measurement_id!r}: "
                f"{sorted(repr(key) for key in unknown)}; use min_C and max_C"
            )
        lower = _temperature_bound(
            bounds.get("min_C"), measurement_id=measurement_id, boundary="min_C"
        )
        upper = _temperature_bound(
            bounds.get("max_C"), measurement_id=measurement_id, boundary="max_C"
        )
        if lower is not None and upper is not None and lower > upper:
            raise ValueError(f"Temperature range for {measurement_id!r} requires min_C <= max_C")
        normalized[measurement_id] = {"min_C": lower, "max_C": upper}
    return normalized


class CombinationMember(TypedDict):
    measurement_id: str
    run_id: str
    cycle_id: str


class CombinationGroup(TypedDict):
    sample_id: str
    members: list[CombinationMember]


def validate_combination_groups(
    value, experiment: Experiment, frame: pd.DataFrame
) -> dict[str, CombinationGroup]:
    """Resolve exact measurement/cycle members without combining repeated cycles.

    None produces one group per observed parent sample/run/cycle. An explicit
    mapping uses group names as keys and nonempty lists of measurement_id/cycle_id
    objects as values. It defines only the requested output groups. Run and parent
    sample identities always come from measurement metadata.
    """
    required = {"measurement_id", "sample_id", "run_id", "cycle_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Combination input is missing identity columns: {sorted(missing)}")
    for name in required:
        if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
            raise ValueError(f"Combination input {name} must contain non-empty identities")
    blank_ids = {name for names in experiment.water_blank_map.values() for name in names}
    available: dict[tuple[str, str], CombinationMember] = {}
    for row in frame[sorted(required)].drop_duplicates().itertuples(index=False):
        measurement_id, cycle_id = str(row.measurement_id), str(row.cycle_id)
        metadata = experiment.measurements.get(measurement_id)
        if metadata is None:
            raise ValueError(f"Unknown measurement in combination input: {measurement_id!r}")
        if (str(row.sample_id), str(row.run_id)) != (metadata.sample_id, metadata.run_id):
            raise ValueError(f"Combination input identities disagree for {measurement_id!r}")
        if measurement_id not in blank_ids:
            available[measurement_id, cycle_id] = {
                "measurement_id": measurement_id,
                "run_id": metadata.run_id,
                "cycle_id": cycle_id,
            }
    if not available:
        raise ValueError("No sample observations are available for combination")
    if value is None:
        defaults: dict[str, CombinationGroup] = {}
        for (measurement_id, _), member in sorted(available.items()):
            sample_id = experiment.measurements[measurement_id].sample_id
            group_id = "/".join(
                quote(name, safe="") for name in (sample_id, member["run_id"], member["cycle_id"])
            )
            defaults.setdefault(group_id, {"sample_id": sample_id, "members": []})[
                "members"
            ].append(member.copy())
        return defaults
    if not isinstance(value, Mapping):
        raise TypeError(
            "combination_groups must map group names to lists of measurement/cycle members"
        )
    if not value:
        raise ValueError("combination_groups must contain at least one group")
    groups: dict[str, CombinationGroup] = {}
    for group_id, members in value.items():
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError("Combination group names must be non-empty strings")
        if not isinstance(members, list) or not members:
            raise TypeError(f"Combination group {group_id!r} requires a nonempty list of members")
        normalized: list[CombinationMember] = []
        seen: set[tuple[str, str]] = set()
        cycles_by_run: dict[str, str] = {}
        parents = set()
        for member in members:
            if not isinstance(member, Mapping) or set(member) != {"measurement_id", "cycle_id"}:
                raise ValueError(
                    "Each combination member requires exactly measurement_id and cycle_id"
                )
            measurement_id, cycle_id = member["measurement_id"], member["cycle_id"]
            if any(
                not isinstance(name, str) or not name.strip() for name in (measurement_id, cycle_id)
            ):
                raise ValueError(
                    "Combination member measurement_id and cycle_id must be non-empty strings"
                )
            if measurement_id in blank_ids:
                raise ValueError(
                    f"Water-blank measurement {measurement_id!r} cannot be a combination member"
                )
            key = (measurement_id, cycle_id)
            if key not in available:
                raise ValueError(f"Unknown or absent combination member: {key!r}")
            if key in seen:
                raise ValueError(f"Duplicate combination member in {group_id!r}: {key!r}")
            seen.add(key)
            resolved = available[key]
            run_id = resolved["run_id"]
            if run_id in cycles_by_run and cycles_by_run[run_id] != cycle_id:
                raise ValueError(f"Combination group {group_id!r} must use only one cycle per run")
            cycles_by_run[run_id] = cycle_id
            parents.add(experiment.measurements[measurement_id].sample_id)
            normalized.append(resolved.copy())
        if len(parents) != 1:
            raise ValueError(f"Combination group {group_id!r} must contain one parent sample")
        groups[group_id] = {"sample_id": parents.pop(), "members": normalized}
    return groups
