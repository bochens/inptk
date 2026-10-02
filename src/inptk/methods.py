"""Validate concentration combination choices and measured temperature ranges."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import Literal, cast


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
