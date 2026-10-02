"""Validated settings for each way of combining dilution measurements."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, fields
from itertools import pairwise
from numbers import Integral
from types import MappingProxyType
from typing import Literal


def _finite(value, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


def _dilution_key(key, name: str) -> float:
    dilution = _finite(key, f"{name} dilution")
    if dilution <= 0:
        raise ValueError(f"{name} dilution keys must be positive")
    return dilution


def _number_mapping(values, name: str, *, minimum=None, strict=False):
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must map dilution factors to numbers")
    result = {}
    for key, value in values.items():
        dilution = _dilution_key(key, name)
        if dilution in result:
            raise ValueError(f"{name} repeats dilution {dilution:g}")
        number = _finite(value, name)
        if minimum is not None and (number < minimum or (strict and number == minimum)):
            relation = "greater than" if strict else "at least"
            raise ValueError(f"{name} values must be {relation} {minimum}")
        result[dilution] = number
    return MappingProxyType(result)


@dataclass(frozen=True)
class Stitch:
    """Automatic OLAF stitching, applied when a group has multiple measurements.

    min_unfrozen is the minimum number of unfrozen droplets at an eligible point.
    overlap_points is the number of cold-end points checked for overlap adjustments;
    zero disables those adjustments while retaining the ordinary dilution handoff.
    Missing and infinite concentrations are always excluded.
    """

    min_unfrozen: int = 3
    overlap_points: int = 4

    def __post_init__(self):
        for name, minimum in (("min_unfrozen", 1), ("overlap_points", 0)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f"{name} must be a whole number >= {minimum}")
            object.__setattr__(self, name, int(value))


@dataclass(frozen=True)
class ManualStitch:
    """Join dilutions at explicit switching temperatures, ordered warm to cold.

    Dilutions are ordered from least to most diluted. At each switch temperature
    and colder, the next dilution is selected. A group with N dilutions requires
    N-1 switches. No automatic droplet cutoff or overlap adjustment is applied.
    Missing selected observations remain missing; another dilution is never used
    as a fallback. All sample/run/cycle groups must have the same dilution factors.
    """

    switch_temperatures_C: tuple[float, ...] | list[float]

    def __post_init__(self):
        values = self.switch_temperatures_C
        if not isinstance(values, (list, tuple)):
            raise TypeError("switch_temperatures_C must be a list of temperatures")
        temperatures = tuple(_finite(value, "switch temperature") for value in values)
        if any(warm <= cold for warm, cold in pairwise(temperatures)):
            raise ValueError("Switch temperatures must be strictly ordered from warm to cold")
        object.__setattr__(self, "switch_temperatures_C", temperatures)


@dataclass(frozen=True)
class MLE:
    """Maximum likelihood estimation: jointly fit counts from different dilutions.

    All mappings use dilution factors as keys and apply across samples/runs/cycles.
    Temperature limits require an explicit mask_mode: drop_rows omits warmer
    rows; rebase_counts also removes the warm frozen baseline and those droplets.
    confidence_drop=None uses the workflow's z**2/2, preserving the default.
    """

    temperature_eligibility_C: Mapping[float, float] | None = None
    mask_mode: Literal["drop_rows", "rebase_counts"] | None = None
    dilution_likelihood_weights: Mapping[float, float] | None = None
    dilution_action_counts: Mapping[float, float] | None = None
    action_weight_lambda: float | None = None
    action_weight_half_life: float | None = None
    confidence_drop: float | None = None

    def __post_init__(self):
        for name, minimum, strict in (
            ("temperature_eligibility_C", None, False),
            ("dilution_likelihood_weights", 0, True),
            ("dilution_action_counts", 0, False),
        ):
            object.__setattr__(
                self,
                name,
                _number_mapping(getattr(self, name), name, minimum=minimum, strict=strict),
            )
        if self.temperature_eligibility_C is None:
            if self.mask_mode is not None:
                raise ValueError("mask_mode requires temperature_eligibility_C")
        elif self.mask_mode not in ("drop_rows", "rebase_counts"):
            raise ValueError("Temperature limits require mask_mode='drop_rows' or 'rebase_counts'")
        if self.dilution_likelihood_weights is not None and self.dilution_action_counts is not None:
            raise ValueError("Use dilution_likelihood_weights or dilution_action_counts, not both")
        if self.action_weight_lambda is not None and self.action_weight_half_life is not None:
            raise ValueError("Use action_weight_lambda or action_weight_half_life, not both")
        for name in ("action_weight_lambda", "action_weight_half_life", "confidence_drop"):
            value = getattr(self, name)
            if value is not None:
                value = _finite(value, name)
                if value < 0 or (name != "action_weight_lambda" and value == 0):
                    raise ValueError(f"Invalid {name}: {value}")
                object.__setattr__(self, name, value)
        has_decay = (
            self.action_weight_lambda is not None or self.action_weight_half_life is not None
        )
        if self.dilution_action_counts is not None and not has_decay:
            raise ValueError("dilution_action_counts requires a decay rate or half-life")
        if has_decay and self.dilution_action_counts is None:
            raise ValueError("Action weighting settings require dilution_action_counts")


DilutionMethod = Stitch | ManualStitch | MLE


def resolve_method(value: str | DilutionMethod, options: dict | None = None) -> DilutionMethod:
    """Resolve string shortcuts and CLI settings; unexpected keys are errors."""
    if isinstance(value, (Stitch, ManualStitch, MLE)):
        if options is not None:
            raise ValueError("Method objects already contain their settings")
        return value
    if options is not None and not isinstance(options, dict):
        raise TypeError("Method options must be a JSON object")
    if value == "stitch":
        return Stitch(**(options or {}))
    if value == "mle":
        return MLE(**(options or {}))
    if value == "manual":
        if not options or "switch_temperatures_C" not in options:
            raise ValueError("Manual stitching requires switch_temperatures_C")
        return ManualStitch(**options)
    raise ValueError("dilution_method must be 'stitch', 'mle', or a method settings object")


def method_name(method: DilutionMethod) -> str:
    return (
        "stitch"
        if isinstance(method, Stitch)
        else "manual"
        if isinstance(method, ManualStitch)
        else "mle"
    )


def method_options(method: DilutionMethod, *, z: float) -> dict:
    """Return JSON-native effective settings for saving and CLI reuse."""
    options = {}
    for item in fields(method):
        value = getattr(method, item.name)
        if isinstance(value, Mapping):
            value = {str(key): list(v) if isinstance(v, tuple) else v for key, v in value.items()}
        if isinstance(value, tuple):
            value = list(value)
        options[item.name] = value
    if isinstance(method, MLE) and method.confidence_drop is None:
        options["confidence_drop"] = z**2 / 2
    return options
