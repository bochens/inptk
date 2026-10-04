"""Shared defaults and validation for Python steps, workflows, and the CLI.

These records are internal: callers use the same explicit keyword arguments in
Python and corresponding CLI flags. Importers never choose analysis settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

from .methods import validate_combination_method
from .tables import UNITS
from .temperature_selection import validate_temperature_selection


def positive_number(name, value):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be finite and positive")


def validate_decrease_policy(policy):
    if policy not in ("stop_at_decrease", "skip_decreases"):
        raise ValueError("decrease_policy must be 'stop_at_decrease' or 'skip_decreases'")


@dataclass(frozen=True)
class EstimationSettings:
    method: Literal["mle", "average"] = "mle"
    fit_step_C: float | None = None
    temperature_step_C: float | None = None
    temperature_start_C: float | None = None
    temperature_end_C: float | None = None
    temperature_method: Literal["latest", "max", "window"] = "latest"
    temperature_window_C: float | None = None
    z: float = 1.96
    water_blank_correction: bool = True

    def __post_init__(self):
        validate_combination_method(self.method)
        validate_temperature_selection(
            self.temperature_step_C,
            self.temperature_method,
            self.temperature_window_C,
            self.temperature_start_C,
            self.temperature_end_C,
        )
        positive_number("z", self.z)
        if not isinstance(self.water_blank_correction, bool):
            raise TypeError("water_blank_correction must be a bool (True or False)")
        if self.fit_step_C is not None:
            positive_number("fit_step_C", self.fit_step_C)
            if self.method != "mle":
                raise ValueError("fit_step_C applies only to method='mle'")


@dataclass(frozen=True)
class AnalysisSettings(EstimationSettings):
    output_basis: str = "suspension"
    differential: bool = False
    decrease_policy: Literal["stop_at_decrease", "skip_decreases"] = "stop_at_decrease"

    def __post_init__(self):
        super().__post_init__()
        if self.output_basis not in UNITS:
            raise ValueError(f"Unknown output_basis {self.output_basis!r}")
        if not isinstance(self.differential, bool):
            raise TypeError("differential must be True or False")
        validate_decrease_policy(self.decrease_policy)


DEFAULTS = AnalysisSettings()
