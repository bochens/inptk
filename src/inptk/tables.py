"""Scientific tables with explicit identities, units, and non-mutating selection."""

from __future__ import annotations

import copy
from typing import ClassVar, Self

import numpy as np
import pandas as pd

IDENTITY_COLUMNS = ("run_id", "sample_id", "cycle_id")
UNITS = {
    "suspension": "INP_per_mL_suspension",
    "sampled_air": "INP_per_L_air",
    "dry_soil": "INP_per_g_dry_soil",
}


class ScientificTable:
    """A validated pandas table. Returned frames are copies of the stored data."""

    required: ClassVar[tuple[str, ...]] = IDENTITY_COLUMNS + ("temperature_C",)

    def __init__(self, data: pd.DataFrame, *, history: list[dict] | None = None):
        frame = pd.DataFrame(data).copy(deep=True)
        missing = set(self.required) - set(frame.columns)
        if missing:
            raise ValueError(f"{type(self).__name__} is missing columns: {sorted(missing)}")
        for name in (*IDENTITY_COLUMNS, "measurement_id"):
            if name in frame:
                if frame[name].isna().any() or frame[name].astype(str).str.strip().eq("").any():
                    raise ValueError(f"{name} must contain non-empty identifiers")
                frame[name] = frame[name].astype(str)
        frame["temperature_C"] = pd.to_numeric(frame["temperature_C"], errors="raise")
        if not np.isfinite(frame["temperature_C"]).all():
            raise ValueError("temperature_C must be finite")
        self._validate(frame)
        self._data = frame.reset_index(drop=True)
        self._history = copy.deepcopy(history or [])

    def _validate(self, frame: pd.DataFrame) -> None:
        pass

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self._data.columns)

    @property
    def history(self) -> list[dict]:
        return copy.deepcopy(self._history)

    @property
    def warnings(self) -> list[str]:
        """Warnings recorded by individual processing steps."""
        return [str(message) for step in self._history for message in step.get("warnings", [])]

    def __len__(self) -> int:
        return len(self._data)

    def to_dataframe(self) -> pd.DataFrame:
        return self._data.copy(deep=True)

    def select(self, **criteria) -> Self:
        """Select labels, or lists of labels, without changing the original table."""
        frame = self.to_dataframe()
        for column, values in criteria.items():
            if column not in frame:
                raise KeyError(f"Unknown column {column!r}")
            selected = list(values) if isinstance(values, (list, tuple, set)) else [values]
            if column in (*IDENTITY_COLUMNS, "measurement_id"):
                selected = [str(value) for value in selected]
            frame = frame[frame[column].isin(selected)]
        return type(self)(frame, history=self.history)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({len(self)} rows, columns={list(self.columns)!r})"


class CountsTable(ScientificTable):
    """Observed counts for physical droplet sets; cycles are never pooled."""

    required = ScientificTable.required + ("measurement_id", "n_total", "n_frozen")

    def _validate(self, frame: pd.DataFrame) -> None:
        for column in ("n_total", "n_frozen"):
            frame[column] = pd.to_numeric(frame[column], errors="raise")
            if not np.isfinite(frame[column]).all():
                raise ValueError(f"{column} must be finite")
            if (frame[column] % 1 != 0).any():
                raise ValueError(f"Observed {column} must contain whole droplet counts")
        if ((frame.n_total <= 0) | (frame.n_frozen < 0) | (frame.n_frozen > frame.n_total)).any():
            raise ValueError("Counts require 0 <= n_frozen <= n_total and n_total > 0")
        if "time_s" in frame:
            frame["time_s"] = pd.to_numeric(frame["time_s"], errors="raise")
            if not np.isfinite(frame.time_s).all():
                raise ValueError("time_s must be finite when supplied")


class FrozenFractionTable(CountsTable):
    """Counts evaluated at temperature thresholds, with a derived frozen fraction."""

    def _validate(self, frame: pd.DataFrame) -> None:
        super()._validate(frame)
        frame["fraction_frozen"] = frame.n_frozen / frame.n_total
        keys = [*IDENTITY_COLUMNS, "measurement_id", "temperature_C"]
        if frame.duplicated(keys).any():
            raise ValueError("Frozen fractions require one row per measurement/cycle/temperature")


class CumulativeSpectrumTable(ScientificTable):
    """Cumulative concentration; error columns are widths, not interval endpoints."""

    required = ScientificTable.required + ("concentration", "unit", "basis")

    def _validate(self, frame: pd.DataFrame) -> None:
        for name in ("concentration", "lower_error", "upper_error"):
            if name in frame:
                frame[name] = pd.to_numeric(frame[name], errors="raise")
        if ("lower_error" in frame) != ("upper_error" in frame):
            raise ValueError("Both lower_error and upper_error must be supplied together")
        for name in ("lower_error", "upper_error"):
            if name in frame and (frame[name].dropna() < 0).any():
                raise ValueError(f"{name} must be nonnegative")
        expected = frame.basis.map(UNITS)
        if expected.isna().any() or not expected.eq(frame.unit).all():
            raise ValueError("Concentration unit does not match its basis")
        if frame.basis.nunique() > 1:
            raise ValueError("A spectrum table must use one concentration basis")
        keys = [*IDENTITY_COLUMNS, "temperature_C"]
        if "measurement_id" in frame:
            keys.append("measurement_id")
        if frame.duplicated(keys).any():
            raise ValueError("A spectrum requires unique identity/temperature rows")
        if "is_extrapolated" not in frame:
            frame["is_extrapolated"] = False
        if "correction_state" not in frame:
            frame["correction_state"] = "uncorrected"


class DifferentialSpectrumTable(ScientificTable):
    """Activity per degree over explicit cold-to-warm temperature intervals."""

    required = ScientificTable.required + (
        "measurement_id",
        "concentration",
        "unit",
        "basis",
        "temperature_bin_left_C",
        "temperature_bin_right_C",
    )

    def _validate(self, frame: pd.DataFrame) -> None:
        for name in ("concentration", "temperature_bin_left_C", "temperature_bin_right_C"):
            frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not frame.basis.eq("suspension").all():
            raise ValueError("Differential output currently uses suspension units")
        if not frame.unit.eq("INP_per_mL_suspension_per_C").all():
            raise ValueError("Differential concentration must be expressed per mL per degree C")
        if not (frame.temperature_bin_left_C < frame.temperature_bin_right_C).all():
            raise ValueError("Differential intervals require cold limit < warm limit")
