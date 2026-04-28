from __future__ import annotations

import numpy as np
import pandas as pd


def flag_below_original_lower_ci(
    corrected_value: np.ndarray,
    original_value: np.ndarray,
    original_lower_error: np.ndarray,
) -> np.ndarray:
    """Return rows where corrected values fall below the original lower error bound."""

    corrected = np.asarray(corrected_value, dtype=float)
    original = np.asarray(original_value, dtype=float)
    lower_error = np.asarray(original_lower_error, dtype=float)
    corrected, original, lower_error = np.broadcast_arrays(corrected, original, lower_error)
    return corrected < (original - lower_error)


def enforce_monotonic_vs_temperature(
    temperature_C: np.ndarray,
    value: np.ndarray,
    lower_ci: np.ndarray | None = None,
    upper_ci: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Ensure values do not decrease as temperature decreases."""

    temp = np.asarray(temperature_C, dtype=float)
    values = np.asarray(value, dtype=float).copy()
    if temp.shape != values.shape:
        raise ValueError("temperature_C and value must have the same shape")

    lower = None if lower_ci is None else np.asarray(lower_ci, dtype=float).copy()
    upper = None if upper_ci is None else np.asarray(upper_ci, dtype=float).copy()
    if lower is not None and lower.shape != values.shape:
        raise ValueError("lower_ci must match value shape")
    if upper is not None and upper.shape != values.shape:
        raise ValueError("upper_ci must match value shape")

    order = np.argsort(temp)[::-1]
    flags = np.zeros(values.shape, dtype=int)
    previous_index: int | None = None
    for index in order:
        if not np.isfinite(values[index]):
            previous_index = index if previous_index is None else previous_index
            continue
        if previous_index is not None and np.isfinite(values[previous_index]):
            if values[index] < values[previous_index]:
                original_value = values[index]
                values[index] = values[previous_index]
                flags[index] = 1
                if upper is not None:
                    previous_upper = (
                        upper[previous_index] if np.isfinite(upper[previous_index]) else 0.0
                    )
                    current_upper = upper[index] if np.isfinite(upper[index]) else 0.0
                    upper[index] = np.sqrt(previous_upper**2 + current_upper**2)
                if lower is not None:
                    lower[index] = lower[previous_index]
                if not np.isfinite(original_value):
                    flags[index] = 2
        previous_index = index
    return values, lower, upper, flags


def qc_blank_corrected_spectrum(
    corrected_df: pd.DataFrame,
    original_df: pd.DataFrame,
    *,
    threshold_percent: float = 10.0,
    error_signal: float = -9999.0,
) -> pd.DataFrame:
    """Apply UFOLAF-style blank-corrected spectrum QC to a dataframe."""

    required = {"temperature_C", "value", "lower_ci", "upper_ci"}
    missing = required - set(corrected_df.columns)
    if missing:
        raise ValueError(f"corrected_df missing required columns: {sorted(missing)}")
    missing_original = required - set(original_df.columns)
    if missing_original:
        raise ValueError(f"original_df missing required columns: {sorted(missing_original)}")
    if threshold_percent < 0:
        raise ValueError("threshold_percent cannot be negative")

    corrected = corrected_df.copy()
    original = original_df.copy()
    merge_keys = ["temperature_C"]
    if "sample_id" in corrected.columns and "sample_id" in original.columns:
        merge_keys.insert(0, "sample_id")
    aligned = corrected.merge(
        original[merge_keys + ["value", "lower_ci"]],
        on=merge_keys,
        suffixes=("", "_original"),
        how="left",
    )
    below = flag_below_original_lower_ci(
        aligned["value"].to_numpy(dtype=float),
        aligned["value_original"].to_numpy(dtype=float),
        aligned["lower_ci_original"].to_numpy(dtype=float),
    )
    if len(below) and below.mean() * 100.0 > threshold_percent:
        corrected.loc[below, ["value", "lower_ci", "upper_ci"]] = error_signal

    valid = corrected["value"].to_numpy(dtype=float) != error_signal
    existing_flags = (
        corrected["qc_flag"].to_numpy(dtype=int, copy=True)
        if "qc_flag" in corrected
        else np.zeros(len(corrected), dtype=int)
    )
    if valid.any():
        fixed_value, fixed_lower, fixed_upper, flags = enforce_monotonic_vs_temperature(
            corrected.loc[valid, "temperature_C"].to_numpy(dtype=float),
            corrected.loc[valid, "value"].to_numpy(dtype=float),
            corrected.loc[valid, "lower_ci"].to_numpy(dtype=float),
            corrected.loc[valid, "upper_ci"].to_numpy(dtype=float),
        )
        corrected.loc[valid, "value"] = fixed_value
        corrected.loc[valid, "lower_ci"] = fixed_lower
        corrected.loc[valid, "upper_ci"] = fixed_upper
        existing_flags[valid] = np.maximum(existing_flags[valid], flags)
    corrected["qc_flag"] = existing_flags
    return corrected


def trim_leading_zeros_by_temperature(
    df: pd.DataFrame,
    *,
    value_column: str = "value",
    tolerance: float = 1e-10,
) -> pd.DataFrame:
    """Drop initial warm-side rows until the first nonzero value."""

    if value_column not in df:
        raise ValueError(f"{value_column!r} column is required")
    values = df[value_column].to_numpy(dtype=float)
    nonzero = np.abs(values) > tolerance
    if not nonzero.any():
        return df.iloc[0:0].copy()
    first_index = np.argmax(nonzero)
    return df.iloc[first_index:].copy()


def sanitize_export_spectrum(
    df: pd.DataFrame,
    *,
    value_column: str = "value",
    lower_column: str = "lower_ci",
    upper_column: str = "upper_ci",
    error_signal: float = -9999.0,
) -> pd.DataFrame:
    """Replace invalid exported values with a configured error signal."""

    for column in (value_column, lower_column, upper_column):
        if column not in df:
            raise ValueError(f"{column!r} column is required")
    clean = df.copy()
    for column in (value_column, lower_column, upper_column):
        values = clean[column].to_numpy(dtype=float)
        invalid = ~np.isfinite(values)
        if column in (value_column, lower_column):
            invalid |= values < 0
        clean.loc[invalid, column] = error_signal
    return clean
