"""Optional temperature-grid views of already selected concentration results."""

from __future__ import annotations

import json
from collections.abc import Hashable
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .tables import CombinedSpectrumTable


def _grid(temperatures: pd.Series, step: float) -> list[float]:
    increment = Decimal(str(step))
    cold = Decimal(str(float(temperatures.min())))
    warm = Decimal(str(float(temperatures.max())))
    first = int((cold / increment).to_integral_value(rounding=ROUND_CEILING))
    last = int((warm / increment).to_integral_value(rounding=ROUND_FLOOR))
    return [float(increment * index) for index in range(last, first - 1, -1)]


def _retained_segments(group: pd.DataFrame) -> list[pd.DataFrame]:
    """Keep native gaps as boundaries even when given all final candidates."""
    segments: list[pd.DataFrame] = []
    indices: list[Hashable] = []
    previous_segment = None
    for index, row in group.sort_values("point_order", kind="stable").iterrows():
        valid = (
            np.isfinite(row.concentration)
            and pd.notna(row.segment_id)
            and bool(row.get("used_in_final", True))
        )
        if not valid or (indices and row.segment_id != previous_segment):
            if indices:
                segments.append(group.loc[indices].copy())
            indices = []
        if valid:
            indices.append(index)
            previous_segment = row.segment_id
    if indices:
        segments.append(group.loc[indices].copy())
    return segments


def _quality(rows: list[pd.Series]) -> int:
    result = 0
    for row in rows:
        value = row.get("qc_flag", 0)
        if pd.notna(value):
            result |= int(value)
    return result


def _record(
    rows: list[pd.Series],
    weights: list[float],
    *,
    temperature: float,
    point_order: int,
    segment_number: int,
    method: str,
    has_errors: bool,
    endpoint_temperatures: list[float],
) -> dict:
    first = rows[0]
    status = (
        "interpolated"
        if len(rows) == 2
        else ("exact" if float(first.temperature_C) == temperature else "sampled")
    )
    def weighted_value(name: str) -> float:
        values = [float(row[name]) for row in rows]
        if all(value == values[0] for value in values):
            return values[0]
        return float(sum(weight * value for weight, value in zip(weights, values)))

    record = {
        "sample_id": first.sample_id,
        "group_id": first.group_id,
        "point_id": f"resampled:{segment_number}:{point_order}",
        "temperature_C": temperature,
        "concentration": weighted_value("concentration"),
        "unit": first.unit,
        "basis": first.basis,
        "point_order": point_order,
        "segment_id": first.segment_id,
        "resampling_method": method,
        "sampling_status": status,
        "is_interpolated": status == "interpolated",
        "is_extrapolated": any(bool(row.get("is_extrapolated", False)) for row in rows),
        "source_point_ids": json.dumps([str(row.point_id) for row in rows]),
        "source_temperatures_C": json.dumps([float(row.temperature_C) for row in rows]),
        "endpoint_temperatures_C": json.dumps(endpoint_temperatures),
        "source_point_orders": json.dumps([int(row.point_order) for row in rows]),
        "qc_flag": _quality(rows),
        "uncertainty_method": "interpolated_interval_endpoints"
        if len(rows) == 2
        else first.get("uncertainty_method", "unspecified"),
        "source_uncertainty_methods": json.dumps(
            [str(row.get("uncertainty_method", "unspecified")) for row in rows]
        ),
        "correction_state": first.get("correction_state", "uncorrected"),
    }
    if has_errors:
        # Linear interpolation of point estimates and corresponding widths is
        # algebraically identical to interpolating lower and upper endpoints.
        # This form avoids cancellation from subtracting nearly equal numbers.
        for name in ("lower_error", "upper_error"):
            record[name] = weighted_value(name)
        if not np.isfinite([record["lower_error"], record["upper_error"]]).all():
            record["qc_flag"] |= 1
    return record


def resample_spectrum(
    spectrum: CombinedSpectrumTable,
    step_C: float,
    method: Literal["sample", "interpolate"] = "sample",
) -> CombinedSpectrumTable:
    """Return a separate final-temperature view without refitting any observations.

    Sampling chooses the latest retained warmer state in native point order.
    Interpolation first selects the latest warmer state at each observed target
    temperature within a retained segment, then connects those endpoint values.
    This keeps temperature jitter from reversing an increasing native sequence.
    Actual source observations and endpoint target temperatures are both recorded.
    Concentrations and interval endpoints are interpolated, never counts or new
    fitted uncertainty. Neither method extends outside a segment or crosses
    excluded rows. Existing source extrapolation flags remain in the output.
    """
    from .tables import CombinedSpectrumTable

    if not isinstance(spectrum, CombinedSpectrumTable):
        raise TypeError("spectrum must be a CombinedSpectrumTable")
    if method not in ("sample", "interpolate"):
        raise ValueError("resampling method must be 'sample' or 'interpolate'")
    if isinstance(step_C, bool) or not np.isfinite(step_C) or step_C <= 0:
        raise ValueError("step_C must be finite and positive")
    source = spectrum.to_dataframe()
    missing = {"point_order", "segment_id"} - set(source)
    if missing:
        raise ValueError(f"Resampling requires retained-segment provenance: {sorted(missing)}")
    has_errors = "lower_error" in source and "upper_error" in source
    records = []
    for _, group in source.groupby("group_id", sort=False):
        point_order = 0
        for segment_number, segment in enumerate(_retained_segments(group)):
            latest = segment.sort_values("point_order", kind="stable")
            observed_temperatures = latest.temperature_C.to_numpy(dtype=float)
            temperatures = np.unique(observed_temperatures)
            by_temperature = np.argsort(-observed_temperatures, kind="stable")
            # For each target, find the last observation among all temperatures
            # at or warmer than it. The cumulative maximum uses native position,
            # not count size or temperature sorting, to define "latest".
            warmer_end = np.searchsorted(
                -observed_temperatures[by_temperature], -temperatures, side="right"
            ) - 1
            positions = np.maximum.accumulate(by_temperature)[warmer_end]
            endpoints = latest.iloc[positions].reset_index(drop=True)
            for temperature in _grid(segment.temperature_C, float(step_C)):
                if method == "sample":
                    eligible = latest.loc[latest.temperature_C.ge(temperature)]
                    rows, weights = [eligible.iloc[-1]], [1.0]
                    endpoint_temperatures = [temperature]
                else:
                    exact = np.flatnonzero(temperatures == temperature)
                    if exact.size:
                        rows, weights = [endpoints.iloc[int(exact[-1])]], [1.0]
                        endpoint_temperatures = [temperature]
                    else:
                        right = int(np.searchsorted(temperatures, temperature))
                        if right == 0 or right == len(temperatures):
                            continue
                        cold, warm = endpoints.iloc[right - 1], endpoints.iloc[right]
                        warm_weight = float(
                            (temperature - temperatures[right - 1])
                            / (temperatures[right] - temperatures[right - 1])
                        )
                        rows, weights = [cold, warm], [1 - warm_weight, warm_weight]
                        endpoint_temperatures = [
                            float(temperatures[right - 1]), float(temperatures[right])
                        ]
                records.append(
                    _record(
                        rows,
                        weights,
                        temperature=temperature,
                        point_order=point_order,
                        segment_number=segment_number,
                        method=method,
                        has_errors=has_errors,
                        endpoint_temperatures=endpoint_temperatures,
                    )
                )
                point_order += 1
    columns = [
        "sample_id",
        "group_id",
        "point_id",
        "temperature_C",
        "concentration",
        "unit",
        "basis",
        "point_order",
        "segment_id",
        "resampling_method",
        "sampling_status",
        "is_interpolated",
        "is_extrapolated",
        "source_point_ids",
        "source_temperatures_C",
        "endpoint_temperatures_C",
        "source_point_orders",
        "qc_flag",
        "uncertainty_method",
        "source_uncertainty_methods",
        "correction_state",
    ]
    if has_errors:
        columns.extend(["lower_error", "upper_error"])
    return CombinedSpectrumTable(
        pd.DataFrame.from_records(records, columns=columns),
        history=spectrum.history
        + [
            {
                "operation": "resample_spectrum",
                "step_C": float(step_C),
                "method": method,
                "extrapolation": False,
                "source_extrapolation_flags": "preserved",
                "interpolation_endpoints": "latest warmer retained state at each observed target",
                "cross_excluded_segments": False,
                "uncertainty": "stored widths for sampled states; interpolated endpoints otherwise",
            }
        ],
    )
