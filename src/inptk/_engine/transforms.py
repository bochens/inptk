from __future__ import annotations

import copy
import warnings
from collections.abc import Mapping
from numbers import Integral
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from .math import (
    PROFILE_LIKELIHOOD_DROP_95,
    binomial_poisson_mle_with_profile_errors,
    cumulative_inp_per_ml_with_errors_from_counts,
    differential_inp_per_ml_per_c_from_counts,
    normalize_inp_air,
    normalize_inp_soil,
    temperature_threshold_edges,
    temperature_thresholds,
    water_blank_corrected_counts,
)
from .models import (
    CountsTable,
    CumulativeNucleusSpectrumTable,
    DifferentialNucleusSpectrumTable,
    MetadataLike,
    NormalizedInpSpectrumTable,
    ProcessingMetadata,
    SampleMetadata,
    SampleType,
    SpectrumBasis,
    TemperatureDependentTable,
    TemperatureFrozenFractionTable,
    processing_metadata_for,
)

TemperatureReductionMethod = Literal["max", "latest", "window_max_count"]
MleMaskMode = Literal["drop_rows", "rebase_counts"]
MLE_MASK_MODES = {"drop_rows", "rebase_counts"}
TableSequence = list[Any] | tuple[Any, ...]
TableMapping = dict[str, Any]


def _is_table_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _is_table_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _require_table_sequence(
    tables: TableSequence,
    allowed_types: tuple[type[Any], ...],
    name: str,
    *,
    allow_empty: bool = True,
) -> None:
    if not allow_empty and len(tables) == 0:
        raise ValueError(f"{name} cannot be empty")
    for index, table in enumerate(tables):
        if not isinstance(table, allowed_types):
            allowed = ", ".join(table_type.__name__ for table_type in allowed_types)
            raise TypeError(f"{name}[{index}] must be {allowed}")


def _map_table_shape(value: Any, mapper: Any) -> Any:
    if isinstance(value, dict):
        return {key: _map_table_shape(nested, mapper) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        return [_map_table_shape(nested, mapper) for nested in value]
    return mapper(value)


def _map_merge_shape(value: Any, merger: Any) -> Any:
    if isinstance(value, dict):
        return {key: _map_merge_shape(nested, merger) for key, nested in value.items()}
    return merger(value)


def _table_sample_ids_from_dataframe(df: pd.DataFrame) -> tuple[str, ...]:
    if "sample_id" not in df:
        return ()
    return tuple(str(value) for value in pd.Series(df["sample_id"]).dropna().unique())


def _source_cycles_from_tables(tables: TableSequence) -> tuple[str, ...]:
    cycles: list[str] = []
    for table in tables:
        processing = getattr(table, "processing_metadata", None)
        generated_by = getattr(processing, "generated_by", None)
        cycles.extend(getattr(generated_by, "source_cycles", ()) or ())
    return tuple(cycles)


def _plain_processing_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain_processing_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain_processing_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain_processing_value(item) for key, item in value.items()}
    return value


def _fraction_frozen_parameters(
    step_C: float,
    method: TemperatureReductionMethod,
    temperature_tolerance_C: float,
    cooling_only: bool,
) -> dict[str, Any]:
    return {
        "step_C": step_C,
        "method": method,
        "temperature_tolerance_C": temperature_tolerance_C,
        "cooling_only": cooling_only,
    }


def _metadata_dilutions(metadata_by_sample_id: dict[str, SampleMetadata]) -> tuple[Any, ...]:
    return tuple(metadata_by_sample_id[sample_id].dilution for sample_id in metadata_by_sample_id)


def _merged_fraction_input_for_stitch_or_mle(
    tables: TableSequence,
) -> TemperatureFrozenFractionTable:
    _require_table_sequence(
        tables,
        (TemperatureFrozenFractionTable,),
        "table",
        allow_empty=False,
    )
    frames = [
        table.to_dataframe()
        for table in tables
        if isinstance(table, TemperatureFrozenFractionTable)
    ]
    combined = pd.concat(frames, ignore_index=True)
    return TemperatureFrozenFractionTable.from_dataframe(
        combined,
        metadata=_merge_table_metadata(tables),
        processing_metadata=processing_metadata_for(
            "merge_fraction_inputs",
            inputs=tuple(tables),
            source_sample_ids=_table_sample_ids_from_dataframe(combined),
        ),
    )


def _merge_table_metadata(tables: TableSequence) -> dict[str, SampleMetadata] | None:
    metadata_by_sample_id: dict[str, SampleMetadata] = {}
    for table in tables:
        metadata = table.metadata
        if isinstance(metadata, dict):
            metadata_by_sample_id.update(copy.deepcopy(metadata))
            continue
        if isinstance(metadata, SampleMetadata):
            sample_ids = pd.Series(table.sample_id).astype(str).unique()
            if metadata.sample_id:
                metadata_by_sample_id[metadata.sample_id] = copy.deepcopy(metadata)
            elif len(sample_ids) == 1:
                metadata_by_sample_id[str(sample_ids[0])] = copy.deepcopy(metadata)
            else:
                raise KeyError("SampleMetadata.sample_id is required for multi-sample tables")
    return metadata_by_sample_id or None


def apply_water_blank_correction(
    table: CountsTable | TemperatureFrozenFractionTable | TableSequence | TableMapping,
    water_blank_frozen: Any,
) -> CountsTable | TemperatureFrozenFractionTable | list[Any] | dict[str, Any]:
    """Return a new count table after same-run water/DI blank correction.

    The water blank frozen count is subtracted from both n_frozen and n_total.
    The physical interpretation is that wells freezing in the water/DI blank are
    treated as unavailable sample wells at that temperature.
    """

    if isinstance(table, dict):
        return {
            key: apply_water_blank_correction(
                nested,
                water_blank_frozen[key]
                if isinstance(water_blank_frozen, dict)
                else water_blank_frozen,
            )
            for key, nested in table.items()
        }
    if isinstance(table, (list, tuple)):
        _require_table_sequence(table, (CountsTable, TemperatureFrozenFractionTable), "table")
        return [
            apply_water_blank_correction(single_table, water_blank_frozen) for single_table in table
        ]

    corrected_frozen, corrected_total = water_blank_corrected_counts(
        table.n_frozen,
        table.n_total,
        water_blank_frozen,
    )
    if isinstance(table, CountsTable):
        return CountsTable(
            sample_id=table.sample_id,
            temperature_C=table.temperature_C,
            n_total=corrected_total,
            n_frozen=corrected_frozen,
            time_s=table.time_s,
            cycle=table.cycle,
            observation_id=table.observation_id,
            metadata=table.metadata,
            processing_metadata=processing_metadata_for(
                "apply_water_blank_correction",
                inputs=(table,),
                parameters={"water_blank_frozen": _plain_processing_value(water_blank_frozen)},
                source_sample_ids=tuple(pd.Series(table.sample_id).astype(str).unique()),
            ),
        )
    if isinstance(table, TemperatureFrozenFractionTable):
        return TemperatureFrozenFractionTable(
            sample_id=table.sample_id,
            temperature_C=table.temperature_C,
            temperature_bin_width_C=table.temperature_bin_width_C,
            temperature_bin_method=table.temperature_bin_method,
            temperature_bin_left_C=table.temperature_bin_left_C,
            temperature_bin_right_C=table.temperature_bin_right_C,
            n_total=corrected_total,
            n_frozen=corrected_frozen,
            obs_count=table.obs_count,
            metadata=table.metadata,
            processing_metadata=processing_metadata_for(
                "apply_water_blank_correction",
                inputs=(table,),
                parameters={"water_blank_frozen": _plain_processing_value(water_blank_frozen)},
                source_sample_ids=tuple(pd.Series(table.sample_id).astype(str).unique()),
            ),
        )
    raise TypeError("table must be a CountsTable or TemperatureFrozenFractionTable")


def counts_to_temperature_frozen_fraction(
    counts: CountsTable | TableSequence | TableMapping,
    *,
    step_C: float = 0.5,
    method: TemperatureReductionMethod = "latest",
    temperature_tolerance_C: float | None = None,
    cooling_only: bool = True,
    **unexpected_kwargs: Any,
) -> TemperatureFrozenFractionTable | list[Any] | dict[str, Any]:
    """Reduce raw count observations to threshold-evaluated frozen-fraction table(s).

    Output rows are labeled by cold temperature thresholds, not centered bins. A
    row labeled ``-7.5`` means the cumulative state when cooling has reached
    ``-7.5 C``, with ``temperature_tolerance_C`` slack for probe jitter.
    ``method="max"`` uses the highest frozen fraction observed up to each
    threshold while preserving paired n_total/n_frozen counts.
    The default is ``method="latest"`` with zero temperature tolerance.
    Explicit ``max`` and ``window_max_count`` default to 0.05 and 0.01 C.
    ``method="latest"`` uses the last qualifying observation in time order
    (coldest qualifying observation when time is absent) and allows decreases.
    ``method="window_max_count"`` chooses the highest frozen count strictly
    within the target temperature +/- tolerance, falling back to the highest
    count warmer than target + tolerance when that window is empty. Ties use
    the last observation, and n_total stays paired with the chosen n_frozen.
    Its table includes four warmer zero rows and the first freezing observation
    rounded to 0.1 C, followed by regular thresholds. This adapts original OLAF's
    SpacedTempCSV.create_temp_csv count-selection rule, independently for each
    measurement/cycle, rather than reproducing OLAF's full processing workflow.
    By default, only the cooling phase is used: rows after each sample/cycle's
    coldest observed temperature are dropped before threshold reduction, so
    post-run warm-up cannot be re-counted at already-visited temperatures.
    Cycles must be processed separately; dictionary inputs preserve their keys.
    """

    if unexpected_kwargs:
        names = ", ".join(sorted(unexpected_kwargs))
        raise TypeError(
            f"Unexpected keyword argument(s): {names}. Cycles are always preserved by read_counts."
        )

    if temperature_tolerance_C is None:
        temperature_tolerance_C = {"latest": 0.0, "max": 0.05, "window_max_count": 0.01}.get(
            method, 0.0
        )

    return _map_table_shape(
        counts,
        lambda single_counts: _counts_to_temperature_frozen_fraction_one(
            single_counts,
            step_C=step_C,
            method=method,
            temperature_tolerance_C=temperature_tolerance_C,
            cooling_only=cooling_only,
        ),
    )


def _counts_to_temperature_frozen_fraction_one(
    counts: CountsTable,
    *,
    step_C: float,
    method: TemperatureReductionMethod,
    temperature_tolerance_C: float,
    cooling_only: bool,
) -> TemperatureFrozenFractionTable:
    if not isinstance(counts, CountsTable):
        raise TypeError("counts must be a CountsTable")
    if method not in ("max", "latest", "window_max_count"):
        raise ValueError("method must be 'max', 'latest', or 'window_max_count'")
    if step_C <= 0:
        raise ValueError("step_C must be positive")
    if temperature_tolerance_C < 0:
        raise ValueError("temperature_tolerance_C cannot be negative")

    reduction_label = f"cold_threshold_{method}_tol_{temperature_tolerance_C:g}C"
    if cooling_only:
        reduction_label += "_cooling_only"
    df = counts.to_dataframe()
    if df.empty:
        empty_table = _empty_temperature_frozen_fraction(
            step_C=step_C,
            temperature_bin_method=reduction_label,
            metadata=counts.metadata,
            processing_metadata=processing_metadata_for(
                "fraction_frozen",
                inputs=(counts,),
                parameters=_fraction_frozen_parameters(
                    step_C,
                    method,
                    temperature_tolerance_C,
                    cooling_only,
                ),
                source_sample_ids=tuple(pd.Series(counts.sample_id).astype(str).unique()),
            ),
        )
        return empty_table

    df = _valid_counts_dataframe(df)
    if df.empty:
        empty_table = _empty_temperature_frozen_fraction(
            step_C=step_C,
            temperature_bin_method=reduction_label,
            metadata=counts.metadata,
            processing_metadata=processing_metadata_for(
                "fraction_frozen",
                inputs=(counts,),
                parameters=_fraction_frozen_parameters(
                    step_C,
                    method,
                    temperature_tolerance_C,
                    cooling_only,
                ),
                source_sample_ids=tuple(pd.Series(counts.sample_id).astype(str).unique()),
            ),
        )
        return empty_table

    df = _with_cycle_key(df)
    cycle_keys = _cycle_keys(df)
    if len(cycle_keys) > 1:
        available = ", ".join(cycle_keys)
        raise ValueError(
            f"Multiple cycles found in one calculation input. Process each separately: {available}"
        )
    selected_df = df.drop(columns="_inptk_cycle_key")
    if cooling_only:
        selected_df = _cooling_phase_counts_dataframe(selected_df)
    return _counts_dataframe_to_temperature_frozen_fraction(
        selected_df,
        step_C=step_C,
        method=method,
        temperature_tolerance_C=temperature_tolerance_C,
        temperature_bin_method=reduction_label,
        metadata=counts.metadata,
        processing_metadata=processing_metadata_for(
            "fraction_frozen",
            inputs=(counts,),
            parameters=_fraction_frozen_parameters(
                step_C,
                method,
                temperature_tolerance_C,
                cooling_only,
            ),
            source_sample_ids=_table_sample_ids_from_dataframe(selected_df),
            source_cycles=tuple(cycle_keys),
        ),
    )


def _cooling_phase_counts_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return count rows through each sample's coldest observation."""

    if df.empty:
        return df.copy()
    frames = [
        _cooling_phase_sample_counts(sample_df)
        for _, sample_df in df.groupby("sample_id", sort=False)
    ]
    return pd.concat(frames, ignore_index=True) if frames else df.iloc[0:0].copy()


def _cooling_phase_sample_counts(sample_df: pd.DataFrame) -> pd.DataFrame:
    if sample_df.empty:
        return sample_df.copy()
    time_values = (
        pd.to_numeric(sample_df["time_s"], errors="coerce").to_numpy(dtype=float)
        if "time_s" in sample_df
        else np.array([], dtype=float)
    )
    if time_values.size and np.isfinite(time_values).any():
        ordered = sample_df.sort_values(
            ["time_s", "temperature_C"],
            ascending=[True, False],
        )
    else:
        ordered = sample_df.copy()
    ordered = ordered.reset_index(drop=True)
    temperatures = pd.to_numeric(ordered["temperature_C"], errors="coerce")
    valid_positions = np.flatnonzero(np.isfinite(temperatures.to_numpy(dtype=float)))
    if valid_positions.size == 0:
        return ordered.iloc[0:0].copy()
    valid_temperatures = temperatures.iloc[valid_positions].to_numpy(dtype=float)
    coldest_valid_offset = int(np.argmin(valid_temperatures))
    coldest_position = int(valid_positions[coldest_valid_offset])
    return ordered.iloc[: coldest_position + 1].copy()


def _counts_dataframe_to_temperature_frozen_fraction(
    df: pd.DataFrame,
    *,
    step_C: float,
    method: TemperatureReductionMethod,
    temperature_tolerance_C: float,
    temperature_bin_method: str,
    metadata: MetadataLike,
    processing_metadata: ProcessingMetadata,
) -> TemperatureFrozenFractionTable:
    if df.empty:
        return _empty_temperature_frozen_fraction(
            step_C=step_C,
            temperature_bin_method=temperature_bin_method,
            metadata=metadata,
            processing_metadata=processing_metadata,
        )

    reduced_frames = [
        _sample_counts_to_temperature_thresholds(
            sample_id,
            sample_df,
            step_C=step_C,
            method=method,
            temperature_tolerance_C=temperature_tolerance_C,
        )
        for sample_id, sample_df in df.groupby("sample_id", sort=False)
    ]
    nonempty_frames = [frame for frame in reduced_frames if not frame.empty]
    result = pd.concat(nonempty_frames, ignore_index=True) if nonempty_frames else pd.DataFrame()
    if result.empty:
        return _empty_temperature_frozen_fraction(
            step_C=step_C,
            temperature_bin_method=temperature_bin_method,
            metadata=metadata,
            processing_metadata=processing_metadata,
        )
    result = result.sort_values(["sample_id", "temperature_C"], ascending=[True, False])
    bin_left, bin_right = temperature_threshold_edges(
        result["temperature_C"].to_numpy(dtype=float, copy=True),
        step_C,
    )
    return TemperatureFrozenFractionTable(
        sample_id=result["sample_id"].to_numpy(dtype=object, copy=True),
        temperature_C=result["temperature_C"].to_numpy(dtype=float, copy=True),
        temperature_bin_width_C=step_C,
        temperature_bin_method=temperature_bin_method,
        temperature_bin_left_C=bin_left,
        temperature_bin_right_C=bin_right,
        n_total=result["n_total"].to_numpy(dtype=float, copy=True),
        n_frozen=result["n_frozen"].to_numpy(dtype=float, copy=True),
        obs_count=result["obs_count"].to_numpy(dtype=int, copy=True),
        metadata=metadata,
        processing_metadata=processing_metadata,
    )


def _sample_counts_to_temperature_thresholds(
    sample_id: Any,
    sample_df: pd.DataFrame,
    *,
    step_C: float,
    method: TemperatureReductionMethod,
    temperature_tolerance_C: float,
) -> pd.DataFrame:
    if sample_df.empty:
        return pd.DataFrame()

    if "time_s" in sample_df:
        sample_df = sample_df.sort_values(["time_s", "temperature_C"], ascending=[True, False])
    else:
        sample_df = sample_df.sort_values("temperature_C", ascending=False)
    sample_df = sample_df.reset_index(drop=True)

    if method == "window_max_count":
        return _sample_counts_to_window_max_count_thresholds(
            sample_id,
            sample_df,
            step_C=step_C,
            temperature_tolerance_C=temperature_tolerance_C,
        )

    thresholds = temperature_thresholds(sample_df["temperature_C"].to_numpy(dtype=float), step_C)
    if thresholds.size == 0:
        return pd.DataFrame()

    temperatures = sample_df["temperature_C"].to_numpy(dtype=float, copy=True)
    n_total = sample_df["n_total"].to_numpy(dtype=float, copy=True)
    n_frozen = sample_df["n_frozen"].to_numpy(dtype=float, copy=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = n_frozen / n_total

    rows: list[dict[str, Any]] = []
    previous_observation_count = 0
    for threshold in thresholds:
        observed_positions = np.flatnonzero(temperatures >= threshold - temperature_tolerance_C)
        if observed_positions.size == 0:
            continue
        selected_position = (
            _latest_fraction_max_position(fraction, observed_positions)
            if method == "max"
            else int(observed_positions[-1])
        )
        obs_count = max(1, int(observed_positions.size) - previous_observation_count)
        rows.append(
            {
                "sample_id": sample_id,
                "temperature_C": float(threshold),
                "n_total": n_total[selected_position],
                "n_frozen": n_frozen[selected_position],
                "obs_count": obs_count,
            }
        )
        previous_observation_count = int(observed_positions.size)
    return pd.DataFrame.from_records(rows)


def _sample_counts_to_window_max_count_thresholds(
    sample_id: Any,
    sample_df: pd.DataFrame,
    *,
    step_C: float,
    temperature_tolerance_C: float,
) -> pd.DataFrame:
    temperatures = sample_df["temperature_C"].to_numpy(dtype=float, copy=True)
    n_total = sample_df["n_total"].to_numpy(dtype=float, copy=True)
    n_frozen = sample_df["n_frozen"].to_numpy(dtype=float, copy=True)
    if temperatures.size == 0:
        return pd.DataFrame()

    positive_positions = np.flatnonzero(n_frozen > 0)
    if positive_positions.size == 0:
        return _sample_counts_to_window_max_count_zero_thresholds(
            sample_id,
            temperatures=temperatures,
            n_total=n_total,
            step_C=step_C,
        )

    first_frozen_position = int(positive_positions[0])
    first_frozen_temperature = round(float(temperatures[first_frozen_position]), 1)
    first_threshold = float(np.ceil(first_frozen_temperature / step_C) * step_C)
    coldest_temperature = float(np.min(temperatures))

    rows: list[dict[str, Any]] = []
    warm_total = n_total[first_frozen_position]
    for offset in range(4, 0, -1):
        rows.append(
            {
                "sample_id": sample_id,
                "temperature_C": first_threshold + offset * step_C,
                "n_total": warm_total,
                "n_frozen": 0.0,
                "obs_count": 1,
            }
        )

    rows.append(
        {
            "sample_id": sample_id,
            "temperature_C": first_frozen_temperature,
            "n_total": n_total[first_frozen_position],
            "n_frozen": n_frozen[first_frozen_position],
            "obs_count": 1,
        }
    )

    threshold = first_threshold
    while threshold - step_C > coldest_temperature:
        threshold -= step_C
        selected_position = _window_max_count_position(
            temperatures,
            n_frozen,
            threshold=threshold,
            temperature_tolerance_C=temperature_tolerance_C,
        )
        if selected_position is None:
            continue
        rows.append(
            {
                "sample_id": sample_id,
                "temperature_C": float(threshold),
                "n_total": n_total[selected_position],
                "n_frozen": n_frozen[selected_position],
                "obs_count": 1,
            }
        )

    return pd.DataFrame.from_records(rows)


def _sample_counts_to_window_max_count_zero_thresholds(
    sample_id: Any,
    *,
    temperatures: np.ndarray,
    n_total: np.ndarray,
    step_C: float,
) -> pd.DataFrame:
    thresholds = temperature_thresholds(temperatures, step_C)
    if thresholds.size == 0:
        return pd.DataFrame()
    warm_total = n_total[0]
    return pd.DataFrame.from_records(
        {
            "sample_id": sample_id,
            "temperature_C": float(threshold),
            "n_total": warm_total,
            "n_frozen": 0.0,
            "obs_count": 1,
        }
        for threshold in thresholds
    )


def _window_max_count_position(
    temperatures: np.ndarray,
    n_frozen: np.ndarray,
    *,
    threshold: float,
    temperature_tolerance_C: float,
) -> int | None:
    exact_band_positions = np.flatnonzero(
        (temperatures > threshold - temperature_tolerance_C)
        & (temperatures < threshold + temperature_tolerance_C)
    )
    selected = _latest_max_count_position(n_frozen, exact_band_positions)
    if selected is not None:
        return selected

    warm_side_positions = np.flatnonzero(temperatures > threshold + temperature_tolerance_C)
    return _latest_max_count_position(n_frozen, warm_side_positions)


def _latest_max_count_position(
    n_frozen: np.ndarray,
    positions: np.ndarray,
) -> int | None:
    if positions.size == 0:
        return None
    window = n_frozen[positions]
    finite = np.isfinite(window)
    if not finite.any():
        return int(positions[-1])
    max_count = np.max(window[finite])
    return int(positions[np.flatnonzero(finite & (window == max_count))[-1]])


def _latest_fraction_max_position(fraction: np.ndarray, positions: np.ndarray) -> int:
    window = fraction[positions]
    finite = np.isfinite(window)
    if not finite.any():
        return int(positions[-1])
    max_fraction = np.max(window[finite])
    return int(positions[np.flatnonzero(finite & (window == max_fraction))[-1]])


def _valid_counts_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df[
        np.isfinite(df["temperature_C"])
        & np.isfinite(df["n_total"])
        & np.isfinite(df["n_frozen"])
        & (df["n_total"] > 0)
        & (df["n_frozen"] >= 0)
        & (df["n_frozen"] <= df["n_total"])
    ].copy()


def _with_cycle_key(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    if "cycle" not in keyed:
        keyed["_inptk_cycle_key"] = "missing"
        return keyed
    keyed["_inptk_cycle_key"] = [
        _normalize_cycle_key(value) for value in keyed["cycle"].to_numpy(dtype=object)
    ]
    return keyed


def _normalize_cycle_key(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        if pd.isna(value):
            return "missing"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        numeric_value = float(value)
        if np.isfinite(numeric_value) and numeric_value.is_integer():
            return str(int(numeric_value))
        return str(numeric_value)

    text = str(value).strip()
    if not text:
        return "missing"
    try:
        numeric_value = float(text)
    except ValueError:
        return text
    if np.isfinite(numeric_value) and numeric_value.is_integer():
        return str(int(numeric_value))
    return str(numeric_value)


def _cycle_keys(df: pd.DataFrame) -> list[str]:
    return [str(value) for value in pd.unique(df["_inptk_cycle_key"])]


def _iter_cycle_dataframes(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    for cycle_key, cycle_df in df.groupby("_inptk_cycle_key", sort=False):
        frames.append((str(cycle_key), cycle_df.drop(columns="_inptk_cycle_key").copy()))
    return frames


def _empty_temperature_frozen_fraction(
    *,
    step_C: float,
    temperature_bin_method: str,
    metadata: MetadataLike,
    processing_metadata: ProcessingMetadata,
) -> TemperatureFrozenFractionTable:
    return TemperatureFrozenFractionTable(
        sample_id=np.array([], dtype=object),
        temperature_C=np.array([], dtype=float),
        temperature_bin_width_C=step_C,
        temperature_bin_method=temperature_bin_method,
        n_total=np.array([], dtype=float),
        n_frozen=np.array([], dtype=float),
        obs_count=np.array([], dtype=int),
        metadata=metadata,
        processing_metadata=processing_metadata,
    )


def temperature_frozen_fraction_to_differential_spectrum(
    table: TemperatureFrozenFractionTable | TableSequence | TableMapping,
    metadata_by_sample_id: dict[str, SampleMetadata] | None = None,
) -> DifferentialNucleusSpectrumTable | list[Any] | dict[str, Any]:
    """Calculate changes in cumulative concentration over adjacent temperatures.

    Output temperature labels are the warm side of each interval; explicit cold
    and warm edges retain its actual width. No interval is invented before the
    first input state. Corrected counts may have changing totals or decreasing
    fractions: negative differences remain visible and carry QC bit 2, while
    nonfinite results carry QC bit 1. Irregular temperature spacing is supported.
    This computes changes in the supplied corrected concentration; it does not
    establish that arbitrary droplet loss is a valid background correction.
    """

    if isinstance(table, (dict, list, tuple)):
        return _map_table_shape(
            table,
            lambda single_table: temperature_frozen_fraction_to_differential_spectrum(
                single_table,
                metadata_by_sample_id,
            ),
        )

    metadata_by_sample_id = resolve_metadata_by_sample_id(table, metadata_by_sample_id)
    frames: list[pd.DataFrame] = []
    for sample_id, sample_df in table.to_dataframe().groupby("sample_id", sort=False):
        sample_key = str(sample_id)
        if sample_key not in metadata_by_sample_id:
            raise KeyError(f"Missing SampleMetadata for sample_id {sample_key!r}")
        metadata = metadata_by_sample_id[sample_key]
        metadata.validate_for_count_to_suspension()
        sample_df = sample_df.sort_values("temperature_C", ascending=False).copy()
        temperatures = sample_df["temperature_C"].to_numpy(dtype=float)
        widths = -np.diff(temperatures)
        k_value = differential_inp_per_ml_per_c_from_counts(
            sample_df["n_frozen"].to_numpy(dtype=float, copy=True),
            sample_df["n_total"].to_numpy(dtype=float, copy=True),
            metadata.well_volume_uL or 0.0,
            widths,
            metadata.dilution or 0.0,
        )
        frames.append(pd.DataFrame({
            "sample_id": sample_df["sample_id"].to_numpy(dtype=object, copy=True)[1:],
            "temperature_C": temperatures[:-1],
            "temperature_bin_left_C": temperatures[1:],
            "temperature_bin_right_C": temperatures[:-1],
            "value": k_value,
            "value_unit": "INP_per_mL_suspension_per_C",
            "basis": "suspension",
            "qc_flag": np.where(np.isfinite(k_value), 0, 1) | np.where(k_value < 0, 2, 0),
        }))

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return DifferentialNucleusSpectrumTable(
            sample_id=np.array([], dtype=object),
            temperature_C=np.array([], dtype=float),
            temperature_bin_left_C=np.array([], dtype=float),
            temperature_bin_right_C=np.array([], dtype=float),
            temperature_bin_method=table.temperature_bin_method,
            value=np.array([], dtype=float),
            value_unit="INP_per_mL_suspension_per_C",
            basis="suspension",
            metadata=metadata_by_sample_id,
            processing_metadata=processing_metadata_for(
                "differential_spec",
                inputs=(table,),
                source_sample_ids=tuple(metadata_by_sample_id),
            ),
        )
    widths = (result["temperature_bin_right_C"] - result["temperature_bin_left_C"]).to_numpy()
    common_width = float(widths[0]) if np.allclose(widths, widths[0], rtol=0, atol=1e-9) else None
    return DifferentialNucleusSpectrumTable(
        sample_id=result["sample_id"].to_numpy(dtype=object, copy=True),
        temperature_C=result["temperature_C"].to_numpy(dtype=float, copy=True),
        temperature_bin_width_C=common_width,
        temperature_bin_method=table.temperature_bin_method,
        temperature_bin_left_C=result["temperature_bin_left_C"].to_numpy(dtype=float),
        temperature_bin_right_C=result["temperature_bin_right_C"].to_numpy(dtype=float),
        value=result["value"].to_numpy(dtype=float, copy=True),
        value_unit="INP_per_mL_suspension_per_C",
        basis="suspension",
        qc_flag=result["qc_flag"].to_numpy(dtype=int, copy=True),
        metadata=metadata_by_sample_id,
        processing_metadata=processing_metadata_for(
            "differential_spec",
            inputs=(table,),
            source_sample_ids=_table_sample_ids_from_dataframe(result),
        ),
    )


def temperature_frozen_fraction_to_cumulative_spectrum(
    table: TemperatureFrozenFractionTable | TableSequence | TableMapping,
    metadata_by_sample_id: dict[str, SampleMetadata] | None = None,
    *,
    z: float = 1.96,
) -> CumulativeNucleusSpectrumTable | list[Any] | dict[str, Any]:
    """Convert each dilution independently to cumulative K(T)."""

    if isinstance(table, (dict, list, tuple)):
        return _map_table_shape(
            table,
            lambda single_table: temperature_frozen_fraction_to_cumulative_spectrum(
                single_table,
                metadata_by_sample_id,
                z=z,
            ),
        )

    metadata_by_sample_id = resolve_metadata_by_sample_id(table, metadata_by_sample_id)
    frames: list[pd.DataFrame] = []
    for sample_id, sample_df in table.to_dataframe().groupby("sample_id", sort=False):
        sample_key = str(sample_id)
        if sample_key not in metadata_by_sample_id:
            raise KeyError(f"Missing SampleMetadata for sample_id {sample_key!r}")
        metadata = metadata_by_sample_id[sample_key]
        metadata.validate_for_count_to_suspension()
        sample_df = sample_df.sort_values("temperature_C", ascending=False).copy()
        inp_per_ml, lower_error, upper_error, finite_mask = (
            cumulative_inp_per_ml_with_errors_from_counts(
                sample_df["n_frozen"].to_numpy(dtype=float, copy=True),
                sample_df["n_total"].to_numpy(dtype=float, copy=True),
                metadata.well_volume_uL or 0.0,
                metadata.dilution or 0.0,
                z=z,
            )
        )
        out = pd.DataFrame(
            {
                "sample_id": sample_df["sample_id"].to_numpy(dtype=object, copy=True),
                "temperature_C": sample_df["temperature_C"].to_numpy(dtype=float, copy=True),
                "value": inp_per_ml,
                "value_unit": "INP_per_mL_suspension",
                "basis": "suspension",
                "lower_ci": lower_error,
                "upper_ci": upper_error,
                "dilution_fold": metadata.dilution,
                "qc_flag": np.where(finite_mask, 0, 1),
            }
        )
        copy_temperature_metadata(sample_df, out)
        frames.append(out)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return CumulativeNucleusSpectrumTable(
            sample_id=np.array([], dtype=object),
            temperature_C=np.array([], dtype=float),
            temperature_bin_width_C=table.temperature_bin_width_C,
            temperature_bin_method=table.temperature_bin_method,
            value=np.array([], dtype=float),
            value_unit="INP_per_mL_suspension",
            basis="suspension",
            metadata=metadata_by_sample_id,
            processing_metadata=processing_metadata_for(
                "cumulative_spec",
                inputs=(table,),
                parameters={"z": z},
                source_sample_ids=tuple(metadata_by_sample_id),
                source_dilutions=_metadata_dilutions(metadata_by_sample_id),
            ),
        )
    return CumulativeNucleusSpectrumTable.from_dataframe(
        result,
        value_unit="INP_per_mL_suspension",
        basis="suspension",
        metadata=metadata_by_sample_id,
        processing_metadata=processing_metadata_for(
            "cumulative_spec",
            inputs=(table,),
            parameters={"z": z},
            source_sample_ids=_table_sample_ids_from_dataframe(result),
            source_dilutions=_metadata_dilutions(metadata_by_sample_id),
        ),
    )


def temperature_frozen_fraction_to_stitched_cumulative_spectrum(
    table: TemperatureFrozenFractionTable | TableSequence | TableMapping,
    metadata_by_sample_id: dict[str, SampleMetadata] | None = None,
    *,
    sample_group_by: Literal["sample_id", "sample_name", "sample_long_name"]
    | dict[str, str]
    | None = None,
    z: float = 1.96,
    min_unfrozen: int = 3,
) -> CumulativeNucleusSpectrumTable | dict[str, Any]:
    """Select disjoint dilution ranges for one cumulative K(T) spectrum.

    Start with the least diluted curve and retain it through its coldest finite
    point with at least min_unfrozen unfrozen droplets. Use the next dilution
    only at colder temperatures. Missing points within a selected range stay
    missing; overlapping curves are never averaged or fitted together.
    """

    if isinstance(min_unfrozen, bool) or not isinstance(min_unfrozen, Integral) or min_unfrozen < 1:
        raise ValueError("min_unfrozen must be a whole number >= 1")

    if isinstance(table, dict):
        return _map_merge_shape(
            table,
            lambda nested: temperature_frozen_fraction_to_stitched_cumulative_spectrum(
                nested,
                metadata_by_sample_id,
                sample_group_by=sample_group_by,
                z=z,
                min_unfrozen=min_unfrozen,
            ),
        )
    if isinstance(table, (list, tuple)):
        table = _merged_fraction_input_for_stitch_or_mle(table)

    metadata_by_sample_id = resolve_metadata_by_sample_id(table, metadata_by_sample_id)
    per_dilution = temperature_frozen_fraction_to_cumulative_spectrum(
        table,
        metadata_by_sample_id,
        z=z,
    )
    source_df = cast(CumulativeNucleusSpectrumTable, per_dilution).to_dataframe()
    counts_df = table.to_dataframe()[["sample_id", "temperature_C", "n_total", "n_frozen"]].copy()
    _raise_on_duplicate_stitch_source_temperatures(source_df, counts_df)
    if source_df.empty:
        return CumulativeNucleusSpectrumTable(
            sample_id=np.array([], dtype=object),
            temperature_C=np.array([], dtype=float),
            temperature_bin_width_C=table.temperature_bin_width_C,
            temperature_bin_method=table.temperature_bin_method,
            value=np.array([], dtype=float),
            value_unit="INP_per_mL_suspension",
            basis="suspension",
            metadata={},
            processing_metadata=processing_metadata_for(
                "cumulative_spec_stitch",
                inputs=(table,),
                parameters={
                    "sample_group_by": _sample_group_by_parameter(sample_group_by),
                    "z": z,
                    "min_unfrozen": min_unfrozen,
                },
            ),
        )

    source_df["source_sample_id"] = source_df["sample_id"].astype(str)
    counts_df["source_sample_id"] = counts_df["sample_id"].astype(str)
    source_df = source_df.merge(
        counts_df.drop(columns="sample_id"),
        on=["source_sample_id", "temperature_C"],
        how="left",
        validate="one_to_one",
    )
    source_df["stitch_group_id"] = [
        _sample_group_id(sample_id, metadata_by_sample_id[sample_id], sample_group_by)
        for sample_id in source_df["source_sample_id"]
    ]

    frames: list[pd.DataFrame] = []
    output_metadata: dict[str, SampleMetadata] = {}
    for group_id, group_df in source_df.groupby("stitch_group_id", sort=False):
        group_sample_ids = group_df["source_sample_id"].astype(str).unique()
        group_metadata = [metadata_by_sample_id[sample_id] for sample_id in group_sample_ids]
        _warn_on_combined_metadata_mismatches(str(group_id), group_metadata, "stitch")
        output_metadata.setdefault(
            str(group_id),
            _combined_source_metadata(str(group_id), group_sample_ids, group_metadata, "stitch"),
        )
        stitched_group = _stitch_cumulative_group(
            group_df,
            str(group_id),
            min_unfrozen=min_unfrozen,
        )
        if not stitched_group.empty:
            frames.append(stitched_group)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return CumulativeNucleusSpectrumTable(
            sample_id=np.array([], dtype=object),
            temperature_C=np.array([], dtype=float),
            temperature_bin_width_C=table.temperature_bin_width_C,
            temperature_bin_method=table.temperature_bin_method,
            value=np.array([], dtype=float),
            value_unit="INP_per_mL_suspension",
            basis="suspension",
            metadata=output_metadata,
            processing_metadata=processing_metadata_for(
                "cumulative_spec_stitch",
                inputs=(table,),
                parameters={
                    "sample_group_by": _sample_group_by_parameter(sample_group_by),
                    "z": z,
                    "min_unfrozen": min_unfrozen,
                },
                source_sample_ids=_table_sample_ids_from_dataframe(source_df),
                source_dilutions=_metadata_dilutions(metadata_by_sample_id),
            ),
        )
    result = result.sort_values(["sample_id", "temperature_C"], ascending=[True, False])
    return CumulativeNucleusSpectrumTable.from_dataframe(
        result,
        value_unit="INP_per_mL_suspension",
        basis="suspension",
        metadata=output_metadata,
        processing_metadata=processing_metadata_for(
            "cumulative_spec_stitch",
            inputs=(table,),
            parameters={
                "sample_group_by": _sample_group_by_parameter(sample_group_by),
                "z": z,
                "min_unfrozen": min_unfrozen,
            },
            source_sample_ids=_table_sample_ids_from_dataframe(source_df),
            source_dilutions=_metadata_dilutions(metadata_by_sample_id),
        ),
    )


def temperature_frozen_fraction_to_binomial_mle_cumulative_spectrum(
    table: TemperatureFrozenFractionTable | TableSequence | TableMapping,
    metadata_by_sample_id: dict[str, SampleMetadata] | None = None,
    *,
    sample_group_by: Literal["sample_id", "sample_name", "sample_long_name"]
    | dict[str, str]
    | None = None,
    enforce_monotone: bool = False,
    confidence_drop: float = PROFILE_LIKELIHOOD_DROP_95,
    temperature_eligibility_C: Mapping[str, float] | None = None,
    mask_mode: MleMaskMode | None = None,
    likelihood_weights: Mapping[str, float] | None = None,
    action_counts: Mapping[str, float] | None = None,
    action_weight_lambda: float | None = None,
    action_weight_half_life: float | None = None,
) -> CumulativeNucleusSpectrumTable | dict[str, Any]:
    """Combine dilution rows with a binomial-Poisson MLE at each temperature.

    Each row contributes observed frozen counts ``x_j`` and total wells ``n_j``.
    The metadata dilution fold maps a shared original-sample concentration K(T)
    to the per-well freezing probability for that dilution:

    p_j(K) = 1 - exp(-K * well_volume_mL / dilution_j)

    When ``sample_group_by`` is omitted, groups are inferred from
    ``sample_long_name``/``sample_name`` by stripping one trailing numeric token,
    falling back to ``sample_id``. Use a dict for sample_id -> group_id mapping,
    or one of the metadata fields listed in the type annotation, when explicit
    grouping is needed.

    Each temperature is fitted separately. ``enforce_monotone=True`` is rejected
    because pooling cumulative counts across temperatures would treat repeated
    states of the same droplets as independent observations. ``lower_ci`` and
    ``upper_ci`` are profile-likelihood error widths, not absolute limits.

    ``temperature_eligibility_C`` maps measurement IDs to the warmest retained
    temperature. For example ``{"Sample_2": -20.0}`` applies the selected
    ``mask_mode`` to Sample_2 above ``-20 C``. ``mask_mode`` is
    required when ``temperature_eligibility_C`` is passed. ``"drop_rows"`` keeps
    the previous behavior: warmer rows are omitted, but retained colder rows keep
    their original cumulative frozen counts. ``"rebase_counts"`` also omits the
    warmer rows, then subtracts the warm-side cumulative frozen baseline from
    retained rows and removes those wells from ``n_total``. Use ``"rebase_counts"``
    when the warm-side events are treated as contamination rather than true
    original-sample INPs. ``likelihood_weights`` maps measurement IDs to
    direct likelihood weights. Alternatively,
    ``action_counts`` can be combined with ``action_weight_lambda`` or
    ``action_weight_half_life`` to compute exponential action-decay weights.
    """

    if enforce_monotone:
        raise ValueError(
            "MLE requires enforce_monotone=False: cumulative counts at adjacent "
            "temperatures describe the same droplets, not independent observations"
        )

    mle_parameters = _mle_processing_parameters(
        sample_group_by=sample_group_by,
        enforce_monotone=enforce_monotone,
        confidence_drop=confidence_drop,
        temperature_eligibility_C=temperature_eligibility_C,
        mask_mode=mask_mode,
        likelihood_weights=likelihood_weights,
        action_counts=action_counts,
        action_weight_lambda=action_weight_lambda,
        action_weight_half_life=action_weight_half_life,
    )

    if isinstance(table, dict):
        return _map_merge_shape(
            table,
            lambda nested: temperature_frozen_fraction_to_binomial_mle_cumulative_spectrum(
                nested,
                metadata_by_sample_id,
                sample_group_by=sample_group_by,
                enforce_monotone=enforce_monotone,
                confidence_drop=confidence_drop,
                temperature_eligibility_C=temperature_eligibility_C,
                mask_mode=mask_mode,
                likelihood_weights=likelihood_weights,
                action_counts=action_counts,
                action_weight_lambda=action_weight_lambda,
                action_weight_half_life=action_weight_half_life,
            ),
        )
    if isinstance(table, (list, tuple)):
        table = _merged_fraction_input_for_stitch_or_mle(table)

    metadata_by_sample_id = resolve_metadata_by_sample_id(table, metadata_by_sample_id)
    source_df = table.to_dataframe()
    if source_df.empty:
        return CumulativeNucleusSpectrumTable(
            sample_id=np.array([], dtype=object),
            temperature_C=np.array([], dtype=float),
            temperature_bin_width_C=table.temperature_bin_width_C,
            temperature_bin_method=table.temperature_bin_method,
            value=np.array([], dtype=float),
            value_unit="INP_per_mL_suspension",
            basis="suspension",
            metadata={},
            processing_metadata=processing_metadata_for(
                "cumulative_spec_mle",
                inputs=(table,),
                parameters=mle_parameters,
            ),
        )

    source_df["source_sample_id"] = source_df["sample_id"].astype(str)
    source_df["mle_group_id"] = [
        _sample_group_id(sample_id, metadata_by_sample_id[sample_id], sample_group_by)
        for sample_id in source_df["source_sample_id"]
    ]
    source_df = _apply_mle_temperature_eligibility(
        source_df,
        temperature_eligibility_C,
        mask_mode,
    )
    source_df["mle_likelihood_weight"] = _mle_likelihood_weights(
        source_df,
        likelihood_weights=likelihood_weights,
        action_counts=action_counts,
        action_weight_lambda=action_weight_lambda,
        action_weight_half_life=action_weight_half_life,
    )

    output_metadata: dict[str, SampleMetadata] = {}
    frames: list[pd.DataFrame] = []
    for group_id, mle_group_df in source_df.groupby("mle_group_id", sort=False):
        metadata_sample_ids = mle_group_df["source_sample_id"].astype(str).unique()
        metadata_rows = [metadata_by_sample_id[sample_id] for sample_id in metadata_sample_ids]
        _warn_on_combined_metadata_mismatches(str(group_id), metadata_rows, "mle")
        output_metadata.setdefault(
            str(group_id),
            _combined_source_metadata(str(group_id), metadata_sample_ids, metadata_rows, "mle"),
        )
        group_result = _independent_mle_cumulative_group(
            mle_group_df,
            str(group_id),
            metadata_by_sample_id,
            confidence_drop=confidence_drop,
        )
        if not group_result.empty:
            frames.append(group_result)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return CumulativeNucleusSpectrumTable(
            sample_id=np.array([], dtype=object),
            temperature_C=np.array([], dtype=float),
            temperature_bin_width_C=table.temperature_bin_width_C,
            temperature_bin_method=table.temperature_bin_method,
            value=np.array([], dtype=float),
            value_unit="INP_per_mL_suspension",
            basis="suspension",
            metadata=output_metadata,
            processing_metadata=processing_metadata_for(
                "cumulative_spec_mle",
                inputs=(table,),
                parameters=mle_parameters,
                source_sample_ids=_table_sample_ids_from_dataframe(source_df),
                source_dilutions=_metadata_dilutions(metadata_by_sample_id),
            ),
        )
    result = result.sort_values(["sample_id", "temperature_C"], ascending=[True, False])
    return CumulativeNucleusSpectrumTable.from_dataframe(
        result,
        value_unit="INP_per_mL_suspension",
        basis="suspension",
        metadata=output_metadata,
        processing_metadata=processing_metadata_for(
            "cumulative_spec_mle",
            inputs=(table,),
            parameters=mle_parameters,
            source_sample_ids=_table_sample_ids_from_dataframe(source_df),
            source_dilutions=_metadata_dilutions(metadata_by_sample_id),
        ),
    )


def cumulative_spectrum_to_normalized_inp_spectrum(
    spectrum: CumulativeNucleusSpectrumTable | TableSequence | TableMapping,
    metadata_by_sample_id: dict[str, SampleMetadata] | None = None,
) -> NormalizedInpSpectrumTable | list[Any] | dict[str, Any]:
    """Normalize cumulative INP/mL suspension values to each sample basis."""

    if isinstance(spectrum, (dict, list, tuple)):
        return _map_table_shape(
            spectrum,
            lambda single_spectrum: cumulative_spectrum_to_normalized_inp_spectrum(
                single_spectrum,
                metadata_by_sample_id,
            ),
        )

    if not np.all(np.asarray(spectrum.basis) == "suspension"):
        raise ValueError("Normalization requires a suspension-basis spectrum")

    metadata_by_sample_id = resolve_metadata_by_sample_id(spectrum, metadata_by_sample_id)
    frames: list[pd.DataFrame] = []
    for sample_id, sample_df in spectrum.to_dataframe().groupby("sample_id", sort=False):
        sample_key = str(sample_id)
        if sample_key not in metadata_by_sample_id:
            raise KeyError(f"Missing SampleMetadata for sample_id {sample_key!r}")
        metadata = metadata_by_sample_id[sample_key]
        value, value_unit, basis = normalize_inp_by_metadata(
            sample_df["value"].to_numpy(dtype=float, copy=True),
            metadata,
        )
        lower_ci = None
        upper_ci = None
        if "lower_ci" in sample_df:
            lower_ci, _, _ = normalize_inp_by_metadata(
                sample_df["lower_ci"].to_numpy(dtype=float, copy=True),
                metadata,
            )
        if "upper_ci" in sample_df:
            upper_ci, _, _ = normalize_inp_by_metadata(
                sample_df["upper_ci"].to_numpy(dtype=float, copy=True),
                metadata,
            )
        out = pd.DataFrame(
            {
                "sample_id": sample_df["sample_id"].to_numpy(dtype=object, copy=True),
                "temperature_C": sample_df["temperature_C"].to_numpy(dtype=float, copy=True),
                "value": value,
                "value_unit": value_unit,
                "basis": basis,
                "lower_ci": lower_ci,
                "upper_ci": upper_ci,
                "qc_flag": sample_df["qc_flag"].to_numpy(dtype=int, copy=True)
                if "qc_flag" in sample_df
                else np.zeros(len(sample_df), dtype=int),
            }
        )
        copy_temperature_metadata(sample_df, out)
        frames.append(out)

    result = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if result.empty:
        return NormalizedInpSpectrumTable(
            sample_id=np.array([], dtype=object),
            temperature_C=np.array([], dtype=float),
            temperature_bin_width_C=spectrum.temperature_bin_width_C,
            temperature_bin_method=spectrum.temperature_bin_method,
            value=np.array([], dtype=float),
            value_unit="unknown",
            basis="other",
            metadata=metadata_by_sample_id,
            processing_metadata=processing_metadata_for(
                "normalize_spec",
                inputs=(spectrum,),
                source_sample_ids=tuple(metadata_by_sample_id),
                source_dilutions=_metadata_dilutions(metadata_by_sample_id),
            ),
        )
    return NormalizedInpSpectrumTable.from_dataframe(
        result,
        metadata=metadata_by_sample_id,
        processing_metadata=processing_metadata_for(
            "normalize_spec",
            inputs=(spectrum,),
            source_sample_ids=_table_sample_ids_from_dataframe(result),
            source_dilutions=_metadata_dilutions(metadata_by_sample_id),
        ),
    )


def temperature_frozen_fraction_to_normalized_inp_spectrum(
    table: TemperatureFrozenFractionTable | TableSequence | TableMapping,
    metadata_by_sample_id: dict[str, SampleMetadata] | None = None,
    *,
    z: float = 1.96,
) -> NormalizedInpSpectrumTable | list[Any] | dict[str, Any]:
    """Convert temperature frozen fractions directly to normalized cumulative INP."""

    if isinstance(table, (dict, list, tuple)):
        return _map_table_shape(
            table,
            lambda single_table: temperature_frozen_fraction_to_normalized_inp_spectrum(
                single_table,
                metadata_by_sample_id,
                z=z,
            ),
        )

    cumulative = temperature_frozen_fraction_to_cumulative_spectrum(
        table,
        metadata_by_sample_id,
        z=z,
    )
    return cumulative_spectrum_to_normalized_inp_spectrum(cumulative, metadata_by_sample_id)


def resolve_metadata_by_sample_id(
    table: TemperatureDependentTable | CountsTable,
    metadata_by_sample_id: dict[str, SampleMetadata] | None,
) -> dict[str, SampleMetadata]:
    """Return a copied sample metadata mapping for a table."""

    if metadata_by_sample_id is not None:
        return _normalize_metadata_mapping(metadata_by_sample_id)
    if isinstance(table.metadata, SampleMetadata):
        metadata = copy.deepcopy(table.metadata)
        if not metadata.sample_id:
            sample_ids = pd.Series(table.sample_id).astype(str).unique()
            if len(sample_ids) != 1:
                raise KeyError("SampleMetadata.sample_id is required for multi-sample tables")
            return {str(sample_ids[0]): metadata}
        return {metadata.sample_id: metadata}
    if isinstance(table.metadata, dict):
        return _normalize_metadata_mapping(table.metadata)
    raise KeyError("SampleMetadata is required on the table or as metadata_by_sample_id")


def copy_temperature_metadata(source: pd.DataFrame, target: pd.DataFrame) -> None:
    """Copy temperature-bin metadata columns between intermediate frames."""

    for column in (
        "temperature_bin_width_C",
        "temperature_bin_method",
        "temperature_bin_left_C",
        "temperature_bin_right_C",
    ):
        if column in source:
            target[column] = source[column].to_numpy(copy=True)


def normalize_inp_by_metadata(
    inp_per_mL: np.ndarray,
    metadata: SampleMetadata,
) -> tuple[np.ndarray, str, SpectrumBasis]:
    """Normalize INP/mL suspension according to the sample type."""

    if metadata.sample_type == "air":
        metadata.validate_for_spectrum_normalization()
        return (
            normalize_inp_air(
                inp_per_mL,
                metadata.suspension_volume_mL or 0.0,
                metadata.air_volume_L or 0.0,
                metadata.filter_fraction_used or 0.0,
            ),
            "INP_per_L_air",
            "sampled_air",
        )
    if metadata.sample_type == "soil":
        metadata.validate_for_spectrum_normalization()
        return (
            normalize_inp_soil(
                inp_per_mL,
                metadata.suspension_volume_mL or 0.0,
                metadata.dry_mass_g or 0.0,
            ),
            "INP_per_g_dry_soil",
            "dry_soil",
        )
    return np.array(inp_per_mL, dtype=float, copy=True), "INP_per_mL_suspension", "suspension"


def _independent_mle_cumulative_group(
    group_df: pd.DataFrame,
    group_id: str,
    metadata_by_sample_id: dict[str, SampleMetadata],
    *,
    confidence_drop: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_df = group_df.sort_values("temperature_C", ascending=False)
    for temperature_C, temperature_df in group_df.groupby("temperature_C", sort=False):
        value, lower_error, upper_error, finite, dilution_fold = _mle_fit_for_rows(
            temperature_df,
            group_id,
            metadata_by_sample_id,
            confidence_drop=confidence_drop,
        )
        rows.append(
            _mle_result_row(
                group_id,
                float(cast(float, temperature_C)),
                value,
                lower_error,
                upper_error,
                dilution_fold,
                0 if finite else 1,
                temperature_df.iloc[0],
            )
        )
    return pd.DataFrame.from_records(rows)


def _mle_fit_for_rows(
    fit_df: pd.DataFrame,
    group_id: str,
    metadata_by_sample_id: dict[str, SampleMetadata],
    *,
    confidence_drop: float,
) -> tuple[float, float, float, bool, float]:
    if fit_df.empty:
        return np.nan, np.nan, np.nan, False, np.nan

    fit_sample_ids = fit_df["source_sample_id"].astype(str).to_numpy(copy=True)
    fit_metadata = [metadata_by_sample_id[sample_id] for sample_id in fit_sample_ids]
    well_volume_uL = _shared_well_volume_uL(fit_metadata, group_id)
    dilution = np.array([metadata.dilution or 0.0 for metadata in fit_metadata], dtype=float)
    if np.any(dilution <= 0):
        raise ValueError(f"All dilutions must be positive for MLE group {group_id!r}")

    likelihood_weight = (
        fit_df["mle_likelihood_weight"].to_numpy(dtype=float, copy=True)
        if "mle_likelihood_weight" in fit_df
        else np.ones(len(fit_df), dtype=float)
    )
    value, lower_error, upper_error, finite = binomial_poisson_mle_with_profile_errors(
        fit_df["n_frozen"].to_numpy(dtype=float, copy=True),
        fit_df["n_total"].to_numpy(dtype=float, copy=True),
        well_volume_uL,
        dilution,
        likelihood_weight=likelihood_weight,
        confidence_drop=confidence_drop,
    )
    dilution_fold = dilution[0] if len(pd.unique(dilution)) == 1 else np.nan
    return value, lower_error, upper_error, finite, dilution_fold


def _mle_result_row(
    group_id: str,
    temperature_C: float,
    value: float,
    lower_error: float,
    upper_error: float,
    dilution_fold: float,
    qc_flag: int,
    source_row: pd.Series,
) -> dict[str, Any]:
    out = {
        "sample_id": group_id,
        "temperature_C": temperature_C,
        "value": value,
        "value_unit": "INP_per_mL_suspension",
        "basis": "suspension",
        "lower_ci": lower_error,
        "upper_ci": upper_error,
        "dilution_fold": dilution_fold,
        "qc_flag": qc_flag,
    }
    _copy_temperature_row_metadata(source_row, out)
    return out


def _mle_processing_parameters(
    *,
    sample_group_by: Literal["sample_id", "sample_name", "sample_long_name"]
    | dict[str, str]
    | None,
    enforce_monotone: bool,
    confidence_drop: float,
    temperature_eligibility_C: Mapping[str, float] | None,
    mask_mode: MleMaskMode | None,
    likelihood_weights: Mapping[str, float] | None,
    action_counts: Mapping[str, float] | None,
    action_weight_lambda: float | None,
    action_weight_half_life: float | None,
) -> dict[str, Any]:
    if likelihood_weights is not None and action_counts is not None:
        raise ValueError(
            "Pass either likelihood_weights or action_counts, not both"
        )
    if action_counts is None and (
        action_weight_lambda is not None or action_weight_half_life is not None
    ):
        raise ValueError(
            "action_weight_lambda/action_weight_half_life require action_counts"
        )
    resolved_lambda = _resolve_action_weight_lambda(
        action_weight_lambda,
        action_weight_half_life,
        require=action_counts is not None,
    )
    temperature_eligibility = _normalize_measurement_mapping(
        temperature_eligibility_C,
        name="temperature_eligibility_C",
    )
    resolved_mask_mode = _resolve_mle_mask_mode(
        mask_mode,
        temperature_eligibility_C=temperature_eligibility_C,
    )
    return {
        "sample_group_by": _sample_group_by_parameter(sample_group_by),
        "enforce_monotone": enforce_monotone,
        "confidence_drop": confidence_drop,
        "temperature_eligibility_C": _plain_processing_value(temperature_eligibility),
        "mask_mode": resolved_mask_mode,
        "likelihood_weights": _plain_processing_value(
            _normalize_measurement_mapping(
                likelihood_weights,
                name="likelihood_weights",
                require_positive_values=True,
            )
        ),
        "action_counts": _plain_processing_value(
            _normalize_measurement_mapping(
                action_counts,
                name="action_counts",
                require_nonnegative_values=True,
            )
        ),
        "action_weight_lambda": resolved_lambda,
        "action_weight_half_life": action_weight_half_life,
    }


def _apply_mle_temperature_eligibility(
    source_df: pd.DataFrame,
    temperature_eligibility_C: Mapping[str, float] | None,
    mask_mode: MleMaskMode | None,
) -> pd.DataFrame:
    eligibility = _normalize_measurement_mapping(
        temperature_eligibility_C,
        name="temperature_eligibility_C",
    )
    if not eligibility or source_df.empty:
        return source_df
    resolved_mask_mode = _resolve_mle_mask_mode(
        mask_mode,
        temperature_eligibility_C=temperature_eligibility_C,
    )

    frames: list[pd.DataFrame] = []
    for sample_id, sample_df in source_df.groupby("source_sample_id", sort=False):
        source_sample_id = str(sample_id)
        warmest_allowed = eligibility.get(source_sample_id)
        if warmest_allowed is None:
            frames.append(sample_df.copy())
            continue

        kept_mask = sample_df["temperature_C"].astype(float) <= float(warmest_allowed)
        if resolved_mask_mode == "drop_rows":
            frames.append(sample_df.loc[kept_mask].copy())
            continue
        if resolved_mask_mode == "rebase_counts":
            frames.append(
                _rebase_mle_masked_counts(
                    sample_df,
                    kept_mask,
                    source_sample_id=source_sample_id,
                    warmest_allowed=float(warmest_allowed),
                )
            )
            continue
        raise AssertionError(f"Unhandled MLE mask mode: {resolved_mask_mode!r}")

    if not frames:
        return source_df.iloc[0:0].copy()
    return pd.concat(frames, ignore_index=True)


def _resolve_mle_mask_mode(
    mask_mode: MleMaskMode | None,
    *,
    temperature_eligibility_C: Mapping[str, float] | None,
) -> MleMaskMode | None:
    if temperature_eligibility_C is None:
        if mask_mode is not None:
            raise ValueError("mask_mode requires temperature_eligibility_C")
        return None
    if mask_mode is None:
        raise ValueError(
            "mask_mode is required when temperature_eligibility_C is passed; "
            "use 'drop_rows' or 'rebase_counts'"
        )
    if mask_mode not in MLE_MASK_MODES:
        raise ValueError("mask_mode must be 'drop_rows' or 'rebase_counts'")
    return mask_mode


def _rebase_mle_masked_counts(
    sample_df: pd.DataFrame,
    kept_mask: pd.Series,
    *,
    source_sample_id: str,
    warmest_allowed: float,
) -> pd.DataFrame:
    kept_df = sample_df.loc[kept_mask].copy()
    if kept_df.empty:
        return kept_df

    masked_df = sample_df.loc[~kept_mask]
    if masked_df.empty:
        return kept_df

    masked_temperatures = masked_df["temperature_C"].to_numpy(dtype=float, copy=True)
    coldest_masked_temperature = float(np.min(masked_temperatures))
    coldest_masked = np.isclose(
        masked_temperatures,
        coldest_masked_temperature,
        rtol=0.0,
        atol=1e-12,
    )
    baseline = float(
        masked_df.loc[coldest_masked, "n_frozen"].to_numpy(dtype=float, copy=True).max()
    )
    if not np.isfinite(baseline):
        raise ValueError(
            f"Cannot rebase MLE temperature mask for sample {source_sample_id!r}: "
            "masked n_frozen values must be finite"
        )
    if baseline < 0:
        raise ValueError(
            f"Cannot rebase MLE temperature mask for sample {source_sample_id!r}: "
            "masked n_frozen baseline cannot be negative"
        )
    if baseline == 0:
        return kept_df

    original_frozen = kept_df["n_frozen"].to_numpy(dtype=float, copy=True)
    original_total = kept_df["n_total"].to_numpy(dtype=float, copy=True)
    tolerance = 1e-9
    nonmonotone = (original_total > baseline + tolerance) & (original_frozen < baseline - tolerance)
    if np.any(nonmonotone):
        raise ValueError(
            f"Cannot rebase MLE temperature mask for sample {source_sample_id!r}: "
            f"kept cumulative n_frozen falls below the masked baseline above "
            f"{warmest_allowed:g} C"
        )

    rebasable = (original_total > baseline + tolerance) & (original_frozen >= baseline - tolerance)
    if not np.any(rebasable):
        return kept_df.iloc[0:0].copy()
    if not np.all(rebasable):
        kept_df = kept_df.loc[rebasable].copy()
        original_frozen = original_frozen[rebasable]
        original_total = original_total[rebasable]

    rebased_frozen = original_frozen - baseline
    rebased_total = original_total - baseline
    if np.any(rebased_frozen < -tolerance):
        raise ValueError(
            f"Cannot rebase MLE temperature mask for sample {source_sample_id!r}: "
            f"kept cumulative n_frozen falls below the masked baseline above "
            f"{warmest_allowed:g} C"
        )
    if np.any(rebased_frozen > rebased_total + tolerance):
        raise ValueError(
            f"Cannot rebase MLE temperature mask for sample {source_sample_id!r}: "
            "rebased n_frozen exceeds rebased n_total"
        )

    rebased_frozen = np.where(np.abs(rebased_frozen) <= tolerance, 0.0, rebased_frozen)
    kept_df["n_frozen"] = rebased_frozen
    kept_df["n_total"] = rebased_total
    if "fraction_frozen" in kept_df.columns:
        kept_df["fraction_frozen"] = kept_df["n_frozen"] / kept_df["n_total"]
    return kept_df


def _mle_likelihood_weights(
    source_df: pd.DataFrame,
    *,
    likelihood_weights: Mapping[str, float] | None,
    action_counts: Mapping[str, float] | None,
    action_weight_lambda: float | None,
    action_weight_half_life: float | None,
) -> np.ndarray:
    if source_df.empty:
        return np.array([], dtype=float)
    if likelihood_weights is not None and action_counts is not None:
        raise ValueError("Pass either likelihood_weights or action_counts, not both")

    direct_weights = _normalize_measurement_mapping(
        likelihood_weights,
        name="likelihood_weights",
        require_positive_values=True,
    )
    actions = _normalize_measurement_mapping(
        action_counts,
        name="action_counts",
        require_nonnegative_values=True,
    )
    action_lambda = _resolve_action_weight_lambda(
        action_weight_lambda,
        action_weight_half_life,
        require=actions is not None,
    )

    weights: list[float] = []
    for measurement_id in source_df["source_sample_id"].astype(str):
        if direct_weights is not None:
            weights.append(direct_weights.get(measurement_id, 1.0))
        elif actions is not None:
            action_count = actions.get(measurement_id, 0.0)
            weights.append(float(np.exp(-float(cast(float, action_lambda)) * action_count)))
        else:
            weights.append(1.0)
    return np.array(weights, dtype=float)


def _resolve_action_weight_lambda(
    action_weight_lambda: float | None,
    action_weight_half_life: float | None,
    *,
    require: bool,
) -> float | None:
    if action_weight_lambda is not None and action_weight_half_life is not None:
        raise ValueError("Pass action_weight_lambda or action_weight_half_life, not both")
    if action_weight_half_life is not None:
        if action_weight_half_life <= 0:
            raise ValueError("action_weight_half_life must be positive")
        return float(np.log(2.0) / action_weight_half_life)
    if action_weight_lambda is not None:
        if action_weight_lambda < 0:
            raise ValueError("action_weight_lambda cannot be negative")
        return float(action_weight_lambda)
    if require:
        raise ValueError(
            "action_counts requires action_weight_lambda or action_weight_half_life"
        )
    return None


def _normalize_measurement_mapping(
    values: Mapping[str, float] | None,
    *,
    name: str,
    require_positive_values: bool = False,
    require_nonnegative_values: bool = False,
) -> dict[str, float] | None:
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must map measurement names to numbers")
    normalized: dict[str, float] = {}
    for measurement_id, value in values.items():
        if not isinstance(measurement_id, str):
            raise TypeError(f"{name} keys must be measurement names (strings)")
        if not measurement_id.strip():
            raise ValueError(f"{name} measurement names must not be empty")
        if isinstance(value, bool):
            raise TypeError(f"{name} values must be finite numbers")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"{name} contains a non-finite value for {measurement_id!r}")
        if require_positive_values and number <= 0:
            raise ValueError(f"{name} values must be positive")
        if require_nonnegative_values and number < 0:
            raise ValueError(f"{name} values cannot be negative")
        normalized[measurement_id] = number
    return normalized


def _normalize_metadata_mapping(metadata: MetadataLike) -> dict[str, SampleMetadata]:
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dict of SampleMetadata objects")
    copied = copy.deepcopy(metadata)
    for key, value in copied.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be sample_id strings")
        if not isinstance(value, SampleMetadata):
            raise TypeError("metadata values must be SampleMetadata objects")
    return copied


def _raise_on_duplicate_stitch_source_temperatures(
    source_df: pd.DataFrame,
    counts_df: pd.DataFrame,
) -> None:
    """Raise a domain error before pandas reports a one-to-one merge failure."""

    for name, frame in (("cumulative", source_df), ("fraction", counts_df)):
        if frame.empty:
            continue
        duplicate_mask = frame.duplicated(["sample_id", "temperature_C"], keep=False)
        if not duplicate_mask.any():
            continue
        examples = _duplicate_key_examples(
            frame.loc[duplicate_mask],
            ["sample_id", "temperature_C"],
        )
        raise ValueError(
            f"Stitch input contains repeated {name} rows for the same source sample "
            f"and temperature: {examples}. INP-toolkit cannot stitch repeated experiments "
            "with the same source sample_id because it cannot tell whether the rows "
            "are replicate measurements or conflicting records. Give each experiment "
            "a unique source sample_id and aggregate replicates before stitching, run "
            "them separately, or use MLE if independent well counts should be fitted "
            "together."
        )


def _duplicate_key_examples(df: pd.DataFrame, columns: list[str]) -> str:
    duplicates = (
        df.loc[df.duplicated(columns, keep=False), columns]
        .drop_duplicates()
        .head(5)
        .to_dict(orient="records")
    )
    return ", ".join(
        "(" + ", ".join(f"{column}={row[column]!r}" for column in columns) + ")"
        for row in duplicates
    )


def _stitch_cumulative_group(
    group_df: pd.DataFrame,
    group_id: str,
    *,
    min_unfrozen: int = 3,
) -> pd.DataFrame:
    """Select one dilution per temperature using the coldest eligible handoff.

    Concentrations and error widths are copied from the selected curve. A gap
    does not cause an early switch, and later dilutions never replace warmer
    selected ranges. Final decrease handling is a separate workflow step.
    """
    group_df = _prepare_olaf_stitch_frame(group_df, min_unfrozen=min_unfrozen)
    if group_df.empty:
        return pd.DataFrame()
    if "source_sample_id" not in group_df:
        raise ValueError("Stitching requires source_sample_id for every input row")
    sources = group_df.loc[np.isfinite(group_df["value"]), "source_sample_id"]
    if not sources.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise ValueError("Finite stitch rows require nonempty source_sample_id names")
    if group_df.duplicated(["temperature_C", "dilution_fold"]).any():
        examples = _duplicate_key_examples(group_df, ["temperature_C", "dilution_fold"])
        raise ValueError(
            f"Stitch group {group_id!r} contains repeated temperature/dilution rows: "
            f"{examples}. Stitching requires at most one measurement at each "
            "temperature and dilution; analyze repeated measurements separately "
            "or use MLE for independent droplet sets."
        )

    temperatures = np.array(sorted(group_df["temperature_C"].unique(), reverse=True), dtype=float)
    dilutions = np.array(sorted(group_df["dilution_fold"].unique()), dtype=float)
    matrices = {
        name: _stitch_matrix(group_df, name, temperatures, dilutions)
        for name in ("value", "lower_ci", "upper_ci")
    }
    result = pd.DataFrame({
        "temperature_C": temperatures,
        "dilution_fold": dilutions[0],
        **{
            name: matrix[dilutions[0]].to_numpy(dtype=float, copy=True)
            for name, matrix in matrices.items()
        },
    })
    for dilution in dilutions[1:]:
        valid_indices = result.index[result["value"].notna()]
        start_index = int(valid_indices[-1]) + 1 if len(valid_indices) else 0
        if start_index >= len(result):
            continue
        result.loc[start_index:, "dilution_fold"] = dilution
        for name, matrix in matrices.items():
            result.loc[start_index:, name] = matrix[dilution].iloc[start_index:].to_numpy(
                dtype=float, copy=True
            )
    result["qc_flag"] = np.where(
        np.isfinite(result[["value", "lower_ci", "upper_ci"]]).all(axis=1), 0, 1
    )
    return pd.DataFrame.from_records([
        _stitch_row_from_result(group_id, row, group_df) for _, row in result.iterrows()
    ])


def _prepare_olaf_stitch_frame(
    group_df: pd.DataFrame,
    *,
    min_unfrozen: int = 3,
) -> pd.DataFrame:
    df = group_df.copy()
    df = df[np.isfinite(df["dilution_fold"])].copy()
    for column in ("value", "lower_ci", "upper_ci", "n_frozen", "n_total"):
        if column in df:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    finite = np.isfinite(df["value"])
    enough_unfrozen = (df["n_total"] - df["n_frozen"]) >= min_unfrozen
    df["stitch_selection_status"] = np.select(
        [~finite, ~enough_unfrozen],
        ["nonfinite_concentration", "insufficient_unfrozen"],
        default="selected",
    )
    df.loc[~(finite & enough_unfrozen), ["value", "lower_ci", "upper_ci"]] = np.nan
    return df.sort_values(["temperature_C", "dilution_fold"], ascending=[False, True])


def _stitch_matrix(
    df: pd.DataFrame,
    column: str,
    temperatures: np.ndarray,
    dilutions: np.ndarray,
) -> pd.DataFrame:
    return (
        df.pivot(index="temperature_C", columns="dilution_fold", values=column)
        .reindex(index=temperatures, columns=dilutions)
        .astype(float)
    )


def _stitch_row_from_result(
    group_id: str,
    row: pd.Series,
    group_df: pd.DataFrame,
) -> dict[str, Any]:
    temperature_C = float(row["temperature_C"])
    selected_dilution = float(row["dilution_fold"])
    source_row = _source_row_for_stitch_result(group_df, temperature_C, selected_dilution)
    exact_source = (
        float(source_row["temperature_C"]) == temperature_C
        and float(source_row["dilution_fold"]) == selected_dilution
    )
    selected_id = (
        str(source_row["source_sample_id"])
        if exact_source and np.isfinite(row["value"])
        else ""
    )
    status = (
        str(source_row["stitch_selection_status"])
        if exact_source else "temperature_unavailable"
    )
    qc_flag = int(row["qc_flag"])
    if exact_source and pd.notna(source_row.get("qc_flag")):
        qc_flag |= int(source_row["qc_flag"])
    out = {
        "sample_id": group_id,
        "temperature_C": temperature_C,
        "value": float(row["value"]) if pd.notna(row["value"]) else np.nan,
        "value_unit": str(source_row.get("value_unit", "INP_per_mL_suspension")),
        "basis": str(source_row.get("basis", "suspension")),
        "lower_ci": float(row["lower_ci"]) if pd.notna(row["lower_ci"]) else np.nan,
        "upper_ci": float(row["upper_ci"]) if pd.notna(row["upper_ci"]) else np.nan,
        "dilution_fold": selected_dilution,
        "qc_flag": qc_flag,
        "source_measurement_id": selected_id,
        "contributing_measurement_ids": [selected_id] if selected_id else [],
        "selection_status": status,
    }
    _copy_temperature_row_metadata(source_row, out)
    return out


def _source_row_for_stitch_result(
    group_df: pd.DataFrame,
    temperature_C: float,
    dilution_fold: Any,
) -> pd.Series:
    candidates = group_df[group_df["temperature_C"] == temperature_C]
    if pd.notna(dilution_fold):
        dilution_candidates = candidates[candidates["dilution_fold"] == float(dilution_fold)]
        if not dilution_candidates.empty:
            return dilution_candidates.iloc[0]
    if not candidates.empty:
        return candidates.iloc[0]
    return group_df.iloc[0]


def _copy_temperature_row_metadata(row: pd.Series, out: dict[str, Any]) -> None:
    for column in (
        "temperature_bin_width_C",
        "temperature_bin_method",
        "temperature_bin_left_C",
        "temperature_bin_right_C",
    ):
        if column in row and pd.notna(row[column]):
            out[column] = row[column]


def _sample_group_id(
    sample_id: str,
    metadata: SampleMetadata,
    sample_group_by: Literal["sample_id", "sample_name", "sample_long_name"]
    | dict[str, str]
    | None,
) -> str:
    if sample_group_by is None:
        return _inferred_sample_group_id(sample_id, metadata)
    if isinstance(sample_group_by, dict):
        if sample_id not in sample_group_by:
            raise KeyError(f"Missing sample group mapping for sample_id {sample_id!r}")
        group_id = str(sample_group_by[sample_id]).strip()
    else:
        if sample_group_by not in ("sample_id", "sample_name", "sample_long_name"):
            raise ValueError(
                "sample_group_by must be sample_id, sample_name, sample_long_name, or a dict"
            )
        group_id = str(getattr(metadata, sample_group_by)).strip()
    if not group_id:
        raise ValueError(f"Empty sample group id for sample_id {sample_id!r}")
    return group_id


def _sample_group_by_parameter(
    sample_group_by: Literal["sample_id", "sample_name", "sample_long_name"]
    | dict[str, str]
    | None,
) -> Any:
    return "inferred" if sample_group_by is None else _plain_processing_value(sample_group_by)


def _inferred_sample_group_id(sample_id: str, metadata: SampleMetadata) -> str:
    label = metadata.sample_long_name or metadata.sample_name
    if label:
        return _strip_trailing_numeric_token(label)
    return sample_id


def _strip_trailing_numeric_token(value: str) -> str:
    parts = str(value).rsplit("_", 1)
    if len(parts) != 2:
        return str(value)
    try:
        float(parts[1])
    except ValueError:
        return str(value)
    return parts[0]


def _shared_well_volume_uL(metadata_rows: list[SampleMetadata], group_id: str) -> float:
    values: list[float] = []
    for metadata in metadata_rows:
        metadata.validate_for_count_to_suspension()
        values.append(float(metadata.well_volume_uL or 0.0))
    if not np.allclose(values, values[0], rtol=0.0, atol=1e-12):
        raise ValueError(f"All wells in MLE group {group_id!r} must use the same well volume")
    return values[0]


def _warn_on_combined_metadata_mismatches(
    group_id: str,
    metadata_rows: list[SampleMetadata],
    operation: str,
) -> None:
    if len(metadata_rows) < 2:
        return
    for label, attr, raw_keys in (
        ("site", None, ("site",)),
        ("start_time", "collection_start", ("start_time",)),
        ("end_time", "collection_end", ("end_time",)),
        ("vol_air_filt", "air_volume_L", ("vol_air_filt", "air_volume_L")),
        (
            "proportion_filter_used",
            "filter_fraction_used",
            ("proportion_filter_used", "filter_fraction_used"),
        ),
        ("vol_susp", "suspension_volume_mL", ("vol_susp", "suspension_volume_mL")),
    ):
        _warn_on_metadata_mismatch(group_id, metadata_rows, operation, label, attr, raw_keys)


def _warn_on_metadata_mismatch(
    group_id: str,
    metadata_rows: list[SampleMetadata],
    operation: str,
    label: str,
    attr: str | None,
    raw_keys: tuple[str, ...],
) -> None:
    values = [
        value
        for value in (
            _metadata_warning_value(metadata, attr, raw_keys) for metadata in metadata_rows
        )
        if not _metadata_warning_missing(value)
    ]
    if len(values) < 2:
        return
    first = values[0]
    if all(_metadata_values_match(first, value) for value in values[1:]):
        return
    display_values = ", ".join(str(value) for value in values)
    warnings.warn(
        f"{operation} group {group_id!r} combines source metadata with different "
        f"{label} values: {display_values}. Different dilutions are expected, but "
        "site, start/end time, sampled air volume, filter fraction, and suspension "
        "volume should usually match when combining one physical sample.",
        UserWarning,
        stacklevel=3,
    )


def _metadata_warning_value(
    metadata: SampleMetadata,
    attr: str | None,
    raw_keys: tuple[str, ...],
) -> Any:
    value = getattr(metadata, attr, None) if attr else None
    if not _metadata_warning_missing(value):
        return value
    normalized_keys = _metadata_warning_key_variants(raw_keys)
    for mapping in (metadata.raw_sample_metadata, metadata.raw_preamble):
        lookup = {str(key).casefold(): item for key, item in (mapping or {}).items()}
        for key in normalized_keys:
            if key in lookup and not _metadata_warning_missing(lookup[key]):
                return lookup[key]
    return None


def _metadata_warning_key_variants(keys: tuple[str, ...]) -> tuple[str, ...]:
    variants: list[str] = []
    for key in keys:
        variants.extend((key, key.replace("_", " "), key.replace(" ", "_")))
    return tuple(dict.fromkeys(value.casefold() for value in variants))


def _metadata_warning_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _combined_source_metadata(
    group_id: str,
    source_sample_ids: np.ndarray,
    source_metadata: list[SampleMetadata],
    source_prefix: str,
) -> SampleMetadata:
    _ = source_prefix
    _ = source_sample_ids
    return SampleMetadata(
        sample_id=group_id,
        sample_name=group_id,
        sample_long_name=group_id,
        sample_type=_shared_sample_type(source_metadata),
        well_volume_uL=_shared_metadata_value(source_metadata, "well_volume_uL"),
        reset_temperature_C=_shared_metadata_value(source_metadata, "reset_temperature_C"),
        air_volume_L=_shared_metadata_value(source_metadata, "air_volume_L"),
        filter_fraction_used=_shared_metadata_value(source_metadata, "filter_fraction_used"),
        suspension_volume_mL=_shared_metadata_value(source_metadata, "suspension_volume_mL"),
        dry_mass_g=_shared_metadata_value(source_metadata, "dry_mass_g"),
        total_cells=_shared_metadata_value(source_metadata, "total_cells"),
        dilution=_shared_metadata_value(source_metadata, "dilution"),
    )


def _shared_sample_type(metadata_rows: list[SampleMetadata]) -> SampleType:
    sample_type = _shared_metadata_value(metadata_rows, "sample_type")
    return cast(SampleType, str(sample_type)) if sample_type else "other"


def _shared_metadata_value(metadata_rows: list[SampleMetadata], field: str) -> Any:
    if not metadata_rows:
        return None
    first = getattr(metadata_rows[0], field)
    for metadata in metadata_rows[1:]:
        if not _metadata_values_match(first, getattr(metadata, field)):
            return None
    return copy.deepcopy(first)


def _metadata_values_match(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    if isinstance(left, (int, float, np.integer, np.floating)) or isinstance(
        right,
        (int, float, np.integer, np.floating),
    ):
        return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))
    return left == right


def _array_or_none(df: pd.DataFrame, column: str) -> np.ndarray | None:
    return df[column].to_numpy(dtype=float, copy=True) if column in df else None


def _scalar_or_none(df: pd.DataFrame, column: str) -> float | None:
    if column not in df:
        return None
    values = pd.Series(df[column]).dropna().unique()
    if len(values) == 0:
        return None
    if len(values) > 1:
        raise ValueError(f"{column} must have one value")
    return float(values[0])


def _scalar_string_or_empty(df: pd.DataFrame, column: str) -> str:
    if column not in df:
        return ""
    values = pd.Series(df[column]).dropna().unique()
    if len(values) == 0:
        return ""
    if len(values) > 1:
        raise ValueError(f"{column} must have one value")
    return str(values[0])
