from __future__ import annotations

import csv
import re
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .models import (
    CountsTable,
    SampleMetadata,
    TemperatureFrozenFractionTable,
    processing_metadata_for,
)


CountInputFormat = Literal["auto", "long", "canonical", "icescopy", "icescopy_wide", "wide"]
CountCyclePolicy = Literal["single", "pooled", "preserve"]
CountColumn = Literal[
    "sample_id",
    "temperature_C",
    "n_total",
    "n_frozen",
    "time_s",
    "cycle",
    "observation_id",
]
CountColumnMap = dict[CountColumn, str]
LONG_COUNT_COLUMNS = {"sample_id", "temperature_C", "n_total", "n_frozen"}
SAMPLE_METADATA_FIELDS = {field.name for field in fields(SampleMetadata)}

__all__ = [
    "infer_dilution_groups",
    "infer_icescopy_dilution_group_map",
    "map_count_columns",
    "metadata_frame",
    "parse_icescopy_wide_temperature_sync",
    "parse_olaf_frozen_at_temp",
    "parse_olaf_frozen_at_temp_csv",
    "parse_sync_wide",
    "read_counts",
    "read_commented_preamble",
    "read_icescopy_freeze_count_timeseries_csv",
    "read_icescopy_sample_metadata",
    "read_icescopy_temperature_sync_csv",
    "read_metadata",
    "read_preamble",
    "read_sync",
    "sample_metadata_to_dataframe",
    "split_metadata_rows",
    "strip_temperature_sync_metadata_rows",
    "tables_to_dataframe",
]

SAMPLE_VALUE_RE = re.compile(r"^(?P<sample>.+?)\s+number\s+(?P<kind>total|frozen)$")
SAMPLE_HEADER_RE = re.compile(r"^(?P<sample>.+?)\s+\(n=(?P<n>[^)]+)\)$")
METADATA_ROW_LABELS = {
    "sample_id",
    "cell_number",
    "sample_name",
    "sample_long_name",
    "collection_start",
    "collection_end",
    "sample_type",
    "dilution",
    "air_volume_L",
    "filter_fraction_used",
    "suspension_volume_mL",
    "dry_mass_g",
    "sample_note",
}


def read_preamble(path: str | Path) -> dict[str, str]:
    """Read leading '# key: value' session preamble lines from a CSV file."""

    preamble: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if _is_sample_metadata_row(body):
                continue
            if ":" in body:
                key, value = body.split(":", 1)
                preamble[key.strip()] = value.strip()
    return preamble


def read_sync(path: str | Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Read an Icescopy temperature-sync CSV with optional commented preamble."""

    preamble = read_preamble(path)
    df = pd.read_csv(path, comment="#")
    return df, preamble


def read_metadata(
    path: str | Path,
) -> tuple[dict[str, str], dict[str, SampleMetadata]]:
    """Read Icescopy commented session/sample metadata from a freeze-count CSV."""

    session_metadata: dict[str, str] = {}
    sample_rows: dict[str, list[str]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            body = line[1:].strip()
            if _is_sample_metadata_row(body):
                values = _split_csv_metadata_row(body)
                sample_rows[values[0]] = values[1:]
                continue
            if ":" in body:
                key, value = body.split(":", 1)
                session_metadata[key.strip()] = value.strip()
                continue

    metadata_by_sample_id: dict[str, SampleMetadata] = {}
    row_count = max((len(values) for values in sample_rows.values()), default=0)
    for index in range(row_count):
        raw_sample_metadata = {
            key: values[index] if index < len(values) else ""
            for key, values in sample_rows.items()
        }
        sample_id = (
            raw_sample_metadata.get("sample_name")
            or raw_sample_metadata.get("sample_id")
            or str(index)
        )
        metadata_by_sample_id[sample_id] = SampleMetadata(
            format_name=session_metadata.get("format_name", ""),
            file_version=session_metadata.get("file_version", ""),
            project_name=session_metadata.get("project_name", ""),
            user_name=session_metadata.get("user_name", ""),
            institution=session_metadata.get("institution", ""),
            date=session_metadata.get("analysis_date", session_metadata.get("date", "")),
            well_volume_uL=session_metadata.get("well_volume_uL"),
            reset_temperature_C=session_metadata.get("reset_temperature_C"),
            sample_id=sample_id,
            sample_name=raw_sample_metadata.get("sample_name", ""),
            sample_long_name=raw_sample_metadata.get("sample_long_name", ""),
            collection_start=raw_sample_metadata.get("collection_start", ""),
            collection_end=raw_sample_metadata.get("collection_end", ""),
            sample_type=raw_sample_metadata.get("sample_type", "other"),
            dilution=raw_sample_metadata.get("dilution"),
            air_volume_L=raw_sample_metadata.get("air_volume_L"),
            filter_fraction_used=raw_sample_metadata.get("filter_fraction_used"),
            suspension_volume_mL=raw_sample_metadata.get("suspension_volume_mL"),
            dry_mass_g=raw_sample_metadata.get("dry_mass_g"),
            total_cells=raw_sample_metadata.get("cell_number"),
            raw_preamble=session_metadata,
            raw_sample_metadata=raw_sample_metadata,
        )
    return session_metadata, metadata_by_sample_id


def read_counts(
    source: str | Path | pd.DataFrame,
    *,
    format: CountInputFormat = "auto",
    columns: CountColumnMap | None = None,
    metadata: Any = None,
    cycle_policy: CountCyclePolicy = "single",
    cycle: Any | None = None,
) -> CountsTable | list[CountsTable] | dict[str, CountsTable | list[CountsTable]]:
    """Read count observations into sample/dilution table(s).

    Supported inputs:
    - canonical long tables with sample_id, temperature_C, n_total, n_frozen
    - arbitrary long tables with a user-supplied columns mapping
    - Icescopy wide freeze_count_timeseries exports with "number total/frozen" columns
    - optional metadata as SampleMetadata, dict[str, SampleMetadata],
      dict[str, dict], metadata DataFrame, or a dict of common metadata defaults

    ``cycle_policy="single"`` selects one cycle and returns a table or dilution
    list. ``cycle_policy="pooled"`` marks all cycles for pooled threshold
    reduction and returns a table or dilution list. ``cycle_policy="preserve"``
    returns ``dict[cycle_id, table_or_dilution_list]``.
    """

    df = source.copy() if isinstance(source, pd.DataFrame) else read_sync(source)[0]
    metadata_source = metadata if metadata is not None else _metadata_from_path(source)
    if columns is not None:
        mapped_df = map_count_columns(df, columns)
        count_tables = _count_tables_from_long_dataframe(mapped_df, metadata_source)
        return _apply_count_cycle_policy(count_tables, cycle_policy=cycle_policy, cycle=cycle)

    resolved_format = _resolve_counts_format(df, format)

    if resolved_format == "long":
        count_tables = _count_tables_from_long_dataframe(df, metadata_source)
        return _apply_count_cycle_policy(count_tables, cycle_policy=cycle_policy, cycle=cycle)

    count_tables = parse_sync_wide(df, metadata=metadata_source)
    return _apply_count_cycle_policy(count_tables, cycle_policy=cycle_policy, cycle=cycle)


def map_count_columns(df: pd.DataFrame, columns: CountColumnMap) -> pd.DataFrame:
    """Map an arbitrary long count table into UFOLAF's canonical column names."""

    unknown = sorted(set(columns) - _count_column_names())
    if unknown:
        raise ValueError(f"Unknown canonical count columns: {', '.join(unknown)}")
    required = {"sample_id", "temperature_C", "n_total", "n_frozen"}
    missing_required = sorted(required - set(columns))
    if missing_required:
        raise ValueError(f"Column mapping is missing: {', '.join(missing_required)}")
    missing_input = sorted({source for source in columns.values() if source not in df.columns})
    if missing_input:
        raise ValueError(f"Input table is missing mapped columns: {', '.join(missing_input)}")

    renamed = {source: target for target, source in columns.items()}
    mapped = df[list(columns.values())].rename(columns=renamed)
    return mapped.loc[:, [column for column in _ordered_count_columns() if column in mapped]]


def _count_tables_from_long_dataframe(
    df: pd.DataFrame,
    metadata: Any,
) -> list[CountsTable]:
    tables: list[CountsTable] = []
    groups = [
        (str(sample_id), sample_df)
        for sample_id, sample_df in df.groupby("sample_id", sort=False)
    ]
    metadata_by_sample = _metadata_mapping_for_sample_ids(
        metadata,
        [sample_id for sample_id, _ in groups],
    )
    for sample_key, sample_df in groups:
        tables.append(
            CountsTable.from_dataframe(
                sample_df.reset_index(drop=True),
                metadata=_metadata_for_sample_id(metadata_by_sample, sample_key),
                processing_metadata=processing_metadata_for(
                    "read_counts_input",
                    parameters={"format": "long"},
                    source_sample_ids=(sample_key,),
                    source_cycles=_cycle_keys(_with_cycle_key(sample_df)),
                ),
            )
        )
    return tables


def _apply_count_cycle_policy(
    tables: list[CountsTable],
    *,
    cycle_policy: CountCyclePolicy,
    cycle: Any | None,
) -> CountsTable | list[CountsTable] | dict[str, CountsTable | list[CountsTable]]:
    if cycle_policy not in ("single", "pooled", "preserve"):
        raise ValueError("cycle_policy must be 'single', 'pooled', or 'preserve'")
    if cycle is not None and cycle_policy != "single":
        raise ValueError("cycle can only be selected when cycle_policy='single'")
    if cycle_policy == "pooled":
        return _single_or_list(
            [
                _with_cycle_policy_processing(
                    table,
                    cycle_policy="pooled",
                    source_cycles=_cycle_keys(_with_cycle_key(table.to_dataframe())),
                )
                for table in tables
            ]
        )
    if cycle_policy == "single":
        return _single_or_list(
            [
                _select_counts_cycle(table, cycle)
                for table in tables
            ]
        )
    return _preserve_count_cycles(tables)


def _preserve_count_cycles(tables: list[CountsTable]) -> dict[str, CountsTable | list[CountsTable]]:
    grouped: dict[str, list[CountsTable]] = {}
    for table in tables:
        keyed = _with_cycle_key(table.to_dataframe())
        for cycle_key, cycle_df in _iter_cycle_dataframes(keyed):
            grouped.setdefault(cycle_key, []).append(
                CountsTable.from_dataframe(
                    cycle_df.reset_index(drop=True),
                    metadata=table.metadata,
                    processing_metadata=processing_metadata_for(
                        "read_counts",
                        inputs=(table,),
                        parameters={"cycle_policy": "preserve"},
                        source_sample_ids=_sample_ids_from_dataframe(cycle_df),
                        source_cycles=(cycle_key,),
                    ),
                )
            )
    return {cycle_key: _single_or_list(cycle_tables) for cycle_key, cycle_tables in grouped.items()}


def _select_counts_cycle(table: CountsTable, cycle: Any | None) -> CountsTable:
    keyed = _with_cycle_key(table.to_dataframe())
    cycle_keys = _cycle_keys(keyed)
    selected_cycle = _selected_cycle_key(cycle_keys, cycle)
    selected_df = keyed[keyed["_ufolaf_cycle_key"] == selected_cycle].drop(
        columns="_ufolaf_cycle_key"
    )
    return CountsTable.from_dataframe(
        selected_df.reset_index(drop=True),
        metadata=table.metadata,
        processing_metadata=processing_metadata_for(
            "read_counts",
            inputs=(table,),
            parameters={"cycle_policy": "single", "cycle": selected_cycle},
            source_sample_ids=_sample_ids_from_dataframe(selected_df),
            source_cycles=(selected_cycle,),
        ),
    )


def _single_or_list(tables: list[CountsTable]) -> CountsTable | list[CountsTable]:
    return tables[0] if len(tables) == 1 else tables


def _with_cycle_policy_processing(
    table: CountsTable,
    *,
    cycle_policy: CountCyclePolicy,
    source_cycles: list[str],
) -> CountsTable:
    return CountsTable(
        sample_id=table.sample_id,
        temperature_C=table.temperature_C,
        n_total=table.n_total,
        n_frozen=table.n_frozen,
        time_s=table.time_s,
        cycle=table.cycle,
        observation_id=table.observation_id,
        metadata=table.metadata,
        processing_metadata=processing_metadata_for(
            "read_counts",
            inputs=(table,),
            parameters={"cycle_policy": cycle_policy},
            source_sample_ids=_sample_ids_from_dataframe(table.to_dataframe()),
            source_cycles=tuple(source_cycles),
        ),
    )


def _sample_ids_from_dataframe(df: pd.DataFrame) -> tuple[str, ...]:
    if "sample_id" not in df:
        return ()
    return tuple(str(value) for value in pd.Series(df["sample_id"]).dropna().unique())


def metadata_frame(metadata_source: Any) -> pd.DataFrame:
    """Return sample metadata as a human-readable DataFrame."""

    metadata_by_sample_id = _metadata_mapping_from_source(metadata_source)
    return pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "sample_name": metadata.sample_name,
                "sample_long_name": metadata.sample_long_name,
                "sample_type": metadata.sample_type,
                "dilution": metadata.dilution,
                "well_volume_uL": metadata.well_volume_uL,
                "air_volume_L": metadata.air_volume_L,
                "filter_fraction_used": metadata.filter_fraction_used,
                "suspension_volume_mL": metadata.suspension_volume_mL,
                "dry_mass_g": metadata.dry_mass_g,
            }
            for sample_id, metadata in metadata_by_sample_id.items()
        ]
    )


def tables_to_dataframe(table_or_tables: Any) -> pd.DataFrame:
    """Return a display DataFrame from one UFOLAF table or a list of tables."""

    if isinstance(table_or_tables, dict):
        frames = []
        for group_id, value in table_or_tables.items():
            frame = tables_to_dataframe(value)
            if not frame.empty:
                frame.insert(0, "group_id", group_id)
            frames.append(frame)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if isinstance(table_or_tables, (list, tuple)):
        frames = [table.to_dataframe() for table in table_or_tables]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not hasattr(table_or_tables, "to_dataframe"):
        raise TypeError("Expected a UFOLAF table or a list of UFOLAF tables")
    return table_or_tables.to_dataframe()


def infer_dilution_groups(
    metadata_by_sample_id: dict[str, SampleMetadata],
) -> dict[str, str]:
    """Infer parent sample IDs from Icescopy long names like CRG_M1_13."""

    return {
        sample_id: _strip_trailing_numeric_token(
            metadata.sample_long_name or metadata.sample_name or sample_id
        )
        for sample_id, metadata in metadata_by_sample_id.items()
    }


def split_metadata_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split sample metadata rows from temperature-sync data rows."""

    if df.empty:
        return df.copy(), df.iloc[0:0].copy()
    first_col = df.columns[0]
    labels = df[first_col].astype(str)
    metadata_mask = labels.isin(METADATA_ROW_LABELS)
    metadata_rows = df.loc[metadata_mask].copy()
    data_rows = df.loc[~metadata_mask].copy()
    return data_rows.reset_index(drop=True), metadata_rows.reset_index(drop=True)


def parse_sync_wide(
    df: pd.DataFrame,
    *,
    metadata: Any = None,
) -> list[CountsTable]:
    """Convert an Icescopy wide temperature-sync table into one CountsTable per sample."""

    data_rows, _ = split_metadata_rows(df)
    time_s = _time_seconds(data_rows)
    sample_columns: dict[str, dict[str, str]] = {}
    for column in data_rows.columns:
        match = SAMPLE_VALUE_RE.match(str(column))
        if not match:
            continue
        sample_key = match.group("sample").strip()
        kind = match.group("kind")
        sample_columns.setdefault(sample_key, {})[kind] = column

    sample_ids = [
        _sample_id_from_header(sample_key)
        for sample_key, columns in sample_columns.items()
        if "total" in columns and "frozen" in columns
    ]
    metadata_by_sample = _metadata_mapping_for_sample_ids(metadata, sample_ids)

    tables: list[CountsTable] = []
    for sample_key, columns in sample_columns.items():
        if "total" not in columns or "frozen" not in columns:
            continue
        sample_id = _sample_id_from_header(sample_key)
        records: list[dict[str, Any]] = []
        for row_index, row in data_rows.iterrows():
            total = _to_float_or_nan(row[columns["total"]])
            frozen = _to_float_or_nan(row[columns["frozen"]])
            if not np.isfinite(total) or not np.isfinite(frozen):
                continue
            records.append(
                {
                    "sample_id": sample_id,
                    "temperature_C": _to_float_or_nan(row.get("temperature_C", np.nan)),
                    "time_s": time_s[row_index],
                    "n_total": total,
                    "n_frozen": frozen,
                    "cycle": row.get("cycle", np.nan),
                    "observation_id": row.get("picture", row.get("image_name", row_index)),
                }
            )
        if records:
            tables.append(
                CountsTable.from_dataframe(
                    pd.DataFrame.from_records(records),
                    metadata=_metadata_for_sample_id(metadata_by_sample, sample_id),
                    processing_metadata=processing_metadata_for(
                        "parse_sync_wide",
                        source_sample_ids=(sample_id,),
                    ),
                )
            )
    if not tables:
        raise ValueError("No sample number total/frozen column pairs found")
    return tables


def _is_sample_metadata_row(body: str) -> bool:
    if "," not in body:
        return False
    key = _split_csv_metadata_row(body)[0]
    return key in METADATA_ROW_LABELS


def _split_csv_metadata_row(body: str) -> list[str]:
    return [value.strip() for value in next(csv.reader([body]))]


def _time_seconds(df: pd.DataFrame) -> np.ndarray:
    if "time_s" in df:
        return pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
    if "timestamp" not in df:
        return np.arange(len(df), dtype=float)
    timestamp = pd.to_datetime(df["timestamp"], errors="coerce")
    if timestamp.notna().sum() == 0:
        return np.arange(len(df), dtype=float)
    start = timestamp.dropna().iloc[0]
    elapsed = (timestamp - start).dt.total_seconds()
    return elapsed.fillna(np.nan).to_numpy(dtype=float)


def _with_cycle_key(df: pd.DataFrame) -> pd.DataFrame:
    keyed = df.copy()
    if "cycle" not in keyed:
        keyed["_ufolaf_cycle_key"] = "missing"
        return keyed
    keyed["_ufolaf_cycle_key"] = [
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
    return [str(value) for value in pd.unique(df["_ufolaf_cycle_key"])]


def _selected_cycle_key(cycle_keys: list[str], cycle: Any | None) -> str:
    if not cycle_keys:
        raise ValueError("No cycles found")
    if cycle is None:
        if len(cycle_keys) == 1:
            return cycle_keys[0]
        available = ", ".join(cycle_keys)
        raise ValueError(
            "Multiple cycles found. Pass cycle=..., use cycle_policy='pooled', "
            f"or use cycle_policy='preserve'. Available cycles: {available}"
        )
    selected = _normalize_cycle_key(cycle)
    if selected not in cycle_keys:
        available = ", ".join(cycle_keys)
        raise ValueError(f"Requested cycle {selected!r} not found. Available cycles: {available}")
    return selected


def _iter_cycle_dataframes(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    for cycle_key, cycle_df in df.groupby("_ufolaf_cycle_key", sort=False):
        frames.append((str(cycle_key), cycle_df.drop(columns="_ufolaf_cycle_key").copy()))
    return frames


def _metadata_from_path(source: str | Path | pd.DataFrame) -> dict[str, SampleMetadata]:
    if isinstance(source, pd.DataFrame):
        return {}
    _, metadata = read_metadata(source)
    return metadata


def _metadata_for_sample_id(metadata: Any, sample_id: str) -> SampleMetadata:
    if isinstance(metadata, SampleMetadata):
        return _metadata_with_sample_id(metadata, sample_id)
    if isinstance(metadata, dict):
        value = metadata.get(sample_id)
        if isinstance(value, SampleMetadata):
            return _metadata_with_sample_id(value, sample_id)
    return SampleMetadata(sample_id=sample_id)


def _metadata_mapping_for_sample_ids(
    metadata_source: Any,
    sample_ids: list[str],
) -> dict[str, SampleMetadata]:
    metadata_by_sample = _metadata_mapping_from_source(metadata_source, sample_ids=sample_ids)
    return {
        sample_id: _metadata_for_sample_id(metadata_by_sample, sample_id)
        for sample_id in sample_ids
    }


def _metadata_mapping_from_source(
    source: Any,
    *,
    sample_ids: list[str] | None = None,
) -> dict[str, SampleMetadata]:
    if source is None:
        return {}
    if isinstance(source, pd.DataFrame):
        return _metadata_mapping_from_dataframe(source, sample_ids=sample_ids)
    if isinstance(source, dict):
        if not source:
            return {}
        if all(isinstance(value, SampleMetadata) for value in source.values()):
            return {
                str(sample_id): _metadata_with_sample_id(metadata, str(sample_id))
                for sample_id, metadata in source.items()
            }
        if _looks_like_metadata_record(source):
            return _metadata_mapping_from_record(source, sample_ids=sample_ids)
        metadata_by_sample_id: dict[str, SampleMetadata] = {}
        for key, value in source.items():
            sample_id = str(key)
            if isinstance(value, SampleMetadata):
                metadata_by_sample_id[sample_id] = _metadata_with_sample_id(value, sample_id)
            elif isinstance(value, dict):
                metadata_by_sample_id[sample_id] = _sample_metadata_from_mapping(
                    value,
                    sample_id,
                )
            else:
                metadata_by_sample_id.update(_metadata_mapping_from_source(value))
        return metadata_by_sample_id
    if isinstance(source, SampleMetadata):
        if source.sample_id:
            return {source.sample_id: source}
        if sample_ids is not None:
            return {sample_id: replace(source, sample_id=sample_id) for sample_id in sample_ids}
        return {}
    if isinstance(source, (list, tuple)):
        metadata_by_sample_id: dict[str, SampleMetadata] = {}
        for table in source:
            metadata_by_sample_id.update(
                _metadata_mapping_from_source(getattr(table, "metadata", None))
            )
        return metadata_by_sample_id
    metadata = getattr(source, "metadata", None)
    if metadata is not None:
        return _metadata_mapping_from_source(metadata)
    return {}


def _metadata_mapping_from_dataframe(
    df: pd.DataFrame,
    *,
    sample_ids: list[str] | None,
) -> dict[str, SampleMetadata]:
    if "sample_id" not in df:
        if sample_ids is not None and len(sample_ids) == 1:
            df = df.copy()
            df["sample_id"] = sample_ids[0]
        else:
            raise ValueError("Metadata DataFrame must include a sample_id column")

    metadata_by_sample_id: dict[str, SampleMetadata] = {}
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        sample_id = _metadata_text(row_dict.get("sample_id"))
        if not sample_id:
            raise ValueError("Metadata DataFrame contains an empty sample_id")
        if sample_id in metadata_by_sample_id:
            raise ValueError(f"Duplicate metadata row for sample_id {sample_id!r}")
        metadata_by_sample_id[sample_id] = _sample_metadata_from_mapping(row_dict, sample_id)
    return metadata_by_sample_id


def _metadata_mapping_from_record(
    values: dict[Any, Any],
    *,
    sample_ids: list[str] | None,
) -> dict[str, SampleMetadata]:
    if sample_ids is None:
        sample_id = _metadata_text(values.get("sample_id"))
        return {sample_id: _sample_metadata_from_mapping(values, sample_id)} if sample_id else {}
    return {
        sample_id: _sample_metadata_from_mapping(values, sample_id)
        for sample_id in sample_ids
    }


def _looks_like_metadata_record(values: dict[Any, Any]) -> bool:
    return any(str(key) in SAMPLE_METADATA_FIELDS for key in values)


def _sample_metadata_from_mapping(values: dict[Any, Any], sample_id: str) -> SampleMetadata:
    raw_sample_metadata = {
        str(key): _metadata_text(value)
        for key, value in values.items()
        if key not in ("raw_preamble", "raw_sample_metadata")
    }
    if isinstance(values.get("raw_sample_metadata"), dict):
        raw_sample_metadata.update(
            {
                str(key): _metadata_text(value)
                for key, value in values["raw_sample_metadata"].items()
            }
        )
    raw_preamble = (
        {str(key): _metadata_text(value) for key, value in values["raw_preamble"].items()}
        if isinstance(values.get("raw_preamble"), dict)
        else {}
    )

    kwargs: dict[str, Any] = {
        str(key): value
        for key, value in values.items()
        if str(key) in SAMPLE_METADATA_FIELDS
        and str(key) not in ("sample_id", "raw_preamble", "raw_sample_metadata")
        and not _is_missing_metadata_value(value)
    }
    kwargs["sample_id"] = sample_id
    kwargs["raw_preamble"] = raw_preamble
    kwargs["raw_sample_metadata"] = raw_sample_metadata
    return SampleMetadata(**kwargs)


def _metadata_with_sample_id(metadata: SampleMetadata, sample_id: str) -> SampleMetadata:
    if metadata.sample_id == sample_id:
        return metadata
    if not metadata.sample_id:
        return replace(metadata, sample_id=sample_id)
    return metadata


def _metadata_text(value: Any) -> str:
    return "" if _is_missing_metadata_value(value) else str(value)


def _is_missing_metadata_value(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False
    if isinstance(missing, (bool, np.bool_)):
        return bool(missing)
    return False


def _count_column_names() -> set[str]:
    return set(_ordered_count_columns())


def _ordered_count_columns() -> tuple[str, ...]:
    return (
        "sample_id",
        "temperature_C",
        "n_total",
        "n_frozen",
        "time_s",
        "cycle",
        "observation_id",
    )


def _resolve_counts_format(df: pd.DataFrame, format: CountInputFormat) -> Literal["long", "wide"]:
    normalized = format.casefold()
    if normalized in ("long", "canonical"):
        _require_long_count_columns(df)
        return "long"
    if normalized in ("icescopy", "icescopy_wide", "wide"):
        _require_wide_count_columns(df)
        return "wide"
    if normalized != "auto":
        raise ValueError(
            "format must be one of auto, long, canonical, icescopy, icescopy_wide, wide"
        )
    if _has_long_count_columns(df):
        return "long"
    if _has_wide_count_columns(df):
        return "wide"
    raise ValueError(
        "Could not detect counts input format. Supported formats are canonical long "
        "columns (sample_id, temperature_C, n_total, n_frozen) or Icescopy wide "
        "'number total'/'number frozen' column pairs."
    )


def _has_long_count_columns(df: pd.DataFrame) -> bool:
    return LONG_COUNT_COLUMNS.issubset(set(df.columns))


def _require_long_count_columns(df: pd.DataFrame) -> None:
    missing = sorted(LONG_COUNT_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Long count table is missing required columns: {', '.join(missing)}")


def _has_wide_count_columns(df: pd.DataFrame) -> bool:
    sample_columns: dict[str, set[str]] = {}
    for column in df.columns:
        match = SAMPLE_VALUE_RE.match(str(column))
        if not match:
            continue
        sample_columns.setdefault(match.group("sample").strip(), set()).add(match.group("kind"))
    return any({"total", "frozen"}.issubset(kinds) for kinds in sample_columns.values())


def _require_wide_count_columns(df: pd.DataFrame) -> None:
    if not _has_wide_count_columns(df):
        raise ValueError("Icescopy wide table must include number total/frozen column pairs")


def _sample_id_from_header(sample_key: str) -> str:
    match = SAMPLE_HEADER_RE.match(sample_key)
    if match:
        return match.group("sample").strip()
    return sample_key.strip()


def _to_float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _strip_trailing_numeric_token(value: str) -> str:
    parts = str(value).rsplit("_", 1)
    if len(parts) != 2:
        return str(value)
    try:
        float(parts[1])
    except ValueError:
        return str(value)
    return parts[0]


def parse_olaf_frozen_at_temp(
    df: pd.DataFrame,
    *,
    n_total_by_sample: dict[str, float],
) -> list[TemperatureFrozenFractionTable]:
    """Convert an OLAF frozen_at_temp table into one table per sample column."""

    if "degC" not in df:
        raise ValueError("OLAF frozen_at_temp data must include a 'degC' column")
    tables: list[TemperatureFrozenFractionTable] = []
    for column in df.columns:
        if column == "degC":
            continue
        if column not in n_total_by_sample:
            raise KeyError(f"Missing n_total for sample column {column!r}")
        records: list[dict[str, Any]] = []
        for _, row in df.iterrows():
            frozen = _to_float_or_nan(row[column])
            if not np.isfinite(frozen):
                continue
            records.append(
                {
                    "sample_id": column,
                    "temperature_C": float(row["degC"]),
                    "n_total": float(n_total_by_sample[column]),
                    "n_frozen": frozen,
                }
            )
        if records:
            result = pd.DataFrame.from_records(records)
            tables.append(
                TemperatureFrozenFractionTable(
                    sample_id=result["sample_id"].to_numpy(dtype=object),
                    temperature_C=result["temperature_C"].to_numpy(dtype=float),
                    n_total=result["n_total"].to_numpy(dtype=float),
                    n_frozen=result["n_frozen"].to_numpy(dtype=float),
                    processing_metadata=processing_metadata_for(
                        "parse_olaf_frozen_at_temp",
                        parameters={"n_total": float(n_total_by_sample[column])},
                        source_sample_ids=(column,),
                    ),
                )
            )
    return tables


read_commented_preamble = read_preamble
read_icescopy_temperature_sync_csv = read_sync
read_icescopy_sample_metadata = read_metadata
read_icescopy_freeze_count_timeseries_csv = read_counts
sample_metadata_to_dataframe = metadata_frame
infer_icescopy_dilution_group_map = infer_dilution_groups
strip_temperature_sync_metadata_rows = split_metadata_rows
parse_icescopy_wide_temperature_sync = parse_sync_wide
parse_olaf_frozen_at_temp_csv = parse_olaf_frozen_at_temp
