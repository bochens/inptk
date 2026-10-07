from __future__ import annotations

import csv
import io
import re
from collections.abc import Iterable
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .._csv import read_csv_with_preamble
from .models import (
    CountsTable,
    SampleMetadata,
    _optional_float,
    _optional_int,
)

SAMPLE_METADATA_FIELDS = {field.name for field in fields(SampleMetadata)}

__all__ = [
    "parse_sync_wide",
    "read_counts",
    "read_preamble",
    "read_sync",
    "split_metadata_rows",
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
    "well_volume_uL",
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
    return _read_sync_data(path), preamble


def _read_sync_data(source):
    return read_csv_with_preamble(
        source,
        dtype={"sample_id": str, "cycle": str, "observation_id": str},
        keep_default_na=False,
        na_values=[""],
    )


def read_icescopy_csv_text(text: str) -> tuple[pd.DataFrame, dict[str, SampleMetadata]]:
    """Parse CSV text with the same count and metadata rules as file input."""
    _, metadata = _read_metadata_lines(io.StringIO(text))
    return _read_sync_data(io.StringIO(text)), metadata


def _read_metadata_lines(
    handle: Iterable[str],
) -> tuple[dict[str, str], dict[str, SampleMetadata]]:
    session_metadata: dict[str, str] = {}
    sample_rows: dict[str, list[str]] = {}
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
            key: values[index] if index < len(values) else "" for key, values in sample_rows.items()
        }
        sample_id = (
            raw_sample_metadata.get("sample_name")
            or raw_sample_metadata.get("sample_id")
            or str(index)
        )
        metadata_by_sample_id[sample_id] = _sample_metadata_from_record(
            session_metadata, raw_sample_metadata, sample_id
        )
    return session_metadata, metadata_by_sample_id


def _sample_metadata_from_record(session_metadata, raw_sample_metadata, sample_id):
    """Read catalog metadata; SampleMetadata normalizes each catalog type."""
    return SampleMetadata(
        format_name=session_metadata.get("format_name", ""),
        file_version=session_metadata.get("file_version", ""),
        project_name=session_metadata.get("project_name", ""),
        user_name=session_metadata.get("user_name", ""),
        institution=session_metadata.get("institution", ""),
        date=session_metadata.get("analysis_date", session_metadata.get("date", "")),
        well_volume_uL=_optional_float(
            raw_sample_metadata.get("well_volume_uL", session_metadata.get("well_volume_uL"))
        ),
        reset_temperature_C=_optional_float(session_metadata.get("reset_temperature_C")),
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
        total_cells=_optional_int(raw_sample_metadata.get("cell_number")),
        raw_preamble=session_metadata,
        raw_sample_metadata=raw_sample_metadata,
    )


def read_counts(
    source: pd.DataFrame,
    *,
    metadata: dict[str, SampleMetadata],
) -> dict[str, list[CountsTable]]:
    """Read an Icescopy wide count table into one table per sample, grouped by cycle.

    metadata maps sample names to their Icescopy header metadata.
    """

    df = source.copy()
    _require_wide_count_columns(df)
    count_tables = parse_sync_wide(df, metadata=metadata)
    return _preserve_count_cycles(count_tables)


def _preserve_count_cycles(tables: list[CountsTable]) -> dict[str, list[CountsTable]]:
    grouped: dict[str, list[CountsTable]] = {}
    for table in tables:
        keyed = _with_cycle_key(table.to_dataframe())
        for cycle_key, cycle_df in _iter_cycle_dataframes(keyed):
            grouped.setdefault(cycle_key, []).append(
                CountsTable.from_dataframe(
                    cycle_df.reset_index(drop=True),
                    metadata=table.metadata,
                )
            )
    return grouped


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
    metadata: dict[str, SampleMetadata],
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
        for row_index, (_, row) in enumerate(data_rows.iterrows()):
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
    if timestamp.isna().any():
        raise ValueError("Icescopy timestamps must be valid and non-missing when supplied")
    start = timestamp.dropna().iloc[0]
    elapsed = (timestamp - start).dt.total_seconds()
    return elapsed.fillna(np.nan).to_numpy(dtype=float)


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


def _iter_cycle_dataframes(df: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    frames: list[tuple[str, pd.DataFrame]] = []
    for cycle_key, cycle_df in df.groupby("_inptk_cycle_key", sort=False):
        frames.append((str(cycle_key), cycle_df.drop(columns="_inptk_cycle_key").copy()))
    return frames


def _metadata_for_sample_id(metadata: dict[str, SampleMetadata], sample_id: str) -> SampleMetadata:
    value = metadata.get(sample_id)
    if isinstance(value, SampleMetadata):
        return _metadata_with_sample_id(value, sample_id)
    return SampleMetadata(sample_id=sample_id)


def _metadata_mapping_for_sample_ids(
    metadata_source: dict[str, SampleMetadata],
    sample_ids: list[str],
) -> dict[str, SampleMetadata]:
    metadata_by_sample = {
        str(sample_id): _metadata_with_sample_id(metadata, str(sample_id))
        for sample_id, metadata in metadata_source.items()
    }
    return {
        sample_id: _metadata_for_sample_id(metadata_by_sample, sample_id)
        for sample_id in sample_ids
    }


def _metadata_with_sample_id(metadata: SampleMetadata, sample_id: str) -> SampleMetadata:
    if metadata.sample_id == sample_id:
        return metadata
    if not metadata.sample_id:
        return replace(metadata, sample_id=sample_id)
    return metadata


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
