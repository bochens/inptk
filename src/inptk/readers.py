"""Translate ordinary count tables and Icescopy exports into an Experiment."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, fields, replace
from pathlib import Path

import pandas as pd

from .experiment import Experiment, MeasurementMetadata, SampleMetadata
from .tables import CountsTable

IDENTIFIERS = ("sample_id", "measurement_id", "run_id", "cycle_id", "cycle", "observation_id")


def _frame(source) -> pd.DataFrame:
    if isinstance(source, pd.DataFrame):
        return source.copy(deep=True)
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.suffix.lower() == ".json":
            return pd.DataFrame(json.loads(path.read_text()))
        return pd.read_csv(
            path,
            comment="#",
            dtype={name: str for name in IDENTIFIERS},
            keep_default_na=False,
            na_values=[""],
        )
    return pd.DataFrame(source)


def _clean(value):
    return None if value is None or pd.isna(value) or value == "" else value


def _metadata(records: pd.DataFrame, run_id: str):
    samples: dict[str, SampleMetadata] = {}
    measurements: dict[str, MeasurementMetadata] = {}
    sample_fields = {item.name for item in fields(SampleMetadata)}
    numeric_fields = {"air_volume_L", "suspension_volume_mL", "filter_fraction_used", "dry_mass_g"}
    for record in records.to_dict("records"):
        if _clean(record.get("sample_id")) is None:
            raise ValueError("Metadata sample_id must be non-empty")
        if "measurement_id" in record and _clean(record["measurement_id"]) is None:
            raise ValueError("Metadata measurement_id must be non-empty")
        sample_id = str(record["sample_id"])
        measurement_id = str(record.get("measurement_id", sample_id))
        values = {key: _clean(value) for key, value in record.items() if key in sample_fields}
        values = {key: value for key, value in values.items() if value is not None}
        values["sample_id"] = sample_id
        for key in numeric_fields & values.keys():
            values[key] = float(values[key])
        sample = SampleMetadata(**values)
        if sample_id in samples and samples[sample_id] != sample:
            raise ValueError(f"Conflicting sample metadata across dilutions of {sample_id!r}")
        samples[sample_id] = sample
        if measurement_id in measurements:
            raise ValueError(f"Duplicate measurement metadata: {measurement_id!r}")
        measurements[measurement_id] = MeasurementMetadata(
            measurement_id=measurement_id,
            sample_id=sample_id,
            run_id=str(_clean(record.get("run_id")) or run_id),
            dilution=float(record["dilution"]),
            droplet_volume_uL=float(record["droplet_volume_uL"]),
        )
    return samples, measurements


def read_counts(source, *, metadata, run_id: str = "1") -> Experiment:
    """Read native counts and measurement metadata from CSV or pandas tables.

    Native counts need measurement_id, temperature_C, n_total, and n_frozen.
    sample_id/run_id can be supplied by metadata. Missing cycle_id means this
    input describes one cycle, labelled '1'; cycles are never detected or pooled.
    Metadata has one row per measurement with sample_id, measurement_id,
    dilution, droplet_volume_uL and any sample normalization inputs.
    """
    data = _frame(source)
    records = _frame(metadata)
    if "sample_id" not in records:
        raise ValueError("Measurement metadata requires sample_id")
    missing = {"dilution", "droplet_volume_uL"} - set(records)
    if missing:
        raise ValueError(f"Measurement metadata is missing {sorted(missing)}")
    samples, measurements = _metadata(records, str(run_id))
    if "measurement_id" not in data:
        if "sample_id" not in data:
            raise ValueError(
                "Counts require measurement_id (or sample_id for one measurement per sample)"
            )
        data["measurement_id"] = data["sample_id"]
    if data.measurement_id.isna().any():
        raise ValueError("Counts measurement_id must be non-empty")
    data["measurement_id"] = data.measurement_id.astype(str)
    unknown = set(data.measurement_id) - set(measurements)
    if unknown:
        raise ValueError(f"Missing measurement metadata for {sorted(unknown)}")
    if "sample_id" not in data:
        data["sample_id"] = data.measurement_id.map(lambda key: measurements[key].sample_id)
    if "run_id" not in data:
        data["run_id"] = data.measurement_id.map(lambda key: measurements[key].run_id)
    if "cycle_id" not in data:
        data["cycle_id"] = data.pop("cycle") if "cycle" in data else "1"
    return Experiment(
        counts=CountsTable(data),
        samples=samples,
        measurements=measurements,
        source={
            "format": "native",
            "path": str(source) if isinstance(source, (str, Path)) else None,
        },
    )


def _icescopy_overrides(metadata, measurement_ids: set[str]) -> dict[str, dict]:
    """Keep only explicitly supplied fields when overriding export metadata."""
    from ._engine.adapters import SAMPLE_METADATA_FIELDS, _is_missing_metadata_value
    from ._engine.models import SampleMetadata as ExportMetadata

    if metadata is None:
        return {}
    if isinstance(metadata, ExportMetadata):
        metadata = asdict(metadata)
        if not metadata["sample_id"]:
            metadata.pop("sample_id")
    if isinstance(metadata, (str, Path)):
        metadata = _frame(metadata)
    if isinstance(metadata, pd.DataFrame):
        if "sample_id" not in metadata and len(measurement_ids) == 1:
            metadata = metadata.assign(sample_id=next(iter(measurement_ids)))
        records = metadata.to_dict("records")
    elif isinstance(metadata, (list, tuple)):
        records = list(metadata)
    elif isinstance(metadata, Mapping):
        if any(key in SAMPLE_METADATA_FIELDS for key in metadata):
            if not _is_missing_metadata_value(metadata.get("sample_id")):
                records = [dict(metadata)]
            else:
                records = [dict(metadata, sample_id=name) for name in measurement_ids]
        else:
            records = []
            for name, values in metadata.items():
                if isinstance(values, ExportMetadata):
                    values = asdict(values)
                if not isinstance(values, Mapping):
                    raise TypeError("Icescopy metadata must map measurement names to metadata")
                if values.get("sample_id") not in (None, "", name):
                    raise ValueError(f"Metadata sample_id disagrees with measurement {name!r}")
                records.append(dict(values, sample_id=name))
    else:
        raise TypeError("Icescopy metadata must contain metadata records or a mapping")

    overrides = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Icescopy metadata records must be mappings")
        name = record.get("sample_id")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Icescopy metadata sample_id must be a non-empty measurement name")
        if name not in measurement_ids:
            raise ValueError(f"Unknown measurement in Icescopy metadata: {name!r}")
        if name in overrides:
            raise ValueError(f"Duplicate metadata row for sample_id {name!r}")
        overrides[name] = {
            key: value
            for key, value in record.items()
            if key in SAMPLE_METADATA_FIELDS
            and key != "sample_id"
            and not _is_missing_metadata_value(value)
            and not (isinstance(value, str) and not value.strip())
        }
    return overrides


def read_icescopy(
    source, *, sample_map: dict[str, str] | None = None, metadata=None, run_id: str = "1"
) -> Experiment:
    """Read an Icescopy export; map measurement labels to original sample IDs.

    Without sample_map, every Icescopy measurement is a separate sample. Names
    are never shortened or interpreted as dilution groups automatically.
    Supplied metadata fields override header metadata for the exact measurement
    name; omitted or missing fields retain their values from the export.
    """
    from ._engine.adapters import read_counts as read_export
    from ._engine.adapters import read_metadata, read_sync, split_metadata_rows
    from ._engine.models import SampleMetadata as ExportMetadata

    if sample_map is not None and (
        not isinstance(sample_map, Mapping)
        or any(
            not isinstance(value, str) or not value.strip()
            for pair in sample_map.items()
            for value in pair
        )
    ):
        raise TypeError("sample_map must map non-empty measurement names to sample names")
    header_metadata: dict[str, ExportMetadata]
    if isinstance(source, pd.DataFrame):
        data, header_metadata = source.copy(deep=True), {}
    else:
        data, _ = read_sync(source)
        _, header_metadata = read_metadata(source)
    data, _ = split_metadata_rows(data)
    if "cycle" in data:
        missing_cycle = data.cycle.isna() | data.cycle.astype(str).str.strip().eq("")
        if missing_cycle.any() and not missing_cycle.all():
            raise ValueError("Some Icescopy cycle labels are missing")
        if missing_cycle.all():
            data = data.drop(columns="cycle")
    has_time = "time_s" in data or "timestamp" in data
    grouped = read_export(data, format="icescopy", metadata=header_metadata)
    known = {
        str(name) for tables in grouped.values() for table in tables for name in table.sample_id
    }
    overrides = _icescopy_overrides(metadata, known)
    frames = []
    records = {}
    for tables in grouped.values():
        for table in tables:
            frame = table.to_dataframe()
            for measurement_id, rows in frame.groupby("sample_id", sort=False):
                measurement_id = str(measurement_id)
                known.add(measurement_id)
                if sample_map is not None and measurement_id not in sample_map:
                    raise ValueError(f"sample_map is missing measurement {measurement_id!r}")
                sample_id = str(sample_map[measurement_id]) if sample_map else measurement_id
                original = table.metadata_for_sample(measurement_id)
                if original is None:
                    raise ValueError(f"Missing Icescopy metadata for {measurement_id!r}")
                original = replace(original, **overrides.get(measurement_id, {}))
                original.validate_for_count_to_suspension()
                records[measurement_id] = {
                    "measurement_id": measurement_id,
                    "sample_id": sample_id,
                    "sample_type": original.sample_type,
                    "dilution": original.dilution,
                    "droplet_volume_uL": original.well_volume_uL,
                    "run_id": str(run_id),
                    "air_volume_L": original.air_volume_L,
                    "suspension_volume_mL": original.suspension_volume_mL,
                    "filter_fraction_used": original.filter_fraction_used,
                    "dry_mass_g": original.dry_mass_g,
                }
                rows = rows.copy()
                rows["measurement_id"] = measurement_id
                rows["sample_id"] = sample_id
                rows["run_id"] = str(run_id)
                if "cycle" in rows and rows["cycle"].notna().any():
                    if rows["cycle"].isna().any():
                        raise ValueError("Some Icescopy cycle labels are missing")
                    rows["cycle_id"] = rows["cycle"].astype(str)
                else:
                    rows["cycle_id"] = "1"
                rows = rows.drop(columns=["cycle"], errors="ignore")
                if not has_time:
                    rows = rows.drop(columns=["time_s"], errors="ignore")
                frames.append(rows)
    if not frames:
        raise ValueError("No count observations found")
    if sample_map is not None and set(sample_map) - known:
        raise ValueError(f"Unknown measurements in sample_map: {sorted(set(sample_map) - known)}")
    experiment = read_counts(pd.concat(frames, ignore_index=True), metadata=list(records.values()))
    return Experiment(
        experiment.counts,
        experiment.samples,
        experiment.measurements,
        source={
            "format": "icescopy",
            "path": str(source) if isinstance(source, (str, Path)) else None,
        },
    )
