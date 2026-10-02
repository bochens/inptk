"""Translate ordinary count tables and Icescopy exports into an Experiment."""

from __future__ import annotations

import json
from dataclasses import fields
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


def read_icescopy(
    source, *, sample_map: dict[str, str] | None = None, metadata=None, run_id: str = "1"
) -> Experiment:
    """Read an Icescopy export; map measurement labels to original sample IDs.

    Without sample_map, every Icescopy measurement is a separate sample. Names
    are never shortened or interpreted as dilution groups automatically.
    """
    from ._engine.adapters import read_counts as read_export

    source_columns = (
        source.columns
        if isinstance(source, pd.DataFrame)
        else pd.read_csv(source, comment="#", nrows=0).columns
    )
    has_time = "time_s" in source_columns or "timestamp" in source_columns
    grouped = read_export(source, format="icescopy", metadata=metadata)
    frames = []
    records = {}
    known = set()
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
