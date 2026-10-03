"""Read count experiments and already calculated reference spectra."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .experiment import Experiment, MeasurementMetadata, SampleMetadata
from .tables import CountsTable

IDENTIFIERS = (
    "sample_id",
    "measurement_id",
    "run_id",
    "cycle_id",
    "cycle",
    "observation_id",
    "picture_id",
)


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
            run_id=str(run_id if _clean(record.get("run_id")) is None else record["run_id"]),
            dilution=float(record["dilution"]),
            droplet_volume_uL=float(record["droplet_volume_uL"]),
        )
    return samples, measurements


def read_counts(
    source,
    *,
    metadata,
    run_id: str = "1",
    water_blank_map: dict[str, list[str]] | None = None,
) -> Experiment:
    """Read native counts and measurement metadata from CSV or pandas tables.

    Native counts need measurement_id, temperature_C, n_total, and n_frozen.
    sample_id/run_id can be supplied by metadata. Missing cycle_id means this
    input describes one cycle, labelled '1'; cycles are never detected or pooled.
    Observation IDs identify source rows within each measurement/run/cycle. Missing
    IDs are assigned in input order; repeated temperatures are preserved.
    Metadata has one row per measurement with sample_id, measurement_id,
    dilution, droplet_volume_uL and any sample normalization inputs.
    With water_blank_map, supply raw counts for samples and water blanks. Map
    each sample measurement name to a list of blank measurement names; import preserves
    the observations unchanged. Do not map counts already corrected by Icescopy.
    Correction assumes the full assay-blank background scales with droplet volume.
    Each sample and blank set needs its own positive droplet_volume_uL; volumes
    and observed droplet counts may differ. The caller supplies blanks with the
    intended material, geometry, preparation and cooling conditions; these are
    not inferred from measurement names.
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
        water_blank_map={} if water_blank_map is None else water_blank_map,
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


def _icescopy_observations(
    source, *, sample_map=None, metadata=None, run_id="1", require_metadata: bool
) -> tuple[pd.DataFrame, list[dict]]:
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
    has_image_identity = "picture" in data or "image_name" in data
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
                if require_metadata:
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
                if "observation_id" in rows:
                    if has_image_identity:
                        rows = rows.rename(columns={"observation_id": "picture_id"})
                    else:
                        # The older adapter fills this field with row numbers.
                        # They are not evidence of a shared image acquisition.
                        rows = rows.drop(columns="observation_id")
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
    return pd.concat(frames, ignore_index=True), list(records.values())


def read_icescopy(
    source, *, sample_map: dict[str, str] | None = None, metadata=None, run_id: str = "1",
    water_blank_map: dict[str, list[str]] | None = None,
) -> Experiment:
    """Read Icescopy counts with the physical metadata required for concentration.

    Map exact measurement names to parent samples explicitly. Supplied metadata
    overrides only the named fields; missing values retain export-header values.
    Use read_observations for counts/fractions before physical metadata is ready.
    For raw exports, water_blank_map assigns physical blank sets explicitly.
    Every complete count row is retained; an image ID is not required.
    """
    data, records = _icescopy_observations(
        source, sample_map=sample_map, metadata=metadata, run_id=run_id, require_metadata=True
    )
    experiment = read_counts(data, metadata=records, water_blank_map=water_blank_map)
    return Experiment(
        experiment.counts,
        experiment.samples,
        experiment.measurements,
        source={
            "format": "icescopy",
            "path": str(source) if isinstance(source, (str, Path)) else None,
        },
        water_blank_map=experiment.water_blank_map,
    )


def read_observations(
    source, *, format: str = "native", metadata=None, sample_map=None, run_id: str = "1"
) -> CountsTable:
    """Read counts for inspection without requiring dilution or volume metadata.

    No fitting, temperature alignment, blank correction or cycle combination is
    performed. Missing parent sample assignments keep each measurement separate.
    Native metadata may supply identities before its physical fields are complete.
    Available metadata and provisional sample assignments are recorded in history.
    Concentration calculation still requires a fully validated Experiment.
    """
    provisional = []
    if format == "icescopy":
        data, records = _icescopy_observations(
            source,
            sample_map=sample_map,
            metadata=metadata,
            run_id=run_id,
            require_metadata=False,
        )
        if sample_map is None:
            provisional = [record["measurement_id"] for record in records]
    elif format == "native":
        if sample_map is not None:
            raise ValueError(
                "sample_map applies only to Icescopy input; native metadata uses sample_id"
            )
        data = _frame(source)
        if "measurement_id" not in data:
            if "sample_id" not in data:
                raise ValueError("Counts require measurement_id or sample_id")
            data["measurement_id"] = data.sample_id
        if (
            data.measurement_id.isna().any()
            or data.measurement_id.astype(str).str.strip().eq("").any()
        ):
            raise ValueError("Counts measurement_id must be non-empty")
        data["measurement_id"] = data.measurement_id.astype(str)
        supplied: dict[str, dict] = {}
        metadata_fields = {
            item.name for model in (SampleMetadata, MeasurementMetadata) for item in fields(model)
        }
        if metadata is not None:
            for item in _frame(metadata).to_dict("records"):
                name = _clean(item.get("measurement_id", item.get("sample_id")))
                if name is None or not str(name).strip():
                    raise ValueError("Metadata requires a non-empty measurement_id or sample_id")
                name = str(name)
                if name in supplied:
                    raise ValueError(f"Duplicate measurement metadata: {name!r}")
                supplied[name] = {
                    str(key): _clean(value) for key, value in item.items() if key in metadata_fields
                }
        if "sample_id" not in data:
            provisional = [
                name
                for name in data.measurement_id.unique()
                if supplied.get(name, {}).get("sample_id") is None
            ]
            data["sample_id"] = data.measurement_id.map(
                lambda name: (
                    supplied.get(name, {}).get("sample_id")
                    if supplied.get(name, {}).get("sample_id") is not None
                    else name
                )
            )
        if "run_id" not in data:
            data["run_id"] = data.measurement_id.map(
                lambda name: (
                    supplied.get(name, {}).get("run_id")
                    if supplied.get(name, {}).get("run_id") is not None
                    else str(run_id)
                )
            )
        if "cycle_id" not in data:
            data["cycle_id"] = data.pop("cycle") if "cycle" in data else "1"
        records = []
        for measurement, rows in data.groupby("measurement_id", sort=False):
            name = str(measurement)
            record: dict = dict(supplied.get(name, {}), measurement_id=name)
            for column in ("sample_id", "run_id"):
                values = rows[column].dropna().astype(str).unique()
                if len(values) != 1:
                    raise ValueError(f"Measurement {name!r} must have one {column}")
                if _clean(record.get(column)) is not None and str(record[column]) != values[0]:
                    raise ValueError(f"Count identities disagree with metadata for {name!r}")
                record[column] = str(values[0])
            records.append(record)
    else:
        raise ValueError("Observation format must be 'native' or 'icescopy'")
    if data.empty:
        raise ValueError("No count observations found")
    return CountsTable(
        data,
        history=[
            {
                "operation": "read_observations",
                "format": format,
                "path": str(source) if isinstance(source, (str, Path)) else None,
                "measurement_metadata": records,
                "provisional_sample_assignments": provisional,
            }
        ],
    )


@dataclass(frozen=True)
class CSUSpectrum:
    """An already calculated CSU/OLAF air spectrum and its supplied metadata.

    table uses INP-toolkit column names and error widths, in INP per litre of air.
    metadata uses standard physical-field names; other header fields remain text.
    source records the path, file hash, and original header. Reading a reference
    never assigns sample identities, blank roles, or metadata to an experiment.
    """

    table: pd.DataFrame
    metadata: dict
    source: dict


def read_csu_csv(source: str | Path) -> CSUSpectrum:
    """Read a CSU INPs_L CSV, including OLAF reference results, without recalculation.

    The file contains key=value header lines followed by degC, INPS_L,
    lower_CI and upper_CI columns. The CI columns are error widths, not endpoints.
    Missing normalization metadata stays missing; importing a comparison curve
    does not require air volume. Applying its metadata is an explicit caller choice.
    """
    path = Path(source)
    content = path.read_bytes()
    lines = content.decode("utf-8-sig").splitlines(keepends=True)
    header = {}
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        columns = [value.strip() for value in next(csv.reader([line]))]
        if columns[0] == "degC":
            break
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise ValueError("CSU CSV requires key=value metadata followed by a degC table")
        key = key.strip()
        if key in header:
            raise ValueError(f"Duplicate CSU metadata field {key!r}")
        header[key] = value.strip()
    else:
        raise ValueError("CSU CSV is missing its degC table header")
    table = pd.read_csv(io.StringIO("".join(lines[index:])))
    table.columns = table.columns.str.strip()
    names = {"degC": "temperature_C", "INPS_L": "concentration",
             "lower_CI": "lower_error", "upper_CI": "upper_error"}
    missing = set(names) - set(table.columns)
    if missing:
        raise ValueError(f"CSU CSV is missing columns: {sorted(missing)}")
    table = table.rename(columns=names)
    for column in names.values():
        table[column] = pd.to_numeric(table[column], errors="raise")
    if table.empty or not np.isfinite(table.temperature_C).all():
        raise ValueError("CSU CSV requires nonempty, finite temperatures")
    if table[["lower_error", "upper_error"]].lt(0).any().any():
        raise ValueError("CSU uncertainty columns must be nonnegative error widths")
    table["basis"] = "sampled_air"
    table["unit"] = "INP_per_L_air"
    metadata: dict[str, str | float] = dict(header)
    for original, standard in {
        "vol_air_filt": "air_volume_L",
        "vol_susp": "suspension_volume_mL",
        "proportion_filter_used": "filter_fraction_used",
    }.items():
        if original not in metadata:
            continue
        if standard in metadata:
            raise ValueError(f"Duplicate CSU metadata for {standard!r}")
        raw = metadata.pop(original)
        if not raw:
            continue
        number = float(raw)
        if not np.isfinite(number) or number <= 0:
            raise ValueError(f"CSU {original} must be finite and positive")
        if standard == "filter_fraction_used" and number > 1:
            raise ValueError("CSU proportion_filter_used must not exceed 1")
        metadata[standard] = number
    return CSUSpectrum(
        table=table,
        metadata=metadata,
        source={"format": "csu", "path": str(path),
                "sha256": hashlib.sha256(content).hexdigest(), "header": header},
    )
