"""Read the saved Icescopy count CSV and its explicitly recorded metadata."""

from __future__ import annotations

import json
from pathlib import Path
from zipfile import BadZipFile, ZipFile

from ._engine.adapters import _sample_metadata_from_record, read_icescopy_csv_text

COUNTS_MEMBER = "freeze_count_timeseries.csv"
METADATA_MEMBER = "session.json"


def read_icescopy_source(source):
    """Read a CSV or exact ZIP members in memory, without extracting other files."""
    path = Path(source)
    info = {"format": "icescopy", "path": str(path)}
    if path.suffix.lower() != ".icescopy":
        data, metadata = read_icescopy_csv_text(path.read_text(encoding="utf-8-sig"))
        return data, metadata, info
    try:
        with ZipFile(path) as archive:
            names = archive.namelist()
            if names.count(COUNTS_MEMBER) != 1:
                raise ValueError(
                    f"Icescopy archive must contain exactly one {COUNTS_MEMBER}; "
                    "calculate and save the temperature-based freeze counts in Icescopy first"
                )
            if names.count(METADATA_MEMBER) > 1:
                raise ValueError("Icescopy archive contains duplicate session.json members")
            text = archive.read(COUNTS_MEMBER).decode("utf-8-sig")
            payload = (json.loads(archive.read(METADATA_MEMBER).decode("utf-8-sig"))
                       if METADATA_MEMBER in names else {})
    except (BadZipFile, RuntimeError, NotImplementedError) as error:
        raise ValueError(f"Cannot read Icescopy archive {path.name!r}: {error}") from error
    if not isinstance(payload, dict):
        raise TypeError("Icescopy session.json must contain an object")
    summary = payload.get("freeze_count_timeseries_summary", {})
    if not isinstance(summary, dict):
        raise TypeError("Icescopy freeze_count_timeseries_summary must contain an object")
    if summary.get("analysis_required"):
        raise ValueError(
            "Icescopy marks its saved freeze counts as requiring analysis; "
            "recalculate and save them in Icescopy before importing"
        )
    records = summary.get("sample_column_metadata", [])
    if not isinstance(records, list):
        raise TypeError("Icescopy sample_column_metadata must be a list")
    session_metadata = payload.get("session_metadata", {})
    if not isinstance(session_metadata, dict):
        raise TypeError("Icescopy session_metadata must contain an object")
    data, metadata = read_icescopy_csv_text(text)
    seen = set()
    for record in records:
        if not isinstance(record, dict):
            raise TypeError("Icescopy sample metadata entries must be objects")
        name = record.get("sample_name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Icescopy count-column metadata requires an exact sample_name")
        if name in seen:
            raise ValueError(f"Duplicate Icescopy count-column metadata for {name!r}")
        seen.add(name)
        # sample_name identifies the literal CSV column prefix in Icescopy's
        # format. Its text is never interpreted to choose a role or parent sample.
        metadata[name] = _sample_metadata_from_record(session_metadata, record, name)
    info["archive_member"] = COUNTS_MEMBER
    if METADATA_MEMBER in names:
        info["metadata_member"] = METADATA_MEMBER
    return data, metadata, info
