from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from .models import (
    ArtifactRef,
    CountsTable,
    CumulativeNucleusSpectrumTable,
    DifferentialNucleusSpectrumTable,
    MetadataLike,
    NormalizedInpSpectrumTable,
    ProcessingMetadata,
    ProcessingStep,
    SampleMetadata,
    TemperatureFrozenFractionTable,
)


TableArtifact = (
    CountsTable
    | TemperatureFrozenFractionTable
    | DifferentialNucleusSpectrumTable
    | CumulativeNucleusSpectrumTable
    | NormalizedInpSpectrumTable
)
ArtifactShape = TableArtifact | list[Any] | dict[str, Any]

MANIFEST_FILE = "manifest.json"
DATA_FILE = "data.csv"
SAMPLE_METADATA_FILE = "sample_metadata.json"
PROCESSING_METADATA_FILE = "processing_metadata.json"


def write_artifact(value: ArtifactShape, path: str | Path, *, overwrite: bool = False) -> None:
    """Write a UFOLAF table/list/dict artifact directory."""

    target = Path(path)
    if target.exists():
        if not overwrite:
            raise FileExistsError(f"{target} already exists. Pass overwrite=True to replace it.")
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
    target.mkdir(parents=True)
    _write_artifact_value(value, target)


def read_artifact(path: str | Path) -> ArtifactShape:
    """Read a UFOLAF artifact directory written by :func:`write_artifact`."""

    source = Path(path)
    if not source.is_dir():
        raise ValueError(f"{source} is not a UFOLAF artifact directory")
    return _read_artifact_value(source)


def export_artifact_csv(value: ArtifactShape, path: str | Path) -> None:
    """Export one artifact shape as a flattened CSV using table data only."""

    from .adapters import tables_to_dataframe

    tables_to_dataframe(value).to_csv(path, index=False)


def _write_artifact_value(value: ArtifactShape, path: Path) -> None:
    if _is_table(value):
        _write_table(value, path)
        return
    if isinstance(value, (list, tuple)):
        items = []
        for index, item in enumerate(value):
            item_name = f"item-{index:04d}"
            _write_artifact_value(item, path / item_name)
            items.append(item_name)
        _write_json(path / MANIFEST_FILE, {"kind": "list", "items": items})
        return
    if isinstance(value, dict):
        items = []
        for index, (key, item) in enumerate(value.items()):
            item_name = f"item-{index:04d}"
            _write_artifact_value(item, path / item_name)
            items.append({"key": str(key), "path": item_name})
        _write_json(path / MANIFEST_FILE, {"kind": "dict", "items": items})
        return
    raise TypeError("Artifact value must be a UFOLAF table, list, or dict")


def _read_artifact_value(path: Path) -> ArtifactShape:
    manifest = _read_json(path / MANIFEST_FILE)
    kind = manifest.get("kind")
    if kind == "table":
        return _read_table(path, str(manifest["table_type"]))
    if kind == "list":
        return [_read_artifact_value(path / item) for item in manifest["items"]]
    if kind == "dict":
        return {
            str(item["key"]): _read_artifact_value(path / str(item["path"]))
            for item in manifest["items"]
        }
    raise ValueError(f"Unknown UFOLAF artifact kind {kind!r} in {path}")


def _write_table(table: TableArtifact, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    table.to_dataframe().to_csv(path / DATA_FILE, index=False)
    _write_json(
        path / MANIFEST_FILE,
        {"kind": "table", "table_type": type(table).__name__},
    )
    _write_json(path / SAMPLE_METADATA_FILE, _metadata_to_jsonable(table.metadata))
    _write_json(
        path / PROCESSING_METADATA_FILE,
        _processing_metadata_to_jsonable(table.processing_metadata),
    )


def _read_table(path: Path, table_type: str) -> TableArtifact:
    df = pd.read_csv(path / DATA_FILE)
    metadata = _metadata_from_jsonable(_read_json(path / SAMPLE_METADATA_FILE))
    processing = _processing_metadata_from_jsonable(_read_json(path / PROCESSING_METADATA_FILE))
    if table_type == "CountsTable":
        return CountsTable.from_dataframe(df, metadata=metadata, processing_metadata=processing)
    if table_type == "TemperatureFrozenFractionTable":
        return TemperatureFrozenFractionTable.from_dataframe(
            df,
            metadata=metadata,
            processing_metadata=processing,
        )
    if table_type == "DifferentialNucleusSpectrumTable":
        return DifferentialNucleusSpectrumTable.from_dataframe(
            df,
            metadata=metadata,
            processing_metadata=processing,
        )
    if table_type == "CumulativeNucleusSpectrumTable":
        return CumulativeNucleusSpectrumTable.from_dataframe(
            df,
            metadata=metadata,
            processing_metadata=processing,
        )
    if table_type == "NormalizedInpSpectrumTable":
        return NormalizedInpSpectrumTable.from_dataframe(
            df,
            metadata=metadata,
            processing_metadata=processing,
        )
    raise ValueError(f"Unsupported UFOLAF table type {table_type!r}")


def _is_table(value: Any) -> bool:
    return isinstance(
        value,
        (
            CountsTable,
            TemperatureFrozenFractionTable,
            DifferentialNucleusSpectrumTable,
            CumulativeNucleusSpectrumTable,
            NormalizedInpSpectrumTable,
        ),
    )


def _metadata_to_jsonable(metadata: MetadataLike) -> dict[str, Any]:
    if metadata is None:
        return {"kind": "none", "value": None}
    if isinstance(metadata, SampleMetadata):
        return {"kind": "single", "value": asdict(metadata)}
    return {
        "kind": "mapping",
        "value": {str(key): asdict(value) for key, value in metadata.items()},
    }


def _metadata_from_jsonable(payload: dict[str, Any]) -> MetadataLike:
    kind = payload.get("kind")
    if kind == "none":
        return None
    if kind == "single":
        return SampleMetadata(**payload["value"])
    if kind == "mapping":
        return {
            str(key): SampleMetadata(**value)
            for key, value in payload["value"].items()
        }
    raise ValueError(f"Unknown metadata payload kind {kind!r}")


def _processing_metadata_to_jsonable(metadata: ProcessingMetadata | None) -> dict[str, Any]:
    return asdict(metadata or ProcessingMetadata())


def _processing_metadata_from_jsonable(payload: dict[str, Any]) -> ProcessingMetadata:
    generated_by = payload.get("generated_by")
    return ProcessingMetadata(
        artifact_id=payload.get("artifact_id", ""),
        generated_by=_processing_step_from_jsonable(generated_by)
        if generated_by is not None
        else None,
        history_snapshot=tuple(
            _processing_step_from_jsonable(step)
            for step in payload.get("history_snapshot", [])
        ),
    )


def _processing_step_from_jsonable(payload: dict[str, Any]) -> ProcessingStep:
    return ProcessingStep(
        operation=payload.get("operation", ""),
        parameters=payload.get("parameters", {}),
        inputs=tuple(_artifact_ref_from_jsonable(item) for item in payload.get("inputs", [])),
        source_sample_ids=tuple(payload.get("source_sample_ids", [])),
        source_cycles=tuple(payload.get("source_cycles", [])),
        source_dilutions=tuple(payload.get("source_dilutions", [])),
        details=payload.get("details", {}),
    )


def _artifact_ref_from_jsonable(payload: dict[str, Any]) -> ArtifactRef:
    return ArtifactRef(
        artifact_id=payload.get("artifact_id", ""),
        table_type=payload.get("table_type", ""),
        role=payload.get("role", "predecessor"),
        sample_ids=tuple(payload.get("sample_ids", [])),
        content_hash=payload.get("content_hash", ""),
    )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
