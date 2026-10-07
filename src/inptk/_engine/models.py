from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

SampleType = Literal["air", "soil", "other", "water blank"]
SAMPLE_TYPES: tuple[str, ...] = ("air", "soil", "other", "water blank")


def _optional_array(values: Any, *, dtype=float) -> np.ndarray | None:
    if values is None:
        return None
    return _array_copy(values, dtype=dtype)


def _required_array(values: Any, *, dtype=float, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required")
    return _array_copy(values, dtype=dtype)


def _array_copy(values: Any, *, dtype=float) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    if array.dtype == object:
        return copy.deepcopy(array)
    return array


def _same_length_or_raise(lengths: dict[str, int]) -> None:
    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        details = ", ".join(f"{name}={length}" for name, length in lengths.items())
        raise ValueError(f"Array lengths do not match: {details}")


def _require_positive(name: str, value: float | None) -> None:
    if value is None:
        raise ValueError(f"{name} is required")
    if value <= 0:
        raise ValueError(f"{name} must be positive")


def _text_or_empty(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.casefold() == "nan" else text


def _optional_float(value: Any) -> float | None:
    text = _text_or_empty(value)
    if not text:
        return None
    converted = float(text)
    if not np.isfinite(converted):
        return None
    return converted


def _optional_int(value: Any) -> int | None:
    converted = _optional_float(value)
    return None if converted is None else int(converted)


def _normalize_sample_type(value: Any) -> SampleType:
    text = _text_or_empty(value).casefold()
    if not text:
        return "other"
    if text in SAMPLE_TYPES:
        return text  # type: ignore[return-value]
    raise ValueError(
        f"Unknown sample_type {value!r}. Use 'air' for aerosol/filter samples normalized "
        "by sampled air volume, 'soil' for dry-soil normalization, or 'other' for "
        "suspension units, or 'water blank' for assay controls."
    )


@dataclass(frozen=True)
class SampleMetadata:
    """Icescopy freeze-count metadata plus INP-toolkit calculation inputs."""

    format_name: str = ""
    file_version: str = ""
    project_name: str = ""
    user_name: str = ""
    institution: str = ""
    date: str = ""
    well_volume_uL: float | None = None
    reset_temperature_C: float | None = None
    sample_id: str = ""
    sample_name: str = ""
    sample_long_name: str = ""
    collection_start: str = ""
    collection_end: str = ""
    sample_type: SampleType = "other"
    dilution: float | None = None
    air_volume_L: float | None = None
    filter_fraction_used: float | None = None
    suspension_volume_mL: float | None = None
    dry_mass_g: float | None = None
    total_cells: int | None = None
    source_header: str = ""
    raw_preamble: dict[str, str] = field(default_factory=dict)
    raw_sample_metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "format_name",
            "file_version",
            "project_name",
            "user_name",
            "institution",
            "date",
            "sample_id",
            "sample_name",
            "sample_long_name",
            "collection_start",
            "collection_end",
            "source_header",
        ):
            object.__setattr__(self, name, _text_or_empty(getattr(self, name)))
        object.__setattr__(self, "sample_type", _normalize_sample_type(self.sample_type))
        if self.sample_type == "water blank":
            object.__setattr__(self, "dilution", 1.0)
            for name in (
                "air_volume_L", "filter_fraction_used", "suspension_volume_mL", "dry_mass_g"
            ):
                object.__setattr__(self, name, None)
        for name in (
            "well_volume_uL",
            "reset_temperature_C",
            "dilution",
            "air_volume_L",
            "filter_fraction_used",
            "suspension_volume_mL",
            "dry_mass_g",
        ):
            object.__setattr__(self, name, _optional_float(getattr(self, name)))
        object.__setattr__(self, "total_cells", _optional_int(self.total_cells))
        object.__setattr__(self, "raw_preamble", dict(self.raw_preamble or {}))
        object.__setattr__(self, "raw_sample_metadata", dict(self.raw_sample_metadata or {}))
        if self.well_volume_uL is not None and self.well_volume_uL <= 0:
            raise ValueError("well_volume_uL must be positive")
        if self.dilution is not None and self.dilution <= 0:
            raise ValueError("dilution must be positive")

    def validate_for_count_to_suspension(self) -> None:
        _require_positive("well_volume_uL", self.well_volume_uL)
        _require_positive("dilution", self.dilution)


MetadataLike = SampleMetadata | dict[str, SampleMetadata] | None


@dataclass(frozen=True)
class ArtifactRef:
    """Serializable reference to an INP-toolkit table artifact."""

    artifact_id: str
    table_type: str
    role: str = "predecessor"
    sample_ids: tuple[str, ...] = ()
    content_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text_or_empty(self.artifact_id))
        object.__setattr__(self, "table_type", _text_or_empty(self.table_type))
        object.__setattr__(self, "role", _text_or_empty(self.role) or "predecessor")
        object.__setattr__(self, "sample_ids", _string_tuple(self.sample_ids))
        object.__setattr__(self, "content_hash", _text_or_empty(self.content_hash))


@dataclass(frozen=True)
class ProcessingStep:
    """One lightweight provenance step in an INP-toolkit processing chain."""

    operation: str
    parameters: dict[str, Any] = field(default_factory=dict)
    inputs: tuple[ArtifactRef, ...] = ()
    source_sample_ids: tuple[str, ...] = ()
    source_cycles: tuple[str, ...] = ()
    source_dilutions: tuple[Any, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "operation", _text_or_empty(self.operation))
        object.__setattr__(self, "parameters", _plain_mapping(self.parameters))
        object.__setattr__(self, "inputs", tuple(copy.deepcopy(self.inputs or ())))
        object.__setattr__(self, "source_sample_ids", _string_tuple(self.source_sample_ids))
        object.__setattr__(self, "source_cycles", _string_tuple(self.source_cycles))
        object.__setattr__(self, "source_dilutions", _plain_tuple(self.source_dilutions))
        object.__setattr__(self, "details", _plain_mapping(self.details))


@dataclass(frozen=True)
class ProcessingMetadata:
    """Provenance for one table: current step plus lightweight history."""

    artifact_id: str = ""
    generated_by: ProcessingStep | None = None
    history_snapshot: tuple[ProcessingStep, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "artifact_id", _text_or_empty(self.artifact_id))
        object.__setattr__(
            self,
            "generated_by",
            copy.deepcopy(self.generated_by) if self.generated_by is not None else None,
        )
        object.__setattr__(
            self,
            "history_snapshot",
            tuple(copy.deepcopy(self.history_snapshot or ())),
        )


def artifact_ref(table: Any, *, role: str = "predecessor") -> ArtifactRef:
    """Return a stable lightweight reference for an INP-toolkit table."""

    processing = getattr(table, "processing_metadata", ProcessingMetadata())
    content_hash = _table_content_hash(table)
    artifact_id = processing.artifact_id or f"{type(table).__name__}:{content_hash[:16]}"
    sample_ids = tuple(pd.Series(table.sample_id).astype(str).dropna().unique())
    return ArtifactRef(
        artifact_id=artifact_id,
        table_type=type(table).__name__,
        role=role,
        sample_ids=sample_ids,
        content_hash=content_hash,
    )


def processing_metadata_for(
    operation: str,
    *,
    inputs: tuple[Any, ...] | list[Any] = (),
    parameters: dict[str, Any] | None = None,
    source_sample_ids: tuple[str, ...] | list[str] = (),
    source_cycles: tuple[str, ...] | list[str] = (),
    source_dilutions: tuple[Any, ...] | list[Any] = (),
    details: dict[str, Any] | None = None,
) -> ProcessingMetadata:
    """Build processing metadata from immediate predecessor tables or refs."""

    input_refs: list[ArtifactRef] = []
    history: list[ProcessingStep] = []
    for value in inputs:
        if isinstance(value, ArtifactRef):
            input_refs.append(value)
            continue
        input_refs.append(artifact_ref(value))
        processing = getattr(value, "processing_metadata", None)
        if isinstance(processing, ProcessingMetadata):
            history.extend(processing.history_snapshot)

    step = ProcessingStep(
        operation=operation,
        parameters=parameters or {},
        inputs=tuple(input_refs),
        source_sample_ids=tuple(source_sample_ids),
        source_cycles=tuple(source_cycles),
        source_dilutions=tuple(source_dilutions),
        details=details or {},
    )
    history.append(step)
    return ProcessingMetadata(generated_by=step, history_snapshot=tuple(history))


def _normalize_metadata(metadata: MetadataLike) -> MetadataLike:
    if metadata is None or isinstance(metadata, SampleMetadata):
        return copy.deepcopy(metadata)
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a SampleMetadata, a dict of SampleMetadata, or None")
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise TypeError("metadata keys must be sample_id strings")
        if not isinstance(value, SampleMetadata):
            raise TypeError("metadata values must be SampleMetadata objects")
    return copy.deepcopy(metadata)


def _normalize_processing_metadata(
    processing_metadata: ProcessingMetadata | None,
) -> ProcessingMetadata:
    if processing_metadata is None:
        return ProcessingMetadata()
    if not isinstance(processing_metadata, ProcessingMetadata):
        raise TypeError("processing_metadata must be a ProcessingMetadata object or None")
    return copy.deepcopy(processing_metadata)


def _metadata_for_sample(metadata: MetadataLike, sample_id: str) -> SampleMetadata | None:
    if metadata is None:
        return None
    if isinstance(metadata, SampleMetadata):
        return metadata
    return metadata.get(sample_id)


def _string_tuple(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        return (values,)
    return tuple(_text_or_empty(value) for value in values)


def _plain_tuple(values: Any) -> tuple[Any, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        return (values,)
    return tuple(_plain_value(value) for value in values)


def _plain_mapping(values: dict[str, Any] | None) -> dict[str, Any]:
    if values is None:
        return {}
    return {str(key): _plain_value(value) for key, value in values.items()}


def _plain_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_plain_value(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain_value(item) for key, item in value.items()}
    return value


def _table_content_hash(table: Any) -> str:
    payload = {
        "table_type": type(table).__name__,
        "data": table.to_dataframe().to_dict(orient="list"),
        "sample_metadata": _metadata_hash_payload(getattr(table, "metadata", None)),
    }
    encoded = json.dumps(_plain_value(payload), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _metadata_hash_payload(metadata: MetadataLike) -> Any:
    if metadata is None:
        return None
    if isinstance(metadata, SampleMetadata):
        return asdict(metadata)
    return {str(key): asdict(value) for key, value in sorted(metadata.items())}


def _dataframe_like_getitem(table: Any, key: Any) -> Any:
    return table.to_dataframe().__getitem__(key)


def _dataframe_like_to_numpy(table: Any, columns: list[str] | tuple[str, ...] | None) -> np.ndarray:
    df = table.to_dataframe()
    if columns is not None:
        df = df.loc[:, list(columns)]
    return df.to_numpy(copy=True)


@dataclass(frozen=True, kw_only=True)
class CountsTable:
    """Raw count observations before temperature binning or concentration conversion."""

    sample_id: Any
    temperature_C: Any
    n_total: Any
    n_frozen: Any
    time_s: Any | None = None
    cycle: Any | None = None
    observation_id: Any | None = None
    metadata: MetadataLike = None
    processing_metadata: ProcessingMetadata | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sample_id", _required_array(self.sample_id, dtype=object, name="sample_id")
        )
        object.__setattr__(
            self,
            "temperature_C",
            _required_array(self.temperature_C, dtype=float, name="temperature_C"),
        )
        object.__setattr__(
            self, "n_total", _required_array(self.n_total, dtype=float, name="n_total")
        )
        object.__setattr__(
            self, "n_frozen", _required_array(self.n_frozen, dtype=float, name="n_frozen")
        )
        object.__setattr__(self, "time_s", _optional_array(self.time_s, dtype=float))
        object.__setattr__(self, "cycle", _optional_array(self.cycle, dtype=object))
        object.__setattr__(
            self, "observation_id", _optional_array(self.observation_id, dtype=object)
        )
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))
        object.__setattr__(
            self,
            "processing_metadata",
            _normalize_processing_metadata(self.processing_metadata),
        )
        lengths = {
            "sample_id": len(self.sample_id),
            "temperature_C": len(self.temperature_C),
            "n_total": len(self.n_total),
            "n_frozen": len(self.n_frozen),
        }
        for name in ("time_s", "cycle", "observation_id"):
            value = getattr(self, name)
            if value is not None:
                lengths[name] = len(value)
        _same_length_or_raise(lengths)

    @property
    def fraction_frozen(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.n_frozen / self.n_total

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        metadata: MetadataLike = None,
        processing_metadata: ProcessingMetadata | None = None,
    ) -> CountsTable:
        return cls(
            sample_id=df["sample_id"].to_numpy(dtype=object),
            temperature_C=df["temperature_C"].to_numpy(dtype=float),
            n_total=df["n_total"].to_numpy(dtype=float),
            n_frozen=df["n_frozen"].to_numpy(dtype=float),
            time_s=df["time_s"].to_numpy(dtype=float) if "time_s" in df else None,
            cycle=df["cycle"].to_numpy(dtype=object) if "cycle" in df else None,
            observation_id=df["observation_id"].to_numpy(dtype=object)
            if "observation_id" in df
            else None,
            metadata=metadata,
            processing_metadata=processing_metadata,
        )

    @property
    def sample_metadata(self) -> MetadataLike:
        return self.metadata

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(self.to_dataframe().columns)

    def __len__(self) -> int:
        return len(self.sample_id)

    def __getitem__(self, key: Any) -> Any:
        return _dataframe_like_getitem(self, key)

    def to_numpy(self, columns: list[str] | tuple[str, ...] | None = None) -> np.ndarray:
        return _dataframe_like_to_numpy(self, columns)

    def artifact_ref(self, *, role: str = "predecessor") -> ArtifactRef:
        return artifact_ref(self, role=role)

    def to_dataframe(self) -> pd.DataFrame:
        data: dict[str, Any] = {
            "sample_id": self.sample_id.copy(),
            "temperature_C": self.temperature_C.copy(),
            "n_total": self.n_total.copy(),
            "n_frozen": self.n_frozen.copy(),
            "fraction_frozen": self.fraction_frozen,
        }
        if self.time_s is not None:
            data["time_s"] = self.time_s.copy()
        if self.cycle is not None:
            data["cycle"] = copy.deepcopy(self.cycle)
        if self.observation_id is not None:
            data["observation_id"] = copy.deepcopy(self.observation_id)
        return pd.DataFrame(data)

    def metadata_for_sample(self, sample_id: str) -> SampleMetadata | None:
        return _metadata_for_sample(self.metadata, sample_id)
