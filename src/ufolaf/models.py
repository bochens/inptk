from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Literal

import numpy as np
import pandas as pd


SampleType = Literal["air", "soil", "other"]
SAMPLE_TYPES: tuple[str, ...] = ("air", "soil", "other")
SpectrumBasis = Literal["suspension", "sampled_air", "dry_soil", "other"]


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


def _optional_unit(values: str | Any | None) -> str | np.ndarray | None:
    if values is None or isinstance(values, str):
        return values
    return _array_copy(values, dtype=object)


def _required_unit(values: str | Any, *, name: str) -> str | np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required")
    converted = _optional_unit(values)
    if converted is None:
        raise ValueError(f"{name} is required")
    return converted


def _add_optional_length(lengths: dict[str, int], name: str, value: Any) -> None:
    if value is not None and not isinstance(value, str):
        lengths[name] = len(value)


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
        "suspension units."
    )


@dataclass(frozen=True)
class SampleMetadata:
    """Icescopy freeze-count metadata plus UFOLAF calculation inputs."""

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

    def validate_for_sample_type(self) -> None:
        self.validate_for_count_to_suspension()
        self.validate_for_spectrum_normalization()

    def validate_for_spectrum_normalization(self) -> None:
        """Validate metadata needed to normalize an already-computed spectrum."""

        if self.sample_type == "air":
            _require_positive("air_volume_L", self.air_volume_L)
            _require_positive("filter_fraction_used", self.filter_fraction_used)
            _require_positive("suspension_volume_mL", self.suspension_volume_mL)
        elif self.sample_type == "soil":
            _require_positive("suspension_volume_mL", self.suspension_volume_mL)
            _require_positive("dry_mass_g", self.dry_mass_g)

    def with_sample_type(self, sample_type: SampleType) -> SampleMetadata:
        return replace(self, sample_type=sample_type)


MetadataLike = SampleMetadata | dict[str, SampleMetadata] | None


@dataclass(frozen=True)
class ArtifactRef:
    """Serializable reference to a UFOLAF table artifact."""

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
    """One lightweight provenance step in a UFOLAF processing chain."""

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
    """Return a stable lightweight reference for a UFOLAF table."""

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


@dataclass(frozen=True, kw_only=True)
class TemperatureDependentTable:
    """Base class for all reduced tables that share a temperature-bin contract."""

    sample_id: Any
    temperature_C: Any
    temperature_bin_width_C: float | None = None
    temperature_bin_method: str = ""
    temperature_bin_left_C: Any | None = None
    temperature_bin_right_C: Any | None = None
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
            self,
            "temperature_bin_left_C",
            _optional_array(self.temperature_bin_left_C, dtype=float),
        )
        object.__setattr__(
            self,
            "temperature_bin_right_C",
            _optional_array(self.temperature_bin_right_C, dtype=float),
        )
        object.__setattr__(self, "metadata", _normalize_metadata(self.metadata))
        object.__setattr__(
            self,
            "processing_metadata",
            _normalize_processing_metadata(self.processing_metadata),
        )
        if self.temperature_bin_width_C is not None and self.temperature_bin_width_C <= 0:
            raise ValueError("temperature_bin_width_C must be positive")
        lengths = {
            "sample_id": len(self.sample_id),
            "temperature_C": len(self.temperature_C),
        }
        _add_optional_length(lengths, "temperature_bin_left_C", self.temperature_bin_left_C)
        _add_optional_length(lengths, "temperature_bin_right_C", self.temperature_bin_right_C)
        _same_length_or_raise(lengths)

    def _temperature_dataframe(self) -> pd.DataFrame:
        data: dict[str, Any] = {
            "sample_id": self.sample_id.copy(),
            "temperature_C": self.temperature_C.copy(),
        }
        if self.temperature_bin_width_C is not None:
            data["temperature_bin_width_C"] = np.repeat(
                self.temperature_bin_width_C, len(self.temperature_C)
            )
        if self.temperature_bin_method:
            data["temperature_bin_method"] = np.repeat(
                self.temperature_bin_method, len(self.temperature_C)
            )
        if self.temperature_bin_left_C is not None:
            data["temperature_bin_left_C"] = self.temperature_bin_left_C.copy()
        if self.temperature_bin_right_C is not None:
            data["temperature_bin_right_C"] = self.temperature_bin_right_C.copy()
        return pd.DataFrame(data)

    def metadata_for_sample(self, sample_id: str) -> SampleMetadata | None:
        return _metadata_for_sample(self.metadata, sample_id)

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


@dataclass(frozen=True, kw_only=True)
class TemperatureFrozenFractionTable(TemperatureDependentTable):
    """Frozen fractions, optionally averaged over repeated cycles of the same droplets.

    For averaged cycles, ``n_frozen`` is a mean count and ``n_total`` remains the
    droplet count of one cycle. ``cycle_count`` records the number of contributing
    cycles at each temperature. ``fraction_frozen_cycle_std`` is their sample
    standard deviation, not a confidence interval or an error of the mean.
    """

    n_total: Any
    n_frozen: Any
    obs_count: Any | None = None
    cycle_count: Any | None = None
    fraction_frozen_cycle_std: Any | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self, "n_total", _required_array(self.n_total, dtype=float, name="n_total")
        )
        object.__setattr__(
            self, "n_frozen", _required_array(self.n_frozen, dtype=float, name="n_frozen")
        )
        object.__setattr__(self, "obs_count", _optional_array(self.obs_count, dtype=int))
        object.__setattr__(self, "cycle_count", _optional_array(self.cycle_count, dtype=int))
        object.__setattr__(
            self, "fraction_frozen_cycle_std", _optional_array(self.fraction_frozen_cycle_std)
        )
        lengths = {
            "sample_id": len(self.sample_id),
            "temperature_C": len(self.temperature_C),
            "n_total": len(self.n_total),
            "n_frozen": len(self.n_frozen),
        }
        _add_optional_length(lengths, "temperature_bin_left_C", self.temperature_bin_left_C)
        _add_optional_length(lengths, "temperature_bin_right_C", self.temperature_bin_right_C)
        _add_optional_length(lengths, "obs_count", self.obs_count)
        _add_optional_length(lengths, "cycle_count", self.cycle_count)
        _add_optional_length(lengths, "fraction_frozen_cycle_std", self.fraction_frozen_cycle_std)
        _same_length_or_raise(lengths)
        if self.cycle_count is not None and np.any(self.cycle_count < 1):
            raise ValueError("cycle_count must be positive")

    @property
    def fraction_frozen(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.n_frozen / self.n_total

    def to_dataframe(self) -> pd.DataFrame:
        data = self._temperature_dataframe()
        data["n_total"] = self.n_total.copy()
        data["n_frozen"] = self.n_frozen.copy()
        data["fraction_frozen"] = self.fraction_frozen
        if self.obs_count is not None:
            data["obs_count"] = self.obs_count.copy()
        if self.cycle_count is not None:
            data["cycle_count"] = self.cycle_count.copy()
        if self.fraction_frozen_cycle_std is not None:
            data["fraction_frozen_cycle_std"] = self.fraction_frozen_cycle_std.copy()
        return data

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        temperature_bin_width_C: float | None = None,
        temperature_bin_method: str = "",
        metadata: MetadataLike = None,
        processing_metadata: ProcessingMetadata | None = None,
    ) -> TemperatureFrozenFractionTable:
        return cls(
            sample_id=df["sample_id"].to_numpy(dtype=object),
            temperature_C=df["temperature_C"].to_numpy(dtype=float),
            temperature_bin_width_C=temperature_bin_width_C
            if temperature_bin_width_C is not None
            else _scalar_or_none(df, "temperature_bin_width_C"),
            temperature_bin_method=temperature_bin_method
            or _scalar_string_or_empty(df, "temperature_bin_method"),
            temperature_bin_left_C=df["temperature_bin_left_C"].to_numpy(dtype=float)
            if "temperature_bin_left_C" in df
            else None,
            temperature_bin_right_C=df["temperature_bin_right_C"].to_numpy(dtype=float)
            if "temperature_bin_right_C" in df
            else None,
            n_total=df["n_total"].to_numpy(dtype=float),
            n_frozen=df["n_frozen"].to_numpy(dtype=float),
            obs_count=df["obs_count"].to_numpy(dtype=int) if "obs_count" in df else None,
            cycle_count=df["cycle_count"].to_numpy(dtype=int) if "cycle_count" in df else None,
            fraction_frozen_cycle_std=df["fraction_frozen_cycle_std"].to_numpy(dtype=float)
            if "fraction_frozen_cycle_std" in df else None,
            metadata=metadata,
            processing_metadata=processing_metadata,
        )


@dataclass(frozen=True, kw_only=True)
class DifferentialNucleusSpectrumTable(TemperatureDependentTable):
    """Differential nucleus spectrum k(T), usually per concentration basis per degree C."""

    value: Any
    value_unit: str | Any
    basis: SpectrumBasis | Any = "suspension"
    lower_ci: Any | None = None
    upper_ci: Any | None = None
    qc_flag: Any | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "basis", _normalize_spectrum_basis(self.basis))
        object.__setattr__(self, "value", _required_array(self.value, dtype=float, name="value"))
        object.__setattr__(self, "value_unit", _required_unit(self.value_unit, name="value_unit"))
        object.__setattr__(self, "lower_ci", _optional_array(self.lower_ci, dtype=float))
        object.__setattr__(self, "upper_ci", _optional_array(self.upper_ci, dtype=float))
        object.__setattr__(self, "qc_flag", _optional_array(self.qc_flag, dtype=int))
        lengths = _spectrum_lengths(self)
        _same_length_or_raise(lengths)

    def to_dataframe(self) -> pd.DataFrame:
        data = _spectrum_dataframe(self)
        data["basis"] = _basis_column(self.basis, len(self.value))
        return data

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        value_unit: str | None = None,
        basis: SpectrumBasis | Any = "suspension",
        metadata: MetadataLike = None,
        processing_metadata: ProcessingMetadata | None = None,
    ) -> DifferentialNucleusSpectrumTable:
        return cls(
            **_temperature_kwargs_from_dataframe(df),
            value=df["value"].to_numpy(dtype=float),
            value_unit=_value_unit_from_dataframe(df, value_unit),
            basis=_basis_from_dataframe(df, basis),
            lower_ci=df["lower_ci"].to_numpy(dtype=float) if "lower_ci" in df else None,
            upper_ci=df["upper_ci"].to_numpy(dtype=float) if "upper_ci" in df else None,
            qc_flag=df["qc_flag"].to_numpy(dtype=int) if "qc_flag" in df else None,
            metadata=metadata,
            processing_metadata=processing_metadata,
        )


@dataclass(frozen=True, kw_only=True)
class CumulativeNucleusSpectrumTable(TemperatureDependentTable):
    """Cumulative nucleus spectrum K(T), usually INP per mL suspension."""

    value: Any
    value_unit: str | Any
    basis: SpectrumBasis | Any = "suspension"
    lower_ci: Any | None = None
    upper_ci: Any | None = None
    dilution_fold: Any | None = None
    qc_flag: Any | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "basis", _normalize_spectrum_basis(self.basis))
        object.__setattr__(self, "value", _required_array(self.value, dtype=float, name="value"))
        object.__setattr__(self, "value_unit", _required_unit(self.value_unit, name="value_unit"))
        object.__setattr__(self, "lower_ci", _optional_array(self.lower_ci, dtype=float))
        object.__setattr__(self, "upper_ci", _optional_array(self.upper_ci, dtype=float))
        object.__setattr__(self, "dilution_fold", _optional_array(self.dilution_fold, dtype=float))
        object.__setattr__(self, "qc_flag", _optional_array(self.qc_flag, dtype=int))
        lengths = _spectrum_lengths(self, extra_names=("dilution_fold",))
        _same_length_or_raise(lengths)

    def to_dataframe(self) -> pd.DataFrame:
        data = _spectrum_dataframe(self, extra_names=("dilution_fold",))
        data["basis"] = _basis_column(self.basis, len(self.value))
        return data

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        value_unit: str | None = None,
        basis: SpectrumBasis | Any = "suspension",
        metadata: MetadataLike = None,
        processing_metadata: ProcessingMetadata | None = None,
    ) -> CumulativeNucleusSpectrumTable:
        return cls(
            **_temperature_kwargs_from_dataframe(df),
            value=df["value"].to_numpy(dtype=float),
            value_unit=_value_unit_from_dataframe(df, value_unit),
            basis=_basis_from_dataframe(df, basis),
            lower_ci=df["lower_ci"].to_numpy(dtype=float) if "lower_ci" in df else None,
            upper_ci=df["upper_ci"].to_numpy(dtype=float) if "upper_ci" in df else None,
            dilution_fold=df["dilution_fold"].to_numpy(dtype=float)
            if "dilution_fold" in df
            else None,
            qc_flag=df["qc_flag"].to_numpy(dtype=int) if "qc_flag" in df else None,
            metadata=metadata,
            processing_metadata=processing_metadata,
        )


@dataclass(frozen=True, kw_only=True)
class NormalizedInpSpectrumTable(TemperatureDependentTable):
    """Temperature-dependent INP concentration after optional sample-basis normalization."""

    value: Any
    value_unit: str | Any
    basis: SpectrumBasis | Any
    lower_ci: Any | None = None
    upper_ci: Any | None = None
    qc_flag: Any | None = None
    replicate_count: Any | None = None
    is_extrapolated: Any | None = None
    correction_state: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "basis", _normalize_spectrum_basis(self.basis))
        object.__setattr__(self, "value", _required_array(self.value, dtype=float, name="value"))
        object.__setattr__(self, "value_unit", _required_unit(self.value_unit, name="value_unit"))
        object.__setattr__(self, "lower_ci", _optional_array(self.lower_ci, dtype=float))
        object.__setattr__(self, "upper_ci", _optional_array(self.upper_ci, dtype=float))
        object.__setattr__(self, "qc_flag", _optional_array(self.qc_flag, dtype=int))
        object.__setattr__(
            self, "replicate_count", _optional_array(self.replicate_count, dtype=int)
        )
        object.__setattr__(
            self, "is_extrapolated", _optional_array(self.is_extrapolated, dtype=bool)
        )
        lengths = _spectrum_lengths(
            self,
            extra_names=(
                "replicate_count",
                "is_extrapolated",
            ),
        )
        _same_length_or_raise(lengths)

    def to_dataframe(self) -> pd.DataFrame:
        data = _spectrum_dataframe(
            self,
            extra_names=(
                "replicate_count",
                "is_extrapolated",
            ),
        )
        data["basis"] = _basis_column(self.basis, len(self.value))
        if self.correction_state:
            data["correction_state"] = np.repeat(self.correction_state, len(self.value))
        return data

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        value_unit: str | None = None,
        basis: SpectrumBasis | Any | None = None,
        correction_state: str = "",
        metadata: MetadataLike = None,
        processing_metadata: ProcessingMetadata | None = None,
    ) -> NormalizedInpSpectrumTable:
        inferred_basis = _basis_from_dataframe(df, basis if basis is not None else "other")
        inferred_correction = correction_state
        if not inferred_correction and "correction_state" in df:
            non_null = df["correction_state"].dropna()
            inferred_correction = str(non_null.iloc[0]) if not non_null.empty else ""
        return cls(
            **_temperature_kwargs_from_dataframe(df),
            value=df["value"].to_numpy(dtype=float),
            value_unit=_value_unit_from_dataframe(df, value_unit),
            basis=inferred_basis,
            lower_ci=df["lower_ci"].to_numpy(dtype=float) if "lower_ci" in df else None,
            upper_ci=df["upper_ci"].to_numpy(dtype=float) if "upper_ci" in df else None,
            qc_flag=df["qc_flag"].to_numpy(dtype=int) if "qc_flag" in df else None,
            replicate_count=df["replicate_count"].to_numpy(dtype=int)
            if "replicate_count" in df
            else None,
            is_extrapolated=df["is_extrapolated"].to_numpy(dtype=bool)
            if "is_extrapolated" in df
            else None,
            correction_state=inferred_correction,
            metadata=metadata,
            processing_metadata=processing_metadata,
        )


def _validate_spectrum_basis(basis: str) -> None:
    if basis not in ("suspension", "sampled_air", "dry_soil", "other"):
        raise ValueError("basis must be one of 'suspension', 'sampled_air', 'dry_soil', 'other'")


def _normalize_spectrum_basis(basis: SpectrumBasis | Any) -> str | np.ndarray:
    if isinstance(basis, str):
        _validate_spectrum_basis(basis)
        return basis
    basis_array = _required_array(basis, dtype=object, name="basis")
    for value in pd.Series(basis_array).dropna().unique():
        _validate_spectrum_basis(str(value))
    return basis_array


def _basis_column(basis: str | np.ndarray, length: int) -> np.ndarray:
    return np.repeat(basis, length) if isinstance(basis, str) else basis.copy()


def _basis_from_dataframe(
    df: pd.DataFrame,
    basis: SpectrumBasis | Any | None,
) -> str | np.ndarray:
    if basis is not None and not (isinstance(basis, str) and "basis" in df):
        return _normalize_spectrum_basis(basis)
    if "basis" not in df:
        return _normalize_spectrum_basis(basis or "other")
    unique_basis = pd.Series(df["basis"]).dropna().unique()
    if len(unique_basis) == 0:
        return _normalize_spectrum_basis(basis or "other")
    if len(unique_basis) == 1:
        return _normalize_spectrum_basis(str(unique_basis[0]))
    return _normalize_spectrum_basis(df["basis"].to_numpy(dtype=object))


def _spectrum_lengths(
    table: TemperatureDependentTable,
    *,
    extra_names: tuple[str, ...] = (),
) -> dict[str, int]:
    lengths = {
        "sample_id": len(table.sample_id),
        "temperature_C": len(table.temperature_C),
        "value": len(getattr(table, "value")),
    }
    _add_optional_length(lengths, "temperature_bin_left_C", table.temperature_bin_left_C)
    _add_optional_length(lengths, "temperature_bin_right_C", table.temperature_bin_right_C)
    _add_optional_length(lengths, "value_unit", getattr(table, "value_unit"))
    _add_optional_length(lengths, "basis", getattr(table, "basis"))
    for name in ("lower_ci", "upper_ci", "qc_flag", *extra_names):
        _add_optional_length(lengths, name, getattr(table, name))
    return lengths


def _spectrum_dataframe(
    table: TemperatureDependentTable,
    *,
    extra_names: tuple[str, ...] = (),
) -> pd.DataFrame:
    data = table._temperature_dataframe()
    data["value"] = getattr(table, "value").copy()
    value_unit = getattr(table, "value_unit")
    data["value_unit"] = (
        value_unit.copy() if not isinstance(value_unit, str) else np.repeat(value_unit, len(data))
    )
    for name in ("lower_ci", "upper_ci", "qc_flag", *extra_names):
        value = getattr(table, name)
        if value is not None:
            data[name] = value.copy()
    return data


def _temperature_kwargs_from_dataframe(df: pd.DataFrame) -> dict[str, Any]:
    return {
        "sample_id": df["sample_id"].to_numpy(dtype=object),
        "temperature_C": df["temperature_C"].to_numpy(dtype=float),
        "temperature_bin_width_C": _scalar_or_none(df, "temperature_bin_width_C"),
        "temperature_bin_method": _scalar_string_or_empty(df, "temperature_bin_method"),
        "temperature_bin_left_C": df["temperature_bin_left_C"].to_numpy(dtype=float)
        if "temperature_bin_left_C" in df
        else None,
        "temperature_bin_right_C": df["temperature_bin_right_C"].to_numpy(dtype=float)
        if "temperature_bin_right_C" in df
        else None,
    }


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


def _value_unit_from_dataframe(df: pd.DataFrame, value_unit: str | None) -> str | np.ndarray:
    if value_unit is not None:
        return value_unit
    if "value_unit" not in df:
        raise ValueError("value_unit must be provided when df lacks a value_unit column")
    unique_units = pd.Series(df["value_unit"]).dropna().unique()
    if len(unique_units) == 1:
        return str(unique_units[0])
    return df["value_unit"].to_numpy(dtype=object)
