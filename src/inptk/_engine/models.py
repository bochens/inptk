from __future__ import annotations

import copy
from dataclasses import dataclass, field
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


def _metadata_for_sample(metadata: MetadataLike, sample_id: str) -> SampleMetadata | None:
    if metadata is None:
        return None
    if isinstance(metadata, SampleMetadata):
        return metadata
    return metadata.get(sample_id)


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
        )

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
