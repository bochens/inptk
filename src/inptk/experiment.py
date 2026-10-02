"""Sample identities, physical droplet sets, and complete analysis results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from .tables import (
    CountsTable,
    CumulativeSpectrumTable,
    DifferentialSpectrumTable,
    FrozenFractionTable,
)


def _positive(name: str, value: float | None) -> float:
    if value is None or not math.isfinite(float(value)) or value <= 0:
        raise ValueError(f"{name} must be a finite positive number")
    return float(value)


@dataclass(frozen=True)
class SampleMetadata:
    """The original sampled material, shared by all its dilution measurements."""

    sample_id: str
    sample_name: str = ""
    sample_type: str = "other"
    air_volume_L: float | None = None
    suspension_volume_mL: float | None = None
    filter_fraction_used: float | None = None
    dry_mass_g: float | None = None

    def __post_init__(self):
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("sample_id must be non-empty text")
        if self.sample_type not in ("air", "soil", "other"):
            raise ValueError("sample_type must be air, soil, or other")
        for name in ("air_volume_L", "suspension_volume_mL", "filter_fraction_used", "dry_mass_g"):
            value = getattr(self, name)
            if value is not None:
                _positive(name, value)
        if self.filter_fraction_used is not None and self.filter_fraction_used > 1:
            raise ValueError("filter_fraction_used must not exceed 1")

    def factor_for(self, basis: str) -> float:
        if basis == "suspension":
            return 1.0
        suspension_volume = _positive("suspension_volume_mL", self.suspension_volume_mL)
        if basis == "sampled_air":
            if self.sample_type != "air":
                raise ValueError(f"Sample {self.sample_id!r} is not an air sample")
            air_volume = _positive("air_volume_L", self.air_volume_L)
            filter_fraction = _positive("filter_fraction_used", self.filter_fraction_used)
            return suspension_volume / (air_volume * filter_fraction)
        if basis == "dry_soil":
            if self.sample_type != "soil":
                raise ValueError(f"Sample {self.sample_id!r} is not a soil sample")
            dry_mass = _positive("dry_mass_g", self.dry_mass_g)
            return suspension_volume / dry_mass
        raise ValueError(f"Unknown concentration basis {basis!r}")


@dataclass(frozen=True)
class MeasurementMetadata:
    """One physical droplet set, followed through every freezing cycle."""

    measurement_id: str
    sample_id: str
    dilution: float
    droplet_volume_uL: float
    run_id: str = "1"

    def __post_init__(self):
        for name in ("measurement_id", "sample_id", "run_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        _positive("dilution", self.dilution)
        _positive("droplet_volume_uL", self.droplet_volume_uL)


@dataclass(frozen=True)
class Experiment:
    """Observations and explicit relationships between physical droplet sets.

    A water_blank_map declares raw sample and blank observations under one common
    water-background concentration per volume for each assigned group. This assumes
    the same prepared-water protocol. Every set needs its own known droplet volume;
    sample and blank volumes may differ. Repeated cycles remain separate.
    """

    counts: CountsTable
    samples: dict[str, SampleMetadata]
    measurements: dict[str, MeasurementMetadata]
    source: dict = field(default_factory=dict)
    water_blank_map: dict[str, list[str]] = field(default_factory=dict)

    def __post_init__(self):
        for key, sample in self.samples.items():
            if key != sample.sample_id:
                raise ValueError("Sample dictionary keys must match sample_id")
        for key, measurement in self.measurements.items():
            if key != measurement.measurement_id:
                raise ValueError("Measurement dictionary keys must match measurement_id")
            if measurement.sample_id not in self.samples:
                raise ValueError(f"Unknown parent sample {measurement.sample_id!r}")
        identities = self.counts.to_dataframe()[
            ["measurement_id", "sample_id", "run_id"]
        ].drop_duplicates()
        for row in identities.itertuples(index=False):
            if row.measurement_id not in self.measurements:
                raise ValueError(f"Missing measurement metadata for {row.measurement_id!r}")
            measurement = self.measurements[row.measurement_id]
            if (row.sample_id, row.run_id) != (measurement.sample_id, measurement.run_id):
                raise ValueError(
                    f"Count identities disagree with metadata for {row.measurement_id!r}"
                )
        self._validate_water_blank_map()

    def _validate_water_blank_map(self) -> None:
        if not isinstance(self.water_blank_map, Mapping):
            raise TypeError(
                "water_blank_map must map measurement names to lists of water-blank names"
            )
        mapping = {}
        for sample_id, assigned in self.water_blank_map.items():
            if not isinstance(sample_id, str) or not sample_id.strip():
                raise ValueError("water_blank_map must contain non-empty measurement names")
            if not isinstance(assigned, list) or not assigned:
                raise ValueError(
                    "water_blank_map values must be non-empty lists of water-blank names"
                )
            if any(not isinstance(name, str) or not name.strip() for name in assigned):
                raise ValueError("water_blank_map must contain non-empty measurement names")
            if len(set(assigned)) != len(assigned):
                raise ValueError(f"Duplicate water-blank names for measurement {sample_id!r}")
            mapping[sample_id] = list(assigned)
        sample_ids = set(mapping)
        blank_ids = {name for assigned in mapping.values() for name in assigned}
        unknown = (sample_ids | blank_ids) - set(self.measurements)
        if unknown:
            raise ValueError(f"Unknown measurements in water_blank_map: {sorted(unknown)}")
        if sample_ids & blank_ids:
            raise ValueError("Water-blank measurements cannot also be sample measurements")
        if mapping:
            unassigned = set(self.measurements) - sample_ids - blank_ids
            if unassigned:
                raise ValueError(
                    f"water_blank_map must assign every nonblank measurement: {sorted(unassigned)}"
                )
        backgrounds: dict[tuple[str, str], frozenset[str]] = {}
        for sample_id, assigned in mapping.items():
            sample = self.measurements[sample_id]
            group = (sample.sample_id, sample.run_id)
            blank_group = frozenset(assigned)
            if group in backgrounds and backgrounds[group] != blank_group:
                raise ValueError(
                    "Measurements of the same sample and run must use the same set of water blanks"
                )
            backgrounds[group] = blank_group
            for blank_id in assigned:
                blank = self.measurements[blank_id]
                if blank.dilution != 1:
                    raise ValueError(f"Water-blank measurement {blank_id!r} must have dilution=1")
                if sample.run_id != blank.run_id:
                    raise ValueError(
                        f"Measurement {sample_id!r} and water blank {blank_id!r} "
                        "must use the same run"
                    )
        object.__setattr__(self, "water_blank_map", mapping)

    def save(self, path: str | Path) -> None:
        from .io import save

        save(self, path)


@dataclass(frozen=True)
class AnalysisResult:
    experiment: Experiment
    frozen_fraction: FrozenFractionTable
    per_dilution: CumulativeSpectrumTable
    combined: CumulativeSpectrumTable
    final: CumulativeSpectrumTable
    differential: DifferentialSpectrumTable | None = None
    final_candidates: CumulativeSpectrumTable | None = None
    settings: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def save(self, path: str | Path) -> None:
        from .io import save

        save(self, path)

    def export_csv(self, path: str | Path) -> None:
        """Export final rows with cycle identities; refuse to overwrite a file."""
        self.final.to_dataframe().to_csv(path, index=False, mode="x")
