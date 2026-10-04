"""Sample identities, physical droplet sets, and complete analysis results."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from .tables import (
    CountsTable,
    CumulativeSpectrumTable,
    CurveSpectrumTable,
    DifferentialSpectrumTable,
    FrozenFractionTable,
    ScientificTable,
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

    A water_blank_map pairs raw sample observations with physical assay blanks.
    Correction assumes the full assay-blank background scales with droplet volume.
    Each set retains its own known volume and observed counts; unequal volumes
    and droplet counts are supported. The caller supplies blanks with the intended
    material, geometry, preparation and cooling conditions, which are not inferred
    from measurement names. Repeated cycles remain separate.
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
class CurveResult:
    """One named curve, its sources, and excluded native points.

    cumulative contains retained points on the analysis temperatures.
    sources identifies physical inputs, independently of contributors at each
    point. Labels and result curves are never additional independent droplets.
    """

    curve_id: str
    cumulative: CurveSpectrumTable
    sources: list[dict]
    excluded: CurveSpectrumTable
    differential: DifferentialSpectrumTable | None = None

    def __post_init__(self):
        if not isinstance(self.curve_id, str) or not self.curve_id.strip():
            raise ValueError("curve_id must be non-empty text")
        for name in ("cumulative", "excluded"):
            table = getattr(self, name)
            if not isinstance(table, CurveSpectrumTable):
                raise TypeError(f"CurveResult {name} must be a CurveSpectrumTable")
            if set(table.to_dataframe().curve_id) - {self.curve_id}:
                raise ValueError(f"CurveResult {name} contains a different curve_id")
        if self.differential is not None and not isinstance(
            self.differential, DifferentialSpectrumTable
        ):
            raise TypeError("CurveResult differential must be a DifferentialSpectrumTable")
        if not isinstance(self.sources, list) or not self.sources:
            raise ValueError("CurveResult sources must be a nonempty list of physical inputs")
        keys = {"measurement_id", "run_id", "cycle_id"}
        seen, cycles = set(), {}
        for source in self.sources:
            if not isinstance(source, Mapping) or not keys.issubset(source):
                raise ValueError("CurveResult sources require measurement_id, run_id and cycle_id")
            identity = tuple(source[key] for key in sorted(keys))
            if any(not isinstance(value, str) or not value.strip() for value in identity):
                raise ValueError("CurveResult source identities must be non-empty text")
            if identity in seen:
                raise ValueError("CurveResult contains duplicate sources")
            seen.add(identity)
            run, cycle = source["run_id"], source["cycle_id"]
            if run in cycles and cycles[run] != cycle:
                raise ValueError("A curve must use only one cycle per run")
            cycles[run] = cycle
        if self.differential is not None:
            if len(self.sources) != 1:
                raise ValueError("Differential output currently requires one physical input")
            frame = self.differential.to_dataframe()
            if any(not frame[key].eq(self.sources[0][key]).all() for key in keys):
                raise ValueError("Differential output does not match the curve source")
        kept, excluded = self.cumulative.to_dataframe(), self.excluded.to_dataframe()
        if set(kept.point_id) & set(excluded.point_id):
            raise ValueError("Retained and excluded points must have different point IDs")

    @property
    def kind(self) -> str:
        """Individual or combined, based on requested physical inputs."""
        return "individual" if len(self.sources) == 1 else "combined"


@dataclass(frozen=True)
class AnalysisResult:
    """Original observations and a dictionary of named output curves."""

    experiment: Experiment
    frozen_fraction: FrozenFractionTable
    curves: dict[str, CurveResult]
    settings: dict = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.experiment, Experiment):
            raise TypeError("AnalysisResult experiment must be an Experiment")
        if not isinstance(self.frozen_fraction, FrozenFractionTable):
            raise TypeError("AnalysisResult frozen_fraction must be a FrozenFractionTable")
        if not isinstance(self.curves, dict) or not self.curves:
            raise ValueError("AnalysisResult curves must be a nonempty dictionary")
        for name, curve in self.curves.items():
            if not isinstance(curve, CurveResult):
                raise TypeError("AnalysisResult curves must contain CurveResult objects")
            if name != curve.curve_id:
                raise ValueError("Curve dictionary keys must match curve_id")
            parents = set()
            for source in curve.sources:
                measurement = self.experiment.measurements.get(source["measurement_id"])
                if measurement is None or measurement.run_id != source["run_id"]:
                    raise ValueError("Curve source disagrees with experiment metadata")
                parents.add(measurement.sample_id)
                if not len(
                    self.counts.select(
                        measurement_id=source["measurement_id"], cycle_id=source["cycle_id"]
                    )
                ):
                    raise ValueError("Curve source cycle has no observations")
            if len(parents) != 1:
                raise ValueError("Each curve must refer to one original sample")
            for table in (curve.cumulative, curve.excluded):
                if table is not None and set(table.to_dataframe().sample_id) - parents:
                    raise ValueError("Curve table disagrees with its original sample")

    @property
    def counts(self) -> CountsTable:
        """Original counts, including blanks; fitted curves never invent counts."""
        return self.experiment.counts

    def to_dataframe(
        self,
        *,
        table: Literal["cumulative", "excluded"] = "cumulative",
        curve_id: str | None = None,
    ):
        """Collect a quantity into a pandas table, retaining curve_id labels."""
        import pandas as pd

        if table not in ("cumulative", "excluded"):
            raise ValueError("table must be 'cumulative' or 'excluded'")
        if curve_id is not None and curve_id not in self.curves:
            raise ValueError(f"Unknown curve {curve_id!r}; available curves: {list(self.curves)}")
        curves = self.curves if curve_id is None else {curve_id: self.curves[curve_id]}
        tables = [getattr(curve, table) for curve in curves.values()]
        return pd.concat([value.to_dataframe() for value in tables], ignore_index=True)

    def save(self, path: str | Path) -> None:
        from .io import save

        save(self, path)

    def export_csv(
        self,
        path: str | Path,
        *,
        table: Literal["cumulative", "excluded"] = "cumulative",
        curve_id: str | None = None,
    ) -> None:
        """Export a quantity for all curves or one named curve; never overwrite."""
        self.to_dataframe(table=table, curve_id=curve_id).to_csv(path, index=False, mode="x")


@dataclass(frozen=True)
class ProcessingResult:
    """Saved tables from one calculation step, with optional experiment context.

    Fractions can be saved before physical metadata is available. Concentration
    steps retain the experiment so subsequent commands reuse its sample metadata.
    """

    tables: dict[str, ScientificTable]
    experiment: Experiment | None = None

    def __post_init__(self):
        if not isinstance(self.tables, dict) or not self.tables:
            raise TypeError("ProcessingResult requires a nonempty table dictionary")
        types = {
            "counts": CountsTable,
            "frozen_fraction": FrozenFractionTable,
            "cumulative": CumulativeSpectrumTable,
            "excluded": CumulativeSpectrumTable,
            "differential": DifferentialSpectrumTable,
        }
        for name, table in self.tables.items():
            if name not in types or not isinstance(table, types[name]):
                raise TypeError(f"Invalid processing table {name!r}")
        if self.experiment is not None and not isinstance(self.experiment, Experiment):
            raise TypeError("ProcessingResult experiment must be an Experiment")

    def save(self, path: str | Path) -> None:
        from .io import save

        save(self, path)
