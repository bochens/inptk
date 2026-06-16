"""Public UFOLAF API.

This module is intentionally a thin facade that exposes short names intended for
normal analysis scripts and notebooks.
"""

from .adapters import (
    infer_dilution_groups,
    map_count_columns,
    metadata_frame,
    parse_olaf_frozen_at_temp,
    parse_sync_wide,
    read_counts,
    read_metadata,
    read_sync,
    tables_to_dataframe,
)
from .blank_math import (
    average_blank_spectra,
    extrapolate_blank_tail,
    subtract_blank_spectrum,
    subtract_filter_blank_spectrum,
)
from .io import export_artifact_csv, read_artifact, write_artifact
from .models import (
    ArtifactRef,
    CountsTable,
    CumulativeNucleusSpectrumTable,
    DifferentialNucleusSpectrumTable,
    NormalizedInpSpectrumTable,
    ProcessingMetadata,
    ProcessingStep,
    SampleMetadata,
    TemperatureDependentTable,
    TemperatureFrozenFractionTable,
    artifact_ref,
    processing_metadata_for,
)
from .pipeline import counts_to_spectrum
from .qc import enforce_monotonic_vs_temperature
from .transforms import (
    apply_water_blank_correction as apply_water_blank,
    counts_to_temperature_frozen_fraction as fraction_frozen,
    cumulative_spectrum_to_normalized_inp_spectrum as normalize_spec,
    temperature_frozen_fraction_to_binomial_mle_cumulative_spectrum as cumulative_spec_mle,
    temperature_frozen_fraction_to_cumulative_spectrum as cumulative_spec,
    temperature_frozen_fraction_to_differential_spectrum as differential_spec,
    temperature_frozen_fraction_to_stitched_cumulative_spectrum as cumulative_spec_stitch,
)

__version__ = "0.1.0"

__all__ = [
    "CountsTable",
    "CumulativeNucleusSpectrumTable",
    "DifferentialNucleusSpectrumTable",
    "ArtifactRef",
    "NormalizedInpSpectrumTable",
    "ProcessingMetadata",
    "ProcessingStep",
    "SampleMetadata",
    "TemperatureDependentTable",
    "TemperatureFrozenFractionTable",
    "artifact_ref",
    "apply_water_blank",
    "average_blank_spectra",
    "cumulative_spec",
    "cumulative_spec_mle",
    "cumulative_spec_stitch",
    "differential_spec",
    "enforce_monotonic_vs_temperature",
    "export_artifact_csv",
    "fraction_frozen",
    "infer_dilution_groups",
    "map_count_columns",
    "metadata_frame",
    "normalize_spec",
    "parse_olaf_frozen_at_temp",
    "parse_sync_wide",
    "processing_metadata_for",
    "counts_to_spectrum",
    "read_artifact",
    "read_counts",
    "read_metadata",
    "read_sync",
    "subtract_filter_blank_spectrum",
    "subtract_blank_spectrum",
    "extrapolate_blank_tail",
    "tables_to_dataframe",
    "write_artifact",
]
