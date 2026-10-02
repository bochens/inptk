"""INP-toolkit: independent analysis of droplet-freezing experiments."""

from .experiment import AnalysisResult, Experiment, MeasurementMetadata, SampleMetadata
from .io import load, save
from .processing import cumulative_spectrum, differential_spectrum, frozen_fraction
from .readers import read_counts, read_icescopy, read_observations
from .resampling import resample_spectrum
from .tables import (
    CombinedSpectrumTable,
    CountsTable,
    CumulativeSpectrumTable,
    DifferentialSpectrumTable,
    FrozenFractionTable,
)
from .workflows import (
    analyze_concentration,
    combine_dilutions,
    convert_concentration,
    finalize_spectrum,
    subtract_blanks,
)

__version__ = "0.2.0"
__all__ = [
    "AnalysisResult",
    "CombinedSpectrumTable",
    "CountsTable",
    "CumulativeSpectrumTable",
    "DifferentialSpectrumTable",
    "Experiment",
    "FrozenFractionTable",
    "MeasurementMetadata",
    "SampleMetadata",
    "analyze_concentration",
    "combine_dilutions",
    "convert_concentration",
    "cumulative_spectrum",
    "differential_spectrum",
    "finalize_spectrum",
    "frozen_fraction",
    "load",
    "read_counts",
    "read_icescopy",
    "read_observations",
    "resample_spectrum",
    "save",
    "subtract_blanks",
]
