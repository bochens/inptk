"""INP-toolkit: independent analysis of droplet-freezing experiments."""

from .experiment import AnalysisResult, CurveResult, Experiment, MeasurementMetadata, SampleMetadata
from .io import load, save
from .processing import cumulative_spectrum, differential_spectrum, frozen_fraction
from .readers import read_counts, read_icescopy, read_observations
from .resampling import resample_spectrum
from .tables import (
    CountsTable,
    CumulativeSpectrumTable,
    CurveSpectrumTable,
    DifferentialSpectrumTable,
    FrozenFractionTable,
)
from .workflows import (
    analyze_concentration,
    convert_concentration,
    estimate_concentration,
    finalize_spectrum,
    subtract_blanks,
)

__version__ = "0.4.0"
__all__ = [
    "AnalysisResult",
    "CountsTable",
    "CumulativeSpectrumTable",
    "CurveResult",
    "CurveSpectrumTable",
    "DifferentialSpectrumTable",
    "Experiment",
    "FrozenFractionTable",
    "MeasurementMetadata",
    "SampleMetadata",
    "analyze_concentration",
    "convert_concentration",
    "cumulative_spectrum",
    "differential_spectrum",
    "estimate_concentration",
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
