"""INP-toolkit: independent analysis of droplet-freezing experiments."""

from .experiment import AnalysisResult, Experiment, MeasurementMetadata, SampleMetadata
from .io import load, save
from .readers import read_counts, read_icescopy
from .tables import (
    CountsTable,
    CumulativeSpectrumTable,
    DifferentialSpectrumTable,
    FrozenFractionTable,
)
from .workflows import analyze_concentration, convert_concentration

__version__ = "0.2.0"
__all__ = [
    "AnalysisResult",
    "CountsTable",
    "CumulativeSpectrumTable",
    "DifferentialSpectrumTable",
    "Experiment",
    "FrozenFractionTable",
    "MeasurementMetadata",
    "SampleMetadata",
    "analyze_concentration",
    "convert_concentration",
    "load",
    "read_counts",
    "read_icescopy",
    "save",
]
