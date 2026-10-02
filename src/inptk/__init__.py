"""INP-toolkit: independent analysis of droplet-freezing experiments."""

from .experiment import AnalysisResult, Experiment, MeasurementMetadata, SampleMetadata
from .io import load, save
from .methods import MLE, ManualStitch, Stitch
from .processing import cumulative_spectrum, differential_spectrum, frozen_fraction
from .readers import read_counts, read_icescopy
from .tables import (
    CountsTable,
    CumulativeSpectrumTable,
    DifferentialSpectrumTable,
    FrozenFractionTable,
)
from .workflows import (
    analyze_concentration,
    combine_dilutions,
    convert_concentration,
    subtract_blanks,
)

__version__ = "0.2.0"
__all__ = [
    "MLE",
    "AnalysisResult",
    "CountsTable",
    "CumulativeSpectrumTable",
    "DifferentialSpectrumTable",
    "Experiment",
    "FrozenFractionTable",
    "ManualStitch",
    "MeasurementMetadata",
    "SampleMetadata",
    "Stitch",
    "analyze_concentration",
    "combine_dilutions",
    "convert_concentration",
    "cumulative_spectrum",
    "differential_spectrum",
    "frozen_fraction",
    "load",
    "read_counts",
    "read_icescopy",
    "save",
    "subtract_blanks",
]
