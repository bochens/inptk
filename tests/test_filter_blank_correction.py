from __future__ import annotations

import numpy as np
import pytest

import ufolaf


def _air_metadata(sample_id: str) -> ufolaf.SampleMetadata:
    return ufolaf.SampleMetadata(
        sample_id=sample_id,
        sample_type="air",
        well_volume_uL=50,
        dilution=1,
        air_volume_L=1000,
        filter_fraction_used=0.5,
        suspension_volume_mL=10,
    )


def _cumulative_spectrum(sample_id: str, value: float) -> ufolaf.CumulativeNucleusSpectrumTable:
    return ufolaf.CumulativeNucleusSpectrumTable(
        sample_id=[sample_id],
        temperature_C=[-10.0],
        value=[value],
        value_unit="INP_per_mL_suspension",
        basis="suspension",
        lower_ci=[0.1],
        upper_ci=[0.2],
        metadata=_air_metadata(sample_id),
    )


def test_filter_blank_correction_accepts_suspension_cumulative_spectra() -> None:
    sample = _cumulative_spectrum("sample", 2.0)
    blank = _cumulative_spectrum("blank", 0.5)

    corrected = ufolaf.subtract_filter_blank_spectrum(sample, blank, apply_qc=False)

    assert isinstance(corrected, ufolaf.CumulativeNucleusSpectrumTable)
    assert corrected.value_unit == "INP_per_mL_suspension"
    assert corrected.basis == "suspension"
    assert np.allclose(corrected.value, [1.5])


def test_filter_blank_correction_rejects_air_normalized_spectra() -> None:
    sample = ufolaf.normalize_spec(_cumulative_spectrum("sample", 2.0))
    blank = ufolaf.normalize_spec(_cumulative_spectrum("blank", 0.5))

    with pytest.raises(ValueError, match="before sample-basis normalization"):
        ufolaf.subtract_filter_blank_spectrum(sample, blank, apply_qc=False)
