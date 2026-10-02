from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import inptk
from inptk import _engine as ufolaf


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


@pytest.mark.parametrize("private_type", ["cumulative", "normalized"])
def test_private_blank_subtraction_crosses_asymmetric_error_directions(private_type):
    cls = (ufolaf.CumulativeNucleusSpectrumTable if private_type == "cumulative"
           else ufolaf.NormalizedInpSpectrumTable)
    def spectrum(name, value, lower, upper):
        return cls(
            sample_id=[name], temperature_C=[-10], value=[value],
            value_unit="INP_per_mL_suspension", basis="suspension",
            lower_ci=[lower], upper_ci=[upper],
        )
    result = ufolaf.subtract_filter_blank_spectrum(
        spectrum("sample", 10., 1., 2.), spectrum("blank", 1., 3., 4.),
        apply_qc=False, extrapolate_missing_cold=False,
    ).to_dataframe()
    assert result.value.iloc[0] == 9
    assert result.lower_ci.iloc[0] == pytest.approx(np.sqrt(1**2 + 4**2))
    assert result.upper_ci.iloc[0] == pytest.approx(np.sqrt(2**2 + 3**2))


def test_public_blank_subtraction_crosses_asymmetric_error_directions():
    def spectrum(name, value, lower, upper):
        return inptk.CumulativeSpectrumTable(pd.DataFrame({
            "run_id": ["1"], "sample_id": [name], "cycle_id": ["1"],
            "temperature_C": [-10], "concentration": [value],
            "unit": ["INP_per_mL_suspension"], "basis": ["suspension"],
            "lower_error": [lower], "upper_error": [upper],
        }))
    result = inptk.subtract_blanks(
        spectrum("sample", 10., 1., 2.), {"sample": spectrum("blank", 1., 3., 4.)}
    ).to_dataframe()
    assert result.concentration.iloc[0] == 9
    assert result.lower_error.iloc[0] == pytest.approx(np.sqrt(1**2 + 4**2))
    assert result.upper_error.iloc[0] == pytest.approx(np.sqrt(2**2 + 3**2))
