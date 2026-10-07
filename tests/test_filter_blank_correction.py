"""Filter-blank subtraction through the public spectrum API."""

import numpy as np
import pandas as pd
import pytest

import inptk


def spectrum(name, value, lower=.1, upper=.2, basis="suspension"):
    return inptk.CumulativeSpectrumTable(pd.DataFrame({
        "run_id": ["1"], "sample_id": [name], "cycle_id": ["1"],
        "temperature_C": [-10], "concentration": [value],
        "unit": ["INP_per_mL_suspension" if basis == "suspension" else "INP_per_L_air"],
        "basis": [basis], "lower_error": [lower], "upper_error": [upper],
    }))


def test_filter_blank_correction_accepts_suspension_cumulative_spectra():
    result = inptk.subtract_blanks(spectrum("sample", 2.0), {"sample": spectrum("blank", .5)})
    assert isinstance(result, inptk.CumulativeSpectrumTable)
    row = result.to_dataframe().iloc[0]
    assert row.concentration == 1.5
    assert row.unit == "INP_per_mL_suspension" and row.basis == "suspension"


def test_filter_blank_correction_rejects_air_normalized_spectra():
    with pytest.raises(ValueError, match="requires suspension concentrations"):
        inptk.subtract_blanks(spectrum("sample", 2.0, basis="sampled_air"),
                             {"sample": spectrum("blank", .5, basis="sampled_air")})


def test_public_blank_subtraction_crosses_asymmetric_error_directions():
    result = inptk.subtract_blanks(
        spectrum("sample", 10., 1., 2.), {"sample": spectrum("blank", 1., 3., 4.)}
    ).to_dataframe()
    assert result.concentration.iloc[0] == 9
    assert result.lower_error.iloc[0] == pytest.approx(np.sqrt(1**2 + 4**2))
    assert result.upper_error.iloc[0] == pytest.approx(np.sqrt(2**2 + 3**2))
