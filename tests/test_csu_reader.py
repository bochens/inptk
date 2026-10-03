"""Read reference spectra without recalculating values or assigning input roles."""

import hashlib

import numpy as np
import pandas as pd
import pytest

import inptk

TABLE = "degC,dilution,INPS_L,lower_CI,upper_CI\n-5,1,2,0.3,0.7\n-6,,4,0.5,1.1\n"


def test_reference_preserves_values_widths_order_header_text_and_source(tmp_path):
    path = tmp_path / "reference.csv"
    content = ("site = 001\nnotes = a=b\nvol_air_filt = 100\n"
               "vol_susp = 5\nproportion_filter_used = 0.5\n\n" + TABLE).encode()
    path.write_bytes(content)
    result = inptk.read_csu_csv(path)
    assert isinstance(result, inptk.CSUSpectrum)
    frame = result.table
    assert frame.temperature_C.tolist() == [-5, -6]
    assert frame.concentration.tolist() == [2, 4]
    assert frame.lower_error.tolist() == [.3, .5]
    assert frame.upper_error.tolist() == [.7, 1.1]
    assert frame.unit.eq("INP_per_L_air").all()
    assert frame.basis.eq("sampled_air").all()
    assert np.isnan(frame.dilution.iloc[1])
    assert result.metadata == {"site": "001", "notes": "a=b", "air_volume_L": 100.0,
                               "suspension_volume_mL": 5.0, "filter_fraction_used": .5}
    assert result.source["header"]["vol_air_filt"] == "100"
    assert result.source["sha256"] == hashlib.sha256(content).hexdigest()
    assert path.read_bytes() == content
    assert "sample_id" not in frame  # A descriptive site label is not an assignment.


def test_reference_can_be_read_without_normalization_metadata(tmp_path):
    path = tmp_path / "reference.csv"
    path.write_text("vol_susp = \n" + TABLE)
    result = inptk.read_csu_csv(path)
    assert result.metadata == {}
    assert len(result.table) == 2


@pytest.mark.parametrize("text, message", [
    ("site = X\n", "missing its degC"),
    ("site = X\nsite = Y\n" + TABLE, "Duplicate CSU"),
    ("unexpected header\n" + TABLE, "key=value"),
    ("degC,INPS_L\n-5,2\n", "missing columns"),
    ("vol_air_filt = -1\n" + TABLE, "finite and positive"),
    ("vol_air_filt = nan\n" + TABLE, "finite and positive"),
    ("proportion_filter_used = 2\n" + TABLE, "must not exceed 1"),
    (TABLE.replace("-5,1,2,0.3", "-5,1,2,-0.3"), "nonnegative error widths"),
    (TABLE.replace("-5,1,2", "inf,1,2"), "finite temperatures"),
])
def test_malformed_reference_fails_explicitly(tmp_path, text, message):
    path = tmp_path / "bad.csv"
    path.write_text(text)
    with pytest.raises(ValueError, match=message):
        inptk.read_csu_csv(path)


def test_reference_reader_preserves_nonfinite_estimates_instead_of_dropping_rows(tmp_path):
    path = tmp_path / "tail.csv"
    path.write_text(TABLE.replace("-6,,4,0.5,1.1", "-6,,nan,nan,inf"))
    frame = inptk.read_csu_csv(path).table
    assert len(frame) == 2
    assert pd.isna(frame.concentration.iloc[1])
    assert np.isinf(frame.upper_error.iloc[1])
