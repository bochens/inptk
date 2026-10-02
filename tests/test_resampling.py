"""Output sampling keeps native states and cannot hide excluded intervals."""

import json

import numpy as np
import pandas as pd
import pytest

from inptk.resampling import resample_spectrum
from inptk.tables import CombinedSpectrumTable


def spectrum(temperatures, values, *, segments=None, kept=None, lower=None, upper=None):
    size = len(temperatures)
    return CombinedSpectrumTable(
        pd.DataFrame(
            {
                "sample_id": "A",
                "group_id": "G",
                "point_id": [f"native:{i}" for i in range(size)],
                "temperature_C": temperatures,
                "concentration": values,
                "unit": "INP_per_mL_suspension",
                "basis": "suspension",
                "point_order": range(size),
                "segment_id": ["s0"] * size if segments is None else segments,
                "used_in_final": [True] * size if kept is None else kept,
                "lower_error": [1.0] * size if lower is None else lower,
                "upper_error": [2.0] * size if upper is None else upper,
                "uncertainty_method": "joint_sample_water_blank_profile_likelihood",
            }
        ),
        history=[{"operation": "native_fixture"}],
    )


def test_sampling_copies_latest_observed_warmer_state_and_preserves_input():
    source = spectrum([-5, -6, -5.8, -7], [1, 2, 3, 4])
    original, history = source.to_dataframe(), source.history
    sampled = resample_spectrum(source, 0.5).to_dataframe().set_index("temperature_C")
    assert sampled.loc[-6, "concentration"] == 3
    assert sampled.loc[-6, "lower_error"] == 1
    assert sampled.loc[-6, "upper_error"] == 2
    assert sampled.loc[-6, "sampling_status"] == "sampled"
    assert json.loads(sampled.loc[-6, "source_point_ids"]) == ["native:2"]
    assert json.loads(sampled.loc[-6, "source_temperatures_C"]) == [-5.8]
    assert not sampled.is_extrapolated.any()
    assert not sampled.is_interpolated.any()
    pd.testing.assert_frame_equal(source.to_dataframe(), original)
    assert source.history == history


def test_interpolation_uses_interval_endpoints_and_latest_duplicate_temperature():
    source = spectrum([-5, -5, -7], [5, 10, 20], lower=[1, 2, 4], upper=[2, 3, 5])
    output = resample_spectrum(source, 1, method="interpolate")
    rows = output.to_dataframe().set_index("temperature_C")
    assert rows.loc[-5, "concentration"] == 10
    assert rows.loc[-5, "sampling_status"] == "exact"
    middle = rows.loc[-6]
    assert middle.concentration == 15
    assert middle.concentration - middle.lower_error == 12
    assert middle.concentration + middle.upper_error == 19
    assert middle.sampling_status == "interpolated"
    assert middle.uncertainty_method == "interpolated_interval_endpoints"
    assert set(json.loads(middle.source_point_ids)) == {"native:1", "native:2"}
    assert not {"n_total", "n_frozen"} & set(output.columns)
    assert output.history[-1]["operation"] == "resample_spectrum"
    assert rows.point_id.is_unique


@pytest.mark.parametrize("method", ["sample", "interpolate"])
def test_grid_does_not_extrapolate_outside_native_segment(method):
    source = spectrum([-5.2, -6.8], [10, 20])
    rows = resample_spectrum(source, 0.5, method=method).to_dataframe()
    assert rows.temperature_C.tolist() == [-5.5, -6, -6.5]
    assert not rows.is_extrapolated.any()


@pytest.mark.parametrize("method", ["sample", "interpolate"])
def test_excluded_or_nonfinite_rows_are_not_bridged_even_with_same_segment_id(method):
    source = spectrum(
        [-5, -6, -7, -8, -9], [10, 12, np.nan, 18, 20], kept=[True, True, False, True, True]
    )
    rows = resample_spectrum(source, 0.5, method=method).to_dataframe()
    assert set(rows.temperature_C) == {-5, -5.5, -6, -8, -8.5, -9}
    assert rows.point_id.is_unique


@pytest.mark.parametrize("method", ["sample", "interpolate"])
def test_retained_segment_ids_prevent_interpolation_across_removed_rows(method):
    source = spectrum([-5, -6, -8, -9], [10, 12, 18, 20], segments=["a", "a", "b", "b"])
    rows = resample_spectrum(source, 0.5, method=method).to_dataframe()
    assert set(rows.temperature_C) == {-5, -5.5, -6, -8, -8.5, -9}
    assert set(rows.segment_id) == {"a", "b"}


def test_singleton_segments_only_supply_their_exact_grid_temperature():
    source = spectrum([-5.2, -6], [10, 20], segments=["a", "b"])
    rows = resample_spectrum(source, 0.5).to_dataframe()
    assert rows.temperature_C.tolist() == [-6]
    empty = resample_spectrum(spectrum([-5.2], [10]), 0.5)
    assert len(empty) == 0
    assert {"point_id", "sampling_status", "source_point_ids"}.issubset(empty.columns)


def test_decimal_grid_edges_do_not_drop_exact_observations():
    rows = resample_spectrum(spectrum([-0.1, -0.3], [1, 2]), 0.1).to_dataframe()
    assert rows.temperature_C.tolist() == [-0.1, -0.2, -0.3]


def test_groups_keep_distinct_point_identity_and_do_not_mix():
    first = spectrum([-5, -7], [10, 20]).to_dataframe()
    second = spectrum([-5, -7], [100, 200]).to_dataframe().assign(group_id="H", sample_id="B")
    output = resample_spectrum(CombinedSpectrumTable(pd.concat([first, second])), 1)
    rows = output.to_dataframe()
    assert not rows.duplicated(["group_id", "point_id"]).any()
    assert rows.groupby("sample_id").concentration.max().to_dict() == {"A": 20, "B": 200}


@pytest.mark.parametrize("step", [0, -1, np.nan, np.inf, True])
def test_invalid_grid_spacing_is_rejected(step):
    with pytest.raises(ValueError, match="step_C"):
        resample_spectrum(spectrum([-5, -6], [1, 2]), step)


def test_resampling_requires_final_segment_provenance():
    source = spectrum([-5, -6], [1, 2]).to_dataframe().drop(columns="segment_id")
    with pytest.raises(ValueError, match="retained-segment provenance"):
        resample_spectrum(CombinedSpectrumTable(source), 0.5)
    with pytest.raises(ValueError, match="method"):
        resample_spectrum(spectrum([-5, -6], [1, 2]), 0.5, method="nearest")


def test_interpolation_uses_latest_warmer_endpoints_so_jitter_does_not_reverse_concentration():
    source = spectrum([-5, -6, -5.8, -7], [1, 2, 3, 4], lower=[1, 2, 3, 4])
    original = source.to_dataframe()
    rows = resample_spectrum(source, 0.5, method="interpolate").to_dataframe()
    np.testing.assert_allclose(rows.concentration, [1, 2.25, 3, 3.5, 4])
    assert np.all(np.diff(rows.concentration) >= 0)
    aligned = rows.loc[rows.temperature_C.eq(-6)].iloc[0]
    assert aligned.concentration == aligned.lower_error == 3
    assert json.loads(aligned.source_point_ids) == ["native:2"]
    assert json.loads(aligned.source_temperatures_C) == [-5.8]
    assert json.loads(aligned.endpoint_temperatures_C) == [-6]
    middle = rows.loc[rows.temperature_C.eq(-6.5)].iloc[0]
    assert json.loads(middle.source_temperatures_C) == [-7, -5.8]
    assert json.loads(middle.endpoint_temperatures_C) == [-7, -6]
    assert json.loads(middle.source_point_ids) == ["native:3", "native:2"]
    pd.testing.assert_frame_equal(source.to_dataframe(), original)


def test_interpolating_a_repeated_native_state_preserves_its_values_exactly():
    source = spectrum([-5, -6, -5.8, -7], [1, 2, 3, 4], lower=[0.1] * 4)
    rows = resample_spectrum(source, 0.1, method="interpolate").to_dataframe()
    plateau = rows.loc[rows.temperature_C.eq(-5.9)].iloc[0]
    assert plateau.concentration == 3
    assert plateau.lower_error == 0.1
    assert json.loads(plateau.source_point_ids) == ["native:2", "native:2"]
    assert json.loads(plateau.endpoint_temperatures_C) == [-6, -5.8]


@pytest.mark.parametrize("method", ["sample", "interpolate"])
def test_existing_source_extrapolation_flags_are_preserved_without_new_extrapolation(method):
    frame = spectrum([-5, -7], [1, 4]).to_dataframe()
    frame["is_extrapolated"] = [True, False]
    source = CombinedSpectrumTable(frame)
    output = resample_spectrum(source, 1, method=method)
    rows = output.to_dataframe()
    assert rows.is_extrapolated.tolist() == [True, True, False]
    assert output.history[-1]["extrapolation"] is False
    assert output.history[-1]["source_extrapolation_flags"] == "preserved"
