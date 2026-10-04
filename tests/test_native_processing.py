"""Processing keeps observations intact and aligns only the required blank states."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from inptk._engine.water_blank_math import fit_concentration
from inptk.experiment import Experiment, MeasurementMetadata, SampleMetadata
from inptk.processing import cumulative_spectrum, differential_spectrum, frozen_fraction
from inptk.tables import CountsTable, FrozenFractionTable
from inptk.water_blank import estimate_point


def experiment(temperatures, frozen, *, blanks=None, blank_temperatures=None, cycles=("01",)):
    metadata = {"M": MeasurementMetadata("M", "sample", 1, 50, "run")}
    samples = {"sample": SampleMetadata("sample")}
    rows = []
    specs = [("M", "sample", temperatures, frozen)]
    if blanks is not None:
        metadata["B"] = MeasurementMetadata("B", "water", 1, 50, "run")
        samples["water"] = SampleMetadata("water")
        specs.append(("B", "water", blank_temperatures or temperatures, blanks))
    for measurement, parent, observed_temperatures, counts in specs:
        for cycle in cycles:
            for index, (temperature, count) in enumerate(zip(observed_temperatures, counts)):
                rows.append({
                    "measurement_id": measurement, "sample_id": parent, "run_id": "run",
                    "cycle_id": cycle, "observation_id": f"{measurement}:{cycle}:{index}",
                    "temperature_C": temperature, "time_s": float(index),
                    "n_frozen": count, "n_total": 32,
                })
    return Experiment(
        CountsTable(pd.DataFrame(rows)), samples, metadata,
        water_blank_map={"M": ["B"]} if blanks is not None else {},
    )


def test_fraction_keeps_duplicate_temperatures_warming_counts_and_original_order():
    source = experiment([-5, -6, -6, -5.9, -7, -6.9], [0, 4, 8, 9, 16, 16])
    before = source.counts.to_dataframe()
    fraction = frozen_fraction(source)
    actual = fraction.to_dataframe()
    pd.testing.assert_frame_equal(actual[before.columns], before)
    np.testing.assert_array_equal(actual.fraction_frozen, before.n_frozen / before.n_total)
    pd.testing.assert_frame_equal(source.counts.to_dataframe(), before)
    assert fraction.history[-1]["observation_selection"] == "none"


@pytest.mark.parametrize("option", ["step_C", "temperature_method", "temperature_tolerance_C"])
def test_fraction_no_longer_accepts_temperature_selection_options(option):
    source = experiment([-5, -6], [0, 4])
    with pytest.raises(TypeError, match=option):
        frozen_fraction(source, **{option: 0.5})


@pytest.mark.parametrize("method", ["mle", "average"])
def test_individual_estimates_keep_native_rows_without_pooling_repeated_temperatures(method):
    source = experiment([-5, -6, -6, -5.9, -7, -6.9], [0, 4, 8, 9, 16, 16])
    actual = cumulative_spectrum(
        frozen_fraction(source), experiment=source, method=method
    ).to_dataframe()
    fitted_counts = [0, 9, 9, 9, 16, 16] if method == "mle" else [0, 4, 8, 9, 16, 16]
    expected = -np.log1p(-np.array(fitted_counts) / 32) / 0.05
    np.testing.assert_allclose(actual.concentration, expected, atol=1e-7, rtol=1e-7)
    assert actual.n_frozen.tolist() == [0, 4, 8, 9, 16, 16]
    assert actual.observation_id.tolist() == source.counts.to_dataframe().observation_id.tolist()
    assert actual.point_order.tolist() == list(range(6))
    assert actual.point_id.is_unique
    assert actual.alignment.eq("native").all()
    np.testing.assert_array_equal(actual.observed_temperature_C, actual.temperature_C)


def test_native_estimates_follow_actual_time_without_mutating_source_row_order():
    original = experiment([-5, -6, -7, -8], [0, 4, 8, 16])
    shuffled = original.counts.to_dataframe().iloc[[3, 0, 2, 1]].reset_index(drop=True)
    source = Experiment(CountsTable(shuffled), original.samples, original.measurements)
    actual = cumulative_spectrum(frozen_fraction(source), experiment=source).to_dataframe()
    assert actual.time_s.tolist() == [0, 1, 2, 3]
    assert actual.observation_id.tolist() == [f"M:01:{index}" for index in range(4)]
    pd.testing.assert_frame_equal(source.counts.to_dataframe(), shuffled)


def test_matching_blank_acquisitions_remain_paired_at_duplicate_temperatures():
    source = experiment([-5, -6, -6, -7], [4, 8, 16, 20], blanks=[0, 1, 5, 8])
    actual = cumulative_spectrum(
        frozen_fraction(source), experiment=source, method="average"
    ).to_dataframe()
    expected = [
        fit_concentration(x, 32, 1, 50, blank_frozen=b, blank_total=32, confidence_drop=1.96**2 / 2)
        for x, b in zip([4, 8, 16, 20], [0, 1, 5, 8])
    ]
    np.testing.assert_allclose(actual.concentration, np.array(expected)[:, 0])
    assert actual.blank_n_frozen.tolist() == [0, 1, 5, 8]
    for position, serialized in enumerate(actual.water_blank_observations):
        assert json.loads(serialized)[0]["observation_id"] == f"B:01:{position}"
    assert actual.alignment.eq("native").all()


def test_unsynchronized_blank_uses_latest_observed_state_and_records_its_actual_temperature():
    source = experiment([-5, -6, -7], [4, 8, 16], blanks=[0, 2, 4],
                        blank_temperatures=[-4.9, -5.9, -7.1])
    actual = cumulative_spectrum(frozen_fraction(source), experiment=source).to_dataframe()
    assert actual.temperature_C.tolist() == [-5, -6, -7]
    assert actual.time_s.tolist() == [0, 1, 2]
    assert actual.blank_n_frozen.tolist() == [0, 2, 2]
    assert actual.alignment.eq("latest").all()
    selected = [json.loads(value)[0] for value in actual.water_blank_observations]
    assert [row["observed_temperature_C"] for row in selected] == [-4.9, -5.9, -5.9]
    assert [row["observation_id"] for row in selected] == ["B:01:0", "B:01:1", "B:01:1"]


def test_combined_exclusions_do_not_hide_missing_individual_blank_coverage():
    source = experiment([-5, -6, -7], [4, 8, 16], blanks=[2, 4], blank_temperatures=[-6, -7])
    with pytest.raises(ValueError, match="no blank extrapolation"):
        cumulative_spectrum(
            frozen_fraction(source), experiment=source,
            temperature_ranges_C={"M": {"min_C": -7, "max_C": -6}},
        )


def test_missing_blank_coverage_for_eligible_native_target_is_rejected():
    source = experiment([-5, -6, -7], [4, 8, 16], blanks=[2, 4], blank_temperatures=[-6, -7])
    with pytest.raises(ValueError, match="no blank extrapolation"):
        cumulative_spectrum(frozen_fraction(source), experiment=source)


def test_repeated_count_states_reuse_fit_but_new_blank_state_or_cycle_does_not():
    source = experiment([-5, -6, -7], [8, 8, 8], blanks=[1, 1, 2], cycles=("01", "1"))
    with patch("inptk.estimation.estimate_point", wraps=estimate_point) as estimator:
        actual = cumulative_spectrum(
            frozen_fraction(source), experiment=source, method="average"
        ).to_dataframe()
    assert estimator.call_count == 4
    assert len(actual) == 6
    assert set(actual.cycle_id) == {"01", "1"}


def test_differential_uses_only_adjacent_cooling_transitions_without_bridging_jitter():
    source = experiment([-5, -6, -6, -5.8, -7], [0, 4, 8, 9, 16])
    fractions = frozen_fraction(source)
    cumulative = cumulative_spectrum(fractions, experiment=source).to_dataframe()
    differential = differential_spectrum(fractions, experiment=source)
    actual = differential.to_dataframe()
    assert actual.observation_id.tolist() == ["M:01:0", "M:01:3"]
    assert actual.next_observation_id.tolist() == ["M:01:1", "M:01:4"]
    expected = [
        cumulative.concentration.iloc[1] - cumulative.concentration.iloc[0],
        (cumulative.concentration.iloc[4] - cumulative.concentration.iloc[3]) / 1.2,
    ]
    np.testing.assert_allclose(actual.concentration, expected)
    omitted = differential.history[-1]["omitted_transitions"]
    assert [row["reason"] for row in omitted] == ["repeated_temperature", "warming"]
    assert omitted[0]["from_observation_id"] == "M:01:1"
    assert omitted[1]["to_observation_id"] == "M:01:3"


def test_differential_ignores_combined_exclusions_and_keeps_cycles_separate():
    source = experiment([-5, -6, -7], [0, 4, 16], cycles=("01", "1"))
    result = differential_spectrum(
        frozen_fraction(source), experiment=source, temperature_ranges_C={"M": {"max_C": -6}}
    ).to_dataframe()
    assert len(result) == 4
    for _, cycle in result.groupby("cycle_id"):
        assert np.isfinite(cycle.concentration).all()
        assert cycle.qc_flag.eq(0).all()


@pytest.mark.parametrize(
    "temperatures,frozen", [([-5], [4]), ([-6, -5], [4, 5]), ([-5, -5], [4, 8])]
)
def test_no_cooling_interval_produces_an_empty_valid_differential_table(temperatures, frozen):
    source = experiment(temperatures, frozen)
    result = differential_spectrum(frozen_fraction(source), experiment=source)
    assert result.to_dataframe().empty


def test_legacy_synthetic_warm_zero_rows_are_not_treated_as_observed_counts():
    source = experiment([-5, -6], [0, 4])
    fractions = FrozenFractionTable(source.counts.to_dataframe(), history=[{
        "operation": "frozen_fraction", "temperature_method": "window_max_count",
    }])
    with pytest.raises(ValueError, match="synthetic warm zero rows"):
        cumulative_spectrum(fractions, experiment=source)
