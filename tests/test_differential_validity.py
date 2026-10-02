"""Differential output follows actual adjacent concentration states and intervals."""

import numpy as np
import pandas as pd
import pytest

import inptk


def experiment(*, temperatures=(-5, -6, -7), totals=(32, 32, 32), frozen=(0, 8, 16)):
    return inptk.read_counts(
        pd.DataFrame({
            "measurement_id": "M", "cycle_id": "1", "temperature_C": temperatures,
            "n_total": totals, "n_frozen": frozen,
        }),
        metadata=[{
            "measurement_id": "M", "sample_id": "S", "dilution": 1,
            "droplet_volume_uL": 50,
        }],
    )


def test_regular_differential_matches_original_fixed_total_count_formula():
    source = experiment()
    fractions = inptk.frozen_fraction(source)
    result = inptk.differential_spectrum(fractions, experiment=source).to_dataframe()
    expected = -np.log(np.array([24 / 32, 16 / 24])) / 0.05
    np.testing.assert_allclose(result.concentration, expected)
    assert result.temperature_bin_left_C.tolist() == [-6, -7]
    assert result.temperature_bin_right_C.tolist() == [-5, -6]
    assert result.temperature_C.tolist() == [-5, -6]


@pytest.mark.parametrize(
    "totals,frozen",
    [((32, 32, 24), (0, 8, 8)), ((32, 32, 28), (0, 12, 8))],
)
def test_changing_corrected_totals_use_each_states_fraction(totals, frozen):
    source = experiment(totals=totals, frozen=frozen)
    fractions = inptk.frozen_fraction(source)
    result = inptk.differential_spectrum(fractions, experiment=source).to_dataframe()
    concentrations = -np.log1p(-np.array(frozen) / np.array(totals)) / 0.05
    np.testing.assert_allclose(result.concentration, np.diff(concentrations))
    np.testing.assert_array_equal(result.qc_flag & 2, np.where(np.diff(concentrations) < 0, 2, 0))


def test_initial_frozen_count_does_not_invent_a_warm_zero_baseline():
    source = experiment(frozen=(1, 8, 16))
    fractions = inptk.frozen_fraction(source)
    result = inptk.differential_spectrum(fractions, experiment=source).to_dataframe()
    assert len(result) == 2
    assert result.temperature_bin_right_C.max() == -5
    assert result.concentration.iloc[0] == pytest.approx(-np.log(24 / 31) / 0.05)


def test_irregular_observations_use_actual_adjacent_interval_edges():
    source = experiment(temperatures=(0, -7.23, -9), frozen=(0, 1, 4))
    # These supplied states are observed at unequal temperature intervals.
    fractions = inptk.FrozenFractionTable(source.counts.to_dataframe())
    observed = fractions.to_dataframe().sort_values("temperature_C", ascending=False)
    result = inptk.differential_spectrum(fractions, experiment=source).to_dataframe()
    np.testing.assert_allclose(result.temperature_bin_left_C, observed.temperature_C.iloc[1:])
    np.testing.assert_allclose(result.temperature_bin_right_C, observed.temperature_C.iloc[:-1])
    widths = -np.diff(observed.temperature_C)
    concentration = -np.log1p(-observed.fraction_frozen.to_numpy()) / 0.05
    np.testing.assert_allclose(result.concentration, np.diff(concentration) / widths)
    assert "temperature_bin_width_C" not in result


def test_saturated_intervals_are_flagged_without_clipping():
    source = experiment(frozen=(0, 32, 32))
    fractions = inptk.frozen_fraction(source)
    result = inptk.differential_spectrum(fractions, experiment=source).to_dataframe()
    assert result.concentration.isna().all()
    assert (result.qc_flag & 1).eq(1).all()


def test_one_state_has_no_differential_interval():
    source = experiment(temperatures=(-5,), totals=(32,), frozen=(8,))
    fractions = inptk.frozen_fraction(source)
    result = inptk.differential_spectrum(fractions, experiment=source).to_dataframe()
    assert result.empty
    assert {"temperature_bin_left_C", "temperature_bin_right_C"}.issubset(result.columns)
