"""Arithmetic concentration means retain shared-blank uncertainty explicitly."""

from math import erfc, sqrt
from statistics import NormalDist

import numpy as np
import pytest

from inptk._engine.math import PROFILE_LIKELIHOOD_DROP_95
from inptk._engine.water_blank_math import average_concentration, fit_concentration


@pytest.mark.parametrize("blank", [False, True])
@pytest.mark.parametrize("frozen", [0, 16, 32])
def test_single_measurement_is_exactly_the_ordinary_fit(blank, frozen):
    options = {"blank_frozen": 4, "blank_total": 32} if blank else {}
    expected = fit_concentration(frozen, 32, 13, 50, **options)
    actual = average_concentration(frozen, 32, 13, 50, **options)
    np.testing.assert_array_equal(actual, expected)


def test_mean_point_is_arithmetic_concentration_mean_with_unequal_counts_and_exposures():
    result = average_concentration([16, 4], [32, 10], [1, 13], [50, 20])
    individuals = -np.log1p(-np.array([16 / 32, 4 / 10])) * [1, 13] / [0.05, 0.02]
    assert result[0] == pytest.approx(np.mean(individuals), rel=1e-10)
    joint = fit_concentration([16, 4], [32, 10], [1, 13], [50, 20])
    assert abs(result[0] - joint[0]) > 1
    assert result[1] > 0 and result[2] > 0 and result[3]


def test_adjusted_bounds_equal_mean_of_individual_interval_endpoints_with_shared_blanks():
    frozen, total, dilution, volume = [16, 24], [32, 40], [1, 13], [50, 20]
    blanks = {
        "blank_frozen": [1, 2], "blank_total": [10, 20], "blank_volume_uL": [50, 20]
    }
    drop = PROFILE_LIKELIHOOD_DROP_95
    adjusted = NormalDist().inv_cdf(erfc(sqrt(drop)) / (2 * len(frozen)))**2 / 2
    individual = np.array([
        fit_concentration(x, n, d, v, confidence_drop=adjusted, **blanks)
        for x, n, d, v in zip(frozen, total, dilution, volume)
    ])
    mean, lower, upper, finite = average_concentration(
        frozen, total, dilution, volume, confidence_drop=drop, **blanks
    )
    assert mean == pytest.approx(individual[:, 0].mean(), rel=1e-10)
    assert mean - lower == pytest.approx((individual[:, 0] - individual[:, 1]).mean())
    assert mean + upper == pytest.approx((individual[:, 0] + individual[:, 2]).mean())
    assert finite


def test_four_versus_thirty_two_blank_droplets_changes_precision_not_the_subtracted_rate():
    small = average_concentration(
        [16, 24], [32, 32], [1, 2], 50, blank_frozen=1, blank_total=4
    )
    large = average_concentration(
        [16, 24], [32, 32], [1, 2], 50, blank_frozen=8, blank_total=32
    )
    individual_points = np.array([1, 2]) * (
        -np.log1p(-np.array([16, 24]) / 32) + np.log1p(-0.25)
    ) / 0.05
    assert small[0] == pytest.approx(individual_points.mean(), rel=1e-8)
    assert large[0] == pytest.approx(small[0], rel=1e-8)
    assert large[1] < small[1] and large[2] < small[2]
    assert small[3] and large[3]


def test_unequal_sample_and_blank_volumes_keep_their_physical_concentrations():
    result = average_concentration(
        [8, 16], [16, 32], [1, 3], [50, 100],
        blank_frozen=1, blank_total=10, blank_volume_uL=20,
    )
    expected = np.array([1, 3]) * (
        -np.log1p(-0.5) / np.array([0.05, 0.1]) + np.log1p(-0.1) / 0.02
    )
    assert result[0] == pytest.approx(expected.mean(), rel=1e-8)
    assert result[3]


def test_identical_points_stay_exactly_identical_when_contributor_count_changes():
    single = average_concentration(1, 32, 1, 50)
    multiple = average_concentration([1, 1, 1], 32, [1, 2, 3], [50, 100, 150])
    assert single[0] == multiple[0]
    assert multiple[3]


def test_shared_blank_intervals_do_not_assume_independent_background_errors():
    options = {"blank_frozen": 1, "blank_total": 4}
    single = average_concentration(16, 32, 1, 50, **options)
    multiple = average_concentration([16, 16], [32, 32], [1, 1], 50, **options)
    assert multiple[0] == single[0]
    # Bonferroni widens the marginal bounds here; duplicating their shared
    # background does not create an artificial square-root precision gain.
    assert multiple[1] >= single[1] and multiple[2] > single[2]


def test_zero_counts_keep_positive_upper_bounds():
    result = average_concentration(
        [0, 0], [32, 10], [1, 13], [50, 20],
        blank_frozen=0, blank_total=4, blank_volume_uL=50,
    )
    assert result[0] == result[1] == 0
    assert result[2] > 0 and result[3]


def test_infinite_and_unidentifiable_measurements_are_not_silently_dropped():
    saturated = average_concentration([32, 4], [32, 32], [1, 13], 50)
    assert np.isinf(saturated[0]) and np.isnan(saturated[1:3]).all()
    assert not saturated[3]
    unidentifiable = average_concentration(
        [32, 4], [32, 32], [1, 13], 50, blank_frozen=32, blank_total=32
    )
    assert np.isnan(unidentifiable[:3]).all() and not unidentifiable[3]


@pytest.mark.parametrize("changes", [
    {"n_frozen": [16, 0.5]}, {"n_total": [32, 0]}, {"dilution": [1, 0]},
    {"well_volume_uL": [50, np.nan]}, {"n_frozen": [[16, 4]]},
    {"n_frozen": [], "n_total": [], "dilution": [], "well_volume_uL": []},
    {"confidence_drop": 0}, {"confidence_drop": np.nan},
    {"blank_frozen": 0}, {"blank_total": 32}, {"blank_volume_uL": 50},
])
def test_invalid_inputs_fail_without_discarding_bad_measurements(changes):
    arguments = {
        "n_frozen": [16, 4], "n_total": [32, 32], "dilution": [1, 13],
        "well_volume_uL": 50,
    }
    arguments.update(changes)
    with pytest.raises(ValueError):
        average_concentration(**arguments)


def test_vector_volumes_require_explicit_blank_volumes_even_when_equal():
    with pytest.raises(ValueError, match="explicit blank_volume_uL"):
        average_concentration(
            [16, 4], [32, 32], [1, 13], [50, 50], blank_frozen=0, blank_total=32
        )


def test_underflowing_adjusted_error_probability_has_a_clear_error():
    with pytest.raises(ValueError, match="finite numerical range"):
        average_concentration([16, 4], [32, 32], [1, 13], 50, confidence_drop=1000)
