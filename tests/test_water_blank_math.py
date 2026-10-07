"""Joint raw-count inference includes each physical water blank exactly once."""

import numpy as np
import pytest
from reference_concentration_math import binomial_poisson_mle_with_profile_errors

from inptk._engine.water_blank_math import joint_water_blank_mle


def fit(frozen, total, dilution=1, blank_frozen=4, blank_total=32, volume=50, **kwargs):
    return joint_water_blank_mle(
        frozen, total, dilution, volume, blank_frozen, blank_total, **kwargs
    )


def grid_log_likelihood(c, background, frozen, total, dilution, blank_frozen, blank_total):
    """Independent direct likelihood for numerical grids; no fitting helpers."""
    c, background = np.broadcast_arrays(np.asarray(c), np.asarray(background))
    result = np.zeros(c.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        for x, n, d in zip(frozen, total, dilution):
            hazard = c / d + background
            if x:
                result += x * np.log1p(-np.exp(-hazard))
            result -= (n - x) * hazard
        if blank_frozen:
            result += blank_frozen * np.log1p(-np.exp(-background))
        result -= (blank_total - blank_frozen) * background
    return result


def test_single_dilution_interior_estimate_matches_analytic_background_subtraction():
    result = fit(16, 32, dilution=13)
    expected = 13 * (-np.log(1 - 16 / 32) + np.log(1 - 4 / 32)) / 0.05
    assert result[0] == pytest.approx(expected, rel=1e-8)
    assert result[1] > 0 and result[2] > 0 and result[3]


def test_interior_fit_agrees_with_independent_two_dimensional_grid():
    frozen, total, dilution = [20, 12], [32, 32], [1, 10]
    estimate, lower, upper, finite = fit(frozen, total, dilution, blank_frozen=2)
    c_grid = np.linspace(0, 2, 1001)
    background_grid = np.linspace(0, 0.8, 801)
    likelihood = grid_log_likelihood(
        c_grid[:, None], background_grid[None, :], frozen, total, dilution, 2, 32
    )
    c_index, _ = np.unravel_index(np.argmax(likelihood), likelihood.shape)
    assert estimate * 0.05 == pytest.approx(c_grid[c_index], abs=0.002)
    assert lower > 0 and upper > 0 and finite


def test_profile_limits_match_independent_background_grid_maximization():
    drop = 1.920729410347062
    frozen, total, dilution = [20, 12], [32, 32], [1, 10]
    estimate, lower, upper, _ = fit(frozen, total, dilution, blank_frozen=2)
    background_grid = np.linspace(0, 0.8, 40001)
    profile_values = [
        np.max(grid_log_likelihood(k * 0.05, background_grid, frozen, total, dilution, 2, 32))
        for k in (estimate, estimate - lower, estimate + upper)
    ]
    assert profile_values[0] - profile_values[1] == pytest.approx(drop, abs=2e-6)
    assert profile_values[0] - profile_values[2] == pytest.approx(drop, abs=2e-6)


def test_independent_sample_rows_do_not_duplicate_the_shared_blank_information():
    # Two independent sample droplet sets at the same dilution have the same
    # likelihood as their summed counts, with the physical blank included once.
    split = fit([12, 18], [32, 40], [5, 5], blank_frozen=3, blank_total=20)
    summed = fit(30, 72, 5, blank_frozen=3, blank_total=20)
    np.testing.assert_allclose(split[:3], summed[:3], rtol=1e-8, atol=1e-8)
    incorrectly_duplicated = fit(30, 72, 5, blank_frozen=6, blank_total=40)
    assert split[1] > incorrectly_duplicated[1]
    assert split[2] > incorrectly_duplicated[2]


def test_zero_frozen_blank_is_not_treated_as_known_zero_background():
    joint = fit(1, 32, blank_frozen=0, blank_total=2)
    known_zero = binomial_poisson_mle_with_profile_errors(1, 32, 50, 1)
    assert joint[0] == pytest.approx(known_zero[0], rel=1e-8)
    assert joint[0] - joint[1] == pytest.approx(0)
    assert known_zero[0] - known_zero[1] > 0
    assert joint[1] > known_zero[1]
    assert joint[2] > 0 and joint[3]


def test_more_independent_blank_droplets_reduce_background_uncertainty():
    small_blank = fit(16, 32, blank_frozen=4, blank_total=32)
    large_blank = fit(16, 32, blank_frozen=4000, blank_total=32000)
    assert large_blank[0] == pytest.approx(small_blank[0], rel=1e-8)
    assert large_blank[1] < small_blank[1]
    assert large_blank[2] < small_blank[2]


def test_all_zero_counts_keep_a_nonzero_upper_concentration_limit():
    drop = 1.920729410347062
    result = fit([0, 0], [32, 16], [1, 10], blank_frozen=0)
    expected_upper = drop / (0.05 * (32 / 1 + 16 / 10))
    assert result[0] == result[1] == 0
    assert result[2] == pytest.approx(expected_upper, rel=1e-8)
    assert result[3]


def test_sample_below_blank_has_zero_boundary_estimate_and_available_upper_limit():
    result = fit(1, 32, blank_frozen=8)
    assert result[0] == result[1] == 0
    assert result[2] > 0 and result[3]


def test_saturation_is_reported_without_fabricating_a_finite_estimate():
    saturated_sample = fit([32, 16], [32, 16], [1, 10], blank_frozen=2)
    assert np.isinf(saturated_sample[0])
    assert np.isnan(saturated_sample[1:3]).all()
    assert not saturated_sample[3]
    saturated_blank = fit(16, 32, blank_frozen=32)
    assert saturated_blank[0] == saturated_blank[1] == 0
    assert saturated_blank[2] > 0 and saturated_blank[3]
    saturated_everything = fit(32, 32, blank_frozen=32)
    assert np.isnan(saturated_everything[:3]).all()
    assert not saturated_everything[3]
    mixed_samples = fit([32, 8], [32, 32], [1, 10], blank_frozen=2)
    assert np.isfinite(mixed_samples[:3]).all() and mixed_samples[3]


def test_droplet_volume_rescales_estimate_and_errors_consistently():
    small = fit([20, 12], [32, 32], [1, 10], blank_frozen=2, volume=25)
    large = fit([20, 12], [32, 32], [1, 10], blank_frozen=2, volume=50)
    np.testing.assert_allclose(small[:3], 2 * np.array(large[:3]), rtol=1e-10)


@pytest.mark.parametrize("changes", [
    {"n_frozen": [0.5]}, {"n_frozen": [-1]}, {"n_frozen": [33]},
    {"n_frozen": [] , "n_total": [], "dilution": []},
    {"n_frozen": [[1, 2]], "n_total": [[32, 32]], "dilution": [[1, 10]]},
    {"n_total": 0}, {"n_total": np.nan}, {"dilution": 0}, {"dilution": np.inf},
    {"well_volume_uL": 0}, {"well_volume_uL": np.inf},
    {"blank_frozen": 0.5}, {"blank_frozen": 33}, {"blank_total": 0},
    {"confidence_drop": 0}, {"confidence_drop": np.nan},
])
def test_invalid_raw_counts_and_settings_fail_clearly(changes):
    arguments = {
        "n_frozen": 16, "n_total": 32, "dilution": 1, "well_volume_uL": 50,
        "blank_frozen": 4, "blank_total": 32,
    }
    arguments.update(changes)
    with pytest.raises(ValueError):
        joint_water_blank_mle(**arguments)



def test_more_comparable_blank_sets_change_precision_not_background_subtraction():
    # Two distinct 32-droplet blanks each have four frozen droplets. Their
    # shared-background observation is 8/64, not eight frozen out of 32.
    one_blank = fit(16, 32, blank_frozen=4, blank_total=32)
    two_blanks = fit(16, 32, blank_frozen=4 + 4, blank_total=32 + 32)
    expected = (-np.log(1 - 16 / 32) + np.log(1 - 4 / 32)) / 0.05
    assert one_blank[0] == pytest.approx(expected, rel=1e-8)
    assert two_blanks[0] == pytest.approx(expected, rel=1e-8)
    assert two_blanks[1] < one_blank[1]
    assert two_blanks[2] < one_blank[2]


def test_pooled_independent_blank_likelihood_equals_product_of_original_blank_terms():
    c = np.linspace(0.1, 1, 11)[:, None]
    b = np.linspace(0.05, 0.5, 13)[None, :]
    sample_only = grid_log_likelihood(c, b, [16, 4], [32, 32], [1, 10], 0, 0)
    separate = sample_only.copy()
    blank_counts = [(2, 16), (5, 32)]
    for frozen, total in blank_counts:
        separate += frozen * np.log1p(-np.exp(-b)) - (total - frozen) * b
    pooled = grid_log_likelihood(c, b, [16, 4], [32, 32], [1, 10], 7, 48)
    np.testing.assert_allclose(separate, pooled, atol=1e-12)



@pytest.mark.parametrize("sample_volume,blank_volume", [(50, 20), (20, 50), (50, 100)])
def test_unequal_volumes_match_single_pair_analytic_concentration_subtraction(
    sample_volume, blank_volume,
):
    result = fit(
        16, 32, dilution=13, blank_frozen=4, blank_total=20,
        volume=sample_volume, blank_volume_uL=blank_volume,
    )
    expected = 13 * (
        -np.log(1 - 16 / 32) / (sample_volume / 1000)
        + np.log(1 - 4 / 20) / (blank_volume / 1000)
    )
    assert result[0] == pytest.approx(expected, rel=1e-8)
    assert result[1] > 0 and result[2] > 0 and result[3]


def test_blanks_with_different_volumes_keep_their_distinct_exposures():
    # 8/16 at 50 uL and 12/16 at 100 uL have exactly the same estimated
    # background concentration: doubling volume doubles the freezing hazard.
    result = fit(
        24, 32, volume=50, blank_frozen=[8, 12], blank_total=[16, 16],
        blank_volume_uL=[50, 100],
    )
    expected = (-np.log(1 - 24 / 32) + np.log(1 - 8 / 16)) / 0.05
    assert result[0] == pytest.approx(expected, rel=1e-8)
    assert result[3]
    incorrectly_pooled = fit(24, 32, blank_frozen=20, blank_total=32, volume=50)
    assert abs(incorrectly_pooled[0] - expected) > 1


def test_distinct_blank_terms_at_equal_volume_match_count_pooling():
    separate = fit(
        [16, 4], [32, 32], [1, 10], blank_frozen=[4, 8], blank_total=[32, 64],
        blank_volume_uL=[50, 50],
    )
    pooled = fit([16, 4], [32, 32], [1, 10], blank_frozen=12, blank_total=96)
    np.testing.assert_allclose(separate[:3], pooled[:3], rtol=1e-8, atol=1e-8)


def volume_grid_log_likelihood(k, background, frozen, total, dilution, volumes,
                               blank_frozen, blank_total, blank_volumes):
    """Evaluate the physical-volume model directly for an independent grid."""
    k, background = np.broadcast_arrays(np.asarray(k), np.asarray(background))
    result = np.zeros(k.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        for x, n, d, volume in zip(frozen, total, dilution, volumes):
            hazard = volume / 1000 * (k / d + background)
            if x:
                result += x * np.log1p(-np.exp(-hazard))
            result -= (n - x) * hazard
        for x, n, volume in zip(blank_frozen, blank_total, blank_volumes):
            hazard = volume / 1000 * background
            if x:
                result += x * np.log1p(-np.exp(-hazard))
            result -= (n - x) * hazard
    return result


def test_unequal_sample_and_blank_volumes_agree_with_direct_likelihood_grid():
    result = fit(
        [16, 3], [32, 40], [1, 10], volume=[50, 20],
        blank_frozen=[2, 1], blank_total=[32, 20], blank_volume_uL=[50, 20],
    )
    k_grid = np.linspace(0, 30, 601)
    background_grid = np.linspace(0, 10, 801)
    likelihood = volume_grid_log_likelihood(
        k_grid[:, None], background_grid[None, :], [16, 3], [32, 40], [1, 10],
        [50, 20], [2, 1], [32, 20], [50, 20],
    )
    index, _ = np.unravel_index(np.argmax(likelihood), likelihood.shape)
    assert result[0] == pytest.approx(k_grid[index], abs=0.05)
    assert result[3]
    fine_background = np.linspace(0, 30, 60001)
    likelihood_at_limits = [
        np.max(volume_grid_log_likelihood(
            k, fine_background, [16, 3], [32, 40], [1, 10],
            [50, 20], [2, 1], [32, 20], [50, 20],
        ))
        for k in (result[0], result[0] - result[1], result[0] + result[2])
    ]
    for bound in likelihood_at_limits[1:]:
        assert likelihood_at_limits[0] - bound == pytest.approx(1.920729410347062, abs=2e-6)


def test_unequal_volumes_and_multiple_blanks_are_invariant_to_row_order():
    original = fit(
        [16, 3], [32, 40], [1, 10], volume=[50, 20],
        blank_frozen=[2, 1], blank_total=[32, 20], blank_volume_uL=[50, 20],
    )
    reordered = fit(
        [3, 16], [40, 32], [10, 1], volume=[20, 50],
        blank_frozen=[1, 2], blank_total=[20, 32], blank_volume_uL=[20, 50],
    )
    np.testing.assert_allclose(original[:3], reordered[:3], rtol=1e-8, atol=1e-8)


def test_vector_sample_volumes_require_explicit_blank_volumes():
    with pytest.raises(ValueError, match="explicit blank_volume_uL"):
        fit([16, 3], [32, 40], [1, 10], volume=[50, 20])


@pytest.mark.parametrize("blank_options", [
    {"blank_frozen": [], "blank_total": [], "blank_volume_uL": []},
    {"blank_frozen": [1, 2], "blank_total": [32, 32], "blank_volume_uL": [50, 0]},
    {"blank_frozen": [[1, 2]], "blank_total": [[32, 32]], "blank_volume_uL": [[50, 50]]},
])
def test_invalid_blank_arrays_are_rejected(blank_options):
    with pytest.raises(ValueError):
        fit(16, 32, **blank_options)


def test_saturated_blank_does_not_preclude_finite_multi_dilution_inference():
    result = fit([32, 1], [32, 32], [1, 100], blank_frozen=32)
    assert result[0] > 0 and np.isfinite(result[:3]).all() and result[3]


def test_large_dilution_zero_count_upper_bound_has_a_finite_analytical_bracket():
    result = fit(0, 32, dilution=1e100, blank_frozen=0)
    expected = 1.920729410347062 * 1e100 / (32 * 0.05)
    assert result[0] == result[1] == 0
    assert result[2] == pytest.approx(expected, rel=1e-8)
    assert result[3]


@pytest.mark.parametrize("dilution", [1, 1e100])
def test_equal_sample_and_blank_concentrations_are_exactly_at_zero_boundary(dilution):
    # These fractions imply the same concentration despite different volumes.
    # A positive numerical remainder could otherwise trigger a false decrease
    # when a later temperature is also at the zero boundary.
    result = fit(
        8, 16, dilution=dilution, volume=50,
        blank_frozen=12, blank_total=16, blank_volume_uL=100,
    )
    assert result[0] == result[1] == 0
    assert result[2] > 0 and result[3]


def test_large_dilution_positive_signal_is_not_classified_as_zero():
    ordinary = fit(16, 32, dilution=1, blank_frozen=4)
    large = fit(16, 32, dilution=1e100, blank_frozen=4)
    assert large[0] > 0 and large[3]
    np.testing.assert_allclose(large[:3], np.array(ordinary[:3]) * 1e100, rtol=1e-8)
