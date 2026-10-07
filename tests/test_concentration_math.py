"""One count likelihood for single and overlapping independent measurements."""

import numpy as np
import pytest
from reference_concentration_math import binomial_poisson_mle_with_profile_errors

from inptk._engine.water_blank_math import fit_concentration


def likelihood(concentration, frozen, total, dilution, volumes):
    """Independent direct no-blank likelihood, in physical concentration units."""
    result = np.zeros_like(np.asarray(concentration, dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        for x, n, d, volume in zip(frozen, total, dilution, volumes):
            hazard = concentration * volume / 1000 / d
            if x:
                result += x * np.log(-np.expm1(-hazard))
            result -= (n - x) * hazard
    return result


@pytest.mark.parametrize("frozen,total,dilution,volume", [
    (1, 32, 1, 50), (16, 32, 13, 50), (9, 10, 100, 2),
])
def test_single_measurement_retains_ordinary_concentration_with_profile_limits(
    frozen, total, dilution, volume,
):
    result = fit_concentration(frozen, total, dilution, volume)
    ordinary = -np.log1p(-frozen / total) * dilution / (volume / 1000)
    assert result[0] == pytest.approx(ordinary, rel=1e-9)
    assert result[1] > 0 and result[2] > 0 and result[3]
    legacy_profile = binomial_poisson_mle_with_profile_errors(
        frozen, total, volume, dilution
    )
    np.testing.assert_allclose(result[:3], legacy_profile[:3], rtol=1e-8)


def test_equal_exposures_with_different_volumes_match_summed_independent_counts():
    # Both sets expose 0.05 mL of original sample per well, despite different
    # well volumes and dilution factors. They are separate physical droplets.
    separate = fit_concentration([12, 18], [32, 40], [1, 2], [50, 100])
    summed = fit_concentration(30, 72, 1, 50)
    np.testing.assert_allclose(separate[:3], summed[:3], rtol=1e-8)


def test_unequal_exposures_match_direct_likelihood_and_interval_crossings():
    frozen, total, dilution, volumes = [20, 3], [32, 40], [1, 10], [50, 20]
    estimate, lower, upper, finite = fit_concentration(frozen, total, dilution, volumes)
    grid = np.linspace(0, 40, 4001)
    values = likelihood(grid, frozen, total, dilution, volumes)
    assert estimate == pytest.approx(grid[np.argmax(values)], abs=0.01)
    at_limits = likelihood(
        np.array([estimate, estimate - lower, estimate + upper]),
        frozen, total, dilution, volumes,
    )
    np.testing.assert_allclose(at_limits[0] - at_limits[1:], 1.920729410347062, atol=1e-8)
    assert finite


def test_zero_counts_have_positive_upper_limit_using_all_actual_exposures():
    result = fit_concentration([0, 0], [32, 20], [1, 10], [50, 20])
    upper = 1.920729410347062 / (32 * 0.05 + 20 * 0.02 / 10)
    assert result[0] == result[1] == 0
    assert result[2] == pytest.approx(upper, rel=1e-9)
    assert result[3]


def test_fully_frozen_counts_are_unbounded_but_mixed_extremes_remain_informative():
    saturated = fit_concentration([32, 20], [32, 20], [1, 10], [50, 20])
    assert np.isinf(saturated[0]) and np.isnan(saturated[1:3]).all()
    assert not saturated[3]
    mixed = fit_concentration([32, 0], [32, 20], [1, 10], [50, 20])
    assert mixed[0] > 0 and np.isfinite(mixed[:3]).all() and mixed[3]


@pytest.mark.parametrize("frozen", [0, 16])
def test_extreme_dilution_scales_point_and_limits_without_losing_information(frozen):
    ordinary = fit_concentration(frozen, 32, 1, 50)
    large = fit_concentration(frozen, 32, 1e100, 50)
    np.testing.assert_allclose(large[:3], np.array(ordinary[:3]) * 1e100, rtol=1e-8)
    assert large[3]


def test_row_order_and_uniform_volume_scaling_preserve_physical_fit():
    original = fit_concentration([20, 3], [32, 40], [1, 10], [50, 20])
    reordered = fit_concentration([3, 20], [40, 32], [10, 1], [20, 50])
    doubled_volumes = fit_concentration([20, 3], [32, 40], [1, 10], [100, 40])
    np.testing.assert_allclose(original[:3], reordered[:3], rtol=1e-10)
    np.testing.assert_allclose(original[:3], np.array(doubled_volumes[:3]) * 2, rtol=1e-10)


@pytest.mark.parametrize("options", [
    {"blank_frozen": 0}, {"blank_total": 32}, {"blank_volume_uL": 50},
])
def test_partial_blank_information_is_rejected(options):
    with pytest.raises(ValueError, match="blank"):
        fit_concentration(16, 32, 1, 50, **options)


@pytest.mark.parametrize("frozen", [1, 8, 16])
def test_identical_concentration_with_added_equal_exposure_has_exactly_equal_point(frozen):
    single = fit_concentration(frozen, 32, 1, 50)
    overlap = fit_concentration([frozen, frozen], [32, 32], [1, 2], [50, 100])
    expected = -np.log1p(-frozen / 32) / 0.05
    assert single[0] == overlap[0] == expected
    assert overlap[1] < single[1] and overlap[2] < single[2]
