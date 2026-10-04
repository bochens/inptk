"""Each independent run keeps its own background in a common-concentration fit."""

import numpy as np
import pytest

from inptk._engine.math import PROFILE_LIKELIHOOD_DROP_95
from inptk._engine.water_blank_math import (
    average_concentration,
    fit_concentration,
    joint_water_blank_mle,
)


@pytest.mark.parametrize("fit", [fit_concentration, average_concentration])
@pytest.mark.parametrize("frozen", [[0, 0], [12, 20], [32, 32]])
def test_one_explicit_group_is_exactly_the_existing_shared_background(fit, frozen):
    options = {"blank_frozen": [1, 2], "blank_total": [10, 20], "blank_volume_uL": [50, 20]}
    expected = fit(frozen, 32, [1, 13], [50, 20], **options)
    actual = fit(
        frozen, 32, [1, 13], [50, 20], **options,
        sample_blank_group=["run A", "run A"], blank_group=["run A", "run A"],
    )
    np.testing.assert_array_equal(actual, expected)


def test_two_backgrounds_recover_common_concentration_from_exact_binomial_fractions():
    result = fit_concentration(
        [15, 35], [20, 40], 1, 50,
        blank_frozen=[5, 24], blank_total=[10, 32],
        sample_blank_group=["A", "B"], blank_group=["A", "B"],
    )
    # Background probabilities are 1/2 and 3/4, and sample probabilities
    # 3/4 and 7/8. Both imply K=log(2)/0.05 mL without sharing a background.
    assert result[0] == pytest.approx(np.log(2) / 0.05, rel=1e-8)
    assert result[1] > 0 and result[2] > 0 and result[3]


def test_multiple_blanks_unequal_counts_volumes_and_dilutions():
    result = fit_concentration(
        [15, 35, 14, 62], [20, 40, 16, 64], [1, 2, 1, 2], [50, 100, 50, 100],
        blank_frozen=[5, 9, 15], blank_total=[10, 12, 20], blank_volume_uL=[50, 100, 50],
        sample_blank_group=["A", "A", "B", "B"], blank_group=["A", "A", "B"],
    )
    assert result[0] == pytest.approx(np.log(2) / 0.05, rel=1e-8)
    assert result[3]


def test_estimate_and_profile_limits_match_independent_background_grid():
    frozen, total = np.array([12, 15]), np.array([20, 24])
    dilution, volume = np.array([1, 2]), np.array([0.05, 0.1])
    blank_x, blank_n = np.array([2, 9]), np.array([10, 20])
    blank_volume = np.array([0.05, 0.05])
    result = fit_concentration(
        frozen, total, dilution, volume * 1000,
        blank_frozen=blank_x, blank_total=blank_n, blank_volume_uL=blank_volume * 1000,
        sample_blank_group=["A", "B"], blank_group=["A", "B"],
    )
    background = np.linspace(1e-7, 40, 20001)

    def grid_profile(concentration):
        likelihood = 0.0
        for x, n, d, v, bx, bn, bv in zip(
            frozen, total, dilution, volume, blank_x, blank_n, blank_volume
        ):
            sample_hazard = v * (concentration / d + background)
            blank_hazard = bv * background
            values = (
                x * np.log(-np.expm1(-sample_hazard)) - (n - x) * sample_hazard
                + bx * np.log(-np.expm1(-blank_hazard)) - (bn - bx) * blank_hazard
            )
            likelihood += float(np.max(values))
        return likelihood

    point, lower, upper, finite = result
    maximum = grid_profile(point)
    concentrations = np.linspace(max(0, point - 1), point + 1, 101)
    grid_best = concentrations[np.argmax([grid_profile(k) for k in concentrations])]
    assert point == pytest.approx(grid_best, abs=0.02)
    assert maximum - grid_profile(point + upper) == pytest.approx(
        PROFILE_LIKELIHOOD_DROP_95, abs=2e-6
    )
    if point > lower:
        assert maximum - grid_profile(point - lower) == pytest.approx(
            PROFILE_LIKELIHOOD_DROP_95, abs=2e-6
        )
    assert finite


def test_group_sample_and_blank_permutations_preserve_the_result():
    frozen, total, dilution, volume = (
        np.array(values) for values in ([10, 13, 19], [20, 30, 40], [1, 2, 4], [50, 20, 100])
    )
    blank_x, blank_n, blank_volume = (
        np.array(values) for values in ([1, 2, 6], [10, 12, 20], [50, 100, 50])
    )
    sample_group, blank_group = np.array(["A", "B", "A"]), np.array(["A", "A", "B"])
    expected = fit_concentration(
        frozen, total, dilution, volume,
        blank_frozen=blank_x, blank_total=blank_n, blank_volume_uL=blank_volume,
        sample_blank_group=sample_group, blank_group=blank_group,
    )
    sample_order, blank_order = [1, 2, 0], [2, 0, 1]
    actual = fit_concentration(
        frozen[sample_order], total[sample_order], dilution[sample_order], volume[sample_order],
        blank_frozen=blank_x[blank_order], blank_total=blank_n[blank_order],
        blank_volume_uL=blank_volume[blank_order],
        sample_blank_group=sample_group[sample_order], blank_group=blank_group[blank_order],
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-8)


def test_fully_saturated_group_does_not_change_identifiable_group_or_its_scale():
    expected = fit_concentration(12, 20, 1, 50, blank_frozen=2, blank_total=10)
    actual = fit_concentration(
        [12, 32], [20, 32], [1, 13], [50, 1e90],
        blank_frozen=[2, 32], blank_total=[10, 32], blank_volume_uL=[50, 1e90],
        sample_blank_group=["informative", "saturated"],
        blank_group=["informative", "saturated"],
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("fit", [fit_concentration, average_concentration])
def test_all_groups_saturated_are_unidentifiable(fit):
    result = fit(
        [10, 20], [10, 20], [1, 13], 50,
        blank_frozen=[4, 32], blank_total=[4, 32],
        sample_blank_group=["A", "B"], blank_group=["A", "B"],
    )
    assert np.isnan(result[:3]).all() and not result[3]


def test_all_samples_saturated_with_an_informative_blank_group_remain_unbounded():
    result = fit_concentration(
        [10, 20], [10, 20], [1, 13], 50,
        blank_frozen=[4, 8], blank_total=[4, 32],
        sample_blank_group=["A", "B"], blank_group=["A", "B"],
    )
    assert np.isinf(result[0]) and np.isnan(result[1:3]).all() and not result[3]


def test_zero_background_groups_and_all_zero_samples_keep_upper_uncertainty():
    result = fit_concentration(
        [0, 0], [10, 20], [1, 13], [50, 20],
        blank_frozen=[0, 0], blank_total=[4, 32], blank_volume_uL=[50, 50],
        sample_blank_group=["A", "B"], blank_group=["A", "B"],
    )
    expected_upper = PROFILE_LIKELIHOOD_DROP_95 / (10 * 0.05 + 20 * 0.02 / 13)
    assert result[0] == result[1] == 0 and result[3]
    assert result[2] == pytest.approx(expected_upper, rel=1e-8)


def test_saturated_blanks_with_partial_samples_are_fitted_at_sample_boundary():
    result = fit_concentration(
        [3, 8], [10, 20], [1, 13], 50,
        blank_frozen=[4, 32], blank_total=[4, 32],
        sample_blank_group=["A", "B"], blank_group=["A", "B"],
    )
    assert result[0] == result[1] == 0 and result[2] > 0 and result[3]


def test_large_dilutions_keep_positive_signal_across_groups():
    options = {
        "blank_frozen": [1, 4], "blank_total": [10, 20],
        "sample_blank_group": ["A", "B"], "blank_group": ["A", "B"],
    }
    expected = fit_concentration([12, 16], [20, 32], [1, 2], 50, **options)
    actual = fit_concentration([12, 16], [20, 32], [1e100, 2e100], 50, **options)
    np.testing.assert_allclose(np.array(actual[:3]) / 1e100, expected[:3], rtol=1e-8)
    assert actual[3]


def test_average_subtracts_only_its_own_groups_direct_blank_concentrations():
    frozen, total = np.array([12, 20, 13]), np.array([20, 32, 30])
    dilution, volume = np.array([1, 2, 13]), np.array([50, 100, 20])
    blank_x, blank_n = np.array([1, 2, 8]), np.array([10, 12, 32])
    blank_volume = np.array([50, 20, 50])
    sample_group, blank_group = ["A", "B", "A"], np.array(["A", "A", "B"])
    backgrounds = {}
    for group in ("A", "B"):
        mask = blank_group == group
        concentrations = -np.log1p(-blank_x[mask] / blank_n[mask]) / (blank_volume[mask] / 1000)
        backgrounds[group] = np.average(concentrations, weights=blank_n[mask] * blank_volume[mask])
    expected = np.mean(dilution * (
        -np.log1p(-frozen / total) / (volume / 1000)
        - np.array([backgrounds[group] for group in sample_group])
    ))
    result = average_concentration(
        frozen, total, dilution, volume,
        blank_frozen=blank_x, blank_total=blank_n, blank_volume_uL=blank_volume,
        sample_blank_group=sample_group, blank_group=blank_group,
    )
    assert result[0] == pytest.approx(expected, rel=1e-12)
    assert result[1] > 0 and result[2] > 0 and result[3]


def test_average_does_not_discard_unidentifiable_group():
    result = average_concentration(
        [12, 20], [20, 20], 1, 50,
        blank_frozen=[2, 32], blank_total=[10, 32],
        sample_blank_group=["A", "B"], blank_group=["A", "B"],
    )
    assert np.isnan(result[:3]).all() and not result[3]


def test_explicit_joint_wrapper_forwards_group_labels():
    options = {"sample_blank_group": ["A", "B"], "blank_group": ["A", "B"]}
    expected = fit_concentration(
        [12, 20], [20, 32], [1, 13], 50,
        blank_frozen=[2, 8], blank_total=[10, 32], **options,
    )
    actual = joint_water_blank_mle(
        [12, 20], [20, 32], [1, 13], 50, [2, 8], [10, 32], **options,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("fit", [fit_concentration, average_concentration])
@pytest.mark.parametrize("groups", [
    {"sample_blank_group": ["A", "B"]},
    {"blank_group": ["A", "B"]},
    {"sample_blank_group": ["A"], "blank_group": ["A", "B"]},
    {"sample_blank_group": "A", "blank_group": ["A", "A"]},
    {"sample_blank_group": ["A", "B"], "blank_group": [["A", "B"]]},
    {"sample_blank_group": ["A", 1], "blank_group": ["A", "B"]},
    {"sample_blank_group": ["A", ""], "blank_group": ["A", "B"]},
    {"sample_blank_group": ["A", "B"], "blank_group": ["A", "C"]},
    {"sample_blank_group": ["A", "A"], "blank_group": ["A", "B"]},
])
def test_invalid_group_assignments_are_rejected_before_fitting(fit, groups):
    with pytest.raises(ValueError):
        fit([12, 20], [20, 32], [1, 13], 50,
            blank_frozen=[2, 8], blank_total=[10, 32], **groups)


@pytest.mark.parametrize("fit", [fit_concentration, average_concentration])
def test_groups_without_blank_observations_are_rejected(fit):
    with pytest.raises(ValueError, match="require blank observations"):
        fit([12, 20], [20, 32], [1, 13], 50,
            sample_blank_group=["A", "B"], blank_group=[])


def test_saturated_samples_in_one_group_still_contribute_when_their_blanks_are_partial():
    partial = fit_concentration(8, 20, 1, 50, blank_frozen=1, blank_total=10)
    options = {
        "blank_frozen": [1, 1], "blank_total": [10, 10],
        "sample_blank_group": ["A", "B"], "blank_group": ["A", "B"],
    }
    joint = fit_concentration([8, 20], [20, 20], 1, 50, **options)
    arithmetic = average_concentration([8, 20], [20, 20], 1, 50, **options)
    assert joint[3] and joint[0] > partial[0]
    assert np.isinf(arithmetic[0]) and not arithmetic[3]
