"""Independent checks for direct frozen-fraction concentration averages."""

from statistics import NormalDist

import numpy as np
import pandas as pd
import pytest
from scipy.stats import binomtest

import inptk
from inptk._engine.water_blank_math import average_concentration


def rate(frozen, total, volume_uL):
    return -np.log1p(-frozen / total) / (volume_uL / 1000)


def rate_bounds(frozen, total, volume_uL, z=1.96):
    level = 2 * NormalDist().cdf(z) - 1
    limits = binomtest(frozen, total).proportion_ci(confidence_level=level, method="wilson")
    return rate(limits.low, 1, volume_uL), rate(limits.high, 1, volume_uL)


@pytest.mark.parametrize("frozen", [0, 1, 16, 30])
def test_single_measurement_uses_direct_fraction_and_transformed_wilson_limits(frozen):
    point, lower, upper, finite = average_concentration(frozen, 32, 13, 50)
    expected = 13 * rate(frozen, 32, 50)
    lo, hi = rate_bounds(frozen, 32, 50)
    assert point == pytest.approx(expected)
    assert lower == pytest.approx(expected - 13 * lo)
    assert upper == pytest.approx(13 * hi - expected)
    assert finite


def test_mean_uses_equal_measurement_weights_and_independent_sample_errors():
    frozen, total = [16, 4], [32, 10]
    dilution, volume = [1, 13], [50, 20]
    point, lower, upper, finite = average_concentration(frozen, total, dilution, volume)
    rates = [rate(x, n, v) for x, n, v in zip(frozen, total, volume)]
    limits = [rate_bounds(x, n, v) for x, n, v in zip(frozen, total, volume)]
    expected_lower = np.hypot(*[
        d * (value - bounds[0]) / 2
        for d, value, bounds in zip(dilution, rates, limits)
    ])
    expected_upper = np.hypot(*[
        d * (bounds[1] - value) / 2
        for d, value, bounds in zip(dilution, rates, limits)
    ])
    assert point == pytest.approx(np.mean(np.asarray(dilution) * rates))
    assert lower == pytest.approx(expected_lower)
    assert upper == pytest.approx(expected_upper)
    assert finite


def test_unequal_sample_and_blank_volumes_use_actual_liquid_volumes():
    result = average_concentration(
        [8, 16], [16, 32], [1, 3], [50, 100],
        blank_frozen=1, blank_total=10, blank_volume_uL=20,
    )
    expected = np.mean([
        rate(8, 16, 50) - rate(1, 10, 20),
        3 * (rate(16, 32, 100) - rate(1, 10, 20)),
    ])
    assert result[0] == pytest.approx(expected)
    assert result[1] > 0 and result[2] > 0 and result[3]


def test_blank_well_count_changes_precision_but_not_direct_correction():
    small = average_concentration([16, 24], [32, 32], [1, 2], 50,
                                  blank_frozen=1, blank_total=4)
    large = average_concentration([16, 24], [32, 32], [1, 2], 50,
                                  blank_frozen=8, blank_total=32)
    assert small[0] == pytest.approx(large[0])
    assert large[1] < small[1] and large[2] < small[2]
    assert small[3] and large[3]


def test_equal_volume_blanks_pool_independent_well_counts():
    options = {"n_frozen": 16, "n_total": 32, "dilution": 1, "well_volume_uL": 50}
    separate = average_concentration(**options, blank_frozen=[1, 7],
        blank_total=[4, 28], blank_volume_uL=[50, 50])
    pooled = average_concentration(**options, blank_frozen=8,
        blank_total=32, blank_volume_uL=50)
    np.testing.assert_allclose(separate[:3], pooled[:3], rtol=1e-12, atol=1e-12)
    assert separate[3] and pooled[3]


def test_different_blank_volumes_use_total_assayed_volume_weights():
    result = average_concentration(16, 32, 2, 50,
        blank_frozen=[1, 8], blank_total=[4, 32], blank_volume_uL=[50, 100])
    weights = np.array([4 * 50, 32 * 100], dtype=float)
    blank_rates = [rate(1, 4, 50), rate(8, 32, 100)]
    expected = 2 * (rate(16, 32, 50) - np.average(blank_rates, weights=weights))
    assert result[0] == pytest.approx(expected)
    assert result[3]


def test_shared_blank_uncertainty_is_counted_once_in_the_mean():
    single = average_concentration(16, 32, 1, 50, blank_frozen=1, blank_total=4)
    shared = average_concentration([16, 16], [32, 32], [1, 1], 50,
                                   blank_frozen=1, blank_total=4)
    independent = average_concentration([16, 16], [32, 32], [1, 1], 50,
        blank_frozen=[1, 1], blank_total=[4, 4], blank_volume_uL=50,
        sample_blank_group=["A", "B"], blank_group=["A", "B"])
    sample = rate(16, 32, 50)
    sample_lo, sample_hi = rate_bounds(16, 32, 50)
    blank = rate(1, 4, 50)
    blank_lo, blank_hi = rate_bounds(1, 4, 50)
    assert shared[0] == pytest.approx(single[0])
    assert shared[1] == pytest.approx(np.hypot(
        (sample - sample_lo) / np.sqrt(2), blank_hi - blank))
    assert shared[2] == pytest.approx(np.hypot(
        (sample_hi - sample) / np.sqrt(2), blank - blank_lo))
    assert independent[1] < shared[1] and independent[2] < shared[2]


def test_negative_correction_is_retained_without_likelihood_clipping():
    point, lower, upper, finite = average_concentration(
        2, 32, 1, 50, blank_frozen=8, blank_total=32)
    assert point == pytest.approx(rate(2, 32, 50) - rate(8, 32, 50))
    assert point < 0 and lower > 0 and upper > 0 and finite


def test_zero_sample_and_blank_keep_both_uncertainty_directions():
    no_blank = average_concentration(0, 32, 1, 50)
    with_blank = average_concentration(0, 32, 1, 50, blank_frozen=0, blank_total=32)
    assert no_blank[0] == no_blank[1] == 0
    assert no_blank[2] > 0 and no_blank[3]
    assert with_blank[0] == 0
    assert with_blank[1] > 0 and with_blank[2] > 0 and with_blank[3]


def test_saturation_is_flagged_without_discarding_other_measurements():
    sample = average_concentration([32, 4], [32, 32], [1, 13], 50)
    assert np.isinf(sample[0]) and np.isnan(sample[1:3]).all() and not sample[3]
    blank = average_concentration(4, 32, 1, 50,
        blank_frozen=[4, 32], blank_total=[16, 32], blank_volume_uL=[50, 100])
    assert not np.isfinite(blank[0]) and np.isnan(blank[1:3]).all() and not blank[3]
    both = average_concentration(32, 32, 1, 50, blank_frozen=32, blank_total=32)
    assert np.isnan(both[:3]).all() and not both[3]


@pytest.mark.parametrize("changes", [
    {"n_frozen": [16, 0.5]}, {"n_total": [32, 0]}, {"dilution": [1, 0]},
    {"well_volume_uL": [50, np.nan]}, {"n_frozen": [[16, 4]]},
    {"n_frozen": [], "n_total": [], "dilution": [], "well_volume_uL": []},
    {"z": 0}, {"z": -1}, {"z": np.nan}, {"z": np.inf}, {"z": 1e308},
    {"blank_frozen": 0}, {"blank_total": 32}, {"blank_volume_uL": 50},
])
def test_invalid_inputs_fail_without_discarding_bad_measurements(changes):
    arguments = {"n_frozen": [16, 4], "n_total": [32, 32],
                 "dilution": [1, 13], "well_volume_uL": 50}
    arguments.update(changes)
    with pytest.raises(ValueError):
        average_concentration(**arguments)


def test_vector_volumes_require_explicit_blank_volumes_even_when_equal():
    with pytest.raises(ValueError, match="explicit blank_volume_uL"):
        average_concentration([16, 4], [32, 32], [1, 13], [50, 50],
                              blank_frozen=0, blank_total=32)


def test_public_average_paths_never_call_the_likelihood_fitter(monkeypatch):
    import inptk._engine.water_blank_math as math

    def forbidden(*args, **kwargs):
        raise AssertionError("Average called the likelihood fitter")

    monkeypatch.setattr(math, "fit_concentration", forbidden)
    rows = []
    for name, frozen in (("A", [2, 8, 14]), ("B", [0, 2, 6]), ("W", [0, 1, 2])):
        rows.extend(
            {"measurement_id": name, "temperature_C": temperature,
             "n_frozen": count, "n_total": 16}
            for temperature, count in zip([-5, -6, -7], frozen)
        )
    metadata = [
        {"measurement_id": "A", "sample_id": "S", "dilution": 1, "droplet_volume_uL": 50},
        {"measurement_id": "B", "sample_id": "S", "dilution": 2, "droplet_volume_uL": 50},
        {"measurement_id": "W", "sample_id": "W", "dilution": 1, "droplet_volume_uL": 50},
    ]
    data = inptk.read_counts(pd.DataFrame(rows), metadata=metadata,
                             water_blank_map={"A": ["W"], "B": ["W"]})
    fractions = inptk.frozen_fraction(data)
    assert len(inptk.estimate_concentration(fractions, experiment=data, method="average"))
    assert len(inptk.cumulative_spectrum(fractions, experiment=data, method="average"))
    assert inptk.suggest_temperature_ranges(data).inputs
