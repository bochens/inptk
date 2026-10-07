import json

import numpy as np
import pandas as pd
import pytest
from analysis_checks import all_points, fit_estimates, input_spectra, retained

import inptk


def observations(specs, temperatures=(-5, -6, -7)):
    rows, metadata = [], []
    for name, total, frozen, dilution, volume in specs:
        metadata.append(
            {
                "measurement_id": name,
                "sample_id": "S",
                "run_id": "R",
                "dilution": dilution,
                "droplet_volume_uL": volume,
            }
        )
        for temperature, count in zip(temperatures, frozen, strict=True):
            rows.append(
                {
                    "measurement_id": name,
                    "cycle_id": "01",
                    "temperature_C": temperature,
                    "n_total": total,
                    "n_frozen": count,
                }
            )
    return inptk.read_counts(pd.DataFrame(rows), metadata=metadata)


@pytest.mark.parametrize("method", ["mle", "average"])
def test_one_workflow_handles_single_overlap_and_gaps_with_explicit_sources(method):
    source = observations(
        [
            ("A", 32, [4, 8, 16], 1, 50),
            ("B", 32, [1, 2, 4], 10, 50),
        ]
    )
    ranges = {"A": {"min_C": -6}, "B": {"min_C": -6, "max_C": -6}}
    result = inptk.analyze_concentration(source, method=method, temperature_ranges_C=ranges)
    combined = fit_estimates(result).to_dataframe().sort_values("temperature_C", ascending=False)
    assert combined.contributor_count.tolist() == [1, 2, 0]
    assert combined.selection_status.tolist() == ["single", "combined", "no_eligible_measurements"]
    assert combined.contributing_measurement_ids.map(json.loads).tolist() == [["A"], ["A", "B"], []]
    assert np.isnan(combined.concentration.iloc[-1])
    per = input_spectra(result).to_dataframe().set_index(["measurement_id", "temperature_C"])
    if method == "average":
        for column in ("concentration", "lower_error", "upper_error"):
            assert combined.iloc[0][column] == per.loc[("A", -5), column]
    else:
        # Later measurements also constrain the earlier part of a joint curve.
        assert combined.concentration.dropna().diff().dropna().ge(0).all()
    excluded = (
        input_spectra(result)
        .to_dataframe()
        .query("selection_status == 'outside_temperature_range'")
    )
    assert excluded.empty
    assert per.concentration.notna().all()
    assert per.n_total.eq(32).all()
    assert len(result.frozen_fraction) == 6
    fractions = inptk.frozen_fraction(source)
    stepwise = inptk.estimate_concentration(
        fractions, experiment=source, method=method, temperature_ranges_C=ranges
    )
    pd.testing.assert_frame_equal(fit_estimates(result).to_dataframe(), stepwise.to_dataframe())


def test_average_is_equal_weight_concentration_mean_while_mle_uses_counts():
    source = observations([("A", 4, [1], 1, 50), ("B", 32, [20], 1, 50)], (-5,))
    mean = (
        fit_estimates(inptk.analyze_concentration(source, method="average")).to_dataframe().iloc[0]
    )
    mle = fit_estimates(inptk.analyze_concentration(source, method="mle")).to_dataframe().iloc[0]
    expected_mean = (-np.log(3 / 4) - np.log(12 / 32)) / (2 * 0.05)
    expected_mle = -np.log(15 / 36) / 0.05
    assert mean.concentration == pytest.approx(expected_mean)
    assert mle.concentration == pytest.approx(expected_mle)
    assert mean.concentration != pytest.approx(mle.concentration)


def test_saturated_measurements_contribute_to_mle_but_not_average():
    source = observations([("A", 4, [4], 1, 50), ("B", 32, [8], 10, 50)], (-5,))
    mle = fit_estimates(inptk.analyze_concentration(source, method="mle")).to_dataframe().iloc[0]
    mean = (
        fit_estimates(inptk.analyze_concentration(source, method="average")).to_dataframe().iloc[0]
    )
    assert np.isfinite(mle.concentration)
    assert mle.contributor_count == 2
    assert mean.concentration == pytest.approx(-10 * np.log(24 / 32) / 0.05)
    assert mean.contributor_count == 1
    assert json.loads(mean.available_measurement_ids) == ["A", "B"]
    assert json.loads(mean.contributing_measurement_ids) == ["B"]
    assert mean.qc_flag == 0


def test_equal_exposure_contributor_change_does_not_create_a_false_decrease():
    source = observations([("A", 32, [1, 1], 1, 50), ("B", 32, [1, 1], 2, 100)], (-5, -6))
    for method in ("mle", "average"):
        result = inptk.analyze_concentration(
            source, method=method, temperature_ranges_C={"B": {"max_C": -6}}
        )
        # The unchanged count state still constrains estimation, but its
        # temperature is past the last observed freezing event.
        assert len(fit_estimates(result)) == 2
        assert fit_estimates(result).to_dataframe().concentration.nunique() == 1
        assert retained(result).to_dataframe().temperature_C.tolist() == [-5]


def test_filter_blank_does_not_require_coverage_at_excluded_temperatures():
    source = observations([("A", 32, [4, 8, 16], 1, 50),
                           ("B", 32, [1, 3, 6], 2, 50)])
    blank = inptk.CumulativeSpectrumTable(
        pd.DataFrame(
            {
                "sample_id": ["filter_blank"],
                "run_id": ["R"],
                "cycle_id": ["01"],
                "temperature_C": [-6],
                "concentration": [1.0],
                "lower_error": [0.1],
                "upper_error": [0.2],
                "unit": ["INP_per_mL_suspension"],
                "basis": ["suspension"],
            }
        )
    )
    result = inptk.analyze_concentration(
        source,
        temperature_ranges_C={"A": {"min_C": -6, "max_C": -6},
                              "B": {"min_C": -6, "max_C": -6}},
        blank_by_curve={"S/R/01": blank},
    )
    assert retained(result).to_dataframe().temperature_C.tolist() == [-6]
    assert all_points(result).to_dataframe().concentration.isna().sum() == 2
