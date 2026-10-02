import json

import numpy as np
import pandas as pd
import pytest

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
    result = inptk.analyze_concentration(
        source, step_C=1, method=method, temperature_ranges_C=ranges
    )
    combined = result.combined.to_dataframe().sort_values("temperature_C", ascending=False)
    assert combined.contributor_count.tolist() == [1, 2, 0]
    assert combined.selection_status.tolist() == ["single", "combined", "no_eligible_measurements"]
    assert combined.contributing_measurement_ids.map(json.loads).tolist() == [["A"], ["A", "B"], []]
    assert np.isnan(combined.concentration.iloc[-1])
    per = result.per_dilution.to_dataframe().set_index(["measurement_id", "temperature_C"])
    for column in ("concentration", "lower_error", "upper_error"):
        assert combined.iloc[0][column] == per.loc[("A", -5), column]
    excluded = result.per_dilution.to_dataframe().query(
        "selection_status == 'outside_temperature_range'"
    )
    assert excluded.concentration.isna().all()
    assert excluded.n_total.eq(32).all()
    assert len(result.frozen_fraction) == 6
    fractions = inptk.frozen_fraction(source, step_C=1)
    stepwise = inptk.combine_dilutions(
        fractions, experiment=source, method=method, temperature_ranges_C=ranges
    )
    pd.testing.assert_frame_equal(result.combined.to_dataframe(), stepwise.to_dataframe())


def test_average_is_equal_weight_concentration_mean_while_mle_uses_counts():
    source = observations([("A", 4, [1], 1, 50), ("B", 32, [20], 1, 50)], (-5,))
    mean = inptk.analyze_concentration(source, method="average").combined.to_dataframe().iloc[0]
    mle = inptk.analyze_concentration(source, method="mle").combined.to_dataframe().iloc[0]
    expected_mean = (-np.log(3 / 4) - np.log(12 / 32)) / (2 * 0.05)
    expected_mle = -np.log(15 / 36) / 0.05
    assert mean.concentration == pytest.approx(expected_mean)
    assert mle.concentration == pytest.approx(expected_mle)
    assert mean.concentration != pytest.approx(mle.concentration)


def test_saturated_measurements_are_not_silently_discarded():
    source = observations([("A", 4, [4], 1, 50), ("B", 32, [8], 10, 50)], (-5,))
    mle = inptk.analyze_concentration(source, method="mle").combined.to_dataframe().iloc[0]
    mean = inptk.analyze_concentration(source, method="average").combined.to_dataframe().iloc[0]
    assert np.isfinite(mle.concentration)
    assert np.isinf(mean.concentration)
    assert mle.contributor_count == mean.contributor_count == 2
    assert mean.qc_flag & 1


def test_equal_exposure_contributor_change_does_not_create_a_false_decrease():
    source = observations([("A", 32, [1, 1], 1, 50), ("B", 32, [1, 1], 2, 100)], (-5, -6))
    for method in ("mle", "average"):
        result = inptk.analyze_concentration(
            source, method=method, step_C=1, temperature_ranges_C={"B": {"max_C": -6}}
        )
        assert len(result.final) == 2
        assert result.final.to_dataframe().concentration.nunique() == 1


def test_filter_blank_does_not_require_coverage_at_excluded_temperatures():
    source = observations([("A", 32, [4, 8, 16], 1, 50)])
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
        step_C=1,
        temperature_ranges_C={"A": {"min_C": -6, "max_C": -6}},
        blank_by_sample={"S": blank},
    )
    assert result.final.to_dataframe().temperature_C.tolist() == [-6]
    assert result.final_candidates.to_dataframe().concentration.isna().sum() == 2
