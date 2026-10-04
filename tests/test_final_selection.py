"""Final cumulative selection removes points without altering observations or fits."""

import inspect

import numpy as np
import pandas as pd
import pytest
from analysis_checks import all_points, fit_estimates, retained

import inptk


def spectrum(values, *, sample="S", cycle="1", run="R", measurement=None):
    data = pd.DataFrame({
        "run_id": run, "sample_id": sample, "cycle_id": cycle,
        "temperature_C": -5 - np.arange(len(values)),
        "point_order": np.arange(len(values)),
        "point_id": [f"point:{index}" for index in range(len(values))],
        "concentration": values, "lower_error": np.arange(len(values)) + 0.1,
        "upper_error": np.arange(len(values)) + 0.2,
        "unit": "INP_per_mL_suspension", "basis": "suspension",
    })
    if measurement is not None:
        data["measurement_id"] = measurement
    return inptk.CumulativeSpectrumTable(data)


@pytest.mark.parametrize(
    "policy,temperatures",
    [("stop_at_decrease", [-5, -6]), ("skip_decreases", [-5, -6, -8, -9])],
)
def test_policies_distinguish_stopping_from_later_recovery(policy, temperatures):
    source = spectrum([0, 8, 4, 8, 12])
    original = source.to_dataframe()
    final = inptk.finalize_spectrum(source, decrease_policy=policy)
    frame = final.to_dataframe()
    assert frame.temperature_C.tolist() == temperatures
    expected = original[original.temperature_C.isin(temperatures)].reset_index(drop=True)
    pd.testing.assert_frame_equal(frame[list(original.columns)], expected)
    assert frame.used_in_final.all()
    assert frame.final_selection_status.eq("kept").all()
    pd.testing.assert_frame_equal(source.to_dataframe(), original)
    event = final.history[-1]
    assert event["decrease_policy"] == policy
    first_drop = event["groups"][0]["excluded"][0]
    assert first_drop["temperature_C"] == -7
    assert first_drop["reference_point_id"] == "point:1"
    assert first_drop["reference_concentration"] == 8


def test_default_stops_at_first_decrease():
    final = inptk.finalize_spectrum(spectrum([0, 8, 4, 10]))
    assert final.to_dataframe().concentration.tolist() == [0, 8]


@pytest.mark.parametrize("policy", ["stop_at_decrease", "skip_decreases"])
def test_selection_uses_explicit_observation_order_independently_of_frame_row_order(policy):
    source = spectrum([0, 8, 4, 10, 12]).to_dataframe()
    expected = inptk.finalize_spectrum(
        inptk.CumulativeSpectrumTable(source), decrease_policy=policy
    ).to_dataframe().sort_values("temperature_C").reset_index(drop=True)
    shuffled = source.sample(frac=1, random_state=11)
    result = inptk.finalize_spectrum(
        inptk.CumulativeSpectrumTable(shuffled), decrease_policy=policy
    ).to_dataframe().sort_values("temperature_C").reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected)


def test_sample_run_cycle_and_measurement_are_separate_curves():
    frames = []
    identities = [
        ("S", "1", "R", "M1"), ("S", "2", "R", "M1"),
        ("S", "1", "R2", "M1"), ("S2", "1", "R", "M1"),
        ("S", "1", "R", "M2"),
    ]
    for number, (sample, cycle, run, measurement) in enumerate(identities):
        frames.append(spectrum(
            [number * 100, number * 100 + 8, number * 100 + 4, number * 100 + 10],
            sample=sample, cycle=cycle, run=run, measurement=measurement,
        ).to_dataframe())
    result = inptk.finalize_spectrum(
        inptk.CumulativeSpectrumTable(pd.concat(frames).sample(frac=1, random_state=9))
    ).to_dataframe()
    groups = result.groupby(["sample_id", "run_id", "cycle_id", "measurement_id"])
    assert len(groups) == 5
    assert groups.size().eq(2).all()
    assert result.temperature_C.isin([-5, -6]).all()


def test_nonfinite_points_never_set_the_baseline_and_equal_values_remain():
    final = inptk.finalize_spectrum(spectrum([np.nan, -2, -2, np.inf, -1, 0, np.nan]))
    assert final.to_dataframe().concentration.tolist() == [-2, -2, -1, 0]
    excluded = final.history[-1]["groups"][0]["excluded"]
    assert [row["reason"] for row in excluded] == ["nonfinite"] * 3


def test_all_nonfinite_points_produce_an_empty_table_and_visible_warning():
    final = inptk.finalize_spectrum(spectrum([np.inf, np.nan, -np.inf]))
    assert len(final) == 0
    assert any("no finite concentration points" in warning for warning in final.warnings)
    assert final.history[-1]["groups"][0]["excluded_count"] == 3


def test_unrecognized_policy_is_rejected():
    with pytest.raises(ValueError, match="decrease_policy"):
        inptk.finalize_spectrum(spectrum([0, 1]), decrease_policy="smooth")


@pytest.mark.parametrize("policy", ["stop_at_decrease", "skip_decreases"])
def test_workflow_selects_after_blank_subtraction_and_unit_conversion(policy):
    source = inptk.read_counts(
        pd.DataFrame({
            "measurement_id": "M", "run_id": "R", "cycle_id": "1",
            "temperature_C": [-5, -6, -7, -8], "n_total": 32,
            "n_frozen": [0, 8, 16, 24],
        }),
        metadata=[{
            "measurement_id": "M", "sample_id": "S", "run_id": "R",
            "dilution": 1, "droplet_volume_uL": 50, "sample_type": "air",
            "air_volume_L": 100, "suspension_volume_mL": 10, "filter_fraction_used": 1,
        }],
    )
    blank = spectrum([0., 1., 12., 12.], sample="blank")
    result = inptk.analyze_concentration(
        source, output_basis="sampled_air",
        blank_by_curve={"S/R/1": blank}, decrease_policy=policy,
    )
    raw = fit_estimates(result).to_dataframe()
    assert np.all(np.diff(raw.concentration) > 0)
    corrected = inptk.subtract_blanks(fit_estimates(result), {"S/R/1": blank})
    converted = inptk.convert_concentration(corrected, source.samples, basis="sampled_air")
    expected = inptk.finalize_spectrum(converted, decrease_policy=policy)
    pd.testing.assert_frame_equal(retained(result).to_dataframe(), expected.to_dataframe())
    candidates = all_points(result).to_dataframe()
    original = converted.to_dataframe()
    pd.testing.assert_frame_equal(candidates[list(original.columns)], original)
    assert candidates.final_selection_status.iloc[2] == "decrease"
    assert candidates.final_selection_status.iloc[3] == (
        "after_decrease" if policy == "stop_at_decrease" else "kept"
    )
    assert result.settings["decrease_policy"] == policy
    assert retained(result).history[-1]["operation"] == "finalize_spectrum"


def test_public_interface_has_no_forced_monotone_fitting_setting():
    for function in (inptk.estimate_concentration, inptk.analyze_concentration):
        assert "enforce_monotone" not in inspect.signature(function).parameters


@pytest.mark.parametrize("policy", ["stop_at_decrease", "skip_decreases"])
def test_raw_blank_contributor_change_keeps_equal_concentration_with_optimizer_roundoff(policy):
    rows = []
    for measurement, temperatures, count in (
        ("A", [-5, -6], 8), ("B", [-6], 11), ("blank", [-5, -6], 4),
    ):
        rows.extend({
            "measurement_id": measurement, "cycle_id": "1",
            "temperature_C": temperature, "n_frozen": count, "n_total": 32,
        } for temperature in temperatures)
    source = inptk.read_counts(
        pd.DataFrame(rows),
        metadata=[
            {"measurement_id": "A", "sample_id": "S", "run_id": "R",
             "dilution": 1, "droplet_volume_uL": 50},
            {"measurement_id": "B", "sample_id": "S", "run_id": "R",
             "dilution": 2, "droplet_volume_uL": 100},
            {"measurement_id": "blank", "sample_id": "water", "run_id": "R",
             "dilution": 1, "droplet_volume_uL": 50},
        ],
        water_blank_map={"A": ["blank"], "B": ["blank"]},
    )
    result = inptk.analyze_concentration(source, decrease_policy=policy)
    combined = fit_estimates(result).to_dataframe()
    final = retained(result).to_dataframe()
    assert len(final) == 2
    expected_concentration = (-np.log(0.75) + np.log(0.875)) / 0.05
    np.testing.assert_allclose(final.concentration, expected_concentration, rtol=1e-9)
    for column in ("concentration", "lower_error", "upper_error"):
        np.testing.assert_array_equal(final[column], combined[column])
    event = retained(result).history[-1]
    assert event["numerical_relative_tolerance"] == 1e-9
    assert event["numerical_absolute_tolerance"] == 0


@pytest.mark.parametrize("policy", ["stop_at_decrease", "skip_decreases"])
@pytest.mark.parametrize("values", [[1e-20, 0.0], [1e-20, 0.99e-20], [1.0, 0.999]])
def test_numerical_equality_does_not_hide_real_decreases_or_tiny_signals(policy, values):
    final = inptk.finalize_spectrum(spectrum(values), decrease_policy=policy)
    assert final.to_dataframe().concentration.tolist() == [values[0]]
    assert final.history[-1]["groups"][0]["excluded"][0]["reason"] == "decrease"



def test_filter_blank_uncertainty_labels_only_corrected_rows_and_retains_source_methods():
    sample = spectrum([4.0, np.nan, 8.0]).to_dataframe()
    sample["uncertainty_method"] = "joint_sample_water_blank_profile_likelihood"
    other = spectrum([2.0, 3.0, 4.0], sample="other").to_dataframe()
    other["uncertainty_method"] = "binomial_Poisson_profile_likelihood"
    source = inptk.CumulativeSpectrumTable(pd.concat([sample, other], ignore_index=True))
    blank = spectrum([1.0, 1.0, 1.0], sample="blank").select(temperature_C=[-5, -7])
    result = inptk.subtract_blanks(source, {"S": blank})
    frame = result.to_dataframe()
    corrected = (frame.sample_id == "S") & frame.concentration.notna()
    assert frame.loc[corrected, "uncertainty_method"].eq(
        "approximate_independent_filter_blank_error_propagation"
    ).all()
    np.testing.assert_array_equal(
        frame.loc[~corrected, "uncertainty_method"],
        source.to_dataframe().loc[~corrected, "uncertainty_method"],
    )
    prior = result.history[-1]["source_uncertainty_methods"]
    assert [(row["sample_id"], row["temperature_C"]) for row in prior] == [("S", -5), ("S", -7)]
    assert all(row["uncertainty_method"] == "joint_sample_water_blank_profile_likelihood"
               for row in prior)
    pd.testing.assert_frame_equal(source.to_dataframe().iloc[:3].reset_index(drop=True), sample)
