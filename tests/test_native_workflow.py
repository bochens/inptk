"""Physical count identities and native-point behavior across complete workflows."""

import json
from dataclasses import replace

import pandas as pd
import pytest
from analysis_checks import (
    fit_estimates,
    input_spectra,
    intervals,
    quantity_for_check,
    retained,
)

import inptk


def experiment(*, two_runs=False, cycles=("01", "01")):
    rows, metadata = [], []
    series = [
        ("a", "R1", cycles[0], "S", 1, 50, 32, [-5, -6, -6, -5.8, -7], [0, 4, 4, 6, 8]),
        ("wa", "R1", cycles[0], "WA", 1, 50, 20, [-5, -6, -6, -5.8, -7], [0, 0, 0, 0, 1]),
    ]
    if two_runs:
        series.extend(
            [
                ("b", "R2", cycles[1], "S", 10, 50, 32, [-5, -6.2, -7], [0, 4, 8]),
                ("wb", "R2", cycles[1], "WB", 1, 25, 20, [-5, -6.2, -7], [0, 1, 2]),
            ]
        )
    for name, run, cycle, sample, dilution, volume, total, temperatures, frozen in series:
        metadata.append(
            {
                "measurement_id": name,
                "sample_id": sample,
                "run_id": run,
                "dilution": dilution,
                "droplet_volume_uL": volume,
            }
        )
        for order, (temperature, count) in enumerate(zip(temperatures, frozen, strict=True)):
            rows.append(
                {
                    "measurement_id": name,
                    "cycle_id": cycle,
                    "temperature_C": temperature,
                    "time_s": order,
                    "n_total": total,
                    "n_frozen": count,
                }
            )
    return inptk.read_counts(
        pd.DataFrame(rows),
        metadata=metadata,
        water_blank_map={"a": ["wa"], **({"b": ["wb"]} if two_runs else {})},
    )


@pytest.mark.parametrize("method", ["mle", "average"])
def test_original_observations_are_primary_and_identical_states_do_not_gain_precision(method):
    source = experiment()
    result = inptk.analyze_concentration(source, method=method)
    pd.testing.assert_frame_equal(
        result.frozen_fraction.to_dataframe().drop(columns="fraction_frozen"),
        source.counts.to_dataframe(),
    )
    assert not hasattr(next(iter(result.curves.values())), "resampled")
    combined = fit_estimates(result).to_dataframe()
    expected = [-5, -5.8, -6, -7] if method == "mle" else [-5, -6, -6, -5.8, -7]
    assert combined.temperature_C.tolist() == expected
    assert combined.point_id.is_unique
    assert "run_id" not in combined and "cycle_id" not in combined
    assert combined.alignment.eq("native").all()
    for column in ("concentration", "lower_error", "upper_error"):
        assert combined[column].iloc[1] == pytest.approx(combined[column].iloc[2], rel=1e-7)
    sources = combined.source_observations.map(json.loads)
    assert all(len(items) == 2 for items in sources)
    assert sources.iloc[1][0]["observation_id"] != sources.iloc[2][0]["observation_id"]
    assert [
        {k: item[k] for k in ("measurement_id", "run_id", "cycle_id")}
        for item in result.curves["S/R1/01"].sources
    ] == [{"measurement_id": "a", "run_id": "R1", "cycle_id": "01"}]


def test_joint_workflow_keeps_cooling_intervals_ending_at_a_repeated_temperature():
    source = experiment()
    result = inptk.analyze_concentration(source, differential=True)
    curve = result.curves["S/R1/01"]
    direct = inptk.differential_spectrum(
        inptk.frozen_fraction(source), experiment=source
    ).to_dataframe()
    actual = curve.differential.to_dataframe()
    assert actual.temperature_bin_right_C.tolist() == [-5.8]
    assert actual.temperature_bin_left_C.tolist() == [-7]
    temperatures = curve.cumulative.to_dataframe().temperature_C
    direct = direct.loc[direct.temperature_bin_left_C.isin(temperatures)
                        & direct.temperature_bin_right_C.isin(temperatures)]
    pd.testing.assert_frame_equal(actual, direct.reset_index(drop=True))


@pytest.mark.parametrize("method", ["mle", "average"])
def test_explicit_cross_run_group_keeps_own_blank_and_cycle_provenance(method):
    source = experiment(two_runs=True, cycles=("01", "02"))
    groups = {
        "both": {
            "inputs": [
                {"measurement_id": "a", "cycle_id": "01"},
                {"measurement_id": "b", "cycle_id": "02"},
            ]
        }
    }
    result = inptk.analyze_concentration(source, method=method, curves=groups)
    frame = fit_estimates(result).to_dataframe()
    assert set(frame.curve_id) == {"both"}
    assert frame.alignment.eq("latest").all()
    assert frame.temperature_C.tolist() == sorted({-5, -6, -5.8, -7, -6.2}, reverse=True)
    a_sources = {("a", "R1", "01"), ("wa", "R1", "01")}
    b_sources = {("b", "R2", "02"), ("wb", "R2", "02")}
    for row in frame.itertuples():
        expected = a_sources | b_sources
        if method == "average":
            expected = (a_sources if -7 <= row.temperature_C <= -5.8 else set())
            if -7 <= row.temperature_C <= -6.2:
                expected |= b_sources
        items = json.loads(row.source_observations)
        assert {(item["measurement_id"], item["run_id"], item["cycle_id"])
                for item in items} == expected
    separate = fit_estimates(inptk.analyze_concentration(source)).to_dataframe()
    assert set(separate.curve_id) == {"S/R1/01", "S/R2/02"}
    assert set(input_spectra(result).to_dataframe().run_id) == {"R1", "R2"}


def test_native_cutoff_follows_observation_order_through_temperature_wiggles():
    curve = inptk.CurveSpectrumTable(
        pd.DataFrame(
            {
                "sample_id": "S",
                "curve_id": "g",
                "point_id": ["a", "b", "c", "d"],
                "point_order": [0, 1, 2, 3],
                "temperature_C": [-5, -6, -5.8, -7],
                "concentration": [1, 2, 3, 4],
                "lower_error": 0.1,
                "upper_error": 0.2,
                "unit": "INP_per_mL_suspension",
                "basis": "suspension",
            }
        )
    )
    final = inptk.finalize_spectrum(curve)
    assert final.to_dataframe().point_id.tolist() == ["a", "b", "c", "d"]
    assert final.to_dataframe().concentration.tolist() == [1, 2, 3, 4]
    assert final.to_dataframe().segment_id.tolist() == ["0"] * 4


def test_grid_is_selected_before_estimation_and_has_no_second_output():
    source = experiment()
    result = inptk.analyze_concentration(source, temperature_step_C=0.5)
    assert result.to_dataframe().temperature_C.tolist() == [-6, -6.5, -7]
    assert fit_estimates(result).to_dataframe().temperature_C.tolist() == [
        -5, -5.5, -6, -6.5, -7
    ]
    assert "temperature_step_C" in result.settings
    assert not hasattr(next(iter(result.curves.values())), "resampled")
    pd.testing.assert_frame_equal(result.counts.to_dataframe(), source.counts.to_dataframe())


def test_native_full_and_stepwise_paths_agree():
    source = experiment()
    result = inptk.analyze_concentration(source, temperature_ranges_C={"a": {"max_C": -5.8}})
    fractions = inptk.frozen_fraction(source)
    per = inptk.cumulative_spectrum(fractions, experiment=source)
    combined = inptk.estimate_concentration(
        fractions, experiment=source, temperature_ranges_C={"a": {"max_C": -5.8}}
    )
    pd.testing.assert_frame_equal(input_spectra(result).to_dataframe(), per.to_dataframe())
    pd.testing.assert_frame_equal(fit_estimates(result).to_dataframe(), combined.to_dataframe())
    pd.testing.assert_frame_equal(
        retained(result).to_dataframe(), inptk.finalize_spectrum(combined).to_dataframe()
    )


def test_initial_grid_results_round_trip(tmp_path):
    source = experiment(two_runs=True)
    groups = {
        "combined": {
            "inputs": [
                {"measurement_id": "a", "cycle_id": "01"},
                {"measurement_id": "b", "cycle_id": "01"},
            ]
        }
    }
    result = inptk.analyze_concentration(source, curves=groups, temperature_step_C=0.5)
    result.save(tmp_path / "saved.inptk")
    restored = inptk.load(tmp_path / "saved.inptk")
    assert restored.settings["curves"] == result.settings["curves"]
    for name in (
        "frozen_fraction",
        "per_dilution",
        "combined",
        "final_candidates",
        "final",
    ):
        pd.testing.assert_frame_equal(
            quantity_for_check(result, name).to_dataframe(),
            quantity_for_check(restored, name).to_dataframe(),
        )


@pytest.mark.parametrize("method", ["mle", "average"])
def test_explicit_group_does_not_fit_unrequested_run_with_incomplete_blank_coverage(method):
    original = experiment(two_runs=True)
    frame = original.counts.to_dataframe()
    frame = frame.loc[~(frame.measurement_id.eq("wb") & frame.temperature_C.lt(-5))]
    source = replace(original, counts=inptk.CountsTable(frame))
    groups = {"requested": {"inputs": [{"measurement_id": "a", "cycle_id": "01"}]}}
    result = inptk.analyze_concentration(source, curves=groups, method=method, differential=True)
    expected = inptk.estimate_concentration(
        inptk.frozen_fraction(source), experiment=source, curves=groups, method=method
    )
    pd.testing.assert_frame_equal(fit_estimates(result).to_dataframe(), expected.to_dataframe())
    assert set(input_spectra(result).to_dataframe().measurement_id) == {"a"}
    assert set(intervals(result).to_dataframe().measurement_id) == {"a"}
    assert result.experiment is source
    pd.testing.assert_frame_equal(
        result.experiment.counts.to_dataframe(), frame.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        result.frozen_fraction.to_dataframe().drop(columns="fraction_frozen"),
        source.counts.to_dataframe(),
    )
    selection = next(
        event
        for event in input_spectra(result).history
        if event["operation"] == "select_curve_inputs"
    )
    assert selection["members"] == [{"measurement_id": "a", "run_id": "R1", "cycle_id": "01"}]
    assert selection["water_blank_context"] == [
        {"measurement_id": "wa", "run_id": "R1", "cycle_id": "01"}
    ]
    with pytest.raises(ValueError, match="no blank extrapolation"):
        inptk.analyze_concentration(source, method=method)


def test_explicit_group_restricts_individual_and_differential_fits_to_selected_cycle():
    original = experiment()
    first = original.counts.to_dataframe()
    second = first.assign(cycle_id="02")
    second.loc[second.measurement_id.eq("a"), "n_frozen"] += 4
    source = replace(
        original, counts=inptk.CountsTable(pd.concat([first, second], ignore_index=True))
    )
    groups = {"cycle two": {"inputs": [{"measurement_id": "a", "cycle_id": "02"}]}}
    result = inptk.analyze_concentration(source, curves=groups, differential=True)
    assert set(input_spectra(result).to_dataframe().cycle_id) == {"02"}
    assert set(intervals(result).to_dataframe().cycle_id) == {"02"}
    assert set(result.frozen_fraction.to_dataframe().cycle_id) == {"01", "02"}
    assert set(result.experiment.counts.to_dataframe().cycle_id) == {"01", "02"}
    single_cycle = inptk.frozen_fraction(source).select(cycle_id="02")
    expected = inptk.cumulative_spectrum(single_cycle, experiment=source)
    pd.testing.assert_frame_equal(input_spectra(result).to_dataframe(), expected.to_dataframe())
