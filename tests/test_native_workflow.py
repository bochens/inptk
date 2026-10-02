"""Physical count identities and native-point behavior across complete workflows."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

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
    assert result.resampled is None
    combined = result.combined.to_dataframe()
    assert combined.temperature_C.tolist() == [-5, -6, -6, -5.8, -7]
    assert combined.point_id.is_unique
    assert "run_id" not in combined and "cycle_id" not in combined
    assert combined.alignment.eq("native").all()
    for column in ("concentration", "lower_error", "upper_error"):
        assert combined[column].iloc[1] == combined[column].iloc[2]
    sources = combined.source_observations.map(json.loads)
    assert all(len(items) == 2 for items in sources)
    assert sources.iloc[1][0]["observation_id"] != sources.iloc[2][0]["observation_id"]
    assert result.settings["combination_groups"]["S/R1/01"]["members"] == [
        {"measurement_id": "a", "run_id": "R1", "cycle_id": "01"}
    ]


@pytest.mark.parametrize("method", ["mle", "average"])
def test_explicit_cross_run_group_keeps_own_blank_and_cycle_provenance(method):
    source = experiment(two_runs=True, cycles=("01", "02"))
    groups = {
        "both": [
            {"measurement_id": "a", "cycle_id": "01"},
            {"measurement_id": "b", "cycle_id": "02"},
        ]
    }
    result = inptk.analyze_concentration(source, method=method, combination_groups=groups)
    frame = result.combined.to_dataframe()
    assert set(frame.group_id) == {"both"}
    assert frame.alignment.eq("latest").all()
    assert frame.temperature_C.tolist() == sorted({-5, -6, -5.8, -7, -6.2}, reverse=True)
    for items in frame.source_observations.map(json.loads):
        assert {(item["measurement_id"], item["run_id"], item["cycle_id"]) for item in items} == {
            ("a", "R1", "01"),
            ("wa", "R1", "01"),
            ("b", "R2", "02"),
            ("wb", "R2", "02"),
        }
    separate = inptk.analyze_concentration(source).combined.to_dataframe()
    assert set(separate.group_id) == {"S/R1/01", "S/R2/02"}
    assert set(result.per_dilution.to_dataframe().run_id) == {"R1", "R2"}


def test_native_cutoff_follows_observation_order_through_temperature_wiggles():
    curve = inptk.CombinedSpectrumTable(
        pd.DataFrame(
            {
                "sample_id": "S",
                "group_id": "g",
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


def test_skipped_points_split_resampling_segments_and_native_values_remain():
    source = experiment()
    native = inptk.analyze_concentration(source)
    sampled = inptk.analyze_concentration(source, output_step_C=0.5, output_method="sample")
    pd.testing.assert_frame_equal(native.final.to_dataframe(), sampled.final.to_dataframe())
    assert sampled.resampled is not None
    assert sampled.resampled.history[-1]["operation"] == "resample_spectrum"
    assert np.isfinite(sampled.resampled.to_dataframe().concentration).all()


def test_native_full_and_stepwise_paths_agree():
    source = experiment()
    result = inptk.analyze_concentration(source, temperature_ranges_C={"a": {"max_C": -5.8}})
    fractions = inptk.frozen_fraction(source)
    per = inptk.cumulative_spectrum(
        fractions, experiment=source, temperature_ranges_C={"a": {"max_C": -5.8}}
    )
    combined = inptk.combine_dilutions(
        fractions, experiment=source, temperature_ranges_C={"a": {"max_C": -5.8}}
    )
    pd.testing.assert_frame_equal(result.per_dilution.to_dataframe(), per.to_dataframe())
    pd.testing.assert_frame_equal(result.combined.to_dataframe(), combined.to_dataframe())
    pd.testing.assert_frame_equal(
        result.final.to_dataframe(), inptk.finalize_spectrum(combined).to_dataframe()
    )


def test_native_and_resampled_results_round_trip(tmp_path):
    source = experiment(two_runs=True)
    groups = {
        "combined": [
            {"measurement_id": "a", "cycle_id": "01"},
            {"measurement_id": "b", "cycle_id": "01"},
        ]
    }
    result = inptk.analyze_concentration(source, combination_groups=groups, output_step_C=0.5)
    result.save(tmp_path / "saved.inptk")
    restored = inptk.load(tmp_path / "saved.inptk")
    assert restored.settings["combination_groups"] == result.settings["combination_groups"]
    for name in (
        "frozen_fraction",
        "per_dilution",
        "combined",
        "final_candidates",
        "final",
        "resampled",
    ):
        pd.testing.assert_frame_equal(
            getattr(result, name).to_dataframe(), getattr(restored, name).to_dataframe()
        )


@pytest.mark.parametrize("method", ["mle", "average"])
def test_explicit_group_does_not_fit_unrequested_run_with_incomplete_blank_coverage(method):
    original = experiment(two_runs=True)
    frame = original.counts.to_dataframe()
    frame = frame.loc[~(frame.measurement_id.eq("wb") & frame.temperature_C.lt(-5))]
    source = replace(original, counts=inptk.CountsTable(frame))
    groups = {"requested": [{"measurement_id": "a", "cycle_id": "01"}]}
    result = inptk.analyze_concentration(
        source, combination_groups=groups, method=method, differential=True
    )
    expected = inptk.combine_dilutions(
        inptk.frozen_fraction(source), experiment=source, combination_groups=groups, method=method
    )
    pd.testing.assert_frame_equal(result.combined.to_dataframe(), expected.to_dataframe())
    assert set(result.per_dilution.to_dataframe().measurement_id) == {"a"}
    assert set(result.differential.to_dataframe().measurement_id) == {"a"}
    assert result.experiment is source
    pd.testing.assert_frame_equal(
        result.experiment.counts.to_dataframe(), frame.reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        result.frozen_fraction.to_dataframe().drop(columns="fraction_frozen"),
        source.counts.to_dataframe(),
    )
    selection = next(
        event for event in result.per_dilution.history
        if event["operation"] == "select_combination_members"
    )
    assert selection["members"] == [
        {"measurement_id": "a", "run_id": "R1", "cycle_id": "01"}
    ]
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
    groups = {"cycle two": [{"measurement_id": "a", "cycle_id": "02"}]}
    result = inptk.analyze_concentration(source, combination_groups=groups, differential=True)
    assert set(result.per_dilution.to_dataframe().cycle_id) == {"02"}
    assert set(result.differential.to_dataframe().cycle_id) == {"02"}
    assert set(result.frozen_fraction.to_dataframe().cycle_id) == {"01", "02"}
    assert set(result.experiment.counts.to_dataframe().cycle_id) == {"01", "02"}
    single_cycle = inptk.frozen_fraction(source).select(cycle_id="02")
    expected = inptk.cumulative_spectrum(single_cycle, experiment=source)
    pd.testing.assert_frame_equal(result.per_dilution.to_dataframe(), expected.to_dataframe())
