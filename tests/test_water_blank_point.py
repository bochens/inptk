"""Native point fitting keeps run backgrounds and physical observations distinct."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from inptk._engine.water_blank_math import average_concentration, fit_concentration
from inptk.experiment import Experiment, MeasurementMetadata, SampleMetadata
from inptk.tables import CountsTable
from inptk.water_blank import analysis_experiment, estimate_point, sample_rows

DROP = 1.920729410347062


def inputs():
    specifications = [
        ("M1", "S", "R1", 1, 50, 15, 20, -7.1),
        ("M2", "S", "R1", 2, 100, 35, 40, -7.1),
        ("M3", "S", "R2", 1, 50, 35, 40, -7.0),
        ("B1", "water1", "R1", 1, 50, 5, 10, -6.9),
        ("B2", "water2", "R2", 1, 50, 24, 32, -7.0),
    ]
    metadata, counts = {}, []
    for name, sample, run, dilution, volume, frozen, total, temperature in specifications:
        metadata[name] = MeasurementMetadata(name, sample, dilution, volume, run)
        counts.append({
            "measurement_id": name, "sample_id": sample, "run_id": run,
            "cycle_id": "01" if run == "R1" else "2",
            "temperature_C": temperature, "n_frozen": frozen, "n_total": total,
            "observation_id": "source-row-007" if run == "R1" else "source-row-12",
        })
    frame = pd.DataFrame(counts)
    experiment = Experiment(
        counts=CountsTable(frame),
        measurements=metadata,
        samples={name: SampleMetadata(name) for name in ("S", "water1", "water2")},
        water_blank_map={"M1": ["B1"], "M2": ["B1"], "M3": ["B2"]},
    )
    return experiment, frame.iloc[:3].copy(), frame.iloc[3:].copy()


@pytest.mark.parametrize(
    "method,solver", [("mle", fit_concentration), ("average", average_concentration)]
)
def test_native_rows_use_each_runs_background_and_each_shared_blank_once(method, solver):
    experiment, samples, blanks = inputs()
    before_samples, before_blanks = samples.copy(deep=True), blanks.copy(deep=True)
    expected = solver(
        [15, 35, 35], [20, 40, 40], [1, 2, 1], [50, 100, 50],
        blank_frozen=[5, 24], blank_total=[10, 32], blank_volume_uL=[50, 50],
        sample_blank_group=["R1", "R1", "R2"], blank_group=["R1", "R2"],
        confidence_drop=DROP,
    )
    actual = estimate_point(samples, blanks, experiment, confidence_drop=DROP, method=method)
    np.testing.assert_array_equal(actual, expected)
    assert actual[0] == pytest.approx(np.log(2) / 0.05, rel=1e-8)
    pd.testing.assert_frame_equal(samples, before_samples)
    pd.testing.assert_frame_equal(blanks, before_blanks)


def test_selecting_one_run_requires_only_its_assigned_blanks():
    experiment, samples, blanks = inputs()
    actual = estimate_point(samples.iloc[:2], blanks.iloc[:1], experiment, confidence_drop=DROP)
    expected = fit_concentration(
        [15, 35], [20, 40], [1, 2], [50, 100],
        blank_frozen=5, blank_total=10, blank_volume_uL=50,
    )
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("role", ["sample", "blank"])
@pytest.mark.parametrize("change_cycle", [False, True])
def test_repeated_observations_and_cycles_cannot_multiply_droplets(role, change_cycle):
    experiment, samples, blanks = inputs()
    frame = samples if role == "sample" else blanks
    repeated = frame.iloc[:1].copy()
    if change_cycle:
        repeated["cycle_id"] = "02"
    repeated["observation_id"] = "another-source-row"
    frame = pd.concat([frame, repeated], ignore_index=True)
    with pytest.raises(ValueError, match="one row per physical measurement"):
        estimate_point(
            frame if role == "sample" else samples,
            frame if role == "blank" else blanks,
            experiment, confidence_drop=DROP,
        )


def test_two_selected_cycles_in_same_run_are_rejected_even_for_different_measurements():
    experiment, samples, blanks = inputs()
    samples.loc[samples.measurement_id.eq("M2"), "cycle_id"] = "02"
    with pytest.raises(ValueError, match="one selected cycle per run"):
        estimate_point(samples, blanks, experiment, confidence_drop=DROP)


@pytest.mark.parametrize("blank_selection", ["missing", "wrong_cycle", "extra"])
def test_selected_blanks_must_match_all_and_only_assigned_measurement_cycles(blank_selection):
    experiment, samples, blanks = inputs()
    if blank_selection == "missing":
        blanks = blanks.iloc[:1]
    elif blank_selection == "wrong_cycle":
        blanks.loc[blanks.measurement_id.eq("B1"), "cycle_id"] = "1"
    else:
        samples = samples.iloc[:2]
    with pytest.raises(ValueError, match="exactly match assigned measurement/cycle pairs"):
        estimate_point(samples, blanks, experiment, confidence_drop=DROP)


def test_mapped_experiment_cannot_silently_skip_blank_correction():
    experiment, samples, _ = inputs()
    with pytest.raises(ValueError, match="missing="):
        estimate_point(samples, pd.DataFrame(), experiment, confidence_drop=DROP)


def test_disabling_correction_keeps_samples_and_runs_but_fits_no_background():
    experiment, samples, _ = inputs()
    uncorrected = analysis_experiment(experiment, water_blank_correction=False)
    assert uncorrected.water_blank_map == {}
    pd.testing.assert_frame_equal(
        sample_rows(experiment.counts.to_dataframe(), experiment).reset_index(drop=True),
        uncorrected.counts.to_dataframe(),
    )
    actual = estimate_point(samples, pd.DataFrame(), uncorrected, confidence_drop=DROP)
    expected = fit_concentration([15, 35, 35], [20, 40, 40], [1, 2, 1], [50, 100, 50])
    np.testing.assert_array_equal(actual, expected)


def test_unassigned_blanks_are_not_silently_used():
    experiment, samples, blanks = inputs()
    experiment = replace(experiment, water_blank_map={})
    with pytest.raises(ValueError, match="require experiment.water_blank_map"):
        estimate_point(samples, blanks, experiment, confidence_drop=DROP)


@pytest.mark.parametrize("column,value,match", [
    ("measurement_id", "missing", "Unknown measurement"),
    ("run_id", "R3", "disagree with metadata"),
    ("sample_id", "wrong", "disagree with metadata"),
    ("cycle_id", "", "nonempty string"),
    ("temperature_C", np.nan, "temperatures must be finite"),
])
def test_invalid_observation_identity_is_rejected(column, value, match):
    experiment, samples, blanks = inputs()
    samples.loc[0, column] = value
    with pytest.raises(ValueError, match=match):
        estimate_point(samples, blanks, experiment, confidence_drop=DROP)


def test_blank_measurement_cannot_be_used_as_sample():
    experiment, _, blanks = inputs()
    with pytest.raises(ValueError, match="not an assigned sample measurement"):
        estimate_point(blanks.iloc[:1], blanks, experiment, confidence_drop=DROP)


def test_unknown_method_is_rejected_instead_of_defaulting_to_mle():
    experiment, samples, blanks = inputs()
    with pytest.raises(ValueError, match="method must be"):
        estimate_point(samples, blanks, experiment, confidence_drop=DROP, method="typo")
