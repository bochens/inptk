"""Report freezing intervals while retaining the observations that constrain fits."""

import json

import numpy as np
import pandas as pd
import pytest
from analysis_checks import all_points

import inptk
from inptk.cli import main
from inptk.reporting import freezing_intervals


def experiment(series, *, water_blank_map=None):
    records, metadata = [], []
    for name, cycle, temperatures, frozen in series:
        metadata.append({"measurement_id": name, "sample_id": "sample",
                         "dilution": 1, "droplet_volume_uL": 50})
        records.extend({"measurement_id": name, "cycle_id": cycle,
                        "temperature_C": temperature, "time_s": i,
                        "n_total": 10, "n_frozen": count}
                       for i, (temperature, count) in enumerate(zip(temperatures, frozen)))
    return inptk.read_counts(
        records, metadata=pd.DataFrame(metadata).drop_duplicates(), water_blank_map=water_blank_map
    )


@pytest.mark.parametrize("method", ["average", "mle"])
def test_only_existing_grid_points_between_events_are_reported(method):
    source = experiment([("A", "1", [0, -5, -6.1, -7, -8.1, -10, -12],
                          [0, 0, 2, 2, 5, 5, 5])])
    original = source.counts.to_dataframe()
    result = inptk.analyze_concentration(
        source, method=method, temperature_step_C=.5,
        temperature_start_C=0, temperature_end_C=-12,
    )
    frame = result.to_dataframe()
    assert frame.temperature_C.tolist() == [-6.5, -7, -7.5, -8]
    assert set(frame.reporting_status) == {"within_freezing_interval"}
    candidates = all_points(result).to_dataframe()
    assert candidates.temperature_C.tolist() == list(np.arange(0, -12.5, -.5))
    assert set(candidates.loc[candidates.temperature_C.gt(-6.1),
                              "final_selection_status"]) == {"before_first_freeze"}
    assert set(candidates.loc[candidates.temperature_C.lt(-8.1),
                              "final_selection_status"]) == {"after_last_freeze"}
    pd.testing.assert_frame_equal(result.counts.to_dataframe(), original)
    estimated = inptk.estimate_concentration(
        inptk.frozen_fraction(source), experiment=source, method=method,
        temperature_step_C=.5, temperature_start_C=0, temperature_end_C=-12,
    )
    pd.testing.assert_frame_equal(candidates[list(estimated.columns)], estimated.to_dataframe())
    if method == "mle":
        fit = next(step for step in result.history if step["operation"] == "estimate_concentration")
        assert fit["joint_curve_fits"] == estimated.history[-1]["joint_curve_fits"]


@pytest.mark.parametrize("method", ["average", "mle"])
def test_blank_events_do_not_extend_interval_and_corrected_zeros_survive(method):
    temperatures = [-5, -6, -7, -8, -9, -10]
    source = experiment([("A", "1", temperatures, [0, 0, 2, 2, 4, 4]),
                         ("blank", "1", temperatures, [0, 1, 3, 3, 5, 6])],
                        water_blank_map={"A": ["blank"]})
    result = inptk.analyze_concentration(source, method=method)
    frame = result.to_dataframe()
    assert frame.temperature_C.tolist() == [-7, -8, -9]
    np.testing.assert_allclose(frame.concentration, 0, atol=1e-6)
    assert frame.upper_error.ge(0).all()


@pytest.mark.parametrize("method", ["average", "mle"])
def test_no_sample_freezing_events_produces_an_empty_report(method):
    source = experiment([("A", "1", [-5, -6, -7], [0, 0, 0])])
    result = inptk.analyze_concentration(source, method=method)
    assert result.to_dataframe().empty
    assert len(result.counts) == 3
    assert result.curves["sample/1/1"].excluded.to_dataframe().final_selection_status.eq(
        "no_freezing_events").all()


def test_one_off_grid_event_does_not_create_an_endpoint_row():
    source = experiment([("A", "1", [-5, -6.1, -8], [0, 3, 3])])
    result = inptk.analyze_concentration(source, temperature_step_C=.5,
                                         temperature_start_C=-5, temperature_end_C=-8)
    assert result.to_dataframe().empty
    assert result.settings["reporting_intervals_C"] == {
        "sample/1/1": {"min_C": -6.1, "max_C": -6.1}}


def test_individual_cycles_and_combined_groups_have_their_own_limits():
    temperatures = [-4, -5, -6, -7, -8, -9, -10, -11, -12]
    source = experiment([("A", "1", temperatures, [0, 1, 1, 3, 3, 3, 3, 3, 3]),
                         ("B", "1", temperatures, [0, 0, 0, 0, 2, 2, 2, 4, 4]),
                         ("A", "2", temperatures, [0, 0, 1, 1, 1, 3, 3, 3, 3])])
    curves = {"A": {"inputs": ["A"], "cycle": "1"},
              "B": {"inputs": ["B"], "cycle": "1"},
              "combined": {"inputs": ["A", "B"], "cycle": "1"},
              "cycle2": {"inputs": ["A"], "cycle": "2"}}
    result = inptk.analyze_concentration(source, curves=curves)
    expected = {"A": (-7, -5), "B": (-11, -8),
                "combined": (-11, -5), "cycle2": (-9, -6)}
    for name, (cold, warm) in expected.items():
        frame = result.curves[name].cumulative.to_dataframe()
        assert (frame.temperature_C.min(), frame.temperature_C.max()) == (cold, warm)


def test_ranges_and_count_recovery_do_not_create_new_freezing_events():
    frame = pd.DataFrame({"measurement_id": "A", "run_id": "R", "cycle_id": "1",
                          "time_s": range(6), "temperature_C": [-5, -6, -7, -8, -9, -10],
                          "n_frozen": [0, 2, 1, 2, 3, 3]})
    groups = {"curve": {"members": [{"measurement_id": "A", "run_id": "R", "cycle_id": "1"}]}}
    assert freezing_intervals(frame, groups, {}) == {"curve": {"min_C": -9, "max_C": -6}}
    assert freezing_intervals(frame, groups, {"A": {"min_C": -8, "max_C": None}}) == {
        "curve": {"min_C": -6, "max_C": -6}}


def test_step_finalization_and_csv_export_share_reporting_limits(tmp_path, capsys):
    source = experiment([("A", "1", [-5, -6, -7, -8, -9], [0, 2, 2, 4, 4])])
    estimated = inptk.estimate_concentration(inptk.frozen_fraction(source), experiment=source)
    final = inptk.finalize_spectrum(estimated)
    expected = inptk.analyze_concentration(source).to_dataframe()
    pd.testing.assert_frame_equal(final.to_dataframe(), expected)
    step = inptk.ProcessingResult({"cumulative": estimated}, experiment=source)
    step.save(tmp_path / "estimated")
    path = tmp_path / "export.csv"
    assert main(["export-csv", str(tmp_path / "estimated"), "--out", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    assert pd.read_csv(path).temperature_C.tolist() == [-6, -7, -8]
    assert main(["finalize", str(tmp_path / "estimated"), "--out", str(tmp_path / "final")]) == 0
    restored = inptk.load(tmp_path / "final")
    pd.testing.assert_frame_equal(restored.tables["cumulative"].to_dataframe(), expected)
