"""Full and explicit useful spans must select identical calculation states."""

import json

import numpy as np
import pandas as pd
import pytest

import inptk
from analysis_checks import all_points
from inptk.cli import main


def experiment(blank=False):
    rows, metadata = [], []
    streams = [("A", [0, 1, 5, 5], 1, 10, 50), ("B", [0, 1, 3, 8], 10, 10, 50)]
    if blank:
        streams.append(("water", [0, 0, 1, 2], 1, 20, 100))
    for name, frozen, dilution, total, volume in streams:
        metadata.append({"measurement_id": name, "sample_id": "G", "run_id": "R",
                         "dilution": dilution, "droplet_volume_uL": volume})
        rows.extend({"measurement_id": name, "run_id": "R", "cycle_id": "1", "time_s": i,
                     "temperature_C": temperature, "n_frozen": count, "n_total": total}
                    for i, (temperature, count) in enumerate(zip([-5.5, -6.2, -7.2, -8.2], frozen)))
    return inptk.read_counts(rows, metadata=metadata,
                            water_blank_map={"A": ["water"], "B": ["water"]} if blank else {})


CURVES = {"G": {"inputs": ["A", "B"], "cycle": "1"}}
GRID = {"temperature_step_C": .5, "temperature_start_C": 0, "temperature_end_C": -10}
VALUES = ["temperature_C", "concentration", "lower_error", "upper_error",
          "contributing_measurement_ids", "source_observations"]


@pytest.mark.parametrize("method", ["mle", "average"])
@pytest.mark.parametrize("rule", ["latest", "max", "window"])
@pytest.mark.parametrize("blank", [False, True])
def test_explicit_full_span_matches_full_values_uncertainty_and_sources(method, rule, blank):
    source = experiment(blank)
    original = source.counts.to_dataframe()
    settings = dict(GRID, method=method, curves=CURVES, temperature_method=rule)
    if rule == "window":
        settings["temperature_window_C"] = 1
    full = inptk.analyze_concentration(source, **settings)
    span = full.settings["resolved_temperature_ranges_C"]["G"]["B"]["full_range_C"]
    assert span == {"min_C": -8, "max_C": -6.5}
    selected = inptk.analyze_concentration(source, temperature_ranges_C={"B": span}, **settings)
    pd.testing.assert_frame_equal(full.to_dataframe()[VALUES], selected.to_dataframe()[VALUES])
    assert selected.to_dataframe().temperature_C.tolist() == [-6.5, -7, -7.5, -8]
    assert selected.settings["resolved_temperature_ranges_C"]["G"]["B"]["selected_range_C"] == span
    pd.testing.assert_frame_equal(original, source.counts.to_dataframe())
    if method == "mle":
        before = next(s for s in full.history if s["operation"] == "estimate_concentration")
        after = next(s for s in selected.history if s["operation"] == "estimate_concentration")
        assert before["joint_curve_fits"] == after["joint_curve_fits"]


@pytest.mark.parametrize("method", ["mle", "average"])
def test_warm_cut_preserves_cold_states_and_does_not_require_an_event_inside_range(method):
    source = experiment()
    full = inptk.analyze_concentration(source, curves=CURVES, method=method, **GRID)
    selected = inptk.analyze_concentration(
        source, curves=CURVES, method=method,
        temperature_ranges_C={"B": {"min_C": -8, "max_C": -7.5}}, **GRID,
    )
    baseline = all_points(full).to_dataframe()
    actual = all_points(selected).to_dataframe()
    for temperature in (-7.5, -8):
        before = baseline.loc[baseline.temperature_C.eq(temperature), "source_observations"].item()
        after = actual.loc[actual.temperature_C.eq(temperature), "source_observations"].item()
        assert before == after
        b = next(s for s in json.loads(after) if s["measurement_id"] == "B")
        assert b["observed_temperature_C"] == -7.2
        assert b["n_frozen"] == 3 and b["n_total"] == 10
        assert b["fraction_frozen"] == .3
    assert -8 in selected.to_dataframe().temperature_C.tolist()


@pytest.mark.parametrize("method", ["mle", "average"])
def test_native_mode_explicit_full_span_is_equivalent(method):
    source = experiment()
    options = {"temperature_step_C": None, "method": method, "curves": CURVES}
    full = inptk.analyze_concentration(source, **options)
    selected = inptk.analyze_concentration(
        source, temperature_ranges_C={"B": {"min_C": -8.2, "max_C": -6.2}}, **options,
    )
    pd.testing.assert_frame_equal(full.to_dataframe()[VALUES], selected.to_dataframe()[VALUES])


def test_cli_full_and_explicit_grid_span_match_saved_and_stepwise_results(tmp_path, capsys):
    source = experiment()
    source.save(tmp_path / "input")
    args = [str(tmp_path / "input"), "--format", "saved", "--method", "mle",
            "--curves", json.dumps(CURVES), "--temperature-step", "0.5",
            "--temperature-start-C", "0", "--temperature-end-C", "-10"]
    for label, ranges in [("full", {}), ("selected", {"B": {"min_C": -8, "max_C": -6.5}})]:
        assert main(["analyze", *args, "--temperature-ranges", json.dumps(ranges),
                     "--out", str(tmp_path / label), "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["settings"]["resolved_temperature_ranges_C"]["G"]["B"]["full_range_C"] == {
            "min_C": -8, "max_C": -6.5}
    full, selected = (inptk.load(tmp_path / label) for label in ("full", "selected"))
    pd.testing.assert_frame_equal(full.to_dataframe()[VALUES], selected.to_dataframe()[VALUES])
    step = inptk.estimate_concentration(
        inptk.frozen_fraction(source), experiment=source, method="mle", curves=CURVES,
        temperature_ranges_C={"B": {"min_C": -8, "max_C": -6.5}}, **GRID,
    )
    pd.testing.assert_frame_equal(
        inptk.finalize_spectrum(step).to_dataframe()[VALUES], selected.to_dataframe()[VALUES]
    )


def test_cut_can_exclude_invalid_history_without_hiding_invalid_retained_counts():
    data = experiment()
    raw = data.counts.to_dataframe()
    raw.loc[raw.measurement_id.eq("B") & raw.temperature_C.eq(-6.2), "n_total"] = 12
    source = inptk.Experiment(inptk.CountsTable(raw), data.samples, data.measurements,
                              water_blank_map={})
    with pytest.raises(ValueError, match="fixed set of wells"):
        inptk.analyze_concentration(source, curves=CURVES, **GRID)
    valid = inptk.analyze_concentration(
        source, curves=CURVES, temperature_ranges_C={"B": {"min_C": -8, "max_C": -7.5}}, **GRID,
    )
    assert np.isfinite(valid.to_dataframe().concentration).all()


def test_average_suggestions_return_grid_limits_and_aligned_threshold_counts():
    source = experiment()
    proposal = inptk.suggest_temperature_ranges(source, curves=CURVES, **GRID)
    report = proposal.observations.to_dataframe()
    for limits in proposal.temperature_ranges_C.values():
        assert limits["min_C"] * 2 == int(limits["min_C"] * 2)
        assert limits["max_C"] * 2 == int(limits["max_C"] * 2)
    assert proposal.settings["range_domain"] == "calculation grid"
    assert report.query("measurement_id == 'B' and temperature_C == -7.5").n_frozen.item() == 3
    result = inptk.estimate_concentration(
        inptk.frozen_fraction(source), experiment=source, curves=CURVES, method="average",
        temperature_ranges_C=proposal.temperature_ranges_C, **GRID,
    ).to_dataframe()
    values = result.dropna(subset=["concentration"]).concentration
    assert values.ge(0).all() and values.diff().dropna().ge(0).all()
