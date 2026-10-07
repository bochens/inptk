"""Each MLE input must obey its own freezing interval before entering the fit."""

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import inptk
import inptk.curve_fit as curve_fit
from analysis_checks import all_points


CURVES = {"sample": {"inputs": ["A", "B", "never"], "cycle": "1"}}
VALUES = ["temperature_C", "concentration", "lower_error", "upper_error"]


def experiment(blank=False):
    records, metadata = [], []
    streams = [("A", [0, 1, 2, 4, 4, 4], 1),
               ("B", [0, 0, 0, 1, 2, 3], 4),
               ("never", [0, 0, 0, 0, 0, 0], 10)]
    if blank:
        streams.append(("water", [0, 0, 0, 1, 1, 2], 1))
    for name, counts, dilution in streams:
        metadata.append({"measurement_id": name, "sample_id": "sample", "run_id": "R",
                         "droplet_volume_uL": 50, "dilution": dilution})
        records.extend({"measurement_id": name, "cycle_id": "1", "run_id": "R",
                        "time_s": i, "temperature_C": -5-i, "n_total": 20, "n_frozen": n}
                       for i, n in enumerate(counts))
    blank_map = {name: ["water"] for name in ("A", "B", "never")} if blank else {}
    return inptk.read_counts(records, metadata=metadata, water_blank_map=blank_map)


def options(rule):
    if rule == "native":
        return {"temperature_step_C": None}
    return {"temperature_step_C": .5, "temperature_start_C": 0,
            "temperature_end_C": -12, "temperature_method": rule,
            **({"temperature_window_C": 1} if rule == "window" else {})}


def fit_details(result):
    return next(step for step in result.history if step["operation"] == "estimate_concentration")[
        "joint_curve_fits"]["sample"]


@pytest.mark.parametrize("rule", ["native", "latest", "max", "window"])
@pytest.mark.parametrize("blank", [False, True])
def test_full_and_explicit_full_ranges_restrict_actual_likelihood_streams(monkeypatch, rule, blank):
    source = experiment(blank)
    original = source.counts.to_dataframe()
    model_type = curve_fit.CurveLikelihood
    captured = []

    def checked_model(streams, *args, **kwargs):
        samples = [stream for stream in streams if stream.sample_exposure > 0]
        assert len(samples) == 2  # The never-frozen input is absent from the fit itself.
        for stream in samples:
            cold, warm = (-8, -6) if stream.sample_exposure == .05 else (-10, -8)
            assert np.all((stream.temperatures >= cold) & (stream.temperatures <= warm))
            assert np.all(stream.frozen > 0)
        if blank:
            controls = [stream for stream in streams if stream.sample_exposure == 0]
            assert len(controls) == 1 and np.any(controls[0].frozen == 0)
        captured.append(streams)
        return model_type(streams, *args, **kwargs)

    monkeypatch.setattr(curve_fit, "CurveLikelihood", checked_model)
    full = inptk.analyze_concentration(source, curves=CURVES, method="mle", **options(rule))
    selected = inptk.analyze_concentration(
        source, curves=CURVES, method="mle", **options(rule),
        temperature_ranges_C={"A": {"min_C": -8, "max_C": -6},
                              "B": {"min_C": -10, "max_C": -8}},
    )
    assert len(captured) == 2
    pd.testing.assert_frame_equal(full.to_dataframe(), selected.to_dataframe())
    assert fit_details(full) == fit_details(selected)
    assert fit_details(full)["physical_droplets"] == (60 if blank else 40)
    for row in all_points(full).to_dataframe().itertuples():
        contributors = json.loads(row.contributing_measurement_ids)
        assert "never" not in contributors
        if row.temperature_C > -8:
            assert "B" not in contributors
        if row.temperature_C < -8:
            assert "A" not in contributors
    pd.testing.assert_frame_equal(source.counts.to_dataframe(), original)


@pytest.mark.parametrize("rule", ["native", "latest", "max", "window"])
def test_never_frozen_input_cannot_lower_an_active_input_or_shrink_its_uncertainty(rule):
    source = experiment()
    alone = inptk.analyze_concentration(
        source, curves={"sample": {"inputs": ["A"], "cycle": "1"}}, **options(rule))
    with_unfrozen = inptk.analyze_concentration(
        source, curves={"sample": {"inputs": ["A", "never"], "cycle": "1"}}, **options(rule))
    pd.testing.assert_frame_equal(alone.to_dataframe()[VALUES], with_unfrozen.to_dataframe()[VALUES])
    assert fit_details(with_unfrozen)["physical_droplets"] == 20
    assert [s["measurement_id"] for s in fit_details(with_unfrozen)["sources"]] == ["A"]


@pytest.mark.parametrize("rule", ["native", "latest", "max", "window"])
def test_manual_range_further_limits_fitted_states_and_keeps_individuals_independent(rule):
    source = experiment()
    curves = {**CURVES, "A": {"inputs": ["A"], "cycle": "1"}}
    full = inptk.analyze_concentration(source, curves=curves, **options(rule))
    restricted = inptk.analyze_concentration(
        source, curves=curves, **options(rule),
        temperature_ranges_C={"A": {"min_C": -7, "max_C": -6},
                              "B": {"min_C": -9, "max_C": -9}},
    )
    pd.testing.assert_frame_equal(full.curves["A"].cumulative.to_dataframe(),
                                  restricted.curves["A"].cumulative.to_dataframe())
    for stream in fit_details(restricted)["sources"]:
        allowed = (-7, -6) if stream["measurement_id"] == "A" else (-9, -9)
        assert all(allowed[0] <= t <= allowed[1] for t in stream["fit_temperatures_C"])


def test_invalid_pre_freeze_and_post_freeze_totals_do_not_enter_full_fit():
    original = experiment()
    raw = original.counts.to_dataframe()
    outside = raw.measurement_id.eq("A") & (raw.temperature_C.gt(-6) | raw.temperature_C.lt(-8))
    raw.loc[outside, "n_total"] = 25
    changed = inptk.Experiment(inptk.CountsTable(raw), original.samples, original.measurements)
    full = inptk.analyze_concentration(original, curves=CURVES, **options("latest"))
    actual = inptk.analyze_concentration(changed, curves=CURVES, **options("latest"))
    pd.testing.assert_frame_equal(full.to_dataframe()[VALUES], actual.to_dataframe()[VALUES])
    assert actual.counts.to_dataframe().loc[outside, "n_total"].eq(25).all()


def test_persistent_cli_full_and_stepwise_manual_full_fit_the_same_inputs(tmp_path):
    source = experiment()
    metadata = [{**vars(source.samples[item.sample_id]), **vars(item)}
                for item in source.measurements.values()]
    common = ["--format", "saved", "--method", "mle", "--curves", json.dumps(CURVES),
              "--temperature-step-C", "0.5", "--temperature-start-C", "0",
              "--temperature-end-C", "-12"]
    ranges = json.dumps({"A": {"min_C": -8, "max_C": -6},
                         "B": {"min_C": -10, "max_C": -8}})
    requests = [
        {"id": 1, "import": {"out": "@raw", "counts": source.counts.to_dataframe().to_dict("records"),
                              "metadata": metadata, "water_blank_map": {}}},
        {"id": 2, "args": ["analyze", "@raw", *common, "--out", "@full"]},
        {"id": 3, "args": ["fractions", "@raw", "--format", "saved", "--out", "@fractions"]},
        {"id": 4, "args": ["estimate", "@fractions", *common, "--temperature-ranges", ranges,
                            "--out", "@estimate"]},
        {"id": 5, "args": ["finalize", "@estimate", "--out", "@final"]},
        {"id": 6, "args": ["save", "@full", "--out", str(tmp_path / "full")]},
        {"id": 7, "args": ["save", "@final", "--out", str(tmp_path / "step")]},
    ]
    process = subprocess.run(
        [sys.executable, "-m", "inptk", "serve"],
        input="\n".join(map(json.dumps, requests)) + "\n", text=True, capture_output=True,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src")},
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    replies = [json.loads(line) for line in process.stdout.splitlines()]
    assert [r["id"] for r in replies] == list(range(1, 8))
    assert all(r["status"] == "ok" for r in replies), replies
    full, step = inptk.load(tmp_path / "full"), inptk.load(tmp_path / "step")
    expected = inptk.analyze_concentration(source, curves=CURVES, **options("latest"))
    pd.testing.assert_frame_equal(full.to_dataframe(), expected.to_dataframe())
    pd.testing.assert_frame_equal(full.to_dataframe(), step.tables["cumulative"].to_dataframe())
    assert fit_details(full)["physical_droplets"] == 40
    assert {s["measurement_id"] for s in fit_details(full)["sources"]} == {"A", "B"}
