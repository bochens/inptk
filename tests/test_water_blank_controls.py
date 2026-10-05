"""Shared controls, original onset, and observation-only blank limits."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import inptk
from inptk import cli
from inptk._engine.curve_likelihood import CurveLikelihood, FreezingSeries
from inptk._engine.water_blank_math import average_concentration
from inptk.cli_store import ResultStore


VALUES = ["concentration", "lower_error", "upper_error"]


def experiment(*, blanks=None, volumes=None, runs=("R",), cycles=("0",), temperatures=None):
    temperatures = temperatures or [-10, -11, -12, -13, -14, -15]
    blanks = {"W1": [0, 0, 1, 2, 3, 4], "W2": [0, 0, 0, 0, 1, 2]} if blanks is None else blanks
    volumes = volumes or {}
    rows, metadata, mapping = [], [], {}
    for run in runs:
        for name, counts in {"S": [1, 3, 6, 9, 12, 14], **blanks}.items():
            identity = name if len(runs) == 1 else run + name
            metadata.append({
                "measurement_id": identity, "sample_id": "sample" if name == "S" else identity,
                "run_id": run, "dilution": 1, "droplet_volume_uL": volumes.get(name, 50),
                "sample_type": "other" if name == "S" else "water blank",
            })
            if name == "S" and blanks:
                mapping[identity] = [w if len(runs) == 1 else run + w for w in blanks]
            for cycle in cycles:
                for index, (temperature, count) in enumerate(
                    zip(temperatures, counts, strict=True)
                ):
                    rows.append({
                        "measurement_id": identity, "cycle_id": cycle, "time_s": index,
                        "temperature_C": temperature, "n_frozen": count, "n_total": 20,
                    })
    return inptk.read_counts(pd.DataFrame(rows), metadata=metadata, water_blank_map=mapping)


def estimates(source, **settings):
    return inptk.estimate_concentration(
        inptk.frozen_fraction(source), experiment=source, **settings
    )


@pytest.mark.parametrize("method", ["average", "mle"])
@pytest.mark.parametrize("grid", [None, 0.5])
def test_never_freezing_control_matches_sample_only_including_uncertainty(method, grid):
    source = experiment(blanks={"W": [0] * 6})
    original = source.counts.to_dataframe()
    actual = estimates(source, method=method, temperature_step_C=grid,
                       water_blank_after_first_freeze=True)
    expected = estimates(experiment(blanks={}), method=method, temperature_step_C=grid)
    np.testing.assert_allclose(actual.to_dataframe()[VALUES], expected.to_dataframe()[VALUES],
                               rtol=1e-9, atol=1e-9, equal_nan=True)
    pd.testing.assert_frame_equal(source.counts.to_dataframe(), original)
    group = next(iter(actual.history[-1]["water_blank_controls"].values()))[0]
    assert group["fixed_zero_throughout"] is True
    assert group["first_freeze_temperature_C"] is None


def test_shared_onset_uses_zero_frozen_second_control_too():
    source = experiment()
    result = estimates(source, method="average", water_blank_after_first_freeze=True)
    frame = result.to_dataframe().set_index("temperature_C")
    no_blank = estimates(experiment(blanks={}), method="average").to_dataframe()
    no_blank = no_blank.set_index("temperature_C")
    np.testing.assert_array_equal(frame.loc[[-10, -11], VALUES], no_blank.loc[[-10, -11], VALUES])
    expected = average_concentration(6, 20, 1, 50,
                                    blank_frozen=[1, 0], blank_total=[20, 20],
                                    blank_volume_uL=[50, 50])
    np.testing.assert_allclose(frame.loc[-12, VALUES].astype(float), expected[:3])
    assert json.loads(frame.loc[-12, "water_blank_ids"]) == ["W1", "W2"]
    details = next(iter(result.history[-1]["water_blank_controls"].values()))
    assert len(details) == 1
    assert details[0]["first_freeze_temperature_C"] == -12
    assert details[0]["measurement_ids"] == ["W1", "W2"]


def test_unequal_volumes_keep_direct_average_and_shared_uncertainty(monkeypatch):
    def unexpected_fit(*args, **kwargs):
        raise AssertionError("Average must not fit")
    monkeypatch.setattr("inptk.water_blank.fit_concentration", unexpected_fit)
    monkeypatch.setattr("inptk.curve_fit.fit_curve", unexpected_fit)
    source = experiment(volumes={"W1": 25, "W2": 100})
    actual = estimates(source, method="average", water_blank_after_first_freeze=True)
    row = actual.to_dataframe().set_index("temperature_C").loc[-14]
    expected = average_concentration(12, 20, 1, 50,
                                    blank_frozen=[3, 1], blank_total=[20, 20],
                                    blank_volume_uL=[25, 100])
    np.testing.assert_allclose(row[VALUES].astype(float), expected[:3])


@pytest.mark.parametrize("method", ["average", "mle"])
def test_one_manual_range_restricts_observations_without_masking_colder_samples(method):
    source = experiment()
    result = estimates(source, method=method, water_blank_temperature_range_C={"min_C": -12})
    frame = result.to_dataframe().set_index("temperature_C")
    assert np.isfinite(frame.loc[-15, "concentration"])
    group = next(iter(result.history[-1]["water_blank_controls"].values()))[0]
    assert group["effective_temperature_range_C"] == {"min_C": -12, "max_C": None}
    if method == "average":
        expected = average_concentration(14, 20, 1, 50,
                                        blank_frozen=[1, 0], blank_total=[20, 20],
                                        blank_volume_uL=[50, 50])
        np.testing.assert_allclose(frame.loc[-15, VALUES].astype(float), expected[:3])
    else:
        fit = next(iter(result.history[-1]["joint_curve_fits"].values()))
        controls = [s for s in fit["sources"] if s["role"] == "blank"]
        assert len(controls) == 2
        assert all(min(s["fit_temperatures_C"]) >= -12 for s in controls)


def test_manual_warm_limit_does_not_assume_zero_for_missing_average_blank():
    result = estimates(experiment(), method="average",
                       water_blank_temperature_range_C={"max_C": -12})
    frame = result.to_dataframe().set_index("temperature_C")
    assert frame.loc[[-10, -11], VALUES].isna().all().all()
    assert np.isfinite(frame.loc[-12, VALUES].astype(float)).all()


def test_raw_onset_does_not_move_with_grid_or_sample_exclusions():
    source = experiment(temperatures=[-10.1, -11.1, -12.1, -13.1, -14.1, -15.1])
    options = dict(method="mle", water_blank_after_first_freeze=True,
                   temperature_step_C=0.5, temperature_start_C=-11.2, temperature_end_C=-14.9,
                   fit_step_C=0.25)
    result = estimates(source, **options)
    group = next(iter(result.history[-1]["water_blank_controls"].values()))[0]
    assert group["first_freeze_temperature_C"] == -12.1
    assert group["effective_temperature_range_C"]["max_C"] == -12.1
    fit = next(iter(result.history[-1]["joint_curve_fits"].values()))
    assert fit["water_blank_controls"] == [group]
    assert all(-14.9 <= t <= -11.2 for s in fit["sources"] for t in s["fit_temperatures_C"])


def test_mle_design_fixes_background_before_onset_at_unrounded_temperature():
    sample = FreezingSeries(np.array([-11.7, -12.2]), np.array([4, 8]), 20, .05,
                            .05, "control", -12.1)
    blank = FreezingSeries(np.array([-11.7, -12.2]), np.array([0, 1]), 20, 0,
                           .05, "control", -12.1)
    model = CurveLikelihood([sample, blank], [-11.7, -12.2], fit_step_C=.25)
    size = len(model.temperatures)
    # The first sample freezing interval has no background coefficients. Both
    # the optimum and confidence bounds optimize this same likelihood matrix.
    np.testing.assert_array_equal(model.events[0, size:], 0)
    assert np.any(model.events[1:, size:] > 0)


def test_default_still_considers_uncertainty_from_zero_observed_blank_freezes():
    source = experiment(blanks={"W": [0] * 6})
    ordinary = estimates(source, method="average").to_dataframe()
    zero = estimates(source, method="average", water_blank_after_first_freeze=True).to_dataframe()
    np.testing.assert_array_equal(ordinary.concentration, zero.concentration)
    assert (ordinary.lower_error > zero.lower_error).all()


@pytest.mark.parametrize("method", ["average", "mle"])
def test_individual_steps_all_in_one_and_save_keep_same_control_settings(tmp_path, method):
    source = experiment()
    settings = dict(method=method, water_blank_after_first_freeze=True,
                    water_blank_temperature_range_C={"min_C": -14})
    result = inptk.analyze_concentration(source, **settings)
    step = inptk.cumulative_spectrum(inptk.frozen_fraction(source), experiment=source, **settings)
    expected = estimates(source, **settings)
    np.testing.assert_allclose(step.to_dataframe()[VALUES], expected.to_dataframe()[VALUES])
    result.save(tmp_path / "result.inptk")
    loaded = inptk.load(tmp_path / "result.inptk")
    assert loaded.settings["water_blank_after_first_freeze"] is True
    assert loaded.settings["water_blank_controls"] == result.settings["water_blank_controls"]
    pd.testing.assert_frame_equal(
        loaded.experiment.counts.to_dataframe(), source.counts.to_dataframe()
    )


def test_cycles_and_runs_use_separate_controls_with_one_shared_requested_range():
    source = experiment(runs=("A", "B"), cycles=("0", "1"))
    frame = source.counts.to_dataframe()
    frame.loc[frame.measurement_id.str.startswith("B") & frame.cycle_id.eq("1")
              & frame.measurement_id.ne("BS"), "n_frozen"] = [0, 0, 0, 1, 2, 3] * 2
    source = replace(source, counts=inptk.CountsTable(frame))
    curves = {"zero": {"inputs": ["AS", "BS"], "cycle": "0"},
              "one": {"inputs": ["AS", "BS"], "cycle": "1"}}
    result = estimates(source, method="average", curves=curves,
                       water_blank_after_first_freeze=True,
                       water_blank_temperature_range_C={"min_C": -14})
    controls = result.history[-1]["water_blank_controls"]
    assert len(controls["one"]) == 2
    assert {(c["run_id"], c["first_freeze_temperature_C"]) for c in controls["one"]} == {
        ("A", -12), ("B", -13)}
    assert all(c["effective_temperature_range_C"]["min_C"] == -14
               for group in controls.values() for c in group)


def test_range_suggestions_use_same_blank_choices():
    source = experiment()
    options = dict(water_blank_after_first_freeze=True,
                   water_blank_temperature_range_C={"min_C": -13}, min_unfrozen=2)
    suggestion = inptk.suggest_temperature_ranges(source, **options)
    assert suggestion.settings["water_blank_after_first_freeze"] is True
    group = next(iter(suggestion.settings["water_blank_controls"].values()))[0]
    assert group["first_freeze_temperature_C"] == -12
    assert group["effective_temperature_range_C"]["min_C"] == -13
    frame = suggestion.observations.to_dataframe().set_index("temperature_C")
    result = estimates(source, method="average", water_blank_after_first_freeze=True,
                       water_blank_temperature_range_C={"min_C": -13}).to_dataframe()
    result = result.set_index("temperature_C")
    retained = frame.in_suggested_range
    np.testing.assert_allclose(frame.loc[retained, VALUES], result.loc[retained, VALUES])


def test_client_flags_capabilities_and_in_memory_steps(capsys):
    parser = cli.build_parser()
    caps = cli._capabilities(parser)
    for command in ("analyze", "estimate", "suggest-ranges"):
        options = {flag for option in caps["commands"][command]["options"]
                   for flag in option["flags"]}
        assert "--water-blank-after-first-freeze" in options
        assert "--water-blank-temperature-range" in options
    store = ResultStore(memory=True)
    store.save(experiment(), "@raw")
    assert cli.main(["estimate", "@raw", "--format", "saved", "--out", "@estimated",
                     "--method", "average", "--water-blank-after-first-freeze",
                     "--water-blank-temperature-range", '{"min_C": -14}', "--json"],
                    store=store, parser=parser) == 0
    capsys.readouterr()
    step = store.load("@estimated")
    settings = step.tables["cumulative"].history[-1]
    assert settings["water_blank_after_first_freeze"] is True
    assert settings["water_blank_temperature_range_C"]["min_C"] == -14
    assert cli.main(["suggest-ranges", "@raw", "--format", "saved", "--summary",
                     "--water-blank-after-first-freeze", "--json"], store=store, parser=parser) == 0
    response = json.loads(capsys.readouterr().out)
    assert response["settings"]["water_blank_after_first_freeze"] is True


@pytest.mark.parametrize("value", [0, "yes", None])
def test_first_freeze_requires_boolean(value):
    with pytest.raises(TypeError, match="water_blank_after_first_freeze"):
        estimates(experiment(), water_blank_after_first_freeze=value)


@pytest.mark.parametrize("value", [{"W1": {"min_C": -12}}, {"min_C": -10, "max_C": -12},
                                    {"max_C": np.nan}])
def test_blank_range_requires_single_valid_range(value):
    with pytest.raises((TypeError, ValueError)):
        estimates(experiment(), water_blank_temperature_range_C=value)


@pytest.mark.parametrize("method", ["average", "mle"])
def test_grid_entirely_before_onset_matches_uncorrected_grid(method):
    source = experiment()
    options = dict(method=method, temperature_step_C=0.5,
                   temperature_start_C=-10, temperature_end_C=-11)
    actual = estimates(source, water_blank_after_first_freeze=True, **options)
    expected = estimates(experiment(blanks={}), **options)
    np.testing.assert_allclose(actual.to_dataframe()[VALUES], expected.to_dataframe()[VALUES],
                               equal_nan=True)
    controls = next(iter(actual.history[-1]["water_blank_controls"].values()))
    assert controls[0]["first_freeze_temperature_C"] == -12


@pytest.mark.parametrize("rule", ["latest", "max", "window"])
def test_grid_control_selection_honors_rule_and_one_onset(rule):
    source = experiment()
    settings = dict(method="average", temperature_step_C=0.5, temperature_method=rule,
                    water_blank_after_first_freeze=True)
    if rule == "window":
        settings["temperature_window_C"] = 1
    actual = estimates(source, **settings).to_dataframe().set_index("temperature_C")
    expected = estimates(experiment(blanks={}), **{k: v for k, v in settings.items()
                                                  if k != "water_blank_after_first_freeze"})
    expected = expected.to_dataframe().set_index("temperature_C")
    np.testing.assert_allclose(actual.loc[[-10, -10.5, -11, -11.5], VALUES],
                               expected.loc[[-10, -10.5, -11, -11.5], VALUES])
    assert json.loads(actual.loc[-12, "water_blank_ids"]) == ["W1", "W2"]


def test_controls_do_not_invent_states_past_original_support():
    source = experiment()
    original = source.counts.to_dataframe()
    trimmed = original.loc[original.measurement_id.eq("S") | original.temperature_C.ge(-13)]
    source = replace(source, counts=inptk.CountsTable(trimmed))
    actual = estimates(source, method="average", water_blank_after_first_freeze=True)
    frame = actual.to_dataframe().set_index("temperature_C")
    assert frame.loc[[-14, -15], VALUES].isna().all().all()
    assert set(frame.loc[[-14, -15], "selection_status"]) == {"missing_blank_observations"}


def test_only_control_range_applies_to_individuals_and_raw_sample_ranges_do_not_move_onset():
    source = experiment()
    raw = source.counts.to_dataframe()
    duplicate = raw.loc[raw.measurement_id.eq("S")].copy()
    duplicate["measurement_id"] = "D"
    metadata = source.measurements.copy()
    metadata["D"] = replace(metadata["S"], measurement_id="D", dilution=2)
    source = replace(source,
                     counts=inptk.CountsTable(pd.concat([raw, duplicate], ignore_index=True)),
                     measurements=metadata, water_blank_map={**source.water_blank_map,
                                                             "D": ["W1", "W2"]})
    result = estimates(source, method="mle", curves={"curve": {"inputs": ["S", "D"], "cycle": "0"}},
                       water_blank_after_first_freeze=True,
                       temperature_ranges_C={"S": {"min_C": -15, "max_C": -14},
                                             "D": {"min_C": -15, "max_C": -14}})
    group = result.history[-1]["water_blank_controls"]["curve"][0]
    assert group["first_freeze_temperature_C"] == -12
    assert group["effective_temperature_range_C"] == {"max_C": -12}


def test_fixed_zero_run_has_no_shared_background_error_term():
    sample_only = average_concentration([6, 6], [20, 20], 1, 50)
    mixed = average_concentration([6, 6], [20, 20], 1, 50,
                     blank_frozen=[1, 0], blank_total=[20, 20], blank_volume_uL=[50, 50],
                     sample_blank_group=["A", "B"], blank_group=["A", "A"], fixed_zero_groups=["B"])
    blank = -np.log1p(-1 / 40) / .05
    assert mixed[0] == pytest.approx(sample_only[0] - blank / 2)
    assert mixed[1] > sample_only[1]
    assert mixed[2] > sample_only[2]
