"""Suggested limits must be safe to apply without silently broadening selection."""

import json

import numpy as np
import pandas as pd
import pytest

import inptk
from inptk.cli import main


def source(frozen=(0, 2, 3, 16, 29, 30, 32), *, temperatures=None, cycles=("01",), blank=None):
    temperatures = temperatures or [-5 - index for index in range(len(frozen))]
    rows, metadata = [], []
    streams = [("001", frozen, temperatures, 32, 50)]
    if blank is not None:
        streams.append(("water", *blank))
    for name, counts, temps, total, volume in streams:
        metadata.append({"measurement_id": name, "sample_id": name, "run_id": "R",
                         "droplet_volume_uL": volume, "dilution": 1})
        for cycle in cycles:
            for index, (temperature, count) in enumerate(zip(temps, counts, strict=True)):
                rows.append({"measurement_id": name, "cycle_id": cycle,
                             "temperature_C": temperature, "time_s": index,
                             "n_total": total, "n_frozen": count})
    return inptk.read_counts(pd.DataFrame(rows), metadata=metadata,
                            water_blank_map={"001": ["water"]} if blank else None)


def test_count_limits_are_inclusive_editable_and_preserve_observations():
    data = source()
    before = data.counts.to_dataframe()
    proposal = inptk.suggest_temperature_ranges(data)
    assert proposal.temperature_ranges_C == {"001": {"min_C": -9, "max_C": -5}}
    assert proposal.inputs["001"]["warm_limit_reason"] == ["observed_temperature_limit"]
    assert proposal.inputs["001"]["cold_limit_reason"] == ["too_few_liquid"]
    report = proposal.observations.to_dataframe()
    assert report.in_suggested_range.tolist() == [True, True, True, True, True, False, False]
    assert report.loc[report.in_suggested_range, "blank_status"].eq("not_applied").all()
    pd.testing.assert_frame_equal(data.counts.to_dataframe(), before)
    limits = proposal.temperature_ranges_C
    limits["001"]["max_C"] = -6
    assert proposal.temperature_ranges_C["001"]["max_C"] == -5
    result = inptk.analyze_concentration(
        data, method="average", temperature_ranges_C=proposal.temperature_ranges_C
    )
    # Suggestions only select contributions to combined curves, not this individual.
    pd.testing.assert_frame_equal(
        result.to_dataframe(), inptk.analyze_concentration(data, method="average").to_dataframe()
    )


def test_repeated_temperature_is_usable_only_when_every_observation_passes():
    data = source((29, 30, 10, 30), temperatures=[-5, -5, -6, -7])
    proposal = inptk.suggest_temperature_ranges(data)
    assert proposal.temperature_ranges_C == {"001": {"min_C": -6, "max_C": -6}}
    report = proposal.observations.to_dataframe()
    assert not report.in_suggested_range.iloc[0]
    assert "another_observation_at_same_temperature_failed" in (
        report.range_exclusion_reasons.iloc[0]
    )


def test_initial_block_is_not_abandoned_for_a_longer_colder_block():
    data = source((3, 4, 5, 30, 8, 9), temperatures=[-5, -5.1, -5.2, -6, -7, -10])
    proposal = inptk.suggest_temperature_ranges(data)
    assert proposal.temperature_ranges_C["001"] == {"min_C": -5.2, "max_C": -5}
    assert proposal.inputs["001"]["kept_observations"] == 3


def test_equal_spans_choose_warmer_block():
    proposal = inptk.suggest_temperature_ranges(source((3, 4, 30, 8, 9)))
    assert proposal.temperature_ranges_C["001"] == {"min_C": -6, "max_C": -5}


@pytest.mark.parametrize("option,value", [
    ("min_frozen", 0), ("min_unfrozen", -1), ("min_frozen", 2.5), ("min_unfrozen", True),
])
def test_thresholds_are_positive_integer_well_counts(option, value):
    with pytest.raises(ValueError, match="positive integer"):
        inptk.suggest_temperature_ranges(source(), **{option: value})


def test_no_usable_range_is_reported_and_cannot_be_applied_as_unrestricted():
    proposal = inptk.suggest_temperature_ranges(source((30, 31, 32)))
    assert proposal.inputs["001"]["status"] == "no_usable_range"
    assert not proposal.observations.to_dataframe().in_suggested_range.any()
    with pytest.raises(ValueError, match="No usable temperature range"):
        _ = proposal.temperature_ranges_C


def test_cycles_must_be_selected_and_duplicate_named_outputs_do_not_duplicate_data():
    data = source(cycles=("01", "02"))
    with pytest.raises(ValueError, match="one cycle per measurement"):
        inptk.suggest_temperature_ranges(data)
    curves = {"one": {"inputs": ["001"], "cycle": "02"},
              "another": {"inputs": ["001"], "cycle": "02"}}
    proposal = inptk.suggest_temperature_ranges(data, curves=curves)
    assert proposal.inputs["001"]["cycle_id"] == "02"
    assert len(proposal.observations) == 7


def test_weak_blank_corrected_signal_is_flagged_but_never_excluded():
    data = source((4, 8, 16), blank=([4, 8, 16], [-5, -6, -7], 32, 50))
    proposal = inptk.suggest_temperature_ranges(data)
    report = proposal.observations.to_dataframe()
    assert report.in_suggested_range.all()
    assert report.blank_status.eq("not_distinguished_from_blank").all()
    assert report.range_exclusion_reasons.eq("[]").all()
    assert report.upper_error.gt(0).all()
    assert proposal.settings["blank_flags_change_ranges"] is False


def test_blank_sample_size_and_volume_are_used_by_the_existing_statistical_model():
    reports = []
    for frozen, total in [(1, 4), (8, 32)]:
        data = source((16,), blank=([frozen], [-5], total, 100))
        proposal = inptk.suggest_temperature_ranges(data)
        actual = proposal.observations.to_dataframe().iloc[0]
        expected = inptk.cumulative_spectrum(
            inptk.frozen_fraction(data), experiment=data, method="average"
        ).to_dataframe().iloc[0]
        for column in ("concentration", "lower_error", "upper_error"):
            assert actual[column] == expected[column]
        reports.append(actual)
    assert reports[0].concentration == pytest.approx(reports[1].concentration)
    assert reports[1].upper_error < reports[0].upper_error
    assert reports[1].lower_error < reports[0].lower_error


def test_missing_blank_coverage_limits_range_and_disabling_correction_restores_support():
    data = source((4, 8, 16), blank=([0, 1], [-6, -7], 32, 50))
    proposal = inptk.suggest_temperature_ranges(data)
    assert proposal.temperature_ranges_C["001"] == {"min_C": -7, "max_C": -6}
    assert proposal.inputs["001"]["warm_limit_reason"] == ["missing_blank_coverage"]
    uncorrected = inptk.suggest_temperature_ranges(data, water_blank_correction=False)
    assert uncorrected.temperature_ranges_C["001"] == {"min_C": -7, "max_C": -5}
    assert uncorrected.observations.to_dataframe().blank_status.eq("not_applied").all()
    assert set(uncorrected.inputs) == {"001"}


@pytest.mark.parametrize("frozen,complete", [((0, 3, 16, 30), True), ((30, 31, 32), False)])
def test_cli_suggestions_match_python_and_do_not_write_or_modify_input(tmp_path, capsys,
                                                                    frozen, complete):
    data = source(frozen)
    path = tmp_path / "experiment.inptk"
    data.save(path)
    before = (path / "analysis.json").read_bytes()
    assert main(["suggest-ranges", str(path), "--format", "saved", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    proposal = inptk.suggest_temperature_ranges(data)
    assert payload["complete"] is complete
    assert payload["inputs"] == proposal.inputs
    assert payload["temperature_ranges_C"] == (proposal.temperature_ranges_C if complete else None)
    assert (path / "analysis.json").read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_cli_thresholds_and_input_selection_are_exposed_in_capabilities(capsys):
    assert main(["capabilities"]) == 0
    payload = json.loads(capsys.readouterr().out)
    flags = {item["name"]: item for item in payload["commands"]["suggest-ranges"]["options"]}
    assert flags["min_frozen"]["default"] == flags["min_unfrozen"]["default"] == 3
    assert "method" not in flags
    assert "curves" in flags and "cycle" in flags


def test_suggestions_end_before_a_decrease_in_corrected_concentration():
    data = source((8, 8, 8), blank=([0, 1, 4], [-5, -6, -7], 32, 50))
    report = inptk.suggest_temperature_ranges(data).observations.to_dataframe()
    assert report.in_suggested_range.tolist() == [True, False, False]
    assert "concentration_decrease" in report.range_exclusion_reasons.iloc[1]
    proposal = inptk.suggest_temperature_ranges(data)
    assert proposal.temperature_ranges_C["001"] == {"min_C": -5, "max_C": -5}


def dilution_series(*, second_dilution=10, second_counts=None):
    rows, metadata = [], []
    for name, dilution, counts in [
        ("A", 1, [0, 2, 12, 29, 30, 31, 32, 32]),
        ("B", second_dilution, second_counts or [0, 0, 3, 8, 12, 20, 29, 30]),
        ("C", 100, [0, 0, 0, 3, 5, 7, 12, 18]),
    ]:
        metadata.append({"measurement_id": name, "sample_id": "sample", "dilution": dilution,
                         "droplet_volume_uL": 50})
        rows.extend({"measurement_id": name, "temperature_C": -5-i,
                     "n_total": 32, "n_frozen": count} for i, count in enumerate(counts))
    return inptk.read_counts(pd.DataFrame(rows), metadata=metadata)


def test_exhaust_previous_dilution_before_switching_even_when_next_is_already_usable():
    data = dilution_series()
    # Input-list order does not determine the dilution sequence.
    curves = {"combined": {"inputs": ["C", "A", "B"]}}
    proposal = inptk.suggest_temperature_ranges(data, curves=curves)
    assert proposal.temperature_ranges_C == {
        "A": {"min_C": -8, "max_C": -5},
        "B": {"min_C": -11, "max_C": -9},
        "C": {"min_C": -12, "max_C": -12},
    }
    assert proposal.inputs["A"]["min_frozen_applied"] is False
    assert proposal.inputs["B"]["warm_limit_reason"] == ["previous_dilution_active"]
    result = inptk.estimate_concentration(
        inptk.frozen_fraction(data), experiment=data, curves=curves, method="average",
        temperature_ranges_C=proposal.temperature_ranges_C,
    ).to_dataframe()
    assert [json.loads(ids) for ids in result.contributing_measurement_ids] == (
        [["A"]] * 4 + [["B"]] * 3 + [["C"]]
    )


def test_equal_dilution_inputs_also_have_nonoverlapping_monotone_ranges():
    proposal = inptk.suggest_temperature_ranges(dilution_series(second_dilution=1))
    assert proposal.temperature_ranges_C["B"] == {"min_C": -11, "max_C": -11}
    assert proposal.temperature_ranges_C["C"] == {"min_C": -12, "max_C": -12}


def test_later_dilution_still_needs_minimum_frozen_count_and_gaps_remain_visible():
    proposal = inptk.suggest_temperature_ranges(
        dilution_series(second_counts=[0, 0, 0, 0, 1, 2, 3, 30])
    )
    assert proposal.temperature_ranges_C["B"] == {"min_C": -11, "max_C": -11}
    assert proposal.inputs["B"]["warm_limit_reason"] == ["too_few_frozen"]


def test_unused_later_dilution_is_reported_instead_of_restored_to_full_range():
    proposal = inptk.suggest_temperature_ranges(
        dilution_series(second_counts=[0, 0, 0, 0, 0, 0, 0, 0])
    )
    assert proposal.inputs["B"]["status"] == "no_usable_range"
    with pytest.raises(ValueError, match="No usable temperature range"):
        _ = proposal.temperature_ranges_C


def test_shared_input_with_different_switching_context_requires_separate_suggestions():
    with pytest.raises(ValueError, match="different curve input sets"):
        inptk.suggest_temperature_ranges(dilution_series(), curves={
            "combined": {"inputs": ["A", "B"]}, "alone": {"inputs": ["B"]},
        })


def two_inputs(first, second, *, second_dilution=2):
    rows, metadata = [], []
    for name, counts, dilution in [("A", first, 1), ("B", second, second_dilution)]:
        metadata.append({"measurement_id": name, "sample_id": "sample",
                         "dilution": dilution, "droplet_volume_uL": 50})
        rows.extend({"measurement_id": name, "temperature_C": -5-i,
                     "n_total": 32, "n_frozen": count} for i, count in enumerate(counts))
    return inptk.read_counts(rows, metadata=metadata)


def unfiltered(data, proposal, *, curves=None, **grid):
    return inptk.estimate_concentration(
        inptk.frozen_fraction(data), experiment=data, method="average", curves=curves,
        temperature_ranges_C=proposal.temperature_ranges_C, **grid,
    ).to_dataframe()


def test_handoff_shortens_previous_range_without_modifying_concentrations():
    data = two_inputs([0, 16, 24, 29, 30], [0, 3, 5, 8, 12])
    proposal = inptk.suggest_temperature_ranges(data)
    assert proposal.temperature_ranges_C == {
        "A": {"min_C": -6, "max_C": -5}, "B": {"min_C": -9, "max_C": -9},
    }
    assert proposal.inputs["A"]["cold_limit_reason"] == ["shortened_for_monotone_handoff"]
    table = unfiltered(data, proposal)
    finite = table.loc[np.isfinite(table.concentration)]
    assert finite.temperature_C.tolist() == [-5, -6, -9]
    assert finite.contributor_count.eq(1).all()
    np.testing.assert_allclose(finite.concentration, -np.log([1, 0.5, 0.625]) / 0.05 * [1, 1, 2])
    assert finite.concentration.diff().dropna().ge(0).all()


def test_impossible_handoff_stops_and_reports_unused_input():
    data = two_inputs([8, 16, 24, 29], [0, 0, 0, 3], second_dilution=1)
    proposal = inptk.suggest_temperature_ranges(data)
    assert proposal.inputs["A"]["range_C"] == {"min_C": -8, "max_C": -5}
    assert proposal.inputs["B"]["range_C"] is None
    report = proposal.observations.to_dataframe()
    assert report.loc[report.measurement_id.eq("B"), "range_exclusion_reasons"].str.contains(
        "no_monotone_continuation|previous_dilution_active"
    ).all()
    with pytest.raises(ValueError, match="No usable temperature range"):
        _ = proposal.temperature_ranges_C


def test_total_count_change_is_checked_through_concentration():
    data = inptk.read_counts(
        {"measurement_id": ["A"] * 3, "temperature_C": [-5, -6, -7],
         "n_total": [32, 32, 64], "n_frozen": [8, 12, 12]},
        metadata=[{"measurement_id": "A", "sample_id": "A", "dilution": 1,
                   "droplet_volume_uL": 50}],
    )
    proposal = inptk.suggest_temperature_ranges(data)
    assert proposal.temperature_ranges_C["A"] == {"min_C": -6, "max_C": -5}
    individual = unfiltered(data, proposal)
    assert individual.concentration.iloc[-1] < individual.concentration.iloc[-2]
    allowed = individual[individual.temperature_C.between(-6, -5)]
    assert allowed.concentration.diff().dropna().ge(0).all()


@pytest.mark.parametrize("method", ["latest", "max", "window"])
def test_monotone_limits_use_the_same_initial_grid_and_rule_as_analysis(method):
    temperatures = [-5, -5.4, -6, -6.4, -7]
    data = source((0, 4, 9, 12, 29), temperatures=temperatures,
                  blank=([0, 1, 7, 7, 8], temperatures, 32, 50))
    grid = {"temperature_step_C": 0.5, "temperature_start_C": -5,
            "temperature_end_C": -7, "temperature_method": method}
    if method == "window":
        grid["temperature_window_C"] = 0.5
    proposal = inptk.suggest_temperature_ranges(data, **grid)
    table = unfiltered(data, proposal, **grid)
    limits = proposal.temperature_ranges_C["001"]
    allowed = table.temperature_C.between(limits["min_C"], limits["max_C"])
    finite = table.loc[allowed & np.isfinite(table.concentration)]
    assert len(finite) >= 2
    assert finite.concentration.diff().dropna().ge(0).all()
    assert finite.contributor_count.eq(1).all()
    for key, value in grid.items():
        assert proposal.settings[key] == value


def test_count_eligible_island_between_grid_targets_does_not_block_later_input():
    data = two_inputs([0, 16, 29, 30, 31, 32], [0, 3, 2, 8, 16, 29], second_dilution=10)
    grid = {"temperature_step_C": 2}
    proposal = inptk.suggest_temperature_ranges(data, **grid)
    assert proposal.inputs["B"]["range_C"] is not None
    finite = unfiltered(data, proposal, **grid).dropna(subset=["concentration"])
    assert finite.concentration.diff().dropna().ge(0).all()


def test_cli_monotone_limits_share_grid_settings_with_python(tmp_path, capsys):
    data = source((8, 8, 8), blank=([0, 1, 4], [-5, -6, -7], 32, 50))
    path = tmp_path / "experiment.inptk"
    data.save(path)
    assert main(["suggest-ranges", str(path), "--format", "saved",
                 "--temperature-step-C", "0.5", "--temperature-start-C", "-5.1",
                 "--temperature-end-C", "-6.9", "--temperature-method", "max", "--json"]) == 0
    actual = json.loads(capsys.readouterr().out)
    expected = inptk.suggest_temperature_ranges(
        data, temperature_step_C=0.5, temperature_start_C=-5.1,
        temperature_end_C=-6.9, temperature_method="max",
    )
    assert actual["temperature_ranges_C"] == expected.temperature_ranges_C
    assert actual["settings"] == expected.settings


def test_unselected_cycles_do_not_change_handoff_limits():
    data = dilution_series()
    curves = {"sample": {"inputs": ["A", "B", "C"], "cycle": "1"}}
    expected = inptk.suggest_temperature_ranges(data, curves=curves)
    original = data.counts.to_dataframe()
    other = original.assign(cycle_id="2", temperature_C=original.temperature_C + 0.125)
    both = inptk.Experiment(inptk.CountsTable(pd.concat([original, other])),
                            data.samples, data.measurements)
    actual = inptk.suggest_temperature_ranges(both, curves=curves)
    assert actual.temperature_ranges_C == expected.temperature_ranges_C


def test_report_keeps_observation_identity_with_shuffled_input_rows():
    data = source((0, 4, 8, 12, 16),
                  blank=([0, 0, 1, 1, 2], [-5, -6, -7, -8, -9], 32, 50))
    original = data.counts.to_dataframe().sample(frac=1, random_state=7)
    shuffled = inptk.Experiment(
        inptk.CountsTable(original), data.samples, data.measurements,
        water_blank_map=data.water_blank_map,
    )
    report = inptk.suggest_temperature_ranges(
        shuffled, temperature_step_C=.5
    ).observations.to_dataframe()
    expected = inptk.cumulative_spectrum(
        inptk.frozen_fraction(shuffled), experiment=shuffled, method="average"
    ).to_dataframe().set_index("observation_id")
    assert report.observation_id.tolist() == original.loc[
        original.measurement_id.eq("001"), "observation_id"
    ].tolist()
    assert len(report) == 5
    columns = ["concentration", "lower_error", "upper_error"]
    pd.testing.assert_frame_equal(
        report.set_index("observation_id")[columns].sort_index(),
        expected[columns].sort_index(),
    )
    for row in report.itertuples():
        assert json.loads(row.blank_observation_ids) == [{
            "measurement_id": "water", "run_id": "R", "cycle_id": "01",
            "observation_id": row.observation_id,
        }]


@pytest.mark.parametrize("method", ["latest", "max", "window"])
@pytest.mark.parametrize("blank", [
    None, ([0, 0, 1, 1, 2, 2, 3], [-5, -6, -7, -8, -9, -10, -11], 20, 25),
])
def test_summary_preserves_ranges_and_reasons_without_report(method, blank, monkeypatch):
    from inptk import ranges

    data = source(blank=blank)
    settings = {"temperature_step_C": 1, "temperature_method": method}
    if method == "window":
        settings["temperature_window_C"] = 1
    full = inptk.suggest_temperature_ranges(data, **settings)

    def no_report_alignment(*args, **kwargs):
        raise AssertionError("A summary must not calculate the per-observation report")

    monkeypatch.setattr(ranges, "align_observations", no_report_alignment)
    summary = inptk.suggest_temperature_ranges(data, include_observations=False, **settings)
    assert summary.observations is None
    assert summary.inputs == full.inputs
    assert summary.settings == full.settings
    assert summary.temperature_ranges_C == full.temperature_ranges_C


def test_cli_summary_matches_full_report(tmp_path, capsys):
    data = source()
    data.save(tmp_path / "source.inptk")
    args = ["suggest-ranges", str(tmp_path / "source.inptk"), "--format", "saved", "--json"]
    assert main(args) == 0
    full = json.loads(capsys.readouterr().out)
    assert main([*args, "--summary"]) == 0
    summary = json.loads(capsys.readouterr().out)
    full.pop("table")
    assert summary == full
