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
    assert proposal.temperature_ranges_C == {"001": {"min_C": -9, "max_C": -7}}
    assert proposal.inputs["001"]["warm_limit_reason"] == ["too_few_frozen"]
    assert proposal.inputs["001"]["cold_limit_reason"] == ["too_few_liquid"]
    report = proposal.observations.to_dataframe()
    assert report.in_suggested_range.tolist() == [False, False, True, True, True, False, False]
    assert report.loc[report.in_suggested_range, "blank_status"].eq("not_applied").all()
    pd.testing.assert_frame_equal(data.counts.to_dataframe(), before)
    limits = proposal.temperature_ranges_C
    limits["001"]["max_C"] = -6
    assert proposal.temperature_ranges_C["001"]["max_C"] == -7
    result = inptk.analyze_concentration(
        data, method="average", temperature_ranges_C=proposal.temperature_ranges_C
    )
    assert result.to_dataframe().temperature_C.tolist() == [-7, -8, -9]


def test_repeated_temperature_is_usable_only_when_every_observation_passes():
    data = source((2, 3, 10, 30), temperatures=[-5, -5, -6, -7])
    proposal = inptk.suggest_temperature_ranges(data)
    assert proposal.temperature_ranges_C == {"001": {"min_C": -6, "max_C": -6}}
    report = proposal.observations.to_dataframe()
    assert not report.in_suggested_range.iloc[1]
    assert "another_observation_at_same_temperature_failed" in (
        report.range_exclusion_reasons.iloc[1]
    )


def test_disjoint_eligible_regions_choose_temperature_span_without_bridging():
    data = source((3, 4, 5, 1, 8, 9), temperatures=[-5, -5.1, -5.2, -6, -7, -10])
    proposal = inptk.suggest_temperature_ranges(data)
    assert proposal.temperature_ranges_C["001"] == {"min_C": -10, "max_C": -7}
    assert proposal.inputs["001"]["kept_observations"] == 2


def test_equal_spans_choose_warmer_block():
    proposal = inptk.suggest_temperature_ranges(source((3, 4, 1, 8, 9)))
    assert proposal.temperature_ranges_C["001"] == {"min_C": -6, "max_C": -5}


@pytest.mark.parametrize("option,value", [
    ("min_frozen", 0), ("min_unfrozen", -1), ("min_frozen", 2.5), ("min_unfrozen", True),
])
def test_thresholds_are_positive_integer_well_counts(option, value):
    with pytest.raises(ValueError, match="positive integer"):
        inptk.suggest_temperature_ranges(source(), **{option: value})


def test_no_usable_range_is_reported_and_cannot_be_applied_as_unrestricted():
    proposal = inptk.suggest_temperature_ranges(source((0, 1, 2)))
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


@pytest.mark.parametrize("frozen,complete", [((0, 3, 16, 30), True), ((0, 1, 2), False)])
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


def test_suggestions_do_not_optimize_away_a_decrease_in_corrected_concentration():
    data = source((8, 8, 8), blank=([0, 1, 4], [-5, -6, -7], 32, 50))
    report = inptk.suggest_temperature_ranges(data).observations.to_dataframe()
    assert report.in_suggested_range.all()
    assert np.all(np.diff(report.concentration) < 0)
