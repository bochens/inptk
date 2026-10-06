"""Average contributors must be reportable physical sample observations."""

import json

import numpy as np
import pandas as pd
import pytest

import inptk


def experiment(streams, *, blanks=None):
    records, metadata = [], []
    blank_ids = {name for assigned in (blanks or {}).values() for name in assigned}
    for name, counts, dilution in streams:
        metadata.append({"measurement_id": name,
                         "sample_id": "water" if name in blank_ids else "sample",
                         "dilution": dilution, "droplet_volume_uL": 50})
        for index, count in enumerate(counts):
            records.append({"measurement_id": name, "cycle_id": "1", "time_s": index,
                            "temperature_C": -5 - index, "n_total": 10, "n_frozen": count})
    return inptk.read_counts(records, metadata=metadata, water_blank_map=blanks)


@pytest.mark.parametrize("grid", [
    {}, {"temperature_step_C": .5},
    {"temperature_step_C": .5, "temperature_method": "max"},
    {"temperature_step_C": .5, "temperature_method": "window", "temperature_window_C": 1},
])
@pytest.mark.parametrize("correct_blank", [False, True])
def test_full_range_uses_only_each_inputs_freezing_interval(grid, correct_blank):
    streams = [("A", [0, 2, 4, 4, 4, 4], 1),
               ("B", [0, 0, 0, 1, 3, 3], 10),
               ("never_frozen", [0, 0, 0, 0, 0, 0], 100)]
    blanks = None
    if correct_blank:
        streams.append(("water", [0, 0, 0, 0, 0, 0], 1))
        blanks = {name: ["water"] for name in ("A", "B", "never_frozen")}
    data = experiment(streams, blanks=blanks)
    original = data.counts.to_dataframe()
    curves = {name: {"inputs": [name]} for name in ("A", "B", "never_frozen")}
    curves["combined"] = {"inputs": ["A", "B", "never_frozen"]}
    estimated = inptk.estimate_concentration(
        inptk.frozen_fraction(data), experiment=data, curves=curves, method="average", **grid
    ).to_dataframe()
    combined = estimated.loc[estimated.curve_id.eq("combined")].set_index("temperature_C")
    for temperature, name in [(-6, "A"), (-7, "A"), (-8, "B"), (-9, "B")]:
        point = combined.loc[temperature]
        individual = estimated.loc[estimated.curve_id.eq(name)
                                   & estimated.temperature_C.eq(temperature)].iloc[0]
        assert point.contributor_count == 1
        assert json.loads(point.contributing_measurement_ids) == [name]
        for column in ("concentration", "lower_error", "upper_error"):
            assert point[column] == individual[column]
        assert {source["measurement_id"]
                for source in json.loads(point.source_observations)
                if source["role"] == "sample"} == {name}
    values = ["concentration", "lower_error", "upper_error"]
    assert combined.loc[[-5, -10], values].isna().all().all()
    assert combined.loc[[-5, -10], "contributor_count"].eq(0).all()
    assert estimated.loc[estimated.curve_id.eq("never_frozen"), "concentration"].isna().all()
    pd.testing.assert_frame_equal(data.counts.to_dataframe(), original)


def test_manual_overlap_averages_only_inputs_that_have_started_freezing():
    data = experiment([("A", [0, 2, 4, 6], 1), ("B", [0, 0, 1, 3], 2)])
    result = inptk.estimate_concentration(
        inptk.frozen_fraction(data), experiment=data, method="average",
        temperature_ranges_C={"A": {"min_C": -8, "max_C": -5},
                              "B": {"min_C": -8, "max_C": -5}},
    ).to_dataframe().set_index("temperature_C")
    assert result.loc[-6, "contributor_count"] == 1
    assert result.loc[-6, "concentration"] == pytest.approx(-np.log(.8) / .05)
    assert result.loc[-7, "contributor_count"] == 2
    assert result.loc[-7, "concentration"] == pytest.approx((-np.log(.6) - 2*np.log(.9)) / .1)


@pytest.mark.parametrize("grid", [
    {"temperature_step_C": None},
    {"temperature_step_C": .5},
    {"temperature_step_C": .5, "temperature_method": "max"},
    {"temperature_step_C": .5, "temperature_method": "window", "temperature_window_C": 1},
])
@pytest.mark.parametrize("selected_range", [False, True])
@pytest.mark.parametrize("correct_blank", [False, True])
def test_average_excludes_all_frozen_inputs_before_mean_and_uncertainty(
    grid, selected_range, correct_blank,
):
    streams = [("A", [0, 2, 10, 10, 10], 1), ("B", [0, 1, 3, 6, 10], 10)]
    blanks = None
    if correct_blank:
        streams.append(("water", [0, 0, 1, 1, 2], 1))
        blanks = {"A": ["water"], "B": ["water"]}
    data = experiment(streams, blanks=blanks)
    original = data.counts.to_dataframe()
    options = {"temperature_ranges_C": {
        "A": {"min_C": -9, "max_C": -5}, "B": {"min_C": -9, "max_C": -5},
    }} if selected_range else {}
    result = inptk.estimate_concentration(
        inptk.frozen_fraction(data), experiment=data, method="average",
        curves={"B": {"inputs": ["B"]}, "combined": {"inputs": ["A", "B"]}},
        **grid, **options,
    ).to_dataframe()
    combined = result.loc[result.curve_id.eq("combined")].set_index("temperature_C")
    single = result.loc[result.curve_id.eq("B")].set_index("temperature_C")
    values = ["concentration", "lower_error", "upper_error"]
    assert combined.loc[-6, "contributor_count"] == 2
    # A's last event at -7 freezes every well. It must not spoil B's finite estimate.
    for temperature in (-7, -8):
        point = combined.loc[temperature]
        assert point.contributor_count == 1
        assert json.loads(point.contributing_measurement_ids) == ["B"]
        np.testing.assert_allclose(point[values].to_numpy(dtype=float),
                                   single.loc[temperature, values].to_numpy(dtype=float))
        assert np.isfinite(point[values].to_numpy(dtype=float)).all()
        assert {row["measurement_id"] for row in json.loads(point.source_observations)
                if row["role"] == "sample"} == {"B"}
    assert combined.loc[-9, values].isna().all()
    assert combined.loc[-9, "contributor_count"] == 0
    assert combined.loc[-9, "selection_status"] == "no_eligible_measurements"
    # Individual spectra expose the direct all-frozen calculation.
    assert np.isinf(single.loc[-9, "concentration"])
    assert single.loc[-9, ["lower_error", "upper_error"]].isna().all()
    pd.testing.assert_frame_equal(data.counts.to_dataframe(), original)


@pytest.mark.parametrize("ranges", [
    {}, {"A": {"min_C": -7, "max_C": -6}, "B": {"min_C": -7, "max_C": -6}},
])
def test_cli_average_excludes_all_frozen_in_full_and_selected_ranges(tmp_path, capsys, ranges):
    from inptk.cli import main

    data = experiment([("A", [0, 2, 10], 1), ("B", [0, 1, 3], 10)])
    source, output = tmp_path / "source.inptk", tmp_path / "average.inptk"
    data.save(source)
    assert main([
        "analyze", str(source), "--format", "saved", "--method", "average",
        "--curves", json.dumps({"combined": {"inputs": ["A", "B"]}}),
        "--temperature-step", "0.5", "--temperature-ranges", json.dumps(ranges),
        "--out", str(output), "--json",
    ]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    table = inptk.load(output).to_dataframe()
    point = table.loc[table.curve_id.eq("combined") & table.temperature_C.eq(-7)].iloc[0]
    assert point.concentration == pytest.approx(-10 * np.log(.7) / .05)
    assert np.isfinite([point.concentration, point.lower_error, point.upper_error]).all()
    assert point.contributor_count == 1
    assert json.loads(point.contributing_measurement_ids) == ["B"]


def test_corrected_zero_with_positive_sample_counts_remains_a_contributor():
    data = experiment([("A", [1, 2], 1), ("B", [3, 4], 1), ("water", [1, 2], 1)],
                      blanks={"A": ["water"], "B": ["water"]})
    result = inptk.estimate_concentration(
        inptk.frozen_fraction(data), experiment=data, method="average",
        curves={"A": {"inputs": ["A"]}, "B": {"inputs": ["B"]},
                "combined": {"inputs": ["A", "B"]}},
    ).to_dataframe()
    combined = result.loc[result.curve_id.eq("combined")]
    signal = result.loc[result.curve_id.eq("B")]
    assert combined.contributor_count.eq(2).all()
    np.testing.assert_allclose(combined.concentration, signal.concentration.to_numpy() / 2)
    assert result.loc[result.curve_id.eq("A"), "concentration"].eq(0).all()
    assert combined.lower_error.gt(0).all()


@pytest.mark.parametrize("method", ["average", "mle"])
def test_individual_estimate_values_outside_events_are_nan(method):
    data = experiment([("A", [0, 2, 2, 4, 4], 1)])
    result = inptk.estimate_concentration(
        inptk.frozen_fraction(data), experiment=data, method=method, temperature_step_C=.5
    ).to_dataframe()
    outside = result.reporting_status.ne("within_freezing_interval")
    assert result.loc[outside, ["concentration", "lower_error", "upper_error"]].isna().all().all()
    assert result.loc[~outside, "concentration"].notna().all()
    assert result.loc[outside, "qc_flag"].eq(1).all()


def test_suggested_limits_do_not_include_unfrozen_prefix_or_postfreeze_tail():
    data = experiment([("A", [0, 2, 4, 4, 4, 4], 1), ("B", [0, 0, 0, 1, 3, 3], 10)])
    proposal = inptk.suggest_temperature_ranges(data, min_frozen=1)
    assert proposal.temperature_ranges_C == {
        "A": {"min_C": -7, "max_C": -6}, "B": {"min_C": -9, "max_C": -8}}
    result = inptk.analyze_concentration(
        data, method="average", temperature_ranges_C=proposal.temperature_ranges_C
    ).to_dataframe()
    assert result.contributor_count.eq(1).all()
    assert result.concentration.diff().dropna().ge(0).all()


@pytest.mark.parametrize("method", ["average", "mle"])
def test_nan_spectra_survive_cli_conversion_save_selection_and_export(method, tmp_path, capsys):
    from inptk.cli import main

    raw = experiment([("A", [0, 2, 2, 4, 4], 1)])
    data = inptk.read_counts(raw.counts.to_dataframe(), metadata=[{
        "measurement_id": "A", "sample_id": "sample", "dilution": 1,
        "droplet_volume_uL": 50, "sample_type": "air", "air_volume_L": 100,
        "suspension_volume_mL": 10, "filter_fraction_used": 1,
    }])
    source, estimated, converted, final = [tmp_path / f"{name}.inptk"
                                           for name in ("source", "estimated", "air", "final")]
    data.save(source)
    assert main(["estimate", str(source), "--format", "saved", "--method", method,
                 "--out", str(estimated), "--json"]) == 0
    capsys.readouterr()
    assert main(["convert", str(estimated), "--output-basis", "sampled_air",
                 "--out", str(converted), "--json"]) == 0
    capsys.readouterr()
    air = inptk.load(converted).tables["cumulative"].to_dataframe()
    assert air.concentration.isna().tolist() == [True, False, False, False, True]
    assert main(["table", str(converted), "--table", "cumulative",
                 "--no-history", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["table"]["rows"][0]["concentration"] == {"$nonfinite": "nan"}
    assert main(["finalize", str(converted), "--out", str(final), "--json"]) == 0
    capsys.readouterr()
    final_points = inptk.load(final).tables["cumulative"].to_dataframe()
    assert final_points.temperature_C.tolist() == [-6, -7, -8]
    output = tmp_path / "spectrum.csv"
    assert main(["export-csv", str(final), "--out", str(output), "--json"]) == 0
    capsys.readouterr()
    assert pd.read_csv(output).temperature_C.tolist() == [-6, -7, -8]
    # NaN output values do not replace or discard observations used by MLE.
    pd.testing.assert_frame_equal(inptk.load(estimated).experiment.counts.to_dataframe(),
                                  data.counts.to_dataframe())
