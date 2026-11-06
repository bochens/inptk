"""Optional removal of warming observations before any analysis."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import inptk
from inptk.cli import main

DATA = Path(__file__).parent / "data"
COUNTS = DATA / "warmup_to_14C_counts.csv"
METADATA = DATA / "warmup_to_14C_metadata.csv"


def counts(temperatures, frozen, *, cycles=None, total=20):
    cycles = cycles or ["0"] * len(temperatures)
    return inptk.CountsTable(
        pd.DataFrame(
            {
                "run_id": "1",
                "sample_id": "S",
                "measurement_id": "S",
                "cycle_id": cycles,
                "temperature_C": temperatures,
                "time_s": np.arange(len(temperatures), dtype=float),
                "n_total": total,
                "n_frozen": frozen,
            }
        )
    )


def kept(table):
    return table.to_dataframe().temperature_C.tolist()


def step(table):
    return table.history[-1]


def test_cooling_jitter_within_the_hysteresis_is_kept():
    temperatures = [-1.0, -2.0, -1.97, -3.0, -2.94, -4.0, -3.81]
    table = inptk.drop_warming(counts(temperatures, [0] * 7))
    assert kept(table) == temperatures
    assert step(table)["omitted_observations"] == []
    assert table.warnings == []


def test_a_real_warm_up_is_dropped_and_listed_with_its_reason():
    table = inptk.drop_warming(counts([-1, -5, -10, -4, 2, 14], [0, 3, 8, 8, 8, 8]))
    assert kept(table) == [-1, -5, -10]
    record = step(table)
    assert record["rows_dropped"] == [
        {"measurement_id": "S", "sample_id": "S", "run_id": "1", "cycle_id": "0",
         "rows_dropped": 3}
    ]
    assert [row["temperature_C"] for row in record["omitted_observations"]] == [-4, 2, 14]
    assert {row["reason"] for row in record["omitted_observations"]} == {"warming"}
    assert table.warnings == [
        "drop_warming left out 3 warming rows (hysteresis 0.2 C): S run 1 cycle 0: 3"
    ]


def test_drift_before_the_cycle_goes_below_zero_is_kept():
    temperatures = [20.0, 18.0, 21.0, 5.0, 6.0, 0.0, 0.5, -1.0, -3.0]
    assert kept(inptk.drop_warming(counts(temperatures, [0] * 9))) == temperatures


def test_a_second_cycle_restarts_the_coldest_temperature_so_far():
    table = inptk.drop_warming(counts(
        [-10, -20, 5, 10, -5, -15, -8],
        [2, 9, 9, 0, 1, 6, 6],
        cycles=["1", "1", "1", "2", "2", "2", "2"],
    ))
    frame = table.to_dataframe()
    assert frame.cycle_id.tolist() == ["1", "1", "2", "2", "2"]
    assert kept(table) == [-10, -20, 10, -5, -15]
    assert [item["rows_dropped"] for item in step(table)["rows_dropped"]] == [1, 1]


def test_freezes_on_dropped_rows_carry_forward_to_the_next_kept_row():
    table = inptk.drop_warming(counts([-1, -5, -3, -2, -6, -7], [0, 2, 4, 5, 3, 6]))
    frame = table.to_dataframe()
    assert frame.temperature_C.tolist() == [-1, -5, -6, -7]
    assert frame.n_frozen.tolist() == [0, 2, 5, 6]
    assert step(table)["carried_frozen_counts"] == [
        {"measurement_id": "S", "run_id": "1", "cycle_id": "0", "observation_id": "4",
         "n_frozen_observed": 3, "n_frozen_used": 5}
    ]


def test_freezes_after_the_last_kept_row_produce_a_warning():
    table = inptk.drop_warming(counts([-1, -5, -10, -4, 2], [0, 3, 8, 9, 11]))
    assert table.to_dataframe().n_frozen.tolist() == [0, 3, 8]
    assert table.warnings[-1] == (
        "S: 3 freeze event(s) in run 1 cycle 0 fall on warming rows after its last kept row "
        "and are not counted"
    )


def test_rows_follow_acquisition_time_and_keep_their_original_order():
    source = counts([-10, -1, 5, -5], [5, 0, 5, 2])
    frame = source.to_dataframe()
    frame["time_s"] = [2.0, 0.0, 3.0, 1.0]
    table = inptk.drop_warming(inptk.CountsTable(frame))
    assert kept(table) == [-10, -1, -5]


def test_the_hysteresis_is_adjustable_and_validated():
    source = counts([-5, -4.7, -4.9], [0, 0, 0])
    assert kept(inptk.drop_warming(source)) == [-5, -4.9]
    assert kept(inptk.drop_warming(source, warming_hysteresis_C=0.5)) == [-5, -4.7, -4.9]
    with pytest.raises(ValueError, match="warming_hysteresis_C"):
        inptk.drop_warming(source, warming_hysteresis_C=-0.1)
    experiment = inptk.read_counts(COUNTS, metadata=METADATA)
    with pytest.raises(TypeError, match="drop_warming"):
        inptk.analyze_concentration(experiment, drop_warming="yes")


def test_mle_on_a_run_that_warms_to_14_C_needs_the_option():
    experiment = inptk.read_counts(COUNTS, metadata=METADATA)
    off = inptk.analyze_concentration(experiment)
    on = inptk.analyze_concentration(experiment, drop_warming=True)
    off_values = off.curves["S/1/0"].cumulative.to_dataframe().set_index("temperature_C")
    on_values = on.curves["S/1/0"].cumulative.to_dataframe().set_index("temperature_C")
    # The warm-up rows are the latest rows at or warmer than every target, so
    # without the option the fully frozen state reaches most temperatures.
    assert off_values.concentration.loc[[-10, -14, -18]].isna().all()
    assert np.isfinite(on_values.concentration.loc[[-10, -12, -14, -16, -18]]).all()
    assert on_values.concentration.loc[-18] > on_values.concentration.loc[-10]
    assert on.settings["drop_warming"] is True
    assert on.settings["warming_hysteresis_C"] == 0.2
    assert "drop_warming" not in off.settings
    assert on.warnings[0].startswith("drop_warming left out 6 warming rows")
    # The saved result keeps every original row; the analyzed rows are recorded.
    assert len(on.counts) == len(experiment.counts) == 18
    assert len(on.frozen_fraction) == 12
    drop = next(s for s in on.frozen_fraction.history if s["operation"] == "drop_warming")
    assert len(drop["omitted_observations"]) == 6


def test_range_suggestions_record_the_option():
    experiment = inptk.read_counts(COUNTS, metadata=METADATA)
    proposal = inptk.suggest_temperature_ranges(experiment, drop_warming=True)
    assert proposal.settings["drop_warming"] is True
    assert proposal.settings["drop_warming_warnings"][0].startswith("drop_warming left out 6")
    assert "drop_warming" not in inptk.suggest_temperature_ranges(experiment).settings


def run_cli(capsys, *arguments):
    code = main([*arguments, "--json"])
    return code, json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("command", ["preview", "fractions", "analyze", "estimate",
                                     "suggest-ranges"])
def test_cli_commands_offer_the_option(command):
    flags = {
        item["name"]: item
        for item in inptk.cli._capabilities(inptk.cli.build_parser())["commands"][command][
            "options"
        ]
    }
    assert flags["drop_warming"]["default"] is False
    assert flags["warming_hysteresis_C"]["default"] == 0.2


def test_cli_analyze_reports_dropped_rows(tmp_path, capsys):
    base = [str(COUNTS), "--metadata", str(METADATA)]
    code, off = run_cli(capsys, "analyze", *base, "--out", str(tmp_path / "off.inptk"))
    assert code == 0 and "drop_warming" not in off["settings"]
    code, on = run_cli(
        capsys, "analyze", *base, "--out", str(tmp_path / "on.inptk"), "--drop-warming",
        "--warming-hysteresis-C", "0.3",
    )
    assert code == 0
    assert on["settings"]["drop_warming"] is True
    assert on["settings"]["warming_hysteresis_C"] == 0.3
    assert on["warnings"][0] == (
        "drop_warming left out 6 warming rows (hysteresis 0.3 C): S run 1 cycle 0: 6"
    )
    assert on["observation_tables"]["counts"]["row_count"] == 18
    assert on["observation_tables"]["frozen_fraction"]["row_count"] == 12


def test_cli_preview_and_estimate_apply_the_option(tmp_path, capsys):
    base = [str(COUNTS), "--metadata", str(METADATA), "--drop-warming"]
    code, preview = run_cli(capsys, "preview", *base)
    assert code == 0
    assert preview["measurements"][0]["observation_count"] == 12
    assert preview["warnings"][0].startswith("drop_warming left out 6")
    code, estimate = run_cli(
        capsys, "estimate", *base, "--individual", "--out", str(tmp_path / "e.inptk")
    )
    assert code == 0
    assert estimate["tables"]["frozen_fraction"]["row_count"] == 12
    assert estimate["warnings"][0].startswith("drop_warming left out 6")
    code, error = run_cli(
        capsys, "analyze", *base, "--warming-hysteresis-C", "-1",
        "--out", str(tmp_path / "x.inptk"),
    )
    assert code == 1 and "warming_hysteresis_C" in error["error"]["message"]
