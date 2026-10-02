"""CLI contracts for concentration combination and explicit measurement ranges."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import inptk


def run_cli(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "inptk", *map(str, arguments)],
        env=dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
            PYTHONDONTWRITEBYTECODE="1",
        ),
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.fixture
def range_source(tmp_path):
    rows, metadata = [], []
    for measurement, dilution, frozen in (
        ("001", 1, [0, 2, 5, 8, 12]),
        ("01", 10, [0, 1, 2, 3, 5]),
    ):
        metadata.append(
            {
                "measurement_id": measurement,
                "sample_id": "007",
                "dilution": dilution,
                "droplet_volume_uL": 50,
            }
        )
        for temperature, count in zip(range(-5, -10, -1), frozen, strict=True):
            rows.append(
                {
                    "measurement_id": measurement,
                    "cycle_id": "01",
                    "temperature_C": temperature,
                    "n_total": 20,
                    "n_frozen": count,
                }
            )
    counts, metadata = pd.DataFrame(rows), pd.DataFrame(metadata)
    counts.to_csv(tmp_path / "counts.csv", index=False)
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    return inptk.read_counts(counts, metadata=metadata)


@pytest.mark.parametrize("range_input", ["inline", "file"])
@pytest.mark.parametrize("method", [None, "mle", "average"])
def test_cli_temperature_ranges_match_python_and_keep_inclusive_boundaries(
    tmp_path, range_source, range_input, method
):
    ranges = {"001": {"min_C": -8, "max_C": -6}, "01": {"min_C": -9, "max_C": -7}}
    argument = json.dumps(ranges)
    if range_input == "file":
        path = tmp_path / "ranges.json"
        path.write_text(argument)
        argument = path
    output = tmp_path / "result.inptk"
    process = run_cli(
        "analyze",
        tmp_path / "counts.csv",
        "--metadata",
        tmp_path / "metadata.csv",
        *([] if method is None else ["--method", method]),
        "--temperature-ranges",
        argument,
        "--step-C",
        "1",
        "--out",
        output,
    )
    assert process.returncode == 0, process.stderr
    result = inptk.load(output)
    expected = inptk.analyze_concentration(
        range_source, method=method or "mle", step_C=1.0, temperature_ranges_C=ranges
    )
    pd.testing.assert_frame_equal(result.combined.to_dataframe(), expected.combined.to_dataframe())
    pd.testing.assert_frame_equal(result.final.to_dataframe(), expected.final.to_dataframe())
    rows = result.combined.to_dataframe().set_index("temperature_C")
    assert rows.contributor_count.to_dict() == {-5.0: 0, -6.0: 1, -7.0: 2, -8.0: 2, -9.0: 1}
    assert rows.loc[-5, "selection_status"] == "no_eligible_measurements"
    assert rows.loc[-6, "source_measurement_id"] == "001"
    assert rows.loc[-9, "source_measurement_id"] == "01"
    assert set(json.loads(rows.loc[-8, "contributing_measurement_ids"])) == {"001", "01"}
    assert result.settings["estimation_method"] == (method or "mle")
    assert result.settings["temperature_ranges_C"] == ranges
    assert "method_options" not in result.settings
    assert "dilution_method" not in result.settings
    assert len(result.experiment.counts) == 10


@pytest.mark.parametrize("ranges", ["[]", '{"1":{"max_C":-7}}', '{"001":{"min_C":-6,"max_C":-8}}'])
def test_cli_rejects_invalid_ranges_without_saving(tmp_path, range_source, ranges):
    output = tmp_path / "result.inptk"
    process = run_cli(
        "analyze",
        tmp_path / "counts.csv",
        "--metadata",
        tmp_path / "metadata.csv",
        "--temperature-ranges",
        ranges,
        "--out",
        output,
    )
    assert process.returncode == 1
    assert process.stderr.startswith("inptk:")
    assert not output.exists()


@pytest.mark.parametrize("flag,value", [("--dilution-method", "mle"), ("--method-options", "{}")])
def test_cli_rejects_removed_method_switches(tmp_path, flag, value):
    output = tmp_path / "result.inptk"
    process = run_cli("analyze", "unused.csv", flag, value, "--out", output)
    assert process.returncode == 2
    assert "unrecognized arguments" in process.stderr
    assert not output.exists()


def test_cli_does_not_offer_synthetic_window_counts_for_concentration():
    process = run_cli("analyze", "--help")
    assert process.returncode == 0
    assert "--temperature-ranges" in process.stdout
    assert "--method {mle,average}" in process.stdout
    assert "window_max_count" not in process.stdout
    assert "--dilution-method" not in process.stdout
    assert "--method-options" not in process.stdout
    assert "--water-blank-model" not in process.stdout
