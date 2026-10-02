import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from analysis_checks import all_points, retained, sampled

import inptk


@pytest.mark.parametrize("sample_id", ["007", "NA"])
def test_cli_matches_python_and_preserves_cycle_ids(tmp_path, sample_id):
    counts = pd.DataFrame(
        {
            "measurement_id": ["001"] * 6,
            "cycle_id": ["01"] * 3 + ["02"] * 3,
            "temperature_C": [-5, -6, -7] * 2,
            "n_total": [20] * 6,
            "n_frozen": [0, 3, 8, 0, 5, 9],
        }
    )
    metadata = pd.DataFrame(
        [{"measurement_id": "001", "sample_id": sample_id, "dilution": 1, "droplet_volume_uL": 50}]
    )
    counts.to_csv(tmp_path / "counts.csv", index=False)
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "inptk",
            "analyze",
            str(tmp_path / "counts.csv"),
            "--metadata",
            str(tmp_path / "metadata.csv"),
            "--out",
            str(tmp_path / "result"),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    restored = inptk.load(tmp_path / "result")
    expected = inptk.analyze_concentration(inptk.read_counts(counts, metadata=metadata))
    pd.testing.assert_frame_equal(
        retained(restored).to_dataframe(), retained(expected).to_dataframe()
    )
    assert set(retained(restored).to_dataframe().sample_id) == {sample_id}
    assert set(retained(restored).to_dataframe().curve_id) == {
        f"{sample_id}/1/01",
        f"{sample_id}/1/02",
    }
    assert (
        restored.settings["observation_processing"]
        == "native; latest warmer alignment only where required"
    )
    assert restored.settings["output_step_C"] is None
    assert sampled(restored) is None
    assert restored.settings["decrease_policy"] == "stop_at_decrease"
    payload = json.loads((tmp_path / "result" / "analysis.json").read_text())
    assert payload["toolkit_version"] == inptk.__version__


def test_no_cycle_policy_or_compatibility_package_in_source():
    from inptk.cli import build_parser

    parser = build_parser()
    assert "cycle-policy" not in parser.format_help()
    assert not (Path(__file__).resolve().parents[1] / "src" / "ufolaf" / "__init__.py").exists()


@pytest.fixture
def selection_source(tmp_path):
    rows, records = [], []
    for index, sample_id in enumerate(("007", "7", "NA")):
        measurement_id = f"Sample_{index}"
        records.append(
            {
                "sample_id": sample_id,
                "measurement_id": measurement_id,
                "dilution": 1,
                "droplet_volume_uL": 50,
            }
        )
        for cycle_id in ("01", "1", "02"):
            for temperature, frozen in zip((-5, -6, -7), (0, 2 + index, 8 + index), strict=True):
                rows.append(
                    {
                        "measurement_id": measurement_id,
                        "cycle_id": cycle_id,
                        "temperature_C": temperature,
                        "n_total": 20,
                        "n_frozen": frozen,
                    }
                )
    counts, metadata = pd.DataFrame(rows), pd.DataFrame(records)
    counts.to_csv(tmp_path / "counts.csv", index=False)
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    source = inptk.read_counts(counts, metadata=metadata)
    source.save(tmp_path / "source.inptk")
    return source


def _run_cli(*arguments):
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


@pytest.mark.parametrize("input_format", ["native", "saved", "icescopy"])
def test_cli_selects_exact_sample_and_cycle_and_saves_selection(
    tmp_path, selection_source, input_format
):
    if input_format == "native":
        arguments = [tmp_path / "counts.csv", "--metadata", tmp_path / "metadata.csv"]
    elif input_format == "saved":
        arguments = [tmp_path / "source.inptk", "--format", "saved"]
    else:
        rows = selection_source.counts.to_dataframe()
        wide = rows.pivot(
            index=["cycle_id", "temperature_C"],
            columns="measurement_id",
            values=["n_total", "n_frozen"],
        )
        wide.columns = [
            f"{measurement} number {'total' if value == 'n_total' else 'frozen'}"
            for value, measurement in wide.columns
        ]
        path = tmp_path / "icescopy.csv"
        path.write_text(
            "# well_volume_uL: 50\n# sample_name,Sample_0,Sample_1,Sample_2\n# dilution,1,1,1\n"
            + wide.reset_index()
            .sort_values(["cycle_id", "temperature_C"], ascending=[True, False])
            .rename(columns={"cycle_id": "cycle"})
            .to_csv(index=False)
        )
        mapping = {key: item.sample_id for key, item in selection_source.measurements.items()}
        (tmp_path / "sample-map.json").write_text(json.dumps(mapping))
        arguments = [path, "--format", "icescopy", "--sample-map", tmp_path / "sample-map.json"]
    output = tmp_path / "selected.inptk"
    process = _run_cli("analyze", *arguments, "--sample", "007", "--cycle", "01", "--out", output)
    assert process.returncode == 0, process.stderr
    result = inptk.load(output)
    assert set(result.experiment.samples) == {"007"}
    assert set(result.experiment.measurements) == {"Sample_0"}
    assert len(result.experiment.counts) == 3
    expected_selection = {"sample_id": ["007"], "cycle_id": ["01"]}
    assert result.experiment.source["selection"] == expected_selection
    assert result.experiment.counts.history[-1] == {"operation": "select", **expected_selection}
    expected = retained(inptk.analyze_concentration(selection_source)).select(curve_id="007/1/01")
    pd.testing.assert_frame_equal(retained(result).to_dataframe(), expected.to_dataframe())
    assert set(retained(result).to_dataframe().curve_id) == {"007/1/01"}
    assert set(retained(result).to_dataframe().sample_id) == {"007"}
    original = inptk.load(tmp_path / "source.inptk")
    assert len(original.counts) == 27
    assert "selection" not in original.source
    assert original.counts.history == []


def test_cli_repeatable_selection_keeps_requested_labels_and_deduplicates_provenance(
    tmp_path, selection_source
):
    output = tmp_path / "selected.inptk"
    process = _run_cli(
        "analyze",
        tmp_path / "source.inptk",
        "--format",
        "saved",
        "--sample",
        "007",
        "--sample",
        "NA",
        "--sample",
        "007",
        "--cycle",
        "02",
        "--cycle",
        "01",
        "--cycle",
        "02",
        "--out",
        output,
    )
    assert process.returncode == 0, process.stderr
    result = inptk.load(output)
    assert set(result.experiment.samples) == {"007", "NA"}
    assert set(result.experiment.measurements) == {"Sample_0", "Sample_2"}
    assert len(result.experiment.counts) == 12
    assert set(retained(result).to_dataframe().curve_id) == {
        "007/1/01",
        "007/1/02",
        "NA/1/01",
        "NA/1/02",
    }
    assert result.experiment.source["selection"] == {
        "sample_id": ["007", "NA"],
        "cycle_id": ["02", "01"],
    }


@pytest.mark.parametrize(
    "flag,value,message",
    [
        ("--sample", "Sample_0", "Unknown sample_id selection"),
        ("--sample", "missing", "Unknown sample_id selection"),
        ("--cycle", "001", "Unknown cycle_id selection"),
    ],
)
def test_cli_unknown_selection_fails_without_creating_output(
    tmp_path, selection_source, flag, value, message
):
    output = tmp_path / "selected.inptk"
    process = _run_cli(
        "analyze", tmp_path / "source.inptk", "--format", "saved", flag, value, "--out", output
    )
    assert process.returncode == 1
    assert message in process.stderr
    assert not output.exists()


@pytest.mark.parametrize("policy", [None, "stop_at_decrease", "skip_decreases"])
def test_cli_final_decrease_policy_matches_python_and_keeps_candidates(tmp_path, policy):
    counts = pd.DataFrame(
        {
            "measurement_id": ["001"] * 5,
            "cycle_id": ["01"] * 5,
            "temperature_C": [-5, -6, -7, -8, -9],
            "n_total": [20, 20, 16, 16, 16],
            "n_frozen": [0, 8, 4, 10, 12],
        }
    )
    metadata = pd.DataFrame(
        [{"measurement_id": "001", "sample_id": "007", "dilution": 1, "droplet_volume_uL": 50}]
    )
    counts.to_csv(tmp_path / "counts.csv", index=False)
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    extra = [] if policy is None else ["--decrease-policy", policy]
    output = tmp_path / "result.inptk"
    process = _run_cli(
        "analyze",
        tmp_path / "counts.csv",
        "--metadata",
        tmp_path / "metadata.csv",
        "--out",
        output,
        *extra,
    )
    assert process.returncode == 0, process.stderr
    actual = inptk.load(output)
    effective_policy = policy or "stop_at_decrease"
    expected = inptk.analyze_concentration(
        inptk.read_counts(counts, metadata=metadata), decrease_policy=effective_policy
    )
    pd.testing.assert_frame_equal(
        retained(actual).to_dataframe(), retained(expected).to_dataframe()
    )
    pd.testing.assert_frame_equal(
        all_points(actual).to_dataframe(), all_points(expected).to_dataframe()
    )
    assert len(all_points(actual)) == 5
    assert retained(actual).to_dataframe().temperature_C.tolist() == (
        [-5, -6] if effective_policy == "stop_at_decrease" else [-5, -6, -8, -9]
    )
    assert actual.settings["decrease_policy"] == effective_policy
