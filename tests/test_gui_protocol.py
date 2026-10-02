"""A separate GUI process can preview, fit, and export without parsing prose."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import inptk
from inptk import cli


def call(*args):
    process = subprocess.run(
        [sys.executable, "-m", "inptk", *map(str, args)],
        env={
            **os.environ,
            "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        text=True,
        capture_output=True,
        check=False,
    )
    payload = json.loads(process.stdout)
    assert payload["protocol_version"] == 1
    assert payload["saved_format_version"] == 2
    assert payload["toolkit_version"] == inptk.__version__
    return process, payload


@pytest.fixture
def inputs(tmp_path):
    counts = pd.DataFrame(
        {
            "measurement_id": ["001"] * 4,
            "cycle_id": ["01"] * 4,
            "observation_id": ["a", "b", "c", "d"],
            "time_s": [0, 1, 2, 3],
            "temperature_C": [-5, -6, -6, -5.8],
            "n_total": [32] * 4,
            "n_frozen": [0, 4, 4, 6],
        }
    )
    path = tmp_path / "original counts 雪.csv"
    counts.to_csv(path, index=False)
    metadata = [
        {
            "measurement_id": "001",
            "sample_id": "007",
            "run_id": "run 01",
            "dilution": 1,
            "droplet_volume_uL": 50,
        }
    ]
    meta_path = tmp_path / "metadata.csv"
    pd.DataFrame(metadata).to_csv(meta_path, index=False)
    return counts, path, metadata, meta_path


def test_capabilities_describe_live_arguments_and_methods():
    process, reply = call("capabilities")
    assert process.returncode == 0
    assert not process.stderr
    assert set(reply["commands"]) == {"capabilities", "preview", "analyze", "export-csv"}
    analyze = {item["name"]: item for item in reply["commands"]["analyze"]["options"]}
    assert analyze["method"]["choices"] == ["mle", "average"]
    assert analyze["method"]["default"] == "mle"
    assert analyze["output_step_C"]["default"] is None
    assert analyze["sample"]["repeatable"]
    assert analyze["no_water_blank_correction"]["type"] == "boolean"
    assert "temperature_method" not in analyze


def test_python_observations_need_no_suspension_metadata(inputs):
    frame, _, _, _ = inputs
    before = frame.copy(deep=True)
    counts = inptk.read_observations(frame)
    fractions = inptk.frozen_fraction(counts).to_dataframe()
    pd.testing.assert_frame_equal(fractions[frame.columns], frame)
    pd.testing.assert_frame_equal(frame, before)
    assert fractions.fraction_frozen.tolist() == [0, 0.125, 0.125, 0.1875]
    assert counts.history[-1]["provisional_sample_assignments"] == ["001"]
    assert "dilution" not in fractions and "concentration" not in fractions


@pytest.mark.parametrize("placement", ["global", "command"])
def test_preview_preserves_observations_and_reports_missing_metadata(inputs, placement):
    frame, path, _, _ = inputs
    args = ["--json", "preview", path] if placement == "global" else ["preview", path, "--json"]
    process, reply = call(*args)
    assert process.returncode == 0
    assert reply["status"] == "ok" and not reply["analysis_performed"]
    observed = pd.DataFrame(reply["table"]["rows"])
    pd.testing.assert_frame_equal(observed[frame.columns], frame)
    assert reply["suspension_metadata"]["missing_fields"] == {
        "001": ["dilution", "droplet_volume_uL"]
    }
    assert not reply["suspension_metadata"]["valid"]
    assert "001" in reply["suspension_metadata"]["error"]
    assert reply["measurements"][0]["cycle_ids"] == ["01"]
    assert reply["provisional_sample_assignments"] == ["001"]


def test_partial_native_metadata_assigns_parent_without_inventing_physical_fields(inputs):
    frame, _, _, _ = inputs
    counts = inptk.read_observations(
        frame,
        metadata=[
            {
                "measurement_id": "001",
                "sample_id": "A",
                "run_id": "run 01",
                "dilution": 13,
            }
        ],
    )
    assert set(counts.to_dataframe().sample_id) == {"A"}
    assert set(counts.to_dataframe().run_id) == {"run 01"}
    assert counts.history[-1]["provisional_sample_assignments"] == []
    metadata = counts.history[-1]["measurement_metadata"][0]
    assert metadata["dilution"] == 13
    assert "droplet_volume_uL" not in metadata


def test_zero_run_identity_is_never_replaced_by_default(inputs):
    frame, _, metadata, _ = inputs
    metadata = [{**metadata[0], "run_id": 0}]
    preview = inptk.read_observations(frame, metadata=metadata)
    full = inptk.read_counts(frame, metadata=metadata)
    pd.testing.assert_frame_equal(preview.to_dataframe(), full.counts.to_dataframe())
    assert set(preview.to_dataframe().run_id) == {"0"}


def test_icescopy_preview_uses_export_metadata_and_matches_full_import(tmp_path):
    path = tmp_path / "freeze_count_timeseries.csv"
    path.write_text(
        "# sample_name,Sample_0\n# dilution,13\n"
        "cycle,temperature_C,time_s,Sample_0 number total,Sample_0 number frozen\n"
        "01,-5,0,32,0\n01,-6,1,32,4\n01,-6,2,32,4\n01,-5.8,3,32,6\n"
    )
    process, preview = call("preview", path, "--format", "icescopy", "--json")
    assert process.returncode == 0
    assert preview["suspension_metadata"]["missing_fields"] == {"Sample_0": ["droplet_volume_uL"]}
    frame = pd.DataFrame(preview["table"]["rows"])
    assert "picture_id" not in frame
    assert frame.temperature_C.tolist() == [-5, -6, -6, -5.8]
    assert set(frame.cycle_id) == {"01"}
    overrides = [{"sample_id": "Sample_0", "well_volume_uL": 50}]
    counts = inptk.read_observations(path, format="icescopy", metadata=overrides)
    strict = inptk.read_icescopy(path, metadata=overrides)
    pd.testing.assert_frame_equal(counts.to_dataframe(), strict.counts.to_dataframe())


@pytest.mark.parametrize("method", ["mle", "average"])
def test_gui_round_trip_preview_fit_saved_preview_and_export(inputs, tmp_path, method):
    frame, path, metadata, meta_path = inputs
    original = path.read_bytes()
    process, preview = call("preview", path, "--metadata", meta_path, "--json")
    assert process.returncode == 0
    assert preview["suspension_metadata"]["valid"]
    groups = {"A result": [{"measurement_id": "001", "cycle_id": "01"}]}
    output = tmp_path / "GUI preview.inptk"
    process, reply = call(
        "analyze",
        path,
        "--metadata",
        meta_path,
        "--method",
        method,
        "--combination-groups",
        json.dumps(groups),
        "--output-step-C",
        "0.5",
        "--out",
        output,
        "--json",
    )
    assert process.returncode == 0, process.stderr
    assert reply["status"] == "ok" and reply["output"] == str(output)
    result = inptk.load(reply["output"])
    expected = inptk.analyze_concentration(
        inptk.read_counts(frame, metadata=metadata),
        method=method,
        combination_groups=groups,
        output_step_C=0.5,
    )
    pd.testing.assert_frame_equal(result.final.to_dataframe(), expected.final.to_dataframe())
    assert reply["tables"]["final"]["row_count"] == len(result.final)
    assert reply["settings"]["estimation_method"] == method
    process, saved_preview = call("preview", output, "--format", "saved", "--json")
    assert process.returncode == 0
    assert saved_preview["table"]["rows"] == preview["table"]["rows"]
    csv = tmp_path / "grid.csv"
    process, exported = call("export-csv", output, "--table", "resampled", "--out", csv, "--json")
    assert process.returncode == 0 and exported["table"] == "resampled"
    assert len(pd.read_csv(csv)) == len(result.resampled)
    process, rejected = call("export-csv", output, "--out", csv, "--json")
    assert process.returncode == 1 and rejected["error"]["code"] == "output_exists"
    assert path.read_bytes() == original


@pytest.mark.parametrize(
    "args,code,status",
    [
        (["analyze", "unused", "--json"], "usage_error", 2),
        (["--json", "invalid-command"], "usage_error", 2),
        (["preview", "/does-not-exist/inptk.csv", "--json"], "file_not_found", 1),
        (
            ["analyze", "unused", "--out", "unused", "--method", "pooled", "--json"],
            "usage_error",
            2,
        ),
    ],
)
def test_gui_errors_are_one_json_object(args, code, status):
    process, reply = call(*args)
    assert process.returncode == status
    assert reply["status"] == "error"
    assert reply["error"]["code"] == code
    assert reply["error"]["message"]
    assert not process.stderr


def test_invalid_counts_do_not_become_a_successful_preview(inputs):
    frame, path, _, _ = inputs
    frame.loc[0, "n_frozen"] = 40
    frame.to_csv(path, index=False)
    process, reply = call("preview", path, "--json")
    assert process.returncode == 1
    assert reply["error"]["code"] == "invalid_input"
    assert "Counts require" in reply["error"]["message"]


@pytest.mark.parametrize(
    "error,code,status",
    [
        (KeyboardInterrupt(), "cancelled", 130),
        (RuntimeError("unexpected solver error"), "internal_error", 1),
    ],
)
def test_gui_reports_interrupt_and_unexpected_failures(
    inputs, tmp_path, monkeypatch, capsys, error, code, status
):
    _, path, _, metadata = inputs

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(cli, "analyze_concentration", fail)
    output = tmp_path / "must-not-exist"
    assert (
        cli.main(
            ["analyze", str(path), "--metadata", str(metadata), "--out", str(output), "--json"]
        )
        == status
    )
    captured = capsys.readouterr()
    reply = json.loads(captured.out)
    assert reply["error"]["code"] == code
    assert not output.exists()
    if code == "internal_error":
        assert "RuntimeError" in captured.err


def test_preview_does_not_fit(inputs, monkeypatch, capsys):
    def fail(*args, **kwargs):
        raise AssertionError("Preview must not fit counts")

    monkeypatch.setattr(cli, "analyze_concentration", fail)
    _, path, _, _ = inputs
    assert cli.main(["preview", str(path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["analysis_performed"] is False


def test_gui_accepts_inline_blank_assignments(inputs, tmp_path):
    frame, path, metadata, metadata_path = inputs
    blanks = frame.assign(measurement_id="water", n_total=20, n_frozen=[0, 0, 1, 1])
    pd.concat([frame, blanks], ignore_index=True).to_csv(path, index=False)
    metadata.append(
        {
            "measurement_id": "water",
            "sample_id": "W",
            "run_id": "run 01",
            "dilution": 1,
            "droplet_volume_uL": 25,
        }
    )
    pd.DataFrame(metadata).to_csv(metadata_path, index=False)
    output = tmp_path / "blank-corrected"
    process, reply = call(
        "analyze",
        path,
        "--metadata",
        metadata_path,
        "--water-blank-map",
        '{"001":["water"]}',
        "--out",
        output,
        "--json",
    )
    assert process.returncode == 0, process.stderr
    assert reply["settings"]["water_blank_correction_applied"]
    assert inptk.load(output).experiment.water_blank_map == {"001": ["water"]}


def test_gui_accepts_same_inline_sample_mapping_for_preview_and_analysis(tmp_path):
    path = tmp_path / "icescopy.csv"
    path.write_text(
        "# well_volume_uL: 50\n# sample_name,Sample_0\n# dilution,1\n"
        "cycle,temperature_C,Sample_0 number total,Sample_0 number frozen\n"
        "01,-5,32,0\n01,-6,32,4\n"
    )
    options = [path, "--format", "icescopy", "--sample-map", '{"Sample_0":"007"}', "--json"]
    process, preview = call("preview", *options)
    assert process.returncode == 0
    assert preview["provisional_sample_assignments"] == []
    assert preview["measurements"][0]["sample_id"] == "007"
    process, analyzed = call("analyze", *options, "--out", tmp_path / "result")
    assert process.returncode == 0
    assert set(inptk.load(analyzed["output"]).experiment.samples) == {"007"}
