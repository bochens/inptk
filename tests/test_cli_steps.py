"""Saved calculation stages and the persistent client use the same scientific API."""

import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

import inptk
from inptk import cli
from inptk.cli_steps import fraction_step, transform_step
from inptk.cli_store import ResultStore


@pytest.fixture
def source(tmp_path):
    rows, metadata = [], []
    for name, dilution, volume, total, frozen in (
        ("001", 1, 50, 20, [1, 3, 6, 8, 14]),
        ("002", 10, 50, 20, [0, 1, 2, 4, 7]),
        ("009", 1, 25, 12, [0, 0, 1, 2, 3]),
    ):
        metadata.append(
            {
                "measurement_id": name,
                "sample_id": "water" if name == "009" else "007",
                "run_id": "run 01",
                "dilution": dilution,
                "droplet_volume_uL": volume,
                "sample_type": "air",
                "filter_fraction_used": 1,
                "air_volume_L": 1000,
                "suspension_volume_mL": 10,
            }
        )
        for i, n in enumerate(frozen):
            rows.append(
                {
                    "measurement_id": name,
                    "cycle_id": "01",
                    "time_s": i,
                    "observation_id": f"frame-{i}",
                    "temperature_C": -5.0 - i,
                    "n_total": total,
                    "n_frozen": n,
                }
            )
    pd.DataFrame(rows).to_csv(tmp_path / "counts.csv", index=False)
    pd.DataFrame(metadata).to_csv(tmp_path / "metadata.csv", index=False)
    experiment = inptk.read_counts(
        pd.DataFrame(rows),
        metadata=metadata,
        water_blank_map={"001": ["009"], "002": ["009"]},
    )
    experiment.save(tmp_path / "source.inptk")
    return experiment


def invoke(capsys, *args, store=None, ok=True):
    status = cli.main(["--json", *map(str, args)], store=store)
    captured = capsys.readouterr()
    reply = json.loads(captured.out)
    assert status == (0 if ok else 1), captured.err + str(reply)
    assert reply["status"] == ("ok" if ok else "error")
    return reply


@pytest.mark.parametrize("method", ["average", "mle"])
def test_saved_steps_equal_python_and_whole_workflow(tmp_path, source, capsys, method):
    fractions, estimated, air, final = [
        tmp_path / f"{name}.inptk" for name in ("fractions", "estimated", "air", "final")
    ]
    invoke(capsys, "fractions", tmp_path / "source.inptk", "--format", "saved", "--out", fractions)
    invoke(
        capsys,
        "estimate",
        fractions,
        "--format",
        "saved",
        "--method",
        method,
        "--temperature-step-C",
        "1",
        "--out",
        estimated,
    )
    step = inptk.load(estimated)
    expected = inptk.estimate_concentration(
        inptk.frozen_fraction(source),
        experiment=source,
        method=method,
        temperature_step_C=1,
    )
    pd.testing.assert_frame_equal(step.tables["cumulative"].to_dataframe(), expected.to_dataframe())
    assert step.tables["cumulative"].history == expected.history
    assert "used_in_final" not in expected.columns
    assert step.experiment.water_blank_map == source.water_blank_map
    invoke(capsys, "convert", estimated, "--output-basis", "sampled_air", "--out", air)
    invoke(capsys, "finalize", air, "--decrease-policy", "skip_decreases", "--out", final)
    result = inptk.load(final)
    whole = inptk.analyze_concentration(
        source,
        method=method,
        temperature_step_C=1,
        output_basis="sampled_air",
        decrease_policy="skip_decreases",
    )
    for name in ("cumulative", "excluded"):
        pd.testing.assert_frame_equal(
            result.tables[name].to_dataframe(), whole.to_dataframe(table=name)
        )
    reply = invoke(capsys, "table", final, "--table", "cumulative")
    assert reply["table"]["rows"]
    assert reply["table"]["dtypes"]
    out = tmp_path / "export.csv"
    invoke(capsys, "export-csv", final, "--table", "counts", "--out", out)
    assert len(pd.read_csv(out)) == len(source.counts)
    rejected = invoke(capsys, "finalize", air, "--out", final, ok=False)
    assert rejected["error"]["code"] == "output_exists"


def test_fraction_metadata_can_be_completed_later(tmp_path, source, capsys):
    path = tmp_path / "fractions.inptk"
    invoke(capsys, "fractions", tmp_path / "counts.csv", "--out", path)
    saved = inptk.load(path)
    assert saved.experiment is None
    assert set(saved.tables["frozen_fraction"].to_dataframe().cycle_id) == {"01"}
    reply = invoke(capsys, "preview", path, "--format", "saved")
    assert not reply["suspension_metadata"]["valid"]
    rejected = invoke(
        capsys, "estimate", path, "--format", "saved", "--out", tmp_path / "bad.inptk", ok=False
    )
    assert "--metadata" in rejected["error"]["message"]
    # Metadata supplies parent IDs, physical parameters and the actual run later.
    # Run IDs in original rows are stable: supply consistent run metadata.
    metadata = pd.read_csv(
        tmp_path / "metadata.csv", dtype={"measurement_id": str, "sample_id": str}
    )
    metadata["run_id"] = "1"
    metadata.to_csv(tmp_path / "complete.csv", index=False)
    out = tmp_path / "estimated.inptk"
    invoke(
        capsys,
        "estimate",
        path,
        "--format",
        "saved",
        "--method",
        "average",
        "--metadata",
        tmp_path / "complete.csv",
        "--water-blank-map",
        '{"001":["009"],"002":["009"]}',
        "--out",
        out,
    )
    result = inptk.load(out)
    assert result.experiment.measurements["001"].sample_id == "007"
    assert result.experiment.water_blank_map == source.water_blank_map
    assert set(result.tables["cumulative"].to_dataframe().sample_id) == {"007"}


def test_steps_reuse_fractions_and_transforms_do_not_estimate(source, capsys, monkeypatch):
    store = ResultStore(memory=True)
    fractions = fraction_step(source.counts, experiment=source)
    store.save(fractions, "@fractions")

    def forbidden(*args, **kwargs):
        pytest.fail("This step repeated a completed calculation")

    monkeypatch.setattr(cli, "frozen_fraction", forbidden)
    invoke(
        capsys,
        "estimate",
        "@fractions",
        "--format",
        "saved",
        "--method",
        "average",
        "--individual",
        "--out",
        "@estimated",
        store=store,
    )
    assert store.load("@estimated").tables["frozen_fraction"] is fractions.tables["frozen_fraction"]
    monkeypatch.setattr(cli, "estimate_concentration", forbidden)
    monkeypatch.setattr(cli, "cumulative_spectrum", forbidden)
    invoke(capsys, "differentiate", "@estimated", "--out", "@differential", store=store)
    invoke(
        capsys,
        "convert",
        "@estimated",
        "--output-basis",
        "sampled_air",
        "--out",
        "@air",
        store=store,
    )
    invoke(capsys, "finalize", "@air", "--out", "@final", store=store)
    assert store.load("@final").experiment is source
    assert len(store.load("@differential").tables["differential"]) == 8
    rejected = invoke(
        capsys,
        "convert",
        "@air",
        "--output-basis",
        "sampled_air",
        "--out",
        "@twice",
        store=store,
        ok=False,
    )
    assert "already converted" in rejected["error"]["message"]
    assert not store.exists("@twice")


def test_estimate_keeps_decreases_and_differentiation_respects_exclusions(capsys):
    source = inptk.read_counts(
        pd.DataFrame(
            {
                "sample_id": ["a"] * 5,
                "temperature_C": [-5, -6, -7, -8, -9],
                "n_frozen": [1, 5, 4, 7, 9],
                "n_total": [20] * 5,
            }
        ),
        metadata=[{"sample_id": "a", "dilution": 1, "droplet_volume_uL": 50}],
    )
    store = ResultStore(memory=True)
    store.save(source, "@source")
    invoke(
        capsys,
        "estimate",
        "@source",
        "--format",
        "saved",
        "--method",
        "average",
        "--individual",
        "--out",
        "@estimated",
        store=store,
    )
    table = store.load("@estimated").tables["cumulative"]
    assert np.any(np.diff(table.to_dataframe().concentration) < 0)
    invoke(
        capsys,
        "finalize",
        "@estimated",
        "--decrease-policy",
        "skip_decreases",
        "--out",
        "@skip",
        store=store,
    )
    final = store.load("@skip")
    assert len(final.tables["excluded"]) == 1
    invoke(capsys, "differentiate", "@skip", "--out", "@diff", store=store)
    diff = store.load("@diff").tables["differential"].to_dataframe()
    assert len(diff) == 2  # Never bridge the excluded -7 C state.
    invoke(
        capsys,
        "finalize",
        "@skip",
        "--decrease-policy",
        "stop_at_decrease",
        "--out",
        "@stop",
        store=store,
    )
    assert len(store.load("@stop").tables["cumulative"]) == 2
    pd.testing.assert_frame_equal(
        table.to_dataframe(), store.load("@estimated").tables["cumulative"].to_dataframe()
    )


def test_persistent_process_protocol_and_error_recovery(tmp_path, source):
    def request(i, *args):
        return json.dumps({"id": i, "args": list(map(str, args))})

    lines = [
        request(1, "fractions", tmp_path / "source.inptk", "--format", "saved", "--out", "@f"),
        request(2, "estimate", "@f", "--format", "saved", "--method", "average", "--out", "@e"),
        request(3, "table", "@e", "--table", "cumulative"),
        request(4, "finalize", "@e", "--out", "@e"),
        '{"id":5,"release":["@e"]}',
        request(6, "table", "@e"),
        "not json",
        request(8, "--json", "serve"),
        request(9, "estimate", "@f", "--format", "saved", "--method", "average", "--out", "@e"),
        request(10, "table", "@e"),
        request(11, "--help"),
        request(12, "nonsense"),
        request(13, "capabilities"),
    ]
    process = subprocess.run(
        [sys.executable, "-m", "inptk", "serve"],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        env=dict(os.environ),
        check=False,
        timeout=60,
    )
    assert process.returncode == 0, process.stderr
    replies = [json.loads(line) for line in process.stdout.splitlines()]
    assert len(replies) == len(lines)
    assert [reply["id"] for reply in replies] == [1, 2, 3, 4, 5, 6, None, 8, 9, 10, 11, 12, 13]
    assert [reply["status"] for reply in replies] == [
        "ok",
        "ok",
        "ok",
        "error",
        "ok",
        "error",
        "error",
        "error",
        "ok",
        "ok",
        "error",
        "error",
        "ok",
    ]
    assert replies[3]["error"]["code"] == "output_exists"
    assert replies[2]["table"]["rows"]
    assert "cumulative" in replies[9]["tables"]
    assert not list(tmp_path.glob("@*"))
    assert not process.stderr


def test_terminal_rejects_memory_references(capsys):
    reply = invoke(capsys, "table", "@data", ok=False)
    assert "serve" in reply["error"]["message"]


def test_named_curve_tables_and_step_roundtrip(tmp_path, source, capsys):
    curves = {"original": {"inputs": ["001", "002"], "cycle": "01"}}
    result = inptk.analyze_concentration(source, method="average", curves=curves)
    result.save(tmp_path / "analysis.inptk")
    reply = invoke(
        capsys, "table", tmp_path / "analysis.inptk", "--table", "cumulative", "--curve", "original"
    )
    assert {row["curve_id"] for row in reply["table"]["rows"]} == {"original"}
    step = transform_step(result, "convert", basis="sampled_air")
    step.save(tmp_path / "step.inptk")
    saved = inptk.load(tmp_path / "step.inptk")
    for name, table in step.tables.items():
        pd.testing.assert_frame_equal(saved.tables[name].to_dataframe(), table.to_dataframe())
        assert saved.tables[name].history == table.history
    rejected = invoke(
        capsys, "differentiate", tmp_path / "step.inptk", "--out", tmp_path / "bad.inptk", ok=False
    )
    assert "--individual" in rejected["error"]["message"]


@pytest.mark.parametrize("flag", ["--hel", "--vers", "-hh"])
def test_client_help_alias_cannot_exit_or_print_prose(monkeypatch, capsys, flag):
    from io import StringIO

    requests = [{"id": 1, "args": [flag]}, {"id": 2, "args": ["capabilities"]}]
    monkeypatch.setattr(sys, "stdin", StringIO("\n".join(map(json.dumps, requests)) + "\n"))
    assert cli.main(["serve"]) == 0
    output = capsys.readouterr()
    replies = [json.loads(line) for line in output.out.splitlines()]
    assert [reply["status"] for reply in replies] == ["error", "ok"]
    assert [reply["id"] for reply in replies] == [1, 2]
    assert not output.err


def test_save_cached_result_without_recalculation(tmp_path, source, capsys, monkeypatch):
    store = ResultStore(memory=True)
    store.save(fraction_step(source.counts, experiment=source), "@fractions")

    def forbidden(*args, **kwargs):
        pytest.fail("Saving must not recalculate")

    monkeypatch.setattr(cli, "frozen_fraction", forbidden)
    out = tmp_path / "saved.inptk"
    invoke(capsys, "save", "@fractions", "--out", out, store=store)
    restored = inptk.load(out)
    pd.testing.assert_frame_equal(
        restored.tables["frozen_fraction"].to_dataframe(),
        store.load("@fractions").tables["frozen_fraction"].to_dataframe(),
    )
    assert restored.experiment.water_blank_map == source.water_blank_map
    reply = invoke(capsys, "save", "@fractions", "--out", out, store=store, ok=False)
    assert reply["error"]["code"] == "output_exists"
