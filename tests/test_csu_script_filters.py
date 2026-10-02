from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import inptk

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "csu_inp_processing.py"


@pytest.fixture
def csu():
    spec = importlib.util.spec_from_file_location("csu_inp_processing", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def saved_analysis(
    tmp_path,
    *,
    groups=(("A", "R1", "01"),),
    policy="stop_at_decrease",
    basis="sampled_air",
    dilutions=(1,),
    method="mle",
):
    counts, metadata = [], {}
    for sample, run, cycle in groups:
        for dilution in dilutions:
            measurement = f"{sample}-{run}-{dilution}"
            metadata[measurement] = {
                "measurement_id": measurement,
                "sample_id": sample,
                "run_id": run,
                "sample_type": "air",
                "dilution": dilution,
                "droplet_volume_uL": 50,
                "suspension_volume_mL": 5,
                "air_volume_L": 100,
                "filter_fraction_used": 1,
            }
            for temperature, frozen in zip((-5, -6, -7, -8), (0, 8, 4, 12), strict=True):
                counts.append(
                    {
                        "measurement_id": measurement,
                        "cycle_id": cycle,
                        "temperature_C": temperature,
                        "n_frozen": frozen,
                        "n_total": 32,
                    }
                )
    source = inptk.read_counts(pd.DataFrame(counts), metadata=list(metadata.values()))
    result = inptk.analyze_concentration(
        source, output_basis=basis, decrease_policy=policy, step_C=1, method=method
    )
    path = tmp_path / "result.inptk"
    result.save(path)
    return path, result


def read_export(path):
    lines = path.read_text().splitlines()
    start = next(i for i, value in enumerate(lines) if value.startswith("degC,"))
    return lines[: start - 1], pd.read_csv(path, skiprows=start)


@pytest.mark.parametrize(
    "policy, temperatures", [("stop_at_decrease", [-5, -6]), ("skip_decreases", [-5, -6, -8])]
)
@pytest.mark.parametrize("method", ["mle", "average"])
def test_export_preserves_saved_final_rows_values_and_error_widths(
    tmp_path, csu, policy, temperatures, method
):
    path, result = saved_analysis(tmp_path, policy=policy, method=method)
    original = (path / "analysis.json").read_bytes()
    output = tmp_path / "export.csv"
    assert csu.main([str(path), "--out", str(output), "--allow-missing-header"]) == 0
    header, actual = read_export(output)
    expected = result.final.to_dataframe()
    assert actual.columns.tolist() == ["degC", "dilution", "INPS_L", "lower_CI", "upper_CI"]
    assert actual.degC.tolist() == temperatures
    np.testing.assert_allclose(actual.INPS_L, expected.concentration)
    np.testing.assert_allclose(actual.lower_CI, expected.lower_error)
    np.testing.assert_allclose(actual.upper_CI, expected.upper_error)
    assert actual.dilution.tolist() == [1] * len(actual)
    assert [line.split(" = ")[0] for line in header] == list(csu.HEADER_ORDER)
    assert "vol_air_filt = 100.0" in header
    assert "vol_susp = 5.0" in header
    assert (path / "analysis.json").read_bytes() == original


def test_combined_points_do_not_invent_a_single_dilution_factor(tmp_path, csu):
    path, result = saved_analysis(tmp_path, dilutions=(1, 10))
    output = tmp_path / "combined.csv"
    csu.main([str(path), "--out", str(output), "--allow-missing-header"])
    _, actual = read_export(output)
    assert actual.dilution.isna().all()
    np.testing.assert_allclose(actual.INPS_L, result.final.to_dataframe().concentration)


def test_multiple_groups_require_exact_sample_run_and_cycle_selection(tmp_path, csu):
    path, result = saved_analysis(
        tmp_path,
        groups=(("01", "001", "01"), ("01", "001", "1"), ("01", "1", "01"), ("1", "001", "01")),
    )
    output = tmp_path / "selected.csv"
    base = [str(path), "--out", str(output), "--allow-missing-header"]
    for selection in ([], ["--sample", "01"], ["--sample", "01", "--run", "001"]):
        with pytest.raises(ValueError, match="Multiple saved sample/run/cycle groups"):
            csu.main(base + selection)
        assert not output.exists()
    assert csu.main(base + ["--sample", "01", "--run", "001", "--cycle", "01"]) == 0
    _, actual = read_export(output)
    expected = result.final.select(sample_id="01", run_id="001", cycle_id="01").to_dataframe()
    np.testing.assert_allclose(actual.INPS_L, expected.concentration)
    assert len(actual) == len(expected)


@pytest.mark.parametrize("selection", [["--sample", "A "], ["--run", "R"], ["--cycle", "1"]])
def test_unknown_selection_is_not_inferred(tmp_path, csu, selection):
    path, _ = saved_analysis(tmp_path)
    output = tmp_path / "missing.csv"
    with pytest.raises(ValueError, match="No saved final rows match"):
        csu.main([str(path), "--out", str(output), "--allow-missing-header", *selection])
    assert not output.exists()


def test_export_never_overwrites_existing_files_or_adds_to_saved_analysis(tmp_path, csu):
    path, _ = saved_analysis(tmp_path)
    output = tmp_path / "existing.csv"
    output.write_text("existing user output\n")
    with pytest.raises(FileExistsError):
        csu.main([str(path), "--out", str(output), "--allow-missing-header"])
    assert output.read_text() == "existing user output\n"
    original = (path / "analysis.json").read_bytes()
    with pytest.raises(ValueError, match="outside the saved analysis folder"):
        csu.main([str(path), "--out", str(path / "extra.csv"), "--allow-missing-header"])
    assert not (path / "extra.csv").exists()
    assert (path / "analysis.json").read_bytes() == original


def test_experiment_and_suspension_result_are_not_silently_processed(tmp_path, csu):
    path, result = saved_analysis(tmp_path, basis="suspension")
    output = tmp_path / "invalid.csv"
    with pytest.raises(ValueError, match="saved final sampled_air spectrum"):
        csu.main([str(path), "--out", str(output), "--allow-missing-header"])
    experiment_path = tmp_path / "experiment.inptk"
    result.experiment.save(experiment_path)
    with pytest.raises(TypeError, match="saved AnalysisResult"):
        csu.main([str(experiment_path), "--out", str(output), "--allow-missing-header"])
    assert not output.exists()


def test_headers_require_explicit_missing_fields_and_preserve_recorded_metadata(tmp_path, csu):
    path, _ = saved_analysis(tmp_path)
    output = tmp_path / "header.csv"
    with pytest.raises(ValueError, match="Missing CSU header fields"):
        csu.main([str(path), "--out", str(output)])
    assert not output.exists()
    headers = {
        "site": "site A",
        "start_time": "start",
        "end_time": "end",
        "filter_color": "white",
        "treatment": "none",
        "notes": "saved analysis",
        "user": "Bo",
        "IS": "IS1",
    }
    options = [part for key, value in headers.items() for part in ("--header", f"{key}={value}")]
    assert csu.main([str(path), "--out", str(output), *options]) == 0
    actual, _ = read_export(output)
    assert "site = site A" in actual
    assert "sample_type = air" in actual
    assert "proportion_filter_used = 1.0" in actual


@pytest.mark.parametrize(
    "header", ["vol_air_filt=200", "sample_type=soil", "vol_susp=1", "proportion_filter_used=0.5"]
)
def test_headers_cannot_relabel_the_saved_normalization(tmp_path, csu, header):
    path, _ = saved_analysis(tmp_path)
    output = tmp_path / "invalid.csv"
    with pytest.raises(ValueError, match="conflicts with saved sample metadata"):
        csu.main([str(path), "--out", str(output), "--allow-missing-header", "--header", header])
    assert not output.exists()


def test_command_line_reports_actionable_export_error_without_traceback(tmp_path):
    path, _ = saved_analysis(tmp_path, basis="suspension")
    process = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(path),
            "--out",
            str(tmp_path / "failed.csv"),
            "--allow-missing-header",
        ],
        env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 2
    assert "run inptk analyze --output-basis sampled_air first" in process.stderr
    assert "Traceback" not in process.stderr
