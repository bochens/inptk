import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import inptk
from inptk.cli import _select_experiment


@pytest.fixture
def raw_source(tmp_path):
    rows, metadata = [], []
    for measurement, sample, dilution, total, frozen in (
        ("M0", "007", 1, 20, [0, 8, 8, 10, 14]),
        ("M1", "007", 10, 20, [0, 1, 2, 3, 5]),
        ("M2", "7", 1, 20, [0, 8, 8, 10, 14]),
        ("W0", "water0", 1, 20, [0, 0, 4, 4, 4]),
        ("W1", "water1", 1, 30, [0, 0, 6, 6, 6]),
    ):
        metadata.append(
            {
                "measurement_id": measurement,
                "sample_id": sample,
                "dilution": dilution,
                "droplet_volume_uL": 20 if measurement == "W0" else 50,
                "run_id": "R1",
            }
        )
        for cycle in ("01", "1"):
            for temperature, count in zip(range(-5, -10, -1), frozen, strict=True):
                rows.append(
                    {
                        "measurement_id": measurement,
                        "cycle_id": cycle,
                        "temperature_C": temperature,
                        "n_total": total,
                        "n_frozen": count,
                    }
                )
    mapping = {"M0": ["W0", "W1"], "M1": ["W1", "W0"], "M2": ["W0"]}
    counts, metadata = pd.DataFrame(rows), pd.DataFrame(metadata)
    counts.to_csv(tmp_path / "counts.csv", index=False)
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    (tmp_path / "water-blank-map.json").write_text(json.dumps(mapping))
    source = inptk.read_counts(counts, metadata=metadata, water_blank_map=mapping)
    source.save(tmp_path / "raw.inptk")
    return source


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


def test_selection_retains_each_associated_raw_blank_once_in_selected_cycle(raw_source):
    selected = _select_experiment(raw_source, ["007"], ["01"])
    rows = selected.counts.to_dataframe()
    assert len(rows) == 20
    assert set(rows.measurement_id) == {"M0", "M1", "W0", "W1"}
    assert set(rows.cycle_id) == {"01"}
    assert rows[rows.measurement_id.eq("W0")].shape[0] == 5
    assert rows[rows.measurement_id.eq("W1")].shape[0] == 5
    assert set(selected.measurements) == {"M0", "M1", "W0", "W1"}
    assert set(selected.samples) == {"007", "water0", "water1"}
    assert selected.water_blank_map == {"M0": ["W0", "W1"], "M1": ["W1", "W0"]}
    assert selected.source["selection"] == {"sample_id": ["007"], "cycle_id": ["01"]}
    assert len(raw_source.counts) == 50


def test_selection_keeps_only_blanks_assigned_to_the_selected_original_sample(raw_source):
    selected = _select_experiment(raw_source, ["7"], ["1"])
    assert set(selected.measurements) == {"M2", "W0"}
    assert set(selected.samples) == {"7", "water0"}
    assert selected.water_blank_map == {"M2": ["W0"]}
    assert len(selected.counts) == 10


@pytest.mark.parametrize("input_format", ["native", "saved"])
@pytest.mark.parametrize("method", ["mle", "average"])
def test_cli_raw_blank_analysis_preserves_context_without_outputting_blank_samples(
    tmp_path, raw_source, input_format, method
):
    if input_format == "native":
        arguments = [
            tmp_path / "counts.csv",
            "--metadata",
            tmp_path / "metadata.csv",
            "--water-blank-map",
            tmp_path / "water-blank-map.json",
        ]
    else:
        arguments = [tmp_path / "raw.inptk", "--format", "saved"]
    output = tmp_path / "result.inptk"
    process = run_cli(
        "analyze",
        *arguments,
        "--method",
        method,
        "--sample",
        "007",
        "--cycle",
        "01",
        "--temperature-ranges",
        '{"M0":{"max_C":-6},"M1":{"min_C":-8}}',
        "--out",
        output,
    )
    assert process.returncode == 0, process.stderr
    result = inptk.load(output)
    assert result.experiment.water_blank_map == {"M0": ["W0", "W1"], "M1": ["W1", "W0"]}
    assert len(result.experiment.counts) == 20
    assert result.experiment.measurements["W0"].droplet_volume_uL == 20
    assert result.experiment.measurements["W1"].droplet_volume_uL == 50
    assert set(result.final_candidates.to_dataframe().sample_id) == {"007"}
    assert set(result.final_candidates.to_dataframe().group_id) == {"007/R1/01"}
    assert set(result.per_dilution.to_dataframe().measurement_id) == {"M0", "M1"}
    expected = inptk.analyze_concentration(
        _select_experiment(raw_source, ["007"], ["01"]),
        method=method,
        temperature_ranges_C={"M0": {"max_C": -6}, "M1": {"min_C": -8}},
    )
    pd.testing.assert_frame_equal(result.final.to_dataframe(), expected.final.to_dataframe())
    pd.testing.assert_frame_equal(
        result.final_candidates.to_dataframe(), expected.final_candidates.to_dataframe()
    )


@pytest.mark.parametrize(
    "input_format,message",
    [
        ("saved", "already contains its water-blank mapping"),
        ("icescopy", "requires raw sample and blank counts in --format native"),
    ],
)
def test_cli_water_blank_map_requires_raw_native_input(tmp_path, raw_source, input_format, message):
    output = tmp_path / "result.inptk"
    process = run_cli(
        "analyze",
        tmp_path / "raw.inptk",
        "--format",
        input_format,
        "--water-blank-map",
        tmp_path / "water-blank-map.json",
        "--out",
        output,
    )
    assert process.returncode == 1
    assert message in process.stderr
    assert not output.exists()


def test_cli_rejects_blank_only_parent_sample_selection(tmp_path, raw_source):
    output = tmp_path / "result.inptk"
    process = run_cli(
        "analyze",
        tmp_path / "raw.inptk",
        "--format",
        "saved",
        "--sample",
        "water0",
        "--out",
        output,
    )
    assert process.returncode == 1
    assert "Water-blank parent samples cannot be selected" in process.stderr
    assert not output.exists()


def test_cli_disabling_water_correction_preserves_blank_context(tmp_path, raw_source):
    output = tmp_path / "uncorrected.inptk"
    process = run_cli(
        "analyze",
        tmp_path / "raw.inptk",
        "--format",
        "saved",
        "--sample",
        "007",
        "--cycle",
        "01",
        "--no-water-blank-correction",
        "--out",
        output,
    )
    assert process.returncode == 0, process.stderr
    result = inptk.load(output)
    assert result.experiment.water_blank_map == {"M0": ["W0", "W1"], "M1": ["W1", "W0"]}
    assert len(result.experiment.counts) == 20
    assert result.experiment.measurements["W0"].droplet_volume_uL == 20
    assert result.experiment.measurements["W1"].droplet_volume_uL == 50
    assert result.settings["water_blank_correction"] is False
    assert result.settings["water_blank_correction_applied"] is False
    assert set(result.per_dilution.to_dataframe().measurement_id) == {"M0", "M1"}
    expected = inptk.analyze_concentration(
        _select_experiment(raw_source, ["007"], ["01"]), water_blank_correction=False
    )
    pd.testing.assert_frame_equal(result.final.to_dataframe(), expected.final.to_dataframe())


@pytest.fixture
def cross_run_source(tmp_path):
    rows, metadata = [], []
    for measurement, sample, run, dilution, volume, total, frozen in (
        ("M0", "007", "R/1", 1, 50, 20, [4, 8, 12]),
        ("M1", "007", "R2", 2, 50, 20, [2, 4, 8]),
        ("W0", "water0", "R/1", 1, 50, 10, [1, 2, 3]),
        ("W1", "water1", "R2", 1, 100, 12, [0, 1, 2]),
    ):
        metadata.append(
            {
                "measurement_id": measurement,
                "sample_id": sample,
                "run_id": run,
                "dilution": dilution,
                "droplet_volume_uL": volume,
            }
        )
        temperatures = [-5.1, -6.1, -7.1] if run == "R/1" else [-5.1, -6.2, -7.2]
        for cycle in ("01", "02"):
            for temperature, count in zip(temperatures, frozen, strict=True):
                rows.append(
                    {
                        "measurement_id": measurement,
                        "cycle_id": cycle,
                        "temperature_C": temperature,
                        "n_total": total,
                        "n_frozen": count,
                    }
                )
    source = inptk.read_counts(
        pd.DataFrame(rows),
        metadata=metadata,
        water_blank_map={"M0": ["W0"], "M1": ["W1"]},
    )
    source.save(tmp_path / "cross-run.inptk")
    return source


def cross_run_groups():
    return {
        "mixed": [
            {"measurement_id": "M0", "cycle_id": "01"},
            {"measurement_id": "M1", "cycle_id": "02"},
        ]
    }


@pytest.mark.parametrize("method", ["mle", "average"])
@pytest.mark.parametrize("group_input", ["inline", "file"])
def test_cli_cross_run_groups_preserve_own_blanks_and_export_saved_tables(
    tmp_path, cross_run_source, method, group_input
):
    groups = cross_run_groups()
    group_argument = json.dumps(groups)
    if group_input == "file":
        group_argument = tmp_path / "groups.json"
        group_argument.write_text(json.dumps(groups))
    output = tmp_path / "result.inptk"
    process = run_cli(
        "analyze",
        tmp_path / "cross-run.inptk",
        "--format",
        "saved",
        "--method",
        method,
        "--combination-groups",
        group_argument,
        "--output-step-C",
        "0.05",
        "--output-method",
        "interpolate",
        "--out",
        output,
    )
    assert process.returncode == 0, process.stderr
    result = inptk.load(output)
    expected = inptk.analyze_concentration(
        cross_run_source,
        method=method,
        combination_groups=groups,
        output_step_C=0.05,
        output_method="interpolate",
    )
    pd.testing.assert_frame_equal(result.final.to_dataframe(), expected.final.to_dataframe())
    pd.testing.assert_frame_equal(
        result.resampled.to_dataframe(), expected.resampled.to_dataframe()
    )
    pd.testing.assert_frame_equal(
        result.experiment.counts.to_dataframe(), cross_run_source.counts.to_dataframe()
    )
    assert result.experiment.water_blank_map == {"M0": ["W0"], "M1": ["W1"]}
    assert result.settings["combination_groups"] == {
        "mixed": {
            "sample_id": "007",
            "members": [
                {"measurement_id": "M0", "run_id": "R/1", "cycle_id": "01"},
                {"measurement_id": "M1", "run_id": "R2", "cycle_id": "02"},
            ],
        }
    }
    combined = result.combined.to_dataframe()
    assert set(combined.group_id) == {"mixed"}
    assert "run_id" not in combined and "cycle_id" not in combined
    contributing = combined.loc[combined.contributor_count.eq(2)]
    assert not contributing.empty
    for observations in contributing.source_observations.map(json.loads):
        blank_sources = [item for item in observations if item["role"] == "blank"]
        assert len(blank_sources) == 2
        assert {
            (item["measurement_id"], item["run_id"], item["cycle_id"]) for item in blank_sources
        } == {("W0", "R/1", "01"), ("W1", "R2", "02")}
    payload = json.loads((output / "analysis.json").read_text())
    assert payload["format_version"] == 2
    for table_name in ("final", "resampled"):
        destination = tmp_path / f"{table_name}.csv"
        table_arguments = [] if table_name == "final" else ["--table", "resampled"]
        exported = run_cli("export-csv", output, *table_arguments, "--out", destination)
        assert exported.returncode == 0, exported.stderr
        assert destination.read_text() == getattr(result, table_name).to_dataframe().to_csv(
            index=False
        )


def test_cli_rejects_group_member_removed_by_cycle_filter(tmp_path, cross_run_source):
    output = tmp_path / "result.inptk"
    process = run_cli(
        "analyze",
        tmp_path / "cross-run.inptk",
        "--format",
        "saved",
        "--cycle",
        "01",
        "--combination-groups",
        json.dumps(cross_run_groups()),
        "--out",
        output,
    )
    assert process.returncode == 1
    assert "Unknown or absent combination member" in process.stderr
    assert not output.exists()


def test_cli_saved_analysis_rerun_uses_original_counts_and_current_settings(
    tmp_path, cross_run_source
):
    prior = inptk.analyze_concentration(
        cross_run_source,
        method="average",
        combination_groups=cross_run_groups(),
        output_step_C=0.05,
        output_method="interpolate",
    )
    previous_path = tmp_path / "previous.inptk"
    prior.save(previous_path)
    original_payload = (previous_path / "analysis.json").read_bytes()
    output = tmp_path / "rerun.inptk"
    process = run_cli("analyze", previous_path, "--format", "saved", "--out", output)
    assert process.returncode == 0, process.stderr
    result = inptk.load(output)
    expected = inptk.analyze_concentration(cross_run_source)
    pd.testing.assert_frame_equal(result.final.to_dataframe(), expected.final.to_dataframe())
    assert result.settings["estimation_method"] == "mle"
    assert result.resampled is None
    assert set(result.combined.to_dataframe().group_id) == {
        "007/R%2F1/01",
        "007/R%2F1/02",
        "007/R2/01",
        "007/R2/02",
    }
    assert (previous_path / "analysis.json").read_bytes() == original_payload
