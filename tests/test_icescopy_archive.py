"""Archive and CSV inputs share counts, metadata, and explicit analysis choices."""

import json
from unittest.mock import patch
from zipfile import ZipFile

import pandas as pd
import pytest

import inptk
from inptk.cli import main

CSV = (
    "cycle,time_s,temperature_C,picture,A number total,A number frozen,"
    "B number total,B number frozen\n"
    "01,0,-5.23,img0,20,2,10,0\n01,1,-5.8,,20,7,10,1\n01,2,-6.4,img2,20,12,10,2\n"
)
RECORDS = [
    {"sample_id": "11", "sample_name": "A", "sample_long_name": "water blank",
     "dilution": "13", "well_volume_uL": "50", "sample_type": "air"},
    {"sample_id": "22", "sample_name": "B", "sample_long_name": "ordinary sample",
     "dilution": "1", "well_volume_uL": "100", "sample_type": "other"},
]


def archive(path, *, records=RECORDS, required=False, csv=CSV):
    with ZipFile(path, "w") as bundle:
        bundle.writestr("freeze_count_timeseries.csv", csv)
        bundle.writestr("session.json", json.dumps({
            "session_metadata": {"project_name": "test"},
            "freeze_count_timeseries_summary": {
                "sample_column_metadata": records, "analysis_required": required,
            },
        }))
        bundle.writestr("grayscale.csv", "must not read this")
        bundle.writestr("../unrelated.txt", "must not extract this")
    return path


def test_archive_equals_csv_preserves_all_rows_and_never_infers_blank_roles(tmp_path):
    path = archive(tmp_path / "input.icescopy")
    csv = tmp_path / "counts.csv"
    csv.write_text("# sample_name,A,B\n# dilution,13,1\n# well_volume_uL,50,100\n"
                   "# sample_type,air,other\n" + CSV)
    original = path.read_bytes()
    unassigned = inptk.read_icescopy(path)
    assert unassigned.water_blank_map == {}
    options = {"sample_map": {"A": "S", "B": "W"}, "water_blank_map": {"A": ["B"]}}
    actual = inptk.read_icescopy(path, **options)
    expected = inptk.read_icescopy(csv, **options)
    pd.testing.assert_frame_equal(actual.counts.to_dataframe(), expected.counts.to_dataframe())
    assert actual.samples == expected.samples and actual.measurements == expected.measurements
    assert actual.measurements["A"].droplet_volume_uL == 50
    assert actual.measurements["B"].droplet_volume_uL == 100
    assert actual.water_blank_map == {"A": ["B"]}
    assert len(actual.counts) == 6 and actual.counts.to_dataframe().picture_id.isna().sum() == 2
    assert actual.counts.to_dataframe().cycle_id.unique().tolist() == ["01"]
    assert actual.source["archive_member"] == "freeze_count_timeseries.csv"
    assert actual.source["metadata_member"] == "session.json"
    pd.testing.assert_frame_equal(
        inptk.analyze_concentration(actual, temperature_step_C=.5).to_dataframe(),
        inptk.analyze_concentration(expected, temperature_step_C=.5).to_dataframe(),
    )
    assert path.read_bytes() == original
    assert sorted(p.name for p in tmp_path.iterdir()) == ["counts.csv", "input.icescopy"]


def test_archive_reads_only_counts_and_session_and_allows_explicit_metadata_overrides(tmp_path):
    path = archive(tmp_path / "input.icescopy")
    original_read = ZipFile.read
    requested = []

    def read(bundle, name, *args, **kwargs):
        requested.append(name)
        return original_read(bundle, name, *args, **kwargs)

    with patch.object(ZipFile, "read", read):
        result = inptk.read_icescopy(path, metadata={"A": {"dilution": 7}})
    assert requested == ["freeze_count_timeseries.csv", "session.json"]
    assert result.measurements["A"].dilution == 7
    assert result.measurements["A"].droplet_volume_uL == 50


def test_archive_without_physical_metadata_can_preview_but_requires_metadata_to_analyze(tmp_path):
    path = archive(tmp_path / "input.icescopy", records=[])
    observations = inptk.read_observations(path, format="icescopy")
    assert len(observations) == 6
    assert observations.history[-1]["archive_member"] == "freeze_count_timeseries.csv"
    with pytest.raises(ValueError, match="dilution|well_volume"):
        inptk.read_icescopy(path)
    supplied = inptk.read_icescopy(path, metadata={"well_volume_uL": 50, "dilution": 1})
    assert len(supplied.counts) == 6


def test_archive_cli_preview_and_analyze_share_the_python_reader(tmp_path, capsys):
    path = archive(tmp_path / "input.icescopy")
    assert main(["preview", str(path), "--format", "icescopy", "--json"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["status"] == "ok" and preview["suspension_metadata"]["valid"]
    out = tmp_path / "result.inptk"
    assert main(["analyze", str(path), "--format", "icescopy", "--water-blank-map",
                 '{"A":["B"]}', "--temperature-step-C", ".5", "--out", str(out), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    expected = inptk.analyze_concentration(
        inptk.read_icescopy(path, water_blank_map={"A": ["B"]}), temperature_step_C=.5,
    )
    pd.testing.assert_frame_equal(inptk.load(out).to_dataframe(), expected.to_dataframe())


@pytest.mark.parametrize("problem, message", [
    ("bad_zip", "Cannot read Icescopy archive"),
    ("missing_csv", "exactly one freeze_count_timeseries.csv"),
    ("duplicate_csv", "exactly one freeze_count_timeseries.csv"),
    ("duplicate_metadata", "Duplicate Icescopy count-column metadata"),
    ("stale", "requiring analysis"),
])
def test_invalid_archives_fail_with_actionable_cli_errors(tmp_path, capsys, problem, message):
    path = tmp_path / "bad.icescopy"
    if problem == "bad_zip":
        path.write_text("not a ZIP")
    elif problem == "missing_csv":
        with ZipFile(path, "w") as bundle:
            bundle.writestr("freeze.csv", "not count data")
    elif problem == "duplicate_csv":
        archive(path)
        with pytest.warns(UserWarning, match="Duplicate name"), ZipFile(path, "a") as bundle:
            bundle.writestr("freeze_count_timeseries.csv", CSV)
    else:
        archive(path, required=problem == "stale",
                records=RECORDS + RECORDS[:1] if problem == "duplicate_metadata" else RECORDS)
    assert main(["preview", str(path), "--format", "icescopy", "--json"]) == 1
    reply = json.loads(capsys.readouterr().out)
    assert reply["status"] == "error"
    assert message in reply["error"]["message"]
