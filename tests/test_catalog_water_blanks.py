"""Catalog control types select blanks independently of names and normalization."""

import csv
import io
import json
from dataclasses import asdict, replace
from zipfile import ZipFile

import pandas as pd
import pytest

import inptk
from inptk._engine.models import SampleMetadata
from inptk.cli import main


@pytest.fixture(params=["csv", "archive"])
def catalog(tmp_path, request):
    records = [
        {"sample_id": "11", "sample_name": "A", "sample_long_name": "water blank",
         "sample_type": "air", "dilution": 13, "well_volume_uL": 50,
         "air_volume_L": 1000, "suspension_volume_mL": 10, "filter_fraction_used": 1},
        {"sample_id": "22", "sample_name": "B", "sample_long_name": "ordinary sample",
         "sample_type": " Water Blank ", "dilution": "inactive", "well_volume_uL": 100,
         "air_volume_L": "inactive", "suspension_volume_mL": -1, "filter_fraction_used": 5,
         "dry_mass_g": "inactive"},
        {"sample_id": "33", "sample_name": "C", "sample_type": "water blank",
         "dilution": 0, "well_volume_uL": 25},
    ]
    counts = pd.DataFrame({
        "cycle": ["01"] * 4, "time_s": [0, 1, 2, 3],
        "temperature_C": [-5.23, -5.8, -6.4, -6.9],
        "picture": ["frame0", "", "frame2", "frame3"],
        "A number total": [20] * 4, "A number frozen": [0, 4, 8, 12],
        "B number total": [10] * 4, "B number frozen": [0, 0, 1, 2],
        "C number total": [8] * 4, "C number frozen": [0, 1, 1, 3],
    })
    header = io.StringIO()
    writer = csv.writer(header)
    for field in dict.fromkeys(key for record in records for key in record):
        writer.writerow([f"# {field}", *(record.get(field, "") for record in records)])
    text = header.getvalue() + counts.to_csv(index=False)
    if request.param == "archive":
        path = tmp_path / "counts.icescopy"
        with ZipFile(path, "w") as bundle:
            bundle.writestr("freeze_count_timeseries.csv", text)
            bundle.writestr("session.json", json.dumps({
                "freeze_count_timeseries_summary": {"sample_column_metadata": records},
            }))
    else:
        path = tmp_path / "counts.csv"
        path.write_text(text)
    return path, counts


def metadata_records(experiment):
    return [
        {**asdict(experiment.samples[item.sample_id]), **asdict(item)}
        for item in experiment.measurements.values()
    ]


def test_catalog_default_retains_counts_roles_and_actual_volumes(catalog):
    path, original_counts = catalog
    original_file = path.read_bytes()
    experiment = inptk.read_icescopy(path)
    assert experiment.water_blank_map == {"A": ["B", "C"]}
    assert experiment.water_blank_ids == {"B", "C"}
    assert experiment.samples["A"].sample_type == "air"  # The long name is not a role.
    for name, volume in (("B", 100), ("C", 25)):
        sample = experiment.samples[name]
        assert sample.sample_type == "water blank"
        assert sample.air_volume_L is None and sample.suspension_volume_mL is None
        assert sample.filter_fraction_used is None and sample.dry_mass_g is None
        assert experiment.measurements[name].dilution == 1
        assert experiment.measurements[name].droplet_volume_uL == volume
    frame = experiment.counts.to_dataframe()
    assert len(frame) == 12 and frame.picture_id.isna().sum() == 3
    assert frame.cycle_id.unique().tolist() == ["01"]
    for name in ("A", "B", "C"):
        assert frame.loc[frame.measurement_id.eq(name), "n_frozen"].tolist() == (
            original_counts[f"{name} number frozen"].tolist()
        )
    assert path.read_bytes() == original_file


@pytest.mark.parametrize("method", ["average", "mle"])
@pytest.mark.parametrize("mapping", [None, {"A": ["C"]}, {}])
def test_catalog_maps_and_empty_override_keep_controls_out_of_curves(catalog, method, mapping):
    path, _ = catalog
    experiment = inptk.read_icescopy(path, water_blank_map=mapping)
    assert experiment.water_blank_map == ({"A": ["B", "C"]} if mapping is None else mapping)
    # Compare with the same selected physical inputs supplied through native metadata.
    selected = {"A"} | {name for names in experiment.water_blank_map.values() for name in names}
    records = [item for item in metadata_records(experiment) if item["measurement_id"] in selected]
    for item in records:
        if item["sample_type"] == "water blank":
            item["sample_type"] = "other"
    expected_input = inptk.read_counts(
        experiment.counts.to_dataframe().query("measurement_id in @selected"),
        metadata=records, water_blank_map=experiment.water_blank_map,
    )
    settings = {"method": method, "temperature_step_C": .5, "output_basis": "sampled_air",
                "differential": False}
    expected = inptk.analyze_concentration(expected_input, **settings)
    actual = inptk.analyze_concentration(experiment, **settings)
    pd.testing.assert_frame_equal(actual.to_dataframe(), expected.to_dataframe())
    assert {source["measurement_id"] for curve in actual.curves.values()
            for source in curve.sources} == {"A"}
    disabled = inptk.analyze_concentration(experiment, water_blank_correction=False, **settings)
    assert {source["measurement_id"] for curve in disabled.curves.values()
            for source in curve.sources} == {"A"}


@pytest.mark.parametrize("method", ["average", "mle"])
def test_cli_preview_and_analyze_use_the_catalog_defaults(catalog, capsys, tmp_path, method):
    path, _ = catalog
    assert main(["preview", str(path), "--format", "icescopy", "--json"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["water_blank_map"] == {"A": ["B", "C"]}
    assert preview["suspension_metadata"]["valid"]
    assert [item["sample_type"] for item in preview["measurement_metadata"]] == (
        ["air", "water blank", "water blank"]
    )
    output = tmp_path / "result.inptk"
    assert main(["analyze", str(path), "--format", "icescopy", "--method", method,
                 "--sample", "A", "--out", str(output), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "ok"
    result = inptk.load(output)
    expected = inptk.analyze_concentration(inptk.read_icescopy(path), method=method)
    pd.testing.assert_frame_equal(result.to_dataframe(), expected.to_dataframe())
    assert result.experiment.water_blank_map == {"A": ["B", "C"]}
    assert result.experiment.samples["B"].sample_type == "water blank"


def test_native_typed_controls_are_assigned_within_each_run(catalog, tmp_path):
    experiment = inptk.read_icescopy(catalog[0])
    counts, metadata = [], []
    for run in ("R1", "R2"):
        frame = experiment.counts.to_dataframe()
        frame["measurement_id"] += run
        frame["run_id"] = run
        counts.append(frame)
        for record in metadata_records(experiment):
            record["measurement_id"] += run
            record["run_id"] = run
            if record["sample_type"] == "water blank":
                record.update(dilution="inactive", air_volume_L="inactive")
            metadata.append(record)
    source = inptk.read_counts(pd.concat(counts), metadata=metadata)
    assert source.water_blank_map == {"AR1": ["BR1", "CR1"], "AR2": ["BR2", "CR2"]}
    off = inptk.read_counts(pd.concat(counts), metadata=metadata, water_blank_map={})
    off.save(tmp_path / "off.inptk")
    restored = inptk.load(tmp_path / "off.inptk")
    assert restored.water_blank_map == {}
    assert restored.water_blank_ids == {"BR1", "CR1", "BR2", "CR2"}
    with pytest.raises(ValueError, match="cannot also be sample"):
        replace(source, water_blank_map={"BR1": ["CR1"], "AR1": ["CR1"], "AR2": ["BR2"]})


@pytest.mark.parametrize("volume", [0, -1, "invalid"])
def test_catalog_blank_volume_still_requires_a_real_positive_value(catalog, volume):
    with pytest.raises(ValueError):
        inptk.read_icescopy(catalog[0], metadata={"B": {"well_volume_uL": volume}})


def test_engine_catalog_normalization_retains_raw_fields():
    raw = {"sample_type": "water blank", "dilution": "inactive"}
    sample = SampleMetadata(sample_type=" Water Blank ", dilution="inactive",
                            well_volume_uL=25, air_volume_L="inactive",
                            raw_sample_metadata=raw)
    assert sample.sample_type == "water blank" and sample.dilution == 1
    assert sample.well_volume_uL == 25 and sample.air_volume_L is None
    assert sample.raw_sample_metadata == raw
    with pytest.raises(ValueError):
        SampleMetadata(sample_type="air", dilution="inactive", well_volume_uL=25)


def test_persistent_client_import_supports_typed_blanks_and_empty_override(catalog, monkeypatch,
                                                                        capsys):
    experiment = inptk.read_icescopy(catalog[0])
    payload = {"counts": experiment.counts.to_dataframe().to_dict("records"),
               "metadata": metadata_records(experiment)}
    requests = [
        {"id": 1, "import": {**payload, "out": "@auto"}},
        {"id": 2, "args": ["preview", "@auto", "--format", "saved"]},
        {"id": 3, "import": {**payload, "out": "@off", "water_blank_map": {}}},
        {"id": 4, "args": ["preview", "@off", "--format", "saved"]},
        {"id": 5, "args": ["analyze", "@off", "--format", "saved", "--method", "average",
                           "--out", "@result"]},
    ]
    monkeypatch.setattr("sys.stdin", io.StringIO("\n".join(map(json.dumps, requests)) + "\n"))
    assert main(["serve"]) == 0
    replies = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["status"] for item in replies] == ["ok"] * 5
    assert replies[1]["water_blank_map"] == {"A": ["B", "C"]}
    assert replies[3]["water_blank_map"] == {}
