"""CSV metadata preambles must not consume text inside table values."""

import json
from zipfile import ZipFile

import pandas as pd
import pytest

import inptk


@pytest.mark.parametrize("sample_id", ["Site #1", "Site\n# second line"])
@pytest.mark.parametrize("sample_last", [False, True])
def test_native_metadata_preserves_hashes_and_quoted_newlines(tmp_path, sample_id, sample_last):
    record = {"measurement_id": "A", "sample_id": sample_id,
              "dilution": 10, "droplet_volume_uL": 50}
    columns = list(record)
    if sample_last:
        columns.remove("sample_id")
        columns.append("sample_id")
    path = tmp_path / "metadata.csv"
    path.write_text("# project: Plate #3\n" + pd.DataFrame([record])[columns].to_csv(index=False),
                    encoding="utf-8-sig")
    experiment = inptk.read_counts(
        [{"measurement_id": "A", "temperature_C": -5, "n_total": 10, "n_frozen": 2}],
        metadata=path,
    )
    assert experiment.measurements["A"].sample_id == sample_id
    assert experiment.measurements["A"].dilution == 10
    assert experiment.measurements["A"].droplet_volume_uL == 50


def test_native_counts_preserve_hash_in_measurement_name(tmp_path):
    path = tmp_path / "counts.csv"
    path.write_text("# project: Plate #3\nmeasurement_id,temperature_C,n_total,n_frozen\n"
                    "Assay #3,-5,10,2\n", encoding="utf-8")
    experiment = inptk.read_counts(path, metadata=[{
        "measurement_id": "Assay #3", "sample_id": "Site #1",
        "dilution": 1, "droplet_volume_uL": 50,
    }])
    row = experiment.counts.to_dataframe().iloc[0]
    assert row.measurement_id == "Assay #3" and row.n_frozen == 2 and row.n_total == 10


ICESCOPY_CSV = (
    "# project_name: Plate #3\n# sample_name,Site #1\n"
    "# dilution,10\n# well_volume_uL,50\n"
    "cycle,time_s,temperature_C,Site #1 number total,Site #1 number frozen\n"
    "1,0,-5,10,1\n1,1,-6,10,2\n"
)


@pytest.mark.parametrize("archive", [False, True])
def test_icescopy_sample_hash_and_leading_metadata_survive(tmp_path, archive):
    path = tmp_path / ("input.icescopy" if archive else "input.csv")
    if archive:
        with ZipFile(path, "w") as bundle:
            bundle.writestr("freeze_count_timeseries.csv", ICESCOPY_CSV)
            bundle.writestr("session.json", "{}")
    else:
        path.write_text(ICESCOPY_CSV, encoding="utf-8")
    experiment = inptk.read_icescopy(path)
    assert experiment.measurements["Site #1"].dilution == 10
    assert experiment.measurements["Site #1"].droplet_volume_uL == 50
    assert experiment.counts.to_dataframe().n_frozen.tolist() == [1, 2]
    result = inptk.analyze_concentration(experiment, method="average")
    assert list(result.curves) == ["Site%20%231/1/1"]


def test_native_json_preserves_utf8_identifiers(tmp_path):
    name = "Site é 冰"
    counts = [{"measurement_id": name, "temperature_C": -5, "n_total": 10, "n_frozen": 2}]
    metadata = [{"measurement_id": name, "sample_id": name, "dilution": 1,
                 "droplet_volume_uL": 50}]
    for filename, records in (("counts.json", counts), ("metadata.json", metadata)):
        (tmp_path / filename).write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    experiment = inptk.read_counts(tmp_path / "counts.json", metadata=tmp_path / "metadata.json")
    assert experiment.measurements[name].sample_id == name
    assert experiment.counts.to_dataframe().measurement_id.tolist() == [name]
