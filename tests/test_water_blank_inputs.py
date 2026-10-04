import json
from dataclasses import replace

import pandas as pd
import pytest

import inptk


def raw_inputs():
    counts, metadata = [], []
    for measurement, sample, dilution, frozen in (
        ("007", "A", 1, (1, 8)),
        ("7", "A", 10, (0, 3)),
        ("B", "water", 1, (0, 2)),
    ):
        metadata.append(
            {
                "measurement_id": measurement,
                "sample_id": sample,
                "run_id": "R1",
                "dilution": dilution,
                "droplet_volume_uL": 50,
            }
        )
        for cycle in ("01", "1"):
            for temperature, number in zip((-5, -6), frozen, strict=True):
                counts.append(
                    {
                        "measurement_id": measurement,
                        "cycle_id": cycle,
                        "temperature_C": temperature,
                        "n_total": 20,
                        "n_frozen": number,
                    }
                )
    return pd.DataFrame(counts), pd.DataFrame(metadata)


def test_native_raw_water_blank_import_preserves_observations_and_exact_names(tmp_path):
    counts, metadata = raw_inputs()
    counts.to_csv(tmp_path / "counts.csv", index=False)
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    mapping = {"007": ["B"], "7": ["B"]}
    source = inptk.read_counts(
        tmp_path / "counts.csv", metadata=tmp_path / "metadata.csv", water_blank_map=mapping
    )
    actual = source.counts.to_dataframe()
    pd.testing.assert_frame_equal(actual[counts.columns], counts)
    assert source.water_blank_map == mapping
    assert set(source.measurements) == {"007", "7", "B"}
    assert set(actual.cycle_id) == {"01", "1"}
    mapping["007"].append("changed")
    assert source.water_blank_map["007"] == ["B"]
    source.save(tmp_path / "source.inptk")
    restored = inptk.load(tmp_path / "source.inptk")
    assert restored.water_blank_map == {"007": ["B"], "7": ["B"]}
    pd.testing.assert_frame_equal(restored.counts.to_dataframe(), actual)
    assert restored.samples == source.samples
    assert restored.measurements == source.measurements


@pytest.mark.parametrize(
    "mapping,match",
    [
        (["007", "B"], "must map measurement names"),
        ({"007": None, "7": ["B"]}, "non-empty lists"),
        ({"007": "", "7": ["B"]}, "non-empty lists"),
        ({"007": "  ", "7": ["B"]}, "non-empty lists"),
        ({"007": "B", "7": ["B"]}, "non-empty lists"),
        ({"007": [], "7": ["B"]}, "non-empty lists"),
        ({"007": ["B", "B"], "7": ["B"]}, "Duplicate water-blank names"),
        ({"007": [None], "7": ["B"]}, "non-empty measurement names"),
        ({7: ["B"], "007": ["B"]}, "non-empty measurement names"),
        ({"007": ["missing"], "7": ["B"]}, "Unknown measurements"),
        ({"unknown": ["B"], "007": ["B"], "7": ["B"]}, "Unknown measurements"),
        ({"007": ["B"], "7": ["B"], "B": ["B"]}, "cannot also be sample measurements"),
        ({"007": ["B"], "B": ["7"]}, "cannot also be sample measurements"),
        ({"007": ["B"]}, "assign every nonblank measurement"),
    ],
)
def test_raw_water_blank_mapping_rejects_invalid_identity_relationships(mapping, match):
    counts, metadata = raw_inputs()
    with pytest.raises((TypeError, ValueError), match=match):
        inptk.read_counts(counts, metadata=metadata, water_blank_map=mapping)


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("dilution", 10, "must have dilution=1"),
        ("run_id", "R2", "must use the same run"),
    ],
)
def test_raw_water_blank_requires_unit_dilution_and_matching_run(field, value, match):
    counts, metadata = raw_inputs()
    metadata.loc[metadata.measurement_id.eq("B"), field] = value
    with pytest.raises(ValueError, match=match):
        inptk.read_counts(counts, metadata=metadata, water_blank_map={"007": ["B"], "7": ["B"]})


def test_direct_experiment_construction_validates_water_blank_map():
    counts, metadata = raw_inputs()
    source = inptk.read_counts(counts, metadata=metadata)
    with pytest.raises(ValueError, match="assign every nonblank measurement"):
        replace(source, water_blank_map={"007": ["B"]})
    mapped = replace(source, water_blank_map={"007": ["B"], "7": ["B"]})
    assert mapped.water_blank_map == {"007": ["B"], "7": ["B"]}


def test_empty_mapping_roundtrip_and_missing_mapping_is_not_reconstructed(tmp_path):
    counts, metadata = raw_inputs()
    metadata.loc[metadata.measurement_id.eq("B"), "dilution"] = 10
    source = inptk.read_counts(counts, metadata=metadata)
    assert source.water_blank_map == {}
    source.save(tmp_path / "source.inptk")
    restored = inptk.load(tmp_path / "source.inptk")
    assert restored.water_blank_map == {}
    pd.testing.assert_frame_equal(restored.counts.to_dataframe(), source.counts.to_dataframe())
    path = tmp_path / "source.inptk" / "analysis.json"
    payload = json.loads(path.read_text())
    payload["experiment"].pop("water_blank_map")
    path.write_text(json.dumps(payload))
    with pytest.raises(KeyError, match="water_blank_map"):
        inptk.load(tmp_path / "source.inptk")


def test_icescopy_reader_accepts_explicit_raw_blank_mapping():
    data = pd.DataFrame({"temperature_C": [-5],
                         "007 number total": [20], "007 number frozen": [4],
                         "B number total": [10], "B number frozen": [1]})
    result = inptk.read_icescopy(data, water_blank_map={"007": ["B"]},
                               metadata={"well_volume_uL": 50, "dilution": 1})
    assert result.water_blank_map == {"007": ["B"]}
    assert result.counts.to_dataframe().n_total.tolist() == [20, 10]


def two_blank_inputs():
    counts, metadata = raw_inputs()
    counts = pd.concat(
        [counts, counts[counts.measurement_id.eq("B")].assign(measurement_id="C", n_total=30)],
        ignore_index=True,
    )
    metadata = pd.concat(
        [metadata, metadata[metadata.measurement_id.eq("B")].assign(measurement_id="C")],
        ignore_index=True,
    )
    return counts, metadata


def test_multiple_blank_sets_preserve_list_mapping_and_accept_different_list_order(tmp_path):
    counts, metadata = two_blank_inputs()
    mapping = {"007": ["B", "C"], "7": ["C", "B"]}
    source = inptk.read_counts(counts, metadata=metadata, water_blank_map=mapping)
    source.save(tmp_path / "raw.inptk")
    restored = inptk.load(tmp_path / "raw.inptk")
    assert restored.water_blank_map == mapping
    assert set(restored.counts.to_dataframe().measurement_id) == {"007", "7", "B", "C"}
    assert len(restored.counts) == len(counts)


def test_dilutions_of_one_original_sample_require_the_same_blank_set():
    counts, metadata = two_blank_inputs()
    with pytest.raises(ValueError, match="same set of water blanks"):
        inptk.read_counts(counts, metadata=metadata, water_blank_map={"007": ["B"], "7": ["C"]})
    metadata.loc[metadata.measurement_id.eq("7"), "sample_id"] = "other"
    source = inptk.read_counts(
        counts, metadata=metadata, water_blank_map={"007": ["B"], "7": ["C"]}
    )
    assert source.water_blank_map == {"007": ["B"], "7": ["C"]}


def test_unequal_known_sample_and_blank_volumes_are_preserved(tmp_path):
    counts, metadata = two_blank_inputs()
    volumes = {"007": 50, "7": 10, "B": 20, "C": 2}
    metadata["droplet_volume_uL"] = metadata.measurement_id.map(volumes)
    source = inptk.read_counts(
        counts, metadata=metadata, water_blank_map={"007": ["B", "C"], "7": ["B", "C"]}
    )
    source.save(tmp_path / "raw.inptk")
    restored = inptk.load(tmp_path / "raw.inptk")
    assert {key: item.droplet_volume_uL for key, item in restored.measurements.items()} == volumes
    pd.testing.assert_frame_equal(restored.counts.to_dataframe()[counts.columns], counts)


@pytest.mark.parametrize("measurement", ["007", "C"])
@pytest.mark.parametrize("volume", [None, 0, -1, float("nan"), float("inf")])
def test_each_sample_and_blank_requires_a_known_positive_volume(measurement, volume):
    counts, metadata = two_blank_inputs()
    metadata["droplet_volume_uL"] = metadata.droplet_volume_uL.astype(float)
    metadata.loc[metadata.measurement_id.eq(measurement), "droplet_volume_uL"] = volume
    with pytest.raises((TypeError, ValueError)):
        inptk.read_counts(
            counts, metadata=metadata, water_blank_map={"007": ["B", "C"], "7": ["B", "C"]}
        )
