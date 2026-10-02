"""Physical observations and explicitly combined curves keep distinct identities."""

import json
from dataclasses import replace

import pandas as pd
import pytest

import inptk
from inptk.methods import resolve_curves


def counts_frame():
    return pd.DataFrame(
        {
            "measurement_id": ["M"] * 4,
            "sample_id": ["S"] * 4,
            "run_id": ["R"] * 4,
            "cycle_id": ["01", "01", "01", "02"],
            "temperature_C": [-5, -5, -6, -5],
            "time_s": [2, 0, 1, 0],
            "n_total": [10] * 4,
            "n_frozen": [0, 1, 2, 0],
        }
    )


def test_observation_ids_preserve_repeated_temperatures_input_order_and_selection():
    original = counts_frame()
    counts = inptk.CountsTable(original)
    frame = counts.to_dataframe()
    assert frame.observation_id.tolist() == ["0", "1", "2", "0"]
    assert frame.temperature_C.tolist() == original.temperature_C.tolist()
    assert frame.time_s.tolist() == [2, 0, 1, 0]
    assert "observation_id" not in original
    chosen = counts.select(cycle_id="01", observation_id=["1", "2"])
    assert chosen.to_dataframe().observation_id.tolist() == ["1", "2"]
    fractions = inptk.FrozenFractionTable(frame)
    assert len(fractions) == 4
    assert fractions.to_dataframe().fraction_frozen.tolist() == [0, 0.1, 0.2, 0]


@pytest.mark.parametrize("ids", [["x", "x", "z", "x"], ["x", None, "z", "x"], ["x", " ", "z", "x"]])
def test_invalid_explicit_observation_ids_are_rejected(ids):
    with pytest.raises(ValueError, match="observation_id|Observation IDs"):
        inptk.CountsTable(counts_frame().assign(observation_id=ids))


def test_native_csv_preserves_explicit_observation_labels(tmp_path):
    counts = counts_frame().assign(observation_id=["001", "010", "100", "001"])
    path = tmp_path / "counts.csv"
    counts.to_csv(path, index=False)
    source = inptk.read_counts(
        path,
        metadata=[
            {
                "measurement_id": "M",
                "sample_id": "S",
                "run_id": "R",
                "dilution": 1,
                "droplet_volume_uL": 50,
            }
        ],
    )
    assert source.counts.to_dataframe().observation_id.tolist() == ["001", "010", "100", "001"]
    source.save(tmp_path / "source.inptk")
    pd.testing.assert_frame_equal(
        source.counts.to_dataframe(), inptk.load(tmp_path / "source.inptk").counts.to_dataframe()
    )


def test_icescopy_missing_or_repeated_pictures_do_not_replace_observation_identity():
    data = pd.DataFrame(
        {
            "temperature_C": [-5, -5, -5.2, -5.2],
            "time_s": [0, 1, 2, 3],
            "picture": ["frame1", None, "frame1", None],
            "cycle": ["01"] * 4,
            "A number total": [10] * 4,
            "A number frozen": [0, 1, 2, 3],
        }
    )
    source = inptk.read_icescopy(data, metadata={"well_volume_uL": 50, "dilution": 1})
    frame = source.counts.to_dataframe()
    assert frame.observation_id.tolist() == ["0", "1", "2", "3"]
    assert frame.picture_id.dropna().tolist() == ["frame1", "frame1"]
    assert frame.picture_id.isna().sum() == 2
    assert frame.n_frozen.tolist() == [0, 1, 2, 3]
    assert frame.temperature_C.tolist() == data.temperature_C.tolist()


def individual_frame():
    return counts_frame().assign(
        point_id=["p0", "p1", "p2", "p0"],
        concentration=[0, 1, 2, 0],
        unit="INP_per_mL_suspension",
        basis="suspension",
        lower_error=0.1,
        upper_error=0.2,
    )


def combined_frame():
    return pd.DataFrame(
        {
            "sample_id": ["S", "S"],
            "curve_id": ["01", "01"],
            "point_id": ["001", "002"],
            "temperature_C": [-5, -5],
            "concentration": [1.0, 2.0],
            "lower_error": [0.1, 0.2],
            "upper_error": [0.2, 0.3],
            "unit": ["INP_per_mL_suspension"] * 2,
            "basis": ["suspension"] * 2,
            "source_observations": [
                json.dumps(
                    [
                        {
                            "measurement_id": "M",
                            "run_id": "R",
                            "cycle_id": "01",
                            "observation_id": "0",
                        }
                    ]
                ),
                "[]",
            ],
        }
    )


def test_individual_spectra_use_point_identity_and_preserve_explicit_temperature_only_validation():
    spectrum = inptk.CumulativeSpectrumTable(individual_frame())
    assert len(spectrum) == 4
    with pytest.raises(ValueError, match="unique identity/point"):
        inptk.CumulativeSpectrumTable(individual_frame().drop(columns="point_id"))
    with pytest.raises(ValueError, match="unique identity/point"):
        inptk.CumulativeSpectrumTable(individual_frame().assign(point_id="same"))


def test_combined_spectrum_has_group_point_identity_without_physical_run_or_cycle():
    table = inptk.CurveSpectrumTable(combined_frame())
    assert isinstance(table, inptk.CumulativeSpectrumTable)
    assert not {"run_id", "cycle_id"} & set(table.columns)
    assert table.select(curve_id="01", point_id="002").to_dataframe().concentration.tolist() == [2]
    with pytest.raises(ValueError, match="unique identity/point"):
        inptk.CurveSpectrumTable(combined_frame().assign(point_id="same"))
    with pytest.raises(ValueError, match="one parent sample"):
        inptk.CurveSpectrumTable(combined_frame().assign(sample_id=["S", "other"]))
    with pytest.raises(ValueError, match="point_id"):
        inptk.CurveSpectrumTable(combined_frame().drop(columns="point_id"))


@pytest.fixture
def grouped_source():
    metadata, rows = [], []
    for measurement, sample, run, dilution in (
        ("M1", "A/B", "R/1", 1),
        ("D1", "A/B", "R/1", 10),
        ("M2", "A/B", "R2", 1),
        ("OTHER", "B", "R2", 1),
        ("W1", "water1", "R/1", 1),
        ("W2", "water2", "R2", 1),
    ):
        metadata.append(
            {
                "measurement_id": measurement,
                "sample_id": sample,
                "run_id": run,
                "dilution": dilution,
                "droplet_volume_uL": 50,
            }
        )
        for cycle in ("01", "02"):
            rows.append(
                {
                    "measurement_id": measurement,
                    "cycle_id": cycle,
                    "temperature_C": -5,
                    "n_total": 10,
                    "n_frozen": 1,
                }
            )
    return inptk.read_counts(
        pd.DataFrame(rows),
        metadata=metadata,
        water_blank_map={"M1": ["W1"], "D1": ["W1"], "M2": ["W2"], "OTHER": ["W2"]},
    )


def test_default_groups_separate_runs_and_cycles_and_escape_ids(grouped_source):
    groups = resolve_curves(None, grouped_source, grouped_source.counts.to_dataframe())
    assert set(groups) == {
        "A%2FB/R%2F1/01",
        "A%2FB/R%2F1/02",
        "A%2FB/R2/01",
        "A%2FB/R2/02",
        "B/R2/01",
        "B/R2/02",
    }
    assert groups["A%2FB/R%2F1/01"] == {
        "sample_id": "A/B",
        "members": [
            {"measurement_id": "D1", "run_id": "R/1", "cycle_id": "01"},
            {"measurement_id": "M1", "run_id": "R/1", "cycle_id": "01"},
        ],
    }
    assert not {
        member["measurement_id"] for group in groups.values() for member in group["members"]
    } & {"W1", "W2"}


def test_explicit_groups_select_members_without_changing_source_or_caller(grouped_source):
    values = {
        "combined": {"inputs": [
            {"measurement_id": "M1", "cycle_id": "01"},
            {"measurement_id": "M2", "cycle_id": "02"},
        ]}
    }
    before = grouped_source.counts.to_dataframe()
    groups = resolve_curves(values, grouped_source, before)
    assert list(groups) == ["combined"]
    assert groups["combined"] == {
        "sample_id": "A/B",
        "members": [
            {"measurement_id": "M1", "run_id": "R/1", "cycle_id": "01"},
            {"measurement_id": "M2", "run_id": "R2", "cycle_id": "02"},
        ],
    }
    groups["combined"]["members"][0]["cycle_id"] = "changed"
    assert values["combined"]["inputs"][0]["cycle_id"] == "01"
    pd.testing.assert_frame_equal(grouped_source.counts.to_dataframe(), before)


@pytest.mark.parametrize(
    "members, message",
    [
        ([{"measurement_id": "M1", "cycle_id": "01"}] * 2, "Duplicate"),
        ([{"measurement_id": "W1", "cycle_id": "01"}], "Water-blank"),
        ([{"measurement_id": "M1", "cycle_id": "1"}], "Unknown or absent"),
        (
            [
                {"measurement_id": "M1", "cycle_id": "01"},
                {"measurement_id": "M1", "cycle_id": "02"},
            ],
            "one cycle per run",
        ),
        (
            [
                {"measurement_id": "M1", "cycle_id": "01"},
                {"measurement_id": "D1", "cycle_id": "02"},
            ],
            "one cycle per run",
        ),
        (
            [
                {"measurement_id": "M1", "cycle_id": "01"},
                {"measurement_id": "OTHER", "cycle_id": "01"},
            ],
            "one parent sample",
        ),
        ([{"measurement_id": "M1", "cycle_id": "01", "run_id": "R/1"}], "exactly"),
        ([{"measurement_id": "M1", "cycle_id": 1}], "non-empty strings"),
    ],
)
def test_groups_reject_invalid_or_nonindependent_members(grouped_source, members, message):
    with pytest.raises((ValueError, TypeError), match=message):
        resolve_curves(
            {"g": {"inputs": members}}, grouped_source, grouped_source.counts.to_dataframe()
        )


@pytest.mark.parametrize("groups", [{}, [], {"": []}, {"g": []}, {"g": {"members": []}}])
def test_invalid_group_shapes_are_rejected(grouped_source, groups):
    with pytest.raises((ValueError, TypeError)):
        resolve_curves(groups, grouped_source, grouped_source.counts.to_dataframe())


def test_filtered_out_members_cannot_reenter_through_group_configuration(grouped_source):
    frame = grouped_source.counts.select(cycle_id="01").to_dataframe()
    with pytest.raises(ValueError, match="Unknown or absent"):
        resolve_curves(
            {"g": {"inputs": [{"measurement_id": "M1", "cycle_id": "02"}]}}, grouped_source, frame
        )


def result_for_archive():
    source = inptk.read_counts(
        counts_frame(),
        metadata=[
            {
                "measurement_id": "M",
                "sample_id": "S",
                "run_id": "R",
                "dilution": 1,
                "droplet_volume_uL": 50,
            }
        ],
    )
    combined = inptk.CurveSpectrumTable(combined_frame())
    return inptk.AnalysisResult(
        source,
        inptk.FrozenFractionTable(source.counts.to_dataframe()),
        curves={
            "01": inptk.CurveResult(
                "01", combined.select(point_id="001"),
                sources=[{"measurement_id": "M", "run_id": "R", "cycle_id": "01"}],
                excluded=combined.select(point_id="002"),
                resampled=inptk.CurveSpectrumTable(
                    combined_frame().assign(point_id=["grid0", "grid1"])
                ),
            )
        },
    )


def test_new_format_roundtrip_retains_original_ids_groups_and_optional_resampling(tmp_path):
    result = result_for_archive()
    target = tmp_path / "result.inptk"
    result.save(target)
    assert json.loads((target / "analysis.json").read_text())["format_version"] == 3
    restored = inptk.load(target)
    assert restored.settings == result.settings
    assert restored.experiment.measurements == result.experiment.measurements
    pd.testing.assert_frame_equal(
        restored.experiment.counts.to_dataframe(), result.experiment.counts.to_dataframe()
    )
    for name in ("cumulative", "excluded", "resampled"):
        assert isinstance(getattr(restored.curves["01"], name), inptk.CurveSpectrumTable)
        pd.testing.assert_frame_equal(
            getattr(restored.curves["01"], name).to_dataframe(),
            getattr(result.curves["01"], name).to_dataframe(),
        )
    output = tmp_path / "sampled.csv"
    restored.export_csv(output, table="resampled")
    assert pd.read_csv(output).point_id.tolist() == ["grid0", "grid1"]
    with pytest.raises(FileExistsError):
        restored.export_csv(output)


def test_export_rejects_absent_resampled_table_and_unknown_table_choice(tmp_path):
    result = result_for_archive()
    result = replace(result, curves={"01": replace(result.curves["01"], resampled=None)})
    with pytest.raises(ValueError, match="No resampled"):
        result.export_csv(tmp_path / "none.csv", table="resampled")
    with pytest.raises(ValueError, match="table must be"):
        result.export_csv(tmp_path / "other.csv", table="combined")
    assert not list(tmp_path.iterdir())


def test_old_saved_format_is_rejected_explicitly(tmp_path):
    result_for_archive().save(tmp_path / "result.inptk")
    path = tmp_path / "result.inptk" / "analysis.json"
    payload = json.loads(path.read_text())
    payload["format_version"] = 1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unsupported.*format"):
        inptk.load(path.parent)


@pytest.mark.parametrize("image_column", [None, "picture", "image_name"])
def test_icescopy_uses_only_real_image_columns_for_picture_identity(image_column):
    data = pd.DataFrame(
        {
            "temperature_C": [-5, -5],
            "A number total": [10, 10],
            "A number frozen": [0, 1],
        }
    )
    if image_column is not None:
        data[image_column] = ["real-image", None]
    source = inptk.read_icescopy(data, metadata={"well_volume_uL": 50, "dilution": 1})
    frame = source.counts.to_dataframe()
    assert frame.observation_id.tolist() == ["0", "1"]
    assert frame.n_frozen.tolist() == [0, 1]
    if image_column is None:
        assert "picture_id" not in frame
    else:
        assert frame.picture_id.dropna().tolist() == ["real-image"]
        assert frame.picture_id.isna().sum() == 1


@pytest.mark.parametrize(
    "slot, wrong_slot",
    [
        ("cumulative", "frozen_fraction"),
        ("excluded", "frozen_fraction"),
        ("differential", "frozen_fraction"),
        ("resampled", "frozen_fraction"),
    ],
)
def test_analysis_result_rejects_wrong_scientific_table_in_each_slot(slot, wrong_slot):
    result = result_for_archive()
    with pytest.raises(TypeError, match=f"CurveResult {slot} must be"):
        replace(result.curves["01"], **{slot: result.frozen_fraction})


@pytest.mark.parametrize(
    "slot, wrong_slot",
    [
        ("cumulative", "frozen_fraction"),
        ("excluded", "frozen_fraction"),
        ("resampled", "frozen_fraction"),
    ],
)
def test_saved_result_rejects_valid_table_payload_in_wrong_slot(tmp_path, slot, wrong_slot):
    target = tmp_path / "result.inptk"
    result_for_archive().save(target)
    path = target / "analysis.json"
    payload = json.loads(path.read_text())
    payload["curves"]["01"]["tables"][slot] = payload[wrong_slot]
    path.write_text(json.dumps(payload))
    with pytest.raises(TypeError, match=f"CurveResult {slot} must be"):
        inptk.load(target)
