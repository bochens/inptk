"""The user-facing result is a named scientific curve, not a processing stage."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import inptk
from inptk import cli


def source(*, dilution=1, cycles=("01",)):
    rows = [
        {
            "measurement_id": name,
            "cycle_id": cycle,
            "temperature_C": temperature,
            "n_total": total,
            "n_frozen": count,
        }
        for name, total, counts in (("first", 10, (1, 4, 6)), ("second", 20, (2, 7, 12)))
        for cycle in cycles
        for temperature, count in zip((-5, -6, -7), counts, strict=True)
    ]
    metadata = [
        {"measurement_id": name, "sample_id": "A", "dilution": factor, "droplet_volume_uL": 50}
        for name, factor in (("first", 1), ("second", dilution))
    ]
    return inptk.read_counts(pd.DataFrame(rows), metadata=metadata)


@pytest.mark.parametrize("method", ["mle", "average"])
@pytest.mark.parametrize("dilution", [1, 10])
def test_individual_and_combined_curves_share_interface_and_preserve_sources(method, dilution):
    data = source(dilution=dilution)
    choices = {
        "first alone": {"inputs": ["first"]},
        "second alone": {"inputs": ["second"]},
        "My sample 雪": {"inputs": ["first", "second"]},
    }
    result = inptk.analyze_concentration(data, curves=choices, method=method)
    assert list(result.curves) == list(choices)
    assert result.curves["first alone"].kind == "individual"
    combined = result.curves["My sample 雪"]
    assert combined.kind == "combined"
    assert [item["measurement_id"] for item in combined.sources] == ["first", "second"]
    assert [item["droplet_volume_uL"] for item in combined.sources] == [50, 50]
    assert not {"run_id", "cycle_id"} & set(combined.cumulative.columns)
    assert combined.cumulative.to_dataframe().contributor_count.eq(2).all()
    pd.testing.assert_frame_equal(result.counts.to_dataframe(), data.counts.to_dataframe())
    pd.testing.assert_frame_equal(
        result.frozen_fraction.to_dataframe().drop(columns="fraction_frozen"),
        data.counts.to_dataframe(),
    )
    if method == "mle" and dilution == 1:
        expected = -np.log1p(-np.array([3, 11, 18]) / 30) / 0.05
        np.testing.assert_allclose(combined.cumulative.to_dataframe().concentration, expected)


def test_repeated_cycles_require_selection_and_remain_separate():
    data = source(cycles=("01", "02"))
    with pytest.raises(ValueError, match="several cycles"):
        inptk.analyze_concentration(data, curves={"A": {"inputs": ["first"]}})
    result = inptk.analyze_concentration(
        data,
        curves={
            "first freezing": {"inputs": ["first"], "cycle": "01"},
            "second freezing": {"inputs": ["first"], "cycle": "02"},
        },
    )
    assert len(result.curves) == 2
    for curve in result.curves.values():
        assert len(curve.sources) == 1
        assert curve.cumulative.to_dataframe().contributor_count.eq(1).all()
    with pytest.raises(ValueError, match="one cycle per run"):
        inptk.analyze_concentration(
            data,
            curves={
                "invalid": {
                    "inputs": [
                        {"measurement_id": "first", "cycle_id": "01"},
                        {"measurement_id": "first", "cycle_id": "02"},
                    ]
                }
            },
        )


@pytest.mark.parametrize(
    "policy, expected",
    [
        ("stop_at_decrease", [-5, -6]),
        ("skip_decreases", [-5, -6, -8]),
    ],
)
def test_cumulative_contains_retained_points_and_excluded_explains_the_rest(policy, expected):
    data = inptk.read_counts(
        pd.DataFrame(
            {
                "measurement_id": ["first"] * 4,
                "temperature_C": [-5, -6, -7, -8],
                "n_total": [10] * 4,
                "n_frozen": [1, 4, 3, 6],
            }
        ),
        metadata=[
            {"measurement_id": "first", "sample_id": "A", "dilution": 1, "droplet_volume_uL": 50}
        ],
    )
    result = inptk.analyze_concentration(
        data, curves={"chosen": {"inputs": ["first"]}}, decrease_policy=policy
    )
    curve = result.curves["chosen"]
    assert curve.cumulative.to_dataframe().temperature_C.tolist() == expected
    assert not curve.excluded.to_dataframe().used_in_final.any()
    assert curve.excluded.to_dataframe().final_selection_status.iloc[0] == "decrease"
    assert len(curve.cumulative) + len(curve.excluded) == len(data.counts)
    assert not hasattr(result, "final") and not hasattr(result, "per_dilution")


def test_names_and_empty_exclusions_roundtrip_and_exports_are_explicit(tmp_path):
    result = inptk.analyze_concentration(
        source(), curves={"Name / 雪": {"inputs": ["first"]}}, output_step_C=0.5
    )
    path = tmp_path / "analysis.inptk"
    result.save(path)
    restored = inptk.load(path)
    assert list(restored.curves) == ["Name / 雪"]
    assert restored.curves["Name / 雪"].sources == result.curves["Name / 雪"].sources
    for quantity in ("cumulative", "excluded", "resampled"):
        pd.testing.assert_frame_equal(
            result.to_dataframe(table=quantity), restored.to_dataframe(table=quantity)
        )
    replay = inptk.analyze_concentration(source(), curves=result.settings["curves"])
    pd.testing.assert_frame_equal(result.to_dataframe(), replay.to_dataframe())
    output = tmp_path / "one.csv"
    restored.export_csv(output, curve_id="Name / 雪")
    assert pd.read_csv(output).curve_id.eq("Name / 雪").all()
    with pytest.raises(FileExistsError):
        restored.export_csv(output)
    with pytest.raises(ValueError, match="Unknown curve"):
        restored.export_csv(tmp_path / "unknown.csv", curve_id="missing")
    assert not (tmp_path / "unknown.csv").exists()


@pytest.mark.parametrize("version", [1, 2, 4])
def test_no_old_or_unknown_format_fallbacks(tmp_path, version):
    path = tmp_path / "analysis.inptk"
    inptk.analyze_concentration(source()).save(path)
    file = path / "analysis.json"
    payload = json.loads(file.read_text())
    payload["format_version"] = version
    file.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="Unsupported"):
        inptk.load(path)


def test_differential_has_one_source_and_does_not_cross_excluded_observations():
    result = inptk.analyze_concentration(
        source(), curves={"first": {"inputs": ["first"]}}, differential=True
    )
    assert set(result.curves["first"].differential.to_dataframe().measurement_id) == {"first"}
    with pytest.raises(ValueError, match="individual curves"):
        inptk.analyze_concentration(source(), differential=True)
    with pytest.raises(ValueError, match="dictionary keys"):
        replace(result, curves={"wrong": result.curves["first"]})


def test_cli_rejects_duplicate_curve_names_instead_of_replacing_them():
    with pytest.raises(ValueError, match="duplicate name"):
        cli._json_object('{"A":{"inputs":["first"]},"A":{"inputs":["second"]}}', "--curves")


def test_unknown_input_name_reports_the_name_without_inventing_a_cycle():
    with pytest.raises(ValueError, match="Unknown or absent curve input: 'missing'"):
        inptk.analyze_concentration(source(), curves={"A": {"inputs": ["missing"]}})
