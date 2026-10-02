import inspect

import numpy as np
import pandas as pd
import pytest
from analysis_checks import all_points, fit_estimates, input_spectra, quantity_for_check, retained

import inptk
from inptk.methods import resolve_curves


def experiment():
    rows, measurements = [], []
    for sample in ("001", "B"):
        for label, dilution in (("neat", 1), ("diluted", 10)):
            mid = sample + "_" + label
            measurements.append(
                {
                    "measurement_id": mid,
                    "sample_id": sample,
                    "dilution": dilution,
                    "droplet_volume_uL": 50,
                    "sample_type": "air",
                    "air_volume_L": 100,
                    "suspension_volume_mL": 5,
                    "filter_fraction_used": 1,
                }
            )
            for cycle in ("01", "02"):
                frozen = [0, 3, 9, 16] if dilution == 1 else [0, 1, 2, 4]
                if cycle == "02":
                    frozen = [value + 1 for value in frozen]
                for temperature, count in zip((-5, -6, -7, -8), frozen):
                    rows.append(
                        {
                            "measurement_id": mid,
                            "cycle_id": cycle,
                            "temperature_C": temperature,
                            "n_total": 32,
                            "n_frozen": count,
                        }
                    )
    return inptk.read_counts(pd.DataFrame(rows), metadata=measurements)


def test_joint_workflow_matches_stepwise_analysis_and_keeps_cycles_separate():
    source = experiment()
    original = source.counts.to_dataframe()
    result = inptk.analyze_concentration(source, method="mle", output_basis="sampled_air")
    assert retained(result).to_dataframe().curve_id.nunique() == 4
    assert result.frozen_fraction.to_dataframe().n_total.eq(32).all()
    assert "cycle_policy" not in inspect.signature(inptk.analyze_concentration).parameters
    for curve_id, group in result.settings["curves"].items():
        members = group["inputs"]
        assert len({member["cycle_id"] for member in members}) == 1
        stepwise = inptk.estimate_concentration(
            inptk.frozen_fraction(source), experiment=source, curves={curve_id: group}
        ).to_dataframe()
        candidate = all_points(result).select(curve_id=curve_id).to_dataframe()
        assert candidate.concentration.diff().dropna().ge(0).all()
        np.testing.assert_allclose(
            candidate[["concentration", "lower_error", "upper_error"]],
            stepwise[["concentration", "lower_error", "upper_error"]] * 0.05,
            rtol=1e-8,
            atol=1e-9,
        )
        selected = inptk.finalize_spectrum(all_points(result).select(curve_id=curve_id))
        actual = retained(result).select(curve_id=curve_id)
        pd.testing.assert_frame_equal(actual.to_dataframe(), selected.to_dataframe())
    pd.testing.assert_frame_equal(original, source.counts.to_dataframe())


def test_results_roundtrip_and_selection_do_not_change_source(tmp_path):
    result = inptk.analyze_concentration(
        experiment(), differential=True, curves={"one": {"inputs": ["001_neat"], "cycle": "01"}}
    )
    result.save(tmp_path / "result")
    restored = inptk.load(tmp_path / "result")
    for name in (
        "frozen_fraction",
        "per_dilution",
        "combined",
        "final",
        "differential",
        "final_candidates",
    ):
        pd.testing.assert_frame_equal(
            quantity_for_check(restored, name).to_dataframe(),
            quantity_for_check(result, name).to_dataframe(),
        )
    assert restored.experiment.samples == result.experiment.samples
    assert restored.settings == result.settings
    frame = retained(result).select(sample_id="001").to_dataframe()
    frame.loc[:, "concentration"] = -5
    assert not retained(result).to_dataframe().concentration.eq(-5).any()
    with pytest.raises(FileExistsError):
        result.save(tmp_path / "result")


def test_independent_runs_with_same_cycle_labels_are_not_combined():
    first = experiment()
    rows = first.counts.to_dataframe()
    metadata = []
    for run in ("R1", "R2"):
        for measurement in first.measurements.values():
            metadata.append(
                {
                    "measurement_id": run + measurement.measurement_id,
                    "sample_id": measurement.sample_id,
                    "run_id": run,
                    "dilution": measurement.dilution,
                    "droplet_volume_uL": 50,
                }
            )
    both = pd.concat(
        [rows.assign(run_id=run, measurement_id=run + rows.measurement_id) for run in ("R1", "R2")]
    )
    result = inptk.analyze_concentration(inptk.read_counts(both, metadata=metadata))
    assert retained(result).to_dataframe().curve_id.nunique() == 8
    for group in resolve_curves(
        result.settings["curves"], result.experiment, result.counts.to_dataframe()
    ).values():
        assert len({(member["run_id"], member["cycle_id"]) for member in group["members"]}) == 1


def test_invalid_counts_and_second_conversion_are_rejected():
    source = experiment()
    data = source.counts.to_dataframe()
    data.loc[0, "n_frozen"] = 33
    with pytest.raises(ValueError, match="Counts require"):
        inptk.CountsTable(data)
    result = inptk.analyze_concentration(source, output_basis="sampled_air")
    with pytest.raises(ValueError, match="already converted"):
        inptk.convert_concentration(retained(result), source.samples, basis="sampled_air")


def test_explicit_group_blank_mapping_preserves_status_and_requires_temperature_coverage():
    source = experiment()
    base = inptk.analyze_concentration(source)
    blank_map = {}
    for cycle in ("01", "02"):
        background = (
            input_spectra(base).select(measurement_id="001_neat", cycle_id=cycle).to_dataframe()
        )
        background["sample_id"] = "blank"
        background["concentration"] = 0.1
        background["lower_error"] = 0.01
        background["upper_error"] = 0.02
        background["is_extrapolated"] = True
        blank_map[f"001/1/{cycle}"] = inptk.CumulativeSpectrumTable(background)
    corrected = inptk.analyze_concentration(source, blank_by_curve=blank_map)
    selected = retained(corrected).select(sample_id="001").to_dataframe()
    assert selected.is_extrapolated.all()
    assert selected.correction_state.eq("blank_corrected").all()
    np.testing.assert_allclose(
        selected.concentration,
        retained(base).select(sample_id="001").to_dataframe().concentration - 0.1,
    )
    assert (
        retained(corrected)
        .select(sample_id="B")
        .to_dataframe()
        .correction_state.eq("uncorrected")
        .all()
    )
    blank_map["001/1/01"] = blank_map["001/1/01"].select(temperature_C=[-5, -6, -7])
    with pytest.raises(ValueError, match="does not cover"):
        inptk.analyze_concentration(source, blank_by_curve=blank_map)


def test_icescopy_import_requires_explicit_parent_mapping():
    frame = pd.DataFrame(
        {
            "temperature_C": [-5, -6, -5, -6],
            "cycle": [1, 1, 2, 2],
            "A_neat number total": [20] * 4,
            "A_neat number frozen": [0, 4, 0, 5],
            "A_diluted number total": [20] * 4,
            "A_diluted number frozen": [0, 1, 0, 2],
        }
    )
    metadata = {
        name: {"sample_id": name, "dilution": dilution, "well_volume_uL": 50}
        for name, dilution in (("A_neat", 1), ("A_diluted", 10))
    }
    imported = inptk.read_icescopy(
        frame, metadata=metadata, sample_map={"A_neat": "A", "A_diluted": "A"}
    )
    assert set(imported.samples) == {"A"}
    assert set(imported.counts.to_dataframe().cycle_id) == {"1", "2"}
    assert (
        len(retained(inptk.analyze_concentration(imported)).to_dataframe().groupby("curve_id")) == 2
    )


def test_icescopy_file_preserves_labels_without_inventing_elapsed_seconds(tmp_path):
    source = tmp_path / "icescopy.csv"
    source.write_text(
        "# well_volume_uL: 50\n"
        "# sample_name,A_neat,A_diluted\n"
        "# dilution,1,10\n"
        "temperature_C,cycle,A_neat number total,A_neat number frozen,"
        "A_diluted number total,A_diluted number frozen\n"
        "-5,01,20,0,20,0\n-6,01,20,4,20,1\n"
        "-5,02,20,0,20,0\n-6,02,20,5,20,2\n"
    )
    result = inptk.read_icescopy(source, sample_map={"A_neat": "A", "A_diluted": "A"})
    assert set(result.counts.to_dataframe().cycle_id) == {"01", "02"}
    assert "time_s" not in result.counts.columns
    assert (
        len(retained(inptk.analyze_concentration(result)).to_dataframe().groupby("curve_id")) == 2
    )


def test_supplied_invalid_timestamps_are_not_replaced_with_row_numbers():
    frame = pd.DataFrame(
        {
            "timestamp": ["invalid", "invalid"],
            "temperature_C": [-5, -6],
            "A number total": [20, 20],
            "A number frozen": [0, 4],
        }
    )
    with pytest.raises(ValueError, match="timestamps must be valid"):
        inptk.read_icescopy(frame, metadata={"A": {"dilution": 1, "well_volume_uL": 50}})


@pytest.mark.parametrize("missing", [None, float("nan"), "", "  "])
def test_icescopy_rejects_partially_missing_cycle_labels(missing):
    frame = pd.DataFrame(
        {
            "temperature_C": [-5, -6, -7],
            "cycle": ["1", missing, "1"],
            "A number total": [20] * 3,
            "A number frozen": [0, 2, 5],
        }
    )
    with pytest.raises(ValueError, match="Some Icescopy cycle labels are missing"):
        inptk.read_icescopy(frame, metadata={"A": {"dilution": 1, "well_volume_uL": 50}})


def test_icescopy_all_missing_cycle_labels_mean_one_cycle():
    frame = pd.DataFrame(
        {
            "temperature_C": [-5, -6],
            "cycle": [None, ""],
            "A number total": [20, 20],
            "A number frozen": [0, 2],
        }
    )
    result = inptk.read_icescopy(frame, metadata={"A": {"dilution": 1, "well_volume_uL": 50}})
    assert set(result.counts.to_dataframe().cycle_id) == {"1"}


@pytest.mark.parametrize(
    "sample_map", [["A"], {"A": None}, {"A": []}, {"A": ""}, {"A": "  "}, {1: "A"}]
)
def test_icescopy_rejects_invalid_sample_maps(sample_map):
    frame = pd.DataFrame({"temperature_C": [-5], "A number total": [20], "A number frozen": [0]})
    with pytest.raises(TypeError, match="sample_map must map non-empty"):
        inptk.read_icescopy(frame, sample_map=sample_map)


@pytest.mark.parametrize("shape", ["mapping", "records", "dataframe", "common"])
def test_icescopy_metadata_overrides_keep_unsupplied_header_values(tmp_path, shape):
    path = tmp_path / "counts.csv"
    path.write_text(
        "# well_volume_uL: 50\n"
        "# sample_name,A,B\n"
        "# sample_type,air,air\n"
        "# dilution,1,10\n"
        "# air_volume_L,100,100\n"
        "# suspension_volume_mL,5,5\n"
        "# filter_fraction_used,1,1\n"
        "temperature_C,A number total,A number frozen,B number total,B number frozen\n"
        "-5,20,0,20,0\n-6,20,2,20,1\n"
    )
    supplied = {"air_volume_L": 200, "well_volume_uL": 25, "dilution": None}
    if shape == "mapping":
        metadata = {"A": supplied}
    elif shape == "records":
        metadata = [dict(supplied, sample_id="A")]
    elif shape == "dataframe":
        metadata = pd.DataFrame([dict(supplied, sample_id="A")])
    else:
        metadata = supplied
    result = inptk.read_icescopy(path, metadata=metadata)
    assert result.measurements["A"].dilution == 1
    assert result.measurements["A"].droplet_volume_uL == 25
    assert result.samples["A"].sample_type == "air"
    assert result.samples["A"].air_volume_L == 200
    assert result.samples["A"].suspension_volume_mL == 5
    assert result.measurements["B"].dilution == 10
    assert result.samples["B"].air_volume_L == (200 if shape == "common" else 100)
    assert result.measurements["B"].droplet_volume_uL == (25 if shape == "common" else 50)
    assert len(retained(inptk.analyze_concentration(result, output_basis="sampled_air"))) > 0


def test_icescopy_metadata_rejects_unknown_measurement_names():
    frame = pd.DataFrame({"temperature_C": [-5], "A number total": [20], "A number frozen": [0]})
    with pytest.raises(ValueError, match="Unknown measurement in Icescopy metadata: 'wrong'"):
        inptk.read_icescopy(frame, metadata={"wrong": {"well_volume_uL": 50}})


@pytest.mark.parametrize(
    "policy,kept,statuses",
    [
        (
            "stop_at_decrease",
            [True, True, False, False, False],
            ["kept", "kept", "decrease", "after_decrease", "after_decrease"],
        ),
        (
            "skip_decreases",
            [True, True, False, True, True],
            ["kept", "kept", "decrease", "kept", "kept"],
        ),
    ],
)
def test_roundtrip_retains_discarded_final_values_and_uncertainty(tmp_path, policy, kept, statuses):
    counts = pd.DataFrame(
        {
            "measurement_id": ["M"] * 5,
            "cycle_id": ["01"] * 5,
            "temperature_C": [-5, -6, -7, -8, -9],
            "n_total": [20, 20, 16, 16, 16],
            "n_frozen": [0, 8, 4, 10, 12],
        }
    )
    source = inptk.read_counts(
        counts,
        metadata=[
            {
                "measurement_id": "M",
                "sample_id": "A",
                "sample_type": "air",
                "droplet_volume_uL": 50,
                "dilution": 1,
                "air_volume_L": 100,
                "suspension_volume_mL": 5,
                "filter_fraction_used": 1,
            }
        ],
    )
    result = inptk.analyze_concentration(
        source, output_basis="sampled_air", decrease_policy=policy, method="average"
    )
    result.save(tmp_path / "result.inptk")
    restored = inptk.load(tmp_path / "result.inptk")
    candidates = all_points(restored).to_dataframe()
    assert candidates.used_in_final.tolist() == kept
    assert candidates.final_selection_status.tolist() == statuses
    assert candidates.temperature_C.tolist() == [-5, -6, -7, -8, -9]
    assert retained(restored).to_dataframe().temperature_C.tolist() == [
        temperature
        for temperature, selected in zip([-5, -6, -7, -8, -9], kept, strict=True)
        if selected
    ]
    assert restored.settings["decrease_policy"] == policy
    for column in ("concentration", "lower_error", "upper_error"):
        np.testing.assert_allclose(
            candidates[column], fit_estimates(result).to_dataframe()[column] * 0.05, rtol=0, atol=0
        )
    pd.testing.assert_frame_equal(candidates, all_points(result).to_dataframe())
    pd.testing.assert_frame_equal(
        retained(restored).to_dataframe(), retained(result).to_dataframe()
    )
    pd.testing.assert_frame_equal(
        restored.experiment.counts.to_dataframe(), source.counts.to_dataframe()
    )


def test_saving_analysis_without_optional_resampled_output(tmp_path):
    result = inptk.analyze_concentration(experiment())
    result.save(tmp_path / "result.inptk")
    restored = inptk.load(tmp_path / "result.inptk")
    assert all(curve.resampled is None for curve in restored.curves.values())
    pd.testing.assert_frame_equal(
        retained(restored).to_dataframe(), retained(result).to_dataframe()
    )
