import inspect

import numpy as np
import pandas as pd
import pytest

import inptk
from inptk import _engine as engine


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


@pytest.mark.parametrize("method", ["stitch", "mle"])
def test_workflow_matches_retained_methods_and_keeps_cycles_separate(method):
    source = experiment()
    original = source.counts.to_dataframe()
    result = inptk.analyze_concentration(source, dilution_method=method, output_basis="sampled_air")
    assert len(result.final.to_dataframe().groupby(["sample_id", "cycle_id"])) == 4
    assert result.frozen_fraction.to_dataframe().n_total.eq(32).all()
    assert "cycle_policy" not in inspect.signature(inptk.analyze_concentration).parameters
    for (sample, cycle), rows in original.groupby(["sample_id", "cycle_id"]):
        fractions = []
        for mid, data in rows.groupby("measurement_id"):
            measurement = source.measurements[mid]
            old_data = data.assign(sample_id=mid, cycle=cycle)
            metadata = engine.SampleMetadata(
                sample_id=mid, sample_name=sample, well_volume_uL=50, dilution=measurement.dilution
            )
            old_counts = engine.CountsTable.from_dataframe(old_data, metadata=metadata)
            fractions.append(engine.fraction_frozen(old_counts, temperature_tolerance_C=0.05))
        combine = (
            engine.cumulative_spec_stitch if method == "stitch" else engine.cumulative_spec_mle
        )
        options = {"confidence_drop": 1.96**2 / 2} if method == "mle" else {}
        expected = combine(
            fractions, sample_group_by={m: sample for m in rows.measurement_id.unique()}, **options
        ).to_dataframe()
        actual = result.final.select(sample_id=sample, cycle_id=cycle).to_dataframe()
        np.testing.assert_allclose(actual.concentration, expected.value * 0.05, equal_nan=True)
        np.testing.assert_allclose(actual.lower_error, expected.lower_ci * 0.05, equal_nan=True)
    pd.testing.assert_frame_equal(original, source.counts.to_dataframe())


def test_results_roundtrip_and_selection_do_not_change_source(tmp_path):
    result = inptk.analyze_concentration(experiment(), differential=True)
    result.save(tmp_path / "result")
    restored = inptk.load(tmp_path / "result")
    for name in ("frozen_fraction", "per_dilution", "combined", "final", "differential"):
        pd.testing.assert_frame_equal(
            getattr(restored, name).to_dataframe(), getattr(result, name).to_dataframe()
        )
    assert restored.experiment.samples == result.experiment.samples
    assert restored.settings == result.settings
    frame = result.final.select(sample_id="001", cycle_id=["01", "02"]).to_dataframe()
    frame.loc[:, "concentration"] = -5
    assert not result.final.to_dataframe().concentration.eq(-5).any()
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
    assert len(result.final.to_dataframe().groupby(["run_id", "sample_id", "cycle_id"])) == 8


def test_invalid_counts_and_second_conversion_are_rejected():
    source = experiment()
    data = source.counts.to_dataframe()
    data.loc[0, "n_frozen"] = 33
    with pytest.raises(ValueError, match="Counts require"):
        inptk.CountsTable(data)
    result = inptk.analyze_concentration(source, output_basis="sampled_air")
    with pytest.raises(ValueError, match="already converted"):
        inptk.convert_concentration(result.final, source.samples, basis="sampled_air")


def test_explicit_blank_mapping_preserves_status_and_missing_cycles_fail():
    source = experiment()
    base = inptk.analyze_concentration(source)
    background = base.combined.select(sample_id="001").to_dataframe()
    background["sample_id"] = "blank"
    background["concentration"] = 0.1
    background["lower_error"] = 0.01
    background["upper_error"] = 0.02
    background["is_extrapolated"] = True
    blank = inptk.CumulativeSpectrumTable(background)
    corrected = inptk.analyze_concentration(source, blank_by_sample={"001": blank})
    selected = corrected.final.select(sample_id="001").to_dataframe()
    assert selected.is_extrapolated.all()
    assert selected.correction_state.eq("blank_corrected").all()
    np.testing.assert_allclose(
        selected.concentration,
        base.final.select(sample_id="001").to_dataframe().concentration - 0.1,
    )
    assert (
        corrected.final.select(sample_id="B")
        .to_dataframe()
        .correction_state.eq("uncorrected")
        .all()
    )
    with pytest.raises(ValueError, match="does not cover"):
        inptk.analyze_concentration(source, blank_by_sample={"001": blank.select(cycle_id="01")})


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
    assert len(inptk.analyze_concentration(imported).final.to_dataframe().groupby("cycle_id")) == 2


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
    assert len(inptk.analyze_concentration(result).final.to_dataframe().groupby("cycle_id")) == 2


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
