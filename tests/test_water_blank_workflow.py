import json

import numpy as np
import pandas as pd
import pytest

import inptk


def make_experiment(measurements, *, temperatures=(-5,), cycles=("01",)):
    metadata, counts, mapping = [], [], {}
    blanks = [name for name, spec in measurements.items() if spec[0] == "blank"]
    for name, (role, volume, dilution, frozen, total) in measurements.items():
        metadata.append(
            {
                "measurement_id": name,
                "sample_id": "S" if role == "sample" else name,
                "droplet_volume_uL": volume,
                "dilution": dilution,
            }
        )
        if role == "sample":
            mapping[name] = list(blanks)
        for cycle in cycles:
            for i, temperature in enumerate(temperatures):
                counts.append(
                    {
                        "measurement_id": name,
                        "cycle_id": cycle,
                        "temperature_C": temperature,
                        "n_frozen": frozen[i] if isinstance(frozen, list) else frozen,
                        "n_total": total,
                    }
                )
    return inptk.read_counts(pd.DataFrame(counts), metadata=metadata, water_blank_map=mapping)


@pytest.mark.parametrize("method", ["stitch", "mle", inptk.ManualStitch([])])
def test_different_blank_well_totals_keep_point_but_change_precision(method):
    results = []
    for frozen, total in ((2, 10), (4, 20)):
        source = make_experiment(
            {
                "sample": ("sample", 50, 1, 16, 32),
                "blank": ("blank", 50, 1, frozen, total),
            }
        )
        result = inptk.analyze_concentration(source, dilution_method=method)
        results.append(result.final.to_dataframe().iloc[0])
        assert set(result.per_dilution.to_dataframe().measurement_id) == {"sample"}
        assert len(result.experiment.counts) == 2
    expected = (-np.log(0.5) + np.log(0.8)) / 0.05
    assert results[0].concentration == pytest.approx(expected, rel=1e-7)
    assert results[1].concentration == pytest.approx(expected, rel=1e-7)
    assert results[1].lower_error < results[0].lower_error
    assert results[1].upper_error < results[0].upper_error


@pytest.mark.parametrize("method", ["stitch", "mle"])
def test_unequal_blank_volumes_keep_separate_likelihood_terms(method):
    source = make_experiment(
        {
            "sample": ("sample", 50, 1, 8, 16),
            "blank50": ("blank", 50, 1, 4, 16),
            "blank100": ("blank", 100, 1, 7, 16),
        }
    )
    result = inptk.analyze_concentration(source, dilution_method=method)
    expected = (-np.log(0.5) + np.log(0.75)) / 0.05
    assert result.final.to_dataframe().concentration.iloc[0] == pytest.approx(expected, rel=1e-7)
    assert json.loads(result.per_dilution.to_dataframe().water_blank_ids.iloc[0]) == [
        "blank100",
        "blank50",
    ]


def test_joint_dilutions_use_each_sample_volume():
    source = make_experiment(
        {
            "neat": ("sample", 50, 1, 6, 10),
            "diluted": ("sample", 100, 2, 17, 25),
            "blank": ("blank", 50, 1, 2, 10),
        }
    )
    result = inptk.analyze_concentration(source, dilution_method="mle")
    assert result.final.to_dataframe().concentration.iloc[0] == pytest.approx(
        -np.log(0.5) / 0.05, rel=1e-7
    )


def test_raw_stages_cycles_differential_and_archive_remain_consistent(tmp_path):
    source = make_experiment(
        {
            "sample": ("sample", 50, 1, [0, 8, 16], 32),
            "blank": ("blank", 50, 1, [0, 1, 2], 10),
        },
        temperatures=(-5, -6, -7),
        cycles=("01", "02"),
    )
    before = source.counts.to_dataframe()
    result = inptk.analyze_concentration(source, differential=True, step_C=1)
    fractions = inptk.frozen_fraction(source, step_C=1)
    combined = inptk.combine_dilutions(fractions, experiment=source)
    pd.testing.assert_frame_equal(
        result.final.to_dataframe(), inptk.finalize_spectrum(combined).to_dataframe()
    )
    pd.testing.assert_frame_equal(before, source.counts.to_dataframe())
    for cycle in ("01", "02"):
        cumulative = result.per_dilution.select(cycle_id=cycle).to_dataframe()
        differential = result.differential.select(cycle_id=cycle).to_dataframe()
        np.testing.assert_allclose(differential.concentration, np.diff(cumulative.concentration))
    result.save(tmp_path / "raw.inptk")
    restored = inptk.load(tmp_path / "raw.inptk")
    assert restored.experiment.water_blank_map == {"sample": ["blank"]}
    pd.testing.assert_frame_equal(result.final.to_dataframe(), restored.final.to_dataframe())


@pytest.mark.parametrize(
    "method",
    [
        inptk.MLE(temperature_eligibility_C={"sample": -6}, mask_mode="rebase_counts"),
        inptk.MLE(likelihood_weights={"sample": 0.5}),
    ],
)
def test_joint_raw_model_rejects_synthetic_or_weighted_counts(method):
    source = make_experiment(
        {"sample": ("sample", 50, 1, 16, 32), "blank": ("blank", 50, 1, 2, 10)}
    )
    with pytest.raises(ValueError, match="rebase_counts|weighted"):
        inptk.analyze_concentration(source, dilution_method=method)


def test_missing_blank_temperature_is_an_error_not_extrapolation():
    source = make_experiment(
        {
            "sample": ("sample", 50, 1, [0, 16], 32),
            "blank": ("blank", 50, 1, [0, 2], 10),
        },
        temperatures=(-5, -6),
    )
    fractions = inptk.frozen_fraction(source, step_C=1).to_dataframe()
    fractions = fractions.loc[
        ~((fractions.measurement_id == "blank") & (fractions.temperature_C == -6))
    ]
    with pytest.raises(ValueError, match="lacks matching"):
        inptk.cumulative_spectrum(inptk.FrozenFractionTable(fractions), experiment=source)


@pytest.mark.parametrize(
    "method, expected_sources",
    [
        (inptk.Stitch(), ["neat", "neat", "diluted"]),
        (inptk.ManualStitch([-6]), ["neat", "diluted", "diluted"]),
    ],
)
def test_raw_blank_stitch_copies_selected_dilution_and_its_full_uncertainty(
    method, expected_sources
):
    source = make_experiment(
        {
            "neat": ("sample", 50, 1, [8, 28, 30], 32),
            "diluted": ("sample", 50, 10, [4, 8, 12], 32),
            "water": ("blank", 100, 1, [1, 1, 1], 10),
        },
        temperatures=(-5, -6, -7),
    )
    result = inptk.analyze_concentration(source, dilution_method=method, step_C=1)
    selected = result.combined.to_dataframe().sort_values("temperature_C", ascending=False)
    assert selected.source_measurement_id.tolist() == expected_sources
    individual = result.per_dilution.to_dataframe().set_index(["measurement_id", "temperature_C"])
    for row in selected.itertuples():
        expected = individual.loc[(row.source_measurement_id, row.temperature_C)]
        np.testing.assert_allclose(
            [row.concentration, row.lower_error, row.upper_error],
            [expected.concentration, expected.lower_error, expected.upper_error],
            rtol=0,
            atol=0,
        )
