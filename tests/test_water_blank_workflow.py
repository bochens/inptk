import json

import numpy as np
import pandas as pd
import pytest
from analysis_checks import all_points, fit_estimates, input_spectra, intervals, retained

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


@pytest.mark.parametrize("method", ["mle", "average"])
def test_different_blank_well_totals_keep_point_but_change_precision(method):
    results = []
    for frozen, total in ((1, 4), (8, 32)):
        source = make_experiment(
            {
                "sample": ("sample", 50, 1, 16, 32),
                "blank": ("blank", 50, 1, frozen, total),
            }
        )
        result = inptk.analyze_concentration(source, method=method)
        results.append(retained(result).to_dataframe().iloc[0])
        assert set(input_spectra(result).to_dataframe().measurement_id) == {"sample"}
        assert len(result.experiment.counts) == 2
        assert result.settings["water_blank_model"] == "volume_scaled"
    expected = (-np.log(0.5) + np.log(0.75)) / 0.05
    assert results[0].concentration == pytest.approx(expected, rel=1e-7)
    assert results[1].concentration == pytest.approx(expected, rel=1e-7)
    assert results[1].lower_error < results[0].lower_error
    assert results[1].upper_error < results[0].upper_error


@pytest.mark.parametrize("method", ["mle", "average"])
def test_unequal_blank_volumes_keep_separate_likelihood_terms(method):
    source = make_experiment(
        {
            "sample": ("sample", 50, 1, 8, 16),
            "blank50": ("blank", 50, 1, 4, 16),
            "blank100": ("blank", 100, 1, 7, 16),
        }
    )
    result = inptk.analyze_concentration(source, method=method)
    expected = (-np.log(0.5) + np.log(0.75)) / 0.05
    assert retained(result).to_dataframe().concentration.iloc[0] == pytest.approx(
        expected, rel=1e-7
    )
    assert json.loads(input_spectra(result).to_dataframe().water_blank_ids.iloc[0]) == [
        "blank100",
        "blank50",
    ]


@pytest.mark.parametrize("method", ["mle", "average"])
def test_joint_dilutions_use_each_sample_volume(method):
    source = make_experiment(
        {
            "neat": ("sample", 50, 1, 6, 10),
            "diluted": ("sample", 100, 2, 17, 25),
            "blank": ("blank", 50, 1, 2, 10),
        }
    )
    result = inptk.analyze_concentration(source, method=method)
    assert retained(result).to_dataframe().concentration.iloc[0] == pytest.approx(
        -np.log(0.5) / 0.05, rel=1e-7
    )


@pytest.mark.parametrize("method", ["mle", "average"])
def test_raw_stages_cycles_differential_and_archive_remain_consistent(tmp_path, method):
    source = make_experiment(
        {
            "sample": ("sample", 50, 1, [0, 8, 16], 32),
            "blank": ("blank", 50, 1, [0, 1, 2], 10),
        },
        temperatures=(-5, -6, -7),
        cycles=("01", "02"),
    )
    before = source.counts.to_dataframe()
    result = inptk.analyze_concentration(source, method=method, differential=True)
    fractions = inptk.frozen_fraction(source)
    combined = inptk.estimate_concentration(fractions, experiment=source, method=method)
    pd.testing.assert_frame_equal(
        retained(result).to_dataframe(), inptk.finalize_spectrum(combined).to_dataframe()
    )
    pd.testing.assert_frame_equal(before, source.counts.to_dataframe())
    assert len(fit_estimates(result)) == 6
    first_cycle_source = inptk.Experiment(
        counts=source.counts.select(cycle_id="01"),
        measurements=source.measurements,
        samples=source.samples,
        water_blank_map=source.water_blank_map,
    )
    first_cycle_result = inptk.analyze_concentration(first_cycle_source, method=method)
    pd.testing.assert_frame_equal(
        fit_estimates(result).select(curve_id="S/1/01").to_dataframe(),
        fit_estimates(first_cycle_result).to_dataframe(),
    )
    for cycle in ("01", "02"):
        cumulative = result.curves[f"S/1/{cycle}"].cumulative.to_dataframe()
        differential = intervals(result).select(cycle_id=cycle).to_dataframe()
        np.testing.assert_allclose(differential.concentration, np.diff(cumulative.concentration))
    result.save(tmp_path / "raw.inptk")
    restored = inptk.load(tmp_path / "raw.inptk")
    assert restored.experiment.water_blank_map == {"sample": ["blank"]}
    pd.testing.assert_frame_equal(
        retained(result).to_dataframe(), retained(restored).to_dataframe()
    )
    pd.testing.assert_frame_equal(
        all_points(result).to_dataframe(), all_points(restored).to_dataframe()
    )
    pd.testing.assert_frame_equal(before, restored.experiment.counts.to_dataframe())


@pytest.mark.parametrize("method", ["mle", "average"])
@pytest.mark.parametrize(
    "options",
    [
        {"mask_mode": "rebase_counts"},
        {"likelihood_weights": {"sample": 0.5}},
        {"action_counts": {"sample": 1}},
        {"confidence_drop": 2},
    ],
)
def test_raw_count_analysis_has_no_rebase_or_weight_controls(method, options):
    source = make_experiment(
        {"sample": ("sample", 50, 1, 16, 32), "blank": ("blank", 50, 1, 2, 10)}
    )
    fractions = inptk.frozen_fraction(source)
    with pytest.raises(TypeError, match=next(iter(options))):
        inptk.analyze_concentration(source, method=method, **options)
    with pytest.raises(TypeError, match=next(iter(options))):
        inptk.estimate_concentration(fractions, experiment=source, method=method, **options)


def test_missing_blank_temperature_is_an_error_not_extrapolation():
    source = make_experiment(
        {
            "sample": ("sample", 50, 1, [0, 16], 32),
            "blank": ("blank", 50, 1, [0, 2], 10),
        },
        temperatures=(-5, -6),
    )
    fractions = inptk.frozen_fraction(source).to_dataframe()
    fractions = fractions.loc[
        ~((fractions.measurement_id == "blank") & (fractions.temperature_C == -6))
    ]
    with pytest.raises(ValueError, match="lacks observed temperature coverage"):
        inptk.cumulative_spectrum(inptk.FrozenFractionTable(fractions), experiment=source)


@pytest.mark.parametrize("method", ["mle", "average"])
def test_raw_blank_ranges_record_zero_one_and_two_contributors(method):
    source = make_experiment(
        {
            "neat": ("sample", 50, 1, [8, 16, 24, 30], 32),
            "diluted": ("sample", 50, 10, [1, 3, 6, 12], 32),
            "water": ("blank", 100, 1, [1, 1, 1, 1], 10),
        },
        temperatures=(-5, -6, -7, -8),
    )
    result = inptk.analyze_concentration(
        source,
        method=method,
        temperature_ranges_C={
            "neat": {"min_C": -6},
            "diluted": {"min_C": -7, "max_C": -6},
        },
    )
    selected = fit_estimates(result).to_dataframe().sort_values("temperature_C", ascending=False)
    assert selected.contributor_count.tolist() == [1, 2, 1, 0]
    assert selected.source_measurement_id.tolist() == ["neat", "", "diluted", ""]
    assert selected.selection_status.tolist() == [
        "single",
        "combined",
        "single",
        "no_eligible_measurements",
    ]
    assert selected.contributing_measurement_ids.map(json.loads).tolist() == [
        ["neat"],
        ["diluted", "neat"],
        ["diluted"],
        [],
    ]
    assert selected.iloc[-1][["concentration", "lower_error", "upper_error"]].isna().all()
    individual = input_spectra(result).to_dataframe().set_index(["measurement_id", "temperature_C"])
    single = selected.loc[selected.contributor_count.eq(1)] if method == "average" else selected[:0]
    for row in single.itertuples():
        expected = individual.loc[(row.source_measurement_id, row.temperature_C)]
        np.testing.assert_allclose(
            [row.concentration, row.lower_error, row.upper_error],
            [expected.concentration, expected.lower_error, expected.upper_error],
            rtol=0,
            atol=0,
        )
    assert result.experiment.water_blank_map == {"neat": ["water"], "diluted": ["water"]}


@pytest.mark.parametrize("method", ["mle", "average"])
def test_excluded_temperatures_do_not_require_missing_blank_observations(method):
    original = make_experiment(
        {
            "sample": ("sample", 50, 1, [8, 16], 32),
            "diluted": ("sample", 50, 2, [4, 8], 32),
            "blank": ("blank", 50, 1, [1, 2], 10),
        },
        temperatures=(-5, -6),
    )
    counts = original.counts.to_dataframe()
    counts = counts.loc[~(counts.measurement_id.eq("blank") & counts.temperature_C.eq(-6))]
    source = inptk.Experiment(
        counts=inptk.CountsTable(counts),
        measurements=original.measurements,
        samples=original.samples,
        water_blank_map=original.water_blank_map,
    )
    ranges = {"sample": {"min_C": -5}, "diluted": {"min_C": -5}}
    actual = inptk.analyze_concentration(source, method=method, temperature_ranges_C=ranges)
    fractions = inptk.frozen_fraction(source)
    stepwise = inptk.estimate_concentration(
        fractions, experiment=source, method=method, temperature_ranges_C=ranges
    )
    pd.testing.assert_frame_equal(stepwise.to_dataframe(), fit_estimates(actual).to_dataframe())
    # A combined exclusion does not waive blank coverage for individual estimates.
    with pytest.raises(ValueError, match="no blank extrapolation"):
        inptk.cumulative_spectrum(fractions, experiment=source, method=method,
                                  temperature_ranges_C=ranges)
    assert (
        fit_estimates(actual).to_dataframe().set_index("temperature_C").loc[-6].contributor_count
        == 0
    )
    assert retained(actual).to_dataframe().temperature_C.tolist() == [-5]
    pd.testing.assert_frame_equal(source.counts.to_dataframe(), counts.reset_index(drop=True))
    with pytest.raises(ValueError, match="lacks observed temperature coverage"):
        inptk.analyze_concentration(source, method=method)


@pytest.mark.parametrize("method", ["mle", "average"])
def test_volume_scaled_blank_assumption_is_saved_without_changing_counts(tmp_path, method):
    source = make_experiment(
        {
            "sample": ("sample", 50, 1, [8, 16], 32),
            "blank": ("blank", 100, 1, [1, 2], 10),
        },
        temperatures=(-5, -6),
    )
    actual = inptk.analyze_concentration(source, method=method)
    actual.save(tmp_path / "volume_scaled.inptk")
    restored = inptk.load(tmp_path / "volume_scaled.inptk")
    assert restored.settings["water_blank_model"] == "volume_scaled"
    assert fit_estimates(restored).history[-1]["water_blank_model"] == "volume_scaled"
    assert input_spectra(restored).history[-1]["water_blank_model"] == "volume_scaled"
    assert restored.experiment.water_blank_map == source.water_blank_map
    pd.testing.assert_frame_equal(
        source.counts.to_dataframe(), restored.experiment.counts.to_dataframe()
    )
    pd.testing.assert_frame_equal(
        fit_estimates(actual).to_dataframe(), fit_estimates(restored).to_dataframe()
    )
