"""Public method choices must change only the controls the caller selected."""

import json

import numpy as np
import pandas as pd
import pytest

import inptk
from inptk.cli import main


def experiment(counts=None, scopes=(("R1", "01", 0),)):
    counts = counts or {1: [1, 8, 17, 19], 10: [0, 1, 3, 7]}
    rows, metadata = [], []
    for run in dict.fromkeys(run for run, _, _ in scopes):
        for dilution in counts:
            metadata.append(
                {
                    "measurement_id": f"{run}_{dilution}",
                    "sample_id": "A",
                    "run_id": run,
                    "dilution": dilution,
                    "droplet_volume_uL": 50,
                }
            )
    for run, cycle, offset in scopes:
        for dilution, frozen in counts.items():
            for temperature, count in zip((-5, -6, -7, -8), frozen):
                rows.append(
                    {
                        "measurement_id": f"{run}_{dilution}",
                        "run_id": run,
                        "cycle_id": cycle,
                        "temperature_C": temperature,
                        "n_total": 20,
                        "n_frozen": min(20, count + offset),
                    }
                )
    return inptk.read_counts(pd.DataFrame(rows), metadata=metadata)


def analyze(source, method, **kwargs):
    return inptk.analyze_concentration(
        source, dilution_method=method, step_C=1, temperature_method="latest", **kwargs
    )


def spectrum(result):
    return result.combined.to_dataframe().set_index("temperature_C").sort_index()


@pytest.mark.parametrize("name", ["stitch", "mle"])
def test_default_config_matches_string_method(name):
    source = experiment()
    config = inptk.Stitch() if name == "stitch" else inptk.MLE()
    default, explicit = analyze(source, name), analyze(source, config)
    for table in ("frozen_fraction", "per_dilution", "combined", "final"):
        pd.testing.assert_frame_equal(
            getattr(default, table).to_dataframe(), getattr(explicit, table).to_dataframe()
        )
    assert explicit.settings == default.settings
    assert explicit.settings["dilution_method"] == name


def test_minimum_unfrozen_count_moves_the_join_at_the_count_boundary():
    source = experiment()
    usual = spectrum(analyze(source, inptk.Stitch(overlap_points=0)))
    stricter = spectrum(analyze(source, inptk.Stitch(min_unfrozen=4, overlap_points=0)))
    assert usual.loc[-7, "dilution_fold"] == 1  # Exactly three unfrozen droplets.
    assert stricter.loc[-7, "dilution_fold"] == 10
    assert usual.loc[-8, "dilution_fold"] == stricter.loc[-8, "dilution_fold"] == 10
    assert usual.loc[-6, "concentration"] == stricter.loc[-6, "concentration"]


def test_zero_overlap_disables_correction_but_preserves_the_main_switch():
    source = experiment({1: [1, 12, 1, 19], 10: [0, 1, 4, 7]})
    usual = spectrum(analyze(source, inptk.Stitch()))
    unadjusted_result = analyze(source, inptk.Stitch(overlap_points=0))
    unadjusted = spectrum(unadjusted_result)
    per_neat = unadjusted_result.per_dilution.select(measurement_id="R1_1").to_dataframe()
    per_neat = per_neat.set_index("temperature_C")
    assert unadjusted.loc[-7, "concentration"] == per_neat.loc[-7, "concentration"]
    assert usual.loc[-7, "dilution_fold"] == 10
    assert unadjusted.loc[-7, "dilution_fold"] == 1
    assert usual.loc[-8, "dilution_fold"] == unadjusted.loc[-8, "dilution_fold"] == 10


def test_single_dilution_bypasses_automatic_stitch_cutoff():
    source = experiment({1: [1, 8, 17, 19]})
    result = analyze(source, inptk.Stitch(min_unfrozen=20, overlap_points=0))
    per = result.per_dilution.to_dataframe().set_index("temperature_C")
    pd.testing.assert_frame_equal(
        spectrum(result)[["concentration", "lower_error", "upper_error"]],
        per[["concentration", "lower_error", "upper_error"]].sort_index(),
    )


def test_manual_switch_boundaries_copy_selected_values_and_uncertainty():
    source = experiment({1: [19, 19, 19, 19], 10: [0, 1, 3, 7], 100: [0, 0, 1, 2]})
    result = analyze(source, inptk.ManualStitch(switch_temperatures_C=[-6, -7]))
    selected = spectrum(result)
    assert selected.loc[[-5, -6, -7, -8], "dilution_fold"].tolist() == [1, 10, 100, 100]
    assert np.isfinite(selected.loc[-5, "concentration"])  # One unfrozen remains usable.
    per = result.per_dilution.to_dataframe().set_index(["measurement_id", "temperature_C"])
    for temperature, row in selected.iterrows():
        measurement = f"R1_{int(row.dilution_fold)}"
        assert row.source_measurement_id == measurement
        np.testing.assert_allclose(
            row[["concentration", "lower_error", "upper_error"]].to_numpy(dtype=float),
            per.loc[
                (measurement, temperature), ["concentration", "lower_error", "upper_error"]
            ].to_numpy(dtype=float),
        )


def test_manual_missing_selected_temperature_does_not_fall_back():
    source = experiment()
    rows = source.counts.to_dataframe()
    rows = rows[~((rows.measurement_id == "R1_10") & (rows.temperature_C > -7))]
    incomplete = inptk.read_counts(rows, metadata=list(source.measurements.values()))
    result = analyze(incomplete, inptk.ManualStitch(switch_temperatures_C=[-6]))
    assert set(spectrum(result).index) == {-5, -6, -7, -8}
    assert np.isnan(spectrum(result).loc[-6, "concentration"])
    assert np.isfinite(spectrum(result).loc[-7, "concentration"])
    assert result.warnings


def test_manual_nonfinite_selected_estimate_is_missing():
    source = experiment({1: [1, 8, 17, 19], 10: [0, 1, 3, 20]})
    result = analyze(source, inptk.ManualStitch(switch_temperatures_C=[-7]))
    assert np.isnan(spectrum(result).loc[-8, "concentration"])
    assert result.warnings


def test_manual_keeps_runs_and_cycles_separate():
    source = experiment(scopes=(("R1", "01", 0), ("R1", "02", 1), ("R2", "01", 2)))
    result = analyze(source, inptk.ManualStitch(switch_temperatures_C=[-7]))
    final = result.final.to_dataframe()
    assert len(final.groupby(["run_id", "sample_id", "cycle_id"])) == 3
    per = result.per_dilution.to_dataframe()
    for identity, group in final.groupby(["run_id", "cycle_id"]):
        run, cycle = identity
        expected = per[
            (per.run_id == run)
            & (per.cycle_id == cycle)
            & (per.measurement_id == f"{run}_10")
            & (per.temperature_C == -7)
        ]
        actual = group.loc[group.temperature_C == -7, "concentration"].iloc[0]
        assert actual == expected.concentration.iloc[0]


def test_manual_rejects_absent_dilution_in_one_cycle_and_monotone_adjustment():
    source = experiment(scopes=(("R1", "01", 0), ("R1", "02", 0)))
    rows = source.counts.to_dataframe()
    rows = rows[~((rows.cycle_id == "02") & (rows.measurement_id == "R1_10"))]
    incomplete = inptk.read_counts(rows, metadata=list(source.measurements.values()))
    with pytest.raises(ValueError):
        analyze(incomplete, inptk.ManualStitch(switch_temperatures_C=[-7]))
    with pytest.raises(ValueError):
        analyze(source, inptk.ManualStitch(switch_temperatures_C=[-7]), enforce_monotone=True)


def test_manual_rejects_multiple_measurements_of_the_selected_dilution():
    source = experiment()
    rows = source.counts.to_dataframe()
    duplicate = rows[rows.measurement_id == "R1_1"].assign(measurement_id="replicate")
    metadata = [vars(value) for value in source.measurements.values()]
    metadata.append(dict(metadata[0], measurement_id="replicate"))
    repeated = inptk.read_counts(pd.concat([rows, duplicate]), metadata=metadata)
    with pytest.raises(ValueError):
        analyze(repeated, inptk.ManualStitch(switch_temperatures_C=[-7]))


def test_manual_requires_the_same_dilution_factors_across_runs():
    source = experiment(scopes=(("R1", "01", 0), ("R2", "01", 0)))
    metadata = [dict(vars(value)) for value in source.measurements.values()]
    for record in metadata:
        if record["measurement_id"] == "R2_10":
            record["dilution"] = 20
    source = inptk.read_counts(source.counts.to_dataframe(), metadata=metadata)
    with pytest.raises(ValueError):
        analyze(source, inptk.ManualStitch(switch_temperatures_C=[-7]))


def test_manual_zero_switches_support_one_dilution():
    result = analyze(experiment({1: [1, 8, 17, 19]}), inptk.ManualStitch(switch_temperatures_C=[]))
    assert spectrum(result).dilution_fold.eq(1).all()


def test_mle_controls_are_effective_and_save_with_json_safe_keys(tmp_path):
    source = experiment()
    method = inptk.MLE(
        temperature_eligibility_C={10: -6},
        mask_mode="drop_rows",
        dilution_likelihood_weights={10: 0.25},
        confidence_drop=1.1,
    )
    result = analyze(source, method)
    options = result.settings["method_options"]
    assert options["confidence_drop"] == 1.1
    assert all(isinstance(key, str) for key in options["temperature_eligibility_C"])
    assert all(isinstance(key, str) for key in options["dilution_likelihood_weights"])
    assert json.loads(json.dumps(options, allow_nan=False)) == options
    result.save(tmp_path / "analysis")
    restored = inptk.load(tmp_path / "analysis")
    assert restored.settings == result.settings
    rerun = analyze(source, inptk.MLE(**restored.settings["method_options"]))
    pd.testing.assert_frame_equal(spectrum(rerun), spectrum(result))
    unweighted = spectrum(analyze(source, inptk.MLE(confidence_drop=1.1)))
    assert spectrum(result).loc[-6, "concentration"] < unweighted.loc[-6, "concentration"]


def test_mle_action_weights_match_direct_weights_and_z_sets_default_confidence():
    source = experiment()
    direct = analyze(source, inptk.MLE(dilution_likelihood_weights={10: 0.25}), z=2.3)
    actions = analyze(
        source, inptk.MLE(dilution_action_counts={10: 2}, action_weight_half_life=1), z=2.3
    )
    pd.testing.assert_frame_equal(spectrum(direct), spectrum(actions))
    assert direct.settings["method_options"]["confidence_drop"] == pytest.approx(2.3**2 / 2)


@pytest.mark.parametrize(
    "name, options",
    [
        ("Stitch", {"min_unfrozen": -1}),
        ("Stitch", {"min_unfrozen": 2.5}),
        ("Stitch", {"overlap_points": -1}),
        ("Stitch", {"overlap_points": True}),
        ("MLE", {"mask_mode": "ignore"}),
        ("MLE", {"temperature_eligibility_C": {10: -6}}),
        ("MLE", {"mask_mode": "drop_rows"}),
        ("MLE", {"dilution_likelihood_weights": {10: 0}}),
        ("MLE", {"dilution_likelihood_weights": {10: np.inf}}),
        ("MLE", {"dilution_likelihood_weights": {"unknown": 0.5}}),
        ("MLE", {"dilution_action_counts": {10: -1}, "action_weight_half_life": 1}),
        ("MLE", {"dilution_action_counts": {10: 2}}),
        ("MLE", {"confidence_drop": 0}),
        ("ManualStitch", {"switch_temperatures_C": [-7, -6]}),
        ("ManualStitch", {"switch_temperatures_C": [-7, -7]}),
        ("ManualStitch", {"switch_temperatures_C": [np.nan]}),
        ("ManualStitch", {"switch_temperatures_C": [-6, -7]}),
    ],
)
def test_invalid_method_options_are_rejected(name, options):
    with pytest.raises((TypeError, ValueError)):
        analyze(experiment(), getattr(inptk, name)(**options))


@pytest.mark.parametrize("method", [42, {}, "unknown", "manual"])
def test_invalid_method_choice_is_rejected(method):
    with pytest.raises((TypeError, ValueError)):
        analyze(experiment(), method)


@pytest.mark.parametrize(
    "name, options",
    [
        ("stitch", {"min_unfrozen": 4, "overlap_points": 0}),
        ("mle", {"dilution_likelihood_weights": {"10": 0.25}, "confidence_drop": 1.1}),
        ("manual", {"switch_temperatures_C": [-7]}),
    ],
)
@pytest.mark.parametrize("from_file", [False, True])
def test_cli_method_options_match_python(tmp_path, name, options, from_file):
    source = experiment()
    source.save(tmp_path / "source")
    payload = json.dumps(options)
    if from_file:
        option_path = tmp_path / "options.json"
        option_path.write_text(payload)
        payload = str(option_path)
    assert (
        main(
            [
                "analyze",
                str(tmp_path / "source"),
                "--format",
                "saved",
                "--out",
                str(tmp_path / "result"),
                "--dilution-method",
                name,
                "--method-options",
                payload,
                "--step-C",
                "1",
                "--temperature-method",
                "latest",
            ]
        )
        == 0
    )
    method_type = {"stitch": inptk.Stitch, "mle": inptk.MLE, "manual": inptk.ManualStitch}[name]
    expected = analyze(source, method_type(**options))
    actual = inptk.load(tmp_path / "result")
    pd.testing.assert_frame_equal(actual.final.to_dataframe(), expected.final.to_dataframe())
    assert actual.settings == expected.settings


@pytest.mark.parametrize("options", [{"unknown": 1}, {"confidence_drop": 1.1}])
def test_cli_rejects_unknown_or_mismatched_method_options(tmp_path, options):
    experiment().save(tmp_path / "source")
    with pytest.raises(SystemExit) as error:
        main(
            [
                "analyze",
                str(tmp_path / "source"),
                "--format",
                "saved",
                "--out",
                str(tmp_path / "result"),
                "--dilution-method",
                "stitch",
                "--method-options",
                json.dumps(options),
            ]
        )
    assert error.value.code != 0
    assert not (tmp_path / "result").exists()


@pytest.mark.parametrize("name", ["stitch", "mle", "manual"])
def test_stepwise_calculation_matches_full_workflow(name):
    source = experiment()
    metadata = [
        dict(
            vars(measurement),
            sample_type="air",
            air_volume_L=100,
            suspension_volume_mL=5,
            filter_fraction_used=1,
        )
        for measurement in source.measurements.values()
    ]
    source = inptk.read_counts(source.counts.to_dataframe(), metadata=metadata)
    method = {
        "stitch": inptk.Stitch(min_unfrozen=4, overlap_points=0),
        "mle": inptk.MLE(dilution_likelihood_weights={10: 0.25}),
        "manual": inptk.ManualStitch(switch_temperatures_C=[-7]),
    }[name]
    fractions = inptk.frozen_fraction(source, step_C=1, temperature_method="latest")
    per_dilution = inptk.cumulative_spectrum(fractions, experiment=source, z=2.3)
    combined = inptk.combine_dilutions(fractions, experiment=source, method=method, z=2.3)
    differential = inptk.differential_spectrum(fractions, experiment=source)
    blank = inptk.CumulativeSpectrumTable(
        combined.to_dataframe().assign(
            sample_id="blank", concentration=0.1, lower_error=0.01, upper_error=0.02
        )
    )
    corrected = inptk.subtract_blanks(combined, {"A": blank})
    final = inptk.convert_concentration(corrected, source.samples, basis="sampled_air")
    full = analyze(
        source,
        method,
        z=2.3,
        differential=True,
        blank_by_sample={"A": blank},
        output_basis="sampled_air",
    )
    for step, expected in (
        (fractions, full.frozen_fraction),
        (per_dilution, full.per_dilution),
        (combined, full.combined),
        (differential, full.differential),
        (final, full.final),
    ):
        pd.testing.assert_frame_equal(step.to_dataframe(), expected.to_dataframe())


def test_frozen_fraction_accepts_counts_without_metadata():
    counts = experiment().counts
    fractions = inptk.frozen_fraction(counts, step_C=1, temperature_method="latest")
    actual = fractions.to_dataframe().set_index(["measurement_id", "temperature_C"])
    observed = counts.to_dataframe().astype({"temperature_C": float})
    observed = observed.set_index(["measurement_id", "temperature_C"])
    pd.testing.assert_series_equal(
        actual.fraction_frozen.sort_index(),
        (observed.n_frozen / observed.n_total).sort_index(),
        check_names=False,
    )


def test_stepwise_spectra_reject_wrong_sample_or_run_context():
    source = experiment()
    fractions = inptk.frozen_fraction(source, step_C=1)
    for column in ("sample_id", "run_id"):
        changed = inptk.FrozenFractionTable(
            fractions.to_dataframe().assign(**{column: "different"})
        )
        for operation in (
            inptk.cumulative_spectrum,
            inptk.combine_dilutions,
            inptk.differential_spectrum,
        ):
            with pytest.raises(ValueError, match="identities disagree"):
                operation(changed, experiment=source)


def test_stepwise_table_exposes_combination_warnings():
    source = experiment({1: [1, 8, 17, 19], 10: [0, 1, 3, 20]})
    method = inptk.ManualStitch(switch_temperatures_C=[-7])
    fractions = inptk.frozen_fraction(source, step_C=1, temperature_method="latest")
    combined = inptk.combine_dilutions(fractions, experiment=source, method=method)
    full = analyze(source, method)
    assert combined.warnings
    assert combined.warnings == full.warnings
    assert combined.select(temperature_C=-8).warnings == combined.warnings
