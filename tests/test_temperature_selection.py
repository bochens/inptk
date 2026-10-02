"""Observation selection retains the counts belonging to each chosen state."""

import pandas as pd
import pytest
from analysis_checks import sampled

import inptk


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


def temperature_selection_experiment():
    # Count-level blank correction can decrease both frozen and available counts.
    rows = pd.DataFrame(
        {
            "measurement_id": ["M"] * 6,
            "run_id": ["R1"] * 6,
            "cycle_id": ["01"] * 6,
            "time_s": range(6),
            "temperature_C": [0, -9, -9.994, -10, -10.006, -11.1],
            "n_total": [32, 32, 31, 20, 19, 18],
            "n_frozen": [0, 1, 11, 10, 9, 12],
        }
    )
    return inptk.read_counts(
        rows,
        metadata=[
            {
                "measurement_id": "M",
                "sample_id": "A",
                "run_id": "R1",
                "dilution": 1,
                "droplet_volume_uL": 50,
            }
        ],
    )


def test_native_fractions_keep_every_corrected_count_pair_and_measured_temperature():
    source = temperature_selection_experiment()
    original = source.counts.to_dataframe()
    fractions = inptk.frozen_fraction(source).to_dataframe()
    pd.testing.assert_frame_equal(fractions[original.columns], original)
    pd.testing.assert_series_equal(
        fractions.fraction_frozen, original.n_frozen / original.n_total, check_names=False
    )


def test_legacy_synthetic_window_rows_cannot_be_used_as_likelihood_observations():
    source = temperature_selection_experiment()
    fractions = inptk.FrozenFractionTable(source.counts.to_dataframe(), history=[{
        "operation": "frozen_fraction", "temperature_method": "window_max_count",
    }])
    for operation in (
        inptk.cumulative_spectrum, inptk.estimate_concentration, inptk.differential_spectrum,
    ):
        with pytest.raises(ValueError, match="synthetic warm zero rows are not raw measurements"):
            operation(fractions, experiment=source)


def test_frozen_fraction_accepts_counts_without_metadata():
    counts = experiment().counts
    fractions = inptk.frozen_fraction(counts)
    actual = fractions.to_dataframe().set_index(["measurement_id", "temperature_C"])
    observed = counts.to_dataframe()
    observed = observed.set_index(["measurement_id", "temperature_C"])
    pd.testing.assert_series_equal(
        actual.fraction_frozen.sort_index(),
        (observed.n_frozen / observed.n_total).sort_index(),
        check_names=False,
    )


def test_stepwise_spectra_reject_wrong_sample_or_run_context():
    source = experiment()
    fractions = inptk.frozen_fraction(source)
    for column in ("sample_id", "run_id"):
        changed = inptk.FrozenFractionTable(
            fractions.to_dataframe().assign(**{column: "different"})
        )
        for operation in (
            inptk.cumulative_spectrum,
            inptk.estimate_concentration,
            inptk.differential_spectrum,
        ):
            with pytest.raises(ValueError, match="identities disagree"):
                operation(changed, experiment=source)


def test_default_workflow_keeps_native_corrected_states_without_a_grid():
    source = temperature_selection_experiment()
    fractions = inptk.frozen_fraction(source)
    row = fractions.to_dataframe().set_index("temperature_C").loc[-10]
    assert (row.n_frozen, row.n_total) == (10, 20)
    result = inptk.analyze_concentration(source)
    assert sampled(result) is None
    assert result.settings["observation_processing"].startswith("native")
    pd.testing.assert_frame_equal(fractions.to_dataframe(), result.frozen_fraction.to_dataframe())


def test_single_observed_temperature_is_kept_without_rounding_to_a_threshold():
    counts = inptk.CountsTable(pd.DataFrame({
        "run_id": ["R"], "sample_id": ["S"], "cycle_id": ["1"], "measurement_id": ["M"],
        "temperature_C": [-10.2], "n_total": [32], "n_frozen": [1],
    }))
    fractions = inptk.frozen_fraction(counts).to_dataframe()
    assert fractions.temperature_C.tolist() == [-10.2]
    assert fractions.n_total.tolist() == [32]
    assert fractions.n_frozen.tolist() == [1]
