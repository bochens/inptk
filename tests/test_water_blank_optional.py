import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import inptk


@pytest.fixture
def incomplete_blank_source():
    rows, metadata = [], []
    for name, parent, dilution, temperatures, frozen in (
        ("M", "A", 1, [-5, -6, -7, -8], [0, 4, 8, 12]),
        ("D", "A", 10, [-5, -6, -7, -8], [0, 1, 2, 4]),
        ("W", "water", 1, [-5], [0]),
    ):
        metadata.append(
            {
                "measurement_id": name,
                "sample_id": parent,
                "dilution": dilution,
                "droplet_volume_uL": 50,
            }
        )
        for temperature, count in zip(temperatures, frozen, strict=True):
            rows.append(
                {
                    "measurement_id": name,
                    "cycle_id": "01",
                    "temperature_C": temperature,
                    "n_total": 20,
                    "n_frozen": count,
                }
            )
    counts = pd.DataFrame(rows)
    full = inptk.read_counts(counts, metadata=metadata, water_blank_map={"M": ["W"], "D": ["W"]})
    ordinary = inptk.read_counts(counts[counts.measurement_id.ne("W")], metadata=metadata[:2])
    return full, ordinary


@pytest.mark.parametrize("ranges", [None, {"M": {"min_C": -6}, "D": {"max_C": -6}}])
def test_disabling_correction_keeps_raw_context_and_matches_sample_only_analysis(
    tmp_path, incomplete_blank_source, ranges
):
    source, ordinary = incomplete_blank_source
    original = source.counts.to_dataframe()
    expected = inptk.analyze_concentration(
        ordinary, temperature_ranges_C=ranges, differential=True
    )
    actual = inptk.analyze_concentration(
        source,
        temperature_ranges_C=ranges,
        differential=True,
        water_blank_correction=False,
    )
    for name in (
        "frozen_fraction",
        "per_dilution",
        "combined",
        "final_candidates",
        "final",
        "differential",
    ):
        pd.testing.assert_frame_equal(
            getattr(actual, name).to_dataframe(), getattr(expected, name).to_dataframe()
        )
    assert actual.experiment is source
    assert actual.experiment.water_blank_map == {"M": ["W"], "D": ["W"]}
    pd.testing.assert_frame_equal(source.counts.to_dataframe(), original)
    assert actual.settings["water_blank_correction"] is False
    assert actual.settings["water_blank_correction_applied"] is False
    assert actual.combined.history[-1]["water_blank_correction"] is False
    actual.save(tmp_path / "disabled.inptk")
    restored = inptk.load(tmp_path / "disabled.inptk")
    assert restored.experiment.water_blank_map == source.water_blank_map
    pd.testing.assert_frame_equal(restored.experiment.counts.to_dataframe(), original)
    assert restored.settings["water_blank_correction"] is False
    with pytest.raises(ValueError, match="lacks observed temperature coverage"):
        inptk.analyze_concentration(source, temperature_ranges_C=ranges)


@pytest.mark.parametrize(
    "function", [inptk.cumulative_spectrum, inptk.combine_dilutions, inptk.differential_spectrum]
)
def test_stepwise_disable_does_not_require_blank_temperature_coverage(
    incomplete_blank_source, function
):
    source, ordinary = incomplete_blank_source
    fractions = inptk.frozen_fraction(source)
    sample_fractions = inptk.frozen_fraction(ordinary)
    expected = function(sample_fractions, experiment=ordinary)
    actual = function(fractions, experiment=source, water_blank_correction=False)
    pd.testing.assert_frame_equal(actual.to_dataframe(), expected.to_dataframe())
    correction_steps = [step for step in actual.history if "water_blank_correction" in step]
    assert correction_steps
    assert all(step["water_blank_correction"] is False for step in correction_steps)


@pytest.mark.parametrize("value", [0, 1, None, "false", np.bool_(False)])
@pytest.mark.parametrize(
    "function",
    [
        inptk.cumulative_spectrum,
        inptk.combine_dilutions,
        inptk.differential_spectrum,
        inptk.analyze_concentration,
    ],
)
def test_water_correction_setting_requires_boolean(incomplete_blank_source, function, value):
    source, _ = incomplete_blank_source
    with pytest.raises(TypeError, match="water_blank_correction must be a bool"):
        if function is inptk.analyze_concentration:
            function(source, water_blank_correction=value)
        else:
            function(inptk.frozen_fraction(source), experiment=source, water_blank_correction=value)


@pytest.mark.parametrize("z", [0, -1, np.nan, np.inf])
def test_raw_cumulative_validates_uncertainty_setting_before_blank_fit(incomplete_blank_source, z):
    source, _ = incomplete_blank_source
    with pytest.raises(ValueError, match="z must be finite and positive"):
        inptk.cumulative_spectrum(inptk.frozen_fraction(source), experiment=source, z=z)


def test_cli_can_disable_saved_raw_blank_correction_with_missing_blank_temperatures(
    tmp_path, incomplete_blank_source
):
    source, _ = incomplete_blank_source
    source.save(tmp_path / "raw.inptk")
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "inptk",
            "analyze",
            str(tmp_path / "raw.inptk"),
            "--format",
            "saved",
            "--no-water-blank-correction",
            "--out",
            str(tmp_path / "result.inptk"),
        ],
        env=dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
            PYTHONDONTWRITEBYTECODE="1",
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    actual = inptk.load(tmp_path / "result.inptk")
    expected = inptk.analyze_concentration(source, water_blank_correction=False)
    pd.testing.assert_frame_equal(actual.final.to_dataframe(), expected.final.to_dataframe())
    assert actual.experiment.water_blank_map == source.water_blank_map
    assert set(actual.experiment.counts.to_dataframe().measurement_id) == {"M", "D", "W"}
    assert set(actual.per_dilution.to_dataframe().measurement_id) == {"M", "D"}
    assert actual.settings["water_blank_correction"] is False


@pytest.mark.parametrize("correction", [False, True])
@pytest.mark.parametrize(
    "function", [inptk.cumulative_spectrum, inptk.combine_dilutions, inptk.differential_spectrum]
)
def test_water_blank_analysis_rejects_synthetic_window_rows_even_when_disabled(
    incomplete_blank_source, function, correction
):
    source, _ = incomplete_blank_source
    fractions = inptk.FrozenFractionTable(source.counts.to_dataframe(), history=[{
        "operation": "frozen_fraction", "temperature_method": "window_max_count",
    }])
    assert len(fractions) > 0
    with pytest.raises(ValueError, match="synthetic warm zero rows are not raw measurements"):
        function(fractions, experiment=source, water_blank_correction=correction)


@pytest.mark.parametrize("correction", [False, True])
def test_full_analysis_no_longer_accepts_temperature_selection_options(
    incomplete_blank_source, correction
):
    source, _ = incomplete_blank_source
    with pytest.raises(TypeError, match="temperature_method"):
        inptk.analyze_concentration(
            source, temperature_method="window_max_count", water_blank_correction=correction
        )
