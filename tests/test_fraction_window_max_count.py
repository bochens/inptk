from __future__ import annotations

import numpy as np
import pytest

from inptk import _engine as ufolaf


def _first_freeze_near_threshold_counts() -> ufolaf.CountsTable:
    return ufolaf.CountsTable(
        sample_id=["sample-1"] * 6,
        temperature_C=[0.0, -6.9, -7.024, -7.14, -7.676, -8.008],
        n_total=[32, 32, 32, 32, 32, 32],
        n_frozen=[0, 0, 1, 2, 3, 4],
    )


def _cool_then_warm_counts() -> ufolaf.CountsTable:
    return ufolaf.CountsTable(
        sample_id=["sample-1"] * 5,
        temperature_C=[20.0, -10.0, -25.0, -10.0, -5.0],
        n_total=[32, 32, 32, 32, 32],
        n_frozen=[0, 2, 20, 20, 20],
        time_s=[0.0, 1.0, 2.0, 3.0, 4.0],
    )


def test_window_max_count_includes_first_frozen_rounded_row() -> None:
    fraction = ufolaf.fraction_frozen(
        _first_freeze_near_threshold_counts(),
        method="window_max_count",
        step_C=0.5,
        temperature_tolerance_C=0.01,
    ).to_dataframe()

    warm_zero_rows = fraction.loc[fraction["temperature_C"].isin([-5.0, -5.5, -6.0, -6.5])]
    assert len(warm_zero_rows) == 4
    assert np.allclose(warm_zero_rows["n_frozen"], 0.0)

    first_frozen_row = fraction.loc[fraction["temperature_C"] == -7.0].iloc[0]
    assert first_frozen_row["n_total"] == 32
    assert first_frozen_row["n_frozen"] == 1

    exact_band_row = fraction.loc[fraction["temperature_C"] == -8.0].iloc[0]
    assert exact_band_row["n_frozen"] == 4


def test_window_max_count_does_not_change_max_threshold_behavior() -> None:
    max_fraction = ufolaf.fraction_frozen(
        _first_freeze_near_threshold_counts(),
        method="max",
        step_C=0.5,
        temperature_tolerance_C=0.01,
    ).to_dataframe()

    max_row = max_fraction.loc[max_fraction["temperature_C"] == -7.0].iloc[0]
    assert max_row["n_frozen"] == 0


def test_fraction_defaults_to_cooling_only_before_threshold_reduction() -> None:
    fraction = ufolaf.fraction_frozen(
        _cool_then_warm_counts(),
        method="latest",
        step_C=5.0,
        temperature_tolerance_C=0.01,
    )
    frame = fraction.to_dataframe()

    threshold_row = frame.loc[frame["temperature_C"] == -10.0].iloc[0]
    assert threshold_row["n_frozen"] == 2
    assert "cooling_only" in str(threshold_row["temperature_bin_method"])
    assert fraction.processing_metadata.generated_by is not None
    assert fraction.processing_metadata.generated_by.parameters["cooling_only"] is True


def test_fraction_can_include_warming_rows_for_legacy_debugging() -> None:
    fraction = ufolaf.fraction_frozen(
        _cool_then_warm_counts(),
        method="latest",
        step_C=5.0,
        temperature_tolerance_C=0.01,
        cooling_only=False,
    ).to_dataframe()

    threshold_row = fraction.loc[fraction["temperature_C"] == -10.0].iloc[0]
    assert threshold_row["n_frozen"] == 20
    assert "cooling_only" not in str(threshold_row["temperature_bin_method"])


def test_window_max_count_rejects_nonpositive_step() -> None:
    with pytest.raises(ValueError, match="step_C must be positive"):
        ufolaf.fraction_frozen(
            _first_freeze_near_threshold_counts(),
            method="window_max_count",
            step_C=0.0,
        )
