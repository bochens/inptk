"""Stitching selects disjoint dilution ranges without changing selected estimates."""

import copy
import inspect

import numpy as np
import pandas as pd
import pytest

from inptk._engine.transforms import (
    _stitch_cumulative_group,
    temperature_frozen_fraction_to_stitched_cumulative_spectrum,
)


def _frame():
    rows = []
    for dilution, measurement, values, lower, upper in (
        (1, "M0", [10, 8, 20, 24], 2.0, 4.0),
        (3, "M1", [11, 9, 22, 25], 10.0, 12.0),
    ):
        for index, (temperature, value) in enumerate(zip((-5, -6, -7, -8), values)):
            rows.append({
                "temperature_C": temperature,
                "dilution_fold": dilution,
                "source_sample_id": measurement,
                "value": float(value),
                "lower_ci": lower,
                "upper_ci": upper,
                "n_total": 32,
                "n_frozen": 30 if dilution == 1 and index == 3 else 5,
            })
    return pd.DataFrame(rows)


def _stitch(frame, **options):
    return _stitch_cumulative_group(frame, "A", **options).set_index("temperature_C")


def test_handoff_copies_exactly_one_curve_and_its_error_bounds():
    frame = _frame()
    original = copy.deepcopy(frame)
    result = _stitch(frame)
    assert result.value.tolist() == [10, 8, 20, 25]
    assert result.dilution_fold.tolist() == [1, 1, 1, 3]
    assert result.source_measurement_id.tolist() == ["M0", "M0", "M0", "M1"]
    for temperature, row in result.iterrows():
        source = frame.loc[
            frame.temperature_C.eq(temperature)
            & frame.source_sample_id.eq(row.source_measurement_id)
        ].iloc[0]
        assert row.value == source.value
        assert row.lower_ci == source.lower_ci
        assert row.upper_ci == source.upper_ci
        assert row.contributing_measurement_ids == [row.source_measurement_id]
        assert row.selection_status == "selected"
    pd.testing.assert_frame_equal(frame, original)


def test_decrease_is_preserved_for_the_separate_final_selection_step():
    result = _stitch(_frame())
    # The other dilution is also available, but a decrease is not an automatic
    # trigger to average, refit, create a plateau, or switch the selected range.
    assert result.loc[-5, "value"] == 10
    assert result.loc[-6, "value"] == 8
    assert result.loc[-7, "value"] == 20
    assert result.loc[-6, "source_measurement_id"] == "M0"


def test_fully_overlapping_eligible_curves_use_only_the_least_diluted():
    frame = _frame()
    frame["n_frozen"] = 5
    result = _stitch(frame)
    assert result.value.tolist() == [10, 8, 20, 24]
    assert result.source_measurement_id.eq("M0").all()


def test_cutoff_is_inclusive_and_changes_only_the_temperature_handoff():
    frame = _frame()
    frame.loc[frame.dilution_fold.eq(1) & frame.temperature_C.eq(-7), "n_frozen"] = 29
    default = _stitch(frame)
    stricter = _stitch(frame, min_unfrozen=4)
    assert default.loc[-7, "source_measurement_id"] == "M0"
    assert stricter.loc[-7, "source_measurement_id"] == "M1"
    assert stricter.loc[-7, "value"] == 22


def test_missing_internal_point_remains_a_gap_despite_another_available_curve():
    frame = _frame()
    frame = frame.loc[~(frame.dilution_fold.eq(1) & frame.temperature_C.eq(-6))]
    result = _stitch(frame)
    assert np.isnan(result.loc[-6, "value"])
    assert result.loc[-6, "selection_status"] == "temperature_unavailable"
    assert result.loc[-6, "contributing_measurement_ids"] == []
    assert result.loc[-6, "source_measurement_id"] == ""
    assert result.loc[-7, "source_measurement_id"] == "M0"
    assert result.loc[-8, "source_measurement_id"] == "M1"


@pytest.mark.parametrize("column,value,status", [
    ("n_frozen", 31, "insufficient_unfrozen"),
    ("value", np.inf, "nonfinite_concentration"),
])
def test_ineligible_internal_points_are_flagged_without_an_early_switch(column, value, status):
    frame = _frame()
    frame.loc[frame.dilution_fold.eq(1) & frame.temperature_C.eq(-6), column] = value
    result = _stitch(frame)
    assert np.isnan(result.loc[-6, "value"])
    assert result.loc[-6, "selection_status"] == status
    assert result.loc[-6, "qc_flag"] & 1
    assert result.loc[-6, "contributing_measurement_ids"] == []
    assert result.loc[-7, "source_measurement_id"] == "M0"


def test_three_dilutions_switch_only_to_colder_ranges():
    frame = _frame()
    additional = []
    for index, temperature in enumerate((-5, -6, -7, -8, -9, -10)):
        row = frame.iloc[-1].to_dict()
        row.update(
            temperature_C=temperature, dilution_fold=9, source_sample_id="M2",
            value=float(40 + index), lower_ci=3.0, upper_ci=5.0,
        )
        additional.append(row)
    result = _stitch(pd.concat([frame, pd.DataFrame(additional)], ignore_index=True))
    assert result.dilution_fold.tolist() == [1, 1, 1, 3, 9, 9]
    assert result.value.tolist() == [10, 8, 20, 25, 44, 45]
    assert result.contributing_measurement_ids.tolist() == [
        ["M0"], ["M0"], ["M0"], ["M1"], ["M2"], ["M2"]
    ]


def test_no_eligible_least_dilution_uses_the_next_curve_from_the_start():
    frame = _frame()
    frame.loc[frame.dilution_fold.eq(1), "n_frozen"] = 31
    result = _stitch(frame)
    assert result.value.tolist() == [11, 9, 22, 25]
    assert result.source_measurement_id.eq("M1").all()


def test_nonfinite_uncertainty_is_preserved_and_flagged():
    frame = _frame()
    frame.loc[frame.dilution_fold.eq(1) & frame.temperature_C.eq(-5), "upper_ci"] = np.nan
    result = _stitch(frame)
    assert result.loc[-5, "value"] == 10
    assert np.isnan(result.loc[-5, "upper_ci"])
    assert result.loc[-5, "qc_flag"] & 1
    assert result.loc[-5, "contributing_measurement_ids"] == ["M0"]


def test_no_overlap_refit_or_value_enforcement_controls_remain():
    for function in (
        _stitch_cumulative_group,
        temperature_frozen_fraction_to_stitched_cumulative_spectrum,
    ):
        parameters = inspect.signature(function).parameters
        assert not {"overlap_points", "refit", "enforce_monotone"} & parameters.keys()


def test_middle_dilution_without_eligible_colder_points_does_not_erase_prior_range():
    frame = _frame()
    frame.loc[frame.dilution_fold.eq(3) & frame.temperature_C.eq(-8), "n_frozen"] = 31
    third = frame.loc[frame.dilution_fold.eq(3)].copy()
    third["dilution_fold"] = 9
    third["source_sample_id"] = "M2"
    third["n_frozen"] = 5
    third["value"] = [40.0, 41.0, 42.0, 43.0]
    result = _stitch(pd.concat([frame, third], ignore_index=True))
    assert result.value.tolist() == [10, 8, 20, 43]
    assert result.source_measurement_id.tolist() == ["M0", "M0", "M0", "M2"]
    assert result.dilution_fold.tolist() == [1, 1, 1, 9]
