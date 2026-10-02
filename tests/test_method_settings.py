"""Temperature ranges use exact measurement names and retain the original counts."""

import json
from types import MappingProxyType

import numpy as np
import pytest

import inptk
from inptk.methods import validate_combination_method, validate_temperature_ranges


def test_ranges_normalize_open_boundaries_without_inventing_omitted_measurements():
    actual = validate_temperature_ranges(
        {"01": {"min_C": -20}, "A": {"max_C": -5}, "B": {}},
        measurement_ids={"01", "A", "B", "omitted"},
    )
    assert actual == {
        "01": {"min_C": -20.0, "max_C": None},
        "A": {"min_C": None, "max_C": -5.0},
        "B": {"min_C": None, "max_C": None},
    }
    assert json.loads(json.dumps(actual, allow_nan=False)) == actual
    assert validate_temperature_ranges(None, measurement_ids={"A"}) == {}
    assert validate_temperature_ranges({}, measurement_ids={"A"}) == {}


def test_range_copy_accepts_readonly_mappings_and_normalizes_numpy_numbers():
    bounds = {"min_C": np.float64(-20), "max_C": np.int64(-5)}
    source = MappingProxyType({"A": MappingProxyType(bounds)})
    actual = validate_temperature_ranges(source, measurement_ids={"A"})
    assert all(type(value) is float for value in actual["A"].values())
    bounds["min_C"] = -30
    assert actual["A"]["min_C"] == -20
    actual["A"]["max_C"] = -1
    assert bounds["max_C"] == -5


def test_equal_boundaries_and_explicit_unlimited_boundaries_are_valid():
    assert validate_temperature_ranges(
        {"A": {"min_C": -10, "max_C": -10}, "B": {"min_C": None, "max_C": None}},
        measurement_ids={"A", "B"},
    ) == {
        "A": {"min_C": -10.0, "max_C": -10.0},
        "B": {"min_C": None, "max_C": None},
    }


@pytest.mark.parametrize("value", [[], "A", 1, True])
def test_ranges_require_a_mapping(value):
    with pytest.raises(TypeError, match="map measurement names"):
        validate_temperature_ranges(value, measurement_ids={"A"})


@pytest.mark.parametrize("name", [1, True, None, "", "  "])
def test_range_names_must_be_nonempty_strings(name):
    with pytest.raises(ValueError, match="non-empty measurement names"):
        validate_temperature_ranges({name: {}}, measurement_ids={"A"})


@pytest.mark.parametrize("name", ["A ", " a", "1", "blank", "parent"])
def test_range_names_must_exactly_match_supplied_sample_measurement_ids(name):
    with pytest.raises(ValueError, match="Unknown measurement"):
        validate_temperature_ranges({name: {}}, measurement_ids={"A", "01"})


@pytest.mark.parametrize("bounds", [None, -10, [], "-10", True])
def test_each_range_requires_a_mapping(bounds):
    with pytest.raises(TypeError, match="range object"):
        validate_temperature_ranges({"A": bounds}, measurement_ids={"A"})


@pytest.mark.parametrize("boundary", ["minimum", "max", "lower_C", 1, None])
def test_unknown_boundaries_are_rejected(boundary):
    with pytest.raises(ValueError, match="Unknown temperature range keys"):
        validate_temperature_ranges({"A": {boundary: -10}}, measurement_ids={"A"})


@pytest.mark.parametrize("boundary", ["min_C", "max_C"])
@pytest.mark.parametrize("value", [True, False, np.bool_(True), "-10", [], {}])
def test_boundaries_require_numbers_or_none(boundary, value):
    with pytest.raises(TypeError, match="finite number or None"):
        validate_temperature_ranges({"A": {boundary: value}}, measurement_ids={"A"})


@pytest.mark.parametrize("boundary", ["min_C", "max_C"])
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_boundaries_are_rejected(boundary, value):
    with pytest.raises(ValueError, match="must be finite"):
        validate_temperature_ranges({"A": {boundary: value}}, measurement_ids={"A"})


def test_reversed_boundaries_are_rejected_without_changing_input():
    source = {"A": {"min_C": -5, "max_C": -20}}
    with pytest.raises(ValueError, match="min_C <= max_C"):
        validate_temperature_ranges(source, measurement_ids={"A"})
    assert source == {"A": {"min_C": -5, "max_C": -20}}


def test_method_selector_classes_are_not_public():
    assert all(not hasattr(inptk, name) for name in ("MLE", "ManualStitch", "Stitch"))


@pytest.mark.parametrize("method", ["mle", "average"])
def test_supported_combination_methods_are_returned_unchanged(method):
    assert validate_combination_method(method) == method


@pytest.mark.parametrize("method", ["stitch", "manual", "single", "MLE", "average ", "", "mean"])
def test_other_combination_method_names_are_rejected(method):
    with pytest.raises(ValueError, match="method must be 'mle' or 'average'"):
        validate_combination_method(method)


@pytest.mark.parametrize("method", [None, True, 1, [], {}, np.array(["mle"])])
def test_combination_method_must_be_a_string(method):
    with pytest.raises(TypeError, match="method must be a string"):
        validate_combination_method(method)
