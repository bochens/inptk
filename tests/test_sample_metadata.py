from __future__ import annotations

import pytest

import ufolaf


def test_sample_type_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unknown sample_type 'filter'"):
        ufolaf.SampleMetadata(sample_type="filter")


def test_sample_type_keeps_blank_as_other() -> None:
    assert ufolaf.SampleMetadata(sample_type="").sample_type == "other"
    assert ufolaf.SampleMetadata(sample_type=None).sample_type == "other"


def test_sample_type_accepts_known_values_case_insensitively() -> None:
    assert ufolaf.SampleMetadata(sample_type=" Air ").sample_type == "air"
    assert ufolaf.SampleMetadata(sample_type="SOIL").sample_type == "soil"
    assert ufolaf.SampleMetadata(sample_type="other").sample_type == "other"
