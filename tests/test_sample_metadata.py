from __future__ import annotations

import pytest

from inptk import _engine as engine


def test_sample_type_rejects_unknown_values() -> None:
    with pytest.raises(ValueError, match="Unknown sample_type 'filter'"):
        engine.SampleMetadata(sample_type="filter")


def test_sample_type_keeps_blank_as_other() -> None:
    assert engine.SampleMetadata(sample_type="").sample_type == "other"
    assert engine.SampleMetadata(sample_type=None).sample_type == "other"


def test_sample_type_accepts_known_values_case_insensitively() -> None:
    assert engine.SampleMetadata(sample_type=" Air ").sample_type == "air"
    assert engine.SampleMetadata(sample_type="SOIL").sample_type == "soil"
    assert engine.SampleMetadata(sample_type="other").sample_type == "other"
