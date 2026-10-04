from __future__ import annotations

import pytest

from inptk import _engine as engine


def _fraction_table(
    sample_id: str,
    *,
    sample_name: str,
    dilution: float = 1.0,
    air_volume_L: float | None = None,
    filter_fraction_used: float | None = None,
    suspension_volume_mL: float | None = None,
) -> engine.TemperatureFrozenFractionTable:
    metadata = engine.SampleMetadata(
        sample_id=sample_id,
        sample_name=sample_name,
        sample_long_name=sample_name,
        sample_type="other",
        well_volume_uL=50,
        dilution=dilution,
        air_volume_L=air_volume_L,
        filter_fraction_used=filter_fraction_used,
        suspension_volume_mL=suspension_volume_mL,
    )
    return engine.TemperatureFrozenFractionTable(
        sample_id=[sample_id],
        temperature_C=[-5.0],
        n_total=[10],
        n_frozen=[2],
        metadata=metadata,
    )


def test_stitch_reports_duplicate_source_sample_temperatures() -> None:
    first = _fraction_table("sample-1", sample_name="filterA_1")
    second = _fraction_table("sample-1", sample_name="filterA_1")

    with pytest.raises(ValueError, match="same source sample and temperature"):
        engine.cumulative_spec_stitch([first, second], sample_group_by="sample_name")


def test_stitch_reports_repeated_group_temperature_dilution() -> None:
    first = _fraction_table("run-1", sample_name="filterA_1")
    second = _fraction_table("run-2", sample_name="filterA_1")

    with pytest.raises(ValueError, match="repeated temperature/dilution rows"):
        engine.cumulative_spec_stitch([first, second])


def test_stitch_warns_on_mismatched_combined_metadata() -> None:
    first = _fraction_table(
        "run-1",
        sample_name="filterA_1",
        dilution=1,
        air_volume_L=1000,
        filter_fraction_used=0.5,
        suspension_volume_mL=10,
    )
    second = _fraction_table(
        "run-2",
        sample_name="filterA_10",
        dilution=10,
        air_volume_L=900,
        filter_fraction_used=0.25,
        suspension_volume_mL=8,
    )

    with pytest.warns(UserWarning) as captured:
        engine.cumulative_spec_stitch([first, second])

    messages = [str(warning.message) for warning in captured]
    assert any("vol_air_filt" in message for message in messages)
    assert any("proportion_filter_used" in message for message in messages)
    assert any("vol_susp" in message for message in messages)


def test_mle_warns_on_mismatched_combined_metadata() -> None:
    first = _fraction_table("run-1", sample_name="filterA_1", dilution=1, air_volume_L=1000)
    second = _fraction_table("run-2", sample_name="filterA_10", dilution=10, air_volume_L=900)

    with pytest.warns(UserWarning, match="mle group 'filterA'.*vol_air_filt"):
        engine.cumulative_spec_mle([first, second])
