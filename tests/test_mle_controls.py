from __future__ import annotations

import numpy as np
import pytest

from inptk import _engine as engine


def _fraction_table(
    sample_id: str,
    *,
    sample_name: str,
    dilution: float,
    temperature_C: float = -5.0,
    n_frozen: float,
) -> engine.TemperatureFrozenFractionTable:
    metadata = engine.SampleMetadata(
        sample_id=sample_id,
        sample_name=sample_name,
        sample_long_name=sample_name,
        sample_type="other",
        well_volume_uL=50,
        dilution=dilution,
    )
    return engine.TemperatureFrozenFractionTable(
        sample_id=[sample_id],
        temperature_C=[temperature_C],
        n_total=[32],
        n_frozen=[n_frozen],
        metadata=metadata,
    )


def _fraction_table_rows(
    sample_id: str,
    *,
    sample_name: str,
    dilution: float,
    temperature_C: list[float],
    n_frozen: list[float],
    n_total: float | list[float] = 32,
) -> engine.TemperatureFrozenFractionTable:
    metadata = engine.SampleMetadata(
        sample_id=sample_id,
        sample_name=sample_name,
        sample_long_name=sample_name,
        sample_type="other",
        well_volume_uL=50,
        dilution=dilution,
    )
    return engine.TemperatureFrozenFractionTable(
        sample_id=[sample_id] * len(temperature_C),
        temperature_C=temperature_C,
        n_total=n_total if isinstance(n_total, list) else [n_total] * len(temperature_C),
        n_frozen=n_frozen,
        metadata=metadata,
    )


def _mle_value(*tables: engine.TemperatureFrozenFractionTable, **kwargs: object) -> float:
    result = engine.cumulative_spec_mle(list(tables), **kwargs)
    return float(result.to_dataframe()["value"].iloc[0])


def test_mle_temperature_eligibility_excludes_warm_high_dilution_event() -> None:
    low = _fraction_table("low", sample_name="filterA_1", dilution=1, n_frozen=0)
    high = _fraction_table("high", sample_name="filterA_1000", dilution=1000, n_frozen=1)

    unmasked = _mle_value(low, high)
    masked = _mle_value(
        low,
        high,
        temperature_eligibility_C={"high": -10.0},
        mask_mode="drop_rows",
    )

    assert unmasked > 0
    assert masked == 0


def test_mle_temperature_eligibility_requires_mask_mode() -> None:
    high = _fraction_table("high", sample_name="filterA_1000", dilution=1000, n_frozen=1)

    with pytest.raises(ValueError, match="mask_mode is required"):
        engine.cumulative_spec_mle([high], temperature_eligibility_C={"high": -10.0})


def test_mle_mask_mode_requires_temperature_eligibility() -> None:
    high = _fraction_table("high", sample_name="filterA_1000", dilution=1000, n_frozen=1)

    with pytest.raises(ValueError, match="mask_mode requires temperature_eligibility_C"):
        engine.cumulative_spec_mle([high], mask_mode="drop_rows")


def test_mle_rebase_counts_removes_warm_masked_frozen_baseline() -> None:
    high = _fraction_table_rows(
        "high",
        sample_name="filterA_1000",
        dilution=1000,
        temperature_C=[-5.0, -10.0],
        n_frozen=[5, 7],
    )

    drop_rows = _mle_value(
        high,
        temperature_eligibility_C={"high": -10.0},
        mask_mode="drop_rows",
    )
    rebased = _mle_value(
        high,
        temperature_eligibility_C={"high": -10.0},
        mask_mode="rebase_counts",
    )

    assert 0 < rebased < drop_rows


def test_mle_rebase_counts_drops_dilution_when_no_wells_remain() -> None:
    high = _fraction_table_rows(
        "high",
        sample_name="filterA_1000",
        dilution=1000,
        temperature_C=[-5.0, -10.0],
        n_frozen=[32, 32],
    )

    result = engine.cumulative_spec_mle(
        [high],
        temperature_eligibility_C={"high": -10.0},
        mask_mode="rebase_counts",
    )

    assert result.to_dataframe().empty


def test_mle_rebase_counts_uses_coldest_masked_baseline() -> None:
    high = _fraction_table_rows(
        "high",
        sample_name="filterA_1000",
        dilution=1000,
        temperature_C=[-5.0, -9.5, -10.0],
        n_frozen=[10, 5, 7],
    )

    rebased = _mle_value(
        high,
        temperature_eligibility_C={"high": -10.0},
        mask_mode="rebase_counts",
    )

    assert rebased > 0


def test_mle_rebase_counts_drops_rows_without_remaining_risk_set() -> None:
    high = _fraction_table_rows(
        "high",
        sample_name="filterA_1000",
        dilution=1000,
        temperature_C=[-5.0, -10.0, -20.0],
        n_total=[32, 32, 1],
        n_frozen=[2, 4, 1],
    )

    result = engine.cumulative_spec_mle(
        [high],
        temperature_eligibility_C={"high": -10.0},
        mask_mode="rebase_counts",
    ).to_dataframe()

    assert result["temperature_C"].tolist() == [-10.0]


def test_mle_direct_likelihood_weight_reduces_high_dilution_influence() -> None:
    low = _fraction_table("low", sample_name="filterA_1", dilution=1, n_frozen=0)
    high = _fraction_table("high", sample_name="filterA_1000", dilution=1000, n_frozen=1)

    unweighted = _mle_value(low, high)
    weighted = _mle_value(low, high, likelihood_weights={"high": 0.05})

    assert 0 < weighted < unweighted


def test_mle_action_count_weights_match_exponential_direct_weights() -> None:
    low = _fraction_table("low", sample_name="filterA_1", dilution=1, n_frozen=0)
    high = _fraction_table("high", sample_name="filterA_1000", dilution=1000, n_frozen=1)

    direct = _mle_value(low, high, likelihood_weights={"high": 0.25})
    from_actions = _mle_value(
        low,
        high,
        action_counts={"low": 0, "high": 2},
        action_weight_half_life=1,
    )

    assert np.isclose(from_actions, direct)


def test_mle_action_counts_require_decay_parameter() -> None:
    low = _fraction_table("low", sample_name="filterA_1", dilution=1, n_frozen=0)

    with pytest.raises(ValueError, match="requires action_weight"):
        engine.cumulative_spec_mle([low], action_counts={"low": 0})


@pytest.mark.parametrize("repeat", [1, 10])
def test_mle_rejects_pooling_repeated_temperature_states(repeat):
    table = _fraction_table_rows(
        "same-droplets",
        sample_name="S",
        dilution=1,
        temperature_C=[-5 - i / repeat for i in range(2 * repeat)],
        n_frozen=[16] * repeat + [8] * repeat,
    )
    with pytest.raises(ValueError, match="same droplets, not independent observations"):
        engine.cumulative_spec_mle(table, enforce_monotone=True)


def test_pointwise_mle_precision_does_not_improve_from_extra_temperature_rows():
    coarse = _fraction_table_rows(
        "same-droplets", sample_name="S", dilution=1,
        temperature_C=[-5, -6], n_frozen=[16, 8],
    )
    dense = _fraction_table_rows(
        "same-droplets", sample_name="S", dilution=1,
        temperature_C=[-5, -5.1, -6, -6.1], n_frozen=[16, 16, 8, 8],
    )
    coarse_result = engine.cumulative_spec_mle(coarse).to_dataframe()
    dense_result = engine.cumulative_spec_mle(dense).to_dataframe()
    common = dense_result[dense_result.temperature_C.isin([-5, -6])]
    for column in ("value", "lower_ci", "upper_ci"):
        np.testing.assert_allclose(coarse_result[column], common[column])
