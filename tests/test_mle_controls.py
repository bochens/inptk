from __future__ import annotations

import numpy as np
import pytest

import ufolaf


def _fraction_table(
    sample_id: str,
    *,
    sample_name: str,
    dilution: float,
    temperature_C: float = -5.0,
    n_frozen: float,
) -> ufolaf.TemperatureFrozenFractionTable:
    metadata = ufolaf.SampleMetadata(
        sample_id=sample_id,
        sample_name=sample_name,
        sample_long_name=sample_name,
        sample_type="other",
        well_volume_uL=50,
        dilution=dilution,
    )
    return ufolaf.TemperatureFrozenFractionTable(
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
) -> ufolaf.TemperatureFrozenFractionTable:
    metadata = ufolaf.SampleMetadata(
        sample_id=sample_id,
        sample_name=sample_name,
        sample_long_name=sample_name,
        sample_type="other",
        well_volume_uL=50,
        dilution=dilution,
    )
    return ufolaf.TemperatureFrozenFractionTable(
        sample_id=[sample_id] * len(temperature_C),
        temperature_C=temperature_C,
        n_total=n_total if isinstance(n_total, list) else [n_total] * len(temperature_C),
        n_frozen=n_frozen,
        metadata=metadata,
    )


def _mle_value(*tables: ufolaf.TemperatureFrozenFractionTable, **kwargs: object) -> float:
    result = ufolaf.cumulative_spec_mle(list(tables), **kwargs)
    return float(result.to_dataframe()["value"].iloc[0])


def test_mle_temperature_eligibility_excludes_warm_high_dilution_event() -> None:
    low = _fraction_table("low", sample_name="filterA_1", dilution=1, n_frozen=0)
    high = _fraction_table("high", sample_name="filterA_1000", dilution=1000, n_frozen=1)

    unmasked = _mle_value(low, high)
    masked = _mle_value(
        low,
        high,
        temperature_eligibility_C={1000: -10.0},
        mask_mode="drop_rows",
    )

    assert unmasked > 0
    assert masked == 0


def test_mle_temperature_eligibility_requires_mask_mode() -> None:
    high = _fraction_table("high", sample_name="filterA_1000", dilution=1000, n_frozen=1)

    with pytest.raises(ValueError, match="mask_mode is required"):
        ufolaf.cumulative_spec_mle([high], temperature_eligibility_C={1000: -10.0})


def test_mle_mask_mode_requires_temperature_eligibility() -> None:
    high = _fraction_table("high", sample_name="filterA_1000", dilution=1000, n_frozen=1)

    with pytest.raises(ValueError, match="mask_mode requires temperature_eligibility_C"):
        ufolaf.cumulative_spec_mle([high], mask_mode="drop_rows")


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
        temperature_eligibility_C={1000: -10.0},
        mask_mode="drop_rows",
    )
    rebased = _mle_value(
        high,
        temperature_eligibility_C={1000: -10.0},
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

    result = ufolaf.cumulative_spec_mle(
        [high],
        temperature_eligibility_C={1000: -10.0},
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
        temperature_eligibility_C={1000: -10.0},
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

    result = ufolaf.cumulative_spec_mle(
        [high],
        temperature_eligibility_C={1000: -10.0},
        mask_mode="rebase_counts",
    ).to_dataframe()

    assert result["temperature_C"].tolist() == [-10.0]


def test_mle_direct_likelihood_weight_reduces_high_dilution_influence() -> None:
    low = _fraction_table("low", sample_name="filterA_1", dilution=1, n_frozen=0)
    high = _fraction_table("high", sample_name="filterA_1000", dilution=1000, n_frozen=1)

    unweighted = _mle_value(low, high)
    weighted = _mle_value(low, high, dilution_likelihood_weights={1000: 0.05})

    assert 0 < weighted < unweighted


def test_mle_action_count_weights_match_exponential_direct_weights() -> None:
    low = _fraction_table("low", sample_name="filterA_1", dilution=1, n_frozen=0)
    high = _fraction_table("high", sample_name="filterA_1000", dilution=1000, n_frozen=1)

    direct = _mle_value(low, high, dilution_likelihood_weights={1000: 0.25})
    from_actions = _mle_value(
        low,
        high,
        dilution_action_counts={1: 0, 1000: 2},
        action_weight_half_life=1,
    )

    assert np.isclose(from_actions, direct)


def test_mle_action_counts_require_decay_parameter() -> None:
    low = _fraction_table("low", sample_name="filterA_1", dilution=1, n_frozen=0)

    with pytest.raises(ValueError, match="requires action_weight"):
        ufolaf.cumulative_spec_mle([low], dilution_action_counts={1: 0})
