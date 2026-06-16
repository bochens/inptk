from __future__ import annotations

from typing import Any, Literal, Mapping

from .transforms import (
    MleMaskMode,
    TemperatureReductionMethod,
    counts_to_temperature_frozen_fraction,
    cumulative_spectrum_to_normalized_inp_spectrum,
    temperature_frozen_fraction_to_binomial_mle_cumulative_spectrum,
    temperature_frozen_fraction_to_cumulative_spectrum,
    temperature_frozen_fraction_to_stitched_cumulative_spectrum,
)


CombineMode = Literal["none", "stitch", "mle"]


def counts_to_spectrum(
    counts: Any,
    *,
    step_C: float = 0.5,
    method: TemperatureReductionMethod = "max",
    temperature_tolerance_C: float = 0.0,
    cooling_only: bool = True,
    combine: CombineMode = "none",
    sample_group_by: Literal["sample_id", "sample_name", "sample_long_name"] | None = None,
    enforce_monotone: bool = False,
    z: float = 1.96,
    normalize: bool = False,
    temperature_eligibility_C: Mapping[Any, float] | None = None,
    mask_mode: MleMaskMode | None = None,
    dilution_likelihood_weights: Mapping[Any, float] | None = None,
    dilution_action_counts: Mapping[Any, float] | None = None,
    action_weight_lambda: float | None = None,
    action_weight_half_life: float | None = None,
) -> Any:
    """Run the common count -> fraction -> spectrum workflow."""

    fraction = counts_to_temperature_frozen_fraction(
        counts,
        step_C=step_C,
        method=method,
        temperature_tolerance_C=temperature_tolerance_C,
        cooling_only=cooling_only,
    )
    if combine == "stitch":
        result = temperature_frozen_fraction_to_stitched_cumulative_spectrum(
            fraction,
            sample_group_by=sample_group_by,
            enforce_monotone=enforce_monotone,
            z=z,
        )
    elif combine == "mle":
        result = temperature_frozen_fraction_to_binomial_mle_cumulative_spectrum(
            fraction,
            sample_group_by=sample_group_by,
            enforce_monotone=enforce_monotone,
            temperature_eligibility_C=temperature_eligibility_C,
            mask_mode=mask_mode,
            dilution_likelihood_weights=dilution_likelihood_weights,
            dilution_action_counts=dilution_action_counts,
            action_weight_lambda=action_weight_lambda,
            action_weight_half_life=action_weight_half_life,
        )
    elif combine == "none":
        result = temperature_frozen_fraction_to_cumulative_spectrum(fraction, z=z)
    else:
        raise ValueError("combine must be 'none', 'stitch', or 'mle'")

    if normalize:
        return cumulative_spectrum_to_normalized_inp_spectrum(result)
    return result
