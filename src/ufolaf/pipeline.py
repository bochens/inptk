from __future__ import annotations

from typing import Any, Literal

from .transforms import (
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
    method: Literal["max", "latest"] = "max",
    temperature_tolerance_C: float = 0.0,
    combine: CombineMode = "none",
    sample_group_by: Literal["sample_id", "sample_name", "sample_long_name"] | None = None,
    enforce_monotone: bool = False,
    z: float = 1.96,
    normalize: bool = False,
) -> Any:
    """Run the common count -> fraction -> spectrum workflow."""

    fraction = counts_to_temperature_frozen_fraction(
        counts,
        step_C=step_C,
        method=method,
        temperature_tolerance_C=temperature_tolerance_C,
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
        )
    elif combine == "none":
        result = temperature_frozen_fraction_to_cumulative_spectrum(fraction, z=z)
    else:
        raise ValueError("combine must be 'none', 'stitch', or 'mle'")

    if normalize:
        return cumulative_spectrum_to_normalized_inp_spectrum(result)
    return result
