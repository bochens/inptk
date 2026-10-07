"""Independent no-blank likelihood solver used to check the production estimator."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

PROFILE_LIKELIHOOD_DROP_95 = 1.920729410347062


def as_float_array(values: Any, *, name: str) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} is required")
    return np.asarray(values, dtype=float)


def validate_counts(n_frozen: np.ndarray, n_total: np.ndarray) -> None:
    if np.any(~np.isfinite(n_frozen)) or np.any(~np.isfinite(n_total)):
        raise ValueError("n_frozen and n_total must be finite")
    if np.any(n_total <= 0):
        raise ValueError("n_total must be positive")
    if np.any(n_frozen < 0):
        raise ValueError("n_frozen cannot be negative")
    if np.any(n_frozen > n_total):
        raise ValueError("n_frozen cannot exceed n_total")


def binomial_poisson_log_likelihood(
    inp_per_mL: float,
    n_frozen: Any,
    n_total: Any,
    well_volume_uL: float,
    dilution: Any = 1.0,
    *,
    likelihood_weight: Any = 1.0,
    include_binomial_constant: bool = False,
) -> float:
    """Return binomial log-likelihood for frozen counts under Poisson occupancy.

    The observed variable is the frozen-well count. Poisson occupancy converts
    the candidate concentration K into a per-well frozen probability; the
    binomial PMF then evaluates the probability of observing n_frozen out of
    n_total wells.

    By default the binomial coefficient is omitted because it does not depend on
    K and therefore cancels for maximum likelihood estimation and profile
    likelihood confidence intervals. ``likelihood_weight`` raises each row's
    likelihood contribution to that power; weights less than 1 discount rows.
    """

    concentration = _scalar_nonnegative_concentration(inp_per_mL, name="inp_per_mL")
    frozen, total, dilution_array, weight = _weighted_count_likelihood_arrays(
        n_frozen,
        n_total,
        dilution,
        likelihood_weight,
    )
    occupancy = _poisson_occupancy_mean(concentration, well_volume_uL, dilution_array)
    unfrozen = total - frozen
    log_frozen_probability = _log_one_minus_exp_neg(occupancy)
    log_unfrozen_probability = -occupancy
    frozen_term = np.zeros_like(frozen, dtype=float)
    frozen_mask = frozen != 0
    frozen_term[frozen_mask] = frozen[frozen_mask] * log_frozen_probability[frozen_mask]
    unfrozen_term = np.zeros_like(unfrozen, dtype=float)
    unfrozen_mask = unfrozen != 0
    unfrozen_term[unfrozen_mask] = unfrozen[unfrozen_mask] * log_unfrozen_probability[unfrozen_mask]
    loglike = float(np.sum(weight * (frozen_term + unfrozen_term)))
    if include_binomial_constant:
        loglike += _binomial_log_constant(frozen, total, weight)
    return loglike


def binomial_poisson_mle_inp_per_ml(
    n_frozen: Any,
    n_total: Any,
    well_volume_uL: float,
    dilution: Any = 1.0,
    *,
    likelihood_weight: Any = 1.0,
    relative_tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> float:
    """Return the MLE of K(T), expressed as INP/mL original suspension."""

    frozen, total, dilution_array, weight = _weighted_count_likelihood_arrays(
        n_frozen,
        n_total,
        dilution,
        likelihood_weight,
    )
    _validate_likelihood_options(relative_tolerance, max_iterations)
    occupancy_scale = _occupancy_scale(well_volume_uL, dilution_array)
    unfrozen = total - frozen
    if np.all(frozen == 0):
        return 0.0
    if np.all(unfrozen == 0):
        return np.inf

    high = _initial_upper_inp_per_ml(frozen, total, well_volume_uL, dilution_array)
    for _ in range(max_iterations):
        if _binomial_poisson_score(high, frozen, total, occupancy_scale, weight) <= 0:
            break
        high *= 2.0
    else:
        raise RuntimeError("Could not bracket finite binomial-Poisson MLE")

    low = 0.0
    for _ in range(max_iterations):
        midpoint = (low + high) / 2.0
        if _binomial_poisson_score(midpoint, frozen, total, occupancy_scale, weight) > 0:
            low = midpoint
        else:
            high = midpoint
        if high - low <= relative_tolerance * max(1.0, midpoint):
            break
    return (low + high) / 2.0


def binomial_poisson_profile_ci_inp_per_ml(
    n_frozen: Any,
    n_total: Any,
    well_volume_uL: float,
    dilution: Any = 1.0,
    *,
    likelihood_weight: Any = 1.0,
    confidence_drop: float = PROFILE_LIKELIHOOD_DROP_95,
    relative_tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> tuple[float, float, float]:
    """Return MLE, lower CI limit, and upper CI limit for K(T).

    ``confidence_drop`` is the log-likelihood drop from the maximum. For a 95%
    profile likelihood interval with one fitted parameter, use 1.920729410347062
    (= chi-square_0.95,df=1 / 2).
    """

    frozen, total, dilution_array, weight = _weighted_count_likelihood_arrays(
        n_frozen,
        n_total,
        dilution,
        likelihood_weight,
    )
    _validate_likelihood_options(relative_tolerance, max_iterations)
    if confidence_drop <= 0:
        raise ValueError("confidence_drop must be positive")

    mle = binomial_poisson_mle_inp_per_ml(
        frozen,
        total,
        well_volume_uL,
        dilution_array,
        likelihood_weight=weight,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )
    loglike_hat = binomial_poisson_log_likelihood(
        mle,
        frozen,
        total,
        well_volume_uL,
        dilution_array,
        likelihood_weight=weight,
    )
    target = loglike_hat - confidence_drop

    if mle == 0.0:
        upper = _solve_profile_upper_from_zero(
            target,
            frozen,
            total,
            well_volume_uL,
            dilution_array,
            weight,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
        )
        return mle, 0.0, upper

    if np.isinf(mle):
        lower = _solve_profile_lower_to_infinity(
            target,
            frozen,
            total,
            well_volume_uL,
            dilution_array,
            weight,
            relative_tolerance=relative_tolerance,
            max_iterations=max_iterations,
        )
        return mle, lower, np.inf

    lower = _solve_profile_crossing(
        0.0,
        mle,
        target,
        frozen,
        total,
        well_volume_uL,
        dilution_array,
        weight,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )
    upper_high = _initial_upper_inp_per_ml(frozen, total, well_volume_uL, dilution_array)
    upper_high = max(upper_high, mle * 2.0)
    for _ in range(max_iterations):
        if (
            binomial_poisson_log_likelihood(
                upper_high,
                frozen,
                total,
                well_volume_uL,
                dilution_array,
                likelihood_weight=weight,
            )
            <= target
        ):
            break
        upper_high *= 2.0
    else:
        raise RuntimeError("Could not bracket upper profile likelihood limit")
    upper = _solve_profile_crossing(
        mle,
        upper_high,
        target,
        frozen,
        total,
        well_volume_uL,
        dilution_array,
        weight,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )
    return mle, lower, upper


def binomial_poisson_mle_with_profile_errors(
    n_frozen: Any,
    n_total: Any,
    well_volume_uL: float,
    dilution: Any = 1.0,
    *,
    likelihood_weight: Any = 1.0,
    confidence_drop: float = PROFILE_LIKELIHOOD_DROP_95,
    relative_tolerance: float = 1e-10,
    max_iterations: int = 200,
) -> tuple[float, float, float, bool]:
    """Return K(T), lower error width, upper error width, and finite flag."""

    mle, lower_limit, upper_limit = binomial_poisson_profile_ci_inp_per_ml(
        n_frozen,
        n_total,
        well_volume_uL,
        dilution,
        likelihood_weight=likelihood_weight,
        confidence_drop=confidence_drop,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )
    lower_error = mle - lower_limit if np.isfinite(mle) and np.isfinite(lower_limit) else np.nan
    upper_error = upper_limit - mle if np.isfinite(mle) and np.isfinite(upper_limit) else np.nan
    return (
        mle,
        lower_error,
        upper_error,
        bool(np.isfinite(mle) and np.isfinite(lower_error) and np.isfinite(upper_error)),
    )


def _poisson_occupancy_mean(
    inp_per_mL: Any,
    well_volume_uL: float,
    dilution: Any,
) -> np.ndarray:
    inp = as_float_array(inp_per_mL, name="inp_per_mL")
    dilution_array = as_float_array(dilution, name="dilution")
    inp, dilution_array = np.broadcast_arrays(inp, dilution_array)
    if np.any(inp < 0):
        raise ValueError("inp_per_mL cannot be negative")
    return inp * _occupancy_scale(well_volume_uL, dilution_array)


def _scalar_nonnegative_concentration(value: Any, *, name: str) -> float:
    array = np.asarray(value, dtype=float)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    concentration = float(array)
    if np.isnan(concentration) or concentration < 0:
        raise ValueError(f"{name} must be nonnegative")
    return concentration


def _occupancy_scale(well_volume_uL: float, dilution: np.ndarray) -> np.ndarray:
    if well_volume_uL <= 0:
        raise ValueError("well_volume_uL must be positive")
    if np.any(dilution <= 0):
        raise ValueError("dilution must be positive")
    return (well_volume_uL / 1000.0) / dilution


def _count_likelihood_arrays(
    n_frozen: Any,
    n_total: Any,
    dilution: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frozen = as_float_array(n_frozen, name="n_frozen")
    total = as_float_array(n_total, name="n_total")
    dilution_array = as_float_array(dilution, name="dilution")
    frozen, total, dilution_array = np.broadcast_arrays(frozen, total, dilution_array)
    validate_counts(frozen, total)
    if np.any(dilution_array <= 0):
        raise ValueError("dilution must be positive")
    return frozen, total, dilution_array


def _weighted_count_likelihood_arrays(
    n_frozen: Any,
    n_total: Any,
    dilution: Any,
    likelihood_weight: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frozen, total, dilution_array = _count_likelihood_arrays(n_frozen, n_total, dilution)
    weight = as_float_array(likelihood_weight, name="likelihood_weight")
    frozen, total, dilution_array, weight = np.broadcast_arrays(
        frozen,
        total,
        dilution_array,
        weight,
    )
    if np.any(~np.isfinite(weight)):
        raise ValueError("likelihood_weight must be finite")
    if np.any(weight < 0):
        raise ValueError("likelihood_weight cannot be negative")
    active = weight > 0
    if not np.any(active):
        raise ValueError("at least one likelihood_weight must be positive")
    return frozen[active], total[active], dilution_array[active], weight[active]


def _log_one_minus_exp_neg(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values, dtype=float)
    zero = values == 0
    result[zero] = -np.inf
    positive = ~zero
    small = positive & (values <= math.log(2.0))
    result[small] = np.log(-np.expm1(-values[small]))
    result[positive & ~small] = np.log1p(-np.exp(-values[positive & ~small]))
    return result


def _binomial_log_constant(
    frozen: np.ndarray,
    total: np.ndarray,
    weight: np.ndarray | None = None,
) -> float:
    if np.any(~np.isclose(frozen, np.rint(frozen))) or np.any(~np.isclose(total, np.rint(total))):
        raise ValueError("binomial constant requires integer counts")
    frozen_int = np.rint(frozen).astype(int)
    total_int = np.rint(total).astype(int)
    weight_array = np.ones_like(frozen, dtype=float) if weight is None else weight
    return float(
        sum(
            float(w)
            * (math.lgamma(int(n) + 1) - math.lgamma(int(x) + 1) - math.lgamma(int(n - x) + 1))
            for x, n, w in zip(frozen_int.flat, total_int.flat, weight_array.flat, strict=True)
        )
    )


def _validate_likelihood_options(relative_tolerance: float, max_iterations: int) -> None:
    if relative_tolerance <= 0:
        raise ValueError("relative_tolerance must be positive")
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")


def _initial_upper_inp_per_ml(
    frozen: np.ndarray,
    total: np.ndarray,
    well_volume_uL: float,
    dilution: np.ndarray,
) -> float:
    fraction = frozen / total
    partial = (fraction > 0) & (fraction < 1)
    if np.any(partial):
        partial_estimates = (
            -np.log1p(-fraction[partial]) / (well_volume_uL / 1000.0) * dilution[partial]
        )
        finite_estimates = partial_estimates[np.isfinite(partial_estimates)]
        if len(finite_estimates) > 0:
            return max(float(np.max(finite_estimates)) * 2.0, 1e-12)
    occupancy_scale = _occupancy_scale(well_volume_uL, dilution)
    return max(float(1.0 / np.max(occupancy_scale)), 1e-12)


def _binomial_poisson_score(
    inp_per_mL: float,
    frozen: np.ndarray,
    total: np.ndarray,
    occupancy_scale: np.ndarray,
    weight: np.ndarray,
) -> float:
    occupancy = inp_per_mL * occupancy_scale
    with np.errstate(over="ignore"):
        denominator = np.expm1(occupancy)
    frozen_term = np.zeros_like(frozen, dtype=float)
    frozen_mask = frozen != 0
    frozen_term[frozen_mask] = (
        frozen[frozen_mask] * occupancy_scale[frozen_mask] / denominator[frozen_mask]
    )
    unfrozen_term = (total - frozen) * occupancy_scale
    return float(np.sum(weight * (frozen_term - unfrozen_term)))


def _solve_profile_upper_from_zero(
    target: float,
    frozen: np.ndarray,
    total: np.ndarray,
    well_volume_uL: float,
    dilution: np.ndarray,
    weight: np.ndarray,
    *,
    relative_tolerance: float,
    max_iterations: int,
) -> float:
    high = _initial_upper_inp_per_ml(frozen, total, well_volume_uL, dilution)
    for _ in range(max_iterations):
        if (
            binomial_poisson_log_likelihood(
                high,
                frozen,
                total,
                well_volume_uL,
                dilution,
                likelihood_weight=weight,
            )
            <= target
        ):
            break
        high *= 2.0
    else:
        raise RuntimeError("Could not bracket upper profile likelihood limit")
    return _solve_profile_crossing(
        0.0,
        high,
        target,
        frozen,
        total,
        well_volume_uL,
        dilution,
        weight,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )


def _solve_profile_lower_to_infinity(
    target: float,
    frozen: np.ndarray,
    total: np.ndarray,
    well_volume_uL: float,
    dilution: np.ndarray,
    weight: np.ndarray,
    *,
    relative_tolerance: float,
    max_iterations: int,
) -> float:
    high = _initial_upper_inp_per_ml(frozen, total, well_volume_uL, dilution)
    for _ in range(max_iterations):
        if (
            binomial_poisson_log_likelihood(
                high,
                frozen,
                total,
                well_volume_uL,
                dilution,
                likelihood_weight=weight,
            )
            >= target
        ):
            break
        high *= 2.0
    else:
        raise RuntimeError("Could not bracket lower profile likelihood limit")
    return _solve_profile_crossing(
        0.0,
        high,
        target,
        frozen,
        total,
        well_volume_uL,
        dilution,
        weight,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )


def _solve_profile_crossing(
    low: float,
    high: float,
    target: float,
    frozen: np.ndarray,
    total: np.ndarray,
    well_volume_uL: float,
    dilution: np.ndarray,
    weight: np.ndarray,
    *,
    relative_tolerance: float,
    max_iterations: int,
) -> float:
    low_value = binomial_poisson_log_likelihood(
        low,
        frozen,
        total,
        well_volume_uL,
        dilution,
        likelihood_weight=weight,
    )
    high_value = binomial_poisson_log_likelihood(
        high,
        frozen,
        total,
        well_volume_uL,
        dilution,
        likelihood_weight=weight,
    )
    low_above = low_value >= target
    high_above = high_value >= target
    if low_above == high_above:
        raise RuntimeError("Profile likelihood bounds do not bracket a crossing")
    for _ in range(max_iterations):
        midpoint = (low + high) / 2.0
        midpoint_value = binomial_poisson_log_likelihood(
            midpoint,
            frozen,
            total,
            well_volume_uL,
            dilution,
            likelihood_weight=weight,
        )
        midpoint_above = midpoint_value >= target
        if midpoint_above == low_above:
            low = midpoint
        else:
            high = midpoint
        if high - low <= relative_tolerance * max(1.0, midpoint):
            break
    return (low + high) / 2.0
