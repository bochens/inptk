"""Pointwise sample/blank likelihood retaining each physical blank once.

This model assumes independent raw droplet sets, known droplet volumes, a
common water-background concentration per mL, and additive freezing hazards. Repeated
cycles or successive temperatures must not be stacked as independent rows.
Profile limits are model-based pointwise intervals, not validated simultaneous
confidence bands or a guarantee of nominal coverage at parameter boundaries.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .math import PROFILE_LIKELIHOOD_DROP_95, validate_counts

_ITERATIONS = 200
_TOLERANCE = 1e-11


def _scalar(value: Any, *, name: str) -> float:
    number = np.asarray(value, dtype=float)
    if number.ndim != 0 or not np.isfinite(number):
        raise ValueError(f"{name} must be a finite scalar")
    return float(number)


def _hazard_scores(hazard: np.ndarray, frozen: np.ndarray, total: np.ndarray):
    """Return first and second log-likelihood derivatives for each hazard."""
    score = -(total - frozen)
    curvature = np.zeros_like(hazard)
    positive = frozen > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        survival = np.exp(-hazard[positive])
        probability = -np.expm1(-hazard[positive])
        score[positive] += frozen[positive] * survival / probability
        curvature[positive] = -frozen[positive] * survival / probability**2
    return score, curvature


def _log_likelihood(hazard: np.ndarray, frozen: np.ndarray, total: np.ndarray) -> float:
    positive = frozen > 0
    unfrozen = total - frozen
    with np.errstate(divide="ignore", invalid="ignore"):
        frozen_term = float(np.sum(frozen[positive] * np.log(-np.expm1(-hazard[positive]))))
    return float(frozen_term - np.dot(unfrozen, hazard))


def joint_water_blank_mle(
    n_frozen: Any,
    n_total: Any,
    dilution: Any,
    well_volume_uL: Any,
    blank_frozen: Any,
    blank_total: Any,
    *,
    blank_volume_uL: Any = None,
    confidence_drop: float = PROFILE_LIKELIHOOD_DROP_95,
) -> tuple[float, float, float, bool]:
    """Return concentration, lower/upper error widths and a finite-result flag.

    The fit searches for nonnegative sample and background concentrations whose
    predicted freezing best matches the observed counts. Each uncertainty bound
    permits a log-likelihood reduction of ``confidence_drop`` while refitting
    the background. These are model-based bounds for one temperature, without a
    guarantee of coverage at small counts or concentration boundaries.

    Sample and blank rows represent independent physical droplet sets at one
    temperature and cycle. Each blank set contributes once, even if it is shared
    by many sample rows. Volumes may differ and are supplied in microlitres.
    ``blank_volume_uL`` defaults to the sample volume only when the supplied
    sample volume is scalar; vector sample volumes require explicit blank volumes.
    Sample and blank droplet counts may differ; their actual totals determine
    their contribution to the fit and uncertainty.

    For sample volume Vs and blank volume Vb in mL, dilution D, original-sample
    concentration K and background concentration B, hazards are Vs*(K/D+B) and
    Vb*B. A hazard h corresponds to frozen probability 1-exp(-h).
    This assumes a common background concentration per unit liquid volume;
    it does not account for a surface background that scales differently.
    K and B are nonnegative, and B is profiled at every candidate K. A completely
    saturated experiment cannot distinguish sample from background (NaNs).
    If all samples but not all blanks are saturated, K is unbounded (infinity).
    Saturated blanks with unsaturated samples are fitted normally.
    """
    supplied_volumes = np.asarray(well_volume_uL, dtype=float)
    if blank_volume_uL is None:
        if supplied_volumes.ndim != 0:
            raise ValueError("Vector sample volumes require explicit blank_volume_uL")
        blank_volume_uL = float(supplied_volumes)
    frozen, total, dilutions, sample_volumes = np.broadcast_arrays(
        np.asarray(n_frozen, dtype=float),
        np.asarray(n_total, dtype=float),
        np.asarray(dilution, dtype=float),
        supplied_volumes,
    )
    frozen, total, dilutions, sample_volumes = (
        np.atleast_1d(values) for values in (frozen, total, dilutions, sample_volumes)
    )
    if frozen.ndim != 1 or frozen.size == 0:
        raise ValueError("Sample counts must be a nonempty one-dimensional sequence")
    validate_counts(frozen, total)
    if np.any(frozen % 1 != 0) or np.any(total % 1 != 0):
        raise ValueError("Raw sample counts must be whole droplet counts")
    if np.any(~np.isfinite(dilutions)) or np.any(dilutions <= 0):
        raise ValueError("Dilution factors must be finite and positive")
    blank_x, blank_n, blank_volumes = np.broadcast_arrays(
        np.asarray(blank_frozen, dtype=float),
        np.asarray(blank_total, dtype=float),
        np.asarray(blank_volume_uL, dtype=float),
    )
    blank_x, blank_n, blank_volumes = (
        np.atleast_1d(values) for values in (blank_x, blank_n, blank_volumes)
    )
    if blank_x.ndim != 1 or blank_x.size == 0:
        raise ValueError("Blank counts must be a nonempty one-dimensional sequence")
    validate_counts(blank_x, blank_n)
    if np.any(blank_x % 1 != 0) or np.any(blank_n % 1 != 0):
        raise ValueError("Raw blank counts must be whole droplet counts")
    for volumes in (sample_volumes, blank_volumes):
        if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
            raise ValueError("Sample and blank droplet volumes must be finite and positive")
    drop = _scalar(confidence_drop, name="confidence_drop")
    if drop <= 0:
        raise ValueError("confidence_drop must be positive")
    if np.all(frozen == total) and np.all(blank_x == blank_n):
        return np.nan, np.nan, np.nan, False
    if np.all(frozen == total):
        return np.inf, np.nan, np.nan, False

    # The largest volume is a numerical reference only: c=K*Vref, beta=B*Vref.
    # Keeping background exposures at most one avoids squaring large ratios.
    all_volumes = np.r_[sample_volumes, blank_volumes]
    reference_volume_uL = float(np.max(all_volumes))
    volume = reference_volume_uL / 1000.0
    if volume == 0:
        raise ValueError("Reference droplet volume is below the finite numerical range")
    all_frozen = np.r_[frozen, blank_x]
    all_total = np.r_[total, blank_n]
    background_exposure = all_volumes / reference_volume_uL
    exposure = np.r_[background_exposure[:len(frozen)] / dilutions, np.zeros(len(blank_x))]
    if np.any(~np.isfinite(background_exposure)) or np.any(~np.isfinite(exposure)):
        raise ValueError("Droplet-volume/dilution ratios exceed the finite numerical range")
    if np.any(background_exposure <= 0) or np.any(exposure[:len(frozen)] <= 0):
        raise ValueError("Droplet-volume/dilution ratios are below the finite numerical range")
    total_frozen = float(np.sum(all_frozen))
    total_droplets = float(np.sum(all_total))
    sample_unfrozen_exposure = float(np.dot(total - frozen, exposure[:len(frozen)]))
    score_tolerance = _TOLERANCE * max(1.0, float(np.dot(all_total, background_exposure)))
    cache: dict[float, tuple[float, float]] = {}

    def fit_background(base_hazard: np.ndarray, upper: float) -> float:
        at_zero, _ = _hazard_scores(base_hazard, all_frozen, all_total)
        if float(np.dot(background_exposure, at_zero)) <= 0:
            return 0.0
        low, high = 0.0, upper
        background = low + (high - low) / 2
        for _ in range(_ITERATIONS):
            scores, curvature = _hazard_scores(
                base_hazard + background * background_exposure, all_frozen, all_total
            )
            score = float(np.dot(background_exposure, scores))
            if abs(score) <= score_tolerance:
                return background
            if score > 0:
                low = background
            else:
                high = background
            if high - low <= _TOLERANCE * max(1.0, background):
                return low + (high - low) / 2
            derivative = float(np.dot(background_exposure**2, curvature))
            candidate = background - score / derivative if derivative < 0 else np.nan
            background = (
                candidate if np.isfinite(candidate) and low < candidate < high
                else low + (high - low) / 2
            )
        raise RuntimeError("Water-blank background profile did not converge")

    if np.all(background_exposure == 1):
        background_at_zero = float(-np.log1p(-total_frozen / total_droplets))
    else:
        upper_background = max(
            1.0, total_frozen / float(np.dot(all_total - all_frozen, background_exposure))
        )
        if not np.isfinite(upper_background):
            raise ValueError("Counts/volumes exceed the finite background concentration range")
        background_at_zero = fit_background(np.zeros_like(all_frozen), upper_background)

    def profile(c: float) -> tuple[float, float]:
        """Return profiled log-likelihood and its derivative with respect to c."""
        if c in cache:
            return cache[c]
        if not np.isfinite(c):
            raise ValueError("Concentration profile exceeds the finite numerical range")
        base_hazard = c * exposure
        background = (
            background_at_zero if c == 0 else fit_background(base_hazard, background_at_zero)
        )
        hazard = base_hazard + background * background_exposure
        scores, _ = _hazard_scores(hazard, all_frozen, all_total)
        result = (
            _log_likelihood(hazard, all_frozen, all_total),
            float(np.dot(scores[:len(frozen)], exposure[:len(frozen)])),
        )
        cache[c] = result
        return result

    log_at_zero, score_at_zero = profile(0.0)
    # Background profiling leaves a small score residual. Scale this boundary
    # tolerance by sample exposure, so very large dilutions still retain a
    # positive sample signal instead of being mistaken for the zero boundary.
    sample_score_tolerance = _TOLERANCE * float(np.dot(total, exposure[:len(frozen)]))
    if score_at_zero <= sample_score_tolerance:
        estimate_c = 0.0
    else:
        low = 0.0
        high = max(1.0, float(np.sum(frozen) / sample_unfrozen_exposure))
        if not np.isfinite(high):
            raise ValueError("Count/dilution scale exceeds the finite concentration range")
        for _ in range(_ITERATIONS):
            midpoint = low + (high - low) / 2
            if profile(midpoint)[1] > 0:
                low = midpoint
            else:
                high = midpoint
            if high - low <= _TOLERANCE * max(1.0, midpoint):
                break
        else:
            raise RuntimeError("Water-blank concentration fit did not converge")
        estimate_c = low + (high - low) / 2
    target = profile(estimate_c)[0] - drop

    if estimate_c == 0 or log_at_zero >= target:
        lower_c = 0.0
    else:
        low, high = 0.0, estimate_c
        for _ in range(_ITERATIONS):
            midpoint = low + (high - low) / 2
            if profile(midpoint)[0] < target:
                low = midpoint
            else:
                high = midpoint
            if high - low <= _TOLERANCE * max(1.0, midpoint):
                break
        else:
            raise RuntimeError("Water-blank lower profile limit did not converge")
        lower_c = low + (high - low) / 2

    # Every frozen-droplet log term is <=0, so the profile is bounded above
    # by -c*sum(unfrozen_sample*sample_exposure). This brackets the upper limit
    # even when the estimate is zero and the dilutions are very large.
    low = estimate_c
    high = max(1.0, estimate_c * 2, -target / sample_unfrozen_exposure)
    for _ in range(_ITERATIONS):
        if profile(high)[0] <= target:
            break
        high *= 2
    else:
        raise RuntimeError("Could not bracket water-blank upper profile limit")
    for _ in range(_ITERATIONS):
        midpoint = low + (high - low) / 2
        if profile(midpoint)[0] > target:
            low = midpoint
        else:
            high = midpoint
        if high - low <= _TOLERANCE * max(1.0, midpoint):
            break
    else:
        raise RuntimeError("Water-blank upper profile limit did not converge")
    upper_c = low + (high - low) / 2
    estimate = estimate_c / volume
    lower_error = max(0.0, estimate_c - lower_c) / volume
    upper_error = max(0.0, upper_c - estimate_c) / volume
    finite = bool(np.isfinite([estimate, lower_error, upper_error]).all())
    return estimate, lower_error, upper_error, finite
