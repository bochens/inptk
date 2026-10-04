"""Direct Average calculations and pointwise likelihood for raw sample/blank counts.

This model assumes independent droplet sets and known droplet volumes. When
blank data are supplied, each group has a common background concentration per mL
that contributes an additive freezing hazard. Repeated cycles or successive
temperatures must not be stacked as independent rows.
Profile limits are model-based pointwise intervals, not validated simultaneous
confidence bands or a guarantee of nominal coverage at parameter boundaries.
"""

from __future__ import annotations

from collections.abc import Callable
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


def _sample_arrays(n_frozen: Any, n_total: Any, dilution: Any, well_volume_uL: Any):
    """Validate every contributing sample before calculating concentrations."""
    frozen, total, dilutions, sample_volumes = np.broadcast_arrays(
        np.asarray(n_frozen, dtype=float),
        np.asarray(n_total, dtype=float),
        np.asarray(dilution, dtype=float),
        np.asarray(well_volume_uL, dtype=float),
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
    if np.any(~np.isfinite(sample_volumes)) or np.any(sample_volumes <= 0):
        raise ValueError("Sample and blank droplet volumes must be finite and positive")
    return frozen, total, dilutions, sample_volumes


def _blank_arrays(
    well_volume_uL: Any, blank_frozen: Any, blank_total: Any, blank_volume_uL: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate supplied blank observations without inventing blank data."""
    with_blank = blank_frozen is not None or blank_total is not None
    if (blank_frozen is None) != (blank_total is None):
        raise ValueError("blank_frozen and blank_total must be supplied together")
    if not with_blank:
        if blank_volume_uL is not None:
            raise ValueError("blank_volume_uL requires blank observations")
        return np.empty(0, dtype=float), np.empty(0, dtype=float), np.empty(0, dtype=float)
    if blank_volume_uL is None:
        supplied_volumes = np.asarray(well_volume_uL, dtype=float)
        if supplied_volumes.ndim != 0:
            raise ValueError("Vector sample volumes require explicit blank_volume_uL")
        blank_volume_uL = float(supplied_volumes)
    blank_x, blank_n, volumes = np.broadcast_arrays(
        np.asarray(blank_frozen, dtype=float),
        np.asarray(blank_total, dtype=float),
        np.asarray(blank_volume_uL, dtype=float),
    )
    blank_x, blank_n, volumes = (np.atleast_1d(values) for values in (blank_x, blank_n, volumes))
    if blank_x.ndim != 1 or blank_x.size == 0:
        raise ValueError("Blank counts must be a nonempty one-dimensional sequence")
    validate_counts(blank_x, blank_n)
    if np.any(blank_x % 1 != 0) or np.any(blank_n % 1 != 0):
        raise ValueError("Raw blank counts must be whole droplet counts")
    if np.any(~np.isfinite(volumes)) or np.any(volumes <= 0):
        raise ValueError("Sample and blank droplet volumes must be finite and positive")
    return blank_x, blank_n, volumes


def _blank_groups(
    sample_blank_group: Any, blank_group: Any, sample_count: int, blank_count: int
) -> tuple[np.ndarray, np.ndarray]:
    """Require exact row labels; physical blank identity remains the caller's job."""
    if sample_blank_group is None and blank_group is None:
        return np.repeat("shared", sample_count), np.repeat("shared", blank_count)
    if not blank_count:
        raise ValueError("Blank group labels require blank observations")
    if sample_blank_group is None or blank_group is None:
        raise ValueError("sample_blank_group and blank_group must be supplied together")
    labels = []
    for name, values, count in (
        ("sample_blank_group", sample_blank_group, sample_count),
        ("blank_group", blank_group, blank_count),
    ):
        array = np.asarray(values, dtype=object)
        if array.ndim != 1 or len(array) != count:
            raise ValueError(f"{name} must have one string label per observation row")
        if any(not isinstance(label, str) or not label for label in array):
            raise ValueError(f"{name} labels must be nonempty strings")
        labels.append(array)
    sample_labels, blank_labels = labels
    if set(sample_labels) != set(blank_labels):
        raise ValueError("Every sample blank group must have blanks, with no unused blank groups")
    return sample_labels, blank_labels


def _group_profile(
    frozen: np.ndarray,
    total: np.ndarray,
    sample_volumes: np.ndarray,
    sample_exposure: np.ndarray,
    blank_x: np.ndarray,
    blank_n: np.ndarray,
    blank_volumes: np.ndarray,
) -> Callable[[float], tuple[float, float]]:
    """Profile one independent background while using the caller's common K scale."""
    with_blank = bool(len(blank_x))
    all_frozen = np.r_[frozen, blank_x]
    all_total = np.r_[total, blank_n]
    all_volumes = np.r_[sample_volumes, blank_volumes]
    # A separate reference for each background keeps its derivatives well scaled
    # even when different runs use very different well volumes.
    background_exposure = all_volumes / float(np.max(all_volumes))
    if np.any(background_exposure <= 0):
        raise ValueError("Droplet-volume ratios are below the finite numerical range")
    exposure = np.r_[sample_exposure, np.zeros(len(blank_x))]
    total_frozen = float(np.sum(all_frozen))
    total_droplets = float(np.sum(all_total))
    score_tolerance = _TOLERANCE * max(1.0, float(np.dot(all_total, background_exposure)))

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

    if not with_blank:
        background_at_zero = 0.0
    elif np.all(background_exposure == 1):
        background_at_zero = float(-np.log1p(-total_frozen / total_droplets))
    else:
        upper_background = max(
            1.0, total_frozen / float(np.dot(all_total - all_frozen, background_exposure))
        )
        if not np.isfinite(upper_background):
            raise ValueError("Counts/volumes exceed the finite background concentration range")
        background_at_zero = fit_background(np.zeros_like(all_frozen), upper_background)

    def profile(c: float) -> tuple[float, float]:
        base_hazard = c * exposure
        background = background_at_zero
        if with_blank and c != 0:
            background = fit_background(base_hazard, background_at_zero)
        hazard = base_hazard + background * background_exposure
        scores, _ = _hazard_scores(hazard, all_frozen, all_total)
        return (
            _log_likelihood(hazard, all_frozen, all_total),
            float(np.dot(scores[:len(frozen)], sample_exposure)),
        )

    return profile


def fit_concentration(
    n_frozen: Any,
    n_total: Any,
    dilution: Any,
    well_volume_uL: Any,
    *,
    confidence_drop: float = PROFILE_LIKELIHOOD_DROP_95,
    blank_frozen: Any = None,
    blank_total: Any = None,
    blank_volume_uL: Any = None,
    sample_blank_group: Any = None,
    blank_group: Any = None,
) -> tuple[float, float, float, bool]:
    """Return concentration, lower/upper error widths and a finite-result flag.

    One or several independent sample sets use the same count likelihood.
    Without blank observations, it fits concentration alone using hazard Vs*K/D.
    With blanks, it also estimates their background concentrations.
    The fit searches for nonnegative concentrations whose predicted freezing
    best matches the observed counts. Each uncertainty bound permits a
    log-likelihood reduction of ``confidence_drop`` while refitting the
    background when present. These are model-based bounds for one temperature,
    without a guarantee of coverage at small counts or concentration boundaries.

    Sample and blank rows represent independent physical droplet sets at one
    temperature, with at most one cycle per physical set. Each blank set contributes
    once, even if it is shared by many sample rows. Volumes may differ and are
    supplied in microlitres.
    ``blank_volume_uL`` defaults to the sample volume only when the supplied
    sample volume is scalar; vector sample volumes require explicit blank volumes.
    Sample and blank droplet counts may differ; their actual totals determine
    their contribution to the fit and uncertainty.

    Omit group labels for one shared background. To combine independent runs,
    supply ``sample_blank_group`` and ``blank_group`` as one-dimensional arrays
    of exact nonempty strings, one label per sample and blank row respectively.
    A sample uses only blanks with its label. Each group has its own background;
    K is shared across groups. Every group needs both samples and blanks, and
    the caller must include each physical blank observation only once.

    For sample volume Vs and blank volume Vb in mL, dilution D, original-sample
    concentration K and background concentration B, hazards are Vs*(K/D+B) and
    Vb*B. A hazard h corresponds to frozen probability 1-exp(-h).
    This assumes a common background concentration per unit liquid volume within each group;
    it does not account for a surface background that scales differently.
    K and each B are nonnegative. Each background is refitted at every candidate
    K. A fully saturated sample-plus-blank group provides no information about K;
    it contributes a constant likelihood while other groups can identify K.
    If all groups are saturated, their contributions cannot be separated (NaNs).
    If all samples are saturated but blanks are absent or include any unfrozen
    droplets, K is unbounded (infinity).
    Saturated blanks with unsaturated samples are fitted normally.
    """
    frozen, total, dilutions, sample_volumes = _sample_arrays(
        n_frozen, n_total, dilution, well_volume_uL
    )
    blank_x, blank_n, blank_volumes = _blank_arrays(
        well_volume_uL, blank_frozen, blank_total, blank_volume_uL
    )
    sample_labels, blank_labels = _blank_groups(
        sample_blank_group, blank_group, len(frozen), len(blank_x)
    )
    with_blank = bool(len(blank_x))
    drop = _scalar(confidence_drop, name="confidence_drop")
    if drop <= 0:
        raise ValueError("confidence_drop must be positive")

    if with_blank:
        informative_groups = []
        for label in dict.fromkeys(sample_labels):
            sample_mask, blank_mask = sample_labels == label, blank_labels == label
            if not (
                np.all(frozen[sample_mask] == total[sample_mask])
                and np.all(blank_x[blank_mask] == blank_n[blank_mask])
            ):
                informative_groups.append(label)
        if not informative_groups:
            return np.nan, np.nan, np.nan, False
        # A fully saturated sample-plus-blank group has supremum log-likelihood
        # zero for any K as its background grows. It provides no information
        # about K, so it must not change another group's numerical scale either.
        sample_mask = np.isin(sample_labels, informative_groups)
        blank_mask = np.isin(blank_labels, informative_groups)
        frozen, total, dilutions, sample_volumes, sample_labels = (
            values[sample_mask]
            for values in (frozen, total, dilutions, sample_volumes, sample_labels)
        )
        blank_x, blank_n, blank_volumes, blank_labels = (
            values[blank_mask] for values in (blank_x, blank_n, blank_volumes, blank_labels)
        )
    if np.all(frozen == total):
        return np.inf, np.nan, np.nan, False

    # The largest informative volume is a numerical reference: c=K*Vref.
    reference_volume_uL = float(np.max(np.r_[sample_volumes, blank_volumes]))
    volume = reference_volume_uL / 1000.0
    if volume == 0:
        raise ValueError("Reference droplet volume is below the finite numerical range")
    exposure = sample_volumes / reference_volume_uL / dilutions
    if np.any(~np.isfinite(exposure)):
        raise ValueError("Droplet-volume/dilution ratios exceed the finite numerical range")
    if np.any(exposure <= 0):
        raise ValueError("Droplet-volume/dilution ratios are below the finite numerical range")
    sample_unfrozen_exposure = float(np.dot(total - frozen, exposure))
    profiles = []
    for label in dict.fromkeys(sample_labels):
        sample_mask, blank_mask = sample_labels == label, blank_labels == label
        profiles.append(_group_profile(
            frozen[sample_mask], total[sample_mask], sample_volumes[sample_mask],
            exposure[sample_mask], blank_x[blank_mask], blank_n[blank_mask],
            blank_volumes[blank_mask],
        ))
    cache: dict[float, tuple[float, float]] = {}

    def profile(c: float) -> tuple[float, float]:
        """Sum independent backgrounds' profiled likelihoods and K derivatives."""
        if c not in cache:
            if not np.isfinite(c):
                raise ValueError("Concentration profile exceeds the finite numerical range")
            values = [group_profile(c) for group_profile in profiles]
            cache[c] = (
                sum(value[0] for value in values), sum(value[1] for value in values)
            )
        return cache[c]

    log_at_zero, score_at_zero = profile(0.0)
    # Background profiling leaves a small score residual. Scale this boundary
    # tolerance by sample exposure, so very large dilutions still retain a
    # positive sample signal instead of being mistaken for the zero boundary.
    sample_score_tolerance = _TOLERANCE * float(np.dot(total, exposure[:len(frozen)]))
    analytic_estimate = None
    effective_volumes = sample_volumes / dilutions / 1000.0
    if (
        not with_blank
        and effective_volumes[0] > 0
        and np.isfinite(effective_volumes).all()
        and np.all(effective_volumes == effective_volumes[0])
    ):
        # Equal original-sample volume per well has a closed-form estimate.
        # Use it directly so changing the number of identical contributors
        # cannot turn a constant curve into a numerical decrease.
        analytic_estimate = float(
            -np.log1p(-np.sum(frozen) / np.sum(total)) / effective_volumes[0]
        )
        estimate_c = analytic_estimate * volume
    elif score_at_zero <= sample_score_tolerance:
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
            raise RuntimeError("Concentration fit did not converge")
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
            raise RuntimeError("Concentration lower profile limit did not converge")
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
        raise RuntimeError("Could not bracket concentration upper profile limit")
    for _ in range(_ITERATIONS):
        midpoint = low + (high - low) / 2
        if profile(midpoint)[0] > target:
            low = midpoint
        else:
            high = midpoint
        if high - low <= _TOLERANCE * max(1.0, midpoint):
            break
    else:
        raise RuntimeError("Concentration upper profile limit did not converge")
    upper_c = low + (high - low) / 2
    estimate = estimate_c / volume if analytic_estimate is None else analytic_estimate
    lower_error = max(0.0, estimate_c - lower_c) / volume
    upper_error = max(0.0, upper_c - estimate_c) / volume
    finite = bool(np.isfinite([estimate, lower_error, upper_error]).all())
    return estimate, lower_error, upper_error, finite


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
    sample_blank_group: Any = None,
    blank_group: Any = None,
) -> tuple[float, float, float, bool]:
    """Fit sample concentration with each supplied physical water blank once.

    This wrapper retains the explicit sample-plus-blank calling convention.
    See ``fit_concentration`` for the model and uncertainty assumptions.
    """
    if blank_frozen is None or blank_total is None:
        raise ValueError("Joint water-blank fitting requires blank observations")
    return fit_concentration(
        n_frozen, n_total, dilution, well_volume_uL,
        confidence_drop=confidence_drop,
        blank_frozen=blank_frozen,
        blank_total=blank_total,
        blank_volume_uL=blank_volume_uL,
        sample_blank_group=sample_blank_group,
        blank_group=blank_group,
    )


def _direct_concentration(frozen, total, volumes_uL, z):
    """Direct concentrations and log-transformed Wilson binomial error widths."""
    fraction = frozen / total
    z_squared = z * z
    if not np.isfinite(z_squared):
        raise ValueError("z exceeds the finite numerical range")
    denominator = 1 + z_squared / total
    center = (fraction + z_squared / (2 * total)) / denominator
    half_width = z * np.sqrt(
        (fraction * (1 - fraction) + z_squared / (4 * total)) / total
    ) / denominator
    lower_fraction = np.maximum(0, center - half_width)
    upper_fraction = np.minimum(1, center + half_width)
    lower_fraction = np.where(frozen == 0, 0, lower_fraction)
    upper_fraction = np.where(frozen == total, 1, upper_fraction)
    volume_mL = volumes_uL / 1000
    with np.errstate(divide="ignore", invalid="ignore"):
        concentration = -np.log1p(-fraction) / volume_mL
        lower = -np.log1p(-lower_fraction) / volume_mL
        upper = -np.log1p(-upper_fraction) / volume_mL
        return (concentration, np.maximum(0, concentration - lower),
                np.maximum(0, upper - concentration))


def average_concentration(
    n_frozen: Any,
    n_total: Any,
    dilution: Any,
    well_volume_uL: Any,
    *,
    z: float = 1.96,
    blank_frozen: Any = None,
    blank_total: Any = None,
    blank_volume_uL: Any = None,
    sample_blank_group: Any = None,
    blank_group: Any = None,
) -> tuple[float, float, float, bool]:
    """Direct blank-corrected concentrations, arithmetic mean and propagated errors.

    Each sample contributes d * (S - B), where S and B are calculated from
    -log(1 - frozen/total) / well_volume_mL. Negative differences are retained.
    Equal-volume independent blank wells in a group use their combined counts.
    Different blank volumes use a mean of these concentrations weighted by
    total assayed volume (well count times well volume).

    Wilson binomial fraction bounds are transformed through the same logarithm.
    Their asymmetric error widths are propagated in quadrature, reversing the
    blank error direction for subtraction. Independent samples each have weight
    d/m; a shared blank appears once with the sum of those weights. These are
    approximate pointwise bounds, not exact confidence intervals or whole-curve
    bands. Counts across repeated cycles must never be supplied as independent
    wells. Saturated/unavailable inputs are flagged, never silently dropped.
    """
    frozen, total, dilutions, volumes = _sample_arrays(
        n_frozen, n_total, dilution, well_volume_uL
    )
    z = _scalar(z, name="z")
    if z <= 0:
        raise ValueError("z must be positive")
    blank_x, blank_n, blank_volumes = _blank_arrays(
        well_volume_uL, blank_frozen, blank_total, blank_volume_uL
    )
    sample_labels, blank_labels = _blank_groups(
        sample_blank_group, blank_group, len(frozen), len(blank_x)
    )
    sample_rate, sample_lower, sample_upper = _direct_concentration(frozen, total, volumes, z)
    weights = dilutions / len(frozen)
    corrected = sample_rate.copy()
    lower_error = float(np.hypot.reduce(weights * sample_lower))
    upper_error = float(np.hypot.reduce(weights * sample_upper))
    for group in dict.fromkeys(blank_labels):
        mask = blank_labels == group
        group_volumes = np.unique(blank_volumes[mask])
        totals = np.array([np.sum(blank_n[mask & (blank_volumes == v)]) for v in group_volumes])
        counts = np.array([np.sum(blank_x[mask & (blank_volumes == v)]) for v in group_volumes])
        rates, lowers, uppers = _direct_concentration(counts, totals, group_volumes, z)
        if not np.isfinite(rates).all():
            return np.nan, np.nan, np.nan, False
        exposures = totals * group_volumes
        blank_weights = exposures / np.sum(exposures)
        background = float(np.dot(blank_weights, rates))
        background_lower = float(np.hypot.reduce(blank_weights * lowers))
        background_upper = float(np.hypot.reduce(blank_weights * uppers))
        assigned = sample_labels == group
        corrected[assigned] -= background
        # The same blank estimate is subtracted from every assigned sample.
        # Its uncertainty is shared, so sum its coefficients before propagation.
        coefficient = float(np.sum(weights[assigned]))
        lower_error = float(np.hypot(lower_error, coefficient * background_upper))
        upper_error = float(np.hypot(upper_error, coefficient * background_lower))
    corrected *= dilutions
    estimate = (float(corrected[0]) if np.all(corrected == corrected[0])
                else float(np.sum(corrected / len(corrected))))
    if not np.isfinite(estimate):
        return estimate, np.nan, np.nan, False
    finite = bool(np.isfinite([lower_error, upper_error]).all())
    return estimate, lower_error, upper_error, finite
