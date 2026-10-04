"""Joint likelihood of first-freezing intervals for fixed physical droplet sets.

For hazard H(T), liquid probability is exp(-H). A droplet freezing between
observations a and b contributes exp(-H(a)) - exp(-H(b)); an unfrozen droplet
contributes exp(-H(last)). Counts of these mutually exclusive outcomes sum to
the original droplet total. Images are not independent trials.

Sample hazard is volume * (K / dilution + B_run). K and each B_run are sums of
nonnegative increments on the observed temperature support. Profile intervals
optimize the same complete likelihood, including all background parameters.
The optional experimental fitting grid instead uses cumulative concentrations
linear between grid points, evaluated at the unchanged observation temperatures.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize  # type: ignore[import-untyped]


@dataclass(frozen=True)
class FreezingSeries:
    """One selected cycle of one physical set, ordered from warm to cold."""

    temperatures: np.ndarray
    frozen: np.ndarray
    total: int
    sample_exposure: float
    blank_exposure: float = 0.0
    background: str = ""


class _Likelihood:
    def __init__(self, linear, events, counts):
        self.linear = linear
        self.events = events
        self.counts = counts

    def value_gradient(self, x):
        hazard = self.events @ x
        # A finite barrier permits optimizer trial steps at the boundary.
        # Valid solutions must give strictly positive mass to observed events.
        safe = np.maximum(hazard, 1e-12)
        probability = -np.expm1(-safe)
        value = self.linear @ x - self.counts @ np.log(probability)
        slope = self.counts * np.exp(-safe) / probability
        # Continue the tangent below the trial-step barrier; the function and
        # supplied gradient must agree even at an infeasible optimizer trial.
        value += slope @ (safe - hazard)
        gradient = self.linear - self.events.T @ slope
        return float(value), gradient

    def fit(self, initial=None):
        if not len(self.counts):
            return np.zeros(len(self.linear)), 0.0
        if initial is None:
            initial = np.full(len(self.linear), max(1.0, self.counts.sum() / len(self.linear)))
        result = minimize(
            self.value_gradient, initial, jac=True, method="L-BFGS-B",
            bounds=[(0, None)] * len(initial),
            options={"ftol": 1e-13, "gtol": 1e-8, "maxiter": 5000, "maxls": 100},
        )
        if not result.success or np.any(self.events @ result.x <= 0):
            raise RuntimeError(f"Joint freezing-curve fit did not converge: {result.message}")
        return result.x, float(result.fun)

    def endpoint(self, direction, initial, limit, *, upper):
        """Optimize a linear concentration over a convex likelihood region.

        A likelihood penalty supplies its supporting hyperplane. This avoids
        linearizing a likelihood-budget constraint at the MLE, where its
        gradient is zero. At a flat optimal face, interpolate its two solutions
        until the original (unpenalized) likelihood reaches the requested drop.
        """
        if not np.any(direction):
            return 0.0
        objective = direction / np.max(direction)
        solutions = {0.0: initial}

        def penalized(weight):
            if weight not in solutions:
                nearest = min(solutions, key=lambda item: abs(item - weight))
                model = _Likelihood(self.linear + weight * objective, self.events, self.counts)
                solutions[weight], _ = model.fit(solutions[nearest])
            return self.value_gradient(solutions[weight])[0] - limit

        if not upper:
            free = objective == 0
            if np.all(self.events[:, free].sum(axis=1) > 0):
                zero_model = _Likelihood(self.linear[free], self.events[:, free], self.counts)
                _, zero_value = zero_model.fit()
                if zero_value <= limit:
                    return 0.0
            boundary = 1.0
            for _ in range(60):
                if penalized(boundary) >= 0:
                    break
                boundary *= 2
            else:
                raise RuntimeError("Could not bracket the lower joint-curve profile bound")
        else:
            ratios = np.divide(
                self.linear, objective, out=np.full_like(objective, np.inf), where=objective > 0
            )
            critical_weight = float(np.min(ratios))
            critical = np.isclose(ratios, critical_weight, rtol=1e-13, atol=0)
            if not np.any(self.events[:, critical]):
                free = ~critical
                model = _Likelihood(
                    np.maximum(0, self.linear[free] - critical_weight * objective[free]),
                    self.events[:, free], self.counts,
                )
                value, _ = model.fit()
                endpoint = np.zeros_like(initial)
                endpoint[free] = value
                remaining = limit - self.value_gradient(endpoint)[0]
                if remaining >= 0:
                    position = np.flatnonzero(critical)[0]
                    endpoint[position] += remaining / self.linear[position]
                    return float(direction @ endpoint)
                boundary = -critical_weight
                solutions[boundary] = endpoint
            else:
                for power in range(1, 45):
                    boundary = -critical_weight * (1 - 2.0 ** -power)
                    if penalized(boundary) >= 0:
                        break
                else:
                    raise RuntimeError("Could not bracket the upper joint-curve profile bound")
        root = brentq(penalized, min(0, boundary), max(0, boundary), xtol=1e-11, rtol=1e-11)
        endpoint = solutions[root]
        if abs(self.value_gradient(endpoint)[0] - limit) > 1e-6:
            inside = [(weight, x) for weight, x in solutions.items()
                      if self.value_gradient(x)[0] <= limit]
            outside = [(weight, x) for weight, x in solutions.items()
                       if self.value_gradient(x)[0] > limit]
            left = min(inside, key=lambda item: abs(item[0] - root))[1]
            right = min(outside, key=lambda item: abs(item[0] - root))[1]
            fraction = brentq(
                lambda f: self.value_gradient(left + f * (right - left))[0] - limit, 0, 1,
            )
            endpoint = left + fraction * (right - left)
        return float(direction @ endpoint)


class CurveLikelihood:
    """Fit a monotone cumulative spectrum and pointwise profile intervals.

    Identical likelihood columns are combined exactly, not averaged. Mass in
    an observationally indistinguishable interval is placed at its cold edge.
    Profiling splits those columns at the queried temperature, so a zero fitted
    increment does not falsely imply zero uncertainty between freezing events.
    """

    def __init__(self, series: list[FreezingSeries], temperatures, *, fit_step_C=None):
        if not series:
            raise ValueError("A joint curve requires at least one physical droplet set")
        observed = np.unique(np.concatenate([
            np.asarray(temperatures, dtype=float), *[s.temperatures for s in series]
        ]))[::-1]
        self.fit_step_C = fit_step_C
        if fit_step_C is None:
            self.temperatures = observed
        else:
            if isinstance(fit_step_C, bool) or not np.isfinite(fit_step_C) or fit_step_C <= 0:
                raise ValueError("fit_step_C must be finite and positive")
            # Trial model: cumulative sample and blank concentrations are linear
            # between regular knots. Evaluate this model at the ORIGINAL temperatures
            # of observed freezing states; never round or interpolate counts.
            lower = int(np.floor(observed[-1] / fit_step_C))
            upper = int(np.ceil(observed[0] / fit_step_C))
            self.temperatures = np.arange(upper, lower - 1, -1, dtype=float) * fit_step_C
        self.backgrounds = sorted({s.background for s in series if s.background})
        size = len(self.temperatures)
        width = size * (1 + len(self.backgrounds))
        linear = np.zeros(width)
        events, counts = [], []
        for stream in series:
            if stream.total <= 0 or np.any(np.diff(stream.temperatures) >= 0):
                raise ValueError(
                    "Joint series require positive totals and distinct cooling temperatures"
                )
            if np.any(np.diff(stream.frozen) < 0) or np.any(stream.frozen < 0) or np.any(
                stream.frozen > stream.total
            ):
                raise ValueError("Joint series require cumulative first-freezing counts")
            previous = np.zeros(width)
            previous_count = 0
            for temperature, frozen in zip(stream.temperatures, stream.frozen, strict=True):
                state = np.zeros(width)
                mask = self._basis(temperature)
                state[:size] = mask * stream.sample_exposure
                if stream.background:
                    component = 1 + self.backgrounds.index(stream.background)
                    state[component * size:(component + 1) * size] = mask * stream.blank_exposure
                number = int(frozen) - previous_count
                if number:
                    events.append(state - previous)
                    counts.append(number)
                    linear += number * previous
                previous, previous_count = state, int(frozen)
            linear += (stream.total - previous_count) * previous
        event_matrix = np.asarray(events, dtype=float).reshape((-1, width))
        event_counts = np.asarray(counts, dtype=float)
        # Zero-cost increments have an unbounded optimum. Their event terms
        # tend to zero; retain the remaining likelihood for identifiable warmer
        # parts. Never report a large optimizer stopping value as a finite fit.
        self.unbounded = linear == 0
        explained = np.any(event_matrix[:, self.unbounded] > 0, axis=1)
        self.events, self.counts = event_matrix[~explained], event_counts[~explained]
        self.linear = linear
        self.usable = ~self.unbounded
        self.physical_droplets = sum(s.total for s in series)
        self._bounds_cache: dict[bytes, tuple[float, float]] = {}
        model, inverse, scale, _ = self._reduced()
        self._column_groups = inverse
        self._group_sizes = np.bincount(inverse)
        solution, self.minimum = model.fit()
        self.increments = np.zeros(width)
        original = np.flatnonzero(self.usable)
        # Choose the coldest member of each equivalent column, deterministically.
        for group in range(len(scale)):
            position = original[np.flatnonzero(inverse == group)[-1]]
            self.increments[position] = solution[group] / scale[group]

    def _basis(self, temperature):
        """Weights of nonnegative cumulative increments at an observed temperature."""
        if self.fit_step_C is None:
            return self.temperatures >= temperature
        if temperature > self.temperatures[0] or temperature < self.temperatures[-1]:
            raise ValueError("Temperature is outside the fitting grid")
        weights = np.ones(len(self.temperatures))
        weights[1:] = np.clip(
            (self.temperatures[:-1] - temperature) / -np.diff(self.temperatures), 0, 1,
        )
        return weights

    def _reduced(self, target=None):
        design = np.vstack([self.linear[self.usable], self.events[:, self.usable]])
        if target is not None:
            design = np.vstack([design, target[self.usable]])
        columns, inverse = np.unique(design.T, axis=0, return_inverse=True)
        linear = columns[:, 0]
        events = columns[:, 1:1 + len(self.counts)].T
        scale = linear + events.T @ self.counts
        model = _Likelihood(linear / scale, events / scale, self.counts)
        direction = columns[:, -1] / scale if target is not None else None
        return model, inverse, scale, direction

    def estimate(self, temperature, confidence_drop):
        target = np.zeros(len(self.linear))
        target[:len(self.temperatures)] = self._basis(temperature)
        if np.any(target[self.unbounded]):
            return np.nan, np.nan, np.nan
        estimate = float(target @ self.increments)
        selected = np.bincount(
            self._column_groups, weights=target[self.usable], minlength=len(self._group_sizes)
        )
        state = np.where(selected == 0, 0, np.where(selected == self._group_sizes, 1, 2))
        # Linear interpolation gives distinct fractional targets within a cell.
        # They must not reuse a step-curve interval's cached profile bounds.
        key = (state.astype(np.uint8).tobytes() if self.fit_step_C is None else target.tobytes())
        key += float(confidence_drop).hex().encode()
        if key not in self._bounds_cache:
            model, inverse, scale, direction = self._reduced(target)
            initial = np.bincount(
                inverse, weights=self.increments[self.usable], minlength=len(scale)
            ) * scale
            limit = self.minimum + confidence_drop
            lower = model.endpoint(direction, initial, limit, upper=False)
            upper = model.endpoint(direction, initial, limit, upper=True)
            self._bounds_cache[key] = lower, upper
        lower, upper = self._bounds_cache[key]
        return estimate, max(0.0, estimate - lower), max(0.0, upper - estimate)
