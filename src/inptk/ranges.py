"""Reviewable count-based temperature ranges for the Average workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from .alignment import align_observations
from .experiment import Experiment
from .methods import CombinationMember, resolve_curves
from .processing import frozen_fraction
from .tables import FrozenFractionTable
from .water_blank import analysis_experiment, estimate_point


@dataclass
class RangeSuggestions:
    """Named range proposals and original observations with selection reasons.

    Inspect ``inputs`` even when an input has no usable range. The
    ``temperature_ranges_C`` property refuses incomplete proposals, so an
    omitted input cannot accidentally regain its unrestricted temperature range.
    Edit the returned range dictionary before passing it to the usual workflow.
    """

    inputs: dict
    observations: FrozenFractionTable
    settings: dict

    @property
    def temperature_ranges_C(self) -> dict[str, dict[str, float]]:
        missing = [name for name, proposal in self.inputs.items() if proposal["range_C"] is None]
        if missing:
            raise ValueError(
                f"No usable temperature range for {missing}. Review thresholds or remove these "
                "inputs from the requested curves and request suggestions again."
            )
        return {name: dict(proposal["range_C"]) for name, proposal in self.inputs.items()}


def _positive_count(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _choose_block(rows):
    # A single temperature limit cannot select different images at the same T.
    # Require every row at that T to pass. Never bridge an ineligible temperature.
    states = rows.groupby("temperature_C", sort=True).range_eligible.all().iloc[::-1]
    blocks, active = [], []
    for temperature, eligible in states.items():
        if eligible:
            active.append(float(temperature))
        elif active:
            blocks.append(active)
            active = []
    if active:
        blocks.append(active)
    if not blocks:
        return None
    # Start at the warm end and stop at the first failing temperature.
    # Never abandon the initial block for a longer, colder block.
    block = blocks[0]
    return {"min_C": block[-1], "max_C": block[0]}


def _sequential_ranges(frame, groups, experiment, *, min_frozen, min_unfrozen):
    """Exhaust each dilution before admitting a more dilute input.

    Equal-dilution inputs share a stage and may overlap. Different dilution
    stages never overlap, including at a shared boundary temperature.
    """
    plans, membership = {}, {}
    for group in groups.values():
        signature = frozenset(member["measurement_id"] for member in group["members"])
        stages = {}
        for member in group["members"]:
            name = member["measurement_id"]
            if name in membership and membership[name] != signature:
                raise ValueError(
                    f"Input {name!r} belongs to different curve input sets. Request Average "
                    "range suggestions separately for each set; their switch points may differ."
                )
            membership[name] = signature
            stages.setdefault(experiment.measurements[name].dilution, []).append(member)
        previous_cold = None
        for stage, dilution in enumerate(sorted(stages)):
            cold_limits = []
            for member in stages[dilution]:
                name = member["measurement_id"]
                rows = frame.loc[
                    frame.measurement_id.eq(name) & frame.run_id.eq(member["run_id"])
                    & frame.cycle_id.eq(member["cycle_id"])
                ].copy()
                # Keep the initial zero/one/two-frozen-well states of the first
                # dilution. Later stages still need a usable positive signal.
                rows["too_few_frozen"] = (rows.n_frozen < min_frozen) & (stage > 0)
                rows["too_few_liquid"] = rows.n_total - rows.n_frozen < min_unfrozen
                rows["previous_dilution_active"] = (
                    False if previous_cold is None else rows.temperature_C.ge(previous_cold)
                )
                rows["blank_coverage"] = True
                for blank in experiment.water_blank_map.get(name, []):
                    observed = frame.loc[
                        frame.measurement_id.eq(blank) & frame.run_id.eq(member["run_id"])
                        & frame.cycle_id.eq(member["cycle_id"])
                    ]
                    rows["blank_coverage"] &= rows.temperature_C.between(
                        observed.temperature_C.min(), observed.temperature_C.max()
                    )
                rows["range_eligible"] = (
                    ~rows.too_few_frozen & ~rows.too_few_liquid & rows.blank_coverage
                    & ~rows.previous_dilution_active
                )
                rows["temperature_eligible"] = rows.groupby(
                    "temperature_C"
                ).range_eligible.transform("all")
                limits = _choose_block(rows)
                plans[name] = (rows, limits, stage == 0)
                if limits is not None:
                    cold_limits.append(limits["min_C"])
            if cold_limits:
                previous_cold = min(cold_limits)
    return plans


def suggest_temperature_ranges(
    experiment: Experiment,
    *,
    curves=None,
    min_frozen: int = 3,
    min_unfrozen: int = 3,
    water_blank_correction: bool = True,
    z: float = 1.96,
) -> RangeSuggestions:
    """Suggest inclusive Average ranges without modifying data or running an average.

    Use each dilution from warm to cold before switching to the next dilution.
    Keep the first dilution's initial observations, including zero frozen wells.
    Later dilutions require ``min_frozen`` wells and start strictly colder than
    the previous stage's cold limit. All require ``min_unfrozen`` liquid wells
    and, when enabled, raw blank coverage. Stop each range at its first failing
    temperature. Equal-dilution inputs may overlap and be averaged together.
    Thresholds default to three wells as an editable heuristic, not a validated
    confidence criterion.
    Repeated temperatures must pass in every observation. Select one cycle per
    measurement using ``curves``; repeated cycles never share an inferred range.

    Within proposed ranges, fit individual sample/blank count states using the
    existing Average model. An interval reaching zero is flagged, never excluded
    for that reason. Concentration decreases never determine these suggestions.
    Confidence intervals are approximate and do not include selection uncertainty.
    """
    min_frozen = _positive_count(min_frozen, "min_frozen")
    min_unfrozen = _positive_count(min_unfrozen, "min_unfrozen")
    if isinstance(z, bool) or not np.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    view = analysis_experiment(experiment, water_blank_correction=water_blank_correction)
    fractions = frozen_fraction(view)
    frame = fractions.to_dataframe()
    groups = resolve_curves(curves, view, frame)
    members: dict[str, CombinationMember] = {}
    for group in groups.values():
        for member in group["members"]:
            name = member["measurement_id"]
            if name in members and members[name] != member:
                raise ValueError(
                    f"Range suggestions require one cycle per measurement: {name!r}. "
                    "Request each cycle separately using curves."
                )
            members[name] = member
    if not members:
        raise ValueError("No sample inputs selected for range suggestions")
    plans = _sequential_ranges(
        frame, groups, view, min_frozen=min_frozen, min_unfrozen=min_unfrozen,
    )
    proposals, reports = {}, []
    cache = {}
    for name, member in members.items():
        rows, limits, first_dilution = plans[name]
        rows["in_suggested_range"] = False
        if limits is not None:
            rows["in_suggested_range"] = rows.temperature_C.between(
                limits["min_C"], limits["max_C"]
            )
        reasons = []
        for row in rows.itertuples():
            excluded = []
            if row.too_few_frozen:
                excluded.append("too_few_frozen")
            if row.too_few_liquid:
                excluded.append("too_few_liquid")
            if not row.blank_coverage:
                excluded.append("missing_blank_coverage")
            if row.previous_dilution_active:
                excluded.append("previous_dilution_active")
            if not excluded and not row.in_suggested_range:
                excluded.append("outside_selected_contiguous_block" if row.temperature_eligible
                                else "another_observation_at_same_temperature_failed")
            reasons.append(json.dumps(excluded))
        rows["range_exclusion_reasons"] = reasons
        rows["blank_status"] = "not_assessed_outside_range"
        rows["concentration_unit"] = "INP_per_mL_suspension"
        for column in ("concentration", "lower_error", "upper_error"):
            rows[column] = np.nan
        rows["blank_observation_ids"] = "[]"
        if limits is not None:
            points = align_observations(
                frame, [member], water_blank_map=view.water_blank_map,
                temperature_ranges_C={name: limits},
            )
            for point in points:
                if point.samples.empty:
                    continue
                sample = point.samples.iloc[0]
                index = rows.index[rows.observation_id.eq(sample.observation_id)][0]
                if not view.water_blank_map:
                    rows.loc[index, "blank_status"] = "not_applied"
                    continue
                key = (name, member["cycle_id"], int(sample.n_frozen), int(sample.n_total),
                       tuple((r.measurement_id, r.n_frozen, r.n_total)
                             for r in point.blanks.itertuples()))
                if key not in cache:
                    cache[key] = estimate_point(
                        point.samples, point.blanks, view, method="average",
                        confidence_drop=float(z)**2 / 2,
                    )
                concentration, lower, upper, finite = cache[key]
                rows.loc[index, ["concentration", "lower_error", "upper_error"]] = (
                    concentration, lower, upper
                )
                lower_bound = concentration - lower
                status = "uncertainty_unavailable" if not finite else (
                    "not_distinguished_from_blank" if lower_bound <= 0 or np.isclose(
                        lower_bound, 0, rtol=0, atol=1e-12
                    ) else "distinguished_from_blank"
                )
                rows.loc[index, "blank_status"] = status
                rows.loc[index, "blank_observation_ids"] = json.dumps([
                    {key: str(getattr(r, key)) for key in
                     ("measurement_id", "run_id", "cycle_id", "observation_id")}
                    for r in point.blanks.itertuples()
                ])
        proposals[name] = {
            "run_id": member["run_id"], "cycle_id": member["cycle_id"], "range_C": limits,
            "dilution": view.measurements[name].dilution,
            "min_frozen_applied": not first_dilution,
            "status": "suggested" if limits is not None else "no_usable_range",
            "kept_observations": int(rows.in_suggested_range.sum()),
            "excluded_observations": int((~rows.in_suggested_range).sum()),
            "warm_limit_reason": _edge_reason(rows, limits, warm=True),
            "cold_limit_reason": _edge_reason(rows, limits, warm=False),
        }
        reports.append(rows)
    settings = {
        "operation": "suggest_temperature_ranges", "intended_method": "average",
        "min_frozen": min_frozen, "min_unfrozen": min_unfrozen, "z": float(z),
        "water_blank_correction": water_blank_correction,
        "rule": "ascending dilution; exhaust each stage's first eligible block before the next",
        "first_dilution": "preserve initial observations; no minimum frozen count",
        "same_dilution": "may overlap; advance after all inputs in the stage end",
        "repeated_temperatures": "all observations must pass",
        "blank_flags_change_ranges": False,
        "uncertainty": "individual pointwise profile bounds; excludes range-selection uncertainty",
        "inputs": proposals,
    }
    return RangeSuggestions(
        proposals, FrozenFractionTable(pd.concat(reports, ignore_index=True),
                                       history=fractions.history + [settings]), settings,
    )


def _edge_reason(rows, limits, *, warm):
    if limits is None:
        return ["no_usable_range"]
    beyond = rows.loc[rows.temperature_C.gt(limits["max_C"]) if warm
                      else rows.temperature_C.lt(limits["min_C"])]
    if beyond.empty:
        return ["observed_temperature_limit"]
    adjacent = beyond.temperature_C.min() if warm else beyond.temperature_C.max()
    return sorted({reason for value in beyond.loc[
        beyond.temperature_C.eq(adjacent), "range_exclusion_reasons"
    ] for reason in json.loads(value)})
