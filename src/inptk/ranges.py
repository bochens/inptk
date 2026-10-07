"""Reviewable monotone temperature ranges for the Average workflow."""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from .alignment import align_observations
from .blank_controls import BlankControls
from .temperature_selection import CountSelector
from .experiment import Experiment
from .methods import CombinationMember, resolve_curves
from .processing import frozen_fraction
from .range_selection import RangePlanner
from .settings import DEFAULTS, EstimationSettings
from .tables import FrozenFractionTable
from .water_blank import analysis_experiment, estimate_point


@dataclass
class RangeSuggestions:
    """Named range proposals and calculation states with selection reasons.

    With a grid, observations retain the selected source IDs and measured
    temperatures in source_observation_id and observed_temperature_C. Without a
    grid, original observation identities and temperatures remain unchanged.
    Inspect ``inputs`` even when an input has no usable range. The
    ``temperature_ranges_C`` property refuses incomplete proposals, so an
    omitted input cannot accidentally regain its unrestricted temperature range.
    Edit the returned range dictionary before passing it to the usual workflow.
    """

    inputs: dict
    observations: FrozenFractionTable | None
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


def _eligible_blocks(rows):
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
    return [{"min_C": block[-1], "max_C": block[0]} for block in blocks]


def _sequential_ranges(frame, groups, experiment, *, min_frozen, min_unfrozen, grid, z, cache, controls):
    """Build count-eligible blocks, then choose monotone nonoverlapping ranges."""
    plans, membership = {}, {}
    for curve_id, group in groups.items():
        control = controls[curve_id]
        signature = tuple(member["measurement_id"] for member in group["members"])
        members = sorted(
            group["members"], key=lambda m: experiment.measurements[m["measurement_id"]].dilution
        )
        bases, rows_by_name = {}, {}
        grid_rows = {}
        if grid["temperature_step_C"] is not None:
            # Count thresholds and suggested boundaries use the same selected
            # target states as calculation, on the complete group's grid.
            points = align_observations(
                frame, members, water_blank_map={}, temperature_ranges_C={}, **grid,
            )
            records = {m["measurement_id"]: [] for m in members}
            for point in points:
                for sample in point.sample_records:
                    row = dict(sample)
                    row["observed_temperature_C"] = row["temperature_C"]
                    row["source_observation_id"] = row["observation_id"]
                    row["temperature_C"] = point.temperature_C
                    row["observation_id"] = f"{point.point_id}:{row['observation_id']}"
                    records[row["measurement_id"]].append(row)
            grid_rows = {
                name: pd.DataFrame(rows, columns=[*frame.columns, "observed_temperature_C",
                                                  "source_observation_id"])
                for name, rows in records.items()
            }
        first_dilution = min(experiment.measurements[m["measurement_id"]].dilution for m in members)
        for member in members:
            name = member["measurement_id"]
            if name in membership and membership[name] != signature:
                raise ValueError(
                    f"Input {name!r} belongs to different curve input sets. Request Average "
                    "range suggestions separately for each set; their switch points may differ."
                )
            membership[name] = signature
            rows = frame.loc[
                frame.measurement_id.eq(name)
                & frame.run_id.eq(member["run_id"])
                & frame.cycle_id.eq(member["cycle_id"])
            ].copy()
            if grid_rows:
                rows = grid_rows[name].copy()
            first = experiment.measurements[name].dilution == first_dilution
            rows["too_few_frozen"] = (rows.n_frozen < min_frozen) & (not first)
            rows["too_few_liquid"] = rows.n_total - rows.n_frozen < min_unfrozen
            rows["blank_coverage"] = True
            if control.active:
                # Pre-onset zero needs no observed blank at that target. Assess
                # native support in arrays; grid trials still use actual selections.
                rows["blank_coverage"] = control.native_coverage(member, rows.temperature_C)
            else:
                for blank in experiment.water_blank_map.get(name, []):
                    observed = frame.loc[
                        frame.measurement_id.eq(blank)
                        & frame.run_id.eq(member["run_id"])
                        & frame.cycle_id.eq(member["cycle_id"])
                    ]
                    rows["blank_coverage"] &= rows.temperature_C.between(
                        observed.temperature_C.min(), observed.temperature_C.max()
                    )
                    if grid_rows and grid["temperature_method"] == "window":
                        selector = CountSelector(observed)
                        rows["blank_coverage"] &= np.array([
                            selector.select(t, method="window", window_C=grid["temperature_window_C"])
                            is not None for t in rows.temperature_C
                        ], dtype=bool)
            rows["range_eligible"] = (
                ~rows.too_few_frozen & ~rows.too_few_liquid & rows.blank_coverage
            )
            rows["temperature_eligible"] = rows.groupby("temperature_C").range_eligible.transform(
                "all"
            )
            bases[name] = _eligible_blocks(rows)
            rows_by_name[name] = rows
        planner = RangePlanner(
            frame, members, experiment, bases, grid, z, cache, control, rows_by_name
        )
        chosen = planner.choose([m["measurement_id"] for m in members])
        previous_cold = None
        for member in members:
            name = member["measurement_id"]
            rows = rows_by_name[name]
            selected = chosen.get(name)
            limits = selected["limits"] if selected is not None else None
            rows["previous_dilution_active"] = (
                False if previous_cold is None else rows.temperature_C.ge(previous_cold)
            )
            rows["monotone_limit_reason"] = ""
            if limits is None:
                rows.loc[rows.temperature_eligible, "monotone_limit_reason"] = (
                    "no_monotone_continuation"
                )
            else:
                eligible = rows.temperature_eligible & ~rows.previous_dilution_active
                rows.loc[eligible & rows.temperature_C.gt(limits["max_C"]),
                         "monotone_limit_reason"] = selected["warm_reason"]
                rows.loc[eligible & rows.temperature_C.lt(limits["min_C"]),
                         "monotone_limit_reason"] = selected["cold_reason"]
                previous_cold = limits["min_C"]
            first = experiment.measurements[name].dilution == first_dilution
            plans[name] = (rows, limits, first)
    return plans


def suggest_temperature_ranges(
    experiment: Experiment,
    *,
    curves=None,
    include_observations: bool = True,
    min_frozen: int = 3,
    min_unfrozen: int = 3,
    water_blank_correction: bool = DEFAULTS.water_blank_correction,
    water_blank_after_first_freeze: bool = DEFAULTS.water_blank_after_first_freeze,
    water_blank_temperature_range_C=None,
    z: float = 1.96,
    temperature_step_C: float | None = DEFAULTS.temperature_step_C,
    temperature_start_C: float | None = DEFAULTS.temperature_start_C,
    temperature_end_C: float | None = DEFAULTS.temperature_end_C,
    temperature_method: str = DEFAULTS.temperature_method,
    temperature_window_C: float | None = DEFAULTS.temperature_window_C,
) -> RangeSuggestions:
    """Suggest nonoverlapping Average ranges for nonnegative, nondecreasing output.

    Exhaust the least diluted input's initial eligible interval, stopping before
    saturation, a negative value or a decrease in directly corrected concentration.
    Initial negative values move the warm limit forward; zero remains eligible.
    Values before the first sample freeze cannot set the first input's cutoff,
    because they are outside the reported combined spectrum. A later input must
    continue at or above the preceding concentration; shorten the preceding range
    if needed for a handoff. Stop when no such continuation exists. Equal dilution
    inputs are ordered as supplied in the curve and also receive nonoverlapping ranges.

    Use the SAME curves, grid, temperature rule and blank correction for analysis.
    Each trial uses the actual aligned sample and blank counts. No concentration
    values are changed and no final decrease filter is needed for these ranges.
    Count thresholds are editable heuristics, not confidence criteria. The first
    dilution retains its initial observations without a minimum frozen count.
    With a grid, count thresholds and proposed limits use selected grid states;
    without a grid, repeated temperatures must pass in every observation. Cycles
    remain separate.
    Intervals reaching zero are flagged, not excluded for that reason. Reported
    confidence intervals do not include the uncertainty of selecting these ranges.
    Set include_observations=False for limits and reasons without constructing the
    per-observation diagnostic report. This does not change the proposed limits.
    """
    if not isinstance(include_observations, bool):
        raise TypeError("include_observations must be a bool (True or False)")
    EstimationSettings(
        method="average", z=z, water_blank_correction=water_blank_correction,
        water_blank_after_first_freeze=water_blank_after_first_freeze,
        water_blank_temperature_range_C=water_blank_temperature_range_C,
        temperature_step_C=temperature_step_C,
        temperature_method=temperature_method,
        temperature_window_C=temperature_window_C,
        temperature_start_C=temperature_start_C,
        temperature_end_C=temperature_end_C,
    )
    grid = {
        "temperature_step_C": temperature_step_C,
        "temperature_start_C": temperature_start_C,
        "temperature_end_C": temperature_end_C,
        "temperature_method": temperature_method,
        "temperature_window_C": temperature_window_C,
    }
    min_frozen = _positive_count(min_frozen, "min_frozen")
    min_unfrozen = _positive_count(min_unfrozen, "min_unfrozen")
    view = analysis_experiment(experiment, water_blank_correction=water_blank_correction)
    fractions = frozen_fraction(view)
    frame = fractions.to_dataframe()
    groups = resolve_curves(curves, view, frame)
    members = _selected_members(groups)
    if not members:
        raise ValueError("No sample inputs selected for range suggestions")
    raw_counts = view.counts.to_dataframe() if (
        view.water_blank_map and (water_blank_after_first_freeze or water_blank_temperature_range_C)
    ) else None
    controls = {
        curve_id: BlankControls(
            view, group["members"], raw_counts=raw_counts,
            after_first_freeze=water_blank_after_first_freeze,
            temperature_range_C=water_blank_temperature_range_C,
        ) for curve_id, group in groups.items()
    }
    controls_by_name = {
        member["measurement_id"]: controls[curve_id]
        for curve_id, group in groups.items() for member in group["members"]
    }
    cache: dict = {}
    plans = _sequential_ranges(
        frame,
        groups,
        view,
        min_frozen=min_frozen,
        min_unfrozen=min_unfrozen,
        grid=grid,
        z=float(z),
        cache=cache,
        controls=controls,
    )
    proposals, reports = {}, []
    for name, member in members.items():
        rows, limits, first_dilution = plans[name]
        rows["in_suggested_range"] = False
        if limits is not None:
            rows["in_suggested_range"] = rows.temperature_C.between(
                limits["min_C"], limits["max_C"]
            )
        rows["range_exclusion_reasons"] = _exclusion_reasons(rows)
        if include_observations:
            rows["blank_status"] = "not_assessed_outside_range"
            rows["concentration_unit"] = "INP_per_mL_suspension"
            for column in ("concentration", "lower_error", "upper_error"):
                rows[column] = np.nan
            rows["blank_observation_ids"] = "[]"
            if limits is not None and not view.water_blank_map:
                rows.loc[rows.in_suggested_range, "blank_status"] = "not_applied"
            elif limits is not None:
                _fill_observation_report(
                    rows, name, member, limits, view=view, frame=frame, plans=plans,
                    grid=grid, temperature_step_C=temperature_step_C,
                    controls_by_name=controls_by_name, cache=cache, z=z,
                )
        proposals[name] = {
            "run_id": member["run_id"],
            "cycle_id": member["cycle_id"],
            "range_C": limits,
            "dilution": view.measurements[name].dilution,
            "min_frozen_applied": not first_dilution,
            "status": "suggested" if limits is not None else "no_usable_range",
            "kept_observations": int(rows.in_suggested_range.sum()),
            "excluded_observations": int((~rows.in_suggested_range).sum()),
            "warm_limit_reason": _edge_reason(rows, limits, warm=True),
            "cold_limit_reason": _edge_reason(rows, limits, warm=False),
        }
        if include_observations:
            reports.append(rows)
    settings = {
        "operation": "suggest_temperature_ranges",
        "intended_method": "average",
        "min_frozen": min_frozen,
        "min_unfrozen": min_unfrozen,
        "z": float(z),
        "water_blank_correction": water_blank_correction,
        "water_blank_after_first_freeze": water_blank_after_first_freeze,
        "water_blank_temperature_range_C": next(iter(controls.values())).limits,
        "water_blank_controls": {key: control.details for key, control in controls.items()},
        "rule": "ascending dilution; nonoverlapping nonnegative monotone intervals and handoffs",
        "negative_concentrations": "adjust contiguous input limits; retain individual values",
        "range_check_starts": "each input's original first freezing event",
        **grid,
        "first_dilution": "retain first frozen observations; at least one frozen sample well",
        "same_dilution": "nonoverlapping; curve input order breaks dilution ties",
        "repeated_temperatures": "all selected states must pass",
        "range_domain": "calculation grid" if temperature_step_C is not None else "native targets",
        "blank_flags_change_ranges": False,
        "uncertainty": "log-transformed Wilson binomial bounds with approximate propagation; "
        "excludes range-selection uncertainty",
        "inputs": proposals,
    }
    return RangeSuggestions(
        proposals,
        FrozenFractionTable(
            pd.concat(reports, ignore_index=True), history=fractions.history + [settings]
        )
        if include_observations
        else None,
        settings,
    )


def _selected_members(groups) -> dict[str, CombinationMember]:
    """Each selected measurement's member; a measurement may use only one cycle."""
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
    return members


def _exclusion_reasons(rows) -> list[str]:
    """Why each observation is outside the suggested range, as JSON lists."""
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
        if row.monotone_limit_reason and not excluded:
            excluded.append(row.monotone_limit_reason)
        if not excluded and not row.in_suggested_range:
            excluded.append(
                "outside_selected_contiguous_block"
                if row.temperature_eligible
                else "another_observation_at_same_temperature_failed"
            )
        reasons.append(json.dumps(excluded))
    return reasons


def _fill_observation_report(
    rows, name, member, limits, *, view, frame, plans, grid, temperature_step_C,
    controls_by_name, cache, z,
):
    """Add Average concentrations and blank status for the observations in range."""
    # Observation IDs are unique within this selected measurement/cycle.
    # Build the report in arrays instead of searching and assigning
    # individual DataFrame cells for every original image.
    gridded = temperature_step_C is not None
    positions = {value: i for i, value in enumerate(
        rows.temperature_C if gridded else rows.observation_id
    )}
    concentrations = np.full((len(rows), 3), np.nan)
    statuses = rows.blank_status.to_numpy(copy=True)
    blank_ids = rows.blank_observation_ids.to_numpy(copy=True)
    points = controls_by_name[name].align(
        frame,
        [member],
        temperature_ranges_C={name: limits},
        **({**grid, "temperature_start_C": plans[name][0].temperature_C.max(),
            "temperature_end_C": plans[name][0].temperature_C.min()}
           if gridded else {}),
    )
    for point in points:
        if point.samples.empty:
            continue
        sample = point.sample_records[0]
        index = positions[point.temperature_C if gridded else sample["observation_id"]]
        key = point.count_key()
        if key not in cache:
            cache[key] = estimate_point(
                point.samples,
                point.blanks,
                view,
                method="average",
                z=float(z),
                blank_state=point.blank_state,
            )
        concentration, lower, upper, finite = cache[key]
        concentrations[index] = (concentration, lower, upper)
        lower_bound = concentration - lower
        status = (
            "uncertainty_unavailable"
            if not finite
            else (
                "not_distinguished_from_blank"
                if lower_bound <= 0 or np.isclose(lower_bound, 0, rtol=0, atol=1e-12)
                else "distinguished_from_blank"
            )
        )
        statuses[index] = status
        blank_ids[index] = json.dumps(
            [
                {
                    key: str(r[key])
                    for key in (
                        "measurement_id",
                        "run_id",
                        "cycle_id",
                        "observation_id",
                    )
                }
                for r in point.blank_records
            ]
        )
    rows[["concentration", "lower_error", "upper_error"]] = concentrations
    rows["blank_status"] = statuses
    rows["blank_observation_ids"] = blank_ids


def _edge_reason(rows, limits, *, warm):
    if limits is None:
        return ["no_usable_range"]
    beyond = rows.loc[
        rows.temperature_C.gt(limits["max_C"]) if warm else rows.temperature_C.lt(limits["min_C"])
    ]
    if beyond.empty:
        return ["observed_temperature_limit"]
    adjacent = beyond.temperature_C.min() if warm else beyond.temperature_C.max()
    return sorted(
        {
            reason
            for value in beyond.loc[beyond.temperature_C.eq(adjacent), "range_exclusion_reasons"]
            for reason in json.loads(value)
        }
    )
