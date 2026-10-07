"""Estimate concentrations from explicitly selected sample and blank observations."""

from __future__ import annotations

import json
from typing import Literal

import numpy as np
import pandas as pd

from .blank_controls import BlankControls
from .context import prepare_fraction_analysis
from .experiment import Experiment
from .methods import curve_specifications, resolve_curves, validate_temperature_ranges
from .reporting import freezing_intervals, mark_reporting_intervals
from .settings import DEFAULTS, EstimationSettings
from .tables import CurveSpectrumTable, FrozenFractionTable
from .water_blank import estimate_point, sample_rows


def _state_key(point) -> tuple:
    return point.count_key()


def _sources(point) -> list[dict]:
    records = []
    for role, rows in (("sample", point.sample_records), ("blank", point.blank_records)):
        for row in rows:
            item: dict[str, object] = {
                key: str(row[key])
                for key in ("measurement_id", "run_id", "cycle_id", "observation_id")
            }
            item.update(
                n_frozen=int(row["n_frozen"]),
                n_total=int(row["n_total"]),
                fraction_frozen=float(row["n_frozen"] / row["n_total"]),
                role=role,
                observed_temperature_C=float(row["temperature_C"]),
                alignment=row.get(
                    "temperature_selection",
                    "exact" if row["temperature_C"] == point.temperature_C else "latest",
                ),
            )
            if "fit_temperature_C" in row:
                item["selected_temperature_C"] = float(row["fit_temperature_C"])
            if "time_s" in row and pd.notna(row["time_s"]):
                item["time_s"] = float(row["time_s"])
            records.append(item)
    return records


def estimate_concentration(
    fractions: FrozenFractionTable,
    *,
    experiment: Experiment,
    method: Literal["mle", "average"] = DEFAULTS.method,
    fit_step_C: float | None = DEFAULTS.fit_step_C,
    temperature_ranges_C=None,
    temperature_step_C: float | None = DEFAULTS.temperature_step_C,
    temperature_start_C: float | None = DEFAULTS.temperature_start_C,
    temperature_end_C: float | None = DEFAULTS.temperature_end_C,
    temperature_method: Literal["latest", "max", "window"] = DEFAULTS.temperature_method,
    temperature_window_C: float | None = DEFAULTS.temperature_window_C,
    curves=None,
    z: float = DEFAULTS.z,
    water_blank_correction: bool = DEFAULTS.water_blank_correction,
    water_blank_after_first_freeze: bool = DEFAULTS.water_blank_after_first_freeze,
    water_blank_temperature_range_C=None,
) -> CurveSpectrumTable:
    """Estimate named curves from original states with separate run backgrounds.

    With no curves supplied, each sample/run/cycle remains separate. Explicit
    curves can span runs but select only one cycle from each run. Observations
    align only where needed, using latest warmer states at observed targets.
    MLE fits selected freezing histories with monotone sample and blank curves;
    Average estimates each target separately. Original count rows stay unchanged.
    temperature_step_C instead selects count pairs on a regular grid before
    either estimator. latest/max use warmer states; window uses a full-width
    centered window. Sample and blank use the same rule, before correction.
    Optional start/end bounds are exact warm/cold grid endpoints; omitted
    bounds use observed limits. Both endpoints are retained without rounding.
    Rows outside original sample freezing limits retain their identities with
    NaN concentrations and errors. Each MLE input contributes only within its
    own original first-to-last freezing interval.
    Combined Average uses positive sample counts inside each input's event
    interval, intersected with user ranges. Input temperature
    ranges select aligned calculation states only; selecting the full useful
    span is equivalent to leaving its limits unset. Ranges apply only to curves
    combining several physical inputs. Individual
    curves use their own freezing interval, independently of those exclusions.
    water_blank_temperature_range_C is one observation range shared by all assigned
    controls; it does not restrict sample temperatures. water_blank_after_first_freeze
    optionally fixes each run/cycle's background at zero before its combined control's
    original first freeze, including uncertainty. Both settings precede estimation.
    """
    EstimationSettings(
        method=method,
        fit_step_C=fit_step_C,
        temperature_step_C=temperature_step_C,
        temperature_start_C=temperature_start_C,
        temperature_end_C=temperature_end_C,
        temperature_method=temperature_method,
        temperature_window_C=temperature_window_C,
        z=z,
        water_blank_correction=water_blank_correction,
        water_blank_after_first_freeze=water_blank_after_first_freeze,
        water_blank_temperature_range_C=water_blank_temperature_range_C,
    )
    fractions, experiment = prepare_fraction_analysis(
        fractions, experiment, water_blank_correction=water_blank_correction
    )
    frame = fractions.to_dataframe()
    source = sample_rows(frame, experiment)
    if source.empty:
        raise ValueError("No sample observations are available for concentration calculation")
    blank_ids = experiment.water_blank_ids
    ranges = validate_temperature_ranges(
        temperature_ranges_C, measurement_ids=set(experiment.measurements) - blank_ids
    )
    groups = resolve_curves(curves, experiment, frame)
    input_groups = {
        (member["measurement_id"], member["run_id"], member["cycle_id"]): {"members": [member]}
        for group in groups.values() for member in group["members"]
    }
    input_intervals = freezing_intervals(source, input_groups, {})
    records, notices, group_alignment, cache = [], [], {}, {}
    joint_fits, controls, resolved_ranges, intervals = {}, {}, {}, {}
    raw_counts = experiment.counts.to_dataframe() if (
        experiment.water_blank_map
        and (water_blank_after_first_freeze or water_blank_temperature_range_C)
    ) else None
    for curve_id, group in groups.items():
        members = group["members"]
        individual = len(members) == 1
        ids = sorted(member["measurement_id"] for member in members)
        supports = {}
        for member in members:
            selected = source[
                (source.measurement_id == member["measurement_id"])
                & (source.cycle_id == member["cycle_id"])
            ]
            supports[member["measurement_id"]] = (
                float(selected.temperature_C.min()),
                float(selected.temperature_C.max()),
            )
        control = BlankControls(
            experiment, members, raw_counts=raw_counts,
            after_first_freeze=water_blank_after_first_freeze,
            temperature_range_C=water_blank_temperature_range_C,
        )
        controls[curve_id] = control.details
        points = control.align(
            frame,
            members,
            temperature_ranges_C={} if individual else ranges,
            temperature_step_C=temperature_step_C,
            temperature_start_C=temperature_start_C,
            temperature_end_C=temperature_end_C,
            temperature_method=temperature_method,
            temperature_window_C=temperature_window_C,
            require_partial_freezing=method == "average",
            sample_freezing_intervals_C={
                member["measurement_id"]: input_intervals[
                    (member["measurement_id"], member["run_id"], member["cycle_id"])
                ] for member in members
            } if method == "mle" or not individual else None,
        )
        resolved_ranges[curve_id] = points[0].range_details
        intervals.update(freezing_intervals(
            source, {curve_id: group},
            {name: details["calculation_limits_C"]
             for name, details in resolved_ranges[curve_id].items()},
        ))
        if method == "mle":
            from dataclasses import replace

            from .curve_fit import fit_curve

            estimates, joint_fits[curve_id] = fit_curve(
                points, experiment, z=z, fit_step_C=fit_step_C, blank_controls=control
            )
            # Evaluate each selected temperature once. Repeated observations
            # remain in the experiment and never become independent droplets.
            by_temperature = {point.temperature_C: point for point in points}
            points = [
                replace(by_temperature[temperature], point_order=index, point_id=f"point:{index}")
                for index, temperature in enumerate(sorted(by_temperature, reverse=True))
            ]
        empty_count = 0
        group_alignment[curve_id] = sorted({point.alignment for point in points})
        for point in points:
            contributors = sorted(str(row["measurement_id"]) for row in point.sample_records)
            available = sorted(
                key for key, (cold, warm) in supports.items() if cold <= point.temperature_C <= warm
            )
            record = {
                "sample_id": group["sample_id"],
                "curve_id": curve_id,
                "curve_kind": "individual" if individual else "combined",
                "point_id": point.point_id,
                "point_order": point.point_order,
                "temperature_C": point.temperature_C,
                "alignment": point.alignment,
                "concentration": np.nan,
                "lower_error": np.nan,
                "upper_error": np.nan,
                "unit": "INP_per_mL_suspension",
                "basis": "suspension",
                "qc_flag": 1,
                "source_measurement_ids": json.dumps(ids),
                "available_measurement_ids": json.dumps(available),
                "contributing_measurement_ids": json.dumps(contributors),
                "contributor_count": len(contributors),
                "source_measurement_id": contributors[0] if len(contributors) == 1 else "",
                "selection_status": "no_eligible_measurements"
                if not contributors
                else "single"
                if len(contributors) == 1
                else "combined",
                "dilution_fold": experiment.measurements[contributors[0]].dilution
                if len(contributors) == 1
                else np.nan,
                "water_blank_ids": json.dumps(
                    sorted(str(row["measurement_id"]) for row in point.blank_records)
                ),
                "source_observations": json.dumps(_sources(point)),
                "uncertainty_method": "joint_curve_profile_likelihood"
                if method == "mle"
                else "propagated_wilson_binomial_bounds",
                "correction_state": "water_blank_corrected"
                if experiment.water_blank_map
                else "uncorrected",
            }
            if contributors:
                if method == "mle":
                    estimate, lower, upper = estimates[point.temperature_C]
                    finite = bool(np.isfinite([estimate, lower, upper]).all())
                    fit = (estimate, lower, upper, finite)
                else:
                    if point.blank_state is not None and point.blank_state.unavailable:
                        record["selection_status"] = "missing_blank_observations"
                    key = _state_key(point)
                    if key not in cache:
                        cache[key] = estimate_point(
                            point.samples,
                            point.blanks,
                            experiment,
                            z=z,
                            method=method,
                            blank_state=point.blank_state,
                        )
                    fit = cache[key]
                record.update(
                    concentration=fit[0],
                    lower_error=fit[1],
                    upper_error=fit[2],
                    qc_flag=0 if fit[3] else 1,
                )
            else:
                empty_count += 1
            record["at_zero_boundary"] = method == "mle" and record["concentration"] == 0
            records.append(record)
        if empty_count:
            notices.append(f"Curve {curve_id!r}: {empty_count} points have no eligible input")
    settings = {
        "operation": "estimate_concentration",
        "reporting_rule": "observed_sample_freezing_interval",
        "reporting_intervals_C": intervals,
        "estimation_method": method,
        "fit_step_C": fit_step_C,
        "curves": curve_specifications(groups),
        "curve_sources": groups,
        "temperature_ranges_C": ranges,
        "temperature_range_scope": "combined_curves_only",
        "resolved_temperature_ranges_C": resolved_ranges,
        "temperature_step_C": temperature_step_C,
        "temperature_start_C": temperature_start_C,
        "temperature_end_C": temperature_end_C,
        "temperature_method": temperature_method,
        "temperature_window_C": temperature_window_C,
        "range_boundaries": "inclusive calculation targets; source rows unchanged",
        "alignment": group_alignment,
        "alignment_rule": temperature_method
        if temperature_step_C is not None
        else "latest observation at or warmer than target, only where needed",
        "z": float(z),
        **({"confidence_drop": float(z**2 / 2)} if method == "mle" else {}),
        "water_blank_correction": water_blank_correction,
        "water_blank_correction_applied": bool(experiment.water_blank_map),
        "water_blank_model": "volume_scaled",
        "water_blank_after_first_freeze": water_blank_after_first_freeze,
        "water_blank_temperature_range_C": control.limits,
        "water_blank_range_scope": "blank_observations_only; shared by assigned controls",
        "water_blank_controls": controls,
        "background_groups": "separate by run and selected cycle",
        "water_blank_map": dict(experiment.water_blank_map),
        "joint_curve_fits": joint_fits,
        **({"average_blank_aggregation": "equal-volume counts pooled; different volumes "
            "weighted by total assayed volume",
            "average_contributors": "sample counts with 0 < n_frozen < n_total within "
            "each input's original first-to-last freezing interval, intersected with selected ranges"}
           if method == "average" else {}),
        "uncertainty_assumption": (
            "Independent physical droplet sets, repeated observations never pooled; "
            + ("joint curve profile bounds have approximate coverage" if method == "mle" else
               "log-transformed Wilson binomial bounds with approximate asymmetric "
               "quadrature propagation; shared blank uncertainty counted once")
        ),
        "warnings": notices,
    }
    return CurveSpectrumTable(
        mark_reporting_intervals(pd.DataFrame.from_records(records), intervals),
        history=fractions.history + [settings]
    )
