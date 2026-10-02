"""Individual processing steps, with identities and preceding history preserved."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .alignment import align_observations
from .experiment import Experiment
from .methods import validate_temperature_ranges
from .tables import (
    CountsTable,
    CumulativeSpectrumTable,
    DifferentialSpectrumTable,
    FrozenFractionTable,
)


def validate_fraction_context(
    fractions: FrozenFractionTable, experiment: Experiment
) -> pd.DataFrame:
    """Validate measurement ownership and return a copy of fraction rows."""
    if not isinstance(fractions, FrozenFractionTable):
        raise TypeError("fractions must be a FrozenFractionTable")
    if not isinstance(experiment, Experiment):
        raise TypeError("experiment must be an Experiment")
    frame = fractions.to_dataframe()
    if frame.empty:
        raise ValueError("Cannot calculate a spectrum from an empty frozen-fraction table")
    identities = frame[["measurement_id", "sample_id", "run_id"]].drop_duplicates()
    for row in identities.itertuples(index=False):
        measurement_id = str(row.measurement_id)
        if measurement_id not in experiment.measurements:
            raise ValueError(f"Missing measurement metadata for {measurement_id!r}")
        measurement = experiment.measurements[measurement_id]
        if (row.sample_id, row.run_id) != (measurement.sample_id, measurement.run_id):
            raise ValueError(
                f"Frozen-fraction identities disagree with metadata for {measurement_id!r}"
            )
        if measurement.sample_id not in experiment.samples:
            raise ValueError(f"Missing sample metadata for {measurement.sample_id!r}")
    return frame


def prepare_fraction_analysis(
    fractions: FrozenFractionTable,
    experiment: Experiment,
    *,
    water_blank_correction: bool,
) -> tuple[FrozenFractionTable, Experiment]:
    """Validate identities and remove blank context only from a disabled analysis view."""
    from .water_blank import analysis_experiment, sample_rows

    validate_fraction_context(fractions, experiment)
    view = analysis_experiment(
        experiment,
        water_blank_correction=water_blank_correction,
    )
    frame = fractions.to_dataframe()
    if any(
        step.get("operation") == "frozen_fraction"
        and step.get("temperature_method") == "window_max_count"
        for step in fractions.history
    ) or (
        "temperature_bin_method" in frame
        and frame.temperature_bin_method.astype(str)
        .str.contains("window_max_count", regex=False)
        .any()
    ):
        raise ValueError(
            "Concentration estimation does not support window_max_count: its synthetic "
            "warm zero rows are not raw measurements; use original observed counts"
        )
    if view is not experiment:
        rows = sample_rows(fractions.to_dataframe(), experiment)
        if rows.empty:
            raise ValueError("No sample observations remain after excluding water-blank sets")
        fractions = FrozenFractionTable(rows, history=fractions.history)
    return fractions, view


def frozen_fraction(source: Experiment | CountsTable) -> FrozenFractionTable:
    """Add frozen fraction to each original observation without selecting temperatures.

    Counts, measured temperatures, observation identities, times and row order
    remain unchanged. Repeated temperatures and warming observations remain
    available. Dilution and droplet-volume metadata are unnecessary for this step.
    """
    counts = source.counts if isinstance(source, Experiment) else source
    if not isinstance(counts, CountsTable):
        raise TypeError("source must be an Experiment or CountsTable")
    frame = counts.to_dataframe()
    if frame.empty:
        raise ValueError("Cannot calculate frozen fractions from an empty counts table")
    return FrozenFractionTable(
        frame,
        history=counts.history + [{
            "operation": "frozen_fraction",
            "temperature_source": "original_observations",
            "observation_selection": "none",
        }],
    )


def cumulative_spectrum(
    fractions: FrozenFractionTable,
    *,
    experiment: Experiment,
    temperature_ranges_C=None,
    z: float = 1.96,
    method: str = "mle",
    water_blank_correction: bool = True,
) -> CumulativeSpectrumTable:
    """Estimate individual spectra using the same method as the named-curve workflow.

    MLE fits each physical input's full freezing trajectory and its raw blanks.
    Average retains separate temperature estimates. All original observation
    rows remain in this table; fitted values are evaluated at their temperatures.
    """
    from urllib.parse import quote

    from .methods import validate_combination_method
    from .water_blank import sample_rows
    from .workflows import estimate_concentration

    method = validate_combination_method(method)
    if method == "average":
        return _pointwise_cumulative_spectrum(
            fractions, experiment=experiment, temperature_ranges_C=temperature_ranges_C,
            z=z, water_blank_correction=water_blank_correction,
        )
    fractions, view = prepare_fraction_analysis(
        fractions, experiment, water_blank_correction=water_blank_correction,
    )
    frame = fractions.to_dataframe()
    source = sample_rows(frame, view)
    keys = ["measurement_id", "run_id", "cycle_id"]
    groups = list(source.groupby(keys, sort=False))
    names = {identity: "/".join(quote(str(part), safe="") for part in identity)
             for identity, _ in groups}
    choices = {names[identity]: {"inputs": [identity[0]], "cycle": identity[2]}
               for identity, _ in groups}
    estimated = estimate_concentration(
        fractions, experiment=experiment, temperature_ranges_C=temperature_ranges_C,
        curves=choices, z=z, method=method, water_blank_correction=water_blank_correction,
    )
    records = []
    originals = frame.set_index([*keys, "observation_id"])
    for identity, rows in groups:
        fit_rows = (
            estimated.select(curve_id=names[identity]).to_dataframe().set_index("temperature_C")
        )
        ordered = rows.sort_values("time_s", kind="stable") if "time_s" in rows else rows
        for index, row in enumerate(ordered.to_dict("records")):
            values = fit_rows.loc[row["temperature_C"]].to_dict()
            values.pop("curve_id")
            row.update(values, point_id=f"point:{index}", point_order=index)
            row["selection_status"] = (
                "selected" if row["contributor_count"] else "outside_temperature_range"
            )
            row["observed_temperature_C"] = row["temperature_C"]
            if view.water_blank_map:
                blanks = [item for item in json.loads(row["source_observations"])
                          if item["role"] == "blank"]
                row["water_blank_observations"] = json.dumps(blanks)
                observed = [originals.loc[tuple(item[key] for key in (*keys, "observation_id"))]
                            for item in blanks]
                row["blank_n_frozen"] = (
                    sum(item.n_frozen for item in observed) if observed else np.nan
                )
                row["blank_n_total"] = (
                    sum(item.n_total for item in observed) if observed else np.nan
                )
            records.append(row)
    settings = {**estimated.history[-1], "operation": "cumulative_spectrum"}
    return CumulativeSpectrumTable(pd.DataFrame(records), history=fractions.history + [settings])


def _pointwise_cumulative_spectrum(
    fractions: FrozenFractionTable,
    *,
    experiment: Experiment,
    temperature_ranges_C=None,
    z: float = 1.96,
    water_blank_correction: bool = True,
) -> CumulativeSpectrumTable:
    """Estimate each measurement at its original observed temperatures.

    Original counts, times and observation identities remain in the output.
    Ranges are inclusive and apply before alignment. Outside-range rows have
    no concentration estimate and do not require a matching water blank.
    Blanks use matching acquisition rows when possible; otherwise they use the
    latest observed state at or warmer than the target, within observed support.
    """
    from .water_blank import estimate_point, sample_rows

    if not np.isfinite(z) or z <= 0:
        raise ValueError("z must be finite and positive")
    fractions, experiment = prepare_fraction_analysis(
        fractions, experiment, water_blank_correction=water_blank_correction,
    )
    frame = fractions.to_dataframe()
    source = sample_rows(frame, experiment)
    blank_ids = {key for ids in experiment.water_blank_map.values() for key in ids}
    ranges = validate_temperature_ranges(
        temperature_ranges_C, measurement_ids=set(experiment.measurements) - blank_ids
    )
    if source.empty:
        raise ValueError("No sample observations are available for concentration calculation")
    records = []
    cache: dict[tuple, tuple[float, float, float, bool]] = {}
    drop = z**2 / 2

    def count_state(rows: pd.DataFrame) -> tuple:
        return tuple(sorted(
            (str(measurement), str(run), str(cycle), float(frozen), float(total))
            for measurement, run, cycle, frozen, total in zip(
                rows.measurement_id, rows.run_id, rows.cycle_id,
                rows.n_frozen.to_numpy(dtype=float), rows.n_total.to_numpy(dtype=float),
            )
        ))

    for (measurement, run, cycle), rows in source.groupby(
        ["measurement_id", "run_id", "cycle_id"], sort=False
    ):
        measurement = str(measurement)
        metadata = experiment.measurements[measurement]
        ordered = rows.sort_values("time_s", kind="stable") if "time_s" in rows else rows
        ordered = ordered.reset_index(drop=True)
        points = align_observations(
            frame,
            [{"measurement_id": measurement, "run_id": str(run), "cycle_id": str(cycle)}],
            water_blank_map=experiment.water_blank_map,
            temperature_ranges_C=ranges,
        )
        for point in points:
            row = ordered.iloc[point.point_order].to_dict()
            eligible = not point.samples.empty
            row.update(
                point_id=point.point_id,
                point_order=point.point_order,
                observed_temperature_C=float(row["temperature_C"]),
                alignment=point.alignment,
                concentration=np.nan,
                lower_error=np.nan,
                upper_error=np.nan,
                unit="INP_per_mL_suspension",
                basis="suspension",
                qc_flag=1,
                dilution_fold=metadata.dilution,
                selection_status="selected" if eligible else "outside_temperature_range",
                uncertainty_method="joint_sample_water_blank_profile_likelihood"
                if experiment.water_blank_map else "binomial_Poisson_profile_likelihood",
                correction_state="water_blank_corrected"
                if experiment.water_blank_map else "uncorrected",
            )
            if experiment.water_blank_map:
                row.update(
                    water_blank_ids=json.dumps(sorted(experiment.water_blank_map[measurement])),
                    water_blank_observations=json.dumps([
                        {
                            "measurement_id": str(blank["measurement_id"]),
                            "run_id": str(blank["run_id"]),
                            "cycle_id": str(blank["cycle_id"]),
                            "observation_id": str(blank["observation_id"]),
                            "observed_temperature_C": float(blank["temperature_C"]),
                        }
                        for blank in point.blanks.to_dict("records")
                    ]),
                    blank_n_frozen=float(point.blanks.n_frozen.sum()) if eligible else np.nan,
                    blank_n_total=float(point.blanks.n_total.sum()) if eligible else np.nan,
                )
            if eligible:
                key = ("mle", float(drop), count_state(point.samples), count_state(point.blanks))
                if key not in cache:
                    cache[key] = estimate_point(
                        point.samples, point.blanks, experiment,
                        confidence_drop=drop, method="mle",
                    )
                fit = cache[key]
                row.update(
                    concentration=fit[0], lower_error=fit[1], upper_error=fit[2],
                    qc_flag=0 if fit[3] else 1,
                )
            row["at_zero_boundary"] = row["concentration"] == 0
            records.append(row)
    return CumulativeSpectrumTable(
        pd.DataFrame.from_records(records),
        history=fractions.history + [{
            "operation": "cumulative_spectrum",
            "estimation_method": "average",
            "z": float(z),
            "temperature_source": "original_observations",
            "alignment_when_needed": "latest",
            "temperature_ranges_C": ranges,
            "range_boundaries": "inclusive",
            "water_blank_correction": water_blank_correction,
            "water_blank_model": "volume_scaled",
            "water_blank_correction_applied": bool(experiment.water_blank_map),
            "water_blank_map": dict(experiment.water_blank_map),
            "uncertainty_assumption": (
                "pointwise count likelihood; sample and blank observations retain "
                "their own totals and volumes; repeated observation states do not "
                "add independent droplets"
            ),
        }],
    )


def differential_spectrum(
    fractions: FrozenFractionTable,
    *,
    experiment: Experiment,
    temperature_ranges_C=None,
    method: str = "mle",
    z: float = 1.96,
    water_blank_correction: bool = True,
) -> DifferentialSpectrumTable:
    """Calculate adjacent concentration changes in observation order.

    Strictly cooling transitions use their measured temperature difference.
    Missing endpoints retain a flagged missing interval. Repeated-temperature
    and warming transitions have no cooling-interval value; their source IDs
    and temperatures are recorded in history. No interval bridges a skipped
    transition or an excluded observation. Cycles remain separate.
    """
    cumulative = cumulative_spectrum(
        fractions,
        experiment=experiment,
        temperature_ranges_C=temperature_ranges_C,
        method=method,
        z=z,
        water_blank_correction=water_blank_correction,
    )
    records, omitted = [], []
    identity_columns = ["run_id", "sample_id", "cycle_id", "measurement_id"]
    for _, group in cumulative.to_dataframe().groupby(identity_columns, sort=False):
        group = group.sort_values("point_order", kind="stable").reset_index(drop=True)
        for position in range(len(group) - 1):
            previous, following = group.iloc[position], group.iloc[position + 1]
            start, end = float(previous.temperature_C), float(following.temperature_C)
            identity = {name: str(previous[name]) for name in identity_columns}
            if end >= start:
                omitted.append({
                    **identity,
                    "from_observation_id": str(previous.observation_id),
                    "to_observation_id": str(following.observation_id),
                    "from_temperature_C": start,
                    "to_temperature_C": end,
                    "reason": "repeated_temperature" if end == start else "warming",
                })
                continue
            endpoints = float(previous.concentration), float(following.concentration)
            value = (endpoints[1] - endpoints[0]) / (start - end) if np.isfinite(
                endpoints
            ).all() else np.nan
            records.append({
                **identity,
                "observation_id": str(previous.observation_id),
                "next_observation_id": str(following.observation_id),
                "point_id": str(previous.point_id),
                "point_order": int(previous.point_order),
                "temperature_C": start,
                "temperature_bin_left_C": end,
                "temperature_bin_right_C": start,
                "concentration": value,
                "unit": "INP_per_mL_suspension_per_C",
                "basis": "suspension",
                "qc_flag": (0 if np.isfinite(value) else 1) | (2 if value < 0 else 0),
            })
    columns = [
        *identity_columns, "observation_id", "next_observation_id", "point_id", "point_order",
        "temperature_C", "temperature_bin_left_C", "temperature_bin_right_C",
        "concentration", "unit", "basis", "qc_flag",
    ]
    return DifferentialSpectrumTable(
        pd.DataFrame.from_records(records, columns=columns),
        history=cumulative.history + [{
            "operation": "differential_spectrum",
            "observation_order": "time_s_then_input_order",
            "omitted_transitions": omitted,
        }],
    )
