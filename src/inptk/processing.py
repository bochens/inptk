"""Individual processing steps, with identities and preceding history preserved."""

from __future__ import annotations

import json
from typing import Literal

import numpy as np
import pandas as pd

from .context import prepare_fraction_analysis
from .experiment import Experiment
from .settings import DEFAULTS
from .tables import (
    CountsTable,
    CumulativeSpectrumTable,
    DifferentialSpectrumTable,
    FrozenFractionTable,
)


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
        history=counts.history
        + [
            {
                "operation": "frozen_fraction",
                "temperature_source": "original_observations",
                "observation_selection": "none",
            }
        ],
    )


def cumulative_spectrum(
    fractions: FrozenFractionTable,
    *,
    experiment: Experiment,
    temperature_step_C: float | None = DEFAULTS.temperature_step_C,
    temperature_start_C: float | None = DEFAULTS.temperature_start_C,
    temperature_end_C: float | None = DEFAULTS.temperature_end_C,
    temperature_method: Literal["latest", "max", "window"] = DEFAULTS.temperature_method,
    temperature_window_C: float | None = DEFAULTS.temperature_window_C,
    z: float = DEFAULTS.z,
    method: Literal["mle", "average"] = DEFAULTS.method,
    fit_step_C: float | None = DEFAULTS.fit_step_C,
    water_blank_correction: bool = DEFAULTS.water_blank_correction,
) -> CumulativeSpectrumTable:
    """Estimate individual spectra using the same method as the named-curve workflow.

    MLE fits each physical input's full freezing trajectory and its raw blanks.
    Average retains separate temperature estimates. All original observation
    rows remain in this table by default; fitted values use their temperatures.
    With temperature_step_C, return selected grid states with original source
    identities and observed temperatures recorded separately. Input-exclusion
    ranges belong to the combined-curve calculation, not this individual step.
    """
    from .estimation import estimate_concentration
    from .water_blank import sample_rows

    fractions, view = prepare_fraction_analysis(
        fractions,
        experiment,
        water_blank_correction=water_blank_correction,
    )
    frame = fractions.to_dataframe()
    keys = ["measurement_id", "run_id", "cycle_id"]
    groups = list(sample_rows(frame, view).groupby(keys, sort=False))
    choices = {
        str(i): {"inputs": [identity[0]], "cycle": identity[2]}
        for i, (identity, _) in enumerate(groups)
    }
    estimated = estimate_concentration(
        fractions,
        experiment=experiment,
        curves=choices,
        method=method,
        z=z,
        fit_step_C=fit_step_C,
        temperature_step_C=temperature_step_C,
        temperature_start_C=temperature_start_C,
        temperature_end_C=temperature_end_C,
        temperature_method=temperature_method,
        temperature_window_C=temperature_window_C,
        water_blank_correction=water_blank_correction,
    )
    return _individual_spectra(estimated, fractions, view)


def _individual_spectra(estimated, fractions, experiment):
    """Attach original count identities to already estimated individual curves."""
    frame = fractions.to_dataframe()
    keys = ["measurement_id", "run_id", "cycle_id"]
    originals = frame.set_index([*keys, "observation_id"])
    settings = estimated.history[-1]
    gridded = settings["temperature_step_C"] is not None
    records = []
    for name, values_frame in estimated.to_dataframe().groupby("curve_id", sort=False):
        members = settings["curve_sources"][name]["members"]
        if len(members) != 1:
            raise ValueError("Individual spectra require one input per curve")
        identity = tuple(members[0][key] for key in keys)
        if not gridded and settings["estimation_method"] == "mle":
            # The model evaluates a temperature once. Preserve original repeated
            # temperatures and observation order in the individual count table.
            rows = frame.loc[(frame[keys] == identity).all(axis=1)]
            rows = rows.sort_values("time_s", kind="stable") if "time_s" in rows else rows
            fits = values_frame.drop_duplicates("temperature_C", keep="last").set_index(
                "temperature_C",
                drop=False,
            )
            pairs = [
                (row, fits.loc[row["temperature_C"]].to_dict()) for row in rows.to_dict("records")
            ]
        else:
            pairs = []
            for values in values_frame.to_dict("records"):
                sources = json.loads(values["source_observations"])
                samples = [item for item in sources if item["role"] == "sample"]
                row = dict(zip(keys, identity, strict=True))
                if samples:
                    item = samples[0]
                    row.update(originals.loc[tuple(item[k] for k in (*keys, "observation_id"))])
                    row["observation_id"] = item["observation_id"]
                    row["source_observation_id"] = item["observation_id"]
                    row["observed_temperature_C"] = item["observed_temperature_C"]
                elif not gridded:
                    # A rejected original row still retains its observation ID.
                    selected = frame.loc[(frame[keys] == identity).all(axis=1)]
                    if "time_s" in selected:
                        selected = selected.sort_values("time_s", kind="stable")
                    row.update(selected.iloc[int(values["point_order"])].to_dict())
                pairs.append((row, values))
        for index, (row, values) in enumerate(pairs):
            values.pop("curve_id", None)
            row.setdefault("observed_temperature_C", row.get("temperature_C", np.nan))
            row.update(values, point_id=f"point:{index}", point_order=index)
            if gridded:
                row["observation_id"] = f"grid:{index}"
            row["selection_status"] = (
                "selected" if row["contributor_count"] else "outside_temperature_range"
            )
            if experiment.water_blank_map:
                blanks = [
                    item
                    for item in json.loads(row["source_observations"])
                    if item["role"] == "blank"
                ]
                row["water_blank_observations"] = json.dumps(blanks)
                observed = [
                    originals.loc[tuple(item[k] for k in (*keys, "observation_id"))]
                    for item in blanks
                ]
                row["blank_n_frozen"] = (
                    sum(item.n_frozen for item in observed) if observed else np.nan
                )
                row["blank_n_total"] = (
                    sum(item.n_total for item in observed) if observed else np.nan
                )
            records.append(row)
    return CumulativeSpectrumTable(
        pd.DataFrame(records),
        history=estimated.history[:-1] + [{**settings, "operation": "cumulative_spectrum"}],
    )


def differential_spectrum(
    fractions: FrozenFractionTable,
    *,
    experiment: Experiment,
    temperature_step_C: float | None = DEFAULTS.temperature_step_C,
    temperature_start_C: float | None = DEFAULTS.temperature_start_C,
    temperature_end_C: float | None = DEFAULTS.temperature_end_C,
    temperature_method: Literal["latest", "max", "window"] = DEFAULTS.temperature_method,
    temperature_window_C: float | None = DEFAULTS.temperature_window_C,
    method: Literal["mle", "average"] = DEFAULTS.method,
    fit_step_C: float | None = DEFAULTS.fit_step_C,
    z: float = DEFAULTS.z,
    water_blank_correction: bool = DEFAULTS.water_blank_correction,
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
        temperature_step_C=temperature_step_C,
        temperature_start_C=temperature_start_C,
        temperature_end_C=temperature_end_C,
        temperature_method=temperature_method,
        temperature_window_C=temperature_window_C,
        method=method,
        fit_step_C=fit_step_C,
        z=z,
        water_blank_correction=water_blank_correction,
    )
    return differentiate_spectrum(cumulative)


def differentiate_spectrum(cumulative: CumulativeSpectrumTable) -> DifferentialSpectrumTable:
    """Calculate adjacent differences from an existing individual cumulative table.

    This step does no fitting. Cooling intervals use both existing endpoints;
    missing endpoints remain missing and warming/repeated temperatures are omitted.
    """
    if not isinstance(cumulative, CumulativeSpectrumTable):
        raise TypeError("cumulative must be a CumulativeSpectrumTable")
    if not cumulative.to_dataframe().basis.eq("suspension").all():
        raise ValueError("Differential output currently requires suspension basis")
    records, omitted = [], []
    identity_columns = ["run_id", "sample_id", "cycle_id", "measurement_id"]
    for _, group in cumulative.to_dataframe().groupby(identity_columns, sort=False):
        group = group.sort_values("point_order", kind="stable").reset_index(drop=True)
        for position in range(len(group) - 1):
            previous, following = group.iloc[position], group.iloc[position + 1]
            start, end = float(previous.temperature_C), float(following.temperature_C)
            identity = {name: str(previous[name]) for name in identity_columns}
            different_segment = (
                "segment_id" in group and previous.segment_id != following.segment_id
            )
            if different_segment or end >= start:
                omitted.append(
                    {
                        **identity,
                        "from_observation_id": str(previous.observation_id),
                        "to_observation_id": str(following.observation_id),
                        "from_temperature_C": start,
                        "to_temperature_C": end,
                        "reason": "excluded_interval"
                        if different_segment
                        else ("repeated_temperature" if end == start else "warming"),
                    }
                )
                continue
            endpoints = float(previous.concentration), float(following.concentration)
            value = (
                (endpoints[1] - endpoints[0]) / (start - end)
                if np.isfinite(endpoints).all()
                else np.nan
            )
            records.append(
                {
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
                }
            )
    columns = [
        *identity_columns,
        "observation_id",
        "next_observation_id",
        "point_id",
        "point_order",
        "temperature_C",
        "temperature_bin_left_C",
        "temperature_bin_right_C",
        "concentration",
        "unit",
        "basis",
        "qc_flag",
    ]
    return DifferentialSpectrumTable(
        pd.DataFrame.from_records(records, columns=columns),
        history=cumulative.history
        + [
            {
                "operation": "differential_spectrum",
                "observation_order": "time_s_then_input_order",
                "omitted_transitions": omitted,
            }
        ],
    )
