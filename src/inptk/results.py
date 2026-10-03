"""Build named result views without changing estimates or source observations."""

from __future__ import annotations

import json
from dataclasses import asdict

from .experiment import CurveResult, Experiment
from .tables import CurveSpectrumTable, DifferentialSpectrumTable


def assemble_curves(
    retained: CurveSpectrumTable,
    candidates: CurveSpectrumTable,
    groups: dict,
    *,
    experiment: Experiment,
    resampled: CurveSpectrumTable | None = None,
    differential: DifferentialSpectrumTable | None = None,
) -> dict[str, CurveResult]:
    """Partition outputs by name; preserve original point order and error widths."""
    results = {}
    for name, group in groups.items():
        cumulative = retained.select(curve_id=name)
        selected = candidates.select(curve_id=name)
        frame = selected.to_dataframe()
        kept_ids = set(cumulative.to_dataframe().point_id)
        excluded = CurveSpectrumTable(
            frame.loc[~frame.point_id.isin(kept_ids)],
            history=selected.history,
        )
        sources = [
            {
                **asdict(experiment.measurements[member["measurement_id"]]),
                "cycle_id": member["cycle_id"],
                "water_blank_ids": list(
                    experiment.water_blank_map.get(member["measurement_id"], [])
                ),
            }
            for member in group["members"]
        ]
        intervals = None
        if differential is not None and len(sources) == 1:
            source = sources[0]
            intervals = differential.select(
                measurement_id=source["measurement_id"],
                cycle_id=source["cycle_id"],
            )
            data = intervals.to_dataframe()
            joint_fit = any(
                entry.get("estimation_method") == "mle" for entry in cumulative.history
            )
            selected_grid = any(
                entry.get("temperature_step_C") is not None for entry in cumulative.history
            )
            if joint_fit or selected_grid:
                # A fitted temperature can represent several original images.
                # Grid selection also has its own state IDs. Keep an interval
                # when both endpoints survived, not by original image IDs.
                temperatures = cumulative.to_dataframe().temperature_C
                keep = data.temperature_bin_left_C.isin(temperatures) & (
                    data.temperature_bin_right_C.isin(temperatures)
                )
            else:
                observed = {
                    item["observation_id"]
                    for value in cumulative.to_dataframe().source_observations
                    for item in json.loads(value)
                    if item["role"] == "sample"
                }
                keep = data.observation_id.isin(observed) & data.next_observation_id.isin(observed)
            data = data.loc[keep]
            intervals = DifferentialSpectrumTable(data, history=intervals.history)
        results[name] = CurveResult(
            curve_id=name,
            cumulative=cumulative,
            sources=sources,
            excluded=excluded,
            resampled=resampled.select(curve_id=name) if resampled is not None else None,
            differential=intervals,
        )
    return results
