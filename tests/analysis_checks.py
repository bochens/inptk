"""Collect named curves and check their independent step-by-step calculations."""

import pandas as pd

import inptk


def retained(result):
    return inptk.CurveSpectrumTable(
        result.to_dataframe(),
        history=next(iter(result.curves.values())).cumulative.history,
    )


def all_points(result):
    frames = []
    for curve in result.curves.values():
        pieces = [
            table.to_dataframe() for table in (curve.cumulative, curve.excluded) if len(table)
        ]
        frame = pd.concat(pieces, ignore_index=True) if pieces else curve.cumulative.to_dataframe()
        frames.append(frame.sort_values("point_order", kind="stable"))
    return inptk.CurveSpectrumTable(
        pd.concat(frames, ignore_index=True),
        history=retained(result).history,
    )


def sampled(result):
    if all(curve.resampled is None for curve in result.curves.values()):
        return None
    return inptk.CurveSpectrumTable(
        result.to_dataframe(table="resampled"),
        history=next(iter(result.curves.values())).resampled.history,
    )


def selected_fractions(result):
    keys = ("measurement_id", "run_id", "cycle_id")
    members = {
        tuple(source[key] for key in keys) for c in result.curves.values() for source in c.sources
    }
    blanks = (
        {
            (blank, run, cycle)
            for measurement, run, cycle in members
            for blank in result.experiment.water_blank_map.get(measurement, [])
        }
        if result.settings["water_blank_correction"]
        else set()
    )
    frame = result.frozen_fraction.to_dataframe()
    use = [
        tuple(row) in members | blanks
        for row in frame[list(keys)].itertuples(index=False, name=None)
    ]
    return inptk.FrozenFractionTable(
        frame.loc[use],
        history=result.frozen_fraction.history
        + [
            {
                "operation": "select_curve_inputs",
                "members": [dict(zip(keys, member, strict=True)) for member in sorted(members)],
                "water_blank_context": [
                    dict(zip(keys, blank, strict=True)) for blank in sorted(blanks)
                ],
            }
        ],
    )


def input_spectra(result):
    return inptk.cumulative_spectrum(
        selected_fractions(result),
        experiment=result.experiment,
        temperature_ranges_C=result.settings["temperature_ranges_C"],
        z=result.settings["z"],
        water_blank_correction=result.settings["water_blank_correction"],
    )


def fit_estimates(result):
    return inptk.estimate_concentration(
        result.frozen_fraction,
        experiment=result.experiment,
        curves=result.settings["curves"],
        method=result.settings["estimation_method"],
        z=result.settings["z"],
        temperature_ranges_C=result.settings["temperature_ranges_C"],
        water_blank_correction=result.settings["water_blank_correction"],
    )


def intervals(result):
    tables = [curve.differential for curve in result.curves.values()]
    if all(table is None for table in tables):
        return None
    return inptk.DifferentialSpectrumTable(
        pd.concat([table.to_dataframe() for table in tables], ignore_index=True),
        history=tables[0].history,
    )


def quantity_for_check(result, name):
    functions = {
        "final": retained,
        "final_candidates": all_points,
        "combined": fit_estimates,
        "per_dilution": input_spectra,
        "resampled": sampled,
        "differential": intervals,
    }
    return result.frozen_fraction if name == "frozen_fraction" else functions[name](result)
