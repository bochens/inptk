"""Time calculations on synthetic sample and blank counts; no files are written.

Run: python examples/benchmark_processing.py --rows 6000 --repeat 3
Add --native to include Average with every original observation.
--wells changes the number of possible freezing events (default: 10).
Times include calculation and result assembly, not import, CLI or file I/O.
"""

import argparse
import json
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd

import inptk


def benchmark_experiment(rows=6000, wells=10):
    progress = np.linspace(0, 1, rows)
    counts = pd.concat([
        pd.DataFrame({
            "measurement_id": name, "cycle_id": "0", "time_s": np.arange(rows),
            "temperature_C": -5 - 20 * progress, "n_total": wells,
            "n_frozen": (fraction * wells * progress).astype(int),
        })
        for name, fraction in (("A", .9), ("B", .7), ("Water", .2))
    ], ignore_index=True)
    metadata = [
        {"measurement_id": name, "sample_id": "Water" if name == "Water" else "Sample",
         "dilution": dilution, "droplet_volume_uL": 50}
        for name, dilution in (("A", 1), ("B", 10), ("Water", 1))
    ]
    return inptk.read_counts(counts, metadata=metadata,
                            water_blank_map={"A": ["Water"], "B": ["Water"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--wells", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    if args.rows < 2 or args.wells < 1 or args.repeat < 1:
        parser.error("Use at least two rows, one well and one repeat")
    experiment = benchmark_experiment(args.rows, args.wells)
    combined = {"Combined": {"inputs": ["A", "B"], "cycle": "0"}}
    curves = {**combined, **{name: {"inputs": [name], "cycle": "0"} for name in ("A", "B")}}
    actions = {
        "suggest_ranges": lambda: inptk.suggest_temperature_ranges(
            experiment, curves=combined, temperature_step_C=.5),
        "average": lambda: inptk.analyze_concentration(
            experiment, curves=curves, method="average", temperature_step_C=.5),
        "mle": lambda: inptk.analyze_concentration(
            experiment, curves=curves, method="mle", temperature_step_C=.5),
    }
    if args.native:
        actions["native_average"] = lambda: inptk.analyze_concentration(
            experiment, curves=curves, method="average")
    for name, action in actions.items():
        durations = []
        for _ in range(args.repeat):
            start = perf_counter()
            action()
            durations.append(perf_counter() - start)
        print(json.dumps({"action": name, "rows_per_input": args.rows,
                          "wells_per_input": args.wells, "repeats": args.repeat,
                          "median_seconds": round(median(durations), 4)}), flush=True)


if __name__ == "__main__":
    main()
