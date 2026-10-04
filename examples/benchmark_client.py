"""Measure the in-memory client path, including JSON transport; writes no files.

Run: python examples/benchmark_client.py --rows 6000 --repeat 3
Requires INP-toolkit in the Python environment. Uses synthetic data from
benchmark_processing.py. Compare its direct calculation timings separately.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from statistics import median
from time import perf_counter

from benchmark_processing import benchmark_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=6000)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.rows < 2 or args.repeat < 1:
        parser.error("Use at least two rows and one repeat")
    experiment = benchmark_experiment(args.rows)
    combined = {"Combined": {"inputs": ["A", "B"], "cycle": "0"}}
    curves = {**combined, **{n: {"inputs": [n], "cycle": "0"} for n in ("A", "B")}}
    payload = {
        "out": "@input", "counts": experiment.counts.to_dataframe().to_dict("list"),
        "metadata": [{**asdict(experiment.samples[m.sample_id]), **asdict(m)}
                     for m in experiment.measurements.values()],
        "water_blank_map": experiment.water_blank_map,
    }
    process = subprocess.Popen(
        [sys.executable, "-m", "inptk", "serve"], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True,
    )
    request_id = 0

    def request(body):
        nonlocal request_id
        request_id += 1
        start = perf_counter()
        process.stdin.write(json.dumps({"id": request_id, **body}, separators=(",", ":")) + "\n")
        process.stdin.flush()
        line = process.stdout.readline()
        if not line:
            raise RuntimeError("Toolkit process closed its output")
        reply = json.loads(line)
        duration = perf_counter() - start
        if reply["status"] != "ok":
            raise RuntimeError(reply)
        return duration, len(line.encode())

    def measure(label, body, repeats=1, release=None):
        times, sizes = [], []
        for _ in range(repeats):
            elapsed, size = request(body)
            times.append(elapsed)
            sizes.append(size)
            if release:
                request({"release": [release]})
        print(json.dumps({"action": label, "median_seconds": round(median(times), 4),
                          "reply_bytes": int(median(sizes))}), flush=True)

    try:
        measure("startup", {"args": ["capabilities"]})
        measure("upload_once", {"import": payload})
        suggestions = ["suggest-ranges", "@input", "--format", "saved", "--curves",
                       json.dumps(combined), "--temperature-step-C", "0.5"]
        measure("suggest_full", {"args": suggestions}, args.repeat)
        measure("suggest_summary", {"args": [*suggestions, "--summary"]}, args.repeat)
        for method in ("average", "mle"):
            operation = ["analyze", "@input", "--format", "saved", "--method", method,
                         "--curves", json.dumps(curves), "--temperature-step-C", "0.5",
                         "--out", "@result"]
            measure(method, {"args": operation}, args.repeat, release="@result")
        request({"args": operation})
        table = ["table", "@result", "--table", "cumulative", "--curve", "Combined"]
        measure("plot_full", {"args": table}, args.repeat)
        measure("plot_columns", {"args": [*table, "--no-history", "--columns",
                "temperature_C", "concentration", "lower_error", "upper_error"]}, args.repeat)
        request({"release": ["@input", "@result"]})
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        process.stdout.close()


if __name__ == "__main__":
    main()
