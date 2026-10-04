"""Check a bundled CLI against Python using real Average and MLE calculations."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

import inptk


def check(executable):
    counts, metadata = [], []
    for name, total, frozen in (("A", 32, [1, 5, 12]), ("Water", 16, [0, 1, 2])):
        metadata.append(
            {"measurement_id": name, "sample_id": name, "dilution": 1, "droplet_volume_uL": 50}
        )
        counts.extend(
            {
                "measurement_id": name,
                "cycle_id": "1",
                "time_s": i,
                "temperature_C": -5 - i,
                "n_total": total,
                "n_frozen": count,
            }
            for i, count in enumerate(frozen)
        )
    blanks = {"A": ["Water"]}
    experiment = inptk.read_counts(counts, metadata=metadata, water_blank_map=blanks)
    requests = [
        {
            "id": 0,
            "import": {
                "out": "@input",
                "counts": counts,
                "metadata": metadata,
                "water_blank_map": blanks,
            },
        }
    ]
    expected = {}
    for method in ("average", "mle"):
        expected[method] = inptk.analyze_concentration(experiment, method=method).to_dataframe()
        requests.extend(
            [
                {
                    "id": method,
                    "args": [
                        "analyze",
                        "@input",
                        "--format",
                        "saved",
                        "--method",
                        method,
                        "--out",
                        f"@{method}",
                    ],
                },
                {
                    "id": method + "-table",
                    "args": ["table", f"@{method}", "--table", "cumulative", "--no-history"],
                },
            ]
        )
    # A clean working directory and environment prevent imports from this checkout.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "CONDA"))}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with tempfile.TemporaryDirectory(prefix="inptk-executable-check-") as folder:
        run = subprocess.run(
            [str(Path(executable).resolve()), "serve"],
            input="\n".join(map(json.dumps, requests)) + "\n",
            text=True,
            capture_output=True,
            check=True,
            cwd=folder,
            env=env,
            timeout=120,
        )
    replies = [json.loads(line) for line in run.stdout.splitlines()]
    if len(replies) != len(requests) or any(r["status"] != "ok" for r in replies):
        raise RuntimeError(f"Executable check failed: {run.stdout}\n{run.stderr}")
    for reply in replies:
        if str(reply["id"]).endswith("-table"):
            method = reply["id"].removesuffix("-table")
            for column in ("temperature_C", "concentration", "lower_error", "upper_error"):
                np.testing.assert_allclose(
                    [row[column] for row in reply["table"]["rows"]],
                    expected[method][column],
                    rtol=1e-10,
                    atol=1e-12,
                )
    print("Executable passed JSON import, Average, joint MLE and uncertainty comparisons.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    check(parser.parse_args().executable)
