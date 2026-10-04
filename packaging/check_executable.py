"""Check a bundled CLI against Python using real Average and MLE calculations."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

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
    if os.name == "nt":
        # Require the bundle to work without a developer Python/Conda on PATH.
        env["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
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
    check_files_and_session(executable, counts, metadata, blanks, env)


def check_files_and_session(executable, counts, metadata, blanks, env):
    """Exercise file transport and saved stages through the actual frozen process."""
    executable = str(Path(executable).resolve())
    with tempfile.TemporaryDirectory(prefix="inptk paths ") as directory:
        folder = Path(directory) / "échantillon 冰"
        folder.mkdir()

        def command(*arguments, success=True):
            result = subprocess.run(
                [executable, "--json", *map(str, arguments)],
                cwd=folder,
                env=env,
                text=True,
                encoding="utf-8",
                capture_output=True,
                timeout=120,
                check=False,
            )
            reply = json.loads(result.stdout)
            if (result.returncode == 0) != success or reply["status"] != (
                "ok" if success else "error"
            ):
                raise RuntimeError(f"Executable command failed: {reply}\n{result.stderr}")
            return reply

        capabilities = command("capabilities")
        assert capabilities["toolkit_version"] == inptk.__version__
        assert capabilities["protocol_version"] == 2
        assert capabilities["saved_format_version"] == 4
        native = folder / "raw counts.csv"
        physical = folder / "physical metadata.csv"
        pd.DataFrame(counts).to_csv(native, index=False)
        pd.DataFrame(metadata).to_csv(physical, index=False)
        common = ["--metadata", physical, "--water-blank-map", json.dumps(blanks)]
        fractions, estimated, converted, final = [
            folder / f"{name}.inptk" for name in ("fractions", "estimate", "convert", "final")
        ]
        command("fractions", native, *common, "--out", fractions)
        command(
            "estimate", fractions, "--format", "saved", "--method", "average", "--out", estimated
        )
        command("convert", estimated, "--output-basis", "suspension", "--out", converted)
        command("finalize", converted, "--out", final)
        expected = inptk.analyze_concentration(
            inptk.read_counts(counts, metadata=metadata, water_blank_map=blanks),
            method="average",
        )
        for column in ("temperature_C", "concentration", "lower_error", "upper_error"):
            np.testing.assert_allclose(
                inptk.load(final).tables["cumulative"].to_dataframe()[column],
                expected.to_dataframe()[column],
                rtol=1e-10,
                atol=1e-12,
            )
        exported = folder / "exported counts.csv"
        command("export-csv", final, "--table", "counts", "--out", exported)
        assert len(pd.read_csv(exported)) == len(counts)
        command("finalize", converted, "--out", final, success=False)

        csv = (
            "cycle,time_s,temperature_C,picture,A number total,A number frozen,"
            "Water number total,Water number frozen\n"
            "01,0,-5,img0,32,1,16,0\n01,1,-6,,32,5,16,1\n01,2,-7,img2,32,12,16,2\n"
        )
        records = [
            {
                "sample_id": name,
                "sample_name": name,
                "dilution": "1",
                "well_volume_uL": "50",
                "sample_type": "other",
            }
            for name in ("A", "Water")
        ]
        project = folder / "input project.icescopy"
        with ZipFile(project, "w") as archive:
            archive.writestr("freeze_count_timeseries.csv", csv)
            archive.writestr(
                "session.json",
                json.dumps(
                    {
                        "freeze_count_timeseries_summary": {
                            "sample_column_metadata": records,
                            "analysis_required": False,
                        },
                    }
                ),
            )
        project_result = folder / "project result.inptk"
        command(
            "analyze",
            project,
            "--format",
            "icescopy",
            "--water-blank-map",
            json.dumps(blanks),
            "--method",
            "mle",
            "--out",
            project_result,
        )
        project_expected = inptk.analyze_concentration(
            inptk.read_icescopy(project, water_blank_map=blanks),
            method="mle",
        )
        np.testing.assert_allclose(
            inptk.load(project_result).to_dataframe().concentration,
            project_expected.to_dataframe().concentration,
            rtol=1e-10,
        )
        icescopy_csv = folder / "icescopy counts.csv"
        icescopy_csv.write_text(
            "# sample_name,A,Water\n# dilution,1,1\n# well_volume_uL,50,50\n" + csv,
            encoding="utf-8",
        )
        command("preview", icescopy_csv, "--format", "icescopy")

        # Interleave failures and success in one long-lived session. Windows clients
        # use pipes and CREATE_NO_WINDOW, keeping the console-enabled standard streams.
        with subprocess.Popen(
            [executable, "serve"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            cwd=folder,
            env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        ) as process:
            requests = [
                {
                    "id": "upload",
                    "import": {
                        "out": "@raw",
                        "counts": counts,
                        "metadata": metadata,
                        "water_blank_map": blanks,
                    },
                },
                {
                    "id": "bad",
                    "args": ["analyze", "@missing", "--format", "saved", "--out", "@bad"],
                },
                {
                    "id": "good",
                    "args": [
                        "analyze",
                        "@raw",
                        "--format",
                        "saved",
                        "--method",
                        "average",
                        "--curves",
                        json.dumps({"Courbe 冰": {"inputs": ["A"], "cycle": "1"}}),
                        "--out",
                        "@good",
                    ],
                },
                {
                    "id": "table",
                    "args": [
                        "table",
                        "@good",
                        "--curve",
                        "Courbe 冰",
                        "--table",
                        "cumulative",
                        "--no-history",
                    ],
                },
                {
                    "id": "save",
                    "args": ["save", "@good", "--out", str(folder / "saved session.inptk")],
                },
                {"id": "release", "release": ["@raw", "@good"]},
                {"id": "gone", "args": ["table", "@good"]},
                {"id": "alive", "args": ["capabilities"]},
            ]
            output, errors = process.communicate(
                "\n".join(map(json.dumps, requests)) + "\n",
                timeout=120,
            )
            replies = [json.loads(line) for line in output.splitlines()]
            assert process.returncode == 0, errors
            assert [reply["id"] for reply in replies] == [r["id"] for r in requests]
            assert [reply["status"] for reply in replies] == [
                "ok",
                "error",
                "ok",
                "ok",
                "ok",
                "ok",
                "error",
                "ok",
            ], replies
            assert "Courbe 冰" in inptk.load(folder / "saved session.inptk").curves
    print("Executable passed saved stages, CSV/project input, Unicode paths and session recovery.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    check(parser.parse_args().executable)
