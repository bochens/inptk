import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import inptk


@pytest.mark.parametrize("sample_id", ["007", "NA"])
def test_cli_matches_python_and_preserves_cycle_ids(tmp_path, sample_id):
    counts = pd.DataFrame(
        {
            "measurement_id": ["001"] * 6,
            "cycle_id": ["01"] * 3 + ["02"] * 3,
            "temperature_C": [-5, -6, -7] * 2,
            "n_total": [20] * 6,
            "n_frozen": [0, 3, 8, 0, 5, 9],
        }
    )
    metadata = pd.DataFrame(
        [{"measurement_id": "001", "sample_id": sample_id, "dilution": 1, "droplet_volume_uL": 50}]
    )
    counts.to_csv(tmp_path / "counts.csv", index=False)
    metadata.to_csv(tmp_path / "metadata.csv", index=False)
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "inptk",
            "analyze",
            str(tmp_path / "counts.csv"),
            "--metadata",
            str(tmp_path / "metadata.csv"),
            "--out",
            str(tmp_path / "result"),
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    restored = inptk.load(tmp_path / "result")
    expected = inptk.analyze_concentration(inptk.read_counts(counts, metadata=metadata))
    pd.testing.assert_frame_equal(restored.final.to_dataframe(), expected.final.to_dataframe())
    assert set(restored.final.to_dataframe().sample_id) == {sample_id}
    assert set(restored.final.to_dataframe().cycle_id) == {"01", "02"}


def test_no_cycle_policy_or_compatibility_package_in_source():
    from inptk.cli import build_parser

    parser = build_parser()
    assert "cycle-policy" not in parser.format_help()
    assert not (Path(__file__).resolve().parents[1] / "src" / "ufolaf" / "__init__.py").exists()
