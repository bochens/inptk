from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType


def _load_csu_script() -> ModuleType:
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "csu_inp_processing.py"
    spec = importlib.util.spec_from_file_location("csu_inp_processing", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load csu_inp_processing.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _args(
    *,
    include_sample: list[str] | None = None,
    exclude_sample: list[str] | None = None,
) -> argparse.Namespace:
    return argparse.Namespace(
        include_sample=include_sample or [],
        exclude_sample=exclude_sample or [],
    )


def test_file_qualified_include_sample_only_matches_that_input_file() -> None:
    csu = _load_csu_script()

    assert csu._sample_allowed(
        {"sample-a"},
        _args(include_sample=["run1.csv::sample-a"]),
        "run1.csv",
    )
    assert not csu._sample_allowed(
        {"sample-a"},
        _args(include_sample=["run1.csv::sample-a"]),
        "run2.csv",
    )


def test_unqualified_include_sample_still_matches_any_input_file() -> None:
    csu = _load_csu_script()

    assert csu._sample_allowed({"sample-a"}, _args(include_sample=["sample-a"]), "run1.csv")
    assert csu._sample_allowed({"sample-a"}, _args(include_sample=["sample-a"]), "run2.csv")


def test_file_qualified_exclude_sample_only_matches_that_input_file() -> None:
    csu = _load_csu_script()

    assert csu._sample_allowed(
        {"sample-a"},
        _args(exclude_sample=["run1.csv::sample-a"]),
        "run2.csv",
    )
    assert not csu._sample_allowed(
        {"sample-a"},
        _args(exclude_sample=["run1.csv::sample-a"]),
        "run1.csv",
    )


def test_csu_export_keeps_repeated_cycles_in_separate_files(tmp_path):
    import pandas as pd

    csu = _load_csu_script()
    counts = tmp_path / "counts.csv"
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "sample_id": ["A"] * 6,
            "cycle": [1, 1, 1, 2, 2, 2],
            "temperature_C": [-5, -6, -7] * 2,
            "n_total": [32] * 6,
            "n_frozen": [0, 2, 8, 0, 3, 9],
        }
    ).to_csv(counts, index=False)
    pd.DataFrame(
        [
            {
                "sample_id": "A",
                "sample_type": "air",
                "well_volume_uL": 50,
                "dilution": 1,
                "suspension_volume_mL": 5,
                "air_volume_L": 100,
                "filter_fraction_used": 1,
            }
        ]
    ).to_csv(metadata, index=False)
    assert (
        csu.main(
            [
                str(counts),
                "--metadata",
                str(metadata),
                "--out-dir",
                str(tmp_path / "out"),
                "--allow-missing-header",
            ]
        )
        == 0
    )
    assert (tmp_path / "out" / "cycle-1" / "A_INPs_L.csv").exists()
    assert (tmp_path / "out" / "cycle-2" / "A_INPs_L.csv").exists()


def test_csu_export_applies_both_final_selection_policies(tmp_path):
    import pandas as pd

    csu = _load_csu_script()
    counts = tmp_path / "counts.csv"
    metadata = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "sample_id": ["A"] * 4,
            "temperature_C": [-5, -6, -7, -8],
            "n_total": [32] * 4,
            "n_frozen": [0, 8, 4, 12],
        }
    ).to_csv(counts, index=False)
    pd.DataFrame(
        [
            {
                "sample_id": "A",
                "sample_type": "air",
                "well_volume_uL": 50,
                "dilution": 1,
                "suspension_volume_mL": 5,
                "air_volume_L": 100,
                "filter_fraction_used": 1,
            }
        ]
    ).to_csv(metadata, index=False)
    for policy, temperatures in (
        ("stop_at_decrease", [-5, -6]),
        ("skip_decreases", [-5, -6, -8]),
    ):
        target = tmp_path / f"{policy}.csv"
        assert (
            csu.main(
                [
                    str(counts),
                    "--metadata",
                    str(metadata),
                    "--out",
                    str(target),
                    "--allow-missing-header",
                    "--step-C",
                    "1",
                    "--decrease-policy",
                    policy,
                ]
            )
            == 0
        )
        lines = target.read_text().splitlines()
        header = next(i for i, line in enumerate(lines) if line.startswith("degC,"))
        exported = pd.read_csv(target, skiprows=header)
        assert exported.degC.tolist() == temperatures
        assert exported.INPS_L.is_monotonic_increasing
