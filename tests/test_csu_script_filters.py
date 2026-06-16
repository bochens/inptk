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
