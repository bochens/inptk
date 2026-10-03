"""Versioned, portable Experiment/AnalysisResult files; no pickle or executable data."""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .experiment import AnalysisResult, CurveResult, Experiment, MeasurementMetadata, SampleMetadata
from .tables import (
    CountsTable,
    CumulativeSpectrumTable,
    CurveSpectrumTable,
    DifferentialSpectrumTable,
    FrozenFractionTable,
)

FORMAT_VERSION = 4
TABLE_TYPES = {
    cls.__name__: cls
    for cls in (
        CountsTable,
        FrozenFractionTable,
        CumulativeSpectrumTable,
        CurveSpectrumTable,
        DifferentialSpectrumTable,
    )
}


def _encode(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {"$nonfinite": str(value)}
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]
    return value


def _decode(value):
    if isinstance(value, dict):
        if set(value) == {"$nonfinite"}:
            return float(value["$nonfinite"])
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


def _table_payload(table):
    return {
        "type": type(table).__name__,
        "columns": list(table.columns),
        "dtypes": {name: str(dtype) for name, dtype in table.to_dataframe().dtypes.items()},
        "rows": table.to_dataframe().to_dict("records"),
        "history": table.history,
    }


def _table_from_payload(payload):
    cls = TABLE_TYPES[payload["type"]]
    return cls(
        pd.DataFrame(payload["rows"], columns=payload["columns"]).astype(payload["dtypes"]),
        history=payload["history"],
    )


def _experiment_payload(experiment):
    return {
        "counts": _table_payload(experiment.counts),
        "samples": {key: asdict(value) for key, value in experiment.samples.items()},
        "measurements": {key: asdict(value) for key, value in experiment.measurements.items()},
        "source": experiment.source,
        "water_blank_map": experiment.water_blank_map,
    }


def _experiment_from_payload(payload):
    return Experiment(
        counts=_table_from_payload(payload["counts"]),
        samples={key: SampleMetadata(**value) for key, value in payload["samples"].items()},
        measurements={
            key: MeasurementMetadata(**value) for key, value in payload["measurements"].items()
        },
        source=payload["source"],
        water_blank_map=payload["water_blank_map"],
    )


def save(value: Experiment | AnalysisResult, path: str | Path) -> None:
    """Save a complete analysis folder. Existing destinations are never overwritten."""
    if isinstance(value, Experiment):
        payload = {"kind": "experiment", "experiment": _experiment_payload(value)}
    elif isinstance(value, AnalysisResult):
        payload = {
            "kind": "analysis",
            "experiment": _experiment_payload(value.experiment),
            "frozen_fraction": _table_payload(value.frozen_fraction),
            "curves": {
                name: {
                    "curve_id": curve.curve_id,
                    "sources": curve.sources,
                    "tables": {
                        quantity: _table_payload(table)
                        for quantity in ("cumulative", "excluded", "differential")
                        if (table := getattr(curve, quantity)) is not None
                    },
                }
                for name, curve in value.curves.items()
            },
            "settings": value.settings,
            "history": value.history,
            "warnings": value.warnings,
        }
    else:
        raise TypeError("save expects an Experiment or AnalysisResult")
    from . import __version__

    payload.update(format="inptk", format_version=FORMAT_VERSION, toolkit_version=__version__)
    text = json.dumps(_encode(payload), indent=2, allow_nan=False)
    target = Path(path)
    target.mkdir(parents=True, exist_ok=False)
    (target / "analysis.json").write_text(text + "\n", encoding="utf-8")


def load(path: str | Path) -> Experiment | AnalysisResult:
    payload = _decode(json.loads((Path(path) / "analysis.json").read_text(encoding="utf-8")))
    if payload.get("format") != "inptk" or payload.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported INP-toolkit file format or version")
    experiment = _experiment_from_payload(payload["experiment"])
    if payload["kind"] == "experiment":
        return experiment
    if payload["kind"] != "analysis":
        raise ValueError(f"Unknown saved object kind {payload['kind']!r}")
    return AnalysisResult(
        experiment=experiment,
        frozen_fraction=_table_from_payload(payload["frozen_fraction"]),
        curves={
            name: CurveResult(
                curve_id=item["curve_id"],
                sources=item["sources"],
                **{key: _table_from_payload(table) for key, table in item["tables"].items()},
            )
            for name, item in payload["curves"].items()
        },
        settings=payload["settings"],
        history=payload["history"],
        warnings=payload["warnings"],
    )
