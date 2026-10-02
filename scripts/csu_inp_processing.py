#!/usr/bin/env python3
"""Export one saved, final INP-toolkit spectrum in CSU INPs_L CSV format.

Run ``inptk analyze --output-basis sampled_air`` first. This script does not
calculate, combine, normalize or select concentration points. CSU lower_CI and
upper_CI retain the saved error widths, not interval endpoints.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

from inptk import AnalysisResult, load  # noqa: E402

HEADER_ORDER = (
    "site",
    "start_time",
    "end_time",
    "filter_color",
    "sample_type",
    "vol_air_filt",
    "proportion_filter_used",
    "vol_susp",
    "treatment",
    "notes",
    "user",
    "IS",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export one saved INP-toolkit final sampled-air spectrum as a CSU INPs_L CSV."
    )
    parser.add_argument("input", help="Saved AnalysisResult folder produced by inptk analyze")
    parser.add_argument(
        "--out", required=True, help="New CSU CSV path; existing files are never replaced"
    )
    for name in ("sample", "run", "cycle"):
        parser.add_argument(
            f"--{name}", help=f"Exact saved {name}_id; required when selection is ambiguous"
        )
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Supply a CSU header field; recorded normalization metadata cannot be changed",
    )
    parser.add_argument(
        "--allow-missing-header",
        action="store_true",
        help="Leave unavailable descriptive CSU header fields blank",
    )
    return parser


def _select_final(result: AnalysisResult, args: argparse.Namespace) -> pd.DataFrame:
    frame = result.final.to_dataframe()
    for name in ("sample", "run", "cycle"):
        requested = getattr(args, name)
        if requested is not None:
            frame = frame.loc[frame[f"{name}_id"].eq(requested)]
    if frame.empty:
        raise ValueError("No saved final rows match the requested sample/run/cycle selection")
    groups = frame[["sample_id", "run_id", "cycle_id"]].drop_duplicates()
    if len(groups) != 1:
        raise ValueError(
            "Multiple saved sample/run/cycle groups remain. Select exact IDs with "
            f"--sample, --run and --cycle. Available groups: {groups.to_dict('records')}"
        )
    if not frame.unit.eq("INP_per_L_air").all() or not frame.basis.eq("sampled_air").all():
        raise ValueError(
            "CSU INPs_L export requires a saved final sampled_air spectrum in INP_per_L_air; "
            "run inptk analyze --output-basis sampled_air first"
        )
    if not np.isfinite(frame.concentration.to_numpy(dtype=float)).all():
        raise ValueError("Saved final concentrations must be finite for CSU export")
    if not {"lower_error", "upper_error"}.issubset(frame.columns):
        raise ValueError("Saved final spectrum must include lower_error and upper_error widths")
    return frame.reset_index(drop=True)


def _header_overrides(values: Sequence[str]) -> dict[str, str]:
    overrides = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--header expects KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        key, raw = key.strip(), raw.strip()
        if key not in HEADER_ORDER:
            raise ValueError(f"Unknown CSU header key {key!r}")
        if "\n" in raw or "\r" in raw:
            raise ValueError(f"CSU header {key!r} must be a single line")
        overrides[key] = raw
    return overrides


def _csu_header(
    result: AnalysisResult, frame: pd.DataFrame, *, overrides: dict[str, str], allow_missing: bool
) -> dict[str, str]:
    sample = result.experiment.samples[str(frame.sample_id.iloc[0])]
    recorded = {
        "sample_type": sample.sample_type,
        "vol_air_filt": sample.air_volume_L,
        "proportion_filter_used": sample.filter_fraction_used,
        "vol_susp": sample.suspension_volume_mL,
    }
    for name, value in recorded.items():
        if name in overrides and value is not None:
            requested = overrides[name]
            if isinstance(value, str):
                matches = requested == value
            else:
                try:
                    matches = float(requested) == value
                except ValueError:
                    matches = False
            if not matches:
                raise ValueError(
                    f"CSU header {name!r} conflicts with saved sample metadata; "
                    "header fields cannot change the analyzed sample or normalization"
                )
    header = {
        key: overrides.get(key, "" if recorded.get(key) is None else str(recorded[key]))
        for key in HEADER_ORDER
    }
    missing = [key for key, value in header.items() if not value]
    if missing and not allow_missing:
        raise ValueError(
            "Missing CSU header fields: "
            + ", ".join(missing)
            + ". Supply --header KEY=VALUE or use --allow-missing-header."
        )
    return header


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = load(args.input)
    if not isinstance(result, AnalysisResult):
        raise TypeError("Input must be a saved AnalysisResult; run inptk analyze first")
    frame = _select_final(result, args)
    header = _csu_header(
        result,
        frame,
        overrides=_header_overrides(args.header),
        allow_missing=args.allow_missing_header,
    )
    data = pd.DataFrame(
        {
            "degC": frame.temperature_C,
            "dilution": frame.dilution_fold if "dilution_fold" in frame else np.nan,
            "INPS_L": frame.concentration,
            "lower_CI": frame.lower_error,
            "upper_CI": frame.upper_error,
        }
    )
    payload = "".join(f"{key} = {header[key]}\n" for key in HEADER_ORDER) + "\n"
    payload += data.to_csv(index=False)
    output = Path(args.out)
    source = Path(args.input).resolve()
    if output.resolve() == source or source in output.resolve().parents:
        raise ValueError("Choose a CSU CSV destination outside the saved analysis folder")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as handle:
        handle.write(payload)
    print(
        f"Exported sample {frame.sample_id.iloc[0]!r}, run {frame.run_id.iloc[0]!r}, "
        f"cycle {frame.cycle_id.iloc[0]!r} to {output}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, TypeError, ValueError) as error:
        print(f"CSU export failed: {error}", file=sys.stderr)
        raise SystemExit(2) from None
