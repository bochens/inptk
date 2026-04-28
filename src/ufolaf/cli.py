from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from . import __version__
from .adapters import metadata_frame, read_counts, read_metadata
from .blank_math import average_blank_spectra, subtract_filter_blank_spectrum
from .io import export_artifact_csv, read_artifact, write_artifact
from .pipeline import counts_to_spectrum
from .transforms import (
    counts_to_temperature_frozen_fraction,
    cumulative_spectrum_to_normalized_inp_spectrum,
    temperature_frozen_fraction_to_binomial_mle_cumulative_spectrum,
    temperature_frozen_fraction_to_cumulative_spectrum,
    temperature_frozen_fraction_to_stitched_cumulative_spectrum,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as exc:  # pragma: no cover - argparse integration path
        parser.exit(1, f"ufolaf: error: {exc}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ufolaf",
        description="UFOLAF table processing and CLI artifact tools.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    _add_read_counts_command(subcommands)
    _add_metadata_command(subcommands)
    _add_fraction_command(subcommands)
    _add_cumulative_command(subcommands)
    _add_stitch_command(subcommands)
    _add_mle_command(subcommands)
    _add_normalize_command(subcommands)
    _add_blank_average_command(subcommands)
    _add_blank_subtract_command(subcommands)
    _add_export_csv_command(subcommands)
    _add_pipeline_command(subcommands)
    return parser


def _add_read_counts_command(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("read-counts", help="Read counts into a UFOLAF artifact")
    parser.add_argument("input", help="Icescopy CSV or canonical count CSV")
    parser.add_argument("--out", required=True, help="Output UFOLAF artifact directory")
    parser.add_argument("--format", choices=("auto", "long", "wide"), default="auto")
    parser.add_argument("--columns", help="JSON object or JSON file mapping canonical columns")
    parser.add_argument("--metadata", help="Optional metadata CSV or JSON")
    parser.add_argument(
        "--cycle-policy",
        choices=("single", "pooled", "preserve"),
        default="single",
    )
    parser.add_argument("--cycle", help="Cycle to select when cycle-policy is single")
    _add_overwrite(parser)
    parser.set_defaults(func=_cmd_read_counts)


def _add_metadata_command(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("metadata", help="Extract Icescopy metadata")
    parser.add_argument("input", help="Icescopy CSV")
    parser.add_argument("--out", required=True, help="Output metadata CSV")
    parser.set_defaults(func=_cmd_metadata)


def _add_fraction_command(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("fraction", help="Reduce counts to frozen fractions")
    parser.add_argument("input", help="Input counts artifact")
    parser.add_argument("--out", required=True, help="Output fraction artifact")
    _add_fraction_options(parser)
    _add_overwrite(parser)
    parser.set_defaults(func=_cmd_fraction)


def _add_cumulative_command(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("cumulative", help="Convert fractions to per-mL K(T)")
    parser.add_argument("input", help="Input fraction artifact")
    parser.add_argument("--out", required=True, help="Output cumulative artifact")
    parser.add_argument("--z", type=float, default=1.96)
    _add_overwrite(parser)
    parser.set_defaults(func=_cmd_cumulative)


def _add_stitch_command(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("stitch", help="OLAF-style serial dilution stitching")
    parser.add_argument("input", help="Input fraction artifact")
    parser.add_argument("--out", required=True, help="Output stitched cumulative artifact")
    parser.add_argument("--sample-group-by", default="inferred")
    parser.add_argument("--enforce-monotone", action="store_true")
    parser.add_argument("--z", type=float, default=1.96)
    _add_overwrite(parser)
    parser.set_defaults(func=_cmd_stitch)


def _add_mle_command(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("mle", help="Binomial-Poisson MLE dilution combine")
    parser.add_argument("input", help="Input fraction artifact")
    parser.add_argument("--out", required=True, help="Output MLE cumulative artifact")
    parser.add_argument("--sample-group-by", default="inferred")
    parser.add_argument("--enforce-monotone", action="store_true")
    parser.add_argument("--confidence-drop", type=float, default=1.920729410347062)
    _add_overwrite(parser)
    parser.set_defaults(func=_cmd_mle)


def _add_normalize_command(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("normalize", help="Normalize cumulative spectra")
    parser.add_argument("input", help="Input cumulative artifact")
    parser.add_argument("--out", required=True, help="Output normalized artifact")
    _add_overwrite(parser)
    parser.set_defaults(func=_cmd_normalize)


def _add_blank_average_command(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("blank-average", help="Average blank spectrum artifacts")
    parser.add_argument("blank", nargs="+", help="Blank artifacts")
    parser.add_argument("--out", required=True, help="Output averaged blank artifact")
    parser.add_argument("--value-method", choices=("mean", "median"), default="mean")
    parser.add_argument("--include-nonpositive", action="store_true")
    _add_overwrite(parser)
    parser.set_defaults(func=_cmd_blank_average)


def _add_blank_subtract_command(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("blank-subtract", help="Subtract filter blank spectrum")
    parser.add_argument("sample", help="Sample spectrum artifact")
    parser.add_argument("blank", help="Blank spectrum artifact")
    parser.add_argument("--out", required=True, help="Output corrected artifact")
    parser.add_argument("--no-extrapolate-missing-cold", action="store_true")
    parser.add_argument("--no-qc", action="store_true")
    parser.add_argument("--threshold-percent", type=float, default=10.0)
    parser.add_argument("--error-signal", type=float, default=-9999.0)
    parser.add_argument("--clamp-zero", action="store_true")
    _add_overwrite(parser)
    parser.set_defaults(func=_cmd_blank_subtract)


def _add_export_csv_command(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("export-csv", help="Export artifact table data to CSV")
    parser.add_argument("input", help="Input UFOLAF artifact")
    parser.add_argument("--out", required=True, help="Output CSV")
    parser.set_defaults(func=_cmd_export_csv)


def _add_pipeline_command(subcommands: argparse._SubParsersAction) -> None:
    parser = subcommands.add_parser("pipeline", help="Run read -> fraction -> spectrum workflow")
    parser.add_argument("input", help="Icescopy CSV or canonical count CSV")
    parser.add_argument("--out", required=True, help="Output UFOLAF artifact directory")
    parser.add_argument("--format", choices=("auto", "long", "wide"), default="auto")
    parser.add_argument("--columns", help="JSON object or JSON file mapping canonical columns")
    parser.add_argument("--metadata", help="Optional metadata CSV or JSON")
    parser.add_argument(
        "--cycle-policy",
        choices=("single", "pooled", "preserve"),
        default="single",
    )
    parser.add_argument("--cycle", help="Cycle to select when cycle-policy is single")
    _add_fraction_options(parser)
    parser.add_argument("--combine", choices=("none", "stitch", "mle"), default="none")
    parser.add_argument("--sample-group-by", default="inferred")
    parser.add_argument("--enforce-monotone", action="store_true")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--z", type=float, default=1.96)
    _add_overwrite(parser)
    parser.set_defaults(func=_cmd_pipeline)


def _add_fraction_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--step-C", type=float, default=0.5)
    parser.add_argument("--method", choices=("max", "latest"), default="max")
    parser.add_argument("--temperature-tolerance-C", type=float, default=0.0)


def _add_overwrite(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--overwrite", action="store_true", help="Replace output if it exists")


def _cmd_read_counts(args: argparse.Namespace) -> None:
    counts = read_counts(
        _read_table_source(args.input),
        format=args.format,
        columns=_load_columns(args.columns),
        metadata=_load_metadata(args.metadata),
        cycle_policy=args.cycle_policy,
        cycle=args.cycle,
    )
    write_artifact(counts, args.out, overwrite=args.overwrite)


def _cmd_metadata(args: argparse.Namespace) -> None:
    _, metadata_by_sample_id = read_metadata(args.input)
    metadata_frame(metadata_by_sample_id).to_csv(args.out, index=False)


def _cmd_fraction(args: argparse.Namespace) -> None:
    fraction = counts_to_temperature_frozen_fraction(
        read_artifact(args.input),
        step_C=args.step_C,
        method=args.method,
        temperature_tolerance_C=args.temperature_tolerance_C,
    )
    write_artifact(fraction, args.out, overwrite=args.overwrite)


def _cmd_cumulative(args: argparse.Namespace) -> None:
    cumulative = temperature_frozen_fraction_to_cumulative_spectrum(
        read_artifact(args.input),
        z=args.z,
    )
    write_artifact(cumulative, args.out, overwrite=args.overwrite)


def _cmd_stitch(args: argparse.Namespace) -> None:
    stitched = temperature_frozen_fraction_to_stitched_cumulative_spectrum(
        read_artifact(args.input),
        sample_group_by=_sample_group_by(args.sample_group_by),
        enforce_monotone=args.enforce_monotone,
        z=args.z,
    )
    write_artifact(stitched, args.out, overwrite=args.overwrite)


def _cmd_mle(args: argparse.Namespace) -> None:
    mle = temperature_frozen_fraction_to_binomial_mle_cumulative_spectrum(
        read_artifact(args.input),
        sample_group_by=_sample_group_by(args.sample_group_by),
        enforce_monotone=args.enforce_monotone,
        confidence_drop=args.confidence_drop,
    )
    write_artifact(mle, args.out, overwrite=args.overwrite)


def _cmd_normalize(args: argparse.Namespace) -> None:
    normalized = cumulative_spectrum_to_normalized_inp_spectrum(read_artifact(args.input))
    write_artifact(normalized, args.out, overwrite=args.overwrite)


def _cmd_blank_average(args: argparse.Namespace) -> None:
    blanks = [read_artifact(path) for path in args.blank]
    average = average_blank_spectra(
        blanks,
        value_method=args.value_method,
        require_positive=not args.include_nonpositive,
    )
    write_artifact(average, args.out, overwrite=args.overwrite)


def _cmd_blank_subtract(args: argparse.Namespace) -> None:
    corrected = subtract_filter_blank_spectrum(
        read_artifact(args.sample),
        read_artifact(args.blank),
        extrapolate_missing_cold=not args.no_extrapolate_missing_cold,
        apply_qc=not args.no_qc,
        threshold_percent=args.threshold_percent,
        error_signal=args.error_signal,
        clamp_zero=args.clamp_zero,
    )
    write_artifact(corrected, args.out, overwrite=args.overwrite)


def _cmd_export_csv(args: argparse.Namespace) -> None:
    export_artifact_csv(read_artifact(args.input), args.out)


def _cmd_pipeline(args: argparse.Namespace) -> None:
    counts = read_counts(
        _read_table_source(args.input),
        format=args.format,
        columns=_load_columns(args.columns),
        metadata=_load_metadata(args.metadata),
        cycle_policy=args.cycle_policy,
        cycle=args.cycle,
    )
    result = counts_to_spectrum(
        counts,
        step_C=args.step_C,
        method=args.method,
        temperature_tolerance_C=args.temperature_tolerance_C,
        combine=args.combine,
        sample_group_by=_sample_group_by(args.sample_group_by),
        enforce_monotone=args.enforce_monotone,
        z=args.z,
        normalize=args.normalize,
    )
    write_artifact(result, args.out, overwrite=args.overwrite)


def _read_table_source(path: str) -> pd.DataFrame | str:
    input_path = Path(path)
    if input_path.suffix.lower() in {".csv", ".txt"}:
        return str(input_path)
    return pd.read_csv(input_path)


def _load_columns(value: str | None) -> dict[str, str] | None:
    if not value:
        return None
    path = Path(value)
    payload = path.read_text(encoding="utf-8") if path.exists() else value
    loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError("--columns must be a JSON object or a path to one")
    return {str(key): str(column) for key, column in loaded.items()}


def _load_metadata(value: str | None) -> Any:
    if not value:
        return None
    path = Path(value)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError("--metadata must be a CSV or JSON file")


def _sample_group_by(value: str | None) -> str | None:
    if value is None or value == "inferred":
        return None
    if value not in ("sample_id", "sample_name", "sample_long_name"):
        raise ValueError(
            "--sample-group-by must be inferred, sample_id, sample_name, or sample_long_name"
        )
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
