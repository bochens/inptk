"""Command-line access to the same workflow used by Python and external apps."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import (
    __version__,
    analyze_concentration,
    frozen_fraction,
    load,
    read_counts,
    read_icescopy,
    read_observations,
)
from .experiment import AnalysisResult, Experiment
from .io import FORMAT_VERSION, _encode, _table_payload
from .tables import CountsTable

CLI_PROTOCOL_VERSION = 2


class _UsageError(Exception):
    """An argparse failure that main can also report as JSON."""


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _UsageError(message)


def _json_flag(parser):
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Return one versioned JSON response on stdout, including failures",
    )


def build_parser():
    parser = _Parser(prog="inptk", description="INP-toolkit droplet-freezing analysis")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _json_flag(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    capabilities = commands.add_parser("capabilities", help="Describe the installed CLI as JSON")
    _json_flag(capabilities)
    preview = commands.add_parser(
        "preview", help="Read original counts and fractions without fitting concentrations"
    )
    preview.add_argument("input")
    preview.add_argument("--format", choices=("native", "icescopy", "saved"), default="native")
    preview.add_argument("--metadata", help="Optional or incomplete measurement metadata")
    preview.add_argument(
        "--sample-map", help="JSON object or file mapping Icescopy measurement names to samples"
    )
    preview.add_argument("--run-id", default="1")
    _json_flag(preview)
    analyze = commands.add_parser(
        "analyze", help="Calculate named concentration curves from original observations"
    )
    _json_flag(analyze)
    analyze.add_argument("input")
    analyze.add_argument("--metadata", help="Native metadata or Icescopy metadata overrides")
    analyze.add_argument(
        "--format",
        choices=("native", "icescopy", "saved"),
        default="native",
        help=(
            "Saved analyses reuse original observations; processing settings use "
            "this command's arguments, not prior settings"
        ),
    )
    analyze.add_argument(
        "--sample-map",
        help="JSON object or file mapping Icescopy measurement names to parent samples",
    )
    analyze.add_argument(
        "--water-blank-map",
        help="JSON object or file mapping native measurements to lists of blank measurements",
    )
    analyze.add_argument(
        "--no-water-blank-correction",
        action="store_true",
        help="Analyze sample counts without water correction while retaining raw blank context",
    )
    analyze.add_argument("--run-id", default="1")
    analyze.add_argument(
        "--sample",
        action="append",
        help="Exact parent sample ID to analyze; repeat to select several",
    )
    analyze.add_argument(
        "--cycle", action="append", help="Exact cycle ID to analyze; repeat to select several"
    )
    analyze.add_argument("--out", required=True)
    analyze.add_argument(
        "--method",
        choices=("mle", "average"),
        default="mle",
        help="Fit eligible counts together (mle) or average their concentration estimates",
    )
    analyze.add_argument(
        "--temperature-ranges",
        help="JSON object or file mapping measurement IDs to inclusive min_C/max_C limits",
    )
    analyze.add_argument(
        "--output-basis", choices=("suspension", "sampled_air", "dry_soil"), default="suspension"
    )
    analyze.add_argument(
        "--curves",
        help="JSON object or file mapping curve names to inputs lists and optional cycle labels",
    )
    analyze.add_argument(
        "--output-step-C",
        type=float,
        help="Optional temperature spacing applied only to the final result",
    )
    analyze.add_argument("--output-method", choices=("sample", "interpolate"), default="sample")
    analyze.add_argument("--z", type=float, default=1.96)
    analyze.add_argument("--differential", action="store_true")
    analyze.add_argument(
        "--decrease-policy",
        choices=("stop_at_decrease", "skip_decreases"),
        default="stop_at_decrease",
        help="Select final cumulative points without changing calculated values",
    )
    export = commands.add_parser("export-csv", help="Export a quantity from named saved curves")
    _json_flag(export)
    export.add_argument("input")
    export.add_argument("--out", required=True)
    export.add_argument("--curve", help="Export one exact curve name; otherwise export all curves")
    export.add_argument(
        "--table", choices=("cumulative", "resampled", "excluded"), default="cumulative"
    )
    return parser


def _select_experiment(experiment, sample_ids, cycle_ids):
    if sample_ids is None and cycle_ids is None:
        return experiment
    original = experiment.counts.to_dataframe()
    blank_ids = {name for assigned in experiment.water_blank_map.values() for name in assigned}
    frame = original.loc[~original.measurement_id.isin(blank_ids)]
    selection = {}
    for column, values in (("sample_id", sample_ids), ("cycle_id", cycle_ids)):
        if values is None:
            continue
        unknown = set(values) - set(frame[column])
        if unknown:
            blank_only = unknown & set(
                original.loc[original.measurement_id.isin(blank_ids), column]
            )
            if column == "sample_id" and blank_only:
                raise ValueError(
                    "Water-blank parent samples cannot be selected as analysis samples: "
                    f"{sorted(blank_only)}"
                )
            raise ValueError(f"Unknown {column} selection: {sorted(unknown)}")
        frame = frame.loc[frame[column].isin(values)]
        selection[column] = list(dict.fromkeys(values))
    measurement_ids = set(frame.measurement_id)
    water_blank_map = {
        key: value for key, value in experiment.water_blank_map.items() if key in measurement_ids
    }
    if water_blank_map:
        keys = ["measurement_id", "run_id", "cycle_id"]
        required = frame[keys].drop_duplicates().copy()
        required["measurement_id"] = required.measurement_id.map(water_blank_map)
        required = required.explode("measurement_id").drop_duplicates()
        blank_rows = pd.MultiIndex.from_frame(original[keys]).isin(
            pd.MultiIndex.from_frame(required.drop_duplicates())
        )
        frame = original.loc[original.index.isin(frame.index) | blank_rows]
        measurement_ids.update(name for assigned in water_blank_map.values() for name in assigned)
    measurements = {
        key: value for key, value in experiment.measurements.items() if key in measurement_ids
    }
    retained_samples = {value.sample_id for value in measurements.values()}
    counts = CountsTable(
        frame, history=experiment.counts.history + [{"operation": "select", **selection}]
    )
    return Experiment(
        counts,
        {key: value for key, value in experiment.samples.items() if key in retained_samples},
        measurements,
        source={**experiment.source, "selection": selection},
        water_blank_map=water_blank_map,
    )


def _json_object(value, option):
    if value is None:
        return None
    payload = value
    if not payload.lstrip().startswith(("{", "[")):
        payload = Path(payload).read_text(encoding="utf-8")

    def unique_keys(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError(f"{option} contains a duplicate name: {key!r}")
            result[key] = item
        return result

    result = json.loads(payload, object_pairs_hook=unique_keys)
    if not isinstance(result, dict):
        raise TypeError(f"{option} must contain a JSON object")
    return result


def _response(command, *, status="ok", warnings=None, **payload):
    return {
        "protocol_version": CLI_PROTOCOL_VERSION,
        "toolkit_version": __version__,
        "saved_format_version": FORMAT_VERSION,
        "command": command,
        "status": status,
        "warnings": [] if warnings is None else warnings,
        **payload,
    }


def _print_json(payload):
    print(json.dumps(_encode(payload), allow_nan=False))


def _capabilities(parser):
    # Derive flags, defaults and choices from the parser used by this executable.
    # The GUI need not scrape --help or duplicate changing scientific defaults.
    commands = {}
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for name, command in action.choices.items():
            options = []
            for option in command._actions:
                if option.dest in ("help", "json"):
                    continue
                options.append(
                    {
                        "name": option.dest,
                        "flags": option.option_strings,
                        "required": option.required,
                        "default": None if option.default == argparse.SUPPRESS else option.default,
                        "choices": None if option.choices is None else list(option.choices),
                        "type": option.type.__name__
                        if option.type
                        else "boolean"
                        if isinstance(option, argparse._StoreTrueAction)
                        else "string",
                        "repeatable": isinstance(option, argparse._AppendAction),
                        "help": option.help,
                    }
                )
            commands[name] = {"options": options}
    return _response(
        "capabilities",
        commands=commands,
        observation_tables=["counts", "frozen_fraction"],
        curve_tables=["cumulative", "excluded", "differential", "resampled"],
        curve_specification={
            "inputs": "Nonempty list of input names or measurement_id/cycle_id objects",
            "cycle": "Optional exact label for input names; required if several cycles exist",
            "default": "One curve per original sample, run and cycle",
        },
        json_number_encoding={"nonfinite_key": "$nonfinite", "values": ["nan", "inf", "-inf"]},
        exit_codes={"success": 0, "processing_error": 1, "usage_error": 2, "cancelled": 130},
    )


def _preview(args):
    blank_map = {}
    if args.format == "saved":
        if args.metadata or args.sample_map:
            raise ValueError("Saved input already contains its metadata and sample mapping")
        source = load(args.input)
        experiment = source.experiment if isinstance(source, AnalysisResult) else source
        counts = experiment.counts
        metadata = [
            {**asdict(experiment.samples[item.sample_id]), **asdict(item)}
            for item in experiment.measurements.values()
        ]
        provisional = []
        blank_map = experiment.water_blank_map
    else:
        mapping = _json_object(args.sample_map, "--sample-map")
        counts = read_observations(
            args.input,
            format=args.format,
            metadata=args.metadata,
            sample_map=mapping,
            run_id=args.run_id,
        )
        imported = counts.history[-1]
        metadata = imported["measurement_metadata"]
        provisional = imported["provisional_sample_assignments"]
    # Check concentration metadata with the same importer, without any fitting.
    # Blank coverage, selected groups/ranges and output normalization are checked
    # only by analyze, after the user has made those decisions.
    missing = {
        item["measurement_id"]: [
            name for name in ("dilution", "droplet_volume_uL") if item.get(name) is None
        ]
        for item in metadata
    }
    missing = {key: value for key, value in missing.items() if value}
    metadata_error = None
    if missing:
        metadata_error = "Missing concentration metadata: " + "; ".join(
            f"{key}: {', '.join(value)}" for key, value in missing.items()
        )
    else:
        try:
            read_counts(counts.to_dataframe(), metadata=metadata, water_blank_map=blank_map)
        except (ValueError, TypeError, KeyError) as error:
            metadata_error = str(error)
    frame = counts.to_dataframe()
    measurements = []
    for (measurement, run, sample), rows in frame.groupby(
        ["measurement_id", "run_id", "sample_id"], sort=False
    ):
        measurements.append(
            {
                "measurement_id": measurement,
                "run_id": run,
                "sample_id": sample,
                "cycle_ids": rows.cycle_id.unique().tolist(),
                "observation_count": len(rows),
                "temperature_min_C": float(rows.temperature_C.min()),
                "temperature_max_C": float(rows.temperature_C.max()),
            }
        )
    return _response(
        "preview",
        table=_table_payload(frozen_fraction(counts)),
        measurements=measurements,
        measurement_metadata=metadata,
        provisional_sample_assignments=provisional,
        water_blank_map=blank_map,
        suspension_metadata={
            "valid": metadata_error is None,
            "error": metadata_error,
            "missing_fields": {key: value for key, value in missing.items() if value},
        },
        analysis_performed=False,
    )


def _error_code(error):
    if isinstance(error, FileExistsError):
        return "output_exists"
    if isinstance(error, FileNotFoundError):
        return "file_not_found"
    if isinstance(error, PermissionError):
        return "permission_denied"
    if isinstance(error, OSError):
        return "io_error"
    return "invalid_input"


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    # An invalid invocation still needs a machine-readable response. Honor the
    # flag only before a '--' separator, where it is an option rather than data.
    option_arguments = arguments[: arguments.index("--")] if "--" in arguments else arguments
    json_mode = "--json" in option_arguments
    command = None
    try:
        args = parser.parse_args(arguments)
        command = args.command
        json_mode = bool(getattr(args, "json", json_mode))
        if command == "capabilities":
            _print_json(_capabilities(parser))
            return 0
        if command == "preview":
            preview = _preview(args)
            if json_mode:
                _print_json(preview)
            else:
                print(
                    f"Read {len(preview['table']['rows'])} original observations "
                    f"from {len(preview['measurements'])} measurements"
                )
                if not preview["suspension_metadata"]["valid"]:
                    print(
                        f"Concentration metadata incomplete or invalid: "
                        f"{preview['suspension_metadata']['error']}"
                    )
                print("No concentrations calculated. Use --json for plot data and metadata.")
            return 0
        if Path(args.out).exists():
            raise FileExistsError(f"Output already exists: {args.out}")
        if command == "export-csv":
            result = load(args.input)
            if not isinstance(result, AnalysisResult):
                raise TypeError("CSV export requires a saved AnalysisResult")
            result.export_csv(args.out, table=args.table, curve_id=args.curve)
            if json_mode:
                _print_json(
                    _response(
                        command,
                        output=str(Path(args.out).resolve()),
                        table=args.table,
                        curve_id=args.curve,
                    )
                )
            return 0
        temperature_ranges = _json_object(args.temperature_ranges, "--temperature-ranges")
        curves = _json_object(args.curves, "--curves")
        if args.format == "native":
            if args.sample_map:
                raise ValueError("--sample-map applies only to Icescopy input")
            if not args.metadata:
                raise ValueError("Native input requires --metadata")
            water_blank_map = _json_object(args.water_blank_map, "--water-blank-map")
            experiment = read_counts(
                args.input,
                metadata=args.metadata,
                run_id=args.run_id,
                water_blank_map=water_blank_map,
            )
        elif args.format == "icescopy":
            if args.water_blank_map:
                raise ValueError(
                    "The Icescopy CSV adapter does not include raw blank context. "
                    "--water-blank-map requires raw sample and blank counts in --format native."
                )
            mapping = _json_object(args.sample_map, "--sample-map")
            overrides = None
            if args.metadata:
                from .readers import _frame

                overrides = _frame(args.metadata)
            experiment = read_icescopy(
                args.input, sample_map=mapping, metadata=overrides, run_id=args.run_id
            )
        else:
            if args.water_blank_map:
                raise ValueError("Saved input already contains its water-blank mapping")
            if args.metadata or args.sample_map:
                raise ValueError("Saved input already contains its metadata and sample mapping")
            experiment = load(args.input)
            if isinstance(experiment, AnalysisResult):
                experiment = experiment.experiment
        experiment = _select_experiment(experiment, args.sample, args.cycle)
        result = analyze_concentration(
            experiment,
            method=args.method,
            temperature_ranges_C=temperature_ranges,
            curves=curves,
            output_basis=args.output_basis,
            output_step_C=args.output_step_C,
            output_method=args.output_method,
            z=args.z,
            differential=args.differential,
            water_blank_correction=not args.no_water_blank_correction,
            decrease_policy=args.decrease_policy,
        )
        result.save(args.out)
        if json_mode:
            _print_json(
                _response(
                    command,
                    output=str(Path(args.out).resolve()),
                    warnings=result.warnings,
                    settings=result.settings,
                    observation_tables={
                        "counts": {"type": "CountsTable", "row_count": len(result.counts)},
                        "frozen_fraction": {
                            "type": "FrozenFractionTable",
                            "row_count": len(result.frozen_fraction),
                        },
                    },
                    curves={
                        name: {
                            "curve_id": curve.curve_id,
                            "kind": curve.kind,
                            "sources": curve.sources,
                            "tables": {
                                quantity: {"type": type(table).__name__, "row_count": len(table)}
                                for quantity in (
                                    "cumulative",
                                    "excluded",
                                    "differential",
                                    "resampled",
                                )
                                if (table := getattr(curve, quantity)) is not None
                            },
                        }
                        for name, curve in result.curves.items()
                    },
                )
            )
        else:
            for warning in result.warnings:
                print(f"Warning: {warning}")
            count = sum(len(curve.cumulative) for curve in result.curves.values())
            print(f"Saved {count} concentration rows in {len(result.curves)} curves to {args.out}")
    except _UsageError as error:
        if json_mode:
            _print_json(
                _response(
                    command, status="error", error={"code": "usage_error", "message": str(error)}
                )
            )
        else:
            parser.print_usage(sys.stderr)
            parser.exit(2, f"inptk: error: {error}\n")
        return 2
    except (ValueError, TypeError, KeyError, OSError, AttributeError) as error:
        if json_mode:
            _print_json(
                _response(
                    command,
                    status="error",
                    error={"code": _error_code(error), "message": str(error)},
                )
            )
            return 1
        parser.exit(1, f"inptk: {error}\n")
    except KeyboardInterrupt:
        if json_mode:
            _print_json(
                _response(
                    command,
                    status="error",
                    error={"code": "cancelled", "message": "Analysis interrupted"},
                )
            )
        else:
            print("inptk: interrupted", file=sys.stderr)
        return 130
    except Exception as error:
        # Preserve diagnostic details for maintainers without making the GUI
        # parse a traceback as its response. Human CLI keeps normal tracebacks.
        if not json_mode:
            raise
        import traceback

        traceback.print_exc(file=sys.stderr)
        _print_json(
            _response(
                command,
                status="error",
                error={"code": "internal_error", "message": str(error)},
            )
        )
        return 1
    return 0
