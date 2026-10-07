"""Command-line access to the same workflow used by Python and external apps."""

import argparse
import json
import sys
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from . import (
    __version__,
    analyze_concentration,
    cumulative_spectrum,
    estimate_concentration,
    frozen_fraction,
    read_counts,
    read_icescopy,
    read_observations,
    suggest_temperature_ranges,
)
from .cli_steps import (
    complete_step_metadata,
    counts_from_step,
    fraction_step,
    select_table,
    table_summary,
    transform_step,
)
from .cli_store import ResultStore
from .experiment import Experiment, ProcessingResult
from .io import FORMAT_VERSION, _encode, _table_frame, _table_payload
from .readers import _default_water_blank_map
from .settings import DEFAULTS
from .tables import CountsTable

CLI_PROTOCOL_VERSION = 2
_CLIENT_WRITER: ContextVar[Callable[[dict], None] | None] = ContextVar(
    "inptk_client_writer", default=None
)


class _UsageError(Exception):
    """An argparse failure that main can also report as JSON."""


class _Parser(argparse.ArgumentParser):
    def _print_message(self, message, file=None):
        # argparse can accept abbreviated or combined help/version flags. Those
        # must not print prose or exit the persistent client's request loop.
        if _CLIENT_WRITER.get() is not None:
            raise _UsageError("Use capabilities for client discovery")
        super()._print_message(message, file)

    def error(self, message):
        raise _UsageError(message)


def _json_flag(parser):
    parser.add_argument(
        "--json",
        action="store_true",
        default=argparse.SUPPRESS,
        help="Return one versioned JSON response on stdout, including failures",
    )


def _analysis_input_arguments(parser):
    parser.add_argument("input")
    parser.add_argument("--metadata", help="Native metadata or Icescopy metadata overrides")
    parser.add_argument(
        "--format",
        choices=("native", "icescopy", "saved"),
        default="native",
        help=(
            "Icescopy input accepts CSV or .icescopy projects. "
            "Saved analyses reuse original observations; processing settings use "
            "this command's arguments, not prior settings"
        ),
    )
    parser.add_argument(
        "--sample-map",
        help="JSON object or file mapping Icescopy measurement names to parent samples",
    )
    parser.add_argument(
        "--water-blank-map",
        help=(
            "JSON object or file mapping inputs to blanks; omitted uses all entries "
            "typed 'water blank' in the same run, {} disables correction"
        ),
    )
    parser.add_argument(
        "--no-water-blank-correction",
        action="store_true",
        help="Analyze sample counts without water correction while retaining raw blank context",
    )
    parser.add_argument(
        "--water-blank-after-first-freeze", action="store_true",
        help="Fix each run/cycle's combined blank background at zero before its first freeze",
    )
    parser.add_argument(
        "--water-blank-temperature-range",
        help="JSON object or file with min_C/max_C observation limits shared by all water blanks",
    )
    parser.add_argument("--run-id", default="1")
    parser.add_argument(
        "--sample",
        action="append",
        help="Exact parent sample ID to analyze; repeat to select several",
    )
    parser.add_argument(
        "--cycle", action="append", help="Exact cycle ID to analyze; repeat to select several"
    )


def build_parser():
    parser = _Parser(prog="inptk", description="INP-toolkit droplet-freezing analysis")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    _json_flag(parser)
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser(
        "serve", help="Keep Python and in-memory results alive for a client"
    )
    _json_flag(serve)
    capabilities = commands.add_parser("capabilities", help="Describe the installed CLI as JSON")
    _json_flag(capabilities)
    preview = commands.add_parser(
        "preview", help="Read original counts and fractions without fitting concentrations"
    )
    _observation_arguments(preview)
    _json_flag(preview)
    analyze = commands.add_parser(
        "analyze", help="Calculate named concentration curves from original observations"
    )
    _json_flag(analyze)
    _analysis_input_arguments(analyze)
    analyze.add_argument("--out", required=True)
    _estimation_arguments(analyze)
    analyze.add_argument(
        "--output-basis",
        choices=("suspension", "sampled_air", "dry_soil"),
        default=DEFAULTS.output_basis,
    )
    analyze.add_argument("--differential", action="store_true")
    analyze.add_argument(
        "--decrease-policy",
        choices=("stop_at_decrease", "skip_decreases"),
        default=DEFAULTS.decrease_policy,
        help="Select final cumulative points without changing calculated values",
    )
    suggest = commands.add_parser(
        "suggest-ranges", help="Suggest Average ranges and flag weak blank-corrected signals"
    )
    _json_flag(suggest)
    _analysis_input_arguments(suggest)
    suggest.add_argument(
        "--summary",
        action="store_true",
        help="Return limits and reasons without the per-observation report",
    )
    suggest.add_argument("--curves", help="JSON object or file selecting named curves and cycles")
    suggest.add_argument(
        "--min-frozen",
        type=int,
        default=3,
        help="Minimum frozen sample wells after the first dilution (default: 3)",
    )
    suggest.add_argument(
        "--min-unfrozen", type=int, default=3, help="Minimum liquid sample wells (default: 3)"
    )
    suggest.add_argument("--z", type=float, default=DEFAULTS.z)
    _temperature_arguments(suggest)
    fractions = commands.add_parser(
        "fractions", help="Calculate and save original frozen fractions"
    )
    _observation_arguments(fractions)
    fractions.add_argument(
        "--water-blank-map", help="Explicit blank assignments with complete metadata"
    )
    fractions.add_argument("--out", required=True)
    _json_flag(fractions)
    estimate = commands.add_parser(
        "estimate", help="Estimate concentrations without final selection"
    )
    _analysis_input_arguments(estimate)
    _estimation_arguments(estimate, individual=True)
    estimate.add_argument("--out", required=True)
    _json_flag(estimate)
    for name, help_text in (
        ("convert", "Convert existing suspension concentrations to air or soil units"),
        ("finalize", "Select final cumulative points without repeating estimation"),
        ("differentiate", "Differentiate existing individual suspension spectra without fitting"),
    ):
        step = commands.add_parser(name, help=help_text)
        step.add_argument("input", help="Saved .inptk calculation step or analysis")
        step.add_argument("--out", required=True)
        _json_flag(step)
        if name == "convert":
            step.add_argument(
                "--output-basis", required=True, choices=("suspension", "sampled_air", "dry_soil")
            )
        if name == "finalize":
            step.add_argument(
                "--decrease-policy",
                choices=("stop_at_decrease", "skip_decreases"),
                default=DEFAULTS.decrease_policy,
            )
    save = commands.add_parser("save", help="Save an existing result without recalculating")
    save.add_argument("input", help="Saved input path or an in-memory @reference")
    save.add_argument("--out", required=True)
    _json_flag(save)
    table = commands.add_parser("table", help="List saved quantities or read a table for plotting")
    table.add_argument("input", help="Saved .inptk experiment, step or analysis")
    table.add_argument(
        "--table", choices=("counts", "frozen_fraction", "cumulative", "excluded", "differential")
    )
    table.add_argument("--curve", help="Select one exact named curve")
    table.add_argument(
        "--columns", nargs="+", help="Return only these table columns, in this order"
    )
    table.add_argument(
        "--no-history", action="store_true", help="Omit processing history from JSON"
    )
    _json_flag(table)
    export = commands.add_parser("export-csv", help="Export a quantity from named saved curves")
    _json_flag(export)
    export.add_argument("input")
    export.add_argument("--out", required=True)
    export.add_argument("--curve", help="Export one exact curve name; otherwise export all curves")
    export.add_argument(
        "--table",
        choices=("counts", "frozen_fraction", "cumulative", "excluded", "differential"),
        default="cumulative",
    )
    return parser


def _observation_arguments(parser):
    parser.add_argument("input")
    parser.add_argument("--format", choices=("native", "icescopy", "saved"), default="native")
    parser.add_argument("--metadata", help="Optional or incomplete measurement metadata")
    parser.add_argument(
        "--sample-map", help="JSON object or file mapping Icescopy inputs to samples"
    )
    parser.add_argument("--run-id", default="1")


def _estimation_arguments(parser, *, individual=False):
    parser.add_argument(
        "--method",
        choices=("mle", "average"),
        default=DEFAULTS.method,
        help="Joint monotone curve fit (mle) or pointwise concentration average",
    )
    parser.add_argument(
        "--temperature-ranges", help="JSON object or file mapping input names to min_C/max_C limits"
    )
    choices = parser.add_mutually_exclusive_group() if individual else parser
    choices.add_argument("--curves", help="JSON object or file selecting named curves and cycles")
    if individual:
        choices.add_argument(
            "--individual",
            action="store_true",
            help="Calculate each physical input separately, including for differentiation",
        )
    _temperature_arguments(parser)
    parser.add_argument("--fit-step-C", type=float, help="MLE curve shape spacing in degrees C")
    parser.add_argument("--z", type=float, default=DEFAULTS.z)


def _estimation_settings(args):
    return {
        "method": args.method,
        "fit_step_C": args.fit_step_C,
        "temperature_step_C": args.temperature_step_C,
        "temperature_start_C": args.temperature_start_C,
        "temperature_end_C": args.temperature_end_C,
        "temperature_method": args.temperature_method,
        "temperature_window_C": args.temperature_window_C,
        "temperature_ranges_C": _json_object(args.temperature_ranges, "--temperature-ranges"),
        "curves": _json_object(args.curves, "--curves"),
        "z": args.z,
        "water_blank_correction": not args.no_water_blank_correction,
        "water_blank_after_first_freeze": args.water_blank_after_first_freeze,
        "water_blank_temperature_range_C": _json_object(
            args.water_blank_temperature_range, "--water-blank-temperature-range"
        ),
    }


def _temperature_arguments(parser):
    parser.add_argument(
        "--temperature-step-C",
        type=float,
        help="Optional count-selection grid spacing before estimation and blank correction",
    )
    parser.add_argument(
        "--temperature-start-C",
        type=float,
        default=DEFAULTS.temperature_start_C,
        help="Warm grid endpoint; defaults to the warmest selected input temperature",
    )
    parser.add_argument(
        "--temperature-end-C",
        type=float,
        default=DEFAULTS.temperature_end_C,
        help="Cold grid endpoint; defaults to the coldest selected input temperature",
    )
    parser.add_argument(
        "--temperature-method",
        choices=("latest", "max", "window"),
        default=DEFAULTS.temperature_method,
        help="Select latest warmer counts, maximum warmer fraction, or maximum count in a window",
    )
    parser.add_argument(
        "--temperature-window-C",
        type=float,
        help="Full centered window width in degrees C; required only for window",
    )


def _select_experiment(experiment, sample_ids, cycle_ids):
    if sample_ids is None and cycle_ids is None:
        return experiment
    original = experiment.counts.to_dataframe()
    blank_ids = experiment.water_blank_ids
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
    writer = _CLIENT_WRITER.get()
    if writer is not None:
        writer(payload)
    else:
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
        curve_tables=["cumulative", "excluded", "differential"],
        saved_kinds=["experiment", "processing", "analysis"],
        client_mode={
            "command": "serve",
            "transport": "one JSON object per line on stdin/stdout",
            "request": {"id": "request ID", "args": ["command", "arguments"]},
            "memory_reference_prefix": "@",
            "import": {
                "id": "request ID",
                "import": {
                    "out": "@input",
                    "counts": "Native count records or an object of column arrays",
                    "metadata": "Complete measurement metadata records or column arrays",
                    "water_blank_map": (
                        "Optional mapping; omitted uses typed blanks by run, {} disables correction"
                    ),
                    "run_id": "Optional default run identity",
                },
            },
            "release": {"id": "request ID", "release": ["@result"]},
        },
        temperature_range_selection={
            "domain": "calculation targets; select counts before applying input limits",
            "boundaries": "inclusive; original source rows unchanged",
            "full_range": "each sample uses its first-to-last freezing interval; explicit full limits are equivalent",
            "resolved_settings": "resolved_temperature_ranges_C, keyed by curve then input",
            "selected_counts": "source_observations in cumulative and excluded curve tables",
        },
        concentration_reporting={
            "rule": "observed_sample_freezing_interval",
            "boundaries": "inclusive; original sample events; blanks do not extend the interval",
            "grid": "existing selected points only; no additional endpoint rows",
            "zeros": "retained inside the interval",
            "outside_values": "NaN concentration and uncertainty; raw observations retained",
            "individual": "every selected point between original first and last sample freezes",
            "exclusion_ranges": "combined curves only",
            "decrease_selection": "combined curves only",
            "fit_observations": "each sample uses its own freezing interval intersected with selected limits",
            "final_tables": "cumulative",
            "diagnostic_tables": "excluded; unfinalized estimates",
            "csv": "cumulative export omits points outside the interval",
        },
        step_sequence=["fractions", "estimate", "convert", "finalize"],
        individual_sequence=["fractions", "estimate --individual", "differentiate"],
        estimation_methods={
            "mle": {
                "fit": "joint_monotone_first_freezing_curve",
                "input": "fixed well totals and cumulative first-freezing counts per cycle",
                "uncertainty": "pointwise profile bounds from the complete curve likelihood",
                "output_order": "selected analysis temperatures, warm to cold",
            },
            "average": {
                "calculation": "direct corrected concentrations; arithmetic mean in overlap",
                "contributors": "sample counts with 0 < n_frozen < n_total within "
                "each input's first-to-last freezing interval and selected ranges",
                "uncertainty": "propagated Wilson binomial bounds; shared blank counted once",
                "output_order": "native observation order, latest-warmer alignment when needed",
            },
        },
        curve_specification={
            "inputs": "Nonempty list of input names or measurement_id/cycle_id objects",
            "cycle": "Optional exact label for input names; required if several cycles exist",
            "default": "One curve per original sample, run and cycle",
        },
        json_number_encoding={"nonfinite_key": "$nonfinite", "values": ["nan", "inf", "-inf"]},
        exit_codes={"success": 0, "processing_error": 1, "usage_error": 2, "cancelled": 130},
    )


def _preview(args, store):
    blank_map = None
    if args.format == "saved":
        if args.metadata or args.sample_map:
            raise ValueError("Saved input already contains its metadata and sample mapping")
        source = store.load(args.input)
        experiment = source if isinstance(source, Experiment) else source.experiment
        counts = counts_from_step(source)
        if experiment is not None:
            metadata = [
                {**asdict(experiment.samples[item.sample_id]), **asdict(item)}
                for item in experiment.measurements.values()
            ]
            provisional = []
            blank_map = experiment.water_blank_map
        else:
            imported = next(step for step in counts.history if "measurement_metadata" in step)
            metadata = imported["measurement_metadata"]
            provisional = imported["provisional_sample_assignments"]
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
    if blank_map is None:
        blank_map = _default_water_blank_map(metadata)
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


def _read_analysis_input(args, store, *, saved=None):
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
        water_blank_map = _json_object(args.water_blank_map, "--water-blank-map")
        mapping = _json_object(args.sample_map, "--sample-map")
        overrides = None
        if args.metadata:
            from .readers import _frame

            overrides = _frame(args.metadata)
        experiment = read_icescopy(
            args.input,
            sample_map=mapping,
            metadata=overrides,
            run_id=args.run_id,
            water_blank_map=water_blank_map,
        )
    else:
        source = store.load(args.input) if saved is None else saved
        if isinstance(source, ProcessingResult) and source.experiment is None:
            if args.sample_map:
                raise ValueError("--sample-map applies only to Icescopy input")
            if not args.metadata:
                raise ValueError(
                    "This fraction step needs --metadata before concentration estimation"
                )
            experiment = complete_step_metadata(
                source, args.metadata, _json_object(args.water_blank_map, "--water-blank-map")
            )
        else:
            if args.water_blank_map:
                raise ValueError("Saved input already contains its water-blank mapping")
            if args.metadata or args.sample_map:
                raise ValueError("Saved input already contains its metadata and sample mapping")
            experiment = source if isinstance(source, Experiment) else source.experiment
    return _select_experiment(experiment, args.sample, args.cycle)


def _fraction_input(args, store):
    blank_map = _json_object(args.water_blank_map, "--water-blank-map")
    if args.format == "saved":
        source = store.load(args.input)
        experiment = source if isinstance(source, Experiment) else source.experiment
        if args.sample_map:
            raise ValueError("Saved input already contains its sample mapping")
        if experiment is not None:
            if args.metadata or args.water_blank_map:
                raise ValueError(
                    "Saved input already contains its metadata and water-blank mapping"
                )
            return fraction_step(experiment.counts, experiment=experiment)
        if args.metadata:
            experiment = complete_step_metadata(source, args.metadata, blank_map)
            return fraction_step(experiment.counts, experiment=experiment)
        if blank_map:
            raise ValueError(
                "Complete physical metadata before assigning blanks; fractions use raw counts"
            )
        return fraction_step(counts_from_step(source))
    counts = read_observations(
        args.input,
        format=args.format,
        metadata=args.metadata,
        sample_map=_json_object(args.sample_map, "--sample-map"),
        run_id=args.run_id,
    )
    records = counts.history[-1]["measurement_metadata"]
    complete = all(
        record.get(key) is not None
        for record in records
        for key in ("dilution", "droplet_volume_uL")
    )
    if complete:
        imported = read_counts(counts.to_dataframe(), metadata=records, water_blank_map=blank_map)
        experiment = Experiment(
            counts,
            imported.samples,
            imported.measurements,
            source=counts.history[-1],
            water_blank_map=imported.water_blank_map,
        )
    else:
        if blank_map:
            raise ValueError(
                "Complete physical metadata before assigning blanks; fractions use raw counts"
            )
        experiment = None
    return fraction_step(counts, experiment=experiment)


def _estimate_step(args, store):
    if args.individual and args.temperature_ranges is not None:
        raise ValueError("--temperature-ranges applies to combined curves, not --individual")
    source = store.load(args.input) if args.format == "saved" else None
    experiment = _read_analysis_input(args, store, saved=source)
    tables = table_summary(source) if source is not None else {}
    if "frozen_fraction" in tables and getattr(source, "experiment", None) is not None:
        fractions = select_table(source, "frozen_fraction")
        if args.sample or args.cycle:
            selected = experiment.counts.to_dataframe()
            fractions = fractions.select(
                measurement_id=list(experiment.measurements),
                cycle_id=selected.cycle_id.unique().tolist(),
            )
    else:
        fractions = frozen_fraction(experiment)
    settings = _estimation_settings(args)
    if args.individual:
        settings.pop("curves")
        settings.pop("temperature_ranges_C")
        estimated = cumulative_spectrum(fractions, experiment=experiment, **settings)
    else:
        estimated = estimate_concentration(fractions, experiment=experiment, **settings)
    return ProcessingResult({"frozen_fraction": fractions, "cumulative": estimated}, experiment)


def _save_step(result, args, json_mode, store):
    store.save(result, args.out)
    warnings = list(
        dict.fromkeys(warning for table in result.tables.values() for warning in table.warnings)
    )
    if json_mode:
        _print_json(
            _response(
                args.command,
                output=store.output_name(args.out),
                tables=table_summary(result),
                warnings=warnings,
            )
        )
    else:
        print(f"Saved {args.command} result to {args.out}")
        for name, table in result.tables.items():
            print(f"  {name}: {len(table)} rows")
        for warning in warnings:
            print(f"Warning: {warning}")


def _write_client_reply(payload, request_id):
    print(
        json.dumps(_encode({**payload, "id": request_id}), allow_nan=False, separators=(",", ":")),
        flush=True,
    )


def serve_client(parser):
    store = ResultStore(memory=True)
    for line in sys.stdin:
        request_id = None
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise TypeError("Client request must be a JSON object")
            request_id = request.get("id")
            if not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
                raise TypeError("Client request id must be a string or integer")
            if set(request) == {"id", "release"}:
                store.release(request["release"])
                reply = _response("release", released=request["release"])
                _write_client_reply(reply, request_id)
                continue
            if set(request) == {"id", "import"}:
                try:
                    experiment = store.import_counts(request["import"])
                    reply = _response(
                        "import",
                        output=request["import"]["out"],
                        tables=table_summary(experiment),
                        measurements=list(experiment.measurements),
                    )
                    del experiment  # The store alone owns the uploaded experiment.
                except (ValueError, TypeError, KeyError, OSError, AttributeError) as error:
                    reply = _response(
                        "import",
                        status="error",
                        error={"code": _error_code(error), "message": str(error)},
                    )
                _write_client_reply(reply, request_id)
                continue
            if set(request) != {"id", "args"}:
                raise ValueError("Client request requires id and one of args, import or release")
            args = request["args"]
            if not isinstance(args, list) or not args or any(not isinstance(a, str) for a in args):
                raise ValueError("args must be a nonempty list of command-line strings")
            if args[0] == "serve" or any(a in ("-h", "--help", "--version") for a in args):
                raise ValueError(
                    "Use capabilities for client discovery; nested serve is not allowed"
                )

            def write_reply(payload, request_id=request_id):
                _write_client_reply(payload, request_id)

            token = _CLIENT_WRITER.set(write_reply)
            try:
                main(["--json", *args], store=store, parser=parser)
            finally:
                _CLIENT_WRITER.reset(token)
        except (ValueError, TypeError) as error:
            reply = _response(
                None, status="error", error={"code": "invalid_request", "message": str(error)}
            )
            _write_client_reply(reply, request_id)
    return 0


def main(argv=None, *, store=None, parser=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser() if parser is None else parser
    store = ResultStore() if store is None else store
    # An invalid invocation still needs a machine-readable response. Honor the
    # flag only before a '--' separator, where it is an option rather than data.
    option_arguments = arguments[: arguments.index("--")] if "--" in arguments else arguments
    json_mode = "--json" in option_arguments
    command = None
    try:
        args = parser.parse_args(arguments)
        command = args.command
        json_mode = bool(getattr(args, "json", json_mode))
        if command == "serve":
            if store.memory:
                raise ValueError("Nested serve is not allowed")
            return serve_client(parser)
        if command == "capabilities":
            _print_json(_capabilities(parser))
            return 0
        if command == "preview":
            preview = _preview(args, store)
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
        if command == "suggest-ranges":
            proposal = suggest_temperature_ranges(
                _read_analysis_input(args, store),
                curves=_json_object(args.curves, "--curves"),
                include_observations=not args.summary,
                min_frozen=args.min_frozen,
                min_unfrozen=args.min_unfrozen,
                z=args.z,
                temperature_step_C=args.temperature_step_C,
                temperature_start_C=args.temperature_start_C,
                temperature_end_C=args.temperature_end_C,
                temperature_method=args.temperature_method,
                temperature_window_C=args.temperature_window_C,
                water_blank_correction=not args.no_water_blank_correction,
                water_blank_after_first_freeze=args.water_blank_after_first_freeze,
                water_blank_temperature_range_C=_json_object(
                    args.water_blank_temperature_range, "--water-blank-temperature-range"
                ),
            )
            complete = all(item["range_C"] is not None for item in proposal.inputs.values())
            _print_json(
                _response(
                    command,
                    complete=complete,
                    temperature_ranges_C=proposal.temperature_ranges_C if complete else None,
                    inputs=proposal.inputs,
                    settings=proposal.settings,
                    **(
                        {"table": _table_payload(proposal.observations)}
                        if proposal.observations is not None
                        else {}
                    ),
                    warnings=[]
                    if complete
                    else [
                        "Some inputs have no usable range. Review thresholds or selected inputs."
                    ],
                )
            )
            return 0
        if command == "table":
            value = store.load(args.input)
            if args.table is None:
                if args.curve or args.columns is not None or args.no_history:
                    raise ValueError("--curve, --columns and --no-history require --table")
                summary = table_summary(value)
                if json_mode:
                    _print_json(_response(command, tables=summary))
                else:
                    for name, info in summary.items():
                        print(f"{name}: {info['row_count']} rows ({info['type']})")
            else:
                table = select_table(value, args.table, args.curve)
                if json_mode:
                    _print_json(
                        _response(
                            command,
                            table_name=args.table,
                            curve_id=args.curve,
                            table=_table_payload(
                                table, columns=args.columns, include_history=not args.no_history
                            ),
                            warnings=table.warnings,
                        )
                    )
                else:
                    print(_table_frame(table, args.columns).to_string(index=False))
            return 0
        if store.exists(args.out):
            raise FileExistsError(f"Output already exists: {args.out}")
        if command == "save":
            result = store.load(args.input)
            store.save(result, args.out)
            if json_mode:
                _print_json(_response(command, output=store.output_name(args.out)))
            else:
                print(f"Saved result to {args.out}")
            return 0
        if command == "fractions":
            _save_step(_fraction_input(args, store), args, json_mode, store)
            return 0
        if command == "estimate":
            _save_step(_estimate_step(args, store), args, json_mode, store)
            return 0
        if command in ("convert", "finalize", "differentiate"):
            result = transform_step(
                store.load(args.input),
                command,
                basis=getattr(args, "output_basis", None),
                decrease_policy=getattr(args, "decrease_policy", None),
            )
            _save_step(result, args, json_mode, store)
            return 0
        if command == "export-csv":
            if str(args.out).startswith("@"):
                raise ValueError(
                    "CSV export requires a file path; use table --json for client data"
                )
            result = store.load(args.input)
            table = select_table(result, args.table, args.curve)
            if args.table == "cumulative":
                from .reporting import reportable_spectrum

                table = reportable_spectrum(table)
            table.to_dataframe().to_csv(args.out, index=False, mode="x")
            if json_mode:
                _print_json(
                    _response(
                        command,
                        output=store.output_name(args.out),
                        table=args.table,
                        curve_id=args.curve,
                    )
                )
            return 0
        experiment = _read_analysis_input(args, store)
        result = analyze_concentration(
            experiment,
            **_estimation_settings(args),
            output_basis=args.output_basis,
            differential=args.differential,
            decrease_policy=args.decrease_policy,
        )
        store.save(result, args.out)
        if json_mode:
            _print_json(
                _response(
                    command,
                    output=store.output_name(args.out),
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
