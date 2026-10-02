"""Command-line access to the same workflow used by Python and external apps."""

import argparse
import json
from pathlib import Path

import pandas as pd

from . import __version__, analyze_concentration, load, read_counts, read_icescopy
from .experiment import Experiment
from .tables import CountsTable


def build_parser():
    parser = argparse.ArgumentParser(
        prog="inptk", description="INP-toolkit droplet-freezing analysis"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser(
        "analyze", help="Combine dilutions into a spectrum for each sample and cycle"
    )
    analyze.add_argument("input")
    analyze.add_argument("--metadata", help="Native metadata or Icescopy metadata overrides")
    analyze.add_argument("--format", choices=("native", "icescopy", "saved"), default="native")
    analyze.add_argument(
        "--sample-map", help="JSON file mapping Icescopy measurement names to parent samples"
    )
    analyze.add_argument(
        "--water-blank-map",
        help="JSON file mapping native sample measurements to lists of water-blank measurements",
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
    analyze.add_argument("--step-C", type=float, default=0.5)
    analyze.add_argument("--temperature-method", choices=("max", "latest"), default="latest")
    analyze.add_argument("--temperature-tolerance-C", type=float)
    analyze.add_argument("--z", type=float, default=1.96)
    analyze.add_argument("--differential", action="store_true")
    analyze.add_argument(
        "--decrease-policy",
        choices=("stop_at_decrease", "skip_decreases"),
        default="stop_at_decrease",
        help="Select final cumulative points without changing calculated values",
    )
    export = commands.add_parser(
        "export-csv", help="Export final concentration rows from a saved analysis"
    )
    export.add_argument("input")
    export.add_argument("--out", required=True)
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


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "export-csv":
            load(args.input).export_csv(args.out)
            return 0
        temperature_ranges = None
        if args.temperature_ranges is not None:
            payload = args.temperature_ranges
            if not payload.lstrip().startswith(("{", "[")):
                payload = Path(payload).read_text(encoding="utf-8")
            temperature_ranges = json.loads(payload)
            if not isinstance(temperature_ranges, dict):
                raise ValueError("--temperature-ranges must contain a JSON object")
        if args.format == "native":
            if args.sample_map:
                raise ValueError("--sample-map applies only to Icescopy input")
            if not args.metadata:
                raise ValueError("Native input requires --metadata")
            water_blank_map = (
                json.loads(Path(args.water_blank_map).read_text(encoding="utf-8"))
                if args.water_blank_map
                else None
            )
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
            mapping = json.loads(Path(args.sample_map).read_text()) if args.sample_map else None
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
            if hasattr(experiment, "experiment"):
                experiment = experiment.experiment
        experiment = _select_experiment(experiment, args.sample, args.cycle)
        result = analyze_concentration(
            experiment,
            method=args.method,
            temperature_ranges_C=temperature_ranges,
            output_basis=args.output_basis,
            step_C=args.step_C,
            temperature_method=args.temperature_method,
            temperature_tolerance_C=args.temperature_tolerance_C,
            z=args.z,
            differential=args.differential,
            water_blank_correction=not args.no_water_blank_correction,
            decrease_policy=args.decrease_policy,
        )
        result.save(args.out)
        for warning in result.warnings:
            print(f"Warning: {warning}")
        print(f"Saved {len(result.final)} concentration rows to {args.out}")
    except (ValueError, TypeError, KeyError, OSError, AttributeError) as error:
        parser.exit(1, f"inptk: {error}\n")
    return 0
