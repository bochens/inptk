#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.exists():
    sys.path.insert(0, str(SRC_ROOT))

import ufolaf  # noqa: E402


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

SAMPLE_METADATA_FIELDS = {field.name for field in fields(ufolaf.SampleMetadata)}


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    header_overrides = _header_overrides(args)
    _validate_output_options(args)
    kind = _resolve_input_kind(args.inputs, args.input_kind)

    if kind == "frozen-at-temp":
        if len(args.inputs) != 1:
            raise ValueError("frozen-at-temp mode accepts exactly one input table")
        fraction = _read_frozen_at_temp_fraction(args, header_overrides)
        source_for_metadata = fraction
    else:
        if args.cycle_policy == "preserve":
            raise ValueError(
                "CSU INPs_L export writes one analysis table. Use --cycle-policy single "
                "or --cycle-policy pooled, not preserve."
            )
        counts = _read_count_inputs(args)
        fraction = ufolaf.fraction_frozen(
            counts,
            step_C=args.step_C,
            method=args.method,
            temperature_tolerance_C=args.temperature_tolerance_C,
        )
        source_for_metadata = counts

    cumulative = _combine_fraction_tables(fraction, args)
    normalized = ufolaf.normalize_spec(cumulative)
    data = _csu_data_frame(
        normalized,
        cumulative,
        sample_id=args.sample_id,
        allow_multiple=bool(args.out_dir),
    )
    if args.out_dir:
        _write_csu_csvs_by_sample(
            data,
            source_for_metadata,
            normalized,
            args=args,
            overrides=header_overrides,
        )
    else:
        header = _csu_header(
            data,
            source_for_metadata,
            normalized,
            args=args,
            overrides=header_overrides,
        )
        _write_csu_csv(data, header, args.out, overwrite=args.overwrite)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert Icescopy/UFOLAF freezing data to the CSU DOES INP Mentor "
            "Program INPs_L CSV format."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help=(
            "Icescopy freeze_count_timeseries/canonical count CSV(s), or one "
            "frozen_at_temp CSV"
        ),
    )
    parser.add_argument("--out", help="Output INPs_L CSV path for one sample/group")
    parser.add_argument(
        "--out-dir",
        help="Directory for one INPs_L CSV per output sample/group",
    )
    parser.add_argument(
        "--input-kind",
        choices=("auto", "counts", "frozen-at-temp"),
        default="auto",
        help="Input type. auto treats degC-wide tables as frozen-at-temp.",
    )

    counts = parser.add_argument_group("count input options")
    counts.add_argument(
        "--format",
        choices=("auto", "long", "canonical", "icescopy", "icescopy_wide", "wide"),
        default="auto",
    )
    counts.add_argument("--columns", help="JSON object or JSON file mapping canonical columns")
    counts.add_argument("--metadata", help="Optional metadata CSV or JSON")
    counts.add_argument(
        "--include-sample",
        action="append",
        default=[],
        metavar="SAMPLE",
        help=(
            "Raw input sample id/name to keep before stitch/MLE. Use FILE::SAMPLE "
            "to qualify a sample from one input file. Repeat this option or use "
            "comma-separated values for multiple samples."
        ),
    )
    counts.add_argument(
        "--exclude-sample",
        action="append",
        default=[],
        metavar="SAMPLE",
        help=(
            "Raw input sample id/name to remove before stitch/MLE. Use FILE::SAMPLE "
            "to qualify a sample from one input file. Repeat this option or use "
            "comma-separated values for multiple samples."
        ),
    )
    counts.add_argument(
        "--cycle-policy",
        choices=("single", "pooled", "preserve"),
        default="single",
    )
    counts.add_argument("--cycle", help="Cycle to select when cycle-policy is single")
    counts.add_argument("--step-C", type=float, default=0.5)
    counts.add_argument("--method", choices=("max", "latest"), default="max")
    counts.add_argument("--temperature-tolerance-C", type=float, default=0.0)

    frozen = parser.add_argument_group("frozen_at_temp input options")
    frozen.add_argument(
        "--dilution-dict",
        help="CSV with sample,dilution columns for frozen-at-temp input",
    )
    frozen.add_argument(
        "--n-total-by-sample",
        help="JSON object/file or CSV with sample,n_total columns",
    )
    frozen.add_argument(
        "--infer-n-total-from-max",
        action="store_true",
        help="For frozen-at-temp input, set n_total to each sample column maximum.",
    )
    frozen.add_argument(
        "--well-volume-uL",
        type=float,
        help="Well volume used to convert frozen fractions to INP/mL.",
    )
    frozen.add_argument(
        "--group-id",
        help="Shared sample/group id for frozen-at-temp serial dilutions.",
    )

    spectrum = parser.add_argument_group("spectrum options")
    spectrum.add_argument("--combine", choices=("stitch", "mle", "none"), default="stitch")
    spectrum.add_argument("--sample-group-by", default="inferred")
    spectrum.add_argument("--enforce-monotone", action="store_true")
    spectrum.add_argument("--z", type=float, default=1.96)
    spectrum.add_argument(
        "--confidence-drop",
        type=float,
        default=1.920729410347062,
        help="Profile likelihood drop for MLE confidence intervals.",
    )
    spectrum.add_argument(
        "--sample-id",
        help="Output sample/group id to write when the pipeline returns multiple groups.",
    )

    header = parser.add_argument_group("CSU header overrides")
    header.add_argument("--header", action="append", default=[], metavar="KEY=VALUE")
    header.add_argument("--site")
    header.add_argument("--start-time")
    header.add_argument("--end-time")
    header.add_argument("--filter-color")
    header.add_argument("--sample-type")
    header.add_argument("--vol-air-filt")
    header.add_argument("--proportion-filter-used")
    header.add_argument("--vol-susp")
    header.add_argument("--treatment")
    header.add_argument("--notes")
    header.add_argument("--user")
    header.add_argument("--is-id", help="Value for the CSU header key IS")
    header.add_argument(
        "--allow-missing-header",
        action="store_true",
        help="Write blank values for missing CSU header fields instead of raising.",
    )

    parser.add_argument("--overwrite", action="store_true")
    return parser


def _validate_output_options(args: argparse.Namespace) -> None:
    if bool(args.out) == bool(args.out_dir):
        raise ValueError("Pass exactly one of --out or --out-dir")


def _resolve_input_kind(paths: Sequence[str], requested: str) -> str:
    if requested != "auto":
        return requested
    kinds = {_detect_input_kind(path) for path in paths}
    if len(kinds) != 1:
        raise ValueError("Do not mix counts and frozen-at-temp inputs in one command")
    return kinds.pop()


def _detect_input_kind(path: str) -> str:
    df = pd.read_csv(path, comment="#", nrows=5)
    if "degC" in df.columns and not {"sample_id", "temperature_C"}.issubset(df.columns):
        return "frozen-at-temp"
    return "counts"


def _read_table_source(path: str) -> pd.DataFrame | str:
    input_path = Path(path)
    if input_path.suffix.lower() in {".csv", ".txt"}:
        return str(input_path)
    return pd.read_csv(input_path)


def _read_count_inputs(args: argparse.Namespace) -> list[Any]:
    metadata = _load_metadata_source(args.metadata)
    columns = _load_columns(args.columns)
    tables: list[Any] = []
    for input_path in args.inputs:
        read = ufolaf.read_counts(
            _read_table_source(input_path),
            format=args.format,
            columns=columns,
            metadata=metadata,
            cycle_policy=args.cycle_policy,
            cycle=args.cycle,
        )
        tables.extend(_filter_sample_tables(_flatten_table_shape(read), args, input_path))
    if not tables:
        raise ValueError("No count tables were read")
    return tables


def _flatten_table_shape(value: Any) -> list[Any]:
    if isinstance(value, dict):
        raise ValueError("Nested count dictionaries are not supported for CSU export")
    if isinstance(value, (list, tuple)):
        tables: list[Any] = []
        for item in value:
            tables.extend(_flatten_table_shape(item))
        return tables
    return [value]


def _filter_sample_tables(
    tables: list[Any],
    args: argparse.Namespace,
    source_path: str,
) -> list[Any]:
    return [
        table
        for table in tables
        if _sample_allowed(_table_sample_labels(table), args, source_path)
    ]


def _sample_allowed(labels: set[str], args: argparse.Namespace, source_path: str) -> bool:
    include, include_by_source = _sample_filter_specs(args.include_sample)
    exclude, exclude_by_source = _sample_filter_specs(args.exclude_sample)
    source_labels = _source_path_labels(source_path)
    include_for_source = include | _samples_for_source(include_by_source, source_labels)
    exclude_for_source = exclude | _samples_for_source(exclude_by_source, source_labels)
    if (include or include_by_source) and labels.isdisjoint(include_for_source):
        return False
    return labels.isdisjoint(exclude_for_source)


def _sample_filter_specs(values: Sequence[str]) -> tuple[set[str], dict[str, set[str]]]:
    samples: set[str] = set()
    samples_by_source: dict[str, set[str]] = {}
    for value in values:
        for source, sample in _iter_sample_filter_parts(str(value)):
            if sample:
                if source:
                    samples_by_source.setdefault(source, set()).add(sample)
                else:
                    samples.add(sample)
    return samples, samples_by_source


def _iter_sample_filter_parts(value: str) -> list[tuple[str, str]]:
    if "::" in value:
        source, sample_values = value.split("::", 1)
        source = source.strip()
        return [
            (source, sample.strip())
            for sample in sample_values.split(",")
            if sample.strip()
        ]
    return [
        ("", sample.strip())
        for sample in value.split(",")
        if sample.strip()
    ]


def _source_path_labels(source_path: str) -> set[str]:
    path = Path(source_path)
    labels = {str(source_path), path.name}
    try:
        labels.add(str(path.expanduser().resolve()))
    except OSError:
        pass
    return {label for label in labels if label}


def _samples_for_source(
    samples_by_source: dict[str, set[str]],
    source_labels: set[str],
) -> set[str]:
    samples: set[str] = set()
    for source, source_samples in samples_by_source.items():
        if source in source_labels:
            samples.update(source_samples)
    return samples


def _table_sample_labels(table: Any) -> set[str]:
    labels: set[str] = set()
    if hasattr(table, "to_dataframe"):
        df = table.to_dataframe()
        if "sample_id" in df:
            labels.update(df["sample_id"].dropna().astype(str))
    metadata = getattr(table, "metadata", None)
    if isinstance(metadata, ufolaf.SampleMetadata):
        labels.update(_sample_labels("", metadata))
    elif isinstance(metadata, dict):
        for sample_id, value in metadata.items():
            if isinstance(value, ufolaf.SampleMetadata):
                labels.update(_sample_labels(str(sample_id), value))
    return {label for label in labels if label}


def _sample_labels(sample_id: str, metadata: Any) -> set[str]:
    labels = {str(sample_id).strip()} if sample_id else set()
    for attribute in ("sample_id", "sample_name", "sample_long_name"):
        value = getattr(metadata, attribute, "")
        if not _missing(value):
            labels.add(str(value).strip())
    return {label for label in labels if label}


def _read_frozen_at_temp_fraction(
    args: argparse.Namespace,
    header_overrides: dict[str, str],
) -> list[Any]:
    if not args.dilution_dict:
        raise ValueError("--dilution-dict is required when --input-kind=frozen-at-temp")

    source = pd.read_csv(args.inputs[0])
    dilution_by_sample = _load_dilution_dict(args.dilution_dict)
    n_total_by_sample = _load_n_total_by_sample(args.n_total_by_sample)
    if n_total_by_sample is None:
        if not args.infer_n_total_from_max:
            raise ValueError(
                "frozen-at-temp input needs --n-total-by-sample or --infer-n-total-from-max"
            )
        n_total_by_sample = _infer_n_total_by_sample(source)

    parsed = ufolaf.parse_olaf_frozen_at_temp(source, n_total_by_sample=n_total_by_sample)
    metadata_by_sample = _load_sample_metadata_mapping(args.metadata)
    group_id = args.group_id or header_overrides.get("site")
    output: list[Any] = []
    skipped: list[str] = []

    for table in parsed:
        sample_id = str(table.sample_id[0])
        dilution = dilution_by_sample.get(sample_id)
        if dilution is None:
            raise KeyError(f"Missing dilution for frozen-at-temp sample {sample_id!r}")
        if not math.isfinite(dilution):
            skipped.append(sample_id)
            continue
        metadata = _metadata_for_frozen_at_temp_sample(
            metadata_by_sample.get(sample_id),
            sample_id=sample_id,
            dilution=dilution,
            group_id=group_id,
            args=args,
            header_overrides=header_overrides,
        )
        if not _sample_allowed(_sample_labels(sample_id, metadata), args, args.inputs[0]):
            continue
        output.append(
            ufolaf.TemperatureFrozenFractionTable.from_dataframe(
                table.to_dataframe(),
                metadata=metadata,
            )
        )

    if skipped:
        print(
            "Skipped non-finite dilution sample(s): " + ", ".join(skipped),
            file=sys.stderr,
        )
    if not output:
        raise ValueError("No finite-dilution frozen-at-temp samples were available")
    return output


def _metadata_for_frozen_at_temp_sample(
    base: Any,
    *,
    sample_id: str,
    dilution: float,
    group_id: str | None,
    args: argparse.Namespace,
    header_overrides: dict[str, str],
) -> Any:
    payload = asdict(base) if isinstance(base, ufolaf.SampleMetadata) else {}
    raw_sample_metadata = dict(payload.get("raw_sample_metadata") or {})
    raw_sample_metadata.update(header_overrides)

    payload["sample_id"] = sample_id
    payload["dilution"] = _first_present(payload.get("dilution"), dilution)
    payload["well_volume_uL"] = _first_present(payload.get("well_volume_uL"), args.well_volume_uL)
    payload["sample_type"] = _first_present(
        payload.get("sample_type"),
        header_overrides.get("sample_type"),
    )
    payload["air_volume_L"] = _first_present(
        payload.get("air_volume_L"),
        _float_or_none(header_overrides.get("vol_air_filt")),
    )
    payload["filter_fraction_used"] = _first_present(
        payload.get("filter_fraction_used"),
        _float_or_none(header_overrides.get("proportion_filter_used")),
    )
    payload["suspension_volume_mL"] = _first_present(
        payload.get("suspension_volume_mL"),
        _float_or_none(header_overrides.get("vol_susp")),
    )
    payload["collection_start"] = _first_present(
        payload.get("collection_start"),
        header_overrides.get("start_time"),
    )
    payload["collection_end"] = _first_present(
        payload.get("collection_end"),
        header_overrides.get("end_time"),
    )
    payload["user_name"] = _first_present(payload.get("user_name"), header_overrides.get("user"))
    if group_id and _missing(payload.get("sample_name")):
        payload["sample_name"] = f"{group_id}_{dilution:g}"
    if group_id and _missing(payload.get("sample_long_name")):
        payload["sample_long_name"] = f"{group_id}_{dilution:g}"
    payload["raw_sample_metadata"] = raw_sample_metadata

    if _missing(payload.get("well_volume_uL")):
        raise ValueError(f"Missing well volume for frozen-at-temp sample {sample_id!r}")
    return ufolaf.SampleMetadata(**_sample_metadata_kwargs(payload))


def _combine_fraction_tables(fraction: Any, args: argparse.Namespace) -> Any:
    sample_group_by = _sample_group_by(args.sample_group_by)
    if args.combine == "stitch":
        return ufolaf.cumulative_spec_stitch(
            fraction,
            sample_group_by=sample_group_by,
            enforce_monotone=args.enforce_monotone,
            z=args.z,
        )
    if args.combine == "mle":
        return ufolaf.cumulative_spec_mle(
            fraction,
            sample_group_by=sample_group_by,
            enforce_monotone=args.enforce_monotone,
            confidence_drop=args.confidence_drop,
        )
    if args.combine == "none":
        return ufolaf.cumulative_spec(fraction, z=args.z)
    raise ValueError("--combine must be stitch, mle, or none")


def _csu_data_frame(
    normalized: Any,
    cumulative: Any,
    *,
    sample_id: str | None,
    allow_multiple: bool,
) -> pd.DataFrame:
    normalized_df = ufolaf.tables_to_dataframe(normalized).reset_index(drop=True)
    cumulative_df = ufolaf.tables_to_dataframe(cumulative).reset_index(drop=True)
    if len(normalized_df) != len(cumulative_df):
        raise ValueError("Normalized and cumulative spectra have different row counts")
    if normalized_df.empty:
        raise ValueError("No output rows were produced")
    if "value_unit" not in normalized_df:
        raise ValueError("Normalized spectrum is missing value_unit")
    units = set(normalized_df["value_unit"].dropna().astype(str).unique())
    if units != {"INP_per_L_air"}:
        raise ValueError(
            "CSU INPs_L output requires air-normalized values. "
            f"Observed value_unit(s): {', '.join(sorted(units)) or 'none'}"
        )

    output = pd.DataFrame(
        {
            "_sample_id": normalized_df["sample_id"].astype(str),
            "degC": normalized_df["temperature_C"].astype(float),
            "dilution": cumulative_df["dilution_fold"].astype(float)
            if "dilution_fold" in cumulative_df
            else float("nan"),
            "INPS_L": normalized_df["value"].astype(float),
            "lower_CI": normalized_df["lower_ci"].astype(float),
            "upper_CI": normalized_df["upper_ci"].astype(float),
        }
    )
    sample_ids = sorted(output["_sample_id"].dropna().unique())
    if sample_id is not None:
        output = output[output["_sample_id"] == sample_id].copy()
        if output.empty:
            raise ValueError(f"No rows found for requested --sample-id {sample_id!r}")
    elif len(sample_ids) > 1 and not allow_multiple:
        raise ValueError(
            "Pipeline produced multiple output sample/group ids. Pass --sample-id "
            "or --out-dir. Available ids: " + ", ".join(sample_ids)
        )
    output = output[output["INPS_L"].notna()].copy()
    if output.empty:
        raise ValueError("No finite INPs_L rows were produced")
    output = output.sort_values(["_sample_id", "degC"], ascending=[True, False])
    return output.reset_index(drop=True)


def _csu_header(
    data: pd.DataFrame,
    source: Any,
    normalized: Any,
    *,
    args: argparse.Namespace,
    overrides: dict[str, str],
) -> dict[str, str]:
    output_sample_id = str(data["_sample_id"].iloc[0])
    source_metadata = _collect_metadata(source)
    final_metadata = _collect_metadata(normalized)
    rows = _matching_source_metadata(output_sample_id, source_metadata, args.sample_group_by)
    if not rows and output_sample_id in final_metadata:
        rows = [final_metadata[output_sample_id]]

    header = {
        "site": _first_present(overrides.get("site"), output_sample_id),
        "start_time": _metadata_or_override(
            overrides,
            "start_time",
            rows,
            "collection_start",
            "start_time",
        ),
        "end_time": _metadata_or_override(
            overrides,
            "end_time",
            rows,
            "collection_end",
            "end_time",
        ),
        "filter_color": _metadata_or_override(
            overrides,
            "filter_color",
            rows,
            None,
            "filter_color",
            "filter color",
        ),
        "sample_type": _metadata_or_override(
            overrides,
            "sample_type",
            rows,
            "sample_type",
            "sample_type",
        ),
        "vol_air_filt": _metadata_or_override(
            overrides,
            "vol_air_filt",
            rows,
            "air_volume_L",
            "vol_air_filt",
            "air_volume_L",
        ),
        "proportion_filter_used": _metadata_or_override(
            overrides,
            "proportion_filter_used",
            rows,
            "filter_fraction_used",
            "proportion_filter_used",
            "filter_fraction_used",
        ),
        "vol_susp": _metadata_or_override(
            overrides,
            "vol_susp",
            rows,
            "suspension_volume_mL",
            "vol_susp",
            "suspension_volume_mL",
        ),
        "treatment": _metadata_or_override(
            overrides,
            "treatment",
            rows,
            None,
            "treatment",
            "sample_treatment",
        ),
        "notes": _metadata_or_override(
            overrides,
            "notes",
            rows,
            None,
            "notes",
            "note",
            "sample_note",
        ),
        "user": _metadata_or_override(overrides, "user", rows, "user_name", "user", "user_name"),
        "IS": _metadata_or_override(
            overrides,
            "IS",
            rows,
            None,
            "IS",
            "is",
            "instrument",
            "instrument_id",
        ),
    }

    missing = [key for key in HEADER_ORDER if _missing(header.get(key))]
    if missing and not args.allow_missing_header:
        raise ValueError(
            "Missing CSU header field(s): "
            + ", ".join(missing)
            + ". Provide them with --header KEY=VALUE or the matching named option."
        )
    return {key: _format_header_value(header.get(key, "")) for key in HEADER_ORDER}


def _write_csu_csv(
    data: pd.DataFrame,
    header: dict[str, str],
    output_path: str,
    *,
    overwrite: bool,
) -> None:
    path = Path(output_path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    export = data.loc[:, ["degC", "dilution", "INPS_L", "lower_CI", "upper_CI"]]
    with path.open("w", encoding="utf-8", newline="") as handle:
        for key in HEADER_ORDER:
            handle.write(f"{key} = {header[key]}\n")
        handle.write("\n")
        export.to_csv(handle, index=False)


def _write_csu_csvs_by_sample(
    data: pd.DataFrame,
    source: Any,
    normalized: Any,
    *,
    args: argparse.Namespace,
    overrides: dict[str, str],
) -> None:
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for sample_id, sample_data in data.groupby("_sample_id", sort=False):
        header = _csu_header(
            sample_data.reset_index(drop=True),
            source,
            normalized,
            args=args,
            overrides=overrides,
        )
        path = output_dir / f"{_safe_filename(str(sample_id))}_INPs_L.csv"
        _write_csu_csv(sample_data, header, str(path), overwrite=args.overwrite)


def _safe_filename(value: str) -> str:
    safe = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in value
    )
    return safe.strip("._") or "sample"


def _header_overrides(args: argparse.Namespace) -> dict[str, str]:
    overrides = _parse_key_value_overrides(args.header)
    named = {
        "site": args.site,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "filter_color": args.filter_color,
        "sample_type": args.sample_type,
        "vol_air_filt": args.vol_air_filt,
        "proportion_filter_used": args.proportion_filter_used,
        "vol_susp": args.vol_susp,
        "treatment": args.treatment,
        "notes": args.notes,
        "user": args.user,
        "IS": args.is_id,
    }
    for key, value in named.items():
        if value is not None:
            overrides[key] = str(value)
    if args.group_id and "site" not in overrides:
        overrides["site"] = str(args.group_id)
    return overrides


def _parse_key_value_overrides(values: list[str]) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"--header value must be KEY=VALUE, got {value!r}")
        key, raw = value.split("=", 1)
        key = key.strip()
        if key not in HEADER_ORDER:
            raise ValueError(f"Unknown CSU header key {key!r}")
        overrides[key] = raw.strip()
    return overrides


def _load_columns(value: str | None) -> dict[str, str] | None:
    if not value:
        return None
    path = Path(value)
    payload = path.read_text(encoding="utf-8") if path.exists() else value
    loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError("--columns must be a JSON object or a path to one")
    return {str(key): str(column) for key, column in loaded.items()}


def _load_metadata_source(value: str | None) -> Any:
    if not value:
        return None
    path = Path(value)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    raise ValueError("--metadata must be a CSV or JSON file")


def _load_sample_metadata_mapping(value: str | None) -> dict[str, Any]:
    source = _load_metadata_source(value)
    if source is None:
        return {}
    if isinstance(source, pd.DataFrame):
        rows = source.to_dict(orient="records")
    elif isinstance(source, list):
        rows = source
    elif isinstance(source, dict):
        if "sample_id" in source:
            rows = [source]
        else:
            rows = [dict(record, sample_id=sample_id) for sample_id, record in source.items()]
    else:
        raise TypeError("--metadata must contain a metadata record or records")

    metadata: dict[str, Any] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError("metadata rows must be objects")
        sample_id = str(row.get("sample_id", "")).strip()
        if not sample_id:
            raise ValueError("metadata rows must include sample_id")
        metadata[sample_id] = ufolaf.SampleMetadata(**_sample_metadata_kwargs(row))
    return metadata


def _sample_metadata_kwargs(record: dict[str, Any]) -> dict[str, Any]:
    raw_preamble = (
        record.get("raw_preamble")
        if isinstance(record.get("raw_preamble"), dict)
        else {}
    )
    raw_sample_metadata = (
        record.get("raw_sample_metadata")
        if isinstance(record.get("raw_sample_metadata"), dict)
        else {}
    )
    raw_sample_metadata = {
        **{str(key): _format_header_value(value) for key, value in record.items()},
        **{str(key): _format_header_value(value) for key, value in raw_sample_metadata.items()},
    }
    kwargs = {
        str(key): value
        for key, value in record.items()
        if str(key) in SAMPLE_METADATA_FIELDS
        and str(key) not in {"raw_preamble", "raw_sample_metadata"}
        and not _missing(value)
    }
    kwargs["raw_preamble"] = raw_preamble
    kwargs["raw_sample_metadata"] = raw_sample_metadata
    return kwargs


def _load_dilution_dict(path: str) -> dict[str, float]:
    frame = pd.read_csv(path)
    if not {"sample", "dilution"}.issubset(frame.columns):
        raise ValueError("--dilution-dict must include sample and dilution columns")
    return {
        str(row["sample"]): float(row["dilution"])
        for _, row in frame.iterrows()
        if not _missing(row["sample"])
    }


def _load_n_total_by_sample(value: str | None) -> dict[str, float] | None:
    if not value:
        return None
    path = Path(value)
    if path.exists() and path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
        if not {"sample", "n_total"}.issubset(frame.columns):
            raise ValueError("--n-total-by-sample CSV must include sample and n_total columns")
        return {str(row["sample"]): float(row["n_total"]) for _, row in frame.iterrows()}
    payload = path.read_text(encoding="utf-8") if path.exists() else value
    loaded = json.loads(payload)
    if not isinstance(loaded, dict):
        raise ValueError("--n-total-by-sample must be a JSON object, JSON file, or CSV")
    return {str(key): float(raw) for key, raw in loaded.items()}


def _infer_n_total_by_sample(frame: pd.DataFrame) -> dict[str, float]:
    if "degC" not in frame:
        raise ValueError("frozen-at-temp input must include degC")
    inferred: dict[str, float] = {}
    for column in frame.columns:
        if column == "degC":
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        maximum = values.max(skipna=True)
        if pd.isna(maximum):
            raise ValueError(f"Cannot infer n_total for empty sample column {column!r}")
        inferred[str(column)] = float(maximum)
    return inferred


def _sample_group_by(value: str | None) -> str | None:
    if value is None or value == "inferred":
        return None
    if value not in {"sample_id", "sample_name", "sample_long_name"}:
        raise ValueError(
            "--sample-group-by must be inferred, sample_id, sample_name, or sample_long_name"
        )
    return value


def _collect_metadata(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        collected: dict[str, Any] = {}
        for item in value.values():
            collected.update(_collect_metadata(item))
        return collected
    if isinstance(value, (list, tuple)):
        collected = {}
        for item in value:
            collected.update(_collect_metadata(item))
        return collected
    metadata = getattr(value, "metadata", None)
    if isinstance(metadata, ufolaf.SampleMetadata):
        key = metadata.sample_id or _single_table_sample_id(value)
        return {key: metadata} if key else {}
    if isinstance(metadata, dict):
        return {
            str(key): item
            for key, item in metadata.items()
            if isinstance(item, ufolaf.SampleMetadata)
        }
    return {}


def _single_table_sample_id(table: Any) -> str:
    if not hasattr(table, "to_dataframe"):
        return ""
    df = table.to_dataframe()
    if "sample_id" not in df:
        return ""
    ids = df["sample_id"].dropna().astype(str).unique()
    return str(ids[0]) if len(ids) == 1 else ""


def _matching_source_metadata(
    output_sample_id: str,
    source_metadata: dict[str, Any],
    sample_group_by: str | None,
) -> list[Any]:
    if not source_metadata:
        return []
    if output_sample_id in source_metadata:
        return [source_metadata[output_sample_id]]
    rows = []
    for sample_id, metadata in source_metadata.items():
        group_id = _source_group_id(str(sample_id), metadata, sample_group_by)
        if group_id == output_sample_id:
            rows.append(metadata)
    return rows


def _source_group_id(sample_id: str, metadata: Any, sample_group_by: str | None) -> str:
    if sample_group_by in (None, "inferred"):
        label = metadata.sample_long_name or metadata.sample_name
        return _strip_trailing_numeric_token(label) if label else sample_id
    if sample_group_by == "sample_id":
        return sample_id
    return str(getattr(metadata, sample_group_by, "")).strip()


def _strip_trailing_numeric_token(value: str) -> str:
    parts = str(value).rsplit("_", 1)
    if len(parts) != 2:
        return str(value)
    try:
        float(parts[1])
    except ValueError:
        return str(value)
    return parts[0]


def _metadata_or_override(
    overrides: dict[str, str],
    key: str,
    rows: list[Any],
    attr: str | None,
    *raw_keys: str,
) -> Any:
    if key in overrides and not _missing(overrides[key]):
        return overrides[key]
    return _metadata_value(rows, attr, *raw_keys)


def _metadata_value(rows: list[Any], attr: str | None, *raw_keys: str) -> Any:
    values = []
    for metadata in rows:
        value = getattr(metadata, attr, None) if attr else None
        if _missing(value):
            value = _first_raw_metadata_value(metadata, raw_keys)
        if not _missing(value):
            values.append(value)
    return _shared_value(values)


def _first_raw_metadata_value(metadata: Any, keys: Sequence[str]) -> Any:
    key_variants = {key: None for key in keys}
    key_variants.update({key.replace("_", " "): None for key in keys})
    key_variants.update({key.replace(" ", "_"): None for key in keys})
    normalized = {key.casefold(): key for key in key_variants}
    for mapping_name in ("raw_sample_metadata", "raw_preamble"):
        mapping = getattr(metadata, mapping_name, {}) or {}
        lookup = {str(key).casefold(): value for key, value in mapping.items()}
        for folded in normalized:
            if folded in lookup and not _missing(lookup[folded]):
                return lookup[folded]
    return None


def _shared_value(values: list[Any]) -> Any:
    if not values:
        return None
    first = values[0]
    for value in values[1:]:
        if not _values_match(first, value):
            raise ValueError(f"Conflicting metadata values: {first!r} vs {value!r}")
    return first


def _values_match(left: Any, right: Any) -> bool:
    left_number = _float_or_none(left)
    right_number = _float_or_none(right)
    if left_number is not None or right_number is not None:
        return left_number is not None and right_number is not None and math.isclose(
            left_number,
            right_number,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    return str(left).strip() == str(right).strip()


def _first_present(*values: Any) -> Any:
    for value in values:
        if not _missing(value):
            return value
    return None


def _float_or_none(value: Any) -> float | None:
    if _missing(value):
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, bool) and missing:
        return True
    text = str(value).strip()
    return text == "" or text.casefold() == "nan"


def _format_header_value(value: Any) -> str:
    return "" if _missing(value) else str(value).strip()


if __name__ == "__main__":
    raise SystemExit(main())
