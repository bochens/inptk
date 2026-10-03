"""CLI step orchestration; all calculations use the public Python functions."""

from __future__ import annotations

import pandas as pd

from .experiment import AnalysisResult, Experiment, ProcessingResult
from .processing import differentiate_spectrum, frozen_fraction
from .tables import CountsTable, CumulativeSpectrumTable
from .workflows import _final_candidates, convert_concentration


def _table_groups(value):
    """Describe available tables using references, without collecting their rows."""
    if isinstance(value, Experiment):
        return {"counts": [(None, value.counts)]}
    if isinstance(value, ProcessingResult):
        tables = {"counts": value.experiment.counts} if value.experiment else {}
        tables.update(value.tables)
        return {name: [(None, table)] for name, table in tables.items()}
    if not isinstance(value, AnalysisResult):
        raise TypeError("Input must be a saved experiment, processing step or analysis")
    groups = {"counts": [(None, value.counts)], "frozen_fraction": [(None, value.frozen_fraction)]}
    for name in ("cumulative", "excluded", "differential"):
        parts = [
            (curve_id, table)
            for curve_id, curve in value.curves.items()
            if (table := getattr(curve, name)) is not None
        ]
        if parts:
            groups[name] = parts
    return groups


def _named_table(curve_id, table):
    if curve_id is None or "curve_id" in table.columns:
        return table
    return type(table)(table.to_dataframe().assign(curve_id=curve_id), history=table.history)


def _collect_tables(parts):
    if len(parts) == 1:
        return _named_table(*parts[0])
    histories = {curve_id: table.history for curve_id, table in parts}
    history = next(iter(histories.values()))
    if any(other != history for other in histories.values()):
        history = [
            {
                "operation": "collect_curves",
                "curve_histories": histories,
                "warnings": list(
                    dict.fromkeys(
                        warning
                        for steps in histories.values()
                        for step in steps
                        for warning in step.get("warnings", [])
                    )
                ),
            }
        ]
    frames = [table.to_dataframe().assign(curve_id=curve_id) for curve_id, table in parts]
    return type(parts[0][1])(pd.concat(frames, ignore_index=True), history=history)


def saved_tables(value):
    """Collect all quantities only for operations that need all of them."""
    return {name: _collect_tables(parts) for name, parts in _table_groups(value).items()}


def select_table(value, name, curve=None):
    groups = _table_groups(value)
    if name not in groups:
        raise ValueError(f"Table {name!r} is unavailable; available tables: {list(groups)}")
    parts = groups[name]
    if curve is None:
        return _collect_tables(parts)
    if not any(label is not None or "curve_id" in table.columns for label, table in parts):
        raise ValueError(f"Table {name!r} has no named curves; omit --curve")
    if isinstance(value, AnalysisResult):
        if curve not in value.curves:
            raise ValueError(f"Unknown curve {curve!r}; available curves: {sorted(value.curves)}")
        for label, table in parts:
            if label == curve:
                return _named_table(label, table)
        # The curve exists but has no table of this quantity, e.g. differential.
        return _collect_tables(parts).select(curve_id=curve)
    table = _collect_tables(parts)
    names = {
        str(item)
        for values in groups.values()
        for _, t in values
        if "curve_id" in t.columns
        for item in t.to_dataframe().curve_id.unique()
    }
    if curve not in names:
        raise ValueError(f"Unknown curve {curve!r}; available curves: {sorted(names)}")
    return table.select(curve_id=curve)


def counts_from_step(value):
    groups = _table_groups(value)
    for name in ("counts", "frozen_fraction"):
        if name in groups:
            source = select_table(value, name)
            if type(source) is CountsTable and "fraction_frozen" not in source.columns:
                return source
            return CountsTable(
                source.to_dataframe().drop(columns="fraction_frozen", errors="ignore"),
                history=source.history,
            )
    raise ValueError("This step has no original counts; provide an experiment or fraction step")


def complete_step_metadata(value, metadata, water_blank_map):
    """Complete physical metadata for fractions saved before it was available."""
    from .readers import _clean, _frame, read_counts

    counts = counts_from_step(value)
    imported = next((step for step in counts.history if "measurement_metadata" in step), {})
    records = {
        item["measurement_id"]: dict(item) for item in imported.get("measurement_metadata", [])
    }
    for item in _frame(metadata).to_dict("records"):
        name = str(item.get("measurement_id", item.get("sample_id", "")))
        if name not in records:
            raise ValueError(f"Metadata names unknown input {name!r}")
        records[name].update({key: val for key, val in item.items() if _clean(val) is not None})
    frame = counts.to_dataframe()
    for name in imported.get("provisional_sample_assignments", []):
        frame.loc[frame.measurement_id.eq(name), "sample_id"] = records[name]["sample_id"]
    return read_counts(frame, metadata=list(records.values()), water_blank_map=water_blank_map)


def fraction_step(counts, *, experiment=None):
    return ProcessingResult({"frozen_fraction": frozen_fraction(counts)}, experiment=experiment)


def transform_step(value, operation, *, basis=None, decrease_policy=None):
    """Convert, select final points, or differentiate without repeating estimation."""
    tables = saved_tables(value)
    if "cumulative" not in tables:
        raise ValueError("This step requires a cumulative concentration table")
    cumulative = tables["cumulative"]
    if not isinstance(cumulative, CumulativeSpectrumTable):
        raise TypeError("This step requires a cumulative concentration table")
    experiment = (
        value.experiment if isinstance(value, (AnalysisResult, ProcessingResult)) else value
    )
    if operation == "convert":
        if experiment is None:
            raise ValueError("Unit conversion requires saved sample metadata")
        for name in ("cumulative", "excluded"):
            if name in tables:
                tables[name] = convert_concentration(tables[name], experiment.samples, basis=basis)
        # Differential concentration has its own units and is not converted.
    elif operation == "finalize":
        if "excluded" in tables:
            cumulative = type(cumulative)(
                pd.concat(
                    [cumulative.to_dataframe(), tables["excluded"].to_dataframe()],
                    ignore_index=True,
                ),
                history=cumulative.history,
            )
        candidates = _final_candidates(cumulative, decrease_policy=decrease_policy)
        tables["cumulative"] = candidates.select(used_in_final=True)
        tables["excluded"] = candidates.select(used_in_final=False)
        tables.pop("differential", None)
    elif operation == "differentiate":
        if "measurement_id" not in cumulative.columns:
            raise ValueError(
                "Differentiation requires individual spectra: run estimate --individual first"
            )
        tables["differential"] = differentiate_spectrum(cumulative)
    else:
        raise ValueError(f"Unknown processing step {operation!r}")
    # Raw counts are already retained by the experiment; don't serialize a copy.
    if experiment is not None:
        tables.pop("counts", None)
    return ProcessingResult(tables, experiment=experiment)


def table_summary(value):
    """Summarize table references without copying rows or processing histories."""
    summary = {}
    for name, parts in _table_groups(value).items():
        columns = list(
            dict.fromkeys(
                column
                for label, table in parts
                for column in (*table.columns, *(("curve_id",) if label is not None else ()))
            )
        )
        summary[name] = {
            "type": type(parts[0][1]).__name__,
            "row_count": sum(len(table) for _, table in parts),
            "columns": columns,
        }
    return summary
