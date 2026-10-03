"""CLI step orchestration; all calculations use the public Python functions."""

from __future__ import annotations

import pandas as pd

from .experiment import AnalysisResult, Experiment, ProcessingResult
from .processing import differentiate_spectrum, frozen_fraction
from .tables import CountsTable, CumulativeSpectrumTable
from .workflows import _final_candidates, convert_concentration


def saved_tables(value):
    """Expose quantities consistently for whole analyses and intermediate steps."""
    if isinstance(value, Experiment):
        return {"counts": value.counts}
    if isinstance(value, ProcessingResult):
        tables = {"counts": value.experiment.counts} if value.experiment else {}
        return {**tables, **value.tables}
    if not isinstance(value, AnalysisResult):
        raise TypeError("Input must be a saved experiment, processing step or analysis")
    tables = {"counts": value.counts, "frozen_fraction": value.frozen_fraction}
    for name in ("cumulative", "excluded", "differential"):
        parts = []
        histories = {}
        table_type = None
        for curve_id, curve in value.curves.items():
            table = getattr(curve, name)
            if table is not None:
                parts.append(table.to_dataframe().assign(curve_id=curve_id))
                histories[curve_id] = table.history
                table_type = type(table)
        if parts:
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
            tables[name] = table_type(pd.concat(parts, ignore_index=True), history=history)
    return tables


def select_table(value, name, curve=None):
    tables = saved_tables(value)
    if name not in tables:
        raise ValueError(f"Table {name!r} is unavailable; available tables: {list(tables)}")
    table = tables[name]
    if curve is not None:
        if "curve_id" not in table.columns:
            raise ValueError(f"Table {name!r} has no named curves; omit --curve")
        names = {
            str(item)
            for t in tables.values()
            if "curve_id" in t.columns
            for item in t.to_dataframe().curve_id.unique()
        }
        if curve not in names:
            raise ValueError(f"Unknown curve {curve!r}; available curves: {sorted(names)}")
        table = table.select(curve_id=curve)
    return table


def counts_from_step(value):
    tables = saved_tables(value)
    for name in ("counts", "frozen_fraction"):
        if name in tables:
            source = tables[name]
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
    cumulative = select_table(value, "cumulative")
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
    return {
        name: {
            "type": type(table).__name__,
            "row_count": len(table),
            "columns": list(table.columns),
        }
        for name, table in saved_tables(value).items()
    }
