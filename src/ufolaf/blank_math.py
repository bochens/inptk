from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .models import (
    CumulativeNucleusSpectrumTable,
    NormalizedInpSpectrumTable,
    SampleMetadata,
    processing_metadata_for,
)
from .qc import qc_blank_corrected_spectrum


FilterBlankSpectrum = CumulativeNucleusSpectrumTable | NormalizedInpSpectrumTable
SpectrumSequence = list[FilterBlankSpectrum] | tuple[FilterBlankSpectrum, ...]
SpectrumMapping = dict[str, Any]


def _is_spectrum_sequence(value: Any) -> bool:
    return isinstance(value, (list, tuple))


def _is_spectrum_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _require_spectrum_sequence(value: Any, name: str) -> SpectrumSequence:
    if not _is_spectrum_sequence(value):
        raise TypeError(
            f"{name} must be a CumulativeNucleusSpectrumTable, "
            "NormalizedInpSpectrumTable, or a list of them"
        )
    for index, spectrum in enumerate(value):
        if not isinstance(
            spectrum,
            (CumulativeNucleusSpectrumTable, NormalizedInpSpectrumTable),
        ):
            raise TypeError(
                f"{name}[{index}] must be a CumulativeNucleusSpectrumTable "
                "or NormalizedInpSpectrumTable"
            )
    return value


def propagate_uncertainty_rss(
    sample_error: np.ndarray,
    blank_error: np.ndarray,
) -> np.ndarray:
    """Propagate independent sample and blank uncertainties by root-sum-of-squares."""

    sample = np.asarray(sample_error, dtype=float)
    blank = np.asarray(blank_error, dtype=float)
    sample, blank = np.broadcast_arrays(sample, blank)
    return np.sqrt(sample**2 + blank**2)


def average_blank_spectra(
    blank_spectra: list[FilterBlankSpectrum],
    *,
    value_method: Literal["mean", "median"] = "mean",
    require_positive: bool = True,
) -> FilterBlankSpectrum:
    """Average replicate filter blank spectra by exact temperature."""

    if not blank_spectra:
        raise ValueError("At least one blank spectrum is required")
    if value_method not in ("mean", "median"):
        raise ValueError("value_method must be 'mean' or 'median'")
    if all(isinstance(spectrum, CumulativeNucleusSpectrumTable) for spectrum in blank_spectra):
        return _average_cumulative_blank_spectra(
            blank_spectra,
            value_method=value_method,
            require_positive=require_positive,
        )
    if any(isinstance(spectrum, CumulativeNucleusSpectrumTable) for spectrum in blank_spectra):
        raise TypeError(
            "blank_spectra must all be CumulativeNucleusSpectrumTable or all be "
            "NormalizedInpSpectrumTable"
        )
    if not all(isinstance(spectrum, NormalizedInpSpectrumTable) for spectrum in blank_spectra):
        raise TypeError(
            "blank_spectra must all be CumulativeNucleusSpectrumTable or all be "
            "NormalizedInpSpectrumTable"
        )
    frames = []
    for idx, spectrum in enumerate(blank_spectra):
        frame = spectrum.to_dataframe().copy()
        frame = _ensure_inp_per_ml_column(frame, name=f"blank_spectra[{idx}]")
        frame["blank_replicate"] = idx
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["value_unit"].nunique(dropna=True) != 1:
        raise ValueError("Blank spectra must use one value_unit")
    if "basis" in combined and combined["basis"].nunique(dropna=True) != 1:
        raise ValueError("Blank spectra must use one basis")
    if require_positive:
        combined = combined[combined["value"].to_numpy(dtype=float) > 0.0].copy()
        if combined.empty:
            raise ValueError("No positive blank spectrum rows remain after filtering")

    value_agg = "mean" if value_method == "mean" else "median"
    grouped = combined.groupby("temperature_C", as_index=False).agg(
        value=("value", value_agg),
        lower_ci=("lower_ci", _rms_or_nan) if "lower_ci" in combined else ("value", _nan_like),
        upper_ci=("upper_ci", _rms_or_nan) if "upper_ci" in combined else ("value", _nan_like),
        replicate_count=("value", "size"),
        value_unit=("value_unit", "first"),
        basis=("basis", "first") if "basis" in combined else ("value_unit", _default_basis),
        temperature_bin_width_C=("temperature_bin_width_C", "first")
        if "temperature_bin_width_C" in combined
        else ("value", _nan_like),
        temperature_bin_method=("temperature_bin_method", "first")
        if "temperature_bin_method" in combined
        else ("value_unit", _empty_string),
        temperature_bin_left_C=("temperature_bin_left_C", "first")
        if "temperature_bin_left_C" in combined
        else ("value", _nan_like),
        temperature_bin_right_C=("temperature_bin_right_C", "first")
        if "temperature_bin_right_C" in combined
        else ("value", _nan_like),
    )
    grouped = grouped.sort_values("temperature_C", ascending=False)
    return NormalizedInpSpectrumTable(
        sample_id=np.repeat("blank", len(grouped)),
        temperature_C=grouped["temperature_C"].to_numpy(dtype=float),
        temperature_bin_width_C=_single_positive_or_none(grouped["temperature_bin_width_C"]),
        temperature_bin_method=str(grouped["temperature_bin_method"].iloc[0])
        if str(grouped["temperature_bin_method"].iloc[0]) != "nan"
        else "",
        temperature_bin_left_C=grouped["temperature_bin_left_C"].to_numpy(dtype=float)
        if grouped["temperature_bin_left_C"].notna().any()
        else None,
        temperature_bin_right_C=grouped["temperature_bin_right_C"].to_numpy(dtype=float)
        if grouped["temperature_bin_right_C"].notna().any()
        else None,
        value=grouped["value"].to_numpy(dtype=float),
        value_unit=str(grouped["value_unit"].iloc[0]),
        basis=str(grouped["basis"].iloc[0]),
        lower_ci=grouped["lower_ci"].to_numpy(dtype=float),
        upper_ci=grouped["upper_ci"].to_numpy(dtype=float),
        replicate_count=grouped["replicate_count"].to_numpy(dtype=int),
        is_extrapolated=np.repeat(False, len(grouped)),
        processing_metadata=processing_metadata_for(
            "average_blank_spectra",
            inputs=tuple(blank_spectra),
            parameters={
                "value_method": value_method,
                "require_positive": require_positive,
            },
        ),
    )


def _average_cumulative_blank_spectra(
    blank_spectra: list[FilterBlankSpectrum],
    *,
    value_method: Literal["mean", "median"],
    require_positive: bool,
) -> CumulativeNucleusSpectrumTable:
    frames = []
    for idx, spectrum in enumerate(blank_spectra):
        if not isinstance(spectrum, CumulativeNucleusSpectrumTable):
            raise TypeError("blank_spectra must all be CumulativeNucleusSpectrumTable")
        frame = _prepare_cumulative_spectrum_frame(
            spectrum,
            name=f"blank_spectra[{idx}]",
        )
        frame["blank_replicate"] = idx
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined["value_unit"].nunique(dropna=True) != 1:
        raise ValueError("Blank spectra must use one value_unit")
    if combined["basis"].nunique(dropna=True) != 1:
        raise ValueError("Blank spectra must use one basis")
    if require_positive:
        combined = combined[combined["value"].to_numpy(dtype=float) > 0.0].copy()
        if combined.empty:
            raise ValueError("No positive blank spectrum rows remain after filtering")

    value_agg = "mean" if value_method == "mean" else "median"
    grouped = combined.groupby("temperature_C", as_index=False).agg(
        value=("value", value_agg),
        lower_ci=("lower_ci", _rms_or_nan) if "lower_ci" in combined else ("value", _nan_like),
        upper_ci=("upper_ci", _rms_or_nan) if "upper_ci" in combined else ("value", _nan_like),
        value_unit=("value_unit", "first"),
        basis=("basis", "first"),
        temperature_bin_width_C=("temperature_bin_width_C", "first")
        if "temperature_bin_width_C" in combined
        else ("value", _nan_like),
        temperature_bin_method=("temperature_bin_method", "first")
        if "temperature_bin_method" in combined
        else ("value_unit", _empty_string),
        temperature_bin_left_C=("temperature_bin_left_C", "first")
        if "temperature_bin_left_C" in combined
        else ("value", _nan_like),
        temperature_bin_right_C=("temperature_bin_right_C", "first")
        if "temperature_bin_right_C" in combined
        else ("value", _nan_like),
        dilution_fold=("dilution_fold", "first")
        if "dilution_fold" in combined
        else ("value", _nan_like),
    )
    grouped = grouped.sort_values("temperature_C", ascending=False)
    return CumulativeNucleusSpectrumTable(
        sample_id=np.repeat("blank", len(grouped)),
        temperature_C=grouped["temperature_C"].to_numpy(dtype=float),
        temperature_bin_width_C=_single_positive_or_none(grouped["temperature_bin_width_C"]),
        temperature_bin_method=str(grouped["temperature_bin_method"].iloc[0])
        if str(grouped["temperature_bin_method"].iloc[0]) != "nan"
        else "",
        temperature_bin_left_C=grouped["temperature_bin_left_C"].to_numpy(dtype=float)
        if grouped["temperature_bin_left_C"].notna().any()
        else None,
        temperature_bin_right_C=grouped["temperature_bin_right_C"].to_numpy(dtype=float)
        if grouped["temperature_bin_right_C"].notna().any()
        else None,
        value=grouped["value"].to_numpy(dtype=float),
        value_unit=str(grouped["value_unit"].iloc[0]),
        basis=str(grouped["basis"].iloc[0]),
        lower_ci=grouped["lower_ci"].to_numpy(dtype=float),
        upper_ci=grouped["upper_ci"].to_numpy(dtype=float),
        dilution_fold=grouped["dilution_fold"].to_numpy(dtype=float)
        if grouped["dilution_fold"].notna().any()
        else None,
        processing_metadata=processing_metadata_for(
            "average_blank_spectra",
            inputs=tuple(blank_spectra),
            parameters={
                "value_method": value_method,
                "require_positive": require_positive,
            },
        ),
    )


def _rms_or_nan(values: pd.Series) -> float:
    values = values.dropna().to_numpy(dtype=float)
    if values.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(values**2)))


def _nan_like(_: pd.Series) -> float:
    return np.nan


def _default_basis(_: pd.Series) -> str:
    return "other"


def _empty_string(_: pd.Series) -> str:
    return ""


def _single_positive_or_none(values: pd.Series) -> float | None:
    clean = values.dropna().unique()
    if len(clean) == 0:
        return None
    if len(clean) > 1:
        raise ValueError("Blank spectra must use one temperature_bin_width_C")
    value = float(clean[0])
    return value if value > 0 else None


def align_spectra_on_temperature(
    sample: NormalizedInpSpectrumTable,
    blank: NormalizedInpSpectrumTable,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return sample and blank rows aligned to exact shared temperatures."""

    sample_df = sample.to_dataframe().set_index("temperature_C", drop=False)
    blank_df = blank.to_dataframe().set_index("temperature_C", drop=False)
    common = sample_df.index.intersection(blank_df.index)
    if common.empty:
        raise ValueError("Sample and blank spectra have no common temperatures")
    common = common.sort_values(ascending=False)
    return sample_df.loc[common].copy(), blank_df.loc[common].copy()


def subtract_filter_blank_spectrum(
    sample: FilterBlankSpectrum | SpectrumSequence | SpectrumMapping,
    blank: FilterBlankSpectrum | SpectrumSequence | SpectrumMapping,
    *,
    extrapolate_missing_cold: bool = True,
    apply_qc: bool = True,
    threshold_percent: float = 10.0,
    error_signal: float = -9999.0,
    clamp_zero: bool = False,
) -> FilterBlankSpectrum | list[Any] | dict[str, Any]:
    """Apply OLAF-style filter blank correction to suspension or normalized spectra."""

    if _is_spectrum_mapping(sample) or _is_spectrum_mapping(blank):
        if _is_spectrum_mapping(sample) and _is_spectrum_mapping(blank):
            missing = set(sample) ^ set(blank)
            if missing:
                raise KeyError(
                    "sample and blank dictionaries must use the same keys: "
                    f"{sorted(missing)}"
                )
            return {
                key: subtract_filter_blank_spectrum(
                    sample[key],
                    blank[key],
                    extrapolate_missing_cold=extrapolate_missing_cold,
                    apply_qc=apply_qc,
                    threshold_percent=threshold_percent,
                    error_signal=error_signal,
                    clamp_zero=clamp_zero,
                )
                for key in sample
            }
        if _is_spectrum_mapping(sample):
            return {
                key: subtract_filter_blank_spectrum(
                    nested,
                    blank,
                    extrapolate_missing_cold=extrapolate_missing_cold,
                    apply_qc=apply_qc,
                    threshold_percent=threshold_percent,
                    error_signal=error_signal,
                    clamp_zero=clamp_zero,
                )
                for key, nested in sample.items()
            }
        return {
            key: subtract_filter_blank_spectrum(
                sample,
                nested_blank,
                extrapolate_missing_cold=extrapolate_missing_cold,
                apply_qc=apply_qc,
                threshold_percent=threshold_percent,
                error_signal=error_signal,
                clamp_zero=clamp_zero,
            )
            for key, nested_blank in blank.items()
        }

    if _is_spectrum_sequence(sample) or _is_spectrum_sequence(blank):
        if _is_spectrum_sequence(sample) and _is_spectrum_sequence(blank):
            sample_tables = _require_spectrum_sequence(sample, "sample")
            blank_tables = _require_spectrum_sequence(blank, "blank")
            if len(sample_tables) != len(blank_tables):
                raise ValueError("sample and blank lists must have the same length")
            return [
                subtract_filter_blank_spectrum(
                    sample_table,
                    blank_table,
                    extrapolate_missing_cold=extrapolate_missing_cold,
                    apply_qc=apply_qc,
                    threshold_percent=threshold_percent,
                    error_signal=error_signal,
                    clamp_zero=clamp_zero,
                )
                for sample_table, blank_table in zip(sample_tables, blank_tables)
            ]
        if _is_spectrum_sequence(sample):
            sample_tables = _require_spectrum_sequence(sample, "sample")
            if not isinstance(
                blank,
                (CumulativeNucleusSpectrumTable, NormalizedInpSpectrumTable),
            ):
                raise TypeError(
                    "blank must be a CumulativeNucleusSpectrumTable or "
                    "NormalizedInpSpectrumTable"
                )
            return [
                subtract_filter_blank_spectrum(
                    sample_table,
                    blank,
                    extrapolate_missing_cold=extrapolate_missing_cold,
                    apply_qc=apply_qc,
                    threshold_percent=threshold_percent,
                    error_signal=error_signal,
                    clamp_zero=clamp_zero,
                )
                for sample_table in sample_tables
            ]
        blank_tables = _require_spectrum_sequence(blank, "blank")
        if not isinstance(
            sample,
            (CumulativeNucleusSpectrumTable, NormalizedInpSpectrumTable),
        ):
            raise TypeError(
                "sample must be a CumulativeNucleusSpectrumTable or "
                "NormalizedInpSpectrumTable"
            )
        return [
            subtract_filter_blank_spectrum(
                sample,
                blank_table,
                extrapolate_missing_cold=extrapolate_missing_cold,
                apply_qc=apply_qc,
                threshold_percent=threshold_percent,
                error_signal=error_signal,
                clamp_zero=clamp_zero,
            )
            for blank_table in blank_tables
        ]

    if isinstance(sample, CumulativeNucleusSpectrumTable) or isinstance(
        blank,
        CumulativeNucleusSpectrumTable,
    ):
        if not isinstance(sample, CumulativeNucleusSpectrumTable) or not isinstance(
            blank,
            CumulativeNucleusSpectrumTable,
        ):
            raise TypeError(
                "sample and blank must both be CumulativeNucleusSpectrumTable for "
                "per-mL suspension correction, or both be NormalizedInpSpectrumTable"
            )
        return _subtract_filter_blank_cumulative_table(
            sample,
            blank,
            extrapolate_missing_cold=extrapolate_missing_cold,
            apply_qc=apply_qc,
            threshold_percent=threshold_percent,
            error_signal=error_signal,
            clamp_zero=clamp_zero,
        )

    if not isinstance(sample, NormalizedInpSpectrumTable) or not isinstance(
        blank,
        NormalizedInpSpectrumTable,
    ):
        raise TypeError(
            "sample and blank must both be CumulativeNucleusSpectrumTable or both "
            "be NormalizedInpSpectrumTable"
        )

    return _subtract_filter_blank_table(
        sample,
        blank,
        extrapolate_missing_cold=extrapolate_missing_cold,
        apply_qc=apply_qc,
        threshold_percent=threshold_percent,
        error_signal=error_signal,
        clamp_zero=clamp_zero,
    )


def subtract_blank_spectrum(
    sample: FilterBlankSpectrum | SpectrumSequence | SpectrumMapping,
    blank: FilterBlankSpectrum | SpectrumSequence | SpectrumMapping,
    *,
    extrapolate_missing_cold: bool = True,
    apply_qc: bool = True,
    threshold_percent: float = 10.0,
    error_signal: float = -9999.0,
    clamp_zero: bool = False,
) -> FilterBlankSpectrum | list[Any] | dict[str, Any]:
    """Backward-compatible public name for filter blank correction."""

    return subtract_filter_blank_spectrum(
        sample,
        blank,
        extrapolate_missing_cold=extrapolate_missing_cold,
        apply_qc=apply_qc,
        threshold_percent=threshold_percent,
        error_signal=error_signal,
        clamp_zero=clamp_zero,
    )


def _subtract_filter_blank_cumulative_table(
    sample: CumulativeNucleusSpectrumTable,
    blank: CumulativeNucleusSpectrumTable,
    *,
    extrapolate_missing_cold: bool,
    apply_qc: bool,
    threshold_percent: float,
    error_signal: float,
    clamp_zero: bool,
) -> CumulativeNucleusSpectrumTable:
    sample_df = _prepare_cumulative_spectrum_frame(sample, name="sample")
    blank = _blank_with_required_temperatures(
        blank,
        sample_df["temperature_C"].to_numpy(dtype=float),
        extrapolate_missing_cold=extrapolate_missing_cold,
    )
    blank_df = _prepare_cumulative_spectrum_frame(blank, name="blank").set_index(
        "temperature_C",
        drop=False,
    )
    _require_unique_temperatures(blank_df, name="blank")

    matched_blank = blank_df.loc[sample_df["temperature_C"].to_numpy(dtype=float)].reset_index(
        drop=True
    )
    corrected = sample_df.copy()
    corrected["correction_state"] = "blank_corrected"
    _require_compatible_confidence_columns(sample_df, matched_blank)

    corrected["value"] = (
        sample_df["value"].to_numpy(dtype=float) - matched_blank["value"].to_numpy(dtype=float)
    )
    if _has_complete_confidence_columns(sample_df):
        corrected["lower_ci"] = propagate_uncertainty_rss(
            sample_df["lower_ci"].to_numpy(dtype=float),
            matched_blank["lower_ci"].to_numpy(dtype=float),
        )
        corrected["upper_ci"] = propagate_uncertainty_rss(
            sample_df["upper_ci"].to_numpy(dtype=float),
            matched_blank["upper_ci"].to_numpy(dtype=float),
        )

    if "is_extrapolated" in sample_df or "is_extrapolated" in matched_blank:
        corrected["is_extrapolated"] = _bool_column(sample_df, "is_extrapolated") | _bool_column(
            matched_blank,
            "is_extrapolated",
        )

    if apply_qc:
        if not _has_complete_confidence_columns(corrected):
            raise ValueError("lower_ci and upper_ci are required when apply_qc=True")
        corrected = qc_blank_corrected_spectrum(
            corrected,
            sample_df,
            threshold_percent=threshold_percent,
            error_signal=error_signal,
        )

    if clamp_zero:
        clamp_mask = corrected["value"].to_numpy(dtype=float) != error_signal
        corrected.loc[clamp_mask, "value"] = np.maximum(
            corrected.loc[clamp_mask, "value"].to_numpy(dtype=float),
            0.0,
        )

    return CumulativeNucleusSpectrumTable.from_dataframe(
        corrected,
        value_unit="INP_per_mL_suspension",
        basis="suspension",
        metadata=sample.metadata,
        processing_metadata=processing_metadata_for(
            "subtract_filter_blank_spectrum",
            inputs=(sample, blank),
            parameters={
                "extrapolate_missing_cold": extrapolate_missing_cold,
                "apply_qc": apply_qc,
                "threshold_percent": threshold_percent,
                "error_signal": error_signal,
                "clamp_zero": clamp_zero,
                "basis": "suspension",
            },
        ),
    )


def _subtract_filter_blank_table(
    sample: NormalizedInpSpectrumTable,
    blank: NormalizedInpSpectrumTable,
    *,
    extrapolate_missing_cold: bool,
    apply_qc: bool,
    threshold_percent: float,
    error_signal: float,
    clamp_zero: bool,
) -> NormalizedInpSpectrumTable:
    _require_suspension_normalized_table(sample, name="sample")
    _require_suspension_normalized_table(blank, name="blank")
    sample_df = _prepare_spectrum_frame(sample, name="sample")
    blank = _blank_with_required_temperatures(
        blank,
        sample_df["temperature_C"].to_numpy(dtype=float),
        extrapolate_missing_cold=extrapolate_missing_cold,
    )
    blank_df = _prepare_spectrum_frame(blank, name="blank").set_index(
        "temperature_C",
        drop=False,
    )
    _require_unique_temperatures(blank_df, name="blank")

    matched_blank = blank_df.loc[sample_df["temperature_C"].to_numpy(dtype=float)].reset_index(
        drop=True
    )
    corrected = sample_df.copy()
    corrected["correction_state"] = "blank_corrected"
    _require_compatible_confidence_columns(sample_df, matched_blank)

    corrected["value"] = (
        sample_df["value"].to_numpy(dtype=float) - matched_blank["value"].to_numpy(dtype=float)
    )
    if _has_complete_confidence_columns(sample_df):
        corrected["lower_ci"] = propagate_uncertainty_rss(
            sample_df["lower_ci"].to_numpy(dtype=float),
            matched_blank["lower_ci"].to_numpy(dtype=float),
        )
        corrected["upper_ci"] = propagate_uncertainty_rss(
            sample_df["upper_ci"].to_numpy(dtype=float),
            matched_blank["upper_ci"].to_numpy(dtype=float),
        )

    if "is_extrapolated" in sample_df or "is_extrapolated" in matched_blank:
        corrected["is_extrapolated"] = _bool_column(sample_df, "is_extrapolated") | _bool_column(
            matched_blank,
            "is_extrapolated",
        )

    if apply_qc:
        if not _has_complete_confidence_columns(corrected):
            raise ValueError("lower_ci and upper_ci are required when apply_qc=True")
        corrected = qc_blank_corrected_spectrum(
            corrected,
            sample_df,
            threshold_percent=threshold_percent,
            error_signal=error_signal,
        )

    if clamp_zero:
        clamp_mask = corrected["value"].to_numpy(dtype=float) != error_signal
        corrected.loc[clamp_mask, "value"] = np.maximum(
            corrected.loc[clamp_mask, "value"].to_numpy(dtype=float),
            0.0,
        )

    return NormalizedInpSpectrumTable.from_dataframe(
        corrected,
        correction_state="blank_corrected",
        metadata=sample.metadata,
        processing_metadata=processing_metadata_for(
            "subtract_filter_blank_spectrum",
            inputs=(sample, blank),
            parameters={
                "extrapolate_missing_cold": extrapolate_missing_cold,
                "apply_qc": apply_qc,
                "threshold_percent": threshold_percent,
                "error_signal": error_signal,
                "clamp_zero": clamp_zero,
                "basis": "suspension",
            },
        ),
    )


def _require_suspension_normalized_table(
    spectrum: NormalizedInpSpectrumTable,
    *,
    name: str,
) -> None:
    frame = spectrum.to_dataframe()
    basis = _single_text(frame, "basis", name=f"{name} spectrum")
    value_unit = _single_text(frame, "value_unit", name=f"{name} spectrum")
    if basis == "suspension" and value_unit == "INP_per_mL_suspension":
        return
    raise ValueError(
        "Filter blank correction must be applied to INP_per_mL_suspension spectra "
        "before sample-basis normalization. Pass CumulativeNucleusSpectrumTable inputs "
        "to subtract_filter_blank_spectrum(), then call normalize_spec()."
    )


def _prepare_spectrum_frame(
    spectrum: NormalizedInpSpectrumTable,
    *,
    name: str,
) -> pd.DataFrame:
    frame = spectrum.to_dataframe().copy()
    required = {"sample_id", "temperature_C", "value", "value_unit", "basis"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} spectrum missing required columns: {sorted(missing)}")
    frame = _ensure_inp_per_ml_column(frame, name=name)
    if frame["inp_per_mL"].isna().any():
        raise ValueError(f"{name} spectrum contains missing inp_per_mL values")
    return frame.sort_values("temperature_C", ascending=False).reset_index(drop=True)


def _prepare_cumulative_spectrum_frame(
    spectrum: CumulativeNucleusSpectrumTable,
    *,
    name: str,
) -> pd.DataFrame:
    frame = spectrum.to_dataframe().copy()
    required = {"sample_id", "temperature_C", "value", "value_unit", "basis"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{name} spectrum missing required columns: {sorted(missing)}")
    _require_suspension_per_ml(frame, name=name)
    if frame["value"].isna().any():
        raise ValueError(f"{name} spectrum contains missing value rows")
    return frame.sort_values("temperature_C", ascending=False).reset_index(drop=True)


def _require_suspension_per_ml(df: pd.DataFrame, *, name: str) -> None:
    basis = _single_text(df, "basis", name=f"{name} spectrum")
    value_unit = _single_text(df, "value_unit", name=f"{name} spectrum")
    if basis != "suspension" or value_unit != "INP_per_mL_suspension":
        raise ValueError(
            f"{name} spectrum must use value_unit='INP_per_mL_suspension' "
            "and basis='suspension'"
        )


def _ensure_inp_per_ml_column(df: pd.DataFrame, *, name: str) -> pd.DataFrame:
    if "inp_per_mL" in df:
        return df
    basis_values = pd.Series(df["basis"]).dropna().astype(str).unique() if "basis" in df else []
    if len(basis_values) == 1 and basis_values[0] == "suspension":
        df = df.copy()
        df["inp_per_mL"] = df["value"].to_numpy(dtype=float)
        return df
    raise ValueError(f"{name} spectrum requires inp_per_mL for filter blank correction")


def _blank_with_required_temperatures(
    blank: FilterBlankSpectrum,
    sample_temperatures_C: np.ndarray,
    *,
    extrapolate_missing_cold: bool,
) -> FilterBlankSpectrum:
    prepared = blank
    blank_temperatures = _table_temperatures(prepared)
    missing = _missing_temperatures(sample_temperatures_C, blank_temperatures)
    if missing and extrapolate_missing_cold:
        min_blank_temperature = min(blank_temperatures)
        cold_missing = np.array(
            [temperature for temperature in missing if temperature < min_blank_temperature],
            dtype=float,
        )
        if cold_missing.size:
            prepared = extrapolate_blank_tail(prepared, cold_missing)
            blank_temperatures = _table_temperatures(prepared)
            missing = _missing_temperatures(sample_temperatures_C, blank_temperatures)
    if missing:
        raise ValueError(
            "Blank spectrum is missing temperatures: "
            f"{_format_temperatures(missing)}. Provide a covering blank spectrum or "
            "enable cold-tail extrapolation."
        )
    return prepared


def _table_temperatures(table: FilterBlankSpectrum) -> np.ndarray:
    temperatures = table.to_dataframe()["temperature_C"].dropna().to_numpy(dtype=float)
    if temperatures.size == 0:
        raise ValueError("Blank spectrum contains no temperatures")
    return temperatures


def _missing_temperatures(required: np.ndarray, available: np.ndarray) -> list[float]:
    available_set = set(float(value) for value in available)
    return sorted(
        {float(value) for value in required if float(value) not in available_set},
        reverse=True,
    )


def _format_temperatures(temperatures: list[float]) -> str:
    return ", ".join(f"{temperature:g}" for temperature in temperatures)


def _require_unique_temperatures(df: pd.DataFrame, *, name: str) -> None:
    duplicated = df["temperature_C"].duplicated()
    if duplicated.any():
        values = sorted(df.loc[duplicated, "temperature_C"].unique(), reverse=True)
        raise ValueError(
            f"{name} spectrum has duplicate temperatures: {_format_temperatures(values)}"
        )


def _require_compatible_confidence_columns(
    sample_df: pd.DataFrame,
    blank_df: pd.DataFrame,
) -> None:
    sample_has_any = "lower_ci" in sample_df or "upper_ci" in sample_df
    blank_has_any = "lower_ci" in blank_df or "upper_ci" in blank_df
    if not sample_has_any and not blank_has_any:
        return
    if not _has_complete_confidence_columns(sample_df):
        raise ValueError("sample spectrum must include both lower_ci and upper_ci")
    if not _has_complete_confidence_columns(blank_df):
        raise ValueError("blank spectrum must include both lower_ci and upper_ci")


def _has_complete_confidence_columns(df: pd.DataFrame) -> bool:
    return "lower_ci" in df and "upper_ci" in df


def _single_text(df: pd.DataFrame, column: str, *, name: str) -> str:
    values = pd.Series(df[column]).dropna().astype(str).unique()
    if len(values) != 1:
        raise ValueError(f"{name} must use one {column}")
    return str(values[0])


def _bool_column(df: pd.DataFrame, column: str) -> np.ndarray:
    if column not in df:
        return np.repeat(False, len(df))
    return df[column].fillna(False).to_numpy(dtype=bool)


def extrapolate_blank_tail(
    blank: FilterBlankSpectrum | SpectrumSequence | SpectrumMapping,
    target_temperatures_C: np.ndarray,
    *,
    tail_points: int = 4,
) -> FilterBlankSpectrum | list[Any] | dict[str, Any]:
    """Linearly extrapolate a blank spectrum to colder target temperatures."""

    if _is_spectrum_mapping(blank):
        return {
            key: extrapolate_blank_tail(
                nested_blank,
                target_temperatures_C,
                tail_points=tail_points,
            )
            for key, nested_blank in blank.items()
        }
    if _is_spectrum_sequence(blank):
        blank_tables = _require_spectrum_sequence(blank, "blank")
        return [
            extrapolate_blank_tail(
                blank_table,
                target_temperatures_C,
                tail_points=tail_points,
            )
            for blank_table in blank_tables
        ]

    if tail_points < 2:
        raise ValueError("tail_points must be at least 2")
    blank_df = blank.to_dataframe().sort_values("temperature_C", ascending=False)
    target = np.asarray(target_temperatures_C, dtype=float)
    if target.size == 0:
        return blank
    min_blank_temp = blank_df["temperature_C"].min()
    extrap_temps = np.array([temp for temp in target if temp < min_blank_temp], dtype=float)
    if extrap_temps.size == 0:
        return blank
    tail = blank_df.tail(tail_points)
    if len(tail) < 2:
        raise ValueError("At least two blank points are required for extrapolation")
    fit = np.polyfit(tail["temperature_C"].to_numpy(dtype=float), tail["value"].to_numpy(float), 1)
    extrap_values = fit[0] * extrap_temps + fit[1]

    extrapolated = {
        "sample_id": "blank",
        "temperature_C": extrap_temps,
        "value": extrap_values,
        "value_unit": blank_df["value_unit"].iloc[0],
        "basis": blank_df["basis"].iloc[0] if "basis" in blank_df else "other",
        "replicate_count": 0,
        "is_extrapolated": True,
    }
    if "lower_ci" in blank_df:
        extrapolated["lower_ci"] = extrap_values * _safe_error_ratio(tail, "lower_ci")
    if "upper_ci" in blank_df:
        extrapolated["upper_ci"] = extrap_values * _safe_error_ratio(tail, "upper_ci")
    extrap_df = pd.DataFrame(extrapolated)
    if "temperature_bin_width_C" in blank_df:
        width_values = blank_df["temperature_bin_width_C"].dropna().unique()
        if len(width_values) == 1:
            width = float(width_values[0])
            extrap_df["temperature_bin_width_C"] = width
            extrap_df["temperature_bin_left_C"] = extrap_temps - width / 2.0
            extrap_df["temperature_bin_right_C"] = extrap_temps + width / 2.0
    if "temperature_bin_method" in blank_df:
        method_values = blank_df["temperature_bin_method"].dropna().unique()
        if len(method_values) == 1:
            extrap_df["temperature_bin_method"] = str(method_values[0])
    if "inp_per_mL" in blank_df:
        fit_ml = np.polyfit(
            tail["temperature_C"].to_numpy(dtype=float),
            tail["inp_per_mL"].to_numpy(dtype=float),
            1,
        )
        extrap_df["inp_per_mL"] = fit_ml[0] * extrap_temps + fit_ml[1]
    if "is_extrapolated" not in blank_df:
        blank_df["is_extrapolated"] = False
    if "replicate_count" not in blank_df:
        blank_df["replicate_count"] = 1
    combined = pd.concat([blank_df, extrap_df], ignore_index=True).sort_values(
        "temperature_C", ascending=False
    )
    combined["is_extrapolated"] = combined["is_extrapolated"].fillna(False)
    combined["replicate_count"] = combined["replicate_count"].fillna(1)
    if isinstance(blank, CumulativeNucleusSpectrumTable):
        return CumulativeNucleusSpectrumTable.from_dataframe(
            combined,
            value_unit="INP_per_mL_suspension",
            basis="suspension",
            metadata=blank.metadata,
            processing_metadata=processing_metadata_for(
                "extrapolate_blank_tail",
                inputs=(blank,),
                parameters={"target_temperatures_C": target.tolist(), "tail_points": tail_points},
            ),
        )
    return NormalizedInpSpectrumTable.from_dataframe(
        combined,
        metadata=blank.metadata,
        processing_metadata=processing_metadata_for(
            "extrapolate_blank_tail",
            inputs=(blank,),
            parameters={"target_temperatures_C": target.tolist(), "tail_points": tail_points},
        ),
    )


def _safe_error_ratio(df: pd.DataFrame, column: str) -> float:
    if column not in df:
        return np.nan
    values = df[column].to_numpy(dtype=float)
    base = np.maximum(df["value"].to_numpy(dtype=float), 1e-12)
    return float(np.nanmean(values / base))
