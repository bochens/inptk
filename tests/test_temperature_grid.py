"""Count selection must preserve physical wells, source pairs, gaps and blanks."""

import json

import numpy as np
import pandas as pd
import pytest

import inptk
from inptk.alignment import align_observations
from inptk.temperature_selection import validate_temperature_selection


def source(temperatures=(-6.9, -7.024, -7.51), frozen=(0, 1, 4), totals=(10, 10, 10)):
    rows = pd.DataFrame({
        "measurement_id": "sample", "run_id": "R", "cycle_id": "0",
        "temperature_C": temperatures, "time_s": range(len(temperatures)),
        "n_frozen": frozen, "n_total": totals,
    })
    return inptk.read_counts(rows, metadata=[{
        "measurement_id": "sample", "sample_id": "S", "run_id": "R",
        "droplet_volume_uL": 50, "dilution": 1,
    }])


def points(data, method="latest", window=None, ranges=None):
    return align_observations(
        data.counts.to_dataframe(),
        [{"measurement_id": "sample", "run_id": "R", "cycle_id": "0"}],
        water_blank_map=data.water_blank_map, temperature_ranges_C=ranges,
        temperature_step_C=.5, temperature_method=method, temperature_window_C=window,
    )


def test_window_uses_nearby_colder_event_latest_and_max_do_not():
    data = source()
    original = data.counts.to_dataframe()
    for method in ("latest", "max"):
        point = points(data, method)[0]
        assert point.temperature_C == -7
        assert point.samples.n_frozen.tolist() == [0]
        assert point.samples.temperature_C.tolist() == [-6.9]
    selected = points(data, "window", .5)[0]
    assert selected.samples.n_frozen.tolist() == [1]
    assert selected.samples.temperature_C.tolist() == [-7.024]
    assert selected.samples.fit_temperature_C.tolist() == [-7]
    pd.testing.assert_frame_equal(original, data.counts.to_dataframe())


def test_max_preserves_count_pair_and_uses_fraction_not_count():
    data = source((-5, -5.5, -6), (8, 9, 7), (10, 20, 20))
    selected = points(data, "max")[-1].samples.iloc[0]
    assert (selected.n_frozen, selected.n_total) == (8, 10)
    latest = points(data)[-1].samples.iloc[0]
    assert (latest.n_frozen, latest.n_total) == (7, 20)
    maximum = inptk.analyze_concentration(
        data, method="average", temperature_step_C=.5, temperature_method="max"
    ).to_dataframe()
    assert np.allclose(maximum.concentration, -np.log(.2) / .05)


def test_window_selects_maximum_count_and_latest_tie_without_synthetic_zeros():
    data = source((-5.01, -5.02, -6.01), (2, 2, 4))
    chosen = points(data, "window", .1)
    # Grid stays inside measured support: -5.5 and -6.0; -5.0 is outside.
    assert chosen[0].samples.empty
    assert chosen[1].samples.n_frozen.tolist() == [4]
    data = source((-4.9, -5, -5.01, -5.1), (0, 2, 2, 2), (10,) * 4)
    assert points(data, "window", .1)[0].samples.temperature_C.tolist() == [-5.01]


def test_range_checks_both_source_and_target():
    data = source()
    chosen = points(data, "window", .5, {"sample": {"min_C": -7}})
    assert chosen[0].samples.n_frozen.tolist() == [0]  # -7.024 is outside range
    assert chosen[1].samples.empty


@pytest.mark.parametrize("method", ["latest", "max", "window"])
def test_whole_curve_fits_selected_counts_and_keeps_original_observations(method):
    data = source((-5, -6, -7), (1, 4, 7))
    kwargs = {"temperature_step_C": 1, "temperature_method": method}
    if method == "window":
        kwargs["temperature_window_C"] = .5
    result = inptk.analyze_concentration(data, **kwargs, differential=True)
    actual = result.to_dataframe().sort_values("temperature_C", ascending=False)
    np.testing.assert_allclose(actual.concentration, -np.log([.9, .6, .3]) / .05, rtol=1e-5)
    details = next(step for step in result.history if step["operation"] == "estimate_concentration")
    fit = next(iter(details["joint_curve_fits"].values()))
    assert fit["physical_droplets"] == 10
    assert fit["sources"][0]["fit_temperatures_C"] == [-5, -6, -7]
    individual = inptk.cumulative_spectrum(inptk.frozen_fraction(data), experiment=data, **kwargs)
    np.testing.assert_allclose(individual.to_dataframe().concentration, actual.concentration)
    assert result.settings["temperature_method"] == method
    pd.testing.assert_frame_equal(result.frozen_fraction.to_dataframe()[list(data.counts.columns)],
                                  data.counts.to_dataframe())


@pytest.mark.parametrize("counts,totals,error", [
    ((8, 7, 9), (10, 10, 10), "decrease"),
    ((1, 2, 4), (10, 12, 12), "fixed set of wells"),
])
def test_selection_cannot_hide_invalid_raw_mle_history(counts, totals, error):
    data = source((-5, -5.2, -6), counts, totals)
    with pytest.raises(ValueError, match=error):
        inptk.analyze_concentration(
            data, temperature_step_C=1, temperature_method="max"
        )


def blank_source():
    rows, metadata = [], []
    for name, sample, total, volume, frozen in [
        ("sample", "S", 20, 50, (4, 6, 8)),
        ("replicate", "S", 20, 50, (4, 6, 8)),
        ("blank", "water", 10, 100, (1, 2, 3)),
    ]:
        metadata.append({"measurement_id": name, "sample_id": sample, "run_id": "R",
                         "droplet_volume_uL": volume, "dilution": 1})
        for time, (temp, count) in enumerate(zip((-5, -6, -7), frozen, strict=True)):
            rows.append({"measurement_id": name, "run_id": "R", "cycle_id": "0", "time_s": time,
                         "temperature_C": temp, "n_total": total, "n_frozen": count})
    return inptk.read_counts(pd.DataFrame(rows), metadata=metadata,
                            water_blank_map={"sample": ["blank"], "replicate": ["blank"]})


@pytest.mark.parametrize("method", ["mle", "average"])
def test_grid_keeps_shared_blank_once_with_unequal_volumes_and_saved_sources(tmp_path, method):
    data = blank_source()
    result = inptk.analyze_concentration(
        data, method=method, temperature_step_C=1, temperature_method="window",
        temperature_window_C=.5,
    )
    actual = result.to_dataframe().sort_values("temperature_C", ascending=False)
    expected = -np.log([.8, .7, .6]) / .05 + np.log([.9, .8, .7]) / .1
    np.testing.assert_allclose(actual.concentration, expected, rtol=1e-5)
    for sources in actual.source_observations.map(json.loads):
        assert sum(s["role"] == "blank" for s in sources) == 1
        assert all(s["alignment"] == "window" for s in sources)
    if method == "mle":
        history = next(s for s in result.history if s["operation"] == "estimate_concentration")
        assert next(iter(history["joint_curve_fits"].values()))["physical_droplets"] == 50
    path = tmp_path / "grid.inptk"
    inptk.save(result, path)
    restored = inptk.load(path)
    assert restored.settings["temperature_window_C"] == .5
    pd.testing.assert_frame_equal(
        restored.experiment.counts.to_dataframe(), data.counts.to_dataframe()
    )


def test_missing_blank_window_is_an_error_not_latest_fallback():
    data = blank_source()
    frame = data.counts.to_dataframe()
    frame["temperature_C"] = frame.temperature_C.astype(float)
    frame.loc[frame.measurement_id.eq("blank") & frame.temperature_C.eq(-6), "temperature_C"] = -6.3
    with pytest.raises(ValueError, match="no eligible window observation"):
        align_observations(
            frame, [{"measurement_id": "sample", "run_id": "R", "cycle_id": "0"}],
            water_blank_map=data.water_blank_map, temperature_ranges_C={},
            temperature_step_C=1, temperature_method="window", temperature_window_C=.1,
        )


@pytest.mark.parametrize("method", ["mle", "average"])
def test_grid_differential_retains_intervals_and_never_bridges_empty_windows(method):
    data = source((-5, -6, -7), (1, 4, 7))
    result = inptk.analyze_concentration(
        data, method=method, temperature_step_C=1, differential=True,
    )
    assert len(next(iter(result.curves.values())).differential) == 2
    result = inptk.analyze_concentration(
        data, method=method, temperature_step_C=.5, temperature_method="window",
        temperature_window_C=.1, differential=True,
    )
    assert len(next(iter(result.curves.values())).differential) == 0


def test_grid_preserves_cycles_and_per_run_blanks():
    from test_alignment import member, stream

    frames = [
        stream("S1", [-5, -6], counts=[2, 5]),
        stream("B1", [-5, -6], counts=[0, 1]),
        stream("S2", [-5, -6], run="R2", counts=[3, 6]),
        stream("B2", [-5, -6], run="R2", counts=[1, 2]),
        stream("S1", [-5, -6], cycle="2", counts=[9, 15]),
    ]
    selected = align_observations(
        pd.concat(frames), [member("S1"), member("S2", run="R2")],
        water_blank_map={"S1": ["B1"], "S2": ["B2"]}, temperature_ranges_C={},
        temperature_step_C=1, temperature_method="max",
    )
    assert selected[-1].samples.n_frozen.tolist() == [5, 6]
    assert selected[-1].blanks.n_frozen.tolist() == [1, 2]
    assert set(selected[-1].samples.cycle_id) == {"1"}


@pytest.mark.parametrize("step,method,width", [
    (None, "max", None), (.5, "window", None), (.5, "latest", .1),
    (0, "latest", None), (True, "latest", None), (.5, "window", np.inf),
    (.5, "window", -1), (.5, "mean", None),
])
def test_invalid_or_irrelevant_settings_are_rejected(step, method, width):
    with pytest.raises(ValueError):
        validate_temperature_selection(step, method, width)
