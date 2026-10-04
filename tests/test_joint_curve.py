"""Scientific checks for the likelihood of physical droplet freezing histories."""

import numpy as np
import pandas as pd
import pytest

import inptk
from inptk._engine.curve_likelihood import CurveLikelihood, FreezingSeries
from inptk._engine.water_blank_math import fit_concentration


def test_two_inputs_match_direct_freezing_interval_likelihood_and_profile():
    from scipy.optimize import minimize, minimize_scalar

    # Three mutually exclusive outcomes: frozen by -5, freeze by -6, remain liquid.
    # Check the engine against those probabilities directly, without its matrices.
    def negative_log_likelihood(values):
        if values[0] <= 0 or values[1] <= values[0]:
            return np.inf
        answer = 0.0
        for exposure, outcomes in [(1.0, [1, 6, 3]), (0.25, [0, 3, 7])]:
            survival = np.exp(-exposure * np.asarray(values))
            probabilities = np.array([1 - survival[0], survival[0] - survival[1], survival[1]])
            answer -= np.asarray(outcomes) @ np.log(probabilities)
        return answer

    expected = minimize(negative_log_likelihood, [0.1, 1.2], method="Nelder-Mead",
                        options={"xatol": 1e-11, "fatol": 1e-11})
    assert expected.success
    model = CurveLikelihood([stream([1, 7], 10, volume=1000),
                             stream([0, 3], 10, volume=1000, dilution=4)], [-5, -6])
    warm = model.estimate(-5, 1.96**2 / 2)[0]
    cold, lower, upper = model.estimate(-6, 1.96**2 / 2)
    np.testing.assert_allclose([warm, cold], expected.x, rtol=2e-6)
    for bound in (cold - lower, cold + upper):
        profile = minimize_scalar(lambda warm, bound=bound: negative_log_likelihood([warm, bound]),
                                  bounds=(1e-12, bound), method="bounded",
                                  options={"xatol": 1e-12})
        assert profile.fun - expected.fun == pytest.approx(1.96**2 / 2, rel=2e-6)


def test_cli_rejects_changing_totals_without_writing_a_result(tmp_path, capsys):
    from inptk.cli import main

    counts = tmp_path / "counts.csv"
    counts.write_text("measurement_id,temperature_C,n_total,n_frozen\nA,-5,20,1\nA,-6,19,3\n")
    metadata = tmp_path / "metadata.csv"
    metadata.write_text("measurement_id,sample_id,dilution,droplet_volume_uL\nA,S,1,50\n")
    output = tmp_path / "result.inptk"
    with pytest.raises(SystemExit) as stopped:
        main(["analyze", str(counts), "--metadata", str(metadata), "--out", str(output)])
    assert stopped.value.code == 1
    assert "changing total" in capsys.readouterr().err
    assert not output.exists()


def stream(frozen, total=20, volume=50, dilution=1, temperatures=None):
    temperatures = np.asarray(
        temperatures if temperatures is not None else -5 - np.arange(len(frozen)), dtype=float
    )
    return FreezingSeries(temperatures, np.asarray(frozen), total, volume / 1000 / dilution)


@pytest.mark.parametrize("counts", [[0, 0, 0], [0, 4, 12], [1, 4, 12], [4, 4, 12]])
def test_single_stream_matches_binomial_profile_at_observed_temperatures(counts):
    data = stream(counts)
    model = CurveLikelihood([data], data.temperatures)
    for temperature, count in zip(data.temperatures, counts, strict=True):
        actual = model.estimate(temperature, 1.96**2 / 2)
        expected = fit_concentration([count], [20], [1], [50], confidence_drop=1.96**2 / 2)
        np.testing.assert_allclose(actual, expected[:3], rtol=2e-6, atol=2e-6)


def test_repeated_images_do_not_shrink_uncertainty():
    original = stream([0, 4, 12], temperatures=[-5, -6, -7])
    more_images = stream([0, 0, 0, 4, 4, 4, 12],
                         temperatures=[-5, -5.2, -5.8, -6, -6.2, -6.8, -7])
    first = CurveLikelihood([original], original.temperatures)
    second = CurveLikelihood([more_images], more_images.temperatures)
    for temperature in original.temperatures:
        np.testing.assert_allclose(first.estimate(temperature, 1.96**2 / 2),
                                   second.estimate(temperature, 1.96**2 / 2),
                                   rtol=2e-6, atol=2e-6)
    assert first.physical_droplets == second.physical_droplets == 20


def test_zero_event_interval_has_positive_upper_uncertainty():
    data = stream([0, 4], temperatures=[-5, -7])
    model = CurveLikelihood([data], [-5, -6, -7])
    estimate, lower_error, upper_error = model.estimate(-6, 1.96**2 / 2)
    assert estimate == 0
    assert lower_error == 0
    assert upper_error > 0


def test_saturation_does_not_invent_a_finite_tail_or_destroy_warm_information():
    data = stream([2, 20])
    model = CurveLikelihood([data], data.temperatures)
    expected = fit_concentration([2], [20], [1], [50], confidence_drop=1.96**2 / 2)
    np.testing.assert_allclose(model.estimate(-5, 1.96**2 / 2), expected[:3], rtol=2e-6)
    assert np.isnan(model.estimate(-6, 1.96**2 / 2)).all()


def experiment(specifications, *, blanks=None):
    rows, metadata = [], []
    for name, total, frozen, dilution, volume in specifications:
        metadata.append({"measurement_id": name, "sample_id": "blank" if name == "W" else "A",
                         "dilution": dilution, "droplet_volume_uL": volume})
        for index, count in enumerate(frozen):
            rows.append({"measurement_id": name, "temperature_C": -5 - index,
                         "n_frozen": count, "n_total": total, "time_s": index})
    return inptk.read_counts(pd.DataFrame(rows), metadata=metadata, water_blank_map=blanks)


def test_new_contributor_does_not_cut_off_the_colder_curve():
    data = experiment([("A", 20, [4, 8, 16], 1, 50), ("B", 20, [0, 0, 2], 10, 50)])
    result = inptk.analyze_concentration(data, temperature_ranges_C={"B": {"max_C": -6}})
    curve = next(iter(result.curves.values()))
    frame = curve.cumulative.to_dataframe()
    assert frame.temperature_C.tolist() == [-5, -6, -7]
    assert frame.concentration.diff().dropna().ge(0).all()
    assert len(curve.excluded) == 0
    assert (frame.lower_error > 0).all() and (frame.upper_error > 0).all()


def test_raw_shared_blank_is_counted_once_and_its_sample_size_changes_uncertainty():
    results = []
    for total, frozen in [(4, [1]), (32, [8])]:
        data = experiment([("A", 20, [10], 1, 50), ("B", 20, [10], 1, 50),
                           ("W", total, frozen, 1, 100)], blanks={"A": ["W"], "B": ["W"]})
        fitted = inptk.estimate_concentration(inptk.frozen_fraction(data), experiment=data)
        results.append(fitted.to_dataframe().iloc[0])
        detail = next(iter(fitted.history[-1]["joint_curve_fits"].values()))
        assert detail["physical_droplets"] == 40 + total
        assert sum(item["role"] == "blank" for item in detail["sources"]) == 1
    assert results[0].concentration == pytest.approx(results[1].concentration, rel=2e-6)
    assert results[1].lower_error < results[0].lower_error
    assert results[1].upper_error < results[0].upper_error


@pytest.mark.parametrize("field,values,message", [
    ("n_total", [20, 19, 18], "changing total"),
    ("n_frozen", [1, 4, 3], "first-freezing"),
])
def test_invalid_event_histories_are_rejected_without_a_pointwise_fallback(field, values, message):
    data = experiment([("A", 20, [1, 4, 8], 1, 50)])
    frame = data.counts.to_dataframe()
    frame[field] = values
    invalid = inptk.read_counts(frame, metadata=[vars(data.measurements["A"])])
    with pytest.raises(ValueError, match=message):
        inptk.analyze_concentration(invalid)


def test_joint_likelihood_is_invariant_to_duplicate_observation_rows_with_new_ids():
    data = experiment([("A", 20, [1, 4, 8], 1, 50)])
    original = inptk.analyze_concentration(data).to_dataframe()
    frame = pd.concat([data.counts.to_dataframe()] * 4, ignore_index=True)
    frame["observation_id"] = frame.index.astype(str)
    repeated = inptk.read_counts(frame, metadata=[vars(data.measurements["A"])])
    fitted = inptk.analyze_concentration(repeated).to_dataframe()
    np.testing.assert_allclose(fitted[["concentration", "lower_error", "upper_error"]],
                               original[["concentration", "lower_error", "upper_error"]],
                               rtol=2e-6, atol=2e-6)


def test_fitting_grid_preserves_off_grid_events_and_matches_direct_probability_fit():
    from scipy.optimize import minimize, minimize_scalar

    temperatures = [-5, -5.4, -6]
    inputs = [stream([1, 4, 7], 10, volume=1000, temperatures=temperatures),
              stream([0, 1, 3], 10, volume=1000, dilution=4, temperatures=temperatures)]
    original = [s.frozen.copy() for s in inputs]
    # K(-5.4) is 40% of the way from the warm grid level to the cold level.
    def objective(levels):
        warm, cold = levels
        if warm <= 0 or cold <= warm:
            return np.inf
        concentrations = np.array([warm, .6 * warm + .4 * cold, cold])
        value = 0.0
        for exposure, outcomes in [(1, [1, 3, 3, 3]), (.25, [0, 1, 2, 7])]:
            survival = np.exp(-exposure * concentrations)
            probabilities = np.r_[1 - survival[0], -np.diff(survival), survival[-1]]
            value -= np.asarray(outcomes) @ np.log(probabilities)
        return value

    independent = minimize(objective, [.1, 1.2], method="Nelder-Mead",
                           options={"xatol": 1e-11, "fatol": 1e-11})
    assert independent.success
    model = CurveLikelihood(inputs, temperatures, fit_step_C=1)
    values = [model.estimate(t, 1.96**2 / 2) for t in temperatures]
    np.testing.assert_allclose([values[0][0], values[-1][0]], independent.x, rtol=3e-6)
    center, lower, upper = values[1]
    assert center == pytest.approx(.6 * independent.x[0] + .4 * independent.x[1], rel=3e-6)
    for endpoint in [center - lower, center + upper]:
        profile = minimize_scalar(
            lambda warm, endpoint=endpoint: objective([warm, (endpoint - .6 * warm) / .4]),
            bounds=(1e-12, endpoint), method="bounded", options={"xatol": 1e-12},
        )
        assert profile.fun - independent.fun == pytest.approx(1.96**2 / 2, rel=3e-5)
    # Fractional temperatures inside the same grid cell have distinct bounds.
    other = model.estimate(-5.2, 1.96**2 / 2)
    assert other[0] < center
    assert other[0] + other[2] < center + upper
    for s, counts in zip(inputs, original, strict=True):
        np.testing.assert_array_equal(s.frozen, counts)
        np.testing.assert_array_equal(s.temperatures, temperatures)


def test_grid_at_observed_temperatures_retains_joint_blank_estimates_and_bounds():
    temperatures = np.array([-5., -6., -7.])
    inputs = [FreezingSeries(temperatures, np.array([1, 5, 12]), 20, .05, .05, "run"),
              FreezingSeries(temperatures, np.array([0, 2, 4]), 20, 0., .05, "run")]
    native = CurveLikelihood(inputs, temperatures)
    grid = CurveLikelihood(inputs, temperatures, fit_step_C=1)
    for temperature in temperatures:
        np.testing.assert_allclose(grid.estimate(temperature, 1.96**2 / 2),
                                   native.estimate(temperature, 1.96**2 / 2), rtol=3e-6)


@pytest.mark.parametrize("step", [0, -1, True, np.nan, np.inf])
def test_invalid_fitting_grid_steps_are_rejected(step):
    with pytest.raises(ValueError, match="finite and positive"):
        CurveLikelihood([stream([0, 4])], [-5, -6], fit_step_C=step)


def test_fitting_resolution_does_not_choose_output_temperatures():
    data = experiment([("A", 20, [1, 5, 12], 1, 50)])
    result = inptk.analyze_concentration(data, fit_step_C=.5)
    assert result.to_dataframe().temperature_C.tolist() == [-5, -6, -7]
    gridded = inptk.analyze_concentration(data, temperature_step_C=.5, fit_step_C=.5)
    assert gridded.to_dataframe().temperature_C.tolist() == [-5, -5.5, -6, -6.5, -7]
    pd.testing.assert_frame_equal(result.counts.to_dataframe(), gridded.counts.to_dataframe())


def test_public_grid_fit_matches_complete_likelihood_and_profile_at_output_targets(tmp_path):
    from inptk.alignment import align_observations
    from inptk.curve_fit import fit_curve
    from inptk.methods import resolve_curves

    data = experiment([("A", 20, [1, 5, 12], 1, 50), ("W", 10, [0, 1, 3], 1, 100)],
                      blanks={"A": ["W"]})
    frame = data.counts.to_dataframe()
    frame["temperature_C"] += .07  # observations intentionally off the reporting grid
    data = inptk.Experiment(inptk.CountsTable(frame), data.samples, data.measurements,
                            water_blank_map=data.water_blank_map)
    members = next(iter(resolve_curves(None, data, frame).values()))["members"]
    points = align_observations(frame, members, water_blank_map=data.water_blank_map,
                                temperature_ranges_C=None, temperature_step_C=.5)
    expected, details = fit_curve(points, data, z=1.96, fit_step_C=.5)
    result = inptk.analyze_concentration(
        data, fit_step_C=.5, temperature_step_C=.5, differential=True,
    )
    actual = result.to_dataframe().set_index("temperature_C")
    assert actual.index.tolist() == list(expected)
    for temperature, estimate in expected.items():
        np.testing.assert_allclose(
            actual.loc[temperature, ["concentration", "lower_error", "upper_error"]]
            .to_numpy(dtype=float), estimate, rtol=1e-7,
        )
    assert details["physical_droplets"] == 30
    pd.testing.assert_frame_equal(result.counts.to_dataframe(), frame)
    assert result.settings["fit_step_C"] == .5
    individual = inptk.cumulative_spectrum(
        inptk.frozen_fraction(data), experiment=data, fit_step_C=.5, temperature_step_C=.5
    )
    np.testing.assert_allclose(individual.to_dataframe().concentration, actual.concentration)
    inptk.save(result, tmp_path / "fit.inptk")
    assert inptk.load(tmp_path / "fit.inptk").settings["fit_step_C"] == .5


@pytest.mark.parametrize("step", [0, -1, True, np.inf, "0.5"])
def test_public_workflow_rejects_invalid_fit_spacing(step):
    data = experiment([("A", 20, [1, 5], 1, 50)])
    with pytest.raises(ValueError, match="fit_step_C"):
        inptk.analyze_concentration(data, fit_step_C=step)


def test_average_does_not_silently_ignore_fit_spacing():
    data = experiment([("A", 20, [1, 5], 1, 50)])
    with pytest.raises(ValueError, match="only to method='mle'"):
        inptk.analyze_concentration(data, method="average", fit_step_C=.5)
