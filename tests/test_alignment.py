"""Native states, conditional temperature matching, and physical blank identity."""

import pandas as pd
import pytest

from inptk.alignment import align_observations


def stream(name, temperatures, *, run="R", cycle="1", times=None, counts=None, pictures=None):
    size = len(temperatures)
    data = {
        "sample_id": "A",
        "measurement_id": name,
        "run_id": run,
        "cycle_id": cycle,
        "temperature_C": temperatures,
        "n_total": 20,
        "n_frozen": list(range(size)) if counts is None else counts,
        "observation_id": [f"{name}:{index}" for index in range(size)],
    }
    if times is not None:
        data["time_s"] = times
    if pictures is not None:
        data["picture_id"] = pictures
    return pd.DataFrame(data)


def member(name, run="R", cycle="1"):
    return {"measurement_id": name, "run_id": run, "cycle_id": cycle}


def align(frames, members, *, blanks=None, ranges=None):
    return align_observations(
        pd.concat(frames, ignore_index=True),
        members,
        water_blank_map=blanks or {},
        temperature_ranges_C=ranges,
    )


def test_single_native_stream_retains_holds_jitter_and_rows_without_pictures():
    source = stream("M", [-5, -6, -6, -5.8, -7, -6.5], pictures=[None] * 6)
    original = source.copy(deep=True)
    points = align([source], [member("M")])
    assert [point.temperature_C for point in points] == source.temperature_C.tolist()
    assert [point.point_order for point in points] == list(range(6))
    assert len({point.point_id for point in points}) == 6
    assert all(point.alignment == "native" for point in points)
    actual = pd.concat([point.samples for point in points], ignore_index=True)
    pd.testing.assert_frame_equal(actual, source)
    pd.testing.assert_frame_equal(source, original)


def test_native_order_uses_time_with_stable_ties_not_temperature_order():
    source = stream("M", [-7, -5, -6, -5.8], times=[3, 0, 1, 1])
    points = align([source], [member("M")])
    assert [point.samples.iloc[0].observation_id for point in points] == [
        "M:1",
        "M:2",
        "M:3",
        "M:0",
    ]
    assert [point.temperature_C for point in points] == [-5, -6, -5.8, -7]


def test_synchronous_sample_and_blank_holds_use_each_native_acquisition():
    temperatures, times = [-5, -6, -6, -7], [0, 1, 2, 3]
    frames = [stream(name, temperatures, times=times) for name in ("M", "D", "W")]
    points = align(frames, [member("M"), member("D")], blanks={"M": ["W"], "D": ["W"]})
    assert len(points) == 4
    for index, point in enumerate(points):
        assert point.alignment == "native"
        assert len(point.samples) == 2
        assert len(point.blanks) == 1
        assert point.blanks.iloc[0].observation_id == f"W:{index}"
        assert set(point.samples.n_frozen) == {index}


def test_matching_unique_temperatures_do_not_require_picture_or_time():
    points = align([stream("M", [-5, -6]), stream("D", [-5, -6])], [member("M"), member("D")])
    assert [point.alignment for point in points] == ["native", "native"]


def test_bare_matching_observation_numbers_do_not_prove_synchronous_repeated_temperatures():
    first = stream("M", [-5, -6, -6]).assign(observation_id=["0", "1", "2"])
    second = stream("D", [-5, -6, -6]).assign(observation_id=["0", "1", "2"])
    points = align([first, second], [member("M"), member("D")])
    assert [point.temperature_C for point in points] == [-5, -6]
    assert all(point.alignment == "latest" for point in points)
    assert points[-1].samples.observation_id.tolist() == ["2", "2"]
    assert points[-1].samples.n_total.tolist() == [20, 20]


def test_cross_run_targets_are_original_sample_union_and_move_matching_blank_state():
    frames = [
        stream("M", [-5, -6, -5.8, -7], run="R1", times=[0, 1, 2, 3]),
        stream("W1", [-5, -6, -5.8, -7, -5.5], run="R1", times=[0, 1, 2, 3, 4]),
        stream("D", [-5, -6.2, -7], run="R2", times=[0, 1, 2]),
        stream("W2", [-5, -6.2, -7], run="R2", times=[0, 1, 2]),
    ]
    points = align(
        frames, [member("M", "R1"), member("D", "R2")], blanks={"M": ["W1"], "D": ["W2"]}
    )
    assert [point.temperature_C for point in points] == [-5, -5.8, -6, -6.2, -7]
    point = next(point for point in points if point.temperature_C == -6.2)
    sample = point.samples.set_index("measurement_id").loc["M"]
    blank = point.blanks.set_index("measurement_id").loc["W1"]
    assert sample.temperature_C == blank.temperature_C == -5.8
    assert sample.observation_id == "M:2"
    assert blank.observation_id == "W1:2"  # The later warmer W1:4 is a different acquisition.
    assert point.alignment == "latest"
    assert len(point.blanks) == 2


def test_picture_match_pairs_native_acquisition_when_timestamp_is_absent():
    sample = stream("M", [-5, -6, -7], pictures=["a", "b", "c"])
    blank = stream("W", [-5, -6, -7, -5.5], pictures=["a", "b", "c", "later"])
    points = align([sample, blank], [member("M")], blanks={"M": ["W"]})
    assert points[1].blanks.iloc[0].observation_id == "W:1"
    assert points[1].alignment == "native"


def test_unsynchronized_blank_uses_its_actual_latest_warmer_row():
    sample = stream("M", [-5, -6, -7])
    blank = stream("W", [-4.9, -5.8, -7.1])
    points = align([sample, blank], [member("M")], blanks={"M": ["W"]})
    assert points[1].blanks.iloc[0].temperature_C == -5.8
    assert points[1].blanks.iloc[0].observation_id == "W:1"
    assert points[1].alignment == "latest"
    assert points[-1].blanks.iloc[0].temperature_C == -5.8


def test_target_range_keeps_the_same_warmer_source_observation():
    points = align(
        [stream("M", [-5, -7], run="R1"), stream("D", [-5, -6, -7], run="R2")],
        [member("M", "R1"), member("D", "R2")],
        ranges={"M": {"max_C": -6}},
    )
    assert points[1].temperature_C == -6
    assert points[1].samples.measurement_id.tolist() == ["M", "D"]
    assert points[1].samples.set_index("measurement_id").loc["M", "temperature_C"] == -5
    assert set(points[-1].samples.measurement_id) == {"M", "D"}


def test_excluded_native_points_remain_and_do_not_require_blank_coverage():
    frames = [stream("M", [-5, -6, -7]), stream("W", [-6, -7])]
    points = align(frames, [member("M")], blanks={"M": ["W"]}, ranges={"M": {"max_C": -6}})
    assert [point.temperature_C for point in points] == [-5, -6, -7]
    assert points[0].samples.empty and points[0].blanks.empty
    with pytest.raises(ValueError, match="lacks observed temperature coverage"):
        align(frames, [member("M")], blanks={"M": ["W"]})


def test_sample_support_is_not_extrapolated_and_fully_excluded_targets_remain():
    frames = [stream("M", [-5, -7], run="R1"), stream("D", [-4, -6, -8], run="R2")]
    members = [member("M", "R1"), member("D", "R2")]
    points = align(frames, members)
    assert points[0].samples.measurement_id.tolist() == ["D"]
    assert points[-1].samples.measurement_id.tolist() == ["D"]
    ranges = {name: {"min_C": -6, "max_C": -6} for name in ("M", "D")}
    points = align(frames, members, ranges=ranges)
    assert [point.temperature_C for point in points] == [-4, -5, -6, -7, -8]
    assert [len(point.samples) for point in points] == [0, 0, 2, 0, 0]


def test_repeated_members_and_same_run_multiple_cycles_are_not_pooled():
    source = stream("M", [-5, -6])
    with pytest.raises(ValueError, match="Repeated physical sample member"):
        align([source], [member("M"), member("M")])
    second = stream("M", [-5, -6], cycle="2")
    with pytest.raises(ValueError, match="cannot combine cycles"):
        align([source, second], [member("M"), member("M", cycle="2")])


@pytest.mark.parametrize("names", [("A", "B"), ("B", "A")])
def test_pairwise_blank_acquisition_survives_missing_rows_in_another_sample(names):
    frames = [
        stream("A", [-5, -6, -5.8, -7], times=[0, 1, 2, 3], counts=[0, 8, 10, 16]),
        stream("B", [-5, -6, -7], times=[0, 1, 3], counts=[0, 8, 16]),
        stream("W", [-5, -6, -5.8, -7], times=[0, 1, 2, 3], counts=[0, 1, 7, 10]),
    ]
    points = align(
        frames, [member(name) for name in names], blanks={"A": ["W"], "B": ["W"]},
        ranges={"A": {"max_C": -5.9}},
    )
    point = next(point for point in points if point.temperature_C == -6)
    full = align(frames, [member(name) for name in names], blanks={"A": ["W"], "B": ["W"]})
    original = next(p for p in full if p.temperature_C == -6)
    pd.testing.assert_frame_equal(point.samples, original.samples)
    pd.testing.assert_frame_equal(point.blanks, original.blanks)
    assert set(point.samples.time_s) == {1, 2}
    assert len(point.blanks) == 1
    assert point.blanks.iloc[0].observation_id == "W:2"


@pytest.mark.parametrize("names", [("A", "B"), ("B", "A")])
def test_conflicting_shared_blank_acquisitions_use_latest_target_state_once(names):
    frames = [
        stream("A", [-5, -6, -5.8, -7], times=[0, 1, 2, 3], counts=[0, 8, 10, 16]),
        stream("B", [-5, -6, -7], times=[0, 1, 3], counts=[0, 8, 16]),
        stream("W", [-5, -6, -5.8, -7], times=[0, 1, 2, 3], counts=[0, 1, 7, 10]),
    ]
    points = align(frames, [member(name) for name in names], blanks={"A": ["W"], "B": ["W"]})
    point = next(point for point in points if point.temperature_C == -6)
    assert set(point.samples.time_s) == {1, 2}
    assert len(point.blanks) == 1
    assert point.blanks.iloc[0].observation_id == "W:2"
    assert point.blanks.iloc[0].n_total == 20
    assert point.alignment == "latest"
