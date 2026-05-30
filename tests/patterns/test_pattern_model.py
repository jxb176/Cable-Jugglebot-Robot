from pathlib import Path

import numpy as np
import pytest

from jugglebot.patterns import (
    PatternProject,
    HandKeyframe,
    ThrowEvent,
    ValidationError,
    build_three_ball_cascade_pattern,
    load_pattern_project,
    save_pattern_project,
)


def test_ballistic_flight_hits_throw_and_catch_points():
    project = PatternProject(
        name="single_throw",
        mode="single_run",
        gravity=9.81,
        events=[
            ThrowEvent(
                id="T1",
                ball="A",
                throw_hand="left",
                catch_hand="right",
                throw_time=0.0,
                catch_time=1.0,
                throw_pos=(-0.2, 0.0, 0.8),
                catch_pos=(0.2, 0.0, 0.8),
            )
        ],
    )
    project.validate()

    start = project.ball_position("A", 0.0)
    midpoint = project.ball_position("A", 0.5)
    end = project.ball_position("A", 1.0)

    np.testing.assert_allclose(start, [-0.2, 0.0, 0.8], atol=1e-9)
    np.testing.assert_allclose(end, [0.2, 0.0, 0.8], atol=1e-9)
    assert midpoint[2] > 0.8


def test_ball_follows_catch_hand_between_catch_and_next_throw():
    project = PatternProject(
        name="carry_test",
        mode="single_run",
        gravity=9.81,
        events=[
            ThrowEvent("A1", "A", "left", "right", 0.0, 1.0, (-0.3, 0.0, 1.0), (0.2, -0.1, 0.8)),
            ThrowEvent("A2", "A", "right", "left", 2.0, 3.0, (0.4, 0.2, 1.1), (-0.2, 0.1, 0.8)),
        ],
    )
    project.validate()

    t = 1.5
    hand_pos = project.hand_position("right", t)
    ball_pos = project.ball_position("A", t)

    np.testing.assert_allclose(ball_pos, hand_pos, atol=1e-9)


def test_ball_stays_attached_to_hand_before_first_throw_and_after_last_catch():
    project = PatternProject(
        name="hand_attachment",
        mode="single_run",
        gravity=9.81,
        events=[
            ThrowEvent("B1", "B", "left", "right", 0.0, 1.0, (-0.3, 0.0, 1.0), (0.2, 0.0, 0.8)),
            ThrowEvent("C1", "C", "left", "left", 1.5, 2.5, (-0.4, 0.0, 1.0), (-0.25, 0.0, 0.78)),
        ],
    )
    project.validate()

    before_throw = project.ball_position("C", 0.25)
    after_catch = project.ball_position("B", 2.0)

    np.testing.assert_allclose(before_throw, project.hand_position("left", 0.25), atol=1e-9)
    np.testing.assert_allclose(after_catch, project.hand_position("right", 2.0), atol=1e-9)


def test_authored_hand_trajectory_shapes_interpolation_between_event_anchors():
    explicit_velocity = (0.18, -0.12, 0.26)
    project = PatternProject(
        name="hand_path_shape",
        mode="single_run",
        gravity=9.81,
        events=[
            ThrowEvent("A1", "A", "right", "right", 0.0, 1.0, (0.3, 0.0, 1.0), (0.2, 0.0, 0.8)),
            ThrowEvent("A2", "A", "right", "right", 2.0, 3.0, (0.3, 0.0, 1.0), (0.2, 0.0, 0.8)),
        ],
        hand_trajectories={
            "right": [
                HandKeyframe("R_mid", "right", 1.5, (0.55, 0.25, 1.25), "cubic", velocity=explicit_velocity),
            ]
        },
    )
    project.validate()

    np.testing.assert_allclose(project.hand_position("right", 1.5), [0.55, 0.25, 1.25], atol=1e-9)
    np.testing.assert_allclose(project.hand_state("right", 1.5).velocity, explicit_velocity, atol=1e-9)
    assert project.to_dict()["hands"]["right"][0]["spline_to_next"] == "cubic"
    assert project.to_dict()["hands"]["right"][0]["velocity"] == list(explicit_velocity)


def test_missing_authored_hand_velocity_is_materialized_after_validation():
    project = PatternProject(
        name="implicit_waypoint_velocity",
        mode="single_run",
        gravity=9.81,
        events=[
            ThrowEvent("A1", "A", "right", "right", 0.0, 1.0, (0.3, 0.0, 1.0), (0.2, 0.0, 0.8)),
            ThrowEvent("A2", "A", "right", "right", 2.0, 3.0, (0.32, 0.0, 1.0), (0.22, 0.0, 0.8)),
        ],
        hand_trajectories={
            "right": [
                HandKeyframe("R_mid", "right", 1.5, (0.5, 0.2, 1.2), "quintic"),
            ]
        },
    )

    project.validate()

    keyframe = project.sorted_hand_trajectory("right")[0]
    assert keyframe.velocity is not None
    np.testing.assert_allclose(keyframe.velocity, project.hand_state("right", 1.5).velocity, atol=1e-9)
    assert "velocity" in project.to_dict()["hands"]["right"][0]


def test_bspline_geometry_is_independent_from_path_speed():
    def make_project(speed0: float, speed1: float) -> PatternProject:
        return PatternProject(
            name="bspline_path_speed",
            mode="single_run",
            gravity=9.81,
            hand_trajectories={
                "right": [
                    HandKeyframe(
                        "R0",
                        "right",
                        0.0,
                        (0.0, 0.0, 0.0),
                        "bspline",
                        velocity=(1.0, 0.0, 0.0),
                        path_speed=speed0,
                        bspline_degree=3,
                        bspline_control_points=6,
                    ),
                    HandKeyframe(
                        "R1",
                        "right",
                        2.0,
                        (1.0, 1.0, 0.0),
                        "quintic",
                        velocity=(0.0, 1.0, 0.0),
                        path_speed=speed1,
                    ),
                ]
            },
        )

    slow = make_project(0.25, 0.25)
    fast = make_project(0.75, 0.75)
    slow.validate()
    fast.validate()

    slow_segment = slow.hand_bspline_segment("right", 0.0, (0.0, 0.0, 0.0))
    fast_segment = fast.hand_bspline_segment("right", 0.0, (0.0, 0.0, 0.0))

    assert slow_segment is not None
    assert fast_segment is not None
    np.testing.assert_allclose(slow_segment["control_points"], fast_segment["control_points"], atol=1e-9)
    np.testing.assert_allclose(slow_segment["curve"], fast_segment["curve"], atol=1e-9)
    assert slow.hand_keyframe_path_speed(slow.sorted_hand_trajectory("right")[0]) == pytest.approx(0.25)
    assert fast.hand_keyframe_path_speed(fast.sorted_hand_trajectory("right")[0]) == pytest.approx(0.75)
    assert not np.allclose(slow.hand_state("right", 1.0).velocity, fast.hand_state("right", 1.0).velocity, atol=1e-6)


def test_missing_bspline_path_speed_is_materialized_after_validation():
    project = PatternProject(
        name="implicit_bspline_speed",
        mode="single_run",
        gravity=9.81,
        hand_trajectories={
            "right": [
                HandKeyframe("R0", "right", 0.0, (0.0, 0.0, 0.0), "bspline"),
                HandKeyframe("R1", "right", 1.0, (1.0, 0.0, 0.0), "quintic"),
            ]
        },
    )

    project.validate()

    keyframe = project.sorted_hand_trajectory("right")[0]
    assert keyframe.path_speed == pytest.approx(1.0)
    assert "path_speed" in project.to_dict()["hands"]["right"][0]


def test_bspline_segment_exposes_configured_degree_and_control_count():
    project = PatternProject(
        name="bspline_degree_and_count",
        mode="single_run",
        gravity=9.81,
        hand_trajectories={
            "right": [
                HandKeyframe(
                    "R0",
                    "right",
                    0.0,
                    (0.0, 0.0, 0.0),
                    "bspline",
                    velocity=(1.0, 0.4, 0.0),
                    path_speed=0.5,
                    bspline_degree=5,
                    bspline_control_points=7,
                ),
                HandKeyframe(
                    "R1",
                    "right",
                    1.5,
                    (1.2, 0.8, 0.4),
                    "quintic",
                    velocity=(0.0, 1.0, 0.0),
                    path_speed=0.5,
                ),
            ]
        },
    )

    project.validate()

    segment = project.hand_bspline_segment("right", 0.0, (0.0, 0.0, 0.0))
    assert segment is not None
    assert segment["degree"] == 5
    assert len(segment["control_points"]) == 7


def test_authored_hand_trajectory_conflict_with_throw_anchor_is_rejected():
    project = PatternProject(
        name="bad_hand_anchor",
        mode="single_run",
        gravity=9.81,
        events=[
            ThrowEvent("A1", "A", "right", "right", 0.0, 1.0, (0.3, 0.0, 1.0), (0.2, 0.0, 0.8)),
        ],
        hand_trajectories={
            "right": [
                HandKeyframe("R0", "right", 0.0, (0.9, 0.0, 1.0), "quintic"),
            ]
        },
    )

    with pytest.raises(ValidationError, match="right hand has conflicting positions"):
        project.validate()


def test_looped_pattern_repeats_after_one_period():
    project = build_three_ball_cascade_pattern()

    state_a = project.sample(0.9)
    state_b = project.sample(0.9 + project.loop_period)

    for hand in state_a.hand_positions:
        np.testing.assert_allclose(state_a.hand_positions[hand], state_b.hand_positions[hand], atol=1e-9)
    for ball in state_a.ball_positions:
        np.testing.assert_allclose(state_a.ball_positions[ball], state_b.ball_positions[ball], atol=1e-9)


def test_throw_and_catch_hand_kinematics_match_ball_conditions():
    project = PatternProject(
        name="throw_catch_conditions",
        mode="single_run",
        gravity=9.81,
        events=[
            ThrowEvent(
                "A1",
                "A",
                "right",
                "right",
                0.0,
                1.0,
                (0.3, 0.1, 1.0),
                (0.1, -0.1, 0.8),
                catch_velocity_scale=0.4,
            ),
            ThrowEvent(
                "A2",
                "A",
                "right",
                "right",
                2.0,
                3.0,
                (0.32, 0.1, 1.02),
                (0.12, -0.08, 0.82),
                catch_velocity_scale=0.4,
            ),
        ],
    )
    project.validate()

    throw_state = project.hand_state("right", 0.0)
    catch_state = project.hand_state("right", 1.0)
    throw_velocity = project._ball_velocity(project.sorted_events()[0], at="throw")
    catch_velocity = 0.4 * project._ball_velocity(project.sorted_events()[0], at="catch")

    np.testing.assert_allclose(throw_state.position, [0.3, 0.1, 1.0], atol=1e-9)
    np.testing.assert_allclose(throw_state.velocity, throw_velocity, atol=1e-9)
    np.testing.assert_allclose(throw_state.acceleration, [0.0, 0.0, 0.0], atol=1e-9)

    np.testing.assert_allclose(catch_state.position, [0.1, -0.1, 0.8], atol=1e-9)
    np.testing.assert_allclose(catch_state.velocity, catch_velocity, atol=1e-9)
    np.testing.assert_allclose(catch_state.acceleration, [0.0, 0.0, 0.0], atol=1e-9)


def test_loop_validation_rejects_ball_overlap_across_boundary():
    project = PatternProject(
        name="bad_loop",
        mode="loop",
        loop_period=1.5,
        gravity=9.81,
        events=[
            ThrowEvent("A1", "A", "left", "right", 0.0, 2.0, (-0.2, 0.0, 0.9), (0.2, 0.0, 0.9)),
        ],
    )

    with pytest.raises(ValidationError, match="overlaps across the loop boundary"):
        project.validate()


def test_validation_rejects_holding_two_balls_in_one_hand():
    project = PatternProject(
        name="bad_hand_capacity",
        mode="single_run",
        gravity=9.81,
        events=[
            ThrowEvent("A1", "A", "right", "right", 0.0, 1.0, (0.3, 0.1, 1.0), (0.2, 0.0, 0.8)),
            ThrowEvent("B1", "B", "right", "right", 0.5, 1.5, (0.28, 0.1, 0.96), (0.18, 0.0, 0.76)),
            ThrowEvent("A2", "A", "right", "right", 2.0, 3.0, (0.3, 0.1, 1.0), (0.2, 0.0, 0.8)),
            ThrowEvent("B2", "B", "right", "right", 2.5, 3.5, (0.28, 0.1, 0.96), (0.18, 0.0, 0.76)),
        ],
    )

    with pytest.raises(ValidationError, match="would hold more than one ball at once"):
        project.validate()


def test_yaml_round_trip_preserves_project(tmp_path):
    project = build_three_ball_cascade_pattern()
    project.hand_trajectories["left"] = [HandKeyframe("L_mid", "left", 1.25, (-0.45, 0.15, 1.18), "quintic")]
    path = tmp_path / "cascade.yaml"

    save_pattern_project(project, path)
    loaded = load_pattern_project(path)

    assert loaded.to_dict() == project.to_dict()


def test_checked_in_three_ball_example_matches_builtin_sample():
    path = Path("src/jugglebot/patterns/examples/three_ball_cascade.yaml")
    loaded = load_pattern_project(path)

    assert loaded.to_dict() == build_three_ball_cascade_pattern().to_dict()
    assert len(loaded.sorted_hand_trajectory("left")) > 0
    assert len(loaded.sorted_hand_trajectory("right")) > 0
    assert all(point.velocity is not None for point in loaded.sorted_hand_trajectory("left"))
    assert all(point.velocity is not None for point in loaded.sorted_hand_trajectory("right"))


@pytest.mark.parametrize(
    ("path_str", "expected_balls", "expected_hand"),
    [
        ("src/jugglebot/patterns/examples/one_ball_one_hand.yaml", ["A"], "right"),
        ("src/jugglebot/patterns/examples/two_balls_one_hand.yaml", ["A", "B"], "right"),
    ],
)
def test_checked_in_single_hand_examples_load_and_validate(path_str, expected_balls, expected_hand):
    project = load_pattern_project(Path(path_str))

    assert project.mode == "loop"
    assert project.ball_ids() == expected_balls
    assert {event.throw_hand for event in project.events} == {expected_hand}
    assert {event.catch_hand for event in project.events} == {expected_hand}
    assert len(project.sorted_hand_trajectory(expected_hand)) > 0
    assert all(point.velocity is not None for point in project.sorted_hand_trajectory(expected_hand))
