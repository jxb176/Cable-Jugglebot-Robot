"""Unconstrained juggling pattern model for interactive authoring."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, Iterable, Literal, Optional, Sequence

import numpy as np
import yaml

HandName = Literal["left", "right"]
SplineName = Literal["cubic", "quintic", "bspline"]
HAND_NAMES: tuple[HandName, HandName] = ("left", "right")
_EPS = 1e-9


class ValidationError(ValueError):
    """Raised when a pattern project is internally inconsistent."""


def _as_hand_name(value: str) -> HandName:
    norm = str(value).strip().lower().replace("-", "_")
    if norm not in HAND_NAMES:
        raise ValidationError(f"hand must be one of {HAND_NAMES}; got {value!r}")
    return norm  # type: ignore[return-value]


def _as_vec3(value: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size != 3:
        raise ValidationError(f"expected a 3-vector; got {value!r}")
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _vec3_array(value: Sequence[float]) -> np.ndarray:
    return np.asarray(value, dtype=float).reshape(3)


def _normalize_vec3(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm <= _EPS:
        return np.zeros(3, dtype=float)
    return value / norm


def _interp_vec3(t: float, t0: float, p0: np.ndarray, t1: float, p1: np.ndarray) -> np.ndarray:
    if t1 <= t0 + _EPS:
        return p1.copy()
    alpha = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
    return p0 + alpha * (p1 - p0)


def _finite_difference_velocity(
    prev_time: Optional[float],
    prev_pos: Optional[np.ndarray],
    cur_time: float,
    cur_pos: np.ndarray,
    next_time: Optional[float],
    next_pos: Optional[np.ndarray],
) -> np.ndarray:
    if prev_time is not None and next_time is not None and next_time > prev_time + _EPS:
        return (next_pos - prev_pos) / (next_time - prev_time)  # type: ignore[operator]
    if next_time is not None and next_time > cur_time + _EPS:
        return (next_pos - cur_pos) / (next_time - cur_time)  # type: ignore[operator]
    if prev_time is not None and cur_time > prev_time + _EPS:
        return (cur_pos - prev_pos) / (cur_time - prev_time)  # type: ignore[operator]
    return np.zeros(3, dtype=float)


def _cubic_hermite_state(
    u: float,
    duration: float,
    p0: np.ndarray,
    v0: np.ndarray,
    p1: np.ndarray,
    v1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h00 = 2.0 * u**3 - 3.0 * u**2 + 1.0
    h10 = u**3 - 2.0 * u**2 + u
    h01 = -2.0 * u**3 + 3.0 * u**2
    h11 = u**3 - u**2

    dh00 = 6.0 * u**2 - 6.0 * u
    dh10 = 3.0 * u**2 - 4.0 * u + 1.0
    dh01 = -6.0 * u**2 + 6.0 * u
    dh11 = 3.0 * u**2 - 2.0 * u

    ddh00 = 12.0 * u - 6.0
    ddh10 = 6.0 * u - 4.0
    ddh01 = -12.0 * u + 6.0
    ddh11 = 6.0 * u - 2.0

    pos = h00 * p0 + h10 * duration * v0 + h01 * p1 + h11 * duration * v1
    vel = (dh00 * p0 + dh10 * duration * v0 + dh01 * p1 + dh11 * duration * v1) / duration
    acc = (ddh00 * p0 + ddh10 * duration * v0 + ddh01 * p1 + ddh11 * duration * v1) / (duration * duration)
    return pos, vel, acc


def _quintic_hermite_state(
    u: float,
    duration: float,
    p0: np.ndarray,
    v0: np.ndarray,
    a0: np.ndarray,
    p1: np.ndarray,
    v1: np.ndarray,
    a1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h00 = 1.0 - 10.0 * u**3 + 15.0 * u**4 - 6.0 * u**5
    h10 = u - 6.0 * u**3 + 8.0 * u**4 - 3.0 * u**5
    h20 = 0.5 * u**2 - 1.5 * u**3 + 1.5 * u**4 - 0.5 * u**5
    h01 = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
    h11 = -4.0 * u**3 + 7.0 * u**4 - 3.0 * u**5
    h21 = 0.5 * u**3 - u**4 + 0.5 * u**5

    dh00 = -30.0 * u**2 + 60.0 * u**3 - 30.0 * u**4
    dh10 = 1.0 - 18.0 * u**2 + 32.0 * u**3 - 15.0 * u**4
    dh20 = u - 4.5 * u**2 + 6.0 * u**3 - 2.5 * u**4
    dh01 = 30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4
    dh11 = -12.0 * u**2 + 28.0 * u**3 - 15.0 * u**4
    dh21 = 1.5 * u**2 - 4.0 * u**3 + 2.5 * u**4

    ddh00 = -60.0 * u + 180.0 * u**2 - 120.0 * u**3
    ddh10 = -36.0 * u + 96.0 * u**2 - 60.0 * u**3
    ddh20 = 1.0 - 9.0 * u + 18.0 * u**2 - 10.0 * u**3
    ddh01 = 60.0 * u - 180.0 * u**2 + 120.0 * u**3
    ddh11 = -24.0 * u + 84.0 * u**2 - 60.0 * u**3
    ddh21 = 3.0 * u - 12.0 * u**2 + 10.0 * u**3

    pos = (
        h00 * p0
        + h10 * duration * v0
        + h20 * duration * duration * a0
        + h01 * p1
        + h11 * duration * v1
        + h21 * duration * duration * a1
    )
    vel = (
        dh00 * p0
        + dh10 * duration * v0
        + dh20 * duration * duration * a0
        + dh01 * p1
        + dh11 * duration * v1
        + dh21 * duration * duration * a1
    ) / duration
    acc = (
        ddh00 * p0
        + ddh10 * duration * v0
        + ddh20 * duration * duration * a0
        + ddh01 * p1
        + ddh11 * duration * v1
        + ddh21 * duration * duration * a1
    ) / (duration * duration)
    return pos, vel, acc


def _cubic_bezier_point(tau: float, p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    omt = 1.0 - tau
    return (
        (omt**3) * p0
        + 3.0 * (omt**2) * tau * p1
        + 3.0 * omt * (tau**2) * p2
        + (tau**3) * p3
    )


def _quintic_scalar_progress_state(
    u: float,
    duration: float,
    length: float,
    speed0: float,
    speed1: float,
) -> tuple[float, float, float]:
    pos, vel, acc = _quintic_hermite_state(
        u,
        duration,
        np.array([0.0], dtype=float),
        np.array([float(speed0)], dtype=float),
        np.array([0.0], dtype=float),
        np.array([float(length)], dtype=float),
        np.array([float(speed1)], dtype=float),
        np.array([0.0], dtype=float),
    )
    return float(pos[0]), float(vel[0]), float(acc[0])


def _clamped_uniform_knots(control_count: int, degree: int) -> np.ndarray:
    if control_count < degree + 1:
        raise ValidationError(f"need at least {degree + 1} control points for degree {degree}")
    interior_count = control_count - degree - 1
    knots = np.zeros(control_count + degree + 1, dtype=float)
    if interior_count > 0:
        interior = np.linspace(0.0, 1.0, interior_count + 2, dtype=float)[1:-1]
        knots[degree + 1 : degree + 1 + interior_count] = interior
    knots[-(degree + 1) :] = 1.0
    return knots


def _find_knot_span(knots: np.ndarray, degree: int, control_count: int, u: float) -> int:
    if u >= knots[control_count] - _EPS:
        return control_count - 1
    low = degree
    high = control_count
    mid = (low + high) // 2
    while u < knots[mid] or u >= knots[mid + 1]:
        if u < knots[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def _de_boor(control_points: np.ndarray, degree: int, knots: np.ndarray, u: float) -> np.ndarray:
    count = len(control_points)
    if count == 1:
        return control_points[0].copy()
    span = _find_knot_span(knots, degree, count, u)
    d = [control_points[span - degree + j].copy() for j in range(degree + 1)]
    for r in range(1, degree + 1):
        for j in range(degree, r - 1, -1):
            idx = span - degree + j
            denom = knots[idx + degree + 1 - r] - knots[idx]
            alpha = 0.0 if abs(denom) <= _EPS else (u - knots[idx]) / denom
            d[j] = (1.0 - alpha) * d[j - 1] + alpha * d[j]
    return d[degree]


def _derivative_control_polygon(
    control_points: np.ndarray,
    degree: int,
    knots: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray]:
    if degree <= 0 or len(control_points) <= 1:
        return np.zeros((0, 3), dtype=float), 0, np.zeros(0, dtype=float)
    derived: list[np.ndarray] = []
    for index in range(len(control_points) - 1):
        denom = knots[index + degree + 1] - knots[index + 1]
        scale = 0.0 if abs(denom) <= _EPS else degree / denom
        derived.append(scale * (control_points[index + 1] - control_points[index]))
    return np.asarray(derived, dtype=float), degree - 1, knots[1:-1].copy()


def _arc_length_lookup(
    control_points: np.ndarray,
    degree: int,
    knots: np.ndarray,
    derivative_control_points: np.ndarray,
    derivative_degree: int,
    derivative_knots: np.ndarray,
    samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    u_grid = np.linspace(0.0, 1.0, max(32, int(samples)), dtype=float)
    speed_u = np.array(
        [
            float(np.linalg.norm(_de_boor(derivative_control_points, derivative_degree, derivative_knots, float(u))))
            if len(derivative_control_points) > 0
            else 0.0
            for u in u_grid
        ],
        dtype=float,
    )
    cumulative = np.zeros_like(u_grid)
    if len(u_grid) > 1:
        du = np.diff(u_grid)
        cumulative[1:] = np.cumsum(0.5 * (speed_u[:-1] + speed_u[1:]) * du)
    return u_grid, cumulative


@dataclass
class ThrowEvent:
    """A single ball flight between a throw and catch hand point."""

    id: str
    ball: str
    throw_hand: HandName
    catch_hand: HandName
    throw_time: float
    catch_time: float
    throw_pos: tuple[float, float, float]
    catch_pos: tuple[float, float, float]
    catch_velocity_scale: float = 0.35

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        self.ball = str(self.ball).strip()
        if not self.id:
            raise ValidationError("event id must be non-empty")
        if not self.ball:
            raise ValidationError("ball id must be non-empty")
        self.throw_hand = _as_hand_name(self.throw_hand)
        self.catch_hand = _as_hand_name(self.catch_hand)
        self.throw_time = float(self.throw_time)
        self.catch_time = float(self.catch_time)
        self.throw_pos = _as_vec3(self.throw_pos)
        self.catch_pos = _as_vec3(self.catch_pos)
        self.catch_velocity_scale = float(self.catch_velocity_scale)
        if self.catch_velocity_scale < 0.0:
            raise ValidationError("catch_velocity_scale must be >= 0")

    @property
    def duration(self) -> float:
        return float(self.catch_time - self.throw_time)

    def to_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "ball": self.ball,
            "throw_hand": self.throw_hand,
            "catch_hand": self.catch_hand,
            "throw_time": float(self.throw_time),
            "catch_time": float(self.catch_time),
            "throw_pos": list(self.throw_pos),
            "catch_pos": list(self.catch_pos),
            "catch_velocity_scale": float(self.catch_velocity_scale),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ThrowEvent":
        return cls(
            id=str(data.get("id", "")),
            ball=str(data.get("ball", "")),
            throw_hand=str(data.get("throw_hand", "left")),
            catch_hand=str(data.get("catch_hand", "right")),
            throw_time=float(data.get("throw_time", 0.0)),
            catch_time=float(data.get("catch_time", 0.0)),
            throw_pos=data.get("throw_pos", (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
            catch_pos=data.get("catch_pos", (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
            catch_velocity_scale=float(data.get("catch_velocity_scale", 0.35)),
        )


@dataclass
class HandKeyframe:
    """Authored hand waypoint used to shape the unconstrained hand path."""

    id: str
    hand: HandName
    time: float
    pos: tuple[float, float, float]
    spline_to_next: SplineName = "quintic"
    velocity: tuple[float, float, float] | None = None
    path_speed: float | None = None
    bspline_degree: int | None = None
    bspline_control_points: int | None = None

    def __post_init__(self) -> None:
        self.id = str(self.id).strip()
        if not self.id:
            raise ValidationError("hand keyframe id must be non-empty")
        self.hand = _as_hand_name(self.hand)
        self.time = float(self.time)
        self.pos = _as_vec3(self.pos)
        spline = str(self.spline_to_next).strip().lower()
        if spline not in {"cubic", "quintic", "bspline"}:
            raise ValidationError(f"spline_to_next must be 'cubic', 'quintic', or 'bspline'; got {self.spline_to_next!r}")
        self.spline_to_next = spline  # type: ignore[assignment]
        if self.velocity is not None:
            self.velocity = _as_vec3(self.velocity)
        if self.path_speed is not None:
            self.path_speed = float(self.path_speed)
            if self.path_speed < 0.0:
                raise ValidationError("path_speed must be >= 0")
        if self.spline_to_next == "bspline":
            self.bspline_degree = 3 if self.bspline_degree is None else int(self.bspline_degree)
            if self.bspline_degree not in {3, 5}:
                raise ValidationError("bspline_degree must be 3 or 5")
            default_control_points = max(self.bspline_degree + 1, 6)
            self.bspline_control_points = (
                default_control_points if self.bspline_control_points is None else int(self.bspline_control_points)
            )
            if self.bspline_control_points < self.bspline_degree + 1:
                raise ValidationError(
                    "bspline_control_points must be at least degree + 1 "
                    f"({self.bspline_degree + 1} for degree {self.bspline_degree})"
                )
        else:
            self.bspline_degree = None
            self.bspline_control_points = None

    def to_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "id": self.id,
            "time": float(self.time),
            "pos": list(self.pos),
            "spline_to_next": self.spline_to_next,
        }
        if self.velocity is not None:
            data["velocity"] = list(self.velocity)
        if self.path_speed is not None:
            data["path_speed"] = float(self.path_speed)
        if self.spline_to_next == "bspline":
            data["bspline_degree"] = int(self.bspline_degree if self.bspline_degree is not None else 3)
            data["bspline_control_points"] = int(
                self.bspline_control_points if self.bspline_control_points is not None else 6
            )
        return data

    @classmethod
    def from_dict(cls, hand: HandName, data: Dict[str, object]) -> "HandKeyframe":
        return cls(
            id=str(data.get("id", "")),
            hand=hand,
            time=float(data.get("time", 0.0)),
            pos=data.get("pos", (0.0, 0.0, 0.0)),  # type: ignore[arg-type]
            spline_to_next=str(data.get("spline_to_next", "quintic")),
            velocity=data.get("velocity"),  # type: ignore[arg-type]
            path_speed=data.get("path_speed"),  # type: ignore[arg-type]
            bspline_degree=data.get("bspline_degree"),  # type: ignore[arg-type]
            bspline_control_points=data.get("bspline_control_points"),  # type: ignore[arg-type]
        )


@dataclass
class HandSplineSample:
    """Hand kinematics sampled from the authored spline path."""

    time: float
    position: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray


@dataclass
class _HandNode:
    time: float
    pos: np.ndarray
    label: str
    velocity: np.ndarray | None = None
    acceleration: np.ndarray | None = None
    spline_to_next: SplineName = "quintic"
    path_speed: float | None = None
    bspline_degree: int | None = None
    bspline_control_points: int | None = None


@dataclass
class _PreparedBSplineSegment:
    control_points: np.ndarray
    degree: int
    knots: np.ndarray
    first_derivative_control_points: np.ndarray
    first_derivative_degree: int
    first_derivative_knots: np.ndarray
    second_derivative_control_points: np.ndarray | None
    second_derivative_degree: int | None
    second_derivative_knots: np.ndarray | None
    arc_u: np.ndarray
    arc_length: np.ndarray
    total_length: float


@dataclass
class SampleState:
    """Instantaneous hand and ball positions."""

    time: float
    hand_positions: Dict[HandName, np.ndarray]
    ball_positions: Dict[str, np.ndarray]


@dataclass
class PatternProject:
    """Editable juggling pattern definition for unconstrained previewing."""

    name: str = "untitled_pattern"
    mode: Literal["loop", "single_run"] = "loop"
    loop_period: float = 3.5
    gravity: float = 9.81
    events: list[ThrowEvent] = field(default_factory=list)
    hand_trajectories: Dict[HandName, list[HandKeyframe]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower().replace("-", "_")
        if mode not in {"loop", "single_run"}:
            raise ValidationError(f"mode must be 'loop' or 'single_run'; got {self.mode!r}")
        self.mode = mode  # type: ignore[assignment]
        self.name = str(self.name).strip() or "untitled_pattern"
        self.loop_period = float(self.loop_period)
        self.gravity = float(self.gravity)
        self.events = [event if isinstance(event, ThrowEvent) else ThrowEvent.from_dict(event) for event in self.events]
        trajectories: Dict[HandName, list[HandKeyframe]] = {hand: [] for hand in HAND_NAMES}
        for raw_hand, raw_keyframes in dict(self.hand_trajectories or {}).items():
            hand = _as_hand_name(raw_hand)
            if not isinstance(raw_keyframes, list):
                raise ValidationError(f"hand trajectory for {hand} must be a list")
            trajectories[hand] = [
                keyframe if isinstance(keyframe, HandKeyframe) else HandKeyframe.from_dict(hand, keyframe)
                for keyframe in raw_keyframes
            ]
        self.hand_trajectories = trajectories

    @property
    def is_loop(self) -> bool:
        return self.mode == "loop"

    def copy(self) -> "PatternProject":
        return PatternProject.from_dict(self.to_dict())

    def sorted_events(self) -> list[ThrowEvent]:
        return sorted(self.events, key=lambda event: (event.throw_time, event.catch_time, event.id))

    def ball_ids(self) -> list[str]:
        return sorted({event.ball for event in self.events})

    def hand_waypoint_ids(self, hand: HandName) -> list[str]:
        hand = _as_hand_name(hand)
        return [keyframe.id for keyframe in self.sorted_hand_trajectory(hand)]

    def sorted_hand_trajectory(self, hand: HandName) -> list[HandKeyframe]:
        hand = _as_hand_name(hand)
        return sorted(self.hand_trajectories.get(hand, []), key=lambda keyframe: (keyframe.time, keyframe.id))

    def timeline_duration(self) -> float:
        if self.is_loop:
            return max(self.loop_period, 1e-6)
        return max(self.sequence_end_time(), 1e-6)

    def sequence_end_time(self) -> float:
        event_end = 0.0
        if self.events:
            event_end = max(max(event.throw_time, event.catch_time) for event in self.events)
        hand_end = 0.0
        for hand in HAND_NAMES:
            keyframes = self.hand_trajectories.get(hand, [])
            if keyframes:
                hand_end = max(hand_end, max(keyframe.time for keyframe in keyframes))
        return max(event_end, hand_end)

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "mode": self.mode,
            "loop_period": float(self.loop_period),
            "gravity": float(self.gravity),
            "events": [event.to_dict() for event in self.sorted_events()],
            "hands": {
                hand: [keyframe.to_dict() for keyframe in self.sorted_hand_trajectory(hand)]
                for hand in HAND_NAMES
                if self.hand_trajectories.get(hand)
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "PatternProject":
        raw_events = data.get("events", [])
        if raw_events is None:
            raw_events = []
        if not isinstance(raw_events, list):
            raise ValidationError("project 'events' must be a list")
        return cls(
            name=str(data.get("name", "untitled_pattern")),
            mode=str(data.get("mode", "loop")),
            loop_period=float(data.get("loop_period", 3.5)),
            gravity=float(data.get("gravity", 9.81)),
            events=[ThrowEvent.from_dict(event) for event in raw_events],  # type: ignore[arg-type]
            hand_trajectories=data.get("hands", {}),  # type: ignore[arg-type]
        )

    def validate(self) -> None:
        if self.gravity <= 0.0:
            raise ValidationError("gravity must be > 0")
        if self.is_loop and self.loop_period <= 0.0:
            raise ValidationError("loop_period must be > 0 for loop mode")

        seen_ids: set[str] = set()
        by_ball: Dict[str, list[ThrowEvent]] = {}

        for hand in HAND_NAMES:
            seen_hand_ids: set[str] = set()
            for keyframe in self.sorted_hand_trajectory(hand):
                if keyframe.id in seen_hand_ids:
                    raise ValidationError(f"duplicate hand keyframe id for {hand}: {keyframe.id}")
                seen_hand_ids.add(keyframe.id)
                if keyframe.time < 0.0:
                    raise ValidationError(f"hand keyframe {keyframe.id} for {hand} has negative time")
                if self.is_loop and keyframe.time > self.loop_period + _EPS:
                    raise ValidationError(
                        f"hand keyframe {keyframe.id} for {hand} exceeds loop_period; "
                        f"looped hand keyframes must be within [0, loop_period]"
                    )

        for event in self.sorted_events():
            if event.id in seen_ids:
                raise ValidationError(f"duplicate event id: {event.id}")
            seen_ids.add(event.id)

            if event.throw_time < 0.0:
                raise ValidationError(f"event {event.id} has negative throw_time")
            if event.catch_time <= event.throw_time + _EPS:
                raise ValidationError(f"event {event.id} must have catch_time > throw_time")

            by_ball.setdefault(event.ball, []).append(event)

        for ball, events in by_ball.items():
            ordered = sorted(events, key=lambda event: (event.throw_time, event.catch_time, event.id))
            for current, nxt in zip(ordered, ordered[1:]):
                if nxt.throw_time < current.catch_time - _EPS:
                    raise ValidationError(
                        f"ball {ball} overlaps itself between {current.id} and {nxt.id}; "
                        f"next throw must be at or after the prior catch"
                    )
                if nxt.throw_hand != current.catch_hand:
                    raise ValidationError(
                        f"ball {ball} continuity breaks between {current.id} and {nxt.id}; "
                        f"next throw hand must equal prior catch hand"
                    )

            if self.is_loop and ordered:
                first = ordered[0]
                last = ordered[-1]
                next_throw_time = first.throw_time + self.loop_period
                if next_throw_time < last.catch_time - _EPS:
                    raise ValidationError(
                        f"ball {ball} overlaps across the loop boundary; "
                        f"loop_period must be long enough to finish the last catch"
                    )
                if first.throw_hand != last.catch_hand:
                    raise ValidationError(
                        f"ball {ball} loop continuity breaks; "
                        f"first throw hand must equal last catch hand"
                    )

        for hand in HAND_NAMES:
            self._hand_nodes(hand)

        self._validate_hand_capacity(by_ball)
        self._materialize_missing_hand_keyframe_parameters()

    def hand_keyframes(self, hand: HandName) -> list[tuple[float, np.ndarray]]:
        hand = _as_hand_name(hand)
        return [(node.time, node.pos.copy()) for node in self._hand_nodes(hand)]

    def hand_position(self, hand: HandName, time_s: float) -> np.ndarray:
        return self.hand_state(hand, time_s).position.copy()

    def hand_keyframe_tangent(self, keyframe: HandKeyframe) -> np.ndarray:
        hand = _as_hand_name(keyframe.hand)
        nodes = self._hand_nodes(hand)
        index = self._find_hand_node_index(nodes, keyframe.time, keyframe.pos)
        if index is None:
            return _normalize_vec3(self.hand_state(hand, keyframe.time).velocity)
        return self._node_tangent_direction(nodes, index)

    def hand_keyframe_path_speed(self, keyframe: HandKeyframe) -> float:
        hand = _as_hand_name(keyframe.hand)
        nodes = self._hand_nodes(hand)
        index = self._find_hand_node_index(nodes, keyframe.time, keyframe.pos)
        if index is None:
            return float(np.linalg.norm(self.hand_state(hand, keyframe.time).velocity))
        return self._node_path_speed(nodes, index)

    def hand_bspline_segment(
        self,
        hand: HandName,
        start_time: float,
        start_pos: Sequence[float],
        samples: int = 96,
    ) -> Dict[str, object] | None:
        hand = _as_hand_name(hand)
        nodes = self._hand_nodes(hand)
        index = self._find_hand_node_index(nodes, start_time, start_pos)
        if index is None or index + 1 >= len(nodes):
            return None
        start = nodes[index]
        if start.spline_to_next != "bspline":
            return None
        prepared = self._prepare_bspline_segment(nodes, index)
        u_samples = np.linspace(0.0, 1.0, max(8, int(samples)), dtype=float)
        curve = np.vstack([self._evaluate_bspline_geometry(prepared, float(u))[0] for u in u_samples])
        return {
            "degree": prepared.degree,
            "control_points": prepared.control_points.copy(),
            "curve": curve,
            "length": float(prepared.total_length),
            "start_time": float(start.time),
            "end_time": float(nodes[index + 1].time),
        }

    def hand_state(self, hand: HandName, time_s: float) -> HandSplineSample:
        hand = _as_hand_name(hand)
        nodes = self._hand_nodes(hand)
        if not nodes:
            zero = np.zeros(3, dtype=float)
            return HandSplineSample(time=float(time_s), position=zero.copy(), velocity=zero.copy(), acceleration=zero.copy())

        t = float(time_s)
        if self.is_loop:
            nodes = self._expand_nodes(nodes, t)
        return self._evaluate_hand_nodes(nodes, t)

    def ball_position(self, ball: str, time_s: float) -> np.ndarray | None:
        ball = str(ball).strip()
        if not ball:
            return None

        occurrences = self._ball_occurrences(ball, float(time_s))
        if not occurrences:
            return None

        t = float(time_s)
        prev_index = -1
        for idx, occurrence in enumerate(occurrences):
            if t < occurrence[0] - _EPS:
                break
            prev_index = idx
            if t <= occurrence[1] + _EPS:
                return self._ballistic_position(occurrence[2], occurrence[0], occurrence[1], t)
        else:
            idx = len(occurrences)

        if prev_index < 0:
            return self.hand_position(occurrences[0][2].throw_hand, t)

        prev = occurrences[prev_index]
        if t <= prev[1] + _EPS:
            return self._ballistic_position(prev[2], prev[0], prev[1], t)

        if idx < len(occurrences):
            next_occurrence = occurrences[idx]
            if t < next_occurrence[0] - _EPS:
                return self.hand_position(prev[2].catch_hand, t)

        return self.hand_position(prev[2].catch_hand, t)

    def sample(self, time_s: float) -> SampleState:
        self.validate()
        hand_positions = {hand: self.hand_position(hand, time_s) for hand in HAND_NAMES}
        ball_positions = {
            ball: position
            for ball in self.ball_ids()
            if (position := self.ball_position(ball, time_s)) is not None
        }
        return SampleState(time=float(time_s), hand_positions=hand_positions, ball_positions=ball_positions)

    def sample_hand_path(self, hand: HandName, samples: int = 240) -> np.ndarray:
        self.validate()
        count = max(2, int(samples))
        t = np.linspace(0.0, self.timeline_duration(), count, dtype=float)
        p = np.vstack([self.hand_state(hand, ti).position for ti in t])
        return np.column_stack((t, p))

    def sample_event_flight(self, event: ThrowEvent, samples: int = 64) -> np.ndarray:
        count = max(2, int(samples))
        t = np.linspace(event.throw_time, event.catch_time, count, dtype=float)
        p = np.vstack([self._ballistic_position(event, event.throw_time, event.catch_time, ti) for ti in t])
        return np.column_stack((t, p))

    def _ball_occurrences(self, ball: str, time_s: float) -> list[tuple[float, float, ThrowEvent]]:
        events = [event for event in self.sorted_events() if event.ball == ball]
        if not events:
            return []
        if not self.is_loop:
            return [(event.throw_time, event.catch_time, event) for event in events]

        period = self.loop_period
        center_cycle = math.floor(float(time_s) / period)
        occurrences: list[tuple[float, float, ThrowEvent]] = []
        for cycle in range(center_cycle - 1, center_cycle + 2):
            offset = cycle * period
            for event in events:
                occurrences.append((event.throw_time + offset, event.catch_time + offset, event))
        occurrences.sort(key=lambda item: (item[0], item[1], item[2].id))
        return occurrences

    def _expand_nodes(
        self,
        nodes: Sequence[_HandNode],
        time_s: float,
    ) -> list[_HandNode]:
        period = self.loop_period
        center_cycle = math.floor(float(time_s) / period)
        expanded: list[_HandNode] = []
        for cycle in range(center_cycle - 1, center_cycle + 2):
            offset = cycle * period
            for node in nodes:
                expanded.append(
                    _HandNode(
                        time=node.time + offset,
                        pos=node.pos.copy(),
                        label=node.label,
                        velocity=(None if node.velocity is None else node.velocity.copy()),
                        acceleration=(None if node.acceleration is None else node.acceleration.copy()),
                        spline_to_next=node.spline_to_next,
                        path_speed=node.path_speed,
                        bspline_degree=node.bspline_degree,
                        bspline_control_points=node.bspline_control_points,
                    )
                )
        expanded.sort(key=lambda item: item.time)
        return expanded

    def _hand_nodes(self, hand: HandName) -> list[_HandNode]:
        raw_nodes: list[_HandNode] = []
        for keyframe in self.sorted_hand_trajectory(hand):
            raw_nodes.append(
                _HandNode(
                    time=keyframe.time,
                    pos=_vec3_array(keyframe.pos),
                    label=f"{keyframe.id}:authored",
                    velocity=(None if keyframe.velocity is None else _vec3_array(keyframe.velocity)),
                    spline_to_next=keyframe.spline_to_next,
                    path_speed=keyframe.path_speed,
                    bspline_degree=keyframe.bspline_degree,
                    bspline_control_points=keyframe.bspline_control_points,
                )
            )
        for event in self.sorted_events():
            if event.throw_hand == hand:
                raw_nodes.append(
                    _HandNode(
                        time=event.throw_time,
                        pos=_vec3_array(event.throw_pos),
                        label=f"{event.id}:throw",
                        velocity=self._ball_velocity(event, at="throw"),
                        acceleration=np.zeros(3, dtype=float),
                        spline_to_next="quintic",
                    )
                )
            if event.catch_hand == hand:
                raw_nodes.append(
                    _HandNode(
                        time=event.catch_time,
                        pos=_vec3_array(event.catch_pos),
                        label=f"{event.id}:catch",
                        velocity=event.catch_velocity_scale * self._ball_velocity(event, at="catch"),
                        acceleration=np.zeros(3, dtype=float),
                        spline_to_next="quintic",
                    )
                )

        merged = self._merge_hand_nodes(raw_nodes, hand=hand)
        for idx, node in enumerate(merged):
            if node.velocity is None:
                prev_node = merged[idx - 1] if idx > 0 else None
                next_node = merged[idx + 1] if idx + 1 < len(merged) else None
                node.velocity = _finite_difference_velocity(
                    None if prev_node is None else prev_node.time,
                    None if prev_node is None else prev_node.pos,
                    node.time,
                    node.pos,
                    None if next_node is None else next_node.time,
                    None if next_node is None else next_node.pos,
                )
            if node.acceleration is None:
                node.acceleration = np.zeros(3, dtype=float)
        for idx, node in enumerate(merged[:-1]):
            if node.spline_to_next == "bspline":
                prepared = self._prepare_bspline_segment(merged, idx)
                self._validate_bspline_progress(merged, idx, prepared)
        return merged

    def _materialize_missing_hand_keyframe_parameters(self) -> None:
        for hand in HAND_NAMES:
            keyframes = self.sorted_hand_trajectory(hand)
            needs_update = any(
                keyframe.velocity is None or keyframe.path_speed is None
                for keyframe in keyframes
            )
            if not keyframes or not needs_update:
                continue

            nodes = self._hand_nodes(hand)
            updated: list[HandKeyframe] = []
            for keyframe in keyframes:
                node_index = self._find_hand_node_index(nodes, keyframe.time, keyframe.pos)
                resolved_velocity = (
                    np.zeros(3, dtype=float)
                    if node_index is None or nodes[node_index].velocity is None
                    else nodes[node_index].velocity
                )
                path_speed = keyframe.path_speed
                if path_speed is None and node_index is not None and self._node_uses_path_speed(nodes, node_index):
                    path_speed = (
                        float(np.linalg.norm(resolved_velocity))
                        if node_index is None
                        else self._node_path_speed(nodes, node_index)
                    )
                updated.append(
                    HandKeyframe(
                        id=keyframe.id,
                        hand=keyframe.hand,
                        time=keyframe.time,
                        pos=keyframe.pos,
                        spline_to_next=keyframe.spline_to_next,
                        velocity=(
                            keyframe.velocity
                            if keyframe.velocity is not None
                            else tuple(float(value) for value in resolved_velocity)
                        ),
                        path_speed=path_speed,
                        bspline_degree=keyframe.bspline_degree,
                        bspline_control_points=keyframe.bspline_control_points,
                    )
                )
            self.hand_trajectories[hand] = updated

    def _find_hand_node_index(
        self,
        nodes: Sequence[_HandNode],
        time_s: float,
        pos: Sequence[float],
    ) -> int | None:
        target_pos = _vec3_array(pos)
        for index, node in enumerate(nodes):
            if abs(node.time - float(time_s)) <= _EPS and np.allclose(node.pos, target_pos, atol=1e-9):
                return index
        return None

    def _node_tangent_direction(self, nodes: Sequence[_HandNode], index: int) -> np.ndarray:
        node = nodes[index]
        prev_node = nodes[index - 1] if index > 0 else None
        next_node = nodes[index + 1] if index + 1 < len(nodes) else None

        if node.velocity is not None:
            tangent = _normalize_vec3(node.velocity)
            if float(np.linalg.norm(tangent)) > _EPS:
                return tangent

        tangent = _finite_difference_velocity(
            None if prev_node is None else prev_node.time,
            None if prev_node is None else prev_node.pos,
            node.time,
            node.pos,
            None if next_node is None else next_node.time,
            None if next_node is None else next_node.pos,
        )
        tangent = _normalize_vec3(tangent)
        if float(np.linalg.norm(tangent)) > _EPS:
            return tangent

        if next_node is not None:
            tangent = _normalize_vec3(next_node.pos - node.pos)
            if float(np.linalg.norm(tangent)) > _EPS:
                return tangent
        if prev_node is not None:
            tangent = _normalize_vec3(node.pos - prev_node.pos)
            if float(np.linalg.norm(tangent)) > _EPS:
                return tangent
        if node.velocity is not None:
            tangent = _normalize_vec3(node.velocity)
            if float(np.linalg.norm(tangent)) > _EPS:
                return tangent
        return np.zeros(3, dtype=float)

    def _node_path_speed(self, nodes: Sequence[_HandNode], index: int) -> float:
        node = nodes[index]
        if node.path_speed is not None:
            return float(node.path_speed)
        if node.velocity is not None:
            speed = float(np.linalg.norm(node.velocity))
            if speed > _EPS:
                return speed

        prev_node = nodes[index - 1] if index > 0 else None
        next_node = nodes[index + 1] if index + 1 < len(nodes) else None
        fallback_velocity = _finite_difference_velocity(
            None if prev_node is None else prev_node.time,
            None if prev_node is None else prev_node.pos,
            node.time,
            node.pos,
            None if next_node is None else next_node.time,
            None if next_node is None else next_node.pos,
        )
        return float(np.linalg.norm(fallback_velocity))

    def _node_uses_path_speed(self, nodes: Sequence[_HandNode], index: int) -> bool:
        return (
            nodes[index].spline_to_next == "bspline"
            or (index > 0 and nodes[index - 1].spline_to_next == "bspline")
        )

    def _prepare_bspline_segment(self, nodes: Sequence[_HandNode], index: int) -> _PreparedBSplineSegment:
        start = nodes[index]
        end = nodes[index + 1]
        degree = 3 if start.bspline_degree is None else int(start.bspline_degree)
        control_count = 6 if start.bspline_control_points is None else int(start.bspline_control_points)
        start_dir = self._node_tangent_direction(nodes, index)
        end_dir = self._node_tangent_direction(nodes, index + 1)
        chord = end.pos - start.pos
        chord_length = float(np.linalg.norm(chord))
        if chord_length <= _EPS:
            raise ValidationError(
                f"hand bspline segment from {start.label} to {end.label} has zero geometric length"
            )
        chord_dir = chord / chord_length
        if float(np.linalg.norm(start_dir)) <= _EPS:
            start_dir = chord_dir
        if float(np.linalg.norm(end_dir)) <= _EPS:
            end_dir = chord_dir

        handle_length = max(0.15 * chord_length, chord_length / max(3, control_count - 1))
        handle_start = start.pos + handle_length * start_dir
        handle_end = end.pos - handle_length * end_dir

        control_points = np.zeros((control_count, 3), dtype=float)
        control_points[0] = start.pos
        control_points[1] = handle_start
        control_points[-2] = handle_end
        control_points[-1] = end.pos
        if control_count > 4:
            interior_count = control_count - 4
            taus = np.linspace(0.0, 1.0, interior_count + 2, dtype=float)[1:-1]
            for offset, tau in enumerate(taus, start=2):
                control_points[offset] = _cubic_bezier_point(float(tau), start.pos, handle_start, handle_end, end.pos)

        knots = _clamped_uniform_knots(control_count, degree)
        first_control, first_degree, first_knots = _derivative_control_polygon(control_points, degree, knots)
        second_control: np.ndarray | None = None
        second_degree: int | None = None
        second_knots: np.ndarray | None = None
        if len(first_control) > 1 and first_degree > 0:
            second_control, second_degree, second_knots = _derivative_control_polygon(first_control, first_degree, first_knots)

        arc_u, arc_length = _arc_length_lookup(
            control_points,
            degree,
            knots,
            first_control,
            first_degree,
            first_knots,
            samples=max(64, control_count * 24),
        )
        total_length = float(arc_length[-1]) if len(arc_length) else 0.0
        if total_length <= _EPS:
            raise ValidationError(
                f"hand bspline segment from {start.label} to {end.label} has zero arc length"
            )
        return _PreparedBSplineSegment(
            control_points=control_points,
            degree=degree,
            knots=knots,
            first_derivative_control_points=first_control,
            first_derivative_degree=first_degree,
            first_derivative_knots=first_knots,
            second_derivative_control_points=second_control,
            second_derivative_degree=second_degree,
            second_derivative_knots=second_knots,
            arc_u=arc_u,
            arc_length=arc_length,
            total_length=total_length,
        )

    def _evaluate_bspline_geometry(
        self,
        prepared: _PreparedBSplineSegment,
        u: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        clamped_u = max(0.0, min(1.0, float(u)))
        pos = _de_boor(prepared.control_points, prepared.degree, prepared.knots, clamped_u)
        if len(prepared.first_derivative_control_points) == 0:
            first = np.zeros(3, dtype=float)
        else:
            first = _de_boor(
                prepared.first_derivative_control_points,
                prepared.first_derivative_degree,
                prepared.first_derivative_knots,
                clamped_u,
            )
        if (
            prepared.second_derivative_control_points is None
            or prepared.second_derivative_knots is None
            or prepared.second_derivative_degree is None
            or len(prepared.second_derivative_control_points) == 0
        ):
            second = np.zeros(3, dtype=float)
        else:
            second = _de_boor(
                prepared.second_derivative_control_points,
                prepared.second_derivative_degree,
                prepared.second_derivative_knots,
                clamped_u,
            )
        return pos, first, second

    def _interpolate_arc_length_state(
        self,
        prepared: _PreparedBSplineSegment,
        distance_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        clamped_s = max(0.0, min(float(distance_s), prepared.total_length))
        if clamped_s <= _EPS:
            u = 0.0
        elif clamped_s >= prepared.total_length - _EPS:
            u = 1.0
        else:
            u = float(np.interp(clamped_s, prepared.arc_length, prepared.arc_u))

        pos, first, second = self._evaluate_bspline_geometry(prepared, u)
        speed_u = float(np.linalg.norm(first))
        if speed_u <= _EPS:
            tangent = np.zeros(3, dtype=float)
            curvature = np.zeros(3, dtype=float)
        else:
            tangent = first / speed_u
            dot_term = float(np.dot(first, second))
            curvature = (second / (speed_u * speed_u)) - (first * dot_term / (speed_u**4))
        return pos, tangent, curvature

    def _validate_bspline_progress(self, nodes: Sequence[_HandNode], index: int, prepared: _PreparedBSplineSegment) -> None:
        start = nodes[index]
        end = nodes[index + 1]
        duration = end.time - start.time
        if duration <= _EPS:
            raise ValidationError(
                f"hand bspline segment from {start.label} to {end.label} must have positive duration"
            )
        speed0 = self._node_path_speed(nodes, index)
        speed1 = self._node_path_speed(nodes, index + 1)
        for u in np.linspace(0.0, 1.0, 41, dtype=float):
            distance_s, distance_dot, _distance_ddot = _quintic_scalar_progress_state(
                float(u),
                duration,
                prepared.total_length,
                speed0,
                speed1,
            )
            if distance_s < -1e-8 or distance_s > prepared.total_length + 1e-8:
                raise ValidationError(
                    f"hand bspline segment from {start.label} to {end.label} leaves its arc-length bounds"
                )
            if distance_dot < -1e-7:
                raise ValidationError(
                    f"hand bspline segment from {start.label} to {end.label} reverses along the curve; "
                    "reduce the endpoint path speeds or lengthen the segment time"
                )

    def _merge_hand_nodes(self, raw_nodes: Iterable[_HandNode], *, hand: HandName) -> list[_HandNode]:
        merged: list[_HandNode] = []
        for node in sorted(raw_nodes, key=lambda item: (item.time, item.label)):
            if merged and abs(node.time - merged[-1].time) <= _EPS:
                existing = merged[-1]
                if not np.allclose(node.pos, existing.pos, atol=1e-9):
                    raise ValidationError(
                        f"{hand} hand has conflicting positions at t={node.time:.6f} "
                        f"between {node.label} and another event"
                    )
                if node.velocity is not None:
                    if existing.velocity is None:
                        existing.velocity = node.velocity.copy()
                    elif not np.allclose(node.velocity, existing.velocity, atol=1e-9):
                        raise ValidationError(
                            f"{hand} hand has conflicting velocities at t={node.time:.6f} "
                            f"between {node.label} and another event"
                        )
                if node.acceleration is not None:
                    if existing.acceleration is None:
                        existing.acceleration = node.acceleration.copy()
                    elif not np.allclose(node.acceleration, existing.acceleration, atol=1e-9):
                        raise ValidationError(
                            f"{hand} hand has conflicting accelerations at t={node.time:.6f} "
                            f"between {node.label} and another event"
                        )
                if node.path_speed is not None:
                    if existing.path_speed is None:
                        existing.path_speed = float(node.path_speed)
                    elif not math.isclose(float(node.path_speed), float(existing.path_speed), abs_tol=1e-9):
                        raise ValidationError(
                            f"{hand} hand has conflicting path speeds at t={node.time:.6f} "
                            f"between {node.label} and another event"
                        )
                if node.bspline_degree is not None:
                    if existing.bspline_degree is None:
                        existing.bspline_degree = int(node.bspline_degree)
                    elif int(node.bspline_degree) != int(existing.bspline_degree):
                        raise ValidationError(
                            f"{hand} hand has conflicting bspline degrees at t={node.time:.6f} "
                            f"between {node.label} and another event"
                        )
                if node.bspline_control_points is not None:
                    if existing.bspline_control_points is None:
                        existing.bspline_control_points = int(node.bspline_control_points)
                    elif int(node.bspline_control_points) != int(existing.bspline_control_points):
                        raise ValidationError(
                            f"{hand} hand has conflicting bspline control counts at t={node.time:.6f} "
                            f"between {node.label} and another event"
                        )
                if existing.spline_to_next != "quintic":
                    existing.spline_to_next = node.spline_to_next
                continue
            merged.append(
                _HandNode(
                    time=float(node.time),
                    pos=node.pos.copy(),
                    label=node.label,
                    velocity=None if node.velocity is None else node.velocity.copy(),
                    acceleration=None if node.acceleration is None else node.acceleration.copy(),
                    spline_to_next=node.spline_to_next,
                    path_speed=node.path_speed,
                    bspline_degree=node.bspline_degree,
                    bspline_control_points=node.bspline_control_points,
                )
            )
        return merged

    def _evaluate_hand_nodes(self, nodes: Sequence[_HandNode], time_s: float) -> HandSplineSample:
        if len(nodes) == 1:
            node = nodes[0]
            zero = np.zeros(3, dtype=float)
            return HandSplineSample(
                time=float(time_s),
                position=node.pos.copy(),
                velocity=(zero.copy() if node.velocity is None else node.velocity.copy()),
                acceleration=(zero.copy() if node.acceleration is None else node.acceleration.copy()),
            )

        t = float(time_s)
        times = [node.time for node in nodes]
        if t < times[0] - _EPS:
            first = nodes[0]
            zero = np.zeros(3, dtype=float)
            return HandSplineSample(
                time=t,
                position=first.pos.copy(),
                velocity=(zero.copy() if first.velocity is None else first.velocity.copy()),
                acceleration=(zero.copy() if first.acceleration is None else first.acceleration.copy()),
            )
        if t > times[-1] + _EPS:
            last = nodes[-1]
            zero = np.zeros(3, dtype=float)
            return HandSplineSample(
                time=t,
                position=last.pos.copy(),
                velocity=(zero.copy() if last.velocity is None else last.velocity.copy()),
                acceleration=(zero.copy() if last.acceleration is None else last.acceleration.copy()),
            )

        idx = min(max(1, bisect_right(times, t)), len(nodes) - 1)
        start = nodes[idx - 1]
        end = nodes[idx]
        duration = end.time - start.time
        if duration <= _EPS:
            zero = np.zeros(3, dtype=float)
            return HandSplineSample(time=t, position=end.pos.copy(), velocity=zero.copy(), acceleration=zero.copy())
        u = (t - start.time) / duration
        if start.spline_to_next == "quintic":
            pos, vel, acc = _quintic_hermite_state(
                u,
                duration,
                start.pos,
                start.velocity if start.velocity is not None else np.zeros(3, dtype=float),
                start.acceleration if start.acceleration is not None else np.zeros(3, dtype=float),
                end.pos,
                end.velocity if end.velocity is not None else np.zeros(3, dtype=float),
                end.acceleration if end.acceleration is not None else np.zeros(3, dtype=float),
            )
        elif start.spline_to_next == "bspline":
            prepared = self._prepare_bspline_segment(nodes, idx - 1)
            distance_s, distance_dot, distance_ddot = _quintic_scalar_progress_state(
                u,
                duration,
                prepared.total_length,
                self._node_path_speed(nodes, idx - 1),
                self._node_path_speed(nodes, idx),
            )
            pos, tangent, curvature = self._interpolate_arc_length_state(prepared, distance_s)
            vel = tangent * distance_dot
            acc = curvature * (distance_dot * distance_dot) + tangent * distance_ddot
        else:
            pos, vel, acc = _cubic_hermite_state(
                u,
                duration,
                start.pos,
                start.velocity if start.velocity is not None else np.zeros(3, dtype=float),
                end.pos,
                end.velocity if end.velocity is not None else np.zeros(3, dtype=float),
            )
        return HandSplineSample(time=t, position=pos, velocity=vel, acceleration=acc)

    def _ballistic_position(
        self,
        event: ThrowEvent,
        throw_time: float,
        catch_time: float,
        time_s: float,
    ) -> np.ndarray:
        duration = catch_time - throw_time
        tau = max(0.0, min(duration, time_s - throw_time))
        p0 = _vec3_array(event.throw_pos)
        p1 = _vec3_array(event.catch_pos)

        vx = (p1[0] - p0[0]) / duration
        vy = (p1[1] - p0[1]) / duration
        vz = (p1[2] - p0[2] + 0.5 * self.gravity * duration * duration) / duration

        return np.array(
            [
                p0[0] + vx * tau,
                p0[1] + vy * tau,
                p0[2] + vz * tau - 0.5 * self.gravity * tau * tau,
            ],
            dtype=float,
        )

    def _ball_velocity(self, event: ThrowEvent, *, at: Literal["throw", "catch"]) -> np.ndarray:
        duration = event.duration
        if duration <= _EPS:
            raise ValidationError(f"event {event.id} must have positive duration")
        p0 = _vec3_array(event.throw_pos)
        p1 = _vec3_array(event.catch_pos)
        vx = (p1[0] - p0[0]) / duration
        vy = (p1[1] - p0[1]) / duration
        vz0 = (p1[2] - p0[2] + 0.5 * self.gravity * duration * duration) / duration
        if at == "throw":
            return np.array([vx, vy, vz0], dtype=float)
        return np.array([vx, vy, vz0 - self.gravity * duration], dtype=float)

    def _validate_hand_capacity(self, by_ball: Dict[str, list[ThrowEvent]]) -> None:
        intervals: Dict[HandName, list[tuple[float, float, str, str]]] = {hand: [] for hand in HAND_NAMES}

        if self.is_loop:
            period = self.loop_period
            for ball, events in by_ball.items():
                ordered = sorted(events, key=lambda event: (event.throw_time, event.catch_time, event.id))
                min_time = min(min(event.throw_time, event.catch_time) for event in ordered)
                max_time = max(max(event.throw_time, event.catch_time) for event in ordered)
                cycle_start = math.floor((-max_time) / period) - 2
                cycle_end = math.ceil((period - min_time) / period) + 2

                occurrences: list[tuple[float, float, ThrowEvent]] = []
                for cycle in range(cycle_start, cycle_end + 1):
                    offset = cycle * period
                    for event in ordered:
                        occurrences.append((event.throw_time + offset, event.catch_time + offset, event))
                occurrences.sort(key=lambda item: (item[0], item[1], item[2].id))

                for current, nxt in zip(occurrences, occurrences[1:]):
                    self._append_hold_interval(
                        intervals[current[2].catch_hand],
                        start=current[1],
                        end=nxt[0],
                        window_start=0.0,
                        window_end=period,
                        ball=ball,
                        label=f"{current[2].id}->{nxt[2].id}",
                    )
        else:
            timeline_end = self.sequence_end_time()
            for ball, events in by_ball.items():
                ordered = sorted(events, key=lambda event: (event.throw_time, event.catch_time, event.id))
                first = ordered[0]
                self._append_hold_interval(
                    intervals[first.throw_hand],
                    start=0.0,
                    end=first.throw_time,
                    window_start=0.0,
                    window_end=timeline_end,
                    ball=ball,
                    label=f"start->{first.id}",
                )

                for current, nxt in zip(ordered, ordered[1:]):
                    self._append_hold_interval(
                        intervals[current.catch_hand],
                        start=current.catch_time,
                        end=nxt.throw_time,
                        window_start=0.0,
                        window_end=timeline_end,
                        ball=ball,
                        label=f"{current.id}->{nxt.id}",
                    )

                last = ordered[-1]
                self._append_hold_interval(
                    intervals[last.catch_hand],
                    start=last.catch_time,
                    end=timeline_end,
                    window_start=0.0,
                    window_end=timeline_end,
                    ball=ball,
                    label=f"{last.id}->end",
                )

        for hand, hand_intervals in intervals.items():
            self._ensure_single_ball_hold(hand, hand_intervals)

    def _append_hold_interval(
        self,
        intervals: list[tuple[float, float, str, str]],
        *,
        start: float,
        end: float,
        window_start: float,
        window_end: float,
        ball: str,
        label: str,
    ) -> None:
        clipped_start = max(float(start), float(window_start))
        clipped_end = min(float(end), float(window_end))
        if clipped_end > clipped_start + _EPS:
            intervals.append((clipped_start, clipped_end, ball, label))

    def _ensure_single_ball_hold(
        self,
        hand: HandName,
        intervals: list[tuple[float, float, str, str]],
    ) -> None:
        current: tuple[float, float, str, str] | None = None
        for interval in sorted(intervals, key=lambda item: (item[0], item[1], item[2], item[3])):
            if current is not None and interval[0] < current[1] - _EPS:
                overlap_start = max(current[0], interval[0])
                overlap_end = min(current[1], interval[1])
                raise ValidationError(
                    f"{hand} hand would hold more than one ball at once between "
                    f"{overlap_start:.6f}s and {overlap_end:.6f}s "
                    f"({current[2]} {current[3]} overlaps {interval[2]} {interval[3]})"
                )
            current = interval


def save_pattern_project(project: PatternProject, path: str | Path) -> None:
    """Persist a pattern project to YAML."""

    project.validate()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(project.to_dict(), handle, sort_keys=False)


def load_pattern_project(path: str | Path) -> PatternProject:
    """Load a pattern project from YAML."""

    in_path = Path(path)
    with in_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValidationError(f"pattern file must contain a mapping; got {type(data).__name__}")
    project = PatternProject.from_dict(data)
    project.validate()
    return project


def build_three_ball_cascade_pattern() -> PatternProject:
    """Default sample used by the interactive editor."""

    left_throw = (-0.32, -0.08, 1.00)
    left_catch = (-0.22, 0.06, 0.78)
    right_throw = (0.32, 0.08, 1.00)
    right_catch = (0.22, -0.06, 0.78)

    events = [
        ThrowEvent("A1", "A", "left", "right", 0.00, 1.50, left_throw, right_catch),
        ThrowEvent("B1", "B", "right", "left", 0.50, 2.00, right_throw, left_catch),
        ThrowEvent("C1", "C", "left", "right", 1.00, 2.50, left_throw, right_catch),
        ThrowEvent("A2", "A", "right", "left", 1.75, 3.25, right_throw, left_catch),
        ThrowEvent("B2", "B", "left", "right", 2.25, 3.75, left_throw, right_catch),
        ThrowEvent("C2", "C", "right", "left", 2.75, 4.25, right_throw, left_catch),
    ]

    project = PatternProject(
        name="three_ball_cascade",
        mode="loop",
        loop_period=3.50,
        gravity=9.81,
        events=events,
        hand_trajectories={
            "left": [
                HandKeyframe("L1", "left", 0.35, (-0.42, -0.18, 1.18), "cubic", velocity=(0.0, 0.0, 0.0)),
                HandKeyframe("L2", "left", 1.45, (-0.30, 0.18, 0.90), "quintic", velocity=(0.10, 0.14, -0.22)),
                HandKeyframe("L3", "left", 2.55, (-0.42, -0.20, 1.16), "cubic", velocity=(0.10, 0.14, -0.22)),
                HandKeyframe("L4", "left", 3.35, (-0.28, 0.16, 0.86), "quintic", velocity=(0.0, 0.0, 0.0)),
            ],
            "right": [
                HandKeyframe("R1", "right", 0.95, (0.42, 0.20, 1.16), "cubic", velocity=(-0.10, -0.14, -0.22)),
                HandKeyframe("R2", "right", 2.10, (0.30, -0.18, 0.88), "quintic", velocity=(-0.13333333333333333, -0.18666666666666668, -0.2933333333333333)),
                HandKeyframe("R3", "right", 3.00, (0.42, 0.18, 1.14), "cubic", velocity=(-0.10, -0.14, -0.22)),
            ],
        },
    )
    project.validate()
    return project
