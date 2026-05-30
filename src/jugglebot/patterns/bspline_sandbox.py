"""Core geometry helpers for the standalone B-spline sandbox."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .model import (
    ValidationError,
    _arc_length_lookup,
    _clamped_uniform_knots,
    _cubic_bezier_point,
    _de_boor,
    _derivative_control_polygon,
)


def _as_point2(value: Sequence[float]) -> np.ndarray:
    point = np.asarray(value, dtype=float).reshape(-1)
    if point.size != 2:
        raise ValidationError(f"expected a 2D point; got {value!r}")
    return point.astype(float, copy=True)


def _point2_to_vec3(value: np.ndarray) -> np.ndarray:
    return np.array([float(value[0]), float(value[1]), 0.0], dtype=float)


def _default_control_polygon(
    start: np.ndarray,
    start_handle: np.ndarray,
    end_handle: np.ndarray,
    end: np.ndarray,
    control_count: int,
) -> np.ndarray:
    count = int(control_count)
    if count < 4:
        raise ValidationError("control_count must be at least 4 for endpoint tangent handles")

    control_points = np.zeros((count, 2), dtype=float)
    control_points[0] = start
    control_points[1] = start_handle
    control_points[-2] = end_handle
    control_points[-1] = end
    if count > 4:
        interior_count = count - 4
        taus = np.linspace(0.0, 1.0, interior_count + 2, dtype=float)[1:-1]
        p0 = _point2_to_vec3(start)
        p1 = _point2_to_vec3(start_handle)
        p2 = _point2_to_vec3(end_handle)
        p3 = _point2_to_vec3(end)
        for offset, tau in enumerate(taus, start=2):
            control_points[offset] = _cubic_bezier_point(float(tau), p0, p1, p2, p3)[:2]
    return control_points


@dataclass(frozen=True)
class BSplineSandboxSample:
    """Sampled curve and metadata for one sandbox state."""

    curve: np.ndarray
    knots: np.ndarray
    length: float


@dataclass
class BSplineSandboxModel:
    """Editable 2D B-spline control polygon for sandbox experiments."""

    degree: int = 3
    control_points: np.ndarray = field(
        default_factory=lambda: _default_control_polygon(
            np.array([-0.85, -0.35], dtype=float),
            np.array([-0.25, -0.90], dtype=float),
            np.array([0.25, -0.20], dtype=float),
            np.array([0.85, 0.45], dtype=float),
            6,
        )
    )

    def __post_init__(self) -> None:
        self.degree = int(self.degree)
        if self.degree < 1:
            raise ValidationError("degree must be at least 1")

        points = np.asarray(self.control_points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValidationError("control_points must be an (N, 2) array")
        if len(points) < max(4, self.degree + 1):
            raise ValidationError(
                f"need at least {max(4, self.degree + 1)} control points for degree {self.degree}"
            )
        self.control_points = points.astype(float, copy=True)

    @property
    def control_count(self) -> int:
        return int(len(self.control_points))

    @property
    def minimum_control_count(self) -> int:
        return max(4, self.degree + 1)

    @property
    def start_point(self) -> np.ndarray:
        return self.control_points[0].copy()

    @property
    def end_point(self) -> np.ndarray:
        return self.control_points[-1].copy()

    @property
    def start_handle(self) -> np.ndarray:
        return self.control_points[1].copy()

    @property
    def end_handle(self) -> np.ndarray:
        return self.control_points[-2].copy()

    @property
    def start_tangent_tip(self) -> np.ndarray:
        return self.start_handle

    @property
    def end_tangent_tip(self) -> np.ndarray:
        end = self.control_points[-1]
        tangent = end - self.control_points[-2]
        return end + tangent

    def set_degree(self, degree: int) -> None:
        degree = int(degree)
        if degree < 1:
            raise ValidationError("degree must be at least 1")
        self.degree = degree
        if self.control_count < self.minimum_control_count:
            self.set_control_count(self.minimum_control_count)

    def set_control_count(self, control_count: int) -> None:
        count = max(self.minimum_control_count, int(control_count))
        if count == self.control_count:
            return
        self.control_points = _default_control_polygon(
            self.control_points[0],
            self.control_points[1],
            self.control_points[-2],
            self.control_points[-1],
            count,
        )

    def reset_default(self) -> None:
        self.degree = 3
        self.control_points = _default_control_polygon(
            np.array([-0.85, -0.35], dtype=float),
            np.array([-0.25, -0.90], dtype=float),
            np.array([0.25, -0.20], dtype=float),
            np.array([0.85, 0.45], dtype=float),
            6,
        )

    def reset_interior_control_points(self) -> None:
        self.control_points = _default_control_polygon(
            self.control_points[0],
            self.control_points[1],
            self.control_points[-2],
            self.control_points[-1],
            self.control_count,
        )

    def move_start_point(self, point: Sequence[float]) -> None:
        self.control_points[0] = _as_point2(point)

    def move_end_point(self, point: Sequence[float]) -> None:
        self.control_points[-1] = _as_point2(point)

    def move_start_tangent_tip(self, point: Sequence[float]) -> None:
        self.control_points[1] = _as_point2(point)

    def move_end_handle(self, point: Sequence[float]) -> None:
        self.control_points[-2] = _as_point2(point)

    def move_end_tangent_tip(self, point: Sequence[float]) -> None:
        tip = _as_point2(point)
        end = self.control_points[-1]
        self.control_points[-2] = end - (tip - end)

    def move_control_point(self, index: int, point: Sequence[float]) -> None:
        if index < 0 or index >= self.control_count:
            raise IndexError(f"control point index {index} out of range")
        self.control_points[index] = _as_point2(point)

    def tangent_vector(self, side: str) -> np.ndarray:
        norm_side = str(side).strip().lower()
        if norm_side == "start":
            return self.control_points[1] - self.control_points[0]
        if norm_side == "end":
            return self.control_points[-1] - self.control_points[-2]
        raise ValidationError(f"unknown tangent side {side!r}")

    def tangent_angle_deg(self, side: str) -> float:
        vector = self.tangent_vector(side)
        return float(np.degrees(np.arctan2(vector[1], vector[0])))

    def tangent_length(self, side: str) -> float:
        return float(np.linalg.norm(self.tangent_vector(side)))

    def knot_vector(self) -> np.ndarray:
        return _clamped_uniform_knots(self.control_count, self.degree)

    def sample(self, samples: int = 256) -> BSplineSandboxSample:
        count = max(2, int(samples))
        control_points3 = np.column_stack((self.control_points, np.zeros(self.control_count, dtype=float)))
        knots = self.knot_vector()
        curve = np.vstack(
            [
                _de_boor(control_points3, self.degree, knots, float(u))[:2]
                for u in np.linspace(0.0, 1.0, count, dtype=float)
            ]
        )

        first_control, first_degree, first_knots = _derivative_control_polygon(control_points3, self.degree, knots)
        _arc_u, arc_length = _arc_length_lookup(
            control_points3,
            self.degree,
            knots,
            first_control,
            first_degree,
            first_knots,
            samples=max(64, self.control_count * 24),
        )
        length = float(arc_length[-1]) if len(arc_length) else 0.0
        return BSplineSandboxSample(curve=curve, knots=knots.copy(), length=length)
