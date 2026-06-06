"""Geometry helpers for the controller 3D scene."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from jugglebot.core.cable_ik import CableRobotGeometry, R_mul_v, q_to_R, v_add
from jugglebot.core.pose_utils import quat_from_rpy_deg


@dataclass(frozen=True, slots=True)
class SceneGeometry:
    anchors_world_mm: np.ndarray
    attach_platform_mm: np.ndarray
    anchor_edge_pairs: tuple[tuple[int, int], ...]
    platform_edge_pairs: tuple[tuple[int, int], ...]
    platform_faces: np.ndarray

    @classmethod
    def from_robot_geometry(cls, geometry: CableRobotGeometry) -> "SceneGeometry":
        anchors = np.asarray(geometry.anchors_world, dtype=float)
        attach = np.asarray(geometry.attach_platform, dtype=float)
        edge_pairs = (
            (0, 2),
            (2, 4),
            (4, 0),
            (1, 3),
            (3, 5),
            (5, 1),
            (0, 1),
            (2, 3),
            (4, 5),
        )
        faces = np.asarray(
            [
                (0, 2, 4),
                (1, 5, 3),
                (0, 1, 2),
                (1, 3, 2),
                (2, 3, 4),
                (3, 5, 4),
                (4, 5, 0),
                (5, 1, 0),
            ],
            dtype=np.int32,
        )
        return cls(
            anchors_world_mm=anchors,
            attach_platform_mm=attach,
            anchor_edge_pairs=edge_pairs,
            platform_edge_pairs=edge_pairs,
            platform_faces=faces,
        )


def pose_to_world_points_mm(pose_mm_deg, geometry: SceneGeometry) -> np.ndarray | None:
    if pose_mm_deg is None:
        return None
    try:
        x_mm = float(pose_mm_deg[0])
        y_mm = float(pose_mm_deg[1])
        z_mm = float(pose_mm_deg[2])
        roll_deg = float(pose_mm_deg[3])
        pitch_deg = float(pose_mm_deg[4])
        yaw_deg = float(pose_mm_deg[5]) if len(pose_mm_deg) > 5 and math.isfinite(float(pose_mm_deg[5])) else 0.0
    except Exception:
        return None

    values = (x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg)
    if not all(math.isfinite(value) for value in values):
        return None

    rotation = q_to_R(quat_from_rpy_deg(roll_deg, pitch_deg, yaw_deg))
    translation = (x_mm, y_mm, z_mm)
    world_points = []
    for local_point in geometry.attach_platform_mm:
        world_points.append(v_add(translation, R_mul_v(rotation, tuple(local_point.tolist()))))
    return np.asarray(world_points, dtype=float)


def pose_center_mm(pose_mm_deg) -> np.ndarray | None:
    if pose_mm_deg is None or len(pose_mm_deg) < 3:
        return None
    center = np.asarray(pose_mm_deg[:3], dtype=float)
    if not np.all(np.isfinite(center)):
        return None
    return center


def edge_segments(points: np.ndarray, edge_pairs: tuple[tuple[int, int], ...]) -> np.ndarray:
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=float)
    segments = []
    for start, end in edge_pairs:
        segments.append(points[start])
        segments.append(points[end])
    return np.asarray(segments, dtype=float)


def cable_segments(anchor_points: np.ndarray, platform_points: np.ndarray | None) -> np.ndarray:
    if platform_points is None or len(platform_points) != len(anchor_points):
        return np.empty((0, 3), dtype=float)
    segments = []
    for anchor, attach in zip(anchor_points, platform_points, strict=True):
        segments.append(anchor)
        segments.append(attach)
    return np.asarray(segments, dtype=float)
