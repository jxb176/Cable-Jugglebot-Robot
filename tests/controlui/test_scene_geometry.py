from __future__ import annotations

import numpy as np

from jugglebot.controlui.scene_geometry import SceneGeometry, cable_segments, edge_segments, pose_center_mm, pose_to_world_points_mm
from jugglebot.core.cable_ik import CableRobotGeometry


def test_zero_pose_maps_to_platform_attachment_points() -> None:
    geometry = SceneGeometry.from_robot_geometry(CableRobotGeometry())

    world_points = pose_to_world_points_mm((0.0, 0.0, 0.0, 0.0, 0.0, 0.0), geometry)

    assert world_points is not None
    assert np.allclose(world_points, geometry.attach_platform_mm)


def test_pose_translation_is_applied_to_platform_points() -> None:
    geometry = SceneGeometry.from_robot_geometry(CableRobotGeometry())

    world_points = pose_to_world_points_mm((10.0, -20.0, 30.0, 0.0, 0.0, 0.0), geometry)

    assert world_points is not None
    assert np.allclose(world_points[0], geometry.attach_platform_mm[0] + np.asarray([10.0, -20.0, 30.0]))


def test_edge_and_cable_segment_shapes_match_expected_pairs() -> None:
    geometry = SceneGeometry.from_robot_geometry(CableRobotGeometry())
    points = geometry.attach_platform_mm

    edges = edge_segments(points, geometry.platform_edge_pairs)
    cables = cable_segments(geometry.anchors_world_mm, points)

    assert edges.shape == (2 * len(geometry.platform_edge_pairs), 3)
    assert cables.shape == (12, 3)


def test_invalid_pose_returns_none() -> None:
    geometry = SceneGeometry.from_robot_geometry(CableRobotGeometry())

    assert pose_to_world_points_mm((float('nan'), 0.0, 0.0, 0.0, 0.0, 0.0), geometry) is None
    assert pose_center_mm((float('nan'), 0.0, 0.0)) is None
