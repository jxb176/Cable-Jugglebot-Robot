"""Pure cable/pose kinematics helpers."""

from __future__ import annotations

import math

import numpy as np

from jugglebot.core.cable_ik import CableRobotGeometry, pose_to_cable_lengths_mm
from jugglebot.core.pose_utils import quat_from_rpy_deg


DEFAULT_GEOM = CableRobotGeometry()


def pose5_to_tq_mm(pose5):
    """Map [x_m, y_m, z_m, roll_rad, pitch_rad] -> (t_mm, q with yaw=0)."""
    x_m, y_m, z_m, roll_rad, pitch_rad = [float(v) for v in pose5]
    t_mm = (1000.0 * x_m, 1000.0 * y_m, 1000.0 * z_m)
    q = quat_from_rpy_deg(math.degrees(roll_rad), math.degrees(pitch_rad), 0.0)
    return t_mm, q


def cable_lengths_m_from_pose5(pose5, geometry: CableRobotGeometry = DEFAULT_GEOM):
    t_mm, q = pose5_to_tq_mm(pose5)
    l_mm = pose_to_cable_lengths_mm(geometry, t_mm, q)
    return np.asarray(l_mm, dtype=float) / 1000.0


def cable_lengths_jacobian_pose5_fd(
    pose5,
    geometry: CableRobotGeometry = DEFAULT_GEOM,
    eps_pos_m: float = 1e-4,
    eps_ang_rad: float = 1e-4,
):
    """
    Finite-difference Jacobian of cable lengths wrt [x,y,z,roll,pitch].
    Returns J shape (6,5), where J[i,j] = dL_i / d pose_j.
    """
    q0 = np.asarray(pose5, dtype=float).copy()
    j_mat = np.zeros((6, 5), dtype=float)
    for j in range(5):
        dq = np.zeros(5, dtype=float)
        dq[j] = eps_pos_m if j < 3 else eps_ang_rad
        lp = cable_lengths_m_from_pose5(q0 + dq, geometry=geometry)
        lm = cable_lengths_m_from_pose5(q0 - dq, geometry=geometry)
        j_mat[:, j] = (lp - lm) / (2.0 * dq[j])
    return j_mat


def solve_pose_from_lengths(
    cable_lengths_m,
    q_seed,
    *,
    geometry: CableRobotGeometry = DEFAULT_GEOM,
    position_clip_m: float = 0.6,
    angle_clip_rad: float | None = None,
    iterations: int = 4,
):
    """
    Solve pose q=[x,y,z,roll,pitch] from measured cable lengths.

    Uses the same clipped least-squares refinement that the previous hardware
    estimator used so the runtime behavior stays close to the old implementation.
    """
    q = np.asarray(q_seed, dtype=float).copy()
    l_meas = np.asarray(cable_lengths_m, dtype=float)
    if not np.all(np.isfinite(l_meas)) or not np.all(np.isfinite(q)):
        raise ValueError("non-finite pose-solver inputs")
    q[2] = max(-position_clip_m, min(position_clip_m, q[2]))
    if angle_clip_rad is None:
        angle_clip_rad = math.radians(60.0)
    for _ in range(max(1, int(iterations))):
        l_pred = cable_lengths_m_from_pose5(q, geometry=geometry)
        residual = l_meas - l_pred
        j_mat = cable_lengths_jacobian_pose5_fd(q, geometry=geometry)
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(j_mat)):
            raise ValueError("non-finite pose-solver Jacobian/residual")
        dq, *_ = np.linalg.lstsq(j_mat, residual, rcond=None)
        if not np.all(np.isfinite(dq)):
            raise ValueError("non-finite pose-solver step")

        dq[0:3] = np.clip(dq[0:3], -0.01, 0.01)
        dq[3:5] = np.clip(dq[3:5], -0.05, 0.05)
        q = q + dq
        q[0:3] = np.clip(q[0:3], -position_clip_m, position_clip_m)
        q[3:5] = np.clip(q[3:5], -angle_clip_rad, angle_clip_rad)
        if float(np.linalg.norm(residual)) < 1e-5:
            break
    return q
