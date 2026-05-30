"""Shared pose and orientation helpers."""

from __future__ import annotations

import math

from jugglebot.core.cable_ik import q_from_axis_angle, q_mul, q_norm


def quat_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float = 0.0):
    """Quaternion for R = Rz(yaw)*Ry(pitch)*Rx(roll)."""
    r = math.radians(float(roll_deg))
    p = math.radians(float(pitch_deg))
    y = math.radians(float(yaw_deg))
    qx = q_from_axis_angle((1.0, 0.0, 0.0), r)
    qy = q_from_axis_angle((0.0, 1.0, 0.0), p)
    qz = q_from_axis_angle((0.0, 0.0, 1.0), y)
    return q_norm(q_mul(q_mul(qz, qy), qx))


def quat_to_rpy_rad(q):
    """Convert quaternion (w,x,y,z) to roll/pitch/yaw radians."""
    w, x, y, z = q_norm((float(q[0]), float(q[1]), float(q[2]), float(q[3])))
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw
