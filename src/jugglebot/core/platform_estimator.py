"""Platform-state estimation from actuator-side feedback."""

from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from jugglebot.core.cable_ik import CableRobotGeometry
from jugglebot.core.kinematics import (
    DEFAULT_GEOM,
    cable_lengths_jacobian_pose5_fd,
    solve_pose_from_lengths,
)
from jugglebot.core.types import ActuatorState


class CablePlatformEstimator:
    """Estimate platform pose/twist from cable-length actuator feedback."""

    def __init__(
        self,
        *,
        axis_ids: Sequence[int],
        mm_per_turn: Sequence[float],
        home_cable_mm: Sequence[float],
        geometry: CableRobotGeometry = DEFAULT_GEOM,
        update_rate_hz: float = 100.0,
    ):
        self.axis_ids = tuple(int(aid) for aid in axis_ids)
        self._axis_index = {aid: i for i, aid in enumerate(self.axis_ids)}
        self._mm_per_turn = np.asarray([float(v) for v in mm_per_turn], dtype=float)
        self._home_cable_mm = np.asarray([float(v) for v in home_cable_mm], dtype=float)
        self._geometry = geometry
        self._update_dt = 1.0 / max(1.0, float(update_rate_hz))
        self._q = np.zeros(5, dtype=float)
        self._qd = np.zeros(5, dtype=float)
        self._jacobian = np.zeros((len(self.axis_ids), 5), dtype=float)
        self._last_perf = 0.0
        self._has_estimate = False

    def get_latest(self):
        if not self._has_estimate:
            return None, None, None
        return self._q.copy(), self._qd.copy(), self._jacobian.copy()

    def update_from_actuator_states(
        self,
        actuator_states: Sequence[ActuatorState],
        *,
        now_perf: float | None = None,
    ):
        if now_perf is None:
            now_perf = time.perf_counter()
        if self._has_estimate and (float(now_perf) - self._last_perf) < self._update_dt:
            return self.get_latest()

        pos_turns = np.full(len(self.axis_ids), np.nan, dtype=float)
        vel_turns_per_s = np.full(len(self.axis_ids), np.nan, dtype=float)
        for axis_state in actuator_states:
            idx = self._axis_index.get(int(axis_state.axis_id))
            if idx is None:
                continue
            if axis_state.position_turns is not None:
                pos_turns[idx] = float(axis_state.position_turns)
            if axis_state.velocity_turns_per_s is not None:
                vel_turns_per_s[idx] = float(axis_state.velocity_turns_per_s)

        if not np.all(np.isfinite(pos_turns)) or not np.all(np.isfinite(vel_turns_per_s)):
            return self.get_latest()

        cable_lengths_m = (self._home_cable_mm + pos_turns * self._mm_per_turn) / 1000.0
        cable_velocities_mps = (vel_turns_per_s * self._mm_per_turn) / 1000.0
        if not np.all(np.isfinite(cable_lengths_m)) or not np.all(np.isfinite(cable_velocities_mps)):
            return self.get_latest()

        q_seed = self._q if self._has_estimate else np.zeros(5, dtype=float)
        try:
            q_new = solve_pose_from_lengths(cable_lengths_m, q_seed, geometry=self._geometry)
            jacobian = cable_lengths_jacobian_pose5_fd(q_new, geometry=self._geometry)
            qd_new, *_ = np.linalg.lstsq(jacobian, cable_velocities_mps, rcond=None)
            qd_new = np.asarray(qd_new, dtype=float)
            if not np.all(np.isfinite(q_new)) or not np.all(np.isfinite(jacobian)) or not np.all(np.isfinite(qd_new)):
                raise ValueError("non-finite estimator result")
        except Exception:
            return self.get_latest()

        self._q = q_new.copy()
        self._qd = qd_new.copy()
        self._jacobian = jacobian.copy()
        self._last_perf = float(now_perf)
        self._has_estimate = True
        return self.get_latest()
