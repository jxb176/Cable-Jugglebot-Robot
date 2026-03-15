"""
Hardware driver using ODrive CAN interface.
"""

import logging
import math
import threading
import time
from collections import deque
from typing import Optional, Callable, List

import numpy as np

from jugglebot.core.cable_ik import (
    CableRobotGeometry,
    pose_to_cable_lengths_mm,
    q_from_axis_angle,
    q_mul,
    q_norm,
)

from .driver_interface import RobotDriver
from .ODriveCANSimple import ODriveCanManager, AxisState


logger = logging.getLogger(__name__)


class HardwareDriver(RobotDriver):
    """
    Hardware driver implementation using ODrive CAN.
    """

    def __init__(
        self,
        canbus: str = "can0",
        axis_ids: List[int] = None,
        mm_per_turn: List[float] = None,
        capstan_radius_m: float = 0.01,
        torque_direction: float = 1.0,
        pose_est_rate_hz: float = 100.0,
        can_bitrate: float = 1_000_000.0,
        can_frame_bits_est: float = 128.0,
    ):
        self.canbus = canbus
        self.axis_ids = axis_ids or [0, 1, 2, 3, 4, 5]
        self.manager: Optional[ODriveCanManager] = None
        self.axes = {}
        self._axis_index = {aid: i for i, aid in enumerate(self.axis_ids)}
        self._position_callback: Optional[Callable[[int, float], None]] = None
        self._velocity_callback: Optional[Callable[[int, float], None]] = None
        self._bus_callback: Optional[Callable[[int, float, float], None]] = None
        self._current_callback: Optional[Callable[[int, float], None]] = None
        self._temp_callback: Optional[Callable[[int, float, float], None]] = None
        self._heartbeat_callback: Optional[Callable[[int, int, int, int], None]] = None
        self._axis_pos_turns = [None] * len(self.axis_ids)
        self._axis_vel_turnsps = [None] * len(self.axis_ids)
        self._axis_current_a = [None] * len(self.axis_ids)
        self._axis_torque_cmd_nm = [0.0] * len(self.axis_ids)
        if mm_per_turn is None:
            self.mm_per_turn = [-62.832] * len(self.axis_ids)
        else:
            if len(mm_per_turn) != len(self.axis_ids):
                raise ValueError("mm_per_turn must match axis_ids length")
            self.mm_per_turn = [float(v) for v in mm_per_turn]
        self.capstan_radius_m = max(1e-9, float(capstan_radius_m))
        self.torque_direction = float(torque_direction)

        self._geom = CableRobotGeometry()
        self._home_q = (1.0, 0.0, 0.0, 0.0)
        self._home_t_mm = (0.0, 0.0, 0.0)
        self._home_cable_mm = pose_to_cable_lengths_mm(self._geom, self._home_t_mm, self._home_q)
        self._pose_est_q = np.zeros(5, dtype=float)  # [x,y,z,roll,pitch], SI units
        self._pose_est_qd = np.zeros(5, dtype=float)
        self._pose_last_perf = 0.0
        self._pose_update_dt = 1.0 / max(1.0, float(pose_est_rate_hz))
        self._lock = threading.Lock()
        self._enc_times_axes = [deque() for _ in self.axis_ids]
        self._can_bitrate = max(1.0, float(can_bitrate))
        self._can_frame_bits_est = max(1.0, float(can_frame_bits_est))

    def start(self):
        """Start the ODrive CAN manager."""
        logger.info(f"Starting ODrive CAN manager on {self.canbus}")
        try:
            self.manager = ODriveCanManager(self.canbus)

            # Register axes
            for aid in self.axis_ids:
                axis = self.manager.add_axis(aid)
                self.axes[aid] = axis

                # Always register handlers; _handle_* methods dispatch only if
                # corresponding external callbacks are set.
                axis.on_encoder(lambda pos, vel, i=aid: self._handle_encoder(i, pos, vel))
                axis.on_bus(lambda vbus, ibus, i=aid: self._handle_bus(i, vbus, ibus))
                axis.on_iq(lambda iq_set, iq_meas, i=aid: self._handle_current(i, iq_meas))
                axis.on_temp(lambda fet, motor, i=aid: self._handle_temp(i, fet, motor))
                axis.on_heartbeat(lambda err, st, proc, i=aid: self._handle_heartbeat(i, err, st, proc))

                logger.info(f"Registered axis {aid}")

        except Exception as e:
            logger.error(f"Failed to start hardware driver: {e}")
            raise

    def stop(self):
        """Stop the ODrive CAN manager."""
        if self.manager:
            try:
                self.manager.close()
            except Exception as e:
                logger.warning(f"Error closing CAN manager: {e}")
            self.manager = None
        logger.info("Hardware driver stopped")

    def set_axis_position(self, axis_id: int, position: float):
        """Set position setpoint."""
        axis = self.axes.get(axis_id)
        if axis:
            axis.set_input_pos(position)

    def set_axis_torque(self, axis_id: int, torque: float):
        """Set torque setpoint."""
        axis = self.axes.get(axis_id)
        if axis:
            trq = float(torque)
            axis.set_input_torque(trq)
            idx = self._axis_index.get(axis_id)
            if idx is not None:
                with self._lock:
                    self._axis_torque_cmd_nm[idx] = trq

    def get_axis_position(self, axis_id: int) -> Optional[float]:
        """Get position feedback in turns."""
        idx = self._axis_index.get(axis_id)
        if idx is None:
            return None
        with self._lock:
            return self._axis_pos_turns[idx]

    def get_axis_velocity(self, axis_id: int) -> Optional[float]:
        """Get velocity feedback in turns/s."""
        idx = self._axis_index.get(axis_id)
        if idx is None:
            return None
        with self._lock:
            return self._axis_vel_turnsps[idx]

    def set_controller_mode(self, axis_id: int, mode: str):
        """Set controller mode."""
        axis = self.axes.get(axis_id)
        if not axis:
            return

        if mode == "position":
            control_mode = 3  # CONTROL_MODE_POSITION
            input_mode = 1    # INPUT_MODE_PASSTHROUGH
        elif mode == "torque":
            control_mode = 1  # CONTROL_MODE_TORQUE
            input_mode = 1    # INPUT_MODE_PASSTHROUGH
        else:
            logger.warning(f"Unknown controller mode: {mode}")
            return

        axis.set_controller_mode(control_mode, input_mode)

    def set_axis_state(self, axis_id: int, state: str):
        """Set axis state."""
        axis = self.axes.get(axis_id)
        if not axis:
            return

        if state == "idle":
            axis.set_axis_state(AxisState.IDLE)
        elif state == "closed_loop":
            axis.set_axis_state(AxisState.CLOSED_LOOP_CONTROL)
        else:
            logger.warning(f"Unknown axis state: {state}")

    def set_absolute_position(self, axis_id: int, position: float):
        """Set absolute position reference."""
        axis = self.axes.get(axis_id)
        if axis:
            axis.set_absolute_position(position)

    def get_axis_torques(self):
        """Return latest commanded axis torques [Nm]."""
        with self._lock:
            return [float(x) for x in self._axis_torque_cmd_nm]

    def get_cable_tensions(self):
        """
        Return inferred cable tensions [N] from commanded torque.
        Positive tension means pulling force along cable.
        """
        scale = self.torque_direction * self.capstan_radius_m
        if abs(scale) < 1e-9:
            return None
        with self._lock:
            return [float(tau) / scale for tau in self._axis_torque_cmd_nm]

    def get_platform_state(self):
        """
        Estimate platform state [x,y,z,roll,pitch], [xd,yd,zd,rolld,pitchd].
        Position is from cable-length FK (yaw fixed to 0); velocity from J*qdot=Ldot least squares.
        Units: meters, radians, m/s, rad/s.
        """
        with self._lock:
            pos = list(self._axis_pos_turns)
            vel = list(self._axis_vel_turnsps)
            q_cached = self._pose_est_q.copy()
            qd_cached = self._pose_est_qd.copy()
            t_cached = self._pose_last_perf

        if any(v is None for v in pos) or any(v is None for v in vel):
            return None, None

        now = time.perf_counter()
        if (now - t_cached) < self._pose_update_dt:
            return q_cached, qd_cached

        L_meas_m = np.array(
            [
                (float(self._home_cable_mm[i]) + float(pos[i]) * float(self.mm_per_turn[i])) / 1000.0
                for i in range(len(self.axis_ids))
            ],
            dtype=float,
        )
        Ldot_meas_mps = np.array(
            [float(vel[i]) * float(self.mm_per_turn[i]) / 1000.0 for i in range(len(self.axis_ids))],
            dtype=float,
        )

        try:
            q_new = self._solve_pose_from_lengths(L_meas_m, q_cached)
            J = self._cable_lengths_jacobian_pose5_fd(q_new)
            qd_new, *_ = np.linalg.lstsq(J, Ldot_meas_mps, rcond=None)
            qd_new = np.asarray(qd_new, dtype=float)
        except Exception as exc:
            logger.debug(f"Platform-state estimator failed: {exc}")
            return q_cached, qd_cached

        with self._lock:
            self._pose_est_q = q_new
            self._pose_est_qd = qd_new
            self._pose_last_perf = now
        return q_new.copy(), qd_new.copy()

    def get_cable_jacobian_plat(self):
        """Return dL/dq for q=[x,y,z,roll,pitch], shape (6,5)."""
        q, _ = self.get_platform_state()
        if q is None:
            return np.zeros((len(self.axis_ids), 5), dtype=float)
        return self._cable_lengths_jacobian_pose5_fd(np.asarray(q, dtype=float))

    def get_comm_stats(self):
        """
        Return communication stats for UI/debugging.
        Keys: can_rx_hz, can_tx_hz, can_msg_hz, can_util_est, pos_fbk_hz.
        """
        can_rx_hz = float("nan")
        can_tx_hz = float("nan")
        can_msg_hz = float("nan")
        can_util_est = float("nan")
        pos_fbk_hz = float("nan")
        pos_fbk_period0_min_s = float("nan")
        pos_fbk_period0_max_s = float("nan")

        if self.manager and hasattr(self.manager, "get_rate_stats"):
            try:
                d = self.manager.get_rate_stats(window_s=1.0)
                can_rx_hz = float(d.get("rx_rate_hz", float("nan")))
                can_tx_hz = float(d.get("tx_rate_hz", float("nan")))
                can_msg_hz = can_rx_hz + can_tx_hz
                can_util_est = (can_msg_hz * self._can_frame_bits_est) / self._can_bitrate
            except Exception:
                pass

        now = time.perf_counter()
        with self._lock:
            tmin = now - 1.0
            total = 0
            for dq in self._enc_times_axes:
                while dq and dq[0] < tmin:
                    dq.popleft()
                total += len(dq)
            if self._enc_times_axes:
                pos_fbk_hz = float(total) / float(len(self._enc_times_axes))

            if self._enc_times_axes and len(self._enc_times_axes[0]) >= 2:
                t0 = self._enc_times_axes[0]
                periods = [float(t0[i] - t0[i - 1]) for i in range(1, len(t0))]
                if periods:
                    pos_fbk_period0_min_s = float(min(periods))
                    pos_fbk_period0_max_s = float(max(periods))

        return {
            "can_rx_hz": can_rx_hz,
            "can_tx_hz": can_tx_hz,
            "can_msg_hz": can_msg_hz,
            "can_util_est": can_util_est,
            "pos_fbk_hz": pos_fbk_hz,
            "pos_fbk_period0_min_s": pos_fbk_period0_min_s,
            "pos_fbk_period0_max_s": pos_fbk_period0_max_s,
        }

    # Callback setters
    def set_position_callback(self, callback: Callable[[int, float], None]):
        self._position_callback = callback

    def set_velocity_callback(self, callback: Callable[[int, float], None]):
        self._velocity_callback = callback

    def set_bus_callback(self, callback: Callable[[int, float, float], None]):
        self._bus_callback = callback

    def set_current_callback(self, callback: Callable[[int, float], None]):
        self._current_callback = callback

    def set_temp_callback(self, callback: Callable[[int, float, float], None]):
        self._temp_callback = callback

    def set_heartbeat_callback(self, callback: Callable[[int, int, int, int], None]):
        self._heartbeat_callback = callback

    # Internal callback handlers
    def _handle_encoder(self, axis_id: int, pos: float, vel: float):
        idx = self._axis_index.get(axis_id)
        if idx is not None:
            with self._lock:
                self._axis_pos_turns[idx] = float(pos)
                self._axis_vel_turnsps[idx] = float(vel)
                self._enc_times_axes[idx].append(time.perf_counter())
        if self._position_callback:
            self._position_callback(axis_id, pos)
        if self._velocity_callback:
            self._velocity_callback(axis_id, vel)

    def _handle_bus(self, axis_id: int, vbus: float, ibus: float):
        if self._bus_callback:
            self._bus_callback(axis_id, vbus, ibus)

    def _handle_current(self, axis_id: int, current: float):
        idx = self._axis_index.get(axis_id)
        if idx is not None:
            with self._lock:
                self._axis_current_a[idx] = float(current)
        if self._current_callback:
            self._current_callback(axis_id, current)

    def _handle_temp(self, axis_id: int, fet: float, motor: float):
        if self._temp_callback:
            self._temp_callback(axis_id, fet, motor)

    def _handle_heartbeat(self, axis_id: int, error: int, state: int, proc_result: int):
        if self._heartbeat_callback:
            self._heartbeat_callback(axis_id, error, state, proc_result)

    @staticmethod
    def _quat_from_rp_rad(roll_rad: float, pitch_rad: float):
        qx = q_from_axis_angle((1.0, 0.0, 0.0), float(roll_rad))
        qy = q_from_axis_angle((0.0, 1.0, 0.0), float(pitch_rad))
        return q_norm(q_mul(qy, qx))

    def _cable_lengths_m_from_pose5(self, pose5):
        x_m, y_m, z_m, roll_rad, pitch_rad = [float(v) for v in pose5]
        t_mm = (1000.0 * x_m, 1000.0 * y_m, 1000.0 * z_m)
        q = self._quat_from_rp_rad(roll_rad, pitch_rad)
        L_mm = pose_to_cable_lengths_mm(self._geom, t_mm, q)
        return np.asarray(L_mm, dtype=float) / 1000.0

    def _cable_lengths_jacobian_pose5_fd(self, pose5, eps_pos_m=1e-4, eps_ang_rad=1e-4):
        q0 = np.asarray(pose5, dtype=float).copy()
        J = np.zeros((len(self.axis_ids), 5), dtype=float)
        for j in range(5):
            dq = np.zeros(5, dtype=float)
            dq[j] = eps_pos_m if j < 3 else eps_ang_rad
            Lp = self._cable_lengths_m_from_pose5(q0 + dq)
            Lm = self._cable_lengths_m_from_pose5(q0 - dq)
            J[:, j] = (Lp - Lm) / (2.0 * dq[j])
        return J

    def _solve_pose_from_lengths(self, L_meas_m, q_seed):
        q = np.asarray(q_seed, dtype=float).copy()
        q[2] = max(-0.6, min(0.6, q[2]))
        for _ in range(4):
            L_pred = self._cable_lengths_m_from_pose5(q)
            r = np.asarray(L_meas_m, dtype=float) - L_pred
            J = self._cable_lengths_jacobian_pose5_fd(q)
            dq, *_ = np.linalg.lstsq(J, r, rcond=None)

            # Keep the solver stable under noisy feedback.
            dq[0:3] = np.clip(dq[0:3], -0.01, 0.01)
            dq[3:5] = np.clip(dq[3:5], -0.05, 0.05)
            q = q + dq

            q[0:3] = np.clip(q[0:3], -0.6, 0.6)
            q[3:5] = np.clip(q[3:5], -math.radians(60.0), math.radians(60.0))
            if float(np.linalg.norm(r)) < 1e-5:
                break
        return q
