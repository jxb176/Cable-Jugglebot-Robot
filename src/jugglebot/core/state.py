"""Mutable runtime mailbox for commands, telemetry, and profile control."""

from __future__ import annotations

import logging
import threading

from jugglebot.core.cable_ik import q_norm
from jugglebot.core.types import (
    RuntimeHealthLevel,
    TimingStats,
    TrajectoryUpdate,
    TrajectoryUpdateMode,
    WatchdogStatus,
)


logger = logging.getLogger("robot")


def _clone_watchdog_status(watchdog_status: WatchdogStatus | None):
    if watchdog_status is None:
        return None
    data = watchdog_status.to_dict()
    level = data.get("level", RuntimeHealthLevel.HEALTHY.value)
    data["level"] = RuntimeHealthLevel(level)
    return WatchdogStatus(**data)


class RuntimeMailbox:
    """Thread-safe shared state for the current runtime process."""

    def __init__(self):
        self.lock = threading.Lock()
        self.controller_ip = None
        self.state = "disable"
        self.state_version = 0
        self.home_pos = [0.0] * 6
        self.home_version = 0
        self.player_thread = None
        self.profile_active = False
        self.axes_pos_estimate = [None] * 6
        self.axes_vel_estimate = [None] * 6
        self.axes_bus_voltage = [None] * 6
        self.axes_bus_current = [None] * 6
        self.axes_motor_current = [None] * 6
        self.axes_torque_cmd_nm = [None] * 6
        self.axes_torque_rsp_nm = [None] * 6
        self.axes_tension_cmd_n = [None] * 6
        self.axes_tension_rsp_n = [None] * 6
        self.axes_temp_fet = [None] * 6
        self.axes_temp_motor = [None] * 6
        self.axes_axis_error = [None] * 6
        self.axes_axis_state = [None] * 6
        self.axes_proc_result = [None] * 6
        self.telem_thread = None
        self.telem_stop = threading.Event()
        self.pret_upper_N = 0.0
        self.pret_lower_N = 0.0
        self.pret_version = 0
        self.spool_gain_version = 0
        self.spool_kp_mult = 1.0
        self.spool_kd_mult = 1.0
        self.hand_t_mm = (0.0, 0.0, 0.0)
        self.hand_q = (1.0, 0.0, 0.0, 0.0)
        self.hand_v_mps = (0.0, 0.0, 0.0)
        self.hand_a_mps2 = (0.0, 0.0, 0.0)
        self.hand_version = 0
        self.hand_est_t_mm = (float("nan"), float("nan"), float("nan"))
        self.hand_est_q = (1.0, 0.0, 0.0, 0.0)
        self.hand_est_v_mps = (float("nan"), float("nan"), float("nan"))
        self.hand_est_w_rps = (float("nan"), float("nan"), float("nan"))
        self.comm_can_rx_hz = float("nan")
        self.comm_can_tx_hz = float("nan")
        self.comm_can_msg_hz = float("nan")
        self.comm_can_util_est = float("nan")
        self.comm_pos_fbk_hz = float("nan")
        self.comm_pos_fbk_period0_min_s = float("nan")
        self.comm_pos_fbk_period0_max_s = float("nan")
        self.pose_profile = []
        self.pending_trajectory_update: TrajectoryUpdate | None = None
        self.trajectory_update_version = 0
        self.snapshot_sequence = 0
        self.control_time_s = None
        self.timing_stats: TimingStats | None = None
        self.watchdog_status: WatchdogStatus | None = None

    def set_hand_pose(self, t_mm, q, v_mps=None, a_mps2=None):
        with self.lock:
            self._set_hand_motion_locked(t_mm, q, v_mps=v_mps, a_mps2=a_mps2)
            self.hand_version += 1

    def set_commanded_motion_sample(self, t_mm, q, v_mps=None, a_mps2=None):
        with self.lock:
            self._set_hand_motion_locked(t_mm, q, v_mps=v_mps, a_mps2=a_mps2)

    def _set_hand_motion_locked(self, t_mm, q, v_mps=None, a_mps2=None):
        self.hand_t_mm = (float(t_mm[0]), float(t_mm[1]), float(t_mm[2]))
        self.hand_q = q_norm((float(q[0]), float(q[1]), float(q[2]), float(q[3])))
        if v_mps is None:
            self.hand_v_mps = (0.0, 0.0, 0.0)
        else:
            self.hand_v_mps = (float(v_mps[0]), float(v_mps[1]), float(v_mps[2]))
        if a_mps2 is None:
            self.hand_a_mps2 = (0.0, 0.0, 0.0)
        else:
            self.hand_a_mps2 = (float(a_mps2[0]), float(a_mps2[1]), float(a_mps2[2]))

    def get_hand_pose(self):
        with self.lock:
            return self.hand_t_mm, self.hand_q

    def get_hand_motion(self):
        with self.lock:
            return self.hand_t_mm, self.hand_q, self.hand_v_mps, self.hand_a_mps2

    def set_hand_estimate(self, t_mm, q, v_mps=None, w_rps=None):
        with self.lock:
            self.hand_est_t_mm = (float(t_mm[0]), float(t_mm[1]), float(t_mm[2]))
            self.hand_est_q = q_norm((float(q[0]), float(q[1]), float(q[2]), float(q[3])))
            if v_mps is None:
                self.hand_est_v_mps = (float("nan"), float("nan"), float("nan"))
            else:
                self.hand_est_v_mps = (float(v_mps[0]), float(v_mps[1]), float(v_mps[2]))
            if w_rps is None:
                self.hand_est_w_rps = (float("nan"), float("nan"), float("nan"))
            else:
                self.hand_est_w_rps = (float(w_rps[0]), float(w_rps[1]), float(w_rps[2]))

    def get_hand_estimate(self):
        with self.lock:
            return self.hand_est_t_mm, self.hand_est_q, self.hand_est_v_mps, self.hand_est_w_rps

    def set_comm_stats(
        self,
        can_rx_hz=None,
        can_tx_hz=None,
        can_msg_hz=None,
        can_util_est=None,
        pos_fbk_hz=None,
        pos_fbk_period0_min_s=None,
        pos_fbk_period0_max_s=None,
    ):
        with self.lock:
            if can_rx_hz is not None:
                self.comm_can_rx_hz = float(can_rx_hz)
            if can_tx_hz is not None:
                self.comm_can_tx_hz = float(can_tx_hz)
            if can_msg_hz is not None:
                self.comm_can_msg_hz = float(can_msg_hz)
            if can_util_est is not None:
                self.comm_can_util_est = float(can_util_est)
            if pos_fbk_hz is not None:
                self.comm_pos_fbk_hz = float(pos_fbk_hz)
            if pos_fbk_period0_min_s is not None:
                self.comm_pos_fbk_period0_min_s = float(pos_fbk_period0_min_s)
            if pos_fbk_period0_max_s is not None:
                self.comm_pos_fbk_period0_max_s = float(pos_fbk_period0_max_s)

    def get_comm_stats(self):
        with self.lock:
            return {
                "can_rx_hz": float(self.comm_can_rx_hz),
                "can_tx_hz": float(self.comm_can_tx_hz),
                "can_msg_hz": float(self.comm_can_msg_hz),
                "can_util_est": float(self.comm_can_util_est),
                "pos_fbk_hz": float(self.comm_pos_fbk_hz),
                "pos_fbk_period0_min_s": float(self.comm_pos_fbk_period0_min_s),
                "pos_fbk_period0_max_s": float(self.comm_pos_fbk_period0_max_s),
            }

    def get_hand_version(self):
        with self.lock:
            return int(self.hand_version)

    def set_control_time_s(self, t_s):
        with self.lock:
            self.control_time_s = None if t_s is None else float(t_s)

    def get_control_time_s(self):
        with self.lock:
            return self.control_time_s

    def set_timing_stats(self, timing_stats: TimingStats | None):
        with self.lock:
            if timing_stats is None:
                self.timing_stats = None
            else:
                self.timing_stats = TimingStats(**timing_stats.to_dict())

    def get_timing_stats(self):
        with self.lock:
            if self.timing_stats is None:
                return None
            return TimingStats(**self.timing_stats.to_dict())

    def set_watchdog_status(self, watchdog_status: WatchdogStatus | None):
        with self.lock:
            self.watchdog_status = _clone_watchdog_status(watchdog_status)

    def get_watchdog_status(self):
        with self.lock:
            return _clone_watchdog_status(self.watchdog_status)

    def next_snapshot_sequence(self):
        with self.lock:
            self.snapshot_sequence += 1
            return int(self.snapshot_sequence)

    def set_controller_ip(self, ip):
        with self.lock:
            self.controller_ip = ip
        logger.info(f"Controller IP set to {ip}")

    def get_controller_ip(self):
        with self.lock:
            return self.controller_ip

    def get_pos_fbk(self):
        with self.lock:
            return list(self.axes_pos_estimate)

    def get_vel_fbk(self):
        with self.lock:
            return list(self.axes_vel_estimate)

    def request_home(self, home_pos):
        if not isinstance(home_pos, (list, tuple)) or len(home_pos) != 6:
            raise ValueError("home_pos must be length-6 list/tuple")
        with self.lock:
            self.home_pos = [float(x) for x in home_pos]
            self.home_version += 1
        logger.info("HOME requested (mm): " + ", ".join(f"{x:.3f}" for x in self.home_pos))

    def get_home_version(self):
        with self.lock:
            return self.home_version

    def get_home_pos(self):
        with self.lock:
            return list(self.home_pos)

    def set_axis_feedback(
        self,
        axis_id: int,
        pos_estimate=None,
        vel_estimate=None,
        bus_voltage=None,
        bus_current=None,
        motor_current=None,
        temp_fet=None,
        temp_motor=None,
        axis_error=None,
        axis_state=None,
        proc_result=None,
    ):
        if not (0 <= int(axis_id) < 6):
            return
        with self.lock:
            if pos_estimate is not None:
                try:
                    self.axes_pos_estimate[axis_id] = float(pos_estimate)
                except Exception:
                    pass
            if vel_estimate is not None:
                try:
                    self.axes_vel_estimate[axis_id] = float(vel_estimate)
                except Exception:
                    pass
            if bus_voltage is not None:
                try:
                    self.axes_bus_voltage[axis_id] = float(bus_voltage)
                except Exception:
                    pass
            if bus_current is not None:
                try:
                    self.axes_bus_current[axis_id] = float(bus_current)
                except Exception:
                    pass
            if motor_current is not None:
                try:
                    self.axes_motor_current[axis_id] = float(motor_current)
                except Exception:
                    pass
            if temp_fet is not None:
                try:
                    self.axes_temp_fet[axis_id] = float(temp_fet)
                except Exception:
                    pass
            if temp_motor is not None:
                try:
                    self.axes_temp_motor[axis_id] = float(temp_motor)
                except Exception:
                    pass
            if axis_error is not None:
                try:
                    self.axes_axis_error[axis_id] = int(axis_error)
                except Exception:
                    pass
            if axis_state is not None:
                try:
                    self.axes_axis_state[axis_id] = int(axis_state)
                except Exception:
                    pass
            if proc_result is not None:
                try:
                    self.axes_proc_result[axis_id] = int(proc_result)
                except Exception:
                    pass

    def get_bus_voltage(self):
        with self.lock:
            return list(self.axes_bus_voltage)

    def get_bus_current(self):
        with self.lock:
            return list(self.axes_bus_current)

    def get_motor_current(self):
        with self.lock:
            return list(self.axes_motor_current)

    def set_axis_torque_telemetry(self, torque_cmd_nm=None, torque_rsp_nm=None):
        with self.lock:
            if torque_cmd_nm is not None:
                for i in range(min(6, len(torque_cmd_nm))):
                    try:
                        self.axes_torque_cmd_nm[i] = float(torque_cmd_nm[i])
                    except Exception:
                        self.axes_torque_cmd_nm[i] = None
            if torque_rsp_nm is not None:
                for i in range(min(6, len(torque_rsp_nm))):
                    try:
                        self.axes_torque_rsp_nm[i] = float(torque_rsp_nm[i])
                    except Exception:
                        self.axes_torque_rsp_nm[i] = None

    def get_axis_torque_command(self):
        with self.lock:
            return list(self.axes_torque_cmd_nm)

    def get_axis_torque_response(self):
        with self.lock:
            return list(self.axes_torque_rsp_nm)

    def set_axis_tension_telemetry(self, tension_cmd_n=None, tension_rsp_n=None):
        with self.lock:
            if tension_cmd_n is not None:
                for i in range(min(6, len(tension_cmd_n))):
                    try:
                        self.axes_tension_cmd_n[i] = float(tension_cmd_n[i])
                    except Exception:
                        self.axes_tension_cmd_n[i] = None
            if tension_rsp_n is not None:
                for i in range(min(6, len(tension_rsp_n))):
                    try:
                        self.axes_tension_rsp_n[i] = float(tension_rsp_n[i])
                    except Exception:
                        self.axes_tension_rsp_n[i] = None

    def get_axis_tension_command(self):
        with self.lock:
            return list(self.axes_tension_cmd_n)

    def get_axis_tension_response(self):
        with self.lock:
            return list(self.axes_tension_rsp_n)

    def get_temp_fet(self):
        with self.lock:
            return list(self.axes_temp_fet)

    def get_temp_motor(self):
        with self.lock:
            return list(self.axes_temp_motor)

    def get_axis_error(self):
        with self.lock:
            return list(self.axes_axis_error)

    def get_axis_state(self):
        with self.lock:
            return list(self.axes_axis_state)

    def get_proc_result(self):
        with self.lock:
            return list(self.axes_proc_result)

    def set_state(self, value: str):
        value = str(value).lower()
        if value not in ("enable", "disable", "estop", "pretension"):
            raise ValueError("invalid state")
        with self.lock:
            self.state = value
            self.state_version += 1
        logger.info(f"State set to: {value} (version {self.state_version})")

    def get_state(self):
        with self.lock:
            return self.state

    def get_state_version(self):
        with self.lock:
            return self.state_version

    def stop_profile(self):
        with self.lock:
            player = self.player_thread
            self.player_thread = None
            hold_pose = (
                tuple(self.hand_t_mm),
                tuple(self.hand_q),
                tuple(self.hand_v_mps),
                tuple(self.hand_a_mps2),
            )
            self.profile_active = False
        if player and player.is_alive():
            player.stop()
            player.join(timeout=1.0)
            logger.info("Profile stopped")
        self.submit_pose_command(
            hold_pose[0],
            hold_pose[1],
            v_mps=hold_pose[2],
            a_mps2=hold_pose[3],
        )

    def set_pose_profile(self, profile_pose: list):
        norm = []
        for row in profile_pose:
            if not isinstance(row, (list, tuple)) or len(row) < 7:
                raise ValueError("each pose profile row must be [t, x,y,z,roll,pitch,yaw]")
            t = float(row[0])
            if len(row) >= 13:
                pose6 = [
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[10]),
                    float(row[11]),
                    float(row[12]),
                ]
                v3 = [float(row[4]), float(row[5]), float(row[6])]
                a3 = [float(row[7]), float(row[8]), float(row[9])]
            else:
                pose6 = [float(x) for x in row[1:7]]
                v3 = [0.0, 0.0, 0.0]
                a3 = [0.0, 0.0, 0.0]
            norm.append((t, pose6, v3, a3))
        with self.lock:
            self.pose_profile = norm
        logger.info(f"Pose profile uploaded with {len(norm)} points")

    def get_pose_profile(self):
        with self.lock:
            return list(self.pose_profile)

    def set_profile_active(self, active: bool):
        with self.lock:
            self.profile_active = bool(active)

    def get_profile_active(self):
        with self.lock:
            return bool(self.profile_active)

    def submit_trajectory_update(self, update: TrajectoryUpdate):
        if not isinstance(update, TrajectoryUpdate):
            raise TypeError("update must be a TrajectoryUpdate")
        with self.lock:
            self.pending_trajectory_update = update
            self.trajectory_update_version += 1
        logger.info(
            f"[TRAJ] queued trajectory update mode={update.mode.value} "
            f"(version {self.trajectory_update_version})"
        )

    def get_pending_trajectory_update(self):
        with self.lock:
            return self.pending_trajectory_update

    def take_pending_trajectory_update(self):
        with self.lock:
            update = self.pending_trajectory_update
            self.pending_trajectory_update = None
            return update

    def get_trajectory_update_version(self):
        with self.lock:
            return int(self.trajectory_update_version)

    def _next_trajectory_sequence_id(self) -> int:
        return int(self.get_trajectory_update_version()) + 1

    def submit_pose_command(
        self,
        t_mm,
        q,
        v_mps=None,
        a_mps2=None,
        *,
        mode: TrajectoryUpdateMode = TrajectoryUpdateMode.REPLACE,
        effective_time_s: float | None = None,
        preserve_continuity: bool = True,
    ):
        from jugglebot.core.types import PoseCommand, TrajectoryCommand, TrajectoryUpdate, TrajectoryUpdateMode, TrajectoryWaypoint
        from jugglebot.core.pose_utils import quat_to_rpy_rad

        roll_rad, pitch_rad, yaw_rad = quat_to_rpy_rad(q)
        sequence_id = self._next_trajectory_sequence_id()
        update = TrajectoryUpdate(
            sequence_id=sequence_id,
            mode=mode,
            trajectory=TrajectoryCommand(
                sequence_id=sequence_id,
                waypoints=(
                    TrajectoryWaypoint(
                        time_from_start_s=0.0,
                        pose=PoseCommand(
                            x_m=float(t_mm[0]) / 1000.0,
                            y_m=float(t_mm[1]) / 1000.0,
                            z_m=float(t_mm[2]) / 1000.0,
                            roll_rad=float(roll_rad),
                            pitch_rad=float(pitch_rad),
                            yaw_rad=float(yaw_rad),
                            linear_velocity_mps=(
                                0.0 if v_mps is None else float(v_mps[0]),
                                0.0 if v_mps is None else float(v_mps[1]),
                                0.0 if v_mps is None else float(v_mps[2]),
                            ),
                            linear_acceleration_mps2=(
                                0.0 if a_mps2 is None else float(a_mps2[0]),
                                0.0 if a_mps2 is None else float(a_mps2[1]),
                                0.0 if a_mps2 is None else float(a_mps2[2]),
                            ),
                        ),
                    ),
                ),
            ),
            effective_time_s=effective_time_s,
            preserve_continuity=bool(preserve_continuity),
        )
        self.submit_trajectory_update(update)

    def start_pose_profile(
        self,
        *,
        mode: TrajectoryUpdateMode = TrajectoryUpdateMode.REPLACE,
        effective_time_s: float | None = None,
        preserve_continuity: bool = True,
    ):
        from jugglebot.core.types import PoseCommand, TrajectoryCommand, TrajectoryUpdate, TrajectoryUpdateMode, TrajectoryWaypoint

        prof = self.get_pose_profile()
        if not prof:
            raise RuntimeError("no pose profile uploaded")
        with self.lock:
            self.player_thread = None
            self.profile_active = True
        logger.info("Pose profile start requested")
        t0 = float(prof[0][0])
        waypoints = []
        for t, pose6, v3, a3 in prof:
            waypoints.append(
                TrajectoryWaypoint(
                    time_from_start_s=float(t) - t0,
                    pose=PoseCommand(
                        x_m=float(pose6[0]) / 1000.0,
                        y_m=float(pose6[1]) / 1000.0,
                        z_m=float(pose6[2]) / 1000.0,
                        roll_rad=float(pose6[3]) * 3.141592653589793 / 180.0,
                        pitch_rad=float(pose6[4]) * 3.141592653589793 / 180.0,
                        yaw_rad=float(pose6[5]) * 3.141592653589793 / 180.0,
                        linear_velocity_mps=(float(v3[0]), float(v3[1]), float(v3[2])),
                        linear_acceleration_mps2=(float(a3[0]), float(a3[1]), float(a3[2])),
                    ),
                )
            )
        sequence_id = self._next_trajectory_sequence_id()
        self.submit_trajectory_update(
            TrajectoryUpdate(
                sequence_id=sequence_id,
                mode=mode,
                trajectory=TrajectoryCommand(
                    sequence_id=sequence_id,
                    waypoints=tuple(waypoints),
                ),
                effective_time_s=effective_time_s,
                preserve_continuity=bool(preserve_continuity),
            )
        )

    def start_telem(self, udp_sock, controller_addr):
        from jugglebot.transport.udp_telemetry import udp_telemetry_sender

        self.stop_telem()
        self.telem_stop.clear()
        t = threading.Thread(
            target=udp_telemetry_sender,
            args=(self, udp_sock, self.telem_stop),
            daemon=True,
        )
        self.telem_thread = t
        t.start()
        logger.info("[UDP] Telemetry thread started")

    def stop_telem(self):
        if self.telem_thread and self.telem_thread.is_alive():
            self.telem_stop.set()
            self.telem_thread.join(timeout=1.0)
            logger.info("[UDP] Telemetry thread stopped")
        self.telem_thread = None

    def request_pretension(self, upper_N: float, lower_N: float):
        with self.lock:
            self.pret_upper_N = float(upper_N)
            self.pret_lower_N = float(lower_N)
            self.pret_version += 1
            self.state = "pretension"
            self.state_version += 1
        logger.info(
            f"[PRET] requested upper={self.pret_upper_N:.3f} N, lower={self.pret_lower_N:.3f} N "
            f"(pret_version {self.pret_version})"
        )

    def get_pretension(self):
        with self.lock:
            return float(self.pret_upper_N), float(self.pret_lower_N)

    def get_pretension_version(self):
        with self.lock:
            return int(self.pret_version)

    def request_spool_gain_multipliers(self, kp=None, kd=None):
        with self.lock:
            if kp is not None:
                self.spool_kp_mult = float(kp)
            if kd is not None:
                self.spool_kd_mult = float(kd)
            self.spool_gain_version += 1
        logger.info(
            "[SPOOL_GAIN] multipliers set: "
            f"kp={self.spool_kp_mult:.3f}, kd={self.spool_kd_mult:.3f}"
        )

    def request_task_gain_multipliers(self, kp_xyz=None, kp_rp=None, kd_xyz=None, kd_rp=None):
        kp_terms = [float(v) for v in (kp_xyz, kp_rp) if v is not None]
        kd_terms = [float(v) for v in (kd_xyz, kd_rp) if v is not None]
        kp = sum(kp_terms) / len(kp_terms) if kp_terms else None
        kd = sum(kd_terms) / len(kd_terms) if kd_terms else None
        self.request_spool_gain_multipliers(kp=kp, kd=kd)

    def get_spool_gain_multipliers(self):
        with self.lock:
            return float(self.spool_kp_mult), float(self.spool_kd_mult)

    def get_task_gain_multipliers(self):
        kp, kd = self.get_spool_gain_multipliers()
        return kp, kp, kd, kd

    def get_spool_gain_version(self):
        with self.lock:
            return int(self.spool_gain_version)

    def get_task_gain_version(self):
        return self.get_spool_gain_version()
