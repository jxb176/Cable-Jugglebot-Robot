# runner.py
import time
import threading
import os
import csv
import logging
from datetime import datetime
import subprocess
import math
import numpy as np
# --- Cable IK ---
from jugglebot.core.cable_ik import (
    CableRobotGeometry,
    pose_to_cable_lengths_mm,
)
from jugglebot.core.controller_step import (
    CableSpaceFallbackConfig,
    NullspaceControllerConfig,
    TaskspaceControllerConfig,
    TaskspaceControllerState,
    compute_cablespace_fallback_commands,
    compute_taskspace_spool_commands,
)
from jugglebot.io.actuator_bus import ActuatorBus
from jugglebot.core.platform_estimator import CablePlatformEstimator
from jugglebot.core.pose_utils import quat_from_rpy_deg
from jugglebot.core.state import RuntimeMailbox
from jugglebot.core.types import ActuatorCommand, ActuatorControlMode, ActuatorState, TimingStats
from jugglebot.core.tension_control import TensionAllocatorConfig, platform_wrench_from_tensions
from jugglebot.core.units import mm_to_turns
from jugglebot.rt.clock import WallClock
from jugglebot.rt.config import RuntimeConfig, parse_runtime_config
from jugglebot.rt.homing import HomingRoutineManager
from jugglebot.rt.manual_input import ManualInputController
from jugglebot.rt.state_machine import RuntimeMode, RuntimeStateMachine, RuntimeTransitionAction
from jugglebot.rt.trajectory_manager import TrajectoryManager
from jugglebot.rt.watchdog import RuntimeWatchdog, WatchdogSample

OUTER_CORR_KP = np.diag([1.0, 1.0, 1.0, 0.35, 0.35])
OUTER_CORR_KD = np.diag([0.15, 0.15, 0.15, 0.05, 0.05])
OUTER_CORR_CABLE_CLIP_M = 0.10

# Geometry (mm)
GEOM = CableRobotGeometry()

# Define the pose that corresponds to your "HOME" physical configuration
# IMPORTANT: this must match how you physically home the platform.
HOME_T_WORLD_MM = (0.0, 0.0, 0.0)
HOME_ROLL_DEG = 0.0
HOME_PITCH_DEG = 0.0
HOME_YAW_DEG = 0.0  # fixed assumption

# Precompute "home" geometric cable lengths in mm (used to convert absolute lengths -> delta lengths)
HOME_Q = quat_from_rpy_deg(HOME_ROLL_DEG, HOME_PITCH_DEG, HOME_YAW_DEG)
HOME_CABLE_MM = pose_to_cable_lengths_mm(GEOM, HOME_T_WORLD_MM, HOME_Q)  # returns mm given your mm geometry

def _expand_axis_values(value, default, axis_count: int = 6):
    if value is None:
        return [float(default)] * axis_count
    if isinstance(value, (list, tuple)):
        if len(value) != axis_count:
            raise ValueError(f"Expected length-{axis_count} list/tuple, got {len(value)}")
        return [float(v) for v in value]
    return [float(value)] * axis_count

# -------- Logging setup --------
def _init_logging():
    logs_dir = os.path.join(os.getcwd(), "Logs")
    os.makedirs(logs_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(logs_dir, f"robot_{ts}.log")
    logger = logging.getLogger("robot")
    logger.setLevel(logging.DEBUG)   #INFO for low level, Set to DEBUG for more verbose logging
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.handlers.clear()
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.propagate = False
    return logger, log_path

logger, LOG_FILE_PATH = _init_logging()


def ensure_can_interface_up(ifname: str, bitrate: int) -> bool:
    try:
        res = subprocess.run(
            ["ip", "link", "show", ifname],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2,
        )
        if res.returncode == 0:
            out = res.stdout.lower()
            if " state up " in out or "<up," in out or "up>" in out:
                logger.info(f"[CAN] Interface {ifname} already UP")
                return True
        else:
            logger.warning(f"[CAN] '{ifname}' not found: {res.stderr.strip()}")

        logger.info(f"[CAN] Bringing up {ifname} @ {bitrate} bps")
        subprocess.run(["ip", "link", "set", ifname, "down"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cfg = subprocess.run(
            ["ip", "link", "set", ifname, "type", "can", "bitrate", str(bitrate)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        if cfg.returncode != 0:
            logger.error(f"[CAN] Failed to configure {ifname}: {cfg.stderr.strip()}")
            return False
        up = subprocess.run(
            ["ip", "link", "set", ifname, "up"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3,
        )
        if up.returncode != 0:
            logger.error(f"[CAN] Failed to bring {ifname} up: {up.stderr.strip()}")
            return False
        return True
    except Exception as e:
        logger.error(f"[CAN] Error: {e}")
        return False



class ControlBridge(threading.Thread):
    """Bridge between RuntimeMailbox and robot driver (hardware or simulation)."""

    def __init__(
        self,
        state: RuntimeMailbox,
        driver: ActuatorBus,
        axis_ids=None,
        diag_log_dir: str | None = None,
        diag_log_hz: float = 100.0,
        config: RuntimeConfig | dict | None = None,
    ):
        super().__init__(daemon=True)
        self.state = state
        self.driver = driver
        self.runtime_config = config if isinstance(config, RuntimeConfig) else parse_runtime_config(config or {})
        self.axis_ids = list(axis_ids or self.runtime_config.hardware.odrive.axis_ids)
        self._stop = threading.Event()
        self.diag_log_dir = diag_log_dir or os.path.join(os.getcwd(), "Logs")
        self.diag_log_hz = max(1.0, float(diag_log_hz))
        self._diag_file = None
        self._diag_writer = None
        self._diag_row_keys = None
        self._diag_log_path = None
        self._diag_start_perf = None
        self._diag_last_log_perf = 0.0
        self._runtime_start_perf = None
        self._last_spool_cmd_mm = [float("nan")] * 6
        self._last_spool_pose_cmd_mm = [float("nan")] * 6
        self._last_spool_null_cmd_mm = [0.0] * 6
        self._last_torque_cmd_nm = [0.0] * 6
        self._last_tension_cmd_N = [0.0] * 6
        self._last_tau_plat_des = np.full(5, np.nan, dtype=float)
        self._last_sigma_ref = float("nan")
        self._last_sigma_meas = float("nan")
        self._last_eta_null_m = 0.0
        self._sim_time_s = float("nan")
        self._sim_rt_factor = float("nan")
        self._sim_time_prev = None
        self._sim_wall_prev = None
        self._clock = WallClock()
        self._control_rate_hz = float(self.runtime_config.robot.control_rate_hz)
        self._loop_period_s = 1.0 / max(1.0, self._control_rate_hz)
        self._missed_deadline_count = 0
        self._loop_command_write_duration_s = 0.0
        self._last_loop_start_perf = None
        self._last_control_step_timings = {
            "trajectory_duration_s": None,
            "kinematics_duration_s": None,
            "tension_solver_duration_s": None,
        }
        spool_cfg = self.runtime_config.controller.spool_space
        tension_cfg = self.runtime_config.tension
        allocation_cfg = self.runtime_config.allocation
        fallback_cfg = spool_cfg.fallback_tension
        mm_per_turn_cfg = self.runtime_config.hardware.odrive.mm_per_turn
        self._mm_per_turn = [float(v) for v in mm_per_turn_cfg]
        capstan_radius_m = float(self.runtime_config.geometry.capstan_radius_m)
        torque_direction = float(self.runtime_config.hardware.odrive.torque_direction)
        self._torque_per_tension = float(torque_direction) * float(capstan_radius_m)
        self._spool_kp_base = _expand_axis_values(
            spool_cfg.kp,
            spool_cfg.kp[0],
            axis_count=len(self.axis_ids),
        )
        self._spool_kd_base = _expand_axis_values(
            spool_cfg.kd,
            spool_cfg.kd[0],
            axis_count=len(self.axis_ids),
        )
        self._spool_torque_limit_nm = _expand_axis_values(
            spool_cfg.torque_limit_nm,
            spool_cfg.torque_limit_nm[0],
            axis_count=len(self.axis_ids),
        )
        self._spool_bias_tension_N = _expand_axis_values(
            spool_cfg.bias_tension_n,
            spool_cfg.bias_tension_n[0],
            axis_count=len(self.axis_ids),
        )
        outer_cfg = spool_cfg.outer_taskspace_correction
        self._outer_corr_kp = np.diag(
            _expand_axis_values(outer_cfg.kp, 1.0, axis_count=5)
        )
        self._outer_corr_kd = np.diag(
            _expand_axis_values(outer_cfg.kd, 0.15, axis_count=5)
        )
        self._outer_corr_cable_clip_m = np.asarray(
            _expand_axis_values(outer_cfg.cable_clip_m, OUTER_CORR_CABLE_CLIP_M, axis_count=len(self.axis_ids)),
            dtype=float,
        )
        null_cfg = spool_cfg.nullspace_tension
        self._null_tension_kp = float(null_cfg.kp)
        self._null_tension_ki = float(null_cfg.ki)
        self._null_eta_limit_m = abs(float(null_cfg.eta_limit_m))
        self._null_sigma_ref_base = float(null_cfg.sigma_ref_n)
        self._null_sigma_rate_limit_Nps = abs(float(null_cfg.sigma_rate_limit_nps))
        self._null_tension_floor_N = float(null_cfg.tension_floor_n)
        self._enable_position_torque_ff = bool(spool_cfg.enable_torque_feedforward)
        self._fallback_torque_ctrl_kp_n_per_mm = float(fallback_cfg.kp_n_per_mm)
        self._fallback_torque_ctrl_kd_n_per_mmps = float(fallback_cfg.kd_n_per_mmps)
        self._fallback_torque_ctrl_bias_n = float(fallback_cfg.bias_n)
        self._fallback_torque_ctrl_min_n = float(fallback_cfg.min_n)
        self._fallback_torque_ctrl_max_n = float(fallback_cfg.max_n)
        self._spool_kp_runtime = list(self._spool_kp_base)
        self._spool_kd_runtime = list(self._spool_kd_base)
        self._tension_allocator_cfg = TensionAllocatorConfig(
            tension_min_n=float(tension_cfg.tension_min_n),
            tension_max_n=float(tension_cfg.tension_max_n),
            regularization_lambda=float(tension_cfg.regularization_lambda),
            iterations=int(allocation_cfg.max_iters),
            alpha_blend=float(tension_cfg.alpha_blend),
            wrench_from_tension_sign=float(allocation_cfg.wrench_from_tension_sign),
        )
        self._nullspace_controller_cfg = NullspaceControllerConfig(
            kp=self._null_tension_kp,
            ki=self._null_tension_ki,
            eta_limit_m=self._null_eta_limit_m,
            sigma_ref_n=self._null_sigma_ref_base,
            sigma_rate_limit_nps=self._null_sigma_rate_limit_Nps,
            tension_floor_n=self._null_tension_floor_N,
        )
        self._taskspace_controller_cfg = TaskspaceControllerConfig(
            axis_ids=tuple(self.axis_ids),
            mm_per_turn=tuple(float(v) for v in self._mm_per_turn),
            home_cable_mm=tuple(float(v) for v in HOME_CABLE_MM),
            geometry=GEOM,
            outer_corr_kp=self._outer_corr_kp.copy(),
            outer_corr_kd=self._outer_corr_kd.copy(),
            outer_corr_cable_clip_m=np.asarray(self._outer_corr_cable_clip_m, dtype=float).copy(),
            spool_bias_tension_n=np.asarray(self._spool_bias_tension_N, dtype=float).copy(),
            torque_per_tension=float(self._torque_per_tension),
            gravity_ff_z_n=float(spool_cfg.gravity_ff_z_n),
            enable_position_torque_ff=self._enable_position_torque_ff,
            tension_allocator=self._tension_allocator_cfg,
            nullspace=self._nullspace_controller_cfg,
        )
        self._cablespace_fallback_cfg = CableSpaceFallbackConfig(
            axis_ids=tuple(self.axis_ids),
            mm_per_turn=tuple(float(v) for v in self._mm_per_turn),
            home_cable_mm=tuple(float(v) for v in HOME_CABLE_MM),
            geometry=GEOM,
            torque_per_tension=float(self._torque_per_tension),
            torque_ctrl_kp_n_per_mm=float(self._fallback_torque_ctrl_kp_n_per_mm),
            torque_ctrl_kd_n_per_mmps=float(self._fallback_torque_ctrl_kd_n_per_mmps),
            torque_ctrl_bias_n=float(self._fallback_torque_ctrl_bias_n),
            torque_ctrl_min_n=float(self._fallback_torque_ctrl_min_n),
            torque_ctrl_max_n=float(self._fallback_torque_ctrl_max_n),
        )
        self._taskspace_controller_state = TaskspaceControllerState(null_sigma_ref_n=float(self._null_sigma_ref_base))
        self._platform_estimator = CablePlatformEstimator(
            axis_ids=self.axis_ids,
            mm_per_turn=self._mm_per_turn,
            home_cable_mm=HOME_CABLE_MM,
            geometry=GEOM,
            update_rate_hz=float(self.runtime_config.estimator.rate_hz),
        )
        self._state_machine = RuntimeStateMachine.from_mailbox(self.state, self.axis_ids)
        self._trajectory_manager = TrajectoryManager()
        self._manual_input_controller = ManualInputController(self.runtime_config.controller.manual_input)
        self._homing_manager = HomingRoutineManager(
            self.runtime_config.homing,
            axis_ids=self.axis_ids,
            mm_per_turn=self._mm_per_turn,
            home_cable_mm=HOME_CABLE_MM,
            geometry=GEOM,
        )
        self._watchdog = RuntimeWatchdog(
            status_report_period_s=float(self.runtime_config.rt.status_report_period_s),
            deadline_warning_margin_s=1e-3 * float(self.runtime_config.rt.deadline_warning_margin_ms),
            deadline_warning_missed_per_report=int(self.runtime_config.rt.deadline_warning_missed_per_report),
            feedback_age_warning_s=1e-3 * float(self.runtime_config.rt.feedback_age_warning_ms),
            transition_grace_s=float(self.runtime_config.rt.transition_grace_s),
        )

    def _supports_position_command_with_ff(self) -> bool:
        caps = getattr(self.driver, "capabilities", None)
        if caps is not None:
            return bool(getattr(caps, "position_command_with_ff", False))
        return hasattr(self.driver, "set_axis_position_command")

    def stop(self):
        self._stop.set()
        if self.driver:
            try:
                self.driver.stop()
            except Exception:
                pass

    def _sync_mailbox_from_actuator_states(self, actuator_states: tuple[ActuatorState, ...]):
        for axis_state in actuator_states:
            self.state.set_axis_feedback(
                axis_state.axis_id,
                pos_estimate=axis_state.position_turns,
                vel_estimate=axis_state.velocity_turns_per_s,
                bus_voltage=axis_state.bus_voltage_v,
                bus_current=axis_state.bus_current_a,
                motor_current=axis_state.current_estimate_a,
                temp_fet=axis_state.temperature_fet_c,
                temp_motor=axis_state.temperature_motor_c,
                axis_error=axis_state.error_flags,
                axis_state=axis_state.axis_state,
                proc_result=axis_state.proc_result,
            )

    def _read_actuator_states(self) -> tuple[ActuatorState, ...]:
        if not hasattr(self.driver, "read_actuator_states"):
            return ()
        try:
            actuator_states = tuple(self.driver.read_actuator_states())
        except Exception as exc:
            logger.error(f"[CTRL] Failed to read actuator states: {exc}")
            return ()
        if actuator_states:
            self._sync_mailbox_from_actuator_states(actuator_states)
        return actuator_states

    def _set_runtime_feedback_telemetry(self, actuator_states: tuple[ActuatorState, ...]):
        torque_rsp = [None] * len(self.axis_ids)
        tension_rsp = [None] * len(self.axis_ids)
        for i, axis_state in enumerate(actuator_states[: len(self.axis_ids)]):
            torque_rsp[i] = axis_state.torque_estimate_nm
            tension_rsp[i] = axis_state.tension_estimate_n
        self.state.set_axis_torque_telemetry(
            torque_cmd_nm=self._last_torque_cmd_nm,
            torque_rsp_nm=torque_rsp,
        )
        self.state.set_axis_tension_telemetry(
            tension_cmd_n=self._last_tension_cmd_N,
            tension_rsp_n=tension_rsp,
        )

    def _update_comm_stats_from_bus(self):
        if not hasattr(self.driver, "get_bus_stats"):
            return
        try:
            cstats = self.driver.get_bus_stats()
        except Exception:
            cstats = None
        if cstats is None:
            return
        try:
            stats_dict = cstats.to_dict()
        except Exception:
            stats_dict = None
        if isinstance(stats_dict, dict):
            self.state.set_comm_stats(
                can_rx_hz=stats_dict.get("can_rx_hz"),
                can_tx_hz=stats_dict.get("can_tx_hz"),
                can_msg_hz=stats_dict.get("can_msg_hz"),
                can_util_est=stats_dict.get("can_util_est"),
                pos_fbk_hz=stats_dict.get("pos_fbk_hz"),
                pos_fbk_period0_min_s=stats_dict.get("pos_fbk_period0_min_s"),
                pos_fbk_period0_max_s=stats_dict.get("pos_fbk_period0_max_s"),
            )

    def _set_platform_estimate_from_feedback(self, actuator_states: tuple[ActuatorState, ...]):
        q_cur, qd_cur, qdd_cur, j_cur = self._platform_estimator.update_from_actuator_states(actuator_states)
        if q_cur is not None and qd_cur is not None:
            self._publish_platform_estimate(q_cur, qd_cur, qdd_cur)
        return q_cur, qd_cur, j_cur

    def _write_command_batch(self, commands):
        if not commands:
            return
        start_perf = self._clock.now_monotonic()
        try:
            self.driver.write_commands(commands)
        except Exception as exc:
            logger.error(f"[CTRL] Failed to write actuator command batch: {exc}")
        finally:
            self._loop_command_write_duration_s += max(0.0, self._clock.now_monotonic() - start_perf)

    def _apply_controller_debug(self, debug):
        self._last_spool_cmd_mm = [float(x) for x in debug.spool_cmd_mm]
        self._last_spool_pose_cmd_mm = [float(x) for x in debug.spool_pose_cmd_mm]
        self._last_spool_null_cmd_mm = [float(x) for x in debug.spool_null_cmd_mm]
        self._last_torque_cmd_nm = [float(x) for x in debug.torque_cmd_nm]
        self._last_tension_cmd_N = [float(x) for x in debug.tension_cmd_n]
        self._last_tau_plat_des = np.asarray(debug.tau_plat_des, dtype=float)
        self._last_sigma_ref = float(debug.sigma_ref_n)
        self._last_sigma_meas = float(debug.sigma_meas_n)
        self._last_eta_null_m = float(debug.eta_null_m)

    def _execute_state_machine_actions(self, result):
        if result.transition is not None:
            self._apply_state_transition(result.transition)
        if result.apply_home:
            self._apply_home()
        if result.apply_pretension_mode:
            self._apply_pretension_mode()
        if result.apply_spool_gain_update:
            self._apply_spool_gain_multipliers()

    def run(self):
        logger.info("[CTRL] Starting control bridge...")
        try:
            # Start the driver
            self.driver.start()
            self._open_diag_log()
            self._runtime_start_perf = self._clock.now_monotonic()

            # main loop at configured control rate
            while not self._stop.is_set():
                loop_start_perf = self._clock.now_monotonic()
                if self._last_loop_start_perf is None:
                    loop_period_s = None
                else:
                    loop_period_s = max(0.0, loop_start_perf - self._last_loop_start_perf)
                self._last_loop_start_perf = loop_start_perf
                loop_deadline_perf = loop_start_perf + self._loop_period_s
                self._loop_command_write_duration_s = 0.0
                self._last_control_step_timings = {
                    "trajectory_duration_s": None,
                    "kinematics_duration_s": None,
                    "tension_solver_duration_s": None,
                }

                read_start_perf = self._clock.now_monotonic()
                actuator_states = self._read_actuator_states()
                read_duration_s = max(0.0, self._clock.now_monotonic() - read_start_perf)
                sm_result = self._state_machine.step(self.state)
                st = sm_result.mode.value
                self._execute_state_machine_actions(sm_result)
                runtime_time_s = self._compute_runtime_time(loop_start_perf)
                sim_time_s, sim_rt_factor = self._compute_sim_timing(loop_start_perf)
                control_time_s = sim_time_s if sim_time_s is not None else runtime_time_s
                self.state.set_timing_state(
                    control_time_s=control_time_s,
                    runtime_time_s=runtime_time_s,
                    sim_time_s=sim_time_s,
                    sim_rt_factor=sim_rt_factor,
                )
                control_now_s = self._sim_time_s if np.isfinite(self._sim_time_s) else loop_start_perf
                self._trajectory_manager.consume_mailbox_updates(self.state, float(control_now_s))
                trajectory_sample, trajectory_status = self._trajectory_manager.sample(float(control_now_s))
                trajectory_sample, _manual_input_status = self._manual_input_controller.step(
                    self.state,
                    now_control_s=float(control_now_s),
                    now_perf_s=loop_start_perf,
                    base_sample=trajectory_sample,
                    allow_streaming=bool(sm_result.allow_taskspace_streaming),
                )
                self.state.set_commanded_motion_sample(
                    trajectory_sample.pose_t_mm,
                    trajectory_sample.pose_q,
                    v_mps=trajectory_sample.linear_velocity_mps,
                    a_mps2=trajectory_sample.linear_acceleration_mps2,
                    w_rps=trajectory_sample.angular_velocity_rps,
                    alpha_rps2=trajectory_sample.angular_acceleration_rps2,
                )
                self.state.set_profile_active(trajectory_status.profile_active)

                observer_start_perf = self._clock.now_monotonic()
                q_cur, qd_cur, j_cur = self._set_platform_estimate_from_feedback(actuator_states)
                observer_duration_s = max(0.0, self._clock.now_monotonic() - observer_start_perf)
                supports_taskspace_controller = self._supports_position_command_with_ff()

                # Stream setpoints if enabled
                if sm_result.allow_taskspace_streaming:
                    try:
                        if supports_taskspace_controller:
                            commands = self._run_taskspace_spool_control(
                                trajectory_sample=trajectory_sample,
                                q_cur=q_cur,
                                qd_cur=qd_cur,
                                j_outer=j_cur,
                                actuator_states=actuator_states,
                            )
                        else:
                            commands = self._run_cablespace_fallback_control(actuator_states=actuator_states)
                        self._write_command_batch(commands)

                    except Exception as e:
                        # IMPORTANT: don't kill the bridge if IK/units blow up
                        logger.error(f"[CTRL] ENABLE streaming error: {e}")

                elif sm_result.allow_pretension_streaming:
                    self._last_tau_plat_des[:] = np.nan
                    upper_N, lower_N = self.state.get_pretension()

                    # Map upper/lower tension to per-axis torque commands
                    torque_cmd = [0.0] * len(self.axis_ids)
                    tension_cmd = [0.0] * len(self.axis_ids)
                    for i in (0, 2, 4):
                        if i < len(self.axis_ids):
                            tension_cmd[i] = upper_N
                            torque_cmd[i] = upper_N * self._torque_per_tension
                    for i in (1, 3, 5):
                        if i < len(self.axis_ids):
                            tension_cmd[i] = lower_N
                            torque_cmd[i] = lower_N * self._torque_per_tension

                    commands = [
                        ActuatorCommand(
                            axis_id=aid,
                            control_mode=ActuatorControlMode.TORQUE,
                            torque_nm=float(torque_cmd[i]),
                        )
                        for i, aid in enumerate(self.axis_ids)
                    ]
                    self._write_command_batch(commands)
                    self._last_tension_cmd_N = [float(x) for x in tension_cmd]
                    self._last_torque_cmd_nm = [float(x) for x in torque_cmd]

                # light heartbeat log
                now = self._clock.now_monotonic()
                self._set_runtime_feedback_telemetry(actuator_states)
                self._update_comm_stats_from_bus()
                feedback_ages = [
                    float(axis_state.feedback_age_s)
                    for axis_state in actuator_states
                    if axis_state.feedback_age_s is not None
                ]
                feedback_age_s = max(feedback_ages) if feedback_ages else None
                bus_utilization_estimate = None
                try:
                    bus_utilization_estimate = self.state.get_comm_stats().get("can_util_est")
                except Exception:
                    bus_utilization_estimate = None
                total_loop_duration_s = max(0.0, self._clock.now_monotonic() - loop_start_perf)
                deadline_margin_s = loop_deadline_perf - self._clock.now_monotonic()
                if deadline_margin_s < 0.0:
                    self._missed_deadline_count += 1
                timing_stats = TimingStats(
                    loop_period_s=loop_period_s,
                    read_duration_s=read_duration_s,
                    observer_duration_s=observer_duration_s,
                    trajectory_duration_s=self._last_control_step_timings["trajectory_duration_s"],
                    kinematics_duration_s=self._last_control_step_timings["kinematics_duration_s"],
                    tension_solver_duration_s=self._last_control_step_timings["tension_solver_duration_s"],
                    command_write_duration_s=self._loop_command_write_duration_s,
                    total_loop_duration_s=total_loop_duration_s,
                    deadline_margin_s=deadline_margin_s,
                    missed_deadline_count=self._missed_deadline_count,
                    feedback_age_s=feedback_age_s,
                    bus_utilization_estimate=bus_utilization_estimate,
                )
                self.state.set_timing_stats(timing_stats)
                watchdog_eval = self._watchdog.observe(
                    WatchdogSample(
                        now_perf_s=now,
                        mode=st,
                        loop_period_s=loop_period_s,
                        total_loop_duration_s=total_loop_duration_s,
                        deadline_margin_s=deadline_margin_s,
                        missed_deadline_count=self._missed_deadline_count,
                        feedback_age_s=feedback_age_s,
                    )
                )
                self.state.set_watchdog_status(watchdog_eval.status)
                self._homing_manager.tick(
                    self.state,
                    now_perf_s=now,
                    runtime_mode=st,
                    allow_taskspace_streaming=bool(sm_result.allow_taskspace_streaming),
                    supports_taskspace_controller=bool(supports_taskspace_controller),
                    q_cur=q_cur,
                    qd_cur=qd_cur,
                    actuator_states=actuator_states,
                    tension_cmd_n=tuple(float(x) for x in self._last_tension_cmd_N),
                    tension_rsp_n=tuple(
                        float(x) if x is not None else float("nan")
                        for x in self.state.get_axis_tension_response()
                    ),
                    sigma_ref_n=float(self._last_sigma_ref),
                    sigma_meas_n=float(self._last_sigma_meas),
                    eta_null_m=float(self._last_eta_null_m),
                    spool_null_cmd_mm=tuple(float(x) for x in self._last_spool_null_cmd_mm),
                    watchdog_status=watchdog_eval.status,
                    profile_active=trajectory_status.profile_active,
                )
                read_ms = 1000.0 * read_duration_s
                observer_ms = 1000.0 * observer_duration_s
                traj_ms = (
                    float("nan")
                    if self._last_control_step_timings["trajectory_duration_s"] is None
                    else 1000.0 * float(self._last_control_step_timings["trajectory_duration_s"])
                )
                kin_ms = (
                    float("nan")
                    if self._last_control_step_timings["kinematics_duration_s"] is None
                    else 1000.0 * float(self._last_control_step_timings["kinematics_duration_s"])
                )
                solver_ms = (
                    float("nan")
                    if self._last_control_step_timings["tension_solver_duration_s"] is None
                    else 1000.0 * float(self._last_control_step_timings["tension_solver_duration_s"])
                )
                write_ms = 1000.0 * self._loop_command_write_duration_s
                loop_ms = 1000.0 * total_loop_duration_s
                deadline_margin_ms = 1000.0 * deadline_margin_s
                feedback_age_ms = (
                    float("nan") if feedback_age_s is None else 1000.0 * float(feedback_age_s)
                )
                if self._diag_writer is not None and (now - self._diag_last_log_perf) >= (1.0 / self.diag_log_hz):
                    self._write_diag_row(now)
                    self._diag_last_log_perf = now
                if watchdog_eval.report_due:
                    log_fn = logger.info
                    if watchdog_eval.log_as_warning:
                        log_fn = logger.warning
                    health_note = (
                        f", watchdog={watchdog_eval.status.message}"
                        if watchdog_eval.status.message
                        else ""
                    )
                    if np.isfinite(self._sim_time_s) and np.isfinite(self._sim_rt_factor):
                        log_fn(
                            f"[CTRL] streaming {len(self.axis_ids)} axes, state={st}, "
                            f"sim_time={self._sim_time_s:.3f}s, rt_factor={self._sim_rt_factor:.3f}x, "
                            f"deadline_margin={deadline_margin_ms:.2f} ms, "
                            f"missed={self._missed_deadline_count} (+{watchdog_eval.missed_since_last_report}), "
                            f"read={read_ms:.2f} ms observer={observer_ms:.2f} ms "
                            f"traj={traj_ms:.2f} ms kin={kin_ms:.2f} ms solver={solver_ms:.2f} ms "
                            f"write={write_ms:.2f} ms loop={loop_ms:.2f} ms fb_age={feedback_age_ms:.2f} ms"
                            f"{health_note}"
                        )
                    else:
                        log_fn(
                            f"[CTRL] streaming {len(self.axis_ids)} axes, state={st}, "
                            f"deadline_margin={deadline_margin_ms:.2f} ms, "
                            f"missed={self._missed_deadline_count} (+{watchdog_eval.missed_since_last_report}), "
                            f"read={read_ms:.2f} ms observer={observer_ms:.2f} ms "
                            f"traj={traj_ms:.2f} ms kin={kin_ms:.2f} ms solver={solver_ms:.2f} ms "
                            f"write={write_ms:.2f} ms loop={loop_ms:.2f} ms fb_age={feedback_age_ms:.2f} ms"
                            f"{health_note}"
                        )

                self._clock.sleep_until(loop_deadline_perf)

        except Exception as e:
            logger.error(f"[CTRL] Bridge error: {e}")
        finally:
            self._close_diag_log()
            if self.driver:
                try:
                    self.driver.stop()
                except Exception:
                    pass
            logger.info("[CTRL] Bridge stopped")

    def _run_taskspace_spool_control(
        self,
        trajectory_sample,
        q_cur=None,
        qd_cur=None,
        j_outer=None,
        actuator_states: tuple[ActuatorState, ...] = (),
    ):
        """Generate spool references from task-space commands and return position-mode actuator commands."""
        traj_start_perf = self._clock.now_monotonic()
        t_mm_cmd = trajectory_sample.pose_t_mm
        q_cmd = trajectory_sample.pose_q
        v_cmd_mps = trajectory_sample.linear_velocity_mps
        a_cmd_mps2 = trajectory_sample.linear_acceleration_mps2
        trajectory_duration_s = max(0.0, self._clock.now_monotonic() - traj_start_perf)
        result = compute_taskspace_spool_commands(
            pose_t_mm=t_mm_cmd,
            pose_q=q_cmd,
            linear_velocity_mps=v_cmd_mps,
            linear_acceleration_mps2=a_cmd_mps2,
            q_cur=q_cur,
            qd_cur=qd_cur,
            j_outer=j_outer,
            actuator_states=actuator_states,
            config=self._taskspace_controller_cfg,
            state=self._taskspace_controller_state,
            now_perf=time.perf_counter(),
        )
        self._taskspace_controller_state = result.state
        self._apply_controller_debug(result.debug)
        self._last_control_step_timings = {
            "trajectory_duration_s": trajectory_duration_s,
            "kinematics_duration_s": result.kinematics_duration_s,
            "tension_solver_duration_s": result.tension_solver_duration_s,
        }
        return list(result.commands)

    def _run_cablespace_fallback_control(self, actuator_states: tuple[ActuatorState, ...] = ()):
        """
        Fallback controller for drivers without platform-state feedback:
        cable-space PD + bias tension.
        """
        traj_start_perf = self._clock.now_monotonic()
        t_mm, q = self.state.get_hand_pose()
        trajectory_duration_s = max(0.0, self._clock.now_monotonic() - traj_start_perf)
        result = compute_cablespace_fallback_commands(
            pose_t_mm=t_mm,
            pose_q=q,
            actuator_states=actuator_states,
            config=self._cablespace_fallback_cfg,
        )
        self._apply_controller_debug(result.debug)
        self._taskspace_controller_state = TaskspaceControllerState(null_sigma_ref_n=float(self._null_sigma_ref_base))
        self._last_control_step_timings = {
            "trajectory_duration_s": trajectory_duration_s,
            "kinematics_duration_s": result.kinematics_duration_s,
            "tension_solver_duration_s": result.tension_solver_duration_s,
        }
        return list(result.commands)

    def _publish_platform_estimate(self, q_cur, qd_cur, qdd_cur=None):
        """
        Publish platform estimate into RuntimeMailbox for GUI telemetry.
        q_cur: [x,y,z,roll,pitch] in SI units.
        qd_cur: [xd,yd,zd,rolld,pitchd] in SI units.
        """
        try:
            q_cur = np.asarray(q_cur, dtype=float)
            qd_cur = np.asarray(qd_cur, dtype=float)
            qdd_cur = np.asarray(qdd_cur, dtype=float) if qdd_cur is not None else None
            t_mm = (1000.0 * float(q_cur[0]), 1000.0 * float(q_cur[1]), 1000.0 * float(q_cur[2]))
            q_est = quat_from_rpy_deg(math.degrees(float(q_cur[3])), math.degrees(float(q_cur[4])), 0.0)
            v_mps = (float(qd_cur[0]), float(qd_cur[1]), float(qd_cur[2]))
            w_rps = (float(qd_cur[3]), float(qd_cur[4]), 0.0)
            a_mps2 = None if qdd_cur is None or qdd_cur.shape[0] < 3 else (float(qdd_cur[0]), float(qdd_cur[1]), float(qdd_cur[2]))
            alpha_rps2 = None if qdd_cur is None or qdd_cur.shape[0] < 5 else (float(qdd_cur[3]), float(qdd_cur[4]), 0.0)
            self.state.set_hand_estimate(t_mm, q_est, v_mps=v_mps, w_rps=w_rps, a_mps2=a_mps2, alpha_rps2=alpha_rps2)
        except Exception:
            pass

    def _apply_spool_gain_multipliers(self):
        kp_mult, kd_mult = self.state.get_spool_gain_multipliers()
        self._spool_kp_runtime = [float(v) * float(kp_mult) for v in self._spool_kp_base]
        self._spool_kd_runtime = [float(v) * float(kd_mult) for v in self._spool_kd_base]
        applied = False
        if hasattr(self.driver, "configure_spool_controller"):
            try:
                applied = bool(
                    self.driver.configure_spool_controller(
                        kp=self._spool_kp_runtime,
                        kd=self._spool_kd_runtime,
                        torque_limit=self._spool_torque_limit_nm,
                    )
                )
            except Exception as e:
                logger.error(f"[CTRL] Failed to configure spool controller: {e}")
        logger.info(
            "[CTRL] Spool gain multipliers applied: "
            f"kp={kp_mult:.3f}, kd={kd_mult:.3f}, runtime_applied={applied}"
        )

    def _reset_runtime_state(self):
        self._last_torque_cmd_nm = [0.0] * 6
        self._last_tension_cmd_N = [0.0] * 6
        self._last_tau_plat_des[:] = np.nan
        self._taskspace_controller_state = TaskspaceControllerState(
            null_sigma_ref_n=float(self._null_sigma_ref_base)
        )
        self._last_sigma_ref = float("nan")
        self._last_sigma_meas = float("nan")
        self._last_eta_null_m = 0.0
        self._last_spool_null_cmd_mm = [0.0] * 6

    def _apply_state_transition(self, transition: RuntimeTransitionAction):
        """Apply a planned runtime-mode transition."""
        try:
            self._write_command_batch(transition.commands)
            for aid in self.axis_ids:
                if transition.mode is RuntimeMode.ENABLE:
                    logger.info(f"[CTRL] axis {aid}: POSITION + CLOSED_LOOP_CONTROL")
                elif transition.mode is RuntimeMode.PRETENSION:
                    logger.info(f"[CTRL] axis {aid}: TORQUE + CLOSED_LOOP_CONTROL")
                elif transition.mode in (RuntimeMode.DISABLE, RuntimeMode.ESTOP):
                    logger.info(f"[CTRL] axis {aid}: IDLE")
            if transition.reset_runtime_state:
                self._reset_runtime_state()

        except Exception as e:
            logger.error(f"[CTRL] _apply_state_transition error: {e}")

    def _apply_home(self):
        """
        HOME intent: do NOT move motors.
        Reset the reported absolute spool position to the GUI-provided home
        values without commanding a motion.
        """
        home_pos_mm = self.state.get_home_pos()  # mm

        # Shift each axis encoder frame to the requested home position.
        home_pos_turns = mm_to_turns(home_pos_mm)
        for i, aid in enumerate(self.axis_ids):
            try:
                self.driver.set_absolute_position(aid, home_pos_turns[i])
                self.driver.set_axis_position(aid, home_pos_turns[i])
                logger.info(f"[HOME] axis {aid}: abs_pos <- {home_pos_mm[i]:.3f} mm ({home_pos_turns[i]:.4f} turns)")
            except Exception as e:
                logger.warning(f"[HOME] axis {aid} set_absolute_position failed: {e}")
        self._last_spool_cmd_mm = [float(x) for x in home_pos_mm]

    def _apply_pretension_mode(self):
        """Put all axes into torque control + closed loop."""
        try:
            commands = self._state_machine.build_mode_commands(RuntimeMode.PRETENSION)
            self._write_command_batch(commands)
            logger.info("[PRET] applied torque control mode to all axes")
        except Exception as e:
            logger.error(f"[PRET] _apply_pretension_mode error: {e}")

    def _open_diag_log(self):
        os.makedirs(self.diag_log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._diag_log_path = os.path.join(self.diag_log_dir, f"control_diag_{ts}.csv")
        self._diag_file = open(self._diag_log_path, "w", newline="", encoding="utf-8")
        self._diag_writer = csv.writer(self._diag_file)
        self._diag_row_keys = None
        self._diag_start_perf = time.perf_counter()
        self._diag_last_log_perf = self._diag_start_perf
        logger.info(f"[CTRL] Diagnostic CSV logging enabled: {self._diag_log_path}")

    def _close_diag_log(self):
        if self._diag_file is not None:
            try:
                self._diag_file.flush()
                self._diag_file.close()
            except Exception:
                pass
            self._diag_file = None
            self._diag_writer = None
            self._diag_row_keys = None

    def _compute_runtime_time(self, now_perf):
        if self._runtime_start_perf is None:
            self._runtime_start_perf = float(now_perf)
        runtime_time_s = max(0.0, float(now_perf) - float(self._runtime_start_perf))
        return runtime_time_s

    def _compute_sim_timing(self, now_perf):
        self._sim_time_s = float("nan")
        self._sim_rt_factor = float("nan")
        if not hasattr(self.driver, "get_sim_time"):
            self._sim_time_prev = None
            self._sim_wall_prev = None
            return None, None
        try:
            sim_time = self.driver.get_sim_time()
        except Exception:
            sim_time = None
        if sim_time is None:
            self._sim_time_prev = None
            self._sim_wall_prev = None
            return None, None

        sim_time = float(sim_time)
        self._sim_time_s = sim_time
        if self._sim_time_prev is not None and self._sim_wall_prev is not None:
            ds = sim_time - self._sim_time_prev
            dw = float(now_perf - self._sim_wall_prev)
            if dw > 1e-6:
                self._sim_rt_factor = ds / dw
        self._sim_time_prev = sim_time
        self._sim_wall_prev = float(now_perf)
        sim_rt_factor = None if not np.isfinite(self._sim_rt_factor) else self._sim_rt_factor
        return sim_time, sim_rt_factor

    def _write_diag_row(self, now_perf):
        if self._diag_writer is None:
            return

        from jugglebot.core.snapshots import build_robot_state_snapshot, flatten_robot_state

        tau_rsp = np.full(5, np.nan, dtype=float)
        tension_rsp = np.asarray(self.state.get_axis_tension_response(), dtype=float)
        _, _, _, J_len_plat = self._platform_estimator.get_latest()
        if J_len_plat is not None:
            try:
                J_len_plat = np.asarray(J_len_plat, dtype=float)
                if J_len_plat.shape == (6, 5) and tension_rsp.shape == (6,) and np.all(np.isfinite(tension_rsp)):
                    tau_rsp = platform_wrench_from_tensions(
                        J_len_plat,
                        tension_rsp,
                        wrench_from_tension_sign=self._tension_allocator_cfg.wrench_from_tension_sign,
                    )
            except Exception:
                pass

        snapshot = build_robot_state_snapshot(
            self.state,
            timestamp_s=time.time(),
            debug={
                "diag_t_rel_s": float(now_perf - self._diag_start_perf),
                "sim_time_s": float(self._sim_time_s),
                "sim_rt_factor": float(self._sim_rt_factor),
                "wrench_cmd": [float(x) for x in np.asarray(self._last_tau_plat_des, dtype=float)],
                "wrench_rsp": [float(x) for x in np.asarray(tau_rsp, dtype=float)],
                "null_sigma_ref_n": float(self._last_sigma_ref),
                "null_sigma_meas_n": float(self._last_sigma_meas),
                "null_eta_m": float(self._last_eta_null_m),
                "spool_pose_cmd_mm": [float(x) for x in self._last_spool_pose_cmd_mm],
                "spool_null_cmd_mm": [float(x) for x in self._last_spool_null_cmd_mm],
                "spool_cmd_mm": [float(x) for x in self._last_spool_cmd_mm],
            },
        )
        flat = flatten_robot_state(snapshot)
        if self._diag_row_keys is None:
            self._diag_row_keys = list(flat.keys())
            self._diag_writer.writerow(self._diag_row_keys)
        self._diag_writer.writerow([flat.get(key, "") for key in self._diag_row_keys])
        if self._diag_file is not None:
            self._diag_file.flush()
