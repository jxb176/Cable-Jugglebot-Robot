# robot_server.py
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
from jugglebot.io.actuator_bus import ActuatorBus
from jugglebot.core.kinematics import cable_lengths_jacobian_pose5_fd, cable_lengths_m_from_pose5
from jugglebot.core.platform_estimator import CablePlatformEstimator
from jugglebot.core.pose_utils import quat_from_rpy_deg, quat_to_rpy_rad
from jugglebot.core.state import RuntimeMailbox
from jugglebot.core.types import ActuatorCommand, ActuatorControlMode, ActuatorState
from jugglebot.core.units import MM_PER_TURN, mm_to_turns

# -------- ODrive CAN configuration --------
ODRIVE_INTERFACE = "can0"
ODRIVE_BITRATE = 1_000_000  # 1 Mbps
AXIS_NODE_IDS = [0, 1, 2, 3, 4, 5]
ODRIVE_COMMAND_RATE_HZ = 500.0
ODRIVE_LOG_RATE_HZ = 2.0

# -------- Capstan / units configuration --------
# Pretension mapping: tension [N] -> capstan torque [Nm]
CAPSTAN_RADIUS_M = 0.010  # 10 mm
MOTOR_TORQUE_DIRECTION = 1  # Positive motor torque should reel in cable and increase tension on hardware.
TORQUE_PER_TENSION = MOTOR_TORQUE_DIRECTION * CAPSTAN_RADIUS_M  # Nm per N  (T = F*r)
TORQUE_CTRL_KP_N_PER_MM = 0.6
TORQUE_CTRL_KD_N_PER_MMPS = 0.02
TORQUE_CTRL_BIAS_N = 12.0
TORQUE_CTRL_MIN_N = 0.0
TORQUE_CTRL_MAX_N = 180.0
DEFAULT_SPOOL_KP_TORQUE_PER_TURN = abs(TORQUE_PER_TENSION) * TORQUE_CTRL_KP_N_PER_MM * abs(MM_PER_TURN[0])
DEFAULT_SPOOL_KD_TORQUE_PER_TURNPS = abs(TORQUE_PER_TENSION) * TORQUE_CTRL_KD_N_PER_MMPS * abs(MM_PER_TURN[0])
#TASK_KP = np.diag([1200.0, 1200.0, 1800.0, 120000.0, 120000.0])
TASK_KP = np.diag([250.0, 250.0, 400.0, 2.5, 2.5])
#TASK_KD = np.diag([80.0, 80.0, 120.0, 0.0, 0.0])
TASK_KD = np.diag([7.5, 7.5, 12.0, 0.05, 0.05])
TASK_KI = np.diag([0.0, 0.0, 0.0, 0.0, 0.0])
TASK_INT_CLIP = np.array([0.0, 0.0, 0.0, 0.35, 0.35], dtype=float)
TASK_TMIN_N = 5.0
TASK_TMAX_N = 180.0
TASK_ALLOC_LAMBDA = 1e-2
TASK_ALLOC_ITERS = 80
TASK_ALLOC_ALPHA = 0.7
TASK_GRAVITY_FF_Z_N = 1.2
OUTER_CORR_KP = np.diag([1.0, 1.0, 1.0, 0.35, 0.35])
OUTER_CORR_KD = np.diag([0.15, 0.15, 0.15, 0.05, 0.05])
OUTER_CORR_CABLE_CLIP_M = 0.10
# Wrench mapping sign convention for tension allocation.
# +1.0 means tau = (+J^T)T, -1.0 means tau = (-J^T)T.
# Keep +1.0 for current sim setup (stable empirically with existing signs/axes).
TASK_WRENCH_FROM_TENSION_SIGN = -1.0

#CLEANUP into ODRIVE library
# ODrive controller modes (CANSimple Set_Controller_Mode)
CONTROL_MODE_TORQUE = 1      # aka "CurrentControl" in some docs
CONTROL_MODE_POSITION = 3
INPUT_MODE_PASSTHROUGH = 1

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


def solve_tensions_least_squares(J_len_plat, tau_plat_des, T_prev):
    """
    Solve:
      min_T || (s*J^T)T - tau ||^2 + lambda ||T - Tref||^2
      s.t. Tmin <= T <= Tmax
    where s = TASK_WRENCH_FROM_TENSION_SIGN.
    """
    J = np.asarray(J_len_plat, dtype=float)
    A = float(TASK_WRENCH_FROM_TENSION_SIGN) * J.T
    tau = np.asarray(tau_plat_des, dtype=float)
    nt = A.shape[1]

    lb = np.full(nt, TASK_TMIN_N, dtype=float)
    ub = np.full(nt, TASK_TMAX_N, dtype=float)
    if T_prev is None:
        Tref = lb.copy()
    else:
        Tref = TASK_ALLOC_ALPHA * np.asarray(T_prev, dtype=float) + (1.0 - TASK_ALLOC_ALPHA) * lb

    T = Tref.copy()
    ATA = A.T @ A
    L = float(np.linalg.norm(ATA, 2) + TASK_ALLOC_LAMBDA)
    step = 1.0 / max(L, 1e-9)

    for _ in range(TASK_ALLOC_ITERS):
        grad = 2.0 * (A.T @ (A @ T - tau)) + 2.0 * TASK_ALLOC_LAMBDA * (T - Tref)
        T = np.clip(T - step * grad, lb, ub)
    return T


def cable_tension_nullspace_basis(J_len_plat, n_prev=None):
    """
    Return a unit basis vector n in Null(J^T) for J shape (6,5).

    The sign is chosen for continuity against n_prev when available.
    """
    J = np.asarray(J_len_plat, dtype=float)
    if J.shape != (6, 5):
        raise ValueError(f"Expected J shape (6,5), got {J.shape}")
    _, s, vh = np.linalg.svd(J.T, full_matrices=True)
    n = np.asarray(vh[-1, :], dtype=float)
    n_norm = float(np.linalg.norm(n))
    if n_norm < 1e-9:
        raise ValueError("Degenerate cable nullspace basis")
    n = n / n_norm
    if n_prev is not None:
        n_prev = np.asarray(n_prev, dtype=float)
        if n_prev.shape == n.shape and np.all(np.isfinite(n_prev)) and float(n_prev @ n) < 0.0:
            n = -n
    return n


def nullspace_sigma_interval(T_particular, n, tension_floor_N):
    """
    Feasible interval for sigma in T = T_particular + n*sigma subject to T >= tension_floor_N.
    Returns (lower, upper, feasible).
    """
    T_particular = np.asarray(T_particular, dtype=float)
    n = np.asarray(n, dtype=float)
    lower = -float("inf")
    upper = float("inf")
    eps = 1e-9
    for Ti, ni in zip(T_particular, n):
        if ni > eps:
            lower = max(lower, (float(tension_floor_N) - float(Ti)) / float(ni))
        elif ni < -eps:
            upper = min(upper, (float(tension_floor_N) - float(Ti)) / float(ni))
        elif float(Ti) < float(tension_floor_N):
            return lower, upper, False
    return lower, upper, lower <= upper


def clamp_sigma_to_interval(sigma_ref, lower, upper):
    sigma_ref = float(sigma_ref)
    if math.isfinite(lower):
        sigma_ref = max(sigma_ref, float(lower))
    if math.isfinite(upper):
        sigma_ref = min(sigma_ref, float(upper))
    return sigma_ref


def rate_limit_scalar(target, current, rate_limit_per_s, dt):
    target = float(target)
    current = float(current)
    rate_limit_per_s = abs(float(rate_limit_per_s))
    dt = max(0.0, float(dt))
    if not math.isfinite(target):
        return current
    if not math.isfinite(current) or dt <= 0.0 or rate_limit_per_s <= 0.0:
        return target
    max_step = rate_limit_per_s * dt
    delta = float(np.clip(target - current, -max_step, max_step))
    return current + delta

def _expand_axis_values(value, default, axis_count: int = 6):
    if value is None:
        return [float(default)] * axis_count
    if isinstance(value, (list, tuple)):
        if len(value) != axis_count:
            raise ValueError(f"Expected length-{axis_count} list/tuple, got {len(value)}")
        return [float(v) for v in value]
    return [float(value)] * axis_count

os.environ.setdefault("CAN_CHANNEL", ODRIVE_INTERFACE)
os.environ.setdefault("CAN_BITRATE", str(ODRIVE_BITRATE))

try:
    import odrive_can as odc
except Exception:
    odc = None

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
        config: dict | None = None,
    ):
        super().__init__(daemon=True)
        self.state = state
        self.driver = driver
        self.axis_ids = axis_ids or [0, 1, 2, 3, 4, 5]
        self.config = config or {}
        self._stop = threading.Event()
        self._T_prev = None
        self._task_err_int = np.zeros(5, dtype=float)
        self._task_last_t = None
        self.diag_log_dir = diag_log_dir or os.path.join(os.getcwd(), "Logs")
        self.diag_log_hz = max(1.0, float(diag_log_hz))
        self._diag_file = None
        self._diag_writer = None
        self._diag_row_keys = None
        self._diag_log_path = None
        self._diag_start_perf = None
        self._diag_last_log_perf = 0.0
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
        controller_cfg = (self.config.get("controller") or {}).get("spool_space") or {}
        mm_per_turn_default = getattr(self.driver, "mm_per_turn", MM_PER_TURN)
        self._mm_per_turn = _expand_axis_values(mm_per_turn_default, MM_PER_TURN[0], axis_count=len(self.axis_ids))
        self._spool_kp_base = _expand_axis_values(
            controller_cfg.get("kp"),
            DEFAULT_SPOOL_KP_TORQUE_PER_TURN,
            axis_count=len(self.axis_ids),
        )
        self._spool_kd_base = _expand_axis_values(
            controller_cfg.get("kd"),
            DEFAULT_SPOOL_KD_TORQUE_PER_TURNPS,
            axis_count=len(self.axis_ids),
        )
        winch_cfg = self.config.get("winches") or {}
        self._spool_torque_limit_nm = _expand_axis_values(
            controller_cfg.get("torque_limit_nm"),
            winch_cfg.get("torque_limit_nm", 1.0),
            axis_count=len(self.axis_ids),
        )
        self._spool_bias_tension_N = _expand_axis_values(
            controller_cfg.get("bias_tension_N"),
            TORQUE_CTRL_BIAS_N,
            axis_count=len(self.axis_ids),
        )
        outer_cfg = controller_cfg.get("outer_taskspace_correction") or {}
        self._outer_corr_kp = np.diag(
            _expand_axis_values(outer_cfg.get("kp"), 1.0, axis_count=5)
        )
        self._outer_corr_kd = np.diag(
            _expand_axis_values(outer_cfg.get("kd"), 0.15, axis_count=5)
        )
        self._outer_corr_cable_clip_m = np.asarray(
            _expand_axis_values(outer_cfg.get("cable_clip_m"), OUTER_CORR_CABLE_CLIP_M, axis_count=len(self.axis_ids)),
            dtype=float,
        )
        null_cfg = controller_cfg.get("nullspace_tension") or {}
        self._null_tension_kp = float(null_cfg.get("kp", 0.001))
        self._null_tension_ki = float(null_cfg.get("ki", 0.0))
        self._null_eta_limit_m = abs(float(null_cfg.get("eta_limit_m", 0.02)))
        self._null_sigma_ref_base = float(null_cfg.get("sigma_ref_N", 0.0))
        self._null_sigma_rate_limit_Nps = abs(float(null_cfg.get("sigma_rate_limit_Nps", 10.0)))
        self._null_tension_floor_N = float(null_cfg.get("tmin_N", TASK_TMIN_N))
        self._eta_null_m = 0.0
        self._eta_null_int = 0.0
        self._sigma_ref_state = float(self._null_sigma_ref_base)
        self._null_basis_prev = None
        self._enable_position_torque_ff = bool(controller_cfg.get("enable_torque_feedforward", True))
        self._spool_kp_runtime = list(self._spool_kp_base)
        self._spool_kd_runtime = list(self._spool_kd_base)
        estimator_cfg = self.config.get("estimator") or {}
        odrive_cfg = (self.config.get("hardware") or {}).get("odrive") or {}
        pose_est_rate_hz = float(estimator_cfg.get("rate_hz", odrive_cfg.get("pose_est_rate_hz", 100.0)))
        self._platform_estimator = CablePlatformEstimator(
            axis_ids=self.axis_ids,
            mm_per_turn=self._mm_per_turn,
            home_cable_mm=HOME_CABLE_MM,
            geometry=GEOM,
            update_rate_hz=pose_est_rate_hz,
        )

        #Apply the current state version to avoid auto applying the default by setting these to -1.  Perhaps reconsider this for desired auto init behavior later on
        self._applied_state_version = state.get_state_version()
        self._applied_home_version = state.get_home_version()
        self._applied_pret_version = state.get_pretension_version()
        self._applied_spool_gain_version = -1

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
        q_cur, qd_cur, j_cur = self._platform_estimator.update_from_actuator_states(actuator_states)
        if q_cur is not None and qd_cur is not None:
            self._publish_platform_estimate(q_cur, qd_cur)
        return q_cur, qd_cur, j_cur

    def _write_command_batch(self, commands):
        if not commands:
            return
        try:
            self.driver.write_commands(commands)
        except Exception as exc:
            logger.error(f"[CTRL] Failed to write actuator command batch: {exc}")

    def _build_state_commands(self, st: str):
        commands = []
        if st == "enable":
            for aid in self.axis_ids:
                commands.append(
                    ActuatorCommand(
                        axis_id=aid,
                        control_mode=ActuatorControlMode.POSITION,
                        apply_control_mode=True,
                        enable=True,
                    )
                )
        elif st == "pretension":
            for aid in self.axis_ids:
                commands.append(
                    ActuatorCommand(
                        axis_id=aid,
                        control_mode=ActuatorControlMode.TORQUE,
                        apply_control_mode=True,
                        enable=True,
                    )
                )
        elif st in ("disable", "estop"):
            for aid in self.axis_ids:
                commands.append(
                    ActuatorCommand(
                        axis_id=aid,
                        control_mode=ActuatorControlMode.DISABLED,
                        enable=False,
                    )
                )
        return commands

    def run(self):
        logger.info("[CTRL] Starting control bridge...")
        try:
            # Start the driver
            self.driver.start()
            self._open_diag_log()

            # main loop (~500 Hz)
            last_log = time.perf_counter()
            while not self._stop.is_set():
                actuator_states = self._read_actuator_states()
                st = self.state.get_state()
                sv = self.state.get_state_version()

                # Apply state transitions when version changes
                if sv != self._applied_state_version:
                    self._apply_state(st)
                    self._applied_state_version = sv

                # Apply HOME request (one-shot) when version changes
                hv = self.state.get_home_version()
                if hv != self._applied_home_version:
                    self._apply_home()
                    self._applied_home_version = hv

                # Apply PRETENSION request when version changes
                pv = self.state.get_pretension_version()
                if pv != self._applied_pret_version:
                    self._apply_pretension_mode()
                    self._applied_pret_version = pv

                gv = self.state.get_spool_gain_version()
                if gv != self._applied_spool_gain_version:
                    self._apply_spool_gain_multipliers()
                    self._applied_spool_gain_version = gv

                q_cur, qd_cur, j_cur = self._set_platform_estimate_from_feedback(actuator_states)

                # Stream setpoints if enabled
                if st == "enable":
                    try:
                        if self._supports_position_command_with_ff():
                            commands = self._run_taskspace_spool_control(
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

                elif st == "pretension":
                    self._last_tau_plat_des[:] = np.nan
                    upper_N, lower_N = self.state.get_pretension()

                    # Map upper/lower tension to per-axis torque commands
                    torque_cmd = [0.0] * len(self.axis_ids)
                    tension_cmd = [0.0] * len(self.axis_ids)
                    for i in (0, 2, 4):
                        if i < len(self.axis_ids):
                            tension_cmd[i] = upper_N
                            torque_cmd[i] = upper_N * TORQUE_PER_TENSION
                    for i in (1, 3, 5):
                        if i < len(self.axis_ids):
                            tension_cmd[i] = lower_N
                            torque_cmd[i] = lower_N * TORQUE_PER_TENSION

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
                now = time.perf_counter()
                self._set_runtime_feedback_telemetry(actuator_states)
                self._update_comm_stats_from_bus()
                self._update_sim_timing(now)
                if self._diag_writer is not None and (now - self._diag_last_log_perf) >= (1.0 / self.diag_log_hz):
                    self._write_diag_row(now)
                    self._diag_last_log_perf = now
                if now - last_log >= 1.0:
                    if np.isfinite(self._sim_time_s) and np.isfinite(self._sim_rt_factor):
                        logger.info(
                            f"[CTRL] streaming {len(self.axis_ids)} axes, state={st}, "
                            f"sim_time={self._sim_time_s:.3f}s, rt_factor={self._sim_rt_factor:.3f}x"
                        )
                    else:
                        logger.info(f"[CTRL] streaming {len(self.axis_ids)} axes, state={st}")
                    last_log = now

                time.sleep(0.002)  # ~500 Hz

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
        q_cur=None,
        qd_cur=None,
        j_outer=None,
        actuator_states: tuple[ActuatorState, ...] = (),
    ):
        """Generate spool references from task-space commands and return position-mode actuator commands."""
        now = time.perf_counter()
        t_mm_cmd, q_cmd, v_cmd_mps, a_cmd_mps2 = self.state.get_hand_motion()
        roll_cmd, pitch_cmd, _ = quat_to_rpy_rad(q_cmd)
        q_ref = np.array(
            [t_mm_cmd[0] / 1000.0, t_mm_cmd[1] / 1000.0, t_mm_cmd[2] / 1000.0, roll_cmd, pitch_cmd],
            dtype=float,
        )
        qd_ref = np.array([float(v_cmd_mps[0]), float(v_cmd_mps[1]), float(v_cmd_mps[2]), 0.0, 0.0], dtype=float)
        qdd_ff = np.array([float(a_cmd_mps2[0]), float(a_cmd_mps2[1]), float(a_cmd_mps2[2]), 0.0, 0.0], dtype=float)
        cable_mm = pose_to_cable_lengths_mm(GEOM, t_mm_cmd, q_cmd)
        pose_ref_m = np.asarray([(cable_mm[i] - HOME_CABLE_MM[i]) / 1000.0 for i in range(6)], dtype=float)
        pose_corr_m = np.zeros(6, dtype=float)
        null_cmd_m = np.zeros(6, dtype=float)
        J_outer = None if j_outer is None else np.asarray(j_outer, dtype=float)

        if q_cur is not None and qd_cur is not None:
            q_cur = np.asarray(q_cur, dtype=float)
            qd_cur = np.asarray(qd_cur, dtype=float)
            e = q_ref - q_cur
            ed = qd_ref - qd_cur
            if J_outer is None or J_outer.shape != (6, 5):
                J_outer = cable_lengths_jacobian_pose5_fd(q_cur)
            cur_lengths_m = cable_lengths_m_from_pose5(q_cur)
            cur_cmd_m = np.asarray(cur_lengths_m, dtype=float) - (np.asarray(HOME_CABLE_MM, dtype=float) / 1000.0)
            cable_corr_m = J_outer @ ((self._outer_corr_kp @ e) + (self._outer_corr_kd @ ed))
            cable_corr_m = np.clip(cable_corr_m, -self._outer_corr_cable_clip_m, self._outer_corr_cable_clip_m)
            pose_corr_m = (cur_cmd_m - pose_ref_m) + cable_corr_m

        J_cmd = cable_lengths_jacobian_pose5_fd(q_ref)
        J_null = J_outer if J_outer is not None and np.shape(J_outer) == (6, 5) else J_cmd

        tau_plat_ff = np.asarray(qdd_ff, dtype=float)
        tau_plat_ff[2] += TASK_GRAVITY_FF_Z_N
        T_particular = solve_tensions_least_squares(J_cmd, tau_plat_ff, self._T_prev)
        self._T_prev = T_particular.copy()
        T_cmd = np.maximum(np.asarray(self._spool_bias_tension_N, dtype=float), T_particular)
        self._last_tau_plat_des = np.asarray(tau_plat_ff, dtype=float)
        self._last_tension_cmd_N = [float(x) for x in T_cmd]
        if self._task_last_t is None:
            dt = 0.0
        else:
            dt = max(0.0, min(0.1, float(now - self._task_last_t)))

        sigma_meas = float("nan")
        sigma_ref = float("nan")
        if np.shape(J_null) == (6, 5):
            try:
                n = cable_tension_nullspace_basis(J_null, self._null_basis_prev)
                self._null_basis_prev = n.copy()
                lower, upper, feasible = nullspace_sigma_interval(
                    T_cmd,
                    n,
                    self._null_tension_floor_N,
                )
                sigma_target = float(self._null_sigma_ref_base)
                if feasible:
                    if math.isfinite(lower) and math.isfinite(upper):
                        sigma_target = 0.5 * (float(lower) + float(upper))
                    else:
                        sigma_target = clamp_sigma_to_interval(self._null_sigma_ref_base, lower, upper)
                    sigma_ref = clamp_sigma_to_interval(
                        rate_limit_scalar(
                            sigma_target,
                            self._sigma_ref_state,
                            self._null_sigma_rate_limit_Nps,
                            dt,
                        ),
                        lower,
                        upper,
                    )
                else:
                    sigma_ref = sigma_target
                self._sigma_ref_state = float(sigma_ref)
                T_meas = np.asarray(
                    [
                        float(axis_state.tension_estimate_n)
                        for axis_state in actuator_states
                        if axis_state.tension_estimate_n is not None
                    ],
                    dtype=float,
                )
                if T_meas is not None:
                    if T_meas.shape == (6,) and np.all(np.isfinite(T_meas)):
                        sigma_meas = float(n @ T_meas)
                if np.isfinite(sigma_meas) and dt > 0.0:
                    sigma_err = float(sigma_ref - sigma_meas)
                    self._eta_null_int += sigma_err * dt
                    eta_dot = self._null_tension_kp * sigma_err + self._null_tension_ki * self._eta_null_int
                    self._eta_null_m += dt * eta_dot
                    self._eta_null_m = float(np.clip(self._eta_null_m, -self._null_eta_limit_m, self._null_eta_limit_m))
                null_cmd_m = n * float(self._eta_null_m)
            except Exception as exc:
                logger.debug(f"[CTRL] Null-space tension controller skipped: {exc}")

        cmd_m = pose_ref_m + pose_corr_m + null_cmd_m
        cmd_mm = [1000.0 * float(v) for v in cmd_m]
        self._last_spool_cmd_mm = [float(x) for x in cmd_mm]
        self._last_spool_pose_cmd_mm = [1000.0 * float(v) for v in (pose_ref_m + pose_corr_m)]
        self._last_spool_null_cmd_mm = [1000.0 * float(v) for v in null_cmd_m]
        self._last_sigma_ref = float(sigma_ref)
        self._last_sigma_meas = float(sigma_meas)
        self._last_eta_null_m = float(self._eta_null_m)
        self._task_last_t = now

        cmd_turns = [float(cmd_mm[i]) / float(self._mm_per_turn[i]) for i in range(len(self.axis_ids))]
        cable_vel_mps = J_cmd @ qd_ref
        vel_turnsps = []
        for i in range(len(self.axis_ids)):
            mm_per_turn = float(self._mm_per_turn[i])
            if abs(mm_per_turn) < 1e-9:
                vel_turnsps.append(0.0)
            else:
                vel_turnsps.append(float(1000.0 * cable_vel_mps[i]) / mm_per_turn)

        torque_ff = [float(TORQUE_PER_TENSION) * float(tension) for tension in T_cmd]
        if not self._enable_position_torque_ff:
            torque_ff = [0.0] * len(torque_ff)
        self._last_torque_cmd_nm = [float(x) for x in torque_ff]
        return [
            ActuatorCommand(
                axis_id=aid,
                control_mode=ActuatorControlMode.POSITION,
                position_turns=float(cmd_turns[i]),
                velocity_ff_turns_per_s=float(vel_turnsps[i]),
                torque_ff_nm=float(torque_ff[i]),
            )
            for i, aid in enumerate(self.axis_ids)
        ]

    def _run_cablespace_fallback_control(self, actuator_states: tuple[ActuatorState, ...] = ()):
        """
        Fallback controller for drivers without platform-state feedback:
        cable-space PD + bias tension.
        """
        self._last_tau_plat_des[:] = np.nan
        t_mm, q = self.state.get_hand_pose()
        cable_mm = pose_to_cable_lengths_mm(GEOM, t_mm, q)
        cmd_mm = [cable_mm[i] - HOME_CABLE_MM[i] for i in range(6)]
        self._last_spool_cmd_mm = [float(x) for x in cmd_mm]
        torque_cmd = [0.0] * len(self.axis_ids)
        tension_cmd = [0.0] * len(self.axis_ids)
        commands = []

        for i, aid in enumerate(self.axis_ids):
            axis_state = actuator_states[i] if i < len(actuator_states) else None
            p_turns = None if axis_state is None else axis_state.position_turns
            v_turnsps = None if axis_state is None else axis_state.velocity_turns_per_s
            if p_turns is None or v_turnsps is None:
                commands.append(
                    ActuatorCommand(
                        axis_id=aid,
                        control_mode=ActuatorControlMode.TORQUE,
                        torque_nm=0.0,
                    )
                )
                continue

            fb_mm = float(p_turns) * MM_PER_TURN[i]
            fb_mmps = float(v_turnsps) * MM_PER_TURN[i]
            err_mm = float(cmd_mm[i]) - fb_mm
            tension_N = (
                TORQUE_CTRL_BIAS_N
                + TORQUE_CTRL_KP_N_PER_MM * err_mm
                - TORQUE_CTRL_KD_N_PER_MMPS * fb_mmps
            )
            tension_N = max(TORQUE_CTRL_MIN_N, min(TORQUE_CTRL_MAX_N, tension_N))
            torque_nm = float(tension_N) * TORQUE_PER_TENSION
            torque_cmd[i] = float(torque_nm)
            tension_cmd[i] = float(tension_N)
            commands.append(
                ActuatorCommand(
                    axis_id=aid,
                    control_mode=ActuatorControlMode.TORQUE,
                    torque_nm=float(torque_nm),
                )
            )
        self._last_torque_cmd_nm = [float(x) for x in torque_cmd]
        self._last_tension_cmd_N = [float(x) for x in tension_cmd]
        return commands

    def _publish_platform_estimate(self, q_cur, qd_cur):
        """
        Publish platform estimate into RuntimeMailbox for GUI telemetry.
        q_cur: [x,y,z,roll,pitch] in SI units.
        qd_cur: [xd,yd,zd,rolld,pitchd] in SI units.
        """
        try:
            q_cur = np.asarray(q_cur, dtype=float)
            qd_cur = np.asarray(qd_cur, dtype=float)
            t_mm = (1000.0 * float(q_cur[0]), 1000.0 * float(q_cur[1]), 1000.0 * float(q_cur[2]))
            q_est = quat_from_rpy_deg(math.degrees(float(q_cur[3])), math.degrees(float(q_cur[4])), 0.0)
            v_mps = (float(qd_cur[0]), float(qd_cur[1]), float(qd_cur[2]))
            w_rps = (float(qd_cur[3]), float(qd_cur[4]), 0.0)
            self.state.set_hand_estimate(t_mm, q_est, v_mps=v_mps, w_rps=w_rps)
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

    def _apply_state(self, st: str):
        """Apply high-level state to all axes."""
        try:
            self._write_command_batch(self._build_state_commands(st))
            for aid in self.axis_ids:
                if st == "enable":
                    logger.info(f"[CTRL] axis {aid}: POSITION + CLOSED_LOOP_CONTROL")
                elif st == "pretension":
                    logger.info(f"[CTRL] axis {aid}: TORQUE + CLOSED_LOOP_CONTROL")
                elif st in ("disable", "estop"):
                    logger.info(f"[CTRL] axis {aid}: IDLE")
            if st in ("disable", "estop"):
                self._last_torque_cmd_nm = [0.0] * 6
                self._last_tension_cmd_N = [0.0] * 6
                self._last_tau_plat_des[:] = np.nan
                self._task_err_int[:] = 0.0
                self._task_last_t = None
                self._eta_null_m = 0.0
                self._eta_null_int = 0.0
                self._null_basis_prev = None
                self._sigma_ref_state = float(self._null_sigma_ref_base)
                self._last_sigma_ref = float("nan")
                self._last_sigma_meas = float("nan")
                self._last_eta_null_m = 0.0
                self._last_spool_null_cmd_mm = [0.0] * 6

        except Exception as e:
            logger.error(f"[CTRL] _apply_state error: {e}")

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
            self._write_command_batch(self._build_state_commands("pretension"))
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

    def _update_sim_timing(self, now_perf):
        self._sim_time_s = float("nan")
        self._sim_rt_factor = float("nan")
        if not hasattr(self.driver, "get_sim_time"):
            self.state.set_control_time_s(None)
            self._sim_time_prev = None
            self._sim_wall_prev = None
            return
        try:
            sim_time = self.driver.get_sim_time()
        except Exception:
            sim_time = None
        if sim_time is None:
            self.state.set_control_time_s(None)
            self._sim_time_prev = None
            self._sim_wall_prev = None
            return

        sim_time = float(sim_time)
        self.state.set_control_time_s(sim_time)
        self._sim_time_s = sim_time
        if self._sim_time_prev is not None and self._sim_wall_prev is not None:
            ds = sim_time - self._sim_time_prev
            dw = float(now_perf - self._sim_wall_prev)
            if dw > 1e-6:
                self._sim_rt_factor = ds / dw
        self._sim_time_prev = sim_time
        self._sim_wall_prev = float(now_perf)

    def _write_diag_row(self, now_perf):
        if self._diag_writer is None:
            return

        from jugglebot.core.snapshots import build_robot_state_snapshot, flatten_robot_state

        tau_rsp = np.full(5, np.nan, dtype=float)
        tension_rsp = np.asarray(self.state.get_axis_tension_response(), dtype=float)
        _, _, J_len_plat = self._platform_estimator.get_latest()
        if J_len_plat is not None:
            try:
                J_len_plat = np.asarray(J_len_plat, dtype=float)
                if J_len_plat.shape == (6, 5) and tension_rsp.shape == (6,) and np.all(np.isfinite(tension_rsp)):
                    tau_rsp = float(TASK_WRENCH_FROM_TENSION_SIGN) * (J_len_plat.T @ tension_rsp)
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

if __name__ == "__main__":
    from jugglebot.transport.axes_logger import axes_state_logger
    from jugglebot.transport.tcp_commands import tcp_command_server

    state = RuntimeMailbox()
    can_ok = ensure_can_interface_up(ODRIVE_INTERFACE, ODRIVE_BITRATE)
    if not can_ok:
        logger.warning("[CAN] Continuing without CAN up")

    odrv_bridge = ControlBridge(state, None)  # <-- driver is None for now
    odrv_bridge.start()

    threading.Thread(target=tcp_command_server, args=(state,), daemon=True).start()
    threading.Thread(target=axes_state_logger, args=(state,), daemon=True).start()

    logger.info("Robot server running. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        odrv_bridge.stop()
