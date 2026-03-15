# robot_server.py
import socket
import json
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
    WinchCalibration,
    pose_to_cable_lengths_mm,
    q_from_axis_angle,
    q_mul,
    q_norm,
)

TCP_CMD_PORT = 5555
UDP_TELEM_PORT = 5556

# -------- ODrive CAN configuration --------
ODRIVE_INTERFACE = "can0"
ODRIVE_BITRATE = 1_000_000  # 1 Mbps
AXIS_NODE_IDS = [0, 1, 2, 3, 4, 5]
ODRIVE_COMMAND_RATE_HZ = 500.0
ODRIVE_LOG_RATE_HZ = 2.0
TELEMETRY_RATE_HZ = 50.0

# -------- Capstan / units configuration --------
# +turns (ODrive) reduces cable length, so use negative mm/turn such that +mm command extends cable.
MM_PER_TURN = [-62.832] * 6  # 2*pi*10mm = 62.832 mm/turn, with sign convention applied
# Pretension mapping: tension [N] -> capstan torque [Nm]
CAPSTAN_RADIUS_M = 0.010  # 10 mm
MOTOR_TORQUE_DIRECTION = 1  # Set to -1 if positive motor torque winds the cable and increases tension, +1 if opposite. This depends on your motor/winch wiring and should be set to ensure that positive torque commands increase tension.
TORQUE_PER_TENSION = MOTOR_TORQUE_DIRECTION * CAPSTAN_RADIUS_M  # Nm per N  (T = F*r)
TORQUE_CTRL_KP_N_PER_MM = 0.6
TORQUE_CTRL_KD_N_PER_MMPS = 0.02
TORQUE_CTRL_BIAS_N = 12.0
TORQUE_CTRL_MIN_N = 0.0
TORQUE_CTRL_MAX_N = 180.0
#TASK_KP = np.diag([1200.0, 1200.0, 1800.0, 120000.0, 120000.0])
TASK_KP = np.diag([500.0, 500.0, 800.0, 5.0, 5.0])
#TASK_KD = np.diag([80.0, 80.0, 120.0, 0.0, 0.0])
TASK_KD = np.diag([15.0, 15.0, 24.0, 0.1, 0.1])
TASK_KI = np.diag([0.0, 0.0, 0.0, 0.0, 0.0])
TASK_INT_CLIP = np.array([0.0, 0.0, 0.0, 0.35, 0.35], dtype=float)
TASK_TMIN_N = 5.0
TASK_TMAX_N = 180.0
TASK_ALLOC_LAMBDA = 1e-2
TASK_ALLOC_ITERS = 80
TASK_ALLOC_ALPHA = 0.7
TASK_GRAVITY_FF_Z_N = 1.2
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

def quat_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float = 0.0):
    """Quaternion for R = Rz(yaw)*Ry(pitch)*Rx(roll)."""
    r = math.radians(float(roll_deg))
    p = math.radians(float(pitch_deg))
    y = math.radians(float(yaw_deg))
    qx = q_from_axis_angle((1.0, 0.0, 0.0), r)
    qy = q_from_axis_angle((0.0, 1.0, 0.0), p)
    qz = q_from_axis_angle((0.0, 0.0, 1.0), y)
    return q_norm(q_mul(q_mul(qz, qy), qx))

# Precompute "home" geometric cable lengths in mm (used to convert absolute lengths -> delta lengths)
HOME_Q = quat_from_rpy_deg(HOME_ROLL_DEG, HOME_PITCH_DEG, HOME_YAW_DEG)
HOME_CABLE_MM = pose_to_cable_lengths_mm(GEOM, HOME_T_WORLD_MM, HOME_Q)  # returns mm given your mm geometry

DEFAULT_WINCH_CAL = WinchCalibration(
    spool_radius_mm=[10.0] * 6,
    gear_ratio=[1.0] * 6,
    sign=[-1.0] * 6,
    zero_length_mm=[0.0] * 6,
)

def turns_to_mm(turns_list, cal: WinchCalibration = DEFAULT_WINCH_CAL):
    """Convert [turns] -> [mm] elementwise using calibration."""
    if not isinstance(turns_list, (list, tuple)) or len(turns_list) != 6:
        raise ValueError("turns_list must be length-6 list/tuple")
    out = []
    for i in range(6):
        trn = float(turns_list[i])
        r = float(cal.spool_radius_mm[i])
        if r <= 0.0:
            raise ValueError(f"spool_radius_mm[{i}] must be > 0")

        spool_turns = trn / float(cal.sign[i]) / float(cal.gear_ratio[i])
        dL = spool_turns * 2.0 * math.pi * r
        L = dL + float(cal.zero_length_mm[i])
        out.append(L)
    return out

def mm_to_turns(mm_list, cal: WinchCalibration = DEFAULT_WINCH_CAL):
    """Convert [mm] -> [turns] elementwise using calibration."""
    if not isinstance(mm_list, (list, tuple)) or len(mm_list) != 6:
        raise ValueError("mm_list must be length-6 list/tuple")
    out = []
    for i in range(6):
        mm = float(mm_list[i])
        r = float(cal.spool_radius_mm[i])
        if r <= 0.0:
            raise ValueError(f"spool_radius_mm[{i}] must be > 0")

        L0 = float(cal.zero_length_mm[i])
        dL = mm - L0
        spool_turns = dL / (2.0 * math.pi * r)
        motor_turns = spool_turns * float(cal.gear_ratio[i])
        out.append(float(cal.sign[i]) * motor_turns)
    return out


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


def pose5_to_tq_mm(pose5):
    """Map [x_m, y_m, z_m, roll_rad, pitch_rad] -> (t_mm, q with yaw=0)."""
    x_m, y_m, z_m, roll_rad, pitch_rad = [float(v) for v in pose5]
    t_mm = (1000.0 * x_m, 1000.0 * y_m, 1000.0 * z_m)
    q = quat_from_rpy_deg(math.degrees(roll_rad), math.degrees(pitch_rad), 0.0)
    return t_mm, q


def cable_lengths_m_from_pose5(pose5):
    t_mm, q = pose5_to_tq_mm(pose5)
    L_mm = pose_to_cable_lengths_mm(GEOM, t_mm, q)
    return np.asarray(L_mm, dtype=float) / 1000.0


def cable_lengths_jacobian_pose5_fd(pose5, eps_pos_m=1e-4, eps_ang_rad=1e-4):
    """
    Finite-difference Jacobian of cable lengths wrt [x,y,z,roll,pitch].
    Returns J shape (6,5), where J[i,j] = dL_i / d pose_j.
    """
    q0 = np.asarray(pose5, dtype=float).copy()
    J = np.zeros((6, 5), dtype=float)
    for j in range(5):
        dq = np.zeros(5, dtype=float)
        dq[j] = eps_pos_m if j < 3 else eps_ang_rad
        Lp = cable_lengths_m_from_pose5(q0 + dq)
        Lm = cable_lengths_m_from_pose5(q0 - dq)
        J[:, j] = (Lp - Lm) / (2.0 * dq[j])
    return J


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

def _coerce_vec6_to_mm(msg, field_name: str):
    vec = msg.get(field_name, [])
    units = (msg.get("units") or "mm").lower()
    if not isinstance(vec, (list, tuple)) or len(vec) != 6:
        raise ValueError(f"{field_name} must be length-6 list")
    vec = [float(x) for x in vec]

    if units == "mm":
        return vec
    elif units == "turns":
        return turns_to_mm(vec)
    else:
        raise ValueError(f"Unknown units '{units}' (expected 'mm' or 'turns')")

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



class RobotState:
    def __init__(self):
        self.lock = threading.Lock()
        self.controller_ip = None
        self.axes_pos_cmd = [0.0] * 6
        self.state = "disable"
        self.state_version = 0
        self.home_pos = [0.0] * 6
        self.home_version = 0
        self.profile = []
        self.player_thread = None
        self.axes_pos_estimate = [None] * 6
        self.axes_vel_estimate = [None] * 6
        self.axes_bus_voltage = [None] * 6
        self.axes_bus_current = [None] * 6
        self.axes_motor_current = [None] * 6
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
        self.task_gain_version = 0
        self.task_kp_xyz_mult = 1.0
        self.task_kp_rp_mult = 1.0
        self.task_kd_xyz_mult = 1.0
        self.task_kd_rp_mult = 1.0
        # --- Hand (platform) command in global coordinates (mm + quaternion) ---
        self.hand_t_mm = (0.0, 0.0, 0.0)
        self.hand_q = (1.0, 0.0, 0.0, 0.0)
        self.hand_v_mps = (0.0, 0.0, 0.0)
        self.hand_a_mps2 = (0.0, 0.0, 0.0)
        self.hand_version = 0
        self.hand_est_t_mm = (float("nan"), float("nan"), float("nan"))
        self.hand_est_q = (1.0, 0.0, 0.0, 0.0)
        self.hand_est_v_mps = (float("nan"), float("nan"), float("nan"))
        self.hand_est_w_rps = (float("nan"), float("nan"), float("nan"))
        self.pose_profile = []  # list of [t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
        self.control_time_s = None

    def set_hand_pose(self, t_mm, q, v_mps=None, a_mps2=None):
        # t_mm: (x,y,z) in mm, q: quaternion (w,x,y,z)
        with self.lock:
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
            self.hand_version += 1

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

    def get_hand_version(self):
        with self.lock:
            return int(self.hand_version)

    def set_control_time_s(self, t_s):
        with self.lock:
            self.control_time_s = None if t_s is None else float(t_s)

    def get_control_time_s(self):
        with self.lock:
            return self.control_time_s

    def set_controller_ip(self, ip):
        with self.lock:
            self.controller_ip = ip
        logger.info(f"Controller IP set to {ip}")

    def get_controller_ip(self):
        with self.lock:
            return self.controller_ip

    def get_pos_cmd(self):
        with self.lock:
            return list(self.axes_pos_cmd)

    def get_pos_fbk(self):
        with self.lock:
            return list(self.axes_pos_estimate)

    def get_vel_fbk(self):
        with self.lock:
            return list(self.axes_vel_estimate)

    #Home methods
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

    #Set methods
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
            proc_result=None
    ):
        """Store measured feedback for a single axis index (0..5)."""
        if not (0 <= int(axis_id) < 6):                         #This hardcodes id's 0-5, this should be upgraded to check against a list of initialized controllers
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
            if bus_current is not None:  # <-- new
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
        """Return list of bus voltage values (may contain None)."""
        with self.lock:
            return list(self.axes_bus_voltage)

    def get_bus_current(self):
        with self.lock:
            return list(self.axes_bus_current)

    def get_motor_current(self):
        with self.lock:
            return list(self.axes_motor_current)

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

    def set_axes(self, positions):
        if not isinstance(positions, (list, tuple)) or len(positions) != 6:
            raise ValueError("positions must be length-6 list/tuple")
        with self.lock:
            self.axes_pos_cmd = [float(x) for x in positions]
        logger.info("Axes target set (mm): " + ", ".join(f"{x:.3f}" for x in self.axes_pos_cmd))

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

    #Profile methods (direct commands to axes)
    def set_profile(self, profile_points):
        if not isinstance(profile_points, (list, tuple)) or len(profile_points) == 0:
            raise ValueError("profile must be a non-empty list")
        prof = []
        for row in profile_points:
            if not isinstance(row, (list, tuple)) or len(row) < 7:
                raise ValueError("each profile row must be [t, a1..a6]")
            t = float(row[0])
            axes = [float(x) for x in row[1:7]]
            prof.append((t, axes))
        times = [p[0] for p in prof]
        if any(t2 < t1 for t1, t2 in zip(times, times[1:])):
            raise ValueError("profile time column must be non-decreasing")
        with self.lock:
            self.profile = prof
        logger.info(f"Profile uploaded: {len(prof)} points, duration {prof[-1][0] - prof[0][0]:.3f}s")

    def get_profile(self):
        """Return the currently stored profile as a list of (t, axes)."""
        with self.lock:
            return list(self.profile)

    def start_profile(self, rate_hz: float):
        """Start executing the uploaded profile at a given rate (Hz)."""
        self.stop_profile()
        prof = self.get_profile()
        if not prof:
            raise RuntimeError("no profile uploaded")
        player = ProfilePlayer(self, prof, rate_hz)
        with self.lock:
            self.player_thread = player
        logger.info(f"Profile start at {rate_hz:.1f} Hz")
        player.start()

    def stop_profile(self):
        """Stop any running profile playback."""
        with self.lock:
            player = self.player_thread
            self.player_thread = None
        if player and player.is_alive():
            player.stop()
            player.join(timeout=1.0)
            logger.info("Profile stopped")

    # Pose Profile methods
    def set_pose_profile(self, profile_pose: list):
        """
        profile_pose rows support:
          [t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
        or
          [t, x_mm, y_mm, z_mm, vx_mps, vy_mps, vz_mps, ax_mps2, ay_mps2, az_mps2, roll_deg, pitch_deg, yaw_deg]
        Stored as list[(t, pose6, v3, a3)].
        """
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

    def start_pose_profile(self, rate_hz: float):
        self.stop_profile()
        prof = self.get_pose_profile()
        if not prof:
            raise RuntimeError("no pose profile uploaded")
        player = PoseProfilePlayer(self, prof, rate_hz)
        with self.lock:
            self.player_thread = player
        logger.info(f"Pose profile start at {rate_hz:.1f} Hz")
        player.start()

    # --- Telemetry lifecycle ---
    def start_telem(self, udp_sock, controller_addr):
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

    #Pretension methods
    def request_pretension(self, upper_N: float, lower_N: float):
        with self.lock:
            self.pret_upper_N = float(upper_N)
            self.pret_lower_N = float(lower_N)
            self.pret_version += 1
            self.state = "pretension"
            self.state_version += 1
        logger.info(f"[PRET] requested upper={self.pret_upper_N:.3f} N, lower={self.pret_lower_N:.3f} N "
                    f"(pret_version {self.pret_version})")

    def get_pretension(self):
        with self.lock:
            return float(self.pret_upper_N), float(self.pret_lower_N)

    def get_pretension_version(self):
        with self.lock:
            return int(self.pret_version)

    def request_task_gain_multipliers(self, kp_xyz=None, kp_rp=None, kd_xyz=None, kd_rp=None):
        with self.lock:
            if kp_xyz is not None:
                self.task_kp_xyz_mult = float(kp_xyz)
            if kp_rp is not None:
                self.task_kp_rp_mult = float(kp_rp)
            if kd_xyz is not None:
                self.task_kd_xyz_mult = float(kd_xyz)
            if kd_rp is not None:
                self.task_kd_rp_mult = float(kd_rp)
            self.task_gain_version += 1
        logger.info(
            "[TASK_GAIN] multipliers set: "
            f"kp_xyz={self.task_kp_xyz_mult:.3f}, kp_rp={self.task_kp_rp_mult:.3f}, "
            f"kd_xyz={self.task_kd_xyz_mult:.3f}, kd_rp={self.task_kd_rp_mult:.3f}"
        )

    def get_task_gain_multipliers(self):
        with self.lock:
            return (
                float(self.task_kp_xyz_mult),
                float(self.task_kp_rp_mult),
                float(self.task_kd_xyz_mult),
                float(self.task_kd_rp_mult),
            )

    def get_task_gain_version(self):
        with self.lock:
            return int(self.task_gain_version)


class ProfilePlayer(threading.Thread):
    """Plays a time-position profile with linear interpolation at fixed rate."""

    def __init__(self, state: RobotState, profile: list[tuple[float, list[float]]], rate_hz: float):
        super().__init__(daemon=True)
        self.state = state
        self._stop = threading.Event()
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
        self.dt = 1.0 / rate_hz

        # Normalize profile times so playback starts at t=0
        if not profile:
            raise ValueError("empty profile")
        t0 = float(profile[0][0])
        norm = []
        for t, axes in profile:
            norm.append((float(t) - t0, [float(x) for x in axes]))
        self.norm_profile = norm
        self.duration = norm[-1][0] if norm else 0.0

    def stop(self):
        self._stop.set()

    def run(self):
        if self.duration <= 0.0:
            # immediate set and exit
            self.state.set_axes(self.norm_profile[-1][1])
            logger.info("[PROFILE] Zero-duration profile applied")
            return

        logger.info(f"[PROFILE] Starting playback at {1.0/self.dt:.1f} Hz, duration {self.duration:.3f}s")
        start = time.perf_counter()
        k = 0
        while not self._stop.is_set():
            t = time.perf_counter() - start
            if t >= self.duration:
                self.state.set_axes(self.norm_profile[-1][1])
                logger.info("[PROFILE] Completed")
                break

            # advance segment index
            while k + 1 < len(self.norm_profile) and self.norm_profile[k + 1][0] <= t:
                k += 1

            t0, p0 = self.norm_profile[k]
            t1, p1 = self.norm_profile[min(k + 1, len(self.norm_profile) - 1)]
            if t1 <= t0:
                alpha = 0.0
            else:
                alpha = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
            axes = [p0[i] + alpha * (p1[i] - p0[i]) for i in range(6)]
            try:
                self.state.set_axes(axes)
            except Exception as e:
                logger.error(f"[PROFILE] set_axes error: {e}")
            time.sleep(self.dt)

        # cleanup: clear player_thread reference if still pointing to us
        with self.state.lock:
            if self.state.player_thread is self:
                self.state.player_thread = None

class PoseProfilePlayer(threading.Thread):
    """
    Plays a time-pose profile:
      [t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
    Converts pose -> quaternion -> IK -> axis mm commands, then calls state.set_axes(mm).
    """

    def __init__(self, state: RobotState, profile, rate_hz: float):
        super().__init__(daemon=True)
        self.state = state
        self._stop = threading.Event()
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
        self.dt = 1.0 / rate_hz

        if not profile:
            raise ValueError("empty pose profile")

        t0 = float(profile[0][0])
        norm = []
        for row in profile:
            t = float(row[0])
            if len(row) >= 4:
                pose6 = [float(x) for x in row[1]]
                v3 = [float(x) for x in row[2]]
                a3 = [float(x) for x in row[3]]
            else:
                pose6 = [float(x) for x in row[1]]
                v3 = [0.0, 0.0, 0.0]
                a3 = [0.0, 0.0, 0.0]
            norm.append((float(t) - t0, pose6, v3, a3))
        self.norm_profile = norm
        self.duration = norm[-1][0] if norm else 0.0
        self._wall_start = None
        self._control_start = None

    def stop(self):
        self._stop.set()

    def _profile_elapsed_s(self):
        control_now = self.state.get_control_time_s()
        if control_now is not None:
            if self._control_start is None:
                self._control_start = float(control_now)
            return float(control_now) - self._control_start
        if self._wall_start is None:
            self._wall_start = time.perf_counter()
        return time.perf_counter() - self._wall_start

    def run(self):
        if self.duration <= 0.0:
            _, pose6, v3, a3 = self.norm_profile[-1]
            self._apply_pose(pose6, v3, a3)
            logger.info("[POSE_PROFILE] Zero-duration pose profile applied")
            return

        logger.info(f"[POSE_PROFILE] Starting playback at {1.0/self.dt:.1f} Hz, duration {self.duration:.3f}s")
        k = 0

        while not self._stop.is_set():
            t = self._profile_elapsed_s()
            if t >= self.duration:
                _, pose6, v3, a3 = self.norm_profile[-1]
                self._apply_pose(pose6, v3, a3)
                logger.info("[POSE_PROFILE] Completed")
                break

            while k + 1 < len(self.norm_profile) and self.norm_profile[k + 1][0] <= t:
                k += 1

            t0, p0, v0, a0 = self.norm_profile[k]
            t1, p1, v1, a1 = self.norm_profile[min(k + 1, len(self.norm_profile) - 1)]
            if t1 <= t0:
                alpha = 0.0
            else:
                alpha = max(0.0, min(1.0, (t - t0) / (t1 - t0)))

            pose = [p0[i] + alpha * (p1[i] - p0[i]) for i in range(6)]
            vel = [v0[i] + alpha * (v1[i] - v0[i]) for i in range(3)]
            acc = [a0[i] + alpha * (a1[i] - a0[i]) for i in range(3)]
            try:
                self._apply_pose(pose, vel, acc)
            except Exception as e:
                logger.error(f"[POSE_PROFILE] apply_pose error: {e}")

            time.sleep(self.dt)

        # cleanup
        with self.state.lock:
            if self.state.player_thread is self:
                self.state.player_thread = None

    def _apply_pose(self, pose6, vel3=None, acc3=None):
        x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = pose6

        # You already have quat_from_rpy_deg, and you already handle pose commands similarly. :contentReference[oaicite:7]{index=7}
        q = quat_from_rpy_deg(roll_deg, pitch_deg, yaw_deg)

        # Reuse your existing "pose -> IK -> axes" path
        # If your RobotState.set_hand_pose() already runs IK and updates axis commands, call it:
        self.state.set_hand_pose((x_mm, y_mm, z_mm), q, v_mps=vel3, a_mps2=acc3)

        # If set_hand_pose currently only stores pose and doesn't compute IK yet,
        # then instead call your cable_ik conversion here (whatever function you wired in).


class ControlBridge(threading.Thread):
    """Bridge between RobotState and robot driver (hardware or simulation)."""

    def __init__(self, state: RobotState, driver, axis_ids=None, diag_log_dir: str | None = None, diag_log_hz: float = 100.0):
        super().__init__(daemon=True)
        self.state = state
        self.driver = driver
        self.axis_ids = axis_ids or [0, 1, 2, 3, 4, 5]
        self._stop = threading.Event()
        self._T_prev = None
        self._task_err_int = np.zeros(5, dtype=float)
        self._task_last_t = None
        self.diag_log_dir = diag_log_dir or os.path.join(os.getcwd(), "Logs")
        self.diag_log_hz = max(1.0, float(diag_log_hz))
        self._diag_file = None
        self._diag_writer = None
        self._diag_log_path = None
        self._diag_start_perf = None
        self._diag_last_log_perf = 0.0
        self._last_spool_cmd_mm = [float("nan")] * 6
        self._last_torque_cmd_nm = [0.0] * 6
        self._last_tension_cmd_N = [0.0] * 6
        self._last_tau_plat_des = np.full(5, np.nan, dtype=float)
        self._sim_time_s = float("nan")
        self._sim_rt_factor = float("nan")
        self._sim_time_prev = None
        self._sim_wall_prev = None
        self._task_kp_runtime = TASK_KP.copy()
        self._task_kd_runtime = TASK_KD.copy()
        self._task_ki_runtime = TASK_KI.copy()

        #Apply the current state version to avoid auto applying the default by setting these to -1.  Perhaps reconsider this for desired auto init behavior later on
        self._applied_state_version = state.get_state_version()
        self._applied_home_version = state.get_home_version()
        self._applied_pret_version = state.get_pretension_version()
        self._applied_task_gain_version = -1

    def stop(self):
        self._stop.set()
        if self.driver:
            try:
                self.driver.stop()
            except Exception:
                pass

    def run(self):
        logger.info("[CTRL] Starting control bridge...")
        try:
            # Start the driver
            self.driver.start()
            self._open_diag_log()

            # Set up callbacks
            self.driver.set_position_callback(lambda aid, pos: self.state.set_axis_feedback(aid, pos_estimate=pos))
            self.driver.set_velocity_callback(lambda aid, vel: self.state.set_axis_feedback(aid, vel_estimate=vel))
            self.driver.set_bus_callback(lambda aid, vbus, ibus: self.state.set_axis_feedback(aid, bus_voltage=vbus, bus_current=ibus))
            self.driver.set_current_callback(lambda aid, curr: self.state.set_axis_feedback(aid, motor_current=curr))
            self.driver.set_temp_callback(lambda aid, fet, motor: self.state.set_axis_feedback(aid, temp_fet=fet, temp_motor=motor))
            self.driver.set_heartbeat_callback(lambda aid, err, st, proc: self.state.set_axis_feedback(aid, axis_error=err, axis_state=st, proc_result=proc))

            # main loop (~500 Hz)
            last_log = time.perf_counter()
            while not self._stop.is_set():
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

                gv = self.state.get_task_gain_version()
                if gv != self._applied_task_gain_version:
                    self._apply_task_gain_multipliers()
                    self._applied_task_gain_version = gv

                # Stream setpoints if enabled
                if st == "enable":
                    try:
                        if hasattr(self.driver, "get_platform_state"):
                            self._run_taskspace_torque_control()
                        else:
                            self._run_cablespace_fallback_control()

                    except Exception as e:
                        # IMPORTANT: don't kill the bridge if IK/units blow up
                        logger.error(f"[CTRL] ENABLE streaming error: {e}")

                elif st == "pretension":
                    self._last_tau_plat_des[:] = np.nan
                    upper_N, lower_N = self.state.get_pretension()

                    # Map upper/lower tension to per-axis torque commands
                    torque_cmd = [0.0] * 6
                    tension_cmd = [0.0] * 6
                    for i in (0, 2, 4):
                        tension_cmd[i] = upper_N
                        torque_cmd[i] = upper_N * TORQUE_PER_TENSION
                    for i in (1, 3, 5):
                        tension_cmd[i] = lower_N
                        torque_cmd[i] = lower_N * TORQUE_PER_TENSION

                    for i, aid in enumerate(self.axis_ids):
                        self.driver.set_axis_torque(aid, torque_cmd[i])
                    self._last_tension_cmd_N = [float(x) for x in tension_cmd]
                    self._last_torque_cmd_nm = [float(x) for x in torque_cmd]

                # light heartbeat log
                now = time.perf_counter()
                if hasattr(self.driver, "get_platform_state"):
                    try:
                        q_cur, qd_cur = self.driver.get_platform_state()
                        if q_cur is not None and qd_cur is not None:
                            self._publish_platform_estimate(q_cur, qd_cur)
                    except Exception:
                        pass
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

    def _run_taskspace_torque_control(self):
        """Task-space controller with Jacobian-based tension allocation."""
        t_mm_cmd, q_cmd, v_cmd_mps, a_cmd_mps2 = self.state.get_hand_motion()
        cable_mm = pose_to_cable_lengths_mm(GEOM, t_mm_cmd, q_cmd)
        cmd_mm = [cable_mm[i] - HOME_CABLE_MM[i] for i in range(6)]
        self._last_spool_cmd_mm = [float(x) for x in cmd_mm]
        roll_cmd, pitch_cmd, _ = quat_to_rpy_rad(q_cmd)
        q_ref = np.array(
            [t_mm_cmd[0] / 1000.0, t_mm_cmd[1] / 1000.0, t_mm_cmd[2] / 1000.0, roll_cmd, pitch_cmd],
            dtype=float,
        )
        qd_ref = np.array([float(v_cmd_mps[0]), float(v_cmd_mps[1]), float(v_cmd_mps[2]), 0.0, 0.0], dtype=float)
        qdd_ff = np.array([float(a_cmd_mps2[0]), float(a_cmd_mps2[1]), float(a_cmd_mps2[2]), 0.0, 0.0], dtype=float)

        q_cur, qd_cur = self.driver.get_platform_state()
        if q_cur is None or qd_cur is None:
            self._run_cablespace_fallback_control()
            return
        self._publish_platform_estimate(q_cur, qd_cur)
        q_cur = np.asarray(q_cur, dtype=float)
        qd_cur = np.asarray(qd_cur, dtype=float)

        e = q_ref - q_cur
        ed = qd_ref - qd_cur
        now = time.perf_counter()
        if self._task_last_t is None:
            dt = 0.002
        else:
            dt = max(1e-4, min(0.05, now - self._task_last_t))
        self._task_last_t = now

        # Integrate only selected channels (roll/pitch by default), with anti-windup clipping.
        self._task_err_int += e * dt
        self._task_err_int = np.clip(self._task_err_int, -TASK_INT_CLIP, TASK_INT_CLIP)

        qdd_fb = self._task_kp_runtime @ e + self._task_kd_runtime @ ed + self._task_ki_runtime @ self._task_err_int
        qdd_cmd = qdd_ff + qdd_fb

        if hasattr(self.driver, "compute_platform_wrench"):
            tau_plat_des = np.asarray(self.driver.compute_platform_wrench(qdd_cmd), dtype=float)
        else:
            tau_plat_des = np.asarray(qdd_cmd, dtype=float)
            tau_plat_des[2] += TASK_GRAVITY_FF_Z_N

        if hasattr(self.driver, "get_cable_jacobian_plat"):
            J_len_plat = np.asarray(self.driver.get_cable_jacobian_plat(), dtype=float)
        else:
            J_len_plat = cable_lengths_jacobian_pose5_fd(q_cur)

        self._last_tau_plat_des = np.asarray(tau_plat_des, dtype=float)
        T_des = solve_tensions_least_squares(J_len_plat, tau_plat_des, self._T_prev)
        self._T_prev = T_des.copy()
        self._last_tension_cmd_N = [float(x) for x in T_des]

        tau_cmd = TORQUE_PER_TENSION * T_des
        self._last_torque_cmd_nm = [float(x) for x in tau_cmd]
        for i, aid in enumerate(self.axis_ids):
            self.driver.set_axis_torque(aid, float(tau_cmd[i]))

    def _run_cablespace_fallback_control(self):
        """
        Fallback controller for drivers without platform-state feedback:
        cable-space PD + bias tension.
        """
        self._last_tau_plat_des[:] = np.nan
        t_mm, q = self.state.get_hand_pose()
        cable_mm = pose_to_cable_lengths_mm(GEOM, t_mm, q)
        cmd_mm = [cable_mm[i] - HOME_CABLE_MM[i] for i in range(6)]
        self._last_spool_cmd_mm = [float(x) for x in cmd_mm]
        fb_pos_turns = self.state.get_pos_fbk()
        fb_vel_turnsps = self.state.get_vel_fbk()
        torque_cmd = [0.0] * 6
        tension_cmd = [0.0] * 6

        for i, aid in enumerate(self.axis_ids):
            p_turns = fb_pos_turns[i] if i < len(fb_pos_turns) else None
            v_turnsps = fb_vel_turnsps[i] if i < len(fb_vel_turnsps) else None
            if p_turns is None or v_turnsps is None:
                self.driver.set_axis_torque(aid, 0.0)
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
            self.driver.set_axis_torque(aid, torque_nm)
            torque_cmd[i] = float(torque_nm)
            tension_cmd[i] = float(tension_N)
        self._last_torque_cmd_nm = [float(x) for x in torque_cmd]
        self._last_tension_cmd_N = [float(x) for x in tension_cmd]

    def _publish_platform_estimate(self, q_cur, qd_cur):
        """
        Publish platform estimate into RobotState for GUI telemetry.
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

    def _apply_task_gain_multipliers(self):
        kp_xyz, kp_rp, kd_xyz, kd_rp = self.state.get_task_gain_multipliers()
        self._task_kp_runtime = TASK_KP.copy()
        self._task_kd_runtime = TASK_KD.copy()
        self._task_ki_runtime = TASK_KI.copy()

        self._task_kp_runtime[0:3, 0:3] *= kp_xyz
        self._task_kp_runtime[3:5, 3:5] *= kp_rp
        self._task_kd_runtime[0:3, 0:3] *= kd_xyz
        self._task_kd_runtime[3:5, 3:5] *= kd_rp
        logger.info(
            "[CTRL] Task gain multipliers applied: "
            f"kp_xyz={kp_xyz:.3f}, kp_rp={kp_rp:.3f}, kd_xyz={kd_xyz:.3f}, kd_rp={kd_rp:.3f}"
        )

    def _apply_state(self, st: str):
        """Apply high-level state to all axes."""
        try:
            for aid in self.axis_ids:
                if st == "enable":
                    self.driver.set_controller_mode(aid, "torque")
                    self.driver.set_axis_state(aid, "closed_loop")
                    logger.info(f"[CTRL] axis {aid}: TORQUE + CLOSED_LOOP_CONTROL")

                elif st == "pretension":
                    self.driver.set_controller_mode(aid, "torque")
                    self.driver.set_axis_state(aid, "closed_loop")
                    logger.info(f"[CTRL] axis {aid}: TORQUE + CLOSED_LOOP_CONTROL")

                elif st in ("disable", "estop"):
                    self.driver.set_axis_state(aid, "idle")
                    logger.info(f"[CTRL] axis {aid}: IDLE")
            if st in ("disable", "estop"):
                self._last_torque_cmd_nm = [0.0] * 6
                self._last_tension_cmd_N = [0.0] * 6
                self._last_tau_plat_des[:] = np.nan
                self._task_err_int[:] = 0.0
                self._task_last_t = None

        except Exception as e:
            logger.error(f"[CTRL] _apply_state error: {e}")

    def _apply_home(self):
        """
        HOME intent: do NOT move motors.
        We reset the estimator's absolute position (pos_estimate) to the GUI-provided
        home positions, and also align the streamed setpoints to those same values
        so the controller doesn't step.
        """
        home_pos_mm = self.state.get_home_pos()  # mm

        # 1) Update command setpoints first
        try:
            self.state.set_axes(home_pos_mm)  # keep cmd in mm
        except Exception as e:
            logger.error(f"[HOME] Failed to set_axes(home_pos_mm): {e}")

        # 2) Send Set_Absolute_Position to each axis (in turns)
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
            for aid in self.axis_ids:
                self.driver.set_controller_mode(aid, "torque")
                self.driver.set_axis_state(aid, "closed_loop")
            logger.info("[PRET] applied torque control mode to all axes")
        except Exception as e:
            logger.error(f"[PRET] _apply_pretension_mode error: {e}")

    def _open_diag_log(self):
        os.makedirs(self.diag_log_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._diag_log_path = os.path.join(self.diag_log_dir, f"control_diag_{ts}.csv")
        self._diag_file = open(self._diag_log_path, "w", newline="", encoding="utf-8")
        self._diag_writer = csv.writer(self._diag_file)
        self._diag_writer.writerow(self._diag_headers())
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

    def _diag_headers(self):
        headers = [
            "t_rel_s",
            "t_wall_s",
            "sim_time_s",
            "sim_rt_factor",
            "state",
            "profile_active",
            "hand_cmd_x_mm",
            "hand_cmd_y_mm",
            "hand_cmd_z_mm",
            "hand_cmd_roll_deg",
            "hand_cmd_pitch_deg",
            "hand_cmd_yaw_deg",
            "hand_cmd_vx_mps",
            "hand_cmd_vy_mps",
            "hand_cmd_vz_mps",
            "hand_cmd_ax_mps2",
            "hand_cmd_ay_mps2",
            "hand_cmd_az_mps2",
            "hand_rsp_x_mm",
            "hand_rsp_y_mm",
            "hand_rsp_z_mm",
            "hand_rsp_roll_deg",
            "hand_rsp_pitch_deg",
            "hand_rsp_yaw_deg",
            "hand_rsp_vx_mps",
            "hand_rsp_vy_mps",
            "hand_rsp_vz_mps",
            "hand_rsp_rollrate_degps",
            "hand_rsp_pitchrate_degps",
            "wrench_cmd_fx_N",
            "wrench_cmd_fy_N",
            "wrench_cmd_fz_N",
            "wrench_cmd_tx_Nm",
            "wrench_cmd_ty_Nm",
            "wrench_rsp_fx_N",
            "wrench_rsp_fy_N",
            "wrench_rsp_fz_N",
            "wrench_rsp_tx_Nm",
            "wrench_rsp_ty_Nm",
        ]
        for i in range(6):
            headers.append(f"spool_cmd_mm_{i + 1}")
        for i in range(6):
            headers.append(f"spool_rsp_mm_{i + 1}")
        for i in range(6):
            headers.append(f"spool_rsp_mmps_{i + 1}")
        for i in range(6):
            headers.append(f"spool_cmd_torque_nm_{i + 1}")
        for i in range(6):
            headers.append(f"spool_rsp_torque_nm_{i + 1}")
        for i in range(6):
            headers.append(f"spool_cmd_tension_N_{i + 1}")
        for i in range(6):
            headers.append(f"spool_rsp_tension_N_{i + 1}")
        for i in range(6):
            headers.append(f"bus_v_{i + 1}")
        for i in range(6):
            headers.append(f"bus_i_{i + 1}")
        for i in range(6):
            headers.append(f"motor_i_{i + 1}")
        for i in range(6):
            headers.append(f"temp_fet_c_{i + 1}")
        for i in range(6):
            headers.append(f"temp_motor_c_{i + 1}")
        return headers

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

    @staticmethod
    def _float_or_nan(x):
        if x is None:
            return float("nan")
        try:
            return float(x)
        except Exception:
            return float("nan")

    def _write_diag_row(self, now_perf):
        if self._diag_writer is None:
            return

        t_rel = float(now_perf - self._diag_start_perf)
        t_wall = time.time()
        state_name = self.state.get_state()
        profile_active = 0
        with self.state.lock:
            if self.state.player_thread is not None:
                profile_active = 1

        hand_t_mm, hand_q, hand_v_cmd_mps, hand_a_cmd_mps2 = self.state.get_hand_motion()
        hand_cmd_roll, hand_cmd_pitch, hand_cmd_yaw = quat_to_rpy_rad(hand_q)
        hand_cmd_roll = math.degrees(hand_cmd_roll)
        hand_cmd_pitch = math.degrees(hand_cmd_pitch)
        hand_cmd_yaw = math.degrees(hand_cmd_yaw)

        hand_rsp = [float("nan")] * 11
        if hasattr(self.driver, "get_platform_state"):
            try:
                q_cur, qd_cur = self.driver.get_platform_state()
                if q_cur is not None and qd_cur is not None:
                    q_cur = np.asarray(q_cur, dtype=float)
                    qd_cur = np.asarray(qd_cur, dtype=float)
                    hand_rsp[0] = 1000.0 * float(q_cur[0])
                    hand_rsp[1] = 1000.0 * float(q_cur[1])
                    hand_rsp[2] = 1000.0 * float(q_cur[2])
                    hand_rsp[3] = math.degrees(float(q_cur[3]))
                    hand_rsp[4] = math.degrees(float(q_cur[4]))
                    hand_rsp[5] = 0.0
                    hand_rsp[6] = float(qd_cur[0])
                    hand_rsp[7] = float(qd_cur[1])
                    hand_rsp[8] = float(qd_cur[2])
                    hand_rsp[9] = math.degrees(float(qd_cur[3]))
                    hand_rsp[10] = math.degrees(float(qd_cur[4]))
            except Exception:
                pass

        fb_pos_turns = self.state.get_pos_fbk()
        fb_vel_turnsps = self.state.get_vel_fbk()
        spool_rsp_mm = [float("nan")] * 6
        spool_rsp_mmps = [float("nan")] * 6
        for i in range(6):
            if i < len(fb_pos_turns):
                p = self._float_or_nan(fb_pos_turns[i])
                spool_rsp_mm[i] = p * MM_PER_TURN[i] if np.isfinite(p) else float("nan")
            if i < len(fb_vel_turnsps):
                v = self._float_or_nan(fb_vel_turnsps[i])
                spool_rsp_mmps[i] = v * MM_PER_TURN[i] if np.isfinite(v) else float("nan")

        torque_rsp = [float("nan")] * 6
        if hasattr(self.driver, "get_axis_torques"):
            try:
                trq = self.driver.get_axis_torques()
                if trq is not None:
                    for i in range(min(6, len(trq))):
                        torque_rsp[i] = self._float_or_nan(trq[i])
            except Exception:
                pass

        tension_rsp = [float("nan")] * 6
        if hasattr(self.driver, "get_cable_tensions"):
            try:
                tr = self.driver.get_cable_tensions()
                if tr is not None:
                    for i in range(min(6, len(tr))):
                        tension_rsp[i] = self._float_or_nan(tr[i])
            except Exception:
                pass
        # Fallback: infer equivalent tension from applied torque feedback.
        for i in range(6):
            if not np.isfinite(tension_rsp[i]) and np.isfinite(torque_rsp[i]):
                tension_rsp[i] = float(torque_rsp[i]) / float(TORQUE_PER_TENSION)

        tau_cmd = np.asarray(self._last_tau_plat_des, dtype=float).copy()
        tau_rsp = np.full(5, np.nan, dtype=float)
        if hasattr(self.driver, "get_cable_jacobian_plat"):
            try:
                J_len_plat = np.asarray(self.driver.get_cable_jacobian_plat(), dtype=float)
                T_rsp = np.asarray(tension_rsp, dtype=float)
                if J_len_plat.shape == (6, 5) and np.all(np.isfinite(T_rsp)):
                    tau_rsp = float(TASK_WRENCH_FROM_TENSION_SIGN) * (J_len_plat.T @ T_rsp)
            except Exception:
                pass

        row = [
            t_rel,
            t_wall,
            self._sim_time_s,
            self._sim_rt_factor,
            state_name,
            profile_active,
            float(hand_t_mm[0]),
            float(hand_t_mm[1]),
            float(hand_t_mm[2]),
            hand_cmd_roll,
            hand_cmd_pitch,
            hand_cmd_yaw,
            float(hand_v_cmd_mps[0]),
            float(hand_v_cmd_mps[1]),
            float(hand_v_cmd_mps[2]),
            float(hand_a_cmd_mps2[0]),
            float(hand_a_cmd_mps2[1]),
            float(hand_a_cmd_mps2[2]),
            hand_rsp[0],
            hand_rsp[1],
            hand_rsp[2],
            hand_rsp[3],
            hand_rsp[4],
            hand_rsp[5],
            hand_rsp[6],
            hand_rsp[7],
            hand_rsp[8],
            hand_rsp[9],
            hand_rsp[10],
            self._float_or_nan(tau_cmd[0]),
            self._float_or_nan(tau_cmd[1]),
            self._float_or_nan(tau_cmd[2]),
            self._float_or_nan(tau_cmd[3]),
            self._float_or_nan(tau_cmd[4]),
            self._float_or_nan(tau_rsp[0]),
            self._float_or_nan(tau_rsp[1]),
            self._float_or_nan(tau_rsp[2]),
            self._float_or_nan(tau_rsp[3]),
            self._float_or_nan(tau_rsp[4]),
        ]

        row.extend([self._float_or_nan(v) for v in self._last_spool_cmd_mm])
        row.extend([self._float_or_nan(v) for v in spool_rsp_mm])
        row.extend([self._float_or_nan(v) for v in spool_rsp_mmps])
        row.extend([self._float_or_nan(v) for v in self._last_torque_cmd_nm])
        row.extend([self._float_or_nan(v) for v in torque_rsp])
        row.extend([self._float_or_nan(v) for v in self._last_tension_cmd_N])
        row.extend([self._float_or_nan(v) for v in tension_rsp])
        row.extend([self._float_or_nan(v) for v in self.state.get_bus_voltage()])
        row.extend([self._float_or_nan(v) for v in self.state.get_bus_current()])
        row.extend([self._float_or_nan(v) for v in self.state.get_motor_current()])
        row.extend([self._float_or_nan(v) for v in self.state.get_temp_fet()])
        row.extend([self._float_or_nan(v) for v in self.state.get_temp_motor()])
        self._diag_writer.writerow(row)
        if self._diag_file is not None:
            self._diag_file.flush()


def udp_telemetry_sender(state: RobotState, udp_sock, stop_event):
    while not stop_event.is_set():
        try:
            controller_ip = state.get_controller_ip()
            if controller_ip:
                controller_addr = (controller_ip, UDP_TELEM_PORT)
                fb_pos_turns = state.get_pos_fbk()
                fb_vel_turnsps = state.get_vel_fbk()

                # Convert to mm + mm/s for the GUI
                fb_pos_mm = []
                fb_vel_mmps = []
                for i in range(6):
                    p = fb_pos_turns[i] if i < len(fb_pos_turns) else None
                    v = fb_vel_turnsps[i] if i < len(fb_vel_turnsps) else None
                    k = MM_PER_TURN[i]
                    fb_pos_mm.append(None if p is None else float(p) * k)
                    fb_vel_mmps.append(None if v is None else float(v) * k)

                bus_v = state.get_bus_voltage() or []
                bus_i = state.get_bus_current() or []
                motor_i = state.get_motor_current() or []
                temp_fet = state.get_temp_fet() or []
                temp_motor = state.get_temp_motor() or []
                axis_state = state.get_axis_state() or []
                axis_error = state.get_axis_error() or []
                hand_cmd_t_mm, hand_cmd_q, _, _ = state.get_hand_motion()
                hand_cmd_roll, hand_cmd_pitch, hand_cmd_yaw = quat_to_rpy_rad(hand_cmd_q)
                hand_est_t_mm, hand_est_q, hand_est_v_mps, hand_est_w_rps = state.get_hand_estimate()
                hand_est_roll, hand_est_pitch, hand_est_yaw = quat_to_rpy_rad(hand_est_q)
                msg = {
                    "t": time.time(),
                    "pos": fb_pos_mm,
                    "vel": fb_vel_mmps,
                    "bus_v": [None if v is None else float(v) for v in bus_v],
                    "bus_i": [None if i is None else float(i) for i in bus_i],
                    "motor_i": [None if x is None else float(x) for x in motor_i],
                    "temp_fet": [None if x is None else float(x) for x in temp_fet],
                    "temp_motor": [None if x is None else float(x) for x in temp_motor],
                    "axis_state": [None if x is None else int(x) for x in axis_state],
                    "axis_error": [None if x is None else int(x) for x in axis_error],
                    "hand_cmd_pose": [
                        float(hand_cmd_t_mm[0]),
                        float(hand_cmd_t_mm[1]),
                        float(hand_cmd_t_mm[2]),
                        math.degrees(float(hand_cmd_roll)),
                        math.degrees(float(hand_cmd_pitch)),
                        math.degrees(float(hand_cmd_yaw)),
                    ],
                    "hand_est_pose": [
                        float(hand_est_t_mm[0]),
                        float(hand_est_t_mm[1]),
                        float(hand_est_t_mm[2]),
                        math.degrees(float(hand_est_roll)),
                        math.degrees(float(hand_est_pitch)),
                        math.degrees(float(hand_est_yaw)),
                    ],
                    "hand_est_vel": [
                        float(hand_est_v_mps[0]),
                        float(hand_est_v_mps[1]),
                        float(hand_est_v_mps[2]),
                        math.degrees(float(hand_est_w_rps[0])),
                        math.degrees(float(hand_est_w_rps[1])),
                        math.degrees(float(hand_est_w_rps[2])),
                    ],
                }
                udp_sock.sendto(json.dumps(msg).encode("utf-8"), controller_addr)
        except Exception as e:
            logger.error(f"[UDP] Error sending telemetry: {e}")
        time.sleep(1.0 / TELEMETRY_RATE_HZ)



def axes_state_logger(state: RobotState):
    while True:
        try:
            pos = state.get_pos_fbk()
            vel = state.get_vel_fbk()
            bus = state.get_bus_voltage()
            busi = state.get_bus_current()
            temp_f = state.get_temp_fet()
            temp_m = state.get_temp_motor()
            st = state.get_state()

            fmt_pos = ", ".join("---" if x is None else f"{x:.3f}" for x in pos)
            fmt_vel = ", ".join("---" if v is None else f"{v:.3f}" for v in vel)
            fmt_bus = ", ".join("---" if b is None else f"{b:.2f}" for b in bus)
            fmt_busi = ", ".join("---" if i is None else f"{i:.2f}" for i in busi)
            fmt_tf = ", ".join("---" if x is None else f"{x:.1f}" for x in temp_f)
            fmt_tm = ", ".join("---" if x is None else f"{x:.1f}" for x in temp_m)

            logger.info(
                f"[LOG] State={st} "
                f"Pos=[{fmt_pos}] "
                f"Vel=[{fmt_vel}] "
                f"BusV=[{fmt_bus}]"
                f"BusI=[{fmt_busi}]"
                f"TempFET=[{fmt_tf}] "
                f"TempMotor=[{fmt_tm}]"
            )
        except Exception as e:
            logger.error(f"[LOG] Error: {e}")
        time.sleep(1.0)


def tcp_command_server(state: RobotState):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("0.0.0.0", TCP_CMD_PORT))
    except OSError as e:
        logger.error(f"[TCP] Bind failed: {e}")
        return
    srv.listen(1)
    logger.info(f"[TCP] Listening on :{TCP_CMD_PORT}")
    while True:
        conn, addr = srv.accept()
        state.set_controller_ip(addr[0])  # <-- save controller IP
        logger.info(f"[TCP] Controller connected from {addr}")
        state.set_controller_ip(addr[0])

        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        controller_addr = (addr[0], UDP_TELEM_PORT)
        state.start_telem(udp_sock, controller_addr)

        try:
            with conn, conn.makefile("r") as f:
                for line in f:
                    try:
                        msg = json.loads(line.strip())
                        mtype = msg.get("type")
                        if mtype == "axes":
                            pos_mm = _coerce_vec6_to_mm(msg, "positions")
                            state.set_axes(pos_mm)
                        elif mtype == "state":
                            state.set_state(msg.get("value", "disable"))
                        elif mtype == "pretension":
                            upper = float(msg.get("upper_N", 0.0))
                            lower = float(msg.get("lower_N", 0.0))
                            state.request_pretension(upper, lower)
                        elif mtype == "task_gain_mult":
                            state.request_task_gain_multipliers(
                                kp_xyz=msg.get("kp_xyz"),
                                kp_rp=msg.get("kp_rp"),
                                kd_xyz=msg.get("kd_xyz"),
                                kd_rp=msg.get("kd_rp"),
                            )
                        elif mtype == "home":
                            home_mm = _coerce_vec6_to_mm(msg, "home_pos")
                            state.request_home(home_mm)
                        elif mtype == "pose":
                            x = float(msg.get("x_mm", 0.0))
                            y = float(msg.get("y_mm", 0.0))
                            z = float(msg.get("z_mm", 0.0))
                            roll = float(msg.get("roll_deg", 0.0))
                            pitch = float(msg.get("pitch_deg", 0.0))

                            q = quat_from_rpy_deg(roll, pitch, 0.0)  # yaw assumed 0
                            state.set_hand_pose((x, y, z), q)
                        elif mtype == "profile_upload":
                            profile = msg.get("profile", [])
                            units = (msg.get("units") or "mm").lower()

                            if units == "mm":
                                profile_mm = profile
                            elif units == "turns":
                                # allow legacy profiles in turns
                                profile_mm = []
                                for row in profile:
                                    if not isinstance(row, (list, tuple)) or len(row) < 7:
                                        raise ValueError("each profile row must be [t, a1..a6]")
                                    t = float(row[0])
                                    axes_turns = [float(x) for x in row[1:7]]
                                    axes_mm = turns_to_mm(axes_turns)
                                    profile_mm.append([t] + axes_mm)
                            else:
                                raise ValueError(f"Unknown units '{units}' (expected 'mm' or 'turns')")

                            # Store profile in mm (RobotState / ProfilePlayer operate in mm)
                            state.set_profile(profile_mm)
                        elif mtype == "profile_start":
                            rate_hz = float(msg.get("rate_hz", 100.0))
                            state.start_profile(rate_hz)
                        elif mtype == "pose_profile_upload":
                            # expected rows:
                            #  [t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
                            # or full feedforward rows:
                            #  [t, x_mm, y_mm, z_mm, vx_mps, vy_mps, vz_mps,
                            #   ax_mps2, ay_mps2, az_mps2, roll_deg, pitch_deg, yaw_deg]
                            profile = msg.get("profile", [])
                            state.set_pose_profile(profile)
                        elif mtype == "pose_profile_start":
                            rate_hz = float(msg.get("rate_hz", 100.0))
                            state.start_pose_profile(rate_hz)
                        elif mtype == "pose_profile_run":
                            profile = msg.get("profile", [])
                            rate_hz = float(msg.get("rate_hz", 100.0))
                            state.set_pose_profile(profile)
                            state.start_pose_profile(rate_hz)
                        elif mtype == "profile_stop":
                            state.stop_profile()
                        else:
                            logger.warning(f"[TCP] Unknown command: {mtype}")
                    except Exception as e:
                        logger.error(f"[TCP] Bad command: {e}")
        except Exception as e:
            logger.error(f"[TCP] Connection error: {e}")
        finally:
            state.stop_profile()
            state.stop_telem()
            logger.info("[TCP] Controller disconnected")


if __name__ == "__main__":
    state = RobotState()
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
