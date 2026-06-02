import sys
import threading
import math
import time
import random
from queue import Queue
import socket
import json
import os
import csv
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton,
    QLabel, QSlider, QHBoxLayout, QDoubleSpinBox, QComboBox,
    QTableWidget, QTableWidgetItem, QSizePolicy, QGridLayout
)
from PyQt6.QtCore import Qt, QTimer

import pyqtgraph as pg

try:
    from jugglebot.planning import load_profile_yaml, build_path_from_profile
except Exception:
    load_profile_yaml = None
    build_path_from_profile = None

# Networking configuration
ROBOT_HOST = "jugglepi.local"  # <-- set to your Raspberry Pi IP or hostname
TCP_CMD_PORT = 5555
UDP_TELEM_PORT = 5556
UDP_RECV_BUFFER_BYTES = 65535


def _float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _float_or_nan(value):
    val = _float_or_none(value)
    return float("nan") if val is None else val


def _vec_get(values, index, default=None):
    try:
        return values[index]
    except Exception:
        return default


def _pose_state_to_legacy_pose(pose_state):
    if not isinstance(pose_state, dict):
        return []
    pos_m = pose_state.get("position_m") or []
    rpy_rad = pose_state.get("orientation_rpy_rad") or []
    return [
        1000.0 * _float_or_nan(_vec_get(pos_m, 0, float("nan"))),
        1000.0 * _float_or_nan(_vec_get(pos_m, 1, float("nan"))),
        1000.0 * _float_or_nan(_vec_get(pos_m, 2, float("nan"))),
        math.degrees(_float_or_nan(_vec_get(rpy_rad, 0, float("nan")))),
        math.degrees(_float_or_nan(_vec_get(rpy_rad, 1, float("nan")))),
        math.degrees(_float_or_nan(_vec_get(rpy_rad, 2, float("nan")))),
    ]


def _pose_state_to_legacy_velocity(pose_state):
    if not isinstance(pose_state, dict):
        return []
    lin = pose_state.get("linear_velocity_mps") or []
    ang = pose_state.get("angular_velocity_rps") or []
    return [
        _float_or_nan(_vec_get(lin, 0, float("nan"))),
        _float_or_nan(_vec_get(lin, 1, float("nan"))),
        _float_or_nan(_vec_get(lin, 2, float("nan"))),
        math.degrees(_float_or_nan(_vec_get(ang, 0, float("nan")))),
        math.degrees(_float_or_nan(_vec_get(ang, 1, float("nan")))),
        math.degrees(_float_or_nan(_vec_get(ang, 2, float("nan")))),
    ]


def _actuator_field_vec(actuators, field_name):
    values = []
    if not isinstance(actuators, list):
        actuators = []
    for i in range(6):
        axis = actuators[i] if i < len(actuators) and isinstance(actuators[i], dict) else {}
        values.append(axis.get(field_name))
    return values


def _normalize_telemetry(telem):
    if not isinstance(telem, dict):
        return {}
    if "pos" in telem or "vel" in telem or "hand_est_pose" in telem:
        return telem

    actuators = telem.get("actuators") or []
    bus_stats = telem.get("bus_stats") or {}
    normalized = {
        "t": telem.get("timestamp_s", time.time()),
        "pos": [1000.0 * _float_or_nan(v) for v in (telem.get("cable_lengths_m") or [])],
        "vel": [1000.0 * _float_or_nan(v) for v in (telem.get("cable_velocities_mps") or [])],
        "temp_fet": _actuator_field_vec(actuators, "temperature_fet_c"),
        "temp_motor": _actuator_field_vec(actuators, "temperature_motor_c"),
        "bus_i": _actuator_field_vec(actuators, "bus_current_a"),
        "motor_i": _actuator_field_vec(actuators, "current_estimate_a"),
        "torque_cmd_nm": list(telem.get("commanded_torques_nm") or []),
        "torque_rsp_nm": list(telem.get("estimated_torques_nm") or []),
        "tension_cmd_n": list(telem.get("commanded_tensions_n") or []),
        "tension_rsp_n": list(telem.get("estimated_tensions_n") or []),
        "bus_v": _actuator_field_vec(actuators, "bus_voltage_v"),
        "axis_state": _actuator_field_vec(actuators, "axis_state"),
        "axis_error": _actuator_field_vec(actuators, "error_flags"),
        "hand_cmd_pose": _pose_state_to_legacy_pose(telem.get("commanded_pose")),
        "hand_est_pose": _pose_state_to_legacy_pose(telem.get("estimated_pose")),
        "hand_est_vel": _pose_state_to_legacy_velocity(telem.get("estimated_pose")),
        "can_rx_hz": bus_stats.get("can_rx_hz"),
        "can_tx_hz": bus_stats.get("can_tx_hz"),
        "can_msg_hz": bus_stats.get("can_msg_hz"),
        "can_util_est": bus_stats.get("can_util_est"),
        "pos_fbk_hz": bus_stats.get("pos_fbk_hz"),
        "pos_fbk_period0_min_s": bus_stats.get("pos_fbk_period0_min_s"),
        "pos_fbk_period0_max_s": bus_stats.get("pos_fbk_period0_max_s"),
        "control_state": telem.get("control_state"),
        "profile_active": telem.get("profile_active"),
    }
    return normalized

def _queue_put_latest(q: Queue, item):
    """Keep only the newest item in the queue."""
    try:
        while True:
            q.get_nowait()
    except Exception:
        pass
    q.put(item)

# --- ODrive AxisState decode (int -> name) --- putting in as class constant now, this should move out when refactored
AXIS_STATE_NAMES = {
    0: "UNDEFINED",
    1: "IDLE",
    2: "STARTUP_SEQUENCE",
    3: "FULL_CALIBRATION_SEQUENCE",
    4: "MOTOR_CALIBRATION",
    6: "ENCODER_INDEX_SEARCH",
    7: "ENCODER_OFFSET_CALIBRATION",
    8: "CLOSED_LOOP_CONTROL",
    9: "LOCKIN_SPIN",
    10: "ENCODER_DIR_FIND",
    11: "HOMING",
    12: "ENCODER_HALL_POLARITY_CALIBRATION",
    13: "ENCODER_HALL_PHASE_CALIBRATION",
    14: "ANTICOGGING_CALIBRATION",
    15: "HARMONIC_CALIBRATION",
    16: "HARMONIC_CALIBRATION_COMMUTATION",
}

def _axis_state_text(state_code):
    if state_code is None:
        return "---"
    try:
        sc = int(state_code)
    except Exception:
        return "---"
    return AXIS_STATE_NAMES.get(sc, f"STATE_{sc}")

# --- ODrive Error decode (bitmask -> names) ---
ODRIVE_ERROR_BITS = {
    0x00000001: "INITIALIZING",
    0x00000002: "SYSTEM_LEVEL",
    0x00000004: "TIMING_ERROR",
    0x00000008: "MISSING_ESTIMATE",
    0x00000010: "BAD_CONFIG",
    0x00000020: "DRV_FAULT",
    0x00000040: "MISSING_INPUT",
    0x00000100: "DC_BUS_OVER_VOLTAGE",
    0x00000200: "DC_BUS_UNDER_VOLTAGE",
    0x00000400: "DC_BUS_OVER_CURRENT",
    0x00000800: "DC_BUS_OVER_REGEN_CURRENT",
    0x00001000: "CURRENT_LIMIT_VIOLATION",
    0x00002000: "MOTOR_OVER_TEMP",
    0x00004000: "INVERTER_OVER_TEMP",
    0x00008000: "VELOCITY_LIMIT_VIOLATION",
    0x00010000: "POSITION_LIMIT_VIOLATION",
    0x01000000: "WATCHDOG_TIMER_EXPIRED",
    0x02000000: "ESTOP_REQUESTED",
    0x04000000: "SPINOUT_DETECTED",
    0x08000000: "BRAKE_RESISTOR_DISARMED",
    0x10000000: "THERMISTOR_DISCONNECTED",
    0x40000000: "CALIBRATION_ERROR",
}

def _decode_odrive_error_mask(err_code):
    """
    Returns:
      short_text: what to show in the table cell
      tooltip: full breakdown (hex + list)
    """
    if err_code is None:
        return "---", "No error data"
    try:
        code = int(err_code)
    except Exception:
        return "---", "Invalid error value"

    if code == 0:
        return "OK", "0x00000000 (no errors)"

    names = []
    # stable ordering by bit value
    for bit in sorted(ODRIVE_ERROR_BITS.keys()):
        if code & bit:
            names.append(ODRIVE_ERROR_BITS[bit])

    # Unknown bits (future firmware, etc.)
    known_mask = 0
    for bit in ODRIVE_ERROR_BITS.keys():
        known_mask |= bit
    unknown = code & (~known_mask)
    if unknown:
        names.append(f"UNKNOWN_BITS:0x{unknown:08X}")

    # Short cell text: single name, or MULTI(n)
    if len(names) == 1:
        short = names[0]
    else:
        short = f"MULTI({len(names)})"

    tooltip = " | ".join([f"0x{code:08X}"] + names)
    return short, tooltip

class CommandClient(threading.Thread):
    """TCP client that reliably sends commands to the robot with auto-reconnect."""
    def __init__(self, host, port, cmd_queue: Queue, status_cb=None):
        super().__init__(daemon=True)
        self.host = host
        self.port = port
        self.cmd_queue = cmd_queue
        self.status_cb = status_cb
        self._stop = threading.Event()
        self._sock = None

    def run(self):
        last_cmd = None
        while not self._stop.is_set():
            try:
                # Connect (block/retry)
                if self.status_cb:
                    self.status_cb("Connecting to robot (TCP)...")
                self._sock = socket.create_connection((self.host, self.port), timeout=5)
                self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if self.status_cb:
                    self.status_cb("Connected (TCP)")
                # If we already have a last command, send it after reconnect
                if last_cmd is not None:
                    self._send_cmd(last_cmd)

                # Main send loop
                while not self._stop.is_set():
                    cmd = self.cmd_queue.get()  # blocks until new command
                    last_cmd = cmd
                    self._send_cmd(cmd)
            except Exception as e:
                if self.status_cb:
                    self.status_cb(f"TCP disconnected: {e}. Reconnecting in 1s...")
                self._close()
                time.sleep(1)

        self._close()

    def _send_cmd(self, cmd_value):
        if not self._sock:
            return
        # Only accept dict commands (axes/state). Ignore non-dicts.
        if not isinstance(cmd_value, dict):
            return
        msg = json.dumps(cmd_value) + "\n"
        self._sock.sendall(msg.encode("utf-8"))

    def stop(self):
        self._stop.set()
        self._close()

    def _close(self):
        try:
            if self._sock:
                self._sock.close()
        except Exception:
            pass
        self._sock = None


def telemetry_listener(udp_port: int, telem_queue: Queue, status_cb=None):
    """Listen for UDP telemetry and push dict into telem_queue."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Reuse addr for quick restarts
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", udp_port))
    if status_cb:
        status_cb(f"Telemetry: listening UDP :{udp_port}")
    while True:
        try:
            data, _ = sock.recvfrom(UDP_RECV_BUFFER_BYTES)
            # New format: {"t": <unix_time>, "pos": [6], "vel": [6]}
            # Legacy: {"t": <unix_time>, "val": <float>}
            telem = json.loads(data.decode("utf-8"))
            telem_queue.put(telem)
        except Exception as e:
            if status_cb:
                status_cb(f"Telemetry error: {e}")
            # brief pause to avoid tight loop on persistent error
            time.sleep(0.05)


class RobotGUI(QWidget):
    def __init__(self, cmd_queue, telem_queue):
        super().__init__()
        self.setWindowTitle("Robot Controller + Telemetry")
        self.resize(800, 600)

        self.cmd_queue = cmd_queue
        self.telem_queue = telem_queue

        # ---- Consistent per-axis color coding (A1..A6) ----
        # Use a fixed palette so colors stay consistent across plots
        self.axis_colors = [
            (255, 80, 80),  # A1 - red
            (80, 200, 120),  # A2 - green
            (80, 160, 255),  # A3 - blue
            (255, 200, 80),  # A4 - amber
            (190, 120, 255),  # A5 - purple
            (80, 220, 220),  # A6 - cyan
        ]

        def axis_pen(i: int, width: int = 2, style=Qt.PenStyle.SolidLine):
            return pg.mkPen(color=self.axis_colors[i], width=width, style=style)

        # History length (number of samples kept in plots)
        self.history_seconds = 20.0

        # --- Layout ---
        layout = QVBoxLayout()

        # Status label
        self.status_label = QLabel("Telemetry: waiting...")
        layout.addWidget(self.status_label)
        self.comm_stats_label = QLabel(
            "Comm: CAN rx=--- Hz tx=--- Hz total=--- Hz util=---% pos_fbk(avg/axis)=--- Hz p0[min,max]=---/--- ms"
        )
        layout.addWidget(self.comm_stats_label)

        # ---- Pose Command (global) ----
        layout.addWidget(QLabel("Pose Command (Global): X/Y/Z (mm), Roll/Pitch (deg). Yaw assumed 0."))

        hand_row = QHBoxLayout()

        def _mk_spin(label, lo, hi, dec, step, suffix):
            col = QVBoxLayout()
            col.addWidget(QLabel(label))
            sp = QDoubleSpinBox()
            sp.setRange(lo, hi)
            sp.setDecimals(dec)
            sp.setSingleStep(step)
            sp.setSuffix(suffix)
            col.addWidget(sp)
            hand_row.addLayout(col)
            return sp

        self.hand_x = _mk_spin("X", -500.0, 500.0, 2, 1.0, " mm")
        self.hand_y = _mk_spin("Y", -500.0, 500.0, 2, 1.0, " mm")
        self.hand_z = _mk_spin("Z", -500.0, 500.0, 2, 1.0, " mm")
        self.hand_roll = _mk_spin("Roll", -45.0, 45.0, 2, 1.0, " deg")
        self.hand_pitch = _mk_spin("Pitch", -45.0, 45.0, 2, 1.0, " deg")

        layout.addLayout(hand_row)

        btns = QHBoxLayout()
        self.btn_hand_send = QPushButton("Send Pose")
        self.btn_hand_send.clicked.connect(self.send_pose)
        btns.addWidget(self.btn_hand_send)

        btns.addStretch(1)
        layout.addLayout(btns)

        self.hand_est_label = QLabel("Pose Estimate: X=--- mm  Y=--- mm  Z=--- mm  Roll=--- deg  Pitch=--- deg")
        layout.addWidget(self.hand_est_label)

        # Pose command/response plot with dual Y axes:
        # left axis in mm (translation), right axis in deg (roll/pitch).
        self.plot_pose = pg.PlotWidget(title="Hand Pose Cmd/Rsp")
        self.plot_pose.setLabel("bottom", "Time", "s")
        self.plot_pose.setLabel("left", "Translation", "mm")
        self.plot_pose.showGrid(x=True, y=True)
        self.plot_pose.addLegend()
        self.plot_pose.showAxis("right")
        self.plot_pose.getAxis("right").setLabel("Rotation", "deg")
        self.pose_rot_vb = pg.ViewBox()
        self.plot_pose.scene().addItem(self.pose_rot_vb)
        self.plot_pose.getAxis("right").linkToView(self.pose_rot_vb)
        self.pose_rot_vb.setXLink(self.plot_pose)
        self.pose_rot_vb.enableAutoRange(axis=pg.ViewBox.YAxis, enable=True)

        def _sync_pose_views():
            vb_left = self.plot_pose.getPlotItem().vb
            self.pose_rot_vb.setGeometry(vb_left.sceneBoundingRect())
            self.pose_rot_vb.linkedViewChanged(vb_left, self.pose_rot_vb.XAxis)

        self.plot_pose.getPlotItem().vb.sigResized.connect(_sync_pose_views)
        _sync_pose_views()

        trans_colors = [
            pg.mkPen((255, 80, 80), width=2),
            pg.mkPen((80, 200, 120), width=2),
            pg.mkPen((80, 160, 255), width=2),
        ]
        self.pose_cmd_xyz_curves = []
        self.pose_rsp_xyz_curves = []
        for i, name in enumerate(("X", "Y", "Z")):
            self.pose_cmd_xyz_curves.append(self.plot_pose.plot(name=f"{name} cmd [mm]", pen=trans_colors[i]))
            self.pose_rsp_xyz_curves.append(
                self.plot_pose.plot(name=f"{name} rsp [mm]", pen=pg.mkPen(trans_colors[i].color(), width=2, style=Qt.PenStyle.DashLine))
            )

        rot_colors = [
            pg.mkPen((255, 200, 80), width=2),
            pg.mkPen((190, 120, 255), width=2),
        ]
        self.pose_cmd_rp_curves = []
        self.pose_rsp_rp_curves = []
        for i, name in enumerate(("Roll", "Pitch")):
            cmd_curve = pg.PlotCurveItem(name=f"{name} cmd [deg]", pen=rot_colors[i])
            rsp_curve = pg.PlotCurveItem(
                name=f"{name} rsp [deg]",
                pen=pg.mkPen(rot_colors[i].color(), width=2, style=Qt.PenStyle.DashLine),
            )
            self.pose_rot_vb.addItem(cmd_curve)
            self.pose_rot_vb.addItem(rsp_curve)
            self.plot_pose.plotItem.legend.addItem(cmd_curve, f"{name} cmd [deg]")
            self.plot_pose.plotItem.legend.addItem(rsp_curve, f"{name} rsp [deg]")
            self.pose_cmd_rp_curves.append(cmd_curve)
            self.pose_rsp_rp_curves.append(rsp_curve)

        # ---- Home position inputs + button ----
        home_layout = QVBoxLayout()

        home_layout.addWidget(QLabel("Home Position (mm)"))

        self.home_spins = []
        row = QHBoxLayout()
        for i in range(6):
            col = QVBoxLayout()
            col.addWidget(QLabel(f"A{i+1}"))
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.01)
            spin.setValue(0.0)  # <-- set your defaults here if you want
            col.addWidget(spin)
            row.addLayout(col)
            self.home_spins.append(spin)
        home_layout.addLayout(row)

        btn_row = QHBoxLayout()
        self.btn_home = QPushButton("HOME")
        self.btn_home.clicked.connect(self.send_home)
        btn_row.addWidget(self.btn_home)
        btn_row.addStretch(1)
        home_layout.addLayout(btn_row)

        layout.addLayout(home_layout)

        # ---- Pretension inputs ----
        pret_layout = QHBoxLayout()

        pret_layout.addWidget(QLabel("Pretension Upper (A1,A3,A5) [N]:"))
        self.pret_upper_spin = QDoubleSpinBox()
        self.pret_upper_spin.setDecimals(2)
        self.pret_upper_spin.setRange(0.0, 500.0)
        self.pret_upper_spin.setSingleStep(1.0)
        self.pret_upper_spin.setValue(3.0)
        pret_layout.addWidget(self.pret_upper_spin)

        pret_layout.addWidget(QLabel("Pretension Lower (A2,A4,A6) [N]:"))
        self.pret_lower_spin = QDoubleSpinBox()
        self.pret_lower_spin.setDecimals(2)
        self.pret_lower_spin.setRange(0.0, 500.0)
        self.pret_lower_spin.setSingleStep(1.0)
        self.pret_lower_spin.setValue(3.0)
        pret_layout.addWidget(self.pret_lower_spin)

        self.btn_pretension = QPushButton("PRETENSION")
        self.btn_pretension.clicked.connect(self.send_pretension)
        pret_layout.addWidget(self.btn_pretension)

        layout.addLayout(pret_layout)

        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Spool Gain Multipliers:"))

        self.spool_kp_mult_spin = QDoubleSpinBox()
        self.spool_kp_mult_spin.setDecimals(2)
        self.spool_kp_mult_spin.setRange(0.0, 20.0)
        self.spool_kp_mult_spin.setSingleStep(0.1)
        self.spool_kp_mult_spin.setValue(1.0)
        self.spool_kp_mult_spin.setPrefix("Spool Kp x")
        gain_layout.addWidget(self.spool_kp_mult_spin)

        self.spool_kd_mult_spin = QDoubleSpinBox()
        self.spool_kd_mult_spin.setDecimals(2)
        self.spool_kd_mult_spin.setRange(0.0, 20.0)
        self.spool_kd_mult_spin.setSingleStep(0.1)
        self.spool_kd_mult_spin.setValue(1.0)
        self.spool_kd_mult_spin.setPrefix("Spool Kd x")
        gain_layout.addWidget(self.spool_kd_mult_spin)

        self.btn_apply_gain_mult = QPushButton("Apply Gains")
        self.btn_apply_gain_mult.clicked.connect(self.send_spool_gain_mult)
        gain_layout.addWidget(self.btn_apply_gain_mult)

        layout.addLayout(gain_layout)

        # State controls
        state_layout = QHBoxLayout()
        self.btn_enable = QPushButton("Enable")
        self.btn_disable = QPushButton("Disable")
        self.btn_estop = QPushButton("ESTOP")
        self.btn_enable.clicked.connect(lambda: self.send_state("enable"))
        self.btn_disable.clicked.connect(lambda: self.send_state("disable"))
        self.btn_estop.clicked.connect(lambda: self.send_state("estop"))
        state_layout.addWidget(self.btn_enable)
        state_layout.addWidget(self.btn_disable)
        state_layout.addWidget(self.btn_estop)
        layout.addLayout(state_layout)

        # Profile controls
        prof_layout = QHBoxLayout()
        self.profile_combo = QComboBox()
        self.profile_refresh_btn = QPushButton("Refresh")
        self.profile_send_btn = QPushButton("Send Profile")
        self.profile_rate = QDoubleSpinBox()
        self.profile_rate.setDecimals(1)
        self.profile_rate.setRange(1.0, 1000.0)
        self.profile_rate.setSingleStep(10.0)
        self.profile_rate.setValue(100.0)
        self.profile_start_btn = QPushButton("Start Profile")
        self.profile_refresh_btn.clicked.connect(self.populate_profile_dropdown)
        self.profile_send_btn.clicked.connect(self.on_send_profile)
        self.profile_start_btn.clicked.connect(self.on_start_profile)
        self.profile_type_combo = QComboBox()
        self.profile_type_combo.addItems(["Axis Profile (mm)", "Pose Profile (XYZ mm, RPY deg)"])
        prof_layout.addWidget(QLabel("Profile CSV:"))
        prof_layout.addWidget(self.profile_combo, 1)
        prof_layout.addWidget(self.profile_type_combo)  # add selector next to the CSV
        prof_layout.addWidget(self.profile_refresh_btn)
        prof_layout.addWidget(self.profile_send_btn)
        prof_layout.addWidget(QLabel("Rate (Hz):"))
        prof_layout.addWidget(self.profile_rate)
        prof_layout.addWidget(self.profile_start_btn)
        layout.addLayout(prof_layout)

        jp_layout = QHBoxLayout()
        jp_layout.addWidget(QLabel("JugglePath:"))
        self.jp_profile_combo = QComboBox()
        self.jp_refresh_btn = QPushButton("Refresh JP")
        self.jp_send_start_btn = QPushButton("Send + Start JugglePath")
        self.jp_refresh_btn.clicked.connect(self.populate_jugglepath_dropdown)
        self.jp_send_start_btn.clicked.connect(self.on_send_start_jugglepath)
        jp_layout.addWidget(self.jp_profile_combo, 1)
        jp_layout.addWidget(self.jp_refresh_btn)
        jp_layout.addWidget(self.jp_send_start_btn)
        layout.addLayout(jp_layout)

        # ---- Axis numeric matrix (6 rows x columns) ----
        self.axis_table_cols = [
            "State",
            "Error",
            "Pos (mm)",
            "Vel (mm/s)",
            "Motor I (A)",
            "Bus V (V)",
            "Bus I (A)",
            "Temp Motor (°C)",
            "Temp FET (°C)",
        ]

        self.axis_table = QTableWidget(6, len(self.axis_table_cols))
        self.axis_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        # Make the table tall enough to show all 6 rows + header
        row_h = self.axis_table.verticalHeader().defaultSectionSize()
        hdr_h = self.axis_table.horizontalHeader().height()
        frame = 2 * self.axis_table.frameWidth()
        self.axis_table.setMinimumHeight(hdr_h + 6 * row_h + frame + 8)

        self.axis_table.setHorizontalHeaderLabels(self.axis_table_cols)
        self.axis_table.setVerticalHeaderLabels([f"A{i+1}" for i in range(6)])
        self.axis_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.axis_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)

        self.axis_table.resizeColumnsToContents()
        # Reserve space for text
        self.axis_table.setColumnWidth(0, 150)  # State
        self.axis_table.setColumnWidth(1, 150)  # Error

        layout.addWidget(QLabel("Axis Data"))
        layout.addWidget(self.axis_table)


        # --- Telemetry plots ---
        plots_grid = QGridLayout()

        # Position plot (6 traces)
        self.plot_pos = pg.PlotWidget(title="Position (mm) — A1..A6")
        self.plot_pos.setLabel("bottom", "Time", "s")
        self.plot_pos.setLabel("left", "Position", "mm")
        self.plot_pos.showGrid(x=True, y=True)
        self.plot_pos.addLegend()
        self.curves_pos = []
        for i in range(6):
            c = self.plot_pos.plot(name=f"A{i + 1}", pen=axis_pen(i, width=2))
            self.curves_pos.append(c)
        plots_grid.addWidget(self.plot_pos, 0, 0)

        # Velocity plot (6 traces)
        self.plot_vel = pg.PlotWidget(title="Velocity (mm/s) — A1..A6")
        self.plot_vel.setLabel("bottom", "Time", "s")
        self.plot_vel.setLabel("left", "Velocity", "mm/s")
        self.plot_vel.showGrid(x=True, y=True)
        self.plot_vel.addLegend()
        self.curves_vel = []
        for i in range(6):
            c = self.plot_vel.plot(name=f"A{i + 1}", pen=axis_pen(i, width=2))
            self.curves_vel.append(c)
        plots_grid.addWidget(self.plot_vel, 0, 1)

        # Tension plot (12 traces: command + response for each axis)
        self.plot_tension = pg.PlotWidget(title="Cable Tension (N) — Command + Response (A1..A6)")
        self.plot_tension.setLabel("bottom", "Time", "s")
        self.plot_tension.setLabel("left", "Tension", "N")
        self.plot_tension.showGrid(x=True, y=True)
        self.plot_tension.addLegend()
        self.curves_tension_cmd = []
        self.curves_tension_rsp = []
        for i in range(6):
            self.curves_tension_cmd.append(
                self.plot_tension.plot(name=f"A{i + 1} Cmd", pen=axis_pen(i, width=2, style=Qt.PenStyle.SolidLine))
            )
            self.curves_tension_rsp.append(
                self.plot_tension.plot(name=f"A{i + 1} Rsp", pen=axis_pen(i, width=2, style=Qt.PenStyle.DashLine))
            )
        plots_grid.addWidget(self.plot_tension, 0, 2)

        # Temperature plot (12 traces: motor+fet for each axis)
        self.plot_temp = pg.PlotWidget(title="Temperatures (°C) — Motor + FET (A1..A6)")
        self.plot_temp.setLabel("bottom", "Time", "s")
        self.plot_temp.setLabel("left", "Temp", "°C")
        self.plot_temp.showGrid(x=True, y=True)
        self.plot_temp.addLegend()
        self.curves_temp_motor = []
        self.curves_temp_fet = []
        for i in range(6):
            self.curves_temp_motor.append(
                self.plot_temp.plot(name=f"A{i + 1} Motor", pen=axis_pen(i, width=2, style=Qt.PenStyle.SolidLine))
            )
            self.curves_temp_fet.append(
                self.plot_temp.plot(name=f"A{i + 1} FET", pen=axis_pen(i, width=2, style=Qt.PenStyle.DashLine))
            )
        plots_grid.addWidget(self.plot_temp, 1, 0)

        # Bus Voltage plot
        self.busv_plot = pg.PlotWidget(title="Bus Voltage (V)")
        self.busv_plot.setLabel('bottom', 'Time', 's')
        self.busv_plot.setLabel('left', 'Voltage', 'V')
        self.busv_plot.showGrid(x=True, y=True)
        self.busv_plot.addLegend()
        self.curves_busv = []
        for i in range(6):
            self.curves_busv.append(self.busv_plot.plot(name=f"A{i + 1}", pen=axis_pen(i, width=2)))
        plots_grid.addWidget(self.busv_plot, 1, 1)

        # Torque plot (12 traces: command + response for each axis)
        self.plot_torque = pg.PlotWidget(title="Torque (Nm) — Command + Response (A1..A6)")
        self.plot_torque.setLabel("bottom", "Time", "s")
        self.plot_torque.setLabel("left", "Torque", "Nm")
        self.plot_torque.showGrid(x=True, y=True)
        self.plot_torque.addLegend()
        self.curves_torque_cmd = []
        self.curves_torque_rsp = []
        for i in range(6):
            self.curves_torque_cmd.append(
                self.plot_torque.plot(name=f"A{i + 1} Cmd", pen=axis_pen(i, width=2, style=Qt.PenStyle.SolidLine))
            )
            self.curves_torque_rsp.append(
                self.plot_torque.plot(name=f"A{i + 1} Rsp", pen=axis_pen(i, width=2, style=Qt.PenStyle.DashLine))
            )
        plots_grid.addWidget(self.plot_torque, 1, 2)

        plots_grid.addWidget(self.plot_pose, 2, 0, 1, 3)
        layout.addLayout(plots_grid)


        self.setLayout(layout)

        # Data buffers
        self.tbuf = []

        self.pos_buf = [[] for _ in range(6)]
        self.vel_buf = [[] for _ in range(6)]

        self.bus_v_buf = [[] for _ in range(6)]

        self.cur_motor_buf = [[] for _ in range(6)]
        self.cur_bus_buf = [[] for _ in range(6)]
        self.torque_cmd_buf = [[] for _ in range(6)]
        self.torque_rsp_buf = [[] for _ in range(6)]
        self.tension_cmd_buf = [[] for _ in range(6)]
        self.tension_rsp_buf = [[] for _ in range(6)]

        self.temp_motor_buf = [[] for _ in range(6)]
        self.temp_fet_buf = [[] for _ in range(6)]

        # --- Heartbeat / state / error buffers (ints) ---
        self.axis_state_buf = [[] for _ in range(6)]
        self.axis_error_buf = [[] for _ in range(6)]

        self.start_time = time.time()
        self.last_pos = [0.0]*6
        self.last_vel = [0.0]*6
        self.hand_cmd_pose = [float("nan")] * 6
        self.hand_est_pose = [float("nan")] * 6
        self.hand_est_vel = [float("nan")] * 6
        self.can_rx_hz = float("nan")
        self.can_tx_hz = float("nan")
        self.can_msg_hz = float("nan")
        self.can_util_est = float("nan")
        self.pos_fbk_hz = float("nan")
        self.pos_fbk_period0_min_s = float("nan")
        self.pos_fbk_period0_max_s = float("nan")
        self.hand_cmd_xyz_buf = [[] for _ in range(3)]
        self.hand_rsp_xyz_buf = [[] for _ in range(3)]
        self.hand_cmd_rp_buf = [[] for _ in range(2)]
        self.hand_rsp_rp_buf = [[] for _ in range(2)]
        self.last_telem_time = 0
        self.telem_timeout = 2.0  # seconds

        # Initialize profile dropdown
        self.populate_profile_dropdown()
        self.populate_jugglepath_dropdown()

        # Timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_gui)
        self.timer.start(100)  # update every 100 ms

        self.conn_timer = QTimer()
        self.conn_timer.timeout.connect(self.check_connection_status)
        self.conn_timer.start(500)  # check every 0.5 s

    def check_connection_status(self):
        now = time.time()
        if self.last_telem_time == 0 or (now - self.last_telem_time) > self.telem_timeout:
            self.status_label.setText("⚠️ Waiting for telemetry…")
        else:
            self.status_label.setText("✅ Connected (telemetry streaming)")

    def _profiles_dir(self) -> str:
        base = os.getcwd()
        pdir = os.path.join(base, "Profiles")
        os.makedirs(pdir, exist_ok=True)
        return pdir

    def _jugglepath_profiles_dir(self) -> Path:
        repo_root = Path(__file__).resolve().parents[1]
        pdir = repo_root / "src" / "jugglebot" / "profiles"
        if pdir.exists():
            return pdir
        return Path(os.getcwd()) / "src" / "jugglebot" / "profiles"

    def send_axes(self, *_):
        positions = [float(sp.value()) for sp in self.axis_spins]
        cmd = {"type": "axes", "positions": positions, "units": "mm"}
        _queue_put_latest(self.cmd_queue, cmd)

    def send_pose(self):
        cmd = {
            "type": "pose",
            "x_mm": float(self.hand_x.value()),
            "y_mm": float(self.hand_y.value()),
            "z_mm": float(self.hand_z.value()),
            "roll_deg": float(self.hand_roll.value()),
            "pitch_deg": float(self.hand_pitch.value()),
            # yaw is assumed 0 in the server
        }
        _queue_put_latest(self.cmd_queue, cmd)
        self.status_label.setText("Sent POSE command")

    def send_state(self, state_value: str):
        cmd = {"type": "state", "value": state_value}
        _queue_put_latest(self.cmd_queue, cmd)

    def send_home(self):
        positions = [float(sp.value()) for sp in self.home_spins]
        cmd = {"type": "home", "home_pos": positions, "units": "mm"}
        _queue_put_latest(self.cmd_queue, cmd)
        self.status_label.setText("HOME command sent")

    def send_pretension(self):
        upper = float(self.pret_upper_spin.value())
        lower = float(self.pret_lower_spin.value())
        cmd = {
            "type": "pretension",
            "upper_N": upper,
            "lower_N": lower,
        }
        _queue_put_latest(self.cmd_queue, cmd)
        self.status_label.setText(f"PRETENSION: upper={upper:.2f} N, lower={lower:.2f} N")

    def send_spool_gain_mult(self):
        kp = float(self.spool_kp_mult_spin.value())
        kd = float(self.spool_kd_mult_spin.value())
        cmd = {
            "type": "spool_gain_mult",
            "kp": kp,
            "kd": kd,
        }
        _queue_put_latest(self.cmd_queue, cmd)
        self.status_label.setText(
            f"SPOOL_GAIN sent: kp={kp:.2f}, kd={kd:.2f}"
        )

    def send_manual_move(self):
        positions = [float(sp.value()) for sp in self.manual_spins]
        cmd = {"type": "axes", "positions": positions, "units": "mm"}
        _queue_put_latest(self.cmd_queue, cmd)
        self.status_label.setText("MOVE command sent")

    def update_manual_to_current(self):
        # Prefer newest telemetry sample; fall back safely if missing
        for i in range(6):
            v = float("nan")
            if i < len(self.pos_buf) and self.pos_buf[i]:
                v = self.pos_buf[i][-1]

            if v == v:  # not NaN
                # block signals so we don't trigger any auto-send on valueChanged
                self.manual_spins[i].blockSignals(True)
                self.manual_spins[i].setValue(float(v))
                self.manual_spins[i].blockSignals(False)

        self.status_label.setText("Manual positions updated from current feedback")

    def _load_csv_as_profile(self, path: str):
        rows = []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = [r for r in reader if any(cell.strip() for cell in r)]
        if not rows:
            raise ValueError("empty CSV")
        start_idx = 0
        try:
            float(rows[0][0])
        except Exception:
            start_idx = 1
        profile_rows = []
        for r in rows[start_idx:]:
            if len(r) < 7:
                raise ValueError("each row must have at least 7 columns: time + 6 axes")
            t = float(r[0])
            axes = [float(x) for x in r[1:7]]
            profile_rows.append([t] + axes)
        times = [row[0] for row in profile_rows]
        if any(t2 < t1 for t1, t2 in zip(times, times[1:])):
            raise ValueError("time column must be non-decreasing")
        return profile_rows

    def _load_csv_as_pose_profile(self, path: str):
        """
        CSV rows: t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg
        """
        rows = []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            rows = [r for r in reader if any(cell.strip() for cell in r)]
        if not rows:
            raise ValueError("empty CSV")

        start_idx = 0
        try:
            float(rows[0][0])
        except Exception:
            start_idx = 1  # header row

        prof = []
        for r in rows[start_idx:]:
            if len(r) < 7:
                raise ValueError("each row must have at least 7 columns: time + x y z roll pitch yaw")
            t = float(r[0])
            vals = [float(x) for x in r[1:7]]
            prof.append([t] + vals)

        times = [row[0] for row in prof]
        if any(t2 < t1 for t1, t2 in zip(times, times[1:])):
            raise ValueError("time column must be non-decreasing")

        return prof

    def populate_profile_dropdown(self):
        pdir = self._profiles_dir()
        csvs = sorted([f for f in os.listdir(pdir) if f.lower().endswith(".csv")])
        self.profile_combo.clear()
        if not csvs:
            self.profile_combo.addItem("(no .csv files in Profiles/)")
            self.profile_combo.setEnabled(False)
        else:
            self.profile_combo.setEnabled(True)
            for f in csvs:
                self.profile_combo.addItem(f)

    def populate_jugglepath_dropdown(self):
        self.jp_profile_combo.clear()
        if load_profile_yaml is None or build_path_from_profile is None:
            self.jp_profile_combo.addItem("(jugglebot.planning unavailable)")
            self.jp_profile_combo.setEnabled(False)
            return
        pdir = self._jugglepath_profiles_dir()
        if not pdir.exists():
            self.jp_profile_combo.addItem("(no src/jugglebot/profiles found)")
            self.jp_profile_combo.setEnabled(False)
            return
        yamls = sorted(list(pdir.glob("*.yaml")))
        if not yamls:
            self.jp_profile_combo.addItem("(no .yaml profiles)")
            self.jp_profile_combo.setEnabled(False)
            return
        self.jp_profile_combo.setEnabled(True)
        for y in yamls:
            self.jp_profile_combo.addItem(y.name)

    def on_send_start_jugglepath(self):
        if not self.jp_profile_combo.isEnabled():
            self.status_label.setText("No JugglePath profile available")
            return
        name = self.jp_profile_combo.currentText()
        if not name or name.startswith("("):
            self.status_label.setText("Select a valid JugglePath profile")
            return
        if load_profile_yaml is None or build_path_from_profile is None:
            self.status_label.setText("jugglebot.planning import failed")
            return

        try:
            profile_path = self._jugglepath_profiles_dir() / name
            profile = load_profile_yaml(str(profile_path))
            rate = float(self.profile_rate.value())
            path, cmd_hz = build_path_from_profile(profile, command_rate_hz=rate)
            result = path.build()
            traj = result.traj
            if traj.shape[0] == 0:
                raise ValueError("generated trajectory is empty")

            rows = []
            # [t, x_mm, y_mm, z_mm, vx,vy,vz, ax,ay,az, roll,pitch,yaw]
            for r in traj:
                rows.append([
                    float(r[0]),
                    1000.0 * float(r[1]),
                    1000.0 * float(r[2]),
                    1000.0 * float(r[3]),
                    float(r[4]),
                    float(r[5]),
                    float(r[6]),
                    float(r[7]),
                    float(r[8]),
                    float(r[9]),
                    0.0,
                    0.0,
                    0.0,
                ])

            _queue_put_latest(
                self.cmd_queue,
                {"type": "pose_profile_run", "profile": rows},
            )
            self.status_label.setText(
                f"Sent+Started JugglePath {name}: {len(rows)} pts"
            )
        except Exception as e:
            self.status_label.setText(f"JugglePath send failed: {e}")

    def on_send_profile(self):
        if not self.profile_combo.isEnabled():
            self.status_label.setText("No CSV profile to send (Profiles/ empty)")
            return
        fname = self.profile_combo.currentText()
        if not fname or fname.startswith("("):
            self.status_label.setText("Select a valid CSV profile")
            return

        path = os.path.join(self._profiles_dir(), fname)
        try:
            is_pose = (self.profile_type_combo.currentIndex() == 1)

            if is_pose:
                profile_rows = self._load_csv_as_pose_profile(path)
                cmd = {"type": "pose_profile_upload", "profile": profile_rows}
                self.status_label.setText(f"Sent POSE profile: {fname} ({len(profile_rows)} pts)")
            else:
                profile_rows = self._load_csv_as_profile(path)
                cmd = {"type": "profile_upload", "profile": profile_rows, "units": "mm"}
                self.status_label.setText(f"Sent AXIS profile: {fname} ({len(profile_rows)} pts)")

            _queue_put_latest(self.cmd_queue, cmd)

        except Exception as e:
            self.status_label.setText(f"Profile send failed: {e}")

    def on_start_profile(self):
        is_pose = (self.profile_type_combo.currentIndex() == 1)

        if is_pose:
            cmd = {"type": "pose_profile_start"}
            self.status_label.setText("Pose profile start requested")
        else:
            cmd = {"type": "profile_start"}
            self.status_label.setText("Axis profile start requested")

        _queue_put_latest(self.cmd_queue, cmd)

    def _append_vec6(self, buf_list, vec):
        """Append a length-6 vector to per-axis buffers. Missing/None -> NaN."""
        if not isinstance(vec, list):
            vec = []
        for i in range(6):
            v = vec[i] if i < len(vec) else None
            if v is None:
                buf_list[i].append(float("nan"))
            else:
                try:
                    buf_list[i].append(float(v))
                except Exception:
                    buf_list[i].append(float("nan"))

    def _append_vec6_int(self, buf_list, vec):
        """Append a length-6 vector to per-axis buffers. Missing/None -> None."""
        if not isinstance(vec, list):
            vec = []
        for i in range(6):
            v = vec[i] if i < len(vec) else None
            if v is None:
                buf_list[i].append(None)
            else:
                try:
                    buf_list[i].append(int(v))
                except Exception:
                    buf_list[i].append(None)

    def _append_vecn(self, buf_list, vec, n):
        if not isinstance(vec, (list, tuple)):
            vec = []
        for i in range(n):
            v = vec[i] if i < len(vec) else None
            if v is None:
                buf_list[i].append(float("nan"))
            else:
                try:
                    buf_list[i].append(float(v))
                except Exception:
                    buf_list[i].append(float("nan"))

    def _trim_history(self):
        if not self.tbuf:
            return
        t_latest = self.tbuf[-1]
        t_min = t_latest - self.history_seconds

        # find first index to keep
        k0 = 0
        for k in range(len(self.tbuf)):
            if self.tbuf[k] >= t_min:
                k0 = k
                break

        if k0 <= 0:
            return

        self.tbuf = self.tbuf[k0:]
        for banks in (
                self.pos_buf, self.vel_buf,
                self.temp_motor_buf, self.temp_fet_buf,
                self.tension_cmd_buf, self.tension_rsp_buf,
                self.torque_cmd_buf, self.torque_rsp_buf,
                self.cur_motor_buf, self.cur_bus_buf,
                self.bus_v_buf,
                self.axis_state_buf, self.axis_error_buf,
        ):
            for i in range(6):
                banks[i] = banks[i][k0:]
        for banks, n in (
                (self.hand_cmd_xyz_buf, 3),
                (self.hand_rsp_xyz_buf, 3),
                (self.hand_cmd_rp_buf, 2),
                (self.hand_rsp_rp_buf, 2),
        ):
            for i in range(n):
                banks[i] = banks[i][k0:]

    def _set_table_cell(self, row: int, col: int, value, fmt=None, align=None, tooltip=None):
        """
        value can be float/int/None/NaN/str. If fmt is provided, it's used for numeric formatting.
        align: optional Qt.AlignmentFlag override for text alignment.
        """
        # --- format text ---
        if value is None:
            text = "---"
        elif isinstance(value, str):
            text = value
        else:
            try:
                v = float(value)
                if v != v:  # NaN
                    text = "---"
                else:
                    text = (fmt.format(v) if fmt else str(v))
            except Exception:
                text = "---"

        item = self.axis_table.item(row, col)
        if item is None:
            item = QTableWidgetItem(text)
            self.axis_table.setItem(row, col, item)
        else:
            item.setText(text)

        # Default: right-align (numbers); allow override (e.g. state text)
        if align is None:
            align = Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        item.setTextAlignment(align)

        # Tooltip (mouseover)
        if tooltip is not None:
            item.setToolTip(str(tooltip))

    def update_gui(self):
        updated = False

        def _safe_float(v, default):
            try:
                return float(v)
            except Exception:
                return default

        while not self.telem_queue.empty():
            telem = self.telem_queue.get()
            if not isinstance(telem, dict):
                continue
            telem = _normalize_telemetry(telem)

            self.last_telem_time = time.time()
            t = float(telem.get("t", time.time()))
            trel = t - self.start_time

            # timebase
            self.tbuf.append(trel)

            # required arrays
            self._append_vec6(self.pos_buf, telem.get("pos", []))
            self._append_vec6(self.vel_buf, telem.get("vel", []))

            # temps (match your keys)
            self._append_vec6(self.temp_fet_buf, telem.get("temp_fet", []))
            self._append_vec6(self.temp_motor_buf, telem.get("temp_motor", []))

            # currents for numeric table
            self._append_vec6(self.cur_bus_buf, telem.get("bus_i", []))
            self._append_vec6(self.cur_motor_buf, telem.get("motor_i", []))

            # torque for live plot
            self._append_vec6(self.torque_cmd_buf, telem.get("torque_cmd_nm", []))
            self._append_vec6(self.torque_rsp_buf, telem.get("torque_rsp_nm", []))
            self._append_vec6(self.tension_cmd_buf, telem.get("tension_cmd_n", []))
            self._append_vec6(self.tension_rsp_buf, telem.get("tension_rsp_n", []))

            #bus voltage
            self._append_vec6(self.bus_v_buf, telem.get("bus_v", []))

            # heartbeat/state/error (match your telemetry keys)
            # Expecting vectors like: "axis_state": [6], "axis_error": [6]
            self._append_vec6_int(self.axis_state_buf, telem.get("axis_state", []))
            self._append_vec6_int(self.axis_error_buf, telem.get("axis_error", []))
            hcp = telem.get("hand_cmd_pose", [])
            hep = telem.get("hand_est_pose", [])
            hev = telem.get("hand_est_vel", [])
            if isinstance(hcp, (list, tuple)) and len(hcp) >= 6:
                self.hand_cmd_pose = [float(hcp[i]) for i in range(6)]
            if isinstance(hep, (list, tuple)) and len(hep) >= 6:
                self.hand_est_pose = [float(hep[i]) for i in range(6)]
            if isinstance(hev, (list, tuple)) and len(hev) >= 6:
                self.hand_est_vel = [float(hev[i]) for i in range(6)]
            self.can_rx_hz = _safe_float(telem.get("can_rx_hz", self.can_rx_hz), self.can_rx_hz)
            self.can_tx_hz = _safe_float(telem.get("can_tx_hz", self.can_tx_hz), self.can_tx_hz)
            self.can_msg_hz = _safe_float(telem.get("can_msg_hz", self.can_msg_hz), self.can_msg_hz)
            self.can_util_est = _safe_float(telem.get("can_util_est", self.can_util_est), self.can_util_est)
            self.pos_fbk_hz = _safe_float(telem.get("pos_fbk_hz", self.pos_fbk_hz), self.pos_fbk_hz)
            self.pos_fbk_period0_min_s = _safe_float(
                telem.get("pos_fbk_period0_min_s", self.pos_fbk_period0_min_s),
                self.pos_fbk_period0_min_s,
            )
            self.pos_fbk_period0_max_s = _safe_float(
                telem.get("pos_fbk_period0_max_s", self.pos_fbk_period0_max_s),
                self.pos_fbk_period0_max_s,
            )
            self._append_vecn(self.hand_cmd_xyz_buf, self.hand_cmd_pose[0:3], 3)
            self._append_vecn(self.hand_rsp_xyz_buf, self.hand_est_pose[0:3], 3)
            self._append_vecn(self.hand_cmd_rp_buf, self.hand_cmd_pose[3:5], 2)
            self._append_vecn(self.hand_rsp_rp_buf, self.hand_est_pose[3:5], 2)

            self._trim_history()
            updated = True

        if not updated:
            return

        def _fmt(v, fmt):
            try:
                vv = float(v)
                return format(vv, fmt) if math.isfinite(vv) else "---"
            except Exception:
                return "---"

        self.hand_est_label.setText(
            "Pose Estimate: "
            f"X={_fmt(self.hand_est_pose[0], '.1f')} mm  "
            f"Y={_fmt(self.hand_est_pose[1], '.1f')} mm  "
            f"Z={_fmt(self.hand_est_pose[2], '.1f')} mm  "
            f"Roll={_fmt(self.hand_est_pose[3], '.2f')} deg  "
            f"Pitch={_fmt(self.hand_est_pose[4], '.2f')} deg"
        )
        util_pct = 100.0 * self.can_util_est if math.isfinite(self.can_util_est) else float("nan")
        p0_min_ms = 1000.0 * self.pos_fbk_period0_min_s if math.isfinite(self.pos_fbk_period0_min_s) else float("nan")
        p0_max_ms = 1000.0 * self.pos_fbk_period0_max_s if math.isfinite(self.pos_fbk_period0_max_s) else float("nan")
        self.comm_stats_label.setText(
            "Comm: "
            f"CAN rx={_fmt(self.can_rx_hz, '.1f')} Hz  "
            f"tx={_fmt(self.can_tx_hz, '.1f')} Hz  "
            f"total={_fmt(self.can_msg_hz, '.1f')} Hz  "
            f"util={_fmt(util_pct, '.1f')}%  "
            f"pos_fbk(avg/axis)={_fmt(self.pos_fbk_hz, '.1f')} Hz  "
            f"p0[min,max]={_fmt(p0_min_ms, '.2f')}/{_fmt(p0_max_ms, '.2f')} ms"
        )

        # ---- Update numeric matrix using newest samples ----
        for i in range(6):
            state_code = self.axis_state_buf[i][-1] if self.axis_state_buf[i] else None
            state_text = _axis_state_text(state_code)
            # Include both code + full name in tooltip (nice for debugging)
            state_tip = f"{state_text} ({state_code})" if state_code is not None else state_text

            error_code = self.axis_error_buf[i][-1] if self.axis_error_buf[i] else None
            error_text, error_tip = _decode_odrive_error_mask(error_code)

            pos_i = self.pos_buf[i][-1] if self.pos_buf[i] else float("nan")
            vel_i = self.vel_buf[i][-1] if self.vel_buf[i] else float("nan")

            motor_i = self.cur_motor_buf[i][-1] if self.cur_motor_buf[i] else float("nan")
            bus_i   = self.cur_bus_buf[i][-1] if self.cur_bus_buf[i] else float("nan")
            busv_i = self.bus_v_buf[i][-1] if self.bus_v_buf[i] else float("nan")

            tm_i = self.temp_motor_buf[i][-1] if self.temp_motor_buf[i] else float("nan")
            tf_i = self.temp_fet_buf[i][-1] if self.temp_fet_buf[i] else float("nan")

            # Cols: State, Error, Pos, Vel, MotorI, BusV, BusI, TempMotor, TempFET
            self._set_table_cell(i, 0, state_text, tooltip=state_text)
            self._set_table_cell(i, 1, error_text, tooltip=error_tip)
            self._set_table_cell(i, 2, pos_i, "{:.2f}")
            self._set_table_cell(i, 3, vel_i, "{:.2f}")
            self._set_table_cell(i, 4, motor_i, "{:.2f}")
            self._set_table_cell(i, 5, busv_i, "{:.2f}")
            self._set_table_cell(i, 6, bus_i, "{:.2f}")
            self._set_table_cell(i, 7, tm_i, "{:.1f}")
            self._set_table_cell(i, 8, tf_i, "{:.1f}")

        # update plots
        x = self.tbuf
        for i in range(6):
            self.curves_pos[i].setData(x, self.pos_buf[i])
            self.curves_vel[i].setData(x, self.vel_buf[i])

            self.curves_temp_motor[i].setData(x, self.temp_motor_buf[i])
            self.curves_temp_fet[i].setData(x, self.temp_fet_buf[i])

            self.curves_torque_cmd[i].setData(x, self.torque_cmd_buf[i])
            self.curves_torque_rsp[i].setData(x, self.torque_rsp_buf[i])
            self.curves_tension_cmd[i].setData(x, self.tension_cmd_buf[i])
            self.curves_tension_rsp[i].setData(x, self.tension_rsp_buf[i])

            self.curves_busv[i].setData(x, self.bus_v_buf[i])

        for i in range(3):
            self.pose_cmd_xyz_curves[i].setData(x, self.hand_cmd_xyz_buf[i])
            self.pose_rsp_xyz_curves[i].setData(x, self.hand_rsp_xyz_buf[i])
        for i in range(2):
            self.pose_cmd_rp_curves[i].setData(x, self.hand_cmd_rp_buf[i])
            self.pose_rsp_rp_curves[i].setData(x, self.hand_rsp_rp_buf[i])

# --- Simulated Robot ---  # (not used when connected to real robot)
# def robot_sim(cmd_queue, telem_queue):
#     ...

if __name__ == "__main__":
    cmd_queue = Queue(maxsize=1)   # keep only the latest command
    telem_queue = Queue()

    # Start telemetry listener thread (UDP)
    telem_thread = threading.Thread(
        target=telemetry_listener,
        args=(UDP_TELEM_PORT, telem_queue, lambda s: print(s)),
        daemon=True,
    )
    telem_thread.start()

    # Start TCP command client
    cmd_client = CommandClient(
        ROBOT_HOST, TCP_CMD_PORT, cmd_queue, status_cb=lambda s: print(s)
    )
    cmd_client.start()

    # Start GUI
    app = QApplication(sys.argv)
    gui = RobotGUI(cmd_queue, telem_queue)
    gui.show()
    sys.exit(app.exec())
