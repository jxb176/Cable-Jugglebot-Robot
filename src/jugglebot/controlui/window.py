"""Main controller GUI window."""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from .channels import STYLE_DASH, ChannelRegistry, build_default_channel_registry
from .history import TelemetryHistory
from .session import LiveRobotSession
from .workspace import PlotWorkspace
try:
    from .view3d import Robot3DView
except Exception:
    Robot3DView = None

try:
    from jugglebot.planning import (
        build_path_from_profile,
        build_traj_from_pattern,
        load_pattern_yaml,
        load_profile_yaml,
    )
except Exception:
    load_profile_yaml = None
    build_path_from_profile = None
    load_pattern_yaml = None
    build_traj_from_pattern = None


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

LIVE_VIEW_INTERVAL_MS = 33
DETAIL_VIEW_INTERVAL_MS = 100
CONNECTION_STATUS_INTERVAL_MS = 500


class RobotControlWindow(QWidget):
    def __init__(self, session: LiveRobotSession, channels: ChannelRegistry | None = None):
        super().__init__()
        self.session = session
        self.history: TelemetryHistory = session.history
        self.channels = channels or build_default_channel_registry()
        self.telem_timeout = 2.0
        self.paused_frame = None
        self.paused_times: list[float] = []
        self.paused_series: dict[str, list[float]] = {}

        self.setWindowTitle("Robot Controller + Telemetry")
        self.resize(1400, 1050)

        layout = QVBoxLayout()

        self.status_label = QLabel("Telemetry: waiting...")
        layout.addWidget(self.status_label)
        self.comm_stats_label = QLabel(
            "Comm: CAN rx=--- Hz tx=--- Hz total=--- Hz util=---% pos_fbk(avg/axis)=--- Hz p0[min,max]=---/--- ms"
        )
        layout.addWidget(self.comm_stats_label)

        scene_header = QHBoxLayout()
        scene_header.addWidget(QLabel("Robot 3D View"))
        scene_header.addStretch(1)
        self.scene_hint_label = QLabel("Estimated = cyan, commanded = amber")
        scene_header.addWidget(self.scene_hint_label)
        layout.addLayout(scene_header)

        self.robot_3d_view = None
        if Robot3DView is not None:
            try:
                self.robot_3d_view = Robot3DView(self)
                layout.addWidget(self.robot_3d_view)
            except Exception as exc:
                fallback = QLabel(f"3D view unavailable: {exc}")
                fallback.setWordWrap(True)
                layout.addWidget(fallback)
        else:
            fallback = QLabel("3D view unavailable: OpenGL view module could not be imported.")
            fallback.setWordWrap(True)
            layout.addWidget(fallback)

        layout.addWidget(QLabel("Pose Command (Global): X/Y/Z (mm), Roll/Pitch (deg). Yaw assumed 0."))

        hand_row = QHBoxLayout()
        self.hand_x = self._make_spin(hand_row, "X", -500.0, 500.0, 2, 1.0, " mm")
        self.hand_y = self._make_spin(hand_row, "Y", -500.0, 500.0, 2, 1.0, " mm")
        self.hand_z = self._make_spin(hand_row, "Z", -500.0, 500.0, 2, 1.0, " mm")
        self.hand_roll = self._make_spin(hand_row, "Roll", -45.0, 45.0, 2, 1.0, " deg")
        self.hand_pitch = self._make_spin(hand_row, "Pitch", -45.0, 45.0, 2, 1.0, " deg")
        layout.addLayout(hand_row)

        btns = QHBoxLayout()
        self.btn_hand_send = QPushButton("Send Pose")
        self.btn_hand_send.clicked.connect(self.send_pose)
        btns.addWidget(self.btn_hand_send)
        btns.addStretch(1)
        layout.addLayout(btns)

        self.hand_est_label = QLabel("Pose Estimate: X=--- mm  Y=--- mm  Z=--- mm  Roll=--- deg  Pitch=--- deg")
        layout.addWidget(self.hand_est_label)

        home_layout = QVBoxLayout()
        home_layout.addWidget(QLabel("Home Position (mm)"))
        self.home_spins = []
        home_row = QHBoxLayout()
        for axis in range(6):
            col = QVBoxLayout()
            col.addWidget(QLabel(f"A{axis + 1}"))
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.01)
            spin.setValue(0.0)
            col.addWidget(spin)
            home_row.addLayout(col)
            self.home_spins.append(spin)
        home_layout.addLayout(home_row)
        home_btn_row = QHBoxLayout()
        self.btn_home = QPushButton("HOME")
        self.btn_home.clicked.connect(self.send_home)
        home_btn_row.addWidget(self.btn_home)
        home_btn_row.addStretch(1)
        home_layout.addLayout(home_btn_row)
        layout.addLayout(home_layout)

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
        prof_layout.addWidget(self.profile_type_combo)
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

        self.axis_table_cols = [
            "State",
            "Error",
            "Pos (mm)",
            "Vel (mm/s)",
            "Motor I (A)",
            "Bus V (V)",
            "Bus I (A)",
            "Temp Motor (C)",
            "Temp FET (C)",
        ]
        self.axis_table = QTableWidget(6, len(self.axis_table_cols))
        self.axis_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row_h = self.axis_table.verticalHeader().defaultSectionSize()
        hdr_h = self.axis_table.horizontalHeader().height()
        frame_w = 2 * self.axis_table.frameWidth()
        self.axis_table.setMinimumHeight(hdr_h + 6 * row_h + frame_w + 8)
        self.axis_table.setHorizontalHeaderLabels(self.axis_table_cols)
        self.axis_table.setVerticalHeaderLabels([f"A{axis + 1}" for axis in range(6)])
        self.axis_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.axis_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.axis_table.resizeColumnsToContents()
        self.axis_table.setColumnWidth(0, 150)
        self.axis_table.setColumnWidth(1, 150)
        layout.addWidget(QLabel("Axis Data"))
        layout.addWidget(self.axis_table)

        self.plot_workspace = PlotWorkspace(self.channels, pen_factory=self._channel_pen, parent=self)
        self.plot_workspace.live_mode_changed.connect(self._on_live_mode_changed)
        self.plot_workspace.configuration_changed.connect(self.update_gui)
        layout.addWidget(self.plot_workspace)
        self.setLayout(layout)

        self.populate_profile_dropdown()
        self.populate_jugglepath_dropdown()

        self.live_timer = QTimer(self)
        self.live_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.live_timer.timeout.connect(self.update_live_views)
        self.live_timer.start(LIVE_VIEW_INTERVAL_MS)

        self.detail_timer = QTimer(self)
        self.detail_timer.timeout.connect(self.update_gui)
        self.detail_timer.start(DETAIL_VIEW_INTERVAL_MS)

        self.conn_timer = QTimer(self)
        self.conn_timer.timeout.connect(self.check_connection_status)
        self.conn_timer.start(CONNECTION_STATUS_INTERVAL_MS)

    def _make_spin(self, parent_layout: QHBoxLayout, label: str, lo: float, hi: float, dec: int, step: float, suffix: str) -> QDoubleSpinBox:
        col = QVBoxLayout()
        col.addWidget(QLabel(label))
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(dec)
        spin.setSingleStep(step)
        spin.setSuffix(suffix)
        col.addWidget(spin)
        parent_layout.addLayout(col)
        return spin

    def _channel_pen(self, key: str, width: int = 2):
        channel = self.channels[key]
        style = Qt.PenStyle.DashLine if channel.style == STYLE_DASH else Qt.PenStyle.SolidLine
        return pg.mkPen(color=channel.color, width=width, style=style)

    def _on_live_mode_changed(self, live: bool) -> None:
        if live:
            self.paused_frame = None
            self.paused_times = []
            self.paused_series = {}
            frame = self.session.latest_frame
            if frame is not None and self.robot_3d_view is not None:
                self.robot_3d_view.set_frame(frame)
        else:
            frame = self.session.latest_frame
            if frame is None:
                return
            self.paused_frame = frame
            self.paused_times = self.history.times()
            self.paused_series = {key: self.history.values(key) for key in self.channels}
        self.update_gui()

    def check_connection_status(self) -> None:
        if self.session.has_recent_telemetry(self.telem_timeout):
            telem_text = "Telemetry: connected"
        else:
            telem_text = "Telemetry: waiting..."
        self.status_label.setText(f"{telem_text} | {self.session.command_status}")

    def _profiles_dir(self) -> str:
        base = os.getcwd()
        pdir = os.path.join(base, "Profiles")
        os.makedirs(pdir, exist_ok=True)
        return pdir

    def _jugglepath_profiles_dir(self) -> Path:
        repo_root = Path(__file__).resolve().parents[3]
        pdir = repo_root / "src" / "jugglebot" / "profiles"
        if pdir.exists():
            return pdir
        return Path(os.getcwd()) / "src" / "jugglebot" / "profiles"

    def send_pose(self) -> None:
        self.session.send_command(
            {
                "type": "pose",
                "x_mm": float(self.hand_x.value()),
                "y_mm": float(self.hand_y.value()),
                "z_mm": float(self.hand_z.value()),
                "roll_deg": float(self.hand_roll.value()),
                "pitch_deg": float(self.hand_pitch.value()),
            }
        )

    def send_state(self, state_value: str) -> None:
        self.session.send_command({"type": "state", "value": state_value})

    def send_home(self) -> None:
        positions = [float(spin.value()) for spin in self.home_spins]
        self.session.send_command({"type": "home", "home_pos": positions, "units": "mm"})

    def send_pretension(self) -> None:
        self.session.send_command(
            {
                "type": "pretension",
                "upper_N": float(self.pret_upper_spin.value()),
                "lower_N": float(self.pret_lower_spin.value()),
            }
        )

    def send_spool_gain_mult(self) -> None:
        self.session.send_command(
            {
                "type": "spool_gain_mult",
                "kp": float(self.spool_kp_mult_spin.value()),
                "kd": float(self.spool_kd_mult_spin.value()),
            }
        )

    def _load_csv_as_profile(self, path: str):
        with open(path, "r", newline="") as file_obj:
            rows = [row for row in csv.reader(file_obj) if any(cell.strip() for cell in row)]
        if not rows:
            raise ValueError("empty CSV")
        start_idx = 0
        try:
            float(rows[0][0])
        except Exception:
            start_idx = 1
        profile_rows = []
        for row in rows[start_idx:]:
            if len(row) < 7:
                raise ValueError("each row must have at least 7 columns: time + 6 axes")
            t_s = float(row[0])
            axes = [float(value) for value in row[1:7]]
            profile_rows.append([t_s] + axes)
        times = [row[0] for row in profile_rows]
        if any(t2 < t1 for t1, t2 in zip(times, times[1:])):
            raise ValueError("time column must be non-decreasing")
        return profile_rows

    def _load_csv_as_pose_profile(self, path: str):
        with open(path, "r", newline="") as file_obj:
            rows = [row for row in csv.reader(file_obj) if any(cell.strip() for cell in row)]
        if not rows:
            raise ValueError("empty CSV")
        start_idx = 0
        try:
            float(rows[0][0])
        except Exception:
            start_idx = 1
        profile_rows = []
        for row in rows[start_idx:]:
            if len(row) < 7:
                raise ValueError("each row must have at least 7 columns: time + x y z roll pitch yaw")
            t_s = float(row[0])
            values = [float(value) for value in row[1:7]]
            profile_rows.append([t_s] + values)
        times = [row[0] for row in profile_rows]
        if any(t2 < t1 for t1, t2 in zip(times, times[1:])):
            raise ValueError("time column must be non-decreasing")
        return profile_rows

    def populate_profile_dropdown(self) -> None:
        pdir = self._profiles_dir()
        csvs = sorted([name for name in os.listdir(pdir) if name.lower().endswith(".csv")])
        self.profile_combo.clear()
        if not csvs:
            self.profile_combo.addItem("(no .csv files in Profiles/)")
            self.profile_combo.setEnabled(False)
            return
        self.profile_combo.setEnabled(True)
        for name in csvs:
            self.profile_combo.addItem(name)

    def populate_jugglepath_dropdown(self) -> None:
        self.jp_profile_combo.clear()
        if (
            load_profile_yaml is None
            or build_path_from_profile is None
            or load_pattern_yaml is None
            or build_traj_from_pattern is None
        ):
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
        for path in yamls:
            self.jp_profile_combo.addItem(path.name)

    def on_send_start_jugglepath(self) -> None:
        if not self.jp_profile_combo.isEnabled():
            return
        name = self.jp_profile_combo.currentText()
        if not name or name.startswith("("):
            return
        if (
            load_profile_yaml is None
            or build_path_from_profile is None
            or load_pattern_yaml is None
            or build_traj_from_pattern is None
        ):
            return

        try:
            profile_path = self._jugglepath_profiles_dir() / name
            rate_hz = float(self.profile_rate.value())
            traj = None
            try:
                profile = load_profile_yaml(str(profile_path))
                path, _cmd_hz = build_path_from_profile(profile, command_rate_hz=rate_hz)
                traj = path.build().traj
            except Exception:
                pattern = load_pattern_yaml(str(profile_path))
                traj, _cmd_hz = build_traj_from_pattern(pattern, hand="right", command_rate_hz=rate_hz, cycles=1)
            if traj is None or traj.shape[0] == 0:
                raise ValueError("generated trajectory is empty")

            rows = []
            for row in traj:
                rows.append(
                    [
                        float(row[0]),
                        1000.0 * float(row[1]),
                        1000.0 * float(row[2]),
                        1000.0 * float(row[3]),
                        float(row[4]),
                        float(row[5]),
                        float(row[6]),
                        float(row[7]),
                        float(row[8]),
                        float(row[9]),
                        0.0,
                        0.0,
                        0.0,
                    ]
                )

            self.session.send_command({"type": "pose_profile_run", "profile": rows})
        except Exception:
            self.status_label.setText(f"JugglePath send failed: {name}")

    def on_send_profile(self) -> None:
        if not self.profile_combo.isEnabled():
            return
        name = self.profile_combo.currentText()
        if not name or name.startswith("("):
            return
        try:
            path = os.path.join(self._profiles_dir(), name)
            is_pose = self.profile_type_combo.currentIndex() == 1
            if is_pose:
                profile_rows = self._load_csv_as_pose_profile(path)
                cmd = {"type": "pose_profile_upload", "profile": profile_rows}
            else:
                profile_rows = self._load_csv_as_profile(path)
                cmd = {"type": "profile_upload", "profile": profile_rows, "units": "mm"}
            self.session.send_command(cmd)
        except Exception:
            self.status_label.setText(f"Profile send failed: {name}")

    def on_start_profile(self) -> None:
        is_pose = self.profile_type_combo.currentIndex() == 1
        cmd = {"type": "pose_profile_start"} if is_pose else {"type": "profile_start"}
        self.session.send_command(cmd)

    def _set_table_cell(self, row: int, col: int, value, fmt: str | None = None, tooltip: str | None = None) -> None:
        if value is None:
            text = "---"
        elif isinstance(value, str):
            text = value
        else:
            try:
                numeric = float(value)
                text = fmt.format(numeric) if fmt is not None and math.isfinite(numeric) else (str(numeric) if math.isfinite(numeric) else "---")
            except Exception:
                text = "---"
        item = self.axis_table.item(row, col)
        if item is None:
            item = QTableWidgetItem(text)
            self.axis_table.setItem(row, col, item)
        else:
            item.setText(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if tooltip is not None:
            item.setToolTip(tooltip)

    def update_live_views(self) -> None:
        if self.session.poll() <= 0:
            return
        frame = self.session.latest_frame
        if frame is None:
            return

        if self.robot_3d_view is not None and self.plot_workspace.is_live_mode:
            self.robot_3d_view.set_frame(frame)

    def update_gui(self) -> None:
        if self.plot_workspace.is_live_mode:
            frame = self.session.latest_frame
            if frame is None:
                if self.session.poll() <= 0:
                    return
                frame = self.session.latest_frame
                if frame is None:
                    return
            times = self.history.times()
            visible_keys = self.plot_workspace.visible_channel_keys()
            series_by_key = {key: self.history.values(key) for key in visible_keys}
        else:
            frame = self.paused_frame
            if frame is None:
                return
            times = self.paused_times
            visible_keys = self.plot_workspace.visible_channel_keys()
            series_by_key = {key: self.paused_series.get(key, []) for key in visible_keys}

        self.hand_est_label.setText(
            "Pose Estimate: "
            f"X={self._fmt(frame.hand_est_pose[0], '.1f')} mm  "
            f"Y={self._fmt(frame.hand_est_pose[1], '.1f')} mm  "
            f"Z={self._fmt(frame.hand_est_pose[2], '.1f')} mm  "
            f"Roll={self._fmt(frame.hand_est_pose[3], '.2f')} deg  "
            f"Pitch={self._fmt(frame.hand_est_pose[4], '.2f')} deg"
        )

        util_pct = 100.0 * frame.comm_stats.can_util_est if math.isfinite(frame.comm_stats.can_util_est) else float("nan")
        p0_min_ms = 1000.0 * frame.comm_stats.pos_fbk_period0_min_s if math.isfinite(frame.comm_stats.pos_fbk_period0_min_s) else float("nan")
        p0_max_ms = 1000.0 * frame.comm_stats.pos_fbk_period0_max_s if math.isfinite(frame.comm_stats.pos_fbk_period0_max_s) else float("nan")
        self.comm_stats_label.setText(
            "Comm: "
            f"CAN rx={self._fmt(frame.comm_stats.can_rx_hz, '.1f')} Hz  "
            f"tx={self._fmt(frame.comm_stats.can_tx_hz, '.1f')} Hz  "
            f"total={self._fmt(frame.comm_stats.can_msg_hz, '.1f')} Hz  "
            f"util={self._fmt(util_pct, '.1f')}%  "
            f"pos_fbk(avg/axis)={self._fmt(frame.comm_stats.pos_fbk_hz, '.1f')} Hz  "
            f"p0[min,max]={self._fmt(p0_min_ms, '.2f')}/{self._fmt(p0_max_ms, '.2f')} ms"
        )

        for axis in range(6):
            state_code = frame.axis_state[axis]
            state_text = self._axis_state_text(state_code)
            error_text, error_tip = self._decode_odrive_error_mask(frame.axis_error[axis])
            self._set_table_cell(axis, 0, state_text, tooltip=state_text)
            self._set_table_cell(axis, 1, error_text, tooltip=error_tip)
            self._set_table_cell(axis, 2, frame.pos_mm[axis], "{:.2f}")
            self._set_table_cell(axis, 3, frame.vel_mmps[axis], "{:.2f}")
            self._set_table_cell(axis, 4, frame.motor_current_a[axis], "{:.2f}")
            self._set_table_cell(axis, 5, frame.bus_voltage_v[axis], "{:.2f}")
            self._set_table_cell(axis, 6, frame.bus_current_a[axis], "{:.2f}")
            self._set_table_cell(axis, 7, frame.temp_motor_c[axis], "{:.1f}")
            self._set_table_cell(axis, 8, frame.temp_fet_c[axis], "{:.1f}")

        self.plot_workspace.render(times, series_by_key)

    @staticmethod
    def _fmt(value, spec: str) -> str:
        try:
            numeric = float(value)
            return format(numeric, spec) if math.isfinite(numeric) else "---"
        except Exception:
            return "---"

    @staticmethod
    def _axis_state_text(state_code) -> str:
        if state_code is None:
            return "---"
        try:
            code = int(state_code)
        except Exception:
            return "---"
        return AXIS_STATE_NAMES.get(code, f"STATE_{code}")

    @staticmethod
    def _decode_odrive_error_mask(err_code) -> tuple[str, str]:
        if err_code is None:
            return "---", "No error data"
        try:
            code = int(err_code)
        except Exception:
            return "---", "Invalid error value"
        if code == 0:
            return "OK", "0x00000000 (no errors)"
        names = []
        for bit in sorted(ODRIVE_ERROR_BITS):
            if code & bit:
                names.append(ODRIVE_ERROR_BITS[bit])
        known_mask = 0
        for bit in ODRIVE_ERROR_BITS:
            known_mask |= bit
        unknown = code & (~known_mask)
        if unknown:
            names.append(f"UNKNOWN_BITS:0x{unknown:08X}")
        short = names[0] if len(names) == 1 else f"MULTI({len(names)})"
        tooltip = " | ".join([f"0x{code:08X}"] + names)
        return short, tooltip
