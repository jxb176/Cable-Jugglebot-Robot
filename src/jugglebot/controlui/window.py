"""Main controller GUI window."""

from __future__ import annotations
import math
import os
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont, QShowEvent
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import pyqtgraph as pg

from .channels import STYLE_DASH, ChannelRegistry, build_default_channel_registry
from .history import HistorySnapshot, TelemetryHistory
from .session import LiveRobotSession
from .spacemouse import SpaceMouseWorker, create_spacemouse_backend
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
PLOT_VIEW_INTERVAL_MS = 100
STATUS_VIEW_INTERVAL_MS = 200
CONNECTION_STATUS_INTERVAL_MS = 500
SPACEMOUSE_POLL_INTERVAL_MS = 20

HOMING_MODE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Manual", "manual"),
    ("Homing Prep", "homing_prep"),
    ("Z-axis Zero", "z_axis_zero"),
    ("XY-axis Zero", "xy_axis_zero"),
)


class SelectAllDoubleSpinBox(QDoubleSpinBox):
    def focusInEvent(self, event) -> None:
        super().focusInEvent(event)
        QTimer.singleShot(0, self._select_all_text)

    def mousePressEvent(self, event) -> None:
        line_edit = self.lineEdit()
        select_all = (
            event.button() == Qt.MouseButton.LeftButton
            and line_edit is not None
            and line_edit.geometry().contains(event.position().toPoint())
        )
        super().mousePressEvent(event)
        if select_all:
            QTimer.singleShot(0, self._select_all_text)

    def _select_all_text(self) -> None:
        line_edit = self.lineEdit()
        if line_edit is not None:
            line_edit.selectAll()


class RefreshOnPopupComboBox(QComboBox):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._popup_refresh_cb = None

    def set_popup_refresh_callback(self, callback) -> None:
        self._popup_refresh_cb = callback

    def showPopup(self) -> None:
        if self._popup_refresh_cb is not None:
            self._popup_refresh_cb()
        super().showPopup()


class RobotControlWindow(QWidget):
    def __init__(
        self,
        session: LiveRobotSession,
        channels: ChannelRegistry | None = None,
        *,
        live_display_seconds: float = 5.0,
    ):
        super().__init__()
        self.session = session
        self.history: TelemetryHistory = session.history
        self.channels = channels or build_default_channel_registry()
        self.telem_timeout = 2.0
        self.paused_frame = None
        self.paused_snapshot: HistorySnapshot | None = None
        self._plots_dirty = True
        self._status_dirty = True
        self._startup_geometry_applied = False
        self._spacemouse_backend = create_spacemouse_backend()
        self._last_spacemouse_sample = None
        self._spacemouse_worker: SpaceMouseWorker | None = None

        self.setWindowTitle("Robot Controller + Telemetry")
        self.resize(1400, 1050)
        self.setMinimumSize(400, 780)

        root_layout = QVBoxLayout()
        root_layout.setSizeConstraint(QLayout.SizeConstraint.SetNoConstraint)

        self.main_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.top_splitter = QSplitter(Qt.Orientation.Horizontal, self.main_splitter)

        self.left_panel_scroll = QScrollArea(self.top_splitter)
        self.left_panel_scroll.setWidgetResizable(True)
        self.left_panel_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.left_panel_container = QWidget(self.left_panel_scroll)
        self.left_panel_layout = QVBoxLayout(self.left_panel_container)
        self.left_panel_layout.setContentsMargins(0, 0, 0, 0)
        self.left_panel_layout.setSpacing(10)
        self.left_panel_scroll.setWidget(self.left_panel_container)

        self.top_right_container = QWidget(self.top_splitter)
        self.top_right_layout = QVBoxLayout(self.top_right_container)
        self.top_right_layout.setContentsMargins(0, 0, 0, 0)
        self.top_right_layout.setSpacing(10)

        self.robot_state_group, robot_state_layout = self._create_section("Robot State")
        self.command_inputs_group, command_inputs_layout = self._create_section("Command Inputs")
        self.command_inputs_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.robot_actions_group, robot_actions_layout = self._create_section("Robot Actions")
        self.scene_group, scene_layout = self._create_section("Robot 3D View")
        self.scene_group.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        link_grid = QGridLayout()
        link_grid.setHorizontalSpacing(10)
        link_grid.setVerticalSpacing(6)
        self.tcp_status_badge = self._make_status_badge("DOWN")
        self.udp_status_badge = self._make_status_badge("DOWN")
        self.can_status_badge = self._make_status_badge("DOWN")
        self.target_value_label = QLabel("ROBOT")
        self.can_util_value_label = QLabel("---%")
        self.runtime_time_value_label = QLabel("--- s")
        self.sim_time_value_label = QLabel("--- s")
        self.status_message_label = QLabel("")
        self.status_message_label.setWordWrap(True)
        link_grid.addWidget(QLabel("TCP"), 0, 0)
        link_grid.addWidget(self.tcp_status_badge, 0, 1)
        link_grid.addWidget(QLabel("UDP"), 0, 2)
        link_grid.addWidget(self.udp_status_badge, 0, 3)
        link_grid.addWidget(QLabel("Target"), 1, 0)
        link_grid.addWidget(self.target_value_label, 1, 1)
        link_grid.addWidget(QLabel("CAN"), 1, 2)
        link_grid.addWidget(self.can_status_badge, 1, 3)
        link_grid.addWidget(QLabel("Util"), 1, 4)
        link_grid.addWidget(self.can_util_value_label, 1, 5)
        link_grid.addWidget(QLabel("Runtime"), 2, 0)
        link_grid.addWidget(self.runtime_time_value_label, 2, 1)
        link_grid.addWidget(QLabel("Sim Time"), 2, 2)
        link_grid.addWidget(self.sim_time_value_label, 2, 3)
        robot_state_layout.addLayout(link_grid)
        robot_state_layout.addWidget(self._make_subsection_label("Pose Feedback"))

        pose_feedback_rows = ["X (mm)", "Y (mm)", "Z (mm)", "Roll (deg)", "Pitch (deg)"]
        self.pose_feedback_table = QTableWidget(len(pose_feedback_rows), 2)
        self.pose_feedback_table.setHorizontalHeaderLabels(["Command Feedback", "Position Feedback"])
        self.pose_feedback_table.setVerticalHeaderLabels(pose_feedback_rows)
        self.pose_feedback_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.pose_feedback_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.pose_feedback_table.resizeColumnsToContents()
        self.pose_feedback_table.setColumnWidth(0, 150)
        self.pose_feedback_table.setColumnWidth(1, 150)
        row_h = self.pose_feedback_table.verticalHeader().defaultSectionSize()
        hdr_h = self.pose_feedback_table.horizontalHeader().height()
        frame_w = 2 * self.pose_feedback_table.frameWidth()
        self.pose_feedback_table.setMinimumHeight(hdr_h + len(pose_feedback_rows) * row_h + frame_w + 8)
        robot_state_layout.addWidget(self.pose_feedback_table)

        self.robot_3d_view = None
        self.scene_hint_label = QLabel("Estimated = cyan, commanded = amber")
        scene_layout.addWidget(self.scene_hint_label)

        if Robot3DView is not None:
            try:
                self.robot_3d_view = Robot3DView(self)
                scene_layout.addWidget(self.robot_3d_view, 1)
            except Exception as exc:
                fallback = QLabel(f"3D view unavailable: {exc}")
                fallback.setWordWrap(True)
                scene_layout.addWidget(fallback)
        else:
            fallback = QLabel("3D view unavailable: OpenGL view module could not be imported.")
            fallback.setWordWrap(True)
            scene_layout.addWidget(fallback)

        pose_specs = (
            ("X", -500.0, 500.0, 2, 1.0, " mm", ".1f"),
            ("Y", -500.0, 500.0, 2, 1.0, " mm", ".1f"),
            ("Z", -500.0, 500.0, 2, 1.0, " mm", ".1f"),
            ("Roll", -45.0, 45.0, 2, 1.0, " deg", ".2f"),
            ("Pitch", -45.0, 45.0, 2, 1.0, " deg", ".2f"),
        )
        self._pose_feedback_formats = tuple(spec[6] for spec in pose_specs)
        self._pose_feedback_units = tuple(spec[5].strip() for spec in pose_specs)

        command_inputs_layout.setSpacing(10)
        command_inputs_row = QHBoxLayout()
        command_inputs_row.setSpacing(16)

        pretension_inputs_layout = QVBoxLayout()
        pretension_inputs_layout.setSpacing(6)
        pretension_inputs_layout.addWidget(self._make_subsection_label("Pretension"))
        pretension_inputs_layout.addWidget(QLabel("Upper Pretension [N]"))
        self.pret_upper_spin = QDoubleSpinBox()
        self.pret_upper_spin.setDecimals(2)
        self.pret_upper_spin.setRange(0.0, 500.0)
        self.pret_upper_spin.setSingleStep(1.0)
        self.pret_upper_spin.setValue(3.0)
        pretension_inputs_layout.addWidget(self.pret_upper_spin)
        pretension_inputs_layout.addWidget(QLabel("Lower Pretension [N]"))
        self.pret_lower_spin = QDoubleSpinBox()
        self.pret_lower_spin.setDecimals(2)
        self.pret_lower_spin.setRange(0.0, 500.0)
        self.pret_lower_spin.setSingleStep(1.0)
        self.pret_lower_spin.setValue(3.0)
        pretension_inputs_layout.addWidget(self.pret_lower_spin)
        pretension_inputs_layout.addStretch(1)
        command_inputs_row.addLayout(pretension_inputs_layout, 1)

        home_inputs_layout = QVBoxLayout()
        home_inputs_layout.setSpacing(6)
        home_inputs_layout.addWidget(self._make_subsection_label("Homing"))
        home_inputs_layout.addWidget(QLabel("Mode"))
        self.homing_mode_combo = QComboBox()
        for label, value in HOMING_MODE_OPTIONS:
            self.homing_mode_combo.addItem(label, value)
        home_inputs_layout.addWidget(self.homing_mode_combo)
        self.home_spool_label = QLabel("Manual Home Spool Position [mm]")
        home_inputs_layout.addWidget(self.home_spool_label)
        self.home_spins = []
        home_grid = QGridLayout()
        home_grid.setHorizontalSpacing(8)
        home_grid.setVerticalSpacing(6)
        home_grid.setColumnStretch(0, 0)
        home_grid.setColumnStretch(1, 0)
        for axis in range(6):
            spin = QDoubleSpinBox()
            spin.setDecimals(4)
            spin.setRange(-100.0, 100.0)
            spin.setSingleStep(0.01)
            spin.setValue(0.0)
            self._set_spin_text_width(spin, "-100.0000")
            self.home_spins.append(spin)
            home_grid.addWidget(QLabel(f"A{axis + 1}"), axis, 0)
            home_grid.addWidget(spin, axis, 1)
        home_inputs_layout.addLayout(home_grid)
        homing_button_row = QHBoxLayout()
        homing_button_row.setSpacing(8)
        self.btn_homing_run = self._make_command_button("Run Homing")
        self.btn_homing_cancel = self._make_command_button("Cancel Homing")
        self.btn_homing_apply = self._make_command_button("Apply Result")
        self.btn_homing_run.clicked.connect(self.send_homing_run)
        self.btn_homing_cancel.clicked.connect(self.send_homing_cancel)
        self.btn_homing_apply.clicked.connect(self.send_homing_apply)
        homing_button_row.addWidget(self.btn_homing_run)
        homing_button_row.addWidget(self.btn_homing_cancel)
        homing_button_row.addWidget(self.btn_homing_apply)
        home_inputs_layout.addLayout(homing_button_row)
        self.homing_status_label = QLabel()
        self.homing_status_label.setWordWrap(True)
        self.homing_status_label.setStyleSheet("font-family: monospace;")
        home_inputs_layout.addWidget(self.homing_status_label)
        home_inputs_layout.addStretch(1)
        command_inputs_row.addLayout(home_inputs_layout, 1)

        manual_pose_layout = QVBoxLayout()
        manual_pose_layout.setSpacing(6)
        manual_pose_layout.addWidget(self._make_subsection_label("Manual Pose"))
        pose_grid = QGridLayout()
        pose_grid.setHorizontalSpacing(12)
        pose_grid.setVerticalSpacing(6)
        pose_grid.setColumnStretch(0, 0)
        pose_grid.setColumnStretch(1, 0)
        pose_grid.addWidget(QLabel("Axis"), 0, 0)
        pose_grid.addWidget(QLabel("Input"), 0, 1)

        self.hand_x = self._make_value_spin(*pose_specs[0][1:6])
        self.hand_y = self._make_value_spin(*pose_specs[1][1:6])
        self.hand_z = self._make_value_spin(*pose_specs[2][1:6])
        self.hand_roll = self._make_value_spin(*pose_specs[3][1:6])
        self.hand_pitch = self._make_value_spin(*pose_specs[4][1:6])
        pose_spins = [self.hand_x, self.hand_y, self.hand_z, self.hand_roll, self.hand_pitch]

        for row, ((axis_label, *_rest), spin) in enumerate(zip(pose_specs, pose_spins), start=1):
            pose_grid.addWidget(QLabel(axis_label), row, 0)
            pose_grid.addWidget(self._make_spin_with_unit(spin, self._pose_feedback_units[row - 1]), row, 1)

        self.btn_hand_send = self._make_command_button("Send Pose")
        self.btn_hand_send.clicked.connect(self.send_pose)
        self.btn_hand_send.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        pose_grid.addWidget(self.btn_hand_send, len(pose_specs) + 1, 1, alignment=Qt.AlignmentFlag.AlignLeft)
        manual_pose_layout.addLayout(pose_grid)
        manual_pose_layout.addStretch(1)
        command_inputs_row.addLayout(manual_pose_layout, 1)

        space_mouse_layout = QVBoxLayout()
        space_mouse_layout.setSpacing(6)
        space_mouse_layout.addWidget(self._make_subsection_label("SpaceMouse"))
        self.btn_space_mouse = self._make_toggle_command_button("SpaceMouse", "#7c3aed", "#6d28d9")
        self.btn_space_mouse.toggled.connect(self._on_space_mouse_toggled)
        space_mouse_layout.addWidget(self.btn_space_mouse, alignment=Qt.AlignmentFlag.AlignLeft)
        space_mouse_layout.addWidget(QLabel("Mode"))
        self.input_mode_combo = QComboBox()
        self.input_mode_combo.addItems(
            [
                "Position",
                "Velocity",
                "Acceleration",
            ]
        )
        self.input_mode_combo.currentIndexChanged.connect(self._on_space_mouse_config_changed)
        space_mouse_layout.addWidget(self.input_mode_combo)
        gain_row = QHBoxLayout()
        gain_row.addWidget(QLabel("Sensitivity"))
        self.input_mode_gain_spin = QDoubleSpinBox()
        self.input_mode_gain_spin.setDecimals(2)
        self.input_mode_gain_spin.setRange(0.05, 20.0)
        self.input_mode_gain_spin.setSingleStep(0.05)
        self.input_mode_gain_spin.setValue(1.0)
        self.input_mode_gain_spin.valueChanged.connect(self._on_space_mouse_config_changed)
        gain_row.addWidget(self.input_mode_gain_spin)
        space_mouse_layout.addLayout(gain_row)
        self.input_mode_status_label = QLabel()
        self.input_mode_status_label.setWordWrap(True)
        space_mouse_layout.addWidget(self.input_mode_status_label)
        self.spacemouse_input_label = QLabel()
        self.spacemouse_input_label.setWordWrap(True)
        self.spacemouse_input_label.setStyleSheet("font-family: monospace;")
        space_mouse_layout.addWidget(self.spacemouse_input_label)
        space_mouse_layout.addStretch(1)
        command_inputs_row.addLayout(space_mouse_layout, 1)

        robot_actions_layout.setSpacing(10)

        self.btn_disable = self._make_state_command_button("Standby", "#667085", "#475467")
        self.btn_pretension = self._make_state_command_button("Pretension", "#d97706", "#b45309")
        self.btn_home = self._make_state_command_button("Home", "#2563eb", "#1d4ed8")
        self.btn_enable = self._make_state_command_button("Enable", "#15803d", "#166534")
        self.btn_estop = self._make_state_command_button("Estop", "#b42318", "#912018")

        self.btn_disable.clicked.connect(lambda: self.send_state("disable"))
        self.btn_pretension.clicked.connect(self.send_pretension)
        self.btn_home.clicked.connect(self.send_home)
        self.btn_enable.clicked.connect(lambda: self.send_state("enable"))
        self.btn_estop.clicked.connect(lambda: self.send_state("estop"))
        self.homing_mode_combo.currentIndexChanged.connect(self._on_homing_mode_changed)

        self._state_buttons_by_state = {
            "disable": self.btn_disable,
            "pretension": self.btn_pretension,
            "enable": self.btn_enable,
            "estop": self.btn_estop,
        }
        self._all_state_buttons = [
            self.btn_disable,
            self.btn_pretension,
            self.btn_home,
            self.btn_enable,
            self.btn_estop,
        ]

        state_button_row = QHBoxLayout()
        state_button_row.setSpacing(8)
        for button in self._all_state_buttons:
            state_button_row.addWidget(button, 1)
        robot_actions_layout.addLayout(state_button_row)
        profile_layout = QVBoxLayout()
        profile_layout.setSpacing(6)
        profile_layout.addWidget(self._make_subsection_label("Profile"))
        self.jp_profile_combo = RefreshOnPopupComboBox(self)
        self.jp_profile_combo.set_popup_refresh_callback(self.populate_jugglepath_dropdown)
        profile_layout.addWidget(QLabel("Profile"))
        profile_layout.addWidget(self.jp_profile_combo)
        self.jp_send_start_btn = self._make_command_button("Run Profile")
        self.jp_send_start_btn.clicked.connect(self.on_send_start_jugglepath)
        profile_layout.addWidget(self.jp_send_start_btn, alignment=Qt.AlignmentFlag.AlignLeft)
        profile_layout.addStretch(1)
        command_inputs_row.addLayout(profile_layout, 1)
        command_inputs_layout.addLayout(command_inputs_row)
        command_inputs_layout.addWidget(self.status_message_label)

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
        robot_state_layout.addWidget(self._make_subsection_label("Axis Data"))
        robot_state_layout.addWidget(self.axis_table)

        self.left_panel_layout.addWidget(self.robot_actions_group)
        self.left_panel_layout.addWidget(self.robot_state_group)
        self.left_panel_layout.addWidget(self.command_inputs_group)
        self.left_panel_layout.addStretch(1)

        self.top_right_layout.addWidget(self.scene_group, 1)

        self.plot_workspace = PlotWorkspace(
            self.channels,
            pen_factory=self._channel_pen,
            live_display_seconds=live_display_seconds,
            parent=self,
        )
        self.plot_workspace.live_mode_changed.connect(self._on_live_mode_changed)
        self.plot_workspace.configuration_changed.connect(self._on_workspace_configuration_changed)
        self.top_splitter.addWidget(self.left_panel_scroll)
        self.top_splitter.addWidget(self.top_right_container)
        self.main_splitter.addWidget(self.top_splitter)
        self.main_splitter.addWidget(self.plot_workspace)
        self.top_splitter.setChildrenCollapsible(False)
        self.main_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(self.main_splitter, 1)
        self.setLayout(root_layout)

        self.populate_jugglepath_dropdown()
        self._update_pose_feedback_labels()
        self._update_input_mode_status()
        self._update_spacemouse_sample_display(None)
        self._update_homing_controls(None)
        self._update_state_button_feedback(None)
        self._update_link_status_indicators(self.session.latest_frame)

        self.live_timer = QTimer(self)
        self.live_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.live_timer.timeout.connect(self.update_live_views)
        self.live_timer.start(LIVE_VIEW_INTERVAL_MS)

        self.plot_timer = QTimer(self)
        self.plot_timer.timeout.connect(self.update_plots)
        self.plot_timer.start(PLOT_VIEW_INTERVAL_MS)

        self.status_timer = QTimer(self)
        self.status_timer.timeout.connect(self.update_status_widgets)
        self.status_timer.start(STATUS_VIEW_INTERVAL_MS)

        self.conn_timer = QTimer(self)
        self.conn_timer.timeout.connect(self.check_connection_status)
        self.conn_timer.start(CONNECTION_STATUS_INTERVAL_MS)

        self.spacemouse_timer = QTimer(self)
        self.spacemouse_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.spacemouse_timer.timeout.connect(self._poll_spacemouse)

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        if self._startup_geometry_applied:
            return
        self._startup_geometry_applied = True
        self._apply_startup_geometry()
        QTimer.singleShot(0, self._apply_initial_splitter_sizes)

    def _apply_startup_geometry(self) -> None:
        handle = self.windowHandle()
        screen = handle.screen() if handle is not None else None
        if screen is None:
            return
        available = screen.availableGeometry()
        left = available.left()
        top = available.top()
        width = max(1, available.width() // 2)
        height = max(1, available.height())
        self.setGeometry(left, top, width, height)

    def _apply_initial_splitter_sizes(self) -> None:
        total_width = max(1, self.width())
        total_height = max(1, self.height())
        self.top_splitter.setSizes([int(total_width * 0.65), int(total_width * 0.35)])
        self.main_splitter.setSizes([int(total_height * 0.52), int(total_height * 0.48)])

    def _create_section(self, title: str) -> tuple[QGroupBox, QVBoxLayout]:
        group = QGroupBox(title, self)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(10, 14, 10, 10)
        group_layout.setSpacing(8)
        return group, group_layout

    def _make_value_spin(self, lo: float, hi: float, dec: int, step: float, _unit_suffix: str) -> QDoubleSpinBox:
        spin = SelectAllDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(dec)
        spin.setSingleStep(step)
        spin.valueChanged.connect(self._update_pose_feedback_labels)
        self._set_spin_text_width(spin, "00000")
        return spin

    def _make_spin_with_unit(self, spin: QDoubleSpinBox, unit: str) -> QWidget:
        container = QWidget(self)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(spin)
        layout.addWidget(QLabel(unit))
        return container

    def _set_spin_text_width(self, spin: QDoubleSpinBox, sample_text: str) -> None:
        width = spin.fontMetrics().horizontalAdvance(sample_text) + 42
        spin.setFixedWidth(width)
        spin.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    def _make_command_button(self, text: str) -> QPushButton:
        button = QPushButton(text.upper(), self)
        font = QFont(button.font())
        font.setBold(True)
        button.setFont(font)
        button.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        return button

    def _make_state_command_button(self, text: str, checked_bg: str, checked_border: str) -> QPushButton:
        button = self._make_command_button(text)
        button.setCheckable(True)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.setStyleSheet(
            f"""
            QPushButton {{
                padding: 6px 10px;
            }}
            QPushButton:checked {{
                background-color: {checked_bg};
                border: 2px solid {checked_border};
                color: white;
            }}
            """
        )
        return button

    def _make_toggle_command_button(self, text: str, checked_bg: str, checked_border: str) -> QPushButton:
        button = self._make_command_button(text)
        button.setCheckable(True)
        button.setStyleSheet(
            f"""
            QPushButton {{
                padding: 6px 10px;
            }}
            QPushButton:checked {{
                background-color: {checked_bg};
                border: 2px solid {checked_border};
                color: white;
            }}
            """
        )
        return button

    def _make_subsection_label(self, text: str) -> QLabel:
        label = QLabel(text, self)
        font = QFont(label.font())
        font.setBold(True)
        label.setFont(font)
        return label

    def _make_status_badge(self, text: str) -> QLabel:
        label = QLabel(text, self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setMinimumWidth(52)
        self._set_status_badge(label, text, None)
        return label

    def _set_status_badge(self, label: QLabel, text: str, up: bool | None) -> None:
        if up is True:
            bg = "#15803d"
            border = "#166534"
            fg = "white"
        elif up is False:
            bg = "#b42318"
            border = "#912018"
            fg = "white"
        else:
            bg = "#98a2b3"
            border = "#667085"
            fg = "white"
        label.setText(text)
        label.setStyleSheet(
            f"QLabel {{ background-color: {bg}; border: 1px solid {border}; border-radius: 4px; color: {fg}; padding: 2px 8px; }}"
        )

    def _channel_pen(self, key: str, width: int = 2):
        channel = self.channels[key]
        style = Qt.PenStyle.DashLine if channel.style == STYLE_DASH else Qt.PenStyle.SolidLine
        return pg.mkPen(color=channel.color, width=width, style=style)

    def _on_live_mode_changed(self, live: bool) -> None:
        if live:
            self.paused_frame = None
            self.paused_snapshot = None
            frame = self.session.latest_frame
            if frame is not None and self.robot_3d_view is not None:
                self.robot_3d_view.set_frame(frame)
        else:
            frame = self.session.latest_frame
            if frame is None:
                return
            self.paused_frame = frame
            self.paused_snapshot = self.history.snapshot(self.channels)
            if self.robot_3d_view is not None:
                self.robot_3d_view.set_frame(frame)
        self._mark_status_dirty()
        self._mark_plots_dirty()
        self.update_status_widgets()
        self.update_plots()

    def _on_workspace_configuration_changed(self) -> None:
        self._mark_plots_dirty()
        self.update_plots()

    def check_connection_status(self) -> None:
        frame = self.session.latest_frame if self.plot_workspace.is_live_mode else self.paused_frame
        self._update_link_status_indicators(frame)

    def _jugglepath_profiles_dir(self) -> Path:
        repo_root = Path(__file__).resolve().parents[3]
        pdir = repo_root / "src" / "jugglebot" / "profiles"
        if pdir.exists():
            return pdir
        return Path(os.getcwd()) / "src" / "jugglebot" / "profiles"

    def send_pose(self) -> None:
        self._deactivate_spacemouse()
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
        if str(state_value).lower() != "enable":
            self._deactivate_spacemouse()
        self.session.send_command({"type": "state", "value": state_value})

    def _update_pose_feedback_labels(self, frame=None) -> None:
        if frame is None or not hasattr(frame, "hand_cmd_pose"):
            cmd_values = [
                self.hand_x.value(),
                self.hand_y.value(),
                self.hand_z.value(),
                self.hand_roll.value(),
                self.hand_pitch.value(),
            ]
            est_values = [None] * len(cmd_values)
        else:
            cmd_values = frame.hand_cmd_pose[:5]
            est_values = frame.hand_est_pose[:5]

        for index, value in enumerate(cmd_values):
            fmt = self._pose_feedback_formats[index]
            self._set_pose_feedback_cell(index, 0, value, fmt)
        for index, value in enumerate(est_values):
            fmt = self._pose_feedback_formats[index]
            self._set_pose_feedback_cell(index, 1, value, fmt)

    def _update_input_mode_status(self) -> None:
        mode = self.input_mode_combo.currentText()
        gain = self.input_mode_gain_spin.value()
        if self.btn_space_mouse.isChecked():
            self.input_mode_status_label.setText(
                f"SpaceMouse {mode.lower()} mode active. Gain = {gain:.2f}. {self._spacemouse_backend.status_text()}"
            )
        else:
            self.input_mode_status_label.setText(
                f"Manual pose input active. SpaceMouse gain = {gain:.2f}. {self._spacemouse_backend.status_text()}"
            )

    def _update_spacemouse_sample_display(self, sample) -> None:
        self._last_spacemouse_sample = sample
        if sample is None:
            values = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            age_text = "---"
            drained = 0
            poll_text = "---"
        else:
            values = (
                float(sample.tx),
                float(sample.ty),
                float(sample.tz),
                float(sample.rx),
                float(sample.ry),
                float(sample.rz),
            )
            device_time_s = getattr(sample, "device_time_s", None)
            if device_time_s is None:
                age_text = self._fmt(getattr(sample, "device_age_ms", None), ".1f")
            else:
                try:
                    import timeit
                    age_text = self._fmt(1000.0 * max(0.0, timeit.default_timer() - float(device_time_s)), ".1f")
                except Exception:
                    age_text = self._fmt(getattr(sample, "device_age_ms", None), ".1f")
            drained = max(0, int(getattr(sample, "reports_drained", 0)))
            poll_text = self._fmt(getattr(sample, "poll_interval_ms", None), ".1f")
        self.spacemouse_input_label.setText(
            "TX {:+.3f}  TY {:+.3f}  TZ {:+.3f}\n"
            "RX {:+.3f}  RY {:+.3f}  RZ {:+.3f}\n"
            "POLL {} ms  AGE {} ms  DRAIN {}".format(*values, poll_text, age_text, drained)
        )

    def _manual_input_mode(self) -> str:
        return self.input_mode_combo.currentText().strip().lower()

    def _send_manual_input_config(self) -> None:
        self.session.send_command(
            {
                "type": "manual_input_config",
                "mode": self._manual_input_mode(),
                "gain": float(self.input_mode_gain_spin.value()),
            }
        )

    def _on_space_mouse_config_changed(self) -> None:
        self._update_input_mode_status()
        self._send_manual_input_config()

    def _send_spacemouse_sample(self, sample) -> None:
        self.session.send_command(
            {
                "type": "manual_input_sample",
                "tx": float(sample.tx),
                "ty": float(sample.ty),
                "tz": float(sample.tz),
                "rx": float(sample.rx),
                "ry": float(sample.ry),
                "rz": float(sample.rz),
            }
        )

    def _on_space_mouse_toggled(self, enabled: bool) -> None:
        if enabled:
            if not self._spacemouse_backend.is_available():
                self.btn_space_mouse.blockSignals(True)
                self.btn_space_mouse.setChecked(False)
                self.btn_space_mouse.blockSignals(False)
                self._update_input_mode_status()
                return
            try:
                self._spacemouse_backend.open()
            except Exception as exc:
                self.btn_space_mouse.blockSignals(True)
                self.btn_space_mouse.setChecked(False)
                self.btn_space_mouse.blockSignals(False)
                self.input_mode_status_label.setText(f"SpaceMouse failed to open: {exc}")
                self._update_spacemouse_sample_display(None)
                return
            self._spacemouse_worker = SpaceMouseWorker(self._spacemouse_backend, sample_cb=self._send_spacemouse_sample)
            self._spacemouse_worker.start()
            self._send_manual_input_config()
            self.session.send_command(
                {
                    "type": "manual_input_enable",
                    "enabled": True,
                    "source": "spacemouse",
                }
            )
            self.spacemouse_timer.start(SPACEMOUSE_POLL_INTERVAL_MS)
        else:
            self.spacemouse_timer.stop()
            if self._spacemouse_worker is not None:
                self._spacemouse_worker.stop()
                self._spacemouse_worker.join(timeout=0.5)
                self._spacemouse_worker = None
            self._spacemouse_backend.close()
            self._update_spacemouse_sample_display(None)
            self.session.send_command(
                {
                    "type": "manual_input_enable",
                    "enabled": False,
                    "source": "spacemouse",
                }
            )
        self._update_input_mode_status()

    def _deactivate_spacemouse(self) -> None:
        if self.btn_space_mouse.isChecked():
            self.btn_space_mouse.setChecked(False)
        else:
            self.spacemouse_timer.stop()
            if self._spacemouse_worker is not None:
                self._spacemouse_worker.stop()
                self._spacemouse_worker.join(timeout=0.5)
                self._spacemouse_worker = None
            self._spacemouse_backend.close()
            self._update_spacemouse_sample_display(None)
            self.session.send_command(
                {
                    "type": "manual_input_enable",
                    "enabled": False,
                    "source": "spacemouse",
                }
            )
            self._update_input_mode_status()

    def _poll_spacemouse(self) -> None:
        if not self.btn_space_mouse.isChecked():
            return
        worker = self._spacemouse_worker
        if worker is None:
            return
        error_text = worker.latest_error()
        if error_text:
            self.input_mode_status_label.setText(f"SpaceMouse read failed: {error_text}")
            self._update_spacemouse_sample_display(None)
            self._deactivate_spacemouse()
            return
        sample = worker.latest_sample()
        if sample is None:
            return
        self._update_spacemouse_sample_display(sample)

    def send_home(self) -> None:
        self._deactivate_spacemouse()
        positions = [float(spin.value()) for spin in self.home_spins]
        self.session.send_command({"type": "home", "home_pos": positions, "units": "mm"})

    def send_homing_run(self) -> None:
        self._deactivate_spacemouse()
        self.session.send_command({"type": "homing_run", "mode": self._selected_homing_mode()})

    def send_homing_cancel(self) -> None:
        self._deactivate_spacemouse()
        self.session.send_command({"type": "homing_cancel"})

    def send_homing_apply(self) -> None:
        self._deactivate_spacemouse()
        self.session.send_command({"type": "homing_apply"})

    def send_pretension(self) -> None:
        self._deactivate_spacemouse()
        self.session.send_command(
            {
                "type": "pretension",
                "upper_N": float(self.pret_upper_spin.value()),
                "lower_N": float(self.pret_lower_spin.value()),
            }
        )

    def populate_jugglepath_dropdown(self) -> None:
        selected = self.jp_profile_combo.currentText()
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
        if selected:
            index = self.jp_profile_combo.findText(selected)
            if index >= 0:
                self.jp_profile_combo.setCurrentIndex(index)

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
        self._deactivate_spacemouse()

        try:
            profile_path = self._jugglepath_profiles_dir() / name
            traj = None
            try:
                profile = load_profile_yaml(str(profile_path))
                path, _cmd_hz = build_path_from_profile(profile)
                traj = path.build().traj
            except Exception:
                pattern = load_pattern_yaml(str(profile_path))
                traj, _cmd_hz = build_traj_from_pattern(pattern, hand="right", cycles=1)
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
            self.status_message_label.setText(f"JugglePath send failed: {name}")

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
        if self.plot_workspace.is_live_mode:
            self._mark_status_dirty()
            self._mark_plots_dirty()
            if self.robot_3d_view is not None:
                self.robot_3d_view.set_frame(frame)

    def update_plots(self) -> None:
        if not self._plots_dirty:
            return

        if self.plot_workspace.is_live_mode:
            visible_keys = self.plot_workspace.visible_channel_keys()
            snapshot = self.history.snapshot(visible_keys)
            times = snapshot.times
            series_by_key = snapshot.series_by_key
        else:
            if self.paused_snapshot is None:
                return
            visible_keys = self.plot_workspace.visible_channel_keys()
            times = self.paused_snapshot.times
            series_by_key = {key: self.paused_snapshot.series_by_key.get(key, []) for key in visible_keys}

        self.plot_workspace.render(times, series_by_key)
        self._plots_dirty = False

    def update_status_widgets(self) -> None:
        if not self._status_dirty:
            return

        if self.plot_workspace.is_live_mode:
            frame = self.session.latest_frame
        else:
            frame = self.paused_frame
        self._update_link_status_indicators(frame)
        if frame is None:
            self._update_homing_controls(None)
            self._update_state_button_feedback(None)
            self._status_dirty = False
            return

        self._update_pose_feedback_labels(frame)
        self._update_state_button_feedback(frame.control_state)
        self._update_homing_controls(frame)

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

        self._status_dirty = False

    def update_gui(self) -> None:
        self.update_status_widgets()
        self.update_plots()

    def _mark_plots_dirty(self) -> None:
        self._plots_dirty = True

    def _mark_status_dirty(self) -> None:
        self._status_dirty = True

    def _update_state_button_feedback(self, control_state: str | None) -> None:
        active = self._state_buttons_by_state.get(control_state or "")
        for button in self._all_state_buttons:
            button.setChecked(button is active)

    def _selected_homing_mode(self) -> str:
        mode = self.homing_mode_combo.currentData()
        return "manual" if mode is None else str(mode)

    def _set_homing_mode_selection(self, mode: str) -> None:
        index = self.homing_mode_combo.findData(str(mode))
        if index < 0 or index == self.homing_mode_combo.currentIndex():
            return
        self.homing_mode_combo.blockSignals(True)
        self.homing_mode_combo.setCurrentIndex(index)
        self.homing_mode_combo.blockSignals(False)

    def _on_homing_mode_changed(self, _index: int | None = None) -> None:
        self._update_homing_controls(None)
        self.session.send_command({"type": "homing_select", "mode": self._selected_homing_mode()})

    def _update_homing_controls(self, frame) -> None:
        homing = {} if frame is None else dict(getattr(frame, "homing", {}) or {})
        selected_mode = str(homing.get("selected_mode", self._selected_homing_mode()))
        self._set_homing_mode_selection(selected_mode)
        current_mode = self._selected_homing_mode()
        is_manual = current_mode == "manual"
        homing_state = str(homing.get("state", "idle"))
        result_available = bool(homing.get("result_available", False))
        routine_active = homing_state == "running"
        self.home_spool_label.setEnabled(is_manual)
        for spin in self.home_spins:
            spin.setEnabled(is_manual and not routine_active)
        self.btn_home.setEnabled(is_manual and not routine_active)
        self.btn_homing_run.setEnabled((not is_manual) and (not routine_active))
        self.btn_homing_cancel.setEnabled(routine_active)
        self.btn_homing_apply.setEnabled(result_available and not routine_active)
        self.homing_status_label.setText(self._format_homing_status_text(homing))

    def _format_homing_status_text(self, homing: dict[str, object]) -> str:
        selected_mode = self._homing_mode_label(str(homing.get("selected_mode", self._selected_homing_mode())))
        active_mode_raw = homing.get("active_mode")
        active_mode = None if active_mode_raw is None else self._homing_mode_label(str(active_mode_raw))
        state = str(homing.get("state", "idle"))
        phase = str(homing.get("phase", "idle"))
        progress = homing.get("progress")
        run_id = homing.get("run_id", 0)
        accepted = homing.get("accepted_samples", 0)
        rejected = homing.get("rejected_samples", 0)
        result_available = bool(homing.get("result_available", False))
        fitted_offset = homing.get("fitted_offset_mm")
        candidate_home = homing.get("candidate_home_pos_mm")
        residual_rms_mm = homing.get("residual_rms_mm")
        message = homing.get("message")
        failure_reason = homing.get("failure_reason")
        status_lines = [
            f"MODE {selected_mode}",
            f"STATE {state.upper()}  PHASE {phase}",
        ]
        if active_mode is not None:
            status_lines.append(f"ACTIVE {active_mode}  RUN {int(run_id) if run_id is not None else 0}")
        show_progress = (
            progress is not None
            and (
                state != "idle"
                or int(accepted or 0) > 0
                or int(rejected or 0) > 0
            )
        )
        if show_progress:
            progress_text = self._fmt(100.0 * float(progress), ".1f")
            status_lines.append(
                f"PROGRESS {progress_text}%  ACCEPTED {int(accepted or 0)}  REJECTED {int(rejected or 0)}"
            )
        if message:
            status_lines.append(str(message))
        if failure_reason:
            status_lines.append(f"REASON {failure_reason}")
        if result_available:
            fit_text = self._format_homing_axis_tuple(fitted_offset)
            candidate_text = self._format_homing_axis_tuple(candidate_home)
            status_lines.append(f"FIT {fit_text}")
            status_lines.append(f"HOME {candidate_text}")
            if residual_rms_mm is not None:
                status_lines.append(f"RMS {self._fmt(residual_rms_mm, '.4f')} mm")
        if not message and not failure_reason and state == "idle":
            status_lines.append("Manual HOME remains available in Manual mode.")
        return "\n".join(status_lines)

    @staticmethod
    def _homing_mode_label(mode: str) -> str:
        for label, value in HOMING_MODE_OPTIONS:
            if value == str(mode):
                return label
        return str(mode)

    def _format_homing_axis_tuple(self, values) -> str:
        if not isinstance(values, (list, tuple)) or len(values) < 6:
            return "---"
        return " ".join(
            f"A{index + 1}={self._fmt(values[index], '.3f')}"
            for index in range(6)
        )

    def _set_pose_feedback_cell(self, row: int, col: int, value, fmt: str) -> None:
        text = self._fmt(value, fmt)
        item = self.pose_feedback_table.item(row, col)
        if item is None:
            item = QTableWidgetItem(text)
            self.pose_feedback_table.setItem(row, col, item)
        else:
            item.setText(text)
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

    def _update_link_status_indicators(self, frame) -> None:
        tcp_up = self.session.command_status.startswith("Connected")
        udp_up = self.session.has_recent_telemetry(self.telem_timeout)
        self._set_status_badge(self.tcp_status_badge, "UP" if tcp_up else "DOWN", tcp_up)
        self._set_status_badge(self.udp_status_badge, "UP" if udp_up else "DOWN", udp_up)
        self.target_value_label.setText(self._target_text(frame))
        self.runtime_time_value_label.setText(self._fmt_time_s(None if frame is None else frame.runtime_time_s))
        sim_time_text = self._fmt_time_s(None if frame is None else frame.sim_time_s)
        self.sim_time_value_label.setText(sim_time_text if self._is_sim_frame(frame) else "--- s")

        can_up = False
        util_pct = float("nan")
        if frame is not None:
            util_pct = 100.0 * frame.comm_stats.can_util_est if math.isfinite(frame.comm_stats.can_util_est) else float("nan")
            can_up = any(
                math.isfinite(value)
                for value in (
                    frame.comm_stats.can_rx_hz,
                    frame.comm_stats.can_tx_hz,
                    frame.comm_stats.can_msg_hz,
                    frame.comm_stats.can_util_est,
                )
            )
        self._set_status_badge(self.can_status_badge, "UP" if can_up else "DOWN", can_up)
        self.can_util_value_label.setText(f"{self._fmt(util_pct, '.1f')}%" if can_up else "---%")

    def _target_text(self, frame) -> str:
        if self._is_sim_frame(frame):
            if frame is not None and frame.sim_rt_factor is not None and math.isfinite(frame.sim_rt_factor):
                return f"SIM ({self._fmt(frame.sim_rt_factor, '.2f')}x)"
            return "SIM"
        host = str(self.session.config.host).strip().lower()
        if host in {"127.0.0.1", "localhost"}:
            return "SIM"
        return "ROBOT"

    def _is_sim_frame(self, frame) -> bool:
        return bool(frame is not None and frame.sim_time_s is not None and math.isfinite(frame.sim_time_s))

    @staticmethod
    def _fmt(value, spec: str) -> str:
        try:
            numeric = float(value)
            return format(numeric, spec) if math.isfinite(numeric) else "---"
        except Exception:
            return "---"

    def _fmt_time_s(self, value) -> str:
        text = self._fmt(value, ".1f")
        return f"{text} s" if text != "---" else "--- s"

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
