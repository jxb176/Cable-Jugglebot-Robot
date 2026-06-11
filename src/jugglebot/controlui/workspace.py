"""Plot workspace widgets for the controller GUI."""

from __future__ import annotations

from collections.abc import Mapping
import math

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy, QVBoxLayout, QWidget
import pyqtgraph as pg
from pyqtgraph.graphicsItems.LegendItem import ItemSample

from .channels import (
    ChannelRegistry,
    PlotDefinition,
    PlotRegistry,
    PresetRegistry,
    build_default_plot_registry,
    build_default_workspace_presets,
    channel_keys_for_plot,
)
from .decimation import decimate_xy


PLOT_GRID_COLUMNS = 2
POSE_ROW_ORDER = ("position", "velocity", "acceleration")
POSE_COLUMN_ORDER = ("linear", "angular")
SPOOL_PLOT_ORDER = ("position", "velocity", "tension", "torque")
SPOOL_PLOT_LABELS = {
    "position": "Length",
    "velocity": "Velocity",
    "tension": "Tension",
    "torque": "Torque",
}
PLOT_MIN_HEIGHT = 140
DEFAULT_LIVE_DISPLAY_SECONDS = 5.0
LIVE_TOTAL_POINTS_PER_PIXEL = 1.0
DETAIL_TOTAL_POINTS_PER_PIXEL = 2.0
MIN_POINTS_PER_TRACE = 48

class PlotPanel(QWidget):
    channel_toggled = pyqtSignal(str, bool)
    channel_isolate_requested = pyqtSignal(str)
    select_all_requested = pyqtSignal(str)
    clear_requested = pyqtSignal(str)

    def __init__(self, spec: PlotDefinition, channels: ChannelRegistry, pen_factory, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._pen_factory = pen_factory
        self._channel_keys = channel_keys_for_plot(channels, spec.plot_id)
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._enabled_keys = {key: channels[key].default_visible for key in self._channel_keys}
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QHBoxLayout()
        title = QLabel(spec.title)
        header.addWidget(title)
        header.addStretch(1)
        self._all_button = QPushButton("All")
        self._all_button.clicked.connect(lambda: self.select_all_requested.emit(self.spec.plot_id))
        header.addWidget(self._all_button)
        self._none_button = QPushButton("None")
        self._none_button.clicked.connect(lambda: self.clear_requested.emit(self.spec.plot_id))
        header.addWidget(self._none_button)
        layout.addLayout(header)

        self.plot = pg.PlotWidget()
        self.plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plot.setLabel("bottom", "Time", "s")
        self.plot.setLabel("left", spec.y_label, spec.unit)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend(offset=(8, 8), sampleType=LegendSample)
        self.plot.setMinimumHeight(PLOT_MIN_HEIGHT)
        layout.addWidget(self.plot)

        for key in self._channel_keys:
            channel = channels[key]
            curve = self.plot.plot(name=channel.label, pen=self._pen_factory(key))
            curve._jugglebot_channel_key = key
            curve._jugglebot_toggle_callback = self._toggle_channel_from_legend
            curve._jugglebot_isolate_callback = self.channel_isolate_requested.emit
            curve.setVisible(channel.default_visible)
            curve.setCurveClickable(True, width=8)
            curve.sigClicked.connect(lambda _curve, ev, channel_key=key: self._on_curve_clicked(channel_key, ev))
            self._curves[key] = curve

    @property
    def channel_keys(self) -> tuple[str, ...]:
        return self._channel_keys

    def set_channel_checked(self, key: str, checked: bool) -> None:
        self._enabled_keys[key] = bool(checked)
        self._curves[key].setVisible(checked)

    def is_channel_enabled(self, key: str) -> bool:
        return self._enabled_keys[key]

    def has_enabled_channels(self) -> bool:
        return any(self._enabled_keys.values())

    def enabled_keys(self) -> tuple[str, ...]:
        return tuple(key for key in self._channel_keys if self._enabled_keys[key])

    def update_series(
        self,
        times: list[float],
        series_by_key: Mapping[str, list[float]],
        *,
        x_range: tuple[float, float] | None = None,
        total_point_budget: int | None = None,
    ) -> None:
        enabled_keys = self.enabled_keys()
        if not enabled_keys:
            return
        pixel_width = self._current_plot_pixel_width()
        per_trace_budget = None
        if total_point_budget is not None:
            per_trace_budget = max(MIN_POINTS_PER_TRACE, int(total_point_budget) // max(1, len(enabled_keys)))
        for key in enabled_keys:
            curve = self._curves[key]
            x_data, y_data = decimate_xy(
                times,
                series_by_key.get(key, ()),
                x_range=x_range,
                pixel_width=pixel_width,
                max_points=per_trace_budget,
            )
            curve.setData(x_data, y_data)

    def _current_plot_pixel_width(self) -> int:
        view_box = self.plot.getPlotItem().getViewBox()
        width = int(round(view_box.sceneBoundingRect().width()))
        return max(0, width)

    def current_x_range(self) -> tuple[float, float] | None:
        x_range = self.plot.getPlotItem().getViewBox().viewRange()[0]
        x_min = float(x_range[0])
        x_max = float(x_range[1])
        if not (math.isfinite(x_min) and math.isfinite(x_max) and x_max > x_min):
            return None
        return x_min, x_max

    def _on_curve_clicked(self, key: str, event) -> None:
        if event.button() != Qt.MouseButton.RightButton:
            return
        self.channel_isolate_requested.emit(key)
        event.accept()

    def _toggle_channel_from_legend(self, key: str) -> None:
        self.channel_toggled.emit(key, not self._enabled_keys[key])


class PlotWorkspace(QWidget):
    live_mode_changed = pyqtSignal(bool)
    configuration_changed = pyqtSignal()

    def __init__(
        self,
        channels: ChannelRegistry,
        plots: PlotRegistry | None = None,
        presets: PresetRegistry | None = None,
        pen_factory=None,
        live_display_seconds: float = DEFAULT_LIVE_DISPLAY_SECONDS,
        parent=None,
    ):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.channels = channels
        self.plots = plots or build_default_plot_registry()
        self.presets = presets or build_default_workspace_presets()
        self._pen_factory = pen_factory
        self._live_display_seconds = max(1.0, float(live_display_seconds))
        self._live_mode = True
        self._enabled_keys = {key: channel.default_visible for key, channel in channels.items()}
        self._active_plot_ids = {channel.plot_id for channel in channels.values() if channel.default_visible}
        self._syncing_x_range = False
        self._suppress_view_tracking = False
        self._shared_x_range: tuple[float, float] | None = None
        self._x_range_mode = "auto"
        self._last_auto_x_range: tuple[float, float] | None = None
        self._pose_row_enabled = {row: True for row in POSE_ROW_ORDER}
        self._pose_column_enabled = {column: True for column in POSE_COLUMN_ORDER}
        self._spool_plot_enabled = {plot_id: True for plot_id in SPOOL_PLOT_ORDER}
        self._active_preset_key: str | None = None

        layout = QVBoxLayout(self)
        self.toolbar_widget = QWidget(self)
        toolbar = QHBoxLayout(self.toolbar_widget)
        toolbar.setContentsMargins(0, 0, 0, 0)
        self.mode_label = QLabel("View: Live")
        toolbar.addWidget(self.mode_label)
        self.mode_button = QPushButton("Pause View")
        self.mode_button.clicked.connect(self.toggle_live_mode)
        toolbar.addWidget(self.mode_button)
        toolbar.addSpacing(12)
        toolbar.addWidget(QLabel("Presets:"))
        self.preset_buttons: dict[str, QPushButton] = {}
        for preset_key, preset in self.presets.items():
            button = QPushButton(preset.label)
            button.setCheckable(True)
            button.clicked.connect(lambda _checked=False, key=preset_key: self.apply_preset(key))
            self.preset_buttons[preset_key] = button
            toolbar.addWidget(button)

        self.pose_controls = QWidget(self)
        pose_controls_layout = QHBoxLayout(self.pose_controls)
        pose_controls_layout.setContentsMargins(0, 0, 0, 0)
        pose_controls_layout.setSpacing(6)
        pose_controls_layout.addWidget(QLabel("Pose rows:"))
        self.pose_row_buttons: dict[str, QPushButton] = {}
        for row in POSE_ROW_ORDER:
            button = QPushButton(row.capitalize())
            button.setCheckable(True)
            button.setChecked(True)
            button.toggled.connect(lambda checked, pose_row=row: self._set_pose_row_enabled(pose_row, checked))
            self.pose_row_buttons[row] = button
            pose_controls_layout.addWidget(button)
        pose_controls_layout.addSpacing(8)
        pose_controls_layout.addWidget(QLabel("Pose axes:"))
        self.pose_column_buttons: dict[str, QPushButton] = {}
        for column in POSE_COLUMN_ORDER:
            button = QPushButton(column.capitalize())
            button.setCheckable(True)
            button.setChecked(True)
            button.toggled.connect(lambda checked, pose_column=column: self._set_pose_column_enabled(pose_column, checked))
            self.pose_column_buttons[column] = button
            pose_controls_layout.addWidget(button)
        toolbar.addSpacing(12)
        toolbar.addWidget(self.pose_controls)

        self.spool_controls = QWidget(self)
        spool_controls_layout = QHBoxLayout(self.spool_controls)
        spool_controls_layout.setContentsMargins(0, 0, 0, 0)
        spool_controls_layout.setSpacing(6)
        spool_controls_layout.addWidget(QLabel("Spool plots:"))
        self.spool_plot_buttons: dict[str, QPushButton] = {}
        for plot_id in SPOOL_PLOT_ORDER:
            button = QPushButton(SPOOL_PLOT_LABELS[plot_id])
            button.setCheckable(True)
            button.setChecked(True)
            button.toggled.connect(lambda checked, spool_plot_id=plot_id: self._set_spool_plot_enabled(spool_plot_id, checked))
            self.spool_plot_buttons[plot_id] = button
            spool_controls_layout.addWidget(button)
        toolbar.addWidget(self.spool_controls)
        toolbar.addStretch(1)
        layout.addWidget(self.toolbar_widget)

        self.empty_label = QLabel("Enable channels or choose a preset to show plots.")
        self.empty_label.hide()
        layout.addWidget(self.empty_label)

        self.plots_container = QWidget(self)
        self.plots_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.plots_layout = QVBoxLayout(self.plots_container)
        self.plots_layout.setContentsMargins(0, 0, 0, 0)

        self.plots_grid = QGridLayout()
        self.plots_grid.setHorizontalSpacing(10)
        self.plots_grid.setVerticalSpacing(10)
        self.plots_layout.addLayout(self.plots_grid)

        self.plots_scroll = QScrollArea(self)
        self.plots_scroll.setWidgetResizable(True)
        self.plots_scroll.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.plots_scroll.setWidget(self.plots_container)
        layout.addWidget(self.plots_scroll, 1)

        self.plot_panels: dict[str, PlotPanel] = {}
        for plot_id, spec in self.plots.items():
            if not channel_keys_for_plot(self.channels, plot_id):
                continue
            panel = PlotPanel(spec, self.channels, pen_factory=self._pen_factory, parent=self.plots_container)
            panel.hide()
            panel.channel_toggled.connect(self._on_channel_toggled)
            panel.channel_isolate_requested.connect(self._isolate_plot_channel)
            panel.select_all_requested.connect(self._enable_plot_channels)
            panel.clear_requested.connect(self._disable_plot_channels)
            view_box = panel.plot.getPlotItem().getViewBox()
            view_box.sigRangeChangedManually.connect(
                lambda _mask, plot_key=plot_id: self._on_manual_range_changed(plot_key)
            )
            self.plot_panels[plot_id] = panel

        self.apply_preset("pose")

    @property
    def is_live_mode(self) -> bool:
        return self._live_mode

    def toggle_live_mode(self) -> None:
        self.set_live_mode(not self._live_mode)

    def set_live_mode(self, live: bool) -> None:
        live = bool(live)
        if self._live_mode == live:
            return
        self._live_mode = live
        if live:
            self.mode_label.setText("View: Live")
            self.mode_button.setText("Pause View")
            self._x_range_mode = "auto"
            self._enable_y_auto_range()
        else:
            self.mode_label.setText("View: Paused")
            self.mode_button.setText("Resume Live")
        self.live_mode_changed.emit(live)

    def visible_channel_keys(self) -> tuple[str, ...]:
        keys = []
        for panel in self.plot_panels.values():
            if not self._panel_is_visible(panel):
                continue
            keys.extend(panel.enabled_keys())
        return tuple(keys)

    def render(self, times: list[float], series_by_key: Mapping[str, list[float]]) -> None:
        visible_panels = [panel for panel in self.plot_panels.values() if self._panel_is_visible(panel)]
        live_auto_x_range = self._live_auto_x_range(times)
        if live_auto_x_range is not None:
            self._apply_x_range(None, live_auto_x_range)
        elif self._x_range_mode == "manual" and self._shared_x_range is not None:
            self._apply_x_range(None, self._shared_x_range)

        for panel in visible_panels:
            x_range = self._panel_render_x_range(panel, live_auto_x_range)
            panel.update_series(
                times,
                series_by_key,
                x_range=x_range,
                total_point_budget=self._panel_point_budget(panel),
            )
        if live_auto_x_range is not None:
            self._apply_x_range(None, live_auto_x_range)

    def apply_preset(self, preset_key: str) -> None:
        preset = self.presets[preset_key]
        self._set_active_preset(preset_key)
        allowed_plots = set(preset.plot_ids)
        self._active_plot_ids = set(allowed_plots)
        for key, channel in self.channels.items():
            enabled = channel.plot_id in allowed_plots
            self._enabled_keys[key] = enabled
            self.plot_panels[channel.plot_id].set_channel_checked(key, enabled)
        self._refresh_plot_layout()
        self.configuration_changed.emit()

    def _set_active_preset(self, preset_key: str | None) -> None:
        self._active_preset_key = preset_key
        for key, button in self.preset_buttons.items():
            button.blockSignals(True)
            button.setChecked(key == preset_key)
            button.blockSignals(False)

    def _enable_plot_channels(self, plot_id: str) -> None:
        panel = self.plot_panels[plot_id]
        self._active_plot_ids.add(plot_id)
        for key in panel.channel_keys:
            self._enabled_keys[key] = True
            panel.set_channel_checked(key, True)
        self._refresh_plot_layout()
        self.configuration_changed.emit()

    def _disable_plot_channels(self, plot_id: str) -> None:
        panel = self.plot_panels[plot_id]
        self._active_plot_ids.add(plot_id)
        for key in panel.channel_keys:
            self._enabled_keys[key] = False
            panel.set_channel_checked(key, False)
        self._refresh_plot_layout()
        self.configuration_changed.emit()

    def _on_channel_toggled(self, key: str, checked: bool) -> None:
        self._enabled_keys[key] = bool(checked)
        plot_id = self.channels[key].plot_id
        self._active_plot_ids.add(plot_id)
        panel = self.plot_panels[plot_id]
        panel.set_channel_checked(key, checked)
        self._refresh_plot_layout()
        self.configuration_changed.emit()

    def _isolate_plot_channel(self, key: str) -> None:
        plot_id = self.channels[key].plot_id
        self._active_plot_ids.add(plot_id)
        panel = self.plot_panels[plot_id]
        for channel_key in panel.channel_keys:
            enabled = channel_key == key
            self._enabled_keys[channel_key] = enabled
            panel.set_channel_checked(channel_key, enabled)
        self._refresh_plot_layout()
        self.configuration_changed.emit()

    def _set_pose_row_enabled(self, row: str, enabled: bool) -> None:
        self._pose_row_enabled[row] = bool(enabled)
        self._refresh_plot_layout()
        self.configuration_changed.emit()

    def _set_pose_column_enabled(self, column: str, enabled: bool) -> None:
        self._pose_column_enabled[column] = bool(enabled)
        self._refresh_plot_layout()
        self.configuration_changed.emit()

    def _set_spool_plot_enabled(self, plot_id: str, enabled: bool) -> None:
        self._spool_plot_enabled[plot_id] = bool(enabled)
        self._refresh_plot_layout()
        self.configuration_changed.emit()

    def _set_all_pose_rows(self, enabled: bool, *, emit: bool) -> None:
        for row, button in self.pose_row_buttons.items():
            self._pose_row_enabled[row] = bool(enabled)
            button.blockSignals(True)
            button.setChecked(bool(enabled))
            button.blockSignals(False)
        if emit:
            self._refresh_plot_layout()
            self.configuration_changed.emit()

    def _set_all_pose_columns(self, enabled: bool, *, emit: bool) -> None:
        for column, button in self.pose_column_buttons.items():
            self._pose_column_enabled[column] = bool(enabled)
            button.blockSignals(True)
            button.setChecked(bool(enabled))
            button.blockSignals(False)
        if emit:
            self._refresh_plot_layout()
            self.configuration_changed.emit()

    def _set_all_spool_plots(self, enabled: bool, *, emit: bool) -> None:
        for plot_id, button in self.spool_plot_buttons.items():
            self._spool_plot_enabled[plot_id] = bool(enabled)
            button.blockSignals(True)
            button.setChecked(bool(enabled))
            button.blockSignals(False)
        if emit:
            self._refresh_plot_layout()
            self.configuration_changed.emit()

    def _panel_is_visible(self, panel: PlotPanel) -> bool:
        if panel.spec.plot_id not in self._active_plot_ids:
            return False
        if panel.spec.plot_id in SPOOL_PLOT_ORDER:
            return self._spool_plot_enabled.get(panel.spec.plot_id, True)
        if panel.spec.domain == "pose" and panel.spec.pose_row is not None:
            if not self._pose_row_enabled.get(panel.spec.pose_row, True):
                return False
            if panel.spec.pose_column is not None:
                return self._pose_column_enabled.get(panel.spec.pose_column, True)
            return True
        return True

    def _refresh_plot_layout(self) -> None:
        self._suppress_view_tracking = True
        try:
            for panel in self.plot_panels.values():
                panel.hide()
            while self.plots_grid.count():
                item = self.plots_grid.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.hide()

            visible_panels = [panel for panel in self.plot_panels.values() if self._panel_is_visible(panel)]
            visible_panels.sort(key=lambda panel: panel.spec.sort_order)
            self.empty_label.setVisible(not visible_panels)

            pose_panels_present = any(
                panel.spec.domain == "pose" and panel.spec.plot_id in self._active_plot_ids
                for panel in self.plot_panels.values()
            )
            spool_panels_present = any(plot_id in self._active_plot_ids for plot_id in SPOOL_PLOT_ORDER)
            self.pose_controls.setVisible(self._active_preset_key == "pose")
            self.spool_controls.setVisible(self._active_preset_key == "spools")
            for button in self.pose_row_buttons.values():
                button.setEnabled(pose_panels_present)
            for button in self.pose_column_buttons.values():
                button.setEnabled(pose_panels_present)
            for button in self.spool_plot_buttons.values():
                button.setEnabled(spool_panels_present)

            next_row = 0
            pose_panels = [panel for panel in visible_panels if panel.spec.domain == "pose"]
            spool_panels = [panel for panel in visible_panels if panel.spec.plot_id in SPOOL_PLOT_ORDER]
            other_panels = [panel for panel in visible_panels if panel not in pose_panels and panel not in spool_panels]

            for index, panel in enumerate(pose_panels):
                row = next_row + (index // PLOT_GRID_COLUMNS)
                col = index % PLOT_GRID_COLUMNS
                self.plots_grid.addWidget(panel, row, col)
                panel.show()
            if pose_panels:
                next_row += (len(pose_panels) + PLOT_GRID_COLUMNS - 1) // PLOT_GRID_COLUMNS

            for panel in spool_panels:
                self.plots_grid.addWidget(panel, next_row, 0, 1, PLOT_GRID_COLUMNS)
                panel.show()
                next_row += 1

            for index, panel in enumerate(other_panels):
                row = next_row + (index // PLOT_GRID_COLUMNS)
                col = index % PLOT_GRID_COLUMNS
                self.plots_grid.addWidget(panel, row, col)
                panel.show()

            if visible_panels:
                if self._x_range_mode == "manual" and self._shared_x_range is not None:
                    self._apply_x_range(None, self._shared_x_range)
        finally:
            self._suppress_view_tracking = False

    def _sync_x_range(self, source_plot_id: str, x_range) -> None:
        if self._syncing_x_range or self._suppress_view_tracking:
            return
        self._x_range_mode = "manual"
        self._capture_shared_x_range(x_range)
        should_pause = self._live_mode
        if should_pause:
            self._live_mode = False
            self.mode_label.setText("View: Paused")
            self.mode_button.setText("Resume Live")
        self._apply_x_range(source_plot_id, x_range)
        if should_pause:
            self.live_mode_changed.emit(False)
        self.configuration_changed.emit()

    def _apply_x_range(self, source_plot_id: str | None, x_range) -> None:
        self._syncing_x_range = True
        try:
            x_min, x_max = float(x_range[0]), float(x_range[1])
            for plot_id, panel in self.plot_panels.items():
                if plot_id == source_plot_id or not self._panel_is_visible(panel):
                    continue
                view_box = panel.plot.getPlotItem().getViewBox()
                view_box.enableAutoRange(axis="x", enable=False)
                view_box.setXRange(x_min, x_max, padding=0.0)
        finally:
            self._syncing_x_range = False

    def _capture_shared_x_range(self, x_range) -> None:
        x_min = float(x_range[0])
        x_max = float(x_range[1])
        if not (math.isfinite(x_min) and math.isfinite(x_max) and x_max > x_min):
            return
        self._shared_x_range = (x_min, x_max)

    def _on_manual_range_changed(self, plot_id: str) -> None:
        panel = self.plot_panels.get(plot_id)
        if panel is None:
            return
        current_range = panel.current_x_range()
        if current_range is None:
            return
        self._sync_x_range(plot_id, current_range)

    def _live_auto_x_range(self, times: list[float]) -> tuple[float, float] | None:
        if not self._live_mode or self._x_range_mode != "auto" or not times:
            self._last_auto_x_range = None
            return None
        x_max = float(times[-1])
        x_min = max(float(times[0]), x_max - self._live_display_seconds)
        live_range = (x_min, x_max)
        self._last_auto_x_range = live_range
        return live_range

    def _panel_render_x_range(
        self,
        panel: PlotPanel,
        live_auto_x_range: tuple[float, float] | None,
    ) -> tuple[float, float] | None:
        if live_auto_x_range is not None:
            return live_auto_x_range
        if self._x_range_mode == "manual" and self._shared_x_range is not None:
            return self._shared_x_range
        return None

    def _panel_point_budget(self, panel: PlotPanel) -> int | None:
        pixel_width = panel._current_plot_pixel_width()
        if pixel_width <= 0:
            return None
        if self._live_mode and self._x_range_mode == "auto":
            scale = LIVE_TOTAL_POINTS_PER_PIXEL
        else:
            scale = DETAIL_TOTAL_POINTS_PER_PIXEL
        return max(MIN_POINTS_PER_TRACE, int(round(pixel_width * scale)))

    def _enable_y_auto_range(self) -> None:
        self._suppress_view_tracking = True
        try:
            for panel in self.plot_panels.values():
                view_box = panel.plot.getPlotItem().getViewBox()
                view_box.enableAutoRange(axis="y", enable=True)
                view_box.updateAutoRange()
        finally:
            self._suppress_view_tracking = False

def _ranges_close(left: tuple[float, float], right: tuple[float, float], *, tol: float = 1e-6) -> bool:
    return abs(left[0] - right[0]) <= tol and abs(left[1] - right[1]) <= tol


class LegendSample(ItemSample):
    def mouseClickEvent(self, event):
        key = getattr(self.item, "_jugglebot_channel_key", None)
        if key is None:
            super().mouseClickEvent(event)
            return

        if event.button() == Qt.MouseButton.RightButton:
            isolate_callback = getattr(self.item, "_jugglebot_isolate_callback", None)
            if callable(isolate_callback):
                isolate_callback(key)
            event.accept()
            self.update()
            return

        if event.button() == Qt.MouseButton.LeftButton:
            toggle_callback = getattr(self.item, "_jugglebot_toggle_callback", None)
            if callable(toggle_callback):
                toggle_callback(key)
            event.accept()
            self.update()
            return

        super().mouseClickEvent(event)
