"""Plot workspace widgets for the controller GUI."""

from __future__ import annotations

from collections.abc import Mapping

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget
import pyqtgraph as pg

from .channels import (
    ChannelRegistry,
    PlotDefinition,
    PlotRegistry,
    PresetRegistry,
    build_default_plot_registry,
    build_default_workspace_presets,
    channel_keys_for_plot,
)


PLOT_GRID_COLUMNS = 3
CHANNEL_BUTTON_COLUMNS = 4


def _button_style(color: tuple[int, int, int]) -> str:
    r, g, b = color
    return (
        "QPushButton {"
        f" border: 1px solid rgb({r}, {g}, {b});"
        " border-radius: 10px;"
        " padding: 4px 8px;"
        " background-color: transparent;"
        "}"
        "QPushButton:checked {"
        f" background-color: rgba({r}, {g}, {b}, 90);"
        "}"
    )


class PlotPanel(QWidget):
    channel_toggled = pyqtSignal(str, bool)
    select_all_requested = pyqtSignal(str)
    clear_requested = pyqtSignal(str)

    def __init__(self, spec: PlotDefinition, channels: ChannelRegistry, pen_factory, parent=None):
        super().__init__(parent)
        self.spec = spec
        self._pen_factory = pen_factory
        self._channel_keys = channel_keys_for_plot(channels, spec.plot_id)
        self._curves: dict[str, pg.PlotDataItem] = {}
        self._buttons: dict[str, QPushButton] = {}

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

        button_grid = QGridLayout()
        button_grid.setHorizontalSpacing(6)
        button_grid.setVerticalSpacing(6)
        for index, key in enumerate(self._channel_keys):
            channel = channels[key]
            button = QPushButton(channel.label)
            button.setCheckable(True)
            button.setChecked(channel.default_visible)
            button.setStyleSheet(_button_style(channel.color))
            button.toggled.connect(lambda checked, channel_key=key: self.channel_toggled.emit(channel_key, checked))
            self._buttons[key] = button
            button_grid.addWidget(button, index // CHANNEL_BUTTON_COLUMNS, index % CHANNEL_BUTTON_COLUMNS)
        layout.addLayout(button_grid)

        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Time", "s")
        self.plot.setLabel("left", spec.y_label, spec.unit)
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.addLegend(offset=(8, 8))
        self.plot.setMinimumHeight(220)
        layout.addWidget(self.plot)

        for key in self._channel_keys:
            channel = channels[key]
            curve = self.plot.plot(name=channel.label, pen=self._pen_factory(key))
            curve.setVisible(channel.default_visible)
            self._curves[key] = curve

    @property
    def channel_keys(self) -> tuple[str, ...]:
        return self._channel_keys

    def set_channel_checked(self, key: str, checked: bool) -> None:
        button = self._buttons[key]
        button.blockSignals(True)
        button.setChecked(checked)
        button.blockSignals(False)
        self._curves[key].setVisible(checked)

    def is_channel_enabled(self, key: str) -> bool:
        return self._buttons[key].isChecked()

    def has_enabled_channels(self) -> bool:
        return any(button.isChecked() for button in self._buttons.values())

    def enabled_keys(self) -> tuple[str, ...]:
        return tuple(key for key in self._channel_keys if self._buttons[key].isChecked())

    def update_series(self, times: list[float], series_by_key: Mapping[str, list[float]]) -> None:
        for key, curve in self._curves.items():
            if not self._buttons[key].isChecked():
                continue
            curve.setData(times, series_by_key.get(key, ()))


class PlotWorkspace(QWidget):
    live_mode_changed = pyqtSignal(bool)
    configuration_changed = pyqtSignal()

    def __init__(
        self,
        channels: ChannelRegistry,
        plots: PlotRegistry | None = None,
        presets: PresetRegistry | None = None,
        pen_factory=None,
        parent=None,
    ):
        super().__init__(parent)
        self.channels = channels
        self.plots = plots or build_default_plot_registry()
        self.presets = presets or build_default_workspace_presets()
        self._pen_factory = pen_factory
        self._live_mode = True
        self._enabled_keys = {key: channel.default_visible for key, channel in channels.items()}

        layout = QVBoxLayout(self)

        toolbar = QHBoxLayout()
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
            button.clicked.connect(lambda _checked=False, key=preset_key: self.apply_preset(key))
            self.preset_buttons[preset_key] = button
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        self.empty_label = QLabel("Enable channels or choose a preset to show plots.")
        self.empty_label.hide()
        layout.addWidget(self.empty_label)

        self.plots_grid = QGridLayout()
        self.plots_grid.setHorizontalSpacing(10)
        self.plots_grid.setVerticalSpacing(10)
        layout.addLayout(self.plots_grid)

        self.plot_panels: dict[str, PlotPanel] = {}
        for plot_id, spec in self.plots.items():
            if not channel_keys_for_plot(self.channels, plot_id):
                continue
            panel = PlotPanel(spec, self.channels, pen_factory=self._pen_factory, parent=self)
            panel.channel_toggled.connect(self._on_channel_toggled)
            panel.select_all_requested.connect(self._enable_plot_channels)
            panel.clear_requested.connect(self._disable_plot_channels)
            self.plot_panels[plot_id] = panel

        self._refresh_plot_layout()

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
        else:
            self.mode_label.setText("View: Paused")
            self.mode_button.setText("Resume Live")
        self.live_mode_changed.emit(live)

    def visible_channel_keys(self) -> tuple[str, ...]:
        keys = []
        for panel in self.plot_panels.values():
            if not panel.isVisible():
                continue
            keys.extend(panel.enabled_keys())
        return tuple(keys)

    def render(self, times: list[float], series_by_key: Mapping[str, list[float]]) -> None:
        for panel in self.plot_panels.values():
            if panel.isVisible():
                panel.update_series(times, series_by_key)

    def apply_preset(self, preset_key: str) -> None:
        preset = self.presets[preset_key]
        allowed_plots = set(preset.plot_ids)
        for key, channel in self.channels.items():
            enabled = channel.plot_id in allowed_plots and channel.default_visible
            self._enabled_keys[key] = enabled
            self.plot_panels[channel.plot_id].set_channel_checked(key, enabled)
        self._refresh_plot_layout()
        self.configuration_changed.emit()

    def _enable_plot_channels(self, plot_id: str) -> None:
        panel = self.plot_panels[plot_id]
        for key in panel.channel_keys:
            self._enabled_keys[key] = True
            panel.set_channel_checked(key, True)
        self._refresh_plot_layout()
        self.configuration_changed.emit()

    def _disable_plot_channels(self, plot_id: str) -> None:
        panel = self.plot_panels[plot_id]
        for key in panel.channel_keys:
            self._enabled_keys[key] = False
            panel.set_channel_checked(key, False)
        self._refresh_plot_layout()
        self.configuration_changed.emit()

    def _on_channel_toggled(self, key: str, checked: bool) -> None:
        self._enabled_keys[key] = bool(checked)
        panel = self.plot_panels[self.channels[key].plot_id]
        panel.set_channel_checked(key, checked)
        self._refresh_plot_layout()
        self.configuration_changed.emit()

    def _refresh_plot_layout(self) -> None:
        while self.plots_grid.count():
            item = self.plots_grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.hide()

        visible_panels = [panel for panel in self.plot_panels.values() if panel.has_enabled_channels()]
        self.empty_label.setVisible(not visible_panels)

        for index, panel in enumerate(visible_panels):
            row = index // PLOT_GRID_COLUMNS
            col = index % PLOT_GRID_COLUMNS
            self.plots_grid.addWidget(panel, row, col)
            panel.show()
