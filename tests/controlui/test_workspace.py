from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")

from PyQt6.QtWidgets import QApplication, QPushButton
import pyqtgraph as pg

from jugglebot.controlui.channels import build_default_channel_registry
from jugglebot.controlui.workspace import PlotWorkspace


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_resume_live_reenables_y_auto_range() -> None:
    app = _app()
    workspace = PlotWorkspace(
        build_default_channel_registry(),
        pen_factory=lambda _key: pg.mkPen("w"),
        live_display_seconds=5.0,
    )
    workspace.show()
    app.processEvents()

    workspace.set_live_mode(False)
    panel = workspace.plot_panels["pose_translation"]
    view_box = panel.plot.getPlotItem().getViewBox()
    view_box.setYRange(-1.0, 1.0, padding=0.0)
    assert view_box.autoRangeEnabled()[1] is False

    workspace.set_live_mode(True)

    for panel in workspace.plot_panels.values():
        y_auto = panel.plot.getPlotItem().getViewBox().autoRangeEnabled()[1]
        assert y_auto is not False

    workspace.close()
    app.processEvents()


def test_manual_x_range_change_switches_workspace_to_paused() -> None:
    app = _app()
    workspace = PlotWorkspace(
        build_default_channel_registry(),
        pen_factory=lambda _key: pg.mkPen("w"),
        live_display_seconds=5.0,
    )
    workspace.show()
    app.processEvents()

    assert workspace.is_live_mode is True

    workspace._sync_x_range("pose_translation", (10.0, 12.0))

    assert workspace.is_live_mode is False
    assert workspace._x_range_mode == "manual"
    assert workspace._shared_x_range == (10.0, 12.0)
    assert workspace.mode_label.text() == "View: Paused"
    assert workspace.mode_button.text() == "Resume Live"

    workspace.close()
    app.processEvents()


def test_live_render_updates_do_not_leave_auto_follow() -> None:
    app = _app()
    workspace = PlotWorkspace(
        build_default_channel_registry(),
        pen_factory=lambda _key: pg.mkPen("w"),
        live_display_seconds=5.0,
    )
    workspace.show()
    app.processEvents()

    times = [0.1 * idx for idx in range(100)]
    series = {key: [0.0 for _ in times] for key in workspace.visible_channel_keys()}

    for _ in range(5):
        workspace.render(times, series)
        app.processEvents()

    assert workspace.is_live_mode is True
    assert workspace._x_range_mode == "auto"
    assert workspace.mode_button.text() == "Pause View"

    workspace.close()
    app.processEvents()


def test_legend_toggle_updates_channel_visibility() -> None:
    app = _app()
    workspace = PlotWorkspace(
        build_default_channel_registry(),
        pen_factory=lambda _key: pg.mkPen("w"),
        live_display_seconds=5.0,
    )
    workspace.show()
    app.processEvents()

    panel = workspace.plot_panels["pose_translation"]
    key = panel.channel_keys[0]
    assert panel.is_channel_enabled(key) is True

    panel._toggle_channel_from_legend(key)
    app.processEvents()
    assert panel.is_channel_enabled(key) is False

    panel._toggle_channel_from_legend(key)
    app.processEvents()
    assert panel.is_channel_enabled(key) is True

    workspace.close()
    app.processEvents()


def test_plot_panel_only_exposes_plot_level_buttons() -> None:
    app = _app()
    workspace = PlotWorkspace(
        build_default_channel_registry(),
        pen_factory=lambda _key: pg.mkPen("w"),
        live_display_seconds=5.0,
    )
    workspace.show()
    app.processEvents()

    panel = workspace.plot_panels["pose_translation"]
    buttons = [button.text() for button in panel.findChildren(QPushButton)]
    assert buttons == ["All", "None"]

    workspace.close()
    app.processEvents()
