"""3D robot visualization for the controller GUI."""

from __future__ import annotations

import numpy as np
from PyQt6.QtOpenGL import QOpenGLWindow
from PyQt6.QtWidgets import QVBoxLayout, QWidget
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLViewWidget import GLViewMixin

from jugglebot.core.cable_ik import CableRobotGeometry

from .models import TelemetryFrame
from .scene_geometry import SceneGeometry, cable_segments, edge_segments, pose_center_mm, pose_to_world_points_mm


def _rgba8(r: int, g: int, b: int, a: int) -> tuple[float, float, float, float]:
    return (r / 255.0, g / 255.0, b / 255.0, a / 255.0)


class _Robot3DWindow(GLViewMixin, QOpenGLWindow):
    def __init__(self, parent=None):
        super().__init__(QOpenGLWindow.UpdateBehavior.NoPartialUpdate, parent)
        self.geometry_model = SceneGeometry.from_robot_geometry(CableRobotGeometry())
        self.setBackgroundColor((245, 247, 250))
        self.setCameraPosition(distance=1800.0, elevation=16.0, azimuth=38.0)

        self._grid = gl.GLGridItem(color=(0.7, 0.73, 0.76, 0.35))
        self._grid.setSize(1200.0, 1200.0, 0.0)
        self._grid.setSpacing(100.0, 100.0, 100.0)
        self._grid.translate(0.0, 0.0, -500.0)
        self.addItem(self._grid)

        self._axis = gl.GLAxisItem()
        self._axis.setSize(x=160.0, y=160.0, z=160.0)
        self.addItem(self._axis)

        anchors = self.geometry_model.anchors_world_mm
        self._anchor_scatter = gl.GLScatterPlotItem(
            pos=anchors,
            color=np.asarray([_rgba8(150, 156, 164, 255)] * len(anchors), dtype=float),
            size=12.0,
            pxMode=True,
        )
        self.addItem(self._anchor_scatter)

        self._anchor_frame = gl.GLLinePlotItem(
            pos=edge_segments(anchors, self.geometry_model.anchor_edge_pairs),
            color=_rgba8(130, 138, 148, 140),
            width=2.0,
            antialias=True,
            mode="lines",
        )
        self.addItem(self._anchor_frame)

        self._cmd_cables = gl.GLLinePlotItem(color=_rgba8(242, 155, 54, 150), width=2.0, antialias=True, mode="lines")
        self._cmd_platform_edges = gl.GLLinePlotItem(color=_rgba8(242, 155, 54, 230), width=3.0, antialias=True, mode="lines")
        self._cmd_center = gl.GLScatterPlotItem(size=10.0, color=np.asarray([_rgba8(242, 155, 54, 255)], dtype=float), pxMode=True)

        self._est_cables = gl.GLLinePlotItem(color=_rgba8(0, 153, 204, 220), width=3.0, antialias=True, mode="lines")
        self._est_platform_edges = gl.GLLinePlotItem(color=_rgba8(0, 153, 204, 255), width=3.0, antialias=True, mode="lines")
        self._est_center = gl.GLScatterPlotItem(size=10.0, color=np.asarray([_rgba8(0, 153, 204, 255)], dtype=float), pxMode=True)
        self._est_platform_mesh = gl.GLMeshItem(
            smooth=False,
            drawEdges=False,
            shader="shaded",
            glOptions="translucent",
            color=_rgba8(0, 153, 204, 70),
        )
        self._est_platform_mesh.setVisible(False)

        for item in (
            self._cmd_cables,
            self._cmd_platform_edges,
            self._cmd_center,
            self._est_cables,
            self._est_platform_edges,
            self._est_center,
            self._est_platform_mesh,
        ):
            self.addItem(item)

    def set_frame(self, frame: TelemetryFrame) -> None:
        commanded = pose_to_world_points_mm(frame.hand_cmd_pose, self.geometry_model)
        estimated = pose_to_world_points_mm(frame.hand_est_pose, self.geometry_model)
        command_center = pose_center_mm(frame.hand_cmd_pose)
        estimate_center = pose_center_mm(frame.hand_est_pose)

        self._cmd_cables.setData(pos=cable_segments(self.geometry_model.anchors_world_mm, commanded))
        self._cmd_platform_edges.setData(pos=edge_segments(commanded, self.geometry_model.platform_edge_pairs))
        self._set_center_point(self._cmd_center, command_center)

        self._est_cables.setData(pos=cable_segments(self.geometry_model.anchors_world_mm, estimated))
        self._est_platform_edges.setData(pos=edge_segments(estimated, self.geometry_model.platform_edge_pairs))
        self._set_center_point(self._est_center, estimate_center)

        if estimated is None:
            self._est_platform_mesh.setVisible(False)
        else:
            self._est_platform_mesh.setMeshData(vertexes=estimated, faces=self.geometry_model.platform_faces)
            self._est_platform_mesh.setVisible(True)

    @staticmethod
    def _set_center_point(item: gl.GLScatterPlotItem, center: np.ndarray | None) -> None:
        if center is None:
            item.setData(pos=np.empty((0, 3), dtype=float))
            return
        item.setData(pos=np.asarray([center], dtype=float))


class Robot3DView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self._window = _Robot3DWindow()
        self._container = QWidget.createWindowContainer(self._window, self)
        self._container.setMinimumHeight(360)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._container)
        self.setMinimumHeight(360)

    def set_frame(self, frame: TelemetryFrame) -> None:
        self._window.set_frame(frame)
