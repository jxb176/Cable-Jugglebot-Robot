"""Application entry point for the controller GUI."""

from __future__ import annotations

import argparse
import os
import sys

from .channels import build_default_channel_registry
from .history import TelemetryHistory
from .models import SessionConfig
from .session import LiveRobotSession


def _configure_qt_graphics() -> None:
    from PyQt6.QtCore import Qt
    from PyQt6.QtGui import QSurfaceFormat
    from PyQt6.QtWidgets import QApplication

    mode = os.environ.get("JUGGLEBOT_QT_OPENGL", "desktop").strip().lower()
    if mode == "software":
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseSoftwareOpenGL, True)
    elif mode == "desktop":
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseDesktopOpenGL, True)

    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.RenderableType.OpenGL)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
    fmt.setVersion(2, 1)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    fmt.setDepthBufferSize(24)
    fmt.setStencilBufferSize(8)
    fmt.setSamples(0)
    QSurfaceFormat.setDefaultFormat(fmt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Jugglebot network control interface")
    parser.add_argument("--host", type=str, default=os.environ.get("JUGGLEBOT_HOST", "jugglepi.local"), help="Robot host/IP")
    parser.add_argument("--tcp-port", type=int, default=int(os.environ.get("JUGGLEBOT_TCP_PORT", "5555")), help="Robot TCP command port")
    parser.add_argument("--udp-port", type=int, default=int(os.environ.get("JUGGLEBOT_UDP_PORT", "5556")), help="Local UDP telemetry listen port")
    parser.add_argument("--history-seconds", type=float, default=20.0, help="Telemetry history window for plots")
    args = parser.parse_args()

    try:
        from PyQt6.QtWidgets import QApplication
        from .window import RobotControlWindow
    except ImportError as exc:
        print("GUI dependencies are missing. Install with: pip install -e .[gui]", file=sys.stderr)
        raise SystemExit(1) from exc

    _configure_qt_graphics()

    channels = build_default_channel_registry()
    history = TelemetryHistory(channels, history_seconds=args.history_seconds)
    session = LiveRobotSession(
        SessionConfig(host=args.host, tcp_port=args.tcp_port, udp_port=args.udp_port),
        history,
    )
    session.start()

    app = QApplication(sys.argv)
    app.aboutToQuit.connect(session.stop)
    window = RobotControlWindow(session=session, channels=channels)
    window.show()
    sys.exit(app.exec())
