"""Compatibility layer for legacy robot-server imports.

The real-time runtime now lives in :mod:`jugglebot.rt.runner`.
"""

from __future__ import annotations

import threading
import time

from jugglebot.core.state import RuntimeMailbox
from jugglebot.rt.runner import (
    AXIS_NODE_IDS,
    LOG_FILE_PATH,
    ODRIVE_BITRATE,
    ODRIVE_COMMAND_RATE_HZ,
    ODRIVE_INTERFACE,
    ODRIVE_LOG_RATE_HZ,
    ControlBridge,
    ensure_can_interface_up,
    logger,
)

__all__ = [
    "AXIS_NODE_IDS",
    "ControlBridge",
    "LOG_FILE_PATH",
    "ODRIVE_BITRATE",
    "ODRIVE_COMMAND_RATE_HZ",
    "ODRIVE_INTERFACE",
    "ODRIVE_LOG_RATE_HZ",
    "ensure_can_interface_up",
    "logger",
]


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
