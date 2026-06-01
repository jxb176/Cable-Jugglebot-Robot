"""Compatibility layer for legacy robot-server imports.

The real-time runtime now lives in :mod:`jugglebot.rt.runner`.
"""

from __future__ import annotations

from jugglebot.rt.runner import (
    LOG_FILE_PATH,
    ControlBridge,
    ensure_can_interface_up,
    logger,
)

__all__ = [
    "ControlBridge",
    "LOG_FILE_PATH",
    "ensure_can_interface_up",
    "logger",
]
