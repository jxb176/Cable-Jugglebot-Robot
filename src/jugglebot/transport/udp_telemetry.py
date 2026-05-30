"""UDP telemetry publisher for the robot command server."""

from __future__ import annotations

import json
import logging
import time

from jugglebot.core.snapshots import build_robot_state_snapshot
from jugglebot.transport.config import TELEMETRY_RATE_HZ, UDP_TELEM_PORT


logger = logging.getLogger("robot")


def udp_telemetry_sender(state, udp_sock, stop_event):
    while not stop_event.is_set():
        try:
            controller_ip = state.get_controller_ip()
            if controller_ip:
                controller_addr = (controller_ip, UDP_TELEM_PORT)
                snapshot = build_robot_state_snapshot(state, timestamp_s=time.time())
                msg = snapshot.to_dict()
                udp_sock.sendto(json.dumps(msg).encode("utf-8"), controller_addr)
        except Exception as e:
            logger.error(f"[UDP] Error sending telemetry: {e}")
        time.sleep(1.0 / TELEMETRY_RATE_HZ)
