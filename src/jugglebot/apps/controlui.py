#!/usr/bin/env python3
"""Launch the prototype network control interface as a packaged app."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import threading
from pathlib import Path
from queue import Queue


def _load_prototype_module():
    repo_root = Path(__file__).resolve().parents[3]
    proto_path = repo_root / "control interface prototype" / "control_interface.py"
    if not proto_path.exists():
        raise FileNotFoundError(f"Prototype control interface not found: {proto_path}")

    spec = importlib.util.spec_from_file_location("jugglebot_control_interface_proto", proto_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec from {proto_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    parser = argparse.ArgumentParser(description="Jugglebot network control interface (prototype)")
    parser.add_argument("--host", type=str, default=os.environ.get("JUGGLEBOT_HOST", "jugglepi.local"), help="Robot host/IP")
    parser.add_argument("--tcp-port", type=int, default=int(os.environ.get("JUGGLEBOT_TCP_PORT", "5555")), help="Robot TCP command port")
    parser.add_argument("--udp-port", type=int, default=int(os.environ.get("JUGGLEBOT_UDP_PORT", "5556")), help="Local UDP telemetry listen port")
    args = parser.parse_args()

    try:
        mod = _load_prototype_module()
    except Exception as exc:
        print(f"Error loading control interface prototype: {exc}", file=sys.stderr)
        sys.exit(1)

    # Override prototype defaults with CLI/runtime configuration.
    mod.ROBOT_HOST = args.host
    mod.TCP_CMD_PORT = int(args.tcp_port)
    mod.UDP_TELEM_PORT = int(args.udp_port)

    cmd_queue = Queue(maxsize=1)
    telem_queue = Queue()

    telem_thread = threading.Thread(
        target=mod.telemetry_listener,
        args=(mod.UDP_TELEM_PORT, telem_queue, lambda s: print(s)),
        daemon=True,
    )
    telem_thread.start()

    cmd_client = mod.CommandClient(
        mod.ROBOT_HOST,
        mod.TCP_CMD_PORT,
        cmd_queue,
        status_cb=lambda s: print(s),
    )
    cmd_client.start()

    app = mod.QApplication(sys.argv)
    app.aboutToQuit.connect(cmd_client.stop)
    gui = mod.RobotGUI(cmd_queue, telem_queue)
    gui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
