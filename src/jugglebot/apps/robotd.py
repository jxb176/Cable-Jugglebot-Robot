#!/usr/bin/env python3
"""
robotd.py - Hardware robot daemon for Cable Jugglebot.

Runs the real-time control loop with ODrive hardware.
"""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from jugglebot.core.robot_server import (
    RobotState,
    ControlBridge,
    tcp_command_server,
    axes_state_logger,
    ensure_can_interface_up,
)
from jugglebot.drivers.hardware_driver import HardwareDriver
from jugglebot.drivers.simulation_driver import SimulationDriver
from jugglebot.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Cable Jugglebot Hardware Daemon")
    parser.add_argument(
        "--config",
        type=str,
        default="default.yaml",
        help="Configuration file name in config/ directory"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    mode = config.get("robot", {}).get("mode", "hardware")
    if mode != "hardware":
        print(f"Error: robotd requires mode=hardware, but config has mode={mode}")
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.get("logging", {}).get("level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting Cable Jugglebot Hardware Daemon")

    # Initialize robot state
    state = RobotState()

    # Setup CAN interface
    can_config = config.get("hardware", {}).get("can", {})
    can_interface = can_config.get("interface", "can0")
    can_bitrate = can_config.get("bitrate", 1000000)

    can_ok = ensure_can_interface_up(can_interface, can_bitrate)
    if not can_ok:
        logger.warning("CAN interface not available, continuing anyway")

    # Setup ODrive bridge
    odrive_config = config.get("hardware", {}).get("odrive", {})
    axis_ids = odrive_config.get("axis_ids", [0, 1, 2, 3, 4, 5])

    # Create hardware driver
    mm_per_turn = odrive_config.get("mm_per_turn", [-62.832] * len(axis_ids))
    capstan_radius_m = float(config.get("geometry", {}).get("capstan_radius_m", 0.01))
    torque_direction = float(odrive_config.get("torque_direction", 1.0))
    pose_est_rate_hz = float(odrive_config.get("pose_est_rate_hz", 100.0))
    driver = HardwareDriver(
        canbus=can_interface,
        axis_ids=axis_ids,
        mm_per_turn=mm_per_turn,
        capstan_radius_m=capstan_radius_m,
        torque_direction=torque_direction,
        pose_est_rate_hz=pose_est_rate_hz,
        can_bitrate=float(can_bitrate),
    )
    odrv_bridge = ControlBridge(state, driver)
    odrv_bridge.start()

    # Start TCP command server
    tcp_thread = threading.Thread(
        target=tcp_command_server,
        args=(state,),
        daemon=True
    )
    tcp_thread.start()

    # Start axes state logger
    logger_thread = threading.Thread(
        target=axes_state_logger,
        args=(state,),
        daemon=True
    )
    logger_thread.start()

    logger.info("Hardware daemon running. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down hardware daemon...")
        odrv_bridge.stop()


if __name__ == "__main__":
    main()
