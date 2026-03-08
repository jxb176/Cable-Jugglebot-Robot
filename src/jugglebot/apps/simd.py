#!/usr/bin/env python3
"""
simd.py - Simulation robot daemon for Cable Jugglebot.

Runs the real-time control loop in simulation mode.
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

from jugglebot.core.robot_server import (
    RobotState,
    ControlBridge,
    tcp_command_server,
    axes_state_logger,
)
from jugglebot.drivers.simulation_driver import SimulationDriver
from jugglebot.config import load_config


def main():
    parser = argparse.ArgumentParser(description="Cable Jugglebot Simulation Daemon")
    parser.add_argument(
        "--config",
        type=str,
        default="default.yaml",
        help="Configuration file name in config/ directory"
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Enable MuJoCo viewer for visualization"
    )
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    mode = config.get("robot", {}).get("mode", "simulation")
    if mode != "simulation":
        print(f"Error: simd requires mode=simulation, but config has mode={mode}")
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.get("logging", {}).get("level", "INFO")),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting Cable Jugglebot Simulation Daemon")

    # Initialize robot state
    state = RobotState()

    # Setup simulation driver
    odrive_config = config.get("hardware", {}).get("odrive", {})
    axis_ids = odrive_config.get("axis_ids", [0, 1, 2, 3, 4, 5])

    # Create simulation driver
    driver = SimulationDriver(axis_ids=axis_ids, enable_viewer=args.viewer)
    sim_bridge = ControlBridge(state, driver)
    sim_bridge.start()

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

    logger.info("Simulation daemon running. Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down simulation daemon...")
        sim_bridge.stop()


if __name__ == "__main__":
    main()
