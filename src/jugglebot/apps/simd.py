#!/usr/bin/env python3
"""
simd.py - Simulation robot daemon for Cable Jugglebot.

Runs the real-time control loop in simulation mode.
"""

import argparse
import logging
import sys
import threading
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from jugglebot.core.robot_server import (
    ControlBridge,
)
from jugglebot.core.state import RuntimeMailbox
from jugglebot.io.simulated_actuator_bus import SimulatedActuatorBus
from jugglebot.transport.axes_logger import axes_state_logger
from jugglebot.transport.tcp_commands import tcp_command_server
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
    parser.add_argument(
        "--auto-enable",
        action="store_true",
        help="Start simulation in enabled state immediately"
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
    state = RuntimeMailbox()

    # Setup simulation driver
    odrive_config = config.get("hardware", {}).get("odrive", {})
    axis_ids = odrive_config.get("axis_ids", [0, 1, 2, 3, 4, 5])
    spool_cfg = config.get("controller", {}).get("spool_space", {})
    winch_cfg = config.get("winches", {})

    # Create simulation actuator bus
    driver = SimulatedActuatorBus(
        axis_ids=axis_ids,
        enable_viewer=args.viewer,
        spool_kp=spool_cfg.get("kp"),
        spool_kd=spool_cfg.get("kd"),
        torque_limit_nm=spool_cfg.get("torque_limit_nm", winch_cfg.get("torque_limit_nm")),
    )
    sim_bridge = ControlBridge(state, driver, config=config)
    sim_bridge.start()

    auto_enable = bool(args.auto_enable or config.get("robot", {}).get("auto_enable_on_startup", False))
    if auto_enable:
        state.set_state("enable")
        logger.info("Auto-enable on startup is active")

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
