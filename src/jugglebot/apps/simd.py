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

from jugglebot.rt.runner import (
    ControlBridge,
)
from jugglebot.core.state import RuntimeMailbox
from jugglebot.io.simulated_actuator_bus import SimulatedActuatorBus
from jugglebot.rt.config import load_runtime_config
from jugglebot.transport.axes_logger import axes_state_logger
from jugglebot.transport.tcp_commands import tcp_command_server


def main():
    parser = argparse.ArgumentParser(description="Cable Jugglebot Simulation Daemon")
    parser.add_argument(
        "--config",
        type=str,
        default="sim.yaml",
        help="Configuration file name in config/ directory (default: sim.yaml)"
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
    config = load_runtime_config(args.config)
    mode = config.robot.mode
    if mode != "simulation":
        print(f"Error: simd requires mode=simulation, but config has mode={mode}")
        sys.exit(1)

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting Cable Jugglebot Simulation Daemon")

    # Initialize robot state
    state = RuntimeMailbox()

    # Setup simulation driver
    axis_ids = list(config.hardware.odrive.axis_ids)
    spool_cfg = config.controller.spool_space

    # Create simulation actuator bus
    driver = SimulatedActuatorBus(
        axis_ids=axis_ids,
        enable_viewer=args.viewer,
        spool_kp=list(spool_cfg.kp),
        spool_kd=list(spool_cfg.kd),
        torque_limit_nm=list(spool_cfg.torque_limit_nm),
    )
    sim_bridge = ControlBridge(state, driver, config=config)
    sim_bridge.start()

    auto_enable = bool(args.auto_enable or config.robot.auto_enable_on_startup)
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
