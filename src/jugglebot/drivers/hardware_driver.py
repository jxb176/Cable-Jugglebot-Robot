"""
Hardware driver using ODrive CAN interface.
"""

import logging
from typing import Optional, Callable, List

from .driver_interface import RobotDriver
from .ODriveCANSimple import ODriveCanManager, AxisState


logger = logging.getLogger(__name__)


class HardwareDriver(RobotDriver):
    """
    Hardware driver implementation using ODrive CAN.
    """

    def __init__(self, canbus: str = "can0", axis_ids: List[int] = None):
        self.canbus = canbus
        self.axis_ids = axis_ids or [0, 1, 2, 3, 4, 5]
        self.manager: Optional[ODriveCanManager] = None
        self.axes = {}
        self._position_callback: Optional[Callable[[int, float], None]] = None
        self._velocity_callback: Optional[Callable[[int, float], None]] = None
        self._bus_callback: Optional[Callable[[int, float, float], None]] = None
        self._current_callback: Optional[Callable[[int, float], None]] = None
        self._temp_callback: Optional[Callable[[int, float, float], None]] = None
        self._heartbeat_callback: Optional[Callable[[int, int, int, int], None]] = None

    def start(self):
        """Start the ODrive CAN manager."""
        logger.info(f"Starting ODrive CAN manager on {self.canbus}")
        try:
            self.manager = ODriveCanManager(self.canbus)

            # Register axes
            for aid in self.axis_ids:
                axis = self.manager.add_axis(aid)
                self.axes[aid] = axis

                # Set up callbacks
                if self._position_callback or self._velocity_callback:
                    axis.on_encoder(lambda pos, vel, i=aid: self._handle_encoder(i, pos, vel))

                if self._bus_callback:
                    axis.on_bus(lambda vbus, ibus, i=aid: self._handle_bus(i, vbus, ibus))

                if self._current_callback:
                    axis.on_iq(lambda iq_set, iq_meas, i=aid: self._handle_current(i, iq_meas))

                if self._temp_callback:
                    axis.on_temp(lambda fet, motor, i=aid: self._handle_temp(i, fet, motor))

                if self._heartbeat_callback:
                    axis.on_heartbeat(lambda err, st, proc, i=aid: self._handle_heartbeat(i, err, st, proc))

                logger.info(f"Registered axis {aid}")

        except Exception as e:
            logger.error(f"Failed to start hardware driver: {e}")
            raise

    def stop(self):
        """Stop the ODrive CAN manager."""
        if self.manager:
            try:
                self.manager.close()
            except Exception as e:
                logger.warning(f"Error closing CAN manager: {e}")
            self.manager = None
        logger.info("Hardware driver stopped")

    def set_axis_position(self, axis_id: int, position: float):
        """Set position setpoint."""
        axis = self.axes.get(axis_id)
        if axis:
            axis.set_input_pos(position)

    def set_axis_torque(self, axis_id: int, torque: float):
        """Set torque setpoint."""
        axis = self.axes.get(axis_id)
        if axis:
            axis.set_input_torque(torque)

    def get_axis_position(self, axis_id: int) -> Optional[float]:
        """Get position feedback - not directly available, use callbacks."""
        return None  # Feedback comes via callbacks

    def get_axis_velocity(self, axis_id: int) -> Optional[float]:
        """Get velocity feedback - not directly available, use callbacks."""
        return None  # Feedback comes via callbacks

    def set_controller_mode(self, axis_id: int, mode: str):
        """Set controller mode."""
        axis = self.axes.get(axis_id)
        if not axis:
            return

        if mode == "position":
            control_mode = 3  # CONTROL_MODE_POSITION
            input_mode = 1    # INPUT_MODE_PASSTHROUGH
        elif mode == "torque":
            control_mode = 1  # CONTROL_MODE_TORQUE
            input_mode = 1    # INPUT_MODE_PASSTHROUGH
        else:
            logger.warning(f"Unknown controller mode: {mode}")
            return

        axis.set_controller_mode(control_mode, input_mode)

    def set_axis_state(self, axis_id: int, state: str):
        """Set axis state."""
        axis = self.axes.get(axis_id)
        if not axis:
            return

        if state == "idle":
            axis.set_axis_state(AxisState.IDLE)
        elif state == "closed_loop":
            axis.set_axis_state(AxisState.CLOSED_LOOP_CONTROL)
        else:
            logger.warning(f"Unknown axis state: {state}")

    def set_absolute_position(self, axis_id: int, position: float):
        """Set absolute position reference."""
        axis = self.axes.get(axis_id)
        if axis:
            axis.set_absolute_position(position)

    # Callback setters
    def set_position_callback(self, callback: Callable[[int, float], None]):
        self._position_callback = callback

    def set_velocity_callback(self, callback: Callable[[int, float], None]):
        self._velocity_callback = callback

    def set_bus_callback(self, callback: Callable[[int, float, float], None]):
        self._bus_callback = callback

    def set_current_callback(self, callback: Callable[[int, float], None]):
        self._current_callback = callback

    def set_temp_callback(self, callback: Callable[[int, float, float], None]):
        self._temp_callback = callback

    def set_heartbeat_callback(self, callback: Callable[[int, int, int, int], None]):
        self._heartbeat_callback = callback

    # Internal callback handlers
    def _handle_encoder(self, axis_id: int, pos: float, vel: float):
        if self._position_callback:
            self._position_callback(axis_id, pos)
        if self._velocity_callback:
            self._velocity_callback(axis_id, vel)

    def _handle_bus(self, axis_id: int, vbus: float, ibus: float):
        if self._bus_callback:
            self._bus_callback(axis_id, vbus, ibus)

    def _handle_current(self, axis_id: int, current: float):
        if self._current_callback:
            self._current_callback(axis_id, current)

    def _handle_temp(self, axis_id: int, fet: float, motor: float):
        if self._temp_callback:
            self._temp_callback(axis_id, fet, motor)

    def _handle_heartbeat(self, axis_id: int, error: int, state: int, proc_result: int):
        if self._heartbeat_callback:
            self._heartbeat_callback(axis_id, error, state, proc_result)
