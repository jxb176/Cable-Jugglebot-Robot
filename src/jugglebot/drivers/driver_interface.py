"""
Abstract driver interface for robot hardware/simulation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable


class RobotDriver(ABC):
    """
    Abstract base class for robot drivers (hardware or simulation).
    """

    @abstractmethod
    def start(self):
        """Start the driver."""
        pass

    @abstractmethod
    def stop(self):
        """Stop the driver."""
        pass

    @abstractmethod
    def set_axis_position(self, axis_id: int, position: float):
        """Set position setpoint for an axis."""
        pass

    @abstractmethod
    def set_axis_torque(self, axis_id: int, torque: float):
        """Set torque setpoint for an axis."""
        pass

    @abstractmethod
    def get_axis_position(self, axis_id: int) -> Optional[float]:
        """Get current position feedback for an axis."""
        pass

    @abstractmethod
    def get_axis_velocity(self, axis_id: int) -> Optional[float]:
        """Get current velocity feedback for an axis."""
        pass

    @abstractmethod
    def set_controller_mode(self, axis_id: int, mode: str):
        """Set controller mode (position, torque, etc.)."""
        pass

    @abstractmethod
    def set_axis_state(self, axis_id: int, state: str):
        """Set axis state (idle, closed_loop, etc.)."""
        pass

    @abstractmethod
    def set_absolute_position(self, axis_id: int, position: float):
        """Set absolute position reference."""
        pass

    # Callbacks for feedback
    def set_position_callback(self, callback: Callable[[int, float], None]):
        """Set callback for position feedback."""
        pass

    def set_velocity_callback(self, callback: Callable[[int, float], None]):
        """Set callback for velocity feedback."""
        pass

    def set_bus_callback(self, callback: Callable[[int, float, float], None]):
        """Set callback for bus voltage/current feedback."""
        pass

    def set_current_callback(self, callback: Callable[[int, float], None]):
        """Set callback for motor current feedback."""
        pass

    def set_temp_callback(self, callback: Callable[[int, float, float], None]):
        """Set callback for temperature feedback."""
        pass

    def set_heartbeat_callback(self, callback: Callable[[int, int, int, int], None]):
        """Set callback for heartbeat feedback."""
        pass

    def get_cable_tensions(self):
        """Optional: return per-cable tension feedback [N] as length-6 list, or None."""
        return None

    def get_axis_torques(self):
        """Optional: return per-axis applied torque feedback [Nm] as length-6 list, or None."""
        return None
