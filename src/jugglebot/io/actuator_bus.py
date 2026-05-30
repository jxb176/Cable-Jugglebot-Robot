"""Hardware-agnostic actuator bus interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from jugglebot.core.types import ActuatorCommand, ActuatorState, BusStats


@dataclass(frozen=True, slots=True)
class ActuatorBusCapabilities:
    position_command_with_ff: bool = False
    torque_command: bool = False
    platform_state_feedback: bool = False
    cable_jacobian_feedback: bool = False
    cable_tension_feedback: bool = False
    axis_torque_feedback: bool = False
    sim_clock: bool = False
    spool_gain_config: bool = False


class ActuatorBus(ABC):
    """Common actuator-side interface used by the runtime controller."""

    axis_ids: tuple[int, ...]
    capabilities: ActuatorBusCapabilities

    @abstractmethod
    def start(self):
        """Start the bus backend."""

    @abstractmethod
    def stop(self):
        """Stop the bus backend."""

    @abstractmethod
    def read_actuator_states(self) -> tuple[ActuatorState, ...]:
        """Return the latest actuator-side feedback snapshot."""

    @abstractmethod
    def write_commands(self, commands: Sequence[ActuatorCommand]):
        """Write actuator commands for the current control step."""

    def get_bus_stats(self) -> BusStats | None:
        """Optional bus/utilization statistics."""
        return None
