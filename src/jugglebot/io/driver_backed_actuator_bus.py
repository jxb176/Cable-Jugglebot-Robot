"""Adapter from the legacy RobotDriver API to the ActuatorBus interface."""

from __future__ import annotations

import threading
import time
from typing import Callable, Sequence

from jugglebot.core.types import (
    ActuatorCommand,
    ActuatorControlMode,
    ActuatorState,
    BusStats,
)
from jugglebot.io.actuator_bus import ActuatorBus, ActuatorBusCapabilities


class DriverBackedActuatorBus(ActuatorBus):
    """Transitional adapter that wraps an existing legacy driver."""

    def __init__(self, driver, axis_ids, capabilities: ActuatorBusCapabilities):
        self._driver = driver
        self.axis_ids = tuple(int(aid) for aid in axis_ids)
        self.capabilities = capabilities
        self.mm_per_turn = getattr(driver, "mm_per_turn", None)
        self._lock = threading.Lock()
        self._feedback = {
            aid: {
                "feedback_timestamp_s": None,
                "position_turns": None,
                "velocity_turns_per_s": None,
                "bus_voltage_v": None,
                "bus_current_a": None,
                "current_estimate_a": None,
                "temperature_fet_c": None,
                "temperature_motor_c": None,
                "error_flags": None,
                "axis_state": None,
                "proc_result": None,
            }
            for aid in self.axis_ids
        }
        self._position_callback: Callable[[int, float], None] | None = None
        self._velocity_callback: Callable[[int, float], None] | None = None
        self._bus_callback: Callable[[int, float, float], None] | None = None
        self._current_callback: Callable[[int, float], None] | None = None
        self._temp_callback: Callable[[int, float, float], None] | None = None
        self._heartbeat_callback: Callable[[int, int, int, int], None] | None = None

    def start(self):
        self._driver.set_position_callback(self._handle_position)
        self._driver.set_velocity_callback(self._handle_velocity)
        self._driver.set_bus_callback(self._handle_bus)
        self._driver.set_current_callback(self._handle_current)
        self._driver.set_temp_callback(self._handle_temp)
        self._driver.set_heartbeat_callback(self._handle_heartbeat)
        self._driver.start()

    def stop(self):
        self._driver.stop()

    def read_actuator_states(self) -> tuple[ActuatorState, ...]:
        with self._lock:
            feedback = {aid: dict(values) for aid, values in self._feedback.items()}
        torque_values = self.get_axis_torques()
        tension_values = self.get_cable_tensions()
        states = []
        now = time.time()
        for index, aid in enumerate(self.axis_ids):
            fb = feedback[aid]
            timestamp_s = fb["feedback_timestamp_s"]
            feedback_age_s = None if timestamp_s is None else max(0.0, now - float(timestamp_s))
            torque_estimate_nm = None
            if torque_values is not None and index < len(torque_values):
                torque_estimate_nm = float(torque_values[index])
            tension_estimate_n = None
            if tension_values is not None and index < len(tension_values):
                tension_estimate_n = float(tension_values[index])
            states.append(
                ActuatorState(
                    axis_id=aid,
                    feedback_timestamp_s=timestamp_s,
                    position_turns=fb["position_turns"],
                    velocity_turns_per_s=fb["velocity_turns_per_s"],
                    torque_estimate_nm=torque_estimate_nm,
                    tension_estimate_n=tension_estimate_n,
                    current_estimate_a=fb["current_estimate_a"],
                    axis_state=fb["axis_state"],
                    error_flags=fb["error_flags"],
                    proc_result=fb["proc_result"],
                    temperature_fet_c=fb["temperature_fet_c"],
                    temperature_motor_c=fb["temperature_motor_c"],
                    bus_voltage_v=fb["bus_voltage_v"],
                    bus_current_a=fb["bus_current_a"],
                    feedback_age_s=feedback_age_s,
                    valid=fb["position_turns"] is not None,
                    stale=feedback_age_s is not None and feedback_age_s > 0.25,
                )
            )
        return tuple(states)

    def write_commands(self, commands: Sequence[ActuatorCommand]):
        for cmd in commands:
            axis_id = int(cmd.axis_id)
            if cmd.enable is False:
                self.set_axis_state(axis_id, "idle")
                continue
            if cmd.enable is True:
                self.set_axis_state(axis_id, "closed_loop")

            if cmd.apply_control_mode:
                if cmd.control_mode == ActuatorControlMode.POSITION:
                    self.set_controller_mode(axis_id, "position")
                elif cmd.control_mode == ActuatorControlMode.TORQUE:
                    self.set_controller_mode(axis_id, "torque")
                elif cmd.control_mode == ActuatorControlMode.DISABLED:
                    continue
                else:
                    raise NotImplementedError(f"Unsupported control mode for legacy driver adapter: {cmd.control_mode}")

            if cmd.control_mode == ActuatorControlMode.POSITION:
                if (
                    cmd.position_turns is not None
                    or cmd.velocity_ff_turns_per_s is not None
                    or cmd.torque_ff_nm is not None
                ):
                    self.set_axis_position_command(
                        axis_id,
                        0.0 if cmd.position_turns is None else float(cmd.position_turns),
                        velocity_ff=0.0 if cmd.velocity_ff_turns_per_s is None else float(cmd.velocity_ff_turns_per_s),
                        torque_ff=0.0 if cmd.torque_ff_nm is None else float(cmd.torque_ff_nm),
                    )
            elif cmd.control_mode == ActuatorControlMode.TORQUE:
                if cmd.torque_nm is not None:
                    self.set_axis_torque(axis_id, float(cmd.torque_nm))
            elif cmd.control_mode != ActuatorControlMode.DISABLED:
                raise NotImplementedError(f"Unsupported control mode for legacy driver adapter: {cmd.control_mode}")

    def get_bus_stats(self) -> BusStats | None:
        if hasattr(self._driver, "get_comm_stats"):
            stats = self._driver.get_comm_stats()
            if isinstance(stats, dict):
                return BusStats(**stats)
        return None

    def get_comm_stats(self):
        stats = self.get_bus_stats()
        return None if stats is None else stats.to_dict()

    # Transitional legacy-control compatibility surface.
    def set_axis_position(self, axis_id: int, position: float):
        self._driver.set_axis_position(axis_id, position)

    def set_axis_position_command(
        self,
        axis_id: int,
        position: float,
        velocity_ff: float = 0.0,
        torque_ff: float = 0.0,
    ):
        self._driver.set_axis_position_command(axis_id, position, velocity_ff=velocity_ff, torque_ff=torque_ff)

    def set_axis_torque(self, axis_id: int, torque: float):
        self._driver.set_axis_torque(axis_id, torque)

    def get_axis_position(self, axis_id: int):
        return self._driver.get_axis_position(axis_id)

    def get_axis_velocity(self, axis_id: int):
        return self._driver.get_axis_velocity(axis_id)

    def set_controller_mode(self, axis_id: int, mode: str):
        self._driver.set_controller_mode(axis_id, mode)

    def set_axis_state(self, axis_id: int, state: str):
        self._driver.set_axis_state(axis_id, state)

    def set_absolute_position(self, axis_id: int, position: float):
        self._driver.set_absolute_position(axis_id, position)

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

    def get_platform_state(self):
        if hasattr(self._driver, "get_platform_state"):
            return self._driver.get_platform_state()
        return None, None

    def get_cable_jacobian_plat(self):
        if hasattr(self._driver, "get_cable_jacobian_plat"):
            return self._driver.get_cable_jacobian_plat()
        return None

    def get_cable_tensions(self):
        if hasattr(self._driver, "get_cable_tensions"):
            return self._driver.get_cable_tensions()
        return None

    def get_axis_torques(self):
        if hasattr(self._driver, "get_axis_torques"):
            return self._driver.get_axis_torques()
        return None

    def get_sim_time(self):
        if hasattr(self._driver, "get_sim_time"):
            return self._driver.get_sim_time()
        return None

    def compute_platform_wrench(self, qdd_cmd):
        if hasattr(self._driver, "compute_platform_wrench"):
            return self._driver.compute_platform_wrench(qdd_cmd)
        return None

    def configure_spool_controller(self, kp=None, kd=None, torque_limit=None):
        if hasattr(self._driver, "configure_spool_controller"):
            return self._driver.configure_spool_controller(kp=kp, kd=kd, torque_limit=torque_limit)
        return False

    # Callback fanout and feedback caching.
    def _update_feedback(self, axis_id: int, **fields):
        now = time.time()
        with self._lock:
            axis_feedback = self._feedback.get(axis_id)
            if axis_feedback is None:
                return
            axis_feedback.update(fields)
            axis_feedback["feedback_timestamp_s"] = now

    def _handle_position(self, axis_id: int, position: float):
        self._update_feedback(axis_id, position_turns=float(position))
        if self._position_callback is not None:
            self._position_callback(axis_id, position)

    def _handle_velocity(self, axis_id: int, velocity: float):
        self._update_feedback(axis_id, velocity_turns_per_s=float(velocity))
        if self._velocity_callback is not None:
            self._velocity_callback(axis_id, velocity)

    def _handle_bus(self, axis_id: int, bus_voltage: float, bus_current: float):
        self._update_feedback(axis_id, bus_voltage_v=float(bus_voltage), bus_current_a=float(bus_current))
        if self._bus_callback is not None:
            self._bus_callback(axis_id, bus_voltage, bus_current)

    def _handle_current(self, axis_id: int, current: float):
        self._update_feedback(axis_id, current_estimate_a=float(current))
        if self._current_callback is not None:
            self._current_callback(axis_id, current)

    def _handle_temp(self, axis_id: int, fet_temp: float, motor_temp: float):
        self._update_feedback(
            axis_id,
            temperature_fet_c=float(fet_temp),
            temperature_motor_c=float(motor_temp),
        )
        if self._temp_callback is not None:
            self._temp_callback(axis_id, fet_temp, motor_temp)

    def _handle_heartbeat(self, axis_id: int, error_flags: int, axis_state: int, proc_result: int):
        self._update_feedback(
            axis_id,
            error_flags=int(error_flags),
            axis_state=int(axis_state),
            proc_result=int(proc_result),
        )
        if self._heartbeat_callback is not None:
            self._heartbeat_callback(axis_id, error_flags, axis_state, proc_result)
