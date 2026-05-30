"""Simulation-backed actuator-bus backend."""

from __future__ import annotations

from jugglebot.drivers.simulation_driver import SimulationDriver
from jugglebot.io.actuator_bus import ActuatorBusCapabilities
from jugglebot.io.driver_backed_actuator_bus import DriverBackedActuatorBus


class SimulatedActuatorBus(DriverBackedActuatorBus):
    def __init__(
        self,
        axis_ids=None,
        enable_viewer: bool = False,
        spool_kp=None,
        spool_kd=None,
        torque_limit_nm=None,
    ):
        axis_ids = axis_ids or [0, 1, 2, 3, 4, 5]
        driver = SimulationDriver(
            axis_ids=axis_ids,
            enable_viewer=enable_viewer,
            spool_kp=spool_kp,
            spool_kd=spool_kd,
            torque_limit_nm=torque_limit_nm,
        )
        super().__init__(
            driver=driver,
            axis_ids=axis_ids,
            capabilities=ActuatorBusCapabilities(
                position_command_with_ff=True,
                torque_command=True,
                platform_state_feedback=False,
                cable_jacobian_feedback=False,
                cable_tension_feedback=True,
                axis_torque_feedback=True,
                sim_clock=True,
                spool_gain_config=True,
            ),
        )
