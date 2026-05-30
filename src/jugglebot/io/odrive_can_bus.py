"""ODrive CAN actuator-bus backend."""

from __future__ import annotations

from jugglebot.io.actuator_bus import ActuatorBusCapabilities
from jugglebot.io.driver_backed_actuator_bus import DriverBackedActuatorBus


class ODriveCanBus(DriverBackedActuatorBus):
    def __init__(
        self,
        canbus: str = "can0",
        axis_ids=None,
        mm_per_turn=None,
        capstan_radius_m: float = 0.01,
        torque_direction: float = 1.0,
        pose_est_rate_hz: float = 100.0,
        can_bitrate: float = 1_000_000.0,
        can_frame_bits_est: float = 128.0,
    ):
        from jugglebot.drivers.hardware_driver import HardwareDriver

        axis_ids = axis_ids or [0, 1, 2, 3, 4, 5]
        driver = HardwareDriver(
            canbus=canbus,
            axis_ids=axis_ids,
            mm_per_turn=mm_per_turn,
            capstan_radius_m=capstan_radius_m,
            torque_direction=torque_direction,
            pose_est_rate_hz=pose_est_rate_hz,
            can_bitrate=can_bitrate,
            can_frame_bits_est=can_frame_bits_est,
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
                sim_clock=False,
                spool_gain_config=False,
            ),
        )
