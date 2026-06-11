from __future__ import annotations

import math

from jugglebot.core.pose_utils import quat_to_rpy_rad
from jugglebot.core.snapshots import build_robot_state_snapshot
from jugglebot.core.state import RuntimeMailbox
from jugglebot.rt.config import (
    ManualInputAccelerationModeConfig,
    ManualInputConfig,
    ManualInputOrientationConfig,
    ManualInputPositionModeConfig,
    ManualInputVelocityModeConfig,
    ManualInputWorkspaceConfig,
    load_runtime_config,
)
from jugglebot.rt.manual_input import ManualInputController
from jugglebot.rt.trajectory_manager import TrajectorySample


def _manual_input_config() -> ManualInputConfig:
    return ManualInputConfig(
        stream_timeout_s=0.2,
        deadband=0.0,
        workspace=ManualInputWorkspaceConfig(radius_m=0.20, z_min_m=-0.05, z_max_m=0.25),
        orientation=ManualInputOrientationConfig(roll_limit_deg=20.0, pitch_limit_deg=20.0),
        position_mode=ManualInputPositionModeConfig(
            linear_xy_scale_m=0.08,
            linear_z_scale_m=0.08,
            angular_scale_deg=12.0,
            filter_tau_s=0.12,
            linear_velocity_limit_mps=0.20,
            angular_velocity_limit_degps=60.0,
        ),
        velocity_mode=ManualInputVelocityModeConfig(
            linear_velocity_limit_mps=0.20,
            angular_velocity_limit_degps=60.0,
            linear_accel_limit_mps2=1.00,
            angular_accel_limit_degps2=180.0,
        ),
        acceleration_mode=ManualInputAccelerationModeConfig(
            linear_accel_limit_mps2=1.00,
            angular_accel_limit_degps2=240.0,
            linear_velocity_limit_mps=0.20,
            angular_velocity_limit_degps=60.0,
        ),
    )


def _sample(
    *,
    x_m: float = 0.0,
    y_m: float = 0.0,
    z_m: float = 0.0,
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
    yaw_rad: float = 0.0,
    linear_velocity_mps=(0.0, 0.0, 0.0),
    angular_velocity_rps=(0.0, 0.0, 0.0),
    linear_acceleration_mps2=(0.0, 0.0, 0.0),
    angular_acceleration_rps2=(0.0, 0.0, 0.0),
) -> TrajectorySample:
    from jugglebot.core.pose_utils import quat_from_rpy_deg

    return TrajectorySample(
        pose_t_mm=(1000.0 * x_m, 1000.0 * y_m, 1000.0 * z_m),
        pose_q=quat_from_rpy_deg(
            math.degrees(roll_rad),
            math.degrees(pitch_rad),
            math.degrees(yaw_rad),
        ),
        linear_velocity_mps=linear_velocity_mps,
        angular_velocity_rps=angular_velocity_rps,
        linear_acceleration_mps2=linear_acceleration_mps2,
        angular_acceleration_rps2=angular_acceleration_rps2,
        sequence_id=1,
    )


def test_runtime_config_includes_manual_input_defaults():
    config = load_runtime_config("sim.yaml")

    manual = config.controller.manual_input
    assert manual.stream_timeout_s > 0.0
    assert manual.workspace.radius_m > 0.0
    assert manual.workspace.z_max_m > manual.workspace.z_min_m
    assert manual.position_mode.filter_tau_s >= 0.0
    assert manual.velocity_mode.linear_velocity_limit_mps > 0.0
    assert manual.acceleration_mode.linear_accel_limit_mps2 > 0.0


def test_position_mode_commands_absolute_target_not_incremental():
    mailbox = RuntimeMailbox()
    controller = ManualInputController(_manual_input_config())
    base_sample = _sample()

    mailbox.set_state("enable")
    mailbox.set_manual_input_config(mode="position", gain=1.0)
    mailbox.set_manual_input_enabled(True)

    sample = base_sample
    for step_idx in range(1, 26):
        now_s = 0.05 * step_idx
        mailbox.submit_manual_input_sample((1.0, 0.0, 0.0, 0.0, 0.0, 0.0), timestamp_s=now_s)
        sample, status = controller.step(
            mailbox,
            now_control_s=now_s,
            now_perf_s=now_s,
            base_sample=base_sample,
            allow_streaming=True,
        )
    assert status.active is True
    assert math.isclose(sample.pose_t_mm[0] / 1000.0, 0.08, abs_tol=0.01)

    for step_idx in range(26, 51):
        now_s = 0.05 * step_idx
        mailbox.submit_manual_input_sample((0.5, 0.0, 0.0, 0.0, 0.0, 0.0), timestamp_s=now_s)
        sample, status = controller.step(
            mailbox,
            now_control_s=now_s,
            now_perf_s=now_s,
            base_sample=base_sample,
            allow_streaming=True,
        )
    assert status.active is True
    assert math.isclose(sample.pose_t_mm[0] / 1000.0, 0.04, abs_tol=0.01)


def test_velocity_mode_integrates_motion_and_respects_limit():
    mailbox = RuntimeMailbox()
    config = _manual_input_config()
    controller = ManualInputController(config)
    base_sample = _sample()

    mailbox.set_state("enable")
    mailbox.set_manual_input_config(mode="velocity", gain=1.0)
    mailbox.set_manual_input_enabled(True)

    sample = base_sample
    for step_idx in range(1, 21):
        now_s = 0.05 * step_idx
        mailbox.submit_manual_input_sample((1.0, 0.0, 0.0, 0.0, 0.0, 0.0), timestamp_s=now_s)
        sample, status = controller.step(
            mailbox,
            now_control_s=now_s,
            now_perf_s=now_s,
            base_sample=base_sample,
            allow_streaming=True,
        )
    assert status.active is True
    assert sample.pose_t_mm[0] > 0.0
    assert abs(sample.linear_velocity_mps[0]) <= config.velocity_mode.linear_velocity_limit_mps + 1e-9


def test_manual_input_timeout_holds_pose_and_reports_status():
    mailbox = RuntimeMailbox()
    controller = ManualInputController(_manual_input_config())
    base_sample = _sample()

    mailbox.set_state("enable")
    mailbox.set_manual_input_config(mode="position", gain=1.0)
    mailbox.set_manual_input_enabled(True)
    mailbox.submit_manual_input_sample((0.5, 0.0, 0.0, 0.0, 0.0, 0.0), timestamp_s=0.0)

    moving_sample, _ = controller.step(
        mailbox,
        now_control_s=0.10,
        now_perf_s=0.10,
        base_sample=base_sample,
        allow_streaming=True,
    )
    held_sample, status = controller.step(
        mailbox,
        now_control_s=0.40,
        now_perf_s=0.40,
        base_sample=base_sample,
        allow_streaming=True,
    )
    assert status.timed_out is True
    assert status.active is False
    assert held_sample.pose_t_mm == moving_sample.pose_t_mm
    assert held_sample.linear_velocity_mps == (0.0, 0.0, 0.0)


def test_non_enable_state_disables_manual_input_and_snapshot_exposes_status():
    mailbox = RuntimeMailbox()
    mailbox.set_manual_input_config(mode="velocity", gain=1.25)
    mailbox.set_manual_input_enabled(True)
    mailbox.submit_manual_input_sample((0.1, 0.2, 0.3, 0.0, 0.0, 0.0), timestamp_s=1.0)
    mailbox.set_manual_input_status(active=True, timed_out=False, workspace_clipped=True, rate_limited=False)
    mailbox.set_state("disable")

    snapshot = mailbox.get_manual_input_snapshot()
    assert snapshot["enabled"] is False

    robot_state = build_robot_state_snapshot(mailbox)
    manual_input_debug = robot_state.debug["manual_input"]
    assert manual_input_debug["enabled"] is False
    assert manual_input_debug["workspace_clipped"] is False


def test_trajectory_sample_preserves_angular_kinematics():
    sample = _sample(
        roll_rad=0.10,
        pitch_rad=-0.20,
        yaw_rad=0.30,
        angular_velocity_rps=(0.4, 0.5, 0.6),
        angular_acceleration_rps2=(0.7, 0.8, 0.9),
    )

    roll_rad, pitch_rad, yaw_rad = quat_to_rpy_rad(sample.pose_q)
    assert math.isclose(roll_rad, 0.10, abs_tol=1e-9)
    assert math.isclose(pitch_rad, -0.20, abs_tol=1e-9)
    assert math.isclose(yaw_rad, 0.30, abs_tol=1e-9)
    assert sample.angular_velocity_rps == (0.4, 0.5, 0.6)
    assert sample.angular_acceleration_rps2 == (0.7, 0.8, 0.9)
