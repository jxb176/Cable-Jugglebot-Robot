"""Runtime-owned manual input controller."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from jugglebot.core.pose_utils import quat_from_rpy_deg, quat_to_rpy_rad
from jugglebot.rt.config import ManualInputConfig
from jugglebot.rt.trajectory_manager import TrajectorySample


def _clamp_unit(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _apply_deadband(value: float, deadband: float) -> float:
    value = _clamp_unit(value)
    deadband = max(0.0, min(0.999, float(deadband)))
    mag = abs(value)
    if mag <= deadband:
        return 0.0
    scaled = (mag - deadband) / max(1e-9, 1.0 - deadband)
    return math.copysign(scaled, value)


@dataclass(slots=True, frozen=True)
class ManualInputStepStatus:
    active: bool = False
    timed_out: bool = False
    workspace_clipped: bool = False
    rate_limited: bool = False


class ManualInputController:
    def __init__(self, config: ManualInputConfig):
        self.config = config
        self._initialized = False
        self._manual_enabled = False
        self._mode = "position"
        self._gain = 1.0
        self._last_control_time_s: float | None = None
        self._reference_pose5 = np.zeros(5, dtype=float)
        self._pose5 = np.zeros(5, dtype=float)
        self._yaw_rad = 0.0
        self._linear_velocity = np.zeros(3, dtype=float)
        self._angular_velocity = np.zeros(3, dtype=float)
        self._linear_acceleration = np.zeros(3, dtype=float)
        self._angular_acceleration = np.zeros(3, dtype=float)

    def step(
        self,
        mailbox,
        *,
        now_control_s: float,
        now_perf_s: float,
        base_sample: TrajectorySample,
        allow_streaming: bool,
    ) -> tuple[TrajectorySample, ManualInputStepStatus]:
        snapshot = mailbox.get_manual_input_snapshot()
        if not self._initialized or not self._manual_enabled or not snapshot["enabled"] or not allow_streaming:
            self._sync_from_sample(base_sample, sync_reference=not self._manual_enabled)

        if not allow_streaming or not snapshot["enabled"]:
            self._manual_enabled = False
            status = ManualInputStepStatus(active=False)
            mailbox.set_manual_input_status(
                active=status.active,
                timed_out=status.timed_out,
                workspace_clipped=status.workspace_clipped,
                rate_limited=status.rate_limited,
            )
            return base_sample, status

        mode = str(snapshot["mode"]).lower()
        gain = max(0.0, float(snapshot["gain"]))
        if not self._manual_enabled:
            self._manual_enabled = True
            self._mode = mode
            self._gain = gain
            self._reference_pose5 = self._pose5.copy()
            self._linear_velocity[:] = 0.0
            self._angular_velocity[:] = 0.0
            self._linear_acceleration[:] = 0.0
            self._angular_acceleration[:] = 0.0
            self._last_control_time_s = float(now_control_s)
        elif mode != self._mode:
            self._mode = mode
            self._gain = gain
            self._reference_pose5 = self._pose5.copy()
            self._linear_velocity[:] = 0.0
            self._angular_velocity[:] = 0.0
            self._linear_acceleration[:] = 0.0
            self._angular_acceleration[:] = 0.0
        else:
            self._gain = gain

        dt = 0.0 if self._last_control_time_s is None else max(0.0, float(now_control_s) - float(self._last_control_time_s))
        self._last_control_time_s = float(now_control_s)

        sample_timestamp_s = snapshot["sample_timestamp_s"]
        timed_out = sample_timestamp_s is None or (float(now_perf_s) - float(sample_timestamp_s)) > float(self.config.stream_timeout_s)
        if timed_out:
            self._linear_velocity[:] = 0.0
            self._angular_velocity[:] = 0.0
            self._linear_acceleration[:] = 0.0
            self._angular_acceleration[:] = 0.0
            output = self._build_sample(base_sample.sequence_id)
            status = ManualInputStepStatus(active=False, timed_out=True)
            mailbox.set_manual_input_status(
                active=status.active,
                timed_out=status.timed_out,
                workspace_clipped=status.workspace_clipped,
                rate_limited=status.rate_limited,
            )
            return output, status

        sample_axes = tuple(_apply_deadband(v, self.config.deadband) for v in snapshot["sample"])
        if mode == "position":
            status = self._step_position_mode(sample_axes, dt)
        elif mode == "velocity":
            status = self._step_velocity_mode(sample_axes, dt)
        else:
            status = self._step_acceleration_mode(sample_axes, dt)

        output = self._build_sample(base_sample.sequence_id)
        mailbox.set_manual_input_status(
            active=status.active,
            timed_out=status.timed_out,
            workspace_clipped=status.workspace_clipped,
            rate_limited=status.rate_limited,
        )
        return output, status

    def _sync_from_sample(self, sample: TrajectorySample, *, sync_reference: bool) -> None:
        pose5, yaw_rad = self._sample_pose5(sample)
        self._pose5 = pose5
        self._yaw_rad = yaw_rad
        self._linear_velocity = np.asarray(sample.linear_velocity_mps, dtype=float)
        self._angular_velocity = np.asarray(sample.angular_velocity_rps, dtype=float)
        self._linear_acceleration = np.asarray(sample.linear_acceleration_mps2, dtype=float)
        self._angular_acceleration = np.asarray(sample.angular_acceleration_rps2, dtype=float)
        if not self._initialized or sync_reference:
            self._reference_pose5 = pose5.copy()
        self._initialized = True

    def _sample_pose5(self, sample: TrajectorySample) -> tuple[np.ndarray, float]:
        roll_rad, pitch_rad, yaw_rad = quat_to_rpy_rad(sample.pose_q)
        pose5 = np.asarray(
            [
                float(sample.pose_t_mm[0]) / 1000.0,
                float(sample.pose_t_mm[1]) / 1000.0,
                float(sample.pose_t_mm[2]) / 1000.0,
                float(roll_rad),
                float(pitch_rad),
            ],
            dtype=float,
        )
        return pose5, float(yaw_rad)

    def _step_position_mode(self, sample_axes: tuple[float, ...], dt: float) -> ManualInputStepStatus:
        cfg = self.config.position_mode
        target_pose = self._reference_pose5.copy()
        target_pose[0] += self._gain * float(cfg.linear_xy_scale_m) * float(sample_axes[0])
        target_pose[1] += self._gain * float(cfg.linear_xy_scale_m) * float(sample_axes[1])
        target_pose[2] += self._gain * float(cfg.linear_z_scale_m) * float(sample_axes[2])
        angle_scale_rad = math.radians(float(cfg.angular_scale_deg))
        target_pose[3] += self._gain * angle_scale_rad * float(sample_axes[3])
        target_pose[4] += self._gain * angle_scale_rad * float(sample_axes[4])

        target_pose, workspace_clipped = self._clamp_pose5(target_pose)
        if dt > 0.0 and float(cfg.filter_tau_s) > 0.0:
            alpha = dt / (float(cfg.filter_tau_s) + dt)
            filtered_target = self._pose5 + alpha * (target_pose - self._pose5)
        elif dt <= 0.0:
            filtered_target = self._pose5.copy()
        else:
            filtered_target = target_pose

        prev_pose = self._pose5.copy()
        pose_next = filtered_target.copy()
        rate_limited = False
        if dt > 0.0:
            max_linear_step = float(cfg.linear_velocity_limit_mps) * dt
            max_angular_step = math.radians(float(cfg.angular_velocity_limit_degps)) * dt
            linear_delta = np.clip(pose_next[:3] - prev_pose[:3], -max_linear_step, max_linear_step)
            angular_delta = np.clip(pose_next[3:5] - prev_pose[3:5], -max_angular_step, max_angular_step)
            rate_limited = bool(
                np.any(np.abs((pose_next[:3] - prev_pose[:3]) - linear_delta) > 1e-12)
                or np.any(np.abs((pose_next[3:5] - prev_pose[3:5]) - angular_delta) > 1e-12)
            )
            pose_next[:3] = prev_pose[:3] + linear_delta
            pose_next[3:5] = prev_pose[3:5] + angular_delta

        pose_next, clipped_after_rate = self._clamp_pose5(pose_next)
        workspace_clipped = workspace_clipped or clipped_after_rate
        self._pose5 = pose_next

        if dt > 0.0:
            linear_velocity = (pose_next[:3] - prev_pose[:3]) / dt
            angular_velocity_2 = (pose_next[3:5] - prev_pose[3:5]) / dt
            linear_acceleration = (linear_velocity - self._linear_velocity[:3]) / dt
            angular_acceleration_2 = (angular_velocity_2 - self._angular_velocity[:2]) / dt
            self._linear_velocity = np.asarray(linear_velocity, dtype=float)
            self._angular_velocity = np.asarray((angular_velocity_2[0], angular_velocity_2[1], 0.0), dtype=float)
            self._linear_acceleration = np.asarray(linear_acceleration, dtype=float)
            self._angular_acceleration = np.asarray((angular_acceleration_2[0], angular_acceleration_2[1], 0.0), dtype=float)
        else:
            self._linear_velocity[:] = 0.0
            self._angular_velocity[:] = 0.0
            self._linear_acceleration[:] = 0.0
            self._angular_acceleration[:] = 0.0

        return ManualInputStepStatus(
            active=True,
            workspace_clipped=workspace_clipped,
            rate_limited=rate_limited,
        )

    def _step_velocity_mode(self, sample_axes: tuple[float, ...], dt: float) -> ManualInputStepStatus:
        cfg = self.config.velocity_mode
        target_linear_velocity = self._gain * float(cfg.linear_velocity_limit_mps) * np.asarray(sample_axes[:3], dtype=float)
        target_angular_velocity = (
            self._gain
            * math.radians(float(cfg.angular_velocity_limit_degps))
            * np.asarray((sample_axes[3], sample_axes[4], 0.0), dtype=float)
        )

        prev_linear_velocity = self._linear_velocity.copy()
        prev_angular_velocity = self._angular_velocity.copy()
        rate_limited = False
        if dt > 0.0:
            linear_delta_limit = float(cfg.linear_accel_limit_mps2) * dt
            angular_delta_limit = math.radians(float(cfg.angular_accel_limit_degps2)) * dt
            linear_delta = np.clip(target_linear_velocity - prev_linear_velocity, -linear_delta_limit, linear_delta_limit)
            angular_delta = np.clip(target_angular_velocity - prev_angular_velocity, -angular_delta_limit, angular_delta_limit)
            rate_limited = bool(
                np.any(np.abs((target_linear_velocity - prev_linear_velocity) - linear_delta) > 1e-12)
                or np.any(np.abs((target_angular_velocity - prev_angular_velocity) - angular_delta) > 1e-12)
            )
            self._linear_velocity = prev_linear_velocity + linear_delta
            self._angular_velocity = prev_angular_velocity + angular_delta
        else:
            self._linear_velocity = target_linear_velocity
            self._angular_velocity = target_angular_velocity

        prev_pose = self._pose5.copy()
        self._pose5[:3] = prev_pose[:3] + self._linear_velocity[:3] * dt
        self._pose5[3] = prev_pose[3] + self._angular_velocity[0] * dt
        self._pose5[4] = prev_pose[4] + self._angular_velocity[1] * dt
        self._pose5, workspace_clipped = self._clamp_pose5(self._pose5)

        if workspace_clipped:
            clipped_linear_velocity = np.where(np.abs(self._pose5[:3] - prev_pose[:3]) <= 1e-12, 0.0, self._linear_velocity[:3])
            self._linear_velocity[:3] = clipped_linear_velocity
            clipped_angular_velocity = np.asarray(
                (
                    0.0 if abs(self._pose5[3] - prev_pose[3]) <= 1e-12 else self._angular_velocity[0],
                    0.0 if abs(self._pose5[4] - prev_pose[4]) <= 1e-12 else self._angular_velocity[1],
                    0.0,
                ),
                dtype=float,
            )
            self._angular_velocity = clipped_angular_velocity

        if dt > 0.0:
            self._linear_acceleration = (self._linear_velocity - prev_linear_velocity) / dt
            self._angular_acceleration = (self._angular_velocity - prev_angular_velocity) / dt
        else:
            self._linear_acceleration[:] = 0.0
            self._angular_acceleration[:] = 0.0

        return ManualInputStepStatus(
            active=True,
            workspace_clipped=workspace_clipped,
            rate_limited=rate_limited,
        )

    def _step_acceleration_mode(self, sample_axes: tuple[float, ...], dt: float) -> ManualInputStepStatus:
        cfg = self.config.acceleration_mode
        target_linear_acceleration = self._gain * float(cfg.linear_accel_limit_mps2) * np.asarray(sample_axes[:3], dtype=float)
        target_angular_acceleration = (
            self._gain
            * math.radians(float(cfg.angular_accel_limit_degps2))
            * np.asarray((sample_axes[3], sample_axes[4], 0.0), dtype=float)
        )

        self._linear_acceleration = target_linear_acceleration
        self._angular_acceleration = target_angular_acceleration
        if dt > 0.0:
            self._linear_velocity = self._linear_velocity + self._linear_acceleration * dt
            self._angular_velocity = self._angular_velocity + self._angular_acceleration * dt

        linear_limit = float(cfg.linear_velocity_limit_mps)
        angular_limit = math.radians(float(cfg.angular_velocity_limit_degps))
        limited_linear_velocity = np.clip(self._linear_velocity, -linear_limit, linear_limit)
        limited_angular_velocity = np.clip(self._angular_velocity, -angular_limit, angular_limit)
        rate_limited = bool(
            np.any(np.abs(limited_linear_velocity - self._linear_velocity) > 1e-12)
            or np.any(np.abs(limited_angular_velocity - self._angular_velocity) > 1e-12)
        )
        self._linear_velocity = limited_linear_velocity
        self._angular_velocity = limited_angular_velocity

        prev_pose = self._pose5.copy()
        self._pose5[:3] = prev_pose[:3] + self._linear_velocity[:3] * dt
        self._pose5[3] = prev_pose[3] + self._angular_velocity[0] * dt
        self._pose5[4] = prev_pose[4] + self._angular_velocity[1] * dt
        self._pose5, workspace_clipped = self._clamp_pose5(self._pose5)

        if workspace_clipped:
            self._linear_velocity[:3] = np.where(np.abs(self._pose5[:3] - prev_pose[:3]) <= 1e-12, 0.0, self._linear_velocity[:3])
            self._angular_velocity = np.asarray(
                (
                    0.0 if abs(self._pose5[3] - prev_pose[3]) <= 1e-12 else self._angular_velocity[0],
                    0.0 if abs(self._pose5[4] - prev_pose[4]) <= 1e-12 else self._angular_velocity[1],
                    0.0,
                ),
                dtype=float,
            )

        return ManualInputStepStatus(
            active=True,
            workspace_clipped=workspace_clipped,
            rate_limited=rate_limited,
        )

    def _clamp_pose5(self, pose5: np.ndarray) -> tuple[np.ndarray, bool]:
        clamped = np.asarray(pose5, dtype=float).copy()
        clipped = False
        radius = float(self.config.workspace.radius_m)
        radial = math.hypot(float(clamped[0]), float(clamped[1]))
        if radial > radius > 0.0:
            scale = radius / radial
            clamped[0] *= scale
            clamped[1] *= scale
            clipped = True
        z_min = float(self.config.workspace.z_min_m)
        z_max = float(self.config.workspace.z_max_m)
        new_z = min(max(float(clamped[2]), z_min), z_max)
        if abs(new_z - float(clamped[2])) > 1e-12:
            clamped[2] = new_z
            clipped = True
        roll_limit_rad = math.radians(float(self.config.orientation.roll_limit_deg))
        pitch_limit_rad = math.radians(float(self.config.orientation.pitch_limit_deg))
        new_roll = min(max(float(clamped[3]), -roll_limit_rad), roll_limit_rad)
        new_pitch = min(max(float(clamped[4]), -pitch_limit_rad), pitch_limit_rad)
        if abs(new_roll - float(clamped[3])) > 1e-12:
            clamped[3] = new_roll
            clipped = True
        if abs(new_pitch - float(clamped[4])) > 1e-12:
            clamped[4] = new_pitch
            clipped = True
        return clamped, clipped

    def _build_sample(self, sequence_id: int | None) -> TrajectorySample:
        pose_q = quat_from_rpy_deg(
            math.degrees(float(self._pose5[3])),
            math.degrees(float(self._pose5[4])),
            math.degrees(float(self._yaw_rad)),
        )
        return TrajectorySample(
            pose_t_mm=(
                1000.0 * float(self._pose5[0]),
                1000.0 * float(self._pose5[1]),
                1000.0 * float(self._pose5[2]),
            ),
            pose_q=pose_q,
            linear_velocity_mps=tuple(float(v) for v in self._linear_velocity[:3]),
            angular_velocity_rps=tuple(float(v) for v in self._angular_velocity[:3]),
            linear_acceleration_mps2=tuple(float(v) for v in self._linear_acceleration[:3]),
            angular_acceleration_rps2=tuple(float(v) for v in self._angular_acceleration[:3]),
            sequence_id=sequence_id,
        )
