"""RT-owned trajectory sampling and update application."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math

from jugglebot.core.pose_utils import quat_from_rpy_deg
from jugglebot.core.types import (
    PoseCommand,
    PoseCommandMode,
    TrajectoryCommand,
    TrajectoryUpdate,
    TrajectoryUpdateMode,
    TrajectoryWaypoint,
)


logger = logging.getLogger("robot")


@dataclass(slots=True, frozen=True)
class TrajectorySample:
    pose_t_mm: tuple[float, float, float]
    pose_q: tuple[float, float, float, float]
    linear_velocity_mps: tuple[float, float, float]
    angular_velocity_rps: tuple[float, float, float]
    linear_acceleration_mps2: tuple[float, float, float]
    angular_acceleration_rps2: tuple[float, float, float]
    sequence_id: int | None = None


@dataclass(slots=True, frozen=True)
class TrajectoryManagerStatus:
    active_sequence_id: int | None
    pending_sequence_id: int | None
    profile_active: bool
    playback_time_s: float
    command_mode: str


@dataclass(slots=True)
class _ActiveTrajectory:
    sequence_id: int
    start_time_s: float
    waypoints: tuple[TrajectoryWaypoint, ...]
    profile_active: bool


@dataclass(slots=True)
class _PendingActivation:
    update: TrajectoryUpdate
    activate_at_s: float | None = None
    remaining_skip_samples: int = 0


def _pose_command_to_sample(pose: PoseCommand, sequence_id: int | None) -> TrajectorySample:
    return TrajectorySample(
        pose_t_mm=(1000.0 * float(pose.x_m), 1000.0 * float(pose.y_m), 1000.0 * float(pose.z_m)),
        pose_q=quat_from_rpy_deg(
            math.degrees(float(pose.roll_rad)),
            math.degrees(float(pose.pitch_rad)),
            math.degrees(float(pose.yaw_rad)),
        ),
        linear_velocity_mps=tuple(float(v) for v in pose.linear_velocity_mps[:3]),
        angular_velocity_rps=tuple(float(v) for v in pose.angular_velocity_rps[:3]),
        linear_acceleration_mps2=tuple(float(v) for v in pose.linear_acceleration_mps2[:3]),
        angular_acceleration_rps2=tuple(float(v) for v in pose.angular_acceleration_rps2[:3]),
        sequence_id=sequence_id,
    )


def _sample_to_pose_command(sample: TrajectorySample) -> PoseCommand:
    from jugglebot.core.pose_utils import quat_to_rpy_rad

    roll_rad, pitch_rad, yaw_rad = quat_to_rpy_rad(sample.pose_q)
    return PoseCommand(
        x_m=float(sample.pose_t_mm[0]) / 1000.0,
        y_m=float(sample.pose_t_mm[1]) / 1000.0,
        z_m=float(sample.pose_t_mm[2]) / 1000.0,
        roll_rad=float(roll_rad),
        pitch_rad=float(pitch_rad),
        yaw_rad=float(yaw_rad),
        linear_velocity_mps=tuple(float(v) for v in sample.linear_velocity_mps),
        angular_velocity_rps=tuple(float(v) for v in sample.angular_velocity_rps),
        linear_acceleration_mps2=tuple(float(v) for v in sample.linear_acceleration_mps2),
        angular_acceleration_rps2=tuple(float(v) for v in sample.angular_acceleration_rps2),
    )


class TrajectoryManager:
    def __init__(self):
        initial_pose = PoseCommand(command_mode=PoseCommandMode.HOLD)
        self._hold_sample = _pose_command_to_sample(initial_pose, None)
        self._active: _ActiveTrajectory | None = None
        self._pending: _PendingActivation | None = None
        self._last_status = TrajectoryManagerStatus(
            active_sequence_id=None,
            pending_sequence_id=None,
            profile_active=False,
            playback_time_s=0.0,
            command_mode=initial_pose.command_mode.value,
        )

    def consume_mailbox_updates(self, state, now_s: float):
        update = state.take_pending_trajectory_update()
        if update is None:
            return
        self._accept_update(update, float(now_s))

    def sample(self, now_s: float) -> tuple[TrajectorySample, TrajectoryManagerStatus]:
        now_s = float(now_s)
        self._activate_pending_if_due(now_s)

        sample, playback_time_s, profile_active, command_mode = self._sample_current(now_s)

        pending = self._pending
        if pending is not None and pending.remaining_skip_samples > 0:
            pending.remaining_skip_samples -= 1

        self._last_status = TrajectoryManagerStatus(
            active_sequence_id=self._active.sequence_id if self._active is not None else sample.sequence_id,
            pending_sequence_id=None if self._pending is None else int(self._pending.update.sequence_id),
            profile_active=bool(profile_active),
            playback_time_s=float(playback_time_s),
            command_mode=str(command_mode),
        )
        return sample, self._last_status

    def _accept_update(self, update: TrajectoryUpdate, now_s: float):
        mode = update.mode
        if mode is TrajectoryUpdateMode.REPLACE:
            self._pending = None
            if update.effective_time_s is not None and float(update.effective_time_s) > now_s:
                self._pending = _PendingActivation(update=update, activate_at_s=float(update.effective_time_s))
                logger.info(f"[TRAJ] queued timed replace sequence={update.sequence_id} at {float(update.effective_time_s):.3f}s")
            else:
                self._activate_update(update, now_s)
            return

        if mode is TrajectoryUpdateMode.APPEND:
            self._pending = None
            self._append_update(update, now_s)
            return

        if mode is TrajectoryUpdateMode.SPLICE_NEXT_CYCLE:
            self._pending = _PendingActivation(update=update, remaining_skip_samples=1)
            logger.info(f"[TRAJ] queued next-cycle splice sequence={update.sequence_id}")
            return

        if mode is TrajectoryUpdateMode.SPLICE_AT_TIME:
            activate_at_s = float(update.effective_time_s) if update.effective_time_s is not None else now_s
            if activate_at_s <= now_s:
                self._activate_update(update, now_s)
            else:
                self._pending = _PendingActivation(update=update, activate_at_s=activate_at_s)
                logger.info(f"[TRAJ] queued timed splice sequence={update.sequence_id} at {activate_at_s:.3f}s")
            return

        logger.warning(f"[TRAJ] Unsupported update mode {mode.value}; treating as replace")
        self._activate_update(update, now_s)

    def _activate_pending_if_due(self, now_s: float):
        pending = self._pending
        if pending is None:
            return
        if pending.remaining_skip_samples <= 0 and pending.activate_at_s is None:
            self._pending = None
            self._activate_update(pending.update, now_s)
            return
        if pending.activate_at_s is not None and now_s >= float(pending.activate_at_s):
            self._pending = None
            self._activate_update(pending.update, now_s)

    def _activate_update(self, update: TrajectoryUpdate, now_s: float):
        reference_sample = self._current_reference_sample(now_s)
        active = self._build_active_from_update(update, now_s, reference_sample)
        if active is None:
            logger.warning("[TRAJ] Ignoring empty trajectory update")
            return
        self._active = active
        logger.info(
            f"[TRAJ] activated sequence={update.sequence_id} mode={update.mode.value} "
            f"waypoints={len(active.waypoints)} start={active.start_time_s:.3f}s"
        )

    def _append_update(self, update: TrajectoryUpdate, now_s: float):
        incoming = self._normalized_waypoints(update.trajectory)
        if not incoming:
            logger.warning("[TRAJ] Ignoring empty append trajectory")
            return

        if self._active is None:
            self._activate_update(
                TrajectoryUpdate(
                    sequence_id=update.sequence_id,
                    mode=TrajectoryUpdateMode.REPLACE,
                    trajectory=update.trajectory,
                    source_timestamp_s=update.source_timestamp_s,
                    effective_time_s=update.effective_time_s,
                    preserve_continuity=update.preserve_continuity,
                    note=update.note,
                ),
                now_s,
            )
            return

        remainder = self._remaining_waypoints(now_s)
        remainder_end_s = float(remainder[-1].time_from_start_s) if remainder else 0.0
        appended = list(remainder)
        if update.preserve_continuity and appended and float(incoming[0].time_from_start_s) > 1e-9:
            appended.append(
                TrajectoryWaypoint(
                    time_from_start_s=float(remainder_end_s),
                    pose=appended[-1].pose,
                )
            )
        for wp in incoming:
            appended.append(
                TrajectoryWaypoint(
                    time_from_start_s=float(remainder_end_s + wp.time_from_start_s),
                    pose=wp.pose,
                )
            )
        self._active = _ActiveTrajectory(
            sequence_id=int(update.sequence_id),
            start_time_s=float(now_s),
            waypoints=tuple(appended),
            profile_active=len(appended) > 1,
        )
        logger.info(
            f"[TRAJ] appended sequence={update.sequence_id} "
            f"remaining_waypoints={len(remainder)} appended_waypoints={len(incoming)}"
        )

    def _build_active_from_update(
        self,
        update: TrajectoryUpdate,
        now_s: float,
        reference_sample: TrajectorySample,
    ) -> _ActiveTrajectory | None:
        waypoints = self._normalized_waypoints(update.trajectory)
        if not waypoints:
            return None

        if update.preserve_continuity and float(waypoints[0].time_from_start_s) > 1e-9:
            waypoints = (
                TrajectoryWaypoint(
                    time_from_start_s=0.0,
                    pose=_sample_to_pose_command(reference_sample),
                ),
            ) + waypoints

        start_time_s = float(update.effective_time_s) if update.effective_time_s is not None else float(now_s)
        return _ActiveTrajectory(
            sequence_id=int(update.sequence_id),
            start_time_s=start_time_s,
            waypoints=waypoints,
            profile_active=len(waypoints) > 1,
        )

    def _normalized_waypoints(self, trajectory: TrajectoryCommand) -> tuple[TrajectoryWaypoint, ...]:
        waypoints = tuple(trajectory.waypoints)
        if not waypoints:
            return ()
        first_t = float(waypoints[0].time_from_start_s)
        normalized = []
        prev_t = None
        for wp in waypoints:
            rel_t = float(wp.time_from_start_s) - first_t
            if prev_t is not None and rel_t < prev_t:
                raise ValueError("trajectory waypoints must be nondecreasing in time")
            normalized.append(TrajectoryWaypoint(time_from_start_s=rel_t, pose=wp.pose))
            prev_t = rel_t
        return tuple(normalized)

    def _current_reference_sample(self, now_s: float) -> TrajectorySample:
        sample, _, _, _ = self._sample_current(now_s)
        return sample

    def _remaining_waypoints(self, now_s: float) -> tuple[TrajectoryWaypoint, ...]:
        active = self._active
        if active is None or not active.waypoints:
            return (
                TrajectoryWaypoint(
                    time_from_start_s=0.0,
                    pose=_sample_to_pose_command(self._hold_sample),
                ),
            )
        elapsed_s = max(0.0, float(now_s) - float(active.start_time_s))
        sample, _, _, _ = self._sample_active(active, now_s)
        remainder = [
            TrajectoryWaypoint(time_from_start_s=0.0, pose=_sample_to_pose_command(sample))
        ]
        for wp in active.waypoints:
            if float(wp.time_from_start_s) > elapsed_s + 1e-9:
                remainder.append(
                    TrajectoryWaypoint(
                        time_from_start_s=float(wp.time_from_start_s) - elapsed_s,
                        pose=wp.pose,
                    )
                )
        return tuple(remainder)

    def _sample_current(self, now_s: float):
        active = self._active
        if active is None or not active.waypoints:
            return self._hold_sample, 0.0, False, "hold"

        sample, elapsed_s, profile_active, command_mode = self._sample_active(active, now_s)
        self._hold_sample = sample
        if not profile_active:
            self._active = None
        return sample, elapsed_s, profile_active, command_mode

    def _sample_active(self, active: _ActiveTrajectory, now_s: float):
        elapsed_s = max(0.0, float(now_s) - float(active.start_time_s))
        last_waypoint = active.waypoints[-1]
        duration_s = float(last_waypoint.time_from_start_s)

        if elapsed_s >= duration_s:
            sample = _pose_command_to_sample(last_waypoint.pose, active.sequence_id)
            return sample, duration_s, False, last_waypoint.pose.command_mode.value

        i0 = 0
        while i0 + 1 < len(active.waypoints) and float(active.waypoints[i0 + 1].time_from_start_s) <= elapsed_s:
            i0 += 1
        wp0 = active.waypoints[i0]
        wp1 = active.waypoints[min(i0 + 1, len(active.waypoints) - 1)]
        t0 = float(wp0.time_from_start_s)
        t1 = float(wp1.time_from_start_s)
        alpha = 0.0 if t1 <= t0 else max(0.0, min(1.0, (elapsed_s - t0) / (t1 - t0)))
        sample = self._interpolate_pose(wp0.pose, wp1.pose, alpha, active.sequence_id)
        return sample, elapsed_s, True, wp0.pose.command_mode.value

    def _interpolate_pose(
        self,
        pose0: PoseCommand,
        pose1: PoseCommand,
        alpha: float,
        sequence_id: int | None,
    ) -> TrajectorySample:
        a = float(alpha)
        x_m = float(pose0.x_m) + a * (float(pose1.x_m) - float(pose0.x_m))
        y_m = float(pose0.y_m) + a * (float(pose1.y_m) - float(pose0.y_m))
        z_m = float(pose0.z_m) + a * (float(pose1.z_m) - float(pose0.z_m))
        roll_rad = float(pose0.roll_rad) + a * (float(pose1.roll_rad) - float(pose0.roll_rad))
        pitch_rad = float(pose0.pitch_rad) + a * (float(pose1.pitch_rad) - float(pose0.pitch_rad))
        yaw_rad = float(pose0.yaw_rad) + a * (float(pose1.yaw_rad) - float(pose0.yaw_rad))
        vel = tuple(
            float(pose0.linear_velocity_mps[i]) + a * (float(pose1.linear_velocity_mps[i]) - float(pose0.linear_velocity_mps[i]))
            for i in range(3)
        )
        ang_vel = tuple(
            float(pose0.angular_velocity_rps[i]) + a * (float(pose1.angular_velocity_rps[i]) - float(pose0.angular_velocity_rps[i]))
            for i in range(3)
        )
        acc = tuple(
            float(pose0.linear_acceleration_mps2[i]) + a * (float(pose1.linear_acceleration_mps2[i]) - float(pose0.linear_acceleration_mps2[i]))
            for i in range(3)
        )
        ang_acc = tuple(
            float(pose0.angular_acceleration_rps2[i]) + a * (float(pose1.angular_acceleration_rps2[i]) - float(pose0.angular_acceleration_rps2[i]))
            for i in range(3)
        )
        return TrajectorySample(
            pose_t_mm=(1000.0 * x_m, 1000.0 * y_m, 1000.0 * z_m),
            pose_q=quat_from_rpy_deg(math.degrees(roll_rad), math.degrees(pitch_rad), math.degrees(yaw_rad)),
            linear_velocity_mps=vel,
            angular_velocity_rps=ang_vel,
            linear_acceleration_mps2=acc,
            angular_acceleration_rps2=ang_acc,
            sequence_id=sequence_id,
        )
