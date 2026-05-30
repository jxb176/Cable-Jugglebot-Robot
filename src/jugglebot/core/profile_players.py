"""Profile playback worker threads."""

from __future__ import annotations

import logging
import threading
import time

from jugglebot.core.pose_utils import quat_from_rpy_deg


logger = logging.getLogger("robot")


class PoseProfilePlayer(threading.Thread):
    """
    Plays a time-pose profile:
      [t, x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg]
    Updates the commanded hand pose consumed by the task-space control loop.
    """

    def __init__(self, state, profile, rate_hz: float):
        super().__init__(daemon=True)
        self.state = state
        self._stop = threading.Event()
        if rate_hz <= 0:
            raise ValueError("rate_hz must be > 0")
        self.dt = 1.0 / rate_hz

        if not profile:
            raise ValueError("empty pose profile")

        t0 = float(profile[0][0])
        norm = []
        for row in profile:
            t = float(row[0])
            if len(row) >= 4:
                pose6 = [float(x) for x in row[1]]
                v3 = [float(x) for x in row[2]]
                a3 = [float(x) for x in row[3]]
            else:
                pose6 = [float(x) for x in row[1]]
                v3 = [0.0, 0.0, 0.0]
                a3 = [0.0, 0.0, 0.0]
            norm.append((float(t) - t0, pose6, v3, a3))
        self.norm_profile = norm
        self.duration = norm[-1][0] if norm else 0.0
        self._wall_start = None
        self._control_start = None

    def stop(self):
        self._stop.set()

    def _profile_elapsed_s(self):
        control_now = self.state.get_control_time_s()
        if control_now is not None:
            if self._control_start is None:
                self._control_start = float(control_now)
            return float(control_now) - self._control_start
        if self._wall_start is None:
            self._wall_start = time.perf_counter()
        return time.perf_counter() - self._wall_start

    def run(self):
        if self.duration <= 0.0:
            _, pose6, v3, a3 = self.norm_profile[-1]
            self._apply_pose(pose6, v3, a3)
            logger.info("[POSE_PROFILE] Zero-duration pose profile applied")
            return

        logger.info(f"[POSE_PROFILE] Starting playback at {1.0/self.dt:.1f} Hz, duration {self.duration:.3f}s")
        k = 0

        while not self._stop.is_set():
            t = self._profile_elapsed_s()
            if t >= self.duration:
                _, pose6, v3, a3 = self.norm_profile[-1]
                self._apply_pose(pose6, v3, a3)
                logger.info("[POSE_PROFILE] Completed")
                break

            while k + 1 < len(self.norm_profile) and self.norm_profile[k + 1][0] <= t:
                k += 1

            t0, p0, v0, a0 = self.norm_profile[k]
            t1, p1, v1, a1 = self.norm_profile[min(k + 1, len(self.norm_profile) - 1)]
            if t1 <= t0:
                alpha = 0.0
            else:
                alpha = max(0.0, min(1.0, (t - t0) / (t1 - t0)))

            pose = [p0[i] + alpha * (p1[i] - p0[i]) for i in range(6)]
            vel = [v0[i] + alpha * (v1[i] - v0[i]) for i in range(3)]
            acc = [a0[i] + alpha * (a1[i] - a0[i]) for i in range(3)]
            try:
                self._apply_pose(pose, vel, acc)
            except Exception as e:
                logger.error(f"[POSE_PROFILE] apply_pose error: {e}")

            time.sleep(self.dt)

        with self.state.lock:
            if self.state.player_thread is self:
                self.state.player_thread = None

    def _apply_pose(self, pose6, vel3=None, acc3=None):
        x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg = pose6

        q = quat_from_rpy_deg(roll_deg, pitch_deg, yaw_deg)
        self.state.set_hand_pose((x_mm, y_mm, z_mm), q, v_mps=vel3, a_mps2=acc3)
