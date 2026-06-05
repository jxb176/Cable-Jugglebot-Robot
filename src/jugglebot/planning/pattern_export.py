"""Export hand trajectories from pattern projects as pose command arrays."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np

from jugglebot.patterns import PatternProject, load_pattern_project


def load_pattern_yaml(path: str) -> PatternProject:
    """Load a pattern project YAML."""
    return load_pattern_project(Path(path))


def build_traj_from_pattern(
    project: PatternProject,
    *,
    hand: str,
    command_rate_hz: float | None = None,
    cycles: int = 1,
) -> Tuple[np.ndarray, float]:
    """
    Sample one hand's authored trajectory into the standard pose trajectory array.

    Returns `(traj, sample_hz)` where `traj` columns match the planning convention:
    `[t,x,y,z,vx,vy,vz,ax,ay,az,jx,jy,jz]`.
    """
    project.validate()

    sample_hz = 500.0 if command_rate_hz is None else float(command_rate_hz)
    if sample_hz <= 0.0:
        raise ValueError("command_rate_hz must be > 0")

    cycles = int(cycles)
    if cycles <= 0:
        raise ValueError("cycles must be >= 1")

    hand = str(hand).strip().lower()
    if hand not in {"left", "right"}:
        raise ValueError(f"hand must be 'left' or 'right'; got {hand!r}")

    has_authored_keyframes = bool(project.sorted_hand_trajectory(hand))
    has_event_anchors = any(
        event.throw_hand == hand or event.catch_hand == hand
        for event in project.sorted_events()
    )
    if not (has_authored_keyframes or has_event_anchors):
        raise ValueError(f"pattern has no trajectory data for hand {hand!r}")

    base_duration = float(project.timeline_duration())
    total_duration = base_duration * cycles if project.is_loop else base_duration

    count = max(2, int(math.floor(total_duration * sample_hz + 1e-9)) + 1)
    times = np.arange(count, dtype=float) / sample_hz
    if times[-1] < total_duration - 1e-9:
        times = np.append(times, total_duration)

    traj = np.zeros((len(times), 13), dtype=float)
    traj[:, 0] = times

    for i, time_s in enumerate(times):
        state = project.hand_state(hand, float(time_s))
        traj[i, 1:4] = state.position
        traj[i, 4:7] = state.velocity
        traj[i, 7:10] = state.acceleration

    return traj, sample_hz
