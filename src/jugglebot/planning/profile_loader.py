"""Load and build JugglePath profiles from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from .jugglepath import JugglePath, State3D


def load_profile_yaml(path: str) -> Dict[str, Any]:
    with Path(path).open("r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("profile must be a mapping/dict at top level")
    if "segments" not in data:
        raise ValueError("profile must include 'segments'")
    return data


def build_path_from_profile(profile: Dict[str, Any], command_rate_hz: float | None = None) -> Tuple[JugglePath, float]:
    start_cfg = profile.get("start", {}) or {}
    p0 = start_cfg.get("p", [0.0, 0.0, 0.0])
    v0 = start_cfg.get("v", [0.0, 0.0, 0.0])
    a0 = start_cfg.get("a", [0.0, 0.0, 0.0])

    if command_rate_hz is None:
        command_rate_hz = float(profile.get("command_rate_hz", 500.0))

    start = State3D(p=p0, v=v0, a=a0)
    path = JugglePath(sample_hz=float(command_rate_hz), start=start)

    segments = profile.get("segments", [])
    if not isinstance(segments, list) or len(segments) == 0:
        raise ValueError("profile 'segments' must be a non-empty list")

    for i, seg in enumerate(segments):
        if not isinstance(seg, dict):
            raise ValueError(f"segment {i} must be a dict")

        p = seg.get("p")
        if p is None:
            raise ValueError(f"segment {i} missing required key 'p'")

        path.add_segment(
            p=p,
            v=seg.get("v"),
            a=seg.get("a"),
            t=seg.get("t"),
            curve=seg.get("curve", "line"),
            time_law=seg.get("time_law", "linear"),
            duration=seg.get("duration"),
            accel_ref=float(seg.get("accel_ref", 1.0)),
            jerk_ref=float(seg.get("jerk_ref", 10.0)),
            v_max=(None if seg.get("v_max") is None else float(seg.get("v_max"))),
        )

    return path, float(command_rate_hz)
