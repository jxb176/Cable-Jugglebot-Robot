#!/usr/bin/env python3
"""Offline trajectory generation CLI for Jugglebot."""

from __future__ import annotations

import argparse
from pathlib import Path

from jugglebot.planning import (
    State3D,
    JugglePath,
    write_pose_cmd_csv,
    write_pose_cmd_full_csv,
    save_trajectory_plot,
    load_profile_yaml,
    build_path_from_profile,
    load_pattern_yaml,
    build_traj_from_pattern,
)


def build_simple_throw(sample_hz: float, throw_v: float) -> JugglePath:
    start = State3D(p=[0.0, 0.0, 0.0], v=[0.0, 0.0, 0.0], a=[0.0, 0.0, 0.0])
    path = JugglePath(start=start, sample_hz=sample_hz)

    accel_ref = 50.0
    jerk_ref = 2000.0

    path.add_segment(
        p=[0.0, 0.0, -0.2],
        v=[0.0, 0.0, 0.0],
        time_law="s_curve",
        accel_ref=5.0,
        jerk_ref=500.0,
    )

    path.add_segment(
        p=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, throw_v],
        time_law="s_curve_monotonic",
        accel_ref=accel_ref,
        jerk_ref=jerk_ref,
    )

    path.add_segment(
        p=[0.0, 0.0, 0.2],
        v=[0.0, 0.0, 0.0],
        time_law="s_curve_monotonic",
        accel_ref=accel_ref,
        jerk_ref=jerk_ref,
    )

    path.add_segment(
        p=[0.0, 0.0, 0.0],
        v=[0.0, 0.0, 0.0],
        time_law="s_curve",
        accel_ref=20.0,
        jerk_ref=1000.0,
        v_max=1.0,
    )

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline trajectories for Jugglebot")
    parser.add_argument("--profile", default="simple_throw", choices=["simple_throw"], help="Profile template")
    parser.add_argument("--profile-file", type=str, default=None, help="YAML profile path defining endpoints/segments")
    parser.add_argument(
        "--pattern-file",
        type=str,
        default=None,
        help="Pattern project YAML to sample directly into a pose trajectory",
    )
    parser.add_argument("--hand", type=str, default="right", help="Hand to export from a pattern file")
    parser.add_argument("--pattern-cycles", type=int, default=1, help="Loop cycles to export from a pattern file")
    parser.add_argument("--sample-hz", type=float, default=100.0, help="Sampling frequency (legacy alias)")
    parser.add_argument("--command-rate-hz", type=float, default=None, help="Command generation rate in Hz")
    parser.add_argument("--throw-v", type=float, default=4.9, help="Throw speed for simple_throw (m/s)")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory for generated CSV files")
    parser.add_argument("--plot", action="store_true", help="Also save trajectory plot PNG")
    args = parser.parse_args()

    cmd_hz = float(args.command_rate_hz) if args.command_rate_hz is not None else float(args.sample_hz)

    if args.profile_file and args.pattern_file:
        parser.error("--profile-file and --pattern-file are mutually exclusive")

    if args.pattern_file:
        project = load_pattern_yaml(args.pattern_file)
        traj, cmd_hz = build_traj_from_pattern(
            project,
            hand=args.hand,
            command_rate_hz=cmd_hz,
            cycles=args.pattern_cycles,
        )
        profile_name = str(project.name)
    elif args.profile_file:
        profile = load_profile_yaml(args.profile_file)
        path, cmd_hz = build_path_from_profile(profile, command_rate_hz=cmd_hz)
        profile_name = str(profile.get("name", Path(args.profile_file).stem))
    elif args.profile == "simple_throw":
        path = build_simple_throw(sample_hz=cmd_hz, throw_v=args.throw_v)
        profile_name = "simple_throw"
    else:
        raise ValueError(f"Unsupported profile: {args.profile}")

    if args.pattern_file:
        result_traj = traj
    else:
        result = path.build()
        result_traj = result.traj

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pose_cmd = out_dir / "pose_cmd.csv"
    pose_cmd_full = out_dir / "pose_cmd_full.csv"

    write_pose_cmd_csv(result_traj, str(pose_cmd))
    write_pose_cmd_full_csv(result_traj, str(pose_cmd_full))
    if args.plot:
        plot_path = out_dir / "trajectory_plot.png"
        try:
            save_trajectory_plot(result_traj, str(plot_path))
            print(f"Wrote: {plot_path}")
        except RuntimeError as exc:
            print(f"Warning: {exc}")
            print("Plot skipped. Install plotting dependency with: pip install matplotlib")

    print(f"Generated trajectory with {result_traj.shape[0]} samples")
    print(f"Profile: {profile_name}")
    print(f"Command rate: {cmd_hz:.3f} Hz")
    print(f"Wrote: {pose_cmd}")
    print(f"Wrote: {pose_cmd_full}")


if __name__ == "__main__":
    main()
