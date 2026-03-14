#!/usr/bin/env python3
"""Upload and play a pose trajectory on a running robot daemon."""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np

from jugglebot.planning import load_pose_cmd_csv, load_pose_cmd_full_csv


def _send_command(sock: socket.socket, cmd: dict) -> None:
    payload = json.dumps(cmd) + "\n"
    sock.sendall(payload.encode("utf-8"))


def _estimate_rate_hz(profile: np.ndarray, default_hz: float) -> float:
    if profile.shape[0] < 2:
        return float(default_hz)
    dt = np.diff(profile[:, 0])
    dt = dt[dt > 1e-9]
    if dt.size == 0:
        return float(default_hz)
    return float(1.0 / np.median(dt))


def _looks_like_full_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8") as f:
            header = f.readline().strip().split(",")
    except Exception:
        return False
    header_set = {h.strip() for h in header}
    needed = {"vx_mps", "vy_mps", "vz_mps", "ax_mps2", "ay_mps2", "az_mps2"}
    return needed.issubset(header_set)


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload and play pose_cmd.csv on a running daemon")
    parser.add_argument("--csv", type=str, default="pose_cmd.csv", help="Path to pose command CSV")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Daemon TCP host")
    parser.add_argument("--port", type=int, default=5555, help="Daemon TCP port")
    parser.add_argument("--rate-hz", type=float, default=None, help="Playback rate (defaults to inferred from timestamps)")
    parser.add_argument(
        "--full-csv",
        action="store_true",
        help="Treat --csv as pose_cmd_full.csv and upload velocity/acceleration feedforward columns",
    )
    parser.add_argument("--auto-enable", action="store_true", help="Send state=enable before starting profile")
    parser.add_argument("--disable-at-end", action="store_true", help="Send state=disable after playback duration")
    parser.add_argument("--start-delay-s", type=float, default=0.25, help="Pause between upload and start command")
    parser.add_argument(
        "--wait-scale",
        type=float,
        default=3.0,
        help="Scale factor on trajectory duration for connection hold time (use >1 when sim runs slower than real-time)",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Do not keep the TCP connection open for profile completion",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    use_full_csv = bool(args.full_csv or _looks_like_full_csv(csv_path))

    if use_full_csv:
        profile = load_pose_cmd_full_csv(str(csv_path))
    else:
        profile = load_pose_cmd_csv(str(csv_path))
    if profile.size == 0:
        print(f"Error: CSV has no rows: {csv_path}")
        sys.exit(1)

    rate_hz = float(args.rate_hz) if args.rate_hz is not None else _estimate_rate_hz(profile, default_hz=500.0)
    duration_s = float(profile[-1, 0] - profile[0, 0]) if profile.shape[0] > 1 else 0.0
    rows = profile.tolist()

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((args.host, args.port))
    except ConnectionRefusedError:
        print(f"Error: could not connect to {args.host}:{args.port} (connection refused).")
        print("Is `simd` running and listening on TCP port 5555?")
        sys.exit(1)

    with sock:

        if args.auto_enable:
            _send_command(sock, {"type": "state", "value": "enable"})
            time.sleep(0.15)

        _send_command(sock, {"type": "pose_profile_upload", "profile": rows})
        time.sleep(max(0.0, float(args.start_delay_s)))
        _send_command(sock, {"type": "pose_profile_start", "rate_hz": rate_hz})

        print(f"Uploaded {len(rows)} trajectory rows from {csv_path}")
        if use_full_csv:
            print("Using full trajectory feedforward columns (v/a)")
        else:
            print("Using pose-only trajectory columns")
        print(f"Started pose profile at {rate_hz:.3f} Hz")

        if not args.no_wait:
            # Keep connection alive while profile plays; daemon stops profiles on disconnect.
            wait_s = max(0.0, duration_s * float(args.wait_scale)) + 0.2
            time.sleep(wait_s)

        if args.disable_at_end:
            if args.no_wait:
                # Still wait before disable when user explicitly opted out of blocking.
                time.sleep(max(0.0, duration_s) + 0.2)
            _send_command(sock, {"type": "state", "value": "disable"})
            print("Sent state=disable")


if __name__ == "__main__":
    main()
