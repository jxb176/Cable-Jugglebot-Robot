"""Trajectory file IO for offline planning and robot playback."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def _strictly_increasing_time_mask(t: np.ndarray) -> np.ndarray:
    keep = np.ones(len(t), dtype=bool)
    if len(t) > 1:
        keep[1:] = np.diff(t) > 1e-9
    return keep


def write_pose_cmd_csv(
    traj: np.ndarray,
    out_path: str = "pose_cmd.csv",
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> None:
    """
    Export trajectory to pose command CSV.

    Input traj columns: [t,x,y,z,vx,vy,vz,ax,ay,az,jx,jy,jz] in SI units.
    Output columns: t,x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg
    """
    if traj.ndim != 2 or traj.shape[1] < 4:
        raise ValueError(f"traj must be (N,>=4). Got {traj.shape}")

    t = traj[:, 0].astype(float)
    xyz_mm = 1000.0 * traj[:, 1:4].astype(float)

    keep = _strictly_increasing_time_mask(t)
    t = t[keep]
    xyz_mm = xyz_mm[keep]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"])
        for i in range(len(t)):
            w.writerow([
                f"{t[i]:.9f}",
                f"{xyz_mm[i,0]:.6f}",
                f"{xyz_mm[i,1]:.6f}",
                f"{xyz_mm[i,2]:.6f}",
                f"{float(roll_deg):.6f}",
                f"{float(pitch_deg):.6f}",
                f"{float(yaw_deg):.6f}",
            ])


def write_pose_cmd_full_csv(
    traj: np.ndarray,
    out_path: str = "pose_cmd_full.csv",
    roll_deg: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> None:
    """
    Export trajectory to full command CSV for control development.

    Input traj columns: [t,x,y,z,vx,vy,vz,ax,ay,az,jx,jy,jz] in SI units.
    """
    if traj.ndim != 2 or traj.shape[1] < 10:
        raise ValueError(f"traj must be (N,>=10). Got {traj.shape}")

    t = traj[:, 0].astype(float)
    p_mm = 1000.0 * traj[:, 1:4].astype(float)
    v = traj[:, 4:7].astype(float)
    a = traj[:, 7:10].astype(float)

    keep = _strictly_increasing_time_mask(t)
    t = t[keep]
    p_mm = p_mm[keep]
    v = v[keep]
    a = a[keep]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "t",
            "x_mm", "y_mm", "z_mm",
            "vx_mps", "vy_mps", "vz_mps",
            "ax_mps2", "ay_mps2", "az_mps2",
            "roll_deg", "pitch_deg", "yaw_deg",
        ])
        for i in range(len(t)):
            w.writerow([
                f"{t[i]:.9f}",
                f"{p_mm[i,0]:.6f}", f"{p_mm[i,1]:.6f}", f"{p_mm[i,2]:.6f}",
                f"{v[i,0]:.6f}", f"{v[i,1]:.6f}", f"{v[i,2]:.6f}",
                f"{a[i,0]:.6f}", f"{a[i,1]:.6f}", f"{a[i,2]:.6f}",
                f"{float(roll_deg):.6f}",
                f"{float(pitch_deg):.6f}",
                f"{float(yaw_deg):.6f}",
            ])


def load_pose_cmd_csv(path: str) -> np.ndarray:
    """Load pose command CSV with columns t,x_mm,y_mm,z_mm,roll_deg,pitch_deg,yaw_deg."""
    rows = []
    with Path(path).open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append([
                float(row["t"]),
                float(row["x_mm"]),
                float(row["y_mm"]),
                float(row["z_mm"]),
                float(row["roll_deg"]),
                float(row["pitch_deg"]),
                float(row.get("yaw_deg", 0.0)),
            ])
    return np.asarray(rows, dtype=float)


def load_pose_cmd_full_csv(path: str) -> np.ndarray:
    """
    Load full pose command CSV with columns:
    t,x_mm,y_mm,z_mm,vx_mps,vy_mps,vz_mps,ax_mps2,ay_mps2,az_mps2,roll_deg,pitch_deg,yaw_deg
    """
    rows = []
    with Path(path).open("r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append([
                float(row["t"]),
                float(row["x_mm"]),
                float(row["y_mm"]),
                float(row["z_mm"]),
                float(row["vx_mps"]),
                float(row["vy_mps"]),
                float(row["vz_mps"]),
                float(row["ax_mps2"]),
                float(row["ay_mps2"]),
                float(row["az_mps2"]),
                float(row.get("roll_deg", 0.0)),
                float(row.get("pitch_deg", 0.0)),
                float(row.get("yaw_deg", 0.0)),
            ])
    return np.asarray(rows, dtype=float)


def save_trajectory_plot(traj: np.ndarray, out_path: str) -> None:
    """Save a simple x/y/z + speed plot for trajectory visualization."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for plotting trajectories") from exc

    if traj.ndim != 2 or traj.shape[1] < 7:
        raise ValueError(f"traj must be (N,>=7). Got {traj.shape}")

    t = traj[:, 0]
    p = traj[:, 1:4]
    v = traj[:, 4:7]
    speed = np.linalg.norm(v, axis=1)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
    axs[0].plot(t, p[:, 0], label="x")
    axs[0].plot(t, p[:, 1], label="y")
    axs[0].plot(t, p[:, 2], label="z")
    axs[0].set_ylabel("position [m]")
    axs[0].legend(loc="upper right")

    axs[1].plot(t, speed, label="|v|")
    axs[1].set_ylabel("speed [m/s]")
    axs[1].set_xlabel("time [s]")
    axs[1].legend(loc="upper right")

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
