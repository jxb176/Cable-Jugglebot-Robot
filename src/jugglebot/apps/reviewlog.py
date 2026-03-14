#!/usr/bin/env python3
"""Interactive matplotlib review for control diagnostic CSV logs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _find_latest_log(log_dir: Path) -> Path:
    matches = sorted(log_dir.glob("control_diag_*.csv"))
    if not matches:
        raise FileNotFoundError(f"No control_diag_*.csv files in {log_dir}")
    return matches[-1]


def _load_log(path: Path):
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Log is empty: {path}")

    data = {}
    for key in rows[0].keys():
        if key == "state":
            data[key] = np.asarray([r[key] for r in rows], dtype=object)
            continue
        if key == "profile_active":
            data[key] = np.asarray([int(float(r[key])) for r in rows], dtype=int)
            continue
        vals = []
        for r in rows:
            v = r[key]
            if v is None or v == "":
                vals.append(np.nan)
            else:
                try:
                    vals.append(float(v))
                except Exception:
                    vals.append(np.nan)
        data[key] = np.asarray(vals, dtype=float)
    return data


def _link_x_axes(axes):
    sync = {"active": False}

    def _on_xlim_changed(ax):
        if sync["active"]:
            return
        sync["active"] = True
        try:
            xlim = ax.get_xlim()
            for other in axes:
                if other is not ax:
                    other.set_xlim(xlim)
        finally:
            sync["active"] = False

    for ax in axes:
        ax.callbacks.connect("xlim_changed", _on_xlim_changed)


def _install_exit_hotkeys(figures):
    def _on_key(event):
        if event.key in ("q", "escape"):
            plt.close("all")

    for fig in figures:
        fig.canvas.mpl_connect("key_press_event", _on_key)


def _select_time_axis(data):
    if "sim_time_s" in data:
        sim_t = data["sim_time_s"]
        finite = np.isfinite(sim_t)
        if finite.sum() > max(10, int(0.7 * len(sim_t))):
            return sim_t, "simulation time [s]"
    return data["t_rel_s"], "wall time [s]"


def _plot_hand(data, t, xlabel):
    fig1, axs1 = plt.subplots(3, 1, sharex=True, num="Hand Translation")
    labels = [("x", "x_mm"), ("y", "y_mm"), ("z", "z_mm")]
    for i, (name, suffix) in enumerate(labels):
        axs1[i].plot(t, data[f"hand_cmd_{suffix}"], label="cmd", linewidth=1.3)
        axs1[i].plot(t, data[f"hand_rsp_{suffix}"], label="rsp", linewidth=1.0)
        axs1[i].set_ylabel(f"{name} [mm]")
        axs1[i].grid(True, alpha=0.3)
    axs1[0].legend(loc="upper right")
    axs1[-1].set_xlabel(xlabel)

    fig2, axs2 = plt.subplots(2, 1, sharex=True, num="Hand Orientation")
    angs = [("roll", "roll_deg"), ("pitch", "pitch_deg")]
    for i, (name, suffix) in enumerate(angs):
        axs2[i].plot(t, data[f"hand_cmd_{suffix}"], label="cmd", linewidth=1.3)
        axs2[i].plot(t, data[f"hand_rsp_{suffix}"], label="rsp", linewidth=1.0)
        axs2[i].set_ylabel(f"{name} [deg]")
        axs2[i].grid(True, alpha=0.3)
    axs2[0].legend(loc="upper right")
    axs2[-1].set_xlabel(xlabel)
    return fig1, fig2, [*axs1, *axs2]


def _plot_spools(data, t, xlabel):
    fig3, axs3 = plt.subplots(2, 3, sharex=True, num="Spool Command/Response")
    axs3 = axs3.flatten()
    for i in range(6):
        ax = axs3[i]
        ax.plot(t, data[f"spool_cmd_mm_{i + 1}"], label="cmd", linewidth=1.2)
        ax.plot(t, data[f"spool_rsp_mm_{i + 1}"], label="rsp", linewidth=1.0)
        ax.set_title(f"Spool {i + 1}")
        ax.set_ylabel("mm")
        ax.grid(True, alpha=0.3)
    axs3[0].legend(loc="upper right")
    for ax in axs3[-3:]:
        ax.set_xlabel(xlabel)
    return fig3, list(axs3)


def _plot_tensions(data, t, xlabel):
    fig4, axs4 = plt.subplots(2, 3, sharex=True, num="Cable Tension Command/Response")
    axs4 = axs4.flatten()
    for i in range(6):
        ax = axs4[i]
        ax.plot(t, data[f"spool_cmd_tension_N_{i + 1}"], label="cmd", linewidth=1.2)
        ax.plot(t, data[f"spool_rsp_tension_N_{i + 1}"], label="rsp", linewidth=1.0)
        ax.set_title(f"Cable {i + 1}")
        ax.set_ylabel("N")
        ax.grid(True, alpha=0.3)
    axs4[0].legend(loc="upper right")
    for ax in axs4[-3:]:
        ax.set_xlabel(xlabel)
    return fig4, list(axs4)


def _plot_torque_and_currents(data, t, xlabel):
    fig5, axs5 = plt.subplots(2, 3, sharex=True, num="Spool Torque, Motor Current")
    axs5 = axs5.flatten()
    for i in range(6):
        ax = axs5[i]
        ax.plot(t, data[f"spool_cmd_torque_nm_{i + 1}"], label="torque cmd [Nm]", linewidth=1.2)
        ax.plot(t, data[f"motor_i_{i + 1}"], label="motor i [A]", linewidth=1.0)
        ax.set_title(f"Spool {i + 1}")
        ax.grid(True, alpha=0.3)
    axs5[0].legend(loc="upper right")
    for ax in axs5[-3:]:
        ax.set_xlabel(xlabel)
    return fig5, list(axs5)


def _plot_wrench(data, t, xlabel):
    needed = [
        "wrench_cmd_fx_N", "wrench_rsp_fx_N",
        "wrench_cmd_fy_N", "wrench_rsp_fy_N",
        "wrench_cmd_fz_N", "wrench_rsp_fz_N",
        "wrench_cmd_tx_Nm", "wrench_rsp_tx_Nm",
        "wrench_cmd_ty_Nm", "wrench_rsp_ty_Nm",
    ]
    if any(k not in data for k in needed):
        return None, []

    fig, axs = plt.subplots(5, 1, sharex=True, num="Platform Wrench Cmd/Rsp")
    labels = [
        ("fx", "N"),
        ("fy", "N"),
        ("fz", "N"),
        ("tx", "Nm"),
        ("ty", "Nm"),
    ]
    for i, (name, unit) in enumerate(labels):
        axs[i].plot(t, data[f"wrench_cmd_{name}_{unit}"], label="cmd", linewidth=1.3)
        axs[i].plot(t, data[f"wrench_rsp_{name}_{unit}"], label="rsp", linewidth=1.0)
        axs[i].set_ylabel(f"{name} [{unit}]")
        axs[i].grid(True, alpha=0.3)
    axs[0].legend(loc="upper right")
    axs[-1].set_xlabel(xlabel)
    return fig, list(axs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive review for control diagnostic logs")
    parser.add_argument("--log", type=str, default=None, help="Path to control_diag_*.csv log")
    parser.add_argument("--log-dir", type=str, default="Logs", help="Directory for auto-selecting latest log")
    args = parser.parse_args()

    if args.log:
        log_path = Path(args.log)
    else:
        log_path = _find_latest_log(Path(args.log_dir))
    if not log_path.exists():
        raise FileNotFoundError(f"Log not found: {log_path}")

    data = _load_log(log_path)
    print(f"Loaded {len(data['t_rel_s'])} samples from {log_path}")
    t, xlabel = _select_time_axis(data)

    fig_hand_t, fig_hand_r, axs_hand = _plot_hand(data, t, xlabel)
    fig_spool, axs_spool = _plot_spools(data, t, xlabel)
    fig_tension, axs_tension = _plot_tensions(data, t, xlabel)
    fig_torque, axs_torque = _plot_torque_and_currents(data, t, xlabel)
    fig_wrench, axs_wrench = _plot_wrench(data, t, xlabel)
    all_axes = [*axs_hand, *axs_spool, *axs_tension, *axs_torque, *axs_wrench]
    _link_x_axes(all_axes)
    figs = [fig_hand_t, fig_hand_r, fig_spool, fig_tension, fig_torque]
    if fig_wrench is not None:
        figs.append(fig_wrench)
    _install_exit_hotkeys(figs)
    print("Hotkeys: press 'q' or 'Esc' in any figure to close all windows.")
    plt.show()


if __name__ == "__main__":
    main()
