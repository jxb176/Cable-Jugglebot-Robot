"""Pure controller-step helpers for spool/task-space control."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from jugglebot.core.cable_ik import CableRobotGeometry, pose_to_cable_lengths_mm
from jugglebot.core.kinematics import cable_lengths_jacobian_pose5_fd, cable_lengths_m_from_pose5
from jugglebot.core.pose_utils import quat_to_rpy_rad
from jugglebot.core.tension_control import (
    TensionAllocatorConfig,
    cable_tension_nullspace_basis,
    clamp_sigma_to_interval,
    nullspace_sigma_interval,
    rate_limit_scalar,
    solve_tensions_least_squares,
)
from jugglebot.core.types import ActuatorCommand, ActuatorControlMode, ActuatorState


@dataclass(slots=True)
class NullspaceControllerConfig:
    kp: float
    ki: float
    eta_limit_m: float
    sigma_ref_n: float
    sigma_rate_limit_nps: float
    tension_floor_n: float


@dataclass(slots=True)
class TaskspaceControllerConfig:
    axis_ids: tuple[int, ...]
    mm_per_turn: tuple[float, ...]
    home_cable_mm: tuple[float, ...]
    geometry: CableRobotGeometry
    outer_corr_kp: np.ndarray
    outer_corr_kd: np.ndarray
    outer_corr_cable_clip_m: np.ndarray
    spool_bias_tension_n: np.ndarray
    torque_per_tension: float
    gravity_ff_z_n: float
    enable_position_torque_ff: bool
    tension_allocator: TensionAllocatorConfig
    nullspace: NullspaceControllerConfig


@dataclass(slots=True)
class CableSpaceFallbackConfig:
    axis_ids: tuple[int, ...]
    mm_per_turn: tuple[float, ...]
    home_cable_mm: tuple[float, ...]
    geometry: CableRobotGeometry
    torque_per_tension: float
    torque_ctrl_kp_n_per_mm: float
    torque_ctrl_kd_n_per_mmps: float
    torque_ctrl_bias_n: float
    torque_ctrl_min_n: float
    torque_ctrl_max_n: float


@dataclass(slots=True)
class TaskspaceControllerState:
    prev_tensions_n: np.ndarray | None = None
    last_perf_s: float | None = None
    null_eta_m: float = 0.0
    null_eta_int: float = 0.0
    null_sigma_ref_n: float = 0.0
    null_basis_prev: np.ndarray | None = None


@dataclass(slots=True)
class ControllerDebugSnapshot:
    spool_cmd_mm: tuple[float, ...]
    spool_pose_cmd_mm: tuple[float, ...]
    spool_null_cmd_mm: tuple[float, ...]
    torque_cmd_nm: tuple[float, ...]
    tension_cmd_n: tuple[float, ...]
    tau_plat_des: tuple[float, ...]
    sigma_ref_n: float
    sigma_meas_n: float
    eta_null_m: float


@dataclass(slots=True)
class ControllerStepResult:
    commands: tuple[ActuatorCommand, ...]
    state: TaskspaceControllerState
    debug: ControllerDebugSnapshot
    kinematics_duration_s: float | None = None
    tension_solver_duration_s: float | None = None


def _safe_measured_tensions(actuator_states: Sequence[ActuatorState], axis_count: int):
    tensions = []
    for axis_state in actuator_states[:axis_count]:
        if axis_state.tension_estimate_n is None:
            return None
        tensions.append(float(axis_state.tension_estimate_n))
    tensions = np.asarray(tensions, dtype=float)
    if tensions.shape != (axis_count,) or not np.all(np.isfinite(tensions)):
        return None
    return tensions


def compute_taskspace_spool_commands(
    *,
    pose_t_mm,
    pose_q,
    linear_velocity_mps,
    linear_acceleration_mps2,
    q_cur,
    qd_cur,
    j_outer,
    actuator_states: Sequence[ActuatorState],
    config: TaskspaceControllerConfig,
    state: TaskspaceControllerState,
    now_perf: float | None = None,
) -> ControllerStepResult:
    """Generate spool references from a task-space command."""
    if now_perf is None:
        now_perf = time.perf_counter()

    t_kin_start = time.perf_counter()
    roll_cmd, pitch_cmd, _ = quat_to_rpy_rad(pose_q)
    q_ref = np.array(
        [pose_t_mm[0] / 1000.0, pose_t_mm[1] / 1000.0, pose_t_mm[2] / 1000.0, roll_cmd, pitch_cmd],
        dtype=float,
    )
    qd_ref = np.array(
        [float(linear_velocity_mps[0]), float(linear_velocity_mps[1]), float(linear_velocity_mps[2]), 0.0, 0.0],
        dtype=float,
    )
    qdd_ff = np.array(
        [
            float(linear_acceleration_mps2[0]),
            float(linear_acceleration_mps2[1]),
            float(linear_acceleration_mps2[2]),
            0.0,
            0.0,
        ],
        dtype=float,
    )
    cable_mm = pose_to_cable_lengths_mm(config.geometry, pose_t_mm, pose_q)
    pose_ref_m = np.asarray(
        [(cable_mm[i] - config.home_cable_mm[i]) / 1000.0 for i in range(len(config.axis_ids))],
        dtype=float,
    )
    pose_corr_m = np.zeros(len(config.axis_ids), dtype=float)
    null_cmd_m = np.zeros(len(config.axis_ids), dtype=float)
    j_outer_arr = None if j_outer is None else np.asarray(j_outer, dtype=float)

    if q_cur is not None and qd_cur is not None:
        q_cur = np.asarray(q_cur, dtype=float)
        qd_cur = np.asarray(qd_cur, dtype=float)
        error = q_ref - q_cur
        error_dot = qd_ref - qd_cur
        if j_outer_arr is None or j_outer_arr.shape != (len(config.axis_ids), 5):
            j_outer_arr = cable_lengths_jacobian_pose5_fd(q_cur, geometry=config.geometry)
        cur_lengths_m = cable_lengths_m_from_pose5(q_cur, geometry=config.geometry)
        cur_cmd_m = np.asarray(cur_lengths_m, dtype=float) - (np.asarray(config.home_cable_mm, dtype=float) / 1000.0)
        cable_corr_m = j_outer_arr @ ((config.outer_corr_kp @ error) + (config.outer_corr_kd @ error_dot))
        cable_corr_m = np.clip(cable_corr_m, -config.outer_corr_cable_clip_m, config.outer_corr_cable_clip_m)
        pose_corr_m = (cur_cmd_m - pose_ref_m) + cable_corr_m

    j_cmd = cable_lengths_jacobian_pose5_fd(q_ref, geometry=config.geometry)
    j_null = j_outer_arr if j_outer_arr is not None and np.shape(j_outer_arr) == (len(config.axis_ids), 5) else j_cmd
    t_solver_start = time.perf_counter()
    kinematics_duration_s = (t_solver_start - t_kin_start)

    tau_plat_ff = np.asarray(qdd_ff, dtype=float)
    tau_plat_ff[2] += float(config.gravity_ff_z_n)
    t_particular = solve_tensions_least_squares(j_cmd, tau_plat_ff, state.prev_tensions_n, config.tension_allocator)
    next_state = TaskspaceControllerState(
        prev_tensions_n=t_particular.copy(),
        last_perf_s=now_perf,
        null_eta_m=float(state.null_eta_m),
        null_eta_int=float(state.null_eta_int),
        null_sigma_ref_n=float(state.null_sigma_ref_n),
        null_basis_prev=None if state.null_basis_prev is None else np.asarray(state.null_basis_prev, dtype=float).copy(),
    )
    t_cmd = np.maximum(np.asarray(config.spool_bias_tension_n, dtype=float), t_particular)

    if state.last_perf_s is None:
        dt = 0.0
    else:
        dt = max(0.0, min(0.1, float(now_perf - float(state.last_perf_s))))

    sigma_meas = float("nan")
    sigma_ref = float("nan")
    if np.shape(j_null) == (len(config.axis_ids), 5):
        try:
            n_vec = cable_tension_nullspace_basis(j_null, next_state.null_basis_prev)
            next_state.null_basis_prev = n_vec.copy()
            lower, upper, feasible = nullspace_sigma_interval(t_cmd, n_vec, config.nullspace.tension_floor_n)
            sigma_target = float(config.nullspace.sigma_ref_n)
            if feasible:
                if math.isfinite(lower) and math.isfinite(upper):
                    sigma_target = 0.5 * (float(lower) + float(upper))
                else:
                    sigma_target = clamp_sigma_to_interval(config.nullspace.sigma_ref_n, lower, upper)
                sigma_ref = clamp_sigma_to_interval(
                    rate_limit_scalar(
                        sigma_target,
                        next_state.null_sigma_ref_n,
                        config.nullspace.sigma_rate_limit_nps,
                        dt,
                    ),
                    lower,
                    upper,
                )
            else:
                sigma_ref = sigma_target
            next_state.null_sigma_ref_n = float(sigma_ref)
            t_meas = _safe_measured_tensions(actuator_states, len(config.axis_ids))
            if t_meas is not None:
                sigma_meas = float(n_vec @ t_meas)
            if np.isfinite(sigma_meas) and dt > 0.0:
                sigma_err = float(sigma_ref - sigma_meas)
                next_state.null_eta_int += sigma_err * dt
                eta_dot = config.nullspace.kp * sigma_err + config.nullspace.ki * next_state.null_eta_int
                next_state.null_eta_m += dt * eta_dot
                next_state.null_eta_m = float(
                    np.clip(next_state.null_eta_m, -config.nullspace.eta_limit_m, config.nullspace.eta_limit_m)
                )
            null_cmd_m = n_vec * float(next_state.null_eta_m)
        except Exception:
            pass
    t_pack_start = time.perf_counter()
    tension_solver_duration_s = (t_pack_start - t_solver_start)

    cmd_m = pose_ref_m + pose_corr_m + null_cmd_m
    cmd_mm = tuple(1000.0 * float(v) for v in cmd_m)
    pose_cmd_mm = tuple(1000.0 * float(v) for v in (pose_ref_m + pose_corr_m))
    null_cmd_mm = tuple(1000.0 * float(v) for v in null_cmd_m)

    cmd_turns = [float(cmd_mm[i]) / float(config.mm_per_turn[i]) for i in range(len(config.axis_ids))]
    cable_vel_mps = j_cmd @ qd_ref
    vel_turnsps = []
    for i in range(len(config.axis_ids)):
        mm_per_turn = float(config.mm_per_turn[i])
        if abs(mm_per_turn) < 1e-9:
            vel_turnsps.append(0.0)
        else:
            vel_turnsps.append(float(1000.0 * cable_vel_mps[i]) / mm_per_turn)

    torque_ff = [float(config.torque_per_tension) * float(tension) for tension in t_cmd]
    if not config.enable_position_torque_ff:
        torque_ff = [0.0] * len(torque_ff)
    commands = tuple(
        ActuatorCommand(
            axis_id=axis_id,
            control_mode=ActuatorControlMode.POSITION,
            position_turns=float(cmd_turns[i]),
            velocity_ff_turns_per_s=float(vel_turnsps[i]),
            torque_ff_nm=float(torque_ff[i]),
        )
        for i, axis_id in enumerate(config.axis_ids)
    )
    debug = ControllerDebugSnapshot(
        spool_cmd_mm=cmd_mm,
        spool_pose_cmd_mm=pose_cmd_mm,
        spool_null_cmd_mm=null_cmd_mm,
        torque_cmd_nm=tuple(float(x) for x in torque_ff),
        tension_cmd_n=tuple(float(x) for x in t_cmd),
        tau_plat_des=tuple(float(x) for x in tau_plat_ff),
        sigma_ref_n=float(sigma_ref),
        sigma_meas_n=float(sigma_meas),
        eta_null_m=float(next_state.null_eta_m),
    )
    kinematics_duration_s += max(0.0, time.perf_counter() - t_pack_start)
    return ControllerStepResult(
        commands=commands,
        state=next_state,
        debug=debug,
        kinematics_duration_s=float(kinematics_duration_s),
        tension_solver_duration_s=float(tension_solver_duration_s),
    )


def compute_cablespace_fallback_commands(
    *,
    pose_t_mm,
    pose_q,
    actuator_states: Sequence[ActuatorState],
    config: CableSpaceFallbackConfig,
) -> ControllerStepResult:
    """
    Fallback controller for drivers without platform-state feedback:
    cable-space PD + bias tension.
    """
    t_kin_start = time.perf_counter()
    cable_mm = pose_to_cable_lengths_mm(config.geometry, pose_t_mm, pose_q)
    cmd_mm = [cable_mm[i] - config.home_cable_mm[i] for i in range(len(config.axis_ids))]
    torque_cmd = [0.0] * len(config.axis_ids)
    tension_cmd = [0.0] * len(config.axis_ids)
    commands = []

    for i, axis_id in enumerate(config.axis_ids):
        axis_state = actuator_states[i] if i < len(actuator_states) else None
        pos_turns = None if axis_state is None else axis_state.position_turns
        vel_turnsps = None if axis_state is None else axis_state.velocity_turns_per_s
        if pos_turns is None or vel_turnsps is None:
            commands.append(
                ActuatorCommand(
                    axis_id=axis_id,
                    control_mode=ActuatorControlMode.TORQUE,
                    torque_nm=0.0,
                )
            )
            continue

        feedback_mm = float(pos_turns) * float(config.mm_per_turn[i])
        feedback_mmps = float(vel_turnsps) * float(config.mm_per_turn[i])
        err_mm = float(cmd_mm[i]) - feedback_mm
        tension_n = (
            float(config.torque_ctrl_bias_n)
            + float(config.torque_ctrl_kp_n_per_mm) * err_mm
            - float(config.torque_ctrl_kd_n_per_mmps) * feedback_mmps
        )
        tension_n = max(float(config.torque_ctrl_min_n), min(float(config.torque_ctrl_max_n), tension_n))
        torque_nm = float(tension_n) * float(config.torque_per_tension)
        torque_cmd[i] = float(torque_nm)
        tension_cmd[i] = float(tension_n)
        commands.append(
            ActuatorCommand(
                axis_id=axis_id,
                control_mode=ActuatorControlMode.TORQUE,
                torque_nm=float(torque_nm),
            )
        )

    debug = ControllerDebugSnapshot(
        spool_cmd_mm=tuple(float(x) for x in cmd_mm),
        spool_pose_cmd_mm=tuple(float(x) for x in cmd_mm),
        spool_null_cmd_mm=tuple(0.0 for _ in config.axis_ids),
        torque_cmd_nm=tuple(float(x) for x in torque_cmd),
        tension_cmd_n=tuple(float(x) for x in tension_cmd),
        tau_plat_des=tuple(float("nan") for _ in range(5)),
        sigma_ref_n=float("nan"),
        sigma_meas_n=float("nan"),
        eta_null_m=0.0,
    )
    return ControllerStepResult(
        commands=tuple(commands),
        state=TaskspaceControllerState(),
        debug=debug,
        kinematics_duration_s=float(time.perf_counter() - t_kin_start),
        tension_solver_duration_s=None,
    )
