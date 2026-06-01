"""Build typed robot snapshots from the mutable runtime mailbox."""

from __future__ import annotations

import time

from jugglebot.core.types import (
    ActuatorState,
    BusStats,
    FaultState,
    PoseState,
    RobotState,
    TimingStats,
    WatchdogStatus,
)
from jugglebot.core.pose_utils import quat_to_rpy_rad
from jugglebot.core.units import MM_PER_TURN


def _float_or_none(value):
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _float_or_nan(value):
    val = _float_or_none(value)
    return float("nan") if val is None else val


def _tuple3(values, default=float("nan")):
    if values is None:
        return (default, default, default)
    out = []
    for i in range(3):
        try:
            out.append(float(values[i]))
        except Exception:
            out.append(default)
    return tuple(out)


def _build_pose_state(t_mm, q, linear_velocity=None, angular_velocity=None, linear_acceleration=None):
    if t_mm is None or q is None:
        return None
    roll, pitch, yaw = quat_to_rpy_rad(q)
    return PoseState(
        position_m=(
            _float_or_nan(t_mm[0]) / 1000.0,
            _float_or_nan(t_mm[1]) / 1000.0,
            _float_or_nan(t_mm[2]) / 1000.0,
        ),
        orientation_rpy_rad=(float(roll), float(pitch), float(yaw)),
        linear_velocity_mps=_tuple3(linear_velocity),
        angular_velocity_rps=_tuple3(angular_velocity),
        linear_acceleration_mps2=_tuple3(linear_acceleration),
    )


def _flatten(value, prefix="", out=None):
    if out is None:
        out = {}
    if isinstance(value, dict):
        for key, item in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            _flatten(item, child_prefix, out)
        return out
    if isinstance(value, list):
        for idx, item in enumerate(value):
            child_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            _flatten(item, child_prefix, out)
        return out
    out[prefix] = value
    return out


def flatten_robot_state(snapshot: RobotState):
    return _flatten(snapshot.to_dict())


def build_robot_state_snapshot(
    state,
    *,
    timestamp_s: float | None = None,
    sequence_id: int | None = None,
    timing: TimingStats | None = None,
    fault_state: FaultState | None = None,
    debug: dict[str, object] | None = None,
    mm_per_turn: list[float] | tuple[float, ...] | None = None,
):
    timestamp_s = time.time() if timestamp_s is None else float(timestamp_s)
    sequence_id = state.next_snapshot_sequence() if sequence_id is None else int(sequence_id)
    if timing is None and hasattr(state, "get_timing_stats"):
        timing = state.get_timing_stats()
    watchdog = state.get_watchdog_status() if hasattr(state, "get_watchdog_status") else None
    axis_mm_per_turn = list(mm_per_turn or MM_PER_TURN)

    with state.lock:
        control_state = str(state.state)
        profile_active = state.player_thread is not None
        axes_pos = list(state.axes_pos_estimate)
        axes_vel = list(state.axes_vel_estimate)
        axes_bus_voltage = list(state.axes_bus_voltage)
        axes_bus_current = list(state.axes_bus_current)
        axes_motor_current = list(state.axes_motor_current)
        axes_temp_fet = list(state.axes_temp_fet)
        axes_temp_motor = list(state.axes_temp_motor)
        axes_axis_error = list(state.axes_axis_error)
        axes_axis_state = list(state.axes_axis_state)
        torque_cmd = list(state.axes_torque_cmd_nm)
        torque_rsp = list(state.axes_torque_rsp_nm)
        tension_cmd = list(state.axes_tension_cmd_n)
        tension_rsp = list(state.axes_tension_rsp_n)
        hand_t_mm = tuple(state.hand_t_mm)
        hand_q = tuple(state.hand_q)
        hand_v_mps = tuple(state.hand_v_mps)
        hand_a_mps2 = tuple(state.hand_a_mps2)
        hand_est_t_mm = tuple(state.hand_est_t_mm)
        hand_est_q = tuple(state.hand_est_q)
        hand_est_v_mps = tuple(state.hand_est_v_mps)
        hand_est_w_rps = tuple(state.hand_est_w_rps)
        comm = {
            "can_rx_hz": float(state.comm_can_rx_hz),
            "can_tx_hz": float(state.comm_can_tx_hz),
            "can_msg_hz": float(state.comm_can_msg_hz),
            "can_util_est": float(state.comm_can_util_est),
            "pos_fbk_hz": float(state.comm_pos_fbk_hz),
            "pos_fbk_period0_min_s": float(state.comm_pos_fbk_period0_min_s),
            "pos_fbk_period0_max_s": float(state.comm_pos_fbk_period0_max_s),
        }

    actuators = []
    cable_lengths_m = []
    cable_velocities_mps = []
    valid = True
    for i in range(6):
        pos_turns = _float_or_none(axes_pos[i] if i < len(axes_pos) else None)
        vel_turns_per_s = _float_or_none(axes_vel[i] if i < len(axes_vel) else None)
        mm_per_turn_i = float(axis_mm_per_turn[i])
        if pos_turns is None:
            cable_lengths_m.append(float("nan"))
            valid = False
        else:
            cable_lengths_m.append((pos_turns * mm_per_turn_i) / 1000.0)
        if vel_turns_per_s is None:
            cable_velocities_mps.append(float("nan"))
            valid = False
        else:
            cable_velocities_mps.append((vel_turns_per_s * mm_per_turn_i) / 1000.0)

        actuators.append(
            ActuatorState(
                axis_id=i,
                position_turns=pos_turns,
                velocity_turns_per_s=vel_turns_per_s,
                torque_estimate_nm=_float_or_none(torque_rsp[i] if i < len(torque_rsp) else None),
                current_estimate_a=_float_or_none(axes_motor_current[i] if i < len(axes_motor_current) else None),
                axis_state=axes_axis_state[i] if i < len(axes_axis_state) else None,
                error_flags=axes_axis_error[i] if i < len(axes_axis_error) else None,
                temperature_fet_c=_float_or_none(axes_temp_fet[i] if i < len(axes_temp_fet) else None),
                temperature_motor_c=_float_or_none(axes_temp_motor[i] if i < len(axes_temp_motor) else None),
                bus_voltage_v=_float_or_none(axes_bus_voltage[i] if i < len(axes_bus_voltage) else None),
                bus_current_a=_float_or_none(axes_bus_current[i] if i < len(axes_bus_current) else None),
                valid=pos_turns is not None,
                stale=False,
            )
        )

    commanded_pose = _build_pose_state(
        hand_t_mm,
        hand_q,
        linear_velocity=hand_v_mps,
        linear_acceleration=hand_a_mps2,
    )
    estimated_pose = _build_pose_state(
        hand_est_t_mm,
        hand_est_q,
        linear_velocity=hand_est_v_mps,
        angular_velocity=hand_est_w_rps,
    )

    return RobotState(
        timestamp_s=timestamp_s,
        sequence_id=sequence_id,
        control_state=control_state,
        profile_active=profile_active,
        actuators=tuple(actuators),
        cable_lengths_m=tuple(cable_lengths_m),
        cable_velocities_mps=tuple(cable_velocities_mps),
        commanded_pose=commanded_pose,
        estimated_pose=estimated_pose,
        commanded_tensions_n=tuple(_float_or_nan(v) for v in tension_cmd),
        estimated_tensions_n=tuple(_float_or_nan(v) for v in tension_rsp),
        commanded_torques_nm=tuple(_float_or_nan(v) for v in torque_cmd),
        estimated_torques_nm=tuple(_float_or_nan(v) for v in torque_rsp),
        fault_state=fault_state or FaultState(),
        timing=timing,
        watchdog=watchdog if isinstance(watchdog, WatchdogStatus) else None,
        bus_stats=BusStats(**comm),
        debug=dict(debug or {}),
        valid=valid,
    )
