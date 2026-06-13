"""Structured telemetry models for the controller GUI."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import time


NAN = float("nan")


@dataclass(frozen=True, slots=True)
class CommStats:
    can_rx_hz: float = NAN
    can_tx_hz: float = NAN
    can_msg_hz: float = NAN
    can_util_est: float = NAN
    pos_fbk_hz: float = NAN
    pos_fbk_period0_min_s: float = NAN
    pos_fbk_period0_max_s: float = NAN


@dataclass(frozen=True, slots=True)
class TelemetryFrame:
    source_id: str
    receipt_time_s: float
    wall_time_s: float
    control_time_s: float | None
    runtime_time_s: float | None
    sim_time_s: float | None
    sim_rt_factor: float | None
    sequence_id: int | None
    control_state: str | None
    profile_active: bool
    pos_mm: tuple[float, ...]
    vel_mmps: tuple[float, ...]
    temp_fet_c: tuple[float, ...]
    temp_motor_c: tuple[float, ...]
    bus_current_a: tuple[float, ...]
    motor_current_a: tuple[float, ...]
    torque_cmd_nm: tuple[float, ...]
    torque_rsp_nm: tuple[float, ...]
    tension_cmd_n: tuple[float, ...]
    tension_rsp_n: tuple[float, ...]
    bus_voltage_v: tuple[float, ...]
    axis_state: tuple[int | None, ...]
    axis_error: tuple[int | None, ...]
    hand_cmd_pose: tuple[float, ...]
    hand_cmd_vel: tuple[float, ...]
    hand_cmd_acc: tuple[float, ...]
    hand_est_pose: tuple[float, ...]
    hand_est_vel: tuple[float, ...]
    hand_est_acc: tuple[float, ...]
    homing: Mapping[str, object] = field(default_factory=dict)
    comm_stats: CommStats = field(default_factory=CommStats)
    debug: Mapping[str, object] = field(default_factory=dict)
    raw: Mapping[str, object] = field(default_factory=dict)

    def preferred_time_s(self) -> float:
        if self.control_time_s is not None and math.isfinite(self.control_time_s):
            return float(self.control_time_s)
        if self.sim_time_s is not None and math.isfinite(self.sim_time_s):
            return float(self.sim_time_s)
        if math.isfinite(self.wall_time_s):
            return float(self.wall_time_s)
        return float(self.receipt_time_s)


@dataclass(frozen=True, slots=True)
class SessionConfig:
    session_id: str = "live"
    host: str = "jugglepi.local"
    tcp_port: int = 5555
    udp_port: int = 5556


def _float_or_none(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _float_or_nan(value) -> float:
    parsed = _float_or_none(value)
    return NAN if parsed is None else parsed


def _int_or_none(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _vec_get(values, index, default=None):
    try:
        return values[index]
    except Exception:
        return default


def _pose_state_to_legacy_pose(pose_state) -> tuple[float, ...]:
    if not isinstance(pose_state, Mapping):
        return (NAN, NAN, NAN, NAN, NAN, NAN)
    pos_m = pose_state.get("position_m") or ()
    rpy_rad = pose_state.get("orientation_rpy_rad") or ()
    return (
        1000.0 * _float_or_nan(_vec_get(pos_m, 0, NAN)),
        1000.0 * _float_or_nan(_vec_get(pos_m, 1, NAN)),
        1000.0 * _float_or_nan(_vec_get(pos_m, 2, NAN)),
        math.degrees(_float_or_nan(_vec_get(rpy_rad, 0, NAN))),
        math.degrees(_float_or_nan(_vec_get(rpy_rad, 1, NAN))),
        math.degrees(_float_or_nan(_vec_get(rpy_rad, 2, NAN))),
    )


def _pose_state_to_legacy_velocity(pose_state) -> tuple[float, ...]:
    if not isinstance(pose_state, Mapping):
        return (NAN, NAN, NAN, NAN, NAN, NAN)
    linear = pose_state.get("linear_velocity_mps") or ()
    angular = pose_state.get("angular_velocity_rps") or ()
    return (
        1000.0 * _float_or_nan(_vec_get(linear, 0, NAN)),
        1000.0 * _float_or_nan(_vec_get(linear, 1, NAN)),
        1000.0 * _float_or_nan(_vec_get(linear, 2, NAN)),
        math.degrees(_float_or_nan(_vec_get(angular, 0, NAN))),
        math.degrees(_float_or_nan(_vec_get(angular, 1, NAN))),
        math.degrees(_float_or_nan(_vec_get(angular, 2, NAN))),
    )


def _pose_state_to_legacy_acceleration(pose_state) -> tuple[float, ...]:
    if not isinstance(pose_state, Mapping):
        return (NAN, NAN, NAN, NAN, NAN, NAN)
    linear = pose_state.get("linear_acceleration_mps2") or ()
    angular = pose_state.get("angular_acceleration_rps2") or ()
    return (
        1000.0 * _float_or_nan(_vec_get(linear, 0, NAN)),
        1000.0 * _float_or_nan(_vec_get(linear, 1, NAN)),
        1000.0 * _float_or_nan(_vec_get(linear, 2, NAN)),
        math.degrees(_float_or_nan(_vec_get(angular, 0, NAN))),
        math.degrees(_float_or_nan(_vec_get(angular, 1, NAN))),
        math.degrees(_float_or_nan(_vec_get(angular, 2, NAN))),
    )


def _actuator_field_vec(actuators, field_name: str, *, cast="float") -> tuple[float | int | None, ...]:
    values = []
    if not isinstance(actuators, list):
        actuators = []
    for index in range(6):
        axis = actuators[index] if index < len(actuators) and isinstance(actuators[index], Mapping) else {}
        raw = axis.get(field_name)
        if cast == "int":
            values.append(_int_or_none(raw))
        else:
            values.append(_float_or_nan(raw))
    return tuple(values)


def _float_tuple(values, count: int, *, scale: float = 1.0) -> tuple[float, ...]:
    values = values if isinstance(values, (list, tuple)) else ()
    out = []
    for index in range(count):
        out.append(scale * _float_or_nan(_vec_get(values, index, NAN)))
    return tuple(out)


def _int_tuple(values, count: int) -> tuple[int | None, ...]:
    values = values if isinstance(values, (list, tuple)) else ()
    out = []
    for index in range(count):
        out.append(_int_or_none(_vec_get(values, index)))
    return tuple(out)


def _mapping_or_empty(value) -> Mapping[str, object]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _normalize_legacy_payload(payload: Mapping[str, object], source_id: str, receipt_time_s: float) -> TelemetryFrame:
    debug = _mapping_or_empty(payload.get("debug"))
    control_time_s = _float_or_none(payload.get("control_time_s"))
    if control_time_s is None:
        control_time_s = _float_or_none(debug.get("control_time_s"))
    runtime_time_s = _float_or_none(payload.get("runtime_time_s"))
    if runtime_time_s is None:
        runtime_time_s = _float_or_none(debug.get("runtime_time_s"))
    sim_time_s = _float_or_none(debug.get("sim_time_s"))
    sim_rt_factor = _float_or_none(payload.get("sim_rt_factor"))
    if sim_rt_factor is None:
        sim_rt_factor = _float_or_none(debug.get("sim_rt_factor"))
    return TelemetryFrame(
        source_id=source_id,
        receipt_time_s=receipt_time_s,
        wall_time_s=_float_or_none(payload.get("t")) or receipt_time_s,
        control_time_s=control_time_s,
        runtime_time_s=runtime_time_s,
        sim_time_s=sim_time_s,
        sim_rt_factor=sim_rt_factor,
        sequence_id=_int_or_none(payload.get("sequence_id")),
        control_state=str(payload.get("control_state")) if payload.get("control_state") is not None else None,
        profile_active=bool(payload.get("profile_active", False)),
        pos_mm=_float_tuple(payload.get("pos"), 6),
        vel_mmps=_float_tuple(payload.get("vel"), 6),
        temp_fet_c=_float_tuple(payload.get("temp_fet"), 6),
        temp_motor_c=_float_tuple(payload.get("temp_motor"), 6),
        bus_current_a=_float_tuple(payload.get("bus_i"), 6),
        motor_current_a=_float_tuple(payload.get("motor_i"), 6),
        torque_cmd_nm=_float_tuple(payload.get("torque_cmd_nm"), 6),
        torque_rsp_nm=_float_tuple(payload.get("torque_rsp_nm"), 6),
        tension_cmd_n=_float_tuple(payload.get("tension_cmd_n"), 6),
        tension_rsp_n=_float_tuple(payload.get("tension_rsp_n"), 6),
        bus_voltage_v=_float_tuple(payload.get("bus_v"), 6),
        axis_state=_int_tuple(payload.get("axis_state"), 6),
        axis_error=_int_tuple(payload.get("axis_error"), 6),
        hand_cmd_pose=_float_tuple(payload.get("hand_cmd_pose"), 6),
        hand_cmd_vel=_float_tuple(payload.get("hand_cmd_vel"), 6),
        hand_cmd_acc=_float_tuple(payload.get("hand_cmd_acc"), 6),
        hand_est_pose=_float_tuple(payload.get("hand_est_pose"), 6),
        hand_est_vel=_float_tuple(payload.get("hand_est_vel"), 6),
        hand_est_acc=_float_tuple(payload.get("hand_est_acc"), 6),
        homing={},
        comm_stats=CommStats(
            can_rx_hz=_float_or_nan(payload.get("can_rx_hz")),
            can_tx_hz=_float_or_nan(payload.get("can_tx_hz")),
            can_msg_hz=_float_or_nan(payload.get("can_msg_hz")),
            can_util_est=_float_or_nan(payload.get("can_util_est")),
            pos_fbk_hz=_float_or_nan(payload.get("pos_fbk_hz")),
            pos_fbk_period0_min_s=_float_or_nan(payload.get("pos_fbk_period0_min_s")),
            pos_fbk_period0_max_s=_float_or_nan(payload.get("pos_fbk_period0_max_s")),
        ),
        debug=debug,
        raw=dict(payload),
    )


def _normalize_snapshot_payload(payload: Mapping[str, object], source_id: str, receipt_time_s: float) -> TelemetryFrame:
    actuators = payload.get("actuators") or []
    bus_stats = _mapping_or_empty(payload.get("bus_stats"))
    debug = _mapping_or_empty(payload.get("debug"))
    control_time_s = _float_or_none(payload.get("control_time_s"))
    if control_time_s is None:
        control_time_s = _float_or_none(debug.get("control_time_s"))
    runtime_time_s = _float_or_none(payload.get("runtime_time_s"))
    if runtime_time_s is None:
        runtime_time_s = _float_or_none(debug.get("runtime_time_s"))
    sim_time_s = _float_or_none(payload.get("sim_time_s"))
    if sim_time_s is None:
        sim_time_s = _float_or_none(debug.get("sim_time_s"))
    sim_rt_factor = _float_or_none(payload.get("sim_rt_factor"))
    if sim_rt_factor is None:
        sim_rt_factor = _float_or_none(debug.get("sim_rt_factor"))
    return TelemetryFrame(
        source_id=source_id,
        receipt_time_s=receipt_time_s,
        wall_time_s=_float_or_none(payload.get("timestamp_s")) or receipt_time_s,
        control_time_s=control_time_s,
        runtime_time_s=runtime_time_s,
        sim_time_s=sim_time_s,
        sim_rt_factor=sim_rt_factor,
        sequence_id=_int_or_none(payload.get("sequence_id")),
        control_state=str(payload.get("control_state")) if payload.get("control_state") is not None else None,
        profile_active=bool(payload.get("profile_active", False)),
        pos_mm=_float_tuple(payload.get("cable_lengths_m"), 6, scale=1000.0),
        vel_mmps=_float_tuple(payload.get("cable_velocities_mps"), 6, scale=1000.0),
        temp_fet_c=_actuator_field_vec(actuators, "temperature_fet_c"),
        temp_motor_c=_actuator_field_vec(actuators, "temperature_motor_c"),
        bus_current_a=_actuator_field_vec(actuators, "bus_current_a"),
        motor_current_a=_actuator_field_vec(actuators, "current_estimate_a"),
        torque_cmd_nm=_float_tuple(payload.get("commanded_torques_nm"), 6),
        torque_rsp_nm=_float_tuple(payload.get("estimated_torques_nm"), 6),
        tension_cmd_n=_float_tuple(payload.get("commanded_tensions_n"), 6),
        tension_rsp_n=_float_tuple(payload.get("estimated_tensions_n"), 6),
        bus_voltage_v=_actuator_field_vec(actuators, "bus_voltage_v"),
        axis_state=_actuator_field_vec(actuators, "axis_state", cast="int"),
        axis_error=_actuator_field_vec(actuators, "error_flags", cast="int"),
        hand_cmd_pose=_pose_state_to_legacy_pose(payload.get("commanded_pose")),
        hand_cmd_vel=_pose_state_to_legacy_velocity(payload.get("commanded_pose")),
        hand_cmd_acc=_pose_state_to_legacy_acceleration(payload.get("commanded_pose")),
        hand_est_pose=_pose_state_to_legacy_pose(payload.get("estimated_pose")),
        hand_est_vel=_pose_state_to_legacy_velocity(payload.get("estimated_pose")),
        hand_est_acc=_pose_state_to_legacy_acceleration(payload.get("estimated_pose")),
        homing=_mapping_or_empty(payload.get("homing")),
        comm_stats=CommStats(
            can_rx_hz=_float_or_nan(bus_stats.get("can_rx_hz")),
            can_tx_hz=_float_or_nan(bus_stats.get("can_tx_hz")),
            can_msg_hz=_float_or_nan(bus_stats.get("can_msg_hz")),
            can_util_est=_float_or_nan(bus_stats.get("can_util_est")),
            pos_fbk_hz=_float_or_nan(bus_stats.get("pos_fbk_hz")),
            pos_fbk_period0_min_s=_float_or_nan(bus_stats.get("pos_fbk_period0_min_s")),
            pos_fbk_period0_max_s=_float_or_nan(bus_stats.get("pos_fbk_period0_max_s")),
        ),
        debug=debug,
        raw=dict(payload),
    )


def normalize_telemetry(payload: Mapping[str, object], source_id: str = "live", receipt_time_s: float | None = None) -> TelemetryFrame:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    receipt_time_s = time.time() if receipt_time_s is None else float(receipt_time_s)
    if "actuators" in payload or "commanded_pose" in payload or "estimated_pose" in payload:
        return _normalize_snapshot_payload(payload, source_id, receipt_time_s)
    return _normalize_legacy_payload(payload, source_id, receipt_time_s)
