"""Shared controller-facing data models."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from enum import Enum


def _serialize_value(value):
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {f.name: _serialize_value(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, tuple):
        return [_serialize_value(v) for v in value]
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    return value


class ModelBase:
    def to_dict(self):
        return _serialize_value(self)


class ActuatorControlMode(str, Enum):
    DISABLED = "disabled"
    POSITION = "position"
    VELOCITY = "velocity"
    TORQUE = "torque"
    CURRENT = "current"


class ActuatorInputMode(str, Enum):
    PASSTHROUGH = "passthrough"
    TRAPEZOIDAL = "trapezoidal"
    TRAJECTORY = "trajectory"
    UNKNOWN = "unknown"


class PoseCommandMode(str, Enum):
    HOLD = "hold"
    POSE = "pose"
    TRAJECTORY = "trajectory"


class TrajectoryUpdateMode(str, Enum):
    REPLACE = "replace"
    APPEND = "append"
    SPLICE_AT_TIME = "splice_at_time"
    SPLICE_NEXT_CYCLE = "splice_next_cycle"


class HomingMode(str, Enum):
    MANUAL = "manual"
    HOMING_PREP = "homing_prep"
    Z_AXIS_ZERO = "z_axis_zero"
    XY_AXIS_ZERO = "xy_axis_zero"


class HomingAction(str, Enum):
    RUN = "run"
    CANCEL = "cancel"
    APPLY = "apply"


class FaultSeverity(str, Enum):
    NONE = "none"
    WARNING = "warning"
    FAULT = "fault"
    ESTOP = "estop"


class RuntimeHealthLevel(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    VIOLATION = "violation"


@dataclass(slots=True)
class FaultState(ModelBase):
    active: bool = False
    severity: FaultSeverity = FaultSeverity.NONE
    code: str | None = None
    source: str | None = None
    message: str | None = None
    timestamp_s: float | None = None


@dataclass(slots=True)
class ActuatorState(ModelBase):
    axis_id: int
    feedback_timestamp_s: float | None = None
    position_turns: float | None = None
    velocity_turns_per_s: float | None = None
    torque_estimate_nm: float | None = None
    tension_estimate_n: float | None = None
    current_estimate_a: float | None = None
    axis_state: int | str | None = None
    error_flags: int | None = None
    proc_result: int | None = None
    temperature_fet_c: float | None = None
    temperature_motor_c: float | None = None
    bus_voltage_v: float | None = None
    bus_current_a: float | None = None
    feedback_age_s: float | None = None
    valid: bool = True
    stale: bool = False


@dataclass(slots=True)
class ActuatorCommand(ModelBase):
    axis_id: int
    control_mode: ActuatorControlMode
    input_mode: ActuatorInputMode = ActuatorInputMode.PASSTHROUGH
    apply_control_mode: bool = False
    timestamp_s: float | None = None
    position_turns: float | None = None
    velocity_turns_per_s: float | None = None
    torque_nm: float | None = None
    current_a: float | None = None
    velocity_ff_turns_per_s: float | None = None
    torque_ff_nm: float | None = None
    position_limit_turns: float | None = None
    velocity_limit_turns_per_s: float | None = None
    torque_limit_nm: float | None = None
    enable: bool | None = None


@dataclass(slots=True)
class PoseCommand(ModelBase):
    timestamp_s: float | None = None
    x_m: float = 0.0
    y_m: float = 0.0
    z_m: float = 0.0
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0
    linear_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_rps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_acceleration_mps2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_acceleration_rps2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    command_mode: PoseCommandMode = PoseCommandMode.POSE


@dataclass(slots=True)
class PoseState(ModelBase):
    position_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_velocity_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity_rps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    linear_acceleration_mps2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_acceleration_rps2: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(slots=True)
class TrajectoryWaypoint(ModelBase):
    time_from_start_s: float
    pose: PoseCommand


@dataclass(slots=True)
class TrajectoryCommand(ModelBase):
    sequence_id: int
    start_time_s: float | None = None
    frame: str = "world"
    waypoints: tuple[TrajectoryWaypoint, ...] = ()


@dataclass(slots=True)
class TrajectoryUpdate(ModelBase):
    sequence_id: int
    mode: TrajectoryUpdateMode
    trajectory: TrajectoryCommand
    source_timestamp_s: float | None = None
    effective_time_s: float | None = None
    preserve_continuity: bool = True
    note: str | None = None


@dataclass(slots=True)
class HomingCommand(ModelBase):
    action: HomingAction
    mode: HomingMode | None = None
    source_timestamp_s: float | None = None


@dataclass(slots=True)
class HomingStatus(ModelBase):
    selected_mode: HomingMode = HomingMode.MANUAL
    active_mode: HomingMode | None = None
    state: str = "idle"
    phase: str = "idle"
    progress: float = 0.0
    run_id: int = 0
    total_points: int = 0
    completed_points: int = 0
    accepted_samples: int = 0
    rejected_samples: int = 0
    result_available: bool = False
    fitted_offset_mm: tuple[float, ...] = (
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    )
    candidate_home_pos_mm: tuple[float, ...] = (
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
    )
    residual_rms_mm: float = float("nan")
    message: str | None = None
    failure_reason: str | None = None


@dataclass(slots=True)
class TimingStats(ModelBase):
    loop_period_s: float | None = None
    read_duration_s: float | None = None
    observer_duration_s: float | None = None
    trajectory_duration_s: float | None = None
    kinematics_duration_s: float | None = None
    tension_solver_duration_s: float | None = None
    command_write_duration_s: float | None = None
    total_loop_duration_s: float | None = None
    deadline_margin_s: float | None = None
    missed_deadline_count: int = 0
    feedback_age_s: float | None = None
    bus_utilization_estimate: float | None = None


@dataclass(slots=True)
class WatchdogStatus(ModelBase):
    level: RuntimeHealthLevel = RuntimeHealthLevel.HEALTHY
    mode: str = "unknown"
    message: str | None = None
    transition_grace_active: bool = False
    deadline_margin_s: float | None = None
    feedback_age_s: float | None = None
    missed_deadline_count: int = 0
    missed_deadline_delta: int = 0
    consecutive_missed_deadlines: int = 0
    low_deadline_margin: bool = False
    excessive_missed_deadlines: bool = False
    stale_feedback: bool = False


@dataclass(slots=True)
class BusStats(ModelBase):
    can_rx_hz: float | None = None
    can_tx_hz: float | None = None
    can_msg_hz: float | None = None
    can_util_est: float | None = None
    pos_fbk_hz: float | None = None
    pos_fbk_period0_min_s: float | None = None
    pos_fbk_period0_max_s: float | None = None


@dataclass(slots=True)
class RobotState(ModelBase):
    timestamp_s: float
    sequence_id: int
    control_time_s: float | None = None
    runtime_time_s: float | None = None
    sim_time_s: float | None = None
    sim_rt_factor: float | None = None
    control_state: str = "unknown"
    profile_active: bool = False
    actuators: tuple[ActuatorState, ...] = ()
    cable_lengths_m: tuple[float, ...] = ()
    cable_velocities_mps: tuple[float, ...] = ()
    commanded_pose: PoseState | None = None
    estimated_pose: PoseState | None = None
    commanded_tensions_n: tuple[float, ...] = ()
    estimated_tensions_n: tuple[float, ...] = ()
    commanded_torques_nm: tuple[float, ...] = ()
    estimated_torques_nm: tuple[float, ...] = ()
    fault_state: FaultState = field(default_factory=FaultState)
    timing: TimingStats | None = None
    watchdog: WatchdogStatus | None = None
    bus_stats: BusStats | None = None
    homing: HomingStatus | None = None
    debug: dict[str, object] = field(default_factory=dict)
    valid: bool = True
