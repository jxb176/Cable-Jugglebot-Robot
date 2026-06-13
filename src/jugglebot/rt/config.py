"""Typed runtime configuration with strict validation."""

from __future__ import annotations

from dataclasses import dataclass

from jugglebot.config import load_config as load_raw_config


class ConfigError(ValueError):
    """Raised when runtime configuration is missing or invalid."""


@dataclass(slots=True, frozen=True)
class RobotConfig:
    name: str
    control_rate_hz: float
    units: str
    mode: str
    auto_enable_on_startup: bool


@dataclass(slots=True, frozen=True)
class GeometryConfig:
    capstan_radius_m: float


@dataclass(slots=True, frozen=True)
class TensionConfig:
    tension_min_n: float
    tension_max_n: float
    regularization_lambda: float
    alpha_blend: float


@dataclass(slots=True, frozen=True)
class FallbackTensionConfig:
    kp_n_per_mm: float
    kd_n_per_mmps: float
    bias_n: float
    min_n: float
    max_n: float


@dataclass(slots=True, frozen=True)
class OuterTaskspaceCorrectionConfig:
    kp: tuple[float, float, float, float, float]
    kd: tuple[float, float, float, float, float]
    cable_clip_m: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class NullspaceTensionConfig:
    kp: float
    ki: float
    eta_limit_m: float
    sigma_ref_n: float
    sigma_rate_limit_nps: float
    tension_floor_n: float


@dataclass(slots=True, frozen=True)
class SpoolSpaceControllerConfig:
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    gravity_ff_z_n: float
    enable_torque_feedforward: bool
    bias_tension_n: tuple[float, ...]
    torque_limit_nm: tuple[float, ...]
    fallback_tension: FallbackTensionConfig
    outer_taskspace_correction: OuterTaskspaceCorrectionConfig
    nullspace_tension: NullspaceTensionConfig


@dataclass(slots=True, frozen=True)
class ManualInputWorkspaceConfig:
    radius_m: float
    z_min_m: float
    z_max_m: float


@dataclass(slots=True, frozen=True)
class ManualInputOrientationConfig:
    roll_limit_deg: float
    pitch_limit_deg: float


@dataclass(slots=True, frozen=True)
class ManualInputPositionModeConfig:
    linear_xy_scale_m: float
    linear_z_scale_m: float
    angular_scale_deg: float
    filter_tau_s: float
    linear_velocity_limit_mps: float
    angular_velocity_limit_degps: float


@dataclass(slots=True, frozen=True)
class ManualInputVelocityModeConfig:
    linear_velocity_limit_mps: float
    angular_velocity_limit_degps: float
    linear_accel_limit_mps2: float
    angular_accel_limit_degps2: float


@dataclass(slots=True, frozen=True)
class ManualInputAccelerationModeConfig:
    linear_accel_limit_mps2: float
    angular_accel_limit_degps2: float
    linear_velocity_limit_mps: float
    angular_velocity_limit_degps: float


@dataclass(slots=True, frozen=True)
class ManualInputConfig:
    stream_timeout_s: float
    deadband: float
    workspace: ManualInputWorkspaceConfig
    orientation: ManualInputOrientationConfig
    position_mode: ManualInputPositionModeConfig
    velocity_mode: ManualInputVelocityModeConfig
    acceleration_mode: ManualInputAccelerationModeConfig


@dataclass(slots=True, frozen=True)
class HomingSettleConfig:
    min_dwell_s: float
    timeout_s: float
    pose_tol_mm: float
    vel_tol_mmps: float
    tension_tol_n: float
    sigma_err_tol_n: float
    null_cmd_tol_mm: float


@dataclass(slots=True, frozen=True)
class HomingPrepConfig:
    enabled: bool
    speed_mps: float
    dwell_s: float
    cycles: int
    z_levels_mm: tuple[float, ...]
    radius_mm: float


@dataclass(slots=True, frozen=True)
class ZAxisZeroConfig:
    points_mm: tuple[float, ...]
    x_mm: float
    y_mm: float


@dataclass(slots=True, frozen=True)
class XYAxisZeroConfig:
    radius_mm: float
    point_count: int
    start_angle_deg: float
    z_levels_mm: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class HomingConfig:
    auto_apply: bool
    require_tension_feedback: bool
    require_taskspace_controller: bool
    settle: HomingSettleConfig
    prep: HomingPrepConfig
    z_axis_zero: ZAxisZeroConfig
    xy_axis_zero: XYAxisZeroConfig


@dataclass(slots=True, frozen=True)
class ControllerConfig:
    spool_space: SpoolSpaceControllerConfig
    manual_input: ManualInputConfig


@dataclass(slots=True, frozen=True)
class AllocationConfig:
    backend: str
    max_iters: int
    wrench_from_tension_sign: float


@dataclass(slots=True, frozen=True)
class EstimatorConfig:
    rate_hz: float
    enable_velocity_filter: bool
    vel_filter_cutoff_hz: float


@dataclass(slots=True, frozen=True)
class HardwareCanConfig:
    interface: str
    bitrate: int


@dataclass(slots=True, frozen=True)
class ODriveConfig:
    axis_ids: tuple[int, ...]
    mm_per_turn: tuple[float, ...]
    torque_direction: float
    input_vel_scale: float
    input_torque_scale: float


@dataclass(slots=True, frozen=True)
class HardwareConfig:
    can: HardwareCanConfig
    odrive: ODriveConfig


@dataclass(slots=True, frozen=True)
class TelemetryUdpConfig:
    enabled: bool
    host: str
    port: int


@dataclass(slots=True, frozen=True)
class LoggingConfig:
    level: str
    log_directory: str
    telemetry_udp: TelemetryUdpConfig


@dataclass(slots=True, frozen=True)
class RuntimeLoopConfig:
    status_report_period_s: float
    deadline_warning_margin_ms: float
    deadline_warning_missed_per_report: int
    feedback_age_warning_ms: float
    transition_grace_s: float


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    robot: RobotConfig
    geometry: GeometryConfig
    tension: TensionConfig
    controller: ControllerConfig
    homing: HomingConfig
    allocation: AllocationConfig
    estimator: EstimatorConfig
    hardware: HardwareConfig
    logging: LoggingConfig
    rt: RuntimeLoopConfig


def _require_mapping(data: dict, key: str, path: str) -> dict:
    value = data.get(key)
    if not isinstance(value, dict):
        raise ConfigError(f"Missing or invalid config section: {path}.{key}")
    return value


def _require_str(data: dict, key: str, path: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"Missing or invalid string config value: {path}.{key}")
    return value


def _require_bool(data: dict, key: str, path: str) -> bool:
    value = data.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"Missing or invalid boolean config value: {path}.{key}")
    return value


def _require_float(data: dict, key: str, path: str) -> float:
    value = data.get(key)
    if isinstance(value, bool) or value is None:
        raise ConfigError(f"Missing or invalid numeric config value: {path}.{key}")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Missing or invalid numeric config value: {path}.{key}") from exc


def _require_positive_float(data: dict, key: str, path: str) -> float:
    value = _require_float(data, key, path)
    if value <= 0.0:
        raise ConfigError(f"Config value must be > 0: {path}.{key}")
    return value


def _require_nonnegative_float(data: dict, key: str, path: str) -> float:
    value = _require_float(data, key, path)
    if value < 0.0:
        raise ConfigError(f"Config value must be >= 0: {path}.{key}")
    return value


def _require_int(data: dict, key: str, path: str) -> int:
    value = data.get(key)
    if isinstance(value, bool) or value is None:
        raise ConfigError(f"Missing or invalid integer config value: {path}.{key}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Missing or invalid integer config value: {path}.{key}") from exc


def _require_positive_int(data: dict, key: str, path: str) -> int:
    value = _require_int(data, key, path)
    if value <= 0:
        raise ConfigError(f"Config value must be > 0: {path}.{key}")
    return value


def _require_float_sequence(data: dict, key: str, path: str, length: int) -> tuple[float, ...]:
    value = data.get(key)
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ConfigError(f"Config value must be a length-{length} list: {path}.{key}")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Config value must be numeric: {path}.{key}") from exc


def _require_nonempty_float_sequence(data: dict, key: str, path: str) -> tuple[float, ...]:
    value = data.get(key)
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ConfigError(f"Config value must be a non-empty list: {path}.{key}")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Config value must be numeric: {path}.{key}") from exc


def _require_int_sequence(data: dict, key: str, path: str) -> tuple[int, ...]:
    value = data.get(key)
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ConfigError(f"Config value must be a non-empty list: {path}.{key}")
    try:
        parsed = tuple(int(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Config value must be integer-valued: {path}.{key}") from exc
    return parsed


def parse_runtime_config(raw: dict) -> RuntimeConfig:
    if not isinstance(raw, dict):
        raise ConfigError("Runtime config must be a mapping")

    robot_cfg = _require_mapping(raw, "robot", "config")
    geometry_cfg = _require_mapping(raw, "geometry", "config")
    tension_cfg = _require_mapping(raw, "tension", "config")
    controller_cfg = _require_mapping(raw, "controller", "config")
    homing_cfg = _require_mapping(raw, "homing", "config")
    allocation_cfg = _require_mapping(raw, "allocation", "config")
    estimator_cfg = _require_mapping(raw, "estimator", "config")
    hardware_cfg = _require_mapping(raw, "hardware", "config")
    logging_cfg = _require_mapping(raw, "logging", "config")
    rt_cfg = _require_mapping(raw, "rt", "config")

    hardware_can_cfg = _require_mapping(hardware_cfg, "can", "config.hardware")
    hardware_odrive_cfg = _require_mapping(hardware_cfg, "odrive", "config.hardware")
    axis_ids = _require_int_sequence(hardware_odrive_cfg, "axis_ids", "config.hardware.odrive")
    axis_count = len(axis_ids)

    spool_space_cfg = _require_mapping(controller_cfg, "spool_space", "config.controller")
    manual_input_cfg = _require_mapping(controller_cfg, "manual_input", "config.controller")
    outer_cfg = _require_mapping(
        spool_space_cfg,
        "outer_taskspace_correction",
        "config.controller.spool_space",
    )
    null_cfg = _require_mapping(
        spool_space_cfg,
        "nullspace_tension",
        "config.controller.spool_space",
    )
    fallback_cfg = _require_mapping(
        spool_space_cfg,
        "fallback_tension",
        "config.controller.spool_space",
    )
    manual_workspace_cfg = _require_mapping(
        manual_input_cfg,
        "workspace",
        "config.controller.manual_input",
    )
    manual_orientation_cfg = _require_mapping(
        manual_input_cfg,
        "orientation",
        "config.controller.manual_input",
    )
    manual_position_cfg = _require_mapping(
        manual_input_cfg,
        "position_mode",
        "config.controller.manual_input",
    )
    manual_velocity_cfg = _require_mapping(
        manual_input_cfg,
        "velocity_mode",
        "config.controller.manual_input",
    )
    manual_accel_cfg = _require_mapping(
        manual_input_cfg,
        "acceleration_mode",
        "config.controller.manual_input",
    )
    homing_settle_cfg = _require_mapping(homing_cfg, "settle", "config.homing")
    homing_prep_cfg = _require_mapping(homing_cfg, "prep", "config.homing")
    homing_z_axis_cfg = _require_mapping(homing_cfg, "z_axis_zero", "config.homing")
    homing_xy_axis_cfg = _require_mapping(homing_cfg, "xy_axis_zero", "config.homing")
    telemetry_udp_cfg = _require_mapping(logging_cfg, "telemetry_udp", "config.logging")

    tension_min_n = _require_nonnegative_float(tension_cfg, "Tmin_N", "config.tension")
    tension_max_n = _require_positive_float(tension_cfg, "Tmax_N", "config.tension")
    if tension_max_n < tension_min_n:
        raise ConfigError("config.tension.Tmax_N must be >= config.tension.Tmin_N")
    workspace_z_min_m = _require_float(
        manual_workspace_cfg,
        "z_min_m",
        "config.controller.manual_input.workspace",
    )
    workspace_z_max_m = _require_float(
        manual_workspace_cfg,
        "z_max_m",
        "config.controller.manual_input.workspace",
    )
    if workspace_z_max_m <= workspace_z_min_m:
        raise ConfigError(
            "config.controller.manual_input.workspace.z_max_m must be > z_min_m"
        )

    fallback_min_n = _require_nonnegative_float(fallback_cfg, "min_n", "config.controller.spool_space.fallback_tension")
    fallback_max_n = _require_positive_float(fallback_cfg, "max_n", "config.controller.spool_space.fallback_tension")
    if fallback_max_n < fallback_min_n:
        raise ConfigError(
            "config.controller.spool_space.fallback_tension.max_n must be >= min_n"
        )

    homing_timeout_s = _require_positive_float(homing_settle_cfg, "timeout_s", "config.homing.settle")
    homing_min_dwell_s = _require_nonnegative_float(homing_settle_cfg, "min_dwell_s", "config.homing.settle")
    if homing_timeout_s < homing_min_dwell_s:
        raise ConfigError("config.homing.settle.timeout_s must be >= min_dwell_s")

    homing_prep_cycles = _require_positive_int(homing_prep_cfg, "cycles", "config.homing.prep")
    homing_z_points_mm = _require_nonempty_float_sequence(homing_z_axis_cfg, "points_mm", "config.homing.z_axis_zero")
    homing_xy_z_levels_mm = _require_nonempty_float_sequence(
        homing_xy_axis_cfg,
        "z_levels_mm",
        "config.homing.xy_axis_zero",
    )
    homing_prep_z_levels_mm = _require_nonempty_float_sequence(
        homing_prep_cfg,
        "z_levels_mm",
        "config.homing.prep",
    )

    wrench_sign = _require_float(allocation_cfg, "wrench_from_tension_sign", "config.allocation")
    if wrench_sign not in (-1.0, 1.0):
        raise ConfigError("config.allocation.wrench_from_tension_sign must be -1.0 or 1.0")

    return RuntimeConfig(
        robot=RobotConfig(
            name=_require_str(robot_cfg, "name", "config.robot"),
            control_rate_hz=_require_positive_float(robot_cfg, "control_rate_hz", "config.robot"),
            units=_require_str(robot_cfg, "units", "config.robot"),
            mode=_require_str(robot_cfg, "mode", "config.robot"),
            auto_enable_on_startup=_require_bool(robot_cfg, "auto_enable_on_startup", "config.robot"),
        ),
        geometry=GeometryConfig(
            capstan_radius_m=_require_positive_float(geometry_cfg, "capstan_radius_m", "config.geometry"),
        ),
        tension=TensionConfig(
            tension_min_n=tension_min_n,
            tension_max_n=tension_max_n,
            regularization_lambda=_require_nonnegative_float(
                tension_cfg, "regularization_lambda", "config.tension"
            ),
            alpha_blend=_require_float(tension_cfg, "alpha_blend", "config.tension"),
        ),
        controller=ControllerConfig(
            spool_space=SpoolSpaceControllerConfig(
                kp=_require_float_sequence(spool_space_cfg, "kp", "config.controller.spool_space", axis_count),
                kd=_require_float_sequence(spool_space_cfg, "kd", "config.controller.spool_space", axis_count),
                gravity_ff_z_n=_require_float(spool_space_cfg, "gravity_ff_z_n", "config.controller.spool_space"),
                enable_torque_feedforward=_require_bool(
                    spool_space_cfg,
                    "enable_torque_feedforward",
                    "config.controller.spool_space",
                ),
                bias_tension_n=_require_float_sequence(
                    spool_space_cfg,
                    "bias_tension_N",
                    "config.controller.spool_space",
                    axis_count,
                ),
                torque_limit_nm=_require_float_sequence(
                    spool_space_cfg,
                    "torque_limit_nm",
                    "config.controller.spool_space",
                    axis_count,
                ),
                fallback_tension=FallbackTensionConfig(
                    kp_n_per_mm=_require_float(
                        fallback_cfg,
                        "kp_n_per_mm",
                        "config.controller.spool_space.fallback_tension",
                    ),
                    kd_n_per_mmps=_require_float(
                        fallback_cfg,
                        "kd_n_per_mmps",
                        "config.controller.spool_space.fallback_tension",
                    ),
                    bias_n=_require_float(
                        fallback_cfg,
                        "bias_n",
                        "config.controller.spool_space.fallback_tension",
                    ),
                    min_n=fallback_min_n,
                    max_n=fallback_max_n,
                ),
                outer_taskspace_correction=OuterTaskspaceCorrectionConfig(
                    kp=_require_float_sequence(
                        outer_cfg,
                        "kp",
                        "config.controller.spool_space.outer_taskspace_correction",
                        5,
                    ),
                    kd=_require_float_sequence(
                        outer_cfg,
                        "kd",
                        "config.controller.spool_space.outer_taskspace_correction",
                        5,
                    ),
                    cable_clip_m=_require_float_sequence(
                        outer_cfg,
                        "cable_clip_m",
                        "config.controller.spool_space.outer_taskspace_correction",
                        axis_count,
                    ),
                ),
                nullspace_tension=NullspaceTensionConfig(
                    kp=_require_float(null_cfg, "kp", "config.controller.spool_space.nullspace_tension"),
                    ki=_require_float(null_cfg, "ki", "config.controller.spool_space.nullspace_tension"),
                    eta_limit_m=_require_nonnegative_float(
                        null_cfg,
                        "eta_limit_m",
                        "config.controller.spool_space.nullspace_tension",
                    ),
                    sigma_ref_n=_require_float(
                        null_cfg,
                        "sigma_ref_N",
                        "config.controller.spool_space.nullspace_tension",
                    ),
                    sigma_rate_limit_nps=_require_nonnegative_float(
                        null_cfg,
                        "sigma_rate_limit_Nps",
                        "config.controller.spool_space.nullspace_tension",
                    ),
                    tension_floor_n=_require_nonnegative_float(
                        null_cfg,
                        "tmin_N",
                        "config.controller.spool_space.nullspace_tension",
                    ),
                ),
            ),
            manual_input=ManualInputConfig(
                stream_timeout_s=_require_positive_float(
                    manual_input_cfg,
                    "stream_timeout_s",
                    "config.controller.manual_input",
                ),
                deadband=_require_nonnegative_float(
                    manual_input_cfg,
                    "deadband",
                    "config.controller.manual_input",
                ),
                workspace=ManualInputWorkspaceConfig(
                    radius_m=_require_positive_float(
                        manual_workspace_cfg,
                        "radius_m",
                        "config.controller.manual_input.workspace",
                    ),
                    z_min_m=workspace_z_min_m,
                    z_max_m=workspace_z_max_m,
                ),
                orientation=ManualInputOrientationConfig(
                    roll_limit_deg=_require_nonnegative_float(
                        manual_orientation_cfg,
                        "roll_limit_deg",
                        "config.controller.manual_input.orientation",
                    ),
                    pitch_limit_deg=_require_nonnegative_float(
                        manual_orientation_cfg,
                        "pitch_limit_deg",
                        "config.controller.manual_input.orientation",
                    ),
                ),
                position_mode=ManualInputPositionModeConfig(
                    linear_xy_scale_m=_require_positive_float(
                        manual_position_cfg,
                        "linear_xy_scale_m",
                        "config.controller.manual_input.position_mode",
                    ),
                    linear_z_scale_m=_require_positive_float(
                        manual_position_cfg,
                        "linear_z_scale_m",
                        "config.controller.manual_input.position_mode",
                    ),
                    angular_scale_deg=_require_nonnegative_float(
                        manual_position_cfg,
                        "angular_scale_deg",
                        "config.controller.manual_input.position_mode",
                    ),
                    filter_tau_s=_require_nonnegative_float(
                        manual_position_cfg,
                        "filter_tau_s",
                        "config.controller.manual_input.position_mode",
                    ),
                    linear_velocity_limit_mps=_require_positive_float(
                        manual_position_cfg,
                        "linear_velocity_limit_mps",
                        "config.controller.manual_input.position_mode",
                    ),
                    angular_velocity_limit_degps=_require_positive_float(
                        manual_position_cfg,
                        "angular_velocity_limit_degps",
                        "config.controller.manual_input.position_mode",
                    ),
                ),
                velocity_mode=ManualInputVelocityModeConfig(
                    linear_velocity_limit_mps=_require_positive_float(
                        manual_velocity_cfg,
                        "linear_velocity_limit_mps",
                        "config.controller.manual_input.velocity_mode",
                    ),
                    angular_velocity_limit_degps=_require_positive_float(
                        manual_velocity_cfg,
                        "angular_velocity_limit_degps",
                        "config.controller.manual_input.velocity_mode",
                    ),
                    linear_accel_limit_mps2=_require_positive_float(
                        manual_velocity_cfg,
                        "linear_accel_limit_mps2",
                        "config.controller.manual_input.velocity_mode",
                    ),
                    angular_accel_limit_degps2=_require_positive_float(
                        manual_velocity_cfg,
                        "angular_accel_limit_degps2",
                        "config.controller.manual_input.velocity_mode",
                    ),
                ),
                acceleration_mode=ManualInputAccelerationModeConfig(
                    linear_accel_limit_mps2=_require_positive_float(
                        manual_accel_cfg,
                        "linear_accel_limit_mps2",
                        "config.controller.manual_input.acceleration_mode",
                    ),
                    angular_accel_limit_degps2=_require_positive_float(
                        manual_accel_cfg,
                        "angular_accel_limit_degps2",
                        "config.controller.manual_input.acceleration_mode",
                    ),
                    linear_velocity_limit_mps=_require_positive_float(
                        manual_accel_cfg,
                        "linear_velocity_limit_mps",
                        "config.controller.manual_input.acceleration_mode",
                    ),
                    angular_velocity_limit_degps=_require_positive_float(
                        manual_accel_cfg,
                        "angular_velocity_limit_degps",
                        "config.controller.manual_input.acceleration_mode",
                    ),
                ),
            ),
        ),
        homing=HomingConfig(
            auto_apply=_require_bool(homing_cfg, "auto_apply", "config.homing"),
            require_tension_feedback=_require_bool(
                homing_cfg,
                "require_tension_feedback",
                "config.homing",
            ),
            require_taskspace_controller=_require_bool(
                homing_cfg,
                "require_taskspace_controller",
                "config.homing",
            ),
            settle=HomingSettleConfig(
                min_dwell_s=homing_min_dwell_s,
                timeout_s=homing_timeout_s,
                pose_tol_mm=_require_nonnegative_float(
                    homing_settle_cfg,
                    "pose_tol_mm",
                    "config.homing.settle",
                ),
                vel_tol_mmps=_require_nonnegative_float(
                    homing_settle_cfg,
                    "vel_tol_mmps",
                    "config.homing.settle",
                ),
                tension_tol_n=_require_nonnegative_float(
                    homing_settle_cfg,
                    "tension_tol_n",
                    "config.homing.settle",
                ),
                sigma_err_tol_n=_require_nonnegative_float(
                    homing_settle_cfg,
                    "sigma_err_tol_n",
                    "config.homing.settle",
                ),
                null_cmd_tol_mm=_require_nonnegative_float(
                    homing_settle_cfg,
                    "null_cmd_tol_mm",
                    "config.homing.settle",
                ),
            ),
            prep=HomingPrepConfig(
                enabled=_require_bool(homing_prep_cfg, "enabled", "config.homing.prep"),
                speed_mps=_require_positive_float(
                    homing_prep_cfg,
                    "speed_mps",
                    "config.homing.prep",
                ),
                dwell_s=_require_nonnegative_float(
                    homing_prep_cfg,
                    "dwell_s",
                    "config.homing.prep",
                ),
                cycles=homing_prep_cycles,
                z_levels_mm=homing_prep_z_levels_mm,
                radius_mm=_require_nonnegative_float(
                    homing_prep_cfg,
                    "radius_mm",
                    "config.homing.prep",
                ),
            ),
            z_axis_zero=ZAxisZeroConfig(
                points_mm=homing_z_points_mm,
                x_mm=_require_float(homing_z_axis_cfg, "x_mm", "config.homing.z_axis_zero"),
                y_mm=_require_float(homing_z_axis_cfg, "y_mm", "config.homing.z_axis_zero"),
            ),
            xy_axis_zero=XYAxisZeroConfig(
                radius_mm=_require_nonnegative_float(
                    homing_xy_axis_cfg,
                    "radius_mm",
                    "config.homing.xy_axis_zero",
                ),
                point_count=_require_positive_int(
                    homing_xy_axis_cfg,
                    "point_count",
                    "config.homing.xy_axis_zero",
                ),
                start_angle_deg=_require_float(
                    homing_xy_axis_cfg,
                    "start_angle_deg",
                    "config.homing.xy_axis_zero",
                ),
                z_levels_mm=homing_xy_z_levels_mm,
            ),
        ),
        allocation=AllocationConfig(
            backend=_require_str(allocation_cfg, "backend", "config.allocation"),
            max_iters=_require_positive_int(allocation_cfg, "max_iters", "config.allocation"),
            wrench_from_tension_sign=wrench_sign,
        ),
        estimator=EstimatorConfig(
            rate_hz=_require_positive_float(estimator_cfg, "rate_hz", "config.estimator"),
            enable_velocity_filter=_require_bool(
                estimator_cfg,
                "enable_velocity_filter",
                "config.estimator",
            ),
            vel_filter_cutoff_hz=_require_positive_float(
                estimator_cfg,
                "vel_filter_cutoff_hz",
                "config.estimator",
            ),
        ),
        hardware=HardwareConfig(
            can=HardwareCanConfig(
                interface=_require_str(hardware_can_cfg, "interface", "config.hardware.can"),
                bitrate=_require_positive_int(hardware_can_cfg, "bitrate", "config.hardware.can"),
            ),
            odrive=ODriveConfig(
                axis_ids=axis_ids,
                mm_per_turn=_require_float_sequence(
                    hardware_odrive_cfg,
                    "mm_per_turn",
                    "config.hardware.odrive",
                    axis_count,
                ),
                torque_direction=_require_float(
                    hardware_odrive_cfg,
                    "torque_direction",
                    "config.hardware.odrive",
                ),
                input_vel_scale=_require_positive_float(
                    hardware_odrive_cfg,
                    "input_vel_scale",
                    "config.hardware.odrive",
                ),
                input_torque_scale=_require_positive_float(
                    hardware_odrive_cfg,
                    "input_torque_scale",
                    "config.hardware.odrive",
                ),
            ),
        ),
        logging=LoggingConfig(
            level=_require_str(logging_cfg, "level", "config.logging"),
            log_directory=_require_str(logging_cfg, "log_directory", "config.logging"),
            telemetry_udp=TelemetryUdpConfig(
                enabled=_require_bool(telemetry_udp_cfg, "enabled", "config.logging.telemetry_udp"),
                host=_require_str(telemetry_udp_cfg, "host", "config.logging.telemetry_udp"),
                port=_require_positive_int(telemetry_udp_cfg, "port", "config.logging.telemetry_udp"),
            ),
        ),
        rt=RuntimeLoopConfig(
            status_report_period_s=_require_positive_float(rt_cfg, "status_report_period_s", "config.rt"),
            deadline_warning_margin_ms=_require_nonnegative_float(
                rt_cfg,
                "deadline_warning_margin_ms",
                "config.rt",
            ),
            deadline_warning_missed_per_report=_require_positive_int(
                rt_cfg,
                "deadline_warning_missed_per_report",
                "config.rt",
            ),
            feedback_age_warning_ms=_require_nonnegative_float(
                rt_cfg,
                "feedback_age_warning_ms",
                "config.rt",
            ),
            transition_grace_s=_require_nonnegative_float(
                rt_cfg,
                "transition_grace_s",
                "config.rt",
            ),
        ),
    )


def load_runtime_config(config_name: str = "robot.yaml") -> RuntimeConfig:
    return parse_runtime_config(load_raw_config(config_name))
