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
class ControllerConfig:
    spool_space: SpoolSpaceControllerConfig


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
    telemetry_udp_cfg = _require_mapping(logging_cfg, "telemetry_udp", "config.logging")

    tension_min_n = _require_nonnegative_float(tension_cfg, "Tmin_N", "config.tension")
    tension_max_n = _require_positive_float(tension_cfg, "Tmax_N", "config.tension")
    if tension_max_n < tension_min_n:
        raise ConfigError("config.tension.Tmax_N must be >= config.tension.Tmin_N")

    fallback_min_n = _require_nonnegative_float(fallback_cfg, "min_n", "config.controller.spool_space.fallback_tension")
    fallback_max_n = _require_positive_float(fallback_cfg, "max_n", "config.controller.spool_space.fallback_tension")
    if fallback_max_n < fallback_min_n:
        raise ConfigError(
            "config.controller.spool_space.fallback_tension.max_n must be >= min_n"
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
            )
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


def load_runtime_config(config_name: str = "default.yaml") -> RuntimeConfig:
    return parse_runtime_config(load_raw_config(config_name))
