"""Plot channel registry for the controller GUI."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterable
from collections.abc import Callable
from dataclasses import dataclass

from .models import TelemetryFrame


AXIS_COLORS: tuple[tuple[int, int, int], ...] = (
    (255, 80, 80),
    (80, 200, 120),
    (80, 160, 255),
    (255, 200, 80),
    (190, 120, 255),
    (80, 220, 220),
)

STYLE_SOLID = "solid"
STYLE_DASH = "dash"


@dataclass(frozen=True, slots=True)
class ChannelDefinition:
    key: str
    label: str
    plot_id: str
    group: str
    unit: str
    color: tuple[int, int, int]
    style: str
    extractor: Callable[[TelemetryFrame], float]
    default_visible: bool = True


ChannelRegistry = OrderedDict[str, ChannelDefinition]


@dataclass(frozen=True, slots=True)
class PlotDefinition:
    plot_id: str
    title: str
    y_label: str
    unit: str
    domain: str = "general"
    pose_row: str | None = None
    pose_column: str | None = None
    sort_order: int = 0


PlotRegistry = OrderedDict[str, PlotDefinition]


@dataclass(frozen=True, slots=True)
class WorkspacePreset:
    key: str
    label: str
    plot_ids: tuple[str, ...]


PresetRegistry = OrderedDict[str, WorkspacePreset]


def build_default_channel_registry() -> ChannelRegistry:
    channels: ChannelRegistry = OrderedDict()

    for axis in range(6):
        color = AXIS_COLORS[axis]
        axis_label = f"A{axis + 1}"
        channels[f"axis.pos_mm.{axis}"] = ChannelDefinition(
            key=f"axis.pos_mm.{axis}",
            label=axis_label,
            plot_id="position",
            group="spools",
            unit="mm",
            color=color,
            style=STYLE_SOLID,
            extractor=lambda frame, index=axis: frame.pos_mm[index],
        )
        channels[f"axis.vel_mmps.{axis}"] = ChannelDefinition(
            key=f"axis.vel_mmps.{axis}",
            label=axis_label,
            plot_id="velocity",
            group="spools",
            unit="mm/s",
            color=color,
            style=STYLE_SOLID,
            extractor=lambda frame, index=axis: frame.vel_mmps[index],
        )
        channels[f"axis.bus_voltage_v.{axis}"] = ChannelDefinition(
            key=f"axis.bus_voltage_v.{axis}",
            label=axis_label,
            plot_id="bus_voltage",
            group="thermals",
            unit="V",
            color=color,
            style=STYLE_SOLID,
            extractor=lambda frame, index=axis: frame.bus_voltage_v[index],
        )
        channels[f"axis.tension_cmd_n.{axis}"] = ChannelDefinition(
            key=f"axis.tension_cmd_n.{axis}",
            label=f"{axis_label} Cmd",
            plot_id="tension",
            group="tension",
            unit="N",
            color=color,
            style=STYLE_SOLID,
            extractor=lambda frame, index=axis: frame.tension_cmd_n[index],
        )
        channels[f"axis.tension_rsp_n.{axis}"] = ChannelDefinition(
            key=f"axis.tension_rsp_n.{axis}",
            label=f"{axis_label} Rsp",
            plot_id="tension",
            group="tension",
            unit="N",
            color=color,
            style=STYLE_DASH,
            extractor=lambda frame, index=axis: frame.tension_rsp_n[index],
        )
        channels[f"axis.torque_cmd_nm.{axis}"] = ChannelDefinition(
            key=f"axis.torque_cmd_nm.{axis}",
            label=f"{axis_label} Cmd",
            plot_id="torque",
            group="torque",
            unit="Nm",
            color=color,
            style=STYLE_SOLID,
            extractor=lambda frame, index=axis: frame.torque_cmd_nm[index],
        )
        channels[f"axis.torque_rsp_nm.{axis}"] = ChannelDefinition(
            key=f"axis.torque_rsp_nm.{axis}",
            label=f"{axis_label} Rsp",
            plot_id="torque",
            group="torque",
            unit="Nm",
            color=color,
            style=STYLE_DASH,
            extractor=lambda frame, index=axis: frame.torque_rsp_nm[index],
        )
        channels[f"axis.temp_motor_c.{axis}"] = ChannelDefinition(
            key=f"axis.temp_motor_c.{axis}",
            label=f"{axis_label} Motor",
            plot_id="temperature",
            group="thermals",
            unit="C",
            color=color,
            style=STYLE_SOLID,
            extractor=lambda frame, index=axis: frame.temp_motor_c[index],
        )
        channels[f"axis.temp_fet_c.{axis}"] = ChannelDefinition(
            key=f"axis.temp_fet_c.{axis}",
            label=f"{axis_label} FET",
            plot_id="temperature",
            group="thermals",
            unit="C",
            color=color,
            style=STYLE_DASH,
            extractor=lambda frame, index=axis: frame.temp_fet_c[index],
        )

    pose_translation = (
        ("x", 0, (255, 80, 80)),
        ("y", 1, (80, 200, 120)),
        ("z", 2, (80, 160, 255)),
    )
    for name, index, color in pose_translation:
        axis_name = name.upper()
        channels[f"pose.cmd_mm.{name}"] = ChannelDefinition(
            key=f"pose.cmd_mm.{name}",
            label=f"{axis_name} cmd [mm]",
            plot_id="pose_translation",
            group="pose",
            unit="mm",
            color=color,
            style=STYLE_SOLID,
            extractor=lambda frame, idx=index: frame.hand_cmd_pose[idx],
        )
        channels[f"pose.rsp_mm.{name}"] = ChannelDefinition(
            key=f"pose.rsp_mm.{name}",
            label=f"{axis_name} rsp [mm]",
            plot_id="pose_translation",
            group="pose",
            unit="mm",
            color=color,
            style=STYLE_DASH,
            extractor=lambda frame, idx=index: frame.hand_est_pose[idx],
        )

    pose_rotation = (
        ("roll", 3, (255, 200, 80)),
        ("pitch", 4, (190, 120, 255)),
    )
    for name, index, color in pose_rotation:
        axis_name = name.capitalize()
        channels[f"pose.cmd_deg.{name}"] = ChannelDefinition(
            key=f"pose.cmd_deg.{name}",
            label=f"{axis_name} cmd [deg]",
            plot_id="pose_rotation",
            group="pose",
            unit="deg",
            color=color,
            style=STYLE_SOLID,
            extractor=lambda frame, idx=index: frame.hand_cmd_pose[idx],
        )
        channels[f"pose.rsp_deg.{name}"] = ChannelDefinition(
            key=f"pose.rsp_deg.{name}",
            label=f"{axis_name} rsp [deg]",
            plot_id="pose_rotation",
            group="pose",
            unit="deg",
            color=color,
            style=STYLE_DASH,
            extractor=lambda frame, idx=index: frame.hand_est_pose[idx],
        )

    pose_velocity_translation = (
        ("x", 0, (255, 80, 80)),
        ("y", 1, (80, 200, 120)),
        ("z", 2, (80, 160, 255)),
    )
    for name, index, color in pose_velocity_translation:
        axis_name = name.upper()
        channels[f"pose.cmd_vel_mmps.{name}"] = ChannelDefinition(
            key=f"pose.cmd_vel_mmps.{name}",
            label=f"{axis_name} cmd vel",
            plot_id="pose_velocity_translation",
            group="pose_velocity",
            unit="mm/s",
            color=color,
            style=STYLE_SOLID,
            extractor=lambda frame, idx=index: frame.hand_cmd_vel[idx],
            default_visible=False,
        )
        channels[f"pose.rsp_vel_mmps.{name}"] = ChannelDefinition(
            key=f"pose.rsp_vel_mmps.{name}",
            label=f"{axis_name} rsp vel",
            plot_id="pose_velocity_translation",
            group="pose_velocity",
            unit="mm/s",
            color=color,
            style=STYLE_DASH,
            extractor=lambda frame, idx=index: frame.hand_est_vel[idx],
            default_visible=False,
        )

    pose_velocity_rotation = (
        ("roll", 3, (255, 200, 80)),
        ("pitch", 4, (190, 120, 255)),
    )
    for name, index, color in pose_velocity_rotation:
        axis_name = name.capitalize()
        channels[f"pose.cmd_vel_degps.{name}"] = ChannelDefinition(
            key=f"pose.cmd_vel_degps.{name}",
            label=f"{axis_name} cmd vel",
            plot_id="pose_velocity_rotation",
            group="pose_velocity",
            unit="deg/s",
            color=color,
            style=STYLE_SOLID,
            extractor=lambda frame, idx=index: frame.hand_cmd_vel[idx],
            default_visible=False,
        )
        channels[f"pose.rsp_vel_degps.{name}"] = ChannelDefinition(
            key=f"pose.rsp_vel_degps.{name}",
            label=f"{axis_name} rsp vel",
            plot_id="pose_velocity_rotation",
            group="pose_velocity",
            unit="deg/s",
            color=color,
            style=STYLE_DASH,
            extractor=lambda frame, idx=index: frame.hand_est_vel[idx],
            default_visible=False,
        )

    pose_accel_translation = (
        ("x", 0, (255, 80, 80)),
        ("y", 1, (80, 200, 120)),
        ("z", 2, (80, 160, 255)),
    )
    for name, index, color in pose_accel_translation:
        axis_name = name.upper()
        channels[f"pose.cmd_acc_mmps2.{name}"] = ChannelDefinition(
            key=f"pose.cmd_acc_mmps2.{name}",
            label=f"{axis_name} cmd acc",
            plot_id="pose_acceleration_translation",
            group="pose_acceleration",
            unit="mm/s^2",
            color=color,
            style=STYLE_SOLID,
            extractor=lambda frame, idx=index: frame.hand_cmd_acc[idx],
            default_visible=False,
        )
        channels[f"pose.rsp_acc_mmps2.{name}"] = ChannelDefinition(
            key=f"pose.rsp_acc_mmps2.{name}",
            label=f"{axis_name} rsp acc",
            plot_id="pose_acceleration_translation",
            group="pose_acceleration",
            unit="mm/s^2",
            color=color,
            style=STYLE_DASH,
            extractor=lambda frame, idx=index: frame.hand_est_acc[idx],
            default_visible=False,
        )

    pose_accel_rotation = (
        ("roll", 3, (255, 200, 80)),
        ("pitch", 4, (190, 120, 255)),
    )
    for name, index, color in pose_accel_rotation:
        axis_name = name.capitalize()
        channels[f"pose.cmd_acc_degps2.{name}"] = ChannelDefinition(
            key=f"pose.cmd_acc_degps2.{name}",
            label=f"{axis_name} cmd acc",
            plot_id="pose_acceleration_rotation",
            group="pose_acceleration",
            unit="deg/s^2",
            color=color,
            style=STYLE_SOLID,
            extractor=lambda frame, idx=index: frame.hand_cmd_acc[idx],
            default_visible=False,
        )
        channels[f"pose.rsp_acc_degps2.{name}"] = ChannelDefinition(
            key=f"pose.rsp_acc_degps2.{name}",
            label=f"{axis_name} rsp acc",
            plot_id="pose_acceleration_rotation",
            group="pose_acceleration",
            unit="deg/s^2",
            color=color,
            style=STYLE_DASH,
            extractor=lambda frame, idx=index: frame.hand_est_acc[idx],
            default_visible=False,
        )

    return channels


def build_default_plot_registry() -> PlotRegistry:
    plots: PlotRegistry = OrderedDict()
    plots["pose_translation"] = PlotDefinition(
        "pose_translation",
        "Linear Position",
        "Position",
        "mm",
        domain="pose",
        pose_row="position",
        pose_column="linear",
        sort_order=0,
    )
    plots["pose_rotation"] = PlotDefinition(
        "pose_rotation",
        "Angular Position",
        "Angle",
        "deg",
        domain="pose",
        pose_row="position",
        pose_column="angular",
        sort_order=1,
    )
    plots["pose_velocity_translation"] = PlotDefinition(
        "pose_velocity_translation",
        "Linear Velocity",
        "Velocity",
        "mm/s",
        domain="pose",
        pose_row="velocity",
        pose_column="linear",
        sort_order=2,
    )
    plots["pose_velocity_rotation"] = PlotDefinition(
        "pose_velocity_rotation",
        "Angular Velocity",
        "Angular Velocity",
        "deg/s",
        domain="pose",
        pose_row="velocity",
        pose_column="angular",
        sort_order=3,
    )
    plots["pose_acceleration_translation"] = PlotDefinition(
        "pose_acceleration_translation",
        "Linear Acceleration",
        "Acceleration",
        "mm/s^2",
        domain="pose",
        pose_row="acceleration",
        pose_column="linear",
        sort_order=4,
    )
    plots["pose_acceleration_rotation"] = PlotDefinition(
        "pose_acceleration_rotation",
        "Angular Acceleration",
        "Angular Acceleration",
        "deg/s^2",
        domain="pose",
        pose_row="acceleration",
        pose_column="angular",
        sort_order=5,
    )
    plots["position"] = PlotDefinition("position", "Cable Position", "Position", "mm", sort_order=10)
    plots["velocity"] = PlotDefinition("velocity", "Cable Velocity", "Velocity", "mm/s", sort_order=11)
    plots["tension"] = PlotDefinition("tension", "Cable Tension", "Tension", "N", sort_order=12)
    plots["torque"] = PlotDefinition("torque", "Torque", "Torque", "Nm", sort_order=13)
    plots["temperature"] = PlotDefinition("temperature", "Temperatures", "Temperature", "C", sort_order=14)
    plots["bus_voltage"] = PlotDefinition("bus_voltage", "Bus Voltage", "Voltage", "V", sort_order=15)
    return plots


def build_default_workspace_presets() -> PresetRegistry:
    presets: PresetRegistry = OrderedDict()
    presets["pose"] = WorkspacePreset(
        "pose",
        "Pose",
        (
            "pose_translation",
            "pose_rotation",
            "pose_velocity_translation",
            "pose_velocity_rotation",
            "pose_acceleration_translation",
            "pose_acceleration_rotation",
        ),
    )
    presets["spools"] = WorkspacePreset("spools", "Spools", ("position", "velocity", "tension", "torque"))
    presets["thermals"] = WorkspacePreset("thermals", "System", ("temperature", "bus_voltage"))
    return presets


def channel_keys_for_plot(channels: ChannelRegistry, plot_id: str) -> tuple[str, ...]:
    return tuple(key for key, channel in channels.items() if channel.plot_id == plot_id)


def channel_keys_for_plots(channels: ChannelRegistry, plot_ids: Iterable[str]) -> tuple[str, ...]:
    allowed = set(plot_ids)
    return tuple(key for key, channel in channels.items() if channel.plot_id in allowed)
