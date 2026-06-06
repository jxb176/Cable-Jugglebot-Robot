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

    return channels


def build_default_plot_registry() -> PlotRegistry:
    plots: PlotRegistry = OrderedDict()
    plots["pose_translation"] = PlotDefinition("pose_translation", "Hand Position", "Position", "mm")
    plots["pose_rotation"] = PlotDefinition("pose_rotation", "Hand Rotation", "Rotation", "deg")
    plots["position"] = PlotDefinition("position", "Cable Position", "Position", "mm")
    plots["velocity"] = PlotDefinition("velocity", "Cable Velocity", "Velocity", "mm/s")
    plots["tension"] = PlotDefinition("tension", "Cable Tension", "Tension", "N")
    plots["torque"] = PlotDefinition("torque", "Torque", "Torque", "Nm")
    plots["temperature"] = PlotDefinition("temperature", "Temperatures", "Temperature", "C")
    plots["bus_voltage"] = PlotDefinition("bus_voltage", "Bus Voltage", "Voltage", "V")
    return plots


def build_default_workspace_presets() -> PresetRegistry:
    presets: PresetRegistry = OrderedDict()
    presets["all"] = WorkspacePreset("all", "All", ("pose_translation", "pose_rotation", "position", "velocity", "tension", "torque", "temperature", "bus_voltage"))
    presets["pose"] = WorkspacePreset("pose", "Pose", ("pose_translation", "pose_rotation"))
    presets["spools"] = WorkspacePreset("spools", "Spools", ("position", "velocity"))
    presets["tension"] = WorkspacePreset("tension", "Tension", ("tension",))
    presets["torque"] = WorkspacePreset("torque", "Torque", ("torque",))
    presets["thermals"] = WorkspacePreset("thermals", "Thermals", ("temperature", "bus_voltage"))
    return presets


def channel_keys_for_plot(channels: ChannelRegistry, plot_id: str) -> tuple[str, ...]:
    return tuple(key for key, channel in channels.items() if channel.plot_id == plot_id)


def channel_keys_for_plots(channels: ChannelRegistry, plot_ids: Iterable[str]) -> tuple[str, ...]:
    allowed = set(plot_ids)
    return tuple(key for key, channel in channels.items() if channel.plot_id in allowed)
