from __future__ import annotations

from jugglebot.controlui.channels import (
    build_default_channel_registry,
    build_default_plot_registry,
    build_default_workspace_presets,
    channel_keys_for_plot,
    channel_keys_for_plots,
)


def test_default_plot_registry_contains_expected_panels() -> None:
    plots = build_default_plot_registry()

    assert tuple(plots) == (
        "pose_translation",
        "pose_rotation",
        "position",
        "velocity",
        "tension",
        "torque",
        "temperature",
        "bus_voltage",
    )


def test_channel_registry_exposes_pose_and_spool_grouping() -> None:
    channels = build_default_channel_registry()

    assert channels["pose.cmd_mm.x"].group == "pose"
    assert channels["axis.pos_mm.0"].group == "spools"
    assert channels["axis.temp_fet_c.0"].group == "thermals"


def test_channel_lookup_helpers_return_expected_keys() -> None:
    channels = build_default_channel_registry()

    pose_keys = channel_keys_for_plot(channels, "pose_translation")
    spool_keys = channel_keys_for_plots(channels, ("position", "velocity"))

    assert len(pose_keys) == 6
    assert "pose.cmd_mm.x" in pose_keys
    assert len(spool_keys) == 12
    assert "axis.vel_mmps.5" in spool_keys


def test_workspace_presets_map_to_expected_plot_sets() -> None:
    presets = build_default_workspace_presets()

    assert presets["pose"].plot_ids == ("pose_translation", "pose_rotation")
    assert presets["spools"].plot_ids == ("position", "velocity")
    assert "temperature" in presets["thermals"].plot_ids
