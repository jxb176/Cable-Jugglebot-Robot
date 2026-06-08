from __future__ import annotations

from jugglebot.controlui.channels import build_default_channel_registry
from jugglebot.controlui.history import TelemetryHistory
from jugglebot.controlui.models import CommStats, TelemetryFrame


def _frame(control_time_s: float, value: float) -> TelemetryFrame:
    zeros6 = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    ints6 = (0, 0, 0, 0, 0, 0)
    pose = (value, 0.0, 0.0, 0.0, 0.0, 0.0)
    return TelemetryFrame(
        source_id="live",
        receipt_time_s=control_time_s,
        wall_time_s=control_time_s,
        control_time_s=control_time_s,
        sim_time_s=None,
        sequence_id=None,
        control_state=None,
        profile_active=False,
        pos_mm=(value, 0.0, 0.0, 0.0, 0.0, 0.0),
        vel_mmps=zeros6,
        temp_fet_c=zeros6,
        temp_motor_c=zeros6,
        bus_current_a=zeros6,
        motor_current_a=zeros6,
        torque_cmd_nm=zeros6,
        torque_rsp_nm=zeros6,
        tension_cmd_n=zeros6,
        tension_rsp_n=zeros6,
        bus_voltage_v=zeros6,
        axis_state=ints6,
        axis_error=ints6,
        hand_cmd_pose=pose,
        hand_cmd_vel=zeros6,
        hand_cmd_acc=zeros6,
        hand_est_pose=pose,
        hand_est_vel=zeros6,
        hand_est_acc=zeros6,
        comm_stats=CommStats(),
        debug={},
        raw={},
    )


def test_history_trims_old_samples_by_time_window() -> None:
    history = TelemetryHistory(build_default_channel_registry(), history_seconds=2.0)

    history.append(_frame(0.0, 1.0))
    history.append(_frame(1.0, 2.0))
    history.append(_frame(3.1, 3.0))

    assert len(history) == 1
    assert history.times() == [3.1]
    assert history.values("axis.pos_mm.0") == [3.0]


def test_history_exposes_latest_frame() -> None:
    history = TelemetryHistory(build_default_channel_registry(), history_seconds=10.0)
    frame = _frame(1.5, 7.0)

    history.append(frame)

    assert history.latest_frame is frame


def test_history_snapshot_reuses_cached_exports_until_version_changes() -> None:
    history = TelemetryHistory(build_default_channel_registry(), history_seconds=10.0)
    history.append(_frame(1.0, 2.0))

    first = history.snapshot(("axis.pos_mm.0",))
    second = history.snapshot(("axis.pos_mm.0",))

    assert second is first
    assert first.times is history.times()
    assert first.series_by_key["axis.pos_mm.0"] is history.values("axis.pos_mm.0")

    history.append(_frame(2.0, 3.0))
    third = history.snapshot(("axis.pos_mm.0",))

    assert third.version > first.version
    assert third is not first
    assert third.times == [0.0, 1.0]
    assert third.series_by_key["axis.pos_mm.0"] == [2.0, 3.0]


def test_history_snapshot_remains_frozen_after_live_history_advances() -> None:
    history = TelemetryHistory(build_default_channel_registry(), history_seconds=10.0)
    history.append(_frame(1.0, 5.0))

    snapshot = history.snapshot(("axis.pos_mm.0",))
    history.append(_frame(2.0, 6.0))

    assert snapshot.times == [0.0]
    assert snapshot.series_by_key["axis.pos_mm.0"] == [5.0]
    assert history.times() == [0.0, 1.0]
    assert history.values("axis.pos_mm.0") == [5.0, 6.0]
