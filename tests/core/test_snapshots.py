from __future__ import annotations

from jugglebot.core.snapshots import build_robot_state_snapshot
from jugglebot.core.state import RuntimeMailbox


def test_snapshot_uses_atomic_timing_state() -> None:
    state = RuntimeMailbox()
    state.set_timing_state(
        control_time_s=12.5,
        runtime_time_s=21.0,
        sim_time_s=12.5,
        sim_rt_factor=0.8,
    )

    snapshot = build_robot_state_snapshot(state, timestamp_s=100.0, sequence_id=7)

    assert snapshot.control_time_s == 12.5
    assert snapshot.runtime_time_s == 21.0
    assert snapshot.sim_time_s == 12.5
    assert snapshot.sim_rt_factor == 0.8
