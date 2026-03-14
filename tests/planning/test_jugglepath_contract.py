import numpy as np

from jugglebot.planning.jugglepath import State3D, JugglePath


def test_monotonic_segment_endpoint_constraints():
    path = JugglePath(sample_hz=200.0, start=State3D([0, 0, 0], [0, 0, 0], [0, 0, 0]))
    path.add_segment(
        p=[0.0, 0.0, 1.0],
        v=[0.0, 0.0, 1.0],
        time_law="s_curve_monotonic",
        accel_ref=1.0,
        jerk_ref=10.0,
    )

    res = path.build()
    traj = res.traj

    np.testing.assert_allclose(traj[0, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(traj[0, 1:4], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(traj[-1, 1:4], [0.0, 0.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(traj[-1, 4:7], [0.0, 0.0, 1.0], atol=1e-6)


def test_wait_requires_near_zero_incoming_state():
    path = JugglePath(sample_hz=100.0, start=State3D([0, 0, 0], [1, 0, 0], [0, 0, 0]))
    path.add_wait(0.5)

    try:
        path.build()
        assert False, "expected wait continuity check to raise"
    except ValueError as exc:
        assert "WAIT segment requires near-zero incoming velocity" in str(exc)


def test_segment_chain_velocity_continuity():
    path = JugglePath(sample_hz=200.0, start=State3D([0, 0, 0], [0, 0, 0], [0, 0, 0]))
    path.add_segment(
        p=[0.0, 0.0, 0.4],
        v=[0.0, 0.0, 0.5],
        time_law="s_curve_monotonic",
        accel_ref=2.0,
        jerk_ref=20.0,
    )
    path.add_segment(
        p=[0.0, 0.0, 0.8],
        v=[0.0, 0.0, 0.0],
        time_law="s_curve_monotonic",
        accel_ref=2.0,
        jerk_ref=20.0,
    )

    res = path.build()
    traj = res.traj

    # All timestamps must be strictly increasing after boundary de-dup.
    assert np.all(np.diff(traj[:, 0]) > 0.0)
