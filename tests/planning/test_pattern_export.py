import pytest

from jugglebot.planning import build_traj_from_pattern, load_pattern_yaml


@pytest.fixture
def one_ball_project():
    return load_pattern_yaml("src/jugglebot/patterns/examples/one_ball_one_hand.yaml")


def test_build_traj_from_pattern_samples_checked_in_loop(one_ball_project):
    traj, sample_hz = build_traj_from_pattern(
        one_ball_project,
        hand="right",
        command_rate_hz=100.0,
        cycles=1,
    )

    assert sample_hz == pytest.approx(100.0)
    assert traj.shape[1] == 13
    assert traj[0, 0] == pytest.approx(0.0)
    assert traj[-1, 0] == pytest.approx(1.5)
    assert traj[0, 1:4] == pytest.approx([0.1, 0.0, 0.0])
    assert traj[-1, 1:4] == pytest.approx([0.1, 0.0, 0.0])
    assert traj[0, 4:7] == pytest.approx([-0.2, 0.0, 4.905])


def test_build_traj_from_pattern_rejects_inactive_hand(one_ball_project):
    with pytest.raises(ValueError, match="no trajectory data"):
        build_traj_from_pattern(one_ball_project, hand="left", command_rate_hz=100.0)
