from pathlib import Path

from jugglebot.planning import load_profile_yaml, build_path_from_profile


def test_yaml_profile_load_and_build():
    profile_path = Path("src/jugglebot/profiles/simple_throw.yaml")
    profile = load_profile_yaml(str(profile_path))
    path, hz = build_path_from_profile(profile)

    assert profile.get("name") == "simple_throw"
    assert hz > 0.0

    result = path.build()
    assert result.traj.shape[1] == 13
    assert result.traj.shape[0] > 10
