from __future__ import annotations

from jugglebot.controlui.models import normalize_telemetry


def test_normalize_snapshot_telemetry_maps_structured_payload() -> None:
    payload = {
        "timestamp_s": 12.5,
        "sequence_id": 42,
        "control_time_s": 4.0,
        "control_state": "enable",
        "profile_active": True,
        "cable_lengths_m": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "cable_velocities_mps": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        "actuators": [
            {
                "temperature_fet_c": 30.0 + index,
                "temperature_motor_c": 40.0 + index,
                "bus_current_a": 1.0 + index,
                "current_estimate_a": 2.0 + index,
                "bus_voltage_v": 24.0,
                "axis_state": 8,
                "error_flags": 0,
            }
            for index in range(6)
        ],
        "commanded_torques_nm": [1, 2, 3, 4, 5, 6],
        "estimated_torques_nm": [6, 5, 4, 3, 2, 1],
        "commanded_tensions_n": [11, 12, 13, 14, 15, 16],
        "estimated_tensions_n": [16, 15, 14, 13, 12, 11],
        "commanded_pose": {
            "position_m": [0.1, 0.2, 0.3],
            "orientation_rpy_rad": [0.0, 0.1, 0.2],
            "linear_velocity_mps": [0.1, 0.2, 0.3],
            "angular_velocity_rps": [0.0, 0.1, 0.2],
            "linear_acceleration_mps2": [1.0, 2.0, 3.0],
            "angular_acceleration_rps2": [0.0, 0.2, 0.4],
        },
        "estimated_pose": {
            "position_m": [0.2, 0.3, 0.4],
            "orientation_rpy_rad": [0.0, 0.2, 0.4],
            "linear_velocity_mps": [1.0, 2.0, 3.0],
            "angular_velocity_rps": [0.0, 0.3, 0.6],
            "linear_acceleration_mps2": [4.0, 5.0, 6.0],
            "angular_acceleration_rps2": [0.0, 0.4, 0.8],
        },
        "bus_stats": {
            "can_rx_hz": 100.0,
            "can_tx_hz": 50.0,
            "can_msg_hz": 150.0,
            "can_util_est": 0.25,
            "pos_fbk_hz": 10.0,
            "pos_fbk_period0_min_s": 0.001,
            "pos_fbk_period0_max_s": 0.002,
        },
        "debug": {"sim_time_s": 4.0},
    }

    frame = normalize_telemetry(payload, source_id="sim")

    assert frame.source_id == "sim"
    assert frame.sequence_id == 42
    assert frame.control_state == "enable"
    assert frame.profile_active is True
    assert frame.pos_mm[0] == 100.0
    assert frame.vel_mmps[5] == 60.0
    assert frame.tension_cmd_n[2] == 13.0
    assert frame.hand_cmd_pose[0] == 100.0
    assert frame.hand_est_pose[2] == 400.0
    assert frame.hand_cmd_vel[1] == 200.0
    assert frame.hand_cmd_acc[2] == 3000.0
    assert frame.hand_est_vel[0] == 1000.0
    assert frame.hand_est_acc[1] == 5000.0
    assert frame.axis_state[0] == 8
    assert frame.comm_stats.can_msg_hz == 150.0
    assert frame.preferred_time_s() == 4.0


def test_normalize_legacy_payload_preserves_flat_fields() -> None:
    payload = {
        "t": 5.0,
        "pos": [1, 2, 3, 4, 5, 6],
        "vel": [6, 5, 4, 3, 2, 1],
        "hand_cmd_vel": [1, 2, 3, 4, 5, 6],
        "hand_cmd_acc": [6, 5, 4, 3, 2, 1],
        "hand_est_pose": [10, 20, 30, 1, 2, 3],
        "hand_est_vel": [7, 8, 9, 10, 11, 12],
        "hand_est_acc": [12, 11, 10, 9, 8, 7],
        "axis_state": [8, 8, 8, 8, 8, 8],
        "axis_error": [0, 0, 0, 0, 0, 0],
    }

    frame = normalize_telemetry(payload)

    assert frame.wall_time_s == 5.0
    assert frame.pos_mm[1] == 2.0
    assert frame.hand_est_pose[5] == 3.0
    assert frame.hand_cmd_vel[3] == 4.0
    assert frame.hand_est_acc[0] == 12.0
    assert frame.axis_state[4] == 8
