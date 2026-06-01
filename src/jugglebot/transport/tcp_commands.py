"""TCP command server for robot and simulation daemons."""

from __future__ import annotations

import json
import logging
import socket

from jugglebot.core.pose_utils import quat_from_rpy_deg
from jugglebot.core.types import TrajectoryUpdateMode
from jugglebot.core.units import coerce_vec6_to_mm
from jugglebot.transport.config import TCP_CMD_PORT, UDP_TELEM_PORT


logger = logging.getLogger("robot")


def _parse_update_mode(raw_value):
    if raw_value is None:
        return TrajectoryUpdateMode.REPLACE
    return TrajectoryUpdateMode(str(raw_value).lower())


def _parse_preserve_continuity(msg):
    raw = msg.get("preserve_continuity", True)
    return bool(raw)


def tcp_command_server(state):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        srv.bind(("0.0.0.0", TCP_CMD_PORT))
    except OSError as e:
        logger.error(f"[TCP] Bind failed: {e}")
        return
    srv.listen(1)
    logger.info(f"[TCP] Listening on :{TCP_CMD_PORT}")
    while True:
        conn, addr = srv.accept()
        state.set_controller_ip(addr[0])
        logger.info(f"[TCP] Controller connected from {addr}")
        state.set_controller_ip(addr[0])

        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        controller_addr = (addr[0], UDP_TELEM_PORT)
        state.start_telem(udp_sock, controller_addr)

        try:
            with conn, conn.makefile("r") as f:
                for line in f:
                    try:
                        msg = json.loads(line.strip())
                        mtype = msg.get("type")
                        if mtype == "state":
                            state.set_state(msg.get("value", "disable"))
                        elif mtype == "pretension":
                            upper = float(msg.get("upper_N", 0.0))
                            lower = float(msg.get("lower_N", 0.0))
                            state.request_pretension(upper, lower)
                        elif mtype == "task_gain_mult":
                            state.request_task_gain_multipliers(
                                kp_xyz=msg.get("kp_xyz"),
                                kp_rp=msg.get("kp_rp"),
                                kd_xyz=msg.get("kd_xyz"),
                                kd_rp=msg.get("kd_rp"),
                            )
                        elif mtype == "spool_gain_mult":
                            state.request_spool_gain_multipliers(
                                kp=msg.get("kp"),
                                kd=msg.get("kd"),
                            )
                        elif mtype == "home":
                            home_mm = coerce_vec6_to_mm(msg, "home_pos")
                            state.request_home(home_mm)
                        elif mtype == "pose":
                            x = float(msg.get("x_mm", 0.0))
                            y = float(msg.get("y_mm", 0.0))
                            z = float(msg.get("z_mm", 0.0))
                            roll = float(msg.get("roll_deg", 0.0))
                            pitch = float(msg.get("pitch_deg", 0.0))
                            update_mode = _parse_update_mode(msg.get("update_mode"))
                            effective_time_s = msg.get("effective_time_s")
                            preserve_continuity = _parse_preserve_continuity(msg)

                            q = quat_from_rpy_deg(roll, pitch, 0.0)
                            state.submit_pose_command(
                                (x, y, z),
                                q,
                                mode=update_mode,
                                effective_time_s=None if effective_time_s is None else float(effective_time_s),
                                preserve_continuity=preserve_continuity,
                            )
                        elif mtype == "pose_profile_upload":
                            profile = msg.get("profile", [])
                            state.set_pose_profile(profile)
                        elif mtype == "pose_profile_start":
                            state.start_pose_profile(
                                mode=_parse_update_mode(msg.get("update_mode")),
                                effective_time_s=None if msg.get("effective_time_s") is None else float(msg.get("effective_time_s")),
                                preserve_continuity=_parse_preserve_continuity(msg),
                            )
                        elif mtype == "pose_profile_run":
                            profile = msg.get("profile", [])
                            state.set_pose_profile(profile)
                            state.start_pose_profile(
                                mode=_parse_update_mode(msg.get("update_mode")),
                                effective_time_s=None if msg.get("effective_time_s") is None else float(msg.get("effective_time_s")),
                                preserve_continuity=_parse_preserve_continuity(msg),
                            )
                        elif mtype == "profile_stop":
                            state.stop_profile()
                        else:
                            logger.warning(f"[TCP] Unknown command: {mtype}")
                    except Exception as e:
                        logger.error(f"[TCP] Bad command: {e}")
        except Exception as e:
            logger.error(f"[TCP] Connection error: {e}")
        finally:
            state.stop_profile()
            state.stop_telem()
            logger.info("[TCP] Controller disconnected")
