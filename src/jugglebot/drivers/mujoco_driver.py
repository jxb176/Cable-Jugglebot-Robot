"""
MuJoCo-based simulation driver for cable robot.
"""

import logging
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from .driver_interface import RobotDriver


logger = logging.getLogger(__name__)

DEFAULT_MM_PER_TURN = -62.832
DEFAULT_TORQUE_PER_TENSION_NM_PER_N = -0.01


class MuJoCoSimulationDriver(RobotDriver):
    """
    MuJoCo-based simulation driver with explicit cable-length feedback.

    MuJoCo simulates only the platform and contacts. Cable lengths are read from
    anchor/platform site geometry, and cable tensions are computed explicitly in
    the driver before applying a generalized wrench to the platform DOFs.
    """

    def __init__(
        self,
        axis_ids: List[int] = None,
        model_path: str = None,
        enable_viewer: bool = False,
        spool_kp: List[float] | None = None,
        spool_kd: List[float] | None = None,
        torque_limit_nm: List[float] | None = None,
        mm_per_turn: List[float] | None = None,
        torque_per_tension_nm_per_N: List[float] | None = None,
    ):
        self.axis_ids = axis_ids or [0, 1, 2, 3, 4, 5]
        self.model_path = model_path or str(
            Path(__file__).parent.parent / "simulation" / "cable_robot_5dof_winch.xml"
        )
        self.enable_viewer = enable_viewer
        self._mj_lock = threading.Lock()
        self.viewer = None

        self.model = None
        self.data = None
        self.running = False

        self._axis_torque_cmd = {aid: 0.0 for aid in self.axis_ids}
        self._axis_pos_cmd = {aid: 0.0 for aid in self.axis_ids}
        self._axis_vel_ff = {aid: 0.0 for aid in self.axis_ids}
        self._axis_torque_ff = {aid: 0.0 for aid in self.axis_ids}
        self._axis_mode = {aid: "torque" for aid in self.axis_ids}
        self._axis_state = {aid: "idle" for aid in self.axis_ids}
        self._axis_pos_offset = {aid: 0.0 for aid in self.axis_ids}
        self._axis_pos_fb = {aid: 0.0 for aid in self.axis_ids}
        self._axis_vel_fb = {aid: 0.0 for aid in self.axis_ids}
        self._axis_torque_applied = {aid: 0.0 for aid in self.axis_ids}
        self._cable_tension_fb = {aid: 0.0 for aid in self.axis_ids}

        self._spool_kp = self._expand_axis_param(spool_kp, 0.377)
        self._spool_kd = self._expand_axis_param(spool_kd, 0.012566)
        self._torque_limit_nm = self._expand_axis_param(torque_limit_nm, 1.0)
        self.mm_per_turn = self._expand_axis_list(mm_per_turn, DEFAULT_MM_PER_TURN)
        self._torque_per_tension = self._expand_axis_param(
            torque_per_tension_nm_per_N,
            DEFAULT_TORQUE_PER_TENSION_NM_PER_N,
        )

        self._position_callback: Optional[Callable[[int, float], None]] = None
        self._velocity_callback: Optional[Callable[[int, float], None]] = None
        self._bus_callback: Optional[Callable[[int, float, float], None]] = None
        self._current_callback: Optional[Callable[[int, float], None]] = None
        self._temp_callback: Optional[Callable[[int, float, float], None]] = None
        self._heartbeat_callback: Optional[Callable[[int, int, int, int], None]] = None

        self.plat_qadr = None
        self.plat_dadr = None
        self.anchor_sids = None
        self.plat_sids = None
        self.cable_tids = None
        self._home_cable_lengths_m = None

    def _expand_axis_param(self, value, default: float):
        if value is None:
            return {aid: float(default) for aid in self.axis_ids}
        if len(value) != len(self.axis_ids):
            raise ValueError("Axis parameter length must match axis_ids")
        return {aid: float(v) for aid, v in zip(self.axis_ids, value)}

    def _expand_axis_list(self, value, default: float):
        if value is None:
            return [float(default)] * len(self.axis_ids)
        if len(value) != len(self.axis_ids):
            raise ValueError("Axis parameter length must match axis_ids")
        return [float(v) for v in value]

    def start(self):
        """Start the MuJoCo simulation."""
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("MuJoCo not installed. Install with: pip install mujoco") from exc

        logger.info(f"Starting MuJoCo simulation with model: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self._setup_ids()

        self.model.opt.timestep = 0.0002
        with self._data_access():
            mujoco.mj_forward(self.model, self.data)
            self._home_cable_lengths_m = self._compute_cable_lengths_m_locked()
            self._refresh_feedback_locked()

        self.running = True
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()

        if self.enable_viewer:
            self._start_viewer()

    def stop(self):
        """Stop the MuJoCo simulation."""
        logger.info("Stopping MuJoCo simulation")
        self.running = False
        if hasattr(self, "sim_thread"):
            self.sim_thread.join(timeout=1.0)

    def _setup_ids(self):
        """Set up all IDs and addresses needed for simulation."""
        PLATFORM_JOINTS = ["jx", "jy", "jz", "jroll", "jpitch"]
        ANCHOR_SITES = ["a1", "a2", "a3", "a4", "a5", "a6"]
        PLAT_SITES = ["p1", "p2", "p3", "p4", "p5", "p6"]
        CABLE_TENDONS = ["cable1", "cable2", "cable3", "cable4", "cable5", "cable6"]

        plat_jids = [self.model.joint(name).id for name in PLATFORM_JOINTS]
        self.plat_qadr = np.array([int(self.model.jnt_qposadr[jid]) for jid in plat_jids], dtype=int)
        self.plat_dadr = np.array([int(self.model.jnt_dofadr[jid]) for jid in plat_jids], dtype=int)

        self.anchor_sids = np.array([self.model.site(name).id for name in ANCHOR_SITES], dtype=int)
        self.plat_sids = np.array([self.model.site(name).id for name in PLAT_SITES], dtype=int)
        self.cable_tids = np.array([self.model.tendon(name).id for name in CABLE_TENDONS], dtype=int)

    def _simulation_loop(self):
        """Main simulation loop: compute cable tensions, apply wrench, then step dynamics."""
        import mujoco

        dt = float(self.model.opt.timestep)

        while self.running:
            with self._data_access():
                mujoco.mj_forward(self.model, self.data)
                lengths_m = self._compute_cable_lengths_m_locked()
                J_plat = self._compute_cable_jacobian_locked()
                qd_plat = self.data.qvel[self.plat_dadr].copy()
                length_rates_mps = J_plat @ qd_plat

                self._refresh_feedback_locked(lengths_m=lengths_m, length_rates_mps=length_rates_mps)

                tensions = np.zeros(len(self.axis_ids), dtype=float)
                torques = np.zeros(len(self.axis_ids), dtype=float)
                for i, aid in enumerate(self.axis_ids):
                    torque_nm = self._compute_axis_torque(aid, i)
                    torque_limit = max(0.0, float(self._torque_limit_nm.get(aid, 0.0)))
                    if torque_limit > 0.0:
                        torque_nm = float(np.clip(torque_nm, -torque_limit, torque_limit))

                    tension_N = self._torque_to_tension(aid, torque_nm)
                    if tension_N < 0.0:
                        tension_N = 0.0

                    torques[i] = torque_nm
                    tensions[i] = tension_N
                    self._axis_torque_applied[aid] = float(torque_nm)
                    self._cable_tension_fb[aid] = float(tension_N)

                self.data.qfrc_applied[:] = 0.0
                self.data.qfrc_applied[self.plat_dadr] = -(J_plat.T @ tensions)

                mujoco.mj_step(self.model, self.data)
                mujoco.mj_forward(self.model, self.data)

                lengths_m = self._compute_cable_lengths_m_locked()
                J_plat = self._compute_cable_jacobian_locked()
                qd_plat = self.data.qvel[self.plat_dadr].copy()
                length_rates_mps = J_plat @ qd_plat
                self._refresh_feedback_locked(lengths_m=lengths_m, length_rates_mps=length_rates_mps)
                self._send_feedback_locked()

            time.sleep(dt)

    def _compute_axis_torque(self, aid: int, idx: int) -> float:
        if self._axis_state.get(aid) != "closed_loop":
            return 0.0

        if self._axis_mode.get(aid) != "position":
            return float(self._axis_torque_cmd.get(aid, 0.0))

        pos_fb = float(self._axis_pos_fb.get(aid, 0.0))
        vel_fb = float(self._axis_vel_fb.get(aid, 0.0))
        pos_err = float(self._axis_pos_cmd.get(aid, pos_fb)) - pos_fb
        vel_err = float(self._axis_vel_ff.get(aid, 0.0)) - vel_fb

        # The host position command sign is defined in spool turns; map control effort
        # into the same motor-torque sign convention used by torque commands.
        motor_direction = float(np.sign(self._torque_per_tension.get(aid, DEFAULT_TORQUE_PER_TENSION_NM_PER_N)))
        if motor_direction == 0.0:
            motor_direction = 1.0

        return (
            motor_direction
            * (
                float(self._spool_kp.get(aid, 0.0)) * pos_err
                + float(self._spool_kd.get(aid, 0.0)) * vel_err
            )
            + float(self._axis_torque_ff.get(aid, 0.0))
        )

    def _torque_to_tension(self, aid: int, torque_nm: float) -> float:
        torque_per_tension = float(self._torque_per_tension.get(aid, DEFAULT_TORQUE_PER_TENSION_NM_PER_N))
        if abs(torque_per_tension) < 1e-9:
            return 0.0
        return float(torque_nm / torque_per_tension)

    def _refresh_feedback_locked(self, lengths_m=None, length_rates_mps=None):
        if lengths_m is None:
            lengths_m = self._compute_cable_lengths_m_locked()
        if length_rates_mps is None:
            J_plat = self._compute_cable_jacobian_locked()
            qd_plat = self.data.qvel[self.plat_dadr].copy()
            length_rates_mps = J_plat @ qd_plat

        for i, aid in enumerate(self.axis_ids):
            delta_mm = 1000.0 * (float(lengths_m[i]) - float(self._home_cable_lengths_m[i]))
            mm_per_turn = float(self.mm_per_turn[i])
            pos_turns = 0.0 if abs(mm_per_turn) < 1e-9 else delta_mm / mm_per_turn
            vel_turnsps = 0.0 if abs(mm_per_turn) < 1e-9 else (1000.0 * float(length_rates_mps[i])) / mm_per_turn
            self._axis_pos_fb[aid] = float(pos_turns + self._axis_pos_offset[aid])
            self._axis_vel_fb[aid] = float(vel_turnsps)

    def _compute_cable_lengths_m_locked(self):
        anchor_xyz = self.data.site_xpos[self.anchor_sids]
        plat_xyz = self.data.site_xpos[self.plat_sids]
        return np.linalg.norm(plat_xyz - anchor_xyz, axis=1)

    def _compute_cable_jacobian_locked(self):
        import mujoco

        J_plat = np.zeros((len(self.axis_ids), len(self.plat_dadr)), dtype=float)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)

        for i in range(len(self.axis_ids)):
            a = self.data.site_xpos[self.anchor_sids[i]].copy()
            p = self.data.site_xpos[self.plat_sids[i]].copy()
            d = p - a
            Li = float(np.linalg.norm(d))
            u = np.zeros(3, dtype=float) if Li < 1e-12 else (d / Li)

            jacp[:] = 0.0
            jacr[:] = 0.0
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.plat_sids[i])
            J_plat[i, :] = (u @ jacp)[self.plat_dadr]

        return J_plat

    def _data_access(self):
        """Lock MuJoCo data access across simulation and passive viewer threads."""
        stack = ExitStack()
        stack.enter_context(self._mj_lock)
        if self.viewer is not None:
            stack.enter_context(self.viewer.lock())
        return stack

    def _send_feedback_locked(self):
        for aid in self.axis_ids:
            if self._position_callback:
                self._position_callback(aid, float(self._axis_pos_fb[aid]))
            if self._velocity_callback:
                self._velocity_callback(aid, float(self._axis_vel_fb[aid]))
            if self._bus_callback:
                self._bus_callback(aid, 24.0, 1.0)
            if self._current_callback:
                self._current_callback(aid, 0.5)
            if self._temp_callback:
                self._temp_callback(aid, 30.0, 35.0)
            if self._heartbeat_callback:
                self._heartbeat_callback(aid, 0, 8, 0)

    def set_axis_position(self, axis_id: int, position: float):
        """Set the axis position command in turns."""
        if axis_id in self.axis_ids:
            self._axis_pos_cmd[axis_id] = float(position)

    def set_axis_position_command(
        self,
        axis_id: int,
        position: float,
        velocity_ff: float = 0.0,
        torque_ff: float = 0.0,
    ):
        """Set the local spool servo reference."""
        if axis_id in self.axis_ids:
            self._axis_pos_cmd[axis_id] = float(position)
            self._axis_vel_ff[axis_id] = float(velocity_ff)
            self._axis_torque_ff[axis_id] = float(torque_ff)

    def set_axis_torque(self, axis_id: int, torque: float):
        """Set spool torque setpoint (Nm)."""
        if axis_id in self.axis_ids:
            self._axis_torque_cmd[axis_id] = float(torque)

    def get_axis_position(self, axis_id: int) -> Optional[float]:
        if axis_id not in self.axis_ids:
            return None
        return float(self._axis_pos_fb[axis_id])

    def get_axis_velocity(self, axis_id: int) -> Optional[float]:
        if axis_id not in self.axis_ids:
            return None
        return float(self._axis_vel_fb[axis_id])

    def get_platform_state(self):
        """Get platform state [x,y,z,roll,pitch], [xd,yd,zd,rolld,pitchd]."""
        if self.data is None:
            return None, None
        with self._data_access():
            q = self.data.qpos[self.plat_qadr].copy()
            qd = self.data.qvel[self.plat_dadr].copy()
        return q, qd

    def get_cable_tensions(self):
        if self.data is None:
            return None
        return [float(self._cable_tension_fb[aid]) for aid in self.axis_ids]

    def get_axis_torques(self):
        if self.data is None:
            return None
        return [float(self._axis_torque_applied[aid]) for aid in self.axis_ids]

    def get_sim_time(self):
        if self.data is None:
            return None
        with self._data_access():
            return float(self.data.time)

    def compute_platform_wrench(self, qdd_cmd):
        """Compute platform generalized wrench tau = (M*qdd_full + bias)[plat_dofs]."""
        if self.data is None:
            return np.zeros(5, dtype=float)
        with self._data_access():
            import mujoco

            qdd_cmd = np.asarray(qdd_cmd, dtype=float)
            qdd_full = np.zeros(self.model.nv, dtype=float)
            qdd_full[self.plat_dadr] = qdd_cmd

            mujoco.mj_forward(self.model, self.data)
            M = np.zeros((self.model.nv, self.model.nv), dtype=float)
            mujoco.mj_fullM(self.model, M, self.data.qM)
            bias = self.data.qfrc_bias.copy()
            tau_full = M @ qdd_full + bias
            return tau_full[self.plat_dadr].copy()

    def get_cable_jacobian_plat(self):
        if self.data is None:
            return np.zeros((len(self.axis_ids), 5), dtype=float)
        with self._data_access():
            return self._compute_cable_jacobian_locked()

    def set_controller_mode(self, axis_id: int, mode: str):
        if axis_id in self.axis_ids:
            self._axis_mode[axis_id] = str(mode)

    def set_axis_state(self, axis_id: int, state: str):
        if axis_id in self.axis_ids:
            self._axis_state[axis_id] = str(state)
            if str(state) == "idle":
                self._axis_torque_cmd[axis_id] = 0.0
                self._axis_torque_ff[axis_id] = 0.0

    def set_absolute_position(self, axis_id: int, position: float):
        """Shift the reported encoder frame without moving the simulated plant."""
        if axis_id not in self.axis_ids:
            return
        current_fb = float(self._axis_pos_fb[axis_id])
        self._axis_pos_offset[axis_id] += float(position) - current_fb
        self._axis_pos_cmd[axis_id] = float(position)

    def set_hand_pose(self, t_mm, q):
        """Unused by plant-only simulation driver (control loop lives in robot_server)."""
        return

    def set_position_callback(self, callback: Callable[[int, float], None]):
        self._position_callback = callback

    def set_velocity_callback(self, callback: Callable[[int, float], None]):
        self._velocity_callback = callback

    def set_bus_callback(self, callback: Callable[[int, float, float], None]):
        self._bus_callback = callback

    def set_current_callback(self, callback: Callable[[int, float], None]):
        self._current_callback = callback

    def set_temp_callback(self, callback: Callable[[int, float, float], None]):
        self._temp_callback = callback

    def set_heartbeat_callback(self, callback: Callable[[int, int, int, int], None]):
        self._heartbeat_callback = callback

    def configure_spool_controller(self, kp=None, kd=None, torque_limit=None):
        if kp is not None:
            self._spool_kp = self._expand_axis_param(kp, 0.377)
        if kd is not None:
            self._spool_kd = self._expand_axis_param(kd, 0.012566)
        if torque_limit is not None:
            self._torque_limit_nm = self._expand_axis_param(torque_limit, 1.0)
        logger.info(
            "Configured simulation spool controller gains: "
            f"kp={list(self._spool_kp.values())}, kd={list(self._spool_kd.values())}, "
            f"torque_limit={list(self._torque_limit_nm.values())}"
        )
        return True

    def _start_viewer(self):
        import mujoco
        import mujoco.viewer

        def viewer_thread():
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                self.viewer = viewer
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                while self.running and viewer.is_running():
                    with self._data_access():
                        viewer.sync()
                    time.sleep(0.01)
                self.viewer = None

        self.viewer_thread = threading.Thread(target=viewer_thread, daemon=True)
        self.viewer_thread.start()
