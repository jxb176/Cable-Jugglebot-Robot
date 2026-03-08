"""
MuJoCo-based simulation driver for cable robot.
"""

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Callable, List, Tuple

import numpy as np

from .driver_interface import RobotDriver


logger = logging.getLogger(__name__)


class MuJoCoSimulationDriver(RobotDriver):
    """
    MuJoCo-based simulation driver with full cable robot dynamics.
    """

    def __init__(self, axis_ids: List[int] = None, model_path: str = None, enable_viewer: bool = False):
        self.axis_ids = axis_ids or [0, 1, 2, 3, 4, 5]
        self.model_path = model_path or str(Path(__file__).parent / "cable_robot_5dof_winch.xml")
        self.enable_viewer = enable_viewer

        # MuJoCo components
        self.model = None
        self.data = None

        # Control parameters (similar to prototype)
        self.Kp = np.diag([5000.0, 5000.0, 6000.0, 500.0, 500.0])
        self.Kd = np.diag([500.0, 500.0, 400.0, 100.0, 100.0])
        self.Tmin = 5.0
        self.Tmax = 200.0
        self.lam = 1e-2
        self.alpha_Tprev = 0.7

        # State
        self.running = False
        self.hand_pose_cmd = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)  # position, quaternion
        self.t_prev = 0.0
        self.T_prev = None

        # Callbacks
        self._position_callback: Optional[Callable[[int, float], None]] = None
        self._velocity_callback: Optional[Callable[[int, float], None]] = None
        self._bus_callback: Optional[Callable[[int, float, float], None]] = None
        self._current_callback: Optional[Callable[[int, float], None]] = None
        self._temp_callback: Optional[Callable[[int, float, float], None]] = None
        self._heartbeat_callback: Optional[Callable[[int, int, int, int], None]] = None

        # Simulation IDs (will be set in start())
        self.plat_qadr = None
        self.plat_dadr = None
        self.spool_qadr = None
        self.spool_dadr = None
        self.act_ids = None
        self.anchor_sids = None
        self.plat_sids = None
        self.r = None  # capstan radii
        self.I_spool = None
        self.b_spool = None

    def start(self):
        """Start the MuJoCo simulation."""
        try:
            import mujoco
        except ImportError:
            raise ImportError("MuJoCo not installed. Install with: pip install mujoco")

        logger.info(f"Starting MuJoCo simulation with model: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # Set up IDs and addresses
        self._setup_ids()

        # Set timestep for real-time control (500 Hz)
        self.model.opt.timestep = 0.002

        self.running = True

        # Start simulation thread
        self.sim_thread = threading.Thread(target=self._simulation_loop, daemon=True)
        self.sim_thread.start()

        if self.enable_viewer:
            self._start_viewer()

    def stop(self):
        """Stop the MuJoCo simulation."""
        logger.info("Stopping MuJoCo simulation")
        self.running = False
        if hasattr(self, 'sim_thread'):
            self.sim_thread.join(timeout=1.0)

    def _setup_ids(self):
        """Set up all the IDs and addresses needed for simulation."""
        import mujoco

        # Joint names
        PLATFORM_JOINTS = ["jx", "jy", "jz", "jroll", "jpitch"]
        SPOOL_JOINTS = ["spool1", "spool2", "spool3", "spool4", "spool5", "spool6"]
        MOTOR_ACTS = ["m1", "m2", "m3", "m4", "m5", "m6"]
        ANCHOR_SITES = ["a1", "a2", "a3", "a4", "a5", "a6"]
        PLAT_SITES = ["p1", "p2", "p3", "p4", "p5", "p6"]

        # Platform joints
        plat_jids = [self.model.joint(n).id for n in PLATFORM_JOINTS]
        self.plat_qadr = np.array([int(self.model.jnt_qposadr[j]) for j in plat_jids], dtype=int)
        self.plat_dadr = np.array([int(self.model.jnt_dofadr[j]) for j in plat_jids], dtype=int)

        # Spool joints
        spool_jids = [self.model.joint(n).id for n in SPOOL_JOINTS]
        self.spool_qadr = np.array([int(self.model.jnt_qposadr[j]) for j in spool_jids], dtype=int)
        self.spool_dadr = np.array([int(self.model.jnt_dofadr[j]) for j in spool_jids], dtype=int)

        # Actuators
        self.act_ids = np.array([self.model.actuator(n).id for n in MOTOR_ACTS], dtype=int)

        # Sites
        self.anchor_sids = np.array([self.model.site(n).id for n in ANCHOR_SITES], dtype=int)
        self.plat_sids = np.array([self.model.site(n).id for n in PLAT_SITES], dtype=int)

        # Capstan radii (from tendon coefficients)
        PAYOUT_TENDONS = ["payout1", "payout2", "payout3", "payout4", "payout5", "payout6"]
        payout_tids = np.array([self.model.tendon(n).id for n in PAYOUT_TENDONS], dtype=int)
        self.r = self._capstan_radii_from_xml_coefs(payout_tids)

        # Spool dynamics
        self.b_spool = self.model.dof_damping[self.spool_dadr].astype(float)
        self.I_spool = self.model.dof_armature[self.spool_dadr].astype(float)

    def _capstan_radii_from_xml_coefs(self, payout_tids):
        """Get capstan radii from tendon coefficients."""
        import mujoco

        mujoco.mj_forward(self.model, self.data)
        q0 = self.data.qpos.copy()
        eps = 1e-6
        r = np.zeros(6, dtype=float)

        for i, tid in enumerate(payout_tids):
            L0 = float(self.data.ten_length[tid])
            self.data.qpos[:] = q0
            self.data.qpos[self.spool_qadr[i]] = q0[self.spool_qadr[i]] + eps
            mujoco.mj_forward(self.model, self.data)
            Lp = float(self.data.ten_length[tid])
            ri = (Lp - L0) / eps
            r[i] = float(np.sign(ri) * max(abs(ri), 1e-12))
            self.data.qpos[:] = q0
            mujoco.mj_forward(self.model, self.data)

        return r

    def _simulation_loop(self):
        """Main simulation loop running at 500 Hz."""
        import mujoco

        dt = float(self.model.opt.timestep)

        while self.running:
            t = float(self.data.time)

            # Get current platform state
            q = self.data.qpos[self.plat_qadr].copy()
            qd = self.data.qvel[self.plat_dadr].copy()

            # Desired pose from hand_pose_cmd
            t_des, q_des = self.hand_pose_cmd
            # Convert mm to meters
            t_des_m = (t_des[0] / 1000.0, t_des[1] / 1000.0, t_des[2] / 1000.0)
            qref = np.array([t_des_m[0], t_des_m[1], t_des_m[2], 0.0, 0.0])  # x,y,z, assume no roll/pitch for now
            qdref = np.zeros(5)  # TODO: compute from pose profile
            qddref = np.zeros(5)

            # Computed acceleration control
            e = qref - q
            ed = qdref - qd
            qdd_cmd = qddref + (self.Kp @ e) + (self.Kd @ ed)

            # Full acceleration vector
            qdd_full = np.zeros(self.model.nv, dtype=float)
            qdd_full[self.plat_dadr] = qdd_cmd

            # Mass matrix and bias forces
            mujoco.mj_forward(self.model, self.data)
            M = self._full_mass_matrix()
            bias = self.data.qfrc_bias.copy()

            tau_full = M @ qdd_full + bias
            tau_plat_des = tau_full[self.plat_dadr]

            # Cable length jacobian
            _, J_len_plat = self._cable_lengths_and_jacobian_plat()

            # Tension allocation
            T_des = self._solve_tensions_least_squares(J_len_plat, tau_plat_des)

            # Motor torques (simplified - no feedforward for now)
            tau_cmd = self.r * T_des

            # Apply torque limits
            umin = np.full(6, -10.0)  # From XML
            umax = np.full(6, 10.0)
            tau_cmd = np.clip(tau_cmd, umin, umax)

            # Apply to actuators
            self.data.ctrl[self.act_ids] = tau_cmd

            # Step simulation
            mujoco.mj_step(self.model, self.data)

            # Send feedback via callbacks
            self._send_feedback()

            # Sleep to maintain 500 Hz
            time.sleep(dt)

    def _full_mass_matrix(self):
        """Compute full mass matrix."""
        import mujoco

        M = np.zeros((self.model.nv, self.model.nv), dtype=float)
        mujoco.mj_fullM(self.model, M, self.data.qM)
        return M

    def _cable_lengths_and_jacobian_plat(self):
        """Compute cable lengths and Jacobian w.r.t platform DOFs."""
        import mujoco

        mujoco.mj_forward(self.model, self.data)

        L = np.zeros(6, dtype=float)
        J_plat = np.zeros((6, len(self.plat_dadr)), dtype=float)

        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)

        for i in range(6):
            a = self.data.site_xpos[self.anchor_sids[i]].copy()
            p = self.data.site_xpos[self.plat_sids[i]].copy()

            d = p - a
            Li = float(np.linalg.norm(d))
            if Li < 1e-12:
                u = np.zeros(3, dtype=float)
            else:
                u = d / Li

            # Jacobian of platform site
            jacp[:] = 0.0
            jacr[:] = 0.0
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.plat_sids[i])

            dL_dqvel = u @ jacp
            L[i] = Li
            J_plat[i, :] = dL_dqvel[self.plat_dadr]

        return L, J_plat

    def _solve_tensions_least_squares(self, J_len_plat, tau_plat_des):
        """Solve for cable tensions using least squares."""
        J = np.asarray(J_len_plat, dtype=float)
        A = J.T
        nt = A.shape[1]

        lb = np.full(nt, self.Tmin, dtype=float)
        ub = np.full(nt, self.Tmax, dtype=float)

        if self.T_prev is None:
            Tref = lb.copy()
        else:
            Tref = self.alpha_Tprev * np.asarray(self.T_prev, dtype=float) + (1.0 - self.alpha_Tprev) * lb

        T = Tref.copy()

        ATA = A.T @ A
        L = float(np.linalg.norm(ATA, 2) + self.lam)
        step = 1.0 / max(L, 1e-9)

        for _ in range(80):  # iters
            grad = 2.0 * (A.T @ (A @ T - tau_plat_des)) + 2.0 * self.lam * (T - Tref)
            T = np.clip(T - step * grad, lb, ub)

        self.T_prev = T.copy()
        return T

    def _send_feedback(self):
        """Send feedback via callbacks."""
        # Spool positions (turns)
        for i, aid in enumerate(self.axis_ids):
            if self._position_callback:
                pos = float(self.data.qpos[self.spool_qadr[i]])
                self._position_callback(aid, pos)

            if self._velocity_callback:
                vel = float(self.data.qvel[self.spool_dadr[i]])
                self._velocity_callback(aid, vel)

            # Fake other feedback
            if self._bus_callback:
                self._bus_callback(aid, 24.0, 1.0)

            if self._current_callback:
                self._current_callback(aid, 0.5)

            if self._temp_callback:
                self._temp_callback(aid, 30.0, 35.0)

            if self._heartbeat_callback:
                self._heartbeat_callback(aid, 0, 8, 0)  # OK status

    def set_axis_position(self, axis_id: int, position: float):
        """Set position setpoint - not used in this driver."""
        pass

    def set_axis_torque(self, axis_id: int, torque: float):
        """Set torque setpoint - handled internally."""
        pass

    def get_axis_position(self, axis_id: int) -> Optional[float]:
        """Get position feedback."""
        if self.data is not None:
            idx = self.axis_ids.index(axis_id)
            return float(self.data.qpos[self.spool_qadr[idx]])
        return None

    def get_axis_velocity(self, axis_id: int) -> Optional[float]:
        """Get velocity feedback."""
        if self.data is not None:
            idx = self.axis_ids.index(axis_id)
            return float(self.data.qvel[self.spool_dadr[idx]])
        return None

    def set_controller_mode(self, axis_id: int, mode: str):
        """Set controller mode - handled internally."""
        pass

    def set_axis_state(self, axis_id: int, state: str):
        """Set axis state - handled internally."""
        pass

    def set_absolute_position(self, axis_id: int, position: float):
        """Set absolute position reference."""
        if self.data is not None:
            idx = self.axis_ids.index(axis_id)
            self.data.qpos[self.spool_qadr[idx]] = position

    def set_hand_pose(self, t_mm, q):
        """Set desired hand pose for simulation control."""
        self.hand_pose_cmd = (t_mm, q)

    # Callback setters
    def set_position_callback(self, callback: Callable[[int, float], None]):
        self._position_callback = callback

    def set_velocity_callback(self, callback: Callable[[int, float], None]):
        self._velocity_callback = callback

    def set_bus_callback(self, callback: Callable[[int, float, float], None]):
        self._bus_callback = callback

    def set_current_callback(self, callback: Callable[[int, float], None]]:
        self._current_callback = callback

    def set_temp_callback(self, callback: Callable[[int, float, float], None]):
        self._temp_callback = callback

    def set_heartbeat_callback(self, callback: Callable[[int, int, int, int], None]):
        self._heartbeat_callback = callback

    def _start_viewer(self):
        """Start the MuJoCo viewer in a separate thread."""
        import mujoco
        import mujoco.viewer

        def viewer_thread():
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
                while self.running and viewer.is_running():
                    time.sleep(0.01)  # Small sleep to not hog CPU

        self.viewer_thread = threading.Thread(target=viewer_thread, daemon=True)
        self.viewer_thread.start()
