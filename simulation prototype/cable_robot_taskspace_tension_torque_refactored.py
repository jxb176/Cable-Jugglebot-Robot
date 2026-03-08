#!/usr/bin/env python3
"""
6-cable robot: task-space trajectory tracking with the same architecture as the single-DoF hand test.

(A) Platform computed-accel (no mj_inverse):
      qdd_cmd      = qdd_ref + Kp*(q_ref-q) + Kd*(qd_ref-qd)
      tau_plat_des = (M*qdd_full + bias)[plat_dofs]

(B) Tension allocation (bounded LS + regularization toward Tmin / previous):
      min_T || (-J_len^T)T - tau_plat_des ||^2 + lam||T - Tref||^2
      s.t. Tmin <= T <= Tmax

(C) Motor torque commands (winch-side):
      tau_cmd = r*T_des + I*thetadd_ref + b*thetad_ref
      where thetad_ref = payout_dot/r, thetadd_ref = payout_ddot/r

Notes
-----
- Cable length Jacobian is computed analytically using mj_jacSite (no finite differences).
- Measured cable tension prefers equality constraint force via efc_force if exposed; otherwise uses spool dynamics.

Inputs
------
- pose_cmd_full.csv: columns t, x_mm,y_mm,z_mm, roll_deg,pitch_deg
  optional: vx_mps,vy_mps,vz_mps and ax_mps2,ay_mps2,az_mps2.
"""

import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt


# -------------------------
# Config (names must match XML)
# -------------------------
XML_PATH  = "cable_robot_5dof_winch.xml"
POSE_CSV  = "pose_cmd_full.csv"

PLATFORM_JOINTS = ["jx", "jy", "jz", "jroll", "jpitch"]

SPOOL_JOINTS = ["spool1", "spool2", "spool3", "spool4", "spool5", "spool6"]
MOTOR_ACTS   = ["m1", "m2", "m3", "m4", "m5", "m6"]

ANCHOR_SITES   = ["a1", "a2", "a3", "a4", "a5", "a6"]
PLAT_SITES     = ["p1", "p2", "p3", "p4", "p5", "p6"]
CABLE_TENDONS  = ["cable1", "cable2", "cable3", "cable4", "cable5", "cable6"]
PAYOUT_TENDONS = ["payout1", "payout2", "payout3", "payout4", "payout5", "payout6"]
EQ_NAMES       = ["eq_cable1", "eq_cable2", "eq_cable3", "eq_cable4", "eq_cable5", "eq_cable6"]


# -------------------------
# Small utils
# -------------------------
def finite_diff(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    y = np.zeros_like(x)
    if len(t) < 3:
        return y
    y[1:-1] = (x[2:] - x[:-2]) / (t[2:] - t[:-2])
    y[0] = (x[1] - x[0]) / max(t[1] - t[0], 1e-12)
    y[-1] = (x[-1] - x[-2]) / max(t[-1] - t[-2], 1e-12)
    return y


def full_mass_matrix(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    M = np.zeros((model.nv, model.nv), dtype=float)
    mujoco.mj_fullM(model, M, data.qM)
    return M


def load_pose_profile_csv(csv_path: str):
    df = pd.read_csv(csv_path)
    t = df["t"].to_numpy(dtype=float)

    q = np.zeros((len(t), 5), dtype=float)
    q[:, 0] = df["x_mm"].to_numpy(dtype=float) / 1000.0
    q[:, 1] = df["y_mm"].to_numpy(dtype=float) / 1000.0
    q[:, 2] = df["z_mm"].to_numpy(dtype=float) / 1000.0
    q[:, 3] = np.unwrap(np.deg2rad(df["roll_deg"].to_numpy(dtype=float)))
    q[:, 4] = np.unwrap(np.deg2rad(df["pitch_deg"].to_numpy(dtype=float)))

    # velocities
    qd = np.zeros_like(q)
    if all(c in df.columns for c in ["vx_mps", "vy_mps", "vz_mps"]):
        qd[:, 0] = df["vx_mps"].to_numpy(dtype=float)
        qd[:, 1] = df["vy_mps"].to_numpy(dtype=float)
        qd[:, 2] = df["vz_mps"].to_numpy(dtype=float)
        # roll/pitch vel if provided? otherwise finite-diff:
        qd[:, 3] = finite_diff(q[:, 3], t)
        qd[:, 4] = finite_diff(q[:, 4], t)
    else:
        for j in range(q.shape[1]):
            qd[:, j] = finite_diff(q[:, j], t)

    # accelerations
    qdd = np.zeros_like(q)
    if all(c in df.columns for c in ["ax_mps2", "ay_mps2", "az_mps2"]):
        qdd[:, 0] = df["ax_mps2"].to_numpy(dtype=float)
        qdd[:, 1] = df["ay_mps2"].to_numpy(dtype=float)
        qdd[:, 2] = df["az_mps2"].to_numpy(dtype=float)
        qdd[:, 3] = finite_diff(qd[:, 3], t)
        qdd[:, 4] = finite_diff(qd[:, 4], t)
    else:
        for j in range(q.shape[1]):
            qdd[:, j] = finite_diff(qd[:, j], t)

    return t, q, qd, qdd


def make_reference_from_profile(t_prof, q_prof, qd_prof, qdd_prof):
    def interp_vec(tt, arr):
        out = np.zeros(arr.shape[1], dtype=float)
        for j in range(arr.shape[1]):
            out[j] = np.interp(tt, t_prof, arr[:, j])
        return out

    def reference(t):
        tt = float(np.clip(t, t_prof[0], t_prof[-1]))
        return (interp_vec(tt, q_prof),
                interp_vec(tt, qd_prof),
                interp_vec(tt, qdd_prof))
    return reference


def solve_tensions_least_squares(J_len_plat, tau_plat_des, T_prev, Tmin, Tmax, lam=1e-2, iters=80, alpha=0.7):
    """
    min_T || (-J^T)T - tau ||^2 + lam||T - Tref||^2, s.t. Tmin<=T<=Tmax

    J_len_plat: (6,5) where J[i,j] = dL_i / d q_plat_j
    tau_plat_des: (5,)
    """
    J = np.asarray(J_len_plat, dtype=float)   # (6,5)
    A = J.T                                  # (5,6)
    nt = A.shape[1]

    lb = np.full(nt, Tmin, dtype=float)
    ub = np.full(nt, Tmax, dtype=float)

    if T_prev is None:
        Tref = lb.copy()
    else:
        Tref = alpha*np.asarray(T_prev, dtype=float) + (1.0-alpha)*lb

    T = Tref.copy()

    # conservative step
    ATA = A.T @ A
    L = float(np.linalg.norm(ATA, 2) + lam)
    step = 1.0 / max(L, 1e-9)

    for _ in range(iters):
        grad = 2.0*(A.T @ (A @ T - tau_plat_des)) + 2.0*lam*(T - Tref)
        T = np.clip(T - step*grad, lb, ub)

    return T


# -------------------------
# Cable kinematics: length + Jacobian w.r.t platform DOFs (analytic)
# -------------------------
def cable_lengths_and_jacobian_plat(model, data, anchor_sids, plat_sids, plat_dadr):
    """
    For each cable i: L_i = || p_plat_i - p_anchor_i ||, anchor site is world-fixed.

    Returns:
      L: (6,)
      J_plat: (6,5) with columns aligned with plat_dadr order (platform DOFs)
    """
    mujoco.mj_forward(model, data)

    L = np.zeros(6, dtype=float)
    J_plat = np.zeros((6, len(plat_dadr)), dtype=float)

    # workspace arrays for jacobians (3 x nv)
    jacp = np.zeros((3, model.nv), dtype=float)
    jacr = np.zeros((3, model.nv), dtype=float)

    for i in range(6):
        a = data.site_xpos[anchor_sids[i]].copy()
        p = data.site_xpos[plat_sids[i]].copy()

        d = p - a
        Li = float(np.linalg.norm(d))
        if Li < 1e-12:
            u = np.zeros(3, dtype=float)
        else:
            u = d / Li

        # translational jacobian of platform site wrt generalized velocities
        jacp[:] = 0.0
        jacr[:] = 0.0
        mujoco.mj_jacSite(model, data, jacp, jacr, plat_sids[i])

        # dL/dqdot = u^T * Jp  (anchor is fixed so no subtraction)
        dL_dqvel = u @ jacp  # (nv,)

        # pick only platform dofs
        L[i] = Li
        J_plat[i, :] = dL_dqvel[plat_dadr]

    return L, J_plat


# -------------------------
# Payout reference from pose profile
# -------------------------
def build_payout_reference_from_pose_profile(model, data, t_prof, q_prof, plat_qadr):
    """
    For each profile sample:
      set platform qpos, forward, get cable lengths
    Then payout = cable_len - eq_offset, where eq_offset is polycoef[0] for tendon equality.

    Returns payout_ref(t) -> (Lp, Lpd, Lpdd) each shape (6,)
    """
    # equality offsets (polycoef[0]) for each tendon equality
    eq_offsets = np.zeros(6, dtype=float)
    for i, eqn in enumerate(EQ_NAMES):
        eq_id = model.equality(eqn).id
        eq_offsets[i] = float(model.eq_data[eq_id, 0])

    # tendon IDs for cable lengths
    cable_tids = np.array([model.tendon(n).id for n in CABLE_TENDONS], dtype=int)

    qpos_save = data.qpos.copy()
    qvel_save = data.qvel.copy()

    L_cable = np.zeros((len(t_prof), 6), dtype=float)

    for k in range(len(t_prof)):
        data.qpos[:] = qpos_save
        for j in range(5):
            data.qpos[plat_qadr[j]] = q_prof[k, j]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)
        L_cable[k, :] = data.ten_length[cable_tids]

    # restore
    data.qpos[:] = qpos_save
    data.qvel[:] = qvel_save
    mujoco.mj_forward(model, data)

    L_payout   = L_cable - eq_offsets[None, :]
    Ld_payout  = np.vstack([finite_diff(L_payout[:, i], t_prof) for i in range(6)]).T
    Ldd_payout = np.vstack([finite_diff(Ld_payout[:, i], t_prof) for i in range(6)]).T

    def interp_vec(tt, arr):
        out = np.zeros(arr.shape[1], dtype=float)
        for j in range(arr.shape[1]):
            out[j] = np.interp(tt, t_prof, arr[:, j])
        return out

    def payout_ref(t):
        tt = float(np.clip(t, t_prof[0], t_prof[-1]))
        return (interp_vec(tt, L_payout),
                interp_vec(tt, Ld_payout),
                interp_vec(tt, Ldd_payout))

    return payout_ref


def capstan_radii_from_xml_coefs(model, data, spool_qadr, payout_tids):
    """
    Robustly estimate r_i = d(payout_len)/d(spool_angle) by perturbing each spool.
    (We *could* hardcode 0.01 from your XML coef, but this keeps it general.)
    """
    mujoco.mj_forward(model, data)
    q0 = data.qpos.copy()
    eps = 1e-6
    r = np.zeros(6, dtype=float)

    for i, tid in enumerate(payout_tids):
        L0 = float(data.ten_length[tid])
        data.qpos[:] = q0
        data.qpos[spool_qadr[i]] = q0[spool_qadr[i]] + eps
        mujoco.mj_forward(model, data)
        Lp = float(data.ten_length[tid])
        ri = (Lp - L0) / eps
        r[i] = float(np.sign(ri) * max(abs(ri), 1e-12))  # keep sign, avoid zeros
        data.qpos[:] = q0
        mujoco.mj_forward(model, data)

    return r


def measured_tensions(model, data, eq_ids, eq_adr_available, r, tau_cmd, I_spool, b_spool, spool_dadr):
    """
    Estimate cable tensions using spool dynamics:
    tau_cable = tau_cmd - I*thdd - b*thd, T = tau_cable / r

    We use dynamics-based measurement since equality constraint forces require explicit naming.
    """
    # fallback dynamics estimate
    thd  = data.qvel[spool_dadr].astype(float)
    thdd = data.qacc[spool_dadr].astype(float)
    tau_cable = tau_cmd - (I_spool * thdd + b_spool * thd)
    T = np.divide(tau_cable, r, out=np.zeros_like(tau_cable), where=np.abs(r) > 1e-12)
    return T


# -------------------------
# Main sim loop
# -------------------------
def run_sim(
    sim_T=2.0,
    Kp=np.diag([5000.0, 5000.0, 6000.0, 500.0, 500.0]),
    Kd=np.diag([ 500.0,  500.0,  400.0, 100.0, 100.0]),
    Tmin=5.0,
    Tmax=200.0,
    lam=1e-2,
    alpha_Tprev=0.7,
    torque_rate_limit=400.0,
):
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)

    # IDs / addresses
    plat_jids = [model.joint(n).id for n in PLATFORM_JOINTS]
    plat_qadr = np.array([int(model.jnt_qposadr[j]) for j in plat_jids], dtype=int)
    plat_dadr = np.array([int(model.jnt_dofadr[j])  for j in plat_jids], dtype=int)

    spool_jids = [model.joint(n).id for n in SPOOL_JOINTS]
    spool_qadr = np.array([int(model.jnt_qposadr[j]) for j in spool_jids], dtype=int)
    spool_dadr = np.array([int(model.jnt_dofadr[j])  for j in spool_jids], dtype=int)

    act_ids     = np.array([model.actuator(n).id for n in MOTOR_ACTS], dtype=int)
    cable_tids  = np.array([model.tendon(n).id for n in CABLE_TENDONS], dtype=int)
    payout_tids = np.array([model.tendon(n).id for n in PAYOUT_TENDONS], dtype=int)

    anchor_sids = np.array([model.site(n).id for n in ANCHOR_SITES], dtype=int)
    plat_sids   = np.array([model.site(n).id for n in PLAT_SITES], dtype=int)

    # equality ids - these constraints enforce cable length = L0 + payout
    # In our XML they're not explicitly named, so we'll use dynamics-based measurement instead
    eq_adr_available = False  # Our XML doesn't name the equality constraints

    # capstan radii
    r = capstan_radii_from_xml_coefs(model, data, spool_qadr, payout_tids)

    # spool dynamics (armature is the rotational inertia term you’re using)
    b_spool = model.dof_damping[spool_dadr].astype(float)
    I_spool = model.dof_armature[spool_dadr].astype(float)

    # profile + references
    t_prof, q_prof, qd_prof, qdd_prof = load_pose_profile_csv(POSE_CSV)
    ref = make_reference_from_profile(t_prof, q_prof, qd_prof, qdd_prof)
    payout_ref = build_payout_reference_from_pose_profile(model, data, t_prof, q_prof, plat_qadr)

    dt   = float(model.opt.timestep)
    nmax = int(sim_T / dt)

    # actuator saturation
    umin = np.zeros(6); umax = np.zeros(6)
    for i, aid in enumerate(act_ids):
        if model.actuator_ctrllimited[aid]:
            lo, hi = model.actuator_ctrlrange[aid]
            umin[i], umax[i] = float(lo), float(hi)
        else:
            umin[i], umax[i] = -np.inf, np.inf

    # logs
    t_log        = np.zeros(nmax)
    q_log        = np.zeros((nmax, 5))
    qd_log       = np.zeros((nmax, 5))
    q_ref_log    = np.zeros((nmax, 5))
    tau_plat_log = np.zeros((nmax, 5))

    T_des_log    = np.zeros((nmax, 6))
    T_meas_log   = np.zeros((nmax, 6))

    tau_cmd_log  = np.zeros((nmax, 6))
    tau_ff_log   = np.zeros((nmax, 6))

    th_log       = np.zeros((nmax, 6))
    thd_log      = np.zeros((nmax, 6))
    thdd_log     = np.zeros((nmax, 6))
    thd_ref_log  = np.zeros((nmax, 6))
    thdd_ref_log = np.zeros((nmax, 6))

    # previous for slew + regularization
    tau_prev = np.zeros(6, dtype=float)
    T_prev   = None

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT]  = True

        for k in range(nmax):
            if not viewer.is_running():
                break

            t = float(data.time)

            # references
            qref, qdref, qddref = ref(t)
            Lp_ref, Lpd_ref, Lpdd_ref = payout_ref(t)

            # current platform state (pre-step)
            q  = data.qpos[plat_qadr].copy()
            qd = data.qvel[plat_dadr].copy()

            # (A) computed-accel in platform coords
            e  = qref - q
            ed = qdref - qd
            qdd_cmd = qddref + (Kp @ e) + (Kd @ ed)

            qdd_full = np.zeros(model.nv, dtype=float)
            qdd_full[plat_dadr] = qdd_cmd

            mujoco.mj_forward(model, data)
            M    = full_mass_matrix(model, data)
            bias = data.qfrc_bias.copy()

            tau_full = M @ qdd_full + bias
            tau_plat_des = tau_full[plat_dadr]
            tau_plat_log[k, :] = tau_plat_des

            # (B) cable length jacobian wrt platform DOFs
            _, J_len_plat = cable_lengths_and_jacobian_plat(model, data, anchor_sids, plat_sids, plat_dadr)  # (6,5)

            # tension allocation
            T_des = solve_tensions_least_squares(
                J_len_plat=J_len_plat,
                tau_plat_des=tau_plat_des,
                T_prev=T_prev,
                Tmin=Tmin, Tmax=Tmax,
                lam=lam, iters=80,
                alpha=alpha_Tprev
            )
            T_prev = T_des.copy()

            # (C) motor torques
            thd_ref  = np.divide(Lpd_ref,  r, out=np.zeros_like(Lpd_ref),  where=np.abs(r) > 1e-12)
            thdd_ref = np.divide(Lpdd_ref, r, out=np.zeros_like(Lpdd_ref), where=np.abs(r) > 1e-12)

            tau_from_tension = r * T_des
            tau_act_ff       = I_spool * thdd_ref + b_spool * thd_ref
            tau_ff           = tau_from_tension + tau_act_ff

            tau_cmd = np.clip(tau_ff, umin, umax)

            # optional slew
            if torque_rate_limit is not None and np.isfinite(torque_rate_limit):
                max_step = float(torque_rate_limit) * dt
                tau_cmd = np.clip(tau_cmd, tau_prev - max_step, tau_prev + max_step)
            tau_prev = tau_cmd.copy()

            # apply & step
            data.ctrl[act_ids] = tau_cmd
            mujoco.mj_step(model, data)

            # measured tensions (using dynamics-based estimate since eq constraints not named)
            T_meas = measured_tensions(model, data, None, eq_adr_available, r, tau_cmd, I_spool, b_spool, spool_dadr)

            # spool states after step
            th_log[k, :]       = data.qpos[spool_qadr]
            thd_log[k, :]      = data.qvel[spool_dadr]
            thdd_log[k, :]     = data.qacc[spool_dadr]
            thd_ref_log[k, :]  = thd_ref
            thdd_ref_log[k, :] = thdd_ref

            # log
            t_log[k]        = t
            q_log[k, :]     = q
            qd_log[k, :]    = qd
            q_ref_log[k, :] = qref

            T_des_log[k, :]   = T_des
            T_meas_log[k, :]  = T_meas
            tau_cmd_log[k, :] = tau_cmd
            tau_ff_log[k, :]  = tau_ff

            viewer.sync()

    n = int(np.max(np.nonzero(t_log))) + 1 if np.any(t_log) else len(t_log)
    return {
        "t": t_log[:n],
        "q": q_log[:n], "qd": qd_log[:n], "qref": q_ref_log[:n],
        "tau_plat": tau_plat_log[:n],
        "T_des": T_des_log[:n], "T_meas": T_meas_log[:n],
        "tau_cmd": tau_cmd_log[:n], "tau_ff": tau_ff_log[:n],
        "th": th_log[:n], "thd": thd_log[:n], "thdd": thdd_log[:n],
        "thd_ref": thd_ref_log[:n], "thdd_ref": thdd_ref_log[:n],
        "r": r, "umin": umin, "umax": umax,
    }


def plot_results(res):
    t = res["t"]
    q, qref = res["q"], res["qref"]
    Tdes, Tmeas = res["T_des"], res["T_meas"]
    tau, tau_ff = res["tau_cmd"], res["tau_ff"]
    th, thd, thdd = res["th"], res["thd"], res["thdd"]
    thd_ref, thdd_ref = res["thd_ref"], res["thdd_ref"]

    labels = ["x", "y", "z", "roll", "pitch"]

    # Platform tracking
    fig, ax = plt.subplots(5, 1, sharex=True)
    for i in range(5):
        ax[i].plot(t, q[:, i], label=labels[i])
        ax[i].plot(t, qref[:, i], "--", label=f"{labels[i]}_ref")
        ax[i].legend(loc="upper right")
    ax[-1].set_xlabel("Time [s]")
    fig.suptitle("Platform tracking")

    # Spool state
    fig, ax = plt.subplots(3, 1, sharex=True)
    for i in range(6):
        ax[0].plot(t, th[:, i], label=f"θ{i+1}")
        ax[1].plot(t, thd[:, i], label=f"θ̇{i+1}")
        ax[2].plot(t, thdd[:, i], label=f"θ̈{i+1}")
    ax[0].set_ylabel("θ [rad]")
    ax[1].set_ylabel("θ̇ [rad/s]")
    ax[2].set_ylabel("θ̈ [rad/s²]")
    ax[2].set_xlabel("Time [s]")
    ax[0].set_title("Spool state (measured from MuJoCo)")

    # Ref overlay for rates/accels
    fig, ax = plt.subplots(2, 1, sharex=True)
    for i in range(6):
        ax[0].plot(t, thd[:, i], label=f"θ̇{i+1}")
        ax[0].plot(t, thd_ref[:, i], "--", label=f"θ̇{i+1}_ref")
        ax[1].plot(t, thdd[:, i], label=f"θ̈{i+1}")
        ax[1].plot(t, thdd_ref[:, i], "--", label=f"θ̈{i+1}_ref")
    ax[0].set_ylabel("θ̇ [rad/s]")
    ax[1].set_ylabel("θ̈ [rad/s²]")
    ax[1].set_xlabel("Time [s]")
    ax[0].set_title("Spool rate/accel tracking (ref from payout profile)")

    # Cable forces
    fig, ax = plt.subplots(6, 1, sharex=True)
    for i in range(6):
        ax[i].plot(t, Tmeas[:, i], label=f"T{i+1}_meas")
        ax[i].plot(t, Tdes[:, i], "--", label=f"T{i+1}_des")
        ax[i].axhline(0.0, linestyle="--")
        ax[i].legend(loc="upper right")
    ax[-1].set_xlabel("Time [s]")
    fig.suptitle("Cable forces (measured vs desired)")

    # Motor torques
    fig, ax = plt.subplots(6, 1, sharex=True)
    for i in range(6):
        ax[i].plot(t, tau[:, i], label=f"τ{i+1}_cmd")
        ax[i].plot(t, tau_ff[:, i], "--", label=f"τ{i+1}_ff (pre-sat)")
        ax[i].legend(loc="upper right")
    ax[-1].set_xlabel("Time [s]")
    fig.suptitle("Motor torques")

    plt.show()


def main():
    res = run_sim(
        sim_T=2.0,
        Tmin=5.0, Tmax=200.0,
        lam=1e-2,
        alpha_Tprev=0.7,
        torque_rate_limit=400.0,
    )
    plot_results(res)


if __name__ == "__main__":
    main()