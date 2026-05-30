"""Pure cable-tension allocation and nullspace helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TensionAllocatorConfig:
    tension_min_n: float
    tension_max_n: float
    regularization_lambda: float
    iterations: int
    alpha_blend: float
    wrench_from_tension_sign: float = -1.0


def solve_tensions_least_squares(
    j_len_plat,
    tau_plat_des,
    t_prev,
    config: TensionAllocatorConfig,
):
    """
    Solve:
      min_T || (s*J^T)T - tau ||^2 + lambda ||T - Tref||^2
      s.t. Tmin <= T <= Tmax
    where s = config.wrench_from_tension_sign.
    """
    j_mat = np.asarray(j_len_plat, dtype=float)
    a_mat = float(config.wrench_from_tension_sign) * j_mat.T
    tau = np.asarray(tau_plat_des, dtype=float)
    nt = a_mat.shape[1]

    lb = np.full(nt, float(config.tension_min_n), dtype=float)
    ub = np.full(nt, float(config.tension_max_n), dtype=float)
    if t_prev is None:
        t_ref = lb.copy()
    else:
        t_ref = float(config.alpha_blend) * np.asarray(t_prev, dtype=float) + (1.0 - float(config.alpha_blend)) * lb

    t_cmd = t_ref.copy()
    ata = a_mat.T @ a_mat
    lipschitz = float(np.linalg.norm(ata, 2) + float(config.regularization_lambda))
    step = 1.0 / max(lipschitz, 1e-9)

    for _ in range(max(1, int(config.iterations))):
        grad = (
            2.0 * (a_mat.T @ (a_mat @ t_cmd - tau))
            + 2.0 * float(config.regularization_lambda) * (t_cmd - t_ref)
        )
        t_cmd = np.clip(t_cmd - step * grad, lb, ub)
    return t_cmd


def cable_tension_nullspace_basis(j_len_plat, n_prev=None):
    """
    Return a unit basis vector n in Null(J^T) for J shape (6,5).

    The sign is chosen for continuity against n_prev when available.
    """
    j_mat = np.asarray(j_len_plat, dtype=float)
    if j_mat.shape != (6, 5):
        raise ValueError(f"Expected J shape (6,5), got {j_mat.shape}")
    _, _, vh = np.linalg.svd(j_mat.T, full_matrices=True)
    n_vec = np.asarray(vh[-1, :], dtype=float)
    n_norm = float(np.linalg.norm(n_vec))
    if n_norm < 1e-9:
        raise ValueError("Degenerate cable nullspace basis")
    n_vec = n_vec / n_norm
    if n_prev is not None:
        n_prev = np.asarray(n_prev, dtype=float)
        if n_prev.shape == n_vec.shape and np.all(np.isfinite(n_prev)) and float(n_prev @ n_vec) < 0.0:
            n_vec = -n_vec
    return n_vec


def nullspace_sigma_interval(t_particular, n_vec, tension_floor_n):
    """
    Feasible interval for sigma in T = T_particular + n*sigma subject to T >= tension_floor_n.
    Returns (lower, upper, feasible).
    """
    t_particular = np.asarray(t_particular, dtype=float)
    n_vec = np.asarray(n_vec, dtype=float)
    lower = -float("inf")
    upper = float("inf")
    eps = 1e-9
    for ti, ni in zip(t_particular, n_vec):
        if ni > eps:
            lower = max(lower, (float(tension_floor_n) - float(ti)) / float(ni))
        elif ni < -eps:
            upper = min(upper, (float(tension_floor_n) - float(ti)) / float(ni))
        elif float(ti) < float(tension_floor_n):
            return lower, upper, False
    return lower, upper, lower <= upper


def clamp_sigma_to_interval(sigma_ref, lower, upper):
    sigma_ref = float(sigma_ref)
    if math.isfinite(lower):
        sigma_ref = max(sigma_ref, float(lower))
    if math.isfinite(upper):
        sigma_ref = min(sigma_ref, float(upper))
    return sigma_ref


def rate_limit_scalar(target, current, rate_limit_per_s, dt):
    target = float(target)
    current = float(current)
    rate_limit_per_s = abs(float(rate_limit_per_s))
    dt = max(0.0, float(dt))
    if not math.isfinite(target):
        return current
    if not math.isfinite(current) or dt <= 0.0 or rate_limit_per_s <= 0.0:
        return target
    max_step = rate_limit_per_s * dt
    delta = float(np.clip(target - current, -max_step, max_step))
    return current + delta


def platform_wrench_from_tensions(j_len_plat, tensions_n, wrench_from_tension_sign: float):
    j_mat = np.asarray(j_len_plat, dtype=float)
    tensions = np.asarray(tensions_n, dtype=float)
    return float(wrench_from_tension_sign) * (j_mat.T @ tensions)
