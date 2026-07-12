from __future__ import annotations

import sys
import time
import gc
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discounted.DP.exact_dp import (  # noqa: E402
    ModelParams, beta_ccdf, required_horizon, solve_discounted_finite_horizon,
)
from discounted.regime_experiments.configs import ExperimentConfig  # noqa: E402


def model_params(cfg: ExperimentConfig) -> ModelParams:
    return ModelParams(cfg.p1, cfg.p2, cfg.c1, cfg.c2, cfg.revenue, cfg.gamma)


def solve_snapshot(cfg: ExperimentConfig, outer: int, tail_tolerance: float) -> tuple[dict, pd.DataFrame]:
    tail = required_horizon(cfg.gamma, tail_tolerance)
    analysis_period = outer + 1
    horizon = analysis_period + tail
    sol = solve_discounted_finite_horizon(
        cfg.p0, model_params(cfg), horizon,
        stored_policy_periods={analysis_period}, snapshot_periods={analysis_period},
    )
    snap = sol["snapshots"][analysis_period]
    n = snap["S"] + snap["F"]
    m = (snap["S"] + 1.0) / (n + 2.0)
    z = (snap["S"] + 1.0 - cfg.p0 * (n + 2.0)) / np.sqrt(
        (n + 2.0) * cfg.p0 * (1.0 - cfg.p0)
    )
    G = snap["discounted_continuation_gap"]
    advantage = -(cfg.c2 - cfg.c1) + (cfg.p2 - cfg.p1) * G
    scale = np.maximum(1.0, np.maximum(np.abs(cfg.c2 - cfg.c1), np.abs((cfg.p2-cfg.p1)*G)))
    tie_tol = 1e-9 * scale
    classification = np.where(advantage > tie_tol, "robust_product2",
                              np.where(advantage < -tie_tol, "robust_product1", "tied"))
    frame = pd.DataFrame({
        "parameter_id": cfg.parameter_id, "S": snap["S"], "F": snap["F"], "n": n,
        "m": m, "z": z, "D": snap["rho"], "value": snap["value"], "G": G,
        "action_advantage": advantage, "optimal_product": np.where(advantage >= 0, 2, 1),
        "classification": classification,
    })
    return sol, frame


def bellman_residual(frame: pd.DataFrame, cfg: ExperimentConfig) -> float:
    by = {(int(r.S), int(r.F)): float(r.value) for r in frame.itertuples()}
    residual = 0.0
    for r in frame[frame.n < frame.n.max()].itertuples():
        vs, vf = by[(r.S + 1, r.F)], by[(r.S, r.F + 1)]
        q1 = cfg.revenue-cfg.c1 + cfg.gamma*(cfg.p1*vs+(1-cfg.p1)*vf)
        q2 = cfg.revenue-cfg.c2 + cfg.gamma*(cfg.p2*vs+(1-cfg.p2)*vf)
        rhs = cfg.gamma*(1-r.D)*r.value + r.D*max(q1, q2)
        residual = max(residual, abs(r.value-rhs))
    return residual


def solve_with_convergence(cfg: ExperimentConfig, smoke: bool = False) -> dict:
    start = time.perf_counter()
    outer = 60 if smoke else (140 if cfg.gamma < 0.99 else 180)
    reliable = 35 if smoke else 90
    tolerances = [3e-5, 1e-5] if smoke else [1e-6, 1e-7]
    sol0, f0 = solve_snapshot(cfg, outer, tolerances[0])
    del sol0
    gc.collect()
    sol1, f1 = solve_snapshot(cfg, outer + (20 if smoke else 40), tolerances[1])
    common0 = f0[f0.n <= reliable].reset_index(drop=True)
    common1 = f1[f1.n <= reliable].reset_index(drop=True)
    policy_change = float(np.mean(common0.optimal_product.to_numpy() != common1.optimal_product.to_numpy()))
    max_g_change = float(np.max(np.abs(common0.G.to_numpy()-common1.G.to_numpy())))
    converged = policy_change == 0.0 and max_g_change <= 2e-5
    f1["reliable_interior"] = f1.n <= reliable
    del sol1, common0, common1
    gc.collect()
    return {
        "config": cfg, "states": f1, "outer": outer + (20 if smoke else 40),
        "reliable": reliable, "remaining_tail": required_horizon(cfg.gamma, tolerances[1]),
        "policy_change_fraction": policy_change, "max_G_change": max_g_change,
        "converged": converged, "bellman_residual": bellman_residual(f1, cfg),
        "runtime": time.perf_counter()-start,
    }


def solve_product1_value(cfg: ExperimentConfig, full: pd.DataFrame) -> float:
    nmax = max(int(full.n.max()), required_horizon(cfg.gamma, 1e-7))
    nxt = np.zeros(nmax + 2)
    for n in range(nmax, -1, -1):
        S = np.arange(n+1); F = n-S; D = beta_ccdf(cfg.p0, S, F)
        q = cfg.revenue-cfg.c1 + cfg.gamma*(cfg.p1*nxt[1:n+2]+(1-cfg.p1)*nxt[:n+1])
        nxt = D*q/(1-cfg.gamma*(1-D))
    return float(nxt[0])


def config_dict(cfg: ExperimentConfig) -> dict:
    row = asdict(cfg)
    row.update({"regime": cfg.regime, "p1_minus_p0": cfg.p1-cfg.p0,
                "p2_minus_p0": cfg.p2-cfg.p0, "Delta_p": cfg.p2-cfg.p1,
                "Delta_c": cfg.c2-cfg.c1})
    return row
