from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import betaincc

from discounted.regime_experiments.configs import ExperimentConfig


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    ix = np.flatnonzero(mask)
    if not len(ix):
        return []
    cuts = np.flatnonzero(np.diff(ix) > 1)
    starts, ends = np.r_[0, cuts+1], np.r_[cuts, len(ix)-1]
    return [(int(ix[a]), int(ix[b])) for a, b in zip(starts, ends)]


def embedded_reach(states: pd.DataFrame, cfg: ExperimentConfig) -> pd.DataFrame:
    out = states.copy()
    out["embedded_reach_probability"] = 0.0
    index = {(int(r.S), int(r.F)): i for i, r in enumerate(out.itertuples())}
    out.loc[index[(0, 0)], "embedded_reach_probability"] = 1.0
    for n in range(int(out.n.max())):
        for r in out[out.n == n].itertuples():
            mass = out.at[r.Index, "embedded_reach_probability"]
            p = cfg.p2 if r.optimal_product == 2 else cfg.p1
            out.at[index[(r.S+1, r.F)], "embedded_reach_probability"] += mass*p
            out.at[index[(r.S, r.F+1)], "embedded_reach_probability"] += mass*(1-p)
    return out


def diagonal_diagnostics(states: pd.DataFrame, cfg: ExperimentConfig, converged: bool) -> pd.DataFrame:
    rows = []
    for n, layer in states[states.reliable_interior].groupby("n", sort=True):
        layer = layer.sort_values("S")
        robust2 = layer.classification.eq("robust_product2").to_numpy()
        tied = layer.classification.eq("tied").to_numpy()
        rr = runs(robust2)
        confirmed_components = len(rr)
        # Tied states between robust-2 blocks do not confirm a violation.
        if len(rr) > 1 and all(tied[a+1:b].any() for (_, a), (b, _) in zip(rr, rr[1:])):
            confirmed_components = 1
        active = layer.optimal_product.eq(2)
        invest = layer[active]
        lower = float(invest.m.min()) if len(invest) else np.nan
        upper = float(invest.m.max()) if len(invest) else np.nan
        width = upper-lower if len(invest) else np.nan
        rows.append({
            "parameter_id": cfg.parameter_id, "n": int(n), "active": bool(len(invest)),
            "product2_components": confirmed_components,
            "interval_property": confirmed_components <= 1,
            "lower_m_boundary": lower, "upper_m_boundary": upper,
            "center": (lower+upper)/2 if len(invest) else np.nan, "width": width,
            "center_minus_p0": (lower+upper)/2-cfg.p0 if len(invest) else np.nan,
            "reach_probability_in_product2_states": float(invest.embedded_reach_probability.sum()),
            "minimum_abs_action_advantage": float(layer.action_advantage.abs().min()),
            "convergence_flag": converged,
        })
    return pd.DataFrame(rows)


def active_summary(diag: pd.DataFrame) -> dict:
    active = diag.loc[diag.active, "n"].astype(int).to_list()
    gaps = []
    if active:
        aset = set(active)
        gaps = [n for n in range(min(active), max(active)+1) if n not in aset]
    return {"last_active_diagonal": max(active, default=-1), "active_diagonals": len(active),
            "active_diagonal_gaps": len(gaps), "reappearance": bool(gaps),
            "max_components": int(diag.product2_components.max()),
            "max_width": float(diag.width.max()) if diag.width.notna().any() else 0.0}


def drift_identity_check(cfg: ExperimentConfig) -> float:
    errors = []
    for S, F in [(0, 0), (2, 3), (20, 10)]:
        n = S+F; z = S+1-cfg.p0*(n+2)
        for p in (cfg.p1, cfg.p2):
            zs = S+2-cfg.p0*(n+3); zf = S+1-cfg.p0*(n+3)
            observation_drift = p*zs+(1-p)*zf-z
            errors.append(abs(observation_drift-(p-cfg.p0)))
            demand = float(betaincc(S+1, F+1, cfg.p0))
            errors.append(abs(demand*observation_drift-demand*(p-cfg.p0)))
    return max(errors)
