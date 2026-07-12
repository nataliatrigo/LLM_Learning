from __future__ import annotations

import numpy as np
import pandas as pd

from discounted.regime_experiments.configs import ExperimentConfig


def discounted_occupancy(states: pd.DataFrame, cfg: ExperimentConfig) -> tuple[float, float]:
    nmax = int(states.n.max())
    HA = np.zeros(nmax+2); H2 = np.zeros(nmax+2)
    for n in range(nmax, -1, -1):
        layer = states[states.n == n].sort_values("S")
        D = layer.D.to_numpy(); act = layer.optimal_product.to_numpy()
        p = np.where(act == 2, cfg.p2, cfg.p1)
        alpha = D/(1-cfg.gamma*(1-D))
        beta = cfg.gamma*D/(1-cfg.gamma*(1-D))
        HA = alpha + beta*(p*HA[1:n+2]+(1-p)*HA[:n+1])
        H2 = alpha*(act == 2) + beta*(p*H2[1:n+2]+(1-p)*H2[:n+1])
    return float(HA[0]), float(H2[0])


def undiscounted_interventions(states: pd.DataFrame, cfg: ExperimentConfig, last_active: int) -> float:
    if last_active < 0:
        return 0.0
    J = np.zeros(last_active+2)
    for n in range(last_active, -1, -1):
        layer = states[states.n == n].sort_values("S")
        act = layer.optimal_product.to_numpy(); p = np.where(act == 2, cfg.p2, cfg.p1)
        J = (act == 2) + p*J[1:n+2]+(1-p)*J[:n+1]
    return float(J[0])


def path_probabilities(states: pd.DataFrame, cfg: ExperimentConfig) -> tuple[float, float | None]:
    maxn = int(states.n.max())
    action = {(int(r.S), int(r.F)): int(r.optimal_product) for r in states.itertuples()}

    def hit_probability(kind: str) -> float:
        mass = {(0, 0): 1.0}; hit_total = 0.0
        for n in range(maxn+1):
            nxt: dict[tuple[int, int], float] = {}
            for (S, F), q in mass.items():
                m = (S+1)/(n+2)
                hit = action[(S, F)] == 2 if kind == "invest" else (
                    (cfg.p0 > .5 and m >= cfg.p0) or (cfg.p0 < .5 and m <= cfg.p0)
                )
                if hit:
                    hit_total += q
                    continue
                p = cfg.p2 if action[(S, F)] == 2 else cfg.p1
                nxt[(S+1,F)] = nxt.get((S+1,F),0)+q*p
                nxt[(S,F+1)] = nxt.get((S,F+1),0)+q*(1-p)
            mass = nxt
        return min(1.0, hit_total)

    return hit_probability("invest"), (None if cfg.p0 == .5 else hit_probability("cross"))
