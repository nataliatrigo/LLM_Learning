"""Configuration for the 1-D binary-belief reputation model with LOGIT demand.

All defaults are overridable from the CLI of run_all.py, e.g.

    python run_all.py --mu 0.6 --beta 16

Everything downstream (grid, tolerances, seeds) lives here so a run is fully
reproducible from this single file plus the CLI flags echoed in SUMMARY.md.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ---- product qualities (success probabilities), p1 < p2 in (0,1)
    p1: float = 0.3
    p2: float = 0.8

    # ---- seller economics (per TRANSACTION; nothing is paid or earned when
    #      the period's user does not engage)
    R: float = 1.0        # revenue per transaction
    c1: float = 0.05      # cost of the low product
    c2: float = 0.65      # cost of the high product
    gamma: float = 0.95   # discount factor

    # ---- logit demand (microfounded by heterogeneous outside options)
    mu: float = 0.55      # center of the outside-option distribution
    beta: float = 8.0     # homogeneity/precision of the outside options

    # ---- numerics
    pi_eps: float = 1e-4          # grid spans pi in [pi_eps, 1 - pi_eps]
    n_grid: int = 3000            # uniform grid in log-odds ell
    vi_tol: float = 1e-10         # sup-norm tolerance for value iteration
    vi_max_iter: int = 200_000    # do NOT stop early; residual is verified

    # ---- simulation (E5)
    sim_paths: int = 200
    sim_horizon: int = 500
    sim_starts: tuple = (0.5, 0.1)
    seed: int = 20260707

    # ---- derived quantities
    @property
    def dp(self) -> float:
        return self.p2 - self.p1

    @property
    def dc(self) -> float:
        return self.c2 - self.c1

    @property
    def pibar(self) -> float:
        """Greedy indifference belief: posterior mean quality equals mu."""
        return (self.mu - self.p1) / self.dp
