"""Configuration for the 1-D binary-belief ("hidden product choice") reputation model.

All model and numerical parameters live in one frozen dataclass so sweeps are
just `dataclasses.replace(cfg, ...)`. Derived quantities (dS, dF, pibar, ...)
are read-only properties, so they always stay consistent with the primitives.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # --- qualities (interior regime p1 < p0 < p2 is assumed throughout) ---
    p1: float = 0.3      # low product success probability
    p2: float = 0.8      # high product success probability
    p0: float = 0.5      # outside option success probability
    # --- seller economics ---
    R: float = 1.0       # revenue per engagement
    c1: float = 0.05     # cost of low product
    c2: float = 0.85      # cost of high product
    gamma: float = 0.95  # discount factor
    # --- user decision rules ---
    eps: float = 0.10    # epsilon-greedy exploration rate
    beta: float = 25.0   # slope of the optional smooth (LOGIT) demand
    # --- grid / value iteration ---
    pi_edge: float = 1e-3   # grid spans pi in [pi_edge, 1 - pi_edge]
    n_grid: int = 2000
    tol: float = 1e-9       # sup-norm stopping rule for value iteration
    max_iter: int = 500_000
    # --- Monte Carlo ---
    n_paths: int = 500
    n_periods: int = 2000
    mc_starts: tuple = (0.15, 0.35, 0.60, 0.85)  # initial beliefs pi_0
    seed: int = 12345
    # --- output directory, relative to BinaryBelief/ ---
    outdir: str = "outputs"

    def __post_init__(self):
        assert 0.0 < self.p1 < self.p0 < self.p2 < 1.0, (
            f"interior regime p1 < p0 < p2 required, got "
            f"p1={self.p1}, p0={self.p0}, p2={self.p2}"
        )
        assert 0.0 < self.gamma < 1.0
        assert self.c1 < self.c2

    # ------------- derived quantities -------------
    @property
    def dS(self) -> float:
        """Log-odds increment on a success: log(p2/p1) > 0."""
        return math.log(self.p2 / self.p1)

    @property
    def dF(self) -> float:
        """Log-odds increment on a failure: log((1-p2)/(1-p1)) < 0."""
        return math.log((1.0 - self.p2) / (1.0 - self.p1))

    @property
    def dc(self) -> float:
        return self.c2 - self.c1

    @property
    def dp(self) -> float:
        return self.p2 - self.p1

    @property
    def threshold(self) -> float:
        """The constant policy threshold dc/dp for the continuation gap."""
        return self.dc / self.dp

    @property
    def pibar(self) -> float:
        """Epsilon-greedy indifference cliff (p0 - p1)/(p2 - p1)."""
        return (self.p0 - self.p1) / (self.p2 - self.p1)

    @property
    def Lambda(self) -> float:
        """Product distinguishability dS - dF = log(p2(1-p1)/(p1(1-p2)))."""
        return self.dS - self.dF

    @property
    def ell_lo(self) -> float:
        return math.log(self.pi_edge / (1.0 - self.pi_edge))

    @property
    def ell_hi(self) -> float:
        return -self.ell_lo


# Configs run by run.py. Comparative-statics sweeps (figures 5-6) are skipped
# for gamma = 0.999 by default (slow value iteration); pass --sweeps-all to
# include them. The gamma = 0.70 config is an illustration of the interior-band
# regime: at the default economics with gamma = 0.95 the product-2 band covers
# nearly the whole belief space (reputation insurance is cheap), so beliefs
# park at the upper band edge; with moderate discounting the band is interior,
# the EG band hugs the cliff pibar, and the "parking" phenomenology is visible.
CONFIGS = (
    Config(),
    Config(gamma=0.999, outdir="outputs_gamma_0999"),
    Config(gamma=0.70, outdir="outputs_gamma_070"),
    Config(gamma=0.80, outdir="outputs_gamma_080"),
    Config(gamma=0.85, outdir="outputs_gamma_085"),
    Config(gamma=0.90, outdir="outputs_gamma_090")
)
