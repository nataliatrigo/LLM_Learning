"""Core model: demand rules, Bellman solver, policy/band extraction, Monte Carlo,
and the fluid (observation-time) ODE.

State: user log-odds belief ell = log(pi/(1-pi)), pi = P(product is the high one).
Bayes updates are additive in ell with constant increments dS (success), dF (failure).

Seller Bellman, in the fixed-point form used for value iteration:
    M(ell) = max_x { R - c_x + gamma * [ p_x V(ell+dS) + (1-p_x) V(ell+dF) ] }
    V(ell) = D(ell) / (1 - gamma * (1 - D(ell))) * M(ell)
Optimal product x*(ell) = 2 iff g(ell) := gamma*[V(ell+dS) - V(ell+dF)] >= dc/dp.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ------------------------------------------------------------------ basics

def sigmoid(ell):
    return 1.0 / (1.0 + np.exp(-np.asarray(ell, dtype=float)))


def logit(pi):
    pi = np.asarray(pi, dtype=float)
    return np.log(pi / (1.0 - pi))


def make_grid(cfg) -> np.ndarray:
    """Uniform grid in ell over [logit(pi_edge), logit(1-pi_edge)]."""
    return np.linspace(cfg.ell_lo, cfg.ell_hi, cfg.n_grid)


def demand(cfg, rule: str, ell):
    """P(user engages with the seller | belief ell), for a given decision rule."""
    pi = sigmoid(ell)
    if rule == "TS":       # Thompson sampling: engage w.p. pi (interior regime)
        return pi
    if rule == "EG":       # epsilon-greedy: cliff at pibar = (p0-p1)/(p2-p1)
        return cfg.eps + (1.0 - cfg.eps) * (pi >= cfg.pibar).astype(float)
    if rule == "LOGIT":    # smooth robustness check
        return sigmoid(cfg.beta * (cfg.p1 + pi * cfg.dp - cfg.p0))
    raise ValueError(f"unknown rule {rule!r}")


def _shift_indices(ell: np.ndarray, delta: float):
    """Linear-interpolation indices/weights for evaluating V(ell + delta) on the
    grid, clamping to the boundary value when the shifted point leaves the grid."""
    target = np.clip(ell + delta, ell[0], ell[-1])
    h = ell[1] - ell[0]
    pos = (target - ell[0]) / h
    j = np.clip(np.floor(pos).astype(int), 0, len(ell) - 2)
    w = np.clip(pos - j, 0.0, 1.0)
    return j, w


def _interp(V: np.ndarray, j: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (1.0 - w) * V[j] + w * V[j + 1]


# ------------------------------------------------------------------ DP solve

@dataclass
class SolveResult:
    rule: str
    ell: np.ndarray
    pi: np.ndarray
    V: np.ndarray
    g: np.ndarray            # continuation gap gamma*(V(ell+dS) - V(ell+dF))
    x: np.ndarray            # optimal product in {1, 2}
    D: np.ndarray            # demand on the grid
    iterations: int
    final_diff: float
    converged: bool
    policy_consistent: bool  # argmax policy == threshold rule g >= dc/dp


def solve_dp(cfg, rule: str, V0: np.ndarray | None = None) -> SolveResult:
    """Value iteration on the fixed-point form; gamma-contraction, sup-norm tol."""
    ell = make_grid(cfg)
    D = np.asarray(demand(cfg, rule, ell), dtype=float)
    jS, wS = _shift_indices(ell, cfg.dS)
    jF, wF = _shift_indices(ell, cfg.dF)
    prefactor = D / (1.0 - cfg.gamma * (1.0 - D))

    gam, R, p1, p2, c1, c2 = cfg.gamma, cfg.R, cfg.p1, cfg.p2, cfg.c1, cfg.c2
    V = np.zeros_like(ell) if V0 is None else np.array(V0, dtype=float)
    diff = np.inf
    for it in range(1, cfg.max_iter + 1):
        VS = _interp(V, jS, wS)
        VF = _interp(V, jF, wF)
        M1 = R - c1 + gam * (p1 * VS + (1.0 - p1) * VF)
        M2 = R - c2 + gam * (p2 * VS + (1.0 - p2) * VF)
        V_new = prefactor * np.maximum(M1, M2)
        diff = float(np.max(np.abs(V_new - V)))
        V = V_new
        if diff < cfg.tol:
            break
    converged = diff < cfg.tol

    # policy and continuation gap at the fixed point
    VS = _interp(V, jS, wS)
    VF = _interp(V, jF, wF)
    M1 = R - c1 + gam * (p1 * VS + (1.0 - p1) * VF)
    M2 = R - c2 + gam * (p2 * VS + (1.0 - p2) * VF)
    g = gam * (VS - VF)
    x = np.where(g >= cfg.threshold, 2, 1)
    x_argmax = np.where(M2 >= M1, 2, 1)
    # the two rules are algebraically identical (M2 - M1 = dp*(g - dc/dp));
    # allow float-rounding flips only at exact ties
    mismatch = (x != x_argmax) & (np.abs(g - cfg.threshold) > 1e-9)
    policy_consistent = not bool(np.any(mismatch))

    return SolveResult(rule, ell, sigmoid(ell), V, g, x, D,
                       it, diff, converged, policy_consistent)


# ------------------------------------------------------------------ band

@dataclass
class Band:
    pi_lo: float
    pi_hi: float
    ell_lo: float
    ell_hi: float
    is_interval: bool
    n_components: int
    empty: bool


def extract_band(cfg, res: SolveResult) -> Band:
    """The pi-interval where product 2 is optimal. Edges are refined by linear
    interpolation of the crossing g(ell) = dc/dp; flags non-interval regions."""
    mask = res.x == 2
    if not mask.any():
        return Band(np.nan, np.nan, np.nan, np.nan, True, 0, True)
    idx = np.flatnonzero(mask)
    n_components = 1 + int(np.sum(np.diff(idx) > 1))
    ell, h = res.ell, res.g - cfg.threshold

    def crossing(a: int, b: int) -> float:
        denom = h[b] - h[a]
        if denom == 0.0:
            return float(ell[a])
        return float(ell[a] - h[a] * (ell[b] - ell[a]) / denom)

    i0, i1 = int(idx[0]), int(idx[-1])
    ell_lo = float(ell[0]) if i0 == 0 else crossing(i0 - 1, i0)
    ell_hi = float(ell[-1]) if i1 == len(ell) - 1 else crossing(i1, i1 + 1)
    return Band(float(sigmoid(ell_lo)), float(sigmoid(ell_hi)),
                ell_lo, ell_hi, n_components == 1, n_components, False)


# ------------------------------------------------------------------ Monte Carlo

@dataclass
class MCResult:
    pi_paths: np.ndarray   # (n_periods + 1, n_total) belief paths
    start_id: np.ndarray   # (n_total,) index into starts
    starts: tuple
    engaged_frac: float    # overall fraction of periods with engagement


def simulate(cfg, rule: str, res: SolveResult, seed_offset: int = 0) -> MCResult:
    """Simulate the belief process under the optimal policy x*(.).

    Each period: engage ~ Bernoulli(D(ell)); if engaged, outcome ~ Bernoulli(p_{x*}),
    ell += dS or dF (clamped to the grid range); if not engaged, ell unchanged.
    """
    rng = np.random.default_rng(cfg.seed + seed_offset)
    n_starts = len(cfg.mc_starts)
    per_start = max(1, cfg.n_paths // n_starts)
    n = per_start * n_starts
    start_id = np.repeat(np.arange(n_starts), per_start)
    ell = np.asarray(logit(np.array(cfg.mc_starts)), dtype=float)[start_id].copy()

    grid = res.ell
    hg = grid[1] - grid[0]
    paths = np.empty((cfg.n_periods + 1, n))
    paths[0] = sigmoid(ell)
    engaged_total = 0
    for t in range(cfg.n_periods):
        D = np.asarray(demand(cfg, rule, ell), dtype=float)
        engage = rng.random(n) < D
        j = np.clip(np.rint((ell - grid[0]) / hg).astype(int), 0, len(grid) - 1)
        p_x = np.where(res.x[j] == 2, cfg.p2, cfg.p1)
        success = rng.random(n) < p_x
        step = np.where(success, cfg.dS, cfg.dF)
        ell = np.where(engage, np.clip(ell + step, grid[0], grid[-1]), ell)
        engaged_total += int(engage.sum())
        paths[t + 1] = sigmoid(ell)
    return MCResult(paths, start_id, cfg.mc_starts,
                    engaged_total / (n * cfg.n_periods))


def stationary_sample(mc: MCResult, burn_frac: float = 0.75, thin: int = 10):
    """Pooled draws from the tail of the paths, as a stationary-distribution proxy."""
    t0 = int(burn_frac * (mc.pi_paths.shape[0] - 1))
    return mc.pi_paths[t0::thin].ravel()


# ------------------------------------------------------------------ fluid check

def fluid_trajectories(cfg, res: SolveResult, starts, n_max: float = 80.0,
                       dn: float = 0.02):
    """Integrate the observation-time ODE  d(ell)/dn = mu1 + a*(ell) * kappa,
    with a*(ell) = 1{x*(ell) = 2} the relaxed control extracted from the DP policy,
    mu1 = p1*dS + (1-p1)*dF (drift of ell per engagement under product 1, < 0) and
    kappa = dp * Lambda.  Note: the drift under product 2 is mu1 + kappa
    = p2*dS + (1-p2)*dF > 0, so mu1 enters with a plus sign here (the extra minus
    in the prompt's mu0 = -(p1*dS + (1-p1)*dF) would flip the product-1 drift
    positive, which contradicts Bayes learning under the low product).
    """
    mu1 = cfg.p1 * cfg.dS + (1.0 - cfg.p1) * cfg.dF
    kappa = cfg.dp * cfg.Lambda
    grid = res.ell
    hg = grid[1] - grid[0]
    steps = int(round(n_max / dn))
    ell = np.asarray(logit(np.array(starts, dtype=float)), dtype=float)
    out = np.empty((steps + 1, len(ell)))
    out[0] = sigmoid(ell)
    for k in range(steps):
        j = np.clip(np.rint((ell - grid[0]) / hg).astype(int), 0, len(grid) - 1)
        a = (res.x[j] == 2).astype(float)
        ell = np.clip(ell + dn * (mu1 + a * kappa), grid[0], grid[-1])
        out[k + 1] = sigmoid(ell)
    return np.linspace(0.0, n_max, steps + 1), out
