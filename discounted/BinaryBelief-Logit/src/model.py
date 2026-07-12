"""Core model: 1-D binary-belief reputation DP with logit demand and
stochastic per-period engagement (belief frozen when no transaction occurs).

STATE. The period's user holds the public belief pi = P(A delivers the high
product p2 rather than p1). Bayes updating on a realized binary outcome is
additive in log-odds ell = log(pi / (1 - pi)):

    success:  ell -> ell + dS,   dS = log(p2 / p1)              > 0
    failure:  ell -> ell + dF,   dF = log((1 - p2) / (1 - p1))  < 0

so ell (equivalently pi = sigmoid(ell)) is the entire state: 1-D, stationary.
If no transaction occurs there is no outcome and the belief stays FROZEN.

DEMAND (microfoundation). Each period's user is greedy: she engages iff the
posterior mean quality p1 + pi * dp (dp = p2 - p1) exceeds her personal
outside option p0_i. Users are heterogeneous, p0_i ~ Logistic(mu, 1/beta),
iid across periods, so the probability that the period's user engages is

    D(pi) = P(p0_i <= p1 + pi*dp) = sigmoid( beta * (p1 + pi*dp - mu) ).

mu centers the outside-option distribution (it plays the role of the single
outside option p0 of the greedy model); beta is the homogeneity/precision.
beta -> infinity recovers the greedy step at pibar = (mu - p1)/dp; beta -> 0
flattens demand toward 1/2. For finite beta, 0 < D < 1 everywhere — no
absorbing regions, but learning is SLOW where D is small (the frozen branch
fires often).

SELLER. Revenue R and cost c_x are paid ONLY when a transaction occurs. If
the user engages, the seller secretly chooses x in {1,2}; the outcome is a
success w.p. p_x and the user updates. Otherwise nothing happens and time
discounts. Bellman equation:

    V(ell) = gamma*(1 - D(ell))*V(ell)
           + D(ell) * max_x { R - c_x
                              + gamma*[ p_x V(ell+dS) + (1-p_x) V(ell+dF) ] }

Value iteration uses the equivalent fixed-point form (solving the frozen
branch in closed form each sweep):

    M(ell) = max_x { R - c_x + gamma*[ p_x V(ell+dS) + (1-p_x) V(ell+dF) ] }
    V(ell) = [ D(ell) / (1 - gamma*(1 - D(ell))) ] * M(ell)

OPTIMAL POLICY. Comparing x = 2 against x = 1 inside M (both actions share
the same two successor states, only the mixing weights differ):

    x*(ell) = 2  iff  g(ell) >= dc/dp,
    g(ell)  = gamma * [ V(ell+dS) - V(ell+dF) ]   (reputational premium).

The threshold dc/dp is a CONSTANT. The investment set is {ell : g >= dc/dp}.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from config import Config


# ---------------------------------------------------------------- basics
def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic function."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-np.abs(x))),
                    np.exp(-np.abs(x)) / (1.0 + np.exp(-np.abs(x))))


def logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def demand(pi: np.ndarray | float, cfg: Config, mode: str = "logit"):
    """Per-period engagement probability at belief pi.

    mode='logit'  : the microfounded logistic demand (see module docstring).
    mode='ts'     : D(pi) = pi (Thompson-sampling benchmark, E6).
    mode='greedy' : hard step at pibar (the beta -> infinity limit, E3).
    """
    if mode == "ts":
        return np.asarray(pi, dtype=float)
    if mode == "greedy":
        return (cfg.p1 + np.asarray(pi) * cfg.dp >= cfg.mu).astype(float)
    return sigmoid(cfg.beta * (cfg.p1 + np.asarray(pi) * cfg.dp - cfg.mu))


# ---------------------------------------------------------------- results
@dataclass
class Band:
    """Investment set {x* = 2} reported as intervals in pi-space."""
    intervals: list          # list of (pi_lo, pi_hi), refined by root-finding
    single_interval: bool    # True also when empty (vacuously a band)
    empty: bool

    @property
    def lo(self) -> float:
        return self.intervals[0][0] if self.intervals else np.nan

    @property
    def hi(self) -> float:
        return self.intervals[-1][1] if self.intervals else np.nan

    @property
    def width(self) -> float:
        return 0.0 if self.empty else sum(b - a for a, b in self.intervals)

    @property
    def center(self) -> float:
        return np.nan if self.empty else 0.5 * (self.lo + self.hi)


@dataclass
class SolveResult:
    ell: np.ndarray
    pi: np.ndarray
    V: np.ndarray
    policy: np.ndarray       # 1 or 2 per grid point
    g: np.ndarray            # reputational premium gamma*[V(ell+dS)-V(ell+dF)]
    threshold: float         # dc/dp, constant
    D: np.ndarray
    band: Band
    n_iter: int
    sup_err: float
    converged: bool
    dS: float
    dF: float


# ---------------------------------------------------------------- solver
def _interp_weights(grid: np.ndarray, points: np.ndarray):
    """Linear-interpolation indices/weights on a uniform grid, clamped to the
    grid ends (justified: V is essentially flat there)."""
    h = grid[1] - grid[0]
    t = (points - grid[0]) / h
    i0 = np.clip(np.floor(t).astype(np.int64), 0, len(grid) - 2)
    w = np.clip(t - i0, 0.0, 1.0)
    return i0, w


def solve_dp(cfg: Config, mode: str = "logit") -> SolveResult:
    """Value iteration on the fixed-point form, to sup-norm tol cfg.vi_tol.

    Where D is small the effective per-iteration contraction is weak, so we
    never stop early: `converged` records whether the residual actually
    reached tolerance and callers should treat False as an error."""
    lo, hi = logit(cfg.pi_eps), logit(1.0 - cfg.pi_eps)
    ell = np.linspace(lo, hi, cfg.n_grid)
    pi = sigmoid(ell)

    dS = np.log(cfg.p2 / cfg.p1)
    dF = np.log((1.0 - cfg.p2) / (1.0 - cfg.p1))
    iS, wS = _interp_weights(ell, ell + dS)
    iF, wF = _interp_weights(ell, ell + dF)

    D = demand(pi, cfg, mode=mode)
    # closed-form resolution of the frozen branch:  V = frozen_factor * M
    frozen_factor = D / (1.0 - cfg.gamma * (1.0 - D))

    V = np.zeros(cfg.n_grid)
    sup_err = np.inf
    for it in range(1, cfg.vi_max_iter + 1):
        VS = V[iS] * (1.0 - wS) + V[iS + 1] * wS
        VF = V[iF] * (1.0 - wF) + V[iF + 1] * wF
        M1 = cfg.R - cfg.c1 + cfg.gamma * (cfg.p1 * VS + (1.0 - cfg.p1) * VF)
        M2 = cfg.R - cfg.c2 + cfg.gamma * (cfg.p2 * VS + (1.0 - cfg.p2) * VF)
        V_new = frozen_factor * np.maximum(M1, M2)
        sup_err = float(np.max(np.abs(V_new - V)))
        V = V_new
        if sup_err < cfg.vi_tol:
            break
    converged = sup_err < cfg.vi_tol

    VS = V[iS] * (1.0 - wS) + V[iS + 1] * wS
    VF = V[iF] * (1.0 - wF) + V[iF + 1] * wF
    g = cfg.gamma * (VS - VF)
    threshold = cfg.dc / cfg.dp
    policy = np.where(g >= threshold, 2, 1)
    band = extract_band(pi, policy, g - threshold)

    return SolveResult(ell=ell, pi=pi, V=V, policy=policy, g=g,
                       threshold=threshold, D=D, band=band,
                       n_iter=it, sup_err=sup_err, converged=converged,
                       dS=dS, dF=dF)


def extract_band(pi: np.ndarray, policy: np.ndarray, f: np.ndarray) -> Band:
    """Investment intervals in pi-space, endpoints refined by the linear root
    of f = g - dc/dp between the two grid points straddling the switch."""
    inv = np.flatnonzero(policy == 2)
    if inv.size == 0:
        return Band(intervals=[], single_interval=True, empty=True)

    # split contiguous runs of investing indices
    cuts = np.flatnonzero(np.diff(inv) > 1)
    starts = np.concatenate(([inv[0]], inv[cuts + 1]))
    ends = np.concatenate((inv[cuts], [inv[-1]]))

    def _root(i_out: int, i_in: int) -> float:
        """Linear root of f between a non-investing and an investing point."""
        f0, f1 = f[i_out], f[i_in]
        if f1 == f0:
            return float(pi[i_in])
        t = -f0 / (f1 - f0)
        return float(pi[i_out] + t * (pi[i_in] - pi[i_out]))

    intervals = []
    for s, e in zip(starts, ends):
        a = _root(s - 1, s) if s > 0 else float(pi[0])
        b = _root(e + 1, e) if e < len(pi) - 1 else float(pi[-1])
        intervals.append((a, b))
    return Band(intervals=intervals, single_interval=len(intervals) == 1,
                empty=False)


# ---------------------------------------------------------------- simulation
@dataclass
class SimResult:
    pi0: float
    pi_paths: np.ndarray        # (n_paths, T+1) beliefs
    engaged: np.ndarray         # (n_paths, T) bool, transaction in period t
    frac_in_band: float         # fraction of (path, t) pairs inside the band
    frac_in_band_t: np.ndarray  # per-period fraction across paths
    escape_t: np.ndarray        # per path: first t with pi_t >= band lower
    #                             endpoint (nan if never; 0 if starts inside)


def simulate(res: SolveResult, cfg: Config, pi0: float,
             rng: np.random.Generator) -> SimResult:
    """Simulate the TRUE mechanism under the optimal policy.

    Each period: the user engages w.p. D(pi_t). If she does not engage the
    belief is frozen (pi_{t+1} = pi_t). If she engages, the seller plays
    x*(ell_t) (nearest-grid lookup), the outcome is a success w.p. p_x, and
    the belief moves by dS or dF (clamped to the grid range)."""
    n, T = cfg.sim_paths, cfg.sim_horizon
    lo, hi = res.ell[0], res.ell[-1]
    h = res.ell[1] - res.ell[0]

    ell = np.full(n, logit(pi0))
    paths = np.empty((n, T + 1))
    engaged = np.empty((n, T), dtype=bool)
    paths[:, 0] = sigmoid(ell)
    for t in range(T):
        pi_t = sigmoid(ell)
        engage = rng.random(n) < demand(pi_t, cfg)
        idx = np.clip(np.rint((ell - lo) / h).astype(np.int64),
                      0, len(res.ell) - 1)
        p_succ = np.where(res.policy[idx] == 2, cfg.p2, cfg.p1)
        success = rng.random(n) < p_succ
        step = np.where(success, res.dS, res.dF)
        ell = np.where(engage, np.clip(ell + step, lo, hi), ell)
        engaged[:, t] = engage
        paths[:, t + 1] = sigmoid(ell)

    in_band = np.zeros_like(paths, dtype=bool)
    for a, b in res.band.intervals:
        in_band |= (paths >= a) & (paths <= b)

    # escape time from the slow-learning trap: first period the belief
    # reaches the band's lower endpoint (nan if it never does within T)
    if res.band.empty:
        escape = np.full(n, np.nan)
    else:
        above = paths >= res.band.lo
        first = above.argmax(axis=1).astype(float)
        first[~above.any(axis=1)] = np.nan
        escape = first
    return SimResult(pi0=pi0, pi_paths=paths, engaged=engaged,
                     frac_in_band=float(in_band.mean()),
                     frac_in_band_t=in_band.mean(axis=0),
                     escape_t=escape)
