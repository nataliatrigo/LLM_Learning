"""Monotone triangular-grid approximation of the no-forgetting fluid HJB.

Only the spatial domain is truncated. Both solves retain the infinite-horizon
discount and bracket the *grid* value using lower/upper continuation values.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np
from scipy.special import betaincc


@dataclass(frozen=True)
class Parameters:
    p0: float = 0.50
    p1: float = 0.35
    p2: float = 0.80
    c1: float = 0.05
    c2: float = 0.65
    revenue: float = 1.0
    gamma: float = 0.98

    def __post_init__(self):
        if not all(math.isfinite(float(v)) for v in vars(self).values()):
            raise ValueError("All parameters must be finite")
        if not 0 < self.p0 < 1:
            raise ValueError("p0 must lie in (0,1)")
        if not 0 <= self.p1 < self.p2 <= 1:
            raise ValueError("Require 0 <= p1 < p2 <= 1")
        if not 0 <= self.c1 <= self.c2 <= self.revenue:
            raise ValueError("Require 0 <= c1 <= c2 <= revenue (nonnegative margins)")
        if not 0 < self.gamma < 1:
            raise ValueError("gamma must lie in (0,1), so delta=-log(gamma)>0")

    @property
    def delta(self):
        return -math.log(self.gamma)

    @property
    def delta_p(self):
        return self.p2 - self.p1

    @property
    def delta_c(self):
        return self.c2 - self.c1

    @property
    def max_reward(self):
        return self.revenue - self.c1


def demand(S, F, params: Parameters):
    """Beta posterior tail, extended to nonnegative real fluid counts."""
    s, f = np.broadcast_arrays(np.asarray(S, dtype=float), np.asarray(F, dtype=float))
    if not (np.isfinite(s).all() and np.isfinite(f).all()) or np.any(s < 0) or np.any(f < 0):
        raise ValueError("S and F must be finite and nonnegative")
    result = np.clip(betaincc(s + 1.0, f + 1.0, params.p0), 0.0, 1.0)
    return float(result) if result.ndim == 0 else result


def _grid_steps(value: float, h: float, name: str) -> int:
    if not math.isfinite(float(value)) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    steps = round(value / h)
    if not math.isclose(steps * h, value, abs_tol=1e-9, rel_tol=1e-10):
        raise ValueError(f"{name} must be an integer multiple of h")
    return int(steps)


@dataclass
class Solution:
    params: Parameters
    h: float
    cutoff_n: float
    report_n: float
    values: list[np.ndarray]
    upper_values: list[np.ndarray]
    advantages: list[np.ndarray]
    advantage_lower: list[np.ndarray]
    advantage_upper: list[np.ndarray]
    actions: list[np.ndarray]
    demand_values: list[np.ndarray]
    residual: float
    value_error_bound: float
    theoretical_error_bound: float
    action_tolerance: float

    def interpolate_field(self, S: float, F: float, field: str = "values") -> float:
        """Piecewise-linear interpolation on the physical (S,F) triangle.

        Interpolating between diagonals in (n,S) coordinates would extrapolate
        at S=0 or F=0; the two-triangle Cartesian construction avoids this.
        """
        if not (math.isfinite(S) and math.isfinite(F)) or min(S, F) < -1e-10:
            raise ValueError("Interpolation requires nonnegative finite states")
        if S + F > self.report_n + 1e-9:
            raise ValueError("State is outside the retained solution; extrapolation is disabled")
        S, F = max(0.0, S), max(0.0, F)
        if S + F > self.report_n:
            # Project only accepted floating-point overflow to the exact edge.
            scale = self.report_n / (S + F)
            S, F = S * scale, F * scale
        rows = getattr(self, field, None)
        if not isinstance(rows, list) or len(rows) != len(self.values):
            raise ValueError(f"Unknown interpolable field: {field}")
        s, f = S / self.h, F / self.h
        # Snap roundoff at grid vertices before choosing a triangle.
        if abs(s - round(s)) < 1e-10:
            s = float(round(s))
        if abs(f - round(f)) < 1e-10:
            f = float(round(f))
        j, k = int(math.floor(s)), int(math.floor(f))
        a, b = s - j, f - k
        level = j + k
        if a + b <= 1.0 + 1e-10:
            terms = [(1.0 - a - b, level, j), (a, level + 1, j + 1), (b, level + 1, j)]
        else:
            terms = [(a + b - 1.0, level + 2, j + 1),
                     (1.0 - b, level + 1, j + 1), (1.0 - a, level + 1, j)]
        return float(sum(w * rows[n][index] for w, n, index in terms if w > 1e-12))

    def action_at(self, S: float, F: float, tolerance: float | None = None) -> int:
        """Nominal feedback control; choose cheaper product 1 near a tie.

        The action map additionally records uncertainty due to the spatial
        boundary, but this deterministic feedback needs a single action.
        """
        threshold = self.action_tolerance if tolerance is None else tolerance
        return 2 if self.interpolate_field(S, F, "advantages") > threshold else 1


def choose_cutoff(params: Parameters, h: float, report_n: float,
                  truncation_tolerance: float) -> float:
    """Choose a buffer using the exact discrete geometric boundary bound."""
    if not math.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite")
    _grid_steps(report_n, h, "report_n")
    if not math.isfinite(truncation_tolerance) or truncation_tolerance <= 0:
        raise ValueError("truncation_tolerance must be positive and finite")
    bound = params.max_reward / params.delta
    layers = max(1, math.ceil(math.log(max(1.0, bound / truncation_tolerance))
                              / math.log1p(params.delta * h)))
    return report_n + layers * h


def solve(params: Parameters, h: float, cutoff_n: float, report_n: float,
          demand_function: Callable | None = None, action_tolerance: float = 1e-8) -> Solution:
    """Backward sweep with simultaneous zero/constant upper continuations.

    No array of size cutoff_n**2 is retained: only two working diagonals and
    the requested interior are stored. The beta demand is evaluated once per
    diagonal and reused in the lower and upper solves.
    """
    if not math.isfinite(h) or h <= 0:
        raise ValueError("h must be positive and finite")
    if not math.isfinite(action_tolerance) or action_tolerance < 0:
        raise ValueError("action_tolerance must be finite and nonnegative")
    outer = _grid_steps(cutoff_n, h, "cutoff_n")
    report = _grid_steps(report_n, h, "report_n")
    if report >= outer:
        raise ValueError("report_n must lie at least one grid layer below cutoff_n")
    bound = params.max_reward / params.delta
    lower_next = np.zeros(outer + 1, dtype=float)
    upper_next = np.full(outer + 1, bound, dtype=float)
    names = ("values", "upper_values", "advantages", "advantage_lower", "advantage_upper",
             "actions", "demand_values")
    fields = {name: [None] * (report + 1) for name in names}
    residual = 0.0
    error_bound = 0.0
    r1, r2 = params.revenue - params.c1, params.revenue - params.c2

    for level in range(outer - 1, -1, -1):
        successes = np.arange(level + 1, dtype=float) * h
        failures = (level - np.arange(level + 1, dtype=float)) * h
        if demand_function is None:
            d = np.asarray(demand(successes, failures, params))
        else:
            d = np.broadcast_to(np.asarray(demand_function(successes, failures), dtype=float), successes.shape)
            if not np.isfinite(d).all() or np.any(d < 0) or np.any(d > 1):
                raise ValueError("Demand must be finite and lie in [0,1]")
        q1 = h * r1 + params.p1 * lower_next[1:] + (1 - params.p1) * lower_next[:-1]
        q2 = h * r2 + params.p2 * lower_next[1:] + (1 - params.p2) * lower_next[:-1]
        u1 = h * r1 + params.p1 * upper_next[1:] + (1 - params.p1) * upper_next[:-1]
        u2 = h * r2 + params.p2 * upper_next[1:] + (1 - params.p2) * upper_next[:-1]
        factor = d / (params.delta * h + d)
        lower = factor * np.maximum(q1, q2)
        upper = factor * np.maximum(u1, u2)

        if level <= report:
            advantage = -params.delta_c + params.delta_p * (lower_next[1:] - lower_next[:-1]) / h
            alo = -params.delta_c + params.delta_p * (lower_next[1:] - upper_next[:-1]) / h
            ahi = -params.delta_c + params.delta_p * (upper_next[1:] - lower_next[:-1]) / h
            actions = np.where((d > 0) & (alo > action_tolerance), 2,
                               np.where((d > 0) & (ahi < -action_tolerance), 1, 0)).astype(np.int8)
            # At zero demand every control has exactly zero reward and drift.
            advantage = np.where(d > 0, advantage, 0.0)
            rhs = d * np.maximum(r1 + (params.p1 * (lower_next[1:] - lower)
                                       + (1 - params.p1) * (lower_next[:-1] - lower)) / h,
                                  r2 + (params.p2 * (lower_next[1:] - lower)
                                       + (1 - params.p2) * (lower_next[:-1] - lower)) / h)
            residual = max(residual, float(np.max(np.abs(params.delta * lower - rhs))))
            error_bound = max(error_bound, float(np.max(upper - lower)))
            for name, row in zip(names, (lower, upper, advantage, alo, ahi, actions, d)):
                fields[name][level] = row.copy()
        lower_next, upper_next = lower, upper

    theoretical = bound * math.exp(-(outer - report) * math.log1p(params.delta * h))
    return Solution(params=params, h=float(h), cutoff_n=float(cutoff_n), report_n=float(report_n),
                    **fields, residual=residual, value_error_bound=error_bound,
                    theoretical_error_bound=theoretical, action_tolerance=action_tolerance)
