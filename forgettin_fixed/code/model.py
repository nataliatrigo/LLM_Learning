"""Exact monopoly MDP with replacement of the opposite outcome.

The state is (m,S): m<=n remembered observations and S remembered successes.
Before capacity, a new result is appended.  At capacity, a new success replaces
a failure and a new failure replaces a success.  At homogeneous boundaries a
same-type result leaves the counts unchanged.  The state has no ordered queue.

Only a purchase from A produces a result; the outside option leaves (m,S)
unchanged.  Calendar-time waiting is eliminated analytically in Bellman updates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import betaincc

from forgetting.Code.model import Primitives


MODEL_FORMAT = "opposite-replacement-v1"
METHOD = "exact_opposite_replacement"


def evidence_matched_n(rho: float) -> int:
    """Match mature geometric evidence 1/(1-rho), rounding half upwards."""
    if not math.isfinite(float(rho)) or not 0 <= float(rho) < 1:
        raise ValueError("rho must be finite and lie in [0,1)")
    return max(1, int(math.floor(1.0 / (1.0 - float(rho)) + 0.5)))


def demand_by_count(primitives: Primitives, m: int) -> np.ndarray:
    """Thompson demand for every retained success count at length m."""
    s = np.arange(m + 1)
    return np.clip(betaincc(1 + s, 1 + m - s, primitives.p0), 0.0, 1.0)


def retained_transition(m, successes, outcome, n: int):
    """Vectorized count transition after an actual A purchase.

    The same public transition is useful for inspection; simulations and
    independent tests may implement their own update to avoid circular checks.
    """
    m, successes, outcome = np.broadcast_arrays(
        np.asarray(m, dtype=np.int64), np.asarray(successes, dtype=np.int64),
        np.asarray(outcome, dtype=np.int64))
    if (np.any((m < 0) | (m > n)) or np.any((successes < 0) | (successes > m))
            or np.any((outcome < 0) | (outcome > 1))):
        raise ValueError("invalid retained state or outcome")
    next_m = np.minimum(n, m + 1)
    next_s = np.where(m < n, successes + outcome,
                      np.clip(successes + 2 * outcome - 1, 0, n))
    return next_m, next_s


@dataclass(slots=True)
class CountSolution:
    n: int
    primitives: Primitives
    values: list[np.ndarray]
    advantages: list[np.ndarray]
    demands: list[np.ndarray]
    diagnostics: dict[str, Any]
    action_tolerance: float = 1e-8
    optimized: bool = True
    _flat_advantage: np.ndarray = field(init=False, repr=False)
    _flat_demand: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Triangular state indexing avoids a per-level loop in every simulated
        # calendar period, including paths at many different startup lengths.
        self._flat_advantage = np.concatenate(self.advantages)
        self._flat_demand = np.concatenate(self.demands)

    @property
    def method(self) -> str:
        return METHOD

    @property
    def status(self) -> str:
        return "exact_count_mdp_numerically_solved"

    def evaluate(self, m, successes) -> tuple[np.ndarray, np.ndarray]:
        """Return product-2 flags and demand for scalar or vector count states.

        Observation lengths above n are capped, while the success argument
        must be the current retained count rather than lifetime successes.
        """
        levels, counts = np.broadcast_arrays(np.asarray(m), np.asarray(successes))
        if (np.any(~np.isfinite(levels)) or np.any(~np.isfinite(counts))
                or np.any(levels != np.floor(levels))
                or np.any(counts != np.floor(counts))):
            raise ValueError("retained lengths and successes must be finite integers")
        levels = np.minimum(np.asarray(levels, dtype=np.int64), self.n)
        counts = np.asarray(counts, dtype=np.int64)
        if np.any(levels < 0) or np.any((counts < 0) | (counts > levels)):
            raise ValueError("success count must lie between zero and retained length")
        index = levels * (levels + 1) // 2 + counts
        action = self.optimized & (self._flat_advantage[index] > self.action_tolerance)
        return action, self._flat_demand[index]

    def product2_intervals(self, m: int | None = None) -> list[tuple[int, int]]:
        """Inclusive integer intervals selecting product 2; no shape assumed."""
        m = self.n if m is None else m
        if not isinstance(m, (int, np.integer)) or not 0 <= m <= self.n:
            raise ValueError("m must be an integer between zero and n")
        active = np.flatnonzero(self.optimized &
                               (self.advantages[m] > self.action_tolerance))
        pieces = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
        return [(int(piece[0]), int(piece[-1])) for piece in pieces if len(piece)]

    def policy_intervals(self, m: int | None = None) -> list[tuple[int, int]]:
        return self.product2_intervals(m)

    def summary_by_count(self, *, include_transient=False) -> list[dict[str, Any]]:
        rows = []
        levels = range(self.n + 1) if include_transient else [self.n]
        errors = self.diagnostics["advantage_error_bounds_by_level"]
        for m in levels:
            for s in range(m + 1):
                gap = float(self.advantages[m][s])
                product2 = bool(self.optimized and gap > self.action_tolerance)
                rows.append({"n": self.n, "p0": self.primitives.p0,
                             "retained_observations": m, "mature": m == self.n,
                             "S": s, "F": m - s,
                             "posterior_mean": (1 + s) / (2 + m),
                             "demand": float(self.demands[m][s]),
                             "value": float(self.values[m][s]),
                             "advantage": gap,
                             "advantage_min": gap, "advantage_max": gap,
                             "product": 2 if product2 else 1,
                             "product2_share": int(product2),
                             "action_min": int(product2), "action_max": int(product2),
                             "action_certified": bool(abs(gap) > errors[m] + self.action_tolerance),
                             "method": self.method})
        return rows

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"format": MODEL_FORMAT, "n": self.n,
                    "primitives": self.primitives.as_dict(),
                    "diagnostics": self.diagnostics,
                    "action_tolerance": self.action_tolerance,
                    "optimized": self.optimized,
                    "state": "(m,S); no ordered history",
                    "replacement": "success replaces failure; failure replaces success; reflecting boundaries"}
        arrays = {"metadata": np.array(json.dumps(metadata))}
        for m in range(self.n + 1):
            arrays[f"value_{m}"] = self.values[m]
            arrays[f"advantage_{m}"] = self.advantages[m]
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "CountSolution":
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.get("format") != MODEL_FORMAT:
                raise ValueError("checkpoint is not an opposite-outcome replacement policy")
            n = int(metadata["n"])
            values = [archive[f"value_{m}"].copy() for m in range(n + 1)]
            advantages = [archive[f"advantage_{m}"].copy() for m in range(n + 1)]
        primitives = Primitives.from_mapping(metadata["primitives"])
        for m in range(n + 1):
            if values[m].shape != (m + 1,) or advantages[m].shape != (m + 1,):
                raise ValueError("malformed count-state policy tables")
        demands = [demand_by_count(primitives, m) for m in range(n + 1)]
        return cls(n, primitives, values, advantages, demands,
                   metadata["diagnostics"], metadata["action_tolerance"],
                   metadata["optimized"])


def _maximand(failure, success, primitives: Primitives, optimize):
    q1 = primitives.net_rewards[0] + primitives.gamma * (
        primitives.p1 * success + (1 - primitives.p1) * failure)
    gap = -primitives.delta_c + primitives.gamma * primitives.delta_p * (success - failure)
    return q1 + (np.maximum(gap, 0.0) if optimize else 0.0), gap


def solve_counts(primitives: Primitives, n: int, *, tolerance=1e-10,
                 action_tolerance=1e-8, max_iterations=100_000,
                 optimize=True) -> CountSolution:
    """Enumerate every count state, solve maturity and backward startup layers.

    There are n+1 mature states and (n+1)(n+2)/2 total states.  No state
    aggregation, interpolation, or large-window approximation is used.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not math.isfinite(action_tolerance) or action_tolerance < 0:
        raise ValueError("action_tolerance must be finite and nonnegative")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    n = int(n)
    demands = [demand_by_count(primitives, m) for m in range(n + 1)]
    d = demands[n]
    waiting = d / (1 - primitives.gamma * (1 - d))
    states = np.arange(n + 1)
    failure = np.maximum(states - 1, 0)
    success = np.minimum(states + 1, n)
    value = np.zeros(n + 1)
    for iteration in range(1, max_iterations + 1):
        q, _ = _maximand(value[failure], value[success], primitives, optimize)
        updated = waiting * q
        delta = float(np.max(np.abs(updated - value)))
        value = updated
        if delta <= tolerance:
            break
    else:
        raise RuntimeError(f"count-state value iteration did not converge for n={n}")
    checked, mature_gap = _maximand(value[failure], value[success], primitives, optimize)
    residual = float(np.max(np.abs(waiting * checked - value)))
    modulus = float(primitives.gamma * np.max(waiting))
    mature_error = residual / (1 - modulus)
    values = [np.empty(0)] * (n + 1)
    gaps = [np.empty(0)] * (n + 1)
    values[n], gaps[n] = value, mature_gap
    value_errors = [0.0] * (n + 1)
    gap_errors = [0.0] * (n + 1)
    value_errors[n] = mature_error
    gap_errors[n] = 2 * primitives.gamma * primitives.delta_p * mature_error
    for m in range(n - 1, -1, -1):
        q, gap = _maximand(values[m + 1][:-1], values[m + 1][1:], primitives, optimize)
        lam = demands[m] / (1 - primitives.gamma * (1 - demands[m]))
        values[m], gaps[m] = lam * q, gap
        value_errors[m] = float(primitives.gamma * np.max(lam) * value_errors[m + 1])
        gap_errors[m] = 2 * primitives.gamma * primitives.delta_p * value_errors[m + 1]
    uncertified_by_level = [int(np.count_nonzero(np.abs(gaps[m]) <= gap_errors[m] + action_tolerance))
                            for m in range(n + 1)]
    minimum_gap = min(float(np.min(np.abs(gap))) for gap in gaps)
    diagnostics = {"method": METHOD, "model_format": MODEL_FORMAT,
                   "mature_states": n + 1, "total_states": (n + 1) * (n + 2) // 2,
                   "iterations": iteration, "bellman_residual": residual,
                   "contraction_modulus": modulus,
                   "value_error_bound": mature_error,
                   "initial_value_error_bound": value_errors[0],
                   "advantage_error_bound": max(gap_errors),
                   "value_error_bounds_by_level": value_errors,
                   "advantage_error_bounds_by_level": gap_errors,
                   "greedy_policy_loss_bound": ((2 * primitives.gamma * mature_error + action_tolerance) / (1 - primitives.gamma) if optimize else None),
                   "minimum_absolute_advantage": minimum_gap,
                   "uncertified_action_states": sum(uncertified_by_level),
                   "uncertified_action_states_by_level": uncertified_by_level,
                   "all_actions_certified": not any(uncertified_by_level),
                   "action_tolerance": action_tolerance, "tolerance": tolerance,
                   "initial_value": float(values[0][0]), "optimized": bool(optimize),
                   "replacement_rule": "opposite outcome with reflecting boundaries",
                   "floating_point_note": "Residual certificates exclude floating-point roundoff; not interval arithmetic."}
    solution = CountSolution(n, primitives, values, gaps, demands, diagnostics,
                             action_tolerance, bool(optimize))
    diagnostics["product2_intervals"] = [list(pair) for pair in solution.product2_intervals()]
    diagnostics["product2_interval_count"] = len(diagnostics["product2_intervals"])
    return solution
