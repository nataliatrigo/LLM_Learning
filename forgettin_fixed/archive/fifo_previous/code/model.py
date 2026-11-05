"""Exact finite FIFO-memory monopoly model.

A history is a binary word, oldest observation first (most significant bit).
The posterior is Beta(1 + successes, 1 + failures).  Nothing is forgotten
while the customer chooses the outside option.  On a purchase, the next
outcome is appended and, once memory is full, the oldest bit is removed.

Ordered words, rather than success counts, are the Markov states.  The mature
state space has 2**n states.  All computations here enumerate that space;
there is no random-eviction or count-only approximation.
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


_BYTE_POPCOUNT = np.array([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def success_counts(histories: np.ndarray | list[int]) -> np.ndarray:
    """Vectorized population counts for unsigned 64-bit history words."""
    words = np.ascontiguousarray(histories, dtype=np.uint64)
    return _BYTE_POPCOUNT[words.view(np.uint8).reshape(words.shape + (8,))].sum(
        axis=-1, dtype=np.uint8
    )


def evidence_matched_n(rho: float) -> int:
    """Match mature total evidence, rounding half upwards: n ~ 1/(1-rho)."""
    if not math.isfinite(rho) or not 0 <= rho < 1:
        raise ValueError("rho must be finite and lie in [0, 1)")
    return max(1, int(math.floor(1.0 / (1.0 - rho) + 0.5)))


def demand_by_count(primitives: Primitives, m: int) -> np.ndarray:
    s = np.arange(m + 1)
    return np.clip(betaincc(1 + s, 1 + m - s, primitives.p0), 0.0, 1.0)


@dataclass(slots=True)
class WindowSolution:
    n: int
    primitives: Primitives
    values: list[np.ndarray]
    advantages: list[np.ndarray]
    demands: list[np.ndarray]
    diagnostics: dict[str, Any]
    action_tolerance: float = 1e-8
    optimized: bool = True
    _count_demands: list[np.ndarray] = field(default_factory=list, repr=False)

    @property
    def method(self) -> str:
        return "exact_fifo"

    @property
    def status(self) -> str:
        return "exact_fifo_numerically_solved"

    def evaluate(
        self, m: np.ndarray | int, histories: np.ndarray | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return prescribed product-2 flags and demand before the choice.

        ``m`` counts retained observations, so values above n are capped.
        Histories must already contain only the last min(m,n) bits.
        """
        levels, words = np.broadcast_arrays(
            np.asarray(m, dtype=np.int64), np.asarray(histories, dtype=np.uint64)
        )
        if np.any(levels < 0):
            raise ValueError("retained observation counts must be nonnegative")
        levels = np.minimum(levels, self.n)
        actions = np.empty(levels.shape, dtype=bool)
        demands = np.empty(levels.shape, dtype=float)
        for level in np.unique(levels):
            take = levels == level
            indices = words[take].astype(np.intp)
            if np.any(words[take] >= np.uint64(1 << int(level))):
                raise ValueError("history word exceeds its retained length")
            actions[take] = self.optimized & (
                self.advantages[int(level)][indices] > self.action_tolerance
            )
            demands[take] = self.demands[int(level)][indices]
        return actions, demands

    def summary_by_count(self, *, include_transient: bool = False) -> list[dict[str, Any]]:
        """Report ranges, not a purported Markov policy on success counts."""
        rows: list[dict[str, Any]] = []
        levels = range(self.n + 1) if include_transient else [self.n]
        action_error = float(self.diagnostics["advantage_error_bound"])
        for m in levels:
            counts = success_counts(np.arange(1 << m, dtype=np.uint64))
            gaps = self.advantages[m]
            actions = self.optimized & (gaps > self.action_tolerance)
            for successes in range(m + 1):
                take = counts == successes
                selected = gaps[take]
                count = int(np.count_nonzero(take))
                p2 = int(np.count_nonzero(actions[take]))
                rows.append({
                    "n": self.n, "p0": self.primitives.p0,
                    "retained_observations": m, "mature": m == self.n,
                    "successes": successes, "histories": count,
                    "posterior_mean": (1 + successes) / (2 + m),
                    "demand": float(demand_by_count(self.primitives, m)[successes]),
                    "product2_histories": p2, "product2_share": p2 / count,
                    "history_dependent_action": 0 < p2 < count,
                    "advantage_min": float(np.min(selected)),
                    "advantage_max": float(np.max(selected)),
                    "value_min": float(np.min(self.values[m][take])),
                    "value_max": float(np.max(self.values[m][take])),
                    "uncertified_action_histories": int(np.count_nonzero(
                        np.abs(selected) <= action_error + self.action_tolerance
                    )),
                })
        return rows

    def save(self, path: str | Path) -> Path:
        """Write all exact history tables and provenance to one NPZ file."""
        metadata = {
            "format": "fifo-window-v1", "n": self.n,
            "primitives": self.primitives.as_dict(),
            "diagnostics": self.diagnostics,
            "action_tolerance": self.action_tolerance,
            "optimized": self.optimized,
            "history_encoding": "oldest bit is most significant; success=1",
        }
        arrays: dict[str, Any] = {"metadata": np.array(json.dumps(metadata))}
        for m in range(self.n + 1):
            arrays[f"value_{m}"] = self.values[m]
            arrays[f"advantage_{m}"] = self.advantages[m]
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "WindowSolution":
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"]))
            if metadata.get("format") != "fifo-window-v1":
                raise ValueError("unrecognized FIFO solution format")
            n = int(metadata["n"])
            primitives = Primitives.from_mapping(metadata["primitives"])
            values = [archive[f"value_{m}"].copy() for m in range(n + 1)]
            gaps = [archive[f"advantage_{m}"].copy() for m in range(n + 1)]
        demands = []
        for m in range(n + 1):
            if values[m].shape != (1 << m,) or gaps[m].shape != (1 << m,):
                raise ValueError("malformed FIFO solution tables")
            counts = success_counts(np.arange(1 << m, dtype=np.uint64))
            demands.append(demand_by_count(primitives, m)[counts])
        return cls(n, primitives, values, gaps, demands,
                   metadata["diagnostics"], metadata["action_tolerance"],
                   metadata["optimized"])


def _maximand(
    value: np.ndarray, primitives: Primitives, *, optimize: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Q values for paired even/odd successors of a given parent word."""
    failure, success = value[0::2], value[1::2]
    q1 = primitives.net_rewards[0] + primitives.gamma * (
        primitives.p1 * success + (1.0 - primitives.p1) * failure
    )
    advantage = -primitives.delta_c + primitives.gamma * primitives.delta_p * (
        success - failure
    )
    if optimize:
        q1 += np.maximum(advantage, 0.0)
    return q1, advantage


def solve_window(
    primitives: Primitives,
    n: int,
    *,
    tolerance: float = 1e-10,
    action_tolerance: float = 1e-8,
    max_iterations: int = 100_000,
    optimize: bool = True,
) -> WindowSolution:
    """Enumerate the exact FIFO MDP, then solve all startup layers.

    The mature Bellman operator is a contraction with modulus
    beta=max_h gamma*lambda(d(h)).  Its residual certifies the sup-norm
    error of every value in the state space.  Values are floating point;
    no interpolation or state aggregation is used.

    n is restricted to <=24 to avoid accidental exponential allocations.
    n=20 entails 1,048,576 mature states and 2,097,151 total states.
    """
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= n <= 24:
        raise ValueError("exact enumeration requires an integer n in [1,24]")
    if not math.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if not math.isfinite(action_tolerance) or action_tolerance < 0:
        raise ValueError("action_tolerance must be finite and nonnegative")
    if not isinstance(max_iterations, int) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    counts = success_counts(np.arange(1 << n, dtype=np.uint64))
    mature_demand = demand_by_count(primitives, n)[counts]
    del counts
    waiting = mature_demand / (1.0 - primitives.gamma * (1.0 - mature_demand))
    modulus = float(primitives.gamma * np.max(waiting))
    value = np.zeros(1 << n, dtype=float)
    half = 1 << (n - 1)
    residual = math.inf
    for iteration in range(1, max_iterations + 1):
        q, _ = _maximand(value, primitives, optimize=optimize)
        updated = np.empty_like(value)
        # States h and h+2**(n-1) have identical successors.  Only the
        # current demand differs, because the dropped oldest bit differs.
        updated[:half] = waiting[:half] * q
        updated[half:] = waiting[half:] * q
        residual = float(np.max(np.abs(updated - value)))
        value = updated
        if residual <= tolerance:
            break
    else:
        raise RuntimeError(f"FIFO value iteration failed for n={n}: residual={residual:g}")

    q, mature_gap_half = _maximand(value, primitives, optimize=optimize)
    checked = np.empty_like(value)
    checked[:half] = waiting[:half] * q
    checked[half:] = waiting[half:] * q
    residual = float(np.max(np.abs(checked - value)))
    value_error = residual / (1.0 - modulus)
    # Two successor errors can enter their difference with opposite signs.
    advantage_error = 2 * primitives.gamma * primitives.delta_p * value_error

    values: list[np.ndarray] = [np.empty(0)] * (n + 1)
    gaps: list[np.ndarray] = [np.empty(0)] * (n + 1)
    demands: list[np.ndarray] = [np.empty(0)] * (n + 1)
    values[n] = value
    gaps[n] = np.tile(mature_gap_half, 2)
    demands[n] = mature_demand
    max_startup_residual = 0.0
    for m in range(n - 1, -1, -1):
        counts_m = success_counts(np.arange(1 << m, dtype=np.uint64))
        demand = demand_by_count(primitives, m)[counts_m]
        waiting_m = demand / (1.0 - primitives.gamma * (1.0 - demand))
        q_m, gap_m = _maximand(values[m + 1], primitives, optimize=optimize)
        values[m] = waiting_m * q_m
        gaps[m] = gap_m
        demands[m] = demand
        max_startup_residual = max(max_startup_residual, float(np.max(
            np.abs(values[m] - waiting_m * q_m)
        )))

    min_abs_gap = min(float(np.min(np.abs(g))) for g in gaps)
    uncertified = sum(int(np.count_nonzero(np.abs(g) <= advantage_error + action_tolerance))
                      for g in gaps)
    # Greedy policy value is within 2*error, plus the declared action tie
    # tolerance accumulated through the waiting-adjusted contraction.
    policy_loss_bound = 2 * value_error + action_tolerance / (1.0 - modulus)
    diagnostics = {
        "method": "exact_ordered_history_enumeration",
        "mature_states": 1 << n, "total_states": (1 << (n + 1)) - 1,
        "iterations": iteration, "bellman_residual": residual,
        "startup_bellman_residual": max_startup_residual,
        "contraction_modulus": modulus,
        "value_error_bound": value_error,
        "advantage_error_bound": advantage_error,
        "greedy_policy_loss_bound": policy_loss_bound if optimize else None,
        "minimum_absolute_advantage": min_abs_gap,
        "uncertified_action_states": uncertified,
        "all_actions_certified": uncertified == 0,
        "action_tolerance": action_tolerance,
        "tolerance": tolerance,
        "initial_value": float(values[0][0]),
        "optimized": optimize,
        "floating_point_note": "Residual bounds use floating-point Bellman evaluations; not interval arithmetic.",
    }
    return WindowSolution(n, primitives, values, gaps, demands, diagnostics,
                          action_tolerance, optimize)
