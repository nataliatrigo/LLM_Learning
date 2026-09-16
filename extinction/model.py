"""Stationary count-state approximation used in the extinction experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.special import betaincc


POLICY_PRODUCT1 = np.int8(0)
POLICY_TIE = np.int8(1)
POLICY_PRODUCT2 = np.int8(2)


@dataclass(frozen=True, slots=True)
class Parameters:
    """Primitives of the baseline discounted reputation model."""

    p0: float = 0.50
    p1: float = 0.35
    p2: float = 0.80
    c1: float = 0.05
    c2: float = 0.65
    revenue: float = 1.0
    gamma: float = 0.98

    def __post_init__(self) -> None:
        probabilities = (self.p0, self.p1, self.p2)
        if not all(0.0 < value < 1.0 for value in probabilities):
            raise ValueError("p0, p1, and p2 must lie strictly between zero and one")
        if not self.p2 > self.p1:
            raise ValueError("product 2 must have a larger success probability")
        if not self.c2 > self.c1:
            raise ValueError("product 2 must have a larger cost")
        if not self.revenue >= self.c2:
            raise ValueError("revenue must weakly exceed both product costs")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must lie in [0,1)")

    @property
    def delta_p(self) -> float:
        return self.p2 - self.p1

    @property
    def delta_c(self) -> float:
        return self.c2 - self.c1

    @property
    def threshold(self) -> float:
        """Continuation-gap threshold ``Delta c / Delta p``."""

        return self.delta_c / self.delta_p


def demand(n: int, p0: float) -> np.ndarray:
    """Thompson-sampling demand on diagonal ``n``, ordered by ``S=0,...,n``."""

    if n < 0 or int(n) != n:
        raise ValueError("n must be a nonnegative integer")
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must lie strictly between zero and one")
    successes = np.arange(int(n) + 1, dtype=float)
    failures = int(n) - successes
    return np.clip(
        np.asarray(betaincc(successes + 1.0, failures + 1.0, p0), dtype=float),
        0.0,
        1.0,
    )


def action_tolerance(
    values: np.ndarray,
    absolute: float = 1e-10,
    relative: float = 1e-8,
) -> float:
    """Scale-aware tolerance used to distinguish strict actions from ties."""

    array = np.asarray(values, dtype=float)
    scale = max(1.0, float(np.max(np.abs(array))) if array.size else 0.0)
    return max(float(absolute), float(relative) * scale)


def solve_stationary_truncation(
    params: Parameters,
    outer_diagonal: int,
    report_diagonal: int,
) -> dict:
    """Solve a truncated approximation to the stationary Bellman equation.

    The Bellman self-loop is rearranged exactly.  Backward recursion starts
    from zero value on diagonal ``outer_diagonal + 1``; only the common,
    prespecified reported interior is retained.
    """

    if outer_diagonal < 1 or int(outer_diagonal) != outer_diagonal:
        raise ValueError("outer_diagonal must be a positive integer")
    if report_diagonal < 0 or int(report_diagonal) != report_diagonal:
        raise ValueError("report_diagonal must be a nonnegative integer")
    if report_diagonal >= outer_diagonal:
        raise ValueError("report_diagonal must be below outer_diagonal")

    outer_diagonal = int(outer_diagonal)
    report_diagonal = int(report_diagonal)
    next_value = np.zeros(outer_diagonal + 2, dtype=float)
    layers: dict[int, dict[str, np.ndarray]] = {}
    maximum_residual = 0.0

    for n in range(outer_diagonal, -1, -1):
        state_demand = demand(n, params.p0)
        denominator = 1.0 - params.gamma * (1.0 - state_demand)
        q1 = params.revenue - params.c1 + params.gamma * (
            params.p1 * next_value[1:]
            + (1.0 - params.p1) * next_value[:-1]
        )
        q2 = params.revenue - params.c2 + params.gamma * (
            params.p2 * next_value[1:]
            + (1.0 - params.p2) * next_value[:-1]
        )
        current_value = state_demand * np.maximum(q1, q2) / denominator

        if n <= report_diagonal:
            gap = params.gamma * (next_value[1:] - next_value[:-1])
            advantage = -params.delta_c + params.delta_p * gap
            rhs = (
                params.gamma * (1.0 - state_demand) * current_value
                + state_demand * np.maximum(q1, q2)
            )
            maximum_residual = max(
                maximum_residual,
                float(np.max(np.abs(current_value - rhs))),
            )
            layers[n] = {
                "value": current_value.copy(),
                "gap": gap.copy(),
                "advantage": advantage.copy(),
                "demand": state_demand.copy(),
            }
        next_value = current_value

    return {
        "parameters": params,
        "outer_diagonal": outer_diagonal,
        "report_diagonal": report_diagonal,
        "layers": layers,
        "maximum_bellman_residual": maximum_residual,
    }


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    cuts = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(indices) - 1]
    return [
        (int(indices[start]), int(indices[end]))
        for start, end in zip(starts, ends, strict=True)
    ]


def _classify_advantage(
    advantage: np.ndarray,
    tolerance_function: Callable[[np.ndarray], float],
) -> dict[str, object]:
    """Classify one diagonal and summarize its robust product-2 geometry."""

    advantage = np.asarray(advantage, dtype=float)
    tolerance = float(tolerance_function(advantage))
    robust_product2 = advantage > tolerance
    robust_product1 = advantage < -tolerance
    tied = ~(robust_product2 | robust_product1)
    components = _runs(robust_product2)
    robust_violation = False
    ambiguous_separation = False
    if len(components) > 1:
        for (_, left_end), (right_start, _) in zip(
            components,
            components[1:],
            strict=True,
        ):
            interior = slice(left_end + 1, right_start)
            robust_violation |= bool(robust_product1[interior].any())
            ambiguous_separation |= bool(tied[interior].any())
    return {
        "tolerance": tolerance,
        "product1": robust_product1,
        "tie": tied,
        "product2": robust_product2,
        "components": components,
        "ambiguous": bool(ambiguous_separation or tied.any()),
        "robust_interval_violation": bool(robust_violation),
    }


def classify_solution(
    solution: dict,
    tolerance_function: Callable[[np.ndarray], float] = action_tolerance,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return state-level actions and diagonal-level extinction diagnostics."""

    params: Parameters = solution["parameters"]
    states: list[dict[str, float | int | str]] = []
    diagonals: list[dict[str, float | int | bool]] = []

    for n in sorted(solution["layers"]):
        layer = solution["layers"][n]
        advantage = np.asarray(layer["advantage"], dtype=float)
        classified = _classify_advantage(advantage, tolerance_function)
        tolerance = float(classified["tolerance"])
        robust_product2 = np.asarray(classified["product2"], dtype=bool)
        robust_product1 = np.asarray(classified["product1"], dtype=bool)
        tied = np.asarray(classified["tie"], dtype=bool)
        components = list(classified["components"])

        for successes in range(n + 1):
            states.append(
                {
                    "n": n,
                    "S": successes,
                    "F": n - successes,
                    "posterior_mean": (successes + 1.0) / (n + 2.0),
                    "value": float(layer["value"][successes]),
                    "G": float(layer["gap"][successes]),
                    "advantage": float(advantage[successes]),
                    "classification": (
                        "product2"
                        if robust_product2[successes]
                        else ("product1" if robust_product1[successes] else "tie")
                    ),
                }
            )

        active = np.flatnonzero(robust_product2)
        diagonals.append(
            {
                "n": n,
                "active": bool(len(active)),
                "robust_product2_states": int(len(active)),
                "components": int(len(components)),
                "ambiguous": bool(classified["ambiguous"]),
                "robust_interval_violation": bool(
                    classified["robust_interval_violation"]
                ),
                "lower_S": int(active[0]) if len(active) else np.nan,
                "upper_S": int(active[-1]) if len(active) else np.nan,
                "lower_posterior_mean": (
                    (active[0] + 1.0) / (n + 2.0) if len(active) else np.nan
                ),
                "upper_posterior_mean": (
                    (active[-1] + 1.0) / (n + 2.0) if len(active) else np.nan
                ),
                "maximum_G": float(np.max(layer["gap"])),
                "extinction_margin": float(np.max(layer["gap"]) - params.threshold),
                "maximum_advantage": float(np.max(advantage)),
                "tolerance": tolerance,
            }
        )

    return pd.DataFrame(states), pd.DataFrame(diagonals)


def solve_stationary_policy_truncation(
    params: Parameters,
    outer_diagonal: int,
    report_diagonal: int,
    tolerance_function: Callable[[np.ndarray], float] = action_tolerance,
) -> dict:
    """Solve the Bellman recursion while retaining compact action layers only.

    This follows :func:`solve_stationary_truncation` exactly, but stores one
    signed-byte action code per reported state instead of four floating-point
    arrays.  It is intended for large statewise policy maps where materializing
    a multi-million-row DataFrame would dominate the calculation.

    Action codes are ``POLICY_PRODUCT1``, ``POLICY_TIE``, and
    ``POLICY_PRODUCT2``.  The returned diagonal diagnostics have the same
    classification columns as :func:`classify_solution`.
    """

    if outer_diagonal < 1 or int(outer_diagonal) != outer_diagonal:
        raise ValueError("outer_diagonal must be a positive integer")
    if report_diagonal < 0 or int(report_diagonal) != report_diagonal:
        raise ValueError("report_diagonal must be a nonnegative integer")
    if report_diagonal >= outer_diagonal:
        raise ValueError("report_diagonal must be below outer_diagonal")

    outer_diagonal = int(outer_diagonal)
    report_diagonal = int(report_diagonal)
    next_value = np.zeros(outer_diagonal + 2, dtype=float)
    actions: dict[int, np.ndarray] = {}
    diagonal_rows: list[dict[str, float | int | bool]] = []
    maximum_residual = 0.0

    for n in range(outer_diagonal, -1, -1):
        state_demand = demand(n, params.p0)
        denominator = 1.0 - params.gamma * (1.0 - state_demand)
        q1 = params.revenue - params.c1 + params.gamma * (
            params.p1 * next_value[1:]
            + (1.0 - params.p1) * next_value[:-1]
        )
        q2 = params.revenue - params.c2 + params.gamma * (
            params.p2 * next_value[1:]
            + (1.0 - params.p2) * next_value[:-1]
        )
        current_value = state_demand * np.maximum(q1, q2) / denominator

        if n <= report_diagonal:
            gap = params.gamma * (next_value[1:] - next_value[:-1])
            advantage = -params.delta_c + params.delta_p * gap
            classified = _classify_advantage(advantage, tolerance_function)
            robust_product1 = np.asarray(classified["product1"], dtype=bool)
            tied = np.asarray(classified["tie"], dtype=bool)
            robust_product2 = np.asarray(classified["product2"], dtype=bool)
            action = np.full(n + 1, POLICY_TIE, dtype=np.int8)
            action[robust_product1] = POLICY_PRODUCT1
            action[robust_product2] = POLICY_PRODUCT2
            actions[n] = action

            rhs = (
                params.gamma * (1.0 - state_demand) * current_value
                + state_demand * np.maximum(q1, q2)
            )
            maximum_residual = max(
                maximum_residual,
                float(np.max(np.abs(current_value - rhs))),
            )
            active = np.flatnonzero(robust_product2)
            components = list(classified["components"])
            diagonal_rows.append(
                {
                    "n": n,
                    "active": bool(len(active)),
                    "robust_product2_states": int(len(active)),
                    "components": int(len(components)),
                    "ambiguous": bool(classified["ambiguous"]),
                    "robust_interval_violation": bool(
                        classified["robust_interval_violation"]
                    ),
                    "lower_S": int(active[0]) if len(active) else np.nan,
                    "upper_S": int(active[-1]) if len(active) else np.nan,
                    "lower_posterior_mean": (
                        (active[0] + 1.0) / (n + 2.0)
                        if len(active)
                        else np.nan
                    ),
                    "upper_posterior_mean": (
                        (active[-1] + 1.0) / (n + 2.0)
                        if len(active)
                        else np.nan
                    ),
                    "maximum_G": float(np.max(gap)),
                    "extinction_margin": float(
                        np.max(gap) - params.threshold
                    ),
                    "maximum_advantage": float(np.max(advantage)),
                    "tolerance": float(classified["tolerance"]),
                    "tied_states": int(np.count_nonzero(tied)),
                }
            )
        next_value = current_value

    diagonals = pd.DataFrame(diagonal_rows).sort_values(
        "n",
        ignore_index=True,
    )
    return {
        "parameters": params,
        "outer_diagonal": outer_diagonal,
        "report_diagonal": report_diagonal,
        "actions": actions,
        "diagonals": diagonals,
        "maximum_bellman_residual": maximum_residual,
    }
