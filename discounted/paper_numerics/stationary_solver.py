"""Truncated-state approximation to the stationary discounted Bellman equation."""

from __future__ import annotations

from dataclasses import dataclass

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

    @property
    def threshold(self) -> float:
        return (self.c2 - self.c1) / (self.p2 - self.p1)


def demand(n: int, p0: float) -> np.ndarray:
    """Demand on diagonal n, ordered by S=0,...,n."""
    successes = np.arange(n + 1, dtype=float)
    failures = n - successes
    return np.clip(betaincc(successes + 1, failures + 1, p0), 0.0, 1.0)


def solve_stationary_truncation(
    params: Parameters,
    outer_diagonal: int,
    report_diagonal: int,
) -> dict:
    """Recurse from V=0 on n=outer_diagonal+1 and retain the interior."""
    if report_diagonal >= outer_diagonal:
        raise ValueError("report_diagonal must be below outer_diagonal")

    next_value = np.zeros(outer_diagonal + 2, dtype=float)
    layers: dict[int, dict[str, np.ndarray]] = {}
    max_residual = 0.0

    for n in range(outer_diagonal, -1, -1):
        d = demand(n, params.p0)
        denominator = 1.0 - params.gamma * (1.0 - d)
        q1 = (
            params.revenue
            - params.c1
            + params.gamma
            * (params.p1 * next_value[1:] + (1.0 - params.p1) * next_value[:-1])
        )
        q2 = (
            params.revenue
            - params.c2
            + params.gamma
            * (params.p2 * next_value[1:] + (1.0 - params.p2) * next_value[:-1])
        )
        current_value = d * np.maximum(q1, q2) / denominator

        if n <= report_diagonal:
            gap = params.gamma * (next_value[1:] - next_value[:-1])
            advantage = (params.p2 - params.p1) * (gap - params.threshold)
            rhs = params.gamma * (1.0 - d) * current_value + d * np.maximum(q1, q2)
            max_residual = max(max_residual, float(np.max(np.abs(current_value - rhs))))
            layers[n] = {
                "value": current_value.copy(),
                "gap": gap.copy(),
                "advantage": advantage.copy(),
                "demand": d.copy(),
            }
        next_value = current_value

    return {
        "parameters": params,
        "outer_diagonal": outer_diagonal,
        "report_diagonal": report_diagonal,
        "layers": layers,
        "maximum_bellman_residual": max_residual,
    }
