"""Exact one-dimensional recursion across Seller-A observation layers."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import betaincc

from forgetting.model import Primitives, solve_boundary


def _components(mask: np.ndarray) -> int:
    padded = np.concatenate(([False], mask, [False]))
    return int(np.count_nonzero((~padded[:-1]) & padded[1:]))


def solve_observation_layers(
    primitives: Primitives,
    rho: float,
    *,
    nodes: int = 1_001,
    terminal_gap: float = 1e-3,
    tolerance: float = 1e-10,
    action_tolerance: float = 1e-8,
    max_iterations: int = 100_000,
    return_policy: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Solve all deterministic observation layers by backward induction.

    Layer ``k`` has total normalized evidence ``m_k=1-rho**k`` and is
    parameterized by its success share ``u in [0,1]``.  The terminal condition
    is the mature-boundary fixed point.  ``terminal_gap`` controls the maximum
    remaining normalized mass at the terminal layer. If ``return_policy`` is
    true, the numerical-grid advantage and demand arrays are retained for path
    simulation; their memory use is proportional to ``nodes`` times the number
    of transient layers.
    """

    if not 0.0 < rho < 1.0:
        raise ValueError("rho must lie in (0,1)")
    if nodes < 101:
        raise ValueError("nodes must be at least 101")
    if not 0.0 < terminal_gap < 1.0:
        raise ValueError("terminal_gap must lie in (0,1)")
    if not math.isfinite(action_tolerance) or action_tolerance < 0.0:
        raise ValueError("action_tolerance must be finite and nonnegative")
    coordinate = np.linspace(0.0, 1.0, nodes)
    # A uniform boundary grid is required because every transient layer uses
    # this same success-share coordinate.
    mature = solve_boundary(
        primitives,
        rho,
        central_nodes=nodes,
        outer_nodes=2,
        half_widths=1e6,
        tolerance=tolerance,
        max_iterations=max_iterations,
    )
    value_next = mature.value.copy()
    terminal_observation = int(
        math.ceil(math.log(terminal_gap) / math.log(rho))
    )
    innovation = 1.0 - rho
    reward1, reward2 = primitives.net_rewards
    rows: list[dict[str, float | int | bool | None]] = []
    advantage_table = (
        np.empty((terminal_observation + 1, nodes), dtype=float)
        if return_policy
        else None
    )
    demand_table = (
        np.empty((terminal_observation + 1, nodes), dtype=float)
        if return_policy
        else None
    )

    for observation in range(terminal_observation - 1, -1, -1):
        remaining_mass = rho**observation
        total_mass = 1.0 - remaining_mass
        next_total_mass = 1.0 - rho * remaining_mass
        success_shift = innovation / next_total_mass
        persistence = 1.0 - success_shift
        success_value = np.interp(
            persistence * coordinate + success_shift, coordinate, value_next
        )
        failure_value = np.interp(
            persistence * coordinate, coordinate, value_next
        )
        advantage = -primitives.delta_c + primitives.gamma * primitives.delta_p * (
            success_value - failure_value
        )
        active = advantage > action_tolerance
        components = _components(active)
        posterior_mean = (
            innovation + total_mass * coordinate
        ) / (2.0 * innovation + total_mass)
        rows.append(
            {
                "rho": rho,
                "gamma": primitives.gamma,
                "observation": observation,
                "mature_boundary": False,
                "effective_sample_size": total_mass / innovation,
                "maximum_advantage": float(np.max(advantage)),
                "product2_active": bool(np.any(active)),
                "product2_components": components,
                "lower_active_posterior_mean": (
                    float(posterior_mean[active][0]) if np.any(active) else None
                ),
                "upper_active_posterior_mean": (
                    float(posterior_mean[active][-1]) if np.any(active) else None
                ),
            }
        )
        alpha = 1.0 + total_mass * coordinate / innovation
        beta = 1.0 + total_mass * (1.0 - coordinate) / innovation
        demand = np.asarray(betaincc(alpha, beta, primitives.p0), dtype=float)
        if return_policy:
            if advantage_table is None or demand_table is None:
                raise RuntimeError("policy tables were not initialized")
            advantage_table[observation] = advantage
            demand_table[observation] = demand
        waiting = demand / (1.0 - primitives.gamma * (1.0 - demand))
        q1 = reward1 + primitives.gamma * (
            primitives.p1 * success_value
            + (1.0 - primitives.p1) * failure_value
        )
        q2 = reward2 + primitives.gamma * (
            primitives.p2 * success_value
            + (1.0 - primitives.p2) * failure_value
        )
        value_next = waiting * np.maximum(q1, q2)

    rows.reverse()
    mature_active = mature.advantage > action_tolerance
    rows.append(
        {
            "rho": rho,
            "gamma": primitives.gamma,
            "observation": None,
            "mature_boundary": True,
            "effective_sample_size": 1.0 / innovation,
            "maximum_advantage": float(np.max(mature.advantage)),
            "product2_active": bool(np.any(mature_active)),
            "product2_components": _components(mature_active),
            "lower_active_posterior_mean": (
                float(mature.posterior_mean[mature_active][0])
                if np.any(mature_active)
                else None
            ),
            "upper_active_posterior_mean": (
                float(mature.posterior_mean[mature_active][-1])
                if np.any(mature_active)
                else None
            ),
        }
    )
    metadata: dict[str, object] = {
        "terminal_observation": terminal_observation,
        "terminal_gap": terminal_gap,
        "nodes": nodes,
        "mature_value": mature.value,
        "coordinate": coordinate,
        "mature_bellman_residual": mature.bellman_residual,
    }
    if return_policy:
        if advantage_table is None or demand_table is None:
            raise RuntimeError("policy tables were not initialized")
        advantage_table[terminal_observation] = mature.advantage
        demand_table[terminal_observation] = mature.demand
        metadata.update(
            {
                "advantage_table": advantage_table,
                "demand_table": demand_table,
            }
        )
    return pd.DataFrame(rows), metadata
