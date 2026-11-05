"""Numerical primitives for observation-time forgetting.

The mature normalized boundary ``x_s + x_f = 1`` is invariant.  Writing
``x = x_s`` reduces the stationary Bellman equation exactly to one dimension:

    V(x) = lambda(D(x)) max_i {
        R-c_i + gamma [p_i V(rho*x + 1-rho) + (1-p_i) V(rho*x)]
    }.

The boundary solver is shared by the paper figures and transient policy layers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Mapping

import numpy as np
from scipy.special import betaincc


@dataclass(frozen=True, slots=True)
class Primitives:
    """Seller and learning primitives used throughout the experiment."""

    p0: float = 0.50
    p1: float = 0.35
    p2: float = 0.80
    c1: float = 0.05
    c2: float = 0.65
    revenue: float = 1.00
    gamma: float = 0.98

    def __post_init__(self) -> None:
        values = asdict(self)
        if not all(math.isfinite(float(value)) for value in values.values()):
            raise ValueError("all primitives must be finite")
        if not 0.0 < self.p0 < 1.0:
            raise ValueError("p0 must lie strictly between zero and one")
        if not 0.0 < self.p1 < self.p2 < 1.0:
            raise ValueError("success probabilities must satisfy 0 < p1 < p2 < 1")
        if not 0.0 <= self.c1 < self.c2 < self.revenue:
            raise ValueError("costs must satisfy 0 <= c1 < c2 < revenue")
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must lie in [0,1)")

    @classmethod
    def from_mapping(cls, values: Mapping[str, float]) -> "Primitives":
        return cls(**{key: float(value) for key, value in values.items()})

    @property
    def delta_p(self) -> float:
        return self.p2 - self.p1

    @property
    def delta_c(self) -> float:
        return self.c2 - self.c1

    @property
    def investment_threshold(self) -> float:
        return self.delta_c / self.delta_p

    @property
    def net_rewards(self) -> tuple[float, float]:
        return self.revenue - self.c1, self.revenue - self.c2

    def as_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class BoundaryGrid:
    """One-dimensional grid refined around the posterior transition region."""

    coordinate: np.ndarray
    central_nodes: int
    outer_nodes: int
    half_widths: float

    @classmethod
    def build(
        cls,
        rho: float,
        p0: float,
        *,
        central_nodes: int,
        outer_nodes: int,
        half_widths: float = 9.0,
    ) -> "BoundaryGrid":
        validate_rho(rho)
        if central_nodes < 101 or outer_nodes < 2:
            raise ValueError("central_nodes >= 101 and outer_nodes >= 2 are required")
        innovation = 1.0 - rho
        width = half_widths * math.sqrt(innovation)
        lower = max(0.0, p0 - width)
        upper = min(1.0, p0 + width)
        pieces = [np.linspace(lower, upper, central_nodes)]
        grading = np.linspace(0.0, 1.0, outer_nodes + 1) ** 2
        if lower > 0.0:
            pieces.append(lower * grading)
        if upper < 1.0:
            pieces.append(upper + (1.0 - upper) * grading)
        coordinate = np.unique(np.concatenate(pieces))
        return cls(coordinate, central_nodes, outer_nodes, half_widths)


@dataclass(frozen=True, slots=True)
class BoundarySolution:
    """Solved mature-boundary value function and action diagnostics."""

    rho: float
    primitives: Primitives
    coordinate: np.ndarray
    posterior_mean: np.ndarray
    demand: np.ndarray
    value: np.ndarray
    advantage: np.ndarray
    iterations: int
    iteration_residual: float
    bellman_residual: float
    optimized: bool


def validate_rho(rho: float) -> None:
    if not math.isfinite(float(rho)) or not 0.0 < float(rho) < 1.0:
        raise ValueError("rho must lie strictly between zero and one")


def boundary_demand(
    coordinate: np.ndarray | Iterable[float], rho: float, p0: float
) -> np.ndarray:
    """Thompson demand on ``(x,1-x)`` in normalized coordinates."""

    validate_rho(rho)
    x = np.asarray(coordinate, dtype=float)
    innovation = 1.0 - rho
    alpha = 1.0 + x / innovation
    beta = 1.0 + (1.0 - x) / innovation
    return np.clip(np.asarray(betaincc(alpha, beta, p0), dtype=float), 0.0, 1.0)


def _bellman_update(
    value: np.ndarray,
    coordinate: np.ndarray,
    demand: np.ndarray,
    primitives: Primitives,
    rho: float,
    *,
    optimize: bool,
) -> tuple[np.ndarray, np.ndarray]:
    innovation = 1.0 - rho
    value_success = np.interp(rho * coordinate + innovation, coordinate, value)
    value_failure = np.interp(rho * coordinate, coordinate, value)
    reward1, reward2 = primitives.net_rewards
    q1 = reward1 + primitives.gamma * (
        primitives.p1 * value_success
        + (1.0 - primitives.p1) * value_failure
    )
    q2 = reward2 + primitives.gamma * (
        primitives.p2 * value_success
        + (1.0 - primitives.p2) * value_failure
    )
    waiting = demand / (1.0 - primitives.gamma * (1.0 - demand))
    maximand = np.maximum(q1, q2) if optimize else q1
    return waiting * maximand, q2 - q1


def solve_boundary(
    primitives: Primitives,
    rho: float,
    *,
    grid: BoundaryGrid | None = None,
    central_nodes: int = 2_001,
    outer_nodes: int = 400,
    half_widths: float = 9.0,
    tolerance: float = 1e-10,
    max_iterations: int = 100_000,
    optimize: bool = True,
) -> BoundarySolution:
    """Solve the exact one-dimensional mature-boundary Bellman equation."""

    validate_rho(rho)
    if tolerance <= 0.0 or max_iterations < 1:
        raise ValueError("tolerance and max_iterations must be positive")
    if grid is None:
        grid = BoundaryGrid.build(
            rho,
            primitives.p0,
            central_nodes=central_nodes,
            outer_nodes=outer_nodes,
            half_widths=half_widths,
        )
    coordinate = np.asarray(grid.coordinate, dtype=float)
    if coordinate[0] != 0.0 or coordinate[-1] != 1.0:
        raise ValueError("the boundary grid must include zero and one")
    demand = boundary_demand(coordinate, rho, primitives.p0)
    value = np.zeros_like(coordinate)
    iteration_residual = math.inf
    for iteration in range(1, max_iterations + 1):
        updated, _ = _bellman_update(
            value, coordinate, demand, primitives, rho, optimize=optimize
        )
        iteration_residual = float(np.max(np.abs(updated - value)))
        value = updated
        if iteration_residual <= tolerance:
            break
    else:
        raise RuntimeError(
            f"boundary value iteration did not converge at rho={rho:g}, "
            f"gamma={primitives.gamma:g}; residual={iteration_residual:.3e}"
        )
    checked, advantage = _bellman_update(
        value, coordinate, demand, primitives, rho, optimize=optimize
    )
    bellman_residual = float(np.max(np.abs(checked - value)))
    innovation = 1.0 - rho
    posterior_mean = (innovation + coordinate) / (1.0 + 2.0 * innovation)
    return BoundarySolution(
        rho=float(rho),
        primitives=primitives,
        coordinate=coordinate,
        posterior_mean=posterior_mean,
        demand=demand,
        value=value,
        advantage=advantage,
        iterations=iteration,
        iteration_residual=iteration_residual,
        bellman_residual=bellman_residual,
        optimized=optimize,
    )


def reachable_boundary_points(rho: float, depth: int = 14) -> np.ndarray:
    """Finite-depth endpoints approximating the mature reachable set ``K_rho``.

    For ``rho < 1/2``, the returned points have Hausdorff error at most
    ``rho**depth``.  For ``rho >= 1/2``, ``K_rho=[0,1]`` and callers should use
    the full solved grid instead.
    """

    validate_rho(rho)
    if depth < 1:
        raise ValueError("depth must be positive")
    if rho >= 0.5:
        return np.array([0.0, 1.0])
    points = np.array([0.0])
    innovation = 1.0 - rho
    for _ in range(depth):
        points = np.concatenate((rho * points, rho * points + innovation))
    # The all-zero and all-one tails give both endpoints of every cylinder.
    points = np.concatenate((points, points + rho**depth))
    return np.unique(np.clip(points, 0.0, 1.0))


def reachable_maximum(
    solution: BoundarySolution, *, reachable_depth: int = 14
) -> tuple[float, float, float]:
    """Maximum action advantage on the mature reachable set.

    Returns ``(maximum, argmax_x, approximation_radius)``.  The radius is zero
    when ``rho >= 1/2`` because the mature reachable set is the full interval.
    """

    if solution.rho >= 0.5:
        index = int(np.argmax(solution.advantage))
        return (
            float(solution.advantage[index]),
            float(solution.coordinate[index]),
            0.0,
        )
    points = reachable_boundary_points(solution.rho, reachable_depth)
    advantages = np.interp(points, solution.coordinate, solution.advantage)
    index = int(np.argmax(advantages))
    return (
        float(advantages[index]),
        float(points[index]),
        float(solution.rho**reachable_depth),
    )


def boundary_summary(
    solution: BoundarySolution,
    *,
    action_tolerance: float = 1e-8,
    reachable_depth: int = 14,
) -> dict[str, float | int | bool | None]:
    """Return tidy mature-boundary diagnostics for one parameter pair."""

    maximum, argmax_x, reachable_radius = reachable_maximum(
        solution, reachable_depth=reachable_depth
    )
    if solution.rho >= 0.5:
        evaluation_x = solution.coordinate
        evaluation_advantage = solution.advantage
    else:
        evaluation_x = reachable_boundary_points(solution.rho, reachable_depth)
        evaluation_advantage = np.interp(
            evaluation_x, solution.coordinate, solution.advantage
        )
    active = evaluation_advantage > action_tolerance
    posterior_mean = (
        1.0 - solution.rho + evaluation_x
    ) / (1.0 + 2.0 * (1.0 - solution.rho))
    return {
        "rho": solution.rho,
        "gamma": solution.primitives.gamma,
        "reachable_maximum_advantage": maximum,
        "reachable_argmax_x": argmax_x,
        "reachable_approximation_radius": reachable_radius,
        "full_boundary_maximum_advantage": float(np.max(solution.advantage)),
        "lower_active_posterior_mean": (
            float(posterior_mean[active][0]) if np.any(active) else None
        ),
        "upper_active_posterior_mean": (
            float(posterior_mean[active][-1]) if np.any(active) else None
        ),
        "active_reachable_share": float(np.mean(active)),
        "product2_recurrent": bool(maximum > action_tolerance),
        "product1_mature": bool(maximum < -action_tolerance),
        "numerically_critical": bool(abs(maximum) <= action_tolerance),
        "iterations": solution.iterations,
        "bellman_residual": solution.bellman_residual,
        "grid_nodes": int(solution.coordinate.size),
    }


def phi_max_free(
    primitives: Primitives,
    rho: float,
    **solver_kwargs: float | int,
) -> tuple[float, float]:
    """Maximum one-shot product-2 advantage under the always-product-1 value.

    At either boundary of the mature product-2 regime this max-free quantity
    is zero, because product 1 is optimal everywhere at the crossing.
    """

    solution = solve_boundary(
        primitives, rho, optimize=False, **solver_kwargs
    )
    maximum, argmax_x, _ = reachable_maximum(solution)
    return maximum, argmax_x


def small_rho_closed_form(
    primitives: Primitives,
) -> dict[str, float | int]:
    """Closed-form ``rho=0`` benchmark used as a numerical anchor.

    At complete forgetting, each product defines a scalar fixed-point value
    ``W_i``.  The optimal continuation value is the larger of those two
    branches, so the action gap must be evaluated at ``max(W_1, W_2)``.
    """

    demand_success = 1.0 - primitives.p0**2
    demand_failure = (1.0 - primitives.p0) ** 2
    waiting_success = demand_success / (
        1.0 - primitives.gamma * (1.0 - demand_success)
    )
    waiting_failure = demand_failure / (
        1.0 - primitives.gamma * (1.0 - demand_failure)
    )
    expected_waiting1 = (
        primitives.p1 * waiting_success
        + (1.0 - primitives.p1) * waiting_failure
    )
    expected_waiting2 = (
        primitives.p2 * waiting_success
        + (1.0 - primitives.p2) * waiting_failure
    )
    value1 = (primitives.revenue - primitives.c1) / (
        1.0
        - primitives.gamma * expected_waiting1
    )
    value2 = (primitives.revenue - primitives.c2) / (
        1.0
        - primitives.gamma * expected_waiting2
    )
    selected_product = 1 if value1 >= value2 else 2
    selected_value = max(value1, value2)
    continuation_gap = (
        primitives.gamma
        * (waiting_success - waiting_failure)
        * selected_value
    )
    advantage = -primitives.delta_c + primitives.delta_p * continuation_gap
    return {
        "demand_success": demand_success,
        "demand_failure": demand_failure,
        "W1": value1,
        "W2": value2,
        "selected_product": selected_product,
        "selected_value": selected_value,
        "continuation_gap": continuation_gap,
        "advantage": advantage,
    }
