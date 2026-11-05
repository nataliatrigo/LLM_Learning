"""Sparse, certified solvers for the mature-boundary fluid experiment.

The stochastic solver uses the original retention ``rho``, calendar discount
``gamma``, and sale reward ``R-c_i``.  Its demand curve and the fluid demand
curve are both evaluated at the same ``rho``.
Policies use 0 for product 1 and 1 for product 2; fixed policies may also
contain probabilities in [0, 1].  All residuals refer to the discretized
equations, and do not bound spatial discretization or model discrepancy.
"""

from __future__ import annotations

import math
from typing import Callable, Mapping

import numpy as np
from scipy.linalg import solve_banded
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import splu

from forgetting.Code.model import Primitives, boundary_demand, validate_rho


DemandFunction = Callable[[np.ndarray], np.ndarray]


def _validate(
    primitives: Primitives, rho: float, tol: float, max_iterations: int
) -> None:
    validate_rho(rho)
    if not 0.0 < primitives.gamma < 1.0:
        raise ValueError("gamma must lie strictly between zero and one")
    if not math.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    if not isinstance(max_iterations, (int, np.integer)) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")


def _uniform_grid(n: int) -> np.ndarray:
    if not isinstance(n, (int, np.integer)) or n < 3:
        raise ValueError("n must be an integer >= 3")
    return np.linspace(0.0, 1.0, n)


def _validate_grid(grid: np.ndarray) -> np.ndarray:
    grid = np.asarray(grid, dtype=float)
    if (
        grid.ndim != 1
        or grid.size < 3
        or not np.all(np.isfinite(grid))
        or grid[0] != 0.0
        or grid[-1] != 1.0
        or np.any(np.diff(grid) <= 0.0)
    ):
        raise ValueError("grid must be increasing, finite, and include 0 and 1")
    return grid


def _demand(
    primitives: Primitives,
    rho: float,
    grid: np.ndarray,
    demand_function: DemandFunction | None,
) -> np.ndarray:
    demand = (
        boundary_demand(grid, rho, primitives.p0)
        if demand_function is None
        else np.broadcast_to(np.asarray(demand_function(grid), dtype=float), grid.shape)
    )
    if not np.all(np.isfinite(demand)) or np.any((demand < 0.0) | (demand > 1.0)):
        raise ValueError("demand_function must return probabilities in [0, 1]")
    return np.asarray(demand, dtype=float)


def _policy_array(policy: np.ndarray, grid: np.ndarray) -> np.ndarray:
    policy = np.asarray(policy, dtype=float)
    if policy.shape != grid.shape or not np.all(np.isfinite(policy)):
        raise ValueError("policy must be a finite array with the same shape as grid")
    if np.any((policy < 0.0) | (policy > 1.0)):
        raise ValueError("policy is the probability of product 2 and must lie in [0, 1]")
    return policy


def _interpolation_matrix(grid: np.ndarray, targets: np.ndarray) -> csr_matrix:
    """Nonnegative row-stochastic linear interpolation, with no dense matrices."""
    lower = np.clip(np.searchsorted(grid, targets, side="right") - 1, 0, grid.size - 2)
    weight = np.clip((targets - grid[lower]) / (grid[lower + 1] - grid[lower]), 0.0, 1.0)
    rows = np.repeat(np.arange(grid.size), 2)
    columns = np.column_stack((lower, lower + 1)).ravel()
    values = np.column_stack((1.0 - weight, weight)).ravel()
    matrix = csr_matrix((values, (rows, columns)), shape=(grid.size, grid.size))
    matrix.eliminate_zeros()
    return matrix


def _kernel(
    primitives: Primitives,
    rho: float,
    grid: np.ndarray,
    demand_function: DemandFunction | None,
) -> dict:
    return {
        "grid": grid,
        "demand": _demand(primitives, rho, grid, demand_function),
        "success": _interpolation_matrix(grid, rho * grid + (1.0 - rho)),
        "failure": _interpolation_matrix(grid, rho * grid),
        "discount": primitives.gamma,
        "discount_loss": 1.0 - primitives.gamma,
        "retention": rho,
    }


def _evaluate_kernel(primitives: Primitives, kernel: Mapping, policy: np.ndarray) -> dict:
    """Solve (I-gP_pi)V=r_pi by sparse LU, then certify its residual."""
    grid, demand = kernel["grid"], kernel["demand"]
    probability = primitives.p1 + primitives.delta_p * policy
    sales_transition = (
        diags(probability) @ kernel["success"]
        + diags(1.0 - probability) @ kernel["failure"]
    ).tocsr()
    reward = demand * (
        primitives.net_rewards[0] - primitives.delta_c * policy
    )
    # The diagonal demand term includes the calendar no-sale self-loop.
    weighted_demand = kernel["discount"] * demand
    matrix = (
        diags(kernel["discount_loss"] + weighted_demand)
        - diags(weighted_demand) @ sales_transition
    ).tocsc()
    factor = splu(matrix)
    value = factor.solve(reward)
    # Sparse iterative refinement is cheap and avoids overstating accuracy.
    for _ in range(2):
        defect = reward - matrix @ value
        if float(np.max(np.abs(defect))) <= 5e-14:
            break
        candidate = value + factor.solve(defect)
        if np.max(np.abs(reward - matrix @ candidate)) >= np.max(np.abs(defect)):
            break
        value = candidate
    residual = float(np.max(np.abs(reward - matrix @ value)))
    return {
        "value": value,
        "residual": residual,
        "value_error_bound": residual / kernel["discount_loss"],
        "factor_nnz": int(factor.L.nnz + factor.U.nnz),
    }


def _stochastic_diagnostics(primitives: Primitives, kernel: Mapping, value: np.ndarray) -> dict:
    success_value = kernel["success"] @ value
    failure_value = kernel["failure"] @ value
    discount = kernel["discount"]
    q1_defect = kernel["demand"] * (
        primitives.net_rewards[0]
        + discount * (
            primitives.p1 * success_value
            + (1.0 - primitives.p1) * failure_value
            - value
        )
    ) - kernel["discount_loss"] * value
    advantage = -primitives.delta_c + discount * primitives.delta_p * (
        success_value - failure_value
    )
    # Conditional regret is measured per sale, so low demand does not conceal
    # suboptimal action choices.
    return {
        "advantage": advantage,
        "gap": advantage,
        "bellman_defect": q1_defect + kernel["demand"] * np.maximum(advantage, 0.0),
    }


def evaluate_stochastic_policy(
    primitives: Primitives,
    rho: float,
    grid: np.ndarray,
    policy: np.ndarray,
    *,
    demand_function: DemandFunction | None = None,
) -> dict:
    """Evaluate a fixed, possibly mixed policy in the original calendar kernel.

    The value-error certificate is for this policy's finite-grid linear system.
    ``optimality_residual`` and ``conditional_regret`` instead describe its
    distance from Bellman optimality and need not be small for a fluid policy.
    """
    _validate(primitives, rho, 1e-10, 1)
    grid = _validate_grid(grid)
    policy = _policy_array(policy, grid)
    kernel = _kernel(primitives, rho, grid, demand_function)
    result = _evaluate_kernel(primitives, kernel, policy)
    diagnostics = _stochastic_diagnostics(primitives, kernel, result["value"])
    advantage = diagnostics["advantage"]
    regret = np.maximum(advantage, 0.0) - policy * advantage
    result.update({
        "grid": grid,
        "x": grid,
        "policy": policy.copy(),
        "demand": kernel["demand"],
        "gap": advantage,
        "advantage": advantage,
        "optimality_residual": float(np.max(np.abs(diagnostics["bellman_defect"]))),
        "conditional_regret": float(np.max(regret)),
        "calendar_regret": float(np.max(kernel["demand"] * regret)),
        "discount": kernel["discount"],
        "discount_loss": kernel["discount_loss"],
        "retention": kernel["retention"],
        "iterations": 1,
    })
    return result


def solve_stochastic(
    primitives: Primitives,
    rho: float,
    n: int,
    tol: float = 1e-10,
    *,
    max_iterations: int = 1000,
    demand_function: DemandFunction | None = None,
) -> dict:
    """Howard iteration for the original stochastic model on a uniform grid."""
    _validate(primitives, rho, tol, max_iterations)
    grid = _uniform_grid(n)
    kernel = _kernel(primitives, rho, grid, demand_function)
    policy = np.zeros(n, dtype=np.int8)
    switch_tolerance = tol * 0.5
    for iteration in range(1, max_iterations + 1):
        evaluation = _evaluate_kernel(primitives, kernel, policy)
        diagnostics = _stochastic_diagnostics(primitives, kernel, evaluation["value"])
        conditional_gap = diagnostics["advantage"]
        residual = float(np.max(np.abs(diagnostics["bellman_defect"])))
        updated = np.where(
            conditional_gap > switch_tolerance, 1,
            np.where(conditional_gap < -switch_tolerance, 0, policy),
        ).astype(np.int8)
        if np.array_equal(updated, policy):
            if residual > max(tol, 1e-12):
                raise RuntimeError(f"stochastic Howard stopped with residual {residual:.3e}")
            break
        policy = updated
    else:
        raise RuntimeError("stochastic Howard iteration did not converge")
    advantage = diagnostics["advantage"]
    regret = np.maximum(advantage, 0.0) - policy * advantage
    return {
        **evaluation,
        "grid": grid,
        "x": grid,
        "value": evaluation["value"],
        "policy": policy,
        "gap": advantage,
        "advantage": advantage,
        "demand": kernel["demand"],
        "residual": residual,
        "policy_evaluation_residual": evaluation["residual"],
        "value_error_bound": residual / kernel["discount_loss"],
        "conditional_regret": float(np.max(regret)),
        "calendar_regret": float(np.max(kernel["demand"] * regret)),
        "iterations": iteration,
        "discount": kernel["discount"],
        "discount_loss": kernel["discount_loss"],
        "retention": kernel["retention"],
    }


def solve_fluid(
    primitives: Primitives,
    rho: float,
    n: int,
    tol: float = 1e-10,
    *,
    max_iterations: int = 20_000,
    demand_function: DemandFunction | None = None,
) -> dict:
    """Monotone upwind Howard solver for delta*v=D*max_i(r_i+lambda*(p_i-x)*v').

    Each action uses its own upwind derivative.  The banded fixed-policy
    systems are M-matrices; inward drifts require no imposed endpoint values.
    """
    _validate(primitives, rho, tol, max_iterations)
    grid = _uniform_grid(n)
    demand = _demand(primitives, rho, grid, demand_function)
    spacing = 1.0 / (n - 1)
    lam, delta = -math.log(rho), -math.log(primitives.gamma)
    rewards = np.asarray(primitives.net_rewards)[:, None]
    velocity = lam * (np.asarray([primitives.p1, primitives.p2])[:, None] - grid)
    drift = demand * velocity
    right = np.maximum(drift, 0.0) / spacing
    left = np.maximum(-drift, 0.0) / spacing
    right[:, -1] = 0.0
    left[:, 0] = 0.0
    policy = np.zeros(n, dtype=np.int8)
    indices = np.arange(n)
    switch_tolerance = tol * 0.5
    for iteration in range(1, max_iterations + 1):
        rr, ll = right[policy, indices], left[policy, indices]
        banded = np.zeros((3, n))
        banded[0, 1:] = -rr[:-1]
        banded[1] = delta + rr + ll
        banded[2, :-1] = -ll[1:]
        value = solve_banded((1, 1), banded, demand * rewards[policy, 0])
        forward = np.r_[np.diff(value) / spacing, 0.0]
        backward = np.r_[0.0, np.diff(value) / spacing]
        slopes = np.where(velocity >= 0.0, forward, backward)
        hamiltonian = rewards + velocity * slopes
        gap = hamiltonian[1] - hamiltonian[0]
        residual = float(np.max(np.abs(delta * value - demand * hamiltonian.max(axis=0))))
        updated = np.where(
            gap > switch_tolerance, 1,
            np.where(gap < -switch_tolerance, 0, policy),
        ).astype(np.int8)
        if np.array_equal(updated, policy):
            if residual > max(tol, 1e-11):
                raise RuntimeError(f"fluid Howard stopped with residual {residual:.3e}")
            break
        policy = updated
    else:
        raise RuntimeError("fluid Howard iteration did not converge")
    changes = np.flatnonzero(policy[1:] != policy[:-1])
    roots = []
    for index in changes:
        difference = gap[index + 1] - gap[index]
        fraction = -gap[index] / difference if difference != 0.0 else 0.5
        roots.append(float(grid[index] + spacing * np.clip(fraction, 0.0, 1.0)))
    starts, ends = np.r_[0, changes + 1], np.r_[changes, n - 1]
    bands = [
        [float(grid[start]), float(grid[end])]
        for start, end in zip(starts, ends) if policy[start] == 1
    ]
    conditional_regret = np.maximum(gap, 0.0) - policy * gap
    return {
        "grid": grid,
        "x": grid,
        "value": value,
        "gap": gap,
        "policy": policy,
        "demand": demand,
        "roots": roots,
        "bands": bands,
        "residual": residual,
        "value_error_bound": residual / delta,
        "conditional_regret": float(np.max(conditional_regret)),
        "iterations": iteration,
        "lambda": lam,
        "delta": delta,
    }


def transfer_policy(
    fluid: Mapping,
    grid: np.ndarray,
    action_tolerance: float = 1e-8,
) -> np.ndarray:
    """Transfer the fluid Hamiltonian gap, with ties assigned to product 1.

    Interpolating binary actions would introduce artificial mixed actions;
    this function instead interpolates the Hamiltonian difference and applies
    the explicit strict threshold ``gap > action_tolerance``.
    """
    grid = _validate_grid(grid)
    if not math.isfinite(action_tolerance) or action_tolerance < 0.0:
        raise ValueError("action_tolerance must be finite and nonnegative")
    source_grid = np.asarray(fluid["grid"] if "grid" in fluid else fluid["x"])
    gap = np.interp(grid, source_grid, np.asarray(fluid["gap"]))
    return (gap > action_tolerance).astype(np.int8)
