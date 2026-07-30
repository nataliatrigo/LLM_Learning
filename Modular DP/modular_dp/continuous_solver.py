"""Interpolated dynamic program for the two continuous forgetting models.

Observation-time forgetting has a genuine idle self-loop at every state and
uses the exact algebraic rearrangement of that term.  Calendar-time forgetting
does not: even though its idle transition fixes the origin, it is solved with
the generic Bellman operator everywhere.  This distinction is controlled by
the model's *global* ``has_exact_idle_self_loop`` contract, never by testing a
single state.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

from .interpolation import InterpolationStencil, TriangularGrid
from .primitives import SellerPrimitives


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ContinuousSolverConfig:
    """Grid and convergence controls for continuous-state value iteration."""

    grid_size: int = 201
    tolerance: float = 1e-10
    max_iterations: int = 50_000
    initial_value: float = 0.0
    action_atol: float = 1e-10
    action_rtol: float = 1e-8

    def __post_init__(self) -> None:
        if isinstance(self.grid_size, bool) or int(self.grid_size) != self.grid_size:
            raise ValueError("grid_size must be an integer")
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least two")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be finite and strictly positive")
        if (
            isinstance(self.max_iterations, bool)
            or int(self.max_iterations) != self.max_iterations
            or self.max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if not np.isfinite(self.initial_value):
            raise ValueError("initial_value must be finite")
        if self.action_atol < 0.0 or self.action_rtol < 0.0:
            raise ValueError("action tolerances must be nonnegative")


@dataclass(frozen=True, slots=True)
class ContinuousTransitionStencils:
    """Precomputed successor interpolation for all grid nodes."""

    success: InterpolationStencil
    failure: InterpolationStencil
    idle: InterpolationStencil


@dataclass(frozen=True, slots=True)
class OffGridEvaluation:
    """Interpolated value and product choice at one normalized state."""

    state: tuple[float, float]
    value: float
    demand: float
    gap: float
    advantage: float
    action2: bool
    robust_action2: bool
    tie: bool
    q1: float
    q2: float
    bellman_rhs: float
    bellman_residual: float


@dataclass(slots=True)
class ContinuousDPSolution:
    """Solution and diagnostics on one triangular normalized-state grid."""

    model: Any
    primitives: SellerPrimitives
    config: ContinuousSolverConfig
    grid: TriangularGrid
    value: FloatArray
    demand: FloatArray
    gap: FloatArray
    advantage: FloatArray
    action2: BoolArray
    robust_action2: BoolArray
    tie: BoolArray
    q1: FloatArray
    q2: FloatArray
    bellman_residual: FloatArray
    iteration_residual: float
    iterations: int
    converged: bool
    bellman_form: str
    stencils: ContinuousTransitionStencils
    action_tolerance: float
    metadata: dict[str, Any]

    @property
    def maximum_bellman_residual(self) -> float:
        return float(np.max(np.abs(self.bellman_residual)))

    @property
    def residual(self) -> FloatArray:
        """Compatibility alias emphasizing that this is the original equation."""

        return self.bellman_residual

    @property
    def grid_size(self) -> int:
        return self.grid.n

    @property
    def optimal_action(self) -> NDArray[np.int8]:
        return np.where(self.action2, 2, 1).astype(np.int8)

    @property
    def robust_optimal_action(self) -> NDArray[np.int8]:
        """Tolerance-filtered policy used in grid-convergence comparisons."""

        return np.where(self.robust_action2, 2, 1).astype(np.int8)

    def interpolate_value(self, state: Sequence[float]) -> float:
        x, y = _state_coordinates(self.model, state)
        return float(self.grid.interpolate(self.value, x, y))

    def evaluate_state(self, state: Sequence[float]) -> OffGridEvaluation:
        """Evaluate an off-grid action using interpolated successor values."""

        x, y = _state_coordinates(self.model, state)
        normalized_state = (x, y)
        value = float(self.grid.interpolate(self.value, x, y))
        success = _state_coordinates(
            self.model, self.model.success_state(normalized_state)
        )
        failure = _state_coordinates(
            self.model, self.model.failure_state(normalized_state)
        )
        idle = _state_coordinates(self.model, self.model.idle_state(normalized_state))
        value_success = float(
            self.grid.interpolate(self.value, success[0], success[1])
        )
        value_failure = float(
            self.grid.interpolate(self.value, failure[0], failure[1])
        )
        value_idle = float(self.grid.interpolate(self.value, idle[0], idle[1]))
        gap = self.primitives.gamma * (value_success - value_failure)
        q1 = self.primitives.revenue - self.primitives.c1 + self.primitives.gamma * (
            self.primitives.p1 * value_success
            + (1.0 - self.primitives.p1) * value_failure
        )
        q2 = self.primitives.revenue - self.primitives.c2 + self.primitives.gamma * (
            self.primitives.p2 * value_success
            + (1.0 - self.primitives.p2) * value_failure
        )
        advantage = q2 - q1
        demand = float(self.model.demand(normalized_state))
        bellman_rhs = (
            self.primitives.gamma * (1.0 - demand) * value_idle
            + demand * max(q1, q2)
        )
        return OffGridEvaluation(
            state=normalized_state,
            value=value,
            demand=demand,
            gap=float(gap),
            advantage=float(advantage),
            action2=bool(advantage > 0.0),
            robust_action2=bool(advantage > self.action_tolerance),
            tie=bool(abs(advantage) <= self.action_tolerance),
            q1=float(q1),
            q2=float(q2),
            bellman_rhs=float(bellman_rhs),
            bellman_residual=float(value - bellman_rhs),
        )


@dataclass(frozen=True, slots=True)
class ContinuousGridComparison:
    """Common-node comparison of two nested continuous-grid solutions."""

    coarse_grid_size: int
    fine_grid_size: int
    common_node_count: int
    nesting_stride: int
    maximum_value_difference: float
    root_mean_square_value_difference: float
    maximum_advantage_difference: float
    raw_action_changes: int
    raw_action_change_share: float
    action_changes: int
    action_change_share: float
    coarse_bellman_residual: float
    fine_bellman_residual: float

    @property
    def robust_action_changes(self) -> int:
        """Explicit alias: ``action_changes`` uses robust classifications."""

        return self.action_changes

    @property
    def robust_action_change_share(self) -> float:
        return self.action_change_share

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _state_coordinates(model: Any, state: Sequence[float]) -> tuple[float, float]:
    coordinates = model.state_coordinates(state)
    if len(coordinates) != 2:
        raise ValueError("state_coordinates must return a pair")
    x, y = float(coordinates[0]), float(coordinates[1])
    tolerance = 1e-12
    if (
        not np.isfinite(x)
        or not np.isfinite(y)
        or x < -tolerance
        or y < -tolerance
        or x + y > 1.0 + tolerance
    ):
        raise ValueError(f"transition returned state outside simplex: {(x, y)}")
    x, y = max(x, 0.0), max(y, 0.0)
    if x + y > 1.0:
        total = x + y
        x, y = x / total, y / total
    return x, y


def _transition_arrays(
    model: Any,
    transition_name: str,
    grid: TriangularGrid,
) -> tuple[FloatArray, FloatArray]:
    """Apply a model transition once per node during stencil construction."""

    array_method = getattr(model, f"{transition_name}_array", None)
    if callable(array_method):
        transitioned = array_method(grid.x, grid.y)
        if len(transitioned) != 2:
            raise ValueError(f"{transition_name}_array must return two arrays")
        x_next, y_next = np.broadcast_arrays(
            np.asarray(transitioned[0], dtype=float),
            np.asarray(transitioned[1], dtype=float),
        )
        if x_next.shape != (grid.node_count,):
            raise ValueError(
                f"{transition_name}_array returned shape {x_next.shape}; "
                f"expected {(grid.node_count,)}"
            )
        return x_next.copy(), y_next.copy()

    transition: Callable[[Sequence[float]], Sequence[float]] = getattr(
        model, transition_name
    )
    x_next = np.empty(grid.node_count, dtype=float)
    y_next = np.empty(grid.node_count, dtype=float)
    for index, (x, y) in enumerate(zip(grid.x, grid.y, strict=True)):
        x_next[index], y_next[index] = _state_coordinates(
            model, transition((float(x), float(y)))
        )
    return x_next, y_next


def _demand_on_grid(model: Any, grid: TriangularGrid) -> FloatArray:
    if hasattr(model, "demand_array"):
        demand = np.asarray(model.demand_array(grid.x, grid.y), dtype=float)
    else:
        demand = np.fromiter(
            (model.demand((float(x), float(y))) for x, y in zip(grid.x, grid.y)),
            dtype=float,
            count=grid.node_count,
        )
    if demand.shape != (grid.node_count,):
        raise ValueError(
            f"demand_array returned shape {demand.shape}; "
            f"expected {(grid.node_count,)}"
        )
    if not np.all(np.isfinite(demand)):
        raise ValueError("demand contains non-finite values")
    if np.any((demand < -1e-13) | (demand > 1.0 + 1e-13)):
        raise ValueError("demand must lie in [0,1]")
    return np.clip(demand, 0.0, 1.0)


def _model_metadata(model: Any) -> dict[str, Any]:
    metadata = getattr(model, "metadata", {})
    if callable(metadata):
        metadata = metadata()
    return dict(metadata)


def _precompute_stencils(
    model: Any, grid: TriangularGrid
) -> tuple[ContinuousTransitionStencils, bool]:
    success_x, success_y = _transition_arrays(model, "success_state", grid)
    failure_x, failure_y = _transition_arrays(model, "failure_state", grid)
    idle_x, idle_y = _transition_arrays(model, "idle_state", grid)

    declared_self_loop = bool(
        getattr(model, "has_exact_idle_self_loop", False)
    )
    if declared_self_loop:
        # The property is a global model contract.  Audit it at every node;
        # never infer it merely because a special state (notably the calendar
        # model's origin) happens to be fixed by the idle transition.
        audit_tolerance = 256.0 * np.finfo(float).eps
        if not (
            np.allclose(idle_x, grid.x, rtol=0.0, atol=audit_tolerance)
            and np.allclose(idle_y, grid.y, rtol=0.0, atol=audit_tolerance)
        ):
            raise ValueError(
                "model declares has_exact_idle_self_loop=True but idle_state "
                "is not the identity on the grid"
            )
        idle_stencil = grid.identity_stencil()
    else:
        idle_stencil = grid.stencil(idle_x, idle_y)

    return (
        ContinuousTransitionStencils(
            success=grid.stencil(success_x, success_y),
            failure=grid.stencil(failure_x, failure_y),
            idle=idle_stencil,
        ),
        declared_self_loop,
    )


def _q_values(
    value: FloatArray,
    stencils: ContinuousTransitionStencils,
    primitives: SellerPrimitives,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    value_success = np.asarray(stencils.success.apply(value), dtype=float)
    value_failure = np.asarray(stencils.failure.apply(value), dtype=float)
    q1 = primitives.revenue - primitives.c1 + primitives.gamma * (
        primitives.p1 * value_success
        + (1.0 - primitives.p1) * value_failure
    )
    q2 = primitives.revenue - primitives.c2 + primitives.gamma * (
        primitives.p2 * value_success
        + (1.0 - primitives.p2) * value_failure
    )
    return q1, q2, value_success, value_failure


def solve_continuous(
    model: Any,
    primitives: SellerPrimitives | ContinuousSolverConfig | None = None,
    config: ContinuousSolverConfig | None = None,
) -> ContinuousDPSolution:
    """Solve a normalized continuous-state forgetting model.

    The preferred call is ``solve_continuous(model, primitives, config)``.
    ``solve_continuous(model, config)`` uses ``model.primitives`` for
    convenience and mirrors the discrete solver's public API.
    """

    if isinstance(primitives, ContinuousSolverConfig) and config is None:
        config = primitives
        primitives = None
    if config is None:
        raise TypeError("a ContinuousSolverConfig is required")
    if primitives is None:
        primitives = getattr(model, "primitives", None)
    if primitives is None:
        raise TypeError("SellerPrimitives must be passed explicitly")
    if not isinstance(primitives, SellerPrimitives):
        raise TypeError("primitives must be a SellerPrimitives instance")
    model_primitives = getattr(model, "primitives", None)
    if model_primitives is not None and model_primitives != primitives:
        raise ValueError(
            "model.primitives and the explicitly supplied SellerPrimitives differ"
        )

    grid = TriangularGrid(config.grid_size)
    demand = _demand_on_grid(model, grid)
    stencils, exact_self_loop = _precompute_stencils(model, grid)
    gamma = primitives.gamma

    if exact_self_loop:
        denominator = 1.0 - gamma * (1.0 - demand)
        if np.any(denominator <= 0.0):
            raise FloatingPointError("nonpositive exact self-loop denominator")
        bellman_form = "exact_idle_self_loop_rearranged"
    else:
        denominator = None
        bellman_form = "generic_idle_transition"

    value = np.full(grid.node_count, config.initial_value, dtype=float)
    iteration_residual = np.inf
    converged = False
    iterations = 0

    for iterations in range(1, config.max_iterations + 1):
        q1, q2, _, _ = _q_values(value, stencils, primitives)
        active_value = np.maximum(q1, q2)
        if exact_self_loop:
            assert denominator is not None
            updated_value = demand * active_value / denominator
        else:
            idle_value = np.asarray(stencils.idle.apply(value), dtype=float)
            updated_value = (
                gamma * (1.0 - demand) * idle_value
                + demand * active_value
            )
        if not np.all(np.isfinite(updated_value)):
            raise FloatingPointError("non-finite value iterate")
        iteration_residual = float(np.max(np.abs(updated_value - value)))
        value = updated_value
        if iteration_residual <= config.tolerance:
            converged = True
            break

    q1, q2, value_success, value_failure = _q_values(value, stencils, primitives)
    gap = gamma * (value_success - value_failure)
    advantage = q2 - q1
    advantage_scale = max(1.0, float(np.max(np.abs(advantage))))
    action_tolerance = max(
        config.action_atol, config.action_rtol * advantage_scale
    )
    # Preserve the mathematical policy separately from a tolerance-filtered
    # classification.  Simulations use ``action2``; convergence diagnostics
    # use ``robust_action2`` so numerical noise at exact ties is not counted as
    # an economically meaningful policy change.
    action2 = advantage > 0.0
    robust_action2 = advantage > action_tolerance
    tie = np.abs(advantage) <= action_tolerance

    # Always evaluate the unrearranged economic Bellman equation.  This makes
    # residuals comparable between solver families and catches an accidental
    # use of the self-loop formula for calendar-time forgetting.
    idle_value = np.asarray(stencils.idle.apply(value), dtype=float)
    original_rhs = (
        gamma * (1.0 - demand) * idle_value + demand * np.maximum(q1, q2)
    )
    bellman_residual = value - original_rhs

    metadata = {
        **_model_metadata(model),
        "solver_type": "continuous_triangular_interpolation",
        "grid_size": grid.n,
        "node_count": grid.node_count,
        "grid_spacing": grid.spacing,
        "cells_per_transition": (
            float((1.0 - float(model.rho)) / grid.spacing)
            if hasattr(model, "rho")
            else float("nan")
        ),
        "interpolation": "piecewise_linear_barycentric",
        "bellman_form": bellman_form,
        "tolerance": config.tolerance,
        "max_iterations": config.max_iterations,
        "iterations": iterations,
        "iteration_residual": iteration_residual,
        "maximum_bellman_residual": float(
            np.max(np.abs(bellman_residual))
        ),
        "converged": converged,
    }
    return ContinuousDPSolution(
        model=model,
        primitives=primitives,
        config=config,
        grid=grid,
        value=value,
        demand=demand,
        gap=gap,
        advantage=advantage,
        action2=action2,
        robust_action2=robust_action2,
        tie=tie,
        q1=q1,
        q2=q2,
        bellman_residual=bellman_residual,
        iteration_residual=iteration_residual,
        iterations=iterations,
        converged=converged,
        bellman_form=bellman_form,
        stencils=stencils,
        action_tolerance=action_tolerance,
        metadata=metadata,
    )


def compare_nested_solutions(
    coarse: ContinuousDPSolution,
    fine: ContinuousDPSolution,
) -> ContinuousGridComparison:
    """Compare value and policy at exactly shared nodes of nested grids."""

    if not coarse.grid.is_nested_in(fine.grid):
        raise ValueError(
            "solutions must be ordered coarse-to-fine with nested resolutions"
        )
    coarse_method = coarse.metadata.get("method")
    fine_method = fine.metadata.get("method")
    coarse_parameter = coarse.metadata.get("parameter_value")
    fine_parameter = fine.metadata.get("parameter_value")
    if (coarse_method, coarse_parameter) != (fine_method, fine_parameter):
        raise ValueError("nested solutions must use the same demand model")
    if coarse.primitives != fine.primitives:
        raise ValueError("nested solutions must use the same seller primitives")

    fine_indices = coarse.grid.indices_in(fine.grid)
    value_difference = coarse.value - fine.value[fine_indices]
    advantage_difference = coarse.advantage - fine.advantage[fine_indices]
    raw_action_changes = int(
        np.count_nonzero(coarse.action2 != fine.action2[fine_indices])
    )
    action_changes = int(
        np.count_nonzero(
            coarse.robust_action2 != fine.robust_action2[fine_indices]
        )
    )
    stride = (fine.grid.n - 1) // (coarse.grid.n - 1)
    return ContinuousGridComparison(
        coarse_grid_size=coarse.grid.n,
        fine_grid_size=fine.grid.n,
        common_node_count=coarse.grid.node_count,
        nesting_stride=stride,
        maximum_value_difference=float(np.max(np.abs(value_difference))),
        root_mean_square_value_difference=float(
            np.sqrt(np.mean(np.square(value_difference)))
        ),
        maximum_advantage_difference=float(
            np.max(np.abs(advantage_difference))
        ),
        raw_action_changes=raw_action_changes,
        raw_action_change_share=raw_action_changes / coarse.grid.node_count,
        action_changes=action_changes,
        action_change_share=action_changes / coarse.grid.node_count,
        coarse_bellman_residual=coarse.maximum_bellman_residual,
        fine_bellman_residual=fine.maximum_bellman_residual,
    )


# A descriptive alias used by experiment scripts and reports.
compare_continuous_grids = compare_nested_solutions


__all__ = [
    "ContinuousDPSolution",
    "ContinuousGridComparison",
    "ContinuousSolverConfig",
    "ContinuousTransitionStencils",
    "OffGridEvaluation",
    "compare_continuous_grids",
    "compare_nested_solutions",
    "solve_continuous",
]
