"""Triangular-state solver for stationary, non-forgetting demand rules.

The implementation deliberately mirrors the recursion in
``discounted/paper_numerics/stationary_solver.py``.  The only model-specific
object is the demand rule; seller primitives and the Bellman algebra are
shared by every discrete experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .primitives import SellerPrimitives


Layer = dict[str, np.ndarray]


@dataclass(frozen=True)
class DiscreteSolverConfig:
    """Numerical controls for the backwards triangular truncation.

    ``boundary_value`` is assigned to diagonal ``outer_diagonal + 1``.  The
    baseline uses zero, just like the paper's existing solver.  Actions are
    classified robustly as product 2 only when its advantage exceeds the
    stated absolute/relative tolerance; states inside the tolerance are ties.
    """

    outer_diagonal: int
    report_diagonal: int
    boundary_value: float = 0.0
    action_atol: float = 1e-10
    action_rtol: float = 1e-8

    def __post_init__(self) -> None:
        if self.outer_diagonal < 1:
            raise ValueError("outer_diagonal must be at least one")
        if self.report_diagonal < 0:
            raise ValueError("report_diagonal must be nonnegative")
        if self.report_diagonal >= self.outer_diagonal:
            raise ValueError("report_diagonal must be below outer_diagonal")
        if not np.isfinite(self.boundary_value):
            raise ValueError("boundary_value must be finite")
        if self.action_atol < 0.0 or self.action_rtol < 0.0:
            raise ValueError("action tolerances must be nonnegative")


@dataclass
class DiscreteSolution:
    """A reported interior of a triangular stationary DP solution."""

    model: Any
    primitives: SellerPrimitives
    config: DiscreteSolverConfig
    layers: dict[int, Layer]
    maximum_bellman_residual: float

    @property
    def outer_diagonal(self) -> int:
        return self.config.outer_diagonal

    @property
    def report_diagonal(self) -> int:
        return self.config.report_diagonal

    @property
    def parameters(self) -> SellerPrimitives:
        """Compatibility alias for the existing paper solver."""

        return self.primitives

    @property
    def metadata(self) -> dict[str, Any]:
        model_metadata = getattr(self.model, "metadata", {})
        if callable(model_metadata):
            model_metadata = model_metadata()
        return {
            **dict(model_metadata),
            "solver_type": "discrete_triangular_truncation",
            "outer_diagonal": self.outer_diagonal,
            "report_diagonal": self.report_diagonal,
            "boundary_value": self.config.boundary_value,
        }

    def as_legacy_dict(self) -> dict[str, Any]:
        """Return the keys used by the original stationary solver."""

        return {
            "parameters": self.primitives,
            "outer_diagonal": self.outer_diagonal,
            "report_diagonal": self.report_diagonal,
            "layers": self.layers,
            "maximum_bellman_residual": self.maximum_bellman_residual,
        }


def _demand_layer(model: Any, n: int) -> np.ndarray:
    successes = np.arange(n + 1, dtype=float)
    failures = n - successes
    if hasattr(model, "demand_array"):
        demand = np.asarray(model.demand_array(successes, failures), dtype=float)
    else:
        demand = np.fromiter(
            (model.demand((int(s), int(n - s))) for s in range(n + 1)),
            dtype=float,
            count=n + 1,
        )
    if demand.shape != (n + 1,):
        raise ValueError(
            f"demand_array returned shape {demand.shape}; expected {(n + 1,)}"
        )
    if not np.all(np.isfinite(demand)):
        raise ValueError(f"non-finite demand on diagonal {n}")
    if np.any((demand < -1e-13) | (demand > 1.0 + 1e-13)):
        raise ValueError(f"demand outside [0,1] on diagonal {n}")
    return np.clip(demand, 0.0, 1.0)


def _layer_action_tolerance(
    advantage: np.ndarray, config: DiscreteSolverConfig
) -> float:
    scale = max(1.0, float(np.max(np.abs(advantage))))
    return max(config.action_atol, config.action_rtol * scale)


def solve_discrete(
    model: Any,
    primitives: SellerPrimitives | DiscreteSolverConfig | None = None,
    config: DiscreteSolverConfig | None = None,
) -> DiscreteSolution:
    """Solve a stationary integer-state model by backwards diagonals.

    This solver is valid only when an idle period leaves the belief state
    exactly unchanged.  It therefore refuses calendar-time forgetting (which
    belongs in the continuous generic Bellman solver) rather than silently
    applying the self-loop rearrangement to the wrong model.
    """

    # Preferred public call: solve_discrete(model, primitives, config).  The
    # two-argument form is retained for convenience when a model was built
    # with a ``primitives`` reference solely to obtain p0.
    if isinstance(primitives, DiscreteSolverConfig) and config is None:
        config = primitives
        primitives = None
    if config is None:
        raise TypeError("a DiscreteSolverConfig is required")
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

    if not bool(getattr(model, "has_exact_idle_self_loop", False)):
        raise ValueError(
            "the triangular recursion requires has_exact_idle_self_loop=True"
        )
    for audit_state in ((0, 0), (1, 0), (0, 1), (2, 3)):
        if tuple(model.idle_state(audit_state)) != audit_state:
            raise ValueError(
                "model declares a self-loop but idle_state(state) differs"
            )

    gamma = float(primitives.gamma)
    if not 0.0 <= gamma < 1.0:
        raise ValueError("gamma must lie in [0,1)")

    next_value = np.full(
        config.outer_diagonal + 2, config.boundary_value, dtype=float
    )
    layers: dict[int, Layer] = {}
    maximum_residual = 0.0

    for n in range(config.outer_diagonal, -1, -1):
        demand = _demand_layer(model, n)
        denominator = 1.0 - gamma * (1.0 - demand)
        if np.any(denominator <= 0.0):
            raise FloatingPointError("nonpositive exact self-loop denominator")

        continuation_success = next_value[1:]
        continuation_failure = next_value[:-1]
        q1 = (
            primitives.revenue
            - primitives.c1
            + gamma
            * (
                primitives.p1 * continuation_success
                + (1.0 - primitives.p1) * continuation_failure
            )
        )
        q2 = (
            primitives.revenue
            - primitives.c2
            + gamma
            * (
                primitives.p2 * continuation_success
                + (1.0 - primitives.p2) * continuation_failure
            )
        )
        value = demand * np.maximum(q1, q2) / denominator

        if n <= config.report_diagonal:
            gap = gamma * (continuation_success - continuation_failure)
            # This algebraic form is exactly q2-q1 and matches the paper code.
            advantage = (primitives.p2 - primitives.p1) * (
                gap - primitives.threshold
            )
            tolerance = _layer_action_tolerance(advantage, config)
            # ``action2`` is the mathematical policy (with an exact tie sent
            # to product 1).  The robust classification is kept separately
            # for convergence/localization diagnostics.
            action2 = advantage > 0.0
            robust_action2 = advantage > tolerance
            tie = np.abs(advantage) <= tolerance
            rhs = gamma * (1.0 - demand) * value + demand * np.maximum(q1, q2)
            maximum_residual = max(
                maximum_residual, float(np.max(np.abs(value - rhs)))
            )
            layers[n] = {
                "value": value.copy(),
                "demand": demand.copy(),
                "gap": gap.copy(),
                "advantage": advantage.copy(),
                "action2": action2.copy(),
                "robust_action2": robust_action2.copy(),
                "tie": tie.copy(),
                "q1": q1.copy(),
                "q2": q2.copy(),
                "action_tolerance": np.full(n + 1, tolerance, dtype=float),
            }
        next_value = value

    return DiscreteSolution(
        model=model,
        primitives=primitives,
        config=config,
        layers=layers,
        maximum_bellman_residual=maximum_residual,
    )


def solve_discrete_truncation(
    model: Any,
    outer_diagonal: int,
    report_diagonal: int,
    primitives: SellerPrimitives | None = None,
    **config_kwargs: Any,
) -> DiscreteSolution:
    """Convenience wrapper mirroring the legacy solver's call signature."""

    config = DiscreteSolverConfig(
        outer_diagonal=outer_diagonal,
        report_diagonal=report_diagonal,
        **config_kwargs,
    )
    return solve_discrete(model, primitives, config)
