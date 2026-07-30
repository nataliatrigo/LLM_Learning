"""Fast Bellman and grid-convergence tests for forgetting solvers."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp.continuous_solver import (  # noqa: E402
    ContinuousSolverConfig,
    compare_nested_solutions,
    solve_continuous,
)
from modular_dp.demand_models import (  # noqa: E402
    CalendarForgettingTS,
    ObservationForgettingTS,
)
from modular_dp.primitives import SellerPrimitives  # noqa: E402


class _IncorrectSelfLoopDeclaration(ObservationForgettingTS):
    """Origin is fixed, but idle is not a global identity transition."""

    def idle_state(self, state):  # type: ignore[no-untyped-def]
        x, y = self.state_coordinates(state)
        return self.rho * x, self.rho * y


class ContinuousSolverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fast_config = ContinuousSolverConfig(
            grid_size=9,
            tolerance=1e-8,
            max_iterations=5_000,
        )

    def test_observation_forgetting_uses_exact_self_loop_equation(self) -> None:
        model = ObservationForgettingTS(0.9)
        result = solve_continuous(model, self.fast_config)
        self.assertTrue(result.converged)
        self.assertEqual(result.bellman_form, "exact_idle_self_loop_rearranged")
        self.assertLess(result.maximum_bellman_residual, 1.1e-8)
        np.testing.assert_array_equal(
            result.stencils.idle.apply(result.value), result.value
        )

    def test_calendar_origin_does_not_trigger_self_loop_rearrangement(self) -> None:
        model = CalendarForgettingTS(0.9)
        self.assertEqual(model.idle_state((0.0, 0.0)), (0.0, 0.0))
        result = solve_continuous(model, self.fast_config)
        self.assertTrue(result.converged)
        self.assertEqual(result.bellman_form, "generic_idle_transition")
        self.assertLess(result.maximum_bellman_residual, 1.1e-8)

        # The origin's local stencil is indeed a self-map.  The solver still
        # uses the generic operator because the model-level property is false.
        origin_idle_value = result.stencils.idle.apply(result.value)[0]
        self.assertEqual(origin_idle_value, result.value[0])

    def test_false_global_self_loop_declaration_is_rejected(self) -> None:
        model = _IncorrectSelfLoopDeclaration(0.9)
        self.assertEqual(model.idle_state((0.0, 0.0)), (0.0, 0.0))
        with self.assertRaisesRegex(ValueError, "not the identity"):
            solve_continuous(model, self.fast_config)

    def test_solver_rejects_mismatched_primitives(self) -> None:
        model_primitives = SellerPrimitives(p0=0.4)
        solver_primitives = SellerPrimitives(p0=0.6)
        model = ObservationForgettingTS(0.9, model_primitives)
        with self.assertRaisesRegex(ValueError, "primitives"):
            solve_continuous(model, solver_primitives, self.fast_config)

    def test_reported_residual_uses_original_bellman_equation(self) -> None:
        result = solve_continuous(ObservationForgettingTS(0.85), self.fast_config)
        primitives = result.primitives
        value_success = result.stencils.success.apply(result.value)
        value_failure = result.stencils.failure.apply(result.value)
        value_idle = result.stencils.idle.apply(result.value)
        q1 = primitives.revenue - primitives.c1 + primitives.gamma * (
            primitives.p1 * value_success
            + (1.0 - primitives.p1) * value_failure
        )
        q2 = primitives.revenue - primitives.c2 + primitives.gamma * (
            primitives.p2 * value_success
            + (1.0 - primitives.p2) * value_failure
        )
        rhs = (
            primitives.gamma * (1.0 - result.demand) * value_idle
            + result.demand * np.maximum(q1, q2)
        )
        np.testing.assert_allclose(
            result.bellman_residual,
            result.value - rhs,
            rtol=0.0,
            atol=2e-15,
        )

    def test_raw_and_robust_action_classifications_are_consistent(self) -> None:
        result = solve_continuous(ObservationForgettingTS(0.9), self.fast_config)
        np.testing.assert_array_equal(result.action2, result.advantage > 0.0)
        np.testing.assert_array_equal(
            result.robust_action2,
            result.advantage > result.action_tolerance,
        )
        np.testing.assert_array_equal(
            result.tie,
            np.abs(result.advantage) <= result.action_tolerance,
        )
        evaluation = result.evaluate_state((0.13, 0.17))
        self.assertEqual(evaluation.action2, evaluation.advantage > 0.0)

    def test_nested_resolution_comparison_on_shared_nodes(self) -> None:
        # gamma=0 isolates the grid plumbing: values and policies at shared
        # nodes must be exactly stable because continuation interpolation is
        # irrelevant.
        primitives = SellerPrimitives(gamma=0.0)
        model = ObservationForgettingTS(0.9, primitives)
        coarse = solve_continuous(
            model,
            ContinuousSolverConfig(grid_size=5, tolerance=1e-12),
        )
        fine = solve_continuous(
            model,
            ContinuousSolverConfig(grid_size=9, tolerance=1e-12),
        )
        comparison = compare_nested_solutions(coarse, fine)
        self.assertEqual(comparison.common_node_count, coarse.grid.node_count)
        self.assertEqual(comparison.nesting_stride, 2)
        self.assertLess(comparison.maximum_value_difference, 1e-14)
        self.assertEqual(comparison.raw_action_changes, 0)
        self.assertEqual(comparison.action_changes, 0)

    def test_off_grid_evaluation_is_finite(self) -> None:
        result = solve_continuous(CalendarForgettingTS(0.8), self.fast_config)
        evaluation = result.evaluate_state((0.21, 0.33))
        for value in (
            evaluation.value,
            evaluation.demand,
            evaluation.gap,
            evaluation.advantage,
            evaluation.bellman_rhs,
            evaluation.bellman_residual,
        ):
            self.assertTrue(np.isfinite(value))
        self.assertGreaterEqual(evaluation.demand, 0.0)
        self.assertLessEqual(evaluation.demand, 1.0)


if __name__ == "__main__":
    unittest.main()
