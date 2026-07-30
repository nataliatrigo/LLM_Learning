"""Accounting, reproducibility, and domain-safety tests for simulations."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp.continuous_solver import ContinuousSolverConfig, solve_continuous  # noqa: E402
from modular_dp.demand_models import (  # noqa: E402
    CalendarForgettingTS,
    ScaledUpdates,
    StandardThompsonSampling,
)
from modular_dp.discrete_solver import DiscreteSolverConfig, solve_discrete  # noqa: E402
from modular_dp.primitives import SellerPrimitives  # noqa: E402
from modular_dp.simulation import (  # noqa: E402
    LayerPolicy,
    simulate_continuous,
    simulate_discrete,
    solve_epsilon_balance_policy,
)


class SimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.primitives = SellerPrimitives()
        self.model = StandardThompsonSampling(self.primitives)
        self.solution = solve_discrete(
            self.model,
            self.primitives,
            DiscreteSolverConfig(outer_diagonal=60, report_diagonal=20),
        )

    def test_discrete_simulation_is_reproducible_and_accounting_holds(self) -> None:
        kwargs = dict(
            paths=12,
            periods=15,
            seed=123,
            representative_paths=2,
            late_thresholds=(10,),
            late_window_start=10,
        )
        first = simulate_discrete(
            self.model, LayerPolicy(self.solution), self.primitives, **kwargs
        )
        second = simulate_discrete(
            self.model, LayerPolicy(self.solution), self.primitives, **kwargs
        )
        np.testing.assert_array_equal(
            first.path_summary.discounted_profit,
            second.path_summary.discounted_profit,
        )
        self.assertTrue(
            np.all(
                first.path_summary.product2_share_calendar.between(0.0, 1.0)
            )
        )
        used = first.representative_trajectories.product2_used
        chosen = first.representative_trajectories.chosen_A
        prescribed = first.representative_trajectories.product2_prescribed
        self.assertTrue(np.all(~used | (chosen & prescribed)))

    def test_scaled_simulation_reports_model_specific_belief_statistics(self) -> None:
        model = ScaledUpdates(0.25, self.primitives)
        solution = solve_discrete(
            model,
            self.primitives,
            DiscreteSolverConfig(outer_diagonal=60, report_diagonal=20),
        )
        result = simulate_discrete(
            model,
            LayerPolicy(solution),
            self.primitives,
            paths=4,
            periods=8,
            seed=7,
            representative_paths=2,
            late_thresholds=(5,),
            late_window_start=5,
        )
        trajectories = result.representative_trajectories
        expected_mean = (1.0 + 0.25 * trajectories.S) / (
            2.0 + 0.25 * (trajectories.S + trajectories.F)
        )
        np.testing.assert_allclose(trajectories.posterior_mean, expected_mean)
        np.testing.assert_allclose(
            trajectories.effective_sample_size,
            0.25 * (trajectories.S + trajectories.F),
        )
        self.assertNotIn(
            "probability_finite_last_product2_use", result.summary.columns
        )

    def test_policy_lookup_fails_instead_of_extrapolating(self) -> None:
        policy = LayerPolicy(self.solution, allow_product1_tail=False)
        with self.assertRaises(RuntimeError):
            policy.action_for_counts(np.array([21]), np.array([0]))

    def test_epsilon_balance_reduction_has_small_residual(self) -> None:
        policy = solve_epsilon_balance_policy(
            0.1, self.primitives, half_width=120, boundary_margin=15
        )
        self.assertLess(policy.bellman_residual, 2e-11)
        self.assertTrue(policy.boundary_safe)

    def test_continuous_simulation_stays_in_normalized_triangle(self) -> None:
        model = CalendarForgettingTS(0.9, self.primitives)
        solution = solve_continuous(
            model,
            self.primitives,
            ContinuousSolverConfig(grid_size=21, tolerance=1e-8),
        )
        result = simulate_continuous(
            model,
            solution,
            self.primitives,
            paths=10,
            periods=20,
            seed=456,
            representative_paths=3,
            late_thresholds=(10,),
            late_window_start=10,
        )
        paths = result.representative_trajectories
        self.assertTrue(np.all(paths.x >= -1e-14))
        self.assertTrue(np.all(paths.y >= -1e-14))
        self.assertTrue(np.all(paths.x + paths.y <= 1.0 + 1e-12))


if __name__ == "__main__":
    unittest.main()
