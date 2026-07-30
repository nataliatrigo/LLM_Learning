"""Tests for policy-localization and demand-sensitivity diagnostics."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp.demand_models import (  # noqa: E402
    EpsilonGreedy,
    ObservationForgettingTS,
    ScaledUpdates,
    StandardThompsonSampling,
)
from modular_dp.diagnostics import (  # noqa: E402
    classify_product2_policy,
    compute_discrete_sensitivity,
    contiguous_components,
    continuous_local_sensitivity,
)
from modular_dp.discrete_solver import (  # noqa: E402
    DiscreteSolution,
    DiscreteSolverConfig,
    solve_discrete,
)
from modular_dp.primitives import SellerPrimitives  # noqa: E402


class PolicyLocalizationTests(unittest.TestCase):
    def test_component_endpoints_and_last_active_diagonal(self) -> None:
        primitives = SellerPrimitives()
        zeros = np.zeros(3)
        fake = DiscreteSolution(
            model=StandardThompsonSampling(primitives),
            primitives=primitives,
            config=DiscreteSolverConfig(outer_diagonal=10, report_diagonal=2),
            layers={
                0: {
                    "gap": zeros[:1],
                    "advantage": zeros[:1],
                    "action2": np.array([False]),
                    "action_tolerance": zeros[:1],
                },
                1: {
                    "gap": zeros[:2],
                    "advantage": zeros[:2],
                    "action2": np.array([False, False]),
                    "action_tolerance": zeros[:2],
                },
                2: {
                    "gap": zeros,
                    "advantage": zeros,
                    "action2": np.array([True, False, True]),
                    "action_tolerance": zeros,
                },
            },
            maximum_bellman_residual=0.0,
        )
        localization = classify_product2_policy(fake)
        last = localization.by_diagonal[2]
        self.assertEqual(last.components, ((0, 0), (2, 2)))
        self.assertEqual(last.lower_s, 0)
        self.assertEqual(last.upper_s, 2)
        self.assertEqual(localization.last_active_diagonal, 2)
        self.assertEqual(localization.distance_to_outer_boundary, 8)

    def test_contiguous_components_empty_and_multiple(self) -> None:
        self.assertEqual(contiguous_components(np.zeros(4, dtype=bool)), ())
        self.assertEqual(
            contiguous_components(np.array([True, True, False, True, False, True])),
            ((0, 1), (3, 3), (5, 5)),
        )


class SensitivityTests(unittest.TestCase):
    def test_standard_ts_has_half_order_decay(self) -> None:
        result = compute_discrete_sensitivity(
            StandardThompsonSampling(),
            max_n=500,
            fit_start=200,
            fit_end=500,
        )
        self.assertGreater(result.fit_observations, 250)
        self.assertGreater(result.estimated_slope, -0.65)
        self.assertLess(result.estimated_slope, -0.35)
        self.assertTrue(np.all(result.g_n > 0.0))

    def test_epsilon_greedy_maximal_jump_does_not_vanish(self) -> None:
        epsilon = 0.2
        result = compute_discrete_sensitivity(
            EpsilonGreedy(epsilon), max_n=200, fit_start=50
        )
        self.assertAlmostEqual(result.max_demand_sensitivity, 1.0 - epsilon)
        self.assertGreater(float(np.min(result.g_n[50:])), 0.0)
        unit_exploration = compute_discrete_sensitivity(
            EpsilonGreedy(1.0), max_n=20
        )
        np.testing.assert_array_equal(unit_exploration.g_n, 0.0)
        self.assertTrue(np.isnan(unit_exploration.estimated_slope))

    def test_scaled_sensitivity_argmax_uses_scaled_belief_mean(self) -> None:
        model = ScaledUpdates(0.25)
        result = compute_discrete_sensitivity(model, max_n=20)
        expected = np.asarray(
            model.posterior_mean_counts(
                result.argmax_s, result.n - result.argmax_s
            )
        )
        np.testing.assert_allclose(result.argmax_posterior_mean, expected)

    def test_continuous_sensitivity_reports_reachable_maximum(self) -> None:
        model = ObservationForgettingTS(0.9)
        coordinates = np.linspace(0.0, 1.0, 21)
        result = continuous_local_sensitivity(model, coordinates, coordinates)
        self.assertEqual(result.q.shape, (21, 21))
        self.assertGreater(result.maximum, 0.0)
        self.assertLessEqual(sum(result.argmax_state), 1.0 + 1e-12)
        self.assertIn("posterior_mean", result.argmax_coordinates)
        self.assertTrue(np.all(np.isnan(result.q[~result.reachable_mask])))

if __name__ == "__main__":
    unittest.main()
