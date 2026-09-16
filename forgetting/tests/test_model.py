from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from forgetting.model import (
    Primitives,
    boundary_demand,
    phi_max_free,
    reachable_boundary_points,
    reachable_maximum,
    small_rho_closed_form,
    solve_boundary,
)
from forgetting.run_experiments import configured_p0_values


class ForgettingModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.primitives = Primitives()

    def test_boundary_demand_is_monotone_and_symmetric(self) -> None:
        coordinate = np.linspace(0.0, 1.0, 201)
        demand = boundary_demand(coordinate, rho=0.95, p0=0.5)
        self.assertTrue(np.all(np.diff(demand) >= -1e-13))
        np.testing.assert_allclose(demand + demand[::-1], 1.0, atol=3e-13)

    def test_p0_comparison_matches_extinction_and_keeps_baseline(self) -> None:
        config = {"p0_comparison": {"values": [0.9, 0.1, 0.3, 0.7]}}
        self.assertEqual(
            configured_p0_values(config, baseline_p0=0.5),
            [0.1, 0.3, 0.5, 0.7, 0.9],
        )

    def test_gamma_zero_rules_out_product_two(self) -> None:
        solution = solve_boundary(
            replace(self.primitives, gamma=0.0),
            rho=0.8,
            central_nodes=401,
            outer_nodes=50,
            tolerance=1e-12,
        )
        np.testing.assert_allclose(
            solution.advantage, -self.primitives.delta_c, atol=1e-13
        )

    def test_reachable_cylinders_have_the_reported_radius(self) -> None:
        rho = 0.4
        depth = 8
        points = reachable_boundary_points(rho, depth)
        self.assertAlmostEqual(float(points[0]), 0.0)
        self.assertAlmostEqual(float(points[-1]), 1.0)
        self.assertEqual(len(points), 2 ** (depth + 1))
        solution = solve_boundary(
            self.primitives,
            rho,
            central_nodes=401,
            outer_nodes=50,
            tolerance=1e-10,
        )
        _, _, radius = reachable_maximum(solution, reachable_depth=depth)
        self.assertAlmostEqual(radius, rho**depth)

    def test_baseline_has_intermediate_persistence_and_high_rho_extinction(
        self,
    ) -> None:
        low = solve_boundary(
            self.primitives,
            0.2,
            central_nodes=801,
            outer_nodes=100,
            tolerance=2e-10,
        )
        middle = solve_boundary(
            self.primitives,
            0.8,
            central_nodes=801,
            outer_nodes=100,
            tolerance=2e-10,
        )
        high = solve_boundary(
            self.primitives,
            0.999,
            central_nodes=1_601,
            outer_nodes=200,
            tolerance=2e-10,
        )
        self.assertLess(reachable_maximum(low)[0], 0.0)
        self.assertGreater(reachable_maximum(middle)[0], 0.0)
        self.assertLess(reachable_maximum(high)[0], 0.0)

    def test_max_free_and_optimal_values_coincide_outside_active_window(self) -> None:
        for rho in (0.2, 0.999):
            optimal = solve_boundary(
                self.primitives,
                rho,
                central_nodes=801,
                outer_nodes=100,
                tolerance=2e-10,
            )
            phi, _ = phi_max_free(
                self.primitives,
                rho,
                central_nodes=801,
                outer_nodes=100,
                tolerance=2e-10,
            )
            self.assertAlmostEqual(
                phi, reachable_maximum(optimal)[0], delta=2e-7
            )

    def test_small_rho_closed_form_matches_known_baseline(self) -> None:
        result = small_rho_closed_form(self.primitives)
        self.assertGreater(result["W1"], result["W2"])
        self.assertEqual(result["selected_product"], 1)
        self.assertEqual(result["selected_value"], result["W1"])
        self.assertAlmostEqual(result["advantage"], -0.24100257069408765, places=10)

    def test_small_rho_closed_form_uses_product_two_when_w2_is_larger(
        self,
    ) -> None:
        result = small_rho_closed_form(replace(self.primitives, c2=0.35))

        self.assertGreater(result["W2"], result["W1"])
        self.assertEqual(result["selected_product"], 2)
        self.assertEqual(result["selected_value"], result["W2"])
        self.assertAlmostEqual(result["W1"], 16.287167952013675, places=12)
        self.assertAlmostEqual(result["W2"], 17.91305096418731, places=12)
        self.assertAlmostEqual(result["advantage"], 0.09483471074380184, places=12)

    def test_bellman_residual_is_small(self) -> None:
        solution = solve_boundary(
            self.primitives,
            0.95,
            central_nodes=801,
            outer_nodes=100,
            tolerance=1e-10,
        )
        self.assertLessEqual(solution.bellman_residual, 1.01e-10)


if __name__ == "__main__":
    unittest.main()
