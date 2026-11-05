"""Checks for unrestricted policy-region detection and the shared solver."""

import unittest

import numpy as np

from compare_beta_logistic_demand import P, demands, detect_regions, solve, advantage
from forgetting.Code.model import BoundaryGrid, solve_boundary


class DemandExperimentTests(unittest.TestCase):
    def test_multiple_disconnected_positive_regions(self):
        x = np.linspace(0, 1, 6)
        roots, regions, zero_segments = detect_regions(x, [1, -1, 1, -1, 1, -1])
        np.testing.assert_allclose(roots, [0.1, 0.3, 0.5, 0.7, 0.9])
        np.testing.assert_allclose([(r['lower'], r['upper']) for r in regions], [(0, 0.1), (0.3, 0.5), (0.7, 0.9)])
        self.assertTrue(regions[0]['lower_included'])
        self.assertFalse(any(r['upper_included'] for r in regions))
        self.assertEqual(zero_segments, [])

    def test_zero_touch_and_zero_plateau_split_strict_regions(self):
        roots, regions, zeros = detect_regions([0, 0.25, 0.5, 0.75, 1], [1, 0, 1, 0, 0])
        self.assertEqual(roots, [0.25, 0.75, 1.0])
        self.assertEqual(len(regions), 2)
        self.assertEqual(zeros, [[0.75, 1.0]])
        self.assertFalse(regions[1]['lower_included'])
        self.assertFalse(regions[1]['upper_included'])
        roots, regions, _ = detect_regions([0, 0.5, 1], [0, -1, 0])
        self.assertEqual(roots, [0.0, 1.0])
        self.assertEqual(regions, [])

    def test_constant_demand_has_known_value_and_no_investment(self):
        x = np.linspace(0, 1, 101)
        result = solve(x, np.full_like(x, 0.4), 0.95, 'constant')
        np.testing.assert_allclose(result.value, 0.4 * (P.revenue-P.c1)/(1-P.gamma), atol=6e-11, rtol=0)
        np.testing.assert_allclose(advantage(result, x), -P.delta_c, atol=1e-12, rtol=0)

    def test_beta_solution_matches_original_boundary_solver(self):
        x = np.linspace(0, 1, 601)
        grid = BoundaryGrid(x, len(x), 2, 1e6)
        for rho in (0.8, 0.999):
            with self.subTest(rho=rho):
                exact, _ = demands(x, rho)
                new = solve(x, exact, rho, 'exact')
                original = solve_boundary(P, rho, grid=grid, tolerance=1e-12)
                np.testing.assert_allclose(new.value, original.value, atol=2e-10, rtol=0)
                np.testing.assert_allclose(advantage(new, x), original.advantage, atol=2e-10, rtol=0)


if __name__ == '__main__':
    unittest.main()
