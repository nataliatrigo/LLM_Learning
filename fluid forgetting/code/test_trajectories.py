"""Independent checks of smooth motion, crossing events, and holding mixtures."""
import math
import unittest

import numpy as np

from simulate_trajectories import (PiecewisePolicy, Primitives,
                                   analytic_constant_demand_check, simulate_path)


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.p = Primitives()
        self.lam = .2
        self.rho = math.exp(-self.lam)
        self.demand = .4
        self.times = np.linspace(0, 100, 1001)

    def policy(self, roots, products):
        return PiecewisePolicy(self.p, self.rho, roots, products,
                               lambda x: np.full_like(x, self.demand))

    def test_constant_demand_analytic_solution(self):
        self.assertLess(analytic_constant_demand_check(), 2e-9)

    def test_attractive_root_hits_and_holds_at_analytic_time(self):
        root, x0 = .55, .2
        path = simulate_path(self.policy([root], [2, 1]), x0, self.times)
        exact_time = -math.log((self.p.p2-root)/(self.p.p2-x0))/(self.lam*self.demand)
        self.assertAlmostEqual(path["holding_time"], exact_time, places=7)
        held = self.times >= exact_time
        np.testing.assert_allclose(path["x"][held], root, atol=1e-13)
        np.testing.assert_allclose(path["drift"][held], 0, atol=1e-13)
        np.testing.assert_allclose(path["alpha"][held], (root-self.p.p1)/self.p.delta_p)

    def test_crossing_continues_then_holds(self):
        policy = self.policy([.2, .55], [1, 2, 1])
        path = simulate_path(policy, .1, self.times)
        self.assertEqual([event["kind"] for event in path["events"]], ["crossing", "attractive"])
        crossing = -math.log((self.p.p1-.2)/(self.p.p1-.1))/(self.lam*self.demand)
        self.assertAlmostEqual(path["events"][0]["time"], crossing, places=7)
        self.assertAlmostEqual(path["x"][-1], .55)
        self.assertGreaterEqual(np.min(np.diff(path["x"])), -1e-12)

    def test_repelling_root_separates_basins_and_tie_uses_product1(self):
        policy = self.policy([.45, .6], [1, 2, 1])
        below = simulate_path(policy, .4, self.times)
        above = simulate_path(policy, .5, self.times)
        exact = simulate_path(policy, .45, self.times)
        self.assertLess(below["x"][-1], .4)
        self.assertAlmostEqual(above["x"][-1], .6)
        self.assertEqual(exact["alpha"][0], 0)
        self.assertLess(exact["x"][-1], .45)

    def test_initial_at_holding_root_and_fixed_equilibrium(self):
        held = simulate_path(self.policy([.55], [2, 1]), .55, self.times)
        self.assertEqual(held["holding_time"], 0)
        np.testing.assert_allclose(held["x"], .55)
        fixed = simulate_path(self.policy([], [1]), self.p.p1, self.times)
        self.assertIsNone(fixed["holding_time"])
        np.testing.assert_allclose(fixed["x"], self.p.p1)
        np.testing.assert_allclose(fixed["drift"], 0)


if __name__ == "__main__":
    unittest.main()
