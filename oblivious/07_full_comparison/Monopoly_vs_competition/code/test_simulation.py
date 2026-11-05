"""Behavioral checks for the new outside-option simulator and count estimators."""

import sys
from pathlib import Path
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.policy import transition_matrix
from simulation import simulate, finite_window, ratio_interval, power_and_sum


class SimulationChecks(unittest.TestCase):
    def setUp(self):
        self.grid = np.linspace(0, 1, 51)
        self.params = Parameters(rho=.9)

    def test_outside_purchases_have_no_profit_and_do_not_update(self):
        result = simulate(self.grid, np.ones(51), np.zeros(51), self.params,
                          periods=50, burn_in=5, replications=3)
        self.assertFalse(result["sold"].any())
        self.assertTrue(np.all(result["before_index"] == 25))
        self.assertTrue(np.all(result["final_index"] == 25))
        self.assertTrue(all(row["seller_profit"] == 0 for row in result["replications"]))
        self.assertIsNone(ratio_interval(result["replications"])["mean"])

    def test_always_selected_product2_earns_correct_net_reward(self):
        result = simulate(self.grid, np.ones(51), np.ones(51), self.params,
                          periods=50, burn_in=5, replications=3)
        self.assertTrue(result["sold"].all())
        self.assertTrue(result["product2"].all())
        for row in result["replications"]:
            self.assertAlmostEqual(row["seller_profit"], self.params.R-self.params.c2)
            self.assertEqual(row["product2_conditional"], 1)

    def test_one_step_transition_agrees_with_independent_kernel(self):
        # Independent paths test demand, action, success, and projection jointly.
        policy = np.full(51, .37)
        demand = np.full(51, .42)
        result = simulate(self.grid, policy, demand, self.params,
                          periods=1, burn_in=0, replications=30000, seed=20171003)
        empirical = np.bincount(result["final_index"], minlength=51)/30000
        expected = transition_matrix(self.grid, demand, policy, self.params).toarray()[25]
        error = abs(empirical-expected)
        bound = 5*np.sqrt(expected*(1-expected)/30000)+1/30000
        self.assertTrue(np.all(error < bound))
        finite = finite_window(transition_matrix(self.grid, demand, policy, self.params),
                               self.grid, demand, policy, self.params, periods=1, burn_in=0)
        profits = np.array([row["seller_profit"] for row in result["replications"]])
        self.assertLess(abs(profits.mean()-finite["outcomes"]["seller_profit"]),
                        5*profits.std(ddof=1)/np.sqrt(len(profits)))

    def test_finite_window_matches_direct_iteration(self):
        policy = np.linspace(.2, .8, 51)
        demand = np.linspace(.1, .9, 51)
        matrix = transition_matrix(self.grid, demand, policy, self.params)
        result = finite_window(matrix, self.grid, demand, policy, self.params,
                               periods=13, burn_in=7)
        law, occupancy = np.eye(51)[25], np.zeros(51)
        for step in range(20):
            if step >= 7:
                occupancy += law/13
            law = law @ matrix
        np.testing.assert_allclose(result["occupation"], occupancy, rtol=0, atol=2e-15)
        power, series = power_and_sum(matrix.toarray(), 13)
        np.testing.assert_allclose(power, np.linalg.matrix_power(matrix.toarray(),13), rtol=0, atol=2e-15)
        direct_series = sum(np.linalg.matrix_power(matrix.toarray(), t) for t in range(13))
        np.testing.assert_allclose(series, direct_series, rtol=0, atol=1e-14)

    def test_ratio_is_sale_weighted_and_keeps_zero_sale_paths(self):
        rows = [dict(product2_sales=0,sales=0), dict(product2_sales=1,sales=1),
                dict(product2_sales=0,sales=9)]
        result = ratio_interval(rows)
        self.assertEqual(result["mean"], .1)
        self.assertEqual(result["n_valid"], 3)
        self.assertGreater(result["se"], 0)


if __name__ == "__main__":
    unittest.main()
