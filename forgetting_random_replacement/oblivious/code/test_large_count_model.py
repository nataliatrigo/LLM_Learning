"""Independent numerical checks of the scalable exact-count solver.

The reference solver uses subtraction-free positive GTH elimination and a
direct discounted-value solve. The tested adapter instead uses iterative
normalized systems and a Bellman contraction on the same exact joint graph.
"""

import unittest

import numpy as np

from forgetting_random_replacement.oblivious.code.large_count_model import (
    LargeCountExperiment, checked_large_stationary)
from forgetting_random_replacement.oblivious.code.model import CountExperiment


class LargeCountModelTests(unittest.TestCase):
    def test_forced_iterative_equilibria_match_positive_gth_and_direct_payoffs(self):
        for capacity, sellers, p0 in ((3, 3, .3), (5, 5, .1)):
            with self.subTest(capacity=capacity, sellers=sellers, p0=p0):
                exact = CountExperiment(capacity, sellers, "marginal", p0).solve()
                iterative = LargeCountExperiment(capacity, sellers, "marginal", p0,
                                                 direct_state_limit=0)
                result = iterative.solve(initial_policy=exact["policy"])
                self.assertTrue(result["verified"])
                self.assertTrue(np.all(result["joint"] > 0))
                np.testing.assert_allclose(result["policy"], exact["policy"], atol=2e-8, rtol=0)
                self.assertLess(float(abs(result["joint"]-exact["joint"]).sum()), 1e-8)
                np.testing.assert_allclose(result["demand"], exact["demand"], atol=2e-9, rtol=0)
                np.testing.assert_allclose(result["value"], exact["value"], atol=2e-7, rtol=0)
                bound = result["diagnostics"]["actual_value_error_bound"]
                exact_bound = exact["diagnostics"]["actual_value_residual"]/(1-iterative.primitives.gamma)
                allowance = bound+exact_bound+5e-11
                self.assertLessEqual(float(abs(result["actual_value"]-exact["actual_value"]).max()), allowance)
                self.assertAlmostEqual(result["summary"]["value_initial"],
                                       exact["summary"]["value_initial"], delta=allowance)
                stationary = result["diagnostics"]["stationary"]
                self.assertEqual(len(stationary["linear_solves"]), 2)
                self.assertIn("reversed", stationary["methods"][1])
                self.assertIn("uniform", stationary["methods"][2])
                self.assertLessEqual(stationary["positive_iteration"]["discrepancy"], 1e-7)
                self.assertLessEqual(stationary["rate_weighted_residual"], 1e-11)
                for metric in ("product2_conditional", "sale_rate", "outside_rate", "profit_calendar"):
                    self.assertAlmostEqual(result["summary"][metric], exact["summary"][metric], delta=2e-8)

    def test_iterative_equilibrium_jacobian_matches_independent_finite_differences(self):
        experiment = LargeCountExperiment(5, 3, "marginal", .3, direct_state_limit=0)
        policy = np.linspace(.2, .8, 6)
        analytic = experiment.calculate(policy)["jacobian"].copy()
        epsilon = 2e-5
        for coordinate in (0, 2, 5):
            direction = np.eye(6)[coordinate]
            upper = experiment.calculate(policy+epsilon*direction)["advantage"].copy()
            lower = experiment.calculate(policy-epsilon*direction)["advantage"].copy()
            derivative = (upper-lower)/(2*epsilon)
            with self.subTest(coordinate=coordinate):
                np.testing.assert_allclose(analytic[:, coordinate], derivative, atol=5e-8, rtol=2e-6)

    def test_rare_positive_count_states_survive_iterative_stationary_solution(self):
        # This 3,003-state chain exercises tails below the algebraic absolute
        # solver tolerance. The positive reference is evaluated only once.
        policy = np.full(11, .5)
        reference = CountExperiment(10, 5, "marginal", .1)
        exact = reference.calculate(policy)
        iterative = LargeCountExperiment(10, 5, "marginal", .1, direct_state_limit=0)
        result = iterative.calculate(policy)
        self.assertTrue(np.all(exact["joint"] > 0))
        self.assertTrue(np.all(result["joint"] > 0))
        self.assertLess(float(exact["joint"].min()), 1e-14)
        self.assertLess(float(abs(result["joint"]-exact["joint"]).sum()), 1e-8)
        np.testing.assert_allclose(result["demand"], exact["demand"], atol=2e-9, rtol=0)
        jump, rates = iterative._jump(policy)
        independently_checked = checked_large_stationary(jump, rates)
        self.assertTrue(independently_checked["summary"]["verified"])
        self.assertTrue(np.all(independently_checked["law"] > 0))
        self.assertLess(float(abs(independently_checked["law"]-exact["joint"]).sum()), 1e-8)
        self.assertLessEqual(independently_checked["summary"]["positive_iteration"]["discrepancy"], 1e-7)


if __name__ == "__main__":
    unittest.main()
