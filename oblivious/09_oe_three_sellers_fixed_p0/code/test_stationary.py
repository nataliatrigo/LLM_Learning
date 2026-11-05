"""Numerical checks of the new multi-seller stationary calculation."""

import unittest

import numpy as np
from scipy.sparse import csr_matrix

from oblivious.oe_p0.code.positive_stationary import dense_positive_invariant
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_three_sellers_fixed_p0.code.kernel import FixedOutsideKernel
from oblivious.oe_three_sellers_fixed_p0.code.stationary import (
    checked_stationary, checked_symmetric_stationary, jump_system, symmetric_lumping,
)


class StationaryChecks(unittest.TestCase):
    def test_full_positive_reference_two_and_three_sellers(self):
        grid = np.linspace(0, 1, 5)
        for sellers in (2, 3):
            with self.subTest(sellers=sellers):
                game = FixedOutsideKernel(grid, Parameters(rho=.7), p0=.5, sellers=sellers, order=64)
                alpha = np.array([0, .2, 1, .7, 0])
                jump, rates = game.jump(alpha)
                result = checked_symmetric_stationary(jump, rates, game.shape)
                full = dense_positive_invariant(jump, rates)
                reverse = dense_positive_invariant(jump, rates, np.arange(game.size)[::-1])
                np.testing.assert_allclose(result["law"], full, atol=2e-13, rtol=2e-11)
                np.testing.assert_allclose(result["law"], reverse, atol=2e-13, rtol=2e-11)
                self.assertLess(result["summary"]["full_rate_weighted_residual"], 1e-12)
                self.assertLess(result["summary"]["marginal_symmetry_l1"], 1e-14)
                expected = 15 if sellers == 2 else 35
                self.assertEqual(result["summary"]["solved_states"], expected)

    def test_rare_exit_rates_survive_rounded_calendar_diagonal(self):
        # The calendar matrix is indistinguishable from the identity by its
        # diagonal alone.  The positive off-diagonal entries still determine it.
        kernel = csr_matrix([[1., 1e-320], [1e-310, 1.]])
        jump, rates = jump_system(kernel)
        np.testing.assert_array_equal(jump.toarray(), [[0, 1], [1, 0]])
        result = checked_stationary(jump, rates)
        expected = np.array([1 / (1 + rates[0] / rates[1]),
                             1 / (1 + rates[1] / rates[0])])
        np.testing.assert_allclose(result["law"], expected, rtol=2e-12, atol=0)
        self.assertGreater(result["law"].min(), 0)
        self.assertLess(result["summary"]["rate_weighted_residual"], 1e-12)

    def test_rare_jump_flows_and_calendar_weighting(self):
        epsilon = 1e-80
        jump = csr_matrix([[0, 1, 0], [epsilon, 0, 1 - epsilon], [0, 1, 0]])
        rates = np.array([1e-280, 1e-250, 1e-200])
        result = checked_stationary(jump, rates)
        expected = np.array([epsilon / rates[0], 1 / rates[1], (1 - epsilon) / rates[2]])
        expected /= expected.sum()
        np.testing.assert_allclose(result["law"], expected, rtol=1e-12, atol=0)
        self.assertGreater(result["law"].min(), 0)

    def test_lumping_rejects_asymmetric_seller_policies(self):
        game = FixedOutsideKernel(np.linspace(0, 1, 5), Parameters(rho=.7), sellers=3, order=64)
        policies = np.array([[0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [0, 0, 0, 0, 0]])
        jump, rates = game.jump(policies)
        with self.assertRaisesRegex(ArithmeticError, "lumping failed"):
            symmetric_lumping(jump, rates, game.shape)

    def test_large_path_positive_iteration_is_required_and_checked(self):
        game = FixedOutsideKernel(np.linspace(0, 1, 5), Parameters(rho=.6), sellers=3, order=64)
        jump, rates = game.jump(np.full(5, .5))
        result = checked_symmetric_stationary(jump, rates, game.shape, reference_max_states=0)
        self.assertIsNotNone(result["summary"]["positive_iteration"])
        self.assertGreater(result["summary"]["positive_iteration"]["iterations"], 0)

    def test_rare_bridge_does_not_pass_by_residual_alone(self):
        epsilon = 1e-10
        jump = csr_matrix([[0, 1 - epsilon, epsilon, 0],
                           [1 - epsilon, 0, 0, epsilon],
                           [epsilon, 0, 0, 1 - epsilon],
                           [0, epsilon, 1 - epsilon, 0]])
        # A direct solve has a very small residual.  Widely separated positive
        # starts expose that this short iteration budget cannot validate it.
        with self.assertRaisesRegex(ArithmeticError, "within its budget"):
            checked_stationary(jump, np.ones(4), reference_max_states=0,
                               positive_max_iterations=100, stability_tolerance=1e-5)

    def test_reducible_kernel_is_rejected(self):
        with self.assertRaisesRegex(ArithmeticError, "communicating components"):
            checked_stationary(csr_matrix(np.eye(3)), np.ones(3))


if __name__ == "__main__":
    unittest.main()
