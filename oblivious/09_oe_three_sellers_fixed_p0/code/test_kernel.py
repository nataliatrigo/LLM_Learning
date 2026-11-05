"""Independent economic timing and demand tests for the new joint kernel."""

from itertools import permutations
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import betainc, betaincc, betaln

from oblivious.oe_p0.code.primitives import Parameters, posterior
from oblivious.oe_distribution_fixed_p0.code.model import OutsideOptionGame
from oblivious.oe_three_sellers_fixed_p0.code.kernel import FixedOutsideKernel


class FixedOutsideKernelTests(unittest.TestCase):
    def test_three_seller_demand_matches_independent_scalar_integrals(self):
        grid = np.linspace(0, 1, 5)
        params = Parameters(rho=.8)
        kernel = FixedOutsideKernel(grid, params, p0=.8, sellers=3, order=64)
        states = [(0, 2, 4), (2, 2, 2), (4, 1, 3), (0, 0, 0)]
        for state in states:
            a, b = posterior(grid[list(state)], params)
            for seller in range(3):
                def integrand(t):
                    pdf = np.exp((a[seller]-1)*np.log(t)
                                 + (b[seller]-1)*np.log1p(-t)
                                 - betaln(a[seller], b[seller]))
                    others = [j for j in range(3) if j != seller]
                    return pdf * np.prod(betainc(a[others], b[others], t))
                expected, error = quad(integrand, .8, 1, epsabs=1e-13, epsrel=1e-12)
                self.assertLess(error, 1e-11)
                assert_allclose(kernel.demand[(seller,) + state], expected, rtol=1e-10, atol=1e-14)
        d = kernel.demand_diagnostics
        self.assertLess(d['raw_partition_error'], 1e-11)
        self.assertLess(d['order_doubling_max_relative_tail'], 1e-9)
        self.assertEqual(d['positive_tail_underflow_entries'], 0)

    def test_monopoly_and_three_way_symmetric_selection_identity(self):
        grid = np.linspace(0, 1, 7)
        params = Parameters(rho=.6)
        for p0 in (0., .1, .5, .8, 1.):
            monopoly = FixedOutsideKernel(grid, params, p0, sellers=1, order=96)
            a, b = posterior(grid, params)
            assert_allclose(monopoly.demand[0], betaincc(a, b, p0), atol=1e-13)
            market = FixedOutsideKernel(grid, params, p0, sellers=3, order=96)
            i = np.arange(len(grid))
            expected = (1-betainc(a, b, p0)**3)/3
            for seller in range(3):
                assert_allclose(market.demand[seller, i, i, i], expected, atol=1e-13)
            assert_allclose(market.demand.sum(axis=0) + market.outside, 1., atol=5e-15)

    def test_demand_is_permutation_equivariant(self):
        kernel = FixedOutsideKernel(np.linspace(0, 1, 9), Parameters(rho=.7), .3, sellers=3, order=128)
        for axes in permutations(range(3)):
            for seller in range(3):
                assert_allclose(kernel.demand[seller].transpose(axes),
                                kernel.demand[axes.index(seller)], atol=1e-14, rtol=1e-12)
            assert_allclose(kernel.outside.transpose(axes), kernel.outside, atol=1e-15)

    def test_two_seller_transition_matches_existing_model(self):
        grid = np.linspace(0, 1, 9)
        params = Parameters(rho=.7)
        kernel = FixedOutsideKernel(grid, params, .8, sellers=2, order=96)
        old = OutsideOptionGame(grid, params, p0=.8, order=96)
        alpha = np.vstack((np.linspace(0, 1, len(grid)), np.linspace(1, 0, len(grid))))
        assert_allclose(kernel.demand[0], old.D, atol=2e-13, rtol=2e-11)
        assert_allclose(kernel.demand[1], old.D.T, atol=2e-13, rtol=2e-11)
        sigma_a = np.broadcast_to(alpha[0, :, None], (len(grid), len(grid)))
        sigma_b = np.broadcast_to(alpha[1, None, :], (len(grid), len(grid)))
        assert_allclose(kernel.transition(alpha).toarray(),
                        old.transition(sigma_a, sigma_b).toarray(), atol=2e-13)

    def test_matrix_free_and_sparse_transition_freeze_other_sellers(self):
        grid = np.array([0., .1, .4, .8, 1.])
        kernel = FixedOutsideKernel(grid, Parameters(rho=.6), .5, sellers=3, order=96)
        alpha = np.array([.2, .8, 1., .3, 0.])
        values = np.random.default_rng(41).normal(size=kernel.shape)
        transition = kernel.transition(alpha)
        assert_allclose(transition.sum(axis=1), 1., atol=4e-15)
        self.assertTrue(np.all(transition.data >= 0))
        assert_allclose(kernel.apply(values, alpha).ravel(), transition @ values.ravel(), atol=1e-14)
        # Every nonholding edge changes exactly one seller's coordinate.
        coo = transition.tocoo()
        from_indices = np.column_stack(np.unravel_index(coo.row, kernel.shape))
        to_indices = np.column_stack(np.unravel_index(coo.col, kernel.shape))
        self.assertTrue(np.all(np.sum(from_indices != to_indices, axis=1) <= 1))
        jump, rates = kernel.jump(alpha)
        assert_allclose(jump.sum(axis=1), 1., atol=4e-15)
        assert_allclose(jump.diagonal(), 0.)
        reconstructed = jump.multiply(rates[:, None]).toarray()
        np.fill_diagonal(reconstructed, 1-rates)
        assert_allclose(reconstructed, transition.toarray(), atol=2e-15)

    def test_rare_exit_rate_is_not_one_minus_holding_probability(self):
        kernel = FixedOutsideKernel(np.linspace(0, 1, 5), Parameters(rho=.8),
                                    1-1e-8, sellers=3, order=64)
        alpha = np.zeros(kernel.n)
        jump, rates = kernel.jump(alpha)
        self.assertGreater(rates[0], 0.)
        self.assertLess(rates[0], 1e-35)
        self.assertEqual(1-kernel.transition(alpha)[0, 0], 0.)
        assert_allclose(jump.sum(axis=1), 1., atol=5e-15)

    def test_exact_outside_endpoint_is_absorbing(self):
        kernel = FixedOutsideKernel(np.linspace(0, 1, 3), Parameters(rho=.6), 1., sellers=3)
        assert_allclose(kernel.transition(np.zeros(3)).toarray(), np.eye(27))
        with self.assertRaisesRegex(ArithmeticError, 'absorbing'):
            kernel.jump(np.zeros(3))


if __name__ == '__main__':
    unittest.main()
