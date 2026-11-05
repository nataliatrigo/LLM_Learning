"""Small full-state and independent-integral audits of the N=4 quotient."""

from functools import lru_cache
from itertools import product
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import betainc, betaln

from oblivious.oe_p0.code.primitives import Parameters, posterior
from .four_seller_kernel import FourSellerKernel


def independent_demand(grid, params, p0):
    a, b = posterior(grid, params)

    @lru_cache(None)
    def focal(own, rivals):
        def integrand(t):
            if t == 0 or t == 1:
                return 0.
            density = np.exp((a[own]-1)*np.log(t)
                             +(b[own]-1)*np.log1p(-t)-betaln(a[own], b[own]))
            return density*np.prod(betainc(a[list(rivals)], b[list(rivals)], t))
        value, error = quad(integrand, p0, 1, epsabs=2e-13, epsrel=2e-12, limit=300)
        if error > 2e-11:
            raise AssertionError(f"Independent quadrature error {error}")
        return value
    return focal


class FourSellerKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.grid = np.linspace(0, 1, 3)
        cls.params = Parameters(rho=.7)
        cls.kernel = FourSellerKernel(cls.grid, cls.params, p0=.3, order=96)

    def test_orbits_exhaust_full_labelled_grid_with_correct_multiplicities(self):
        kernel = self.kernel
        states = np.asarray(list(product(range(3), repeat=4)))
        ids = kernel.orbit_index(states)
        assert_allclose(np.bincount(ids), kernel.orbit_sizes)
        self.assertEqual(kernel.size, 15)
        self.assertEqual(kernel.full_size, 81)
        self.assertEqual(int(kernel.orbit_sizes.sum()), 81)
        assert_allclose(kernel.orbit_index(kernel.representatives), np.arange(kernel.size))

    def test_positive_demands_match_independent_scalar_integrals(self):
        grid = np.linspace(0, 1, 5)
        params = Parameters(rho=.8)
        kernel = FourSellerKernel(grid, params, p0=.8, order=96)
        focal = independent_demand(grid, params, .8)
        for state in ((0, 1, 3, 4), (0, 0, 0, 0), (2, 2, 2, 2), (1, 1, 3, 4)):
            row = int(kernel.orbit_index(state))
            for slot in range(4):
                rivals = tuple(state[k] for k in range(4) if k != slot)
                assert_allclose(kernel.demand[row, slot], focal(state[slot], rivals),
                                rtol=2e-10, atol=2e-14)
        assert_allclose(kernel.demand.sum(axis=1)+kernel.outside, 1, atol=5e-15)

    def test_direct_full_calendar_lumps_to_identical_quotient(self):
        kernel, grid, p = self.kernel, self.grid, self.params
        focal = independent_demand(grid, p, .3)
        alpha = np.array([.2, .8, .1])
        states = list(product(range(3), repeat=4))
        ids = {state: row for row, state in enumerate(states)}
        full = np.zeros((81, 81))
        a, b = posterior(grid, p)
        for row, state in enumerate(states):
            full[row, row] += np.prod(betainc(a[list(state)], b[list(state)], .3))
            for slot, own in enumerate(state):
                rivals = tuple(sorted(state[k] for k in range(4) if k != slot))
                selected = focal(own, rivals)
                q = p.p1+(p.p2-p.p1)*alpha[own]
                for success, probability in ((True, q), (False, 1-q)):
                    target = p.rho*grid[own]+(1-p.rho if success else 0)
                    lower = int(np.clip(np.searchsorted(grid, target, side="right")-1, 0, 1))
                    upper_weight = (target-grid[lower])/(grid[lower+1]-grid[lower])
                    for destination, weight in ((lower, 1-upper_weight), (lower+1, upper_weight)):
                        state_next = list(state)
                        state_next[slot] = destination
                        full[row, ids[tuple(state_next)]] += selected*probability*weight
        orbit_ids = kernel.orbit_index(np.asarray(states))
        aggregate = np.zeros((81, kernel.size))
        for labelled_column, orbit_column in enumerate(orbit_ids):
            aggregate[:, orbit_column] += full[:, labelled_column]
        quotient = kernel.transition(alpha).toarray()
        assert_allclose(aggregate, quotient[orbit_ids], rtol=2e-10, atol=2e-13)
        assert_allclose(quotient.sum(axis=1), 1, atol=5e-15)
        self.assertTrue(np.all(quotient >= 0))

    def test_marginal_operator_preserves_correlated_joint_rivals(self):
        kernel = self.kernel
        orbit_law = np.arange(1., kernel.size+1)**2
        orbit_law /= orbit_law.sum()
        joint = np.empty((3,)*4)
        for state in product(range(3), repeat=4):
            row = int(kernel.orbit_index(state))
            joint[state] = orbit_law[row]/kernel.orbit_sizes[row]
        own = joint.sum(axis=(1, 2, 3))
        assert_allclose(kernel.own_marginal@orbit_law, own, atol=2e-15)
        rivals = joint.sum(axis=0)
        focal = independent_demand(self.grid, self.params, .3)
        expected = np.zeros(3)
        selected = np.zeros(3)
        independent_iid = np.zeros(3)
        for x in range(3):
            for rival_state in product(range(3), repeat=3):
                value = focal(x, tuple(sorted(rival_state)))
                expected[x] += value*rivals[rival_state]
                selected[x] += value*joint[(x,)+rival_state]
                independent_iid[x] += value*np.prod(own[list(rival_state)])
        assert_allclose(kernel.marginal_demand@orbit_law, expected, rtol=2e-11, atol=2e-13)
        assert_allclose(kernel.sale_marginal@orbit_law, selected, rtol=2e-11, atol=2e-13)
        self.assertGreater(float(np.max(abs(expected-independent_iid))), 1e-4)

    def test_calendar_derivatives_match_finite_differences(self):
        kernel = self.kernel
        alpha = np.array([.25, .45, .75])
        for k, delta in enumerate(kernel.transition_derivatives()):
            plus, minus = alpha.copy(), alpha.copy()
            plus[k] += 1e-5; minus[k] -= 1e-5
            finite = (kernel.transition(plus)-kernel.transition(minus))/(2e-5)
            assert_allclose(delta.toarray(), finite.toarray(), atol=2e-11, rtol=2e-9)
            assert_allclose(delta.sum(axis=1), 0, atol=2e-16)

    def test_jump_reconstructs_calendar_without_subtractive_exit_rates(self):
        kernel = self.kernel
        alpha = np.array([.2, .8, .1])
        jump, rates = kernel.jump(alpha)
        assert_allclose(jump.sum(axis=1), 1, atol=5e-15)
        assert_allclose(jump.diagonal(), 0)
        reconstructed = jump.multiply(rates[:, None]).toarray()
        reconstructed[np.diag_indices(kernel.size)] = 1-rates
        assert_allclose(reconstructed, kernel.transition(alpha).toarray(), atol=5e-15)

    def test_absorbing_outside_endpoint_and_symmetric_zero_outside(self):
        for p0 in (0., 1.):
            kernel = FourSellerKernel(self.grid, self.params, p0=p0, order=96)
            if p0 == 1:
                assert_allclose(kernel.transition(np.zeros(3)).toarray(), np.eye(kernel.size))
                with self.assertRaisesRegex(ArithmeticError, "absorbing"):
                    kernel.jump(np.zeros(3))
            else:
                diagonal = kernel.orbit_index(np.repeat(np.arange(3)[:, None], 4, axis=1))
                assert_allclose(kernel.demand[diagonal], .25, atol=2e-15)
                assert_allclose(kernel.outside, 0)

    def test_rare_exit_is_directly_positive(self):
        kernel = FourSellerKernel(self.grid, self.params, p0=1-1e-8, order=64)
        jump, rates = kernel.jump(np.zeros(3))
        self.assertGreater(rates[0], 0.)
        self.assertLess(rates[0], 1e-25)
        self.assertEqual(1-kernel.transition(np.zeros(3))[0, 0], 0.)
        assert_allclose(jump.sum(axis=1), 1, atol=5e-15)


if __name__ == "__main__":
    unittest.main()
