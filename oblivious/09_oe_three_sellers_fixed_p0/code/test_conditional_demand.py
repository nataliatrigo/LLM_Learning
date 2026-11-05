"""Independent analytical and numerical checks for fixed-Beta seller demand."""

import math
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.special import betaincc
from scipy.stats import beta

from oblivious.oe_p0.code.primitives import Parameters, posterior
from oblivious.oe_three_sellers_fixed_p0.code.conditional_demand import (
    demand_by_seller_count,
)
from oblivious.oe_three_sellers_fixed_p0.code.kernel import fixed_outside_demand


def integer_beta_reference(sellers, p0):
    """Integrate 2*t*(2*t-t**2)**(N-1) by its polynomial expansion."""
    return math.fsum(
        2 * math.comb(sellers - 1, k) * 2**(sellers - 1 - k) * (-1)**k
        * (1 - p0**(sellers + k + 1)) / (sellers + k + 1)
        for k in range(sellers)
    )


class ConditionalDemandTests(unittest.TestCase):
    def test_uniform_closed_form_including_unsorted_repeated_seller_counts(self):
        counts = np.array([20, 1, 4, 2, 4])
        for p0 in (0., .1, .7, 1.):
            with self.subTest(p0=p0):
                actual, _ = demand_by_seller_count((1, 1), (1, 1), counts, p0=p0)
                assert_allclose(actual, (1 - p0**counts) / counts,
                                atol=3e-13, rtol=3e-12)

    def test_asymmetric_integer_betas_match_independent_polynomial_integral(self):
        counts = [1, 2, 4, 7]
        for p0 in (.2, .7):
            with self.subTest(p0=p0):
                actual, _ = demand_by_seller_count((2, 1), (1, 2), counts, p0=p0)
                expected = [integer_beta_reference(n, p0) for n in counts]
                assert_allclose(actual, expected, atol=3e-13, rtol=3e-12)

    def test_noninteger_betas_match_independent_adaptive_quadrature(self):
        own, rival = (2.3, 4.7), (1.4, 3.8)
        counts = [1, 2, 7, 20]
        for p0 in (0., .4):
            with self.subTest(p0=p0):
                actual, _ = demand_by_seller_count(own, rival, counts, p0=p0)
                expected = [
                    quad(lambda t: beta.pdf(t, *own) * beta.cdf(t, *rival)**(n - 1),
                         p0, 1, epsabs=2e-12, epsrel=2e-12, limit=300)[0]
                    for n in counts
                ]
                assert_allclose(actual, expected, atol=3e-10, rtol=3e-9)

    def test_insufficient_order_rejects_difficult_endpoint_shapes(self):
        # Shapes below one produce endpoint singularities beyond those in
        # the experiment's posterior family. Fail explicitly if unresolved.
        with self.assertRaises(ArithmeticError):
            demand_by_seller_count((.7, .8), (1.4, .8), [1, 2, 7, 20], p0=0.)

    def test_symmetric_rare_tail_uses_survival_probability_without_cancellation(self):
        # Here the CDF rounds to one, but selection still has positive mass.
        shapes, p0 = (2., 40.), .95
        counts = np.array([1, 2, 5, 20])
        tail = betaincc(*shapes, p0)
        self.assertGreater(tail, 0.)
        self.assertEqual(1 - tail, 1.)
        expected = -np.expm1(counts * np.log1p(-tail)) / counts
        actual, _ = demand_by_seller_count(shapes, shapes, counts, p0=p0)
        self.assertTrue(np.all(actual > 0))
        assert_allclose(actual, expected, atol=0, rtol=2e-10)

    def test_monopoly_tail_and_outside_endpoint_ignore_rival_beta(self):
        own = (2.3, 4.7)
        for rival in ((1., 1.), (.5, 30.)):
            with self.subTest(rival=rival):
                actual, _ = demand_by_seller_count(own, rival, [1], p0=.8)
                assert_allclose(actual, [betaincc(*own, .8)], atol=0, rtol=2e-11)
                actual, _ = demand_by_seller_count(own, rival, [1, 2, 20], p0=1.)
                assert_allclose(actual, np.zeros(3), atol=0, rtol=0)

    def test_demand_decreases_with_fixed_added_rivals_and_outside_quality(self):
        counts = np.arange(1, 21)
        previous = np.ones(len(counts))
        for p0 in (0., .1, .5, .9, 1.):
            actual, _ = demand_by_seller_count((3., 2.), (2., 4.), counts, p0=p0)
            self.assertTrue(np.all(actual >= 0))
            self.assertTrue(np.all(actual <= 1))
            self.assertTrue(np.all(np.diff(actual) <= 2e-13))
            self.assertTrue(np.all(actual <= previous + 2e-13))
            previous = actual

    def test_matches_existing_joint_state_demand_without_solving_equilibrium(self):
        grid, params = np.array([0., .5, 1.]), Parameters(rho=.6)
        a, b = posterior(grid, params)
        own, rival = (a[0], b[0]), (a[1], b[1])
        for sellers in (1, 2, 3):
            with self.subTest(sellers=sellers):
                joint, _, _ = fixed_outside_demand(grid, params, p0=.8, sellers=sellers)
                state = (0,) + (1,) * (sellers - 1)
                actual, _ = demand_by_seller_count(own, rival, [sellers], p0=.8)
                assert_allclose(actual, [joint[(0,) + state]], atol=2e-13, rtol=2e-10)

    def test_rejects_invalid_shapes_counts_outside_quality_and_order(self):
        valid = dict(own_beta=(2., 3.), rival_beta=(1., 2.), seller_counts=[1, 2])
        invalid = [
            ('own_beta', (0., 1.)), ('own_beta', (1.,)),
            ('rival_beta', (-1., 2.)), ('rival_beta', (1., np.nan)),
            ('rival_beta', (1., 2., 3.)),
            ('seller_counts', []), ('seller_counts', [0, 1]),
            ('seller_counts', [-1]), ('seller_counts', [1, 2.5]),
            ('seller_counts', [[1, 2]]), ('seller_counts', [1, np.inf]),
            ('p0', -.1), ('p0', 1.1), ('p0', np.nan),
            ('order', 0), ('order', 7), ('order', 8.5),
        ]
        for key, value in invalid:
            with self.subTest(key=key, value=value):
                options = dict(valid)
                options[key] = value
                with self.assertRaises(ValueError):
                    demand_by_seller_count(**options)


if __name__ == '__main__':
    unittest.main()
