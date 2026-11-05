"""Pinned search must preserve the frozen model and its certificate."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from oblivious.oe_p0.code.primitives import Parameters
from .model import Experiment
from .search_only import FastExperiment


class PinnedSearchTests(unittest.TestCase):
    def test_values_law_and_exact_jacobian_match_frozen_search(self):
        for sellers, closure, rho, p0 in ((2, 'marginal', .7, .5),
                                           (2, 'conditional', .8, .9),
                                           (3, 'marginal', .7, .5),
                                           (3, 'conditional', .8, .9)):
            with self.subTest(sellers=sellers, closure=closure, rho=rho, p0=p0):
                options = dict(n=7, sellers=sellers, closure=closure, p0=p0,
                               params=Parameters(rho=rho, gamma=.95))
                original, fast = Experiment(**options), FastExperiment(**options)
                alpha = np.array([.2, .3, .8, .7, .6, .4, .1])
                expected, actual = original.calculate(alpha), fast.calculate(alpha)
                for key in ('value', 'advantage', 'demand', 'reduced_law', 'regret'):
                    assert_allclose(actual[key], expected[key], atol=3e-11, rtol=3e-7, err_msg=key)
                assert_allclose(actual['jacobian'], expected['jacobian'], atol=2e-9, rtol=3e-6)

    def test_pinned_jacobian_matches_finite_differences(self):
        for closure in ('marginal', 'conditional'):
            with self.subTest(closure=closure):
                fast = FastExperiment(n=5, sellers=3, closure=closure, p0=.8,
                                      params=Parameters(rho=.7, gamma=.95))
                alpha = np.array([.2, .8, .7, .4, .1])
                analytical = fast.calculate(alpha)['jacobian']
                numerical = np.empty((5, 5))
                for column in range(5):
                    delta = np.zeros(5)
                    delta[column] = 2e-6
                    plus = fast.calculate(alpha+delta)['advantage'].copy()
                    minus = fast.calculate(alpha-delta)['advantage'].copy()
                    numerical[:, column] = (plus-minus)/4e-6
                assert_allclose(analytical, numerical, atol=2e-7, rtol=3e-5)

    def test_jacobian_is_lazy_cached_and_certificate_is_inherited(self):
        fast = FastExperiment(n=5, sellers=3, closure='marginal', p0=.5,
                              params=Parameters(rho=.6, gamma=0))
        alpha = np.zeros(5)
        result = fast.calculate(alpha)
        self.assertNotIn('jacobian', result)
        self.assertIs(fast.calculate(alpha.copy()), result)
        first = result['jacobian']
        self.assertIs(result['jacobian'], first)
        self.assertIs(FastExperiment.certify, Experiment.certify)
        self.assertIs(FastExperiment.solve, Experiment.solve)
        certificate = fast.solve(initial_policy=alpha)
        self.assertTrue(certificate['verified'])
        self.assertLess(certificate['diagnostics']['conditional_regret'], 1e-7)
        self.assertEqual(certificate['summary']['product2_conditional'], 0.)

    def test_old_lazy_jacobian_survives_releasing_its_large_factor(self):
        options = dict(n=5, sellers=3, closure='marginal', p0=.5,
                       params=Parameters(rho=.7))
        original, fast = Experiment(**options), FastExperiment(**options)
        alpha = np.array([.2, .8, .7, .3, .1])
        pending = fast.calculate(alpha)
        fast.calculate(np.full(5, .5))
        # The previous LU was released to bound peak memory, but the old
        # result remains mathematically valid and may refactor on demand.
        assert_allclose(pending['jacobian'], original.calculate(alpha)['jacobian'],
                        atol=2e-9, rtol=3e-6)


if __name__ == '__main__':
    unittest.main()
