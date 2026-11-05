"""Independent sensitivities, closure, and economic timing checks."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.special import betaincc
from scipy.sparse import eye
from scipy.sparse.linalg import spsolve

from oblivious.oe_p0.code.primitives import Parameters, posterior
from oblivious.oe_p0.code.policy import solve_dp
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment
from oblivious.oe_distribution_fixed_p0.code.conditional_model import FixedOutsideConditionalExperiment
from oblivious.oe_three_sellers_fixed_p0.code.model import Experiment
from oblivious.oe_three_sellers_fixed_p0.code.stationary import checked_stationary


class ExperimentSensitivityTests(unittest.TestCase):
    def test_analytic_equilibrium_jacobian_matches_finite_differences(self):
        # Change the number of sellers AND the closure. A conditional-demand
        # derivative must differentiate its own-state denominator as well.
        for sellers, n in ((2, 7), (3, 5)):
            for closure in ('marginal', 'conditional'):
                with self.subTest(sellers=sellers, closure=closure):
                    experiment = Experiment(n=n, sellers=sellers, closure=closure,
                                            p0=.5, params=Parameters(rho=.6, gamma=.95))
                    alpha = .25 + .5*np.linspace(0, 1, n)
                    analytical = experiment.calculate(alpha)['jacobian'].copy()
                    numerical = np.empty((n, n))
                    step = 2e-6
                    for column in range(n):
                        delta = np.zeros(n)
                        delta[column] = step
                        plus = experiment.calculate(alpha+delta)['advantage'].copy()
                        minus = experiment.calculate(alpha-delta)['advantage'].copy()
                        numerical[:, column] = (plus-minus)/(2*step)
                    assert_allclose(analytical, numerical, atol=2e-7, rtol=3e-5)

    def test_reduced_chain_law_and_beliefs_match_independent_full_chain(self):
        for closure in ('marginal', 'conditional'):
            with self.subTest(closure=closure):
                experiment = Experiment(n=5, sellers=3, closure=closure, p0=.8,
                                        params=Parameters(rho=.6, gamma=.9))
                alpha = np.array([.1, .9, .7, .4, .2])
                calculated = experiment.calculate(alpha)
                jump, rates = experiment.game.jump(alpha)
                independent = checked_stationary(jump, rates)['law']
                quotient_law = calculated['reduced_law'][experiment.inverse]/experiment.counts[experiment.inverse]
                assert_allclose(quotient_law, independent, atol=2e-12, rtol=2e-8)
                assert_allclose(calculated['demand'], experiment.belief_demand(independent),
                                atol=2e-12, rtol=2e-8)


class BeliefClosureTests(unittest.TestCase):
    def test_three_seller_marginal_preserves_correlated_joint_rival_law(self):
        experiment = Experiment(n=5, sellers=3, closure='marginal', p0=.3,
                                params=Parameters(rho=.6, gamma=.9))
        own = np.full(5, .2)
        rival = np.array([.1, .2, .4, .2, .1])
        joint = np.zeros((5, 5, 5))
        for y in range(5):
            joint[:, y, y] = own*rival[y]
        demand = experiment.game.demand[0]
        exact = np.array([sum(demand[x, y, y]*rival[y] for y in range(5)) for x in range(5)])
        incorrect_independent = np.einsum('xyz,y,z->x', demand, rival, rival)
        assert_allclose(experiment.belief_demand(joint), exact, atol=1e-14)
        self.assertGreater(float(np.max(abs(exact-incorrect_independent))), .01)

    def test_conditional_closure_retains_every_positive_row_without_floor(self):
        experiment = Experiment(n=5, sellers=3, closure='conditional', p0=.8,
                                params=Parameters(rho=.6, gamma=.9))
        joint = np.zeros((5, 5, 5))
        joint[0, 4, 4] = 1e-200
        joint[1, 1, 2] = .25
        joint[2, 2, 1] = .25
        joint[3, 0, 0] = .5
        demand = experiment.belief_demand(joint)
        self.assertAlmostEqual(demand[0]/experiment.game.demand[0, 0, 4, 4], 1., places=13)
        assert_allclose(demand[1:4], [experiment.game.demand[0, 1, 1, 2],
                                     experiment.game.demand[0, 2, 2, 1],
                                     experiment.game.demand[0, 3, 0, 0]], atol=1e-14)
        # Only an exactly zero row uses the joint unconditional completion.
        unconditional = np.sum(experiment.game.demand[0, 4]*joint.sum(axis=0))
        self.assertAlmostEqual(demand[4], unconditional, places=14)

    def test_two_seller_fixed_policy_matches_historical_model(self):
        params = Parameters(rho=.6, gamma=.9)
        alpha = np.array([.2, .8, .6, .3, .1])
        for closure, original_type in (('marginal', FixedOutsideMarginalExperiment),
                                       ('conditional', FixedOutsideConditionalExperiment)):
            with self.subTest(closure=closure):
                new = Experiment(n=5, sellers=2, closure=closure, p0=.5, params=params)
                calculated = new.calculate(alpha)
                original = original_type(n=5, params=params, p0=.5, order=256).evaluate(alpha)
                assert_allclose(calculated['demand'], original['demand'], atol=3e-12, rtol=1e-9)
                assert_allclose(calculated['value'], original['value'], atol=2e-10, rtol=1e-9)
                assert_allclose(calculated['advantage'], original['advantage'], atol=2e-11, rtol=1e-9)


class EconomicCertificateTests(unittest.TestCase):
    def test_monopoly_policy_and_value_match_exact_beta_tail_dp(self):
        params = Parameters(rho=.6)
        experiment = Experiment(n=7, sellers=1, closure='monopoly', p0=.8, params=params)
        a, b = posterior(experiment.grid, params)
        exact_demand = betaincc(a, b, .8)
        expected = solve_dp(experiment.grid, exact_demand, params)
        result = experiment.solve()
        self.assertTrue(result['verified'])
        assert_allclose(result['demand'], exact_demand, atol=1e-13)
        assert_allclose(result['policy'], expected['policy'], atol=1e-12)
        assert_allclose(result['value'], expected['value'], atol=3e-10)
        assert_allclose(result['actual_value'], expected['value'], atol=1e-9)

    def test_actual_discounted_value_solves_full_calendar_system(self):
        experiment = Experiment(n=5, sellers=3, closure='marginal', p0=.8,
                                params=Parameters(rho=.6, gamma=.92))
        alpha = np.array([.1, .8, .7, .2, .1])
        result = experiment.certify(alpha)
        p = experiment.params
        reward = experiment.game.demand[0]*(p.R-p.c1-(p.c2-p.c1)*alpha[:, None, None])
        calendar = experiment.game.transition(alpha)
        exact = spsolve(eye(experiment.game.size, format='csr')-p.gamma*calendar, reward.ravel())
        assert_allclose(result['actual_value'].ravel(), exact, atol=2e-10, rtol=1e-10)
        self.assertAlmostEqual(result['summary']['value_initial'], exact.reshape(experiment.game.shape)[2, 2, 2], places=9)
        # A fixed-policy evaluation is not automatically an equilibrium.
        self.assertFalse(result['verified'])

    def test_myopic_conditional_certificate_checks_all_rows_and_market_partition(self):
        experiment = Experiment(n=7, sellers=3, closure='conditional', p0=.9,
                                params=Parameters(rho=.8, gamma=0))
        result = experiment.certify(np.zeros(7))
        self.assertTrue(result['verified'])
        own = result['joint'].sum(axis=(1, 2))
        self.assertTrue(np.all(own > 0))
        self.assertLess(result['diagnostics']['conditional_row_stability_l1'], 1e-7)
        self.assertLess(result['diagnostics']['sale_law_l1'], 1e-9)
        summary = result['summary']
        self.assertAlmostEqual(summary['outside_rate']+3*summary['sale_rate'], 1., places=12)
        self.assertEqual(summary['product2_conditional'], 0.)
        assert_allclose(result['actual_value'], .95*experiment.game.demand[0], atol=1e-14)
        self.assertAlmostEqual(summary['value_initial'], .95*experiment.game.demand[0, 3, 3, 3], places=14)


if __name__ == '__main__':
    unittest.main()
