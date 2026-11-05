"""Economic comparisons against analytical myopic and sale-time identities."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment
from oblivious.oe_distribution_fixed_p0.code.run_experiment import economic_outcomes, solve_monopoly


class FixedOutsideEconomicComparison(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = Parameters(rho=.6, gamma=0.)
        cls.n = 7
        cls.experiment = FixedOutsideMarginalExperiment(n=cls.n, params=cls.params, p0=.5)
        cls.monopoly = solve_monopoly(cls.params, cls.n, .5)
        cls.competition = cls.experiment.evaluate(np.zeros(cls.n))
        full = cls.experiment.game.evaluate(np.zeros((cls.n, cls.n)), np.zeros((cls.n, cls.n)))
        cls.competition['actual_values'] = full['values']
        cls.rows, cls.policies = economic_outcomes(cls.params, cls.n, .5,
            cls.monopoly, cls.competition, cls.experiment.game)

    def test_analytical_initial_values_include_outside_and_per_seller_denominator(self):
        monopoly, competition = self.rows
        # Symmetric Beta at x=.5: F(.5)=.5. With 2 sellers, each wins
        # (1-F(.5)^2)/2 = .375; a sole seller wins 1-F(.5) = .5.
        self.assertAlmostEqual(monopoly['value_initial'], .5*.95, places=12)
        self.assertAlmostEqual(competition['value_initial'], .375*.95, places=12)
        self.assertEqual(monopoly['product2_conditional'], 0.)
        self.assertEqual(competition['product2_conditional'], 0.)

    def test_calendar_flows_count_own_sales_and_industry_totals_separately(self):
        for row in self.rows:
            self.assertAlmostEqual(row['profit_calendar'], .95*row['sale_rate'], places=12)
            self.assertAlmostEqual(row['sale_weighted_reputation'], .35, places=11)
            self.assertLess(row['sale_law_l1'], 1e-10)
            sellers = 2 if row['model']=='competition' else 1
            self.assertAlmostEqual(row['industry_profit_calendar'], sellers*row['profit_calendar'], places=12)
            self.assertAlmostEqual(row['outside_rate'] + sellers*row['sale_rate'], 1., places=11)

    def test_actual_value_is_not_replaced_by_perceived_marginal_value(self):
        comp = self.competition
        center = self.n//2
        assert_allclose(comp['actual_values'][0], .95*self.experiment.game.D, atol=2e-14)
        assert_allclose(comp['actual_values'][1], .95*self.experiment.game.D.T, atol=2e-14)
        self.assertAlmostEqual(self.rows[1]['value_initial'], comp['actual_values'][0, center, center], places=12)
        self.assertGreater(abs(self.rows[1]['value_initial']-self.rows[1]['perceived_value_initial']), 1e-3)

    def test_unverified_policy_cannot_be_reported_as_an_equilibrium(self):
        unverified = dict(self.competition, verified=False)
        with self.assertRaisesRegex(ArithmeticError, 'Unverified policy'):
            economic_outcomes(self.params, self.n, .5, self.monopoly,
                              unverified, self.experiment.game)


if __name__ == '__main__':
    unittest.main()
