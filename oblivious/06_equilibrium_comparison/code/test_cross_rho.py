"""Independent checks for cross-rho calendar accounting and path-level CIs."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.stats import t

from oblivious.equilibrium_comparison.code.cross_rho import summarize_profile
from oblivious.equilibrium_comparison.code.test_analysis import accounting_fixture


class CrossRhoAccounting(unittest.TestCase):
    def setUp(self):
        self.pair, simulation = accounting_fixture()
        self.raw = simulation['profiles']['oe']

    def test_profit_averages_both_sellers_per_calendar_period(self):
        result = summarize_profile(self.pair['grid'], self.raw, start=2, length=4)
        for row in result['replications']:
            # Two product-2 and two product-1 purchases in each four-period path.
            self.assertEqual(row['product2_market'], .5)
            self.assertAlmostEqual(row['seller_profit'], .325)
        self.assertAlmostEqual(result['summary']['seller_profit']['mean'], .325)
        self.assertAlmostEqual(result['summary']['seller_profit']['se'], 0.)

    def test_cdf_pools_sellers_before_replication_interval(self):
        result = summarize_profile(self.pair['grid'], self.raw, start=0, length=8)
        d = result['distribution']
        # Each path has the same pooled mass, although its sellers differ.
        assert_allclose(d['cdf_replications'], [[5/16, 11/16, 1.], [5/16, 11/16, 1.]])
        assert_allclose(d['cdf_low'], d['cdf_high'])
        assert_allclose(d['joint'].sum(), 1.)
        pooled = (d['joint'].sum(axis=0) + d['joint'].sum(axis=1)) / 2
        assert_allclose(d['cdf'], np.cumsum(pooled))

    def test_ci_uses_independent_path_means_and_not_periods(self):
        # Product-2 fractions .25 and .75: sample SE .25, not a Bernoulli SE.
        self.raw['product'] = np.array([[2, 1, 1, 1], [2, 2, 2, 1]], dtype=np.uint8)
        self.raw['success'] = self.raw['success'][:, :4]
        self.raw['before'] = self.raw['before'][:, :4]
        self.raw['profit'] = self.raw['profit'][:, :4]
        result = summarize_profile(self.pair['grid'], self.raw, start=0, length=4)
        estimate = result['summary']['product2_market']
        self.assertEqual(estimate['n_valid'], 2)
        self.assertEqual(estimate['se'], .25)
        assert_allclose([estimate['ci_low'], estimate['ci_high']],
                        [.5 - t.ppf(.975, 1) * .25, .5 + t.ppf(.975, 1) * .25])

    def test_empirical_cdf_uses_actual_continuous_states(self):
        self.raw['before'][:] = .49
        d = summarize_profile(self.pair['grid'], self.raw, start=0, length=8)['distribution']
        assert_allclose(d['cdf'], [0, 1, 1])
        self.raw['before'][:] = .51
        d = summarize_profile(self.pair['grid'], self.raw, start=0, length=8)['distribution']
        assert_allclose(d['cdf'], [0, 0, 1])

    def test_invalid_window_does_not_silently_truncate(self):
        with self.assertRaises(ValueError):
            summarize_profile(self.pair['grid'], self.raw, start=5, length=5)


if __name__ == '__main__':
    unittest.main()
