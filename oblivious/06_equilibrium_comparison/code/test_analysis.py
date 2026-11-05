"""Independent accounting and uncertainty checks for the comparison report."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import eye
from scipy.stats import t as student_t

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.equilibrium_comparison.code.analysis import (
    cdf_distance, interval, product_usage_evolution, summarize_window,
)


def accounting_fixture():
    """Small hand-recorded calendar events, deliberately imbalanced by seller."""
    grid = np.array([0., .5, 1.])
    params = Parameters(gamma=.5)
    marginal = np.array([.2, .5, .3])
    nu = np.outer(marginal, marginal)
    pair = dict(params=params, grid=grid, profiles={
        name: dict(policy=np.full(3, .37), nu=nu.copy(), mu=marginal.copy(), p0=.5,
                   transition=eye(9, format="csr"))
        for name in ("oe", "p0")
    })
    selected = np.array([[1, 1, 0, 0, 0, 1, 1, 0],
                         [0, 1, 1, 1, 0, 1, 0, 0]], dtype=np.uint8)
    product = np.array([[2, 2, 2, 1, 2, 1, 1, 2],
                        [1, 2, 2, 1, 1, 2, 2, 1]], dtype=np.uint8)
    success = np.array([[1, 0, 1, 0, 1, 0, 0, 1],
                        [0, 1, 1, 1, 0, 0, 1, 0]], dtype=bool)
    states = np.array([
        [[.5, .5], [0, .5], [0, 1], [.5, 1], [1, .5], [.5, 0], [0, 0], [1, 1]],
        [[.5, .5], [.5, 0], [1, 0], [1, .5], [.5, 1], [0, .5], [0, 0], [1, 1]],
    ])
    profit = np.zeros((2, 8, 2))
    profit[np.arange(2)[:, None], np.arange(8)[None, :], selected] = (
        params.R - np.where(product == 2, params.c2, params.c1))
    raw = dict(before=states, selected=selected, product=product,
               success=success, profit=profit, theta=.1 + .8 * states)
    simulation = {"profiles": {name: {key: value.copy() for key, value in raw.items()}
                               for name in ("oe", "p0")}}
    return pair, simulation


class DistributionDistances(unittest.TestCase):
    def test_exact_cdf_metric_handles_repeated_grid_values(self):
        grid = np.array([0., .5, 1.])
        empirical = np.array([0., 0., .5, .5, 1., 1., 1., 1.])
        self.assertEqual(cdf_distance(empirical, grid, [.25, .25, .5]), 0.)
        self.assertAlmostEqual(cdf_distance(empirical, grid, [.1, .7, .2]), .3)

    def test_cdf_metric_includes_left_limits_at_prediction_jumps(self):
        # F_emp(.5)=F_pred(.5)=1, but immediately before .5 they differ by 1.
        self.assertEqual(cdf_distance(np.array([.4, .4]), np.array([0., .5, 1.]), [0., 1., 0.]), 1.)
        # Reverse jump ordering must also be captured, including the endpoint.
        self.assertEqual(cdf_distance(np.array([1., 1.]), np.array([0., .5, 1.]), [0., 1., 0.]), 1.)


class ReplicationUncertainty(unittest.TestCase):
    def test_intervals_use_independent_replications_and_student_t(self):
        result = interval([1., 2., 3., 4.])
        expected_se = np.sqrt(5 / 3) / 2
        expected_halfwidth = student_t.ppf(.975, 3) * expected_se
        self.assertEqual(result["mean"], 2.5)
        self.assertAlmostEqual(result["se"], expected_se)
        self.assertAlmostEqual(result["ci_low"], 2.5 - expected_halfwidth)
        self.assertAlmostEqual(result["ci_high"], 2.5 + expected_halfwidth)
        self.assertEqual(result["n_valid"], 4)
        self.assertEqual(interval([np.nan])["n_valid"], 0)
        self.assertIsNone(interval([2.])["se"])

    def test_matched_profile_paths_have_exactly_zero_paired_intervals(self):
        pair, simulation = accounting_fixture()
        result = summarize_window(pair, simulation, mode="grid", window="baseline",
                                  start=2, length=4, short_periods=2)
        self.assertTrue(result["differences"])
        for row in result["differences"]:
            self.assertEqual(row["n_valid"], 2)
            assert_allclose([row[key] for key in ("mean", "se", "ci_low", "ci_high")], 0., atol=0)
        profits = next(r for r in result["summary"] if r["profile"] == "oe" and r["metric"] == "profit_A")
        self.assertGreater(profits["se"], 0.)


class ProductUsageEvolution(unittest.TestCase):
    def test_each_period_counts_all_paths_from_the_initial_purchase(self):
        _, simulation = accounting_fixture()
        product = simulation['profiles']['oe']['product']
        curve = product_usage_evolution(product, smoothing_window=3)
        assert_allclose(curve['time'], np.arange(8))
        assert_allclose(curve['q2'], [.5, 1, 1, 0, .5, .5, .5, .5])
        self.assertEqual(curve['n_replications'], 2)
        self.assertFalse(curve['burn_in_discarded'])

    def test_trailing_windows_keep_the_initial_period_and_exclude_future_events(self):
        # Across paths q2(t) is [0, 1, .5, 0]. A two-period trailing mean is
        # [0, .5, .75, .25], including a partial window at t=0.
        product = np.array([[1, 2, 2, 1], [1, 2, 1, 1]])
        curve = product_usage_evolution(product, smoothing_window=2)
        assert_allclose(curve['smoothed'], [0, .5, .75, .25])
        assert_allclose(curve['window_size'], [1, 2, 2, 2])
        changed_future = product.copy()
        changed_future[:, 2:] = 2
        changed = product_usage_evolution(changed_future, smoothing_window=2)
        assert_allclose(changed['smoothed'][:2], curve['smoothed'][:2])

    def test_uncertainty_uses_independent_paths_not_pooled_path_periods(self):
        # Final five-period frequencies are .2, .4, .6, .8 across four paths.
        product = np.array([[2, 1, 1, 1, 1], [2, 2, 1, 1, 1],
                            [2, 2, 2, 1, 1], [2, 2, 2, 2, 1]])
        curve = product_usage_evolution(product, smoothing_window=5)
        expected_se = np.sqrt(1 / 15) / 2
        halfwidth = student_t.ppf(.975, 3) * expected_se
        self.assertAlmostEqual(curve['smoothed'][-1], .5)
        self.assertAlmostEqual(curve['se'][-1], expected_se)
        assert_allclose([curve['ci_low'][-1], curve['ci_high'][-1]],
                        [.5 - halfwidth, .5 + halfwidth])

    def test_no_smoothing_and_window_longer_than_path_have_defined_endpoints(self):
        product = np.array([[1, 2, 2], [2, 1, 2]])
        unsmoothed = product_usage_evolution(product, smoothing_window=1)
        assert_allclose(unsmoothed['smoothed'], unsmoothed['q2'])
        cumulative = product_usage_evolution(product, smoothing_window=100)
        assert_allclose(cumulative['smoothed'], [.5, .5, 2 / 3])
        assert_allclose(cumulative['window_size'], [1, 2, 3])


class EconomicAccounting(unittest.TestCase):
    def setUp(self):
        self.pair, self.simulation = accounting_fixture()
        self.result = summarize_window(self.pair, self.simulation,
            mode="grid", window="baseline", start=2, length=4,
            short_periods=2, include_distributions=True)
        self.rows = [r for r in self.result["replications"] if r["profile"] == "oe"]

    def test_product_usage_distinguishes_purchase_and_calendar_denominators(self):
        a, b = self.rows
        self.assertEqual(a["market_share_A"], .75)
        self.assertEqual(a["market_share_B"], .25)
        self.assertAlmostEqual(a["product2_conditional_A"], 2 / 3)
        self.assertEqual(a["product2_calendar_A"], .5)
        self.assertEqual(a["product2_conditional_B"], 0.)
        self.assertEqual(a["product2_market"], .5)
        self.assertAlmostEqual(b["product2_conditional_B"], 2 / 3)
        for row in self.rows:
            self.assertEqual(row["product2_market"], row["product2_calendar_A"] + row["product2_calendar_B"])
            self.assertAlmostEqual(row["industry_profit"], .65)
            self.assertAlmostEqual(row["industry_profit"], row["profit_A"] + row["profit_B"])
            self.assertAlmostEqual(row["expected_success_probability"], .575)
            self.assertEqual(row["success_rate"] + row["failure_rate"], 1.)

    def test_discounted_profits_start_at_common_initial_state_not_burn_in(self):
        # First replication has only seller B selected in its first two
        # periods, both using product 2. The stationary window selects A often.
        row = self.rows[0]
        self.assertEqual(row["discounted_profit_A"], 0.)
        self.assertAlmostEqual(row["discounted_profit_B"], .35 + .5 * .35)
        self.assertGreater(row["profit_A"], 0.)
        self.assertAlmostEqual(row["discounted_industry_profit"], .525)

    def test_distributions_use_before_states_and_not_purchase_weighted_states(self):
        distribution = self.result["distributions"]["oe"]
        # Both sellers, averaged across paths: mass .25 at 0, .375 at .5,
        # .375 at 1. Selected-only reputations need not have these masses.
        expected = np.array([[.25, .375, .375], [.25, .375, .375]])
        assert_allclose(distribution["mass"], expected)
        assert_allclose(distribution["cdf"], np.cumsum(expected, axis=1))
        self.assertAlmostEqual(distribution["joint"].sum(), 1.)
        self.assertEqual(self.rows[0]["empirical_stationarity_residual"], 0.)

    def test_no_purchases_gives_missing_conditional_usage_not_zero(self):
        # A seller never selected has no observed conditional product usage.
        self.simulation["profiles"]["oe"]["selected"][:, 2:6] = 0
        result = summarize_window(self.pair, self.simulation, mode="grid", window="test",
                                  start=2, length=4, short_periods=2)
        rows = [r for r in result["replications"] if r["profile"] == "oe"]
        for row in rows:
            self.assertTrue(np.isnan(row["product2_conditional_B"]))
            self.assertEqual(row["product2_calendar_B"], 0.)


if __name__ == "__main__":
    unittest.main()
