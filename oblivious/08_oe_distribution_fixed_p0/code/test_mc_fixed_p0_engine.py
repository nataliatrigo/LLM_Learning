"""Independent event-protocol and aggregation checks; no equilibrium solves."""

from types import SimpleNamespace
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from oblivious.oe_p0.code.primitives import interpolation_matrix
from oblivious.oe_three_sellers_fixed_p0.code.choice_dynamics import simulate_step
from oblivious.oe_distribution_fixed_p0.code.mc_fixed_p0_engine import simulate_calendar_paths


class StreamingCalendarSimulation(unittest.TestCase):
    def test_matches_independent_step_protocol_for_mixed_policies(self):
        """Match every count and final state to the existing N-seller engine."""
        grid = np.linspace(0, 1, 7)
        policy = np.array([0., .2, 1., .65, .4, .8, 0.])
        rho, n_paths, periods, seed = .7, 103, 41, 20261026
        for sellers in (1, 2):
            with self.subTest(sellers=sellers):
                shape = (len(grid),)*sellers
                probabilities = np.random.default_rng(84+sellers).random((sellers+1,)+shape)
                probabilities /= probabilities.sum(axis=0, keepdims=True)
                kernel = SimpleNamespace(sellers=sellers, n=len(grid), grid=grid,
                    outside=probabilities[0], demand=probabilities[1:],
                    params=SimpleNamespace(p1=.35, p2=.8),
                    failure=interpolation_matrix(grid, rho*grid),
                    success=interpolation_matrix(grid, rho*grid+1-rho))
                game = SimpleNamespace(kernel=kernel, policy=policy)
                rng = np.random.default_rng(seed)
                states = np.full((n_paths, sellers), len(grid)//2, dtype=int)
                expected_counts = np.zeros((6, n_paths), dtype=int)
                period_counts, path_history = [], []
                for date in range(periods):
                    states, choice, product, _ = simulate_step(game, states, rng.random((n_paths, 4)))
                    own, b = choice == 1, choice == 2
                    own2, b2 = own & (product == 2), b & (product == 2)
                    events = np.array([own, own2, b, b2, choice == 0, product == 2])
                    expected_counts += events
                    period_counts.append(events.sum(axis=1))
                    path_history.append(events)
                result = simulate_calendar_paths(grid=grid, policy=policy,
                    selection=probabilities, rho=rho, p1=.35, p2=.8,
                    n_paths=n_paths, periods=periods, seed=seed, bin_width=13)
                columns = ["own_sale_count", "own_product2_count", "b_sale_count",
                           "b_product2_count", "outside_count", "customer_product2_count"]
                assert_array_equal(result["by_period"][columns], period_counts)
                assert_array_equal(result["paths"][columns].to_numpy().T, expected_counts)
                assert_array_equal(result["paths"].final_own_index, states[:, 0])
                if sellers == 2:
                    assert_array_equal(result["paths"].final_b_index, states[:, 1])
                for row in result["binned"].itertuples():
                    cluster = np.asarray(path_history[row.bin_start:row.bin_stop]).sum(axis=0)
                    own, own2, b, b2, outside, customer2 = cluster
                    estimate = own2.sum()/own.sum()
                    se = (own2-estimate*own).std(ddof=1)/(np.sqrt(n_paths)*own.mean())
                    self.assertAlmostEqual(row.product2_given_sale, estimate)
                    self.assertAlmostEqual(row.product2_given_sale_se, se)
                    self.assertAlmostEqual(row.customer_product2_rate, customer2.mean()/row.periods_in_bin)
                    self.assertAlmostEqual(row.own_sale_rate*row.product2_given_sale, row.own_product2_rate)
                    assert_array_equal(np.array([getattr(row, c) for c in columns]), cluster.sum(axis=1))

    def test_outside_freezes_both_reputations_and_keeps_calendar_denominator(self):
        grid = np.linspace(0, 1, 7)
        selection = np.zeros((3, 7, 7)); selection[0] = 1.
        result = simulate_calendar_paths(grid=grid, policy=np.ones(7),
            selection=selection, rho=.7, p1=.35, p2=.8,
            n_paths=17, periods=32, seed=22, bin_width=13, initial_index=[1, 5])
        assert_array_equal(result["paths"].outside_count, 32)
        assert_array_equal(result["paths"].final_own_index, 1)
        assert_array_equal(result["paths"].final_b_index, 5)
        assert_array_equal(result["binned"].periods_in_bin, [13, 13, 6])
        assert_array_equal(result["binned"].customer_product2_rate, 0.)
        assert_array_equal(result["binned"].outside_rate, 1.)
        self.assertTrue(result["binned"].product2_given_sale.isna().all())

    def test_customer_counts_the_rival_without_creating_an_own_sale(self):
        grid = np.linspace(0, 1, 5)
        selection = np.zeros((3, 5, 5)); selection[2] = 1.
        result = simulate_calendar_paths(grid=grid, policy=np.array([np.zeros(5), np.ones(5)]),
            selection=selection, rho=.7, p1=.35, p2=.8,
            n_paths=53, periods=29, seed=83)
        assert_array_equal(result["paths"].own_sale_count, 0)
        assert_array_equal(result["paths"].b_sale_count, 29)
        assert_array_equal(result["paths"].customer_product2_count, 29)
        assert_array_equal(result["paths"].final_own_index, 2)
        assert_array_equal(result["binned"].customer_product2_rate, 1.)
        self.assertTrue(result["binned"].product2_given_sale.isna().all())

    def test_invalid_selection_partition_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "sum to one"):
            simulate_calendar_paths(grid=np.linspace(0, 1, 5), policy=np.ones(5),
                selection=np.full((2, 5), .4), rho=.7, p1=.35, p2=.8,
                n_paths=2, periods=1, seed=0)


if __name__ == "__main__":
    unittest.main()
