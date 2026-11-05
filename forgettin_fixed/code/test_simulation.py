"""Independent count-memory bookkeeping and seeded simulation validation."""
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.special import betaincc

from forgettin_fixed.code.model import Primitives, solve_counts
from forgettin_fixed.code.simulation import (
    CountState, _mean_interval, _ratio_interval,
    simulate_calendar, simulate_interactions,
)


def scalar_advance(m, successes, incoming, capacity):
    """Independent scalar rule with explicit type of the replaced result."""
    if m < capacity:
        return m + 1, successes + int(incoming)
    failures = m - successes
    if incoming and failures:
        return m, successes + 1
    if not incoming and successes:
        return m, successes - 1
    return m, successes


class SimulationTests(unittest.TestCase):
    def test_opposite_replacement_reflects_at_boundaries(self):
        state = CountState(SimpleNamespace(n=3), 5)
        state.lengths[:] = 3
        state.counts[:] = [0, 3, 0, 3, 2]
        state.advance([True, True, True, True, False], [False, True, True, False, False])
        np.testing.assert_array_equal(state.counts, [0, 3, 1, 2, 2])
        np.testing.assert_array_equal(state.lengths, [3] * 5)

    def test_scalar_bookkeeping_and_frozen_outside_paths(self):
        rng = np.random.default_rng(7319)
        for n in (1, 3, 20, 1000):
            with self.subTest(n=n):
                state = CountState(SimpleNamespace(n=n), 7)
                reference = [(0, 0)] * 7
                for _ in range(2 * n + 19):
                    buyers = rng.random(7) < .57
                    successes = rng.random(7) < .61
                    old_lengths, old_counts = state.lengths.copy(), state.counts.copy()
                    state.advance(buyers, successes)
                    for path in range(7):
                        if buyers[path]:
                            reference[path] = scalar_advance(
                                *reference[path], successes[path], n
                            )
                        self.assertEqual((state.lengths[path], state.counts[path]),
                                         reference[path])
                    np.testing.assert_array_equal(state.lengths[~buyers], old_lengths[~buyers])
                    np.testing.assert_array_equal(state.counts[~buyers], old_counts[~buyers])
                    self.assertTrue(np.all((state.counts >= 0) & (state.counts <= state.lengths)))

    def test_interaction_records_precede_outcome(self):
        for n in (1, 3, 20, 1000):
            with self.subTest(n=n):
                policy = solve_counts(Primitives(), n)
                records, summary = simulate_interactions(
                    policy, paths=4, interactions=75, seed=724, burn_in=7
                )
                for path in range(4):
                    m, successes = 0, 0
                    for time in range(75):
                        row = records[time, path]
                        self.assertEqual((row['m_before'], row['S_before']), (m, successes))
                        self.assertEqual(row['product2'],
                                         policy.advantages[m][successes] > policy.action_tolerance)
                        demand = betaincc(1 + successes, 1 + m - successes,
                                         policy.primitives.p0)
                        self.assertAlmostEqual(float(row['demand']), demand, places=7)
                        m, successes = scalar_advance(m, successes, row['success'], n)
                    self.assertEqual(summary.loc[path, 'final_remembered_successes'], successes)
                    self.assertEqual(summary.loc[path, 'final_window_length'], m)
                    self.assertEqual(summary.loc[path, 'post_burn_in_frequency'],
                                     records['product2'][7:, path].mean())

    def test_calendar_matches_independent_scalar_replay(self):
        paths, periods, seed, bin_width = 7, 61, 98217, 8
        for n in (1, 3, 20, 1000):
            with self.subTest(n=n):
                primitives = Primitives()
                policy = solve_counts(primitives, n)
                actual, bins, path_summary = simulate_calendar(
                    policy, n_paths=paths, max_periods=periods,
                    seed=seed, bin_width=bin_width,
                )
                memory = [(0, 0)] * paths
                observed, total_successes = np.zeros(paths, int), np.zeros(paths, int)
                all_buyers = np.zeros((periods, paths), dtype=bool)
                all_p2 = np.zeros_like(all_buyers)
                rng = np.random.default_rng(seed)
                for time in range(periods):
                    buyer_uniforms, outcome_uniforms = rng.random(paths), rng.random(paths)
                    self.assertEqual(actual.loc[time, 'mean_observation_n'], observed.mean())
                    self.assertEqual(actual.loc[time, 'mean_remembered_successes'],
                                     np.mean([successes for _, successes in memory]))
                    self.assertEqual(actual.loc[time, 'mean_window_length'],
                                     np.mean([m for m, _ in memory]))
                    for path, (m, successes) in enumerate(memory):
                        p2 = policy.advantages[m][successes] > policy.action_tolerance
                        demand = betaincc(1 + successes, 1 + m - successes, primitives.p0)
                        buyer = buyer_uniforms[path] < demand
                        incoming = outcome_uniforms[path] < (primitives.p2 if p2 else primitives.p1)
                        all_buyers[time, path] = buyer
                        all_p2[time, path] = buyer and p2
                        if buyer:
                            memory[path] = scalar_advance(m, successes, incoming, n)
                            observed[path] += 1
                            total_successes[path] += incoming
                    self.assertEqual(actual.loc[time, 'a_chosen_count'], all_buyers[time].sum())
                    self.assertEqual(actual.loc[time, 'product2_chosen_count'], all_p2[time].sum())
                np.testing.assert_array_equal(path_summary.final_observation_n, observed)
                np.testing.assert_array_equal(path_summary.final_successes, total_successes)
                np.testing.assert_array_equal(path_summary.final_remembered_successes,
                                              [successes for _, successes in memory])
                np.testing.assert_array_equal(path_summary.product2_count, all_p2.sum(axis=0))
                for b, row in bins.iterrows():
                    start, stop = b * bin_width, min((b + 1) * bin_width, periods)
                    a, p2 = all_buyers[start:stop], all_p2[start:stop]
                    self.assertEqual(row.a_chosen_count, a.sum())
                    self.assertEqual(row.product2_chosen_count, p2.sum())
                    self.assertAlmostEqual(row.a_market_share, a.mean())
                    self.assertAlmostEqual(row.product2_unconditional_rate, p2.mean())
                    if a.any():
                        self.assertAlmostEqual(row.product2_given_a_rate, p2.sum() / a.sum())

    def test_cluster_interval_uses_path_totals(self):
        numerator = np.array([1., 5., 2., 0.])
        denominator = np.array([3., 8., 5., 1.])
        estimate, se, lower, upper = _ratio_interval(numerator, denominator)
        expected = numerator.sum() / denominator.sum()
        expected_se = np.std(numerator - expected * denominator, ddof=1) / (
            np.sqrt(len(numerator)) * denominator.mean()
        )
        self.assertAlmostEqual(estimate, expected)
        self.assertAlmostEqual(se, expected_se)
        self.assertLess(lower, estimate)
        self.assertGreater(upper, estimate)
        self.assertTrue(np.isnan(_ratio_interval([0, 0], [0, 0])[0]))
        self.assertTrue(np.isnan(_mean_interval([1])[1]))


if __name__ == '__main__':
    unittest.main()
