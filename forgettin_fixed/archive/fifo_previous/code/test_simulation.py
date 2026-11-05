"""Independent FIFO bookkeeping and seeded calendar-path validation."""

from collections import deque
from types import SimpleNamespace
import unittest

import numpy as np
from scipy.special import betaincc

from forgettin_fixed.code.model import Primitives, solve_window
from forgettin_fixed.code.large_window import solve_large_window
from forgettin_fixed.code.simulation import (
    _FIFOState, simulate_calendar, simulate_interactions,
)


def _word(history):
    result = 0
    for outcome in history:
        result = 2 * result + int(outcome)
    return result


class SimulationTests(unittest.TestCase):
    def test_projected_policy_executes_actual_fifo(self):
        policy = solve_large_window(5, Primitives(), prefix_bits=2)
        state = _FIFOState(policy, 6)
        histories = [deque(maxlen=5) for _ in range(6)]
        rng = np.random.default_rng(32891)
        for _ in range(40):
            actions, demands = state.evaluate()
            queue_actions, queue_demands = policy.evaluate_queue(state.queue, state.lengths)
            np.testing.assert_array_equal(actions, queue_actions)
            np.testing.assert_array_equal(demands, queue_demands)
            for path, history in enumerate(histories):
                m, successes = len(history), sum(history)
                expected_demand = betaincc(1 + successes, 1 + m - successes,
                                           policy.primitives.p0)
                self.assertAlmostEqual(demands[path], expected_demand)
                self.assertEqual(state.counts[path], successes)
                if m == policy.n:
                    # Evaluate true shifted ordered queues explicitly, using
                    # only the planner's lifted terminal value function.
                    value_successors = []
                    for outcome in (0, 1):
                        nxt = list(history)[1:] + [outcome]
                        prefix = _word(nxt[:policy.prefix_bits])
                        tail_successes = sum(nxt[policy.prefix_bits:])
                        value_successors.append(policy.mature_value[prefix, tail_successes])
                    gap = -policy.primitives.delta_c + policy.primitives.gamma * (
                        policy.primitives.delta_p *
                        (value_successors[1] - value_successors[0])
                    )
                    self.assertEqual(actions[path], gap > policy.action_tolerance)
            buyers, successes = rng.random(6) < .7, rng.random(6) < .6
            state.advance(buyers, successes)
            for path in np.flatnonzero(buyers):
                histories[path].append(int(successes[path]))

    def test_projected_residual_covers_full_ordered_state_space(self):
        """Enumerate all true FIFO words independently of projected contexts."""
        n, k, primitives = 5, 2, Primitives()
        policy = solve_large_window(n, primitives, prefix_bits=k, tolerance=1e-11)
        exact = solve_window(primitives, n, tolerance=1e-12)
        words = list(range(1 << n))
        lifted = np.array([policy.mature_value[
            h >> (n-k), (h & ((1 << (n-k)) - 1)).bit_count()
        ] for h in words])
        errors = []
        for h in words:
            failure = (h << 1) & ((1 << n) - 1)
            success = failure | 1
            demand = betaincc(1+h.bit_count(), 1+n-h.bit_count(), primitives.p0)
            wait = demand / (1-primitives.gamma*(1-demand))
            q_values = [primitives.net_rewards[i] + primitives.gamma * (
                prob*lifted[success] + (1-prob)*lifted[failure]
            ) for i, prob in enumerate((primitives.p1, primitives.p2))]
            errors.append(abs(wait*max(q_values)-lifted[h]))
        self.assertAlmostEqual(max(errors), policy.diagnostics['fifo_bellman_residual_sup'])
        self.assertLessEqual(np.max(np.abs(lifted-exact.values[n])),
                             policy.diagnostics['uniform_value_error_bound'])

    def test_fifo_eviction_and_outside_option(self):
        rng = np.random.default_rng(7319)
        for n in (1, 3, 20, 100):
            with self.subTest(n=n):
                if n <= 20:
                    # Bookkeeping needs only the presence of this interface;
                    # avoid allocating the large n=20 policy in a unit test.
                    policy = SimpleNamespace(n=n, evaluate=lambda m, h: None)
                else:
                    policy = SimpleNamespace(n=n)
                state = _FIFOState(policy, 7)
                histories = [deque(maxlen=n) for _ in range(7)]
                for _ in range(2 * n + 19):
                    buyers = rng.random(7) < .57
                    success = rng.random(7) < .61
                    old_lengths = state.lengths.copy()
                    old_counts = state.counts.copy()
                    old_words = None if state.codes is None else state.codes.copy()
                    old_queue = None if state.queue is None else state.queue.copy()
                    state.advance(buyers, success)
                    for path in range(7):
                        if buyers[path]:
                            histories[path].append(int(success[path]))
                        self.assertEqual(state.lengths[path], len(histories[path]))
                        self.assertEqual(state.counts[path], sum(histories[path]))
                        if state.codes is not None:
                            self.assertEqual(int(state.codes[path]), _word(histories[path]))
                        else:
                            np.testing.assert_array_equal(
                                state.queue[path, :len(histories[path])], list(histories[path])
                            )
                    np.testing.assert_array_equal(state.lengths[~buyers], old_lengths[~buyers])
                    np.testing.assert_array_equal(state.counts[~buyers], old_counts[~buyers])
                    if state.codes is not None:
                        np.testing.assert_array_equal(state.codes[~buyers], old_words[~buyers])
                    else:
                        np.testing.assert_array_equal(state.queue[~buyers], old_queue[~buyers])

    def test_interaction_records_are_before_outcome(self):
        primitives = Primitives()
        for n in (1, 2, 3):
            with self.subTest(n=n):
                policy = solve_window(primitives, n)
                records, summary = simulate_interactions(
                    policy, paths=4, interactions=65, seed=724, burn_in=7
                )
                for path in range(4):
                    history = deque(maxlen=n)
                    for time in range(65):
                        row = records[time, path]
                        self.assertEqual(row['m_before'], len(history))
                        self.assertEqual(row['S_before'], sum(history))
                        m, h = len(history), _word(history)
                        self.assertEqual(row['product2'],
                                         policy.advantages[m][h] > policy.action_tolerance)
                        demand = betaincc(1 + sum(history), 1 + m - sum(history),
                                         primitives.p0)
                        self.assertAlmostEqual(float(row['demand']), demand, places=7)
                        history.append(int(row['success']))
                    self.assertEqual(summary.loc[path, 'final_remembered_successes'],
                                     sum(history))
                    self.assertEqual(summary.loc[path, 'final_window_length'], len(history))
                    self.assertEqual(summary.loc[path, 'post_burn_in_frequency'],
                                     records['product2'][7:, path].mean())

    def test_calendar_matches_independent_scalar_replay(self):
        """Replay random draws with Python deques and direct posterior calls."""
        paths, periods, seed, bin_width = 7, 47, 98217, 8
        for n in (1, 2, 3):
            with self.subTest(n=n):
                primitives = Primitives()
                policy = solve_window(primitives, n)
                actual, bins, path_summary = simulate_calendar(
                    policy, n_paths=paths, max_periods=periods,
                    seed=seed, bin_width=bin_width,
                )
                histories = [deque(maxlen=n) for _ in range(paths)]
                observed = np.zeros(paths, dtype=int)
                total_successes = np.zeros(paths, dtype=int)
                all_buyers = np.zeros((periods, paths), dtype=bool)
                all_p2 = np.zeros_like(all_buyers)
                rng = np.random.default_rng(seed)
                for time in range(periods):
                    buyer_uniforms = rng.random(paths)
                    outcome_uniforms = rng.random(paths)
                    self.assertEqual(actual.loc[time, 'mean_observation_n'], observed.mean())
                    self.assertEqual(actual.loc[time, 'mean_remembered_successes'],
                                     np.mean([sum(h) for h in histories]))
                    self.assertEqual(actual.loc[time, 'mean_window_length'],
                                     np.mean([len(h) for h in histories]))
                    for path, history in enumerate(histories):
                        m, h = len(history), _word(history)
                        p2 = policy.advantages[m][h] > policy.action_tolerance
                        demand = betaincc(1 + sum(history), 1 + m - sum(history),
                                         primitives.p0)
                        buyer = buyer_uniforms[path] < demand
                        success = outcome_uniforms[path] < (primitives.p2 if p2 else primitives.p1)
                        all_buyers[time, path] = buyer
                        all_p2[time, path] = buyer and p2
                        if buyer:
                            history.append(int(success))
                            observed[path] += 1
                            total_successes[path] += success
                    self.assertEqual(actual.loc[time, 'a_chosen_count'],
                                     all_buyers[time].sum())
                    self.assertEqual(actual.loc[time, 'product2_chosen_count'],
                                     all_p2[time].sum())
                np.testing.assert_array_equal(path_summary['final_observation_n'], observed)
                np.testing.assert_array_equal(path_summary['final_successes'], total_successes)
                np.testing.assert_array_equal(path_summary['final_remembered_successes'],
                                              [sum(h) for h in histories])
                np.testing.assert_array_equal(path_summary['product2_count'], all_p2.sum(axis=0))
                for b, row in bins.iterrows():
                    start, end = b * bin_width, min((b + 1) * bin_width, periods)
                    self.assertEqual(row['a_chosen_count'], all_buyers[start:end].sum())
                    self.assertEqual(row['product2_chosen_count'], all_p2[start:end].sum())
                    self.assertAlmostEqual(row['a_market_share'], all_buyers[start:end].mean())
                    if all_buyers[start:end].any():
                        self.assertAlmostEqual(row['product2_given_a_rate'],
                                               all_p2[start:end].sum() / all_buyers[start:end].sum())


if __name__ == '__main__':
    unittest.main()
