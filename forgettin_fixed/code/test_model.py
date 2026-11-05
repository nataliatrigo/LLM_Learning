"""Independent calendar-time checks of the count-state replacement model."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import betaincc

from forgettin_fixed.code.model import (
    CountSolution, Primitives, evidence_matched_n, retained_transition, solve_counts,
)


def dense_calendar_policy_iteration(p, n):
    """Independent state graph with outside-option self transitions explicit."""
    states = [(m, s) for m in range(n + 1) for s in range(m + 1)]
    lookup = {state: index for index, state in enumerate(states)}
    demand = np.array([betaincc(1 + s, 1 + m - s, p.p0) for m, s in states])
    next_states = []
    for m, s in states:
        if m < n:
            next_states.append((lookup[m + 1, s], lookup[m + 1, s + 1]))
        else:
            next_states.append((lookup[n, max(0, s - 1)], lookup[n, min(n, s + 1)]))
    actions = np.zeros(len(states), dtype=np.int8)
    for iteration in range(100):
        matrix = np.diag(1 - demand)
        reward = demand * np.where(actions, p.net_rewards[1], p.net_rewards[0])
        for index, (failure, success) in enumerate(next_states):
            probability = p.p2 if actions[index] else p.p1
            matrix[index, failure] += demand[index] * (1 - probability)
            matrix[index, success] += demand[index] * probability
        value = np.linalg.solve(np.eye(len(states)) - p.gamma * matrix, reward)
        gap = np.array([-p.delta_c + p.gamma * p.delta_p * (value[success] - value[failure])
                        for failure, success in next_states])
        next_actions = gap > 0
        if np.array_equal(actions, next_actions):
            return states, value, gap
        actions = next_actions
    raise AssertionError("independent policy iteration failed")


class CountModelTests(unittest.TestCase):
    def test_matches_dense_calendar_policy_iteration(self):
        for n in range(1, 6):
            for p0 in (.1, .5, .9):
                with self.subTest(n=n, p0=p0):
                    p = replace(Primitives(), p0=p0)
                    solution = solve_counts(p, n, tolerance=1e-12)
                    states, value, gap = dense_calendar_policy_iteration(p, n)
                    got = np.array([solution.values[m][s] for m, s in states])
                    gaps = np.array([solution.advantages[m][s] for m, s in states])
                    np.testing.assert_allclose(got, value, atol=1e-9, rtol=0)
                    np.testing.assert_allclose(gaps, gap, atol=1e-9, rtol=0)
                    self.assertLess(np.max(np.abs(got - value)),
                                    solution.diagnostics["value_error_bound"] + 1e-11)

    def test_startup_growth_and_reflecting_boundaries(self):
        m, s = retained_transition([0, 1, 2, 3, 3, 3, 3],
                                   [0, 1, 1, 0, 0, 3, 3],
                                   [1, 0, 1, 0, 1, 0, 1], 3)
        np.testing.assert_array_equal(m, [1, 2, 3, 3, 3, 3, 3])
        np.testing.assert_array_equal(s, [1, 1, 2, 0, 1, 2, 3])
        with self.assertRaises(ValueError):
            retained_transition(2, 3, 1, 3)

    def test_exact_known_mature_bands(self):
        for n, expected in ((20, [(4, 14)]), (100, [(39, 58)]), (1000, [(484, 512)])):
            with self.subTest(n=n):
                solution = solve_counts(Primitives(), n)
                self.assertEqual(solution.product2_intervals(), expected)
                self.assertEqual(solution.diagnostics["mature_states"], n + 1)
                self.assertTrue(solution.diagnostics["all_actions_certified"])
                self.assertEqual(len(solution.values[n]), n + 1)

    def test_count_is_complete_state_and_initial_prior(self):
        solution = solve_counts(Primitives(), 20)
        # No history parameter exists: both arrangements with 12 successes
        # are the same state, with product 2 and identical Beta(13,9) demand.
        action, demand = solution.evaluate([20, 20], [12, 12])
        np.testing.assert_array_equal(action, [True, True])
        self.assertEqual(demand[0], demand[1])
        self.assertEqual(solution.demands[0][0], .5)
        self.assertEqual(solution.evaluate(0, 0)[0].shape, ())
        for m in range(20):
            s = np.arange(m + 1)
            p = solution.primitives
            vf, vs = solution.values[m + 1][s], solution.values[m + 1][s + 1]
            d = solution.demands[m]
            q1 = .95 + .98 * (.35 * vs + .65 * vf)
            q2 = .35 + .98 * (.8 * vs + .2 * vf)
            np.testing.assert_allclose(solution.values[m], d / (1 - .98 * (1 - d)) * np.maximum(q1, q2), atol=1e-12)

    def test_save_load_and_validation(self):
        original = solve_counts(Primitives(), 20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.npz"
            self.assertEqual(original.save(path), path)
            loaded = CountSolution.load(path)
            self.assertEqual(loaded.diagnostics, original.diagnostics)
            self.assertEqual(loaded.method, "exact_opposite_replacement")
            np.testing.assert_array_equal(loaded.evaluate([0, 20, 25], [0, 12, 20])[0], [True, True, False])
            for m in range(21):
                np.testing.assert_array_equal(loaded.values[m], original.values[m])
        for args in ((1, 2), (-1, 0), (20, -1), (1.5, 0), (2, .5)):
            with self.assertRaises(ValueError):
                original.evaluate(*args)
        for n in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                solve_counts(Primitives(), n)

    def test_equivalence_and_fixed_product_benchmark(self):
        self.assertEqual(evidence_matched_n(.95), 20)
        self.assertEqual(evidence_matched_n(.99), 100)
        self.assertEqual(evidence_matched_n(.999), 1000)
        self.assertEqual(evidence_matched_n(.6), 3)
        optimal = solve_counts(Primitives(), 5)
        fixed = solve_counts(Primitives(), 5, optimize=False)
        self.assertGreaterEqual(optimal.values[0][0], fixed.values[0][0])
        self.assertFalse(fixed.evaluate(np.full(6, 5), np.arange(6))[0].any())


if __name__ == "__main__":
    unittest.main()
