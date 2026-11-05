"""Independent full-slot checks of exact aggregation under random replacement."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.special import betaincc

from forgetting_random_replacement.code.model import (
    CountSolution, Primitives, conditional_successors, evidence_matched_n,
    retained_transition, solve_counts,
)


def full_slot_calendar_policy_iteration(p, n):
    """Resolve every binary arrangement, with explicit random old-slot choice.

    This independent graph does not use count transition formulas, the waiting
    prefactor, or the solver's Bellman update.  The outside option is explicit.
    """
    states = [(m, word) for m in range(n + 1) for word in range(1 << m)]
    lookup = {state: index for index, state in enumerate(states)}
    demand = np.array([betaincc(1 + h.bit_count(), 1 + m - h.bit_count(), p.p0)
                       for m, h in states])
    transitions = np.zeros((2, len(states), len(states)))
    for row, (m, word) in enumerate(states):
        for outcome in (0, 1):
            if m < n:
                transitions[outcome, row, lookup[m + 1, (word << 1) | outcome]] = 1
            else:
                for slot in range(n):
                    changed = (word & ~(1 << slot)) | (outcome << slot)
                    transitions[outcome, row, lookup[m, changed]] += 1 / n
    action = np.zeros(len(states), dtype=bool)
    for _ in range(100):
        probability = np.where(action, p.p2, p.p1)
        matrix = np.diag(1 - demand) + demand[:, None] * (
            probability[:, None] * transitions[1] + (1 - probability[:, None]) * transitions[0])
        reward = demand * np.where(action, p.net_rewards[1], p.net_rewards[0])
        value = np.linalg.solve(np.eye(len(states)) - p.gamma * matrix, reward)
        gap = -p.delta_c + p.gamma * p.delta_p * ((transitions[1] - transitions[0]) @ value)
        updated_action = gap > 0
        if np.array_equal(updated_action, action):
            return states, value, gap
        action = updated_action
    raise AssertionError("independent full-slot policy iteration failed")


class RandomReplacementModelTests(unittest.TestCase):
    def test_matches_independent_full_slot_calendar_model(self):
        for n in range(1, 5):
            for p0 in (.1, .5, .9):
                with self.subTest(n=n, p0=p0):
                    p = replace(Primitives(), p0=p0)
                    solution = solve_counts(p, n, tolerance=1e-12)
                    states, value, gap = full_slot_calendar_policy_iteration(p, n)
                    got = np.array([solution.values[m][word.bit_count()] for m, word in states])
                    gaps = np.array([solution.advantages[m][word.bit_count()] for m, word in states])
                    np.testing.assert_allclose(got, value, atol=1e-9, rtol=0)
                    np.testing.assert_allclose(gaps, gap, atol=1e-9, rtol=0)

    def test_full_slot_transition_lumps_exactly_by_counts(self):
        for n in range(1, 6):
            for word in range(1 << n):
                s = word.bit_count()
                for probability in (.35, .8):
                    counts = np.zeros(n + 1)
                    for outcome, weight in ((0, 1 - probability), (1, probability)):
                        for slot in range(n):
                            changed = (word & ~(1 << slot)) | (outcome << slot)
                            counts[changed.bit_count()] += weight / n
                    expected = np.zeros(n + 1)
                    expected[min(s + 1, n)] += probability * (1 - s / n)
                    expected[max(s - 1, 0)] += (1 - probability) * s / n
                    expected[s] += probability * s / n + (1 - probability) * (1 - s / n)
                    np.testing.assert_allclose(counts, expected, atol=1e-14)
                    self.assertAlmostEqual(counts.sum(), 1)

    def test_example_18_successes_new_failure_and_growth(self):
        uniform = np.array([0, .4, .899999, .9, .999999])
        m, s = retained_transition(20, 18, 0, 20, uniform)
        np.testing.assert_array_equal(m, np.full(5, 20))
        np.testing.assert_array_equal(s, [17, 17, 17, 18, 18])
        m, s = retained_transition([0, 1, 2, 3, 3, 3, 3],
                                   [0, 1, 1, 0, 0, 3, 3],
                                   [1, 0, 1, 0, 1, 0, 1], 3, .5)
        np.testing.assert_array_equal(m, [1, 2, 3, 3, 3, 3, 3])
        np.testing.assert_array_equal(s, [1, 1, 2, 0, 1, 2, 3])
        # At S/n=.5, a success/failure can both be replaced by the same type.
        np.testing.assert_array_equal(retained_transition(20, 10, 1, 20, [0, .5])[1], [10, 11])
        np.testing.assert_array_equal(retained_transition(20, 10, 0, 20, [0, .5])[1], [9, 10])

    def test_conditional_continuation_before_removal_is_observed(self):
        value = np.array([0., 2., 5., 9.])
        vf, vs = conditional_successors(value, 3)
        for s in range(4):
            self.assertAlmostEqual(vf[s], (s / 3) * value[max(s - 1, 0)] + (1 - s / 3) * value[s])
            self.assertAlmostEqual(vs[s], (1 - s / 3) * value[min(s + 1, 3)] + (s / 3) * value[s])
        p = replace(Primitives(), gamma=0)
        solution = solve_counts(p, 20)
        np.testing.assert_allclose(solution.values[-1], .95 * solution.demands[-1])
        self.assertEqual(solution.product2_intervals(), [])

    def test_initial_prior_startup_and_fixed_product(self):
        solution = solve_counts(Primitives(), 20)
        self.assertEqual(solution.demands[0][0], .5)
        for m in range(20):
            p = solution.primitives
            vf, vs = solution.values[m + 1][:-1], solution.values[m + 1][1:]
            q1 = p.net_rewards[0] + p.gamma * (p.p1 * vs + (1 - p.p1) * vf)
            q2 = p.net_rewards[1] + p.gamma * (p.p2 * vs + (1 - p.p2) * vf)
            d = solution.demands[m]
            np.testing.assert_allclose(solution.values[m], d / (1 - p.gamma * (1 - d)) * np.maximum(q1, q2))
        fixed = solve_counts(Primitives(), 20, optimize=False)
        self.assertGreaterEqual(solution.values[0][0], fixed.values[0][0])
        self.assertFalse(fixed.evaluate(np.full(21, 20), np.arange(21))[0].any())

    def test_serialization_capacity_and_input_checks(self):
        solution = solve_counts(Primitives(), 20)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.npz"
            self.assertEqual(solution.save(path), path)
            loaded = CountSolution.load(path)
            self.assertEqual(loaded.diagnostics, solution.diagnostics)
            self.assertEqual(loaded.method, "exact_random_replacement")
            for m in range(21):
                np.testing.assert_array_equal(loaded.values[m], solution.values[m])
                np.testing.assert_array_equal(loaded.evaluate(m, np.arange(m + 1))[0],
                                              solution.evaluate(m, np.arange(m + 1))[0])
        self.assertEqual(evidence_matched_n(.95), 20)
        self.assertEqual(evidence_matched_n(.99), 100)
        self.assertEqual(evidence_matched_n(.999), 1000)
        for args in ((20, 21, 0, 20, .5), (20, 18, 0, 20, 1.),
                     (20, 18, 0, 20, -.1), (20, 18.5, 0, 20, .5)):
            with self.assertRaises(ValueError):
                retained_transition(*args)
        for n in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                solve_counts(Primitives(), n)


if __name__ == "__main__":
    unittest.main()
