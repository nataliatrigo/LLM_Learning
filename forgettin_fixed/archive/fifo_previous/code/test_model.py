"""Independent finite-state calendar-time checks of the FIFO solver."""

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.special import betaincc

from forgettin_fixed.code.model import (
    Primitives, WindowSolution, evidence_matched_n, solve_window, success_counts,
)


def independent_policy_iteration(primitives, n):
    """Build the original calendar-time chain, without eliminating waiting."""
    states = [(m, h) for m in range(n + 1) for h in range(1 << m)]
    indices = {state: i for i, state in enumerate(states)}
    size = len(states)
    actions = np.zeros(size, dtype=int)
    demands = np.array([betaincc(1 + h.bit_count(), 1 + m - h.bit_count(),
                                primitives.p0) for m, h in states])
    successors = []
    for m, h in states:
        next_m = min(n, m + 1)
        base = (h << 1) & ((1 << next_m) - 1)
        successors.append((indices[next_m, base], indices[next_m, base | 1]))
    for _ in range(100):
        transition = np.diag(1 - demands)
        reward = demands * np.where(actions, primitives.net_rewards[1],
                                    primitives.net_rewards[0])
        for i, (failure, success) in enumerate(successors):
            prob = primitives.p2 if actions[i] else primitives.p1
            transition[i, failure] += demands[i] * (1 - prob)
            transition[i, success] += demands[i] * prob
        value = np.linalg.solve(np.eye(size) - primitives.gamma * transition, reward)
        gaps = np.array([-primitives.delta_c + primitives.gamma * primitives.delta_p *
                         (value[success] - value[failure])
                         for failure, success in successors])
        improved = (gaps > 0).astype(int)
        if np.array_equal(actions, improved):
            return states, value, gaps
        actions = improved
    raise AssertionError("independent policy iteration did not converge")


class WindowModelTests(unittest.TestCase):
    def test_calendar_policy_iteration(self):
        for n in range(1, 5):
            for p0 in (.1, .5, .9):
                with self.subTest(n=n, p0=p0):
                    p = replace(Primitives(), p0=p0)
                    solution = solve_window(p, n, tolerance=1e-12)
                    states, exact, gaps = independent_policy_iteration(p, n)
                    approximate = np.array([solution.values[m][h] for m, h in states])
                    actual_gaps = np.array([solution.advantages[m][h] for m, h in states])
                    np.testing.assert_allclose(approximate, exact, atol=1e-9, rtol=0)
                    np.testing.assert_allclose(actual_gaps, gaps, atol=1e-9, rtol=0)
                    self.assertLessEqual(np.max(np.abs(approximate - exact)),
                                         solution.diagnostics["value_error_bound"] + 1e-11)

    def test_fifo_is_not_count_markov(self):
        solution = solve_window(Primitives(), 3)
        # 001 and 100 both contain one success. Appending failure removes
        # different oldest outcomes and hence yields different posteriors.
        self.assertEqual(0b001.bit_count(), 0b100.bit_count())
        self.assertNotEqual(((0b001 << 1) & 7).bit_count(),
                            ((0b100 << 1) & 7).bit_count())
        self.assertNotEqual(solution.advantages[3][0b001],
                            solution.advantages[3][0b100])
        # A dropped oldest bit does not change the seller's continuation gap.
        self.assertEqual(solution.advantages[3][0b001],
                         solution.advantages[3][0b101])

    def test_startup_empty_prior(self):
        p = Primitives()
        solution = solve_window(p, 3)
        self.assertEqual(solution.demands[0][0], 1 - p.p0)
        for m in range(3):
            for h in range(1 << m):
                q = [p.net_rewards[i] + p.gamma * (
                    probability * solution.values[m + 1][2 * h + 1]
                    + (1 - probability) * solution.values[m + 1][2 * h])
                    for i, probability in enumerate((p.p1, p.p2))]
                d = solution.demands[m][h]
                self.assertAlmostEqual(solution.values[m][h],
                                       d / (1 - p.gamma * (1 - d)) * max(q))

    def test_save_load_evaluate(self):
        solution = solve_window(Primitives(), 4)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "policy.npz"
            solution.save(path)
            restored = WindowSolution.load(path)
            self.assertEqual(restored.diagnostics, solution.diagnostics)
            actions, demand = restored.evaluate([0, 1, 4, 9], [0, 1, 5, 9])
            for i, (m, h) in enumerate([(0, 0), (1, 1), (4, 5), (4, 9)]):
                self.assertEqual(actions[i], solution.advantages[m][h] > 1e-8)
                self.assertEqual(demand[i], solution.demands[m][h])
            self.assertEqual(restored.evaluate(0, 0)[0].shape, ())
            with self.assertRaises(ValueError):
                restored.evaluate(1, 2)

    def test_success_counts_and_equivalence(self):
        words = np.array([0, 1, 7, 13, 2**63, 2**64 - 1], dtype=np.uint64)
        np.testing.assert_array_equal(success_counts(words), [0, 1, 3, 3, 1, 64])
        self.assertEqual(evidence_matched_n(.95), 20)
        self.assertEqual(evidence_matched_n(.99), 100)
        self.assertEqual(evidence_matched_n(.6), 3)

    def test_fixed_product_one(self):
        optimal = solve_window(Primitives(), 4)
        fixed = solve_window(Primitives(), 4, optimize=False)
        self.assertGreaterEqual(optimal.values[0][0], fixed.values[0][0])
        self.assertFalse(fixed.evaluate(np.arange(5), np.zeros(5, dtype=int))[0].any())


if __name__ == "__main__":
    unittest.main()
