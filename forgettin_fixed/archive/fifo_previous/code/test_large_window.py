"""Check FIFO lifting, the global residual certificate and exact limiting case."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
from scipy.special import betaincc

from forgetting.Code.model import Primitives
from forgettin_fixed.code.large_window import LargeWindowPolicy, solve_large_window


def exact_layers(n, p):
    histories = np.arange(1 << n)
    count = np.array([i.bit_count() for i in histories])
    demand = betaincc(1 + count, 1 + n - count, p.p0)
    waiting = demand / (1 - p.gamma * (1 - demand))
    failure = (histories << 1) & ((1 << n) - 1)
    value = np.zeros(1 << n)
    for _ in range(10_000):
        q1 = p.net_rewards[0] + p.gamma * (p.p1 * value[failure + 1] + (1 - p.p1) * value[failure])
        q2 = p.net_rewards[1] + p.gamma * (p.p2 * value[failure + 1] + (1 - p.p2) * value[failure])
        updated = waiting * np.maximum(q1, q2)
        if np.max(np.abs(updated - value)) < 1e-12:
            value = updated
            break
        value = updated
    gaps = [None] * (n + 1)
    gaps[n] = q2 - q1
    for m in range(n - 1, -1, -1):
        vf, vs = value[0::2], value[1::2]
        q1 = p.net_rewards[0] + p.gamma * (p.p1 * vs + (1 - p.p1) * vf)
        q2 = p.net_rewards[1] + p.gamma * (p.p2 * vs + (1 - p.p2) * vf)
        gaps[m] = q2 - q1
        c = np.array([i.bit_count() for i in range(1 << m)])
        d = betaincc(1 + c, 1 + m - c, p.p0)
        value = d / (1 - p.gamma * (1 - d)) * np.maximum(q1, q2)
    return gaps


class LargeWindowTests(unittest.TestCase):
    def setUp(self):
        self.p = Primitives(c2=.6665, gamma=.9)

    def test_one_hidden_bit_recovers_exact_ordered_mdp(self):
        n = 6
        policy = solve_large_window(n, self.p, prefix_bits=n - 1, tolerance=1e-12)
        exact = exact_layers(n, self.p)
        for m in range(n + 1):
            histories = np.arange(1 << m)
            count = np.array([i.bit_count() for i in histories])
            prefix = histories if m < n else histories >> 1
            bit = histories & 1
            _, _, gap = policy.evaluate_state(m, count, prefix, bit)
            np.testing.assert_allclose(gap, exact[m], atol=2e-7, rtol=2e-7)
        self.assertLess(policy.diagnostics["fifo_bellman_residual_sup"], 1e-10)

    def test_global_residual_equals_exhaustive_fifo_residual(self):
        n, k = 7, 3
        policy = solve_large_window(n, self.p, prefix_bits=k, tolerance=1e-12)
        histories = np.arange(1 << n)
        counts = np.array([i.bit_count() for i in histories])

        def lift(h):
            prefix = h >> (n - k)
            c = np.array([int(i).bit_count() for i in h])
            pc = np.array([int(i).bit_count() for i in prefix])
            return policy.mature_value[prefix, c - pc]

        f = (histories << 1) & ((1 << n) - 1)
        vf, vs = lift(f), lift(f + 1)
        q1 = self.p.net_rewards[0] + self.p.gamma * (self.p.p1 * vs + (1 - self.p.p1) * vf)
        q2 = self.p.net_rewards[1] + self.p.gamma * (self.p.p2 * vs + (1 - self.p.p2) * vf)
        d = betaincc(1 + counts, 1 + n - counts, self.p.p0)
        tv = d / (1 - self.p.gamma * (1 - d)) * np.maximum(q1, q2)
        exact_res = np.max(np.abs(tv - lift(histories)))
        self.assertAlmostEqual(exact_res, policy.diagnostics["fifo_bellman_residual_sup"], places=12)
        _, _, gap = policy.evaluate_state(n, counts, histories >> (n - k), (histories >> (n - k - 1)) & 1)
        np.testing.assert_allclose(gap, q2 - q1, atol=1e-12)

    def test_queue_orientation_and_save_load(self):
        policy = solve_large_window(7, self.p, prefix_bits=3)
        queue = np.array([[1, 0, 1, 0, 1, 1, 0], [1, 1, 0, 0, 0, 0, 0]], dtype=np.uint8)
        lengths = np.array([7, 2])
        action, demand = policy.evaluate_queue(queue, lengths)
        direct = policy.evaluate_state(lengths, [4, 2], [5, 3], [0, 0])
        np.testing.assert_array_equal(action, direct[0])
        np.testing.assert_allclose(demand, direct[1])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "policy.npz"
            policy.save(path)
            restored = LargeWindowPolicy.load(path)
            result = restored.evaluate_queue(queue, lengths)
            np.testing.assert_array_equal(result[0], action)
            np.testing.assert_allclose(result[1], demand)


if __name__ == "__main__":
    unittest.main()
