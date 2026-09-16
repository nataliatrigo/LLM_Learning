from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from extinction.model import (
    POLICY_PRODUCT1,
    POLICY_PRODUCT2,
    POLICY_TIE,
    Parameters,
)
from extinction.run_experiments import (
    compare_compact_policies,
    product2_policy_points,
)


class PolicyFigureTests(unittest.TestCase):
    def test_policy_points_use_observation_counts_and_posterior_means(self) -> None:
        solution = {
            "report_diagonal": 2,
            "actions": {
                0: np.array([POLICY_PRODUCT1], dtype=np.int8),
                1: np.array([POLICY_PRODUCT2, POLICY_TIE], dtype=np.int8),
                2: np.array(
                    [POLICY_PRODUCT1, POLICY_PRODUCT2, POLICY_TIE],
                    dtype=np.int8,
                ),
            },
        }

        observations, posterior_means = product2_policy_points(solution)
        np.testing.assert_array_equal(observations, [1, 2])
        np.testing.assert_allclose(
            posterior_means,
            [1.0 / 3.0, 0.5],
            rtol=0.0,
            atol=0.0,
        )

    def test_compact_comparison_distinguishes_resolved_and_censored_tails(
        self,
    ) -> None:
        report_diagonal = 5
        actions = {
            n: np.full(n + 1, POLICY_PRODUCT1, dtype=np.int8)
            for n in range(report_diagonal + 1)
        }
        actions[0][0] = POLICY_PRODUCT2
        actions[1][0] = POLICY_PRODUCT2
        diagonals = pd.DataFrame(
            {
                "n": np.arange(report_diagonal + 1),
                "active": [True, True, False, False, False, False],
                "ambiguous": False,
                "robust_interval_violation": False,
            }
        )

        def solution(outer: int) -> dict:
            return {
                "parameters": Parameters(),
                "outer_diagonal": outer,
                "report_diagonal": report_diagonal,
                "actions": actions,
                "diagonals": diagonals,
                "maximum_bellman_residual": 0.0,
            }

        resolved = compare_compact_policies(
            solution(10),
            solution(12),
            minimum_inactive_tail=4,
        )
        self.assertTrue(bool(resolved["cutoff_resolved"]))
        self.assertEqual(resolved["last_active_diagonal"], 1)
        self.assertEqual(resolved["first_inactive_diagonal_after_last_active"], 2)

        censored = compare_compact_policies(
            solution(10),
            solution(12),
            minimum_inactive_tail=5,
        )
        self.assertFalse(bool(censored["cutoff_resolved"]))


if __name__ == "__main__":
    unittest.main()
