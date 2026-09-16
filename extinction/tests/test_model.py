from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np

from extinction.model import (
    POLICY_PRODUCT1,
    POLICY_PRODUCT2,
    POLICY_TIE,
    Parameters,
    action_tolerance,
    classify_solution,
    demand,
    solve_stationary_policy_truncation,
    solve_stationary_truncation,
)


class ExtinctionModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parameters = Parameters()

    def test_demand_is_monotone_symmetric_and_bounded(self) -> None:
        for n in (0, 1, 20, 100):
            values = demand(n, p0=0.5)
            self.assertEqual(values.shape, (n + 1,))
            self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all((0.0 <= values) & (values <= 1.0)))
            self.assertTrue(np.all(np.diff(values) >= -1e-14))
            np.testing.assert_allclose(
                values + values[::-1],
                1.0,
                rtol=0.0,
                atol=3e-13,
            )

    def test_stationary_solution_has_complete_finite_layers(self) -> None:
        report_diagonal = 15
        solution = solve_stationary_truncation(
            self.parameters,
            outer_diagonal=50,
            report_diagonal=report_diagonal,
        )

        self.assertEqual(set(solution["layers"]), set(range(report_diagonal + 1)))
        for n, layer in solution["layers"].items():
            for field in ("value", "gap", "advantage", "demand"):
                values = np.asarray(layer[field])
                self.assertEqual(values.shape, (n + 1,))
                self.assertTrue(np.all(np.isfinite(values)))
            self.assertTrue(np.all((0.0 <= layer["demand"]) & (layer["demand"] <= 1.0)))

        self.assertLessEqual(solution["maximum_bellman_residual"], 1e-10)

    def test_backward_solution_matches_small_grid_value_iteration(self) -> None:
        """Check the closed-form self-loop solve against Bellman iteration."""

        outer_diagonal = 8
        report_diagonal = 7
        solution = solve_stationary_truncation(
            self.parameters,
            outer_diagonal=outer_diagonal,
            report_diagonal=report_diagonal,
        )
        values = {
            n: np.zeros(n + 1, dtype=float)
            for n in range(outer_diagonal + 2)
        }
        for _ in range(10_000):
            maximum_change = 0.0
            updated = {outer_diagonal + 1: values[outer_diagonal + 1]}
            for n in range(outer_diagonal, -1, -1):
                state_demand = demand(n, self.parameters.p0)
                next_value = updated[n + 1]
                q1 = self.parameters.revenue - self.parameters.c1 + (
                    self.parameters.gamma
                    * (
                        self.parameters.p1 * next_value[1:]
                        + (1.0 - self.parameters.p1) * next_value[:-1]
                    )
                )
                q2 = self.parameters.revenue - self.parameters.c2 + (
                    self.parameters.gamma
                    * (
                        self.parameters.p2 * next_value[1:]
                        + (1.0 - self.parameters.p2) * next_value[:-1]
                    )
                )
                current = (
                    self.parameters.gamma * (1.0 - state_demand) * values[n]
                    + state_demand * np.maximum(q1, q2)
                )
                maximum_change = max(
                    maximum_change,
                    float(np.max(np.abs(current - values[n]))),
                )
                updated[n] = current
            values = updated
            if maximum_change < 1e-13:
                break
        else:
            self.fail("small-grid value iteration did not converge")

        for n in range(report_diagonal + 1):
            np.testing.assert_allclose(
                solution["layers"][n]["value"],
                values[n],
                rtol=0.0,
                atol=2e-11,
            )

    def test_gamma_zero_rules_out_product_two(self) -> None:
        parameters = replace(self.parameters, gamma=0.0)
        solution = solve_stationary_truncation(
            parameters,
            outer_diagonal=30,
            report_diagonal=10,
        )
        expected_advantage = -(parameters.c2 - parameters.c1)
        for layer in solution["layers"].values():
            np.testing.assert_allclose(
                layer["advantage"], expected_advantage, rtol=0.0, atol=1e-14
            )

        states, diagonals = classify_solution(solution, action_tolerance)
        self.assertEqual(len(states), sum(n + 1 for n in range(11)))
        self.assertEqual(len(diagonals), 11)
        self.assertTrue((states["classification"] == "product1").all())
        self.assertFalse(diagonals["active"].any())
        self.assertTrue((diagonals["robust_product2_states"] == 0).all())

    def test_action_tolerance_has_an_absolute_floor_and_relative_scale(self) -> None:
        self.assertAlmostEqual(action_tolerance(np.zeros(5)), 1e-8)
        self.assertAlmostEqual(action_tolerance(np.array([-1_000.0, 2.0])), 1e-5)

    def test_diagonal_maxima_and_investment_advantage_are_consistent(self) -> None:
        solution = solve_stationary_truncation(
            self.parameters,
            outer_diagonal=60,
            report_diagonal=30,
        )
        states, diagonals = classify_solution(solution, action_tolerance)

        maximum_g_by_diagonal = states.groupby("n", sort=True).G.max()
        np.testing.assert_allclose(
            diagonals.maximum_G,
            maximum_g_by_diagonal,
            rtol=0.0,
            atol=0.0,
        )
        investment_advantage = states.G - self.parameters.threshold
        np.testing.assert_allclose(
            states.advantage,
            self.parameters.delta_p * investment_advantage,
            rtol=0.0,
            atol=2e-15,
        )
        np.testing.assert_allclose(
            diagonals.extinction_margin,
            maximum_g_by_diagonal - self.parameters.threshold,
            rtol=0.0,
            atol=0.0,
        )

    def test_report_diagonal_must_be_strictly_inside_truncation(self) -> None:
        with self.assertRaises(ValueError):
            solve_stationary_truncation(
                self.parameters,
                outer_diagonal=12,
                report_diagonal=12,
            )

    def test_compact_policy_matches_full_solution_classification(self) -> None:
        full = solve_stationary_truncation(
            self.parameters,
            outer_diagonal=80,
            report_diagonal=30,
        )
        states, expected_diagonals = classify_solution(full, action_tolerance)
        compact = solve_stationary_policy_truncation(
            self.parameters,
            outer_diagonal=80,
            report_diagonal=30,
            tolerance_function=action_tolerance,
        )

        code_by_label = {
            "product1": POLICY_PRODUCT1,
            "tie": POLICY_TIE,
            "product2": POLICY_PRODUCT2,
        }
        for n, expected in states.groupby("n", sort=True):
            expected_codes = expected.classification.map(code_by_label).to_numpy(
                dtype=np.int8
            )
            np.testing.assert_array_equal(compact["actions"][int(n)], expected_codes)

        columns = [
            "active",
            "robust_product2_states",
            "components",
            "ambiguous",
            "robust_interval_violation",
            "lower_S",
            "upper_S",
            "maximum_G",
            "extinction_margin",
            "maximum_advantage",
            "tolerance",
        ]
        for column in columns:
            left = compact["diagonals"][column].to_numpy()
            right = expected_diagonals[column].to_numpy()
            if left.dtype.kind == "f":
                np.testing.assert_allclose(left, right, rtol=0.0, atol=0.0)
            else:
                np.testing.assert_array_equal(left, right)
        self.assertLessEqual(compact["maximum_bellman_residual"], 1e-10)

    def test_baseline_reproduces_the_reported_last_active_diagonal(self) -> None:
        solution = solve_stationary_truncation(
            self.parameters,
            outer_diagonal=800,
            report_diagonal=450,
        )
        _, diagonals = classify_solution(solution, action_tolerance)
        active = diagonals.loc[diagonals.active, "n"]
        self.assertEqual(int(active.max()), 435)
        self.assertFalse(bool(diagonals.loc[diagonals.n == 436, "active"].iloc[0]))
        self.assertLessEqual(solution["maximum_bellman_residual"], 1e-12)


if __name__ == "__main__":
    unittest.main()
