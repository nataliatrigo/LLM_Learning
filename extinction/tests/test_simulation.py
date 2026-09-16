from __future__ import annotations

from dataclasses import replace
import unittest

import numpy as np
import pandas as pd

from extinction.model import (
    Parameters,
    solve_stationary_policy_truncation,
    solve_stationary_truncation,
)
from extinction.simulation import (
    simulate_calendar_paths,
)


class ExtinctionSimulationTests(unittest.TestCase):
    def _solution(
        self,
        *,
        gamma: float = 0.98,
        report_diagonal: int = 16,
    ) -> dict:
        return solve_stationary_truncation(
            replace(Parameters(), gamma=gamma),
            outer_diagonal=60,
            report_diagonal=report_diagonal,
        )





    def test_calendar_simulation_is_reproducible_and_accounts_for_state(self) -> None:
        solution = self._solution()
        kwargs = {
            "n_paths": 96,
            "seed": 31415,
            "max_periods": 12,
            "bin_width": 5,
        }
        first = simulate_calendar_paths(solution, **kwargs)
        second = simulate_calendar_paths(solution, **kwargs)
        for first_frame, second_frame in zip(first, second, strict=True):
            pd.testing.assert_frame_equal(first_frame, second_frame)

        period_summary, binned_summary, paths = first
        np.testing.assert_array_equal(period_summary.period, np.arange(12))
        self.assertEqual(len(binned_summary), 3)
        self.assertEqual(len(paths), 96)

        # Every Seller-A selection produces exactly one observation; choosing
        # the outside option leaves the count state unchanged.
        np.testing.assert_array_equal(
            paths.final_observation_n,
            paths.a_chosen_count,
        )
        np.testing.assert_array_equal(
            paths.final_successes + paths.final_failures,
            paths.final_observation_n,
        )
        self.assertTrue((paths.product2_count <= paths.a_chosen_count).all())
        self.assertTrue(paths.final_observation_n.between(0, 12).all())

        self.assertEqual(period_summary.loc[0, "mean_observation_n"], 0.0)
        self.assertEqual(period_summary.loc[0, "observation_n_q10"], 0.0)
        self.assertEqual(period_summary.loc[0, "observation_n_q50"], 0.0)
        self.assertEqual(period_summary.loc[0, "observation_n_q90"], 0.0)
        self.assertTrue(
            (
                period_summary.mean_observation_n
                <= period_summary.period
            ).all()
        )
        self.assertEqual(
            int(period_summary.a_chosen_count.sum()),
            int(paths.a_chosen_count.sum()),
        )
        self.assertEqual(
            int(period_summary.product2_chosen_count.sum()),
            int(paths.product2_count.sum()),
        )

    def test_calendar_conditional_rates_use_the_actual_a_denominator(self) -> None:
        n_paths = 128
        period_summary, binned_summary, paths = simulate_calendar_paths(
            self._solution(),
            n_paths=n_paths,
            seed=2718,
            max_periods=12,
            bin_width=4,
        )

        for frame in (period_summary, binned_summary):
            nonempty = frame.a_chosen_count > 0
            expected = (
                frame.loc[nonempty, "product2_chosen_count"]
                / frame.loc[nonempty, "a_chosen_count"]
            )
            np.testing.assert_allclose(
                frame.loc[nonempty, "product2_given_a_rate"],
                expected,
                rtol=0.0,
                atol=0.0,
            )
            self.assertTrue(
                frame.loc[nonempty, "product2_given_a_rate"].between(0.0, 1.0).all()
            )
            self.assertTrue(
                (
                    frame.loc[nonempty, "product2_given_a_ci_lower"]
                    <= frame.loc[nonempty, "product2_given_a_rate"]
                ).all()
            )
            self.assertTrue(
                (
                    frame.loc[nonempty, "product2_given_a_rate"]
                    <= frame.loc[nonempty, "product2_given_a_ci_upper"]
                ).all()
            )
            self.assertTrue(frame.a_market_share.between(0.0, 1.0).all())
            self.assertTrue(
                frame.product2_unconditional_rate.between(0.0, 1.0).all()
            )
            np.testing.assert_allclose(
                frame.product2_unconditional_rate,
                frame.product2_chosen_count / (
                    n_paths
                    if "period" in frame.columns
                    else n_paths * frame.periods_in_bin
                ),
                rtol=0.0,
                atol=0.0,
            )

        np.testing.assert_array_equal(
            binned_summary.a_chosen_count,
            [
                period_summary.loc[
                    row.bin_start : row.bin_stop_exclusive - 1,
                    "a_chosen_count",
                ].sum()
                for row in binned_summary.itertuples()
            ],
        )
        self.assertEqual(
            int(binned_summary.product2_chosen_count.sum()),
            int(paths.product2_count.sum()),
        )

    def test_calendar_simulation_accepts_the_compact_policy_solution(self) -> None:
        full = self._solution()
        compact = solve_stationary_policy_truncation(
            Parameters(),
            outer_diagonal=60,
            report_diagonal=16,
        )
        kwargs = {
            "n_paths": 64,
            "seed": 1234,
            "max_periods": 12,
            "bin_width": 4,
        }
        full_results = simulate_calendar_paths(full, **kwargs)
        compact_results = simulate_calendar_paths(compact, **kwargs)
        for full_frame, compact_frame in zip(
            full_results,
            compact_results,
            strict=True,
        ):
            pd.testing.assert_frame_equal(full_frame, compact_frame)

    def test_calendar_gamma_zero_never_uses_product_two(self) -> None:
        period_summary, binned_summary, paths = simulate_calendar_paths(
            self._solution(gamma=0.0, report_diagonal=10),
            n_paths=64,
            seed=11,
            max_periods=10,
            bin_width=3,
        )
        np.testing.assert_array_equal(period_summary.product2_chosen_count, 0)
        np.testing.assert_array_equal(binned_summary.product2_chosen_count, 0)
        np.testing.assert_array_equal(paths.product2_count, 0)
        self.assertTrue(paths.last_product2_period_within_window.isna().all())

    def test_calendar_simulation_refuses_to_leave_policy_region(self) -> None:
        with self.assertRaises(ValueError):
            simulate_calendar_paths(
                self._solution(report_diagonal=4),
                n_paths=8,
                seed=1,
                max_periods=8,
                bin_width=2,
            )




if __name__ == "__main__":
    unittest.main()
