"""Unit tests for the dynamic Monte Carlo panel infrastructure."""

from __future__ import annotations

from pathlib import Path
import math
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp.demand_models import StandardThompsonSampling  # noqa: E402
from modular_dp.discrete_solver import (  # noqa: E402
    DiscreteSolverConfig,
    solve_discrete,
)
from modular_dp.dynamic_panels import (  # noqa: E402
    _dynamic_figure_stems,
    _method_specifications,
    make_common_random_numbers,
    make_dynamic_panel_figures,
    period_monte_carlo_statistics,
    quality_maintenance_mix,
    simulate_discrete_dynamic_panel,
    summarize_dynamic_series,
)
from modular_dp.primitives import SellerPrimitives  # noqa: E402
from modular_dp.simulation import LayerPolicy  # noqa: E402


class PeriodMonteCarloStatisticsTests(unittest.TestCase):
    def test_exact_statistics_distinguish_conditional_calendar_and_prescribed_use(
        self,
    ) -> None:
        statistics = period_monte_carlo_statistics(
            demand_probability=np.array([0.1, 0.2, 0.3, 0.4]),
            chosen_a=np.array([True, False, True, False]),
            action2=np.array([True, True, False, False]),
            profit=np.array([0.35, 0.0, 0.95, 0.0]),
            posterior_mean=np.array([0.2, 0.4, 0.6, 0.8]),
            effective_sample_size=np.array([0.0, 1.0, 2.0, 3.0]),
            effective_concentration=np.array([2.0, 3.0, 4.0, 5.0]),
            confidence_level=0.95,
        )

        # A is selected on two paths. Product 2 is prescribed on two paths,
        # but it is actually used on only one of the selected paths.
        self.assertEqual(statistics["A_selected_count"], 2)
        self.assertEqual(statistics["product2_used_count"], 1)
        self.assertAlmostEqual(statistics["realized_A_share_mean"], 2 / 4)
        self.assertAlmostEqual(
            statistics["product2_share_conditional_on_A_mean"], 1 / 2
        )
        self.assertAlmostEqual(statistics["product2_share_calendar_mean"], 1 / 4)
        self.assertAlmostEqual(statistics["product2_prescribed_share_mean"], 2 / 4)

        # Demand is an ex-ante probability, whereas market share is realized.
        self.assertAlmostEqual(statistics["mean_demand_probability_mean"], 0.25)
        self.assertAlmostEqual(statistics["calendar_profit_mean"], 0.325)
        self.assertAlmostEqual(statistics["posterior_mean_mean"], 0.5)
        self.assertAlmostEqual(statistics["effective_sample_size_mean"], 1.5)
        self.assertAlmostEqual(statistics["effective_concentration_mean"], 3.5)

        self.assertEqual(
            statistics["product2_share_conditional_on_A_valid_count"], 2
        )
        self.assertEqual(statistics["product2_share_calendar_valid_count"], 4)
        self.assertEqual(statistics["product2_prescribed_share_valid_count"], 4)
        for metric in (
            "realized_A_share",
            "product2_share_conditional_on_A",
            "product2_share_calendar",
            "product2_prescribed_share",
        ):
            self.assertLessEqual(
                statistics[f"{metric}_ci_lower"], statistics[f"{metric}_mean"]
            )
            self.assertGreaterEqual(
                statistics[f"{metric}_ci_upper"], statistics[f"{metric}_mean"]
            )
            self.assertGreaterEqual(statistics[f"{metric}_ci_lower"], 0.0)
            self.assertLessEqual(statistics[f"{metric}_ci_upper"], 1.0)

    def test_conditional_product2_use_is_nan_when_no_path_selects_a(self) -> None:
        statistics = period_monte_carlo_statistics(
            demand_probability=np.zeros(3),
            chosen_a=np.zeros(3, dtype=bool),
            action2=np.array([True, False, True]),
            profit=np.zeros(3),
            posterior_mean=np.full(3, 0.5),
            effective_sample_size=np.zeros(3),
            effective_concentration=np.full(3, 2.0),
            confidence_level=0.95,
        )

        prefix = "product2_share_conditional_on_A"
        self.assertEqual(statistics[f"{prefix}_valid_count"], 0)
        for suffix in ("mean", "standard_error", "ci_lower", "ci_upper"):
            self.assertTrue(math.isnan(float(statistics[f"{prefix}_{suffix}"])))
        self.assertEqual(statistics["A_selected_count"], 0)
        self.assertEqual(statistics["product2_used_count"], 0)
        self.assertEqual(statistics["product2_share_calendar_mean"], 0.0)
        self.assertAlmostEqual(statistics["product2_prescribed_share_mean"], 2 / 3)


class DynamicPanelSimulationTests(unittest.TestCase):
    def test_quality_maintenance_mix_distinguishes_feasibility(self) -> None:
        self.assertEqual(quality_maintenance_mix(0.1, 0.35, 0.8), 0.0)
        self.assertAlmostEqual(quality_maintenance_mix(0.5, 0.35, 0.8), 1 / 3)
        self.assertAlmostEqual(quality_maintenance_mix(0.7, 0.35, 0.8), 7 / 9)
        self.assertTrue(math.isnan(quality_maintenance_mix(0.9, 0.35, 0.8)))

    def test_dynamic_figures_group_methods_into_one_plot_per_p0(self) -> None:
        p0_values = [0.1, 0.3, 0.5, 0.7, 0.9]
        methods = [
            ("standard_ts", "Standard Thompson sampling", "standard_ts"),
            ("scaled_eta_0p5", "Scaled updates, eta=0.5", "scaled_updates"),
            ("scaled_eta_2", "Scaled updates, eta=2", "scaled_updates"),
            (
                "temperature_0p5",
                "Mean-preserving temperature, T=0.5",
                "mean_preserving_temperature",
            ),
            (
                "temperature_2",
                "Mean-preserving temperature, T=2",
                "mean_preserving_temperature",
            ),
            ("epsilon_0p05", "Epsilon-greedy, epsilon=0.05", "epsilon_greedy"),
            ("epsilon_0p1", "Epsilon-greedy, epsilon=0.1", "epsilon_greedy"),
            ("epsilon_0p2", "Epsilon-greedy, epsilon=0.2", "epsilon_greedy"),
            (
                "observation_forgetting_0p95",
                "Observation-time forgetting, rho=0.95",
                "observation_forgetting_ts",
            ),
            (
                "calendar_forgetting_0p95",
                "Calendar-time forgetting, rho=0.95",
                "calendar_forgetting_ts",
            ),
        ]
        rows = []
        for method_key, method_label, method in methods:
            for p0 in p0_values:
                rows.append(
                    {
                        "method": method,
                        "method_spec_key": method_key,
                        "method_spec_label": method_label,
                        "series_key": f"{method_key}__p0_{p0:g}",
                        "series_label": f"p0={p0:g}",
                        "p0": p0,
                    }
                )
        frame = pd.DataFrame(rows)

        self.assertEqual(
            _dynamic_figure_stems(frame),
            [
                "13_dynamic_p0_0p1_across_methods",
                "14_dynamic_p0_0p3_across_methods",
                "15_dynamic_p0_0p5_across_methods",
                "16_dynamic_p0_0p7_across_methods",
                "17_dynamic_p0_0p9_across_methods",
            ],
        )

        with patch("modular_dp.dynamic_panels._plot_four_panel") as plot:
            make_dynamic_panel_figures(frame, Path("/tmp/unused"), preliminary=False)

        self.assertEqual(plot.call_count, 10)
        colors_by_method: dict[str, object] = {}
        complete_keys = {
            method_key
            for method_key, _, _ in methods
            if method_key not in {"epsilon_0p05", "epsilon_0p2"}
        }
        for call, p0 in zip(plot.call_args_list[:5], p0_values, strict=True):
            plotted = call.args[0]
            self.assertEqual(set(plotted.p0), {p0})
            self.assertEqual(set(plotted.method_spec_key), complete_keys)
            self.assertEqual(len(call.kwargs["series_order"]), len(complete_keys))
            self.assertIn(f"p0={p0:g}", call.kwargs["title"])
            for method_key in complete_keys:
                row = plotted[plotted.method_spec_key == method_key].iloc[0]
                color = call.kwargs["colors"][row.series_key]
                if method_key in colors_by_method:
                    self.assertEqual(color, colors_by_method[method_key])
                colors_by_method[method_key] = color
        self.assertEqual(
            list(plot.call_args_list[0].args[0].series_label),
            [
                "Standard TS",
                "Scaled updates, η=0.5",
                "Scaled updates, η=2",
                "Temperature, T=0.5",
                "Temperature, T=2",
                "Epsilon-greedy, ε=0.1",
                "Observation forgetting, ρ=0.95",
                "Calendar forgetting, ρ=0.95",
            ],
        )
        focused_keys = {
            "standard_ts",
            "scaled_eta_0p5",
            "scaled_eta_2",
            "epsilon_0p1",
            "observation_forgetting_0p95",
        }
        for call, p0 in zip(plot.call_args_list[5:], p0_values, strict=True):
            plotted = call.args[0]
            self.assertEqual(set(plotted.p0), {p0})
            self.assertEqual(set(plotted.method_spec_key), focused_keys)
            self.assertEqual(len(call.kwargs["series_order"]), len(focused_keys))
            self.assertIn("focused learning rules", call.kwargs["title"])
            self.assertIn(f"p0={p0:g}", call.kwargs["title"])

    def test_common_random_numbers_are_reproducible(self) -> None:
        first = make_common_random_numbers(paths=7, periods=5, seed=1234)
        second = make_common_random_numbers(paths=7, periods=5, seed=1234)
        different = make_common_random_numbers(paths=7, periods=5, seed=1235)

        np.testing.assert_array_equal(first.choice, second.choice)
        np.testing.assert_array_equal(first.outcome, second.outcome)
        self.assertEqual(first.choice.shape, (5, 7))
        self.assertEqual(first.outcome.shape, (5, 7))
        self.assertEqual(first.paths, 7)
        self.assertEqual(first.periods, 5)
        self.assertTrue(np.all((first.choice >= 0.0) & (first.choice < 1.0)))
        self.assertTrue(np.all((first.outcome >= 0.0) & (first.outcome < 1.0)))
        self.assertFalse(np.array_equal(first.choice, different.choice))
        self.assertFalse(np.array_equal(first.outcome, different.outcome))

    def test_discrete_panel_is_reproducible_and_satisfies_accounting(self) -> None:
        primitives = SellerPrimitives(p0=0.3)
        model = StandardThompsonSampling(primitives)
        solution = solve_discrete(
            model,
            primitives,
            DiscreteSolverConfig(outer_diagonal=45, report_diagonal=8),
        )
        random_numbers = make_common_random_numbers(paths=32, periods=8, seed=77)
        kwargs = dict(
            model=model,
            policy=LayerPolicy(solution),
            primitives=primitives,
            random_numbers=random_numbers,
            seed=77,
            confidence_level=0.95,
            panel_group="unit_test",
            method_spec_key="standard_ts",
            method_spec_label="Standard Thompson sampling",
            series_key="standard_ts__p0_0p3",
            series_label="p0=0.30",
            policy_solver="test triangular DP",
        )
        first = simulate_discrete_dynamic_panel(**kwargs)
        second = simulate_discrete_dynamic_panel(**kwargs)

        assert_frame_equal(first, second, check_exact=True)
        np.testing.assert_array_equal(first["period"], np.arange(1, 9))
        self.assertEqual(len(first), 8)
        self.assertTrue((first["p0"] == 0.3).all())
        self.assertTrue((first["paths"] == 32).all())
        self.assertTrue((first["periods"] == 8).all())
        self.assertTrue((first["seed"] == 77).all())
        self.assertTrue((first["rolling_window"] == 0).all())
        self.assertTrue(first["common_random_numbers"].all())
        self.assertTrue((first["method_spec_key"] == "standard_ts").all())

        np.testing.assert_allclose(
            first["realized_A_share_mean"], first["A_selected_count"] / 32
        )
        np.testing.assert_allclose(
            first["product2_share_calendar_mean"],
            first["product2_used_count"] / 32,
        )
        selected = first["A_selected_count"].to_numpy()
        conditional = first["product2_share_conditional_on_A_mean"].to_numpy()
        expected_conditional = np.divide(
            first["product2_used_count"].to_numpy(),
            selected,
            out=np.full(len(first), np.nan),
            where=selected > 0,
        )
        np.testing.assert_allclose(conditional, expected_conditional, equal_nan=True)
        self.assertTrue(
            (
                first["product2_share_calendar_mean"]
                <= first["realized_A_share_mean"] + 1e-15
            ).all()
        )
        self.assertTrue(
            (
                first["product2_share_calendar_mean"]
                <= first["product2_prescribed_share_mean"] + 1e-15
            ).all()
        )

        for metric in (
            "realized_A_share",
            "mean_demand_probability",
            "product2_share_calendar",
            "product2_prescribed_share",
            "posterior_mean",
        ):
            self.assertTrue(first[f"{metric}_mean"].between(0.0, 1.0).all())
        self.assertTrue(
            first["calendar_profit_mean"].between(
                0.0, max(primitives.net_rewards)
            ).all()
        )

        summary = summarize_dynamic_series(first, late_window_start=6)
        self.assertEqual(len(summary), 1)
        row = summary.iloc[0]
        self.assertEqual(row["late_window_start"], 6)
        self.assertAlmostEqual(
            row["overall_product2_share_calendar"],
            first["product2_share_calendar_mean"].mean(),
        )
        self.assertAlmostEqual(
            row["late_product2_share_calendar"],
            first.loc[first.period >= 6, "product2_share_calendar_mean"].mean(),
        )
        self.assertAlmostEqual(
            row["overall_product2_share_conditional_on_A"],
            first["product2_used_count"].sum() / first["A_selected_count"].sum(),
        )
        late = first[first.period >= 6]
        self.assertAlmostEqual(
            row["late_product2_share_conditional_on_A"],
            late["product2_used_count"].sum() / late["A_selected_count"].sum(),
        )
        expected_mix = max(
            0.0,
            (primitives.p0 - primitives.p1) / (primitives.p2 - primitives.p1),
        )
        self.assertTrue(row["quality_maintenance_mix_feasible"])
        self.assertAlmostEqual(row["quality_maintenance_mix"], expected_mix)
        self.assertAlmostEqual(
            row["late_product2_gap_to_quality_maintenance_mix"],
            row["late_product2_share_conditional_on_A"] - expected_mix,
        )

    def test_representative_catalog_has_eight_unique_specifications(self) -> None:
        specifications = _method_specifications(
            {
                "representative_methods": {
                    "scaled_eta": [0.5, 2.0],
                    "temperature": [0.5, 2.0],
                    "epsilon": [0.1],
                    "rho": [0.95],
                }
            }
        )

        self.assertEqual(len(specifications), 8)
        keys = [specification.key for specification in specifications]
        self.assertEqual(len(set(keys)), 8)
        self.assertEqual(
            sum(
                specification.solver_kind == "discrete"
                for specification in specifications
            ),
            6,
        )
        self.assertEqual(
            sum(
                specification.solver_kind == "continuous"
                for specification in specifications
            ),
            2,
        )


if __name__ == "__main__":
    unittest.main()
