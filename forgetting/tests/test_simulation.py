from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forgetting.layers import solve_observation_layers
from forgetting.model import Primitives, boundary_demand
from forgetting.run_experiments import (
    plot_monte_carlo_p0_panels,
)
from forgetting.simulation import _policy_and_demand, simulate_calendar_paths


class ForgettingSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.primitives = Primitives()
        self.fast_options = {
            "policy_nodes": 201,
            "terminal_gap": 0.1,
            "solver_tolerance": 1e-9,
        }

    def simulate(self, **overrides):
        options = {
            "n_paths": 96,
            "seed": 20260809,
            "max_periods": 12,
            "bin_width": 5,
            **self.fast_options,
            **overrides,
        }
        return simulate_calendar_paths(self.primitives, 0.8, **options)

    def test_simulation_is_reproducible(self) -> None:
        first = self.simulate()
        second = self.simulate()
        for first_frame, second_frame in zip(first, second, strict=True):
            pd.testing.assert_frame_equal(first_frame, second_frame)

    def test_calendar_accounting_and_discounted_state_are_consistent(self) -> None:
        periods, bins, paths = self.simulate(n_paths=128, seed=31415)
        np.testing.assert_array_equal(periods.period, np.arange(12))
        self.assertEqual(len(bins), 3)
        self.assertEqual(len(paths), 128)

        np.testing.assert_array_equal(paths.final_observation_n, paths.a_chosen_count)
        np.testing.assert_array_equal(
            paths.final_successes + paths.final_failures,
            paths.final_observation_n,
        )
        self.assertTrue((paths.product2_count <= paths.a_chosen_count).all())
        np.testing.assert_allclose(
            paths.final_normalized_success_mass
            + paths.final_normalized_failure_mass,
            1.0 - np.power(0.8, paths.final_observation_n),
            rtol=0.0,
            atol=2e-15,
        )
        self.assertTrue((paths.final_normalized_success_mass >= 0.0).all())
        self.assertTrue((paths.final_normalized_failure_mass >= -2e-15).all())

        self.assertEqual(periods.loc[0, "mean_observation_n"], 0.0)
        self.assertTrue((periods.mean_observation_n <= periods.period).all())
        self.assertEqual(
            int(periods.a_chosen_count.sum()), int(paths.a_chosen_count.sum())
        )
        self.assertEqual(
            int(periods.product2_chosen_count.sum()),
            int(paths.product2_count.sum()),
        )

    def test_reported_rates_use_their_actual_denominators(self) -> None:
        n_paths = 128
        periods, bins, _ = self.simulate(n_paths=n_paths, seed=2718, bin_width=4)
        for frame in (periods, bins):
            nonempty = frame.a_chosen_count > 0
            np.testing.assert_allclose(
                frame.loc[nonempty, "product2_given_a_rate"],
                frame.loc[nonempty, "product2_chosen_count"]
                / frame.loc[nonempty, "a_chosen_count"],
                rtol=0.0,
                atol=0.0,
            )
            denominator = (
                n_paths
                if "period" in frame.columns
                else n_paths * frame.periods_in_bin
            )
            np.testing.assert_allclose(
                frame.product2_unconditional_rate,
                frame.product2_chosen_count / denominator,
                rtol=0.0,
                atol=0.0,
            )
            self.assertTrue(frame.a_market_share.between(0.0, 1.0).all())
            self.assertTrue(
                frame.product2_unconditional_rate.between(0.0, 1.0).all()
            )

    def test_gamma_zero_never_uses_product_two(self) -> None:
        periods, bins, paths = simulate_calendar_paths(
            replace(self.primitives, gamma=0.0),
            0.8,
            n_paths=64,
            seed=7,
            max_periods=10,
            bin_width=3,
            **self.fast_options,
        )
        np.testing.assert_array_equal(periods.product2_chosen_count, 0)
        np.testing.assert_array_equal(bins.product2_chosen_count, 0)
        np.testing.assert_array_equal(paths.product2_count, 0)
        self.assertTrue(paths.last_product2_period_within_window.isna().all())

    def test_outside_option_leaves_discounted_evidence_unchanged(self) -> None:
        class FakeGenerator:
            def __init__(self) -> None:
                self.draws = iter(
                    (
                        np.array([0.0, 0.999]),
                        np.array([0.0, 0.999]),
                        np.array([0.999, 0.999]),
                        np.array([0.0, 0.0]),
                    )
                )

            def random(self, size: int) -> np.ndarray:
                draw = next(self.draws)
                self_test.assertEqual(len(draw), size)
                return draw

        self_test = self
        with patch(
            "forgetting.simulation.np.random.default_rng",
            return_value=FakeGenerator(),
        ):
            _, _, paths = simulate_calendar_paths(
                replace(self.primitives, gamma=0.0),
                0.8,
                n_paths=2,
                seed=1,
                max_periods=2,
                bin_width=1,
                **self.fast_options,
            )
        np.testing.assert_array_equal(paths.final_observation_n, [1, 0])
        np.testing.assert_array_equal(paths.final_successes, [1, 0])
        np.testing.assert_allclose(
            paths.final_normalized_success_mass,
            [0.2, 0.0],
            atol=2e-16,
        )

    def test_policy_tables_are_opt_in_and_cover_the_mature_boundary(self) -> None:
        _, summary_only = solve_observation_layers(
            self.primitives, 0.8, nodes=201, terminal_gap=0.1, tolerance=1e-9
        )
        self.assertNotIn("advantage_table", summary_only)
        _, policy = solve_observation_layers(
            self.primitives,
            0.8,
            nodes=201,
            terminal_gap=0.1,
            tolerance=1e-9,
            return_policy=True,
        )
        terminal = int(policy["terminal_observation"])
        self.assertEqual(
            np.asarray(policy["advantage_table"]).shape,
            (terminal + 1, 201),
        )
        self.assertEqual(
            np.asarray(policy["demand_table"]).shape,
            (terminal + 1, 201),
        )
        np.testing.assert_allclose(
            np.asarray(policy["demand_table"])[0],
            1.0 - self.primitives.p0,
            atol=2e-15,
        )
        np.testing.assert_allclose(
            np.asarray(policy["demand_table"])[-1],
            boundary_demand(
                np.asarray(policy["coordinate"]), 0.8, self.primitives.p0
            ),
            atol=2e-15,
        )
        self.assertLessEqual(0.8**terminal, 0.1)
        self.assertGreater(0.8 ** (terminal - 1), 0.1)

    def test_policy_interpolation_and_terminal_clamping_are_exact(self) -> None:
        policy: dict[str, object] = {
            "coordinate": np.array([0.0, 0.5, 1.0]),
            "terminal_observation": 2,
            "advantage_table": np.array(
                [
                    [-1.0, -1.0, -1.0],
                    [0.0, 2.0, 4.0],
                    [0.0, 1.0, 2.0],
                ]
            ),
            "demand_table": np.array(
                [
                    [0.2, 0.2, 0.2],
                    [0.0, 0.5, 1.0],
                    [1.0, 0.5, 0.0],
                ]
            ),
        }
        observations = np.array([0, 1, 1, 2, 3], dtype=np.int32)
        success_mass = np.array([0.0, 0.125, 0.0625, 0.5625, 0.0875])
        product2, demand = _policy_and_demand(
            observations,
            success_mass,
            rho=0.5,
            policy=policy,
            action_tolerance=0.5,
        )
        np.testing.assert_array_equal(product2, [False, True, False, True, False])
        np.testing.assert_allclose(demand, [0.2, 0.25, 0.125, 0.25, 0.9])

    def test_two_panel_plot_writes_png_and_pdf(self) -> None:
        periods, _, _ = self.simulate(n_paths=32, max_periods=4, bin_width=2)
        periods.insert(0, "p0", self.primitives.p0)
        with TemporaryDirectory() as temporary_directory:
            figure_dir = Path(temporary_directory)
            with patch("forgetting.run_experiments.FIGURES", figure_dir):
                plot_monte_carlo_p0_panels(periods)
            self.assertTrue(
                (figure_dir / "monte_carlo_forgetting_p0_panels.png").is_file()
            )
            self.assertTrue(
                (figure_dir / "monte_carlo_forgetting_p0_panels.pdf").is_file()
            )


    def test_invalid_simulation_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.simulate(n_paths=0)
        with self.assertRaises(ValueError):
            self.simulate(confidence_level=1.0)
        with self.assertRaises(ValueError):
            self.simulate(action_tolerance=np.nan)


if __name__ == "__main__":
    unittest.main()
