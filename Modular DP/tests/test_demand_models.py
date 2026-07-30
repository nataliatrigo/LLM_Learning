"""Unit tests for the modular belief and demand interface."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.special import betaincc


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp.demand_models import (  # noqa: E402
    CalendarForgettingTS,
    EpsilonGreedy,
    MeanPreservingTemperature,
    ObservationForgettingTS,
    ScaledUpdates,
    StandardThompsonSampling,
)
from modular_dp.primitives import SellerPrimitives  # noqa: E402


class SellerPrimitiveTests(unittest.TestCase):
    def test_baseline_parameters_and_threshold(self) -> None:
        primitives = SellerPrimitives()
        self.assertEqual(primitives.p0, 0.50)
        self.assertEqual(primitives.p1, 0.35)
        self.assertEqual(primitives.p2, 0.80)
        self.assertEqual(primitives.c1, 0.05)
        self.assertEqual(primitives.c2, 0.65)
        self.assertEqual(primitives.revenue, 1.00)
        self.assertEqual(primitives.gamma, 0.98)
        self.assertAlmostEqual(
            primitives.threshold,
            (primitives.c2 - primitives.c1) / (primitives.p2 - primitives.p1),
        )


class DiscreteDemandTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = [(0, 0), (1, 0), (0, 1), (7, 3), (18, 22), (100, 40)]

    def test_standard_ts_reproduces_legacy_demand(self) -> None:
        model = StandardThompsonSampling()
        for successes, failures in self.states:
            legacy = float(betaincc(successes + 1, failures + 1, 0.50))
            self.assertAlmostEqual(model.demand((successes, failures)), legacy, places=14)
        np.testing.assert_allclose(
            model.demand_diagonal(25),
            betaincc(np.arange(26) + 1, 25 - np.arange(26) + 1, 0.50),
            rtol=0.0,
            atol=2e-15,
        )
        self.assertAlmostEqual(StandardThompsonSampling(p0=0.4).p0, 0.4)

    def test_eta_one_and_temperature_one_reproduce_standard_ts(self) -> None:
        standard = StandardThompsonSampling()
        scaled = ScaledUpdates(1.0)
        temperature = MeanPreservingTemperature(1.0)
        for n in (0, 1, 10, 75):
            expected = standard.demand_diagonal(n)
            np.testing.assert_allclose(scaled.demand_diagonal(n), expected, atol=2e-15)
            np.testing.assert_allclose(
                temperature.demand_diagonal(n), expected, atol=2e-15
            )

    def test_epsilon_one_is_constant_half(self) -> None:
        model = EpsilonGreedy(1.0)
        successes, failures = np.meshgrid(np.arange(20), np.arange(20))
        np.testing.assert_array_equal(
            model.demand_counts(successes, failures),
            np.full_like(successes, 0.5, dtype=float),
        )

    def test_epsilon_greedy_uses_uniform_exploration_and_robust_tie(self) -> None:
        model = EpsilonGreedy(0.20, tie_tolerance=1e-10)
        self.assertEqual(model.demand((2, 0)), 0.90)
        self.assertEqual(model.demand((0, 2)), 0.10)
        self.assertEqual(model.demand((5, 5)), 0.50)

        close_threshold = SellerPrimitives(p0=0.5 + 5e-11)
        robust_model = EpsilonGreedy(
            0.20, primitives=close_threshold, tie_tolerance=1e-10
        )
        self.assertEqual(robust_model.demand((5, 5)), 0.50)

    def test_discrete_transitions_and_state_coordinates(self) -> None:
        model = ScaledUpdates(0.5)
        self.assertEqual(model.success_state((3, 4)), (4, 4))
        self.assertEqual(model.failure_state((3, 4)), (3, 5))
        self.assertEqual(model.idle_state((3, 4)), (3, 4))
        self.assertEqual(model.state_coordinates((3, 4)), (3, 4))
        self.assertTrue(model.has_exact_idle_self_loop)

    def test_discrete_belief_statistics_follow_each_sampling_rule(self) -> None:
        scaled = ScaledUpdates(0.25)
        self.assertAlmostEqual(scaled.posterior_mean((1, 0)), 1.25 / 2.25)
        self.assertAlmostEqual(scaled.effective_sample_size((1, 0)), 0.25)
        self.assertAlmostEqual(scaled.effective_concentration((1, 0)), 2.25)

        temperature = MeanPreservingTemperature(4.0)
        self.assertAlmostEqual(temperature.posterior_mean((3, 1)), 2.0 / 3.0)
        self.assertAlmostEqual(temperature.effective_sample_size((3, 1)), 1.0)
        self.assertAlmostEqual(temperature.effective_concentration((3, 1)), 1.5)

    def test_all_discrete_demands_are_probabilities(self) -> None:
        successes, failures = np.meshgrid(np.arange(51), np.arange(51))
        models = [
            StandardThompsonSampling(),
            ScaledUpdates(0.25),
            ScaledUpdates(4.0),
            MeanPreservingTemperature(0.25),
            MeanPreservingTemperature(4.0),
            EpsilonGreedy(0.0),
            EpsilonGreedy(0.2),
            EpsilonGreedy(1.0),
        ]
        for model in models:
            demand = model.demand_counts(successes, failures)
            self.assertTrue(np.all(np.isfinite(demand)), model.method)
            self.assertTrue(np.all((0.0 <= demand) & (demand <= 1.0)), model.method)


class ForgettingDemandTests(unittest.TestCase):
    def test_beta_coordinate_round_trip_and_posterior_statistics(self) -> None:
        model = ObservationForgettingTS(0.8)
        state = model.from_beta((2.0, 3.0))
        np.testing.assert_allclose(state, (0.2, 0.4), atol=2e-15)
        np.testing.assert_allclose(model.to_beta(state), (2.0, 3.0), atol=2e-15)
        self.assertAlmostEqual(model.posterior_mean(state), 0.4)
        self.assertAlmostEqual(model.effective_concentration(state), 5.0)
        self.assertAlmostEqual(model.effective_sample_size(state), 3.0)

    def test_observation_and_calendar_transitions(self) -> None:
        state = (0.30, 0.20)
        observation = ObservationForgettingTS(0.8)
        calendar = CalendarForgettingTS(0.8)
        np.testing.assert_allclose(observation.success_state(state), (0.44, 0.16))
        np.testing.assert_allclose(observation.failure_state(state), (0.24, 0.36))
        self.assertEqual(observation.idle_state(state), state)
        np.testing.assert_allclose(calendar.idle_state(state), (0.24, 0.16))
        self.assertTrue(observation.has_exact_idle_self_loop)
        self.assertFalse(calendar.has_exact_idle_self_loop)

    def test_forgetting_transitions_stay_in_reachable_triangle(self) -> None:
        boundary_states = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (0.3, 0.7)]
        for model in (ObservationForgettingTS(0.8), CalendarForgettingTS(0.99)):
            for state in boundary_states:
                successors = (
                    model.success_state(state),
                    model.failure_state(state),
                    model.idle_state(state),
                )
                for x, y in successors:
                    self.assertGreaterEqual(x, -1e-15)
                    self.assertGreaterEqual(y, -1e-15)
                    self.assertLessEqual(x + y, 1.0 + 1e-14)

    def test_forgetting_demands_are_probabilities_and_vectorized(self) -> None:
        x_values = np.linspace(0.0, 1.0, 31)
        x, y = np.meshgrid(x_values, x_values)
        mask = x + y <= 1.0
        for model in (ObservationForgettingTS(0.8), CalendarForgettingTS(0.99)):
            demand = model.demand_xy(x[mask], y[mask])
            self.assertEqual(demand.shape, x[mask].shape)
            self.assertTrue(np.all(np.isfinite(demand)))
            self.assertTrue(np.all((0.0 <= demand) & (demand <= 1.0)))
            self.assertAlmostEqual(model.demand((0.0, 0.0)), 0.5)


if __name__ == "__main__":
    unittest.main()
