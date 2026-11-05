"""Analytic and independent ODE checks for the deterministic simulator."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.integrate import solve_ivp
from scipy.special import betaincc

from simulation import simulate_paths, summarize_paths


def fake_solution(report_n=20.0):
    params = SimpleNamespace(
        p0=0.5, p1=0.35, p2=0.8, c1=0.05, c2=0.65,
        revenue=1.0, delta=0.03,
    )

    def checked_action(S, F):
        if min(S, F) < 0.0 or S + F > report_n + 1e-12:
            raise AssertionError("policy evaluated outside the retained triangle")
        return 1

    return SimpleNamespace(params=params, report_n=report_n, action_at=checked_action)


class DeterministicFluidTrajectories(unittest.TestCase):
    def test_constant_demand_matches_states_and_exact_discounted_rewards(self):
        solution = fake_solution()
        with patch("simulation._demand", return_value=0.4):
            paths = simulate_paths(solution, [(0.2, 0.3)], horizon=4.0, dt=0.13)
        summary = summarize_paths(paths)
        for _, row in summary.iterrows():
            product = 2 if row.policy == "product2" else 1
            p = solution.params.p2 if product == 2 else solution.params.p1
            cost = solution.params.c2 if product == 2 else solution.params.c1
            assert_allclose([row.final_time, row.final_S, row.final_F],
                            [4.0, 0.2 + 1.6*p, 0.3 + 1.6*(1-p)], atol=2e-14)
            expected = 0.4*(1-cost)*(-np.expm1(-0.03*4))/0.03
            self.assertAlmostEqual(row.discounted_profit, expected, places=13)
            self.assertAlmostEqual(row.product2_time_share, float(product == 2), places=13)
            self.assertEqual(row.termination_reason, "horizon")

    def test_boundary_stops_without_extrapolation(self):
        solution = fake_solution(report_n=1.0)
        with patch("simulation._demand", return_value=0.4):
            paths = simulate_paths(solution, [(0.1, 0.1)], horizon=10.0, dt=0.3)
        terminal = paths.groupby("policy").tail(1)
        assert_allclose(terminal["n"], 1.0, atol=1e-14)
        assert_allclose(terminal["t"], 2.0, atol=1e-14)
        self.assertTrue((terminal["termination_reason"] == "report_boundary").all())
        self.assertTrue((terminal["product"] == 0).all())
        self.assertLessEqual(paths["n"].max(), 1.0 + 1e-14)

    def test_two_hundred_time_units_fit_extended_domain_at_maximum_demand(self):
        solution = fake_solution(report_n=240.0)
        for demand in (0.4, 1.0):
            with self.subTest(demand=demand):
                with patch("simulation._demand", return_value=demand):
                    paths = simulate_paths(solution, [(10.0, 10.0)],
                                           horizon=200.0, dt=0.37)
                summary = summarize_paths(paths)
                self.assertEqual(set(summary["policy"]),
                                 {"optimal", "product1", "product2"})
                self.assertTrue((summary["termination_reason"] == "horizon").all())
                assert_allclose(summary["final_time"], 200.0, atol=2e-12, rtol=0)
                # Even demand one cannot leave the interior before t=200:
                # n(t) <= n(0) + t = 220, below the report boundary at 240.
                self.assertLessEqual(paths["n"].max(), 220.0 + 2e-11)
                self.assertGreaterEqual(paths[["S", "F"]].to_numpy().min(), 0.0)
                for policy, group in paths.groupby("policy"):
                    increments_t = np.diff(group["t"])
                    self.assertTrue((increments_t > 0.0).all())
                    rates_n = np.diff(group["n"]) / increments_t
                    assert_allclose(rates_n, demand, atol=2e-11, rtol=0)
                    self.assertLessEqual(rates_n.max(), 1.0 + 2e-11)
                for _, row in summary.iterrows():
                    product = 2 if row.policy == "product2" else 1
                    p = solution.params.p2 if product == 2 else solution.params.p1
                    cost = solution.params.c2 if product == 2 else solution.params.c1
                    assert_allclose([row.final_S, row.final_F],
                                    [10.0 + 200.0*demand*p,
                                     10.0 + 200.0*demand*(1.0-p)],
                                    atol=2e-11, rtol=0)
                    expected = demand*(1.0-cost)*(-np.expm1(-0.03*200.0))/0.03
                    self.assertAlmostEqual(row.discounted_profit, expected, places=12)

    def test_time_refinement_converges_to_independent_scipy_ode(self):
        solution = fake_solution()
        p = solution.params

        def rhs(t, y):
            D = betaincc(y[0]+1, y[1]+1, p.p0)
            return [D*p.p2, D*(1-p.p2), np.exp(-p.delta*t)*D*(1-p.c2)]

        reference = solve_ivp(rhs, (0.0, 4.0), [0.0, 0.0, 0.0], rtol=1e-12, atol=1e-14).y[:, -1]
        errors = []
        for dt in (0.1, 0.05):
            paths = simulate_paths(solution, [(0.0, 0.0)], horizon=4.0, dt=dt)
            terminal = paths.loc[paths["policy"] == "product2"].iloc[-1]
            errors.append(float(np.max(np.abs(terminal[["S", "F", "discounted_profit"]].to_numpy(float) - reference))))
        self.assertLess(errors[1], errors[0] / 3.0)
        self.assertLess(errors[1], 1e-4)

    def test_invalid_or_empty_domains_are_rejected(self):
        solution = fake_solution(report_n=1.0)
        for states, horizon, dt in [([], 1, 0.1), ([(-1, 0)], 1, 0.1),
                                    ([(1, 1)], 1, 0.1), ([(0, 0)], 0, 0.1),
                                    ([(0, 0)], 1, 0), ([(0, np.nan)], 1, 0.1)]:
            with self.subTest(states=states, horizon=horizon, dt=dt):
                with self.assertRaises(ValueError):
                    simulate_paths(solution, states, horizon, dt)


if __name__ == "__main__":
    unittest.main()
