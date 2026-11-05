"""Independent checks for crossed fluid-oblivious responses and realized paths."""

import math
import unittest

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code.model import outside_demand_probability
from oblivious.fluid_oblivious.code.trajectories import simulate_pair
from oblivious.fluid_oblivious_initial_conditions.code.initial_conditions import (
    contains,
    phase_limit,
)
from oblivious.fluid_asymmetric.code.model import (
    analyse_pair,
    independent_transformed_demand,
    intersect_intervals,
    simulate_pure_pair_log,
)


def interval(lo, hi, left=True, right=True):
    return dict(lo=lo, hi=hi, left_closed=left, right_closed=right)


class IntervalIntersectionTests(unittest.TestCase):
    def test_touching_closed_intervals_keep_the_singleton(self):
        result = intersect_intervals([interval(0., .5)], [interval(.5, 1.)])
        self.assertEqual(result, [interval(.5, .5)])
        self.assertTrue(contains(result, .5))

    def test_touching_intervals_with_an_open_end_have_no_intersection(self):
        self.assertEqual(
            intersect_intervals([interval(0., .5, right=False)], [interval(.5, 1.)]),
            [],
        )
        self.assertEqual(
            intersect_intervals([interval(0., .5)], [interval(.5, 1., left=False)]),
            [],
        )

    def test_disconnected_sets_preserve_each_endpoint_choice(self):
        first = [interval(0., .3, right=False), interval(.6, 1., left=False)]
        second = [interval(.2, .8, right=False)]
        self.assertEqual(
            intersect_intervals(first, second),
            [interval(.2, .3, right=False), interval(.6, .8, left=False, right=False)],
        )
        self.assertEqual(intersect_intervals([], second), [])


class IndependentDemandTests(unittest.TestCase):
    def test_smoothed_quadrature_matches_adaptive_integrals_at_endpoint_states(self):
        own = np.array([[0., .001, .35], [.99, .999, 1.]])
        for p0, rho in ((.8, .9), (.9, .7), (.9, .8), (.7, .9)):
            params = Parameters(rho=rho)
            for conjecture in (params.p1, params.p2):
                with self.subTest(p0=p0, rho=rho, conjecture=conjecture):
                    vector = independent_transformed_demand(
                        params, p0, conjecture, own, order=256,
                    )
                    scalar = np.array([
                        independent_transformed_demand(params, p0, conjecture, float(x), order=256)
                        for x in own.flat
                    ]).reshape(own.shape)
                    adaptive = np.array([
                        outside_demand_probability(float(x), conjecture, params, p0)
                        for x in own.flat
                    ]).reshape(own.shape)
                    self.assertEqual(vector.shape, own.shape)
                    np.testing.assert_allclose(vector, scalar, atol=1e-15, rtol=1e-14)
                    np.testing.assert_allclose(vector, adaptive, atol=2e-13, rtol=2e-10)


class CrossedResponseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = Parameters(rho=.9)
        cls.pair = analyse_pair(cls.params, .8, n=1001, order=128)

    def test_real_crossed_responses_fill_the_intermediate_initial_states(self):
        pair = self.pair
        self.assertTrue(pair["qualifies"])
        self.assertTrue(contains(pair["common_initial_intervals"], .55))
        self.assertFalse(contains(pair["common_initial_intervals"], .5))
        self.assertFalse(contains(pair["common_initial_intervals"], .6))
        a, b = pair["response_a"], pair["response_b"]
        self.assertEqual((a["conjecture"], a["target"]), (.8, .35))
        self.assertEqual((b["conjecture"], b["target"]), (.35, .8))
        for response in (a, b):
            self.assertTrue(response["diagnostics"]["target_optimal"])
            self.assertTrue(response["diagnostics"]["action_valid"])
            self.assertLess(response["diagnostics"]["hjb_residual"], 1e-8)
            self.assertTrue(response["basin"]["stationary"])
        self.assertLess(a["diagnostics"]["hamiltonian_gap"], 0.)
        self.assertGreater(b["diagnostics"]["hamiltonian_gap"], 0.)

    def test_pure_pair_is_rejected_when_high_target_is_not_optimal(self):
        pair = analyse_pair(self.params, .7, n=1001, order=128)
        self.assertFalse(pair["qualifies"])
        self.assertEqual(pair["common_initial_intervals"], [])
        self.assertFalse(pair["response_b"]["diagnostics"]["target_optimal"])
        self.assertLess(pair["response_b"]["diagnostics"]["hamiltonian_gap"], 0.)

    def test_actual_joint_paths_follow_phase_limits_and_swap_with_labels(self):
        params, pair = self.params, self.pair
        a, b = pair["response_a"], pair["response_b"]
        times = np.linspace(0., 3000., 121)
        ordinary = simulate_pair(
            params, .8, a, response_b=b, initial=(.55, .55), times=times,
            max_step=25., demand_order=128,
        )["by_time"]
        swapped = simulate_pair(
            params, .8, b, response_b=a, initial=(.55, .55), times=times,
            max_step=25., demand_order=128,
        )["by_time"]
        self.assertEqual(phase_limit(params, a, .55), params.p1)
        self.assertEqual(phase_limit(params, b, .55), params.p2)
        self.assertTrue(np.all(np.diff(ordinary.x_a) <= 1e-10))
        self.assertTrue(np.all(np.diff(ordinary.x_b) >= -1e-10))
        self.assertTrue(np.all(ordinary.x_a >= params.p1-1e-10))
        self.assertTrue(np.all(ordinary.x_b <= params.p2+1e-10))
        self.assertLess(ordinary.x_a.iloc[-1], .55)
        self.assertAlmostEqual(ordinary.x_b.iloc[-1], params.p2, places=8)
        np.testing.assert_allclose(ordinary.x_a, swapped.x_b, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(ordinary.x_b, swapped.x_a, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(ordinary.demand_a, swapped.demand_b, atol=2e-9, rtol=2e-9)
        # An independent operational-time identity checks the realized demand
        # counters and drift together, without mistaking x(T) for the limit.
        lam = -math.log(params.rho)
        expected_a = params.p1+(.55-params.p1)*np.exp(-lam*ordinary.cumulative_own_sales)
        expected_b = params.p2+(.55-params.p2)*np.exp(-lam*ordinary.cumulative_b_sales)
        np.testing.assert_allclose(ordinary.x_a, expected_a, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(ordinary.x_b, expected_b, atol=2e-9, rtol=2e-9)

    def test_log_coordinates_match_event_paths_and_reach_long_run_targets(self):
        params, pair = self.params, self.pair
        short_times = np.linspace(0., 3000., 61)
        event_path = simulate_pair(
            params, .8, pair["response_a"], response_b=pair["response_b"],
            initial=(.55, .55), times=short_times, max_step=25., demand_order=128,
        )["by_time"]
        transformed = simulate_pure_pair_log(
            params, .8, pair, .55, np.r_[short_times, 1e6], order=128,
        )
        np.testing.assert_allclose(
            transformed["x_a"][:-1], event_path.x_a, atol=2e-9, rtol=2e-9,
        )
        np.testing.assert_allclose(
            transformed["x_b"][:-1], event_path.x_b, atol=2e-9, rtol=2e-9,
        )
        self.assertLess(abs(transformed["x_a"][-1]-params.p1), 1e-7)
        self.assertLess(abs(transformed["x_b"][-1]-params.p2), 1e-7)

    def test_log_coordinates_reject_a_policy_switch_on_the_proposed_orbit(self):
        # A starts with product 1, then must change to product 2 at .4.
        # Integrating product 1 forever would invent an inadmissible orbit.
        pair = dict(
            response_a=dict(policy=dict(roots=[.4], products=[2, 1])),
            response_b=dict(policy=dict(roots=[], products=[2])),
        )
        with self.assertRaisesRegex(ValueError, "policy switch"):
            simulate_pure_pair_log(self.params, .8, pair, .55, [0., 3000.])


class AsymmetricIntegratorTests(unittest.TestCase):
    def test_two_products_and_unequal_demands_match_analytical_paths(self):
        params = Parameters(rho=math.exp(-.2))
        times = np.linspace(0., 100., 101)
        result = simulate_pair(
            params, .5, dict(roots=[], products=[1]),
            response_b=dict(roots=[], products=[2]), initial=(.55, .55),
            times=times, demand_function=lambda a, b: (.2, .6, .2),
        )["by_time"]
        expected_a = params.p1+(.55-params.p1)*np.exp(-.04*times)
        expected_b = params.p2+(.55-params.p2)*np.exp(-.12*times)
        np.testing.assert_allclose(result.x_a, expected_a, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(result.x_b, expected_b, atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(result.cumulative_own_sales, .2*times, atol=1e-10)
        np.testing.assert_allclose(result.cumulative_b_sales, .6*times, atol=1e-10)
        np.testing.assert_allclose(result.alpha_a, 0.)
        np.testing.assert_allclose(result.alpha_b, 1.)


if __name__ == "__main__":
    unittest.main()
