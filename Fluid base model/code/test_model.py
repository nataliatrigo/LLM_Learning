"""Analytic benchmarks for the infinite-horizon fluid discretization.

Run from this directory with ``python -m unittest -v test_model``.
The constant-demand examples have closed-form values and policies, independently
of the numerical solver's recurrence.
"""

from dataclasses import replace
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from model import Parameters, demand, solve
from diagnostics import load_solution, save_solution


class FluidModelTests(unittest.TestCase):
    def setUp(self):
        self.params = Parameters(gamma=.8)

    @staticmethod
    def constant_demand(rate):
        return lambda S, F: np.full(np.broadcast(S, F).shape, rate)

    def test_beta_demand_matches_known_distributions_and_real_symmetry(self):
        params = replace(self.params, p0=.27)
        self.assertAlmostEqual(float(demand(0., 0., params)), 1 - params.p0)
        self.assertAlmostEqual(float(demand(1., 0., params)), 1 - params.p0**2)
        self.assertAlmostEqual(float(demand(0., 1., params)), (1 - params.p0)**2)
        states = np.array([0., .125, .7, 2.25, 30.])
        np.testing.assert_allclose(demand(states, states, self.params), .5,
                                   rtol=0, atol=2e-14)

    def test_beta_demand_respects_success_and_failure_information(self):
        levels = np.array([0., .1, .5, 1., 4.])
        success = demand(levels, .4, self.params)
        failure = demand(.4, levels, self.params)
        self.assertTrue(np.all(np.diff(success) > 0))
        self.assertTrue(np.all(np.diff(failure) < 0))
        self.assertTrue(np.all((success >= 0) & (success <= 1)))

    def test_higher_p0_reduces_demand_and_discounted_grid_value(self):
        # With nonnegative rewards, increasing the purchase threshold slows
        # the same observation-time state paths and discounts their rewards
        # more heavily. Check every retained state, not just the origin.
        previous = None
        previous_demand = None
        for p0 in (.1, .3, .5, .7, .9):
            params = replace(self.params, p0=p0)
            result = solve(params, h=.5, cutoff_n=8., report_n=3.)
            current_demand = demand(np.array([0., .25, 3.]),
                                    np.array([2., .5, 0.]), params)
            self.assertAlmostEqual(result.demand_values[0][0], 1-p0)
            if previous is not None:
                self.assertTrue(np.all(current_demand < previous_demand))
                for first, second in zip(previous.values, result.values):
                    self.assertTrue(np.all(second <= first + 1e-12))
                self.assertLess(result.values[0][0], previous.values[0][0])
            previous, previous_demand = result, current_demand

    def test_constant_demand_has_exact_finite_boundary_value(self):
        rate, h, cutoff = .37, .25, 6.
        result = solve(self.params, h=h, cutoff_n=cutoff, report_n=2.,
                       demand_function=self.constant_demand(rate))
        q = rate / (rate + self.params.delta*h)
        infinite_value = rate*self.params.max_reward/self.params.delta
        total_steps = round(cutoff/h)
        for k, row in enumerate(result.values):
            remaining = total_steps - k
            expected = infinite_value * (-math.expm1(remaining*math.log(q)))
            np.testing.assert_allclose(row, expected, rtol=2e-13, atol=2e-13)
            np.testing.assert_array_less(row - 2e-13, infinite_value)
            upper = result.upper_values[k]
            self.assertTrue(np.all(upper + 2e-13 >= infinite_value))
            expected_gap = q**remaining*self.params.max_reward/self.params.delta
            np.testing.assert_allclose(upper-row, expected_gap,
                                       rtol=3e-12, atol=3e-13)

    def test_constant_demand_always_selects_cheapest_product(self):
        result = solve(self.params, h=.25, cutoff_n=8., report_n=2.,
                       demand_function=self.constant_demand(.4))
        for S, F in [(0., 0.), (.3, .6), (1.25, .5), (0., 2.)]:
            self.assertEqual(result.action_at(S, F), 1)
        for row in result.advantages:
            np.testing.assert_allclose(row, -self.params.delta_c,
                                       rtol=0, atol=2e-13)

    def test_nominal_action_uses_configured_indifference_tolerance(self):
        params = replace(self.params, c2=self.params.c1)
        reference = solve(params, h=.5, cutoff_n=12., report_n=2.,
                          action_tolerance=0.)
        advantage = reference.interpolate_field(0., 0., 'advantages')
        self.assertGreater(advantage, 1e-6)
        tolerant = solve(params, h=.5, cutoff_n=12., report_n=2.,
                         action_tolerance=2*advantage)
        self.assertEqual(reference.action_at(0., 0.), 2)
        self.assertEqual(tolerant.action_at(0., 0.), 1)
        self.assertEqual(tolerant.action_at(0., 0., tolerance=0.), 2)
        self.assertAlmostEqual(tolerant.interpolate_field(0., 0., 'advantages'),
                               advantage, places=14)
        for first, second in zip(reference.values, tolerant.values):
            np.testing.assert_array_equal(first, second)

    def test_saved_solution_roundtrip_preserves_values_certificates_and_policy(self):
        original = solve(replace(self.params, p0=.43), h=.25, cutoff_n=6.,
                         report_n=2., action_tolerance=.0123)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'solution.npz'
            save_solution(original, path)
            restored = load_solution(path)
        self.assertEqual(restored.params, original.params)
        for name in ('h', 'cutoff_n', 'report_n', 'residual',
                     'value_error_bound', 'theoretical_error_bound',
                     'action_tolerance'):
            self.assertEqual(getattr(restored, name), getattr(original, name))
        for name in ('values', 'upper_values', 'advantages', 'advantage_lower',
                     'advantage_upper', 'actions', 'demand_values'):
            expected, actual = getattr(original, name), getattr(restored, name)
            self.assertEqual(len(actual), len(expected))
            for first, second in zip(expected, actual):
                np.testing.assert_array_equal(first, second)
        for S, F in [(0., 0.), (.13, .31), (.7, 1.3), (2., 0.)]:
            self.assertEqual(restored.action_at(S, F), original.action_at(S, F))
            self.assertEqual(restored.interpolate_field(S, F),
                             original.interpolate_field(S, F))

    def test_no_sales_has_zero_value_despite_positive_terminal_bound(self):
        result = solve(self.params, h=.5, cutoff_n=4., report_n=2.,
                       demand_function=self.constant_demand(0.))
        total_steps = round(result.cutoff_n/result.h)
        for k, row in enumerate(result.values):
            if k < total_steps:
                np.testing.assert_array_equal(row, 0.)
                np.testing.assert_array_equal(result.upper_values[k], 0.)
        self.assertEqual(result.value_error_bound, 0.)

    def test_real_demand_value_is_bounded_and_certified(self):
        result = solve(self.params, h=.25, cutoff_n=8., report_n=2.)
        value_cap = self.params.max_reward/self.params.delta
        for lower, upper in zip(result.values, result.upper_values):
            self.assertTrue(np.all(np.isfinite(lower)))
            self.assertTrue(np.all(lower >= -1e-13))
            self.assertTrue(np.all(lower <= upper + 1e-12))
            self.assertTrue(np.all(upper <= value_cap + 1e-12))
        self.assertLess(result.residual, 1e-10)
        report_steps = round(result.report_n/result.h)
        actual_gap = max(float(np.max(upper-lower))
                         for lower, upper in zip(result.values[:report_steps+1],
                                                 result.upper_values[:report_steps+1]))
        self.assertAlmostEqual(result.value_error_bound, actual_gap, places=11)
        geometric_bound = value_cap*(1+self.params.delta*result.h)**(
            -(result.cutoff_n-result.report_n)/result.h)
        self.assertLessEqual(result.value_error_bound, geometric_bound+1e-11)

    def test_larger_cutoff_tightens_interval_around_analytic_infinite_value(self):
        options = dict(h=.5, report_n=1.,
                       demand_function=self.constant_demand(.4))
        small = solve(self.params, cutoff_n=4., **options)
        large = solve(self.params, cutoff_n=8., **options)
        self.assertGreater(large.values[0][0], small.values[0][0])
        self.assertLess(large.upper_values[0][0], small.upper_values[0][0])
        self.assertLess(large.value_error_bound, small.value_error_bound)

    def test_piecewise_linear_interpolation_reproduces_affine_fields(self):
        result = solve(self.params, h=.5, cutoff_n=4., report_n=2.)
        for k, row in enumerate(result.values):
            S = result.h*np.arange(k+1)
            F = result.h*(k-np.arange(k+1))
            row[:] = 2 + 3*S - 4*F
        for S, F in [(0., 0.), (.13, .31), (.4, .45), (.9, .25),
                     (.6, .8), (1.5, .5), (0., 1.35), (1.35, 0.)]:
            with self.subTest(S=S, F=F):
                self.assertAlmostEqual(result.interpolate_field(S, F),
                                       2 + 3*S - 4*F, places=12)

    def test_interpolation_rejects_extrapolation(self):
        result = solve(self.params, h=.5, cutoff_n=4., report_n=2.)
        for S, F in [(-.01, 0.), (0., -.01), (5., 0.), (3., 2.),
                     (float('nan'), 0.), (0., float('inf'))]:
            with self.subTest(S=S, F=F), self.assertRaises(ValueError):
                result.interpolate_field(S, F)

    def test_interpolation_projects_accepted_roundoff_at_report_edge(self):
        result = solve(self.params, h=.5, cutoff_n=4., report_n=2.)
        for k, row in enumerate(result.values):
            S = result.h*np.arange(k+1)
            F = result.h*(k-np.arange(k+1))
            row[:] = 2 + 3*S - 4*F
        states = [(2.+1e-10, 0.), (1.1, .9+1e-10),
                  (-1e-11, 2.+1e-11), (2.+1e-11, -1e-11)]
        for S, F in states:
            clipped_S, clipped_F = max(0., S), max(0., F)
            scale = min(1., result.report_n/(clipped_S+clipped_F))
            expected = 2 + 3*clipped_S*scale - 4*clipped_F*scale
            with self.subTest(S=S, F=F):
                self.assertAlmostEqual(result.interpolate_field(S, F),
                                       expected, places=11)

    def test_invalid_parameters_are_rejected(self):
        for changes in [dict(gamma=0.), dict(gamma=1.), dict(gamma=float('nan')),
                        dict(p1=-.1), dict(p2=1.1), dict(p1=.8, p2=.35),
                        dict(c1=-.1), dict(c2=1.1), dict(c1=.7, c2=.6)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.params, **changes)

    def test_invalid_meshes_are_rejected(self):
        for settings in [dict(h=0., cutoff_n=4., report_n=2.),
                         dict(h=-.5, cutoff_n=4., report_n=2.),
                         dict(h=.3, cutoff_n=4., report_n=1.5),
                         dict(h=.5, cutoff_n=4., report_n=1.1),
                         dict(h=.5, cutoff_n=4., report_n=4.),
                         dict(h=.5, cutoff_n=4., report_n=-.5)]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                solve(self.params, **settings)


if __name__ == '__main__':
    unittest.main()
