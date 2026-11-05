"""Independent analytical checks for actual-state deterministic paths."""

import math
import unittest

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.fluid_oblivious.code.trajectories import actual_selection, simulate_pair


class TrajectoryTests(unittest.TestCase):
    def setUp(self):
        self.params = Parameters(rho=math.exp(-.2))

    def test_fixed_product_matches_exact_reputation_and_sale_counters(self):
        times = np.linspace(0, 100, 201)
        result = simulate_pair(self.params, .5, dict(roots=[], products=[1]),
            initial=(.1, .9), times=times, demand_function=lambda a, b: (.2, .6, .2))
        frame = result["by_time"]
        np.testing.assert_allclose(frame.x_a, self.params.p1+(.1-self.params.p1)*np.exp(-.04*times), atol=1e-9)
        np.testing.assert_allclose(frame.x_b, self.params.p1+(.9-self.params.p1)*np.exp(-.12*times), atol=1e-9)
        np.testing.assert_allclose(frame.cumulative_own_sales, .2*times, atol=1e-10)
        self.assertEqual(result["diagnostics"]["own_product2_given_sale"], 0.)
        delta = -math.log(self.params.gamma)
        exact_profit = .2*(self.params.R-self.params.c1)*(1-math.exp(-delta*times[-1]))/delta
        self.assertAlmostEqual(result["diagnostics"]["value_initial_a"], exact_profit, places=9)

    def test_product2_given_sale_differs_from_calendar_rate(self):
        result = simulate_pair(self.params, .5, dict(roots=[], products=[2]),
            times=[0., 25., 50.], demand_function=lambda a, b: (.2, .6, .2))
        np.testing.assert_allclose(result["binned"].product2_given_sale, 1.)
        np.testing.assert_allclose(result["binned"].own_product2_rate, .2)
        np.testing.assert_allclose(result["binned"].customer_product2_rate, .8)

    def test_simultaneous_attractive_events_hold_without_chatter(self):
        root, x0, demand = .6, .4, .3
        hit_time = -math.log((self.params.p2-root)/(self.params.p2-x0))/(.2*demand)
        result = simulate_pair(self.params, .5, dict(roots=[root], products=[2, 1]),
            initial=(x0, x0), times=np.linspace(0, 100, 401),
            demand_function=lambda a, b: (demand, demand, .4))
        events = result["events"]
        self.assertEqual(len(events), 2)
        self.assertAlmostEqual(events[0]["time"], hit_time, places=7)
        self.assertAlmostEqual(events[1]["time"], hit_time, places=7)
        frame = result["by_time"]
        held = frame.time >= hit_time
        np.testing.assert_allclose(frame.x_a[held], root, atol=1e-13)
        alpha = (root-self.params.p1)/(self.params.p2-self.params.p1)
        np.testing.assert_allclose(frame.alpha_a[held], alpha)
        expected_share = (hit_time+alpha*(100-hit_time))/100
        self.assertAlmostEqual(result["diagnostics"]["own_product2_given_sale"], expected_share, places=8)

    def test_crossing_changes_product_before_attractive_hold(self):
        result = simulate_pair(self.params, .5, dict(roots=[.2, .6], products=[1, 2, 1]),
            initial=(.1, .1), times=np.linspace(0, 150, 301),
            demand_function=lambda a, b: (.4, .4, .2))
        self.assertEqual([event["kind"] for event in result["events"]],
                         ["crossing", "crossing", "attractive", "attractive"])
        self.assertAlmostEqual(result["by_time"].x_a.iloc[-1], .6, places=12)

    def test_actual_selection_uses_both_reputations_and_partitions_mass(self):
        params = Parameters(rho=.7)
        symmetric = actual_selection(.5, .5, params, .5)
        self.assertAlmostEqual(symmetric[0], symmetric[1], places=14)
        self.assertAlmostEqual(symmetric.sum(), 1., places=14)
        # Independently evaluate the original competition model's adaptive
        # scalar integrals; this catches a frozen-conjecture demand on paths.
        from oblivious.oe_distribution_fixed_p0.code.model import outside_demand_probability
        asymmetric = actual_selection(.4, .7, params, .5)
        expected_a = outside_demand_probability(.4, .7, params, .5)
        expected_b = outside_demand_probability(.7, .4, params, .5)
        self.assertAlmostEqual(asymmetric[0], expected_a, places=10)
        self.assertAlmostEqual(asymmetric[1], expected_b, places=10)
        self.assertAlmostEqual(asymmetric.sum(), 1., places=12)
        self.assertNotAlmostEqual(asymmetric[0], actual_selection(.4, .2, params, .5)[0], places=5)

    def test_repelling_threshold_separates_opposite_paths(self):
        result = simulate_pair(self.params, .5, dict(roots=[.5], products=[1, 2]),
            initial=(.4, .6), times=np.linspace(0, 200, 401),
            demand_function=lambda a, b: (.4, .4, .2))
        frame = result["by_time"]
        self.assertTrue(np.all(np.diff(frame.x_a) <= 1e-12))
        self.assertTrue(np.all(np.diff(frame.x_b) >= -1e-12))
        self.assertLess(abs(frame.x_a.iloc[-1]-self.params.p1), 1e-7)
        self.assertLess(abs(frame.x_b.iloc[-1]-self.params.p2), 1e-7)
        self.assertEqual(result["events"], [])

    def test_no_sale_conditional_fraction_is_undefined(self):
        result = simulate_pair(self.params, 1., dict(roots=[], products=[2]),
                               times=[0., 25., 50.])
        np.testing.assert_allclose(result["by_time"].x_a, .5)
        self.assertTrue(result["by_time"].product2_given_sale.isna().all())
        self.assertTrue(result["binned"].product2_given_sale.isna().all())
        np.testing.assert_allclose(result["binned"].own_product2_rate, 0.)

    def test_explicit_validated_holding_choice_is_respected_at_repelling_tie(self):
        root = .5
        alpha = (root-self.params.p1)/(self.params.p2-self.params.p1)
        # This tests only trajectory semantics. The HJB solver must establish
        # whether a supplied repelling-tie mixture is actually optimal.
        policy = dict(roots=[root], products=[1, 2],
                      holding_points=[dict(x=root, alpha=alpha)])
        result = simulate_pair(self.params, .5, policy, initial=(root, root),
                               times=[0., 25., 50.])
        np.testing.assert_allclose(result["by_time"].x_a, root, atol=1e-13)
        np.testing.assert_allclose(result["by_time"].alpha_a, alpha, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
