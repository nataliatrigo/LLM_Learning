"""Topology and consistency tests for common-initial-state equilibria."""
import math
from types import SimpleNamespace
import unittest

try:
    from .initial_conditions import initial_conditions_for_equilibrium, contains, phase_limit
except ImportError:
    from initial_conditions import initial_conditions_for_equilibrium, contains, phase_limit


class InitialConditionsTests(unittest.TestCase):
    params = SimpleNamespace(p1=.35, p2=.8)

    def analyse(self, roots, products, target, holding=()):
        policy = dict(roots=roots, products=products, holding_points=list(holding))
        return initial_conditions_for_equilibrium(self.params, policy, target)

    def test_constant_pure_policy_has_global_initial_conditions(self):
        result = self.analyse([], [1], .35)
        self.assertTrue(result["stationary"])
        self.assertEqual(result["intervals"], [dict(lo=0., hi=1., left_closed=True, right_closed=True)])
        self.assertEqual(result["measure"], 1.)

    def test_repelling_boundary_product1_selector_and_nextafter(self):
        policy = dict(roots=[.5], products=[1, 2])
        low = initial_conditions_for_equilibrium(self.params, policy, .35)
        high = initial_conditions_for_equilibrium(self.params, policy, .8)
        self.assertEqual(low["text"], "[0, 0.5]")
        self.assertEqual(high["text"], "(0.5, 1]")
        for x in (math.nextafter(.5, 0.), .5, math.nextafter(.5, 1.)):
            expected = .35 if x <= .5 else .8
            self.assertEqual(phase_limit(self.params, policy, x), expected)
            self.assertEqual(contains(low["intervals"], x), x <= .5)
            self.assertEqual(contains(high["intervals"], x), x > .5)

    def test_attractive_root_holds_and_has_global_initial_conditions(self):
        result = self.analyse([.5], [2, 1], .5)
        self.assertTrue(result["stationary"])
        self.assertLess(result["stationarity_defect"], 1e-15)
        self.assertEqual(result["text"], "[0, 1]")
        self.assertTrue(result["root_actions"][0]["holding"])

    def test_crossing_roots_do_not_create_false_stationary_states(self):
        policy = dict(roots=[.2, .5, .9], products=[2, 1, 2, 1])
        low = initial_conditions_for_equilibrium(self.params, policy, .35)
        high = initial_conditions_for_equilibrium(self.params, policy, .8)
        self.assertEqual(low["text"], "[0, 0.5]")
        self.assertEqual(high["text"], "(0.5, 1]")
        self.assertEqual(phase_limit(self.params, policy, .2), .35)
        self.assertEqual(phase_limit(self.params, policy, .9), .8)
        invalid = initial_conditions_for_equilibrium(self.params, policy, .2)
        self.assertFalse(invalid["stationary"])
        self.assertEqual(invalid["intervals"], [])

    def test_explicit_repelling_holding_creates_singleton_equilibrium(self):
        alpha = (.5-self.params.p1)/(self.params.p2-self.params.p1)
        holding = [dict(x=.5, alpha=alpha)]
        result = self.analyse([.5], [1, 2], .5, holding)
        self.assertEqual(result["text"], "{0.5}")
        self.assertEqual(result["measure"], 0.)
        low = self.analyse([.5], [1, 2], .35, holding)
        high = self.analyse([.5], [1, 2], .8, holding)
        self.assertEqual(low["text"], "[0, 0.5)")
        self.assertEqual(high["text"], "(0.5, 1]")

    def test_degenerate_switch_at_pure_target_is_stationary(self):
        result = self.analyse([.35], [1, 2], .35)
        self.assertTrue(result["stationary"])
        self.assertEqual(result["text"], "[0, 0.35]")
        global_result = self.analyse([.35], [2, 1], .35)
        self.assertEqual(global_result["text"], "[0, 1]")
        upper = self.analyse([.8], [1, 2], .8)
        self.assertEqual(upper["text"], "[0.8, 1]")

    def test_input_validation_rejects_false_holding_and_bad_policy(self):
        with self.assertRaises(ValueError):
            self.analyse([.5], [1, 2], .5, [dict(x=.5, alpha=0.)])
        with self.assertRaises(ValueError):
            self.analyse([.5, .4], [1, 2, 1], .35)
        with self.assertRaises(ValueError):
            self.analyse([.5], [1], .35)
        with self.assertRaises(ValueError):
            phase_limit(self.params, dict(roots=[], products=[1]), 1.01)


if __name__ == "__main__":
    unittest.main()
