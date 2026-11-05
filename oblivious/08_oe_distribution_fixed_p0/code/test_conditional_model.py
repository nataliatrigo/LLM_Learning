"""Independent checks for conditional OE with the same fixed external p0.

Run: .venv/bin/python -B -m unittest oblivious.oe_distribution_fixed_p0.code.test_conditional_model -v
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from oblivious.oe_conditional_distribution.code.model import ConditionalExperiment
from oblivious.oe_distribution_fixed_p0.code.conditional_model import (
    FixedOutsideConditionalExperiment, _ConditionalResponse, conditional_from_joint,
)
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment
from oblivious.oe_distribution_fixed_p0.code.test_model import reference_events, reference_stationary
from oblivious.oe_p0.code.primitives import Parameters


class FixedOutsideConditional(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.e = FixedOutsideConditionalExperiment(n=7, params=Parameters(rho=.6))
        cls.alpha = np.array([.15, .7, .2, .85, .1, .6, .3])
        cls.result = cls.e.evaluate(cls.alpha)

    def test_partition_and_full_joint_transition_match_independent_events(self):
        e = self.e
        assert_allclose(e.game.D+e.game.D.T+e.game.outside, 1, atol=3e-15, rtol=0)
        a = np.broadcast_to(self.alpha[:, None], (e.n, e.n))
        matrix, _ = reference_events(e.grid, e.game.D, e.game.outside, a, a.T, e.params)
        assert_allclose(e.game.transition(a, a.T).toarray(), matrix, atol=9e-16, rtol=0)
        assert_allclose(matrix.sum(axis=1), 1, atol=3e-15, rtol=0)
        reference = reference_stationary(matrix).reshape(e.n, e.n)
        assert_allclose(self.result["nu"], reference, atol=3e-11, rtol=0)
        # The outside event has zero reward and places all its mass on the
        # identical joint state, including neither seller's feedback update.
        selected = matrix-np.diag(e.game.outside.ravel())
        assert_allclose(selected.sum(axis=1), (e.game.D+e.game.D.T).ravel(), atol=4e-16)

    def test_same_primitives_as_marginal_but_actual_rowwise_conditional_demand(self):
        e, r = self.e, self.result
        marginal = FixedOutsideMarginalExperiment(n=e.n, params=e.params, p0=e.p0)
        assert_array_equal(e.game.D, marginal.game.D)
        conditional = r["nu"]/r["mu_a"][:, None]
        assert_allclose(r["conditional_mu"], conditional, atol=2e-13, rtol=0)
        assert_allclose(r["demand"], np.sum(conditional*e.game.D, axis=1), atol=2e-13, rtol=0)
        # One-coordinate observation updates do not imply a product calendar
        # law: selection rates depend on both reputations.
        self.assertGreater(np.sum(abs(r["nu"]-np.outer(r["mu_a"], r["mu_b"]))), 1e-3)
        self.assertGreater(np.max(abs(r["demand"]-e.game.D@r["mu_b"])), 1e-3)
        self.assertLess(r["conditional_diagnostics"]["independent_order_supported_row_l1_max"], 1e-10)
        self.assertTrue(r["conditional_diagnostics"]["independent_order_verified"])

    def test_independent_joint_law_exactly_reduces_to_marginal_demand(self):
        own = np.arange(1, 8, dtype=float); own /= own.sum()
        rival = own[::-1].copy()
        result = conditional_from_joint(np.outer(own, rival), self.e.game.D)
        assert_allclose(result["conditional_mu"], np.broadcast_to(rival, (7, 7)), atol=2e-15)
        assert_allclose(result["conditional_demand"], self.e.game.D@rival, atol=2e-15)

    def test_tiny_positive_rows_have_no_mass_floor_and_zero_rows_are_explicit(self):
        for tiny in (1e-40, 1e-240, 1e-320):
            with self.subTest(tiny=tiny):
                joint = np.diag([tiny, .4, .6])
                result = conditional_from_joint(joint, self.e.game.D[:3, :3])
                assert_array_equal(result["conditional_mu"][0], [1, 0, 0])
                self.assertEqual(result["zero_rows"], [])
        joint = np.diag([0., .4, .6])
        result = conditional_from_joint(joint, self.e.game.D[:3, :3])
        assert_array_equal(result["conditional_mu"][0], joint.sum(axis=0))
        self.assertEqual(result["zero_rows"], [0])

    def test_analytic_conditional_jacobian_matches_centered_differences(self):
        for rho in (.6, .95, .97):
            e = FixedOutsideConditionalExperiment(n=7, params=Parameters(rho=rho))
            engine = _ConditionalResponse(e)
            alpha = np.linspace(.2, .8, 7)
            jac = engine.calculate(alpha)["jacobian"].copy()
            for k in (0, 3, 6):
                with self.subTest(rho=rho, k=k):
                    delta = np.zeros(7); delta[k] = 1e-5
                    plus = engine.calculate(alpha+delta)["advantage"].copy()
                    minus = engine.calculate(alpha-delta)["advantage"].copy()
                    assert_allclose(jac[:, k], (plus-minus)/2e-5, atol=3e-8, rtol=8e-5)

    def test_zero_threshold_recovers_original_conditional_certificate(self):
        e = FixedOutsideConditionalExperiment(n=7, params=Parameters(rho=.6), p0=0)
        original = ConditionalExperiment(n=7, params=e.params)
        r, old = e.evaluate(self.alpha), original.evaluate(self.alpha)
        for key in ("nu", "mu_a", "mu_b", "conditional_mu", "demand", "value", "advantage"):
            assert_allclose(r[key], old[key], atol=3e-12, rtol=0)
        self.assertEqual(r["verified"], old["verified"])
        for name in ("dp_diagnostics", "conditional_diagnostics"):
            self.assertEqual(r[name]["verified"], old[name]["verified"])
        self.assertTrue(r["conditional_diagnostics"]["independent_order_verified"])
        myopic = Parameters(rho=.6, gamma=0)
        for experiment in (ConditionalExperiment(n=7, params=myopic),
                           FixedOutsideConditionalExperiment(n=7, params=myopic, p0=0)):
            self.assertTrue(experiment.evaluate(np.zeros(7))["verified"])

    def test_myopic_solution_and_false_conjecture_rejection(self):
        e = FixedOutsideConditionalExperiment(n=7, params=Parameters(rho=.6, gamma=0))
        result = e.solve(initial_policy=np.full(7, .4), max_evaluations=300,
                         picard_iterations=2)
        self.assertTrue(result["verified"], result["summary"])
        assert_array_equal(result["policy"], np.zeros(7))
        assert_allclose(result["value"], result["demand"]*(e.params.R-e.params.c1), atol=2e-13)
        false = e.evaluate(np.zeros(7), conjectured_mu=np.full((7, 7), 1/7))
        self.assertFalse(false["verified"])
        self.assertFalse(false["conditional_diagnostics"]["verified"])

    def test_configuration_retains_outside_timing_and_conditional_closure(self):
        config = self.e.configuration()
        self.assertEqual(config["p0"], .5)
        self.assertIn("Conditional", config["closure"])
        self.assertIn("both reputations freeze", config["outside_event"])
        self.assertEqual(config["positive_mass_floor"], 0.)
        self.assertIn("exactly zero", config["zero_mass_convention"])
        with self.assertRaisesRegex(ValueError, "no observations"):
            FixedOutsideConditionalExperiment(n=3, p0=1)


if __name__ == "__main__":
    unittest.main()
