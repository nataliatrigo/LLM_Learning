"""Independent numerical and economic checks for Conditional OE.

Run from the repository root:
    .venv/bin/python -B -m unittest oblivious.oe_conditional_distribution.code.test_model -v

These tests certify finite-grid calculations and the stated conditional-belief
closure. They do not identify an equilibrium of the full-information game or
solve the seller's history-dependent filtering problem.
"""

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_conditional_distribution.code.model import (
    ConditionalExperiment, conditional_from_joint, _ConditionalResponse,
    _independent_conditional_stability,
)


def reference_joint_matrix(grid, demand, alpha, params):
    """Enumerate selection, randomized product, feedback and grid projection."""
    n = len(grid)
    result = np.zeros((n*n, n*n))
    for i in range(n):
        for j in range(n):
            for seller, own, rival, sale in ((0, i, j, demand[i, j]),
                                            (1, j, i, demand[j, i])):
                for product, probability in ((0, 1-alpha[own]), (1, alpha[own])):
                    success = (params.p1, params.p2)[product]
                    for outcome, likelihood in ((0, 1-success), (1, success)):
                        target = params.rho*grid[own] + (1-params.rho)*outcome
                        left = max(0, min(n-2, int(np.searchsorted(grid, target,
                                                                 side="right")-1)))
                        weight = (target-grid[left])/(grid[left+1]-grid[left])
                        for next_own, interpolation in ((left, 1-weight),
                                                        (left+1, weight)):
                            next_index = (next_own*n+rival if seller == 0
                                          else rival*n+next_own)
                            result[i*n+j, next_index] += sale*probability*likelihood*interpolation
    return result


def reference_stationary(matrix):
    """Dense normalized invariant solve, suitable for the small test kernel."""
    linear = matrix.T-np.eye(len(matrix))
    linear[-1] = 1.
    rhs = np.zeros(len(matrix))
    rhs[-1] = 1.
    law = np.linalg.solve(linear, rhs)
    if law.min() < -1e-12:
        raise AssertionError("Reference stationary solution has negative mass")
    return law


class ConditionalBeliefConstruction(unittest.TestCase):
    def setUp(self):
        self.demand = np.array([[.5, .1, .05], [.9, .5, .25], [.95, .75, .5]])
        self.joint = np.array([[.20, .02, .03], [.02, .30, .03], [.03, .03, .34]])

    def test_independent_joint_law_reduces_to_unconditional_belief(self):
        mu_a, mu_b = np.array([.2, .3, .5]), np.array([.4, .5, .1])
        result = conditional_from_joint(np.outer(mu_a, mu_b), self.demand)
        assert_allclose(result["conditional_mu"], np.broadcast_to(mu_b, (3, 3)), atol=2e-15)
        assert_allclose(result["conditional_demand"], self.demand@mu_b, atol=2e-15)
        assert_allclose(result["mu_a"], mu_a, atol=2e-15)
        assert_allclose(result["mu_b"], mu_b, atol=2e-15)

    def test_correlated_calendar_law_is_conditioned_on_own_state(self):
        result = conditional_from_joint(self.joint, self.demand)
        own_mass = np.array([.25, .35, .4])
        expected = self.joint/own_mass[:, None]
        assert_allclose(result["conditional_mu"], expected, atol=2e-15)
        assert_allclose(result["conditional_demand"], (expected*self.demand).sum(axis=1), atol=2e-15)
        unconditional = self.demand@self.joint.sum(axis=0)
        self.assertGreater(np.max(abs(result["conditional_demand"]-unconditional)), .1)
        # Conditioning on a sale would weight the rival law by demand again.
        sale_belief = self.joint*self.demand
        sale_belief /= sale_belief.sum(axis=1)[:, None]
        self.assertGreater(np.max(abs(sale_belief-expected)), .05)

    def test_positive_tiny_rows_are_not_treated_as_off_path(self):
        for tiny in (1e-40, 1e-240, 1e-320):
            with self.subTest(mass=tiny):
                joint = np.array([[tiny, 0., 0.], [0., .4, .1], [0., .1, .4]])
                result = conditional_from_joint(joint, self.demand)
                assert_array_equal(result["conditional_mu"][0], [1., 0., 0.])
                self.assertEqual(result["conditional_demand"][0], .5)
                self.assertEqual(len(result["conditioning_zero_rows"]), 0)

    def test_exact_zero_row_has_explicit_calendar_marginal_convention(self):
        joint = np.array([[0., 0., 0.], [0., .4, .1], [0., .1, .4]])
        result = conditional_from_joint(joint, self.demand)
        assert_array_equal(result["conditioning_zero_rows"], [0])
        assert_allclose(result["conditional_mu"][0], joint.sum(axis=0), atol=0)
        self.assertTrue(result["off_path_convention"])
        for fallback in (np.array([.2, .3, .5]),
                         np.array([[.2, .3, .5], [.1, .2, .7], [.4, .5, .1]])):
            changed = conditional_from_joint(joint, self.demand, off_path=fallback)
            assert_allclose(changed["conditional_mu"][0],
                            fallback if fallback.ndim == 1 else fallback[0], atol=0)
            assert_allclose(changed["conditional_mu"][1:], result["conditional_mu"][1:], atol=0)

    def test_invalid_joint_laws_demands_and_offpath_probabilities_rejected(self):
        invalid_joint = [np.ones((3, 3)), np.zeros((3, 3)), np.ones((2, 3))/6,
                         self.joint + np.diag([np.nan, 0., 0.])]
        negative = self.joint.copy(); negative[0, 0] = -.1
        invalid_joint.append(negative)
        for joint in invalid_joint:
            with self.subTest(joint=joint), self.assertRaises(ValueError):
                conditional_from_joint(joint, self.demand)
        for bad in (np.ones((2, 2))*.5, np.full((3, 3), 1.1),
                    np.full((3, 3), -.1), np.full((3, 3), np.inf)):
            with self.subTest(demand=bad), self.assertRaises(ValueError):
                conditional_from_joint(self.joint, bad)
        zero_row_joint = np.array([[0., 0., 0.], [0., .4, .1], [0., .1, .4]])
        for bad in (np.ones(3), np.array([-.1, .6, .5]), np.full(3, np.nan)):
            with self.subTest(off_path=bad), self.assertRaises(ValueError):
                conditional_from_joint(zero_row_joint, self.demand, off_path=bad)


class ConditionalEconomics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = ConditionalExperiment(n=7, params=Parameters(rho=.6))
        cls.alpha = np.array([.15, .7, .2, .85, .1, .6, .3])
        cls.result = cls.experiment.evaluate(cls.alpha)
        cls.myopic = ConditionalExperiment(n=7, params=Parameters(rho=.6, gamma=0.))
        cls.myopic_result = cls.myopic.evaluate(np.zeros(7))

    def test_joint_calendar_law_matches_independent_enumeration(self):
        exp, result = self.experiment, self.result
        matrix = reference_joint_matrix(exp.grid, exp.game.D, self.alpha, exp.params)
        assert_allclose(matrix.sum(axis=1), 1., atol=3e-15)
        reference = reference_stationary(matrix).reshape(7, 7)
        assert_allclose(result["nu"], reference, atol=2e-11, rtol=0)
        assert_allclose(result["mu_a"], reference.sum(axis=1), atol=2e-11, rtol=0)
        assert_allclose(result["mu_b"], reference.sum(axis=0), atol=2e-11, rtol=0)
        assert_allclose(result["nu"], result["nu"].T, atol=2e-11, rtol=0)
        rows, columns = np.nonzero(matrix > 0)
        self.assertTrue(np.all((rows//7 == columns//7) | (rows % 7 == columns % 7)))
        production = exp.game.transition(np.broadcast_to(self.alpha[:, None], (7, 7)),
                                         np.broadcast_to(self.alpha[None, :], (7, 7)))
        assert_allclose(production.toarray(), matrix, atol=3e-15, rtol=0)

    def test_decision_demand_uses_actual_conditional_law_not_unconditional_marginal(self):
        exp, result = self.experiment, self.result
        expected = result["nu"]/result["mu_a"][:, None]
        expected_demand = (expected*exp.game.D).sum(axis=1)
        assert_allclose(result["conditional_mu"], expected, atol=2e-13, rtol=0)
        assert_allclose(result["conditional_demand"], expected_demand, atol=2e-13, rtol=0)
        assert_allclose(result["demand"], expected_demand, atol=2e-13, rtol=0)
        self.assertGreater(np.max(abs(expected_demand-exp.game.D@result["mu_b"])), .01)
        self.assertAlmostEqual(result["mu_a"]@result["demand"], .5, places=12)
        self.assertLess(result["conditional_diagnostics"]["closure_supported_row_l1_max"], 1e-12)

    def test_mixed_policy_bellman_uses_expected_cost_and_no_sale_self_loop(self):
        exp, result = self.experiment, self.result
        p, grid, value = exp.params, exp.grid, result["value"]
        success = np.interp(p.rho*grid+1-p.rho, grid, value)
        failure = np.interp(p.rho*grid, grid, value)
        q1 = p.R-p.c1+p.gamma*(p.p1*success+(1-p.p1)*failure)
        q2 = p.R-p.c2+p.gamma*(p.p2*success+(1-p.p2)*failure)
        d = result["demand"]
        updated = p.gamma*(1-d)*value+d*((1-self.alpha)*q1+self.alpha*q2)
        assert_allclose(value, updated, atol=2e-11, rtol=0)
        assert_allclose(result["advantage"], q2-q1, atol=2e-12, rtol=0)

    def test_conditional_projection_preserves_actual_stationary_own_marginal(self):
        exp, result = self.experiment, self.result
        matrix = reference_joint_matrix(exp.grid, exp.game.D, self.alpha, exp.params)
        # Average the actual next-state law over the conditional current rival
        # state and sum out the rival's next state. This property holds at the
        # stationary closure, without asserting that own-state paths are Markov.
        own_kernel = np.einsum("ij,ijab->ia", result["conditional_mu"],
                               matrix.reshape(7, 7, 7, 7))
        assert_allclose(own_kernel.sum(axis=1), 1., atol=3e-15)
        assert_allclose(result["mu_a"]@own_kernel, result["mu_a"], atol=2e-12, rtol=0)

    def test_myopic_pure_equilibrium_and_strictly_suboptimal_mixing(self):
        exp, result = self.myopic, self.myopic_result
        self.assertTrue(result["verified"])
        self.assertTrue(result["candidate_verified"])
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["failure_reasons"], [])
        assert_allclose(result["advantage"], -.6, atol=2e-13)
        assert_allclose(result["value"], result["demand"]*.95, atol=2e-13)
        bad = exp.evaluate(np.full(7, .4))
        self.assertFalse(bad["verified"])
        self.assertFalse(bad["candidate_verified"])
        self.assertEqual(bad["status"], "unresolved")
        self.assertTrue(bad["failure_reasons"])
        self.assertLess(bad["conditional_residual"], 1e-12)
        self.assertGreater(bad["dp_diagnostics"]["mixed_indifference_residual"], .5)
        assert_allclose(bad["value"], bad["demand"]*(.95-.6*.4), atol=2e-13)

    def test_full_conditional_distribution_checked_even_if_demands_unchanged(self):
        exp, result = self.myopic, self.myopic_result
        wrong = result["conditional_mu"].copy()
        row = 3
        nodes = np.array([1, 3, 5])
        d = exp.game.D[row, nodes]
        direction = np.array([d[1]-d[2], d[2]-d[0], d[0]-d[1]])
        direction /= max(abs(direction))
        scale = .2*min(wrong[row, nodes]/np.maximum(abs(direction), 1e-15))
        wrong[row, nodes] += scale*direction
        assert_allclose(wrong.sum(axis=1), 1., atol=2e-15)
        assert_allclose((wrong*exp.game.D).sum(axis=1), result["demand"], atol=2e-15)
        self.assertGreater(abs(wrong-result["conditional_mu"]).sum(), 1e-5)
        rejected = exp.evaluate(np.zeros(7), conjectured_mu=wrong)
        self.assertTrue(rejected["dp_diagnostics"]["verified"])
        self.assertFalse(rejected["verified"])
        self.assertGreater(rejected["conditional_diagnostics"]["closure_supported_row_l1_max"], 1e-5)
        assert_allclose(rejected["nu"], result["nu"], atol=2e-11, rtol=0)

    def test_invalid_policies_and_conjectures_rejected(self):
        for bad in (-.01, 1.01, np.nan, np.inf):
            with self.subTest(policy=bad), self.assertRaises(ValueError):
                self.myopic.evaluate(np.full(7, bad))
        for conjecture in (np.ones(7)/7, np.ones((7, 7)), np.full((7, 7), np.nan)):
            with self.subTest(conjecture=conjecture), self.assertRaises(ValueError):
                self.myopic.evaluate(np.zeros(7), conjectured_mu=conjecture)

    def test_myopic_solver_returns_freshly_verified_pure_policy(self):
        result = self.myopic.solve(initial_policy=np.full(7, .5), max_evaluations=300)
        self.assertTrue(result["verified"])
        self.assertEqual(result["status"], "verified")
        assert_allclose(result["policy"], 0., atol=1e-10, rtol=0)
        self.assertLess(result["dp_diagnostics"]["conditional_regret"], 1e-10)
        self.assertLess(result["conditional_diagnostics"]["closure_supported_row_l1_max"], 1e-10)


class ConditionalClosureCertificates(unittest.TestCase):
    def setUp(self):
        self.exp = ConditionalExperiment(n=3, params=Parameters(rho=.6, gamma=0.))
        self.reference_summary = self.exp.evaluate(np.zeros(3))["stationary_diagnostics"]

    def stationary_fixture(self, joint):
        # Isolate closure certification, including off-path branches that do
        # not arise in the full-support small numerical game used above.
        return dict(nu=joint, mu_a=joint.sum(axis=1), mu_b=joint.sum(axis=0),
                    summary=deepcopy(self.reference_summary))

    @staticmethod
    def stable_order_fixture():
        return dict(independent_order_supported_row_l1_max=0.,
                    independent_order_demand_max=0.,
                    independent_order_support_agrees=True,
                    independent_order_joint_l1=0.,
                    independent_order_method="Isolated conditional-closure fixture")

    def test_tiny_supported_row_cannot_hide_behind_weighted_closure_residual(self):
        joint = np.array([[1e-240, 0., 0.], [0., .4, .1], [0., .1, .4]])
        beliefs = conditional_from_joint(joint, self.exp.game.D)["conditional_mu"]
        beliefs[0] = [0., 1., 0.]
        with patch("oblivious.oe_conditional_distribution.code.model.checked_joint_stationary",
                   return_value=self.stationary_fixture(joint)), \
             patch("oblivious.oe_conditional_distribution.code.model._independent_conditional_stability",
                   return_value=self.stable_order_fixture()):
            result = self.exp.evaluate(np.zeros(3), conjectured_mu=beliefs)
        diagnostic = result["conditional_diagnostics"]
        self.assertLess(diagnostic["closure_weighted_l1"], 1e-200)
        self.assertAlmostEqual(diagnostic["closure_supported_row_l1_max"], 2.)
        self.assertEqual(diagnostic["closure_unsupported_row_l1_max"], 0.)
        self.assertTrue(diagnostic["independent_order_verified"])
        self.assertTrue(result["dp_diagnostics"]["verified"])
        self.assertFalse(result["verified"])

    def test_exact_zero_row_has_separate_offpath_convention_residual(self):
        joint = np.array([[0., 0., 0.], [0., .4, .1], [0., .1, .4]])
        beliefs = conditional_from_joint(joint, self.exp.game.D)["conditional_mu"]
        beliefs[0] = [1., 0., 0.]
        with patch("oblivious.oe_conditional_distribution.code.model.checked_joint_stationary",
                   return_value=self.stationary_fixture(joint)), \
             patch("oblivious.oe_conditional_distribution.code.model._independent_conditional_stability",
                   return_value=self.stable_order_fixture()):
            result = self.exp.evaluate(np.zeros(3), conjectured_mu=beliefs)
        diagnostic = result["conditional_diagnostics"]
        self.assertEqual(diagnostic["closure_weighted_l1"], 0.)
        self.assertEqual(diagnostic["closure_supported_row_l1_max"], 0.)
        self.assertAlmostEqual(diagnostic["closure_unsupported_row_l1_max"], 2.)
        self.assertTrue(diagnostic["independent_order_verified"])
        self.assertTrue(result["dp_diagnostics"]["verified"])
        self.assertFalse(result["verified"])

    def test_conditional_instability_rejected_even_with_tiny_global_joint_error(self):
        row_failure = self.stable_order_fixture()
        row_failure.update(independent_order_supported_row_l1_max=.3,
                           independent_order_demand_max=.1,
                           independent_order_joint_l1=1e-240)
        support_failure = self.stable_order_fixture()
        support_failure["independent_order_support_agrees"] = False
        for unstable in (row_failure, support_failure):
            with self.subTest(unstable=unstable), \
                 patch("oblivious.oe_conditional_distribution.code.model._independent_conditional_stability",
                       return_value=unstable):
                result = self.exp.evaluate(np.zeros(3))
            self.assertTrue(result["dp_diagnostics"]["verified"])
            self.assertEqual(result["conditional_residual"], 0.)
            self.assertFalse(result["conditional_diagnostics"]["independent_order_verified"])
            self.assertFalse(result["verified"])
            self.assertEqual(result["status"], "unresolved")

    def test_independent_order_helper_detects_relative_error_in_tiny_row(self):
        tiny = 1e-240
        primary_joint = np.array([[tiny, 0., 0.], [0., .4, .1], [0., .1, .4]])
        alternate_joint = np.array([[tiny/2, tiny/2, 0.],
                                    [tiny/2, .4, .1], [0., .1, .4]])
        primary = conditional_from_joint(primary_joint, self.exp.game.D)
        i, j = np.triu_indices(3)
        alternate_unordered_mass = alternate_joint[i, j]*np.where(i == j, 1, 2)
        actual_kernel = self.exp.game.transition(np.zeros((3, 3)), np.zeros((3, 3)))
        with patch("oblivious.oe_conditional_distribution.code.model._positive_invariant",
                   return_value=alternate_unordered_mass):
            stability = _independent_conditional_stability(self.exp.game, actual_kernel, primary)
        self.assertTrue(stability["independent_order_support_agrees"])
        self.assertLess(stability["independent_order_joint_l1"], 1e-200)
        self.assertAlmostEqual(stability["independent_order_supported_row_l1_max"], 1.)
        self.assertGreater(stability["independent_order_demand_max"], .01)


class ConditionalSearchSensitivities(unittest.TestCase):
    def test_nonmyopic_equilibrium_satisfies_independent_action_maximum(self):
        exp = ConditionalExperiment(n=7, params=Parameters(rho=.6))
        result = exp.solve(max_evaluations=500)
        self.assertTrue(result["verified"])
        self.assertTrue(result["converged"])
        value, alpha, p = result["value"], result["policy"], exp.params
        success = np.interp(p.rho*exp.grid+1-p.rho, exp.grid, value)
        failure = np.interp(p.rho*exp.grid, exp.grid, value)
        q1 = p.R-p.c1+p.gamma*(p.p1*success+(1-p.p1)*failure)
        q2 = p.R-p.c2+p.gamma*(p.p2*success+(1-p.p2)*failure)
        selected_value = (1-alpha)*q1+alpha*q2
        assert_allclose(selected_value, np.maximum(q1, q2), atol=1e-9, rtol=0)
        d = result["demand"]
        assert_allclose(value, p.gamma*(1-d)*value+d*np.maximum(q1, q2),
                        atol=1e-9, rtol=0)
        self.assertGreater(alpha.max(), .9)
        self.assertLess(alpha.min(), .1)

    def test_analytic_policy_jacobian_matches_fresh_conditional_certificates(self):
        exp = ConditionalExperiment(n=5, params=Parameters(rho=.6, gamma=.95))
        alpha = np.array([.15, .3, .55, .7, .8])
        data = _ConditionalResponse(exp).calculate(alpha)
        certified = exp.evaluate(alpha)
        assert_allclose(data["demand"], certified["demand"], atol=2e-12, rtol=0)
        assert_allclose(data["advantage"], certified["advantage"], atol=2e-11, rtol=0)
        # The finite differences recompute the *conditional* joint closure;
        # holding the conditional law or marginal fixed would miss terms.
        step = 2e-6
        numerical = np.empty((5, 5))
        for column in range(5):
            delta = np.eye(5)[column]*step
            plus = exp.evaluate(alpha+delta)["advantage"]
            minus = exp.evaluate(alpha-delta)["advantage"]
            numerical[:, column] = (plus-minus)/(2*step)
        assert_allclose(data["jacobian"], numerical, atol=2e-7, rtol=2e-5)

    def test_solver_does_not_promote_a_failed_final_certificate(self):
        exp = ConditionalExperiment(n=3, params=Parameters(rho=.6, gamma=0.))
        failed = exp.evaluate(np.full(3, .4))
        self.assertFalse(failed["verified"])
        # Even if the exploratory equations succeed, failed independent final
        # certification must keep the returned search result unresolved.
        with patch.object(exp, "evaluate", side_effect=lambda *_a, **_kw: deepcopy(failed)):
            result = exp.solve(initial_policy=np.zeros(3), max_evaluations=80,
                               picard_iterations=0)
        self.assertFalse(result["verified"])
        self.assertFalse(result["converged"])
        self.assertEqual(result["status"], "unresolved")
        self.assertTrue(result["failure_reasons"])


if __name__ == "__main__":
    unittest.main()
