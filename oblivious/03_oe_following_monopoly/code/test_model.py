"""Independent checks for the true-quality scalar consistency experiment.

The economic checks use the original forgetting Bellman equation, explicit
joint-state sums, and stationary flow identities rather than solver internals.
Run with ``python -m unittest oblivious.oe_following_monopoly.code.test_model -v``.
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from oblivious.oe_p0.code.core import (
    ScalarExperiment,
    checked_joint_stationary,
    fixed_demand,
    posterior_mean,
)
from oblivious.oe_p0.code.policy import solve_dp, transition_matrix
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_following_monopoly.code.event_chain import purchase_chain_statistics
from oblivious.oe_following_monopoly.code.model import QualityExperiment
from forgetting.Code.model import BoundaryGrid, Primitives, solve_boundary


class MonopolyEquationChecks(unittest.TestCase):
    def test_reused_dp_matches_original_forgetting_on_the_same_grid(self):
        grid = np.linspace(0, 1, 7)
        for rho in (.6, .95, .999):
            for p0 in (.35, .59, .8):
                with self.subTest(rho=rho, p0=p0):
                    params = Parameters(rho=rho)
                    reference = solve_boundary(
                        Primitives(p0=p0), rho,
                        grid=BoundaryGrid(grid, len(grid), 0, 9.),
                        tolerance=1e-12,
                    )
                    demand = fixed_demand(grid, p0, params)
                    actual = solve_dp(grid, demand, params, tol=1e-10)
                    assert_allclose(demand, reference.demand, atol=2e-15, rtol=0)
                    assert_allclose(actual["value"], reference.value,
                                    atol=8e-10, rtol=0)
                    assert_allclose(actual["advantage"], reference.advantage,
                                    atol=2e-10, rtol=0)
                    assert_array_equal(actual["policy"], reference.advantage >= 0)


class TrueQualityDefinitionChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.purchase = QualityExperiment(
            n=5, params=Parameters(rho=.7), weighting="purchase", order=64)
        cls.calendar = QualityExperiment(
            n=5, params=Parameters(rho=.7), weighting="calendar", order=64)
        cls.policy = np.array([0., .2, .5, .8, 1.])
        # Deliberately asymmetric and nonstationary: this catches seller-axis
        # transposition and accidental replacement of a joint law by marginals.
        cls.nu = np.zeros((5, 5))
        cls.nu[0, 4], cls.nu[4, 0], cls.nu[2, 1] = .6, .1, .3

    def hand_sums(self, experiment):
        p = experiment.params
        calendar_quality = purchased_quality = share = 0.
        for i in range(experiment.n):
            for j in range(experiment.n):
                mass = self.nu[i, j]
                quality = p.p1 + (p.p2-p.p1)*self.policy[j]
                demand_b = experiment.game.selection[1, i, j]
                calendar_quality += mass*quality
                purchased_quality += mass*demand_b*quality
                share += mass*demand_b
        return calendar_quality, purchased_quality/share, share

    def test_purchase_weighted_quality_normalizes_by_own_sales(self):
        expected_calendar, expected_purchase, expected_share = self.hand_sums(self.purchase)
        actual = self.purchase.quality_statistics(self.policy, self.nu)
        self.assertAlmostEqual(actual["calendar_quality_B"], expected_calendar, places=13)
        self.assertAlmostEqual(actual["purchase_quality_B"], expected_purchase, places=13)
        self.assertAlmostEqual(actual["market_share_B"], expected_share, places=13)
        self.assertAlmostEqual(actual["generated_quality"], expected_purchase, places=13)
        # This law is asymmetric; assuming the denominator is 1/2 is wrong.
        self.assertGreater(abs(expected_share-.5), .1)

    def test_calendar_weighting_is_a_distinct_closure(self):
        a = self.purchase.quality_statistics(self.policy, self.nu)
        b = self.calendar.quality_statistics(self.policy, self.nu)
        self.assertGreater(abs(a["generated_quality"]-b["generated_quality"]), .01)
        self.assertAlmostEqual(b["generated_quality"], self.hand_sums(self.calendar)[0], places=13)
        self.assertAlmostEqual(a["purchase_quality_B"], b["purchase_quality_B"], places=13)
        self.assertAlmostEqual(a["calendar_quality_B"], b["calendar_quality_B"], places=13)

    def test_scalar_hook_uses_the_selected_true_quality_definition(self):
        invariant = dict(nu=self.nu, mu_a=self.nu.sum(axis=1), mu_b=self.nu.sum(axis=0))
        for exp in (self.purchase, self.calendar):
            with self.subTest(weighting=exp.weighting):
                expected = exp.quality_statistics(self.policy, self.nu)["generated_quality"]
                self.assertAlmostEqual(exp._generated_scalar(self.policy, invariant), expected, places=13)

    def test_true_quality_bounds_are_product_success_probabilities(self):
        for exp in (self.purchase, self.calendar):
            self.assertEqual(tuple(exp.p0_bounds), (exp.params.p1, exp.params.p2))

    def test_constant_policies_have_exact_true_quality(self):
        for exp in (self.purchase, self.calendar):
            for alpha in (0., .37, 1.):
                with self.subTest(weighting=exp.weighting, alpha=alpha):
                    result = exp.quality_statistics(np.full(exp.n, alpha), self.nu)
                    expected = exp.params.p1 + alpha*(exp.params.p2-exp.params.p1)
                    for key in ("calendar_quality_B", "purchase_quality_B", "generated_quality"):
                        self.assertAlmostEqual(result[key], expected, places=13)

    def test_rejects_unknown_weighting_and_infeasible_policies(self):
        for weighting in ("posterior", "", None):
            with self.subTest(weighting=weighting), self.assertRaises(ValueError):
                QualityExperiment(n=5, params=Parameters(rho=.7), weighting=weighting, order=32)
        for policy in (np.zeros(4), np.zeros((5, 1)), np.full(5, -.01),
                       np.full(5, 1.01), np.full(5, np.nan)):
            with self.subTest(shape=policy.shape), self.assertRaises(ValueError):
                self.purchase.quality_statistics(policy, self.nu)


class QualityCertificateChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiments = [QualityExperiment(
            n=5, params=Parameters(rho=.6, gamma=0), weighting=weighting, order=64)
            for weighting in ("purchase", "calendar")]

    def test_pure_evaluation_returns_quality_instead_of_posterior_mean(self):
        for exp in self.experiments:
            with self.subTest(weighting=exp.weighting):
                result = exp.evaluate(.43)
                assert_array_equal(result["policy"], np.zeros(exp.n))
                self.assertAlmostEqual(result["F"], exp.params.p1, places=12)
                self.assertAlmostEqual(result["G"], exp.params.p1-.43, places=12)
                old_closure = result["mu_b"] @ posterior_mean(exp.grid, exp.params)
                self.assertGreater(abs(result["F"]-old_closure), .001)
                self.assertTrue(result["verified"])

    def test_consistency_does_not_certify_nonindifferent_mixing(self):
        for exp in self.experiments:
            with self.subTest(weighting=exp.weighting):
                alpha = np.full(exp.n, .5)
                true_quality = .5*(exp.params.p1+exp.params.p2)
                result = exp.evaluate_mixed(true_quality, alpha)
                self.assertLess(abs(result["G"]), 1e-12)
                self.assertAlmostEqual(result["F"], true_quality, places=12)
                assert_allclose(result["advantage"], -(exp.params.c2-exp.params.c1),
                                atol=1e-13, rtol=0)
                self.assertGreater(result["dp_diagnostics"]["mixed_indifference_residual"], .1)
                self.assertFalse(result["verified"])
                self.assertFalse(result["candidate_verified"])

    def test_original_p0_experiment_keeps_its_original_closure(self):
        exp = ScalarExperiment(n=5, params=Parameters(rho=.6, gamma=0), order=64)
        result = exp.evaluate(.43)
        expected = result["mu_b"] @ posterior_mean(exp.grid, exp.params)
        self.assertAlmostEqual(result["F"], expected, places=13)
        self.assertGreater(abs(result["F"]-exp.params.p1), .001)
        self.assertNotEqual(tuple(exp.p0_bounds), (exp.params.p1, exp.params.p2))

    def test_purchase_quality_obeys_stationary_reputation_flow(self):
        exp = QualityExperiment(n=7, params=Parameters(rho=.7), weighting="purchase", order=64)
        alpha = np.array([.05, .8, .3, .9, .2, .4, .1])
        a = np.broadcast_to(alpha[:, None], (exp.n, exp.n))
        b = a.T
        stationary = checked_joint_stationary(exp.game, exp.game.transition(a, b))
        nu = stationary["nu"]
        selected_mass_b = nu*exp.game.selection[1]
        share_b = selected_mass_b.sum()
        event_reputation_b = (selected_mass_b*exp.grid[None, :]).sum()/share_b
        statistics = exp.quality_statistics(alpha, nu)
        self.assertAlmostEqual(share_b, .5, places=11)
        # Stationarity and x' = rho*x+(1-rho)*Y imply E[D_B(q_B-x_B)]=0.
        self.assertAlmostEqual(statistics["purchase_quality_B"], event_reputation_b, places=10)

        # At a seller's own sales, the embedded reputation chain is independent
        # of rival demand. Solve this small chain independently by least squares.
        event_kernel = transition_matrix(exp.grid, np.ones(exp.n), alpha, exp.params).toarray()
        lhs = np.vstack((event_kernel.T-np.eye(exp.n), np.ones(exp.n)))
        rhs = np.r_[np.zeros(exp.n), 1.]
        event_law = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
        quality = exp.params.p1 + (exp.params.p2-exp.params.p1)*alpha
        self.assertAlmostEqual(statistics["purchase_quality_B"], event_law @ quality, places=10)

    def test_fast_event_chain_matches_actual_game_even_with_asymmetric_rival(self):
        exp = QualityExperiment(n=5, params=Parameters(rho=.7), weighting="purchase", order=64)
        own_b = np.array([.1, .8, .4, .6, .2])
        for own_a in (own_b, np.zeros(exp.n)):
            with self.subTest(symmetric=np.array_equal(own_a, own_b)):
                a = np.broadcast_to(own_a[:, None], (exp.n, exp.n))
                b = np.broadcast_to(own_b[None, :], (exp.n, exp.n))
                stationary = checked_joint_stationary(exp.game, exp.game.transition(a, b))
                sale_mass = np.sum(stationary["nu"]*exp.game.selection[1], axis=0)
                expected_law = sale_mass/sale_mass.sum()
                fast = purchase_chain_statistics(exp.grid, own_b, exp.params)
                assert_allclose(fast["sale_stationary"], expected_law, atol=2e-10, rtol=0)
                quality = exp.params.p1 + (exp.params.p2-exp.params.p1)*own_b
                self.assertAlmostEqual(fast["generated_quality"], expected_law @ quality, places=10)
                self.assertTrue(fast["stationary_verified"])
                self.assertLess(fast["stationary_residual"], 1e-11)
                self.assertLess(abs(fast["quality_reputation_difference"]), 1e-10)

    def test_event_chain_constant_policy_and_invalid_inputs(self):
        grid, params = np.linspace(0, 1, 7), Parameters(rho=.999)
        for alpha in (0., .37, 1.):
            expected = params.p1+(params.p2-params.p1)*alpha
            result = purchase_chain_statistics(grid, np.full(len(grid), alpha), params)
            self.assertAlmostEqual(result["generated_quality"], expected, places=13)
            self.assertAlmostEqual(result["mean_reputation_at_sale"], expected, places=10)
        for policy in (np.zeros(6), np.full(7, -.01), np.full(7, np.nan)):
            with self.subTest(shape=policy.shape), self.assertRaises(ValueError):
                purchase_chain_statistics(grid, policy, params)


if __name__ == "__main__":
    unittest.main()
