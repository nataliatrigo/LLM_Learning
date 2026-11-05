"""Independent finite-grid checks for the new fixed-outside-option model.

Run: .venv/bin/python -B -m unittest oblivious.oe_distribution_fixed_p0.code.test_model -v
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import beta

from oblivious.oe_distribution_fixed_p0.code.model import (
    FixedOutsideMarginalExperiment, OutsideOptionGame, _OutsideResponse,
    outside_demand_matrix, outside_demand_probability,
)
from oblivious.oe_p0.code.primitives import Parameters, demand_matrix, posterior


def reference_events(grid, demand, outside, sigma_a, sigma_b, params):
    """Enumerate outside, selected seller, product, outcome, and interpolation.

    This dense reference does not call production transition or interpolation
    methods. Both policies are indexed in the global (A,B) axes.
    """
    n = len(grid)
    matrix = np.zeros((n*n, n*n))
    rewards = np.zeros((2, n*n))
    for i in range(n):
        for j in range(n):
            row = i*n+j
            matrix[row, row] += outside[i, j]
            for seller, own, rival, selected, alpha in (
                    (0, i, j, demand[i, j], sigma_a[i, j]),
                    (1, j, i, demand[j, i], sigma_b[i, j])):
                for product, probability in ((0, 1-alpha), (1, alpha)):
                    success = (params.p1, params.p2)[product]
                    cost = (params.c1, params.c2)[product]
                    rewards[seller, row] += selected*probability*(params.R-cost)
                    for outcome, chance in ((0, 1-success), (1, success)):
                        target = params.rho*grid[own]+(1-params.rho)*outcome
                        left = int(np.clip(np.searchsorted(grid, target, side="right")-1, 0, n-2))
                        right_weight = (target-grid[left])/(grid[left+1]-grid[left])
                        for new_own, weight in ((left, 1-right_weight), (left+1, right_weight)):
                            col = new_own*n+rival if seller == 0 else rival*n+new_own
                            matrix[row, col] += selected*probability*chance*weight
    return matrix, rewards


def reference_stationary(matrix):
    linear = matrix.T-np.eye(len(matrix))
    linear[-1] = 1
    rhs = np.zeros(len(matrix)); rhs[-1] = 1
    return np.linalg.solve(linear, rhs)


class OutsideDemand(unittest.TestCase):
    def test_event_partition_and_monopoly_bound_across_concentrations(self):
        grid = np.linspace(0, 1, 11)
        for rho in (.6, .95, .99, .999):
            with self.subTest(rho=rho):
                p = Parameters(rho=rho)
                demand, diagnostics = outside_demand_matrix(grid, p, .5)
                a, b = posterior(grid, p)
                f = beta.cdf(.5, a, b)
                s = beta.sf(.5, a, b)
                assert_allclose(demand+demand.T+np.outer(f, f), 1, atol=3e-15, rtol=0)
                assert_allclose(demand.diagonal(), s*(1-.5*s), atol=0, rtol=4e-15)
                self.assertTrue(np.all(demand <= s[:, None]+2e-15))
                old, _ = demand_matrix(grid, p, 128)
                self.assertTrue(np.all(demand <= old+1e-9))
                self.assertAlmostEqual(demand[5, 5], .375, places=13)
                self.assertLess(diagnostics["order_doubling_max_abs"], 2e-9)
                self.assertLess(diagnostics["raw_relative_selected_mass_error"], 2e-7)

    def test_independent_scaled_quad_checks_common_and_rare_sales(self):
        grid = np.linspace(0, 1, 5)
        for rho in (.6, .95, .99, .999):
            p = Parameters(rho=rho)
            demand, _ = outside_demand_matrix(grid, p, .5)
            for i, j in ((0, 0), (0, 2), (2, 2), (3, 4), (4, 0), (4, 4)):
                with self.subTest(rho=rho, i=i, j=j):
                    direct = outside_demand_probability(grid[i], grid[j], p, .5)
                    assert_allclose(demand[i, j], direct, atol=1e-316, rtol=3e-8)

    def test_zero_threshold_exactly_recovers_existing_two_seller_game(self):
        grid = np.linspace(0, 1, 7)
        p = Parameters(rho=.8)
        original, _ = demand_matrix(grid, p, 128)
        outside, diagnostics = outside_demand_matrix(grid, p, 0, 128)
        assert_array_equal(outside, original)
        self.assertEqual(diagnostics["p0"], 0)

    def test_high_rho_rare_diagonal_is_preserved_and_underflow_disclosed(self):
        grid = np.linspace(0, 1, 21)
        p = Parameters(rho=.999)
        demand, diagnostics = outside_demand_matrix(grid, p, .5)
        self.assertGreater(demand[0, 0], 0.)
        a, b = posterior(0., p)
        survival = beta.sf(.5, a, b)
        assert_allclose(demand[0, 0], survival, atol=0, rtol=3e-15)
        self.assertGreater(diagnostics["positive_tail_underflow_entries"], 0)
        self.assertEqual(demand[0, -1], 0.)
        self.assertLess(diagnostics["order_doubling_max_relative_tail"], 2e-6)

    def test_invalid_threshold_order_and_no_observation_comparison(self):
        for p0 in (-.1, 1.1, np.nan, np.inf):
            with self.subTest(p0=p0), self.assertRaises(ValueError):
                outside_demand_matrix(np.linspace(0, 1, 3), Parameters(), p0)
        for order in (7, 8.5):
            with self.subTest(order=order), self.assertRaises(ValueError):
                outside_demand_matrix(np.linspace(0, 1, 3), Parameters(), order=order)
        demand, _ = outside_demand_matrix(np.linspace(0, 1, 3), Parameters(), 1)
        assert_array_equal(demand, np.zeros((3, 3)))
        with self.assertRaisesRegex(ValueError, "no observations"):
            FixedOutsideMarginalExperiment(n=3, p0=1)


class OutsideEconomics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = FixedOutsideMarginalExperiment(n=7, params=Parameters(rho=.6))
        cls.alpha = np.array([.15, .7, .2, .85, .1, .6, .3])
        cls.result = cls.experiment.evaluate(cls.alpha)

    def test_outside_hold_joint_chain_values_and_marginal_match_dense_reference(self):
        e, r = self.experiment, self.result
        sa = np.broadcast_to(self.alpha[:, None], (7, 7))
        sb = sa.T
        matrix, rewards = reference_events(e.grid, e.game.D, e.game.outside, sa, sb, e.params)
        assert_allclose(e.game.transition(sa, sb).toarray(), matrix, atol=8e-16, rtol=0)
        assert_allclose(matrix.sum(axis=1), 1, atol=2e-15, rtol=0)
        nu = reference_stationary(matrix).reshape(7, 7)
        assert_allclose(r["nu"], nu, atol=3e-11, rtol=0)
        assert_allclose(r["mu"], nu.sum(axis=0), atol=3e-11, rtol=0)
        values = np.linalg.solve(np.eye(49)-e.params.gamma*matrix, rewards.T).T.reshape(2, 7, 7)
        checked = e.game.evaluate(sa, sb)
        assert_allclose(checked["values"], values, atol=2e-9, rtol=0)
        assert_allclose(e.game.rewards(sa, sb).reshape(2, 49), rewards, atol=2e-16, rtol=0)
        # The no-sale event is added to precisely the original joint state.
        without_outside = matrix-np.diag(e.game.outside.ravel())
        assert_allclose(without_outside.sum(axis=1), (e.game.D+e.game.D.T).ravel(), atol=3e-16)

    def test_oblivious_value_includes_all_unselected_calendar_periods(self):
        e, r = self.experiment, self.result
        p, value = e.params, r["value"]
        success = np.interp(p.rho*e.grid+1-p.rho, e.grid, value)
        failure = np.interp(p.rho*e.grid, e.grid, value)
        q1 = p.R-p.c1+p.gamma*(p.p1*success+(1-p.p1)*failure)
        q2 = p.R-p.c2+p.gamma*(p.p2*success+(1-p.p2)*failure)
        updated = p.gamma*(1-r["demand"])*value+r["demand"]*((1-self.alpha)*q1+self.alpha*q2)
        assert_allclose(value, updated, atol=2e-11, rtol=0)
        assert_allclose(r["advantage"], q2-q1, atol=2e-12, rtol=0)
        self.assertEqual(e.configuration()["p0"], .5)
        self.assertIn("External", e.configuration()["p0_role"])

    def test_full_state_best_response_includes_outside_continuation(self):
        p = Parameters(rho=.7, gamma=.9)
        grid = np.linspace(0, 1, 3)
        game = OutsideOptionGame(grid, p, .5)
        opponent = np.array([[.1, .4, .8], [.3, .7, .2], [.5, .9, .6]])
        candidates = [reference_events(grid, game.D, game.outside, np.full((3, 3), k), opponent, p)
                      for k in (0, 1)]
        value = np.zeros(9)
        for _ in range(1000):
            q = np.array([rewards[0]+p.gamma*(matrix@value) for matrix, rewards in candidates])
            updated = q.max(axis=0)
            if np.max(abs(updated-value)) < 1e-12:
                value = updated
                break
            value = updated
        else:
            self.fail("Independent dense best response did not converge")
        response = game.best_response(opponent)
        assert_allclose(response["value"].ravel(), value, atol=2e-9, rtol=0)
        assert_array_equal(response["sigma"].ravel(), q[1] >= q[0])
        response_b = game.best_response(opponent.T, seller=1)
        assert_allclose(response_b["value"], response["value"].T, atol=0, rtol=0)

    def test_myopic_solver_has_analytical_cheap_product_policy(self):
        e = FixedOutsideMarginalExperiment(n=7, params=Parameters(rho=.6, gamma=0))
        r = e.solve(initial_policy=np.full(7, .4), max_evaluations=300)
        self.assertTrue(r["verified"], r["summary"])
        assert_array_equal(r["policy"], np.zeros(7))
        assert_allclose(r["value"], r["demand"]*(e.params.R-e.params.c1), atol=2e-13)
        assert_allclose(r["advantage"], e.params.c1-e.params.c2, atol=2e-13)
        self.assertLess(r["marginal_residual"], 1e-10)
        rejected = e.evaluate(np.full(7, .4))
        self.assertFalse(rejected["verified"])

    def test_policy_sensitivities_match_finite_differences(self):
        for rho in (.6, .95):
            e = FixedOutsideMarginalExperiment(n=7, params=Parameters(rho=rho))
            alpha = np.linspace(.2, .8, 7)
            engine = _OutsideResponse(e)
            analytical = engine.calculate(alpha)["jacobian"].copy()
            for k in (0, 3, 6):
                with self.subTest(rho=rho, k=k):
                    delta = np.zeros(7); delta[k] = 1e-5
                    plus = engine.calculate(alpha+delta)["advantage"].copy()
                    minus = engine.calculate(alpha-delta)["advantage"].copy()
                    assert_allclose(analytical[:, k], (plus-minus)/(2e-5), atol=2e-8, rtol=5e-5)

    def test_high_rho_sensitivities_do_not_square_tiny_exit_rates(self):
        e = FixedOutsideMarginalExperiment(n=7, params=Parameters(rho=.999))
        with np.errstate(divide="raise", invalid="raise", over="raise"):
            response = _OutsideResponse(e).calculate(np.linspace(.2, .8, 7))
        self.assertTrue(np.isfinite(response["jacobian"]).all())
        self.assertTrue(np.isfinite(response["mu"]).all())
        self.assertGreater(response["mu"][0], .999)
        self.assertLess(response["stationary_residual"], 1e-11)


if __name__ == "__main__":
    unittest.main()
