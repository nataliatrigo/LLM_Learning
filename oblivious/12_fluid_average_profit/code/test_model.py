"""Independent checks of the average-profit objective and global searches."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import betainc, betaincc

from oblivious.oe_p0.code.primitives import Parameters, posterior
from oblivious.oe_distribution_fixed_p0.code.model import outside_demand_probability
from oblivious.fluid_average_profit.code.model import (
    AverageProfitModel, PreparedCompetition, find_competitive_equilibria, global_maxima,
)


class GlobalSearchTests(unittest.TestCase):
    def test_distinct_tied_global_maxima_survive(self):
        # There are two separated peaks, with a strict local minimum between.
        result = global_maxima(
            lambda u: -((np.asarray(u) - .2) * (np.asarray(u) - .8)) ** 2,
            points=513, profit_atol=1e-12, profit_rtol=1e-10,
            u_tolerance=1e-6,
        )
        solutions = sorted(row["u"] for row in result["maximizers"])
        self.assertEqual(len(solutions), 2)
        np.testing.assert_allclose(solutions, [.2, .8], atol=2e-7, rtol=0)
        self.assertAlmostEqual(result["max_value"], 0., places=12)

    def test_both_endpoints_are_eligible(self):
        result = global_maxima(
            lambda u: -(np.asarray(u) * (1 - np.asarray(u))) ** 2,
            points=257, profit_atol=1e-12, profit_rtol=1e-10,
            u_tolerance=1e-6,
        )
        self.assertEqual(sorted(row["u"] for row in result["maximizers"]), [0., 1.])


class AverageProfitTests(unittest.TestCase):
    def setUp(self):
        self.params = Parameters(rho=.9)
        self.model = AverageProfitModel(self.params, .5, order=128)

    def test_reputation_margin_and_monopoly_match_definitions(self):
        u = np.array([0., .17, .59, 1.])
        q = self.params.p1 + (self.params.p2 - self.params.p1) * u
        margin = self.params.R - self.params.c1 - (self.params.c2 - self.params.c1) * u
        a, b = posterior(q, self.params)
        np.testing.assert_allclose(self.model.q(u), q, atol=0, rtol=1e-15)
        np.testing.assert_allclose(self.model.margin(u), margin, atol=0, rtol=1e-15)
        np.testing.assert_allclose(self.model.monopoly_demand(u), betaincc(a, b, .5), rtol=2e-13)
        np.testing.assert_allclose(self.model.monopoly_profit(u), betaincc(a, b, .5) * margin, rtol=2e-13)

    def test_discount_does_not_enter_average_objective(self):
        alternative = AverageProfitModel(replace(self.params, gamma=.2), .5, order=128)
        u = np.linspace(0, 1, 17)
        np.testing.assert_array_equal(self.model.monopoly_profit(u), alternative.monopoly_profit(u))
        np.testing.assert_array_equal(self.model.competition_profit(u, .37), alternative.competition_profit(u, .37))

    def test_rho_remains_in_demand(self):
        alternative = AverageProfitModel(replace(self.params, rho=.6), .5, order=128)
        self.assertGreater(abs(float(self.model.monopoly_demand(.1)) - float(alternative.monopoly_demand(.1))), .01)

    def test_competition_matches_independent_scaled_adaptive_quadrature(self):
        for p0, own, rival in [(.1, .12, .87), (.5, .63, .28), (.8, .9, .17), (.9, .02, .81)]:
            with self.subTest(p0=p0, own=own, rival=rival):
                model = AverageProfitModel(self.params, p0, order=128)
                expected = outside_demand_probability(float(model.q(own)), float(model.q(rival)), self.params, p0)
                actual = float(model.competition_demand(own, rival))
                self.assertAlmostEqual(actual / expected, 1., delta=2e-8)
                self.assertAlmostEqual(actual, expected, delta=2e-10)

    def test_near_equal_reputations_have_no_quadrature_jump(self):
        # A direct Legendre integral at these fractional Beta shapes differs
        # from the exact diagonal by ~2e-9 at order 256, enough to create a
        # false profitable deviation. Check both sides independently.
        params = replace(self.params, rho=.6)
        model = AverageProfitModel(params, .8, order=128)
        rival = .964302535305371
        for own in (rival - 1e-8, rival, rival + 1e-8):
            with self.subTest(own=own):
                expected = outside_demand_probability(float(model.q(own)), float(model.q(rival)), params, .8)
                actual = float(model.competition_demand(own, rival))
                self.assertAlmostEqual(actual, expected, delta=2e-12)

    def test_prepared_global_scan_matches_shared_production_demand(self):
        params = replace(self.params, rho=.6)
        model = AverageProfitModel(params, .8, order=128)
        prepared = PreparedCompetition(model, 129)
        for rival in (0., .964302535305371, 1.):
            with self.subTest(rival=rival):
                np.testing.assert_allclose(
                    prepared.profits(rival),
                    model.competition_profit(prepared.grid, rival),
                    atol=2e-12, rtol=0,
                )

    def test_outside_option_event_partition(self):
        own, rival = np.array([.1, .3, .8]), np.array([.9, .6, .2])
        a, b = posterior(self.model.q(own), self.params)
        c, d = posterior(self.model.q(rival), self.params)
        outside = betainc(a, b, .5) * betainc(c, d, .5)
        np.testing.assert_allclose(
            self.model.competition_demand(own, rival)
            + self.model.competition_demand(rival, own) + outside,
            np.ones(3), atol=3e-10, rtol=0,
        )

    def test_symmetric_foc_holds_rival_fixed(self):
        u, h = .48, 2e-5
        own_derivative = (self.model.competition_profit(u + h, u)
                          - self.model.competition_profit(u - h, u)) / (2 * h)
        diagonal_derivative = (self.model.competition_profit(u + h, u + h)
                               - self.model.competition_profit(u - h, u - h)) / (2 * h)
        self.assertAlmostEqual(float(self.model.symmetric_foc(u)), float(own_derivative), delta=3e-7)
        self.assertGreater(abs(float(own_derivative - diagonal_derivative)), .1)

    def test_diagonal_profit_maximum_is_not_a_best_response_equilibrium(self):
        diagonal = global_maxima(
            lambda u: self.model.competition_profit(u, u), points=257,
            profit_atol=1e-12, profit_rtol=1e-10, u_tolerance=1e-6,
        )
        diagonal_u = diagonal["maximizers"][0]["u"]
        unilateral = global_maxima(
            lambda v: self.model.competition_profit(v, diagonal_u), points=257,
            profit_atol=1e-12, profit_rtol=1e-10, u_tolerance=1e-6,
        )
        best_u = unilateral["maximizers"][0]["u"]
        gain = unilateral["max_value"] - float(self.model.competition_profit(diagonal_u, diagonal_u))
        self.assertGreater(best_u - diagonal_u, .2)
        self.assertGreater(gain, .04)


class EquilibriumSearchTests(unittest.TestCase):
    settings = dict(own_points=101, search_points=41, quadrature_order=128,
                    profit_atol=1e-12, profit_rtol=1e-9,
                    u_tolerance=1e-6, optimizer_xatol=1e-12)

    class PreparedPolynomial:
        def __init__(self, model, points):
            self.model, self.grid = model, np.linspace(0., 1., points)

        def profits(self, rival_u):
            return self.model.competition_profit(self.grid, rival_u)

    def test_gain_minimum_is_found_without_a_sign_change_or_foc_seed(self):
        class Game:
            def competition_profit(self, own, rival):
                return -(np.asarray(own) - .413) ** 2

            def symmetric_foc(self, u):
                # Disable the auxiliary candidate generator to exercise the
                # residual-minimization path in isolation.
                return 1.

        with patch("oblivious.fluid_average_profit.code.model.PreparedCompetition", self.PreparedPolynomial):
            result = find_competitive_equilibria(Game(), self.settings)
        self.assertTrue(all(row["deviation_gain"] > 0 for row in result["scan"]))
        self.assertEqual(len(result["equilibria"]), 1)
        equilibrium = result["equilibria"][0]
        self.assertEqual(equilibrium["source"], "refined_gain_minimum")
        self.assertAlmostEqual(equilibrium["u"], .413, delta=2e-7)
        self.assertLessEqual(equilibrium["deviation_gain"], equilibrium["profit_tolerance"])

    def test_interior_foc_local_minimum_is_rejected_by_global_deviation(self):
        class Game:
            def competition_profit(self, own, rival):
                return (np.asarray(own) - .4) ** 2

            def symmetric_foc(self, u):
                return 2 * (u - .4)

        with patch("oblivious.fluid_average_profit.code.model.PreparedCompetition", self.PreparedPolynomial):
            result = find_competitive_equilibria(Game(), self.settings)
        rejected = [row for row in result["candidates"] if abs(row["u"] - .4) < 1e-6]
        self.assertEqual(len(rejected), 1)
        self.assertFalse(rejected[0]["verified"])
        self.assertAlmostEqual(rejected[0]["deviation_gain"], .36, places=10)
        self.assertEqual([row["u"] for row in result["equilibria"]], [1.])


if __name__ == "__main__":
    unittest.main()
