"""Analytical payoff and probability checks for independent departures."""
import math
import unittest

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code.model import outside_demand_probability
from oblivious.fluid_oblivious_initial_conditions.code.independent_response import (
    FixedRivalLegendreDemand, constant_action_deviation, excursion_then_hold_deviation,
    stationary_holding_value,
)


class IndependentResponseTests(unittest.TestCase):
    def test_direct_demand_agrees_with_adaptive_integral(self):
        for rho,p0,z in [(.9,.7,.6),(.9,.8,.568),(.7,.9,.45),(.8,.9,.65)]:
            params=Parameters(rho=rho)
            engine=FixedRivalLegendreDemand(params,p0,z,512)
            for x in (params.p1,z,params.p2):
                expected=outside_demand_probability(x,z,params,p0)
                self.assertAlmostEqual(engine(x),expected,delta=2e-11)

    def test_constant_demand_has_exact_discounted_payoff_and_path(self):
        params=Parameters(rho=.9)
        d,z,horizon=.37,.55,20.
        for product in (1,2):
            result=constant_action_deviation(params,.8,z,product,horizon=horizon,
                                             demand_function=lambda _:d)
            reward=params.R-(params.c1 if product==1 else params.c2)
            delta,lam=-math.log(params.gamma),-math.log(params.rho)
            exact=d*reward*(-math.expm1(-delta*horizon))/delta
            self.assertAlmostEqual(result["finite_horizon_value"],exact,delta=2e-10)
            q=params.p1 if product==1 else params.p2
            state=q+(z-q)*math.exp(-lam*d*horizon)
            self.assertAlmostEqual(result["terminal_reputation"],state,delta=2e-11)
            full=d*reward/delta
            lo,hi=result["infinite_value_interval_from_truncation"]
            self.assertLessEqual(lo-1e-10,full)
            self.assertGreaterEqual(hi+1e-10,full)

    def test_exact_diagonal_holding_value_matches_direct_quadrature(self):
        params=Parameters(rho=.9)
        for z in (params.p1,.5679644430953444,params.p2):
            holding=stationary_holding_value(params,.8,z)
            demand=FixedRivalLegendreDemand(params,.8,z,256)(z)
            self.assertAlmostEqual(holding["demand"],demand,delta=2e-13)
            self.assertAlmostEqual(holding["value"],demand*holding["reward"]/-math.log(params.gamma),delta=2e-12)

    def test_finite_payoff_rejects_a_suboptimal_mix_under_constant_demand(self):
        params=Parameters(rho=.9)
        z,d=.55,.4
        result=constant_action_deviation(params,.8,z,1,horizon=200.,
                                         demand_function=lambda _:d)
        alpha=(z-params.p1)/(params.p2-params.p1)
        holding=d*(params.R-params.c1-(params.c2-params.c1)*alpha)/-math.log(params.gamma)
        self.assertGreater(result["finite_horizon_value"]-holding,4.)

    def test_excursion_and_holding_have_exact_payoff_under_constant_demand(self):
        params=Parameters(rho=.9)
        z,target,d=.55,.65,.4
        result=excursion_then_hold_deviation(params,.8,z,target,demand_function=lambda _:d)
        delta,lam=-math.log(params.gamma),-math.log(params.rho)
        time=math.log((params.p2-z)/(params.p2-target))/(lam*d)
        alpha=(target-params.p1)/(params.p2-params.p1)
        terminal=d*(params.R-params.c1-(params.c2-params.c1)*alpha)/delta
        exact=d*(params.R-params.c2)*(-math.expm1(-delta*time))/delta+math.exp(-delta*time)*terminal
        self.assertTrue(result["target_reached"])
        self.assertAlmostEqual(result["travel_time"],time,delta=5e-9)
        self.assertAlmostEqual(result["value"],exact,delta=2e-10)

    def test_unreached_target_still_has_feasible_holding_continuation(self):
        params=Parameters(rho=.9)
        result=excursion_then_hold_deviation(params,.8,.55,.8,horizon=.1,demand_function=lambda _:.4)
        self.assertFalse(result["target_reached"])
        self.assertTrue(params.p1<result["reached_state"]<params.p2)
        self.assertGreater(result["continuation_value"],0.)

    def test_rejects_invalid_inputs(self):
        params=Parameters(rho=.9)
        with self.assertRaises(ValueError):
            constant_action_deviation(params,.8,.55,3)
        with self.assertRaises(ValueError):
            stationary_holding_value(params,.8,.2)
        with self.assertRaises(ValueError):
            FixedRivalLegendreDemand(params,.8,.5,8)


if __name__=="__main__":
    unittest.main()
