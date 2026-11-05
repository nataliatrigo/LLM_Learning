"""Checks for the fluid monopoly benchmark and denominator accounting."""
import unittest

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.fluid_oblivious.code.monopoly import solve_monopoly,stationary_states,simulate_monopoly


class MonopolyTests(unittest.TestCase):
    def test_low_outside_option_retains_product1_stationary_state(self):
        params=Parameters(rho=.7)
        response=solve_monopoly(params,.1,n=1001)
        states,_=stationary_states(params,.1,response)
        self.assertEqual(len(states),1)
        self.assertEqual(states[0]["kind"],"pure_product1")
        self.assertAlmostEqual(states[0]["x_star"],params.p1)
        self.assertEqual(states[0]["alpha"],0.)

    def test_mixed_holding_matches_existing_monopoly_threshold(self):
        params=Parameters(rho=.7)
        response=solve_monopoly(params,.5,n=4001)
        states,_=stationary_states(params,.5,response)
        self.assertEqual(len(states),1)
        state=states[0]
        self.assertEqual(state["kind"],"mixed")
        self.assertAlmostEqual(state["x_star"],.5547361,delta=2e-5)
        self.assertLess(abs(state["diagnostics"]["conditional_holding_defect"]),2e-5)
        path=simulate_monopoly(params,.5,response,times=[0.,1.,10.],initial=state["x_star"])
        np.testing.assert_allclose(path["by_time"].x_a,state["x_star"],atol=1e-12)
        np.testing.assert_allclose(path["by_time"].alpha_a,state["alpha"],atol=1e-12)

    def test_one_seller_customer_flow_is_not_doubled(self):
        params=Parameters(rho=.7)
        response=solve_monopoly(params,.5,n=1001)
        result=simulate_monopoly(params,.5,response,times=np.arange(101.),bin_width=25)
        frame,bins=result["by_time"],result["binned"]
        np.testing.assert_allclose(frame.customer_product2_rate,frame.own_product2_rate,atol=1e-14)
        np.testing.assert_allclose(bins.customer_product2_rate,bins.own_product2_rate,atol=1e-14)
        np.testing.assert_allclose(frame.b_sale_rate,0.,atol=1e-14)
        np.testing.assert_allclose(frame.own_sale_rate+frame.outside_rate,1.,atol=1e-14)
        np.testing.assert_allclose(bins.product2_given_sale,
            np.diff(frame.cumulative_own_product2.iloc[::25])/np.diff(frame.cumulative_own_sales.iloc[::25]),atol=1e-10)

    def test_multiple_optimal_stationary_states_are_all_retained(self):
        params=Parameters(rho=.7)
        response=solve_monopoly(params,.9,n=4001)
        states,rejected=stationary_states(params,.9,response)
        self.assertEqual([state["kind"] for state in states],
                         ["pure_product1","pure_product2"])
        np.testing.assert_allclose([state["x_star"] for state in states],
                                   [params.p1,params.p2],atol=1e-12)
        self.assertTrue(any(state["kind"]=="mixed" for state in rejected))


if __name__=="__main__":
    unittest.main()
