"""Independent demand checks and stationary-equilibrium regression cases."""
import unittest

import numpy as np

from oblivious.fluid_oblivious.code.model import (
    FixedRivalDemand, Parameters, audit_response, demand_at_states,
    find_equilibria, outside_demand_probability, solve_response,
    symmetric_demand_derivative,
)
from oblivious.oe_distribution_fixed_p0.code.model import outside_demand_matrix


class DemandTests(unittest.TestCase):
    def test_prepared_demand_matches_original_matrix(self):
        grid=np.linspace(0,1,11)
        for rho,p0 in [(.7,.5),(.99,.9)]:
            p=Parameters(rho=rho)
            original,_=outside_demand_matrix(grid,p,p0,128)
            prepared=FixedRivalDemand(p,p0,grid,128)
            for column in [0,3,5,8,10]:
                np.testing.assert_allclose(prepared(grid[column]),original[:,column],
                                           rtol=2e-10,atol=2e-12)

    def test_actual_selection_partition_symmetry_and_adaptive_audit(self):
        for rho,p0,x,y in [(.7,.5,.35,.8),(.95,.7,.5,.65),(.99,.9,.35,.8)]:
            p=Parameters(rho=rho)
            da,db,out=demand_at_states(x,y,p,p0)
            ba,bb,out2=demand_at_states(y,x,p,p0)
            self.assertAlmostEqual(da+db+out,1.,places=14)
            self.assertAlmostEqual(da,bb,places=14)
            self.assertAlmostEqual(db,ba,places=14)
            expected=outside_demand_probability(x,y,p,p0)
            self.assertLess(abs(da-expected),2e-10)
            self.assertLess(abs(da-expected)/max(expected,1e-300),2e-7)
            self.assertGreater(da,0.)

    def test_own_derivative_holds_rival_fixed(self):
        for rho,p0,z in [(.7,.5,.55),(.99,.8,.65)]:
            p=Parameters(rho=rho)
            _,derivative=symmetric_demand_derivative(p,p0,z,256)
            step=2e-6
            finite=(outside_demand_probability(z+step,z,p,p0)
                    -outside_demand_probability(z-step,z,p,p0))/(2*step)
            self.assertAlmostEqual(derivative,finite,delta=2e-7)


class EquilibriumTests(unittest.TestCase):
    def test_constant_demand_response_has_exact_value(self):
        p=Parameters(rho=.8)
        n=201;demand=.37
        response=solve_response(p,.5,.6,n=n,prepared=lambda _:np.full(n,demand))
        exact=demand*(p.R-p.c1)/(-np.log(p.gamma))
        np.testing.assert_allclose(response["value"],exact,rtol=2e-12,atol=2e-12)
        self.assertTrue(np.all(response["policy"]==0))

    def test_mixed_equilibrium_satisfies_full_response_consistency(self):
        p=Parameters(rho=.7)
        result=find_equilibria(p,.5,n=1001)
        self.assertEqual(len(result["equilibria"]),1)
        eq=result["equilibria"][0]
        self.assertEqual(eq["kind"],"mixed")
        self.assertTrue(0<eq["alpha"]<1)
        self.assertAlmostEqual(eq["x_star"],p.p1+(p.p2-p.p1)*eq["alpha"],places=13)
        self.assertLess(eq["diagnostics"]["consistency_error"],5e-8)
        self.assertLess(abs(eq["diagnostics"]["conditional_holding_defect"]),2e-5)
        self.assertTrue(audit_response(p,.5,eq["response"])["passed"])

    def test_multiple_pure_equilibria_are_retained(self):
        p=Parameters(rho=.7)
        result=find_equilibria(p,.9,n=1001)
        self.assertEqual([eq["kind"] for eq in result["equilibria"]],
                         ["pure_product1","pure_product2"])
        self.assertTrue(any(eq["kind"]=="mixed" for eq in result["search"]["rejected_candidates"]))

    def test_no_candidate_is_not_reported_as_nonexistence_proof(self):
        result=find_equilibria(Parameters(rho=.99),.1,n=1001)
        self.assertEqual(result["equilibria"],[])
        self.assertEqual(result["search"]["status"],"no_equilibrium_found")
        self.assertIn("not a nonexistence proof",result["search"]["scope"])


if __name__=="__main__":
    unittest.main()
