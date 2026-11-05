"""Independent derivative and repeated-zero checks for the root audit."""
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, betaln

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.fluid_oblivious_initial_conditions.code.root_certificate import (
    _isolate_sampled_roots, enumerate_stationary_candidates, stationary_conditions,
)


class RootCertificateTests(unittest.TestCase):
    def params(self,rho=.9):
        return Parameters(R=1.,c1=.05,c2=.65,p1=.35,p2=.8,gamma=.98,rho=rho)

    def test_analytic_diagonal_derivative_matches_five_point_difference(self):
        params = self.params()
        z,h = .57,1e-4
        f = lambda x:stationary_conditions(params,.8,x,order=256)["residual"]
        finite = (f(z-2*h)-8*f(z-h)+8*f(z+h)-f(z+2*h))/(12*h)
        analytic = stationary_conditions(params,.8,z,order=256)["derivative"]
        self.assertAlmostEqual(finite,analytic,places=8)

    def test_repeated_zero_is_found_without_a_sign_change(self):
        grid = np.linspace(.35,.8,129)
        function = lambda z:(z-.5273)**2
        roots = _isolate_sampled_roots(function,grid,function(grid),zero_tolerance=1e-12)
        self.assertEqual(len(roots),1)
        self.assertAlmostEqual(roots[0]["z"],.5273,places=8)

    def test_holding_curvature_holds_the_rival_fixed(self):
        params = self.params()
        z,p0,h = .56796444309534,.8,2e-4
        rival_a,rival_b = 1+z/(1-params.rho),1+(1-z)/(1-params.rho)
        delta = -np.log(params.gamma)
        def holding(x):
            a,b = 1+x/(1-params.rho),1+(1-x)/(1-params.rho)
            demand = quad(lambda t:np.exp((a-1)*np.log(t)+(b-1)*np.log1p(-t)-betaln(a,b))
                          *betainc(rival_a,rival_b,t),p0,1,epsabs=1e-13,epsrel=1e-12)[0]
            reward = params.R-params.c1-(params.c2-params.c1)*(x-params.p1)/(params.p2-params.p1)
            return demand*reward/delta
        finite = (-holding(z+2*h)+16*holding(z+h)-30*holding(z)
                  +16*holding(z-h)-holding(z-2*h))/(12*h*h)
        analytic = stationary_conditions(params,p0,z,order=256)["holding_value_curvature"]
        self.assertAlmostEqual(finite,analytic,places=4)
        self.assertGreater(analytic,100.)

    def test_independent_scan_enumerates_two_roots_and_their_maximum(self):
        audit = enumerate_stationary_candidates(self.params(),.7,scan_points=129,
                                                orders=(64,128),return_samples=False)
        self.assertEqual(len(audit["interior_roots"]),2)
        self.assertEqual(len(audit["critical_points"]),1)
        self.assertAlmostEqual(audit["interior_roots"][0]["z"],.404435378,places=7)
        self.assertAlmostEqual(audit["interior_roots"][1]["z"],.786887316,places=7)
        self.assertGreater(audit["critical_points"][0]["residual"],.5)
        self.assertTrue(audit["root_count_stable"])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            stationary_conditions(self.params(),1.,.5)
        with self.assertRaises(ValueError):
            stationary_conditions(self.params(),.8,1.2)
        with self.assertRaises(ValueError):
            enumerate_stationary_candidates(self.params(),.8,orders=(128,64))


if __name__ == "__main__":
    unittest.main()
