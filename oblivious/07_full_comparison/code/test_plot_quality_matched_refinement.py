"""Check that the matching quality conditions on the rival's own sales."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SPEC = importlib.util.spec_from_file_location(
    "plot_quality_matched_refinement_test",
    Path(__file__).with_name("plot_quality_matched_refinement.py"),
)
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)

PARAMETERS = dict(R=1., c1=.05, c2=.65, p1=.35, p2=.8, gamma=.98, rho=.9)


class RivalSalesQualityTests(unittest.TestCase):
    def test_correlated_joint_law_weights_rival_columns_by_rival_demand(self):
        candidate = dict(
            grid=np.array([0., .5, 1.]),
            policy=np.array([0., .25, 1.]),
            nu=np.array([[.20, .03, .02],
                         [.03, .30, .07],
                         [.02, .07, .26]]),
        )
        demand = np.array([[.5, .2, .1], [.8, .5, .3], [.9, .7, .5]])
        with patch.object(PLOT, "demand_matrix", return_value=(demand, {})):
            result = PLOT.rival_sales_quality(candidate, PARAMETERS, order=32)
        # Hand sums of columns of nu*D.T give rival B's sales masses:
        # .100+.006+.002=.108; .024+.150+.021=.195;
        # .018+.049+.130=.197. The three masses total .500.
        # Product 2 mass is .195*.25+.197=.24575.
        self.assertAlmostEqual(result["rival_sale_rate"], .5, places=12)
        self.assertAlmostEqual(result["rival_product2_conditional"], .4915, places=12)
        self.assertAlmostEqual(result["rival_success_probability"], .571175, places=12)
        # Calendar marginal weights (.25,.40,.35) instead give .4500;
        # using row sums after D.T would give .4085. Both are wrong here.
        self.assertNotAlmostEqual(result["rival_product2_conditional"], .45, places=6)
        self.assertNotAlmostEqual(result["rival_product2_conditional"], .4085, places=6)

    def test_equal_selection_reduces_to_sale_weighted_product_success(self):
        marginal = np.array([.2, .5, .3])
        candidate = dict(
            grid=np.array([0., .5, 1.]),
            policy=np.array([.1, .5, .9]),
            nu=np.outer(marginal, marginal),
        )
        parameters = dict(PARAMETERS, p1=.2, p2=.8)
        with patch.object(PLOT, "demand_matrix", return_value=(np.full((3, 3), .5), {})):
            result = PLOT.rival_sales_quality(candidate, parameters, order=32)
        # Equal sale probabilities leave rival reputation weights unchanged.
        # Among rival sales, product 2 mass is .2*.1+.5*.5+.3*.9=.54.
        # Thus 46% have success .2 and 54% have success .8: .524 overall.
        self.assertAlmostEqual(result["rival_sale_rate"], .5, places=12)
        self.assertAlmostEqual(result["rival_product2_conditional"], .54, places=12)
        self.assertAlmostEqual(result["rival_success_probability"], .524, places=12)


if __name__ == "__main__":
    unittest.main()
