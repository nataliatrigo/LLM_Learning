"""Independent timing, stationarity, and Bellman checks for monopoly curves."""

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.stats import beta


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forgetting.Code.model import BoundaryGrid, Primitives, solve_boundary


SPEC = importlib.util.spec_from_file_location(
    "plot_monopoly_p0_refinement_test",
    Path(__file__).with_name("plot_monopoly_p0_refinement.py"),
)
PLOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLOT)

PARAMETERS = dict(R=1., c1=.05, c2=.65, p1=.35, p2=.8, gamma=.98)


def independent_kernels(case, rho, parameters):
    """Reconstruct interpolation and calendar waiting without model helpers."""
    x = np.asarray(case["grid"])
    alpha = np.asarray(case["policy"])
    demand = np.asarray(case["demand"])
    identity = np.eye(len(x))
    success = np.column_stack([
        np.interp(rho*x + 1-rho, x, basis) for basis in identity
    ])
    failure = np.column_stack([
        np.interp(rho*x, x, basis) for basis in identity
    ])
    quality = parameters["p1"] + (parameters["p2"]-parameters["p1"])*alpha
    event = quality[:, None]*success + (1-quality[:, None])*failure
    calendar = (1-demand[:, None])*identity + demand[:, None]*event
    return event, calendar


class MonopolyUsageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rho = .9
        cls.p0 = .5
        cls.case = PLOT.monopoly_case(cls.rho, cls.p0, 51, PARAMETERS)

    def test_myopic_seller_uses_only_the_cheaper_product(self):
        parameters = dict(PARAMETERS, gamma=0.)
        for p0 in (.25, .5, .75):
            with self.subTest(p0=p0):
                case = PLOT.monopoly_case(.8, p0, 31, parameters)
                x = np.asarray(case["grid"])
                demand = beta.sf(p0, 1+x/.2, 1+(1-x)/.2)
                np.testing.assert_array_equal(case["policy"], np.zeros(len(x)))
                np.testing.assert_allclose(case["demand"], demand, atol=1e-14)
                np.testing.assert_allclose(
                    case["value"], demand*(parameters["R"]-parameters["c1"]),
                    atol=1e-10,
                )
                self.assertEqual(case["outcomes"]["product2_conditional"], 0.)
                self.assertEqual(case["outcomes"]["product2_calendar"], 0.)
                self.assertTrue(case["diagnostics"]["verified"])

    def test_usage_conditions_on_sales_and_differs_from_calendar_usage(self):
        case = self.case
        eta = np.asarray(case["sale_stationary"])
        mu = np.asarray(case["calendar_stationary"])
        demand = np.asarray(case["demand"])
        alpha = np.asarray(case["policy"])
        sale_rate = float(mu @ demand)
        calendar_usage = float((mu*demand) @ alpha)
        conditional_usage = float(eta @ alpha)
        self.assertTrue(np.any(alpha > .5))
        self.assertGreater(conditional_usage, .001)
        self.assertGreater(sale_rate, 0.)
        self.assertLess(sale_rate, 1.)
        self.assertAlmostEqual(conditional_usage, calendar_usage/sale_rate, places=9)
        np.testing.assert_allclose(eta, mu*demand/sale_rate, atol=1e-10, rtol=1e-8)
        self.assertAlmostEqual(case["outcomes"]["sale_rate"], sale_rate, places=10)
        self.assertAlmostEqual(
            case["outcomes"]["product2_conditional"], conditional_usage, places=10,
        )
        self.assertAlmostEqual(
            case["outcomes"]["product2_calendar"], calendar_usage, places=10,
        )
        self.assertGreater(conditional_usage-calendar_usage, .001)

    def test_both_stationary_laws_match_independently_constructed_kernels(self):
        event, calendar = independent_kernels(self.case, self.rho, PARAMETERS)
        for key, kernel in (("sale_stationary", event),
                            ("calendar_stationary", calendar)):
            with self.subTest(law=key):
                law = np.asarray(self.case[key])
                self.assertTrue(np.all(law >= 0))
                self.assertTrue(np.all(kernel >= 0))
                self.assertAlmostEqual(float(law.sum()), 1., places=12)
                np.testing.assert_allclose(kernel.sum(axis=1), 1., atol=1e-14)
                np.testing.assert_allclose(law @ kernel, law, atol=1e-10, rtol=0)
        self.assertTrue(self.case["diagnostics"]["verified"])

    def test_value_matches_original_solver_and_independent_policy_evaluation(self):
        case = self.case
        x = np.asarray(case["grid"])
        demand = np.asarray(case["demand"])
        alpha = np.asarray(case["policy"])
        _, calendar = independent_kernels(case, self.rho, PARAMETERS)
        reward = demand*(PARAMETERS["R"]-PARAMETERS["c1"]
                         -(PARAMETERS["c2"]-PARAMETERS["c1"])*alpha)
        evaluated = np.linalg.solve(np.eye(len(x))-PARAMETERS["gamma"]*calendar, reward)
        primitives = Primitives(
            p0=self.p0, revenue=PARAMETERS["R"],
            **{key: PARAMETERS[key] for key in ("c1", "c2", "p1", "p2", "gamma")},
        )
        original = solve_boundary(
            primitives, self.rho, grid=BoundaryGrid(x, len(x), 0, 9.), tolerance=1e-11,
        )
        np.testing.assert_allclose(case["value"], original.value, atol=1e-7, rtol=0)
        np.testing.assert_allclose(case["value"], evaluated, atol=1e-7, rtol=0)
        # This maximization verifies optimality independently of the chosen policy.
        v_success = np.interp(self.rho*x+1-self.rho, x, evaluated)
        v_failure = np.interp(self.rho*x, x, evaluated)
        q1 = PARAMETERS["R"]-PARAMETERS["c1"]+PARAMETERS["gamma"]*(
            PARAMETERS["p1"]*v_success+(1-PARAMETERS["p1"])*v_failure)
        q2 = PARAMETERS["R"]-PARAMETERS["c2"]+PARAMETERS["gamma"]*(
            PARAMETERS["p2"]*v_success+(1-PARAMETERS["p2"])*v_failure)
        optimal = demand/(1-PARAMETERS["gamma"]*(1-demand))*np.maximum(q1, q2)
        np.testing.assert_allclose(evaluated, optimal, atol=1e-8, rtol=0)


if __name__ == "__main__":
    unittest.main()
