"""Bounded checks of movable endpoint and interior OE decisions.

Run from the project root:
    OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m unittest discover \
        -s 'oblivious/07_full_comparison/code' -p test_oe_active_set.py -v
"""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose


SPEC = importlib.util.spec_from_file_location(
    "comparison_oe_active_set", Path(__file__).with_name("oe_active_set.py"))
ACTIVE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACTIVE)


class AffineExperiment:
    """Independently certified affine complementarity test problem."""

    def __init__(self, constant, jacobian, reject_certificate=False):
        self.constant = np.asarray(constant, dtype=float)
        self.jacobian = np.asarray(jacobian, dtype=float)
        self.grid = np.linspace(0., 1., len(self.constant))
        self.dp_tol = 1e-10
        self.reject_certificate = reject_certificate
        self.certified = []

    def advantage(self, rule):
        return self.constant + self.jacobian @ rule

    def evaluate(self, rule):
        rule = np.asarray(rule, dtype=float)
        self.certified.append(rule.copy())
        advantage = self.advantage(rule)
        regret = np.maximum(advantage, 0.) - rule * advantage
        pure_optimal = np.all(advantage[rule == 0.] <= self.dp_tol) and np.all(
            advantage[rule == 1.] >= -self.dp_tol)
        mixed = np.flatnonzero((rule > 0.) & (rule < 1.))
        indifferent = np.max(np.abs(advantage[mixed]), initial=0.) <= self.dp_tol
        verified = bool(pure_optimal and indifferent and not self.reject_certificate)
        complementarity = np.max(abs(rule - np.clip(rule + advantage, 0., 1.)))
        return dict(policy=rule.copy(), advantage=advantage, verified=verified,
                    candidate_verified=verified, mixed_indices=mixed.tolist(),
                    dp_diagnostics=dict(conditional_regret=float(np.max(regret)),
                                        complementarity_residual=float(complementarity)))


class AffineResponse:
    def __init__(self, experiment):
        self.experiment = experiment
        self.calls = 0

    def calculate(self, rule):
        self.calls += 1
        return dict(advantage=self.experiment.advantage(rule),
                    jacobian=self.experiment.jacobian)


class ActiveSetDecisions(unittest.TestCase):
    def solve(self, experiment, initial, budget=80):
        with patch.object(ACTIVE, "_JointResponse", AffineResponse):
            return ACTIVE.solve_active_set(experiment, initial, max_evaluations=budget)

    def test_endpoints_can_enter_a_genuine_mixed_solution(self):
        expected = np.array([.25, .7, .3])
        experiment = AffineExperiment(expected, -np.eye(3))
        initial = np.array([0., 1., .5])
        result = self.solve(experiment, initial)
        self.assertTrue(result["verified"])
        assert_allclose(result["policy"], expected, atol=1e-12, rtol=0.)
        self.assertEqual(result["mixed_indices"], [0, 1, 2])
        assert_allclose(initial, [0., 1., .5], atol=0., rtol=0.)

    def test_pure_equilibrium_is_not_restricted_to_a_threshold(self):
        experiment = AffineExperiment([.3, -.2, .3], np.zeros((3, 3)))
        result = self.solve(experiment, np.full(3, .5))
        self.assertTrue(result["verified"])
        assert_allclose(result["policy"], [1., 0., 1.], atol=0., rtol=0.)
        self.assertFalse(result["search_metadata"]["pure_strategy_restriction"])

    def test_false_mixed_face_can_be_abandoned_to_fix_other_node(self):
        # At (0,.6,0), node 1 is indifferent but node 0 prefers product 2.
        # Solving the released two-node indifference system points outside
        # the feasible square.  Moving node 1 to zero instead satisfies all
        # pure inequalities.  This captures the failed real candidate's
        # branch geometry without encoding its reputation or policy support.
        jacobian = np.array([[.05, .035, 0.], [.05, .037, 0.], [0., 0., 0.]])
        constant = np.array([.004-.035*.6, -.037*.6, -.3])
        experiment = AffineExperiment(constant, jacobian)
        result = self.solve(experiment, np.array([0., .6, 0.]))
        self.assertTrue(result["verified"])
        assert_allclose(result["policy"], np.zeros(3), atol=0., rtol=0.)
        self.assertTrue(any(row["stage"] == "active face pivot" for row in result["history"]))

    def test_wrong_endpoint_can_switch_to_opposite_pure_action(self):
        experiment = AffineExperiment([.2, -.2, -.2], np.zeros((3, 3)))
        result = self.solve(experiment, np.zeros(3))
        self.assertTrue(result["verified"])
        assert_allclose(result["policy"], [1., 0., 0.], atol=0., rtol=0.)

    def test_every_endpoint_is_checked_after_a_mixed_face_changes(self):
        # Dropping the initial mixed action to zero fixes node 0, but makes
        # node 2's existing product-2 action suboptimal.  It must be revisited.
        jacobian = np.array([[.05, .035, 0.], [.05, .037, 0.], [0., .004, 0.]])
        constant = np.array([.004-.035*.6, -.037*.6, -.0012])
        experiment = AffineExperiment(constant, jacobian)
        result = self.solve(experiment, np.array([0., .6, 1.]))
        self.assertTrue(result["verified"])
        assert_allclose(result["policy"], np.zeros(3), atol=0., rtol=0.)
        self.assertTrue(result["search_metadata"]["all_nodes_rechecked"])

    def test_failed_full_certificate_is_never_reported_as_equilibrium(self):
        experiment = AffineExperiment([-.2, -.2, -.2], np.zeros((3, 3)),
                                     reject_certificate=True)
        result = self.solve(experiment, np.zeros(3))
        self.assertFalse(result["verified"])
        self.assertFalse(result["candidate_verified"])
        self.assertEqual(len(experiment.certified), 1)

    def test_budget_is_strict_for_exploratory_calls_and_certificate_still_runs(self):
        experiment = AffineExperiment([.2, -.2, -.2], np.zeros((3, 3)))
        result = self.solve(experiment, np.zeros(3), budget=1)
        self.assertFalse(result["verified"])
        self.assertEqual(result["search_metadata"]["exploratory_evaluations"], 1)
        self.assertEqual(result["search_metadata"]["full_certifications"], 1)
        self.assertEqual(len(experiment.certified), 1)

    def test_invalid_policy_and_budgets_are_rejected(self):
        experiment = AffineExperiment([-.2, -.2, -.2], np.zeros((3, 3)))
        for initial in ([0., 0.], [0., np.nan, 0.], [0., -.1, 0.], [0., 1.1, 0.]):
            with self.subTest(initial=initial), self.assertRaises(ValueError):
                self.solve(experiment, np.asarray(initial))
        for budget in (0, -1, 1.5):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.solve(experiment, np.zeros(3), budget=budget)


class ActiveSetEconomicCertificate(unittest.TestCase):
    def test_small_real_joint_model_keeps_full_independent_certificate(self):
        from oblivious.oe_p0.code.primitives import Parameters
        from oblivious.oe_distribution.code.model import JointMarginalExperiment
        experiment = JointMarginalExperiment(n=7, params=Parameters(rho=.6, gamma=0.))
        result = ACTIVE.solve_active_set(experiment, np.full(7, .4), max_evaluations=80)
        self.assertTrue(result["verified"], result["dp_diagnostics"])
        assert_allclose(result["policy"], np.zeros(7), atol=0., rtol=0.)
        self.assertTrue(result["stationary_diagnostics"]["verified"])
        self.assertEqual(result["search_metadata"]["full_certifications"], 1)


if __name__ == "__main__":
    unittest.main()
