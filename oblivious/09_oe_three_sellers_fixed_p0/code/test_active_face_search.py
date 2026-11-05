"""Selected exact derivatives and full-grid incentives for optional recovery."""

from hashlib import sha256
import json
import unittest
from unittest.mock import patch

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from . import active_face_search as entry
from .model import Experiment as FrozenExperiment


class ToyFace(entry.ActiveFaceExperiment):
    """Analytic incentives expose an initially omitted profitable deviation."""

    def __init__(self):
        self.n, self.tolerance, self.calls, self.certificate_calls = 3, 1e-7, 0, 0

    def calculate_active(self, alpha, columns):
        self.calls += 1
        advantage = np.array([.5, -.5, .25])
        return dict(policy=np.asarray(alpha).copy(), advantage=advantage,
                    jacobian=np.zeros((self.n, len(columns))),
                    regret=entry.incentive_violations(alpha, advantage).max())

    def certify(self, alpha):
        self.certificate_calls += 1
        self.full_certificate_regret = entry.incentive_violations(alpha, np.array([.5, -.5, .25])).max()
        return dict(verified=self.full_certificate_regret <= self.tolerance, policy=alpha.copy())


class ActiveFaceChecks(unittest.TestCase):
    def experiment(self, sellers=3, closure='marginal'):
        return entry.ActiveFaceExperiment(n=5, sellers=sellers, closure=closure,
                    p0=.5, params=Parameters(rho=.7), order=64)

    def test_selected_columns_match_full_frozen_analytic_jacobian(self):
        alpha = np.array([.2, .35, .6, .4, .1])
        for sellers, closure in ((2, 'marginal'), (3, 'marginal'), (2, 'conditional')):
            with self.subTest(sellers=sellers, closure=closure):
                experiment = self.experiment(sellers, closure)
                full = experiment.calculate(alpha)
                selected = experiment.calculate_active(alpha, [1, 3])
                np.testing.assert_allclose(selected['advantage'], full['advantage'], rtol=1e-11, atol=1e-11)
                np.testing.assert_allclose(selected['jacobian'], full['jacobian'][:, [1, 3]],
                                           rtol=2e-9, atol=2e-9)

    def test_selected_columns_match_central_finite_differences(self):
        experiment = self.experiment()
        alpha, columns, step = np.array([.2, .35, .6, .4, .1]), [1, 3], 1e-5
        exact = experiment.calculate_active(alpha, columns)['jacobian'].copy()
        for j, column in enumerate(columns):
            upper, lower = alpha.copy(), alpha.copy()
            upper[column] += step
            lower[column] -= step
            numeric = (experiment.calculate_active(upper, columns)['advantage']
                       - experiment.calculate_active(lower, columns)['advantage'])/(2*step)
            np.testing.assert_allclose(exact[:, j], numeric, rtol=2e-5, atol=2e-7)

    def test_derivative_solves_have_only_requested_rhs_columns(self):
        original_splu, shapes = entry.splu, []
        class TracedFactor:
            def __init__(self, factor):
                self.factor = factor
            def solve(self, rhs):
                shapes.append(np.asarray(rhs).shape)
                return self.factor.solve(rhs)
        def traced(*args, **kwargs):
            return TracedFactor(original_splu(*args, **kwargs))
        experiment = self.experiment()
        with patch.object(entry, 'splu', side_effect=traced):
            selected = experiment.calculate_active(np.full(5, .4), [1, 3])
            self.assertEqual(selected['jacobian'].shape, (5, 2))
        matrix_rhs = [shape for shape in shapes if len(shape) == 2]
        self.assertEqual(len(matrix_rhs), 2)
        self.assertTrue(all(shape[1] == 2 for shape in matrix_rhs), shapes)

    def test_same_policy_reuses_evaluation_when_active_columns_change(self):
        experiment = self.experiment()
        alpha = np.full(5, .4)
        first = experiment.calculate_active(alpha, [1])
        calls = experiment.calls
        second = experiment.calculate_active(alpha, [1, 3])
        self.assertEqual(experiment.calls, calls)
        np.testing.assert_array_equal(first['advantage'], second['advantage'])
        np.testing.assert_allclose(first['jacobian'][:, 0], second['jacobian'][:, 0], atol=1e-12)

    def test_real_low_grid_solution_passes_full_certificate_and_logs_new_evaluations(self):
        reference = self.experiment().solve(max_evaluations=80)
        experiment, events = self.experiment(), []
        result = experiment.solve_active_face(reference['policy'], [1, 3],
                                             max_evaluations=30, progress=events.append)
        self.assertTrue(result['verified'])
        self.assertTrue(result['diagnostics']['stationary']['verified'])
        self.assertLessEqual(result['diagnostics']['conditional_regret'], experiment.tolerance)
        evaluations = [event for event in events if event['stage'] == 'active-face evaluation']
        self.assertEqual(len(evaluations), result['search_calls'])
        self.assertEqual(len({event['calls'] for event in evaluations}), len(evaluations))
        self.assertEqual(evaluations[0]['active'], [1, 3])
        self.assertEqual(len(evaluations[0]['alpha_active']), 2)
        self.assertIn('full_regret', evaluations[0])
        json.dumps(result['history'], allow_nan=False)

    def test_full_grid_violation_expands_initial_face_before_certification(self):
        experiment = ToyFace()
        result = experiment.solve_active_face(np.array([.5, 0., 0.]), [0], max_evaluations=100)
        self.assertTrue(result['verified'])
        self.assertEqual(experiment.certificate_calls, 1)
        self.assertLessEqual(experiment.full_certificate_regret, experiment.tolerance)
        expansions = [x for x in result['history'] if x['stage'] == 'expand active face']
        self.assertEqual(expansions[0]['added'], [2])
        json.dumps(result['history'], allow_nan=False)
        np.testing.assert_allclose(result['policy'], [1, 0, 1], atol=4e-7)

    def test_expansion_budget_failure_cannot_return_uncertified_face_solution(self):
        experiment = ToyFace()
        with self.assertRaisesRegex(entry.ActiveFaceFailure, 'expansion budget'):
            experiment.solve_active_face(np.array([1., 0., 0.]), [0], max_expansions=0)
        self.assertEqual(experiment.certificate_calls, 0)

    def test_full_certificate_rejection_is_an_explicit_failure(self):
        experiment = ToyFace()
        experiment.certify = lambda policy:dict(verified=False)
        with self.assertRaisesRegex(entry.ActiveFaceFailure, 'certificate rejected'):
            experiment.solve_active_face(np.array([1., 0., 1.]), [0])

    def test_final_certifier_is_inherited_unchanged_and_sources_are_fingerprinted(self):
        self.assertIs(entry.ActiveFaceExperiment.certify, FrozenExperiment.certify)
        for name, expected in entry.source_reference().items():
            self.assertEqual(sha256((entry.ROOT/name).read_bytes()).hexdigest(), expected)

    def test_invalid_active_columns_are_rejected(self):
        experiment = self.experiment()
        for columns in ([1, 1], [-1], [5], [1.0], [True]):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                experiment.calculate_active(np.full(5, .5), columns)


if __name__ == '__main__':
    unittest.main()
