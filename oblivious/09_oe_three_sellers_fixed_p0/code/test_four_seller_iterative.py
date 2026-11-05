"""Iterative equations, derivatives and audit metadata against direct solves."""

import unittest

import numpy as np

from . import four_seller_model as model
from . import four_seller_iterative as iterative


class FourSellerIterativeTests(unittest.TestCase):
    def test_pinned_equations_and_actual_certificate_match_direct(self):
        direct = model.FourSellerExperiment(n=5, p0=.3, order=64)
        alpha = np.linspace(.15, .85, 5)
        original = direct.calculate(alpha)
        reference = direct.certify(alpha)
        old_pinned, old_factor = model._pinned_jump, model.splu
        restore = iterative.install()
        try:
            experiment = model.FourSellerExperiment(n=5, p0=.3, order=64)
            candidate = experiment.calculate(alpha)
            result = experiment.certify(alpha)
            np.testing.assert_allclose(candidate['reduced_law'], original['reduced_law'], atol=1e-11)
            np.testing.assert_allclose(result['joint'], reference['joint'], atol=1e-11)
            np.testing.assert_allclose(candidate['advantage'], original['advantage'], atol=1e-10)
            self.assertIn('matrix-free GMRES', result['diagnostics']['stationary']['methods'][0])
            self.assertEqual(len(result['diagnostics']['stationary']['linear_solves']), 2)
            self.assertEqual(experiment.configuration()['certification']['invariant_tolerance'], 1e-11)
            self.assertIn('GMRES', experiment.configuration()['certification']['sparse_linear_solver']['method'])
        finally:
            restore()
        self.assertIs(model._pinned_jump, old_pinned)
        self.assertIs(model.splu, old_factor)

    def test_iterative_analytic_derivative_matches_finite_differences(self):
        restore = iterative.install()
        try:
            experiment = model.FourSellerExperiment(n=5, p0=.1, order=64)
            alpha = np.linspace(.15, .85, 5)
            jacobian = experiment.calculate(alpha)['jacobian']
            numerical = np.empty_like(jacobian)
            for k in range(5):
                plus, minus = alpha.copy(), alpha.copy()
                plus[k] += 1e-5; minus[k] -= 1e-5
                numerical[:, k] = (experiment.calculate(plus)['advantage']
                                   -experiment.calculate(minus)['advantage'])/2e-5
            np.testing.assert_allclose(jacobian, numerical, atol=1e-8, rtol=1e-7)
        finally:
            restore()

    def test_install_restores_all_globals_and_rejects_nested_install(self):
        originals = (model._pinned_jump, model.splu, model.checked_quotient_stationary, model.CERTIFICATION)
        restore = iterative.install()
        try:
            with self.assertRaisesRegex(RuntimeError, 'already installed'):
                iterative.install()
        finally:
            restore()
        self.assertEqual((model._pinned_jump, model.splu, model.checked_quotient_stationary,
                          model.CERTIFICATION), originals)


if __name__ == '__main__':
    unittest.main()
