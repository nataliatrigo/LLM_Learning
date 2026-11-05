"""Independent derivative, labelled-payoff and certification checks for N=4."""

from itertools import product
from unittest.mock import patch
import unittest

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve

from . import four_seller_model as model


class FourSellerModelTests(unittest.TestCase):
    def test_exact_equilibrium_derivative_against_central_differences(self):
        experiment = model.FourSellerExperiment(n=5, p0=.3, order=64)
        alpha = np.linspace(.15, .85, 5)
        analytic = experiment.calculate(alpha)['jacobian']
        numerical = np.empty_like(analytic)
        for k in range(5):
            plus, minus = alpha.copy(), alpha.copy()
            plus[k] += 1e-5
            minus[k] -= 1e-5
            numerical[:, k] = (experiment.calculate(plus)['advantage']
                               - experiment.calculate(minus)['advantage'])/2e-5
        np.testing.assert_allclose(analytic, numerical, atol=1e-8, rtol=1e-7)

    def test_mean_quotient_value_matches_independent_labelled_focal_value(self):
        experiment = model.FourSellerExperiment(n=3, p0=.3, order=64)
        alpha = np.asarray([.2, .7, .4])
        result = experiment.certify(alpha)
        game, p = experiment.game, experiment.params
        states = np.asarray(list(product(range(3), repeat=4)))
        size = len(states)
        rows, columns, weights = [], [], []
        focal_reward = np.zeros(size)
        for row, state in enumerate(states):
            orbit = int(game.orbit_index(state))
            rows.append(row); columns.append(row); weights.append(game.outside[orbit])
            for seller in range(4):
                own = state[seller]
                rival = int(game.rival_index(np.delete(state, seller)))
                demand = game.focal_demand[own, rival]
                if seller == 0:
                    focal_reward[row] = demand*(p.R-p.c1-(p.c2-p.c1)*alpha[own])
                success = p.p1+(p.p2-p.p1)*alpha[own]
                for operator, probability in ((game.success, success), (game.failure, 1-success)):
                    start, end = operator.indptr[own:own+2]
                    for target, interpolation in zip(operator.indices[start:end], operator.data[start:end]):
                        destination = state.copy(); destination[seller] = target
                        column = int(np.ravel_multi_index(destination, (3,)*4))
                        rows.append(row); columns.append(column)
                        weights.append(demand*probability*interpolation)
        full = csr_matrix((weights, (rows, columns)), shape=(size, size))
        focal = spsolve(eye(size, format='csr')-p.gamma*full, focal_reward)
        initial = np.ravel_multi_index((1,)*4, (3,)*4)
        self.assertAlmostEqual(focal[initial], result['summary']['value_initial'], places=8)
        # A focal payoff generally differs at asymmetric states, so the saved
        # quotient value must explicitly retain its mean-payoff description.
        quotient_focal = result['actual_value'][game.orbit_index(states)]
        self.assertGreater(float(np.max(abs(quotient_focal-focal))), .1)
        self.assertIn('Mean', result['diagnostics']['actual_value_scope'])

    def test_own_sale_law_and_market_accounting_are_independently_consistent(self):
        experiment = model.FourSellerExperiment(n=5, p0=.1, order=64)
        result = experiment.certify(np.linspace(.1, .9, 5))
        diagnostics = result['diagnostics']
        self.assertLess(diagnostics['sale_law_l1'], 1e-11)
        self.assertLess(diagnostics['market_partition_error'], 1e-12)
        self.assertLess(diagnostics['stationary_profit_accounting_error'], 1e-12)
        self.assertLess(diagnostics['actual_value_residual'], 1e-11)
        self.assertFalse(result['verified'])  # An arbitrary policy is not an OE.

    def test_stationary_residual_cannot_bypass_independent_positive_validation(self):
        experiment = model.FourSellerExperiment(n=3, p0=.1, order=64)
        jump, rates = experiment.game.jump(np.full(3, .5))
        with patch.object(model, '_positive_iterates', side_effect=ArithmeticError('budget exhausted')):
            with self.assertRaisesRegex(ArithmeticError, 'budget exhausted'):
                model.checked_quotient_stationary(jump, rates)

    def test_search_releases_old_factor_but_can_materialize_old_jacobian(self):
        experiment = model.FourSellerExperiment(n=3, p0=.1, order=64)
        old = experiment.calculate(np.asarray([.2, .5, .8]))
        experiment.calculate(np.asarray([.3, .4, .7]))
        self.assertTrue(np.all(np.isfinite(old['jacobian'])))

    def test_scope_and_tolerances_are_explicit(self):
        experiment = model.FourSellerExperiment(n=3, p0=.1, order=64)
        config = experiment.configuration()
        self.assertEqual(config['sellers'], 4)
        self.assertEqual(config['certification']['invariant_tolerance'], 1e-11)
        self.assertEqual(config['certification']['stability_tolerance'], 1e-7)
        self.assertIn('correlations preserved', config['marginal_belief'])
        self.assertIn('Mean across four', config['actual_value_scope'])
        for kwargs in ({'sellers': 3}, {'closure': 'conditional'}, {'tolerance': 1e-6}, {'n': 4}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                model.FourSellerExperiment(**kwargs)


if __name__ == '__main__':
    unittest.main()
