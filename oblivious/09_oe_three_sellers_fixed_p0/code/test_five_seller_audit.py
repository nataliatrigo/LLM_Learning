"""Scientific cross-checks of orbit accounting and the N5 payoff scope."""

from itertools import product
from copy import deepcopy
import unittest

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.linalg import spsolve

from oblivious.oe_p0.code.primitives import Parameters
from .five_seller_kernel import FiveSellerKernel
from .five_seller_audit import (orbit_observables, independent_marginal_beliefs,
                                _adaptive_quadrature, _stationary_certificate, frozen)


def labeled_data(game):
    states = np.asarray(list(product(range(game.n), repeat=5)))
    orbits = game.orbit_index(states)
    demand = np.empty(states.shape)
    for row, state in enumerate(states):
        canonical = np.sort(state)
        for seller, own in enumerate(state):
            demand[row,seller] = game.demand[orbits[row], np.flatnonzero(canonical == own)[0]]
    return states, orbits, demand


def labeled_transition(game, states, demands, policy):
    """Slow labeled oracle; independent of production transition assembly."""
    lookup = {tuple(state):row for row,state in enumerate(states)}
    matrix = np.zeros((len(states),len(states)))
    p = game.params
    for row,state in enumerate(states):
        matrix[row,row] += game.outside[game.orbit_index(state)]
        for seller, own in enumerate(state):
            probability = p.p1+(p.p2-p.p1)*policy[own]
            for observed, event_probability in ((1,probability),(0,1-probability)):
                next_x = p.rho*game.grid[own]+(1-p.rho)*observed
                upper = min(np.searchsorted(game.grid,next_x,side='right'),game.n-1)
                lower = max(upper-1,0)
                weight = (next_x-game.grid[lower])/(game.grid[upper]-game.grid[lower])
                for destination,interpolation in ((lower,1-weight),(upper,weight)):
                    target = state.copy(); target[seller] = destination
                    matrix[row,lookup[tuple(target)]] += demands[row,seller]*event_probability*interpolation
    return csr_matrix(matrix)


class OrbitAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = Parameters(rho=.6)
        cls.game = FiveSellerKernel(np.linspace(0,1,3),cls.params,p0=.3,order=128)
        cls.states,cls.orbits,cls.demands = labeled_data(cls.game)
        rng = np.random.default_rng(7751)
        cls.law = rng.random(cls.game.size)
        cls.law /= cls.law.sum()

    def test_orbit_accounting_matches_all_labeled_states(self):
        g = self.game
        policy = np.asarray([.7,.2,.9])
        marginal,sales,reward = orbit_observables(g.representatives,self.law,g.demand,policy,self.params)
        full_law = self.law[self.orbits]/g.orbit_sizes[self.orbits]
        expected_marginal = np.bincount(self.states[:,0],weights=full_law,minlength=g.n)
        expected_sales = np.bincount(self.states[:,0],weights=full_law*self.demands[:,0],minlength=g.n)
        margins = self.params.R-self.params.c1-(self.params.c2-self.params.c1)*policy
        full_mean_reward = (self.demands*margins[self.states]).sum(axis=1)/5
        np.testing.assert_allclose(marginal,expected_marginal,atol=2e-16)
        np.testing.assert_allclose(sales,expected_sales,atol=2e-16)
        np.testing.assert_allclose(reward[self.orbits],full_mean_reward,atol=2e-16)
        self.assertAlmostEqual(full_law.sum(),1.)

    def test_marginal_belief_keeps_correlated_four_rival_distribution(self):
        g = self.game
        full_law = self.law[self.orbits]/g.orbit_sizes[self.orbits]
        rival_law = full_law.reshape((g.n,)*5).sum(axis=0)
        expected = np.zeros(g.n)
        for x in range(g.n):
            for rivals in product(range(g.n),repeat=4):
                state = np.asarray((x,)+rivals)
                demand = g.demand[g.orbit_index(state),np.sum(state < x)]
                expected[x] += rival_law[rivals]*demand
        actual = independent_marginal_beliefs(g,self.law)
        np.testing.assert_allclose(actual,expected,atol=2e-16)
        np.testing.assert_allclose(actual,g.marginal_demand@self.law,atol=2e-16)
        one_rival = rival_law.sum(axis=(1,2,3))
        independent = np.zeros(g.n)
        for x in range(g.n):
            for rivals in product(range(g.n),repeat=4):
                state = np.asarray((x,)+rivals)
                demand = g.demand[g.orbit_index(state),np.sum(state < x)]
                independent[x] += np.prod(one_rival[list(rivals)])*demand
        self.assertGreater(np.max(abs(actual-independent)),1e-3)

    def test_scalar_quadrature_agrees_at_extreme_and_unequal_states(self):
        result = _adaptive_quadrature(self.game)
        self.assertEqual(result['seller_integrals_checked'],25)
        self.assertLess(result['maximum_absolute_error'],2e-10)

    def test_discounted_quotient_mean_equals_each_seller_only_at_symmetric_initial(self):
        g,p = self.game,self.params
        policy = np.asarray([1.,.3,0.])
        full = labeled_transition(g,self.states,self.demands,policy)
        quotient = g.transition(policy)
        reward = self.demands*(p.R-p.c1-(p.c2-p.c1)*policy[self.states])
        full_values = spsolve(eye(len(self.states))-p.gamma*full,reward)
        _,_,mean_reward = orbit_observables(g.representatives,self.law,g.demand,policy,p)
        mean_values = spsolve(eye(g.size)-p.gamma*quotient,mean_reward)
        np.testing.assert_allclose(full_values.mean(axis=1),mean_values[self.orbits],atol=1e-12)
        symmetric = np.flatnonzero(np.all(self.states == 1,axis=1))[0]
        np.testing.assert_allclose(full_values[symmetric],mean_values[self.orbits[symmetric]],atol=1e-12)
        unequal = np.flatnonzero(np.all(self.states == [0,2,2,2,2],axis=1))[0]
        self.assertGreater(abs(full_values[unequal,0]-mean_values[self.orbits[unequal]]),1.)

    def test_quotient_mass_must_not_be_multiplied_by_orbit_size(self):
        g = self.game
        marginal,_,_ = orbit_observables(g.representatives,self.law,g.demand,np.zeros(g.n),self.params)
        self.assertAlmostEqual(marginal.sum(),1.)
        wrong,_,_ = orbit_observables(g.representatives,self.law*g.orbit_sizes,g.demand,np.zeros(g.n),self.params)
        self.assertGreater(wrong.sum(),2.)


class RecordedStationaryCertificateTests(unittest.TestCase):
    def setUp(self):
        self.record = dict(
            verified=True, tolerance=1e-11, stability_tolerance=1e-7,
            residual=1e-15, rate_weighted_residual=1e-15, row_sum_error=1e-15,
            stability_l1=8e-8, independent_calendar_l1=1e-14,
            independent_jump_l1=1e-14,
            positive_iteration=dict(iterations=1000, discrepancy=8e-8,
                                    calendar_discrepancy=8e-8, jump_discrepancy=7e-8),
            methods=['GMRES original order and middle reference',
                     'GMRES reversed order and edge reference',
                     'positive lazy jump iteration, uniform and concentrated starts'])

    def test_full_independent_certificate_is_accepted(self):
        _stationary_certificate(self.record)

    def test_small_residual_does_not_replace_independent_checks(self):
        for key in ['independent_calendar_l1','independent_jump_l1','stability_l1']:
            with self.subTest(key=key):
                record = deepcopy(self.record)
                record[key] = 1.01e-7
                with self.assertRaises(frozen.AuditFailure):
                    _stationary_certificate(record)

    def test_positive_two_start_discrepancy_is_checked(self):
        record = deepcopy(self.record)
        record['positive_iteration']['discrepancy'] = 1.01e-7
        with self.assertRaises(frozen.AuditFailure):
            _stationary_certificate(record)

    def test_missing_positive_start_is_rejected(self):
        record = deepcopy(self.record)
        record['methods'][-1] = 'positive lazy iteration from uniform start only'
        with self.assertRaises(frozen.AuditFailure):
            _stationary_certificate(record)

    def test_looser_tolerances_are_rejected(self):
        for key in ['tolerance','stability_tolerance']:
            with self.subTest(key=key):
                record = deepcopy(self.record)
                record[key] *= 2
                with self.assertRaises(frozen.AuditFailure):
                    _stationary_certificate(record)


if __name__ == '__main__':
    unittest.main()
