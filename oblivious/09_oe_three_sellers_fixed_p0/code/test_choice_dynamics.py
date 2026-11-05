"""Check simulated buyer events against the saved game's exact calendar kernel."""

from dataclasses import asdict
import unittest

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from .kernel import FixedOutsideKernel
from . import choice_dynamics as dynamics


class ChoiceDynamicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.params = Parameters(rho=.63)
        cls.kernel = FixedOutsideKernel(np.linspace(0, 1, 5), cls.params,
                                        p0=.4, sellers=2, order=64)
        cls.policy = np.array([.1, .3, .65, .85, .2])
        cls.game = dynamics.SavedGame(
            kernel=cls.kernel, policy=cls.policy,
            joint=np.full((5, 5), 1 / 25),
            configuration=dict(n=5, sellers=2, closure='marginal', p0=.4,
                               parameters=asdict(cls.params)),
            summary=dict(p0=.4, rho=.63, n=5, sellers=2, closure='marginal'))
        cls.replicates = 80_000
        cls.states = np.tile([1, 3], (cls.replicates, 1))
        cls.sample = dynamics.simulate_step(
            cls.game, cls.states,
            np.random.default_rng(7941).random((cls.replicates, 4)))

    def assert_binomial_agreement(self, observed, probability, count):
        self.assertGreater(count, 100)
        tolerance = 6 * np.sqrt(probability * (1 - probability) / count) + 3 / count
        self.assertLessEqual(abs(observed - probability), tolerance)

    def test_selection_probabilities_are_realized_joint_events(self):
        states = np.array([[0, 4], [1, 3], [2, 2], [4, 0]])
        actual = dynamics.selection_probabilities(self.game, states)
        expected = np.array([
            [self.kernel.outside[tuple(state)]] +
            [self.kernel.demand[seller][tuple(state)] for seller in range(2)]
            for state in states])
        np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=2e-15)
        np.testing.assert_allclose(actual.sum(axis=1), 1, atol=2e-15)
        np.testing.assert_allclose(actual[0, 1:], actual[-1, 1:][::-1], atol=2e-15)

    def test_one_step_simulation_matches_independent_sparse_transition_row(self):
        next_states, choices, _, _ = self.sample
        ids = np.ravel_multi_index(next_states.T, self.kernel.shape)
        observed = np.bincount(ids, minlength=self.kernel.size) / self.replicates
        row = np.ravel_multi_index((1, 3), self.kernel.shape)
        expected = self.kernel.transition(self.policy).getrow(row).toarray().ravel()
        tolerance = (6 * np.sqrt(expected * (1 - expected) / self.replicates)
                     + 3 / self.replicates)
        self.assertTrue(np.all(abs(observed - expected) <= tolerance),
                        msg=f'Maximum transition error: {max(abs(observed-expected))}')
        event_expected = dynamics.selection_probabilities(self.game, np.array([[1, 3]]))[0]
        event_observed = np.bincount(choices, minlength=3) / self.replicates
        for observed_probability, probability in zip(event_observed, event_expected):
            self.assert_binomial_agreement(observed_probability, probability, self.replicates)

    def test_outside_and_nonselected_sellers_freeze(self):
        next_states, choices, products, successes = self.sample
        outside = choices == 0
        self.assertTrue(np.any(outside))
        np.testing.assert_array_equal(next_states[outside], self.states[outside])
        self.assertTrue(np.all(products[outside] == 0))
        self.assertTrue(np.all(successes[outside] == -1))
        for seller in range(2):
            not_selected = choices != seller + 1
            np.testing.assert_array_equal(next_states[not_selected, seller],
                                          self.states[not_selected, seller])
        self.assertTrue(np.all((next_states >= 0) & (next_states < 5)))

    def test_mixed_actions_and_product_specific_outcomes(self):
        _, choices, products, successes = self.sample
        for seller in range(2):
            selected = choices == seller + 1
            count = int(selected.sum())
            own_index = (1, 3)[seller]
            self.assert_binomial_agreement(np.mean(products[selected] == 2),
                                           self.policy[own_index], count)
        for product, probability in ((1, self.params.p1), (2, self.params.p2)):
            used = products == product
            self.assert_binomial_agreement(np.mean(successes[used]), probability,
                                           int(used.sum()))

    def test_off_grid_update_is_a_lottery_between_neighbors(self):
        states = np.tile([1, 3], (2, 1))
        probabilities = dynamics.selection_probabilities(self.game, states[:1])[0]
        choose_a = probabilities[0] + probabilities[1] / 2
        # Product 2, successful observation. The target .5275 lies strictly
        # between grid nodes .5 and .75; opposite uniforms must select both.
        uniforms = np.array([[choose_a, 0., 0., 0.],
                             [choose_a, 0., 0., .999]])
        updated, choices, products, successes = dynamics.simulate_step(
            self.game, states, uniforms)
        np.testing.assert_array_equal(choices, [1, 1])
        np.testing.assert_array_equal(products, [2, 2])
        np.testing.assert_array_equal(successes, [1, 1])
        np.testing.assert_array_equal(updated[:, 0], [2, 3])
        np.testing.assert_array_equal(updated[:, 1], [3, 3])

    def test_transient_choices_and_final_law_match_forward_propagation(self):
        periods = 3
        actual = dynamics.exact_transient(self.game, periods)
        transition = self.kernel.transition(self.policy)
        law = np.zeros(self.kernel.size)
        law[np.ravel_multi_index((2, 2), self.kernel.shape)] = 1
        event_matrix = np.column_stack([
            self.kernel.outside.ravel(),
            *(self.kernel.demand[seller].ravel() for seller in range(2))])
        # Symmetry makes the expected own reputation equal the first seller's
        # expectation; derive it directly instead of reusing the reporter mean.
        own_reputation = self.kernel.grid[np.indices(self.kernel.shape)[0]].ravel()
        choices, mean_reputations = [], []
        for _ in range(periods):
            choices.append(law @ event_matrix)
            mean_reputations.append(law @ own_reputation)
            law = np.asarray(transition.T @ law).ravel()
        np.testing.assert_allclose(actual['choices'], choices, atol=3e-15, rtol=3e-15)
        np.testing.assert_allclose(actual['mean_reputation'], mean_reputations,
                                   atol=3e-15, rtol=3e-15)
        np.testing.assert_allclose(actual['final_law'], law, atol=3e-15, rtol=3e-15)
        self.assertGreaterEqual(np.min(actual['final_law']), 0)
        self.assertAlmostEqual(float(np.sum(actual['final_law'])), 1, places=13)
        np.testing.assert_allclose(np.sum(actual['choices'], axis=1), 1, atol=3e-15)
        np.testing.assert_allclose(actual['cumulative_sales_per_seller'],
                                   np.cumsum(np.asarray(choices)[:, 1:].sum(axis=1)) / 2,
                                   atol=3e-15, rtol=3e-15)

    def test_path_reproducibility_and_initial_condition(self):
        first = dynamics.simulate_path(self.game, periods=100, seed=184)
        second = dynamics.simulate_path(self.game, periods=100, seed=184)
        self.assertEqual(first['states'].shape, (101, 2))
        np.testing.assert_array_equal(first['states'][0], [2, 2])
        for key in ('states', 'choices', 'products', 'successes'):
            np.testing.assert_array_equal(first[key], second[key])
        for seller in range(2):
            frozen = first['choices'] != seller + 1
            np.testing.assert_array_equal(first['states'][1:, seller][frozen],
                                          first['states'][:-1, seller][frozen])


if __name__ == '__main__':
    unittest.main()
