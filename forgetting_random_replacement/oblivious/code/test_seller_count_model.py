"""Independent regression checks for exact quotients beyond five sellers."""
from math import comb
import unittest

import numpy as np

from forgetting_random_replacement.oblivious.code.kernel import CountKernel
from forgetting_random_replacement.oblivious.code.model import CountExperiment
from forgetting_random_replacement.oblivious.code.seller_count_model import (
    ExtendedCountKernel, ExtendedCountExperiment)
from forgetting_random_replacement.oblivious.code.test_model import (
    adaptive_selection, labelled_reference, quotient_from_labels)


class ExtendedSellerCountTests(unittest.TestCase):
    def test_historical_kernel_equivalence(self):
        for capacity, sellers, p0 in ((1, 1, .1), (3, 2, .5), (3, 5, .3), (5, 5, .9)):
            with self.subTest(capacity=capacity, sellers=sellers, p0=p0):
                old = CountKernel(capacity, sellers, p0)
                new = ExtendedCountKernel(capacity, sellers, p0)
                for name in ("states", "rivals", "selection", "outside", "focal_demand", "marginal_demand"):
                    np.testing.assert_allclose(getattr(old, name), getattr(new, name), atol=1e-14, rtol=0)
                policy = np.linspace(.1, .9, capacity+1)
                np.testing.assert_allclose(old.transition(policy).toarray(),
                                           new.transition(policy).toarray(), atol=1e-14, rtol=0)

    def test_six_sellers_match_explicit_labelled_binary_slot_chain(self):
        capacity, sellers, p0 = 1, 6, .4
        policy = np.array([.13, .79])
        game = ExtendedCountKernel(capacity, sellers, p0, order=32)
        states, transition, selection, outside, law = labelled_reference(
            capacity, sellers, p0, tuple(policy), .35, .8)
        expected, membership = quotient_from_labels(states, transition, game.states)
        np.testing.assert_allclose(game.transition(policy).toarray(), expected, atol=1e-12, rtol=0)
        np.testing.assert_array_equal(game.orbit_index(states), membership)
        np.testing.assert_array_equal(game.orbit_sizes, np.bincount(membership))
        quotient_law = np.bincount(membership, weights=law, minlength=game.size)
        own = np.bincount(states[:, 0], weights=law, minlength=2)
        sold = np.bincount(states[:, 0], weights=law*selection[:, 0], minlength=2)
        np.testing.assert_allclose(game.own_marginal@quotient_law, own, atol=1e-12, rtol=0)
        np.testing.assert_allclose(game.sale_marginal@quotient_law, sold, atol=1e-12, rtol=0)

    def test_large_seller_indexing_and_demand_against_adaptive_integrals(self):
        game = ExtendedCountKernel(1, 50, .5)
        self.assertEqual(game.size, 51)
        self.assertEqual(game.orbit_index(np.zeros((1, 50), dtype=int))[0], 0)
        self.assertEqual(game.orbit_index(np.ones((1, 50), dtype=int))[0], 50)
        self.assertTrue(np.all(np.isfinite(game.log_orbit_sizes)))
        self.assertTrue(np.all(game.orbit_sizes > 0))
        np.testing.assert_allclose(game.selection.sum(axis=1)+game.outside, 1, atol=1e-13, rtol=0)
        for own in (0, 1):
            for column in (0, 24, 49):
                expected = adaptive_selection(1, own, tuple(game.rivals[column]), .5)
                self.assertAlmostEqual(game.focal_demand[own, column], expected, delta=2e-13)

    def test_initial_mixture_matches_enumerated_and_large_binomial_laws(self):
        old = CountExperiment(3, 5, "marginal", .5)
        new = ExtendedCountExperiment(3, 5, "marginal", .5)
        values = np.sin(np.arange(new.game.size))+np.arange(new.game.size)**2
        self.assertAlmostEqual(new._initial(values), old._initial(values), delta=2e-12)

        many = ExtendedCountExperiment(1, 50, "marginal", .5)
        indicators = np.eye(many.game.size)
        actual = np.array([many._initial(indicator) for indicator in indicators])
        expected = np.array([comb(50, k)/2**50 for k in range(51)])
        np.testing.assert_allclose(actual, expected, atol=5e-15, rtol=5e-13)
        self.assertAlmostEqual(actual@np.arange(51), 25, delta=1e-12)

        even = ExtendedCountExperiment(2, 25, "marginal", .5)
        values = np.cos(np.arange(even.game.size))
        index = even.game.orbit_index(np.ones((1, 25), dtype=int))[0]
        self.assertEqual(even._initial(values), values[index])

    def test_state_budget_rejects_large_quotient_before_allocation(self):
        with self.assertRaisesRegex(ValueError, "state budget"):
            ExtendedCountKernel(8, 20, .5, max_states=250_000)


if __name__ == "__main__":
    unittest.main()
