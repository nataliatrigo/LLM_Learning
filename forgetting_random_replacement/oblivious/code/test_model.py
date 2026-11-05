"""Independent checks of count-state oblivious equilibria.

The reference chain enumerates labelled sellers and binary memory slots.  Its
selection probabilities use adaptive integration rather than the production
quadrature, and its transition graph explicitly replaces each possible slot.
"""

from dataclasses import replace
from functools import lru_cache
from itertools import product
import unittest

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, betaln

from forgetting.Code.model import Primitives
from forgetting_random_replacement.code.model import solve_counts
from forgetting_random_replacement.oblivious.code.kernel import CountKernel


@lru_cache(maxsize=None)
def adaptive_selection(capacity, own, rivals, p0):
    """Integrate one labelled seller's winning event independently."""
    def integrand(draw):
        density = np.exp(own*np.log(draw)+(capacity-own)*np.log1p(-draw)
                         - betaln(1+own, 1+capacity-own))
        return density*np.prod([betainc(1+other, 1+capacity-other, draw)
                                for other in rivals])

    return quad(integrand, p0, 1, epsabs=2e-13, epsrel=2e-13, limit=200)[0]


def slot_outcome_law(capacity, successes, outcome):
    """Enumerate replaced binary slots; no count transition formula is used."""
    word = (1 << successes)-1
    law = np.zeros(capacity+1)
    for slot in range(capacity):
        replaced = (word & ~(1 << slot)) | (outcome << slot)
        law[replaced.bit_count()] += 1/capacity
    return law


@lru_cache(maxsize=None)
def labelled_reference(capacity, sellers, p0, policy_tuple, p1, p2):
    """Full labelled count chain constructed from explicit slot replacements."""
    states = np.array(list(product(range(capacity+1), repeat=sellers)), dtype=int)
    lookup = {tuple(state): index for index, state in enumerate(states)}
    selection = np.zeros((len(states), sellers))
    outside = np.prod(betainc(1+states, 1+capacity-states, p0), axis=1)
    transition = np.diag(outside)
    policy = np.asarray(policy_tuple)
    for row, state in enumerate(states):
        for seller, own in enumerate(state):
            rivals = tuple(np.delete(state, seller))
            demand = adaptive_selection(capacity, int(own), rivals, p0)
            selection[row, seller] = demand
            probability = p1+(p2-p1)*policy[own]
            for outcome, weight in ((0, 1-probability), (1, probability)):
                word = (1 << int(own))-1
                for slot in range(capacity):
                    changed_word = (word & ~(1 << slot)) | (outcome << slot)
                    changed = state.copy()
                    changed[seller] = changed_word.bit_count()
                    transition[row, lookup[tuple(changed)]] += demand*weight/capacity
    system = transition.T-np.eye(len(states))
    system[-1, :] = 1
    rhs = np.zeros(len(states)); rhs[-1] = 1
    law = np.linalg.solve(system, rhs)
    return states, transition, selection, outside, law


def quotient_from_labels(states, transition, quotient_states):
    lookup = {tuple(state): index for index, state in enumerate(quotient_states)}
    membership = np.array([lookup[tuple(sorted(state))] for state in states])
    result = np.zeros((len(quotient_states), len(quotient_states)))
    for row, representative in enumerate(quotient_states):
        labelled_row = np.flatnonzero(np.all(states == representative, axis=1))[0]
        np.add.at(result[row], membership, transition[labelled_row])
    return result, membership


class CountObliviousTests(unittest.TestCase):
    def test_count_transitions_match_every_binary_memory_arrangement(self):
        for capacity in (2, 3):
            game = CountKernel(capacity, 2, .5)
            for word in range(1 << capacity):
                successes = word.bit_count()
                for outcome, operator in ((0, game.failure), (1, game.success)):
                    expected = np.zeros(capacity+1)
                    for slot in range(capacity):
                        changed = (word & ~(1 << slot)) | (outcome << slot)
                        expected[changed.bit_count()] += 1/capacity
                    with self.subTest(capacity=capacity, word=word, outcome=outcome):
                        np.testing.assert_allclose(operator.getrow(successes).toarray()[0],
                                                   expected, atol=2e-15, rtol=0)

    def test_demand_events_and_all_rival_correlations_use_adaptive_integrals(self):
        for sellers in (1, 2, 5):
            for p0 in (0., .3, .8):
                game = CountKernel(3, sellers, p0, order=32)
                np.testing.assert_allclose(game.selection.sum(axis=1)+game.outside,
                                           1, atol=2e-14, rtol=0)
                # Test every focal state against every possible rival multiset,
                # including repeated rivals and the boundary counts.
                for own in range(4):
                    for column, rivals in enumerate(game.rivals):
                        expected = adaptive_selection(3, own, tuple(rivals), p0)
                        with self.subTest(sellers=sellers, p0=p0, own=own,
                                          rivals=tuple(rivals)):
                            self.assertAlmostEqual(game.focal_demand[own, column],
                                                   expected, delta=3e-13)

    def test_quotient_calendar_chain_matches_explicit_slot_graph(self):
        capacity, p0, alpha = 2, .4, np.array([.04, .47, .93])
        for sellers in (2, 3):
            game = CountKernel(capacity, sellers, p0, order=32)
            states, full, selection, outside, law = labelled_reference(
                capacity, sellers, p0, tuple(alpha), .35, .8)
            expected, membership = quotient_from_labels(states, full, game.states)
            np.testing.assert_allclose(game.transition(alpha).toarray(), expected,
                                       atol=3e-13, rtol=0)
            np.testing.assert_allclose(full.sum(axis=1), 1, atol=3e-13, rtol=0)
            np.testing.assert_allclose(game.orbit_index(states), membership, atol=0, rtol=0)
            np.testing.assert_array_equal(np.bincount(membership), game.orbit_sizes)
            # Every labelled representative in an orbit must have the same
            # aggregate successors, rather than merely agreeing for one row.
            for row in range(len(states)):
                aggregate = np.bincount(membership, weights=full[row], minlength=game.size)
                np.testing.assert_allclose(aggregate, expected[membership[row]],
                                           atol=3e-13, rtol=0)
            quotient_law = np.bincount(membership, weights=law, minlength=game.size)
            own = np.bincount(states[:, 0], weights=law, minlength=capacity+1)
            sold = np.bincount(states[:, 0], weights=law*selection[:, 0], minlength=capacity+1)
            np.testing.assert_allclose(game.own_marginal@quotient_law, own, atol=2e-13, rtol=0)
            np.testing.assert_allclose(game.sale_marginal@quotient_law, sold, atol=2e-13, rtol=0)
            np.testing.assert_allclose(quotient_law@game.transition(alpha), quotient_law,
                                       atol=3e-13, rtol=0)
            np.testing.assert_allclose(quotient_law@game.outside, law@outside,
                                       atol=3e-13, rtol=0)

    def test_analytic_transition_sensitivity_matches_independent_slot_graph(self):
        game = CountKernel(2, 3, .4, order=32)
        alpha = np.array([.04, .47, .93])
        for coordinate in range(3):
            direction = np.eye(3)[coordinate]
            reconstructed = []
            for policy in (alpha-.01*direction, alpha+.01*direction):
                states, full, *_ = labelled_reference(2, 3, .4, tuple(policy), .35, .8)
                reconstructed.append(quotient_from_labels(states, full, game.states)[0])
            derivative = (reconstructed[1]-reconstructed[0])/.02
            np.testing.assert_allclose(game.transition_derivative(coordinate).toarray(),
                                       derivative, atol=4e-13, rtol=0)

    def test_solver_beliefs_match_labelled_calendar_conditioning(self):
        from forgetting_random_replacement.oblivious.code.model import CountExperiment

        capacity, p0, alpha = 2, .6, np.array([.05, .55, .9])
        for closure in ("marginal", "conditional"):
            experiment = CountExperiment(capacity, sellers=2, closure=closure, p0=p0, order=32)
            result = experiment.calculate(alpha)
            states, full, selection, outside, law = labelled_reference(
                capacity, 2, p0, tuple(alpha), .35, .8)
            _, membership = quotient_from_labels(states, full, experiment.game.states)
            quotient_law = np.bincount(membership, weights=law, minlength=experiment.game.size)
            np.testing.assert_allclose(result["joint"], quotient_law, atol=2e-11, rtol=0)
            demand = np.empty(capacity+1)
            for own in range(capacity+1):
                if closure == "conditional":
                    same_count = states[:, 0] == own
                    demand[own] = law[same_count]@selection[same_count, 0]/law[same_count].sum()
                else:
                    demand[own] = sum(
                        probability*adaptive_selection(capacity, own, (int(state[1]),), p0)
                        for state, probability in zip(states, law))
            np.testing.assert_allclose(result["demand"], demand, atol=2e-11, rtol=0)

    def test_monopoly_matches_existing_exact_full_startup_solver_at_maturity(self):
        from forgetting_random_replacement.oblivious.code.model import CountExperiment

        for capacity, p0 in ((2, .1), (3, .5), (10, .9)):
            primitives = replace(Primitives(), p0=p0)
            existing = solve_counts(primitives, capacity, tolerance=1e-12)
            result = CountExperiment(capacity, sellers=1, closure="monopoly", p0=p0,
                                     primitives=primitives).solve()
            self.assertTrue(result["summary"]["verified"])
            np.testing.assert_allclose(result["value"], existing.values[-1], atol=2e-8, rtol=0)
            np.testing.assert_allclose(result["advantage"], existing.advantages[-1], atol=2e-8, rtol=0)
            active = abs(existing.advantages[-1]) > 2e-8
            np.testing.assert_allclose(result["policy"][active],
                                       (existing.advantages[-1][active] > 0), atol=1e-10, rtol=0)

    def test_actual_discounted_value_matches_labelled_calendar_evaluation(self):
        from forgetting_random_replacement.oblivious.code.model import CountExperiment

        p = replace(Primitives(), p0=.5)
        capacity, sellers = 3, 2
        for closure in ("marginal", "conditional"):
            experiment = CountExperiment(capacity, sellers=sellers, closure=closure,
                                         p0=p.p0, primitives=p, order=32)
            result = experiment.solve()
            self.assertTrue(result["summary"]["verified"])
            alpha = np.asarray(result["policy"])
            states, transition, selection, outside, law = labelled_reference(
                capacity, sellers, p.p0, tuple(alpha), p.p1, p.p2)
            individual_reward = selection*(p.revenue-p.c1-(p.c2-p.c1)*alpha[states])
            labelled_values = np.linalg.solve(np.eye(len(states))-p.gamma*transition,
                                              individual_reward)
            _, membership = quotient_from_labels(states, transition, experiment.game.states)
            quotient_values = np.empty(experiment.game.size)
            for orbit in range(experiment.game.size):
                members = membership == orbit
                quotient_values[orbit] = labelled_values[members].mean()
            np.testing.assert_allclose(result["actual_value"], quotient_values,
                                       atol=3e-8, rtol=0)
            initial = np.all((states == capacity//2) | (states == capacity//2+1), axis=1)
            initial_value = labelled_values[initial, 0].mean()
            self.assertAlmostEqual(result["summary"]["value_initial"], initial_value, delta=3e-8)
            sale_rate = law@selection[:, 0]
            product2_calendar = law@(selection[:, 0]*alpha[states[:, 0]])
            self.assertAlmostEqual(result["summary"]["sale_rate"], sale_rate, delta=2e-10)
            self.assertAlmostEqual(result["summary"]["product2_conditional"],
                                   product2_calendar/sale_rate, delta=2e-10)
            self.assertAlmostEqual(result["summary"]["outside_rate"], law@outside, delta=2e-10)


if __name__ == "__main__":
    unittest.main()
