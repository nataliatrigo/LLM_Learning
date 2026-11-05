"""Checks for the sparse positive stationary-law kernel and rare-state laws."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix, diags, eye, kron

from oblivious.oe_p0.code.core import _jump_system, checked_joint_stationary
from oblivious.oe_p0.code.game import Game
from oblivious.oe_p0.code.positive_stationary import (
    dense_positive_invariant, positive_invariant,
)
from oblivious.oe_p0.code.primitives import Parameters


class PositiveStationaryChecks(unittest.TestCase):
    def test_sparse_matches_dense_for_directed_graphs_and_permutations(self):
        rng = np.random.default_rng(24601)
        for n in (2, 9, 31):
            weights = rng.random((n, n)) * (rng.random((n, n)) < .15)
            weights[np.arange(n), (np.arange(n) + 1) % n] += .1
            np.fill_diagonal(weights, 0)
            weights /= weights.sum(axis=1)[:, None]
            jump = csr_matrix(weights)
            rates = np.exp(rng.uniform(-12, 0, n))
            reference = dense_positive_invariant(jump, rates)
            for perm in (None, np.arange(n)[::-1], rng.permutation(n)):
                with self.subTest(n=n, permutation=None if perm is None else perm.tolist()):
                    actual = positive_invariant(jump, rates, perm)
                    assert_allclose(actual, reference, atol=0, rtol=3e-13)
                    assert_allclose(actual, dense_positive_invariant(jump, rates, perm),
                                    atol=0, rtol=3e-13)
                    self.assertAlmostEqual(actual.sum(), 1., places=14)

    def test_nearly_reducible_birth_death_retains_rare_masses(self):
        # Adjacent rare flows remain representable; subtracting from a unit
        # diagonal would erase these 1e-12 flows. pi spans 240 decades.
        n = 21
        target = np.logspace(-240, 0, n)
        target /= target.sum()
        # Orient the 1e-12 rate toward the rare state, the .25 rate away.
        up = np.full(n - 1, .25)
        down = .25 * target[:-1] / target[1:]
        off = diags((down, up), (-1, 1), shape=(n, n), format="csr")
        rates = np.asarray(off.sum(axis=1)).ravel()
        jump = diags(1 / rates) @ off
        for permutation in (None, np.arange(n)[::-1], np.random.default_rng(7).permutation(n)):
            actual = positive_invariant(jump, rates, permutation)
            self.assertTrue(np.all(actual > 0))
            assert_allclose(actual, target, atol=0, rtol=1e-12)

    def test_checked_joint_law_checks_tiny_supported_conditional_rows(self):
        n = 5
        marginal = np.array([1e-150, 1e-90, 1e-20, .3, .7])
        refresh = csr_matrix(np.tile(marginal, (n, 1)))
        identity = eye(n, format="csr")
        matrix = .5 * (kron(refresh, identity) + kron(identity, refresh))
        game = Game(np.linspace(0, 1, n), Parameters(rho=.99), order=32)
        answer = checked_joint_stationary(game, matrix, force_positive=True)
        actual = answer["nu"]
        exact = np.outer(marginal, marginal)
        self.assertTrue(np.all(actual > 0))
        assert_allclose(actual, exact, atol=0, rtol=2e-12)
        conditioned = actual / actual.sum(axis=1)[:, None]
        assert_allclose(conditioned, np.broadcast_to(marginal, actual.shape),
                        atol=0, rtol=2e-12)

    def test_compiler_unavailable_has_small_dense_fallback(self):
        jump = csr_matrix([[0., 1.], [1., 0.]])
        rates = np.array([.2, .8])
        with patch("oblivious.oe_p0.code.positive_stationary._kernel", side_effect=RuntimeError("no compiler")):
            with self.assertWarns(RuntimeWarning):
                answer = positive_invariant(jump, rates, np.array([1, 0]))
        assert_allclose(answer, [.8, .2], atol=1e-15)

    def test_invalid_permutation_and_reducibility_fail_explicitly(self):
        jump = csr_matrix([[0., 1.], [1., 0.]])
        with self.assertRaises(ValueError):
            positive_invariant(jump, np.ones(2), [0, 0])
        with self.assertRaises(ArithmeticError):
            positive_invariant(eye(2, format="csr"), np.ones(2))

    def test_saved_n51_policy_matches_dense_and_every_conditional_row(self):
        archive = (Path(__file__).resolve().parents[3] / "oblivious/04_oe_conditional_distribution" / "data" /
                   "rho_forgetting" / "arrays" / "rho_0p99_n051.npz")
        if not archive.exists():
            self.skipTest("Saved n51 reference is not present in this checkout")
        with np.load(archive) as saved:
            grid = saved["root/selected/grid"]
            policy = saved["root/selected/policy"]
        game = Game(grid, Parameters(rho=.99), order=128)
        sigma = np.broadcast_to(policy[:, None], (game.n, game.n))
        matrix = game.transition(sigma, sigma.T)
        i, j = np.triu_indices(game.n)
        groups = np.empty((game.n, game.n), dtype=int)
        groups[i, j] = np.arange(len(i))
        groups[j, i] = np.arange(len(i))
        membership = csr_matrix((np.ones(game.n ** 2),
                                 (np.arange(game.n ** 2), groups.ravel())),
                                shape=(game.n ** 2, len(i)))
        jump, rates = _jump_system((matrix[i * game.n + j] @ membership).tocsr())
        reference = dense_positive_invariant(jump, rates)
        for permutation in (None, np.arange(len(rates))[::-1]):
            actual = positive_invariant(jump, rates, permutation)
            assert_allclose(actual, reference, atol=0, rtol=2e-12)
            laws = []
            for law in (reference, actual):
                nu = np.zeros((game.n, game.n))
                shares = law / np.where(i == j, 1., 2.)
                nu[i, j] = shares
                nu[j, i] = shares
                laws.append(nu / nu.sum(axis=1)[:, None])
            self.assertLess(np.abs(laws[0] - laws[1]).sum(axis=1).max(), 1e-12)


if __name__ == "__main__":
    unittest.main()
