"""Independent finite-horizon checks: timing, reducibility, and pair lifting."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from oblivious.oe_p0.code.game import Game
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0_temporal.code.occupation import (
    finite_window_law, finite_window_occupation, lift_symmetric_pair_law,
    symmetric_pair_kernel,
)


def dense_power_reference(matrix, initial, periods, burn_in):
    """Independent dense powers, including nonstationary finite windows."""
    laws = [initial @ np.linalg.matrix_power(matrix, t)
            for t in range(burn_in, burn_in + periods)]
    return np.mean(laws, axis=0)


class FiniteWindowTests(unittest.TestCase):
    def test_nonstationary_laws_equal_explicit_dense_powers(self):
        matrix = np.array([[.7, .2, .1], [.1, .6, .3], [.2, .1, .7]])
        initial = np.array([.8, .2, 0.])
        for burn_in in (0, 1, 7):
            for periods in (1, 2, 19):
                with self.subTest(burn_in=burn_in, periods=periods):
                    result = finite_window_occupation(matrix, initial, periods, burn_in,
                                                      tolerance=0, early_stop=False)
                    expected = dense_power_reference(matrix, initial, periods, burn_in)
                    assert_allclose(result["law"], expected, atol=2e-15, rtol=0)
                    assert_allclose(result["burn_in_law"], initial @ np.linalg.matrix_power(matrix, burn_in), atol=2e-15)
                    assert_allclose(result["final_law"], initial @ np.linalg.matrix_power(matrix, burn_in + periods), atol=2e-15)
                    self.assertEqual(result["summary"]["propagated_steps"], burn_in + periods)
                    self.assertEqual(result["summary"]["occupation_truncation_l1_bound"], 0)

    def test_pre_selection_window_excludes_final_transition(self):
        matrix = np.array([[0., 1., 0.], [0., 0., 1.], [0., 0., 1.]])
        initial = np.array([1., 0., 0.])
        first = finite_window_occupation(matrix, initial, periods=2, burn_in=0)
        second = finite_window_occupation(matrix, initial, periods=2, burn_in=1)
        assert_allclose(first["law"], [.5, .5, 0.], atol=0)
        assert_allclose(second["law"], [0., .5, .5], atol=0)
        assert_allclose(first["final_law"], [0., 0., 1.], atol=0)
        self.assertFalse(np.array_equal(first["law"], first["final_law"]))

    def test_identity_keeps_selected_initial_law_and_stops_safely(self):
        initial = np.array([.25, .75, 0.])
        result = finite_window_occupation(np.eye(3), initial, periods=100000, burn_in=10000)
        assert_allclose(result["law"], initial, atol=0)
        assert_allclose(result["burn_in_law"], initial, atol=0)
        assert_allclose(result["final_law"], initial, atol=0)
        self.assertTrue(result["summary"]["early_stopped"])
        self.assertEqual(result["summary"]["propagated_steps"], 1)
        self.assertEqual(result["summary"]["occupation_truncation_l1_bound"], 0)

    def test_periodic_chain_is_not_made_lazy_or_stationary(self):
        result = finite_window_occupation([[0., 1.], [1., 0.]], [1., 0.],
                                         periods=5, burn_in=1)
        assert_allclose(result["law"], [2/5, 3/5], atol=0)
        self.assertFalse(result["summary"]["early_stopped"])
        self.assertEqual(result["summary"]["propagated_steps"], 6)

    def test_reducible_transient_chain_preserves_absorption_weights(self):
        matrix = np.array([[.5, .2, .3], [0., 1., 0.], [0., 0., 1.]])
        initial = np.array([1., 0., 0.])
        for burn_in in (0, 3, 20):
            result = finite_window_occupation(matrix, initial, periods=23, burn_in=burn_in)
            expected = dense_power_reference(matrix, initial, 23, burn_in)
            assert_allclose(result["law"], expected, atol=1e-10, rtol=0)

    def test_tail_error_is_bounded_in_nonstationary_window(self):
        matrix = np.array([[.95, .05], [.1, .9]])
        initial = np.array([1., 0.])
        result = finite_window_occupation(matrix, initial, periods=200, burn_in=0,
                                         tolerance=2e-5)
        full = finite_window_occupation(matrix, initial, periods=200, burn_in=0,
                                       tolerance=0, early_stop=False)
        self.assertTrue(result["summary"]["early_stopped"])
        bound = result["summary"]["occupation_truncation_l1_bound"]
        actual_error = np.sum(np.abs(result["law"] - full["law"]))
        self.assertGreater(actual_error, 1e-12)
        self.assertLessEqual(actual_error, bound + 2e-14)
        self.assertLessEqual(bound, 2e-5)
        self.assertFalse(result["summary"]["arithmetic_roundoff_certified"])
        self.assertIsNone(result["summary"]["arithmetic_roundoff_error_bound"])

    def test_tail_bound_can_cover_unfinished_burn_in(self):
        matrix = np.array([[.8, .2], [.4, .6]])
        result = finite_window_occupation(matrix, [1., 0.], periods=13, burn_in=500,
                                         tolerance=1e-8)
        self.assertTrue(result["summary"]["early_stopped"])
        self.assertLess(result["summary"]["tail_replacement_time"], 500)
        expected = dense_power_reference(matrix, np.array([1., 0.]), 13, 500)
        error = np.sum(np.abs(result["law"] - expected))
        self.assertLessEqual(error, result["summary"]["occupation_truncation_l1_bound"] + 1e-13)

    def test_rejects_invalid_inputs(self):
        for matrix in (np.array([[.5, .1], [0., 1.]]),
                       np.array([[1.1, -.1], [0., 1.]]),
                       np.ones((2, 3))):
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    finite_window_occupation(matrix, [1., 0.])
        for kwargs in (dict(periods=0), dict(burn_in=-1), dict(periods=1.5),
                       dict(tolerance=-1), dict(periods=True)):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    finite_window_occupation(np.eye(2), [1., 0.], **kwargs)


class SymmetricPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = Game(np.linspace(0, 1, 7), Parameters(rho=.8), order=32)
        cls.policy = np.array([0., .1, 1., .3, .8, 0., 1.])

    def test_lumped_propagation_matches_original_joint_kernel(self):
        game, policy = self.game, self.policy
        sigma_a = np.broadcast_to(policy[:, None], (game.n, game.n))
        sigma_b = np.broadcast_to(policy[None, :], (game.n, game.n))
        matrix = game.transition(sigma_a, sigma_b)
        result = finite_window_law(game, policy, periods=21, burn_in=7, tolerance=0)
        initial = np.zeros(game.n ** 2)
        initial[(game.n // 2) * game.n + game.n // 2] = 1
        full = finite_window_occupation(matrix, initial, periods=21, burn_in=7,
                                       tolerance=0, early_stop=False)
        assert_allclose(result["nu"].ravel(), full["law"], atol=3e-16, rtol=0)
        assert_allclose(result["final_law"].ravel(), full["final_law"], atol=3e-16, rtol=0)
        assert_allclose(result["mu_a"], result["mu_b"], atol=2e-16, rtol=0)
        self.assertEqual(result["summary"]["solved_states"], 28)
        self.assertEqual(result["summary"]["full_states"], 49)

    def test_one_step_lifting_is_exact_for_arbitrary_symmetric_law(self):
        game, policy = self.game, self.policy
        matrix = game.transition(np.tile(policy[:, None], (1, game.n)),
                                 np.tile(policy[None, :], (game.n, 1)))
        reduced, membership, pairs, _ = symmetric_pair_kernel(matrix, game.n)
        weights = np.arange(1., reduced.shape[0] + 1)
        weights /= weights.sum()
        lifted = lift_symmetric_pair_law(weights, game.n, pairs)
        assert_allclose(membership.T @ lifted.ravel(), weights, atol=0)
        expected = matrix.T @ lifted.ravel()
        actual = lift_symmetric_pair_law(reduced.T @ weights, game.n, pairs)
        assert_allclose(actual.ravel(), expected, atol=2e-17, rtol=0)

    def test_off_grid_equal_initial_state_uses_product_interpolation(self):
        result = finite_window_law(self.game, self.policy, periods=1, burn_in=0,
                                   initial=(.25, .25), tolerance=0)
        marginal = np.zeros(self.game.n)
        marginal[1:3] = .5
        assert_allclose(result["nu"], np.outer(marginal, marginal), atol=2e-16, rtol=0)
        assert_allclose(result["initial_law"], result["nu"], atol=0)

    def test_rejects_asymmetric_initial_and_kernel(self):
        with self.assertRaises(ValueError):
            finite_window_law(self.game, self.policy, initial=(.2, .3))
        matrix = np.eye(4)
        matrix[1] = [1, 0, 0, 0]
        with self.assertRaises(ValueError):
            symmetric_pair_kernel(matrix, 2)


if __name__ == "__main__":
    unittest.main()
