"""Event-level and distributional checks for the comparison simulations."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from oblivious.oe_p0.code.game import Game
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.equilibrium_comparison.code.simulation import policy_probability, simulate_pair


def example_pair(policy=None, *, rho=.95):
    grid = np.linspace(0, 1, 11)
    if policy is None:
        policy = np.array([0., 0., .3, 1., 1., .37, 0., 0., 1., .8, 0.])
    return {
        "params": Parameters(rho=rho), "grid": grid, "order": 64,
        "profiles": {
            "oe": {"grid": grid, "policy": np.asarray(policy)},
            "p0": {"grid": grid, "policy": np.asarray(policy).copy()},
        },
    }


class PolicyExtensionTests(unittest.TestCase):
    def test_extensions_preserve_every_stored_mixture_and_node(self):
        pair = example_pair()
        grid, policy = pair["grid"], pair["profiles"]["oe"]["policy"]
        for mode in ("grid", "nearest", "linear"):
            assert_array_equal(policy_probability(grid, policy, grid, mode), policy)
        values = policy_probability(grid, policy, np.linspace(0, 1, 203), "nearest")
        self.assertTrue(set(np.unique(values)).issubset(set(policy)))
        self.assertIn(.37, values)

    def test_nearest_ties_lower_and_linear_is_explicit_additional_mixing(self):
        grid, policy = np.array([0., .5, 1.]), np.array([0., 1., 0.])
        assert_array_equal(policy_probability(grid, policy, [.25, .75], "nearest"), [0., 1.])
        assert_array_equal(policy_probability(grid, policy, [.25, .75], "linear"), [.5, .5])
        with self.assertRaises(ValueError):
            policy_probability(grid, policy, [.25], "grid")
        for states in ([np.nan], [-.1], [1.01]):
            with self.assertRaises(ValueError):
                policy_probability(grid, policy, states)


class SimulationTests(unittest.TestCase):
    def test_recorded_events_obey_economic_accounting_in_all_modes(self):
        pair = example_pair()
        params = pair["params"]
        for mode in ("grid", "nearest", "linear"):
            result = simulate_pair(pair, periods=80, burn_in=20, replications=3,
                                   short_periods=25, mode=mode, seed=17)
            raw = result["profiles"]["oe"]
            selected = raw["selected"]
            chosen = selected[..., None]
            unchosen = (1 - selected)[..., None]
            before_own = np.take_along_axis(raw["before"], chosen, axis=2)[..., 0]
            after_own = np.take_along_axis(raw["after"], chosen, axis=2)[..., 0]
            assert_array_equal(np.take_along_axis(raw["before"], unchosen, axis=2),
                               np.take_along_axis(raw["after"], unchosen, axis=2))
            assert_array_equal(raw["before"][:, 1:], raw["after"][:, :-1])
            assert_array_equal(raw["before"][:, 0], np.full((3, 2), .5))
            assert_array_equal(selected, (raw["theta"][..., 0] <= raw["theta"][..., 1]))
            assert_array_equal(np.take_along_axis(raw["profit"], unchosen, axis=2), 0.)
            actual_market_profit = raw["profit"].sum(axis=2)
            assert_allclose(actual_market_profit,
                            params.R - np.where(raw["product"] == 2, params.c2, params.c1))
            self.assertTrue(np.all((raw["after"] >= 0) & (raw["after"] <= 1)))
            exact_target = params.rho * before_own + (1 - params.rho) * raw["success"]
            if mode == "grid":
                self.assertTrue(np.isin(raw["after"], pair["grid"]).all())
                self.assertLessEqual(np.abs(after_own - exact_target).max(), .1 + 1e-15)
            else:
                assert_array_equal(after_own, exact_target)
                self.assertTrue(np.any(~np.isin(raw["after"], pair["grid"])))
            self.assertEqual(result["metadata"]["primary_window"], [20, 100])

    def test_common_uniforms_give_identical_paths_for_identical_policies(self):
        pair = example_pair()
        for mode in ("grid", "nearest", "linear"):
            result = simulate_pair(pair, periods=90, replications=3, burn_in=10,
                                   short_periods=10, seed=982, mode=mode)
            for key, values in result["profiles"]["oe"].items():
                assert_array_equal(values, result["profiles"]["p0"][key])
            self.assertFalse(np.array_equal(result["uniforms"][0], result["uniforms"][1]))

    def test_seed_reproduction_and_horizon_extension_preserve_time_prefix(self):
        pair = example_pair()
        arguments = dict(periods=20, burn_in=5, replications=2,
                         short_periods=10, mode="nearest", seed=753)
        first = simulate_pair(pair, **arguments)
        repeat = simulate_pair(pair, **arguments)
        longer = simulate_pair(pair, **arguments, total_periods=50)
        for key in first["profiles"]["oe"]:
            assert_array_equal(first["profiles"]["oe"][key], repeat["profiles"]["oe"][key])
            assert_array_equal(first["profiles"]["oe"][key], longer["profiles"]["oe"][key][:, :25])
        assert_array_equal(first["uniforms"], longer["uniforms"][:, :25])
        self.assertEqual(first["replication_seeds"], longer["replication_seeds"])

    def test_one_step_law_matches_the_solver_kernel_and_samples_mixtures(self):
        # A large independent one-step ensemble tests the entire joint kernel
        # row, not a serially correlated trajectory. Non-node targets exercise
        # stochastic projection; policy .37 must remain genuinely mixed.
        pair = example_pair(np.full(11, .37), rho=.65)
        count = 40_000
        result = simulate_pair(pair, periods=1, burn_in=0, short_periods=0,
                               replications=count, seed=18427, mode="grid",
                               assignments={"oe": ("oe", "oe")})
        raw = result["profiles"]["oe"]
        game = Game(pair["grid"], pair["params"], order=64)
        policy = np.broadcast_to(pair["profiles"]["oe"]["policy"][:, None], (11, 11))
        row = game.transition(policy, policy.T).getrow(5 * 11 + 5).toarray().ravel()
        states = np.searchsorted(pair["grid"], raw["after"][:, 0])
        observed = np.bincount(states[:, 0] * 11 + states[:, 1], minlength=121) / count
        tolerance = 6 * np.sqrt(row * (1 - row) / count) + 1 / count
        self.assertTrue(np.all(np.abs(observed - row) <= tolerance))
        self.assertLess(abs(np.mean(raw["product"] == 2) - .37), .015)
        self.assertLess(abs(np.mean(raw["selected"] == 0) - .5), .015)
        # In the economic game a selected seller pays for its product even
        # on failure; averaging event profits must obey this exact identity.
        assert_allclose(raw["profit"].sum(axis=2),
                        .95 - .60 * (raw["product"] == 2), atol=1e-15)

    def test_pure_policies_cross_assignments_and_input_checks(self):
        pair = example_pair(np.zeros(11))
        pair["profiles"]["p0"]["policy"] = np.ones(11)
        result = simulate_pair(pair, periods=30, burn_in=0, short_periods=10,
                               replications=2, assignments={"cross": ("oe", "p0")})
        raw = result["profiles"]["cross"]
        assert_array_equal(raw["product"], raw["selected"] + 1)
        with self.assertRaises(ValueError):
            simulate_pair(pair, initial=(.47, .5), periods=1, short_periods=0)
        with self.assertRaises(ValueError):
            simulate_pair(pair, periods=10, burn_in=10, total_periods=10)
        with self.assertRaises(ValueError):
            simulate_pair(pair, periods=1.5)


if __name__ == "__main__":
    unittest.main()
