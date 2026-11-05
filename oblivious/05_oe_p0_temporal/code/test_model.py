"""Finite-window closure checks independent of the stationary benchmark."""

import unittest

import numpy as np

from oblivious.oe_p0.code.core import posterior_mean
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0_temporal.code.model import TemporalExperiment


class TemporalModelTests(unittest.TestCase):
    def test_one_period_is_initial_posterior_not_stationary_mean(self):
        experiment = TemporalExperiment(n=5, params=Parameters(rho=.6, gamma=0),
                                        periods=1, burn_in=0, order=64)
        result = experiment.evaluate_mixed(.5, np.zeros(5))
        self.assertTrue(result["candidate_verified"])
        self.assertAlmostEqual(result["F"], .5, places=14)
        self.assertGreater(abs(result["stationary_F"] - result["F"]), .01)
        self.assertAlmostEqual(result["mu_b"] @ posterior_mean(experiment.grid, experiment.params),
                               result["stationary_F"], places=14)
        self.assertAlmostEqual(result["temporal_nu"][2,2], 1.)

    def test_two_period_analytical_mean_and_burn_in(self):
        parameters = Parameters(rho=.6, gamma=0)
        # Both start at .5; B sells with probability 1/2. The projection
        # preserves the expectation of reputation exactly.
        next_reputation = .5 + .5 * (1-parameters.rho) * (parameters.p1-.5)
        next_posterior = float(posterior_mean(next_reputation, parameters))
        first = TemporalExperiment(n=5, params=parameters, periods=2, burn_in=0, order=64)
        second = TemporalExperiment(n=5, params=parameters, periods=1, burn_in=1, order=64)
        self.assertAlmostEqual(first._pure_probe(.5)["F"], (.5+next_posterior)/2, places=13)
        self.assertAlmostEqual(second._pure_probe(.5)["F"], next_posterior, places=13)

    def test_pure_search_requires_fresh_certificate_and_rejects_bad_mix(self):
        experiment = TemporalExperiment(n=5, params=Parameters(rho=.6, gamma=0),
                                        periods=2, burn_in=1, order=64)
        search = experiment.search(p0_points=5)
        self.assertEqual(len(search["candidates"]), 1)
        candidate = search["candidates"][0]
        self.assertTrue(candidate["candidate_verified"])
        self.assertTrue(candidate["stationary_diagnostics"]["verified"])
        self.assertTrue(candidate["temporal_diagnostics"]["verified"])
        self.assertLess(abs(candidate["G"]), 1e-12)
        bad = experiment.evaluate_mixed(candidate["p0"], np.full(5,.5))
        self.assertFalse(bad["verified"])
        self.assertFalse(bad["candidate_verified"])
        with self.assertRaises(ValueError):
            experiment.search(pure_seed={})

    def test_detected_mixture_matches_independent_full_chain_average(self):
        experiment = TemporalExperiment(n=11, params=Parameters(rho=.95),
                                        periods=1000, burn_in=100)
        search = experiment.search(p0_points=21)
        mixed = [candidate for candidate in search["candidates"] if candidate["mixed_nodes"]]
        self.assertTrue(mixed)
        candidate = mixed[0]
        alpha = candidate["policy"]
        kernel = experiment.game.transition(np.broadcast_to(alpha[:,None], (11,11)),
                                            np.broadcast_to(alpha[None,:], (11,11))).T
        law = np.zeros(121); law[60] = 1.
        average = np.zeros(121)
        for period in range(1100):
            if period >= 100:
                average += law/1000
            law = kernel @ law
        independent = average.reshape(11,11).sum(axis=0) @ posterior_mean(experiment.grid, experiment.params)
        self.assertLess(abs(independent-candidate["p0"]), 1e-7)
        self.assertLess(np.abs(average.reshape(11,11)-candidate["temporal_nu"]).sum(), 1e-10)
        self.assertLess(candidate["dp_diagnostics"]["mixed_indifference_residual"], 1e-10)
        self.assertTrue(candidate["candidate_verified"])


if __name__ == "__main__":
    unittest.main()
