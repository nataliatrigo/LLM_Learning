"""Independent transient-law and finite-window bookkeeping checks."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.sparse import eye

from oblivious.oe_p0.code.game import Game
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.equilibrium_comparison.code.inputs import predictions


from oblivious.equilibrium_comparison.code import validate_transients as module


def geometric_chain_pair():
    """Center moves to (1,1) with probability 1/2 each period and stays there."""
    grid = np.array([0., .5, 1.])
    params = Parameters()
    transition = eye(9, format='lil')
    transition[4, 4] = .5
    transition[4, 8] = .5
    nu = np.zeros((3, 3)); nu[2, 2] = 1.
    pair = dict(params=params, grid=grid, game=Game(grid, params), profiles={})
    for key in ('oe', 'p0'):
        pair['profiles'][key] = dict(grid=grid, policy=np.zeros(3), nu=nu.copy(),
            p0=(1-params.rho+1)/(3-2*params.rho), label=key, transition=transition.tocsr())
        pair['profiles'][key]['predicted'] = predictions(pair, key)
    return pair


class TransientValidationTests(unittest.TestCase):
    def test_window_uses_before_state_and_true_geometric_transient(self):
        pair = geometric_chain_pair()
        result = module.validate_transients(pair, {'first': (0, 3), 'skip': (1, 2)}, checkpoints=(0, 1, 3))
        for profile in result['profiles'].values():
            self.assertAlmostEqual(profile['windows']['first']['expected_metrics']['mean_reputation_B'], 17/24)
            self.assertAlmostEqual(profile['windows']['skip']['expected_metrics']['mean_reputation_B'], 13/16)
            self.assertAlmostEqual(profile['checkpoints']['1']['metrics']['mean_reputation_B'], .75)
            self.assertAlmostEqual(profile['checkpoints']['3']['metrics']['mean_reputation_B'], .9375)
            self.assertAlmostEqual(profile['checkpoints']['3']['l1_distance_to_stationary'], .25)
            self.assertEqual(profile['final_probability_sum'], 1.)
            self.assertEqual(profile['saved_stationarity_residual_max'], 0.)
            self.assertEqual(profile['windows']['first']['expected_metrics']['product2_market'], 0.)
        json.dumps(result, allow_nan=False)

    def test_ci_comparison_distinguishes_stationary_from_finite_expectation(self):
        pair = geometric_chain_pair()
        rows = [dict(mode='grid', profile='p0', window='first', metric='mean_reputation_B',
                     mean=.71, se=.02, ci_low=.67, ci_high=.75, n_valid=20)]
        with tempfile.TemporaryDirectory() as folder:
            result = module.write_validation(pair, rows, folder, {'first': (0, 3)}, checkpoints=(3,))
            row = next(r for r in result['monte_carlo_comparisons'] if r['profile']=='p0' and r['metric']=='mean_reputation_B')
            self.assertTrue(row['ci_contains_finite_expectation'])
            self.assertFalse(row['ci_contains_stationary_prediction'])
            self.assertAlmostEqual(row['initial_condition_bias'], 17/24-1)
            for suffix in ('json', 'csv', 'txt'):
                self.assertTrue((Path(folder)/('transient_validation.'+suffix)).is_file())

    def test_single_profile_and_invalid_window(self):
        pair = geometric_chain_pair()
        pair['profiles'] = {'p0': pair['profiles']['p0']}
        result = module.validate_transients(pair, {'one': (0, 1)}, checkpoints=())
        self.assertEqual(set(result['profiles']), {'p0'})
        with self.assertRaises(ValueError):
            module.validate_transients(pair, {'bad': (0, 0)})


if __name__ == '__main__':
    unittest.main()
