"""Guard target-p0 recomputation, source proposals and resume integrity."""

from dataclasses import asdict
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.stats import beta

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code import multi_p0_run as runner


class MultiP0RunnerTests(unittest.TestCase):
    def test_target_p0_is_resolved_again_against_analytic_myopic_values(self):
        # A deliberately wrong historical all-product2 policy is a proposal,
        # not a target-p0 solution. In this analytic myopic benchmark each
        # newly certified model must instead choose product1 at every state.
        params = Parameters(rho=.6, gamma=0.)
        n = 7
        seed = dict(grid=np.linspace(0, 1, n), policy=np.ones(n))
        initial_sales = []
        for p0 in (.1, .3, .7, .8):
            with self.subTest(p0=p0), TemporaryDirectory(dir=runner.HOME/'data') as temporary:
                directory = Path(temporary)
                (directory/'arrays').mkdir()
                (directory/'provenance').mkdir()
                manifest = dict(configuration=dict(parameters={key:value for key,value in asdict(params).items()
                                                               if key != 'rho'}),
                                numerical_sources=runner.source_fingerprints(), requested_grids=[n],
                                completed=[], checkpoint_sha256={}, reference_input_sha256={}, failures={})
                args = SimpleNamespace(p0=p0, rho=.6, order=128, tol=1e-7, dp_tol=1e-10,
                                       max_evaluations=100)
                with patch.object(runner, 'baseline_seed', return_value=(seed, {})):
                    marginal = runner.run_grid(args, directory, manifest, n, 'marginal', lambda _:None)
                    conditional = runner.run_grid(args, directory, manifest, n, 'conditional', lambda _:None)
                shape = 1+.5/(1-params.rho)
                cdf = beta.cdf(p0, shape, shape)
                expected = {'monopoly':.95*(1-cdf), 'competition':.95*(1-cdf*cdf)/2,
                            'conditional_competition':.95*(1-cdf*cdf)/2}
                for case in (marginal, conditional):
                    self.assertEqual(case['p0'], p0)
                    for row in case['outcomes']:
                        self.assertAlmostEqual(row['value_initial'], expected[row['model']], places=12)
                        self.assertEqual(row['product2_conditional'], 0.)
                        sellers = 1 if row['model'] == 'monopoly' else 2
                        self.assertAlmostEqual(row['outside_rate']+sellers*row['sale_rate'], 1., places=11)
                np.testing.assert_array_equal(seed['policy'], np.ones(n))
                initial_sales.append(marginal['outcomes'][0]['value_initial'])
                # A committed checkpoint is re-read only through its digest.
                tag = manifest['completed'][0]
                artifact = directory/'arrays'/(tag+'.json')
                artifact.write_text(artifact.read_text()+' ')
                with self.assertRaisesRegex(ArithmeticError, 'Checkpoint changed'):
                    runner.run_grid(args, directory, manifest, n, 'marginal', lambda _:None)
        self.assertTrue(all(a > b for a,b in zip(initial_sales, initial_sales[1:])))

    def test_resume_rejects_another_p0_before_skipping_completed_work(self):
        args = SimpleNamespace(p0=.7, rho=.6, order=256, tol=1e-7, dp_tol=1e-10,
                               grids=[51,101], max_evaluations=250, resume=True)
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory/'manifest.json').write_text(json.dumps(dict(configuration=dict(p0=.3))))
            with patch.object(runner, 'arguments', return_value=args), \
                    patch.object(runner, 'campaign_directory', return_value=directory):
                with self.assertRaisesRegex(ArithmeticError, 'Saved numerical calibration differs'):
                    runner.main()


if __name__ == '__main__':
    unittest.main()
