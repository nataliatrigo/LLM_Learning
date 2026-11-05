"""Guard 201/251 campaign isolation and checked policy-only warm starts."""

from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code import sweep201_run as worker
from oblivious.oe_distribution_fixed_p0.code import sweep201_batch as batch
from oblivious.oe_distribution_fixed_p0.code.refine_case import atomic_bundle


class Sweep201Tests(unittest.TestCase):
    def test_new_rho_and_difficult_outside_values_run_first(self):
        jobs = batch.ordered_jobs(worker.P0S,worker.RHOS)
        self.assertEqual(jobs[:3],[(.8,.9),(.9,.9),(.1,.9)])
        self.assertTrue(all(rho == .9 for _,rho in jobs[:6]))
        self.assertEqual(len(jobs),len(set(jobs)))
        self.assertEqual(set(jobs),{(p0,rho) for p0 in worker.P0S for rho in worker.RHOS})

    def test_point_five_is_fresh_and_campaign_is_isolated(self):
        args = worker.arguments(['--p0','.5','--rho','.9'])
        self.assertEqual(args.grids,[201,251])
        self.assertEqual(worker.campaign_directory(.5,.9),
                         worker.runner.HOME/'data/p0_sweep_201/p0_0p5/rho_0p9')
        with self.assertRaises(SystemExit):
            worker.arguments(['--p0','.5','--rho','.9','--grids','101'])

    def test_same_p0_seed_checks_digest_and_exact_target(self):
        runner = worker.runner
        args = SimpleNamespace(p0=.8,rho=.9,order=256,tol=1e-7,dp_tol=1e-10)
        with TemporaryDirectory(dir=runner.HOME/'data') as temporary:
            home = Path(temporary)
            directory = home/'data/p0_sweep/p0_0p8/rho_0p9'
            (directory/'arrays').mkdir(parents=True)
            tag = 'rho_0p9_n101_marginal'
            policy = dict(grid=np.linspace(0,1,5),policy=np.full(5,.3),verified=True,
                          actual_values=np.full((2,5,5),987654.))
            case = dict(p0=.8,rho=.9,n=101,
                        configuration=dict(parameters=asdict(Parameters(rho=.9))),
                        competition=policy)
            digests = atomic_bundle(directory,tag,case)
            manifest = dict(configuration=dict(p0=.8,rho=.9,order=256,tol=1e-7,dp_tol=1e-10,
                            parameters={k:v for k,v in asdict(Parameters()).items() if k != 'rho'}),
                            numerical_sources=runner.source_fingerprints(),completed=[tag],
                            checkpoint_sha256=digests,reference_input_sha256={})
            runner.write_json(directory/'manifest.json',manifest)
            with patch.object(runner,'HOME',home), \
                    patch.object(runner,'source_fingerprints',return_value=manifest['numerical_sources']):
                seed,inputs = worker.checked_same_p0_seed(args,201,'marginal')
                np.testing.assert_array_equal(seed['policy'],policy['policy'])
                self.assertEqual(set(seed),{'grid','policy'})
                self.assertIn(str(Path(worker.__file__).resolve().relative_to(runner.ROOT)),inputs)
                manifest['configuration']['p0'] = .7
                runner.write_json(directory/'manifest.json',manifest)
                with self.assertRaisesRegex(ArithmeticError,'calibration differs'):
                    worker.checked_same_p0_seed(args,201,'marginal')
                manifest['configuration']['p0'] = .8
                runner.write_json(directory/'manifest.json',manifest)
                path = directory/'arrays'/(tag+'.json')
                path.write_text(path.read_text()+' ')
                with self.assertRaisesRegex(ArithmeticError,'Checkpoint changed'):
                    worker.checked_same_p0_seed(args,201,'marginal')

    def test_point_five_seed_uses_checked_baseline_loader(self):
        args = SimpleNamespace(p0=.5,rho=.8)
        sentinel = dict(grid=np.array([.5]),policy=np.array([.123]),value_initial=999999.)
        with patch.object(worker.runner,'_sweep201_original_baseline_seed',create=True,
                          return_value=(sentinel,{})) as loader:
            proposal,inputs = worker.checked_same_p0_seed(args,251,'conditional')
        loader.assert_called_once_with(.8,251,'conditional')
        self.assertEqual(set(proposal),{'grid','policy'})
        np.testing.assert_array_equal(proposal['policy'],sentinel['policy'])
        self.assertTrue(inputs)


if __name__ == '__main__':
    unittest.main()
