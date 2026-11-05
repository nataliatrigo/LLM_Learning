"""Check that prefix plots use actual early states and preserve original histories."""
import unittest
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from forgetting.Code.experiments.interaction_paths import simulate
from forgetting.Code.model import Primitives
from forgetting.Code.experiments.states import replay_prefix
from forgetting.__main__ import COMMANDS, main


class InitialStateTests(unittest.TestCase):
    def test_long_run_continuation_matches_existing_simulator(self):
        from forgetting.Code.layers import solve_observation_layers
        from forgetting.Code.experiments.policy_summary import continue_customer

        solver = dict(nodes=101, terminal_gap=.01, tolerance=1e-9,
                      action_tolerance=1e-6, max_iterations=100000)
        primitives = Primitives()
        records = replay_prefix(primitives, .8, solver, seed=17,
                                original_paths=3, paths=5, count=80)
        _, policy = solve_observation_layers(primitives, .8, **solver, return_policy=True)
        meta = dict(rho=.8, seller=primitives.as_dict(), seed=17,
                    original_paths=3, shape=[80, 5], policy_solver=solver)
        for prefix_length in (10, 30):
            with self.subTest(prefix_length=prefix_length):
                table = policy['advantage_table']
                if prefix_length >= policy['terminal_observation']:
                    table = table[-1]
                result = continue_customer(meta, records[:prefix_length, 0], table,
                    policy['coordinate'], policy['terminal_observation'],
                    interactions=80, burn_in=60)
                states = records['x_before'][60:, 0] / (1-.8)
                self.assertAlmostEqual(result['S_min'], states.min())
                self.assertAlmostEqual(result['S_max'], states.max())
                self.assertEqual(result['min_at_interaction'], 61+states.argmin())
                self.assertEqual(result['max_at_interaction'], 61+states.argmax())
                self.assertEqual(result['late_product2_count'], records['product2'][60:, 0].sum())

    def test_every_command_exposes_help_without_running_experiments(self):
        for command in COMMANDS:
            with self.subTest(command=command), redirect_stdout(StringIO()) as output:
                with self.assertRaises(SystemExit) as result:
                    main([command, '--help'])
                self.assertEqual(result.exception.code, 0)
                self.assertIn('usage:', output.getvalue())

    def test_initial_command_reuses_records_and_counts_only_customer_one(self):
        with TemporaryDirectory() as temporary:
            source = Path(temporary) / 'source'
            output = Path(temporary) / 'output'
            directory = source / 'data' / 'p0_0.5_rho_0.8'
            directory.mkdir(parents=True)
            records = np.zeros((10, 2), dtype=[
                ('x_before', 'f8'), ('product2', '?'), ('success', '?')])
            records['x_before'][:, 0] = np.linspace(0, .8, 10)
            records['x_before'][:, 1] = .9
            np.save(directory / 'observations.npy', records)
            manifest = {
                'shape': [10, 2], 'burn_in': 0, 'seed': 17, 'p0': .5,
                'rho': .8, 'seller': Primitives().as_dict(), 'policy_solver': {},
            }
            (directory / 'records_manifest.json').write_text(json.dumps(manifest))
            with redirect_stdout(StringIO()):
                main(['initial', '--source', str(source), '--output', str(output),
                      '--count', '8', '--paths', '2', '--reuse-records'])
            result_directory = output / 'data' / directory.name
            np.testing.assert_array_equal(
                np.load(result_directory / 'observations.npy'), records[:8])
            histogram = np.loadtxt(result_directory / 'state_histogram.csv',
                                   delimiter=',', skiprows=1)
            self.assertEqual(histogram[:, 2].sum(), 8)
            self.assertAlmostEqual(histogram[-1, 1], 5.)
            self.assertEqual(len(list((output / 'figures').rglob('*.png'))), 3)
            np.testing.assert_array_equal(
                np.load(directory / 'observations.npy'), records)

    def test_replay_preserves_source_trajectories_when_adding_paths(self):
        solver = dict(nodes=101, terminal_gap=.1, tolerance=1e-9,
                      action_tolerance=1e-6, max_iterations=100000)
        for original_paths in (1, 3):
            with self.subTest(original_paths=original_paths):
                source = simulate(Primitives(), .8, paths=original_paths,
                                  interactions=25, seed=17, trace_window=(1, 25), **solver)
                records = replay_prefix(Primitives(), .8, solver, seed=17,
                                        original_paths=original_paths, paths=5, count=25)
                np.testing.assert_array_equal(records['product2'][:, :original_paths], source['choices'])
                for trajectory, trace in source['trace'].groupby('trajectory'):
                    np.testing.assert_array_equal(records['x_before'][:, trajectory-1], trace.x_before)
                np.testing.assert_array_equal(records['x_before'][0], np.zeros(5))
                np.testing.assert_allclose(records['x_before'][1:],
                    .8*records['x_before'][:-1] + (1-.8)*records['success'][:-1])
                counts, _ = np.histogram(records['x_before'], np.linspace(0, 1, 101))
                self.assertEqual(counts.sum(), 25*5)


if __name__ == '__main__':
    unittest.main()
