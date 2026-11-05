"""Checks for independently saved Monte Carlo rho scenarios."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from forgetting.Code.experiments import run_experiments as experiments
from forgetting.Code.experiments.run_monte_carlo import rho_values, run


class MonteCarloRunnerTests(unittest.TestCase):
    def test_rho_validation(self):
        self.assertEqual(rho_values([.8, .4, .8]), [.4, .8])
        for values in ([], [0], [1], [float('nan')], [.5, -1]):
            with self.assertRaises(ValueError):
                rho_values(values)

    def test_scenarios_are_separate_and_match_combined_output(self):
        config = experiments.read_config(experiments.CONFIG_DIR / 'config_policy_comparisons.json')
        config['p0_comparison']['values'] = [.3, .5]
        config['simulation'].update(paths=32, periods=4, bin_width=2,
                                    policy_nodes=101, terminal_gap=.1)
        with TemporaryDirectory() as temporary, \
                patch.multiple(experiments, OUTPUTS=Path(temporary),
                               DATA=Path(temporary), FIGURES=Path(temporary)):
            output = Path(temporary)
            run(config, [.4, .8], output)
            combined = pd.read_csv(output / 'data/monte_carlo_by_rho_by_period.csv')
            self.assertEqual(set(combined.rho), {.4, .8})
            self.assertEqual(set(combined.p0), {.3, .5})
            self.assertEqual(len(combined), 16)
            for rho in (.4, .8):
                directory = output / 'data' / f'rho_{rho}'
                separate = pd.read_csv(directory / 'data/monte_carlo_calendar_by_period.csv')
                pd.testing.assert_frame_equal(separate, combined[combined.rho == rho].reset_index(drop=True))
                paths = pd.read_csv(directory / 'data/monte_carlo_calendar_path_summary.csv')
                self.assertEqual(len(paths), 64)
                np.testing.assert_allclose(
                    paths.final_normalized_success_mass + paths.final_normalized_failure_mass,
                    1 - rho**paths.final_observation_n, atol=1e-14)
                manifest = json.loads((directory / 'manifest.json').read_text())
                self.assertEqual(manifest['config']['simulation']['rho'], rho)
                self.assertFalse((output / f'rho_{rho}').exists())
                for figure in manifest['figures']:
                    self.assertTrue((directory / figure).is_file())
            manifest = json.loads((output / 'manifest.json').read_text())
            for figure in manifest['figures']:
                self.assertTrue((output / figure).is_file())
            self.assertTrue((output / 'figures/monte_carlo_forgetting_by_rho.png').is_file())
            self.assertEqual(len(list(output.rglob('*.png'))), 3)
            self.assertEqual(len(list((output / 'figures').glob('*.png'))), 3)
            self.assertFalse(list(output.rglob('*.pdf')))
