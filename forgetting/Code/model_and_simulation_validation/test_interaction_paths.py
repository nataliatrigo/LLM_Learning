"""Interaction indexing, policy reuse, and trajectory diagnostics."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np

from forgetting.Code.experiments.interaction_paths import simulate, diagnostics, save_outputs
from forgetting.Code.model import Primitives
from forgetting.Code.simulation import _policy_and_demand


class InteractionPathTests(unittest.TestCase):
    def small(self, n=20):
        return simulate(Primitives(), .8, paths=3, interactions=n,
                        trace_window=(1, 20), nodes=101, terminal_gap=.1, seed=17)

    def test_existing_policy_and_state_updates_and_prefix_reproducibility(self):
        with patch('forgetting.Code.experiments.interaction_paths._policy_and_demand',
                   wraps=_policy_and_demand) as policy:
            result = self.small()
            self.assertEqual(policy.call_count, 20)
        longer = self.small(30)
        np.testing.assert_array_equal(result['choices'], longer['choices'][:20])
        trace = result['trace']
        for _, path in trace.groupby('trajectory'):
            np.testing.assert_allclose(path.x_before.iloc[0], 0)
            np.testing.assert_allclose(path.y_before.iloc[0], 0)
            np.testing.assert_allclose(path.x_after, .8*path.x_before + .2*path.success)
            np.testing.assert_allclose(path.y_after, .8*path.y_before + .2*(1-path.success), atol=1e-14)
            np.testing.assert_allclose(path.x_before.to_numpy()[1:], path.x_after.to_numpy()[:-1])
            np.testing.assert_allclose(path.x_after + path.y_after, 1-.8**path.interaction)
        np.testing.assert_allclose(result['cumulative'][-1], result['choices'].mean(axis=0))

    def test_zero_discount_never_chooses_product2(self):
        result = simulate(replace(Primitives(), gamma=0), .8, paths=3,
                          interactions=20, trace_window=(1,20), nodes=101, terminal_gap=.1)
        self.assertFalse(result['choices'].any())

    def test_diagnostic_interaction_boundaries_and_weighting(self):
        choices = np.column_stack((np.arange(12) < 6, np.arange(12) >= 6))
        result = {'choices': choices, 'final_x': np.zeros(2), 'final_y': np.ones(2)}
        report = diagnostics(result, burn_in=2, block_size=2, lengths=(6,12))
        summary = report['summary']
        np.testing.assert_allclose(summary.first_segment_frequency, [1,0])
        np.testing.assert_allclose(summary.second_segment_frequency, [0,1])
        np.testing.assert_allclose(summary.post_burn_in_frequency, [.4,.6])
        np.testing.assert_allclose(summary.F_6, [1,0])
        np.testing.assert_allclose(summary.F_12, [.5,.5])
        self.assertEqual(report['statistics']['first_segment'], [3,6])
        self.assertEqual(report['statistics']['second_segment'], [7,12])
        self.assertEqual(report['blocks'].start_interaction.min(), 3)
        self.assertEqual(report['blocks'].end_interaction.max(), 12)
        self.assertEqual(len(report['blocks']), 10)

    def test_outputs_are_individual_and_png_only(self):
        result = self.small()
        report = diagnostics(result, burn_in=2, block_size=2, lengths=(10,20))
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_outputs(result, report, output, {})
            self.assertEqual(len(list((output/'figures').glob('*.png'))), 6)
            self.assertFalse(list(output.rglob('*.pdf')))
            self.assertTrue((output/'data/trajectory_summary.csv').is_file())
