"""Focused tests for plots built from saved policy tables."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import pandas as pd


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp import plotting  # noqa: E402


class ObservationForgettingPolicyPlotTests(unittest.TestCase):
    def test_focused_plot_uses_observation_clock_and_independent_memory_limits(
        self,
    ) -> None:
        frame = pd.DataFrame(
            [
                {
                    "method": "observation_forgetting_ts",
                    "parameter_value": 0.8,
                    "effective_sample_size": 2.0,
                    "posterior_mean": 0.55,
                    "p0": 0.5,
                    "robust_action2": True,
                },
                {
                    "method": "observation_forgetting_ts",
                    "parameter_value": 0.8,
                    "effective_sample_size": 3.0,
                    "posterior_mean": 0.45,
                    "p0": 0.5,
                    "robust_action2": False,
                },
                {
                    "method": "observation_forgetting_ts",
                    "parameter_value": 0.95,
                    "effective_sample_size": 10.0,
                    "posterior_mean": 0.60,
                    "p0": 0.5,
                    "robust_action2": True,
                },
                {
                    "method": "calendar_forgetting_ts",
                    "parameter_value": 0.8,
                    "effective_sample_size": 2.0,
                    "posterior_mean": 0.70,
                    "p0": 0.5,
                    "robust_action2": True,
                },
            ]
        )

        with patch("modular_dp.plotting._save") as save:
            plotting.plot_observation_forgetting_policy_regions(
                frame, Path("/tmp/unused")
            )

        save.assert_called_once()
        figure, stem = save.call_args.args
        try:
            self.assertEqual(stem.name, "24_policy_observation_forgetting")
            self.assertEqual(len(figure.axes), 2)
            self.assertIn("max $n_{\\rm eff}$=5", figure.axes[0].get_title())
            self.assertIn("max $n_{\\rm eff}$=20", figure.axes[1].get_title())
            self.assertAlmostEqual(figure.axes[0].get_xlim()[1], 5.1)
            self.assertAlmostEqual(figure.axes[1].get_xlim()[1], 20.4)
            self.assertEqual(len(figure.axes[0].collections[1].get_offsets()), 1)
        finally:
            plotting.plt.close(figure)


if __name__ == "__main__":
    unittest.main()
