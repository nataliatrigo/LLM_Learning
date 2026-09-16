"""Regression checks for disconnected positive-investment windows."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forgetting.model import Primitives, solve_boundary, reachable_maximum
from forgetting.run_experiments import refine_positive_windows, plot_positive_windows


class PositiveWindowTests(unittest.TestCase):
    def refine(self, objective, rhos):
        rhos = np.asarray(rhos)
        frame = pd.DataFrame({"p0": .9, "rho": rhos,
                              "reachable_maximum_advantage": objective(rhos)})
        config = {"root_refinement": {"rho_tolerance": 1e-9,
                                       "max_bisection_iterations": 40}}
        with patch('forgetting.run_experiments.solve_summary',
                   side_effect=lambda config, p, rho: {
                       'reachable_maximum_advantage': float(objective(rho))}):
            return refine_positive_windows(frame, config, Primitives())

    def test_four_crossings_remain_two_windows(self):
        windows = self.refine(lambda x: -(x-.2)*(x-.4)*(x-.46)*(x-.48),
                              [.1,.3,.42,.45,.47,.49,.6])
        np.testing.assert_allclose(windows.rho_lower, [.2,.46], atol=1e-8)
        np.testing.assert_allclose(windows.rho_upper, [.4,.48], atol=1e-8)
        self.assertEqual(windows.crossing_count.tolist(), [4,4])
        self.assertFalse(((windows.rho_lower < .45) & (windows.rho_upper > .45)).any())
        with TemporaryDirectory() as temp:
            def inspect(fig, name):
                segments = fig.axes[0].collections[0].get_segments()
                self.assertEqual(len(segments), 2)
                for segment in segments:
                    self.assertFalse(segment[0,1] < .45 < segment[1,1])
                fig.savefig(Path(temp)/'windows.png')
                plt.close(fig)
            with patch('forgetting.run_experiments.save_figure', side_effect=inspect):
                plot_positive_windows(windows)

    def test_single_window_and_roots_on_grid(self):
        windows = self.refine(lambda x: (x-.2)*(.8-x), [.1,.2,.5,.8,.9])
        np.testing.assert_allclose(windows[['rho_lower','rho_upper']], [[.2,.8]])
        self.assertEqual(int(windows.iloc[0].window_count), 1)

    def test_inactive_and_censored_windows(self):
        inactive = self.refine(lambda x: np.zeros_like(x)-1, [.1,.5,.9])
        self.assertEqual(int(inactive.iloc[0].window_count), 0)
        self.assertTrue(inactive.rho_lower.isna().all())
        active = self.refine(lambda x: np.zeros_like(x)+1, [.1,.5,.9])
        self.assertTrue(active.iloc[0].lower_censored)
        self.assertTrue(active.iloc[0].upper_censored)
        np.testing.assert_allclose(active[['rho_lower','rho_upper']], [[.1,.9]])

    def test_actual_gap_is_negative_only_on_reachable_states(self):
        primitives = Primitives(p0=.9, c2=.6665)
        signs = []
        for rho in (.43,.45,.47):
            solution = solve_boundary(primitives, rho, central_nodes=4001, tolerance=1e-12)
            maximum = reachable_maximum(solution, reachable_depth=16)[0]
            signs.append(np.sign(maximum))
            if rho == .45:
                self.assertLess(maximum, -4e-4)
                self.assertGreater(solution.advantage.max(), 5e-4)
                active = solution.coordinate[solution.advantage > 0]
                self.assertTrue(np.all((active > rho) & (active < 1-rho)))
        self.assertEqual(signs, [1,-1,1])
