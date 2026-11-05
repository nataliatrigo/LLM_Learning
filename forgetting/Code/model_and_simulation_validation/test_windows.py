"""Regression checks for disconnected positive-investment windows."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forgetting.Code.model import Primitives, solve_boundary, reachable_maximum
from forgetting.Code.experiments.run_experiments import (
    refine_positive_windows, plot_positive_windows, draw_positive_windows,
    investment_scenarios, run_investment_sweeps,
)


class PositiveWindowTests(unittest.TestCase):
    def test_no_fill_or_connections_across_p0(self):
        windows = pd.DataFrame({
            "p0": [.1, .5, .9], "rho_lower": [.2, np.nan, .6],
            "rho_upper": [.4, np.nan, .8],
            "lower_censored": [False, False, False],
            "upper_censored": [False, False, True],
        })
        fig, ax = plt.subplots()
        try:
            draw_positive_windows(ax, windows)
            self.assertEqual(len(ax.lines), 0)
            self.assertEqual(len(ax.patches), 0)
            np.testing.assert_allclose(ax.collections[0].get_segments(),
                                       [[[.1, .2], [.1, .4]], [[.9, .6], [.9, .8]]])
            self.assertEqual(len(ax.collections), 5)  # Intervals and endpoint markers only.
        finally:
            plt.close(fig)

    def test_sweeps_hold_other_parameter_fixed_and_reuse_baseline(self):
        base = Primitives()
        config = {"investment_window_sweeps": {"c2": [.65, .68], "p2": [.75, .8]}}
        windows = self.refine(lambda x: (x-.2)*(.8-x), [.1, .5, .9])
        with TemporaryDirectory() as temp, \
                patch('forgetting.Code.experiments.run_experiments.DATA', Path(temp)), \
                patch('forgetting.Code.experiments.run_experiments.FIGURES', Path(temp)), \
                patch('forgetting.Code.experiments.run_experiments.run_map', return_value=pd.DataFrame()) as run_map, \
                patch('forgetting.Code.experiments.run_experiments.refine_positive_windows', return_value=windows):
            stems = run_investment_sweeps(config, base, windows)
            self.assertEqual(len(stems), 2)
            self.assertEqual(run_map.call_count, 2)
            scenarios = [call.args[1] for call in run_map.call_args_list]
            self.assertEqual([(p.c2, p.p2) for p in scenarios], [(.68, .8), (.65, .75)])
            for stem in stems:
                self.assertTrue((Path(temp) / f'{stem}.png').is_file())
            data = pd.read_csv(Path(temp) / 'positive_window_by_p0_vary_c2.csv')
            self.assertEqual(set(data.c2), {.65, .68})
            self.assertEqual(set(data.p2), {.8})

    def test_invalid_scenarios_fail_before_solving(self):
        for sweep in ({"p2": [.3]}, {"c2": [1.]}, {"c2": []}, {"p0": [.5]}):
            with self.assertRaises(ValueError):
                investment_scenarios({"investment_window_sweeps": sweep}, Primitives())

    def refine(self, objective, rhos):
        rhos = np.asarray(rhos)
        frame = pd.DataFrame({"p0": .9, "rho": rhos,
                              "reachable_maximum_advantage": objective(rhos)})
        config = {"root_refinement": {"rho_tolerance": 1e-9,
                                       "max_bisection_iterations": 40}}
        with patch('forgetting.Code.experiments.run_experiments.solve_summary',
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
            with patch('forgetting.Code.experiments.run_experiments.save_figure', side_effect=inspect):
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
