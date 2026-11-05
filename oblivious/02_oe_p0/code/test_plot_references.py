"""Regression checks for plotting archived reports with retired references."""

from pathlib import Path
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
from numpy.testing import assert_allclose

from oblivious.oe_p0.code import run_experiment as scalar_runner
from oblivious.oe_distribution.code import run_experiment as marginal_runner


class ObliviousPlotReferences(unittest.TestCase):
    def setUp(self):
        self.grid = np.array([0., .5, 1.])
        self.policy = np.array([0., .4, 1.])
        self.reference = np.array([0., .6, 1.])
        self.retired = np.array([1., .8, 0.])
        self.figures = {}
        self.references = [
            dict(key="oblivious", model="OE_REAL_MARGINAL", source_grid=3),
            dict(key="retired", model="MPE", source_grid=3),
            dict(key="excluded", model="OE", source_grid=3),
            # An unclassified historical row must not be treated as OE.
            dict(key="unclassified", source_grid=3),
        ]

    def tearDown(self):
        plt.close("all")

    def capture(self, fig, folder, name):
        self.figures[name] = fig

    def assert_oblivious_policies_only(self, name):
        for fig in self.figures.values():
            text = "\n".join(artist.get_text() for artist in fig.findobj(Text))
            self.assertNotIn("mpe", text.lower())
        lines = self.figures[name].axes[0].lines
        self.assertEqual(len(lines), 2)
        assert_allclose(lines[0].get_ydata(), self.policy)
        assert_allclose(lines[1].get_ydata(), self.reference)

    def test_scalar_figures_filter_historical_full_state_references(self):
        report = dict(rho=.9, n=3, p0=.5,
            demand=dict(arrays=dict(grid=self.grid, d_fixed=self.grid,
                                    d_marginal=self.grid, d_conditional=self.grid)),
            comparisons=dict(summary=dict(references=self.references), arrays=dict(
                grid=self.grid, scalar_policy=self.policy,
                oblivious_sigma_a=np.repeat(self.reference[:, None], 3, axis=1),
                retired_sigma_a=np.repeat(self.retired[:, None], 3, axis=1))))
        chosen = {.9: dict(n=3, selected=dict(p0=.5))}
        with patch.object(scalar_runner, "save_figure", self.capture):
            scalar_runner.diagnostic_figures(chosen, {(.9, 3, 0): report}, Path("unused"))
        self.assert_oblivious_policies_only("08_reference_policies")

    def test_marginal_figures_show_only_the_current_experiment(self):
        marginal = np.array([.2, .3, .5])
        evaluation = dict(grid=self.grid, policy=self.policy, verified=True,
            advantage=np.array([-.2, 0., .3]),
            mu=marginal, mu_a=marginal, mu_b=marginal,
            nu=np.array([[.05, .05, .1], [.05, .1, .15], [.1, .15, .25]]))
        case = dict(rho=.9, n=3, selected=evaluation, candidates=[evaluation])
        report = dict(rho=.9, n=3,
            demand=dict(arrays=dict(d_optimized=self.grid,
                                    d_real_marginal=self.grid, d_conditional=self.grid)),
            comparisons=dict(summary=dict(references=self.references), arrays=dict(
                oblivious_policy=self.reference, retired_policy=self.retired)))
        with patch.object(marginal_runner, "save_figure", self.capture):
            marginal_runner.make_figures([case], [report], Path("unused"))
        for fig in self.figures.values():
            text = "\n".join(artist.get_text() for artist in fig.findobj(Text))
            self.assertNotIn("mpe", text.lower())
            self.assertNotIn("Original", text)
        lines = self.figures["01_policy"].axes[0].lines
        self.assertEqual(len(lines), 1)
        assert_allclose(lines[0].get_ydata(), self.policy)
        self.assertEqual(lines[0].get_label(), "OE distribution")


if __name__ == "__main__":
    unittest.main()
