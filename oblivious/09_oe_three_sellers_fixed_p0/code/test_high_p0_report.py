"""A finer grid must distinguish policy equality from product-mix equality."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import expanded_report as expanded
from . import high_p0_report as high
from . import report as base
from . import test_expanded_report as fixtures

P0S, RHOS = fixtures.P0S, fixtures.RHOS


class HighP0ReportTests(unittest.TestCase):
    def fixture(self, root):
        return fixtures.ExpandedReportTests().fixture(root)

    def test_policy_equality_is_checked_at_original_nodes_even_when_product_mix_matches(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, rows, policies, _ = self.fixture(Path(temporary))
            # Identical policies can have different model-specific sale weights.
            result = high.policy_diagnostics(rows, policies, P0S, RHOS)
            self.assertEqual(len(result['within_grid_pairs']), 72)
            self.assertEqual(len(result['refinement_by_model']), 36)
            self.assertTrue(all(row['exact_policy_equal'] for row in result['within_grid_pairs']))
            target = next(row for row in policies if row['p0'] == .8 and row['rho'] == .6
                          and row['n'] == 41 and row['sellers'] == 3)
            target['product2_probability'] += .1
            changed = high.policy_diagnostics(rows, policies, P0S, RHOS)
            observed = next(row for row in changed['within_grid_pairs'] if row['p0'] == .8
                and row['rho'] == .6 and row['n'] == 41 and row['first_sellers'] == 1 and row['second_sellers'] == 3)
            self.assertFalse(observed['exact_policy_equal'])
            self.assertAlmostEqual(observed['alpha_max_abs_difference'], .1)
            self.assertAlmostEqual(observed['product2_difference_pp'], 2.)

    def test_interval_diagnostics_keep_disconnected_runs_and_do_not_interpolate_endpoints(self):
        intervals = high.action_intervals([0., .25, .5, .75, 1.], [0., .7, .6, 0., 1.])
        self.assertEqual(intervals, [[.25, .5], [1., 1.]])
        self.assertEqual(high.action_intervals([0., 1.], [0., 0.]), [])

    def test_policy_diagnostics_follow_the_finest_common_pair_if_refinement_reaches_51(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, rows, policies, _ = self.fixture(Path(temporary))
            for sellers, closure, *_ in base.DEFAULT_MODELS:
                row = next(row for row in rows if row['p0'] == .8 and row['rho'] == .6
                           and row['n'] == 41 and row['sellers'] == sellers)
                rows.append(dict(row, n=51))
                policies.extend(dict(p0=.8, rho=.6, n=51, sellers=sellers, closure=closure,
                    reputation=float(x), product2_probability=.451) for x in np.linspace(0, 1, 51))
            result = high.policy_diagnostics(rows, policies, P0S, RHOS)
            selected = [row for row in result['within_grid_pairs'] if row['p0'] == .8 and row['rho'] == .6]
            self.assertEqual({row['n'] for row in selected}, {41, 51})
            refinement = next(row for row in result['refinement_by_model'] if row['p0'] == .8 and row['rho'] == .6)
            self.assertEqual((refinement['coarse_n'], refinement['fine_n']), (41, 51))

    def test_horizontal_references_share_validation_omissions_and_case_specific_grids(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, figures, rows, _, manifest = self.fixture(Path(temporary))
            manifest['case_grids'][expanded.case_key(.1, .6)] = [21, 31]
            row = next(row for row in rows if row['p0'] == .8 and row['rho'] == .9
                       and row['n'] == 41 and row['sellers'] == 3)
            row['product2_conditional'] += .1
            cases, _ = expanded.select_cases(rows, manifest, P0S, RHOS)
            observed = {}

            def capture(fig, directory, stem):
                observed['lines'] = [len(ax.lines) for ax in fig.axes]
                observed['legend'] = [text.get_text() for text in fig.legends[0].texts]
                base.plt.close(fig)
                return []

            with patch.object(base, 'save', side_effect=capture):
                _, result = high.render_references(rows, figures, P0S, RHOS, cases)
            self.assertEqual(observed['lines'], [12, 12, 12, 10])
            self.assertIn('Monopoly reference (N=1)', observed['legend'])
            self.assertEqual(result['omitted_p0_by_rho']['0.9'], [.8])
            self.assertEqual(next(point for point in result['plotted_cases'] if point['p0'] == .1
                and point['rho'] == .6)['n'], 31)
            self.assertEqual(next(point for point in result['plotted_cases'] if point['p0'] == .8
                and point['rho'] == .6)['n'], 41)

    def test_fresh_closed_audited_campaign_only_and_reporting_sources_are_fingerprinted(self):
        helper = fixtures.ExpandedReportTests()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, figures, _, _, manifest = self.fixture(root)
            with patch.object(base, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'complete independent audit'):
                    high.build_report(data, figures)
                helper.audit(data, manifest)
                with patch.object(expanded, 'render', return_value=[]), patch.object(high, 'render_references', return_value=([], {})):
                    report = high.build_report(data, figures)
                self.assertTrue(report['verified'])
                self.assertTrue(any(name.endswith('high_p0_report.py') for name in report['reporting_source_sha256']))
                self.assertTrue((data/high.DIAGNOSTIC_FILE).exists())
                base.verify_inputs(report['generated_data_sha256'])
                self.assertIn('dos grillas más finas', (figures/'README.md').read_text())
                with self.assertRaisesRegex(ValueError, 'new output directory'):
                    high.build_report(data, figures)
                with self.assertRaisesRegex(ValueError, 'already has a published report'):
                    high.build_report(data, root/'different_figures')


if __name__ == '__main__':
    unittest.main()
