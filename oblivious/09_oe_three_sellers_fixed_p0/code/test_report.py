"""Safeguards against mixed seller counts, grids, beliefs and stale exports."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from oblivious.oe_three_sellers_fixed_p0.code import report


class ReportTests(unittest.TestCase):
    def write_csv(self, path, rows):
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def fixture(self, root, grids=(21, 31)):
        data, figures = root/'data', root/'figures'
        data.mkdir()
        rows, policies = [], []
        for p0 in report.P0S:
            for rho in report.RHOS:
                for n in grids:
                    for sellers, closure, *_ in report.MODELS:
                        sale = .5 if sellers == 1 else .3 if sellers == 2 else .2
                        rows.append(dict(p0=p0, rho=rho, n=n, sellers=sellers, closure=closure,
                            verified=True, product2_conditional=.4, sale_rate=sale,
                            outside_rate=1-sellers*sale, profit_calendar=.2, value_initial=10.))
                        policies.extend(dict(p0=p0, rho=rho, n=n, sellers=sellers, closure=closure,
                            reputation=float(x), product2_probability=.4 if n == 21 else .41)
                            for x in np.linspace(0, 1, n))
        (root/'engine.py').write_text('certified numerical engine')
        manifest = dict(status='complete', configuration=dict(p0s=report.P0S, rhos=report.RHOS, grids=list(grids),
            models=[list(model[:2]) for model in report.MODELS]),
            numerical_sources={'engine.py': report.digest(root/'engine.py')})
        self.save(data, rows, policies, manifest)
        return data, figures, rows, policies, manifest

    def save(self, data, rows, policies, manifest):
        self.write_csv(data/'summary.csv', rows)
        self.write_csv(data/'policies.csv', policies)
        manifest['input_sha256'] = {name: report.digest(data/name) for name in ('summary.csv', 'policies.csv')}
        (data/'manifest.json').write_text(json.dumps(manifest))

    def test_hashes_and_whole_market_normalization_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, _, rows, policies, manifest = self.fixture(root)
            with patch.object(report, 'ROOT', root):
                report.read_inputs(data)
                (data/'summary.csv').write_text('changed CSV')
                with self.assertRaisesRegex(ValueError, 'Input fingerprint differs'):
                    report.read_inputs(data)
                rows[-1]['outside_rate'] = 1-rows[-1]['sale_rate']
                self.save(data, rows, policies, manifest)
                with self.assertRaisesRegex(ValueError, r'outside\+N\*sale_rate=1'):
                    report.read_inputs(data)

    def test_original_policy_nodes_and_common_grid_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, figures, rows, policies, manifest = self.fixture(root)
            # One calibration refined to 51 cannot replace the shared 31 grid.
            extra = dict(rows[-1], n=51)
            rows.append(extra)
            policies.extend(dict(p0=extra['p0'], rho=extra['rho'], n=51, sellers=extra['sellers'],
                closure=extra['closure'], reputation=float(x), product2_probability=.9)
                for x in np.linspace(0, 1, 51))
            self.save(data, rows, policies, manifest)
            with patch.object(report, 'ROOT', root), patch.object(report, 'render', return_value=[]) as render:
                result = report.build_report(data, figures)
                self.assertEqual((result['displayed_n'], result['validation_coarse_n']), (31, 21))
                self.assertEqual(render.call_args.args[-2:], (31, 21))
                self.assertIn(.41, {row['product2_probability'] for row in render.call_args.args[1] if row['n'] == 31})
                policies[1]['reputation'] += 1e-6
                self.save(data, rows, policies, manifest)
                with self.assertRaisesRegex(ValueError, 'original uniform reputation grid'):
                    report.read_inputs(data)

    def test_one_failed_three_seller_model_excludes_rho_for_every_p0(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, figures, rows, policies, manifest = self.fixture(root)
            target = next(row for row in rows if row['p0'] == .8 and row['rho'] == .8
                          and row['n'] == 31 and row['sellers'] == 3 and row['closure'] == 'conditional')
            target['profit_calendar'] = .2/.94
            self.save(data, rows, policies, manifest)
            with patch.object(report, 'ROOT', root), patch.object(report, 'render', return_value=[]) as render:
                result = report.build_report(data, figures)
            self.assertEqual(result['common_retained_rhos'], [.6])
            self.assertEqual(result['status'], 'partial_rhos')
            self.assertEqual(render.call_args.args[4], [.6])
            failed = result['excluded_rhos']['0.8'][0]
            self.assertEqual((failed['p0'], failed['sellers'], failed['closure']), (.8, 3, 'conditional'))
            self.assertAlmostEqual(failed['change'], .06)

    def test_a_single_grid_produces_explicit_diagnostics_and_no_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, figures, _, _, _ = self.fixture(root, grids=(31,))
            with patch.object(report, 'ROOT', root), patch.object(report, 'render') as render:
                result = report.build_report(data, figures)
            self.assertFalse(result['verified'])
            self.assertEqual(result['status'], 'missing_validation_grid')
            self.assertEqual(result['figure_files'], {})
            self.assertIn('No se generaron plots verificados', (figures/'README.md').read_text())
            render.assert_not_called()

    def test_seller_count_facets_use_original_grid_and_shared_monopoly(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, figures, rows, _, _ = self.fixture(root)
            for row in rows:
                if row['n'] == 31:
                    row['product2_conditional'] = (row['p0']+.01*row['sellers']
                                                  +(.02 if row['closure'] == 'conditional' else 0))
            captured = []

            def capture(fig, directory, stem):
                captured.append(fig)
                self.assertEqual(stem, report.SELLER_COUNT_STEM)
                return [directory/(stem+'.png'), directory/(stem+'.pdf')]

            with patch.object(report, 'save', side_effect=capture):
                paths = report.render_product2_by_seller_count(
                    rows, figures, report.P0S, report.RHOS, 31, 21, models=report.MODELS)
            self.assertEqual(len(paths), 2)
            self.assertEqual(len(captured[0].axes), 4)
            for index, rho in enumerate(report.RHOS):
                left, right = captured[0].axes[2*index:2*index+2]
                for p0, marginal, conditional in zip(report.P0S, left.lines, right.lines):
                    np.testing.assert_array_equal(marginal.get_xdata(), [1, 2, 3])
                    self.assertAlmostEqual(marginal.get_ydata()[0], 100*(p0+.01))
                    self.assertEqual(marginal.get_ydata()[0], conditional.get_ydata()[0])
                    self.assertAlmostEqual(conditional.get_ydata()[1]-marginal.get_ydata()[1], 2.)
            report.plt.close(captured[0])

    def test_marginal_only_scope_uses_three_models_and_one_count_plot_column(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, figures, rows, policies, manifest = self.fixture(root)
            rows = [row for row in rows if row['closure'] != 'conditional']
            policies = [row for row in policies if row['closure'] != 'conditional']
            manifest['configuration']['models'] = [list(model[:2]) for model in report.DEFAULT_MODELS]
            self.save(data, rows, policies, manifest)
            with patch.object(report, 'ROOT', root), patch.object(report, 'render', return_value=[]) as render:
                result = report.build_report(data, figures)
            self.assertEqual(len(result['models']), 3)
            self.assertFalse(any(model['closure'] == 'conditional' for model in result['models']))
            self.assertEqual(render.call_args.kwargs['models'], report.DEFAULT_MODELS)
            self.assertNotIn('condicional', (figures/'README.md').read_text())
            captured = []

            def capture(fig, directory, stem):
                captured.append(fig)
                return [directory/(stem+'.png'), directory/(stem+'.pdf')]

            with patch.object(report, 'save', side_effect=capture):
                report.render_product2_by_seller_count(rows, figures, report.P0S, report.RHOS, 31, 21)
            self.assertEqual(len(captured[0].axes), len(report.RHOS))
            self.assertTrue(all('Marginal' in ax.get_title() for ax in captured[0].axes))
            report.plt.close(captured[0])

    def test_running_campaign_does_not_write_any_report_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, figures, rows, policies, manifest = self.fixture(root)
            manifest['status'] = 'running'
            self.save(data, rows, policies, manifest)
            with patch.object(report, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'still running'):
                    report.build_report(data, figures)
            self.assertFalse(figures.exists())
            self.assertFalse((data/'report_manifest.json').exists())

    def test_available_independent_audit_must_match_manifest_inputs_and_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, figures, _, _, manifest = self.fixture(root)
            (root/'audit.py').write_text('independent audit source')
            audit = dict(status='complete_verified', verified=True,
                manifest_snapshot_sha256=report.digest(data/'manifest.json'),
                campaign_input_sha256=manifest['input_sha256'],
                numerical_sources=manifest['numerical_sources'],
                audit_sources={'audit.py': report.digest(root/'audit.py')},
                audited_model_grid_cases=60, expected_model_grid_cases=60)
            path = data/'independent_audit.json'
            path.write_text(json.dumps(audit))
            with patch.object(report, 'ROOT', root), patch.object(report, 'render', return_value=[]):
                result = report.build_report(data, figures)
                self.assertTrue(result['independent_audit']['verified'])
                self.assertEqual(result['source_input_sha256']['data/independent_audit.json'], report.digest(path))
                audit['manifest_snapshot_sha256'] = 'stale snapshot'
                path.write_text(json.dumps(audit))
                with self.assertRaisesRegex(ValueError, 'snapshot is stale'):
                    report.build_report(data, figures)
                audit['manifest_snapshot_sha256'] = report.digest(data/'manifest.json')
                audit['campaign_input_sha256'] = {}
                path.write_text(json.dumps(audit))
                with self.assertRaisesRegex(ValueError, 'current campaign inputs'):
                    report.build_report(data, figures)
                audit['campaign_input_sha256'] = manifest['input_sha256']
                path.write_text(json.dumps(audit))
                (root/'audit.py').write_text('changed since review')
                with self.assertRaisesRegex(ValueError, 'Input fingerprint differs: audit.py'):
                    report.build_report(data, figures)


if __name__ == '__main__':
    unittest.main()
