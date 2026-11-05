"""Case-specific grid and metric-specific publication safeguards."""

import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from . import expanded_report as expanded
from . import report as base


P0S, RHOS = [.1, .3, .5, .7, .8, .9], [.6, .7, .8, .9]


class ExpandedReportTests(unittest.TestCase):
    def write_csv(self, path, rows):
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def fixture(self, root):
        data, figures = root/'data', root/'figures'
        data.mkdir()
        rows, policies = [], []
        for p0 in P0S:
            for rho in RHOS:
                for n in (21, 31, 41):
                    for sellers, closure, *_ in base.DEFAULT_MODELS:
                        rows.append(dict(p0=p0, rho=rho, n=n, sellers=sellers, closure=closure,
                            verified=True, product2_conditional=p0/2+.01*sellers,
                            sale_rate=.5/sellers, outside_rate=.5, profit_calendar=.2, value_initial=10.))
                        policies.extend(dict(p0=p0, rho=rho, n=n, sellers=sellers, closure=closure,
                            reputation=float(x), product2_probability=.4+n/1000)
                            for x in np.linspace(0, 1, n))
        (root/'engine.py').write_text('frozen numerical source')
        (root/'audit.py').write_text('independent equation audit')
        manifest = dict(status='complete', configuration=dict(p0s=P0S, rhos=RHOS, grids=[21, 31, 41],
            models=[list(model[:2]) for model in base.DEFAULT_MODELS]),
            case_grids={expanded.case_key(p0, rho): [21, 31, 41] for p0 in P0S for rho in RHOS},
            numerical_sources={'engine.py': base.digest(root/'engine.py')})
        self.save(data, rows, policies, manifest)
        return data, figures, rows, policies, manifest

    def save(self, data, rows, policies, manifest):
        self.write_csv(data/'summary.csv', rows)
        self.write_csv(data/'policies.csv', policies)
        manifest['input_sha256'] = {name: base.digest(data/name) for name in ('summary.csv', 'policies.csv')}
        (data/'manifest.json').write_text(json.dumps(manifest))

    def audit(self, data, manifest):
        root = data.parent
        (data/'independent_audit.json').write_text(json.dumps(dict(status='complete_verified', verified=True,
            manifest_snapshot_sha256=base.digest(data/'manifest.json'), campaign_input_sha256=manifest['input_sha256'],
            numerical_sources=manifest['numerical_sources'], audit_sources={'audit.py': base.digest(root/'audit.py')},
            expected_model_grid_cases=216, audited_model_grid_cases=216)))

    def test_each_case_uses_its_own_finest_pair_common_to_all_three_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, _, rows, _, manifest = self.fixture(root)
            case = expanded.case_key(.9, .9)
            manifest['case_grids'][case].append(51)
            rows.extend(dict(row, n=51) for row in list(rows)
                        if row['p0'] == .9 and row['rho'] == .9 and row['n'] == 41)
            cases, comparisons = expanded.select_cases(rows, manifest, P0S, RHOS)
            self.assertEqual((cases[case]['validation_coarse_n'], cases[case]['displayed_n']), (41, 51))
            self.assertEqual((cases[expanded.case_key(.8, .9)]['validation_coarse_n'],
                              cases[expanded.case_key(.8, .9)]['displayed_n']), (31, 41))
            self.assertEqual(len(comparisons), 72)
            rows = [row for row in rows if not (row['p0'] == .9 and row['rho'] == .9 and row['n'] == 51 and row['sellers'] == 3)]
            cases, _ = expanded.select_cases(rows, manifest, P0S, RHOS)
            self.assertEqual(cases[case]['displayed_n'], 41)

    def test_failed_profit_does_not_hide_valid_p2_or_other_rho_cases(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, rows, _, manifest = self.fixture(Path(temporary))
            target = next(row for row in rows if row['p0'] == .8 and row['rho'] == .9 and row['n'] == 41 and row['sellers'] == 3)
            target['profit_calendar'] = .2/.9
            cases, _ = expanded.select_cases(rows, manifest, P0S, RHOS)
            key = expanded.case_key(.8, .9)
            self.assertFalse(cases[key]['metrics']['profit_calendar']['passed'])
            self.assertTrue(cases[key]['metrics']['product2_conditional']['passed'])
            self.assertAlmostEqual(cases[key]['metrics']['profit_calendar']['max_change'], .1)
            retained = expanded.metric_retention(cases, P0S, RHOS)
            self.assertIn(.9, retained['product2_conditional']['by_p0']['0.8'])
            self.assertNotIn(.9, retained['profit_calendar']['by_p0']['0.8'])
            self.assertIn(.9, retained['profit_calendar']['by_p0']['0.7'])
            lookup = {(r['p0'], r['rho'], r['n'], r['sellers'], r['closure']): r for r in rows}
            values = expanded.metric_values(lookup, cases, .8, RHOS, 1, 'monopoly', 'profit_calendar')
            self.assertTrue(np.isnan(values[-1]))
            self.assertTrue(np.isfinite(values[:-1]).all())

    def test_missing_validation_grid_becomes_a_metric_gap(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, _, rows, _, manifest = self.fixture(Path(temporary))
            rows = [row for row in rows if not (row['p0'] == .9 and row['rho'] == .9 and row['sellers'] == 3 and row['n'] > 21)]
            cases, _ = expanded.select_cases(rows, manifest, P0S, RHOS)
            case = cases[expanded.case_key(.9, .9)]
            self.assertEqual(case['displayed_n'], 21)
            self.assertIsNone(case['validation_coarse_n'])
            self.assertTrue(all(not metric['passed'] for metric in case['metrics'].values()))

    def test_live_check_only_writes_nothing_and_publication_requires_fresh_audit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, figures, rows, policies, manifest = self.fixture(root)
            manifest['status'] = 'running'
            self.save(data, rows, policies, manifest)
            before = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}
            with patch.object(base, 'ROOT', root):
                result = expanded.build_report(data, figures, check_only=True)
                self.assertFalse(result['verified'])
                self.assertTrue(result['diagnostic_only'])
                self.assertEqual(len(result['case_grids']), 24)
                with self.assertRaisesRegex(ValueError, 'still running'):
                    expanded.build_report(data, figures)
            after = {str(path.relative_to(root)): path.read_bytes() for path in root.rglob('*') if path.is_file()}
            self.assertEqual(before, after)
            manifest['status'] = 'complete'
            self.save(data, rows, policies, manifest)
            with patch.object(base, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'complete independent audit'):
                    expanded.build_report(data, figures)
                self.audit(data, manifest)
                with patch.object(expanded, 'render', return_value=[]):
                    result = expanded.build_report(data, figures)
                self.assertTrue(result['independent_audit']['verified'])
                manifest['new_revision'] = 1
                (data/'manifest.json').write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, 'snapshot is stale'):
                    expanded.build_report(data, figures)

    def test_six_colors_four_facets_and_only_validated_p2_curves(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, figures, rows, policies, manifest = self.fixture(root)
            target = next(row for row in rows if row['p0'] == .8 and row['rho'] == .9 and row['n'] == 41 and row['sellers'] == 3)
            target['product2_conditional'] += .06
            cases, _ = expanded.select_cases(rows, manifest, P0S, RHOS)
            observed = {}

            def capture(fig, directory, stem):
                if stem == base.SELLER_COUNT_STEM:
                    observed['facets'] = len(fig.axes)
                    observed['colors'] = [line.get_color() for line in fig.axes[0].lines]
                    observed['last_labels'] = [line.get_label() for line in fig.axes[-1].lines]
                    observed['caption'] = fig.axes[-1].get_title()
                elif stem == base.STEMS[0]:
                    self.assertEqual(len(fig.axes), 4)
                base.plt.close(fig)
                return []

            figures.mkdir()
            with patch.object(base, 'save', side_effect=capture):
                expanded.render(rows, policies, figures, P0S, RHOS, cases)
            self.assertEqual(observed['facets'], 4)
            self.assertEqual(len(set(observed['colors'])), 6)
            self.assertNotIn('p₀ = 0.8', observed['last_labels'])
            self.assertIn('Omitted p₀: 0.8', observed['caption'])


if __name__ == '__main__':
    unittest.main()
