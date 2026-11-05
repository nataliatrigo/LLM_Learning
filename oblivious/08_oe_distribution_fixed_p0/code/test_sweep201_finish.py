"""Guards and artifact checks for the authorized background finalizer."""

from contextlib import ExitStack
import csv
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

from oblivious.oe_distribution_fixed_p0.code import sweep201_finish as finish


MODULE = 'oblivious.oe_distribution_fixed_p0.code.sweep201_finish'
READER = 'oblivious.oe_distribution_fixed_p0.code.multi_p0_report'


class Sweep201FinishTests(unittest.TestCase):
    def environment(self, root):
        stack = ExitStack()
        stack.enter_context(patch(MODULE+'.ROOT', root))
        stack.enter_context(patch(MODULE+'.HOME', root))
        stack.enter_context(patch(READER+'.ROOT', root))
        return stack

    def test_another_controller_or_running_target_cannot_be_resumed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root/'sweep_manifest.json'
            path.write_text(json.dumps(dict(status='running')))
            with patch(MODULE+'.active_controllers', return_value=[]):
                with self.assertRaisesRegex(ValueError, 'controller is still running'):
                    finish.require_closed_campaign(root)
                path.write_text(json.dumps(dict(status='interrupted')))
                case = root/'p0_0p1/rho_0p9'
                case.mkdir(parents=True)
                (case/'manifest.json').write_text(json.dumps(dict(status='interrupted', current_grid=201)))
                with self.assertRaisesRegex(ValueError, 'target campaign is still running'):
                    finish.require_closed_campaign(root)
                (case/'manifest.json').write_text(json.dumps(dict(status='interrupted', current_grid=None)))
                finish.require_closed_campaign(root)
            with patch(MODULE+'.active_controllers', return_value=[{'pid': 123}]):
                with self.assertRaisesRegex(ValueError, 'process is active'):
                    finish.require_closed_campaign(root)

    def test_exclusive_lock_blocks_a_second_pipeline(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with finish.exclusive_finalizer(root):
                with self.assertRaisesRegex(ValueError, 'already holds'):
                    with finish.exclusive_finalizer(root):
                        self.fail('Second finalizer entered')
            with finish.exclusive_finalizer(root):
                pass

    def test_running_controller_guard_writes_no_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root/'data'/finish.CAMPAIGN
            data.mkdir(parents=True)
            (data/'sweep_manifest.json').write_text(json.dumps(dict(status='running')))
            before = {str(path.relative_to(root)): path.read_bytes()
                      for path in root.rglob('*') if path.is_file()}
            with self.environment(root), patch(MODULE+'.subprocess.run') as process:
                with self.assertRaisesRegex(ValueError, 'controller is still running'):
                    finish.finish()
                process.assert_not_called()
            after = {str(path.relative_to(root)): path.read_bytes()
                     for path in root.rglob('*') if path.is_file()}
            self.assertEqual(before, after)

    def test_failed_compute_or_audit_prevents_report_and_index(self):
        for codes, failed_stage in (([2], 'compute'), ([0, 3], 'audit')):
            with self.subTest(failed_stage=failed_stage), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                data = root/'data'/finish.CAMPAIGN
                data.mkdir(parents=True)
                (data/'sweep_manifest.json').write_text('{}')
                process = Mock(side_effect=[SimpleNamespace(returncode=code) for code in codes])
                with self.environment(root), patch(MODULE+'.require_closed_campaign'), \
                        patch(MODULE+'.prior_figures', return_value={}), \
                        patch(MODULE+'.source_hashes', return_value={}), \
                        patch(MODULE+'.subprocess.run', process), \
                        patch(MODULE+'.verify_artifacts') as verify, \
                        patch(MODULE+'.write_index') as index:
                    self.assertEqual(finish.finish(), codes[-1])
                state = json.loads((data/'finalization_manifest.json').read_text())
                self.assertEqual(state['status'], 'failed')
                self.assertEqual(state['failed_stage'], failed_stage)
                self.assertEqual([stage['exit_code'] for stage in state['stages']], codes)
                self.assertEqual(process.call_count, len(codes))
                self.assertNotIn('sweep201_report', str(process.call_args_list))
                verify.assert_not_called()
                index.assert_not_called()

    def write_csv(self, path, rows):
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def fixture(self, root):
        data, figures = root/'data', root/'figures'
        data.mkdir()
        models = [model for model, *_ in finish.MODELS]
        all_rows = [dict(p0=p0, rho=rho, n=n, model=model, verified=True)
                    for p0 in finish.P0S for rho in finish.RHOS for n in finish.GRIDS for model in models]
        displayed = [row for row in all_rows if row['n'] == 201 and row['rho'] == .6]
        policies = [dict(**row, reputation=float(x), product2_probability=.4)
                    for row in displayed for x in np.linspace(0, 1, 201)]
        exports = {'comparison_summary_all_grids.csv': all_rows,
                   'comparison_summary.csv': displayed, 'comparison_policies.csv': policies}
        for name, rows in exports.items():
            self.write_csv(data/name, rows)
        (root/'engine.py').write_text('verified engine')
        paths = []
        for p0 in finish.P0S:
            directory = figures/finish.p0_name(p0)
            directory.mkdir(parents=True)
            for stem in finish.STEMS:
                png, pdf = directory/(stem+'.png'), directory/(stem+'.pdf')
                Image.new('RGB', (120, 120), 'white').save(png)
                pdf.write_bytes(b'%PDF-1.4\nfixture')
                paths.extend((png, pdf))
        report = dict(status='complete_with_sensitivity_exclusions',
            all_required_cases_certified=True, requested_p0s=finish.P0S,
            requested_rhos=finish.RHOS, required_grids=finish.GRIDS, displayed_n=201,
            validation_n=251, independent_audit={'verified': True}, common_retained_rhos=[.6],
            source_input_sha256={'engine.py': finish.digest(root/'engine.py')},
            reporting_source_sha256={'engine.py': finish.digest(root/'engine.py')},
            generated_data_sha256={str((data/name).relative_to(root)): finish.digest(data/name) for name in exports},
            figure_files={str(path.relative_to(root)): finish.digest(path) for path in paths})
        (data/'comparison_report_manifest.json').write_text(json.dumps(report))
        return data, figures, report, policies

    def test_artifacts_require_144_results_original_201_nodes_and_exact_48_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data, figures, report, policies = self.fixture(root)
            with self.environment(root):
                _, checks = finish.verify_artifacts(data, figures)
                self.assertEqual((checks['all_grid_rows'], checks['displayed_rows'], checks['policy_rows']),
                                 (144, 18, 3618))
                self.assertEqual((checks['png_count'], checks['pdf_count']), (24, 24))
                index = finish.write_index(figures, report)
                self.assertEqual(index.read_text().count('[PNG]('), 24)
                self.assertEqual(index.read_text().count('[PDF]('), 24)
                policies[1]['reputation'] += 1e-6
                self.write_csv(data/'comparison_policies.csv', policies)
                report['generated_data_sha256']['data/comparison_policies.csv'] = finish.digest(data/'comparison_policies.csv')
                (data/'comparison_report_manifest.json').write_text(json.dumps(report))
                with self.assertRaisesRegex(ValueError, 'original 201 nodes'):
                    finish.verify_artifacts(data, figures)


if __name__ == '__main__':
    unittest.main()
