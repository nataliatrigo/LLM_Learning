"""The automatic publisher waits, fails closed, and preserves old figures."""

import fcntl
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from . import expanded_finish as finish


class ExpandedFinishTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.home = self.root/'experiment'
        self.data, self.figures = self.home/'data/sweep', self.home/'figures/sweep'
        self.data.mkdir(parents=True)
        self.source = self.root/'source.py'
        self.source.write_text('frozen numerical source\n')
        self.sources = {'source.py':finish.io.digest(self.source)}
        for owner, name, value in ((finish, 'HOME', self.home), (finish, 'ROOT', self.root),
                                    (finish.io, 'ROOT', self.root)):
            patcher = patch.object(owner, name, value)
            patcher.start(); self.addCleanup(patcher.stop)
        patcher = patch.object(finish, 'source_snapshot', return_value=dict(self.sources))
        patcher.start(); self.addCleanup(patcher.stop)
        roots = [(self.home/'figures/marginal_pilot', 26),
                 (self.root/'oblivious/08_oe_distribution_fixed_p0/figures/p0_sweep', 48)]
        for directory, count in roots:
            directory.mkdir(parents=True)
            for i in range(count):
                (directory/(str(i)+('.png' if i % 2 else '.pdf'))).write_bytes(b'historical bytes')
        config = dict(p0s=list(finish.campaign.P0S), rhos=list(finish.campaign.RHOS),
                      grids=[3, 5, 7], models=[list(x) for x in finish.campaign.MODELS])
        config['case_grids'] = {finish.campaign.pair_key(p, r):[3, 5, 7]
                               for p in config['p0s'] for r in config['rhos']}
        completed = finish.campaign.expected(config)
        self.manifest = dict(status='complete', configuration=config, completed=completed,
                             numerical_sources=self.sources, input_sha256={})
        for stem in completed:
            for suffix in ('.json', '.npz'):
                path = self.data/(stem+suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'stored numerical result')
                self.manifest['input_sha256'][str(path.relative_to(self.root))] = finish.io.digest(path)
        finish.io.write_json(self.data/'manifest.json', self.manifest)

    def audit(self):
        audit = dict(verified=True, status='complete_verified', expected_model_grid_cases=216,
                     audited_model_grid_cases=216, numerical_sources=self.sources, audit_sources=self.sources,
                     campaign_input_sha256=self.manifest['input_sha256'],
                     manifest_snapshot_sha256=finish.io.digest(self.data/'manifest.json'))
        finish.io.write_json(self.data/'independent_audit.json', audit)

    def report(self):
        self.figures.mkdir(parents=True, exist_ok=True)
        cases = {}
        for p0 in finish.campaign.P0S:
            for rho in finish.campaign.RHOS:
                key = finish.campaign.pair_key(p0, rho)
                cases[key] = dict(p0=p0, rho=rho, metrics={metric:dict(passed=True) for metric in finish.METRICS})
        first = next(iter(cases))
        cases[first]['metrics']['product2_conditional']['passed'] = False
        retention = {metric:dict(valid_cases=[k for k,c in cases.items() if c['metrics'][metric]['passed']],
                                 excluded_cases={k:c['metrics'][metric] for k,c in cases.items()
                                                 if not c['metrics'][metric]['passed']}) for metric in finish.METRICS}
        paths = []
        for p0 in finish.campaign.P0S:
            directory = self.figures/('p0_'+str(float(p0)).replace('.', 'p'))
            directory.mkdir(parents=True, exist_ok=True)
            paths.extend(directory/(stem+suffix) for stem in finish.STEMS for suffix in ('.png', '.pdf'))
        paths.extend(self.figures/('05_product2_by_seller_count'+suffix) for suffix in ('.png', '.pdf'))
        for path in paths:
            if path.suffix == '.png':
                Image.new('RGB', (2, 2), 'white').save(path)
            else:
                path.write_bytes(b'%PDF-1.4\nfixture')
        generated = self.data/'metric_retention.json'
        finish.io.write_json(generated, retention)
        report = dict(verified=True, diagnostic_only=False, source_campaign_status='complete',
                      status='partial_gridstable', requested_p0s=finish.campaign.P0S,
                      requested_rhos=finish.campaign.RHOS,
                      models=[dict(sellers=n, closure=c) for n,c in finish.campaign.MODELS],
                      case_grids=cases, metric_retention=retention, all_metrics_grid_stable=False,
                      source_input_sha256=dict(self.manifest['input_sha256']),
                      reporting_source_sha256=self.sources,
                      generated_data_sha256=finish.fingerprint([generated]), figure_files=finish.fingerprint(paths))
        for target in (self.data/'report_manifest.json', self.figures/'report_manifest.json'):
            finish.io.write_json(target, report)
        return report

    def test_complete_pipeline_accepts_honest_metric_gaps_and_preserves_all_74_old_figures(self):
        original = finish.historical_snapshot()
        calls = []
        def stage(data, figures, name, state, target):
            calls.append(name)
            state['status'] = name
            self.audit() if name == 'audit' else self.report()
        with patch.object(finish, 'run_stage', side_effect=stage):
            self.assertEqual(finish.finish('sweep'), 0)
        state = json.loads((self.data/'finalization_manifest.json').read_text())
        self.assertEqual(calls, ['audit', 'report'])
        self.assertEqual(state['status'], 'complete_verified')
        self.assertEqual(state['expected_model_grid_cases'], 216)
        self.assertEqual(len(state['figure_files']), 50)
        self.assertFalse(state['all_metrics_grid_stable'])
        self.assertFalse(state['compute_started_by_finalizer'])
        self.assertEqual(finish.historical_snapshot(), original)

    def test_closed_failed_computation_never_runs_audit_or_report(self):
        self.manifest['status'] = 'partial_failed'
        finish.io.write_json(self.data/'manifest.json', self.manifest)
        with patch.object(finish, 'run_stage') as stage:
            self.assertEqual(finish.finish('sweep'), 1)
        stage.assert_not_called()
        state = json.loads((self.data/'finalization_manifest.json').read_text())
        self.assertEqual(state['status'], 'compute_failed')

    def test_waiting_polls_existing_compute_and_does_not_start_another_controller(self):
        self.manifest['status'] = 'running'
        finish.io.write_json(self.data/'manifest.json', self.manifest)
        def stop(_):
            self.manifest['status'] = 'paused_by_user'
            finish.io.write_json(self.data/'manifest.json', self.manifest)
        with patch.object(finish.time, 'sleep', side_effect=stop) as sleep, patch.object(finish, 'run_stage') as stage:
            self.assertEqual(finish.finish('sweep', 20), 1)
        sleep.assert_called_once_with(20)
        stage.assert_not_called()

    def test_stage_failure_stops_before_publication(self):
        with patch.object(finish, 'run_stage', side_effect=RuntimeError('audit rejected')) as stage:
            self.assertEqual(finish.finish('sweep'), 1)
        self.assertEqual(stage.call_count, 1)
        self.assertEqual(json.loads((self.data/'finalization_manifest.json').read_text())['status'], 'failed')

    def test_single_finalizer_lock(self):
        with (self.data/'.finish.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                finish.finish('sweep')

    def test_corrupt_png_and_false_metric_retention_are_rejected(self):
        self.audit()
        report = self.report()
        report['metric_retention']['product2_conditional']['valid_cases'] = list(report['case_grids'])
        for path in (self.data/'report_manifest.json', self.figures/'report_manifest.json'):
            finish.io.write_json(path, report)
        with self.assertRaisesRegex(RuntimeError, 'omissions|retention'):
            finish.validate_report(self.data, self.figures, self.manifest, 216)
        report = self.report()
        png = next(name for name in report['figure_files'] if name.endswith('.png'))
        (self.root/png).write_bytes(b'invalid PNG bytes')
        report['figure_files'][png] = finish.io.digest(self.root/png)
        for path in (self.data/'report_manifest.json', self.figures/'report_manifest.json'):
            finish.io.write_json(path, report)
        with self.assertRaises(Exception):
            finish.validate_report(self.data, self.figures, self.manifest, 216)

    def test_subprocess_command_environment_logs_and_exit_are_checked(self):
        state = {}
        target = self.data/'finalization_manifest.json'
        with patch.object(finish.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            finish.run_stage(self.data, self.figures, 'audit', state, target)
        command = run.call_args.args[0]
        self.assertEqual(command[-2:], ['--name', 'sweep'])
        self.assertIn('oblivious.oe_three_sellers_fixed_p0.code.expanded_audit', command)
        self.assertEqual(run.call_args.kwargs['env']['OPENBLAS_NUM_THREADS'], '1')
        self.assertEqual(state['stages']['audit']['exit_code'], 0)
        with patch.object(finish.subprocess, 'run', return_value=SimpleNamespace(returncode=2)):
            with self.assertRaisesRegex(RuntimeError, 'report failed'):
                finish.run_stage(self.data, self.figures, 'report', state, target)


if __name__ == '__main__':
    unittest.main()
