"""Wait for the existing sweep, then audit and publish verified comparisons.

This process never launches or resumes computation. Grid-sensitive metric points
remain explicit omissions in the reporter; successful finalization does not
assert that every metric passed its comparison between grids.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

from . import expanded_run as campaign
from . import run_experiment as io

HOME, ROOT = io.HOME, io.ROOT
MODULE = 'oblivious.oe_three_sellers_fixed_p0.code.'
METRICS = {'product2_conditional', 'outside_rate', 'profit_calendar', 'value_initial'}
STEMS = ('01_policy_comparison', '02_product2_usage_comparison',
         '03_profit_comparison', '04_outside_option_comparison')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def fingerprint(paths):
    return {str(Path(path).resolve().relative_to(ROOT)):io.digest(path) for path in paths}


def source_snapshot():
    return fingerprint([HOME/'code'/name for name in (
        'expanded_finish.py', 'expanded_run.py', 'expanded_audit.py', 'audit.py',
        'expanded_report.py', 'report.py', 'run_experiment.py')])


def historical_snapshot():
    groups = [(HOME/'figures/marginal_pilot', 26),
              (ROOT/'oblivious/08_oe_distribution_fixed_p0/figures/p0_sweep', 48)]
    paths = []
    for directory, expected_count in groups:
        selected = sorted(path for path in directory.rglob('*') if path.suffix.lower() in ('.png', '.pdf'))
        require(len(selected) == expected_count, 'Historical figure inventory differs: '+str(directory))
        paths.extend(selected)
    return fingerprint(paths)


def check_scope(manifest, data=None):
    config = manifest['configuration']
    require(sorted(config['p0s']) == sorted(campaign.P0S)
            and sorted(config['rhos']) == sorted(campaign.RHOS), 'Requested p0/rho scope differs')
    require({tuple(item) for item in config['models']} == set(campaign.MODELS)
            and len(config['models']) == 3, 'Expected monopoly and marginal OE with two and three sellers')
    expected = campaign.expected(config)
    require(len(expected) == len(set(expected)), 'Duplicate configured model/grid cases')
    require(len(manifest['completed']) == len(set(manifest['completed']))
            and set(manifest['completed']) == set(expected), 'Closed campaign has incomplete model/grid cases')
    if data is not None:
        raw_paths = {str((data/(stem+suffix)).relative_to(ROOT))
                     for stem in expected for suffix in ('.json', '.npz')}
        require(raw_paths <= set(manifest['input_sha256']), 'Raw checkpoint fingerprints are incomplete')
    require(bool(manifest['numerical_sources']), 'Numerical source fingerprints are missing')
    io.check_hashes(manifest['numerical_sources'])
    io.check_hashes(manifest['input_sha256'])
    return len(expected)


def validate_report(data, figures, manifest, expected_count):
    report_path = data/'report_manifest.json'
    report = json.loads(report_path.read_text())
    require((figures/'report_manifest.json').read_bytes() == report_path.read_bytes(),
            'Data and figure report manifests differ')
    require(report.get('verified') is True and report.get('diagnostic_only') is False
            and report.get('source_campaign_status') == 'complete', 'Report is not a verified publication')
    require(report.get('status') in {'complete_gridstable', 'partial_gridstable', 'no_gridstable_metrics'},
            'Unrecognized grid-stability report status')
    require(sorted(report['requested_p0s']) == sorted(manifest['configuration']['p0s'])
            and sorted(report['requested_rhos']) == sorted(manifest['configuration']['rhos']), 'Report scope differs')
    require({(item['sellers'], item['closure']) for item in report['models']} == set(campaign.MODELS)
            and len(report['models']) == 3, 'Report seller-count comparison differs')
    require(len(report['case_grids']) == len(campaign.P0S)*len(campaign.RHOS), 'Report case map is incomplete')
    all_passed = True
    for p0 in campaign.P0S:
        for rho in campaign.RHOS:
            case = report['case_grids'][campaign.pair_key(p0, rho)]
            require(case['p0'] == p0 and case['rho'] == rho and set(case['metrics']) == METRICS,
                    'Report case parameters or metric map differ')
            for metric in METRICS:
                passed = case['metrics'][metric]['passed']
                require(isinstance(passed, bool), 'Metric validation is not explicit')
                all_passed &= passed
    require(report.get('all_metrics_grid_stable') is all_passed
            and (report['status'] == 'complete_gridstable') == all_passed,
            'Report overstates metric grid stability')
    require(set(report['metric_retention']) == METRICS, 'Metric retention map is incomplete')
    for metric in METRICS:
        retained = report['metric_retention'][metric]
        valid, excluded = set(retained['valid_cases']), set(retained['excluded_cases'])
        require(not valid & excluded and valid | excluded == set(report['case_grids']),
                'Metric omissions are not explicitly accounted for')
        require(valid == {key for key, case in report['case_grids'].items() if case['metrics'][metric]['passed']},
                'Metric retention contradicts its grid checks')
    for field in ('source_input_sha256', 'reporting_source_sha256', 'generated_data_sha256', 'figure_files'):
        require(isinstance(report.get(field), dict) and bool(report[field]), 'Missing report fingerprints: '+field)
        io.check_hashes(report[field])
    expected_files = set()
    for p0 in campaign.P0S:
        directory = figures/('p0_'+str(float(p0)).replace('.', 'p'))
        expected_files.update(str(path.relative_to(ROOT)) for stem in STEMS
                              for suffix in ('.png', '.pdf') for path in [directory/(stem+suffix)])
    expected_files.update(str((figures/('05_product2_by_seller_count'+suffix)).relative_to(ROOT))
                          for suffix in ('.png', '.pdf'))
    require(set(report['figure_files']) == expected_files, 'Expected 25 PNG/PDF figure pairs')
    for name in expected_files:
        path = ROOT/name
        if path.suffix == '.png':
            with Image.open(path) as picture:
                require(picture.width > 0 and picture.height > 0, 'Empty PNG: '+name)
                picture.verify()
            with Image.open(path) as picture:
                picture.load()
        else:
            require(path.read_bytes().startswith(b'%PDF-'), 'Invalid PDF header: '+name)
    audit = json.loads((data/'independent_audit.json').read_text())
    require(audit.get('verified') is True and audit.get('status') == 'complete_verified'
            and audit.get('expected_model_grid_cases') == expected_count
            and audit.get('audited_model_grid_cases') == expected_count
            and audit.get('manifest_snapshot_sha256') == io.digest(data/'manifest.json'),
            'Independent audit is incomplete or stale')
    require(audit.get('numerical_sources') == manifest['numerical_sources']
            and audit.get('campaign_input_sha256') == manifest['input_sha256'],
            'Independent audit does not match closed computation')
    io.check_hashes(audit['audit_sources'])
    io.check_hashes(audit['numerical_sources'])
    return report


def run_stage(data, figures, name, state, target):
    module = 'expanded_audit' if name == 'audit' else 'expanded_report'
    command = [sys.executable, '-B', '-m', MODULE+module]
    command += ['--name', data.name] if name == 'audit' else ['--data', str(data), '--figures', str(figures)]
    log_path = data/'logs'/('finish_'+name+'.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)
    state['status'] = name
    stage = dict(started_utc=io.now(), command=command, log=str(log_path.relative_to(ROOT)))
    state.setdefault('stages', {})[name] = stage
    io.write_json(target, state)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                       MPLCONFIGDIR='/tmp/llm_learning_expanded_finish_mpl')
    with log_path.open('a') as log:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, check=False)
    stage.update(finished_utc=io.now(), exit_code=result.returncode)
    io.write_json(target, state)
    require(result.returncode == 0, name+' failed; inspect '+str(log_path))


def finish(name='rho_p0_sweep', poll=20):
    require('/' not in name and '..' not in name and name, 'Use a simple campaign name')
    require(1 <= poll <= 60, 'Poll interval must be 1..60 seconds')
    data, figures = HOME/'data'/name, HOME/'figures'/name
    require((data/'manifest.json').is_file(), 'Existing computation manifest is required')
    target = data/'finalization_manifest.json'
    with (data/'.finish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        previous = json.loads(target.read_text()) if target.exists() else None
        state = dict(status='waiting_compute', started_utc=io.now(), campaign=name,
                     compute_started_by_finalizer=False, stages={})
        if previous is not None:
            state['previous_attempt'] = previous
        try:
            state['finalization_source_sha256'] = source_snapshot()
            state['historical_figure_sha256'] = historical_snapshot()
            io.write_json(target, state)
            while True:
                io.check_hashes(state['finalization_source_sha256'])
                manifest = json.loads((data/'manifest.json').read_text())
                state.update(compute_status=manifest['status'], completed_model_grid_cases=len(manifest['completed']),
                             checked_utc=io.now())
                io.write_json(target, state)
                if manifest['status'] == 'complete':
                    break
                if manifest['status'] not in {'created', 'running'}:
                    state['status'] = 'compute_failed'
                    raise RuntimeError('Computation stopped with status '+manifest['status'])
                time.sleep(poll)
            expected_count = check_scope(manifest, data)
            state.update(expected_model_grid_cases=expected_count,
                         closed_compute_manifest_sha256=io.digest(data/'manifest.json'),
                         numerical_source_sha256=manifest['numerical_sources'],
                         closed_input_sha256=manifest['input_sha256'])
            for stage in ('audit', 'report'):
                io.check_hashes(state['finalization_source_sha256'])
                io.check_hashes(manifest['numerical_sources'])
                io.check_hashes(manifest['input_sha256'])
                require(io.digest(data/'manifest.json') == state['closed_compute_manifest_sha256'],
                        'Closed computation manifest changed')
                run_stage(data, figures, stage, state, target)
            report = validate_report(data, figures, manifest, expected_count)
            io.check_hashes(state['finalization_source_sha256'])
            io.check_hashes(manifest['numerical_sources'])
            io.check_hashes(manifest['input_sha256'])
            require(io.digest(data/'manifest.json') == state['closed_compute_manifest_sha256'], 'Closed manifest changed')
            require(historical_snapshot() == state['historical_figure_sha256'], 'Historical figure files changed')
            state.update(status='complete_verified', finished_utc=io.now(),
                         report_status=report['status'], all_metrics_grid_stable=report['all_metrics_grid_stable'],
                         report_manifest_sha256=io.digest(data/'report_manifest.json'),
                         figure_files=report['figure_files'], verified=True,
                         scope='Verified saved finite-grid computation and report; metric omissions remain explicit')
            io.write_json(target, state)
            return 0
        except BaseException as error:
            if state['status'] != 'compute_failed':
                state['status'] = 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed'
            state.update(verified=False, error=repr(error), stopped_utc=io.now())
            try:
                if state.get('historical_figure_sha256'):
                    require(historical_snapshot() == state['historical_figure_sha256'], 'Historical figure files changed')
            except BaseException as historical_error:
                state['historical_preservation_error'] = repr(historical_error)
            io.write_json(target, state)
            if isinstance(error, KeyboardInterrupt):
                raise
            return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='rho_p0_sweep')
    parser.add_argument('--poll', type=float, default=20)
    args = parser.parse_args(argv)
    return finish(args.name, args.poll)


if __name__ == '__main__':
    raise SystemExit(main())
