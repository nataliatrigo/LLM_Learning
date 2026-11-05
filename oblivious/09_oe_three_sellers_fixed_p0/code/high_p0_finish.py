"""Wait for an existing refinement, independently audit it, and publish plots.

This controller never starts or resumes model computation. It reuses the frozen
expanded finalizer's strict scope and publication checks, with a new reporter.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from . import expanded_finish as frozen
from . import run_experiment as io

HOME, ROOT = io.HOME, io.ROOT
MODULE = 'oblivious.oe_three_sellers_fixed_p0.code.'
require = frozen.require


def source_snapshot():
    return frozen.fingerprint([HOME/'code'/name for name in (
        'high_p0_finish.py', 'high_p0_report.py', 'expanded_finish.py',
        'expanded_audit.py', 'audit.py', 'expanded_report.py', 'report.py',
        'expanded_run.py', 'run_experiment.py')])


def historical_snapshot(figures):
    """Record actual prior publications, excluding only this new output tree."""
    figures = Path(figures).resolve()
    groups = [HOME/'figures', ROOT/'oblivious/08_oe_distribution_fixed_p0/figures/p0_sweep']
    paths = sorted({path.resolve() for directory in groups for path in directory.rglob('*')
                    if path.is_file() and not path.resolve().is_relative_to(figures)})
    return frozen.fingerprint(paths)


def run_stage(data, figures, stage, state, target):
    module = 'expanded_audit' if stage == 'audit' else 'high_p0_report'
    command = [sys.executable, '-B', '-m', MODULE+module]
    command += ['--name', data.name] if stage == 'audit' else ['--data', str(data), '--figures', str(figures)]
    log = data/'logs'/('high_p0_finish_'+stage+'.log')
    log.parent.mkdir(parents=True, exist_ok=True)
    state['status'] = stage
    record = dict(command=command, started_utc=io.now(), log=str(log.relative_to(ROOT)))
    state['stages'][stage] = record
    io.write_json(target, state)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
        MPLCONFIGDIR='/tmp/llm_learning_high_p0_finish_mpl')
    with log.open('a') as handle:
        result = subprocess.run(command, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
            stdout=handle, stderr=subprocess.STDOUT, check=False)
    record.update(exit_code=result.returncode, finished_utc=io.now())
    io.write_json(target, state)
    require(result.returncode == 0, stage+' failed; inspect '+str(log))


def validate(data, figures, manifest, count):
    report = frozen.validate_report(data, figures, manifest, count)
    references = report.get('primary_comparison_references', {})
    points = references.get('plotted_cases', [])
    valid = set(report['metric_retention']['product2_conditional']['valid_cases'])
    require({frozen.campaign.pair_key(point['p0'], point['rho']) for point in points} == valid
        and len(points) == len(valid), 'Monopoly references do not match retained P2 cases')
    for point in points:
        case = report['case_grids'][frozen.campaign.pair_key(point['p0'], point['rho'])]
        require(point['n'] == case['displayed_n'] and point['validation_coarse_n'] == case['validation_coarse_n'],
            'Monopoly reference uses a different case grid')
        require(len(point['product2_percent_by_sellers']) == 3
            and point['monopoly_reference_percent'] == point['product2_percent_by_sellers'][0],
            'Horizontal monopoly reference differs from the N1 outcome')
    diagnostic = json.loads((data/'high_p0_policy_diagnostics.json').read_text())
    require(len(diagnostic['within_grid_pairs']) == 72 and len(diagnostic['refinement_by_model']) == 36,
        'High-p0 policy diagnostics are incomplete')
    return report


def finish(name='high_p0_refinement_41', poll=20, expected_count=180):
    require(name and '/' not in name and '..' not in name, 'Use a simple campaign name')
    require(1 <= poll <= 60, 'Poll interval must be 1..60 seconds')
    data, figures = HOME/'data'/name, HOME/'figures'/name
    require((data/'manifest.json').is_file(), 'Existing computation manifest is required')
    target = data/'finalization_manifest.json'
    with (data/'.finish.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = dict(status='waiting_compute', started_utc=io.now(), campaign=name, stages={},
            compute_started_by_finalizer=False, finalization_source_sha256=source_snapshot(),
            historical_file_sha256=historical_snapshot(figures))
        if target.exists():
            state['previous_attempt'] = json.loads(target.read_text())
        try:
            while True:
                io.check_hashes(state['finalization_source_sha256'])
                manifest = json.loads((data/'manifest.json').read_text())
                state.update(compute_status=manifest['status'], checked_utc=io.now(),
                    completed_model_grid_cases=len(manifest['completed']))
                io.write_json(target, state)
                if manifest['status'] == 'complete':
                    break
                require(manifest['status'] in {'created', 'running'}, 'Computation closed with status '+manifest['status'])
                time.sleep(poll)
            require(not manifest.get('active_workers') and manifest.get('current_case') is None,
                'Closed campaign still reports active workers')
            count = frozen.check_scope(manifest, data)
            require(count == expected_count, 'Refinement size differs from the expected scope')
            closed_sha = io.digest(data/'manifest.json')
            state.update(expected_model_grid_cases=count, closed_compute_manifest_sha256=closed_sha)
            for stage in ('audit', 'report'):
                io.check_hashes(state['finalization_source_sha256'])
                io.check_hashes(manifest['numerical_sources'])
                io.check_hashes(manifest['input_sha256'])
                require(io.digest(data/'manifest.json') == closed_sha, 'Closed manifest changed')
                if (data/'report_manifest.json').exists():
                    validate(data, figures, manifest, count)
                    state['stages'][stage] = dict(status='existing_publication_verified')
                else:
                    run_stage(data, figures, stage, state, target)
            report = validate(data, figures, manifest, count)
            io.check_hashes(state['finalization_source_sha256'])
            io.check_hashes(manifest['input_sha256'])
            require(io.digest(data/'manifest.json') == closed_sha, 'Closed manifest changed')
            require(historical_snapshot(figures) == state['historical_file_sha256'], 'Historical publications changed')
            state.update(status='complete_verified', verified=True, finished_utc=io.now(),
                report_status=report['status'], all_metrics_grid_stable=report['all_metrics_grid_stable'],
                report_manifest_sha256=io.digest(data/'report_manifest.json'), figure_files=report['figure_files'])
            io.write_json(target, state)
            return 0
        except BaseException as error:
            state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                verified=False, error=repr(error), stopped_utc=io.now())
            io.write_json(target, state)
            if isinstance(error, KeyboardInterrupt):
                raise
            return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='high_p0_refinement_41')
    parser.add_argument('--poll', type=float, default=20)
    parser.add_argument('--expected-count', type=int, default=180)
    args = parser.parse_args()
    return finish(args.name, args.poll, args.expected_count)


if __name__ == '__main__':
    raise SystemExit(main())
