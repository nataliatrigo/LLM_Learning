"""Independently audit the complete six-p0 n201/n251 comparison.

The event, invariant-law, value, optimality and export checks are reused from
the immutable historical raw-checkpoint audit. This entrypoint supplies only
the new campaign location and its mandatory complete 24-case scope. It never
calls an equilibrium, stationary-law, dynamic-programming or GTH solver.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import load_bundle
from oblivious.oe_distribution_fixed_p0.code import multi_p0_audit as raw
from oblivious.oe_distribution_fixed_p0.code.multi_p0_run import (
    number_tag, source_fingerprints, stability,
)
from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    HOME, ROOT, case_tag, now, write_json,
)


P0S = (.1, .3, .5, .7, .8, .9)
RHOS = (.6, .7, .8, .9)
GRIDS = (201, 251)
DATA = HOME/'data/p0_sweep_201'
OUTPUT = DATA/'independent_review.json'
CLOSURES = ('marginal', 'conditional')


def campaign_directory(p0, rho):
    return DATA/('p0_'+number_tag(p0))/('rho_'+number_tag(rho))


def required_tags(rho):
    return {case_tag(rho, n)+'_'+closure for n in GRIDS for closure in CLOSURES}


def closed_manifest(directory, p0, rho, expected_sources):
    """Reject partial work before doing any expensive audit reconstruction."""
    path = directory/'manifest.json'
    raw.require(path.is_file(), 'Current n201/n251 campaign is missing: '+str(path))
    manifest = json.loads(path.read_text())
    raw.require(manifest.get('current_grid') is None
                and manifest.get('current_model') is None
                and manifest.get('status') not in (None, 'running'),
                'Independent review requires a closed campaign: '+str(path))
    completed = manifest.get('completed', [])
    raw.require(len(completed) == len(set(completed))
                and set(completed) == required_tags(rho),
                'All n201/n251 marginal/conditional checkpoints are required: '+str(path))
    raw.require(manifest.get('numerical_sources') == expected_sources,
                'Immutable numerical source set differs: '+str(path))
    raw.require(sorted(manifest.get('requested_grids', [])) == list(GRIDS),
                'The new comparison requires exactly grids 201 and 251: '+str(path))
    configuration = manifest['configuration']
    raw.require(configuration['p0'] == p0 and configuration['rho'] == rho
                and configuration['parameters'] == {
                    k:v for k,v in asdict(Parameters()).items() if k != 'rho'},
                'Campaign economic calibration differs: '+str(path))
    return manifest


def verify_execution_history(master, add_input):
    """Check every phase's sources, including phases without a saved result."""
    history = master.get('execution_entrypoint_history', [])
    current = master.get('execution_entrypoint')
    raw.require(isinstance(history, list), 'Execution entrypoint history must be a list')
    if history:
        raw.require(current == history[-1], 'Current execution entrypoint differs from last history record')
    phases = history if history else ([current] if current is not None else [])
    for index, phase in enumerate(phases):
        raw.require(isinstance(phase, dict), f'Malformed execution phase {index}')
        sources = phase.get('source_input_sha256')
        raw.require(isinstance(sources, dict) and sources,
                    f'Execution phase {index} has no source fingerprints')
        for role, filename in (('scheduler', 'sweep201_batch.py'), ('worker', 'sweep201_run.py')):
            expected = phase.get('immutable_'+role+'_sha256')
            raw.require(isinstance(expected, str) and len(expected) == 64
                        and expected == master.get(role+'_sha256'),
                        f'Execution phase {index} immutable {role} differs from master')
            add_input(HOME/'code'/filename, expected)
        for source, expected in sources.items():
            raw.require(isinstance(source, str) and source
                        and not Path(source).is_absolute() and '..' not in Path(source).parts
                        and isinstance(expected, str) and len(expected) == 64,
                        f'Malformed execution phase {index} source fingerprint')
            add_input(ROOT/source, expected)
    return len(phases)


def audit_saved_checkpoints(job):
    """Read-only pair worker; partial inputs are for unpublished diagnostics.

    Only run_audit can publish a review, after its complete-scope preflight.
    """
    p0, rho, directory, manifest, tags = job
    directory = Path(directory)
    config, cache, cases, records = manifest['configuration'], raw.TargetCache(), {}, []
    for n in GRIDS:
        for closure in CLOSURES:
            tag = case_tag(rho, n)+'_'+closure
            if tag not in tags:
                continue
            prefix = directory/'arrays'/tag
            for suffix in ('.json', '.npz'):
                file = prefix.with_suffix(suffix)
                raw.require(raw.digest(file) == manifest['checkpoint_sha256'][file.name],
                            'Committed checkpoint hash differs: '+str(file))
            case = load_bundle(prefix)
            configuration = case['configuration']
            raw.require(case['n'] == n and case['rho'] == rho and case['p0'] == p0
                        and configuration['n'] == n and configuration['p0'] == p0
                        and configuration['parameters'] == asdict(Parameters(rho=rho))
                        and configuration['order'] == config['order']
                        and configuration['tolerance'] == config['tol']
                        and configuration['dp_tolerance'] == config['dp_tol']
                        and configuration.get('closure', '').lower().startswith('conditional')
                        == (closure == 'conditional'),
                        'Checkpoint calibration/tag differs: '+tag)
            for source, expected in case.get('reference_input_sha256', {}).items():
                raw.require(manifest['reference_input_sha256'].get(source) == expected
                            and raw.digest(ROOT/source) == expected,
                            'Checkpoint reference differs from manifest/source: '+source)
            if closure == 'conditional':
                raw.require((n, 'marginal') in cases,
                            'Conditional audit requires its same-grid marginal checkpoint')
                marginal = cases[n, 'marginal']
                for key in ('policy', 'mu', 'value', 'demand', 'advantage', 'sale_stationary', 'grid'):
                    raw.require(np.array_equal(case['monopoly'][key], marginal['monopoly'][key]),
                                'Same-grid closures must share their freshly certified monopoly: '+key)
                for suffix in ('.json', '.npz'):
                    path = directory/'arrays'/(case_tag(rho, n)+'_marginal'+suffix)
                    key = str(path.relative_to(ROOT))
                    raw.require(case.get('reference_input_sha256', {}).get(key)
                                == manifest['checkpoint_sha256'][path.name],
                                'Conditional checkpoint must fingerprint its same-grid marginal input')
            result = raw.audit_case(case, closure, cache.get(configuration))
            result.update(checkpoint=str(prefix.relative_to(ROOT)),
                checkpoint_sha256={tag+s:manifest['checkpoint_sha256'][tag+s]
                                   for s in ('.json', '.npz')})
            records.append(result)
            cases[n, closure] = case
            for suffix in ('.json', '.npz'):
                file = prefix.with_suffix(suffix)
                raw.require(raw.digest(file) == manifest['checkpoint_sha256'][file.name],
                            'Committed checkpoint changed during audit: '+str(file))
            print(f'AUDITED p0={p0:g} rho={rho:g} n={n} {closure}', flush=True)
    raw.require(len(records) == len(tags) and len(set(tags)) == len(tags),
                'Per-pair audit did not cover its exact committed tag set')
    comparisons = []
    for closure in CLOSURES:
        if (201, closure) in cases and (251, closure) in cases:
            comparisons.extend(stability(cases[201, closure], cases[251, closure]))
    return dict(records=records, comparisons=comparisons,
                cached_target_game_count=len(cache.values))


def audit_jobs(jobs, workers=6):
    """Keep result order deterministic while evaluating independent pairs."""
    raw.require(isinstance(workers, int) and not isinstance(workers, bool)
                and 1 <= workers <= 6, 'Audit workers must be an integer from 1 to 6')
    variables = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS')
    previous = {key:os.environ.get(key) for key in variables}
    try:
        os.environ.update({key:'1' for key in variables})
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
            return list(pool.map(audit_saved_checkpoints, jobs))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def run_audit(output=OUTPUT, workers=6):
    """Require and verify the entire authorized sweep; never publish a subset."""
    expected_sources = source_fingerprints()
    inputs, audited_bundles, records, statuses, comparisons = {}, {}, [], [], []
    entrypoint = Path(__file__).resolve()

    def add_input(path, expected=None):
        path = Path(path)
        raw.require(path.resolve().is_relative_to(ROOT.resolve()),
                    'Audit input must lie in the repository: '+str(path))
        key, actual = str(path.relative_to(ROOT)), raw.digest(path)
        raw.require(expected is None or actual == expected, 'Input/source hash differs: '+key)
        raw.require(key not in inputs or inputs[key] == actual, 'Input changed during audit: '+key)
        inputs[key] = actual

    add_input(entrypoint)
    add_input(Path(raw.__file__).resolve())
    master_path = DATA/'sweep_manifest.json'
    raw.require(master_path.is_file(), 'Complete n201/n251 sweep manifest is required')
    master = json.loads(master_path.read_text())
    raw.require(master.get('status') == 'complete_certified'
                and master.get('configuration') == dict(
                    p0s=list(P0S), rhos=list(RHOS), grids=list(GRIDS),
                    order=256, tol=1e-7, dp_tol=1e-10)
                and master.get('completed_p0_rho_pairs') == 24
                and master.get('certified_checkpoint_bundles') == 96
                and master.get('certified_model_grid_solutions') == 144
                and master.get('numerical_sources') == expected_sources,
                'Sweep manifest does not describe the complete certified requested comparison')
    add_input(master_path)
    add_input(HOME/'code/sweep201_batch.py', master['scheduler_sha256'])
    add_input(HOME/'code/sweep201_run.py', master['worker_sha256'])
    execution_phase_count = verify_execution_history(master, add_input)
    campaigns = []
    for p0 in P0S:
        for rho in RHOS:
            directory = campaign_directory(p0, rho)
            manifest = closed_manifest(directory, p0, rho, expected_sources)
            add_input(directory/'manifest.json')
            campaigns.append((p0, rho, directory, manifest))

    jobs = []
    for p0, rho, directory, manifest in campaigns:
        for source, expected in manifest['numerical_sources'].items():
            add_input(ROOT/source, expected)
        for source, expected in manifest['reference_input_sha256'].items():
            add_input(ROOT/source, expected)
        for n in GRIDS:
            for closure in CLOSURES:
                tag = case_tag(rho, n)+'_'+closure
                prefix = directory/'arrays'/tag
                for suffix in ('.json', '.npz'):
                    path = prefix.with_suffix(suffix)
                    expected = manifest['checkpoint_sha256'][path.name]
                    add_input(path, expected)
                    audited_bundles[str(path.relative_to(ROOT))] = expected
        jobs.append((p0, rho, directory, manifest, sorted(required_tags(rho))))
        statuses.append(dict(p0=p0, rho=rho, status=manifest['status'],
            completed=list(manifest['completed']), required_grids=list(GRIDS),
            failures=manifest.get('failures', {}), missing_base_checkpoints=[]))

    cached_target_game_count = 0
    for result in audit_jobs(jobs, workers):
        records.extend(result['records'])
        comparisons.extend(result['comparisons'])
        cached_target_game_count += result['cached_target_game_count']

    raw.require(len(campaigns) == 24 and len(records) == 96 and len(comparisons) == 72,
                'The full six-p0/four-rho/two-grid scope was not audited')
    for source, expected in inputs.items():
        raw.require(raw.digest(ROOT/source) == expected,
                    'Audited input changed before publication: '+source)
    maxima = {}
    for record in records:
        for name, value in record['residuals'].items():
            maxima[name] = max(maxima.get(name, 0.), value)
    relative = str(entrypoint.relative_to(ROOT))
    review = dict(created_utc=now(), verified=True,
        scope='Independent complete six-p0 saved finite-grid event/economic audit; '
              'original positive and reversed-order certificates required; '
              'no fresh GTH, equilibrium or dynamic-programming solves',
        limitations='The finite-grid certificates and n201/n251 sensitivity do not prove '
                    'uniqueness, continuous-state convergence or full-information '
                    'Markov-perfect equilibrium',
        p0s=list(P0S), rhos=list(RHOS), grids=list(GRIDS),
        publication_grid=201, validation_grid=251, required_case_count=24,
        execution_phase_count=execution_phase_count,
        publication_requirement_base_checkpoints_complete=True,
        model_grid_outcome_count=144, audited_bundle_count=len(records),
        audited_supplemental_monopoly_count=0, audited_total_bundle_count=len(records),
        audited_file_count=len(audited_bundles), audited_input_count=len(inputs),
        cached_target_game_count=cached_target_game_count,
        audit_workers=workers, worker_linear_algebra_threads=1,
        record_order='Requested p0, rho, grid, closure order; deterministic across worker counts',
        audit_source=relative, audit_source_sha256=inputs[relative],
        audit_entrypoint=relative, audit_entrypoint_sha256=inputs[relative],
        reused_raw_audit_source=str(Path(raw.__file__).resolve().relative_to(ROOT)),
        reused_raw_audit_source_sha256=raw.digest(Path(raw.__file__)),
        campaign_statuses=statuses, cases=records, residual_maxima=maxima,
        supplemental_monopoly_exclusions=[],
        grid_stability=comparisons,
        grid_stability_passed_all=all(row['passed'] for row in comparisons),
        grid_sensitivity_limits=dict(probability_absolute=.05, earnings_relative_symmetric=.05),
        audited_bundle_sha256=audited_bundles, audited_input_sha256=inputs,
        thresholds=raw.LIMITS)
    write_json(Path(output), review)
    return review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, choices=range(1, 7), default=6)
    args = parser.parse_args()
    review = run_audit(args.output, workers=args.workers)
    print(json.dumps({key:review[key] for key in (
        'verified', 'audited_bundle_count', 'model_grid_outcome_count',
        'grid_stability_passed_all', 'residual_maxima')}), flush=True)


if __name__ == '__main__':
    main()
