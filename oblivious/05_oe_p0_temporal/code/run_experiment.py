"""Solve temporal p0 and compare it with the saved stationary endogenous p0."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from oblivious.oe_p0.code.core import posterior_mean
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import (
    case_tag, load_bundle, now, save_bundle, select_candidate, write_json,
)
from .model import TemporalExperiment
from .occupation import finite_window_law, finite_window_occupation


ROOT = Path(__file__).resolve().parents[3]
RHOS = [.6, .7, .8, .9, .95, .97, .99]


def source_hashes():
    paths = list((ROOT / 'oblivious/05_oe_p0_temporal/code').glob('*.py'))
    paths += [ROOT / 'oblivious/02_oe_p0/code' / name for name in
              ('core.py', 'game.py', 'policy.py', 'primitives.py', 'run_experiment.py')]
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if not p.name.startswith('test_') and p.name != 'reporting.py'}


def reference_case(config, rho, n):
    base = ROOT / 'oblivious/02_oe_p0/data' / config['reference_run'] / 'arrays' / case_tag(rho, n)
    case = load_bundle(base)
    params = asdict(Parameters(rho=rho))
    if case['configuration']['parameters'] != params or case['n'] != n:
        raise ValueError('Reference parameters/grid do not match the temporal experiment')
    selected = case['selected']
    if not all((selected.get('candidate_verified'), selected.get('verified'),
                selected['dp_diagnostics']['verified'], selected['stationary_diagnostics']['verified'])):
        raise ValueError('The original reference must be a saved verified candidate')
    provenance = dict(path=str(base.relative_to(ROOT)),
        sha256={suffix: sha256(base.with_suffix(suffix).read_bytes()).hexdigest()
                for suffix in ('.json', '.npz')})
    return case, provenance


def independent_validation(experiment, candidate, config):
    """Recompute the full joint chain, without lumping or early truncation."""
    n = experiment.n
    alpha = candidate['policy']
    matrix = experiment.game.transition(np.broadcast_to(alpha[:, None], (n, n)),
                                        np.broadcast_to(alpha[None, :], (n, n)))
    initial = np.zeros(n*n)
    initial[(n//2)*n+n//2] = 1.
    validation = finite_window_occupation(matrix, initial, periods=config['periods'],
        burn_in=config['burn_in'], tolerance=0., early_stop=False)
    # The utility's generic result is a vector occupation law.
    law = np.asarray(validation['law']).reshape(n, n)
    scalar = float(law.sum(axis=0) @ posterior_mean(experiment.grid, experiment.params))
    difference = float(np.abs(law-candidate['temporal_nu']).sum())
    residual = abs(scalar-candidate['p0'])
    verified = difference <= 1e-8 and residual <= config['consistency_tol']
    if not verified:
        raise ArithmeticError(f'Independent temporal validation failed: L1={difference:g}, G={residual:g}')
    return dict(verified=verified, method='Full joint kernel, all periods, no symmetry reduction or early stop',
                F=scalar, consistency_residual=residual, occupation_l1_difference=difference,
                diagnostics=validation['summary'])


def solve_case(config, rho, n):
    started = time.monotonic()
    old, provenance = reference_case(config, rho, n)
    experiment = TemporalExperiment(n=n, params=Parameters(rho=rho), order=config['order'],
        periods=config['periods'], burn_in=config['burn_in'], initial=.5,
        temporal_tolerance=config['temporal_tolerance'])
    prefix = f'rho={rho:g}, n={n}'
    print(f'{prefix}: solving temporal consistency', flush=True)
    search = experiment.search(p0_points=config['p0_points'], consistency_tol=config['consistency_tol'],
        progress=lambda message: print(f'  {prefix}: {message}', flush=True))
    # A narrow mixed branch can be missed by the finite sign-change scan.
    # Saved original candidates are only proposals: all optimality, stationarity
    # and NEW temporal-consistency conditions are freshly checked below.
    reference_trials = []
    for index, proposed in enumerate(old['candidates']):
        checked = experiment.evaluate_mixed(proposed['p0'], proposed['policy'], config['consistency_tol'])
        reference_trials.append(dict(original_candidate=index, p0=checked['p0'],
                                     F_temporal=checked['F'], G=checked['G'],
                                     candidate_verified=checked['candidate_verified']))
        if checked['candidate_verified'] and not any(
                abs(checked['p0']-found['p0']) <= 5*config['consistency_tol']
                and np.max(np.abs(checked['policy']-found['policy'])) <= 1e-6
                for found in search['candidates']):
            checked['proposal_source'] = 'Original stationary candidate, freshly reverified under temporal closure'
            search['candidates'].append(checked)
            search['evaluations'].append(checked)
    search['candidates'].sort(key=lambda item: (item['p0'], tuple(item['policy'])))
    search['summary'].update(verified_candidates=len(search['candidates']),
        verified_mixed_candidates=sum(c['mixed_nodes'] > 0 for c in search['candidates']),
        reference_candidates_reverified=len(reference_trials))
    search['search_log'].append(dict(method='Fresh temporal recertification of original candidates',
                                     trials=reference_trials))
    selected = select_candidate(search['candidates']) if search['candidates'] else None
    if selected is None:
        raise ArithmeticError(f'{prefix}: no verified temporal candidate found')
    validation = independent_validation(experiment, selected, config)
    reference = old['selected']
    reference_law = finite_window_law(experiment.game, reference['policy'], periods=config['periods'],
        burn_in=config['burn_in'], initial=.5, tolerance=config['temporal_tolerance'])
    reference_scalar = float(reference_law['mu_b'] @ posterior_mean(experiment.grid, experiment.params))
    # Same new policy evaluated in longer/delayed windows, not new equilibria.
    sensitivity = []
    for burn, length in ((config['burn_in'], 2*config['periods']),
                         (2*config['burn_in'], config['periods'])):
        measured = finite_window_law(experiment.game, selected['policy'], periods=length,
            burn_in=burn, initial=.5, tolerance=config['temporal_tolerance'])
        value = float(measured['mu_b'] @ posterior_mean(experiment.grid, experiment.params))
        sensitivity.append(dict(burn_in=burn, periods=length, F=value,
                                residual_at_fixed_policy=value-selected['p0'],
                                interpretation='Fixed selected policy; no equilibrium re-solved',
                                diagnostics=measured['summary']))
    case = dict(rho=rho, n=n, selected=selected, candidates=search['candidates'],
        configuration=experiment.configuration(), demand_audit=experiment.demand_audit(),
        map_rows=[{key: row.get(key) for key in ('p0', 'F', 'G', 'strategy_mode', 'exploratory')}
                  for row in search['evaluations']],
        search_summary=search['summary'], search_log=search['search_log'],
        failures=search['failures'], mixing_branches=search['mixing_branches'],
        reference=reference, reference_map_rows=old['map_rows'],
        reference_candidates=old['candidates'], reference_candidate_trials=reference_trials,
        reference_provenance=provenance,
        reference_temporal=dict(F=reference_scalar, nu=reference_law['nu'],
                                diagnostics=reference_law['summary']),
        independent_validation=validation, horizon_sensitivity=sensitivity,
        elapsed_seconds=time.monotonic()-started)
    print(f"{prefix}: p0={selected['p0']:.9f}, original={reference['p0']:.9f}, "
          f"residual={abs(selected['G']):.3g}, {case['elapsed_seconds']:.1f}s", flush=True)
    return case


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='temporal_n51')
    parser.add_argument('--rhos', nargs='+', type=float, default=RHOS)
    parser.add_argument('--grids', nargs='+', type=int, default=[51])
    parser.add_argument('--reference-run', default='rho_forgetting_mixed')
    parser.add_argument('--periods', type=int, default=100000)
    parser.add_argument('--burn-in', type=int, default=10000)
    parser.add_argument('--order', type=int, default=128)
    parser.add_argument('--p0-points', type=int, default=41)
    parser.add_argument('--consistency-tol', type=float, default=1e-7)
    parser.add_argument('--temporal-tolerance', type=float, default=1e-10)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    if (args.name in ('', '.', '..') or Path(args.name).name != args.name
            or Path(args.reference_run).name != args.reference_run
            or len(set(args.rhos)) != len(args.rhos) or len(set(args.grids)) != len(args.grids)
            or any(not .5 < rho < 1 for rho in args.rhos)
            or any(n < 3 or n % 2 != 1 for n in args.grids)
            or args.periods < 1 or args.burn_in < 0 or args.workers < 1
            or args.order < 8 or args.p0_points < 3
            or not 0 < args.temporal_tolerance < args.consistency_tol < .01):
        parser.error('Invalid name, grid, horizon or numerical settings')
    return args


def main():
    args = arguments()
    directory = ROOT / 'oblivious/05_oe_p0_temporal/data' / args.name
    figures = ROOT / 'oblivious/05_oe_p0_temporal/figures' / args.name
    manifest_path = directory / 'manifest.json'
    if args.resume or args.plots_only:
        manifest = json.loads(manifest_path.read_text())
        config = manifest['configuration']
        if not args.plots_only and source_hashes() != manifest['source_sha256']:
            raise SystemExit('Sources changed; use a new run name')
    else:
        if directory.exists() or figures.exists():
            raise SystemExit('Existing results protected; use --resume or a new name')
        config = {key: value for key, value in vars(args).items() if key not in ('resume', 'plots_only')}
        config['initial'] = [.5, .5]
        config['parameters'] = asdict(Parameters())
        (directory / 'arrays').mkdir(parents=True)
        manifest = dict(status='incomplete', started_utc=now(), configuration=config,
                        source_sha256=source_hashes(), completed_cases=[])
        write_json(manifest_path, manifest)
    figures.mkdir(parents=True, exist_ok=True)
    cases = []
    pending = []
    for rho in config['rhos']:
        for n in config['grids']:
            tag = case_tag(rho, n)
            if tag in manifest['completed_cases']:
                case = load_bundle(directory / 'arrays' / tag)
                _, provenance = reference_case(config, rho, n)
                if provenance != case['reference_provenance']:
                    raise RuntimeError('Saved reference changed')
                cases.append(case)
            else:
                pending.append((rho, n))
    from .reporting import export_results, make_figures
    try:
        if pending and args.plots_only:
            raise RuntimeError('Finish all cases before --plots-only')
        if pending:
            # Complete the motivating rho=.95 early; checkpoint every finished rate.
            pending.sort(key=lambda pair: (pair[0] != .95, pair[0], pair[1]))
            with ProcessPoolExecutor(max_workers=config['workers']) as pool:
                futures = {pool.submit(solve_case, config, rho, n): (rho, n) for rho, n in pending}
                for future in as_completed(futures):
                    case = future.result()
                    tag = case_tag(case['rho'], case['n'])
                    save_bundle(directory / 'arrays' / tag, case)
                    cases.append(case)
                    manifest['completed_cases'].append(tag)
                    write_json(manifest_path, manifest)
                    export_results(cases, directory, config)
        cases.sort(key=lambda case: (case['rho'], case['n']))
        export_results(cases, directory, config)
        make_figures(cases, figures)
        if not args.plots_only:
            if source_hashes() != manifest['source_sha256']:
                raise RuntimeError('Source files changed during the run')
            reporting_path = ROOT / 'oblivious/05_oe_p0_temporal/code/reporting.py'
            manifest.update(status='complete', completed_utc=now(),
                            reporting_source_sha256=sha256(reporting_path.read_bytes()).hexdigest())
            write_json(manifest_path, manifest)
    except BaseException as exc:
        if not args.plots_only:
            manifest.update(status='incomplete', last_error=dict(type=type(exc).__name__, message=str(exc), time=now()))
            write_json(manifest_path, manifest)
        raise


if __name__ == '__main__':
    main()
