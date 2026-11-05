"""Independently checkpoint uniform-grid refinement for one fixed-p0 case.

Example: OPENBLAS_NUM_THREADS=1 .venv/bin/python -B \
    oblivious/08_oe_distribution_fixed_p0/code/refine_case.py --rho .95

Interpolated policies are search seeds only. Every saved grid has a fresh,
unchanged demand, monopoly, joint stationary, and OE optimality certificate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment
from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    HOME, audit_demand, case_tag, check_bundle, economic_outcomes, now,
    numerical_sources, require, save_bundle, solve_competition, solve_monopoly,
    write_csv, write_json,
)


def source_fingerprints():
    return dict(numerical_sources(), **{
        str(Path(__file__).relative_to(ROOT)): sha256(Path(__file__).read_bytes()).hexdigest()})


def convergence(coarse, fine):
    """Apply predeclared probability (5pp) and relative money (5%) limits."""
    rows = []
    for earlier, later in zip(coarse['outcomes'], fine['outcomes']):
        require(earlier['model'] == later['model'], 'Outcome models differ')
        differences = {key: abs(later[key]-earlier[key]) for key in
                       ('product2_conditional', 'product2_calendar', 'sale_rate', 'outside_rate')}
        relative = {key: abs(later[key]-earlier[key])/max(abs(later[key]), np.finfo(float).tiny)
                    for key in ('profit_calendar', 'value_initial')}
        rows.append(dict(rho=fine['rho'], model=later['model'], coarse_n=coarse['n'],
                         fine_n=fine['n'], passed=all(value <= .05 for value in
                         list(differences.values())+list(relative.values())),
                         **{key+'_absolute_difference': value for key, value in differences.items()},
                         **{key+'_relative_difference': value for key, value in relative.items()}))
    return rows


def atomic_bundle(directory, tag, bundle):
    """Publish both files before the atomic manifest commit exposes a case."""
    temporary = directory/'arrays'/('_pending_'+tag)
    save_bundle(temporary, bundle)
    for suffix in ('.json', '.npz'):
        os.replace(temporary.with_suffix(suffix), (directory/'arrays'/tag).with_suffix(suffix))
    return {tag+suffix: sha256((directory/'arrays'/(tag+suffix)).read_bytes()).hexdigest()
            for suffix in ('.json', '.npz')}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rho', required=True, type=float)
    parser.add_argument('--grids', nargs='+', type=int, default=[101, 151, 201])
    parser.add_argument('--name', default='p0_0p5_refinement')
    parser.add_argument('--p0', type=float, default=.5)
    parser.add_argument('--order', type=int, default=256)
    parser.add_argument('--tol', type=float, default=1e-7)
    parser.add_argument('--dp-tol', type=float, default=1e-10)
    parser.add_argument('--max-evaluations', type=int, default=500)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--early-stop', action='store_true',
                        help='Optionally stop after two new grids pass; default completes all requested grids')
    args = parser.parse_args()
    if (not .5 < args.rho < 1 or args.rho == .999 or args.p0 != .5
            or Path(args.name).name != args.name or args.name in ('', '.', '..')
            or len(args.grids) < 2 or len(set(args.grids)) != len(args.grids)
            or any(n <= 51 or n % 2 == 0 for n in args.grids)
            or args.order < 8 or args.tol <= 0 or args.dp_tol <= 0 or args.max_evaluations < 1):
        parser.error('Use a distinct output name, p0=.5, .5<rho<1 except .999, and >=2 odd grids>51')
    return args


def main():
    args = arguments()
    rho_directory = 'rho_'+str(args.rho).replace('.', 'p')
    directory = HOME/'data'/args.name/rho_directory
    path = directory/'manifest.json'
    historical = HOME/'data/p0_0p5'
    historical_manifest = json.loads((historical/'manifest.json').read_text())
    original_tag = case_tag(args.rho, 51)
    original = check_bundle(historical, original_tag, historical_manifest)
    require(original['competition']['verified'] and original['monopoly']['verified'],
            'Original seed is not certified')
    config = dict(rho=args.rho, p0=args.p0, grids=sorted(args.grids), order=args.order,
                  tol=args.tol, dp_tol=args.dp_tol, max_evaluations=args.max_evaluations,
                  parameters={key:value for key,value in asdict(Parameters()).items() if key != 'rho'},
                  probability_tolerance=.05, money_relative_tolerance=.05,
                  comparison_denominator='absolute finer-grid outcome',
                  early_stop=args.early_stop,
                  stopping_rule='Complete requested grids unless --early-stop; minimum two new certified grids')
    if args.resume:
        manifest = json.loads(path.read_text())
        require(manifest['configuration'] == config, 'Saved configuration differs')
        require(manifest['numerical_sources'] == source_fingerprints(),
                'Numerical sources changed; use another output name')
    else:
        require(not directory.exists(), 'Existing results protected; use --resume or another name')
        (directory/'arrays').mkdir(parents=True)
        manifest = dict(status='running', created_utc=now(), configuration=config,
            numerical_sources=source_fingerprints(), completed=[], failures={}, checkpoint_sha256={},
            historical_seed=dict(directory=str(historical.relative_to(ROOT)), tag=original_tag,
                manifest_sha256=sha256((historical/'manifest.json').read_bytes()).hexdigest(),
                checkpoint_sha256={original_tag+s:historical_manifest['checkpoint_sha256'][original_tag+s]
                                   for s in ('.json', '.npz')},
                numerical_sources=historical_manifest['numerical_sources']), convergence=[])
        write_json(path, manifest)
    cases = []
    previous = original
    params = Parameters(rho=args.rho, **config['parameters'])
    for n in sorted(args.grids):
        tag = case_tag(args.rho, n)
        if tag in manifest['completed']:
            case = check_bundle(directory, tag, manifest)
        else:
            started = time.monotonic()
            manifest.update(status='running', current_grid=n, updated_utc=now())
            write_json(path, manifest)
            print(f'[{now()}] refine rho={args.rho:g}, n={n}, previous n={previous["n"]}', flush=True)
            try:
                experiment = FixedOutsideMarginalExperiment(n=n, params=params, p0=args.p0,
                    order=args.order, tol=args.tol, dp_tol=args.dp_tol)
                print('  auditing demand', flush=True)
                demand_audit = audit_demand(experiment)
                print('  solving/certifying monopoly', flush=True)
                monopoly = solve_monopoly(params, n, args.p0, args.dp_tol)
                seed = np.interp(experiment.grid, previous['competition']['grid'],
                                 previous['competition']['policy'])
                competition = solve_competition(experiment, monopoly,
                    [(f'verified_grid_{previous["n"]}', seed)], args.max_evaluations)
                outcomes, policy_rows = economic_outcomes(params, n, args.p0,
                                                          monopoly, competition, experiment.game)
                case = dict(rho=args.rho, n=n, p0=args.p0, configuration=experiment.configuration(),
                    monopoly=monopoly, competition=competition, demand_audit=demand_audit,
                    outcomes=outcomes, policy_rows=policy_rows, elapsed_seconds=time.monotonic()-started,
                    refinement_metadata=dict(seed_grid=previous['n'], seed_is_proposal_only=True,
                        certificate='Fresh unchanged model/optimality/stationarity/demand checks',
                        previous_completed_tag=case_tag(args.rho, previous['n'])))
                require(manifest['numerical_sources'] == source_fingerprints(),
                        'Sources changed during case; checkpoint not committed')
                manifest['checkpoint_sha256'].update(atomic_bundle(directory, tag, case))
                manifest['completed'].append(tag)
                manifest['failures'].pop(tag, None)
                print(f'  CERTIFIED in {case["elapsed_seconds"]:.1f}s; ' +
                      '; '.join(f'{row["model"]}: P2={row["product2_conditional"]:.6g}, '
                                f'profit={row["profit_calendar"]:.6g}' for row in outcomes), flush=True)
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
                manifest['failures'][tag] = str(exc)
                manifest.update(status='stopped_without_certificate', updated_utc=now())
                write_json(path, manifest)
                raise
        cases.append(case)
        manifest['convergence'] = [row for coarse, fine in zip([original]+cases, cases)
                                   for row in convergence(coarse, fine)]
        write_csv(directory/'convergence.csv', manifest['convergence'])
        write_csv(directory/'summary_all_grids.csv', [row for item in cases for row in item['outcomes']])
        write_csv(directory/'summary.csv', case['outcomes'])
        write_csv(directory/'policies.csv', case['policy_rows'])
        comparison = convergence(previous, case)
        previous = case
        manifest.update(updated_utc=now(), current_grid=None)
        write_json(path, manifest)
        print('  refinement comparison: '+json.dumps(comparison), flush=True)
        if args.early_stop and len(cases) >= 2 and all(row['passed'] for row in comparison):
            manifest.update(status='grid_stable_under_declared_thresholds', converged_grids=[cases[-2]['n'],n])
            break
    else:
        stable = len(cases) >= 2 and all(row['passed'] for row in comparison)
        manifest.update(status='grid_stable_under_declared_thresholds' if stable else 'certified_but_grid_sensitive',
                        converged_grids=[cases[-2]['n'],cases[-1]['n']] if stable else None)
    manifest.update(finished_utc=now(), updated_utc=now())
    write_json(path, manifest)
    print(f'{manifest["status"]}: {directory}', flush=True)


if __name__ == '__main__':
    main()
