"""Solve Conditional OE beside the saved fixed-p0 marginal comparisons.

Each rho has an independent checkpoint directory. The old monopoly solution
is reused unchanged; the conditional competition, demand audit and actual
joint discounted values are newly computed at each requested grid size.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code.conditional_model import (
    FixedOutsideConditionalExperiment,
)
from oblivious.oe_distribution_fixed_p0.code.refine_case import atomic_bundle
from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    HOME, ROOT, audit_demand, case_tag, check_bundle, economic_outcomes,
    now, numerical_sources, require, write_csv, write_json,
)

RHOS = (.6, .7, .8, .9, .95, .97)


def source_fingerprints():
    extra = [Path(__file__), HOME/'code/conditional_model.py',
             HOME/'code/refine_case.py',
             ROOT/'oblivious/04_oe_conditional_distribution/code/model.py']
    return dict(numerical_sources(), **{
        str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
        for path in extra})


def marginal_reference(rho, n):
    if n == 51:
        directory = HOME/'data/p0_0p5'
    else:
        campaign = 'p0_0p5_refinement' if rho in (.95, .97) else 'p0_0p5_common_grid'
        directory = HOME/'data'/campaign/('rho_'+str(rho).replace('.', 'p'))
    path = directory/'manifest.json'
    manifest = json.loads(path.read_text())
    tag = case_tag(rho, n)
    case = check_bundle(directory, tag, manifest)
    require(case['monopoly']['verified'] and case['competition']['verified'],
            'Marginal reference lacks certificates')
    require(case['configuration']['parameters'] == vars(Parameters(rho=rho))
            and case['p0'] == .5 and case['n'] == n, 'Marginal reference calibration differs')
    inputs = {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()}
    for suffix in ('.json', '.npz'):
        artifact = directory/'arrays'/(tag+suffix)
        inputs[str(artifact.relative_to(ROOT))] = sha256(artifact.read_bytes()).hexdigest()
    return case, inputs


def stability(coarse, fine):
    a = next(row for row in coarse['outcomes'] if row['model'] == 'conditional_competition')
    b = next(row for row in fine['outcomes'] if row['model'] == 'conditional_competition')
    probability = {key: abs(b[key]-a[key]) for key in
                   ('product2_conditional', 'product2_calendar', 'sale_rate')}
    relative = {key: abs(b[key]-a[key])/max(abs(a[key]), abs(b[key]))
                if a[key] or b[key] else 0. for key in ('profit_calendar', 'value_initial')}
    return dict(rho=fine['rho'], model='conditional_competition', coarse_n=coarse['n'],
        fine_n=fine['n'], passed=all(value <= .05 for value in
            [*probability.values(), *relative.values()]),
        max_probability_change_pp=100*max(probability.values()),
        max_earnings_relative_change_pct=100*max(relative.values()),
        **{key+'_absolute_change':value for key,value in probability.items()},
        **{key+'_relative_change':value for key,value in relative.items()})


def solve_case(experiment, reference, previous, max_evaluations):
    seeds = [('same_grid_marginal', reference['competition']['policy'])]
    if previous is not None:
        seeds.insert(0, ('previous_conditional_grid', np.interp(
            experiment.grid, previous['competition']['grid'], previous['competition']['policy'])))
    attempts, seen = [], set()
    for name, alpha in seeds:
        key = np.asarray(alpha).tobytes()
        if key in seen:
            continue
        seen.add(key)
        print(f'  conditional start={name}', flush=True)
        try:
            result = experiment.solve(initial_policy=alpha, prefer_local=True,
                max_evaluations=max_evaluations,
                progress=lambda message: print('    '+message, flush=True))
            attempts.append(dict(initialization=name, verified=bool(result['verified']),
                dp=result['dp_diagnostics'], conditional=result['conditional_diagnostics'],
                search=result.get('search_metadata', {})))
            if result['verified']:
                result['attempts'] = attempts
                return result
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            attempts.append(dict(initialization=name, verified=False, error=str(error)))
            print('    failed: '+str(error), flush=True)
    raise ArithmeticError('No certified Conditional OE solution: '+json.dumps(attempts))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rho', type=float, required=True, choices=RHOS)
    parser.add_argument('--name', default='p0_0p5_conditional_fixed')
    parser.add_argument('--grids', type=int, nargs='+', default=[51, 101])
    parser.add_argument('--order', type=int, default=256)
    parser.add_argument('--tol', type=float, default=1e-7)
    parser.add_argument('--dp-tol', type=float, default=1e-10)
    parser.add_argument('--max-evaluations', type=int, default=400)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (Path(args.name).name != args.name or args.name in ('', '.', '..')
            or sorted(args.grids) != [51, 101] or args.order < 8
            or args.tol <= 0 or args.dp_tol <= 0 or args.max_evaluations < 1):
        parser.error('Use a campaign name, grids 51 101 and positive numerical settings')
    return args


def main():
    args = arguments()
    directory = HOME/'data'/args.name/('rho_'+str(args.rho).replace('.', 'p'))
    path = directory/'manifest.json'
    parameters = Parameters(rho=args.rho)
    config = dict(rho=args.rho, p0=.5, grids=[51,101], order=args.order,
        tol=args.tol, dp_tol=args.dp_tol, max_evaluations=args.max_evaluations,
        parameters={key:value for key,value in vars(parameters).items() if key != 'rho'},
        model='Conditional OE with known fixed outside option',
        probability_change_limit=.05, relative_earnings_change_limit=.05)
    if args.resume:
        manifest = json.loads(path.read_text())
        require(manifest['configuration'] == config, 'Saved configuration differs')
        require(manifest['numerical_sources'] == source_fingerprints(), 'Numerical sources changed')
    else:
        require(not directory.exists(), 'Existing results protected; use --resume')
        (directory/'arrays').mkdir(parents=True)
        manifest = dict(created_utc=now(), status='running', configuration=config,
            numerical_sources=source_fingerprints(), completed=[], failures={},
            checkpoint_sha256={}, reference_input_sha256={}, current_grid=None)
        write_json(path, manifest)
    cases, previous = [], None
    for n in (51, 101):
        tag = case_tag(args.rho, n)
        if tag in manifest['completed']:
            case = check_bundle(directory, tag, manifest)
        else:
            started = time.monotonic()
            manifest.update(status='running', current_grid=n, updated_utc=now())
            write_json(path, manifest)
            print(f'[{now()}] conditional fixed p0=.5, rho={args.rho:g}, n={n}', flush=True)
            try:
                reference, inputs = marginal_reference(args.rho, n)
                manifest['reference_input_sha256'].update(inputs)
                experiment = FixedOutsideConditionalExperiment(n=n, params=parameters,
                    p0=.5, order=args.order, tol=args.tol, dp_tol=args.dp_tol)
                require(experiment.configuration()['parameters'] == reference['configuration']['parameters'],
                        'Conditional and marginal primitives differ')
                print('  auditing outside demand', flush=True)
                demand_audit = audit_demand(experiment)
                competition = solve_case(experiment, reference, previous, args.max_evaluations)
                print('  evaluating actual joint discounted values', flush=True)
                alpha = competition['policy']
                full = experiment.game.evaluate(np.broadcast_to(alpha[:,None], (n,n)),
                    np.broadcast_to(alpha[None,:], (n,n)), tol=1e-10)
                competition['actual_values'] = full['values']
                competition['actual_value_diagnostics'] = {key:value for key,value in full.items()
                                                          if key != 'values'}
                outcomes, policies = economic_outcomes(parameters, n, .5,
                    reference['monopoly'], competition, experiment.game)
                outcomes = [dict(row, model='conditional_competition') for row in outcomes
                            if row['model'] == 'competition']
                policies = [dict(row, model='conditional_competition') for row in policies
                            if row['model'] == 'competition']
                case = dict(rho=args.rho, n=n, p0=.5, configuration=experiment.configuration(),
                    monopoly=reference['monopoly'], competition=competition,
                    demand_audit=demand_audit, outcomes=outcomes, policy_rows=policies,
                    elapsed_seconds=time.monotonic()-started, reference_input_sha256=inputs,
                    refinement_metadata=dict(seed_is_proposal_only=True,
                        seed_grid=previous['n'] if previous else n,
                        certificate='Fresh conditional OE certificate with positive joint law and independent rowwise conditional-order check',
                        monopoly='Unmodified certified same-grid historical reference',
                        value='Fresh actual joint-environment discounted evaluation'))
                require(manifest['numerical_sources'] == source_fingerprints(),
                        'Sources changed during calculation')
                for source, expected in inputs.items():
                    require(sha256((ROOT/source).read_bytes()).hexdigest() == expected,
                            'Reference changed during calculation')
                manifest['checkpoint_sha256'].update(atomic_bundle(directory, tag, case))
                manifest['completed'].append(tag)
                manifest['failures'].pop(tag, None)
                print(f'  CERTIFIED in {case["elapsed_seconds"]:.1f}s: '+json.dumps(outcomes), flush=True)
            except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
                manifest['failures'][tag] = str(error)
                manifest.update(status='stopped_without_certificate', current_grid=None, updated_utc=now())
                write_json(path, manifest)
                raise
        cases.append(case)
        previous = case
        write_csv(directory/'summary_all_grids.csv', [row for item in cases for row in item['outcomes']])
        write_csv(directory/'summary.csv', case['outcomes'])
        write_csv(directory/'policies.csv', case['policy_rows'])
        manifest.update(current_grid=None, updated_utc=now())
        write_json(path, manifest)
    comparison = stability(cases[0], cases[1])
    write_csv(directory/'grid_stability.csv', [comparison])
    manifest.update(status='grid_stable_under_declared_thresholds' if comparison['passed']
                    else 'certified_but_grid_sensitive', grid_stability=comparison,
                    current_grid=None, finished_utc=now())
    write_json(path, manifest)
    print(manifest['status']+': '+json.dumps(comparison), flush=True)


if __name__ == '__main__':
    main()
