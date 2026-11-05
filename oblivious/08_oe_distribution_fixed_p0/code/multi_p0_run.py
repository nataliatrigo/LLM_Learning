"""Fresh monopoly, marginal OE and conditional OE at a chosen outside p0.

Historical p0=.5 policies are proposals only. Each new model/grid has its own
atomic checkpoint and fresh unchanged numerical certificates. Appending n151
with --resume retains earlier checkpoints and never changes their calibration.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time
import traceback

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment
from oblivious.oe_distribution_fixed_p0.code.conditional_model import FixedOutsideConditionalExperiment
from oblivious.oe_distribution_fixed_p0.code.refine_case import atomic_bundle
from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    HOME, ROOT, audit_demand, case_tag, check_bundle, economic_outcomes,
    now, numerical_sources, require, save_bundle, solve_monopoly, write_csv, write_json,
)

RHOS = (.6, .7, .8, .9, .95, .97)
BASE_P0S = (.1, .3, .5, .7, .9)
PROBABILITY_METRICS = ('product2_conditional', 'product2_calendar', 'sale_rate', 'outside_rate')
EARNINGS_METRICS = ('profit_calendar', 'value_initial')


def number_tag(value):
    return str(float(value)).replace('.', 'p')


def campaign_directory(p0, rho):
    return HOME/'data/p0_sweep'/('p0_'+number_tag(p0))/('rho_'+number_tag(rho))


def source_fingerprints():
    extra = (Path(__file__), HOME/'code/conditional_model.py', HOME/'code/refine_case.py',
             ROOT/'oblivious/04_oe_conditional_distribution/code/model.py',
             ROOT/'oblivious/02_oe_p0/code/run_experiment.py')
    return dict(numerical_sources(), **{
        str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest() for path in extra})


def input_hashes(directory, tag):
    return {str((directory/'arrays'/(tag+suffix)).relative_to(ROOT)):
            sha256((directory/'arrays'/(tag+suffix)).read_bytes()).hexdigest()
            for suffix in ('.json', '.npz')}


def baseline_seed(rho, n, closure):
    """Verify immutable .5 inputs, returning a policy proposal, never an outcome."""
    seed_n = 51 if n < 101 else 101
    if closure == 'conditional':
        directory = HOME/'data/p0_0p5_conditional_fixed'/('rho_'+number_tag(rho))
    elif seed_n == 51:
        directory = HOME/'data/p0_0p5'
    else:
        campaign = 'p0_0p5_refinement' if rho in (.95, .97) else 'p0_0p5_common_grid'
        directory = HOME/'data'/campaign/('rho_'+number_tag(rho))
    path = directory/'manifest.json'
    manifest = json.loads(path.read_text())
    old_driver = 'oblivious/08_oe_distribution_fixed_p0/code/run_experiment.py'
    known_old_driver = 'b153368e3a3127de13c564e085d7e3e847a5ffdc1ae71123cc6b606976c1e0f8'
    for source, expected in manifest['numerical_sources'].items():
        actual = sha256((ROOT/source).read_bytes()).hexdigest()
        require(actual == expected or (source == old_driver and expected == known_old_driver),
                'Historical seed numerical source changed: '+source)
    tag = case_tag(rho, seed_n)
    require(tag in manifest['completed'], 'Historical seed is incomplete: '+tag)
    case = check_bundle(directory, tag, manifest)
    require(case['p0'] == .5 and case['rho'] == rho and case['n'] == seed_n
            and case['configuration']['parameters'] == asdict(Parameters(rho=rho))
            and case['competition']['verified'], 'Historical seed calibration/certificate differs')
    if closure == 'conditional':
        require(case['competition']['conditional_diagnostics']['independent_order_verified'],
                'Historical conditional seed lacks reversed-order certificate')
    inputs = input_hashes(directory, tag)
    inputs[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
    for source, expected in manifest.get('reference_input_sha256', {}).items():
        require(sha256((ROOT/source).read_bytes()).hexdigest() == expected,
                'Historical seed reference changed: '+source)
        inputs[source] = expected
    return case['competition'], inputs


def actual_values(experiment, competition):
    alpha, n = competition['policy'], experiment.n
    full = experiment.game.evaluate(np.broadcast_to(alpha[:, None], (n, n)),
                                    np.broadcast_to(alpha[None, :], (n, n)), tol=1e-10)
    competition['actual_values'] = full['values']
    competition['actual_value_diagnostics'] = {k:v for k,v in full.items() if k != 'values'}


def solve_from_proposals(experiment, proposals, max_evaluations, log):
    """Try exact policies first, followed by the established local solver."""
    attempts, unique, seen = [], [], set()
    for name, old in proposals:
        alpha = np.interp(experiment.grid, old['grid'], old['policy'])
        key = alpha.tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, alpha))
        log('  fresh proposal certificate: '+name)
        try:
            candidate = experiment.evaluate(alpha)
            attempts.append(dict(initialization=name, stage='fresh_certificate',
                verified=bool(candidate['verified']), dp=candidate['dp_diagnostics'],
                conditional=candidate.get('conditional_diagnostics', {})))
            if candidate['verified']:
                candidate['attempts'] = attempts
                candidate['search_metadata'] = dict(method='Fresh certificate of historical or prior-grid proposal',
                    seed_is_proposal_only=True, unique=False, exhaustive=False)
                return candidate
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            attempts.append(dict(initialization=name, stage='fresh_certificate', verified=False,
                                 error=str(error)))
    best = None
    for name, alpha in unique:
        log('  exact local solve: '+name)
        try:
            candidate = experiment.solve(initial_policy=alpha, prefer_local=True,
                max_evaluations=max_evaluations, progress=lambda message: log('    '+message))
            attempts.append(dict(initialization=name, stage='local_solve',
                verified=bool(candidate['verified']), dp=candidate['dp_diagnostics'],
                conditional=candidate.get('conditional_diagnostics', {}),
                search=candidate.get('search_metadata', {})))
            if best is None or candidate['dp_diagnostics']['conditional_regret'] < best['dp_diagnostics']['conditional_regret']:
                best = candidate
            if candidate['verified']:
                candidate['attempts'] = attempts
                return candidate
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            attempts.append(dict(initialization=name, stage='local_solve', verified=False, error=str(error)))
            log('    failed: '+str(error))
    error = ArithmeticError('No certified solution; attempts='+json.dumps(attempts))
    error.attempts = attempts
    error.best_candidate = best
    raise error


def stability(coarse, fine):
    earlier = {row['model']:row for row in coarse['outcomes']}
    records = []
    for b in fine['outcomes']:
        a = earlier[b['model']]
        probabilities = {k:abs(b[k]-a[k]) for k in PROBABILITY_METRICS}
        money = {k:abs(b[k]-a[k])/max(abs(a[k]), abs(b[k]))
                 if a[k] or b[k] else 0. for k in EARNINGS_METRICS}
        records.append(dict(p0=fine['p0'], rho=fine['rho'], model=b['model'],
            coarse_n=coarse['n'], fine_n=fine['n'],
            passed=all(v <= .05+1e-12 for v in [*probabilities.values(), *money.values()]),
            **{k+'_absolute_change':v for k,v in probabilities.items()},
            **{k+'_relative_change':v for k,v in money.items()}))
    return records


def run_grid(args, directory, manifest, n, closure, log):
    tag = case_tag(args.rho, n)+'_'+closure
    if tag in manifest['completed']:
        return check_bundle(directory, tag, manifest)
    started = time.monotonic()
    manifest.update(status='running', current_grid=n, current_model=closure, updated_utc=now())
    write_json(directory/'manifest.json', manifest)
    params = Parameters(rho=args.rho, **manifest['configuration']['parameters'])
    cls = FixedOutsideConditionalExperiment if closure == 'conditional' else FixedOutsideMarginalExperiment
    experiment = cls(n=n, params=params, p0=args.p0, order=args.order,
                     tol=args.tol, dp_tol=args.dp_tol)
    log(f'[{now()}] p0={args.p0:g}, rho={args.rho:g}, n={n}, {closure}')
    log('  fresh demand audit')
    demand_audit = audit_demand(experiment)
    seed, inputs = baseline_seed(args.rho, n, closure)
    proposals = []
    earlier = [g for g in manifest['requested_grids'] if g < n
               and case_tag(args.rho, g)+'_'+closure in manifest['completed']]
    if earlier:
        prior_tag = case_tag(args.rho, max(earlier))+'_'+closure
        prior = check_bundle(directory, prior_tag, manifest)
        proposals.append(('same_p0_previous_grid', prior['competition']))
        inputs.update(input_hashes(directory, prior_tag))
    proposals.append(('historical_p0_0p5_proposal', seed))
    if closure == 'conditional':
        marginal_tag = case_tag(args.rho, n)+'_marginal'
        reference = check_bundle(directory, marginal_tag, manifest)
        monopoly = reference['monopoly']
        proposals.append(('fresh_same_p0_same_grid_marginal', reference['competition']))
        inputs.update(input_hashes(directory, marginal_tag))
    else:
        log('  fresh monopoly solve')
        monopoly = solve_monopoly(params, n, args.p0, args.dp_tol)
        proposals.append(('fresh_monopoly', monopoly))
    manifest['reference_input_sha256'].update(inputs)
    write_json(directory/'manifest.json', manifest)
    try:
        competition = solve_from_proposals(experiment, proposals, args.max_evaluations, log)
    except ArithmeticError as error:
        if getattr(error, 'best_candidate', None) is not None:
            diagnostic = directory/'provenance'/('uncertified_'+tag)
            save_bundle(diagnostic, dict(configuration=experiment.configuration(),
                competition=error.best_candidate, attempts=error.attempts,
                explanation='Uncertified search diagnostic; not an equilibrium checkpoint'))
        raise
    if closure == 'conditional':
        require(competition['conditional_diagnostics']['independent_order_verified'],
                'Fresh conditional reversed-order certificate missing')
    log('  fresh actual joint discounted evaluation')
    actual_values(experiment, competition)
    outcomes, policies = economic_outcomes(params, n, args.p0, monopoly, competition, experiment.game)
    if closure == 'conditional':
        outcomes = [dict(row, model='conditional_competition') for row in outcomes if row['model'] == 'competition']
        policies = [dict(row, model='conditional_competition') for row in policies if row['model'] == 'competition']
    for row in outcomes:
        sellers = 1 if row['model'] == 'monopoly' else 2
        require(abs(row['outside_rate']+sellers*row['sale_rate']-1) < 1e-9,
                'Market outside selection identity failed')
    case = dict(p0=args.p0, rho=args.rho, n=n, configuration=experiment.configuration(),
        monopoly=monopoly, competition=competition, demand_audit=demand_audit,
        outcomes=outcomes, policy_rows=policies, elapsed_seconds=time.monotonic()-started,
        reference_input_sha256=inputs,
        refinement_metadata=dict(seed_is_proposal_only=True,
            demand='Fresh target-p0 demand and independent audit',
            monopoly='Fresh target-p0 same-grid monopoly',
            certificate='Fresh unchanged finite-grid optimality and positive joint stationary certificate',
            value='Fresh actual joint-environment discounted evaluation'))
    require(manifest['numerical_sources'] == source_fingerprints(), 'Sources changed during calculation')
    for source, expected in inputs.items():
        require(sha256((ROOT/source).read_bytes()).hexdigest() == expected, 'Seed input changed: '+source)
    manifest['checkpoint_sha256'].update(atomic_bundle(directory, tag, case))
    manifest['completed'].append(tag)
    manifest['failures'].pop(tag, None)
    manifest.update(current_grid=None, current_model=None, updated_utc=now())
    write_json(directory/'manifest.json', manifest)
    log(f'  CERTIFIED {tag} in {case["elapsed_seconds"]:.1f}s: '+json.dumps(outcomes))
    return case


def export_completed(directory, manifest):
    cases = [check_bundle(directory, tag, manifest) for tag in manifest['completed']]
    write_csv(directory/'summary_all_grids.csv', [row for case in cases for row in case['outcomes']])
    write_csv(directory/'policies_all_grids.csv', [row for case in cases for row in case['policy_rows']])
    comparisons = []
    for closure in ('marginal', 'conditional'):
        selected = sorted((case for tag,case in zip(manifest['completed'],cases)
                           if tag.endswith('_'+closure)), key=lambda case:case['n'])
        for coarse, fine in zip(selected, selected[1:]):
            comparisons.extend(stability(coarse, fine))
    write_csv(directory/'grid_stability.csv', comparisons)
    manifest['grid_stability'] = comparisons
    return comparisons


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p0', type=float, required=True)
    parser.add_argument('--rho', type=float, required=True, choices=RHOS)
    parser.add_argument('--grids', type=int, nargs='+', default=[51, 101])
    parser.add_argument('--order', type=int, default=256)
    parser.add_argument('--tol', type=float, default=1e-7)
    parser.add_argument('--dp-tol', type=float, default=1e-10)
    parser.add_argument('--max-evaluations', type=int, default=250)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (not 0 < args.p0 < 1 or args.p0 == .5 or len(set(args.grids)) != len(args.grids)
            or any(n < 3 or n % 2 == 0 for n in args.grids) or args.order < 8
            or args.tol <= 0 or args.dp_tol <= 0 or args.max_evaluations < 1):
        parser.error('Use 0<p0<1 other than reused .5, distinct odd grids, and positive numerical settings')
    return args


def main():
    args = arguments()
    directory = campaign_directory(args.p0, args.rho)
    path = directory/'manifest.json'
    configuration = dict(p0=args.p0, rho=args.rho, order=args.order, tol=args.tol,
        dp_tol=args.dp_tol, parameters={k:v for k,v in asdict(Parameters()).items() if k != 'rho'})
    if args.resume:
        manifest = json.loads(path.read_text())
        require(manifest['configuration'] == configuration, 'Saved numerical calibration differs')
        require(manifest['numerical_sources'] == source_fingerprints(), 'Numerical sources changed')
        for source, expected in manifest['reference_input_sha256'].items():
            require(sha256((ROOT/source).read_bytes()).hexdigest() == expected, 'Saved proposal input changed: '+source)
        manifest['requested_grids'] = sorted(set(manifest['requested_grids']) | set(args.grids))
    else:
        require(not directory.exists(), 'Existing results protected; use --resume')
        (directory/'arrays').mkdir(parents=True)
        manifest = dict(created_utc=now(), configuration=configuration,
            numerical_sources=source_fingerprints(), requested_grids=sorted(args.grids),
            completed=[], checkpoint_sha256={}, failures={}, reference_input_sha256={},
            status='running', current_grid=None, current_model=None, attempts=[])
    (directory/'provenance').mkdir(exist_ok=True)
    manifest['attempts'].append(dict(started_utc=now(), grids=sorted(args.grids), max_evaluations=args.max_evaluations))
    write_json(path, manifest)
    log_path = directory/'provenance/solver.log'

    def log(message):
        with log_path.open('a') as stream:
            stream.write(message+'\n')
        print(message, flush=True)

    try:
        for n in sorted(args.grids):
            for closure in ('marginal', 'conditional'):
                run_grid(args, directory, manifest, n, closure, log)
                export_completed(directory, manifest)
                write_json(path, manifest)
    except Exception as error:
        log(traceback.format_exc())
        tag = case_tag(args.rho, manifest['current_grid'])+'_'+manifest['current_model'] if manifest.get('current_grid') else 'campaign'
        manifest['failures'][tag] = dict(error=str(error), failed_utc=now())
        manifest.update(status='stopped_without_certificate', current_grid=None, current_model=None, updated_utc=now())
        export_completed(directory, manifest)
        write_json(path, manifest)
        raise
    comparisons = export_completed(directory, manifest)
    complete = all(case_tag(args.rho,n)+'_'+closure in manifest['completed']
                   for n in manifest['requested_grids'] for closure in ('marginal','conditional'))
    manifest.update(status=('complete_grid_checks_passed' if comparisons and all(r['passed'] for r in comparisons)
                            else 'complete_certified_with_grid_sensitivity') if complete else 'partial_certified',
                    current_grid=None, current_model=None, finished_utc=now())
    write_json(path, manifest)
    log(manifest['status'])


if __name__ == '__main__':
    main()
