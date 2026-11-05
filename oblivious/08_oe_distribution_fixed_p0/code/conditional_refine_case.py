"""Optional n151 refinement of the fixed-p0 conditional OE at rho=.97.

This separate campaign requires a completed certified conditional n101 seed.
Interpolation provides only a search proposal: every accepted n151 policy
receives the original fresh positive and reversed-order conditional certificate.
Neither the original conditional campaign nor a historical monopoly is changed.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time
import traceback

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code.conditional_model import FixedOutsideConditionalExperiment
from oblivious.oe_distribution_fixed_p0.code.conditional_run import (
    marginal_reference, source_fingerprints as base_sources, stability,
)
from oblivious.oe_distribution_fixed_p0.code.refine_case import atomic_bundle
from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    HOME, ROOT, audit_demand, case_tag, check_bundle, economic_outcomes,
    now, require, write_csv, write_json,
)


def source_fingerprints():
    path = Path(__file__)
    return dict(base_sources(), **{
        str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()})


def conditional_seed(rho=.97):
    """Read n101 only after its writer closes and verify every source hash."""
    directory = HOME/'data/p0_0p5_conditional_fixed'/('rho_'+str(rho).replace('.', 'p'))
    path = directory/'manifest.json'
    manifest = json.loads(path.read_text())
    require(manifest.get('current_grid') is None
            and manifest['status'] in ('grid_stable_under_declared_thresholds',
                                       'certified_but_grid_sensitive'),
            'Conditional n101 campaign must be completed before refinement')
    require(manifest['numerical_sources'] == base_sources(), 'Conditional seed sources differ')
    tag = case_tag(rho, 101)
    require(tag in manifest['completed'], 'Certified conditional n101 seed missing')
    case = check_bundle(directory, tag, manifest)
    require(case['competition']['verified']
            and case['competition']['conditional_diagnostics']['independent_order_verified'],
            'Conditional seed lacks positive/reversed-order certificates')
    require(case['configuration']['parameters'] == vars(Parameters(rho=rho))
            and case['p0'] == .5 and case['n'] == 101,
            'Conditional seed calibration differs')
    inputs = {str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()}
    for suffix in ('.json', '.npz'):
        artifact = directory/'arrays'/(tag+suffix)
        inputs[str(artifact.relative_to(ROOT))] = sha256(artifact.read_bytes()).hexdigest()
    for source, expected in manifest.get('reference_input_sha256', {}).items():
        require(sha256((ROOT/source).read_bytes()).hexdigest() == expected,
                'Conditional seed historical reference changed')
        inputs[source] = expected
    for artifact, expected in manifest.get('diagnostic_artifact_sha256', {}).items():
        source = directory/artifact
        require(sha256(source.read_bytes()).hexdigest() == expected,
                'Conditional seed diagnostic artifact changed')
        inputs[str(source.relative_to(ROOT))] = expected
    return case, inputs


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rho', type=float, choices=[.97], default=.97)
    parser.add_argument('--grid', type=int, choices=[151], default=151)
    parser.add_argument('--name', default='p0_0p5_conditional_refinement')
    parser.add_argument('--order', type=int, default=256)
    parser.add_argument('--tol', type=float, default=1e-7)
    parser.add_argument('--dp-tol', type=float, default=1e-10)
    parser.add_argument('--max-evaluations', type=int, default=400)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight', action='store_true',
                        help='Audit only demand against historical n151, without starting a campaign')
    args = parser.parse_args()
    if (Path(args.name).name != args.name or args.name in ('', '.', '..')
            or args.order < 8 or args.tol <= 0 or args.dp_tol <= 0
            or args.max_evaluations < 1):
        parser.error('Use a campaign basename and positive numerical settings')
    return args


def main():
    args = arguments()
    parameters = Parameters(rho=args.rho)
    if args.preflight:
        reference, _ = marginal_reference(args.rho, args.grid)
        experiment = FixedOutsideConditionalExperiment(n=args.grid, params=parameters,
            p0=.5, order=args.order, tol=args.tol, dp_tol=args.dp_tol)
        require(experiment.configuration()['parameters'] == reference['configuration']['parameters'],
                'Historical monopoly calibration differs')
        print(json.dumps(dict(configuration=experiment.configuration(),
                              demand_audit=audit_demand(experiment))), flush=True)
        return

    # Validate both immutable inputs before creating a campaign directory.
    seed, seed_inputs = conditional_seed(args.rho)
    reference, monopoly_inputs = marginal_reference(args.rho, args.grid)
    inputs = dict(seed_inputs, **monopoly_inputs)
    directory = HOME/'data'/args.name/('rho_'+str(args.rho).replace('.', 'p'))
    path = directory/'manifest.json'
    config = dict(rho=args.rho, p0=.5, grids=[args.grid], comparison_grids=[101, args.grid],
        order=args.order, tol=args.tol, dp_tol=args.dp_tol,
        max_evaluations=args.max_evaluations,
        parameters={key:value for key,value in vars(parameters).items() if key != 'rho'},
        model='Conditional OE with known fixed outside option',
        probability_change_limit=.05, relative_earnings_change_limit=.05,
        conditional_seed_campaign='p0_0p5_conditional_fixed',
        monopoly_reference_campaign='p0_0p5_refinement')
    if args.resume:
        manifest = json.loads(path.read_text())
        require(manifest['configuration'] == config, 'Saved configuration differs')
        require(manifest['numerical_sources'] == source_fingerprints(), 'Numerical sources changed')
        require(manifest['reference_input_sha256'] == inputs, 'Saved reference inputs differ')
    else:
        require(not directory.exists(), 'Existing results protected; use --resume')
        (directory/'arrays').mkdir(parents=True)
        manifest = dict(created_utc=now(), status='running', configuration=config,
            numerical_sources=source_fingerprints(), completed=[], failures={},
            checkpoint_sha256={}, reference_input_sha256=inputs, current_grid=None)
        write_json(path, manifest)
    provenance = directory/'provenance'
    provenance.mkdir(exist_ok=True)
    log_path = provenance/'solver.log'

    def log(message):
        with log_path.open('a') as stream:
            stream.write(message+'\n')
        print(message, flush=True)

    tag = case_tag(args.rho, args.grid)
    if tag in manifest['completed']:
        case = check_bundle(directory, tag, manifest)
    else:
        started = time.monotonic()
        manifest.update(status='running', current_grid=args.grid, updated_utc=now())
        write_json(path, manifest)
        log(f'[{now()}] conditional fixed p0=.5, rho={args.rho:g}, n={args.grid}')
        try:
            n = args.grid
            experiment = FixedOutsideConditionalExperiment(n=n, params=parameters,
                p0=.5, order=args.order, tol=args.tol, dp_tol=args.dp_tol)
            require(experiment.configuration()['parameters'] == reference['configuration']['parameters'],
                    'Conditional and historical monopoly primitives differ')
            log('  auditing outside demand')
            demand_audit = audit_demand(experiment)
            initial = np.interp(experiment.grid, seed['competition']['grid'],
                                seed['competition']['policy'])
            log('  conditional start=certified_n101_interpolation (proposal only)')
            competition = experiment.solve(initial_policy=initial, prefer_local=True,
                max_evaluations=args.max_evaluations,
                progress=lambda message: log('    '+message))
            require(competition['verified']
                    and competition['conditional_diagnostics']['independent_order_verified'],
                    'No fresh positive/reversed-order conditional n151 certificate')
            log('  evaluating actual joint discounted values')
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
                refinement_metadata=dict(seed_is_proposal_only=True, seed_grid=101,
                    certificate='Fresh conditional OE certificate with positive joint law and independent rowwise conditional-order check',
                    monopoly='Unmodified certified n151 historical reference',
                    value='Fresh actual joint-environment discounted evaluation'))
            require(manifest['numerical_sources'] == source_fingerprints(),
                    'Sources changed during calculation')
            for source, expected in inputs.items():
                require(sha256((ROOT/source).read_bytes()).hexdigest() == expected,
                        'Reference changed during calculation')
            manifest['checkpoint_sha256'].update(atomic_bundle(directory, tag, case))
            manifest['completed'].append(tag)
            manifest['failures'].pop(tag, None)
            log(f'  CERTIFIED in {case["elapsed_seconds"]:.1f}s: '+json.dumps(outcomes))
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            log(traceback.format_exc())
            manifest['failures'][tag] = str(error)
            manifest.update(status='stopped_without_certificate', current_grid=None, updated_utc=now())
            write_json(path, manifest)
            raise

    comparison = stability(seed, case)
    write_csv(directory/'summary_all_grids.csv', [*seed['outcomes'], *case['outcomes']])
    write_csv(directory/'summary.csv', case['outcomes'])
    write_csv(directory/'policies.csv', case['policy_rows'])
    write_csv(directory/'grid_stability.csv', [comparison])
    manifest.update(status='grid_stable_under_declared_thresholds' if comparison['passed']
                    else 'certified_but_grid_sensitive', grid_stability=comparison,
                    current_grid=None, finished_utc=now())
    log(manifest['status']+': '+json.dumps(comparison))
    diagnostics = provenance/'verification_diagnostics.json'
    write_json(diagnostics, dict(archived_utc=now(), rho=args.rho, n=args.grid,
        dp=case['competition']['dp_diagnostics'],
        conditional=case['competition']['conditional_diagnostics'],
        stationary=case['competition']['stationary_diagnostics'],
        demand_audit=case['demand_audit'],
        actual_value_diagnostics=case['competition']['actual_value_diagnostics'],
        grid_stability=comparison))
    runner_copy = provenance/Path(__file__).name
    shutil.copyfile(Path(__file__), runner_copy)
    manifest['diagnostic_artifact_sha256'] = {
        str(item.relative_to(directory)): sha256(item.read_bytes()).hexdigest()
        for item in (log_path, diagnostics, runner_copy)}
    write_json(path, manifest)


if __name__ == '__main__':
    main()
