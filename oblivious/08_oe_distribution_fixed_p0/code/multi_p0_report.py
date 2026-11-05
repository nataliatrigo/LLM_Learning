"""Compare certified monopoly/marginal/conditional solutions across fixed p0.

This module never solves a model. It verifies saved checkpoints and numerical
sources, screens discretization sensitivity, and displays n101 for every p0.
Supplemental n151 checkpoints may validate n101 but never replace its curves.
The existing p0=.5 campaign is read in place and remains unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np

from oblivious.oe_p0.code.run_experiment import case_tag, now, write_csv, write_json
from oblivious.oe_distribution_fixed_p0.code.run_experiment import HOME, ROOT, check_bundle
from oblivious.oe_distribution_fixed_p0.code.common_grid_report import read_cases as read_marginal_cases
from oblivious.oe_distribution_fixed_p0.code.conditional_comparison_report import (
    EARNINGS_METRICS, MODELS, PROBABILITY_METRICS, RHOS, comparable_configuration,
    digest, displayed_records, read_conditional_cases, render_figures,
    sensitivity_selection,
)


P0S = [.1, .3, .5, .7, .8]
PROBABILITY_ROUNDOFF_TOLERANCE = 1e-12


def validate_market_outside_rows(rows):
    """Allow endpoint roundoff while requiring the whole-market accounting identity."""
    tolerance = PROBABILITY_ROUNDOFF_TOLERANCE
    for row in rows:
        sellers = 1 if row['model'] == 'monopoly' else 2
        outside, sales = float(row['outside_rate']), float(row['sale_rate'])
        if (not np.isfinite([outside, sales]).all()
                or not -tolerance <= outside <= 1+tolerance
                or not -tolerance <= sales <= 1+tolerance
                or abs(outside+sellers*sales-1) > 1e-9):
            raise ValueError(f"Outside rate is not a whole-market probability: {row['model']}, rho={row['rho']}")


def rendering_rows(rows):
    """Copy endpoint-rounded display values; keep every stored/exported value raw."""
    validate_market_outside_rows(rows)
    return [dict(row, outside_rate=float(np.clip(row['outside_rate'], 0, 1)),
                 sale_rate=float(np.clip(row['sale_rate'], 0, 1))) for row in rows]


def p0_name(p0):
    return 'p0_' + str(float(p0)).replace('.', 'p')


def shared_economics(configuration):
    """Only the external benchmark may vary across the p0 campaign."""
    result = comparable_configuration(configuration)
    result.pop('p0')
    return result


def verify_inputs(inputs):
    for source, expected in inputs.items():
        path = ROOT/source
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Input fingerprint differs: {source}')


def read_independent_audit(data, p0s, rhos):
    """Link an available audit and reject stale hashes or incomplete coverage."""
    path = Path(data)/'independent_review.json'
    if not path.exists():
        return None, {}
    audit = json.loads(path.read_text())
    requested = set(p0s)-{.5}
    audited = set(audit.get('p0s', []))
    if (audit.get('verified') is not True or not audited <= set(P0S)-{.5}
            or not requested <= audited or not set(rhos) <= set(audit.get('rhos', []))):
        raise ValueError('Independent audit is unverified or does not cover the current p0/rho selection')
    inputs = dict(audit.get('audited_input_sha256', {}))
    if not inputs:
        raise ValueError('Independent audit has no fingerprinted inputs')
    verify_inputs(inputs)
    source = audit['audit_source']
    source_path = (ROOT/source).resolve()
    if not source_path.is_relative_to(ROOT.resolve()):
        raise ValueError('Independent audit source must be inside the repository')
    source_hash = audit['audit_source_sha256']
    verify_inputs({source: source_hash})
    inputs[source] = source_hash
    review_hash = digest(path)
    inputs[str(path.relative_to(ROOT))] = review_hash
    summary = dict(path=str(path.relative_to(ROOT)), sha256=review_hash, verified=True,
        audit_source=source, audit_source_sha256=source_hash,
        p0s=audit['p0s'], rhos=audit['rhos'],
        audited_bundle_count=audit['audited_bundle_count'],
        audited_file_count=audit['audited_file_count'],
        audited_input_count=audit['audited_input_count'], residual_maxima=audit['residual_maxima'],
        scope=audit.get('scope'), limitations=audit.get('limitations'))
    return summary, inputs


def sweep_execution_provenance(sweep):
    """Verify archived execution and diagnostic-only supplemental certificates."""
    inputs = {}
    for field in ('prior_run_input_sha256', 'supplemental_monopoly_validation_sha256'):
        fingerprints = sweep.get(field, {})
        verify_inputs(fingerprints)
        inputs.update(fingerprints)
    completed = sweep.get('all_supplemental_151_models_completed')
    summary = dict(status=sweep['status'],
        prior_run_input_sha256=sweep.get('prior_run_input_sha256', {}),
        supplemental_monopoly_validation_sha256=sweep.get('supplemental_monopoly_validation_sha256', {}),
        globally_excluded_rhos=sweep.get('globally_excluded_rhos', []),
        global_exclusion_witnesses=sweep.get('global_exclusion_witnesses', {}),
        witness_scope='Diagnostic witnesses for the full current-p0 campaign; ordinary saved validation pairs determine this report selection',
        all_supplemental_151_models_completed=completed,
        supplemental_refinement_completion=sweep.get('supplemental_refinement_completion', []),
        supplemental_stop_reason=(
            'Optional remaining n151 refinements were stopped after a certified sensitivity failure already excluded their rho from the common-all-p0/all-model comparison. All requested 51/101 model results are preserved. Missing conditional n151 results in those excluded rho cases are not required for publication.'
            if completed is False else None))
    return summary, inputs


def validate_saved_case(case, p0, rho, n, closure, baseline=None):
    """Reject mixed calibration, wrong grids, or exports detached from policies."""
    config = case['configuration']
    if (case['p0'] != p0 or config['p0'] != p0 or case['rho'] != rho
            or config['parameters']['rho'] != rho or case['n'] != n or config['n'] != n):
        raise ValueError(f'Checkpoint p0/rho/grid mismatch: p0={p0}, rho={rho}, n={n}')
    economics = shared_economics(config)
    if baseline is not None and economics != baseline:
        raise ValueError(f'Shared economic configuration differs: p0={p0}, rho={rho}, n={n}')
    if not case['demand_audit'].get('verified', False):
        raise ValueError(f'Unverified demand audit: p0={p0}, rho={rho}, n={n}')
    if closure == 'conditional' and 'conditional' not in config.get('closure', '').lower():
        raise ValueError('Conditional closure is missing from configuration')
    if closure == 'marginal' and 'marginal' not in config.get('closure', '').lower():
        raise ValueError('Marginal closure is missing from configuration')
    selected_models = ('monopoly', 'competition') if closure == 'marginal' else ('conditional_competition',)
    outcome_rows = [row for row in case['outcomes'] if row['model'] in selected_models]
    policy_rows = [row for row in case['policy_rows'] if row['model'] in selected_models]
    if len(outcome_rows) != len(selected_models):
        raise ValueError('Missing or duplicate model outcome in checkpoint')
    grid = np.linspace(0, 1, n)
    params = config['parameters']
    for model in selected_models:
        result = case['monopoly' if model == 'monopoly' else 'competition']
        policy = np.asarray(result['policy'])
        if (not result.get('verified', False) or policy.shape != (n,)
                or not np.array_equal(result['grid'], grid) or not np.isfinite(policy).all()
                or np.any((policy < 0) | (policy > 1))):
            raise ValueError(f'Unverified or malformed saved policy: {model}')
        row = next(item for item in outcome_rows if item['model'] == model)
        selected = sorted((item for item in policy_rows if item['model'] == model),
                          key=lambda item: item['reputation'])
        if (not row.get('verified', False) or len(selected) != n
                or any(item['p0'] != p0 or item['rho'] != rho or item['n'] != n
                       for item in [row, *selected])
                or not np.array_equal([item['reputation'] for item in selected], grid)
                or not np.array_equal([item['product2_probability'] for item in selected], policy)):
            raise ValueError(f'Saved export metadata/policy mismatch: {model}')
        if any(not np.isfinite(row[metric]) for metric in PROBABILITY_METRICS+EARNINGS_METRICS):
            raise ValueError(f'Nonfinite saved outcome: {model}')
        if any(not -PROBABILITY_ROUNDOFF_TOLERANCE <= row[metric] <= 1+PROBABILITY_ROUNDOFF_TOLERANCE
               for metric in PROBABILITY_METRICS):
            raise ValueError(f'Saved probability outside [0, 1]: {model}')
        if model == 'monopoly':
            values = np.asarray(result['value'])
            if values.shape != (n,) or not np.isfinite(values).all():
                raise ValueError('Malformed monopoly value')
            actual_initial = float(values[n//2])
        else:
            values = np.asarray(result['actual_values'])
            if values.shape != (2, n, n) or not np.isfinite(values).all():
                raise ValueError('Actual-game competitive values are missing')
            actual_initial = float(values[:, n//2, n//2].mean())
        expected_product2 = row['sale_rate']*row['product2_conditional']
        expected_profit = row['sale_rate']*(params['R']-params['c1'])-(params['c2']-params['c1'])*expected_product2
        if (not np.isclose(row['value_initial'], actual_initial, rtol=0, atol=1e-12)
                or not np.isclose(row['product2_calendar'], expected_product2, rtol=0, atol=1e-12)
                or not np.isclose(row['profit_calendar'], expected_profit, rtol=0, atol=1e-12)):
            raise ValueError(f'Saved outcome accounting or actual-value identity differs: {model}')
    validate_market_outside_rows(outcome_rows)
    return outcome_rows, policy_rows, economics


def read_new_campaign(data, p0, rhos, *, baseline=None):
    rows, policies, inputs, statuses = [], [], {}, {}
    for rho in rhos:
        directory = Path(data)/p0_name(p0)/('rho_'+str(rho).replace('.', 'p'))
        path = directory/'manifest.json'
        state = dict(status='missing', source_directory=str(directory.relative_to(ROOT)),
                     available_grids={}, missing_checkpoints=[])
        statuses[str(rho)] = state
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        inputs[str(path.relative_to(ROOT))] = digest(path)
        declared = manifest['configuration']
        if declared['p0'] != p0 or declared['rho'] != rho:
            raise ValueError(f'Manifest p0/rho mismatch: {directory}')
        sources = manifest['numerical_sources']
        if not sources:
            raise ValueError(f'Numerical source fingerprints are missing: {directory}')
        verify_inputs(sources)
        inputs.update(sources)
        references = manifest.get('reference_input_sha256', {})
        verify_inputs(references)
        inputs.update(references)
        state.update(status=manifest['status'], current_grid=manifest.get('current_grid'),
                     failures=manifest.get('failures', {}), numerical_sources=sources)
        if manifest['status'] == 'running' or manifest.get('current_grid') is not None:
            state['status'] = 'running_not_published'
            continue
        for closure in ('marginal', 'conditional'):
            state['available_grids'][closure] = []
            for n in (51, 101, 151):
                tag = case_tag(rho, n)+'_'+closure
                if tag not in manifest['completed']:
                    state['missing_checkpoints'].append(tag)
                    continue
                case = check_bundle(directory, tag, manifest)
                for source, expected in case.get('reference_input_sha256', {}).items():
                    if references.get(source) != expected:
                        raise ValueError(f'Case/reference provenance differs: {directory}, {tag}')
                found_rows, found_policies, economics = validate_saved_case(
                    case, p0, rho, n, closure, baseline)
                baseline = economics if baseline is None else baseline
                if (case['configuration']['order'] != declared['order']
                        or case['configuration']['tolerance'] != declared['tol']
                        or case['configuration']['dp_tolerance'] != declared['dp_tol']
                        or {key: value for key, value in case['configuration']['parameters'].items()
                            if key != 'rho'} != declared['parameters']):
                    raise ValueError(f'Manifest/checkpoint configuration mismatch: {directory}, {tag}')
                for suffix in ('.json', '.npz'):
                    artifact = directory/'arrays'/(tag+suffix)
                    inputs[str(artifact.relative_to(ROOT))] = manifest['checkpoint_sha256'][artifact.name]
                rows.extend(found_rows)
                policies.extend(found_policies)
                state['available_grids'][closure].append(n)
    return rows, policies, inputs, statuses, baseline


def read_existing_half(rhos):
    """Read the original certified p0=.5 bundles, preserving all old artifacts."""
    marginal, marginal_status, inputs, configuration = read_marginal_cases(
        rhos, 51, 101, 'p0_0p5_common_grid', 'p0_0p5_refinement', {})
    if configuration is None:
        raise ValueError('Existing p0=.5 marginal calibration is unavailable')
    for state in marginal_status.values():
        inputs.update(state.get('numerical_sources', {}))
    conditional, conditional_status, more = read_conditional_cases(
        'p0_0p5_conditional_fixed', rhos, 51, 101, comparable_configuration(configuration))
    inputs.update(more)
    refined, refinement_status, more = read_conditional_cases(
        'p0_0p5_conditional_refinement', [.97] if .97 in rhos else [], 101, 151,
        comparable_configuration(configuration), required_grids=[151])
    inputs.update(more)
    conditional.extend(refined)
    rows = [row for case in marginal for row in case['outcomes']]
    policies = [row for case in marginal for row in case['policy_rows']]
    rows += [row for case in conditional for row in case['outcomes']
             if row['model'] == 'conditional_competition']
    policies += [row for case in conditional for row in case['policy_rows']
                 if row['model'] == 'conditional_competition']
    validate_market_outside_rows(rows)
    return rows, policies, inputs, dict(marginal=marginal_status, conditional=conditional_status,
                                       conditional_refinement=refinement_status), shared_economics(configuration)


def select_stable(rows, rhos):
    """Prefer the original grid check; use a saved finer check when it is needed."""
    default = sensitivity_selection(rows, rhos, 51, 101)
    lookup = {(row['rho'], row['model'], row['n']): row for row in rows}
    pairs = {}
    for rho in rhos:
        for model, *_ in MODELS:
            selected = next((row for row in default['comparisons'] if row['rho'] == rho
                             and row['model'] == model and row['used_for_selection']), None)
            if (selected is None or not selected['passed']) and (rho, model, 151) in lookup:
                pairs[rho, model] = (101, 151)
    return sensitivity_selection(rows, rhos, 51, 101, validation_pairs=pairs)


def common_rhos(selections, rhos):
    return [rho for rho in rhos if all(rho in item['included_rhos'] for item in selections.values())]


def validate_display_records(rows, policies, p0, rhos, grid=101):
    """A report must have exactly the same p0 and grid in every displayed curve."""
    fine, fine_policies = displayed_records(rows, policies, grid)
    fine = [row for row in fine if row['rho'] in rhos]
    fine_policies = [row for row in fine_policies if row['rho'] in rhos]
    models = {item[0] for item in MODELS}
    expected = {(rho, model) for rho in rhos for model in models}
    keys = [(row['rho'], row['model']) for row in fine]
    if (len(keys) != len(expected) or set(keys) != expected
            or any(row['p0'] != p0 or row['n'] != grid for row in fine+fine_policies)):
        raise ValueError('Displayed records mix p0/grid or omit/duplicate a required model')
    for rho, model in expected:
        selected = sorted((row for row in fine_policies if row['rho'] == rho and row['model'] == model),
                          key=lambda row: row['reputation'])
        if len(selected) != grid or not np.array_equal([row['reputation'] for row in selected], np.linspace(0, 1, grid)):
            raise ValueError('Displayed policy is not the original common grid')
    validate_market_outside_rows(fine)
    return fine, fine_policies


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', '--data-name', dest='data_name', default='p0_sweep')
    parser.add_argument('--p0s', type=float, nargs='+', default=P0S)
    parser.add_argument('--rhos', type=float, nargs='+', default=RHOS)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if (Path(args.data_name).name != args.data_name or not args.data_name
            or args.data_name in {'.', '..'} or len(set(args.p0s)) != len(args.p0s)
            or len(set(args.rhos)) != len(args.rhos) or not args.p0s or not args.rhos
            or any(p0 not in P0S for p0 in args.p0s) or any(rho not in RHOS for rho in args.rhos)):
        parser.error('Use a simple campaign name, current p0 values (0.1,0.3,0.5,0.7,0.8), and retained rho cases')
    args.p0s, args.rhos = sorted(args.p0s), sorted(args.rhos)
    return args


def main():
    args = arguments()
    data = HOME/'data'/args.data_name
    all_rows, all_policies, selections, statuses, inputs = [], [], {}, {}, {}
    baseline = None
    baseline_file = ROOT/'Fluid base model/config.json'
    baseline_p0s = json.loads(baseline_file.read_text())['p0_comparison']['values']
    inputs[str(baseline_file.relative_to(ROOT))] = digest(baseline_file)
    sweep_manifest = data/'sweep_manifest.json'
    sweep_execution = None
    if sweep_manifest.exists():
        sweep = json.loads(sweep_manifest.read_text())
        if sweep['status'] == 'running' and not args.check_only:
            raise SystemExit('Sweep is still running; no final figures published')
        inputs[str(sweep_manifest.relative_to(ROOT))] = digest(sweep_manifest)
        sweep_execution, sweep_inputs = sweep_execution_provenance(sweep)
        inputs.update(sweep_inputs)
        expected = sweep.get('baseline_configuration_sha256')
        if expected is not None:
            if digest(baseline_file) != expected:
                raise ValueError('Baseline p0 configuration fingerprint differs')
            inputs[str(baseline_file.relative_to(ROOT))] = expected
    # Read p0=.5 first to enforce its established economics across every new run.
    order = ([.5] if .5 in args.p0s else [])+[p0 for p0 in args.p0s if p0 != .5]
    for p0 in order:
        if p0 == .5:
            rows, policies, more, state, economics = read_existing_half(args.rhos)
            if baseline is not None and baseline != economics:
                raise ValueError('Existing p0=.5 economic calibration differs')
            baseline = economics
        else:
            rows, policies, more, state, economics = read_new_campaign(data, p0, args.rhos, baseline=baseline)
            baseline = economics if baseline is None else baseline
        inputs.update(more)
        statuses[str(p0)] = state
        selections[str(p0)] = select_stable(rows, args.rhos)
        all_rows.extend(rows)
        all_policies.extend(policies)
    independent_audit, audit_inputs = read_independent_audit(data, args.p0s, args.rhos)
    inputs.update(audit_inputs)
    retained = common_rhos(selections, args.rhos)
    overview = dict(status='complete' if retained == args.rhos else 'partial' if retained else 'no_common_stable_rhos',
                    requested_p0s=args.p0s, requested_rhos=args.rhos, displayed_n=101,
                    common_retained_rhos=retained, selections=selections, per_p0_provenance=statuses,
                    independent_audit=independent_audit, sweep_execution=sweep_execution)
    if args.check_only:
        print(json.dumps(overview, indent=2))
        return
    if not retained:
        raise SystemExit('No common certified, grid-stable rho cases; no figures published.\n'+json.dumps(overview, indent=2))
    verify_inputs(inputs)
    paths, coincidences, displayed_rows, displayed_policies = [], {}, [], []
    for p0 in args.p0s:
        rows = [row for row in all_rows if row['p0'] == p0]
        policies = [row for row in all_policies if row['p0'] == p0]
        fine, fine_policies = validate_display_records(rows, policies, p0, retained)
        displayed_rows.extend(fine)
        displayed_policies.extend(fine_policies)
        finer = [item for item in selections[str(p0)]['selected_validation_pairs']
                 if item['rho'] in retained and item['explicit_finer_check']]
        note = ('n = 101 checked against n = 151 where needed; see validation table.' if finer else None)
        if retained != args.rhos:
            note = (note+'\n' if note else '')+'Common certified, grid-stable ρ cases across all displayed p₀.'
        generated, coincidence = render_figures(rendering_rows(rows), policies, HOME/'figures'/args.data_name/p0_name(p0),
                                                retained, p0=p0, n=101, validation_note=note)
        paths.extend(generated)
        coincidences[str(p0)] = coincidence
    data.mkdir(parents=True, exist_ok=True)
    comparisons = [dict(p0=float(p0), **row) for p0, selection in selections.items()
                   for row in selection['comparisons']]
    generated_data = {
        'comparison_summary_all_grids.csv': all_rows,
        'comparison_summary.csv': displayed_rows,
        'comparison_policies.csv': displayed_policies,
        'comparison_grid_stability.csv': comparisons,
    }
    for name, rows in generated_data.items():
        write_csv(data/name, rows)
    write_json(data/'comparison_grid_sensitivity_selection.json', selections)
    write_json(data/'comparison_coincidence.json', coincidences)
    report = dict(created_utc=now(), **overview,
        models=[dict(model=model, label=label) for model, label, *_ in MODELS],
        p0_selection=dict(original_baseline_values=baseline_p0s, current_requested_values=args.p0s,
            user_adjustment='User replaced p0=0.9 with p0=0.8; original baseline configuration and archived p0=0.9 checkpoints remain unchanged and are excluded from this report'),
        shared_economic_configuration=baseline, source_input_sha256=inputs,
        reporting_source_sha256={str(Path(__file__).relative_to(ROOT)): digest(__file__),
            str((HOME/'code/conditional_comparison_report.py').relative_to(ROOT)): digest(HOME/'code/conditional_comparison_report.py'),
            str((HOME/'code/common_grid_report.py').relative_to(ROOT)): digest(HOME/'code/common_grid_report.py')},
        figure_files={str(path.relative_to(ROOT)): digest(path) for path in paths},
        generated_data_sha256={str((data/name).relative_to(ROOT)): digest(data/name) for name in generated_data},
        value_source='Actual game values; not perceived oblivious values',
        outside_option_measure='Whole market: monopoly 1-sale_rate; symmetric competition 1-2*sale_rate',
        probability_endpoint_roundoff=dict(tolerance=PROBABILITY_ROUNDOFF_TOLERANCE,
            validation='Saved probabilities may exceed an endpoint only within the stated roundoff tolerance; market accounting tolerance is 1e-9',
            exported_values='Raw saved probabilities are retained in all CSV exports and sensitivity checks',
            display='Only copied outside_rate and sale_rate values are clipped to [0,1] for compatibility with the original renderer'),
        display_scope='Every plotted/exported curve uses its original n101 solution; n151 only validates',
        scientific_scope='Observed finite-grid stability; not a proof of continuous convergence or uniqueness')
    verify_inputs(inputs)
    write_json(data/'comparison_report_manifest.json', report)
    for p0 in args.p0s:
        write_json(HOME/'figures'/args.data_name/p0_name(p0)/'figure_manifest.json', report)
    lines = [f"Comparación de p0: {', '.join(f'{p0:g}' for p0 in args.p0s)}.",
             'Por indicación del usuario, p0=0.8 reemplaza a p0=0.9; la configuración base original se conserva.',
             'Modelos: monopolio, OE marginal y OE condicional; todas las curvas usan n101.',
             'Rhos comunes certificados y estables: '+', '.join(f'{rho:g}' for rho in retained)+'.',
             'Fuera del conjunto común: '+(', '.join(f'{rho:g}' for rho in args.rhos if rho not in retained) or 'ninguno')+'.',
             'Filtro: máximo 5 pp en probabilidades y 5% relativo simétrico en beneficio/valor.',
             'Se conserva cada comparación 51→101; cuando falla y existe n151, se valida 101→151.',
             'La opción externa significa que ningún vendedor vende; una compra al rival no cuenta como outside.',
             'Los resultados originales p0=0.5 se reutilizan sin modificarlos.']
    if sweep_execution and sweep_execution['all_supplemental_151_models_completed'] is False:
        lines += [
            'Los suplementos n151 no se completaron para todos los modelos: se detuvieron los restantes cuando otro resultado certificado ya excluía ese rho de la comparación común.',
            'Los resultados solicitados en 51/101 se conservan; un condicional n151 pendiente en un rho ya excluido no se exige para publicar.',
            'Los certificados suplementarios de monopolio son diagnósticos de validación; ninguna curva n151 reemplaza a n101.',
        ]
    if sweep_execution:
        for rho, witness in sweep_execution['global_exclusion_witnesses'].items():
            if float(rho) in args.rhos:
                lines.append(f'Testigo de exclusión en la campaña completa, rho={rho}: {witness}')
    for p0 in args.p0s:
        excluded = selections[str(p0)]['excluded_rhos']
        if excluded:
            lines.append(f'p0={p0:g}: exclusiones por datos incompletos o sensibilidad: '+', '.join(excluded)+'.')
    (data/'COMPARACION.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
