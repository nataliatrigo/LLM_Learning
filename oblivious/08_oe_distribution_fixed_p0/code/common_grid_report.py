"""Compare saved, certified fixed-p0 solutions on one common reputation grid.

The default report uses n=101, screened against historical n=51 for both models. Saved
solutions for rho=.95/.97 are reused in place; no policy is interpolated here.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from oblivious.oe_p0.code.run_experiment import case_tag, now, write_csv, write_json
from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    HOME, ROOT, RHOS, check_bundle, numerical_sources,
)
from oblivious.oe_distribution_fixed_p0.code.reporting import (
    MODELS, grid_sensitivity_selection, make_figures,
)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def campaign_name(value):
    if Path(value).name != value or value in ('', '.', '..'):
        raise ValueError('Campaign names must be single directory names')
    return value


def scientific_configuration(configuration):
    """Require the same economics, clock, closure and verification tolerances."""
    result = {key: value for key, value in configuration.items()
              if key not in ('n', 'order', 'parameters')}
    result['parameters'] = {key: value for key, value in configuration['parameters'].items()
                            if key != 'rho'}
    return result


def validate_case(case, rho, n, manifest):
    config, declared = case['configuration'], manifest['configuration']
    if (case['rho'] != rho or case['n'] != n or case['p0'] != .5
            or config['n'] != n or config['p0'] != .5
            or config['parameters']['rho'] != rho or declared['rho'] != rho
            or declared['p0'] != .5 or config['order'] != declared['order']
            or config['tolerance'] != declared['tol']
            or config['dp_tolerance'] != declared['dp_tol']
            or {key: value for key, value in config['parameters'].items() if key != 'rho'}
               != declared['parameters']):
        raise ValueError(f'Checkpoint/configuration mismatch: rho={rho}, n={n}')
    if not case['demand_audit'].get('verified', False):
        return 'Demand audit is not verified'
    expected_grid = np.linspace(0, 1, n)
    for model, *_ in MODELS:
        solution = case[model]
        if not solution.get('verified', False):
            return f'{model} has no certificate'
        if (np.asarray(solution['grid']).shape != (n,)
                or not np.array_equal(solution['grid'], expected_grid)
                or np.asarray(solution['policy']).shape != (n,)
                or not np.isfinite(solution['policy']).all()
                or np.any((solution['policy'] < 0) | (solution['policy'] > 1))):
            raise ValueError(f'Malformed saved policy: rho={rho}, n={n}, model={model}')
        outcomes = [row for row in case['outcomes'] if row['model'] == model]
        policies = [row for row in case['policy_rows'] if row['model'] == model]
        if (len(outcomes) != 1 or not outcomes[0].get('verified', False)
                or outcomes[0]['rho'] != rho or outcomes[0]['n'] != n
                or outcomes[0]['p0'] != .5 or len(policies) != n):
            raise ValueError(f'Malformed saved export: rho={rho}, n={n}, model={model}')
        policies = sorted(policies, key=lambda row: row['reputation'])
        if (any(row['rho'] != rho or row['n'] != n or row['p0'] != .5 for row in policies)
                or not np.array_equal([row['reputation'] for row in policies], expected_grid)
                or not np.array_equal([row['product2_probability'] for row in policies],
                                      solution['policy'])):
            raise ValueError(f'Policy export differs from stored solution: rho={rho}, n={n}, model={model}')
    return None


def read_cases(rhos, coarse_grid, grid, name, refinement_name, overrides):
    cases, statuses, inputs, baseline = [], {}, {}, None
    expected_sources = numerical_sources()
    historical_directory, historical_manifest = HOME/'data/p0_0p5', None
    if coarse_grid == 51:
        historical_path = historical_directory/'manifest.json'
        historical_bytes = historical_path.read_bytes()
        historical_manifest = json.loads(historical_bytes)
        historical_hash = sha256(historical_bytes).hexdigest()
        inputs[str(historical_path.relative_to(ROOT))] = historical_hash
        # The original sweep driver predates the default-rho/reporting edits.
        # All demand, game, policy and invariant-law kernels must still match.
        historical_driver = 'oblivious/08_oe_distribution_fixed_p0/code/run_experiment.py'
        known_driver_hash = 'b153368e3a3127de13c564e085d7e3e847a5ffdc1ae71123cc6b606976c1e0f8'
        for source, expected in expected_sources.items():
            saved = historical_manifest['numerical_sources'].get(source)
            if saved != expected and not (source == historical_driver and saved == known_driver_hash):
                raise ValueError(f'Historical numerical kernel differs: {source}')
    for rho in rhos:
        campaign = overrides.get(rho, refinement_name if rho in (.95, .97) else name)
        # A higher quadrature order resolves the same demand formula. New-grid
        # pairs share that campaign; historical n51 retains its audited order.
        if rho == .99 and rho not in overrides:
            for alternative in ('p0_0p5_common_grid_order2048', 'p0_0p5_common_grid_order512'):
                if (HOME/'data'/alternative/'rho_0p99/manifest.json').exists():
                    campaign = alternative
                    break
        directory = HOME/'data'/campaign/('rho_'+str(rho).replace('.', 'p'))
        path = directory/'manifest.json'
        state = dict(rho=rho, campaign=campaign, source_directory=str(directory.relative_to(ROOT)),
                     required_grids=[coarse_grid, grid], available_verified_grids=[],
                     checkpoint_sha256={}, per_grid_provenance={}, reasons=[])
        statuses[str(rho)] = state
        if not path.exists():
            state.update(status='pending', source_status='missing_manifest')
            state['reasons'].append('Source manifest does not exist yet')
            continue
        encoded = path.read_bytes()
        manifest = json.loads(encoded)
        inputs[str(path.relative_to(ROOT))] = sha256(encoded).hexdigest()
        state.update(source_status=manifest['status'], current_grid=manifest.get('current_grid'),
                     quadrature_order=manifest['configuration']['order'],
                     source_failures=manifest.get('failures', {}))
        for artifact, expected in manifest.get('diagnostic_artifact_sha256', {}).items():
            diagnostic = directory/artifact
            if digest(diagnostic) != expected:
                raise ValueError(f'Diagnostic artifact changed: {diagnostic}')
            inputs[str(diagnostic.relative_to(ROOT))] = expected
        state['diagnostic_artifact_sha256'] = manifest.get('diagnostic_artifact_sha256', {})
        if 'grid_sensitivity_diagnostic' in manifest:
            state['grid_sensitivity_diagnostic'] = manifest['grid_sensitivity_diagnostic']
        source_hashes = manifest['numerical_sources']
        if any(source_hashes.get(source) != expected for source, expected in expected_sources.items()):
            raise ValueError(f'Numerical source fingerprints differ: {directory}')
        for source, expected in source_hashes.items():
            if digest(ROOT/source) != expected:
                raise ValueError(f'Numerical source changed: {source}')
        state['numerical_sources'] = source_hashes
        if historical_manifest is not None:
            seed = manifest['historical_seed']
            if (seed['manifest_sha256'] != historical_hash
                    or seed['numerical_sources'] != historical_manifest['numerical_sources']
                    or seed['tag'] != case_tag(rho, 51)
                    or any(seed['checkpoint_sha256'].get(case_tag(rho,51)+suffix)
                           != historical_manifest['checkpoint_sha256'][case_tag(rho,51)+suffix]
                           for suffix in ('.json', '.npz'))):
                raise ValueError(f'Historical seed provenance differs: rho={rho}')
        pair = []
        for n in (coarse_grid, grid):
            tag = case_tag(rho, n)
            case_directory = historical_directory if n == 51 else directory
            case_manifest = historical_manifest if n == 51 else manifest
            if tag not in case_manifest['completed']:
                failure = case_manifest.get('failures', {}).get(tag)
                state['reasons'].append(f'n={n}: '+(failure or 'No committed certificate yet'))
                continue
            case = check_bundle(case_directory, tag, case_manifest)
            for suffix in ('.json', '.npz'):
                artifact = case_directory/'arrays'/(tag+suffix)
                expected = case_manifest['checkpoint_sha256'][artifact.name]
                inputs[str(artifact.relative_to(ROOT))] = expected
                state['checkpoint_sha256'][artifact.name] = expected
            normalized = dict(case_manifest, configuration=dict(case_manifest['configuration'], rho=rho))
            reason = validate_case(case, rho, n, normalized)
            if reason:
                state['reasons'].append(f'n={n}: {reason}')
                continue
            scientific = scientific_configuration(case['configuration'])
            if baseline is None:
                baseline = scientific
            elif scientific != baseline:
                raise ValueError(f'Economics/model/tolerances differ: rho={rho}, n={n}')
            pair.append(case)
            state['available_verified_grids'].append(n)
            state['per_grid_provenance'][str(n)] = dict(
                source_directory=str(case_directory.relative_to(ROOT)),
                source_manifest_sha256=inputs[str((case_directory/'manifest.json').relative_to(ROOT))],
                quadrature_order=case['configuration']['order'],
                numerical_sources=case_manifest['numerical_sources'],
                historical_driver_hash_preserved=(n == 51))
        if coarse_grid > 51 and len({case['configuration']['order'] for case in pair}) > 1:
            raise ValueError(f'Coarse and fine quadrature orders differ: rho={rho}')
        if manifest['status'] == 'running':
            state['reasons'].append('Source worker is still running')
        state['status'] = 'complete' if len(pair) == 2 and not state['reasons'] else 'incomplete'
        # Partial certified grids remain available in the audit CSV, but cannot
        # supply a plotted fine-grid result without their complete comparison.
        cases.extend(pair)
    return sorted(cases, key=lambda case:(case['rho'], case['n'])), statuses, inputs, baseline


def stability_rows(selection):
    rows = []
    for item in selection['comparisons']:
        probability, relative = item['absolute_probability_changes'], item['relative_changes']
        passed = (len(probability) == len(selection['probability_metrics'])
                  and len(relative) == len(selection['relative_metrics'])
                  and all(np.isfinite(value) and value <= selection['probability_change_limit'] + 1e-12
                          for value in probability.values())
                  and all(np.isfinite(value) and value <= selection['relative_earnings_change_limit'] + 1e-12
                          for value in relative.values()))
        rows.append(dict(rho=item['rho'], model=item['model'], coarse_n=item['coarse_n'],
            fine_n=item['fine_n'], passed=passed,
            both_models_included=item['rho'] in selection['included_rhos'],
            max_probability_change_pp=100*max(probability.values(), default=np.nan),
            max_earnings_relative_change_pct=100*max(relative.values(), default=np.nan),
            **{key+'_absolute_change': value for key, value in probability.items()},
            **{key+'_relative_change': value for key, value in relative.items()}))
    return rows


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='p0_0p5_common_grid')
    parser.add_argument('--refinement-name', default='p0_0p5_refinement')
    parser.add_argument('--rhos', type=float, nargs='+', default=RHOS)
    parser.add_argument('--grid', type=int, default=101)
    parser.add_argument('--coarse-grid', type=int, default=51)
    parser.add_argument('--source', action='append', default=[], metavar='RHO=CAMPAIGN',
                        help='Override one rho source campaign; may be repeated')
    parser.add_argument('--allow-incomplete', action='store_true',
                        help='Explicitly exclude incomplete rho pairs and report their reasons')
    parser.add_argument('--check-only', action='store_true', help='Validate sources without writing a report')
    args = parser.parse_args()
    try:
        campaign_name(args.name)
        campaign_name(args.refinement_name)
        overrides = {}
        for item in args.source:
            key, campaign = item.split('=', 1)
            rho = float(key)
            if rho not in args.rhos or rho in overrides:
                raise ValueError('Source rho must appear exactly once among requested rhos')
            overrides[rho] = campaign_name(campaign)
        if (len(set(args.rhos)) != len(args.rhos) or not args.rhos
                or any(not .5 < rho < 1 or rho == .999 for rho in args.rhos)
                or not 51 <= args.coarse_grid < args.grid
                or args.coarse_grid % 2 != 1 or args.grid % 2 != 1):
            raise ValueError('Use unique .5<rho<1 except .999 and two increasing odd grids >=51')
    except ValueError as exc:
        parser.error(str(exc))
    args.overrides = overrides
    return args


def main():
    args = arguments()
    cases, statuses, inputs, configuration = read_cases(
        sorted(args.rhos), args.coarse_grid, args.grid, args.name, args.refinement_name, args.overrides)
    incomplete = [rho for rho, state in statuses.items() if state['status'] != 'complete']
    if incomplete and not args.allow_incomplete:
        raise SystemExit('Incomplete rho comparisons; wait or use --allow-incomplete for explicit exclusions:\n'
                         +json.dumps({rho: statuses[rho] for rho in incomplete}, indent=2))
    rows = [row for case in cases for row in case['outcomes']]
    policies = [row for case in cases for row in case['policy_rows']]
    complete_rows = [row for row in rows if statuses[str(row['rho'])]['status'] == 'complete']
    selection = grid_sensitivity_selection(complete_rows, p0=.5, n=args.grid)
    for rho in incomplete:
        selection['excluded_rhos'][rho] = [dict(reason=reason) for reason in statuses[rho]['reasons']]
    selection.update(requested_rhos=sorted(args.rhos), coarse_grid=args.coarse_grid, fine_grid=args.grid)
    comparisons = stability_rows(selection)
    for rho, state in statuses.items():
        state['presentation_status'] = ('included' if float(rho) in selection['included_rhos']
            else 'grid_sensitive' if state['status'] == 'complete' else 'excluded_incomplete')
    overview = []
    for rho, state in statuses.items():
        checked = [row for row in comparisons if row['rho'] == float(rho)]
        overview.append(dict(rho=float(rho), status=state['presentation_status'],
            campaign=state['campaign'], quadrature_order=state.get('quadrature_order'),
            coarse_n=args.coarse_grid, fine_n=args.grid,
            max_probability_change_pp=max((row['max_probability_change_pp'] for row in checked), default=None),
            max_earnings_relative_change_pct=max((row['max_earnings_relative_change_pct'] for row in checked), default=None),
            reason=json.dumps(selection['excluded_rhos'].get(rho, []))))
    lines = ['Comparación con grilla común: modelo con opción externa fija p0=0.5',
        f'Grilla presentada: {args.grid}; contraste de sensibilidad: {args.coarse_grid}→{args.grid}.',
        'Cada curva utiliza su solución guardada en esa grilla; no se interpolan políticas.',
        'Uso/ventas: límite 5 puntos porcentuales. Beneficio/valor: límite relativo 5% simétrico.',
        'La estabilidad observada no demuestra convergencia al continuo ni unicidad.',
        f"Rhos incluidos: {selection['included_rhos']}",
        f"Rhos excluidos: {list(selection['excluded_rhos'])}", '',
        'rho | estado | máximo cambio uso/ventas (pp) | máximo cambio beneficio/valor (%)']
    for row in overview:
        probability, earnings = row['max_probability_change_pp'], row['max_earnings_relative_change_pct']
        lines.append(f"{row['rho']:g} | {row['status']} | "
                     +(f'{probability:.6g} | {earnings:.6g}' if probability is not None else 'sin comparación certificada | sin comparación certificada'))
    for rho, reasons in selection['excluded_rhos'].items():
        lines.extend(['', f'rho={rho}: '+json.dumps(reasons, ensure_ascii=False)])
        diagnostic = statuses[rho].get('grid_sensitivity_diagnostic')
        if diagnostic:
            lines.append('Diagnóstico adicional; no constituye una solución competitiva certificada: '
                         +json.dumps(diagnostic, ensure_ascii=False))
    if args.check_only:
        print('\n'.join(lines))
        return
    if not selection['included_rhos']:
        raise SystemExit('No rho passes the complete, verified grid comparison; no figures published')
    # Inputs must still match the inspected snapshot before publishing outputs.
    for source, expected in inputs.items():
        if digest(ROOT/source) != expected:
            raise ValueError(f'Source changed while preparing report; retry: {source}')
    data, figures = HOME/'data'/args.name, HOME/'figures'/args.name
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    accepted = set(selection['included_rhos'])
    write_csv(data/'summary_all_grids.csv', rows)
    write_csv(data/'summary.csv', [row for row in rows if row['n'] == args.grid and row['rho'] in accepted])
    write_csv(data/'policies.csv', [row for row in policies if row['n'] == args.grid and row['rho'] in accepted])
    write_csv(data/'grid_stability.csv', comparisons)
    write_csv(data/'overview.csv', overview)
    paths = make_figures(complete_rows, policies, figures, p0=.5, n=args.grid) if accepted else []
    write_json(data/'grid_sensitivity_selection.json', selection)
    write_json(figures/'grid_sensitivity_selection.json', selection)
    report = dict(created_utc=now(), status='complete' if not incomplete else 'partial_with_explicit_exclusions',
        p0=.5, common_grid=args.grid, comparison_grid=args.coarse_grid,
        scientific_configuration=configuration, requested_rhos=sorted(args.rhos),
        source_input_sha256=inputs, per_rho_provenance=statuses,
        grid_sensitivity_selection=selection, stability_comparisons=comparisons,
        policy_source='Unmodified stored grid policies; lines only connect reputation nodes',
        quadrature_scope=('Same demand formula and independent audits; historical n51 retains its original order; '
                          'new n101/n151 pairs must use the same order'),
        scientific_scope='Observed discretization stability; no continuous convergence or uniqueness proof',
        reporting_source_sha256={str(Path(__file__).relative_to(ROOT)): digest(__file__),
            str((HOME/'code/reporting.py').relative_to(ROOT)): digest(HOME/'code/reporting.py')},
        figure_files={path.name: digest(path) for path in paths})
    write_json(data/'report_manifest.json', report)
    write_json(figures/'figure_manifest.json', report)
    (data/'RESULTADOS.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
