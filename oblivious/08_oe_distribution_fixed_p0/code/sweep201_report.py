"""Compare the six-p0 campaign on original n201 policies, validated at n251.

This reporting entrypoint never solves a model. It reads only the isolated
p0_sweep_201 campaign and requires a dedicated independent review before
publishing the four established figures for each p0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np

from oblivious.oe_p0.code.run_experiment import case_tag, now, write_csv, write_json
from oblivious.oe_distribution_fixed_p0.code.run_experiment import HOME, ROOT, check_bundle
from oblivious.oe_distribution_fixed_p0.code.conditional_comparison_report import (
    EARNINGS_METRICS, MODELS, PROBABILITY_METRICS, digest, legend, render_figures, save, style,
)
from oblivious.oe_distribution_fixed_p0.code.multi_p0_report import (
    PROBABILITY_ROUNDOFF_TOLERANCE, common_rhos, p0_name, rendering_rows,
    validate_display_records, validate_saved_case, verify_inputs,
)
from oblivious.oe_distribution_fixed_p0.code.multi_p0_run import source_fingerprints


P0S = [.1, .3, .5, .7, .8, .9]
RHOS = [.6, .7, .8, .9]
DISPLAY_GRID = 201
VALIDATION_GRID = 251
GRIDS = [DISPLAY_GRID, VALIDATION_GRID]
CAMPAIGN = 'p0_sweep_201'
SENSITIVITY_LIMIT = .05


def read_campaign(data, p0, rhos, *, baseline=None):
    """Read both required grids without substituting old or finer policies."""
    rows, policies, inputs, statuses = [], [], {}, {}
    expected_sources = source_fingerprints()
    for rho in rhos:
        directory = Path(data)/p0_name(p0)/('rho_'+str(float(rho)).replace('.', 'p'))
        path = directory/'manifest.json'
        required = {case_tag(rho, n)+'_'+closure for n in GRIDS
                    for closure in ('marginal', 'conditional')}
        state = dict(status='missing', source_directory=str(directory.relative_to(ROOT)),
                     available_grids={}, missing_checkpoints=sorted(required))
        statuses[str(rho)] = state
        if not path.is_file():
            continue
        manifest = json.loads(path.read_text())
        inputs[str(path.relative_to(ROOT))] = digest(path)
        declared = manifest['configuration']
        if declared['p0'] != p0 or declared['rho'] != rho:
            raise ValueError(f'Manifest p0/rho mismatch: {directory}')
        if sorted(manifest.get('requested_grids', [])) != GRIDS:
            raise ValueError(f'Manifest does not request the n201/n251 campaign: {directory}')
        sources = manifest.get('numerical_sources', {})
        if sources != expected_sources:
            raise ValueError(f'Immutable numerical source set differs: {directory}')
        references = manifest.get('reference_input_sha256', {})
        verify_inputs(sources)
        verify_inputs(references)
        inputs.update(sources)
        inputs.update(references)
        completed = manifest.get('completed', [])
        if len(completed) != len(set(completed)) or set(completed)-required:
            raise ValueError(f'Duplicate or unexpected completed checkpoint: {directory}')
        state.update(source_status=manifest['status'], current_grid=manifest.get('current_grid'),
                     current_model=manifest.get('current_model'),
                     failures=manifest.get('failures', {}), numerical_sources=sources,
                     missing_checkpoints=sorted(required-set(completed)))
        if (manifest['status'] == 'running' or manifest.get('current_grid') is not None
                or manifest.get('current_model') is not None):
            state['status'] = 'running_not_published'
            continue
        for closure in ('marginal', 'conditional'):
            state['available_grids'][closure] = []
            for n in GRIDS:
                tag = case_tag(rho, n)+'_'+closure
                if tag not in completed:
                    continue
                case = check_bundle(directory, tag, manifest)
                for source, expected in case.get('reference_input_sha256', {}).items():
                    if references.get(source) != expected:
                        raise ValueError(f'Case/reference provenance differs: {directory}, {tag}')
                found_rows, found_policies, economics = validate_saved_case(
                    case, p0, rho, n, closure, baseline)
                baseline = economics if baseline is None else baseline
                configuration = case['configuration']
                if (configuration['order'] != declared['order']
                        or configuration['tolerance'] != declared['tol']
                        or configuration['dp_tolerance'] != declared['dp_tol']
                        or {key: value for key, value in configuration['parameters'].items()
                            if key != 'rho'} != declared['parameters']):
                    raise ValueError(f'Manifest/checkpoint configuration mismatch: {directory}, {tag}')
                for suffix in ('.json', '.npz'):
                    artifact = directory/'arrays'/(tag+suffix)
                    inputs[str(artifact.relative_to(ROOT))] = manifest['checkpoint_sha256'][artifact.name]
                rows.extend(found_rows)
                policies.extend(found_policies)
                state['available_grids'][closure].append(n)
        state['status'] = 'complete' if required.issubset(completed) else 'incomplete'
    return rows, policies, inputs, statuses, baseline


def read_sweep_execution(data, *, require_complete):
    """Fingerprint the dedicated controller and worker that produced the data."""
    path = Path(data)/'sweep_manifest.json'
    if not path.is_file():
        if require_complete:
            raise ValueError('Complete n201/n251 sweep manifest is required before publication')
        return None, {}
    sweep = json.loads(path.read_text())
    if (sweep.get('configuration') != dict(p0s=P0S, rhos=RHOS, grids=GRIDS,
                                          order=256, tol=1e-7, dp_tol=1e-10)
            or sweep.get('numerical_sources') != source_fingerprints()):
        raise ValueError('Sweep manifest differs from the requested n201/n251 comparison')
    complete = (sweep.get('status') == 'complete_certified'
                and sweep.get('completed_p0_rho_pairs') == 24
                and sweep.get('certified_checkpoint_bundles') == 96
                and sweep.get('certified_model_grid_solutions') == 144)
    if require_complete and not complete:
        raise ValueError('Sweep manifest is not complete and certified for all 24 cases')
    sources = {str((HOME/'code/sweep201_batch.py').relative_to(ROOT)): sweep['scheduler_sha256'],
               str((HOME/'code/sweep201_run.py').relative_to(ROOT)): sweep['worker_sha256']}
    history = sweep.get('execution_entrypoint_history', [])
    current = sweep.get('execution_entrypoint')
    if not isinstance(history, list) or (history and current != history[-1]):
        raise ValueError('Current execution entrypoint differs from the final recorded phase')
    phases = history if history else ([] if current is None else [current])
    for phase in phases:
        if (not isinstance(phase, dict)
                or phase.get('immutable_scheduler_sha256') != sweep['scheduler_sha256']
                or phase.get('immutable_worker_sha256') != sweep['worker_sha256']):
            raise ValueError('Execution phase immutable worker/scheduler fingerprints differ')
        declared_sources = phase.get('source_input_sha256')
        if not isinstance(declared_sources, dict) or not declared_sources:
            raise ValueError('Execution phase source fingerprints are missing')
        for source, expected in declared_sources.items():
            if (not (ROOT/source).resolve().is_relative_to(ROOT.resolve())
                    or source in sources and sources[source] != expected):
                raise ValueError('Execution phase source fingerprint conflicts: '+source)
        verify_inputs(declared_sources)
        sources.update(declared_sources)
    verify_inputs(sources)
    sources[str(path.relative_to(ROOT))] = digest(path)
    summary = {key: sweep.get(key) for key in (
        'status', 'configuration', 'completed_p0_rho_pairs', 'certified_checkpoint_bundles',
        'certified_model_grid_solutions', 'scheduler_sha256', 'worker_sha256',
        'execution_entrypoint', 'execution_entrypoint_history')}
    summary.update(path=str(path.relative_to(ROOT)), sha256=digest(path),
                   verified_execution_phase_count=len(phases))
    return summary, sources


def select_stable(rows, rhos):
    """Keep displayed n201 independent of the forward n201→n251 check."""
    lookup = {}
    for row in rows:
        key = (row['rho'], row['model'], row['n'])
        if key in lookup:
            raise ValueError(f'Duplicate saved outcome: {key}')
        lookup[key] = row
    comparisons, included, excluded = [], [], {}
    for rho in rhos:
        reasons = []
        for model, *_ in MODELS:
            pair = [lookup.get((rho, model, n)) for n in GRIDS]
            if any(row is None or not row.get('verified', False) for row in pair):
                reasons.append(dict(model=model, coarse_n=DISPLAY_GRID, fine_n=VALIDATION_GRID,
                                    reason='Missing verified n201/n251 comparison'))
                continue
            probability, relative, failures = {}, {}, []
            for metric in PROBABILITY_METRICS+EARNINGS_METRICS:
                first, second = (float(row[metric]) for row in pair)
                if not np.isfinite([first, second]).all():
                    failures.append(dict(model=model, metric=metric, reason='Nonfinite outcome'))
                    continue
                difference = abs(second-first)
                if metric in PROBABILITY_METRICS:
                    change = probability[metric] = difference
                else:
                    denominator = max(abs(first), abs(second))
                    change = relative[metric] = difference/denominator if denominator else 0.
                if change > SENSITIVITY_LIMIT+1e-12:
                    failures.append(dict(model=model, metric=metric, change=change,
                                         limit=SENSITIVITY_LIMIT))
            passed = (not failures and len(probability) == len(PROBABILITY_METRICS)
                      and len(relative) == len(EARNINGS_METRICS))
            comparisons.append(dict(rho=rho, model=model, coarse_n=DISPLAY_GRID,
                fine_n=VALIDATION_GRID, displayed_n=DISPLAY_GRID, used_for_selection=True,
                comparison_role='forward_validation', passed=passed,
                max_probability_change_pp=100*max(probability.values(), default=np.nan),
                max_earnings_relative_change_pct=100*max(relative.values(), default=np.nan),
                **{metric+'_absolute_change': value for metric, value in probability.items()},
                **{metric+'_relative_change': value for metric, value in relative.items()}))
            reasons.extend(failures)
        if reasons:
            excluded[str(rho)] = reasons
        else:
            included.append(rho)
    return dict(included_rhos=included, excluded_rhos=excluded, requested_rhos=list(rhos),
        displayed_grid=DISPLAY_GRID, default_validation_pair=GRIDS,
        probability_change_limit=SENSITIVITY_LIMIT, relative_earnings_change_limit=SENSITIVITY_LIMIT,
        probability_metrics=list(PROBABILITY_METRICS), relative_metrics=list(EARNINGS_METRICS),
        relative_definition='abs(fine-coarse)/max(abs(fine),abs(coarse)); zero when both are zero',
        comparisons=comparisons,
        interpretation='Finite-grid stability of saved certified n201/n251 results; n251 never replaces n201; no continuous-state convergence or uniqueness proof')


def read_required_audit(data, required_inputs):
    """Require exact new-campaign coverage, current fingerprints and both grids."""
    path = Path(data)/'independent_review.json'
    if not path.is_file():
        raise ValueError('The dedicated n201/n251 independent review is required before publication')
    audit = json.loads(path.read_text())
    expected = {(p0, rho, n, closure) for p0 in P0S for rho in RHOS for n in GRIDS
                for closure in ('marginal', 'conditional')}
    records = audit.get('cases', [])
    keys = [(item.get('p0'), item.get('rho'), item.get('n'), item.get('closure'))
            for item in records]
    if (audit.get('verified') is not True or audit.get('p0s') != P0S
            or audit.get('rhos') != RHOS or audit.get('grids') != GRIDS
            or audit.get('publication_grid') != DISPLAY_GRID
            or audit.get('validation_grid') != VALIDATION_GRID
            or audit.get('publication_requirement_base_checkpoints_complete') is not True
            or audit.get('audited_bundle_count') != len(expected)
            or len(keys) != len(expected) or set(keys) != expected
            or any(item.get('verified') is not True for item in records)):
        raise ValueError('Independent review does not certify all 24 p0/rho cases on n201 and n251')
    inputs = dict(audit.get('audited_input_sha256', {}))
    if not inputs or any(inputs.get(source) != expected_hash
                         for source, expected_hash in required_inputs.items()):
        raise ValueError('Independent review does not cover every current checkpoint/source/reference input')
    for path_field, hash_field in (('audit_source', 'audit_source_sha256'),
                                   ('audit_entrypoint', 'audit_entrypoint_sha256')):
        source, expected_hash = audit.get(path_field), audit.get(hash_field)
        if (not source or not expected_hash
                or not (ROOT/source).resolve().is_relative_to(ROOT.resolve())):
            raise ValueError('Independent review must identify its repository source and entrypoint')
        if source in inputs and inputs[source] != expected_hash:
            raise ValueError('Independent review source/entrypoint fingerprint conflicts')
        inputs[source] = expected_hash
    verify_inputs(inputs)
    review_hash = digest(path)
    inputs[str(path.relative_to(ROOT))] = review_hash
    summary = {key: audit.get(key) for key in (
        'verified', 'p0s', 'rhos', 'grids', 'publication_grid', 'validation_grid',
        'audit_source', 'audit_source_sha256', 'audit_entrypoint', 'audit_entrypoint_sha256',
        'audited_bundle_count', 'audited_file_count', 'audited_input_count',
        'residual_maxima', 'scope', 'limitations', 'publication_requirement_base_checkpoints_complete')}
    summary.update(path=str(path.relative_to(ROOT)), sha256=review_hash)
    return summary, inputs


def preserved_historical_artifacts():
    """Fingerprint prior publications; this isolated campaign never writes them."""
    paths = set()
    historical = HOME/'data/p0_sweep'
    for directory in (historical, historical/'p0_0p9'):
        paths.update(path for path in directory.glob('comparison*') if path.is_file())
        if (directory/'COMPARACION.txt').is_file():
            paths.add(directory/'COMPARACION.txt')
    figures = HOME/'figures/p0_sweep'
    if figures.exists():
        paths.update(path for path in figures.rglob('*') if path.is_file()
                     and path.suffix in ('.png', '.pdf', '.json'))
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def outside_axis_limits(rows):
    values = 100*np.asarray([row['outside_rate'] for row in rows])
    if len(values) and np.min(values) >= 98:
        lower = float(np.min(values))
        return max(0., lower-max(.1*(100-lower), .001)), 100.
    return None


def render_outside_zoom(rows, directory, rhos, p0):
    """Label a near-100% zoom explicitly; exported probabilities stay raw."""
    rows = rendering_rows(rows)
    limits = outside_axis_limits(rows)
    metadata = dict(metric='outside_rate', scale='percentage of calendar periods',
        criterion='Zoom only when every displayed whole-market outside rate is at least 98%',
        zoom_applied=limits is not None, saved_csv_values='Raw probabilities remain unchanged')
    if limits is None:
        return metadata
    lookup = {(row['rho'], row['model']): row for row in rows}
    positions = np.arange(len(rhos))
    with plt.rc_context({'font.size': 9, 'axes.titlesize': 10, 'legend.fontsize': 9}):
        fig, ax = plt.subplots(figsize=(7.5, 4.8), layout='constrained')
        for model, label, color, linestyle, marker in MODELS:
            ax.plot(positions, [100*lookup[rho, model]['outside_rate'] for rho in rhos],
                    color=color, ls=linestyle, marker=marker, lw=1.7,
                    ms=6 if model == 'conditional_competition' else 5,
                    markerfacecolor=color if model == 'monopoly' else 'none', label=label)
        ax.set(ylabel='Outside option (% of periods)',
               title='Buyer chooses outside: no seller makes a sale',
               xlabel='Persistence ρ (equally spaced cases)', ylim=limits)
        ax.set_xticks(positions, [f'{rho:g}' for rho in rhos])
        if len(rhos) > 1:
            ax.set_xlim(-.2, len(rhos)-.8)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.ticklabel_format(axis='y', style='plain', useOffset=False)
        style(ax)
        legend(fig, [ax])
        overlap = all(abs(lookup[rho, 'competition']['outside_rate']
                          - lookup[rho, 'conditional_competition']['outside_rate']) <= 5e-12
                      for rho in rhos)
        note = '\nMarginal and conditional OE overlap at plotted precision.' if overlap else ''
        fig.suptitle('Outside option choice (zoomed vertical axis)\n'
                     f'Fixed outside benchmark p₀ = {p0:g} | n = {DISPLAY_GRID} | verified grid solutions'
                     +note+'\nn = 201 checked against n = 251.', fontsize=12)
        save(fig, directory, '04_outside_option_comparison')
    metadata['vertical_axis_limits_percent'] = list(limits)
    return metadata


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true',
                        help='Inspect completion and sensitivity without publishing figures')
    return parser.parse_args()


def main():
    args = arguments()
    data = HOME/'data'/CAMPAIGN
    figures = HOME/'figures'/CAMPAIGN
    all_rows, all_policies, inputs, statuses, selections = [], [], {}, {}, {}
    baseline = None
    for p0 in P0S:
        rows, policies, more, state, economics = read_campaign(data, p0, RHOS, baseline=baseline)
        baseline = economics if baseline is None else baseline
        all_rows.extend(rows)
        all_policies.extend(policies)
        inputs.update(more)
        statuses[str(p0)] = state
        selections[str(p0)] = select_stable(rows, RHOS)
    complete = all(state['status'] == 'complete'
                   for states in statuses.values() for state in states.values())
    retained = common_rhos(selections, RHOS)
    overview = dict(status=('incomplete' if not complete else 'complete' if retained == RHOS
                            else 'complete_with_sensitivity_exclusions' if retained
                            else 'no_common_stable_rhos'),
        requested_p0s=P0S, requested_rhos=RHOS, required_grids=GRIDS,
        displayed_n=DISPLAY_GRID, validation_n=VALIDATION_GRID,
        all_required_cases_certified=complete, required_case_count=len(P0S)*len(RHOS),
        common_retained_rhos=retained, selections=selections, per_p0_provenance=statuses)
    overview['sweep_execution'], execution_inputs = read_sweep_execution(
        data, require_complete=not args.check_only)
    inputs.update(execution_inputs)
    if args.check_only:
        if complete and (data/'independent_review.json').exists():
            overview['independent_audit'], _ = read_required_audit(data, inputs)
        else:
            overview['independent_audit'] = None
        print(json.dumps(overview, indent=2))
        return
    if not complete:
        raise SystemExit('The n201/n251 campaign is incomplete; no figures published.\n'+json.dumps(overview, indent=2))
    independent, audit_inputs = read_required_audit(data, inputs)
    inputs.update(audit_inputs)
    overview['independent_audit'] = independent
    if not retained:
        raise SystemExit('No common certified, grid-stable rho cases; no figures published.\n'+json.dumps(overview, indent=2))
    preserved = preserved_historical_artifacts()
    verify_inputs(inputs)
    paths, coincidences, displayed_rows, displayed_policies, outside_display = [], {}, [], [], {}
    for p0 in P0S:
        rows = [row for row in all_rows if row['p0'] == p0]
        policies = [row for row in all_policies if row['p0'] == p0]
        displayed, displayed_policy = validate_display_records(
            rows, policies, p0, retained, grid=DISPLAY_GRID)
        displayed_rows.extend(displayed)
        displayed_policies.extend(displayed_policy)
        note = 'n = 201 checked against n = 251.'
        if retained != RHOS:
            note += '\nCommon certified, grid-stable ρ cases across all displayed p₀.'
        directory = figures/p0_name(p0)
        generated, coincidence = render_figures(rendering_rows(rows), policies, directory,
            retained, p0=p0, n=DISPLAY_GRID, validation_note=note)
        paths.extend(generated)
        coincidences[str(p0)] = coincidence
        outside_display[str(p0)] = render_outside_zoom(displayed, directory, retained, p0)
    data.mkdir(parents=True, exist_ok=True)
    exports = {
        'comparison_summary_all_grids.csv': all_rows,
        'comparison_summary.csv': displayed_rows,
        'comparison_policies.csv': displayed_policies,
        'comparison_grid_stability.csv': [dict(p0=float(p0), **row)
            for p0, selection in selections.items() for row in selection['comparisons']],
    }
    for name, records in exports.items():
        write_csv(data/name, records)
    write_json(data/'comparison_grid_sensitivity_selection.json', selections)
    write_json(data/'comparison_coincidence.json', coincidences)
    lines = [
        'Nueva comparación aislada en p0_sweep_201; las publicaciones anteriores se conservan.',
        'p0: '+', '.join(f'{p0:g}' for p0 in P0S)+'.',
        'Rhos solicitados: '+', '.join(f'{rho:g}' for rho in RHOS)+'.',
        'Modelos: monopolio, OE marginal y OE condicional; 24 casos y 144 resultados modelo/grilla certificados.',
        'Todas las curvas y políticas publicadas usan sus 201 nodos originales; 251 solamente valida.',
        'Rhos comunes certificados y estables: '+', '.join(f'{rho:g}' for rho in retained)+'.',
        'Fuera del conjunto común: '+(', '.join(f'{rho:g}' for rho in RHOS if rho not in retained) or 'ninguno')+'.',
        'Filtro 201→251: máximo 5 puntos porcentuales en probabilidades y 5% relativo simétrico en beneficio/valor.',
        'Cambio relativo: abs(fino-grueso)/max(abs(fino),abs(grueso)); cero cuando ambos son cero.',
        'Cada rho publicado pasa el filtro para los seis p0 y los tres modelos.',
        'Producto 2 es el porcentaje de las ventas propias; outside significa que ningún vendedor vende.',
        'Los beneficios descontados son valores del juego real desde reputación inicial 0.5.',
        'La auditoría independiente dedicada verifica todas las fuentes, referencias, checkpoints e identidades numéricas.',
        'Es evidencia de estabilidad en grillas finitas; no prueba convergencia continua ni unicidad.',
    ]
    for p0, selection in selections.items():
        for rho, reasons in selection['excluded_rhos'].items():
            for reason in reasons:
                if 'metric' in reason and 'change' in reason:
                    units = 'pp' if reason['metric'] in PROBABILITY_METRICS else '% relativo'
                    lines.append(f"p0={p0}, rho={rho}, {reason['model']}: {reason['metric']} cambia {100*reason['change']:.6g} {units}; límite 5 {units}.")
                else:
                    lines.append(f"p0={p0}, rho={rho}, {reason['model']}: {reason['reason']}.")
    for p0, display in outside_display.items():
        if display['zoom_applied']:
            lower, upper = display['vertical_axis_limits_percent']
            lines.append(f'Outside p0={p0}: eje vertical ampliado explícitamente de {lower:.6g}% a {upper:g}%; las probabilidades no cambian.')
    (data/'COMPARACION.txt').write_text('\n'.join(lines)+'\n')
    generated_names = [*exports, 'comparison_grid_sensitivity_selection.json',
                       'comparison_coincidence.json', 'COMPARACION.txt']
    source_names = [Path(__file__), HOME/'code/multi_p0_report.py',
                    HOME/'code/conditional_comparison_report.py', HOME/'code/common_grid_report.py']
    report = dict(created_utc=now(), **overview,
        models=[dict(model=model, label=label) for model, label, *_ in MODELS],
        shared_economic_configuration=baseline, source_input_sha256=inputs,
        reporting_source_sha256={str(path.relative_to(ROOT)): digest(path) for path in source_names},
        figure_files={str(path.relative_to(ROOT)): digest(path) for path in paths},
        generated_data_sha256={str((data/name).relative_to(ROOT)): digest(data/name) for name in generated_names},
        preserved_historical_artifact_sha256=preserved, outside_option_display=outside_display,
        model_grid_outcome_count=len(all_rows), displayed_outcome_count=len(displayed_rows),
        value_source='Actual game values; not perceived oblivious values',
        outside_option_measure='Whole market: monopoly 1-sale_rate; symmetric competition 1-2*sale_rate',
        probability_endpoint_roundoff=dict(tolerance=PROBABILITY_ROUNDOFF_TOLERANCE,
            exported_values='Raw saved probabilities retained in exports and sensitivity checks',
            display='Only copied outside_rate and sale_rate are clipped to [0,1] for the original renderer'),
        display_scope='Every curve and exported policy uses its original n201 nodes; n251 only validates',
        scientific_scope='Observed finite-grid stability; not a proof of continuous convergence or uniqueness')
    verify_inputs(inputs)
    verify_inputs(preserved)
    write_json(data/'comparison_report_manifest.json', report)
    for p0 in P0S:
        write_json(figures/p0_name(p0)/'figure_manifest.json', report)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
