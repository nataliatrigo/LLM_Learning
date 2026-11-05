"""Publish the later p0=.9 request without rewriting the five-p0 comparison.

This module only reads certified checkpoints, checks saved sensitivity pairs,
and renders the established four comparisons on their original n101 grids.
The dedicated independent review is mandatory before figures are published.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from oblivious.oe_p0.code.run_experiment import now, write_csv, write_json
from oblivious.oe_distribution_fixed_p0.code.run_experiment import HOME, ROOT
from oblivious.oe_distribution_fixed_p0.code.conditional_comparison_report import (
    MODELS, PROBABILITY_METRICS, digest, legend, render_figures, save, style,
)
from oblivious.oe_distribution_fixed_p0.code.multi_p0_report import (
    PROBABILITY_ROUNDOFF_TOLERANCE, read_existing_half, read_new_campaign,
    rendering_rows, select_stable, validate_display_records, verify_inputs,
)
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


P0 = .9
COMMON_RHOS = [.6, .7, .8]
RHOS = [*COMMON_RHOS, .9]
SWEEP_CASE = 'p0_0p9'


def read_required_audit(data, rhos, required_inputs):
    """Reject missing, stale, or incomplete independent checkpoint reviews."""
    path = Path(data)/'independent_review.json'
    if not path.is_file():
        raise ValueError('The dedicated p0=.9 independent review is required before publication')
    audit = json.loads(path.read_text())
    if (audit.get('verified') is not True or audit.get('p0s') != [P0]
            or not set(rhos) <= set(audit.get('rhos', []))
            or audit.get('publication_requirement_base_checkpoints_complete') is not True):
        raise ValueError('Independent review is unverified or does not cover the requested p0=.9 base grids/rhos')
    inputs = dict(audit.get('audited_input_sha256', {}))
    if not inputs or any(inputs.get(source) != expected
                         for source, expected in required_inputs.items()):
        raise ValueError('Independent review does not cover every current checkpoint/source/reference input')
    for path_field, hash_field in (('audit_source', 'audit_source_sha256'),
                                   ('audit_entrypoint', 'audit_entrypoint_sha256')):
        source, expected = audit.get(path_field), audit.get(hash_field)
        if not source or not expected or not (ROOT/source).resolve().is_relative_to(ROOT.resolve()):
            raise ValueError('Independent review must identify its repository source and entrypoint')
        if source in inputs and inputs[source] != expected:
            raise ValueError('Independent review source/entrypoint fingerprint conflicts')
        inputs[source] = expected
    verify_inputs(inputs)
    review_hash = digest(path)
    inputs[str(path.relative_to(ROOT))] = review_hash
    summary = {key: audit.get(key) for key in (
        'verified', 'p0s', 'rhos', 'audit_source', 'audit_source_sha256',
        'audit_entrypoint', 'audit_entrypoint_sha256', 'audited_bundle_count',
        'audited_file_count', 'audited_input_count', 'residual_maxima', 'scope',
        'limitations', 'publication_requirement_base_checkpoints_complete')}
    summary.update(path=str(path.relative_to(ROOT)), sha256=review_hash)
    return summary, inputs


def read_prior_comparison():
    """Validate the published comparison and fingerprint its preserved artifacts."""
    data = HOME/'data/p0_sweep'
    path = data/'comparison_report_manifest.json'
    prior = json.loads(path.read_text())
    if (prior.get('requested_p0s') != [.1, .3, .5, .7, .8]
            or prior.get('common_retained_rhos') != COMMON_RHOS
            or prior.get('displayed_n') != 101):
        raise ValueError('The prior five-p0 comparison differs from the established comparison scope')
    inputs = {}
    for field in ('source_input_sha256', 'reporting_source_sha256',
                  'figure_files', 'generated_data_sha256'):
        declared = prior.get(field, {})
        if not declared:
            raise ValueError('The prior comparison has incomplete artifact provenance: '+field)
        verify_inputs(declared)
        inputs.update(declared)
    # Only the five declared publications belong to the prior comparison.
    # The added .9 folder now shares the sweep root and must not protect its
    # own previous output while regenerating it.
    prior_figures = {ROOT/source for source in prior['figure_files']}
    prior_sidecars = {figure.parent/'figure_manifest.json' for figure in prior_figures}
    paths = sorted(set(data.glob('comparison*')) | {data/'COMPARACION.txt'}
                   | prior_figures | prior_sidecars)
    protected = {str(file.relative_to(ROOT)): digest(file) for file in paths if file.is_file()}
    inputs.update(protected)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest(path),
                p0s=prior['requested_p0s'], common_rhos=prior['common_retained_rhos'],
                displayed_n=101, modification='Existing data, report and figures are preserved byte-for-byte'), inputs, protected


def exclusion_lines(selection):
    """Produce explicit metric/missing-data reasons without changing the filter."""
    lines = []
    for rho, reasons in selection['excluded_rhos'].items():
        for reason in reasons:
            label = f"rho={rho}, {reason['model']}"
            if 'metric' in reason:
                units = 'puntos porcentuales' if reason['metric'] in PROBABILITY_METRICS else '% relativo'
                lines.append(f"{label}: {reason['metric']} cambia {100*reason['change']:.6g} {units}; límite 5 {units}.")
            else:
                lines.append(f"{label}: {reason['reason']}.")
    return lines


def outside_axis_limits(rows):
    """Show near-certain outside choices on an explicitly labeled percentage zoom."""
    values = 100*np.asarray([row['outside_rate'] for row in rows])
    if len(values) and np.min(values) >= 98:
        lower = float(np.min(values))
        return max(0., lower-max(.1*(100-lower), .001)), 100.
    return None


def render_outside_zoom(rows, directory, rhos, validation_note=None):
    """Replace only the new extension's outside panel when a zoom is useful."""
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
            values = [100*lookup[rho, model]['outside_rate'] for rho in rhos]
            ax.plot(positions, values, color=color, ls=linestyle, marker=marker,
                    ms=6 if model == 'conditional_competition' else 5, lw=1.7,
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
                     f'Fixed outside benchmark p₀ = {P0:g} | n = 101 | verified grid solutions'
                     +note+ ('\n'+validation_note if validation_note else ''), fontsize=12)
        save(fig, directory, '04_outside_option_comparison')
    metadata['vertical_axis_limits_percent'] = list(limits)
    return metadata


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rhos', type=float, nargs='+', default=RHOS,
                        choices=RHOS, help='Rho cases for the p0=.9 extension; the prior five-p0 scope stays unchanged')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if len(set(args.rhos)) != len(args.rhos):
        parser.error('Rho cases must be unique')
    args.rhos = sorted(args.rhos)
    return args


def main():
    args = arguments()
    data = HOME/'data/p0_sweep'/SWEEP_CASE
    figures = HOME/'figures/p0_sweep'/SWEEP_CASE
    previous, inputs, protected = read_prior_comparison()
    _, _, baseline_inputs, _, baseline = read_existing_half(args.rhos)
    inputs.update(baseline_inputs)
    rows, policies, target_inputs, statuses, economics = read_new_campaign(
        HOME/'data/p0_sweep', P0, args.rhos, baseline=baseline)
    inputs.update(target_inputs)
    selection = select_stable(rows, args.rhos)
    retained = selection['included_rhos']
    independent, audit_inputs = read_required_audit(data, args.rhos, target_inputs)
    inputs.update(audit_inputs)
    overview = dict(status='complete' if retained == args.rhos else 'partial' if retained else 'no_stable_rhos',
                    requested_p0s=[P0], requested_rhos=args.rhos, displayed_n=101,
                    retained_rhos=retained, selection=selection,
                    per_rho_provenance=statuses, independent_audit=independent,
                    prior_five_p0_comparison=previous)
    verify_inputs(inputs)
    if args.check_only:
        print(json.dumps(overview, indent=2))
        return
    if not retained:
        raise SystemExit('No certified, grid-stable rho cases; no figures published.\n'+json.dumps(overview, indent=2))
    fine, fine_policies = validate_display_records(rows, policies, P0, retained)
    finer = any(item['rho'] in retained and item['explicit_finer_check']
                for item in selection['selected_validation_pairs'])
    note = 'n = 101 checked against n = 151 where needed; see validation table.' if finer else None
    if retained != args.rhos:
        note = (note+'\n' if note else '')+'Certified, grid-stable ρ cases; excluded cases are listed in the report.'
    paths, coincidence = render_figures(rendering_rows(rows), policies, figures, retained,
                                       p0=P0, n=101, validation_note=note)
    outside_display = render_outside_zoom(fine, figures, retained, validation_note=note)
    data.mkdir(parents=True, exist_ok=True)
    exports = {
        'comparison_summary_all_grids.csv': rows,
        'comparison_summary.csv': fine,
        'comparison_policies.csv': fine_policies,
        'comparison_grid_stability.csv': [dict(p0=P0, **row) for row in selection['comparisons']],
    }
    for name, records in exports.items():
        write_csv(data/name, records)
    write_json(data/'comparison_grid_sensitivity_selection.json', selection)
    write_json(data/'comparison_coincidence.json', coincidence)
    lines = [
        'Extensión solicitada posteriormente: p0=0.9 se añade a la comparación; p0=0.8 se conserva.',
        'Modelos: monopolio, OE marginal y OE condicional; todas las curvas usan sus soluciones originales de n101.',
        'Rhos solicitados para la extensión p0=0.9: '+', '.join(f'{rho:g}' for rho in args.rhos)+'.',
        'Rhos certificados y estables incluidos: '+', '.join(f'{rho:g}' for rho in retained)+'.',
        'Fuera de esta extensión por sensibilidad o datos incompletos: '+(', '.join(selection['excluded_rhos']) or 'ninguno')+'.',
        'Filtro sin cambios: máximo 5 puntos porcentuales en probabilidades y 5% relativo simétrico en beneficio/valor.',
        'Se conserva el contraste 51→101; si falla y hay un suplemento certificado n151 se usa 101→151.',
        'Las grillas n151 validan; no reemplazan las curvas n101.',
        'Producto 2 se mide como porcentaje de las ventas propias.',
        'Outside significa que ningún vendedor vende: 1-sale_rate en monopolio y 1-2*sale_rate en competencia.',
        'Los beneficios descontados son valores del juego real desde reputación inicial 0.5.',
        'Los datos, informes y figuras de los cinco p0 anteriores no se modifican.',
        'La auditoría independiente dedicada verifica fuentes, referencias, checkpoints e identidades numéricas.',
        *exclusion_lines(selection),
    ]
    if outside_display['zoom_applied']:
        lower, upper = outside_display['vertical_axis_limits_percent']
        lines.append(f'El plot de outside amplía explícitamente el eje vertical: {lower:.6g}% a {upper:g}%; las probabilidades no cambian.')
    lines += [f"Outside: rho={row['rho']:g}, {row['model']} = {100*row['outside_rate']:.8f}% de los períodos."
              for row in fine]
    (data/'COMPARACION.txt').write_text('\n'.join(lines)+'\n')
    generated_names = [*exports, 'comparison_grid_sensitivity_selection.json',
                       'comparison_coincidence.json', 'COMPARACION.txt']
    source_names = [Path(__file__), HOME/'code/multi_p0_report.py',
                    HOME/'code/conditional_comparison_report.py', HOME/'code/common_grid_report.py']
    report = dict(created_utc=now(), **overview,
        p0_selection=dict(previous_values=[.1, .3, .5, .7, .8], added_value=P0,
            resulting_values=[.1, .3, .5, .7, .8, .9],
            user_adjustment='The user later requested p0=0.9 in addition to the preserved p0=0.8 results'),
        models=[dict(model=model, label=label) for model, label, *_ in MODELS],
        shared_economic_configuration=economics, source_input_sha256=inputs,
        reporting_source_sha256={str(path.relative_to(ROOT)): digest(path) for path in source_names},
        figure_files={str(path.relative_to(ROOT)): digest(path) for path in paths},
        generated_data_sha256={str((data/name).relative_to(ROOT)): digest(data/name) for name in generated_names},
        preserved_prior_artifact_sha256=protected,
        outside_option_display=outside_display,
        value_source='Actual game values; not perceived oblivious values',
        outside_option_measure='Whole market: monopoly 1-sale_rate; symmetric competition 1-2*sale_rate',
        probability_endpoint_roundoff=dict(tolerance=PROBABILITY_ROUNDOFF_TOLERANCE,
            exported_values='Raw saved probabilities are retained in exports and sensitivity checks',
            display='Only copied outside_rate and sale_rate are clipped to [0,1] for the original renderer'),
        display_scope='Every plotted/exported curve uses its original n101 solution; n151 only validates',
        scientific_scope='Observed finite-grid stability; not a proof of continuous convergence or uniqueness')
    verify_inputs(inputs)
    verify_inputs(protected)
    write_json(data/'comparison_report_manifest.json', report)
    write_json(figures/'figure_manifest.json', report)
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
