"""Case-specific grids and metric-specific validation for the marginal sweep.

All seller counts share a grid within a (p0,rho) comparison. Each metric uses
the same numerical threshold, separately: unsupported points are explicit gaps.
The frozen pilot reporter and its historical publications stay unchanged.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from . import report as base


P0_COLORS = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00', '#333333')
METRICS = ('product2_conditional', 'outside_rate', 'profit_calendar', 'value_initial')
PROBABILITIES = ('product2_conditional', 'outside_rate')
STABILITY_FILE, RETENTION_FILE = 'expanded_grid_stability.csv', 'metric_retention.json'


def case_key(p0, rho):
    return f'p0_{p0:g}/rho_{rho:g}'.replace('.', 'p')


def read_diagnostic_snapshot(data):
    """Read a live CSV snapshot without issuing a publication certificate."""
    manifest = json.loads((Path(data)/'manifest.json').read_text())
    rows, policies = [], []
    for name, target in (('summary.csv', rows), ('policies.csv', policies)):
        path = Path(data)/name
        if not path.exists():
            continue
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                for field in ('p0', 'rho'):
                    row[field] = float(row[field])
                for field in ('n', 'sellers'):
                    row[field] = int(row[field])
                if name == 'summary.csv':
                    row['verified'] = str(row['verified']).lower() in ('true', '1')
                    for field in (*METRICS, 'sale_rate'):
                        row[field] = float(row[field])
                else:
                    for field in ('reputation', 'product2_probability'):
                        row[field] = float(row[field])
                target.append(row)
    config = manifest['configuration']
    return manifest, rows, policies, {}, sorted(config['p0s']), sorted(config['rhos']), base.DEFAULT_MODELS


def select_cases(rows, manifest, p0s, rhos):
    """Select the finest two certified grids shared by all N in each case."""
    lookup = {}
    for row in rows:
        key = (row['p0'], row['rho'], row['n'], row['sellers'], row['closure'])
        base.require(key not in lookup, 'Duplicate saved outcome: '+str(key))
        lookup[key] = row
    planned = manifest.get('case_grids', manifest['configuration'].get('case_grids', {}))
    cases, comparisons = {}, []
    for p0 in p0s:
        for rho in rhos:
            key = case_key(p0, rho)
            requested = sorted(planned.get(key, manifest['configuration']['grids']))
            available = [{row['n'] for row in rows if row['p0'] == p0 and row['rho'] == rho
                and row['sellers'] == sellers and row['closure'] == closure and row['verified']
                and row['n'] in requested} for sellers, closure, *_ in base.DEFAULT_MODELS]
            common = sorted(set.intersection(*available))
            fine = common[-1] if common else None
            coarse = common[-2] if len(common) > 1 else None
            case = dict(p0=p0, rho=rho, planned_grids=requested, common_certified_grids=common,
                        displayed_n=fine, validation_coarse_n=coarse, metrics={})
            for metric in METRICS:
                case['metrics'][metric] = dict(passed=coarse is not None, limit=.05,
                    units='absolute probability' if metric in PROBABILITIES else 'symmetric relative',
                    max_change=None, failed_sellers=[], reason=None if coarse is not None else 'Missing two common certified grids')
            if coarse is not None:
                changes = {metric: [] for metric in METRICS}
                for sellers, closure, *_ in base.DEFAULT_MODELS:
                    earlier, later = [lookup[p0, rho, n, sellers, closure] for n in (coarse, fine)]
                    record = dict(p0=p0, rho=rho, sellers=sellers, closure=closure,
                                  coarse_n=coarse, fine_n=fine, displayed_n=fine)
                    for metric in METRICS:
                        first, second = earlier[metric], later[metric]
                        difference = abs(second-first)
                        change = difference if metric in PROBABILITIES else (
                            difference/max(abs(first), abs(second)) if first or second else 0.)
                        passed = bool(np.isfinite([first, second, change]).all() and change <= .05+1e-12)
                        record[metric+'_change'], record[metric+'_passed'] = float(change), passed
                        changes[metric].append(float(change))
                        if not passed:
                            case['metrics'][metric]['passed'] = False
                            case['metrics'][metric]['failed_sellers'].append(sellers)
                    record['all_metrics_passed'] = all(record[metric+'_passed'] for metric in METRICS)
                    comparisons.append(record)
                for metric in METRICS:
                    case['metrics'][metric]['max_change'] = max(changes[metric])
            cases[key] = case
    return cases, comparisons


def metric_retention(cases, p0s, rhos):
    return {metric: dict(
        valid_cases=[key for key, case in cases.items() if case['metrics'][metric]['passed']],
        excluded_cases={key: case['metrics'][metric] for key, case in cases.items() if not case['metrics'][metric]['passed']},
        by_p0={str(p0): [rho for rho in rhos if cases[case_key(p0, rho)]['metrics'][metric]['passed']] for p0 in p0s},
        by_rho={str(rho): [p0 for p0 in p0s if cases[case_key(p0, rho)]['metrics'][metric]['passed']] for rho in rhos}
    ) for metric in METRICS}


def facets(rhos, *, panel_height=2.9):
    columns = min(2, len(rhos))
    count = (len(rhos)+columns-1)//columns
    return base.plt.subplots(count, columns, figsize=(5.5*columns, panel_height*count+.7),
        sharex=True, sharey=True, squeeze=False, layout='constrained')


def decorate(fig, axes, title, subtitle):
    handles = {}
    for ax in np.asarray(axes).ravel():
        if not ax.get_visible():
            continue
        base.style(ax)
        artists, labels = ax.get_legend_handles_labels()
        handles.update(zip(labels, artists))
    if handles:
        fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle(title+'\n'+subtitle, fontsize=11)


def metric_values(lookup, cases, p0, rhos, sellers, closure, metric):
    """NaNs leave gaps rather than connecting unsupported rho points."""
    return [lookup[p0, rho, cases[case_key(p0, rho)]['displayed_n'], sellers, closure][metric]
        if cases[case_key(p0, rho)]['metrics'][metric]['passed'] else np.nan for rho in rhos]


def render(rows, policies, figures, p0s, rhos, cases):
    lookup = {(r['p0'], r['rho'], r['n'], r['sellers'], r['closure']): r for r in rows}
    paths, positions = [], np.arange(len(rhos))
    with base.plt.rc_context({'font.size': 9, 'axes.titlesize': 9, 'legend.fontsize': 8}):
        for p0 in p0s:
            directory = Path(figures)/('p0_'+str(float(p0)).replace('.', 'p'))
            directory.mkdir(parents=True, exist_ok=True)
            fig, axes = facets(rhos)
            for ax, rho in zip(axes.ravel(), rhos):
                grid = cases[case_key(p0, rho)]['displayed_n']
                if grid is None:
                    ax.text(.5, .5, 'No common certified grid', ha='center', va='center', transform=ax.transAxes)
                else:
                    for sellers, closure, label, color, line, marker in base.DEFAULT_MODELS:
                        selected = sorted((r for r in policies if r['p0'] == p0 and r['rho'] == rho
                            and r['n'] == grid and r['sellers'] == sellers and r['closure'] == closure),
                            key=lambda row: row['reputation'])
                        ax.plot([r['reputation'] for r in selected], [r['product2_probability'] for r in selected],
                                color=color, ls=line, lw=1.7, label=label)
                ax.set(title=f'ρ = {rho:g} | original n = {grid if grid else "missing"}',
                       xlabel='Own reputation x', ylabel='Pr(product 2 | own sale, x)', ylim=(-.03, 1.03), xlim=(0, 1))
                ax.tick_params(labelbottom=True)
            for ax in axes.ravel()[len(rhos):]:
                ax.set_visible(False)
            decorate(fig, axes, f'N=1 vs N=2 vs N=3 | fixed p₀ = {p0:g}',
                     'Certified finite-grid policies; pointwise threshold stability is not established.')
            paths.extend(base.save(fig, directory, base.STEMS[0]))
            specifications = [
                (base.STEMS[1], [('product2_conditional', 'Product 2 (% of own sales)', 'Own-sales product mix', 100)]),
                (base.STEMS[2], [('profit_calendar', 'Profit per seller / calendar period', 'Stationary profit', 1),
                    ('value_initial', 'Discounted profit per seller', 'Actual game value; initial reputations = 0.5', 1)]),
                (base.STEMS[3], [('outside_rate', 'Outside option (% of periods)', 'Whole market: no seller sells', 100)]),
            ]
            for stem, panels in specifications:
                fig, axes = base.plt.subplots(1, len(panels), figsize=(6.7*len(panels), 5.1),
                    squeeze=False, layout='constrained')
                for ax, (metric, ylabel, title, scale) in zip(axes.ravel(), panels):
                    for sellers, closure, label, color, line, marker in base.DEFAULT_MODELS:
                        values = scale*np.asarray(metric_values(lookup, cases, p0, rhos, sellers, closure, metric))
                        ax.plot(positions, values, color=color, ls=line, marker=marker, ms=5,
                                markerfacecolor=color if sellers == 1 else 'none', label=label)
                    omitted = [rho for rho in rhos if not cases[case_key(p0, rho)]['metrics'][metric]['passed']]
                    ax.set(ylabel=ylabel, title=title+'\nOmitted ρ: '+(', '.join(f'{rho:g}' for rho in omitted) or 'none'),
                           xlabel='Persistence ρ (equally spaced cases)')
                    ax.set_xticks(positions, [f'{rho:g}' for rho in rhos])
                    if scale == 100:
                        ax.set_ylim(0, 100)
                    if len(rhos) > 1:
                        ax.set_xlim(-.15, len(rhos)-.85)
                    if len(omitted) == len(rhos):
                        ax.text(.5, .5, 'No grid-stable metric points', ha='center', transform=ax.transAxes)
                decorate(fig, axes, f'N=1 vs N=2 vs N=3 | fixed p₀ = {p0:g}',
                    'Only metric-validated points; case-specific grids and checks are listed in the report.')
                paths.extend(base.save(fig, directory, stem))
        fig, axes = facets(rhos, panel_height=3.2)
        for ax, rho in zip(axes.ravel(), rhos):
            omitted = []
            for p0, color in zip(p0s, P0_COLORS):
                case = cases[case_key(p0, rho)]
                if not case['metrics']['product2_conditional']['passed']:
                    omitted.append(p0)
                    continue
                grid = case['displayed_n']
                values = [100*lookup[p0, rho, grid, sellers, closure]['product2_conditional']
                          for sellers, closure, *_ in base.DEFAULT_MODELS]
                ax.plot([1, 2, 3], values, color=color, marker='o', ms=5, lw=1.7, label=f'p₀ = {p0:g}')
            ax.set(title=f'ρ = {rho:g} | Marginal OE\nOmitted p₀: '+(', '.join(f'{p0:g}' for p0 in omitted) or 'none'),
                   xticks=[1, 2, 3], xlim=(.85, 3.15), ylim=(0, 100),
                   xlabel='Number of sellers N', ylabel='Product 2 (% of own sales)')
            ax.tick_params(labelbottom=True)
            if len(omitted) == len(p0s):
                ax.text(.5, .5, 'No grid-stable P2 cases', ha='center', transform=ax.transAxes)
        for ax in axes.ravel()[len(rhos):]:
            ax.set_visible(False)
        decorate(fig, axes, 'Product 2 use as the number of sellers changes',
                 'Same fixed p₀ and grid within each N comparison; each case passes its own P2 grid check.')
        paths.extend(base.save(fig, Path(figures), base.SELLER_COUNT_STEM))
    return paths


def write_index(figures, data, report):
    lines = ['# Uno, dos y tres sellers: OE marginal con p₀ fijo', '',
        '**La estabilidad se comprueba por caso y por métrica.** Cada comparación N=1,2,3 usa '
        'la misma grilla dentro de un (p₀,ρ); otros casos pueden usar otra grilla. '
        'Se contrastan las dos grillas comunes certificadas más finas de cada caso.', '',
        'Los puntos que no pasan el filtro de su métrica se omiten y aparecen como gaps. '
        'Un beneficio estacionario sensible a la grilla no invalida automáticamente un punto de producto 2 estable.', '',
        f'**Comparación principal:** [PNG]({base.SELLER_COUNT_STEM}.png) · [PDF]({base.SELLER_COUNT_STEM}.pdf). '
        'Cada curva mantiene p₀ fijo al cambiar N.', '',
        'Producto 2: porcentaje de ventas propias. Outside: períodos sin ventas en todo el mercado. '
        'Beneficios por seller; valor inicial descontado del juego real. '
        'Las políticas son soluciones certificadas en nodos originales; no se demuestra estabilidad puntual de sus umbrales.', '',
        '| p₀ | Políticas | Producto 2 | Beneficios | Outside |', '|---|---|---|---|---|']
    for p0 in report['requested_p0s']:
        directory = 'p0_'+str(float(p0)).replace('.', 'p')
        cells = [f'[PNG]({directory}/{stem}.png) · [PDF]({directory}/{stem}.pdf)' for stem in base.STEMS]
        lines.append('| '+f'{p0:g}'+' | '+' | '.join(cells)+' |')
    lines += ['', 'Cada casilla indica el cambio máximo entre N=1,2,3: probabilidades en puntos porcentuales; '
              'beneficio/valor en porcentaje relativo simétrico. ✓ pasa el límite de 5; ✗ se omite.', '',
              '| p₀ | ρ | Grillas | Producto 2 | Outside | Beneficio | Valor inicial |', '|---|---|---|---|---|---|---|']
    for case in report['case_grids'].values():
        cells = []
        for metric in METRICS:
            result = case['metrics'][metric]
            cells.append(('✓ ' if result['passed'] else '✗ ')
                         +(f"{100*result['max_change']:.3g}" if result['max_change'] is not None else 'faltan grillas'))
        lines.append(f"| {case['p0']:g} | {case['rho']:g} | {case['validation_coarse_n']} → {case['displayed_n']} | "+' | '.join(cells)+' |')
    stability = Path(os.path.relpath(Path(data)/STABILITY_FILE, figures)).as_posix()
    retention = Path(os.path.relpath(Path(data)/RETENTION_FILE, figures)).as_posix()
    lines += ['', f'Detalles: [contrastes por modelo]({stability}) · [retención por métrica]({retention}).', '',
        'OE marginal conserva la distribución conjunta de los rivales; no supone independencia iid. '
        'La auditoría independiente verifica los resultados guardados. Los contrastes entre grillas '
        'son evidencia local finita y no prueban convergencia continua, unicidad ni equilibrio Markov perfecto.', '']
    path = Path(figures)/'README.md'
    path.write_text('\n'.join(lines))
    return path


def build_report(data=None, figures=None, *, check_only=False):
    data = Path(data if data is not None else base.HOME/'data/rho_p0_sweep').resolve()
    figures = Path(figures if figures is not None else base.HOME/'figures/rho_p0_sweep').resolve()
    source_status = json.loads((data/'manifest.json').read_text()).get('status')
    loaded = (read_diagnostic_snapshot(data) if check_only and source_status in ('created', 'running')
              else base.read_inputs(data))
    manifest, rows, policies, inputs, p0s, rhos, models = loaded
    base.require(tuple(models) == base.DEFAULT_MODELS, 'The expanded comparison is marginal-only')
    base.require(len(p0s) <= len(P0_COLORS), 'Use at most six distinct p0 values')
    cases, comparisons = select_cases(rows, manifest, p0s, rhos)
    retention = metric_retention(cases, p0s, rhos)
    all_passed = all(case['metrics'][metric]['passed'] for case in cases.values() for metric in METRICS)
    any_passed = any(case['metrics'][metric]['passed'] for case in cases.values() for metric in METRICS)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='complete_gridstable' if all_passed else 'partial_gridstable' if any_passed else 'no_gridstable_metrics',
        diagnostic_only=check_only, source_campaign_status=manifest.get('status'),
        requested_p0s=p0s, requested_rhos=rhos,
        models=[dict(sellers=m[0], closure=m[1], label=m[2]) for m in models],
        case_grids=cases, metric_retention=retention, all_metrics_grid_stable=all_passed,
        probability_change_limit=.05, symmetric_relative_earnings_change_limit=.05,
        relative_change_definition='abs(fine-coarse)/max(abs(fine),abs(coarse)); zero if both zero',
        policy_measure='Original nodes on each case finest common certified grid; no pointwise grid-stability claim',
        product2_measure='Percentage among own sales', outside_measure='Whole-market outside probability',
        value_source='Actual game discounted value at initial reputations 0.5', source_input_sha256=inputs,
        reporting_source_sha256={base.location(base.__file__): base.digest(base.__file__), base.location(__file__): base.digest(__file__)},
        scientific_scope='Saved finite-grid certificates plus independent audit; metric stability recorded separately; no continuous convergence or uniqueness claim')
    if check_only:
        report.update(verified=False, independent_audit=None,
            diagnostic_note='Read-only snapshot; no figures or metadata written; live CSVs provisional until the complete independent audit')
        return report
    base.require(manifest.get('status') == 'complete', 'Complete campaign is required before publication')
    independent, audit_inputs = base.read_optional_audit(data, manifest)
    base.require(independent is not None, 'The complete independent audit is required before publication')
    report.update(independent_audit=independent, verified=True,
                  verification_scope='Independent finite-grid data audit; metric stability is certified separately')
    inputs.update(audit_inputs)
    base.verify_inputs(inputs)
    figures.mkdir(parents=True, exist_ok=True)
    paths = render(rows, policies, figures, p0s, rhos, cases)
    stability, retained = data/STABILITY_FILE, data/RETENTION_FILE
    base.write_csv(stability, comparisons)
    base.write_json(retained, retention)
    index = write_index(figures, data, report)
    report['figure_files'] = {base.location(path): base.digest(path) for path in paths}
    report['generated_data_sha256'] = {base.location(path): base.digest(path) for path in (stability, retained, index)}
    report['primary_seller_count_comparison'] = {suffix: base.location(figures/(base.SELLER_COUNT_STEM+'.'+suffix)) for suffix in ('png', 'pdf')}
    base.verify_inputs(inputs)
    base.write_json(data/'report_manifest.json', report)
    base.write_json(figures/'report_manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--figures', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    result = build_report(args.data, args.figures, check_only=args.check_only)
    print(json.dumps(result if args.check_only else {key: result[key] for key in (
        'status', 'verified', 'requested_p0s', 'requested_rhos', 'all_metrics_grid_stable')}, indent=2))


if __name__ == '__main__':
    main()
