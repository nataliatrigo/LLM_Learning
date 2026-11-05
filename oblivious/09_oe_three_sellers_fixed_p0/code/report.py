"""Report certified N=1,2,3 fixed-p0 results without solving any model."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HOME = Path(__file__).resolve().parents[1]
ROOT = HOME.parents[1]
P0S, RHOS = [.1, .5, .8], [.6, .8]
MODELS = ((1, 'monopoly', 'N=1 monopoly', '#0072B2', '-', 'o'),
          (2, 'marginal', 'N=2 marginal OE', '#D55E00', '--', 's'),
          (2, 'conditional', 'N=2 conditional OE', '#D55E00', '-.', '^'),
          (3, 'marginal', 'N=3 marginal OE', '#009E73', '--', 's'),
          (3, 'conditional', 'N=3 conditional OE', '#009E73', '-.', '^'))
DEFAULT_MODELS = tuple(model for model in MODELS if model[1] != 'conditional')
PROBABILITIES = ('product2_conditional', 'sale_rate', 'outside_rate')
EARNINGS = ('profit_calendar', 'value_initial')
STEMS = ('01_policy_comparison', '02_product2_usage_comparison',
         '03_profit_comparison', '04_outside_option_comparison')
SELLER_COUNT_STEM = '05_product2_by_seller_count'


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def write_csv(path, rows):
    with Path(path).open('w', newline='') as handle:
        if rows:
            writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
            writer.writeheader()
            writer.writerows(rows)


def location(path):
    path = Path(path).resolve()
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def verify_inputs(inputs):
    for name, expected in inputs.items():
        path = Path(name)
        path = path if path.is_absolute() else ROOT/path
        require(path.is_file() and digest(path) == expected, 'Input fingerprint differs: '+name)


def read_optional_audit(data, manifest):
    """Link an available independent review, rejecting stale or partial audits."""
    path = Path(data)/'independent_audit.json'
    if not path.is_file():
        return None, {}
    audit = json.loads(path.read_text())
    require(audit.get('status') == 'complete_verified' and audit.get('verified') is True,
            'Independent audit is not complete and verified')
    require(audit.get('manifest_snapshot_sha256') == digest(Path(data)/'manifest.json'),
            'Independent audit manifest snapshot is stale')
    require(audit.get('campaign_input_sha256') == manifest.get('input_sha256'),
            'Independent audit does not match the current campaign inputs')
    require(audit.get('numerical_sources') == manifest.get('numerical_sources'),
            'Independent audit numerical source fingerprints differ from the campaign')
    inputs = {}
    for field in ('audit_sources', 'numerical_sources'):
        declared = audit.get(field)
        require(isinstance(declared, dict) and bool(declared),
                'Independent audit has no fingerprinted '+field)
        verify_inputs(declared)
        for source, expected in declared.items():
            require(source not in inputs or inputs[source] == expected,
                    'Independent audit source fingerprint conflicts: '+source)
            inputs[source] = expected
    audit_hash = digest(path)
    inputs[location(path)] = audit_hash
    summary = {key: audit.get(key) for key in (
        'status', 'verified', 'audited_utc', 'manifest_snapshot_sha256',
        'expected_model_grid_cases', 'audited_model_grid_cases', 'mutable_csv_fingerprints_checked', 'scope')}
    summary.update(path=location(path), sha256=audit_hash,
                   audit_sources=audit['audit_sources'], numerical_sources=audit['numerical_sources'])
    return summary, inputs


def read_inputs(data):
    """Verify immutable numerical/reference files and both runner CSV exports."""
    data = Path(data).resolve()
    path = data/'manifest.json'
    manifest = json.loads(path.read_text())
    require(manifest.get('status') != 'running' and manifest.get('current_grid') is None
            and manifest.get('current_model') is None, 'The pilot is still running; no report is published')
    configuration = manifest.get('configuration', {})
    requested_models = [tuple(model) for model in configuration.get(
        'models', [(m[0], m[1]) for m in DEFAULT_MODELS])]
    requested_set = set(requested_models)
    require(len(requested_models) == len(requested_set)
            and requested_set in ({(m[0], m[1]) for m in DEFAULT_MODELS},
                                  {(m[0], m[1]) for m in MODELS}),
            'Use the complete N=1,2,3 marginal comparison, optionally with both conditional models')
    models = tuple(model for model in MODELS if (model[0], model[1]) in requested_set)
    inputs = {location(path): digest(path)}
    declared_csv = set()
    source_count = 0
    for field in ('numerical_sources', 'source_sha256', 'reference_input_sha256',
                  'input_sha256', 'source_input_sha256', 'generated_data_sha256'):
        declared = manifest.get(field, {})
        require(isinstance(declared, dict), 'Fingerprint map is malformed: '+field)
        for name, expected in declared.items():
            candidate = Path(name)
            if not candidate.is_absolute():
                candidate = ROOT/candidate if (ROOT/candidate).is_file() else data/candidate
            candidate = candidate.resolve()
            require(candidate.is_file() and digest(candidate) == expected, 'Input fingerprint differs: '+str(candidate))
            key = location(candidate)
            require(key not in inputs or inputs[key] == expected, 'Conflicting input fingerprints: '+key)
            inputs[key] = expected
            if candidate in (data/'summary.csv', data/'policies.csv'):
                declared_csv.add(candidate.name)
            if candidate.suffix in ('.py', '.cpp'):
                source_count += 1
    require(source_count > 0, 'The manifest must fingerprint numerical source files')
    require(declared_csv == {'summary.csv', 'policies.csv'},
            'The manifest must fingerprint both summary.csv and policies.csv')
    with (data/'summary.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    with (data/'policies.csv').open(newline='') as handle:
        policies = list(csv.DictReader(handle))
    for records in (rows, policies):
        for row in records:
            for name in ('p0', 'rho'):
                row[name] = float(row[name])
            for name in ('n', 'sellers'):
                row[name] = int(row[name])
            require((row['sellers'], row['closure']) in requested_set,
                    'Unknown model/closure in runner exports')
    for row in rows:
        row['verified'] = str(row['verified']).lower() in ('true', '1')
        for name in PROBABILITIES+EARNINGS:
            row[name] = float(row[name])
        require(np.isfinite([row[name] for name in PROBABILITIES+EARNINGS]).all(),
                'Nonfinite reported outcome')
        require(all(-1e-12 <= row[name] <= 1+1e-12 for name in PROBABILITIES),
                'Reported probability is outside [0,1]')
        require(abs(row['outside_rate']+row['sellers']*row['sale_rate']-1) <= 1e-9,
                'Outside must refer to the whole market: outside+N*sale_rate=1')
    for row in policies:
        for name in ('reputation', 'product2_probability'):
            row[name] = float(row[name])
            require(np.isfinite(row[name]) and -1e-12 <= row[name] <= 1+1e-12,
                    'Malformed reported policy')
    keys = [(r['p0'], r['rho'], r['n'], r['sellers'], r['closure']) for r in rows]
    require(len(keys) == len(set(keys)), 'Duplicate reported model/grid outcome')
    groups = {}
    for row in policies:
        key = (row['p0'], row['rho'], row['n'], row['sellers'], row['closure'])
        groups.setdefault(key, []).append(row)
    require(set(groups) <= set(keys), 'Policy export has no corresponding outcome')
    for row, key in zip(rows, keys):
        if not row['verified']:
            continue
        selected = sorted(groups.get(key, []), key=lambda item: item['reputation'])
        require(len(selected) == row['n'] and np.array_equal(
            [item['reputation'] for item in selected], np.linspace(0, 1, row['n'])),
            'Certified policy is not the original uniform reputation grid')
    p0s = sorted(configuration.get('p0s', manifest.get('p0s', P0S)))
    rhos = sorted(configuration.get('rhos', manifest.get('rhos', RHOS)))
    require(bool(p0s) and len(p0s) == len(set(p0s)) and all(0 < p0 < 1 for p0 in p0s)
            and bool(rhos) and len(rhos) == len(set(rhos)) and all(.5 < rho < 1 for rho in rhos),
            'Invalid requested p0/rho scope')
    return manifest, rows, policies, inputs, p0s, rhos, models


def select_grids(rows, p0s, rhos, models=DEFAULT_MODELS):
    """Use one common original grid and its next smaller certified common grid."""
    available = []
    for p0 in p0s:
        for rho in rhos:
            for sellers, closure, *_ in models:
                available.append({row['n'] for row in rows if row['p0'] == p0 and row['rho'] == rho
                                  and row['sellers'] == sellers and row['closure'] == closure and row['verified']})
    common = sorted(set.intersection(*available)) if available else []
    return (common[-1] if common else None, common[-2] if len(common) > 1 else None, common)


def sensitivity(rows, p0s, rhos, coarse, grid, models=DEFAULT_MODELS):
    lookup = {(r['p0'], r['rho'], r['n'], r['sellers'], r['closure']): r for r in rows}
    comparisons, excluded = [], {}
    if coarse is None or grid is None:
        return [], {}, []
    for rho in rhos:
        reasons = []
        for p0 in p0s:
            for sellers, closure, *_ in models:
                pair = [lookup[p0, rho, n, sellers, closure] for n in (coarse, grid)]
                changes = {}
                failed = []
                for metric in PROBABILITIES+EARNINGS:
                    first, second = [r[metric] for r in pair]
                    difference = abs(second-first)
                    change = difference if metric in PROBABILITIES else (
                        difference/max(abs(first), abs(second)) if first or second else 0.)
                    changes[metric+('_absolute_change' if metric in PROBABILITIES else '_relative_change')] = change
                    if change > .05+1e-12:
                        failed.append(dict(metric=metric, change=change, limit=.05))
                comparisons.append(dict(p0=p0, rho=rho, sellers=sellers, closure=closure,
                    coarse_n=coarse, fine_n=grid, displayed_n=grid, passed=not failed, **changes))
                reasons.extend(dict(p0=p0, sellers=sellers, closure=closure, **item) for item in failed)
        if reasons:
            excluded[str(rho)] = reasons
    retained = [rho for rho in rhos if str(rho) not in excluded]
    return comparisons, excluded, retained


def save(fig, directory, stem):
    paths = []
    for suffix in ('png', 'pdf'):
        path = directory/(stem+'.'+suffix)
        fig.savefig(path, dpi=180, bbox_inches='tight')
        paths.append(path)
    plt.close(fig)
    return paths


def style(ax):
    ax.grid(alpha=.2)
    ax.spines[['top', 'right']].set_visible(False)


def finish_figure(fig, axes, p0, grid, coarse):
    handles = {}
    for ax in np.asarray(axes).ravel():
        style(ax)
        artists, labels = ax.get_legend_handles_labels()
        handles.update(zip(labels, artists))
    fig.legend(handles.values(), handles.keys(), loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle(f'N=1 vs N=2 vs N=3 | fixed p₀ = {p0:g}\n'
                 f'n = {grid} | checked n = {coarse} → {grid} | joint rival beliefs, no iid factorization', fontsize=11)


def render_product2_by_seller_count(rows, figures, p0s, rhos, grid, coarse, models=DEFAULT_MODELS):
    """Hold p0 fixed within a curve and compare N on the same retained grid."""
    lookup = {(r['p0'], r['rho'], r['sellers'], r['closure']): r
              for r in rows if r['n'] == grid}
    colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00')
    model_set = {(m[0], m[1]) for m in models}
    closures = [closure for closure in ('marginal', 'conditional')
                if (2, closure) in model_set and (3, closure) in model_set]
    fig, axes = plt.subplots(len(rhos), len(closures), figsize=(5.4*len(closures), 3.5*len(rhos)),
                             sharex=True, sharey=True, squeeze=False, layout='constrained')
    for row_index, rho in enumerate(rhos):
        for column, closure in enumerate(closures):
            ax = axes[row_index, column]
            for index, p0 in enumerate(p0s):
                values = [100*lookup[p0, rho, sellers, 'monopoly' if sellers == 1 else closure]['product2_conditional']
                          for sellers in (1, 2, 3)]
                ax.plot([1, 2, 3], values, color=colors[index % len(colors)], marker='o',
                        lw=1.8, ms=5, label=f'p₀ = {p0:g}')
            ax.set(title=f'ρ = {rho:g} | {closure.capitalize()} OE',
                   xticks=[1, 2, 3], xlim=(.85, 3.15), ylim=(0, 100),
                   xlabel='Number of sellers N')
            if column == 0:
                ax.set_ylabel('Product 2 (% of own sales)')
            style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=len(p0s), frameon=False)
    note = ('N=1 monopoly is shared across the two belief closures.' if len(closures) == 2
            else 'Marginal OE retains joint rival beliefs; no iid factorization.')
    fig.suptitle('Product 2 use as the number of sellers changes\n'
                 f'n = {grid} | checked n = {coarse} → {grid} | same fixed p₀ within each curve\n'
                 +note, fontsize=11)
    return save(fig, Path(figures), SELLER_COUNT_STEM)


def render(rows, policies, figures, p0s, rhos, grid, coarse, models=DEFAULT_MODELS):
    paths = []
    lookup = {(r['p0'], r['rho'], r['sellers'], r['closure']): r for r in rows if r['n'] == grid}
    positions = np.arange(len(rhos))
    with plt.rc_context({'font.size': 9, 'axes.titlesize': 10, 'legend.fontsize': 8}):
        for p0 in p0s:
            directory = Path(figures)/('p0_'+str(float(p0)).replace('.', 'p'))
            directory.mkdir(parents=True, exist_ok=True)
            fig, axes = plt.subplots(1, len(rhos), figsize=(5.4*len(rhos), 4),
                                     sharex=True, sharey=True, squeeze=False, layout='constrained')
            for ax, rho in zip(axes.ravel(), rhos):
                for sellers, closure, label, color, line, marker in models:
                    chosen = sorted((r for r in policies if r['p0'] == p0 and r['rho'] == rho and r['n'] == grid
                                     and r['sellers'] == sellers and r['closure'] == closure), key=lambda r: r['reputation'])
                    ax.plot([r['reputation'] for r in chosen], [r['product2_probability'] for r in chosen],
                            color=color, ls=line, lw=1.6, marker=marker if closure == 'conditional' else None,
                            markevery=max(1, grid//10), ms=4, markerfacecolor='none', label=label)
                ax.set(title=f'ρ = {rho:g}', xlabel='Own reputation x', ylim=(-.03, 1.03), xlim=(0, 1))
            axes[0, 0].set_ylabel('Pr(product 2 | own sale, x)')
            finish_figure(fig, axes, p0, grid, coarse)
            paths.extend(save(fig, directory, STEMS[0]))
            specifications = [
                (STEMS[1], [('product2_conditional', 'Product 2 (% of own sales)', 'Product mix when this seller sells', 100)]),
                (STEMS[2], [('profit_calendar', 'Profit per seller / calendar period', 'Stationary profit', 1),
                            ('value_initial', 'Discounted profit per seller', 'Actual game value; initial reputations = 0.5', 1)]),
                (STEMS[3], [('outside_rate', 'Outside option (% of periods)', 'Whole market: no seller makes a sale', 100)]),
            ]
            for stem, panels in specifications:
                fig, axes = plt.subplots(1, len(panels), figsize=(6.7*len(panels), 4.8),
                                         squeeze=False, layout='constrained')
                for ax, (metric, ylabel, title, scale) in zip(axes.ravel(), panels):
                    for sellers, closure, label, color, line, marker in models:
                        values = [scale*lookup[p0, rho, sellers, closure][metric] for rho in rhos]
                        ax.plot(positions, values, color=color, ls=line, marker=marker, ms=5,
                                markerfacecolor=color if sellers == 1 else 'none', label=label)
                    ax.set(ylabel=ylabel, title=title, xlabel='Persistence ρ (equally spaced cases)')
                    ax.set_xticks(positions, [f'{rho:g}' for rho in rhos])
                    if scale == 100:
                        ax.set_ylim(0, 100)
                    if len(rhos) > 1:
                        ax.set_xlim(-.15, len(rhos)-.85)
                finish_figure(fig, axes, p0, grid, coarse)
                paths.extend(save(fig, directory, stem))
        paths.extend(render_product2_by_seller_count(rows, figures, p0s, rhos, grid, coarse, models=models))
    return paths


def build_report(data=None, figures=None):
    data = Path(data if data is not None else HOME/'data/marginal_pilot').resolve()
    figures = Path(figures if figures is not None else HOME/'figures/marginal_pilot').resolve()
    manifest, rows, policies, inputs, p0s, rhos, models = read_inputs(data)
    independent, audit_inputs = read_optional_audit(data, manifest)
    inputs.update(audit_inputs)
    grid, coarse, common = select_grids(rows, p0s, rhos, models=models)
    checks, excluded, retained = sensitivity(rows, p0s, rhos, coarse, grid, models=models)
    conditional_included = any(m[1] == 'conditional' for m in models)
    status = ('no_common_certified_grid' if grid is None else 'missing_validation_grid' if coarse is None
              else 'no_common_stable_rhos' if not retained else 'complete' if retained == rhos else 'partial_rhos')
    figures.mkdir(parents=True, exist_ok=True)
    stability = data/'grid_stability.csv'
    write_csv(stability, checks)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), status=status,
        verified=bool(retained), requested_p0s=p0s, requested_rhos=rhos,
        common_certified_grids=common, displayed_n=grid, validation_coarse_n=coarse,
        common_retained_rhos=retained, excluded_rhos=excluded,
        probability_change_limit=.05, symmetric_relative_earnings_change_limit=.05,
        relative_change_definition='abs(fine-coarse)/max(abs(fine),abs(coarse)); zero if both zero',
        models=[dict(sellers=m[0], closure=m[1], label=m[2]) for m in models],
        source_input_sha256=inputs, reporting_source_sha256={location(__file__): digest(__file__)},
        source_campaign_status=manifest.get('status'),
        independent_audit=independent,
        belief_closure=('Marginal OE retains the joint law of all rivals independent of own state; no iid factorization'
                        +('; conditional OE uses its own-state conditional joint law' if conditional_included else '')),
        value_source='Actual calendar-time game values from initial reputations 0.5, not perceived OE values',
        product2_measure='Percentage among this seller own sales',
        outside_measure='Whole-market probability that no seller sells; outside+N*sale_rate=1',
        scientific_scope='Certified finite-grid outcomes and metric sensitivity; no claim of continuous convergence, uniqueness or full-state MPE')
    paths = render(rows, policies, figures, p0s, retained, grid, coarse, models=models) if retained else []
    lines = ['# N=1, N=2 y N=3 con el mismo p₀', '']
    if retained:
        lines += [f'Curvas originales de {grid} nodos, contrastadas con {coarse} nodos. '
                  f'ρ comunes que pasan el filtro para todos los p₀ y los {len(models)} modelos: '
                  +', '.join(f'{rho:g}' for rho in retained)+'.', '',
                  'Producto 2: porcentaje de las ventas propias. Outside: porcentaje de períodos sin '
                  'venta de ningún seller. Beneficios por seller; valores descontados del juego real.', '',
                  'El OE marginal conserva la distribución conjunta de los rivales. No se supone independencia entre rivales.'
                  +(' El condicional usa su distribución conjunta dado el estado propio.' if conditional_included else ''), '',
                  '**Comparación principal: producto 2 al pasar de uno a dos y tres sellers:** '
                  f'[PNG]({SELLER_COUNT_STEM}.png) · [PDF]({SELLER_COUNT_STEM}.pdf). '
                  'Cada curva mantiene p₀ fijo; las filas separan ρ. '
                  +('Las columnas separan los cierres marginal y condicional; el monopolio es común a ambas.'
                    if conditional_included else 'Se comparan monopolio y OE marginal con dos y tres sellers.'), '',
                  '| p₀ | Políticas | Producto 2 | Beneficios | Outside |', '|---|---|---|---|---|']
        for p0 in p0s:
            directory = 'p0_'+str(float(p0)).replace('.', 'p')
            cells = [f'[PNG]({directory}/{stem}.png) · [PDF]({directory}/{stem}.pdf)' for stem in STEMS]
            lines.append('| '+f'{p0:g}'+' | '+' | '.join(cells)+' |')
    else:
        lines += ['**Diagnóstico: todavía no hay una comparación común certificada y estable entre grillas.**', '',
                  f'Estado: `{status}`. No se generaron plots verificados para esta ejecución.',
                  'Consulte `grid_stability.csv` y `report_manifest.json` en la carpeta de datos.']
    lines += ['', 'Filtro: máximo 5 puntos porcentuales en probabilidades y 5% relativo simétrico '
              'en beneficios/valores. Es evidencia finita; no demuestra convergencia continua ni unicidad.', '']
    index = figures/'README.md'
    index.write_text('\n'.join(lines))
    report['figure_files'] = {location(path): digest(path) for path in paths}
    report['primary_seller_count_comparison'] = (
        {suffix: location(figures/(SELLER_COUNT_STEM+'.'+suffix)) for suffix in ('png', 'pdf')}
        if retained else None)
    report['generated_data_sha256'] = {location(stability): digest(stability), location(index): digest(index)}
    verify_inputs(inputs)
    write_json(data/'report_manifest.json', report)
    write_json(figures/'report_manifest.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--figures', type=Path)
    args = parser.parse_args()
    result = build_report(args.data, args.figures)
    print(json.dumps({key: result[key] for key in ('status', 'verified', 'displayed_n',
                                                  'validation_coarse_n', 'common_retained_rhos')}, indent=2))


if __name__ == '__main__':
    main()
