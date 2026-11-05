"""Plot successor demand from certified checkpoints; never solve equilibria.

Run with python -m oblivious.oe_three_sellers_fixed_p0.code.successor_demand.
The frozen campaign/reporting sources and their published figures are untouched.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from . import expanded_report as expanded
from . import report as base


P0 = .8
POLICY_TOLERANCE = 1e-7
GRID_CHANGE_LIMIT = .05
STATE_CHANGE_LIMIT = .05
STATE_QUANTILES = np.asarray([.1, .25, .5, .75, .9])
PROBABILITY_AREA_SCALE = 900.  # scatter area in points squared per unit probability
METRICS = ('average_successor_demand', 'outcome_sensitivity')
SHADE = '#f0d28b'
NAME = 'successor_demand_p0_0p8'


def successor_diagnostics(grid, demand, rho, x=None):
    """Equal-weight mean and success-minus-failure gap, in probability units."""
    grid, demand = np.asarray(grid), np.asarray(demand)
    x = grid if x is None else np.asarray(x)
    base.require(grid.ndim == 1 and len(grid) >= 2 and demand.shape == grid.shape,
                 'Demand must be a vector on the saved grid')
    base.require(np.isfinite(grid).all() and np.all(np.diff(grid) > 0)
                 and grid[0] == 0 and grid[-1] == 1, 'Require a sorted [0,1] grid')
    base.require(np.isfinite(demand).all() and np.all((demand >= 0) & (demand <= 1)),
                 'Demand must contain finite probabilities')
    base.require(0 <= rho <= 1 and np.isfinite(x).all() and np.all((x >= 0) & (x <= 1)),
                 'Successor states must remain on the saved grid domain')
    xs, xf = rho*x+1-rho, rho*x
    ds, df = np.interp(xs, grid, demand), np.interp(xf, grid, demand)
    return dict(x=x, x_success=xs, x_failure=xf, demand_success=ds, demand_failure=df,
                average_successor_demand=(ds+df)/2, outcome_sensitivity=ds-df)


def disagreement_cells(grid, policies):
    """Shade midpoint cells of disagreeing nodes, including mixed policies.

    Cell boundaries are a visual convention, not continuous policy thresholds.
    Differences below the saved equilibrium tolerance are numerical noise.
    """
    mask = np.ptp(np.asarray(policies), axis=0) > POLICY_TOLERANCE
    edges = np.r_[grid[0], (grid[:-1]+grid[1:])/2, grid[-1]]
    nodes = np.flatnonzero(mask)
    groups = np.split(nodes, np.flatnonzero(np.diff(nodes) != 1)+1) if len(nodes) else []
    cells = [dict(left=float(edges[group[0]]), right=float(edges[group[-1]+1]),
                  original_nodes=grid[group].tolist()) for group in groups]
    return mask, cells


def state_distribution(grid, joint, sellers):
    """Own calendar-time marginal of the saved joint law, without sale weights."""
    n = len(grid)
    base.require(joint.shape == (n,)*sellers and np.isfinite(joint).all()
                 and np.all(joint >= 0) and abs(float(joint.sum())-1) < 1e-10,
                 'Invalid saved calendar-time stationary distribution')
    mass = joint.reshape(n, -1).sum(axis=1)
    base.require(abs(float(mass.sum())-1) < 1e-10, 'Own-state probabilities must sum to one')
    cdf = np.cumsum(mass)
    quantiles = grid[np.searchsorted(cdf, STATE_QUANTILES)]
    return dict(mass=mass, quantiles=quantiles)


def distribution_comparison(coarse, fine):
    """Compare discrete laws on different grids via transport and quantiles.

    W1 is the exact integral of the difference between the two step CDFs.
    Node probabilities themselves are not interpolated or compared pointwise.
    """
    coordinates = np.union1d(coarse['grid'], fine['grid'])
    cdfs = []
    for model in (coarse, fine):
        indices = np.searchsorted(model['grid'], coordinates, side='right')
        cdfs.append(np.r_[0., np.cumsum(model['stationary']['mass'])][indices])
    gap = np.abs(cdfs[0]-cdfs[1])
    transport = float(gap[:-1] @ np.diff(coordinates))
    quantile_change = float(np.max(abs(coarse['stationary']['quantiles']-fine['stationary']['quantiles'])))
    return dict(state_wasserstein1=transport, state_cdf_sup=float(gap.max()),
                state_max_quantile_change=quantile_change,
                state_frequency_passed=bool(max(transport, quantile_change) <= STATE_CHANGE_LIMIT+1e-12))


def load_checkpoint(data, p0, rho, n, sellers, closure, inputs):
    stem = data/expanded.case_key(p0, rho)/f'n{n}_N{sellers}_{closure}'
    record_path, checkpoint = stem.with_suffix('.json'), stem.with_suffix('.npz')
    for path in (record_path, checkpoint):
        key = base.location(path)
        base.require(key in inputs and inputs[key] == base.digest(path),
                     'Checkpoint is not fingerprinted by the audited campaign: '+key)
    record = json.loads(record_path.read_text())
    config = record['configuration']
    base.require((config['p0'], config['parameters']['rho'], config['n'],
                  config['sellers'], config['closure']) == (p0, rho, n, sellers, closure)
                 and record['summary']['verified'] is True, 'Wrong or uncertified checkpoint')
    base.require(record['checkpoint_sha256'] == base.digest(checkpoint), 'Checkpoint hash differs')
    with np.load(checkpoint, allow_pickle=False) as arrays:
        grid, demand, policy, joint = [arrays[key].copy() for key in ('grid', 'demand', 'policy', 'joint')]
    base.require(np.array_equal(grid, np.linspace(0, 1, n)), 'Require original common grid')
    base.require(policy.shape == grid.shape and np.isfinite(policy).all()
                 and np.all((policy >= 0) & (policy <= 1)), 'Invalid saved policy')
    return dict(grid=grid, demand=demand, policy=policy,
                stationary=state_distribution(grid, joint, sellers),
                diagnostics=successor_diagnostics(grid, demand, rho))


def prepare(data):
    manifest, rows, _, inputs, p0s, rhos, models = base.read_inputs(data)
    base.require(P0 in p0s and models == base.DEFAULT_MODELS, 'Require N=1,2,3 marginal at p0=.8')
    base.require(manifest['status'] == 'complete' and not manifest.get('active_workers')
                 and manifest.get('current_case') is None, 'Require a completed, inactive campaign')
    audit, audit_inputs = base.read_optional_audit(data, manifest)
    base.require(audit is not None, 'Require the completed independent audit')
    inputs.update(audit_inputs)
    selections, _ = expanded.select_cases(rows, manifest, [P0], rhos)
    cases, records, stability = [], [], []
    for rho in rhos:
        selection = selections[expanded.case_key(P0, rho)]
        n, coarse_n = selection['displayed_n'], selection['validation_coarse_n']
        base.require(n is not None and coarse_n is not None, 'Need two common certified grids')
        saved = [load_checkpoint(data, P0, rho, n, sellers, closure, inputs)
                 for sellers, closure, *_ in models]
        grid = saved[0]['grid']
        base.require(all(np.array_equal(model['grid'], grid) for model in saved),
                     'The three models must share their original own-reputation grid')
        mask, cells = disagreement_cells(grid, [model['policy'] for model in saved])
        included = {metric: True for metric in METRICS}
        state_included = True
        for model, (sellers, closure, *_rest) in zip(saved, models):
            coarse = load_checkpoint(data, P0, rho, coarse_n, sellers, closure, inputs)
            # Compare the two saved demand functions at identical successor states.
            control = successor_diagnostics(coarse['grid'], coarse['demand'], rho, x=grid)
            comparison = dict(p0=P0, rho=rho, sellers=sellers, coarse_n=coarse_n, fine_n=n)
            comparison.update(distribution_comparison(coarse, model))
            state_included &= comparison['state_frequency_passed']
            for metric in METRICS:
                change = float(np.max(abs(model['diagnostics'][metric]-control[metric])))
                passed = change <= GRID_CHANGE_LIMIT+1e-12
                comparison[metric+'_max_abs_change'] = change
                comparison[metric+'_passed'] = passed
                included[metric] &= passed
            stability.append(comparison)
        case = dict(rho=rho, n=n, coarse_n=coarse_n, grid=grid, models=saved,
                    disagreement_mask=mask, disagreement_cells=cells, included=included,
                    state_frequency_included=state_included)
        cases.append(case)
        for model, (sellers, closure, *_rest) in zip(saved, models):
            for index, x in enumerate(grid):
                records.append(dict(p0=P0, rho=rho, n=n, sellers=sellers, closure=closure,
                    x=float(x), perceived_demand=float(model['demand'][index]),
                    product2_probability=float(model['policy'][index]),
                    stationary_state_probability=float(model['stationary']['mass'][index]),
                    state_frequency_included=state_included,
                    policy_disagreement=bool(mask[index]),
                    **{key: float(values[index]) for key, values in model['diagnostics'].items() if key != 'x'},
                    **{metric+'_included': included[metric] for metric in METRICS}))
    return cases, records, stability, inputs, audit


def diagnostic_limits(cases):
    """Use common scales across rho, with enough headroom for every curve."""
    return [(0, max(.1, np.ceil(10*1.05*max(
        float(np.max(model['diagnostics'][metric]))
        for case in cases for model in case['models']))/10)) for metric in METRICS]


def draw_row(axes, case, limits):
    for ax in axes:
        for cell in case['disagreement_cells']:
            ax.axvspan(cell['left'], cell['right'], color=SHADE, alpha=.42, lw=0, zorder=0)
        ax.set(xlim=(0, 1), xlabel='Own reputation x', xticks=np.linspace(0, 1, 6))
        base.style(ax)
    for model, (_sellers, _closure, _label, color, line, _marker) in zip(case['models'], base.DEFAULT_MODELS):
        for ax, metric in zip(axes[:2], METRICS):
            if case['included'][metric]:
                ax.plot(case['grid'], model['diagnostics'][metric], color=color, ls=line, lw=1.8, zorder=4)
                if case['state_frequency_included']:
                    ax.scatter(case['grid'], model['diagnostics'][metric],
                               s=PROBABILITY_AREA_SCALE*model['stationary']['mass'],
                               color=color, alpha=.32, linewidths=.45, zorder=3, clip_on=False)
        axes[2].plot(case['grid'], model['policy'], color=color, ls=line, lw=1.7)
        changed = case['disagreement_mask']
        axes[2].plot(case['grid'][changed], model['policy'][changed], color=color,
                     marker='x', ms=4, linestyle='none')
    axes[0].set(ylim=limits[0], ylabel='Demand probability')
    axes[1].set(ylim=limits[1], ylabel='Demand difference')
    axes[2].set(ylim=(-.06, 1.06), ylabel='Pr(product 2 | own sale, x)', yticks=[0, .5, 1])
    for ax, metric in zip(axes[:2], METRICS):
        if not case['included'][metric]:
            ax.text(.5, .5, 'Omitted: demand grid check failed', ha='center', transform=ax.transAxes)
    if not np.any(case['disagreement_mask']):
        axes[2].text(.05, .92, 'Policies coincide on this grid', transform=axes[2].transAxes,
                     fontsize=8, va='top')


def draw_state_frequencies(ax, case):
    """Separate probability lanes keep overlapping models' masses readable."""
    for cell in case['disagreement_cells']:
        ax.axvspan(cell['left'], cell['right'], color=SHADE, alpha=.42, lw=0, zorder=0)
    if case['state_frequency_included']:
        for lane, model, spec in zip((2, 1, 0), case['models'], base.DEFAULT_MODELS):
            color = spec[3]
            ax.axhline(lane, color=color, alpha=.2, lw=.6, zorder=1)
            ax.scatter(case['grid'], np.full(case['n'], lane),
                       s=PROBABILITY_AREA_SCALE*model['stationary']['mass'],
                       color=color, alpha=.7, linewidths=.4, zorder=2, clip_on=False)
    else:
        ax.text(.5, .5, 'State frequencies omitted: grid check failed',
                transform=ax.transAxes, ha='center', fontsize=8)
    ax.set(xlim=(0, 1), ylim=(-.7, 2.7), yticks=[0, 1, 2],
           yticklabels=['N=3', 'N=2', 'N=1'], xticks=np.linspace(0, 1, 6), xlabel='Own reputation x')
    ax.set_title('Own-state frequency (% of calendar periods)', fontsize=8, pad=3)
    ax.tick_params(axis='y', length=0, labelsize=8)
    ax.spines[['top', 'right', 'left']].set_visible(False)
    for tick, spec in zip(reversed(ax.get_yticklabels()), base.DEFAULT_MODELS):
        tick.set_color(spec[3])


def make_figure(cases, limits=None):
    overview = len(cases) > 1
    limits = diagnostic_limits(cases) if limits is None else limits
    fig = base.plt.figure(figsize=(14.5, 3.45*len(cases)+1.35), layout='constrained')
    grid = fig.add_gridspec(len(cases), 3, width_ratios=[1, 1, .9])
    # Use mathtext for outcome subscripts to keep the PDF font portable.
    titles = (r'Average successor demand'+'\n'+r'$[d(x_s)+d(x_f)]/2$',
              'Demand sensitivity to the outcome\n'+r'$d(x_s)-d(x_f)$', 'Saved product policies')
    for row, case in enumerate(cases):
        policy_grid = grid[row, 2].subgridspec(2, 1, height_ratios=[3.3, 1.2])
        row_axes = [fig.add_subplot(grid[row, 0]), fig.add_subplot(grid[row, 1]),
                    fig.add_subplot(policy_grid[0, 0])]
        frequencies = fig.add_subplot(policy_grid[1, 0], sharex=row_axes[2])
        draw_row(row_axes, case, limits)
        row_axes[2].set_xlabel('')
        row_axes[2].tick_params(labelbottom=False)
        draw_state_frequencies(frequencies, case)
        for ax, title in zip(row_axes, titles):
            ax.set_title((title+'\n' if not overview or case is cases[0] else '')
                         +f'ρ = {case["rho"]:g} | n = {case["n"]}', fontsize=10)
    handles = [Line2D([], [], color=color, ls=line, lw=2, label=label)
               for _sellers, _closure, label, color, line, _marker in base.DEFAULT_MODELS]
    handles.append(Patch(facecolor=SHADE, alpha=.55, label='Policies differ at saved nodes'))
    handles.extend(row_axes[0].scatter([], [], s=PROBABILITY_AREA_SCALE*probability,
                   color='#666666', alpha=.5, label=f'{100*probability:g}% of periods')
                   for probability in (.01, .05, .1))
    fig.legend(handles=handles, loc='outside lower center', ncol=4, frameon=False, fontsize=9)
    fig.suptitle('Successor-demand diagnostics | fixed p₀ = 0.8\n'
        +r'$x_s=\rho x+1-\rho,\quad x_f=\rho x$'
        +' | Linear interpolation of saved perceived demand\n'
        +'Circle area = stationary probability of own reputation x (calendar periods)', fontsize=12)
    return fig


def write_index(figures, data, cases, stability):
    relative_data = Path(os.path.relpath(data, figures)).as_posix()
    lines = ['# Demanda en estados sucesores: p₀ = 0.8', '',
        '[Vista conjunta](00_successor_demand_overview.png) · '
        '[PDF con una página por ρ](successor_demand_by_rho.pdf)', '',
        'Se reutilizan las demandas percibidas y políticas guardadas de monopolio '
        '(N=1) y OE marginal (N=2,3) en `high_p0_refinement_41`, sin resolver equilibrios. '
        'Los colores y estilos provienen de la comparación vigente de políticas.', '',
        'En los mismos nodos originales x de los tres modelos se calculan '
        '`x_s = rho*x + 1-rho` y `x_f = rho*x`; se interpola linealmente la demanda '
        'guardada de cada modelo en esos estados. La primera columna muestra '
        '`(d(x_s)+d(x_f))/2`: el nivel de oportunidades futuras con pesos iguales '
        'para éxito y fracaso. No pondera por las probabilidades de éxito del producto. '
        'La segunda muestra `d(x_s)-d(x_f)`: la ganancia de demanda por un éxito '
        'frente a un fracaso, en unidades de probabilidad y sin dividir por `1-rho`.', '',
        'El sombreado marca celdas alrededor de nodos donde la diferencia máxima '
        'entre probabilidades de producto 2 supera `1e-7`. Incluye políticas mixtas. '
        'Los bordes son puntos medios entre nodos vecinos, solo para visualizar: '
        'no son umbrales continuos estimados. La tercera columna muestra las políticas '
        'originales y cruces en los nodos con diferencias.', '',
        'Los círculos sobre las dos curvas de demanda y en las franjas inferiores '
        'marcan las frecuencias de la reputación propia. Su **área** es proporcional '
        'a la probabilidad estacionaria del nodo, con la misma escala para todos '
        'los modelos y rho. Las franjas separan N=1,2,3 para que las probabilidades '
        'se vean incluso si sus curvas se superponen. Se calcula '
        '`mu_N(x) = sum_{reputaciones rivales} joint_N(x, rivales)` a partir de '
        'la distribución conjunta guardada. Son porcentajes de todos los períodos '
        'de calendario, incluidos los períodos sin venta propia.', '',
        '| ρ | Grilla | Control | Figura | Nodos con diferencias |', '|---|---:|---:|---|---|']
    for case in cases:
        nodes = case['grid'][case['disagreement_mask']]
        label = ', '.join(f'{x:g}' for x in nodes) or 'Ninguno'
        name = f'rho_{case["rho"]:g}'.replace('.', 'p')
        lines.append(f'| {case["rho"]:g} | {case["n"]} | {case["coarse_n"]} | '
                     f'[PNG]({name}.png) | {label} |')
    lines += ['', 'Cada diagnóstico tiene su propio contraste de grillas: las demandas '
        'de la grilla de control se evalúan en los mismos estados sucesores de la '
        'grilla publicada. Si algún modelo cambia más de 0.05 en un diagnóstico, '
        'se omiten las tres curvas de ese diagnóstico para ese ρ. '
        'Este filtro es independiente de los filtros de mezcla de productos o beneficios.', '',
        '| Diagnóstico | Máximo cambio absoluto entre grillas |', '|---|---:|']
    for metric in METRICS:
        lines.append(f'| {metric} | {max(row[metric+"_max_abs_change"] for row in stability):.9f} |')
    state_retained = sum(case['state_frequency_included'] for case in cases)
    lines += ['', 'Las frecuencias tienen un control separado entre las grillas: '
        'distancia Wasserstein-1 entre las marginales discretas y cambio máximo '
        'de los cuantiles 10/25/50/75/90%, ambos con tolerancia 0.05 en unidades '
        'de reputación. Se ocultan los círculos de los tres modelos si alguno '
        'falla. No se interpolan masas nodales ni se afirma estabilidad puntual '
        'del modo o de cada probabilidad de nodo.', '',
        f'Frecuencias retenidas: {state_retained}/{len(cases)} valores de rho. '
        f'Máximo Wasserstein-1: {max(row["state_wasserstein1"] for row in stability):.6f}; '
        f'máximo cambio de cuantil: {max(row["state_max_quantile_change"] for row in stability):.6f}.', '']
    retained = sum(all(case['included'].values()) for case in cases)
    lines += ['', f'{retained} de {len(cases)} valores de ρ pasan ambos contrastes. La auditoría y los contrastes '
        'son verificaciones del modelo finito; no certifican umbrales continuos ni unicidad.', '',
        f'Datos: [todos los nodos]({relative_data}/successor_demand.csv) · '
        f'[resumen de frecuencias]({relative_data}/state_frequency_summary.csv) · '
        f'[contraste por modelo]({relative_data}/grid_stability.csv) · '
        f'[manifiesto y hashes]({relative_data}/manifest.json).', '',
        'Reproducción desde la raíz del repositorio:', '', '```bash',
        'MPLBACKEND=Agg MPLCONFIGDIR=/tmp/successor-demand-mpl .venv/bin/python -B '
        '-m oblivious.oe_three_sellers_fixed_p0.code.successor_demand '
        '--name successor_demand_p0_0p8_new', '```', '']
    path = figures/'README.md'
    path.write_text('\n'.join(lines))
    return path


def build_report(source=None, *, name=NAME):
    base.require(bool(name) and Path(name).name == name and name not in ('.', '..'),
                 'Output name must be a single directory name')
    source = Path(source or base.HOME/'data/high_p0_refinement_41').resolve()
    data, figures = base.HOME/'data'/name, base.HOME/'figures'/name
    base.require(not data.exists() and not figures.exists(), 'Choose new directories to preserve existing outputs')
    cases, records, stability, inputs, audit = prepare(source)
    sources = {base.location(path): base.digest(path) for path in
               (Path(__file__), Path(base.__file__), Path(expanded.__file__))}
    base.verify_inputs(inputs)
    data.mkdir(parents=True)
    figures.mkdir(parents=True)
    base.write_csv(data/'successor_demand.csv', records)
    base.write_csv(data/'grid_stability.csv', stability)
    state_summary = []
    for case in cases:
        for model, spec in zip(case['models'], base.DEFAULT_MODELS):
            mass = model['stationary']['mass']
            mode = int(np.argmax(mass))
            state_summary.append(dict(p0=P0, rho=case['rho'], n=case['n'], sellers=spec[0],
                included=case['state_frequency_included'],
                own_reputation_mean=float(case['grid'] @ mass),
                modal_grid_node=float(case['grid'][mode]), modal_node_probability=float(mass[mode]),
                q10=float(model['stationary']['quantiles'][0]),
                q90=float(model['stationary']['quantiles'][-1]),
                probability_at_policy_disagreement_nodes=float(mass[case['disagreement_mask']].sum())))
    base.write_csv(data/'state_frequency_summary.csv', state_summary)
    outputs = [data/'successor_demand.csv', data/'grid_stability.csv', data/'state_frequency_summary.csv']
    limits = diagnostic_limits(cases)
    with base.plt.rc_context({'font.size': 9, 'axes.titlesize': 10}):
        fig = make_figure(cases, limits)
        overview = figures/'00_successor_demand_overview.png'
        fig.savefig(overview, dpi=180, bbox_inches='tight')
        base.plt.close(fig)
        outputs.append(overview)
        pdf = figures/'successor_demand_by_rho.pdf'
        with PdfPages(pdf, metadata={'Title': 'Successor demand, fixed p0=0.8'}) as document:
            for case in cases:
                fig = make_figure([case], limits)
                path = figures/(f'rho_{case["rho"]:g}'.replace('.', 'p')+'.png')
                fig.savefig(path, dpi=180, bbox_inches='tight')
                document.savefig(fig, bbox_inches='tight')
                base.plt.close(fig)
                outputs.append(path)
        outputs.append(pdf)
    outputs.append(write_index(figures, data, cases, stability))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), status='complete',
        source_campaign=base.location(source), independent_audit=audit,
        equilibrium_solves=0, p0=P0, rhos=[case['rho'] for case in cases],
        models=[dict(sellers=m[0], closure=m[1], label=m[2], color=m[3], linestyle=m[4])
                for m in base.DEFAULT_MODELS],
        interpolation='Linear interpolation of each saved perceived-demand vector at rho*x+1-rho and rho*x',
        average_definition='Equal weights: (d(success)+d(failure))/2, not product-weighted expected demand',
        sensitivity_definition='d(success)-d(failure), in probability units, not a derivative',
        state_frequency_definition='Own calendar-time marginal of the saved joint law, summed over all rival coordinates; includes periods without own sales',
        circle_area_definition=f'{PROBABILITY_AREA_SCALE:g} points squared times stationary node probability; same scale across N and rho',
        state_grid_stability_limit=STATE_CHANGE_LIMIT,
        state_grid_stability_quantiles=STATE_QUANTILES.tolist(),
        state_grid_stability_definition='Require W1 and maximum quantile change <= limit for all three models; no pointwise node-mass or modal-location convergence claim',
        policy_disagreement_tolerance=POLICY_TOLERANCE,
        shading_definition='Midpoint cells around original nodes with max policy probability difference > tolerance; not continuous thresholds',
        grid_stability_limit=GRID_CHANGE_LIMIT,
        grid_stability_definition='Maximum absolute change on common fine-grid successor states; each metric requires all three models to pass',
        cases=[dict(rho=case['rho'], n=case['n'], coarse_n=case['coarse_n'], included=case['included'],
                    state_frequency_included=case['state_frequency_included'],
                    disagreement_cells=case['disagreement_cells']) for case in cases],
        source_input_sha256=inputs, reporting_source_sha256=sources,
        generated_files_sha256={base.location(path): base.digest(path) for path in outputs})
    base.verify_inputs(inputs)
    base.verify_inputs(sources)
    base.write_json(data/'manifest.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--name', default=NAME)
    args = parser.parse_args()
    result = build_report(args.source, name=args.name)
    print(json.dumps({key: result[key] for key in ('status', 'equilibrium_solves', 'p0', 'rhos', 'cases')}, indent=2))


if __name__ == '__main__':
    main()
