"""Buyer choice shares and finite-grid game dynamics from saved OE policies.

No equilibrium is solved here. Calendar-time transitions are exactly those of
the certified grid game, including its interpolation lotteries after a sale.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from . import report as base
from . import expanded_report as expanded
from .kernel import FixedOutsideKernel


SELLER_COLORS = ('#0072B2', '#D55E00', '#009E73')
MODEL_COLORS = SELLER_COLORS
OUTSIDE_COLOR = '#b8bec6'
LABELS = ('Outside', 'Seller A', 'Seller B', 'Seller C')


@dataclass
class SavedGame:
    kernel: FixedOutsideKernel
    policy: np.ndarray
    joint: np.ndarray
    configuration: dict
    summary: dict


def selection_probabilities(game, states):
    """Realized-state choice probabilities; columns outside, A, B, C."""
    states = np.asarray(states)
    kernel = game.kernel
    if (states.ndim != 2 or states.shape[1] != kernel.sellers
            or not np.issubdtype(states.dtype, np.integer)
            or np.any(states < 0) or np.any(states >= kernel.n)):
        raise ValueError('Require integer grid coordinates, shape (replicates,N)')
    coordinates = tuple(states.T)
    probabilities = np.column_stack([kernel.outside[coordinates]] + [
        kernel.demand[(seller,) + coordinates] for seller in range(kernel.sellers)])
    return probabilities


def simulate_step(game, states, uniforms):
    """One calendar period, using four independent uniforms per replicate.

    Choices are 0=outside, 1=A, etc. Products are 0=outside, 1/2 otherwise;
    successes are -1=outside, 0=failure, 1=success.
    """
    probabilities = selection_probabilities(game, states)
    states = np.asarray(states)
    uniforms = np.asarray(uniforms, dtype=float)
    if uniforms.shape != (len(states), 4) or np.any(~np.isfinite(uniforms)) or np.any(
            (uniforms < 0) | (uniforms >= 1)):
        raise ValueError('Require four finite independent uniforms in [0,1) per replicate')
    cdf = np.cumsum(probabilities, axis=1)
    cdf[:, -1] = 1.
    choices = np.sum(uniforms[:, :1] >= cdf, axis=1)
    products = np.zeros(len(states), dtype=np.int8)
    successes = np.full(len(states), -1, dtype=np.int8)
    next_states = states.copy()
    kernel = game.kernel
    for seller in range(kernel.sellers):
        selected = np.flatnonzero(choices == seller + 1)
        own = states[selected, seller]
        product2 = uniforms[selected, 1] < game.policy[own]
        products[selected] = 1 + product2
        quality = np.where(product2, kernel.params.p2, kernel.params.p1)
        successes[selected] = uniforms[selected, 2] < quality
        for success, operator in ((0, kernel.failure), (1, kernel.success)):
            events = selected[successes[selected] == success]
            source = states[events, seller]
            # Both interpolation neighbors are stored, including zero weights.
            destinations = operator.indices.reshape(kernel.n, 2)[source]
            weights = operator.data.reshape(kernel.n, 2)[source]
            neighbor = (uniforms[events, 3] >= weights[:, 0]).astype(int)
            next_states[events, seller] = destinations[np.arange(len(events)), neighbor]
    return next_states, choices, products, successes


def simulate_path(game, periods, seed):
    if int(periods) != periods or periods < 1:
        raise ValueError('Use a positive integer horizon')
    kernel = game.kernel
    rng = np.random.default_rng(seed)
    states = np.full((periods + 1, kernel.sellers), kernel.n // 2, dtype=int)
    choices = np.empty(periods, dtype=int)
    products = np.empty(periods, dtype=np.int8)
    successes = np.empty(periods, dtype=np.int8)
    for t in range(periods):
        following, choice, product, success = simulate_step(
            game, states[t:t+1], rng.random((1, 4)))
        states[t+1] = following[0]
        choices[t], products[t], successes[t] = choice[0], product[0], success[0]
    return dict(states=states, choices=choices, products=products, successes=successes)


def exact_transient(game, periods):
    """Forward-propagate the full joint law from x_A=x_B=x_C=.5.

    Choices and mean reputation are measured BEFORE each period's transition.
    Expected cumulative own sales include the current period's choice.
    """
    if int(periods) != periods or periods < 1:
        raise ValueError('Use a positive integer horizon')
    kernel = game.kernel
    transition = kernel.transition(game.policy)
    row_error = float(np.max(np.abs(np.asarray(transition.sum(axis=1)).ravel()-1)))
    base.require(row_error < 1e-10 and np.all(transition.data >= 0), 'Invalid calendar transition')
    choice_rewards = np.vstack([kernel.outside.ravel(), kernel.demand.reshape(kernel.sellers, -1)])
    coordinates = np.indices(kernel.shape).reshape(kernel.sellers, -1)
    reputation_reward = kernel.grid[coordinates].mean(axis=0)
    law = np.zeros(kernel.size)
    initial = np.ravel_multi_index((kernel.n//2,) * kernel.sellers, kernel.shape)
    law[initial] = 1.
    choices = np.empty((periods, kernel.sellers+1))
    mean_reputation = np.empty(periods)
    maximum_mass_error, minimum_mass = 0., 0.
    forward = transition.T.tocsr()
    for t in range(periods):
        choices[t] = choice_rewards @ law
        mean_reputation[t] = reputation_reward @ law
        law = forward @ law
        maximum_mass_error = max(maximum_mass_error, abs(float(law.sum())-1))
        minimum_mass = min(minimum_mass, float(law.min()))
    base.require(maximum_mass_error < 1e-9 and minimum_mass >= -1e-15,
                 'Transient law lost probability mass or positivity')
    symmetry_error = float(np.max(np.ptp(choices[:, 1:], axis=1)))
    base.require(symmetry_error < 1e-10, 'Symmetric initial condition lost seller symmetry')
    return dict(choices=choices, mean_reputation=mean_reputation,
        cumulative_sales_per_seller=np.cumsum(choices[:, 1:].mean(axis=1)), final_law=law,
        maximum_mass_error=maximum_mass_error, minimum_mass=minimum_mass,
        row_sum_error=row_error, seller_symmetry_error=symmetry_error)


def load_saved_game(data, p0, rho, n, sellers):
    closure = 'monopoly' if sellers == 1 else 'marginal'
    stem = Path(data)/expanded.case_key(p0, rho)/f'n{n}_N{sellers}_{closure}'
    record = json.loads(stem.with_suffix('.json').read_text())
    checkpoint = stem.with_suffix('.npz')
    base.require(base.digest(checkpoint) == record['checkpoint_sha256'], 'Checkpoint hash differs')
    config = record['configuration']
    base.require((config['n'], config['sellers'], config['p0'], config['parameters']['rho'])
                 == (n, sellers, p0, rho) and record['summary']['verified'], 'Wrong or uncertified saved game')
    with np.load(checkpoint) as arrays:
        grid, policy, joint = [arrays[key].copy() for key in ('grid', 'policy', 'joint')]
    kernel = FixedOutsideKernel(grid, Parameters(**config['parameters']), p0, sellers, config['order'])
    game = SavedGame(kernel, policy, joint, config, record['summary'])
    shares = np.asarray([np.sum(joint*kernel.outside)] + [
        np.sum(joint*kernel.demand[seller]) for seller in range(sellers)])
    expected = np.asarray([game.summary['outside_rate']] + [game.summary['sale_rate']]*sellers)
    share_error = float(np.max(abs(shares-expected)))
    base.require(share_error < 1e-9, 'Stationary event shares differ from saved summary')
    residual = float(np.max(abs(kernel.transition(policy).T @ joint.ravel()-joint.ravel())))
    base.require(residual < 1e-9, 'Saved joint law is not invariant under reconstructed transition')
    return game, dict(stationary_choice_max_error=share_error, stationary_residual=residual,
                     configuration=config, checkpoint_sha256=record['checkpoint_sha256'])


def stationary_share_rows(rows, cases):
    records = []
    for row in rows:
        case = cases[expanded.case_key(row['p0'], row['rho'])]
        if row['n'] != case['displayed_n']:
            continue
        shares = [row['sale_rate'] if index < row['sellers'] else 0. for index in range(3)]
        base.require(abs(sum(shares)+row['outside_rate']-1) < 1e-10, 'Choice shares must sum to one')
        records.append(dict(p0=row['p0'], rho=row['rho'], n=row['n'], sellers=row['sellers'],
            included=case['metrics']['outside_rate']['passed'], seller_A=shares[0], seller_B=shares[1],
            seller_C=shares[2], outside=row['outside_rate'], coarse_n=case['validation_coarse_n'],
            maximum_grid_change=case['metrics']['outside_rate']['max_change']))
    return records


def plot_stationary(records, figures, p0s, rhos):
    fig, axes = base.plt.subplots(2, 2, figsize=(14, 8), sharey=True, layout='constrained')
    lookup = {(row['p0'], row['rho'], row['sellers']): row for row in records}
    colors, fields = (*SELLER_COLORS, OUTSIDE_COLOR), ('seller_A', 'seller_B', 'seller_C', 'outside')
    for ax, rho in zip(axes.ravel(), rhos):
        for j, p0 in enumerate(p0s):
            for sellers in (1, 2, 3):
                x = j*4+sellers-2
                row = lookup[p0, rho, sellers]
                if not row['included']:
                    ax.text(x, 4, '×', ha='center', color='#777777', fontsize=15)
                    continue
                bottom = 0.
                for field, color in zip(fields, colors):
                    height = 100*row[field]
                    ax.bar(x, height, bottom=bottom, width=.8, color=color,
                           edgecolor='white', linewidth=.35)
                    bottom += height
        ax.set(title=f'ρ = {rho:g}', ylim=(0, 101), ylabel='Elecciones (% de períodos)',
               xticks=np.arange(len(p0s))*4, xticklabels=[f'{p0:g}' for p0 in p0s], xlabel='Outside fijo p₀')
        ax.text(.01, .98, 'En cada grupo: N = 1, 2, 3 (izq. → der.)', transform=ax.transAxes,
                va='top', fontsize=8, bbox=dict(facecolor='white', alpha=.85, edgecolor='none'))
        ax.set_axisbelow(True)
        ax.grid(axis='y', alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('¿A quién elige el comprador? Frecuencias estacionarias\n'
                 'Cada barra suma 100%: seller A + B + C + outside', fontsize=13)
    fig.legend([Patch(color=color) for color in colors], (*LABELS[1:], LABELS[0]),
               loc='outside lower center', ncol=4, frameon=False)
    return base.save(fig, figures, '01_stationary_buyer_choices')


def plot_transients(results, figures, p0s, rho, periods):
    fig, axes = base.plt.subplots(len(p0s), 3, figsize=(14, 4*len(p0s)), layout='constrained')
    axes = np.atleast_2d(axes)
    time = np.arange(1, periods+1)
    for row_index, p0 in enumerate(p0s):
        for sellers, color in zip((1, 2, 3), MODEL_COLORS):
            game, transient = results[p0, sellers]
            label = 'Monopolio' if sellers == 1 else f'{sellers} sellers (OE marginal)'
            axes[row_index, 0].plot(time, 100*transient['choices'][:, 1:].mean(axis=1), color=color, label=label)
            axes[row_index, 0].axhline(100*game.summary['sale_rate'], color=color, ls='--', lw=1, alpha=.8)
            axes[row_index, 1].plot(time, transient['mean_reputation'], color=color, label=label)
            axes[row_index, 2].plot(time, transient['cumulative_sales_per_seller'], color=color, label=label)
        axes[row_index, 0].set(title=f'p₀ = {p0:g} | Probabilidad de elegir a cada seller',
                              ylabel='% de períodos')
        axes[row_index, 1].set(title='Reputación propia esperada', ylabel='Reputación', ylim=(0, 1))
        axes[row_index, 2].set(title='Ventas acumuladas esperadas por seller', ylabel='Número de ventas')
        for ax in axes[row_index]:
            ax.set(xlabel='Período', xlim=(1, periods))
            base.style(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.append(Line2D([], [], color='#555555', ls='--'))
    labels.append('Referencia estacionaria (primer panel)')
    fig.legend(handles, labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle(f'Evolución promedio exacta desde reputaciones 0.5 | ρ = {rho:g}\n'
                 'Políticas fijas; expectativas sobre la ley conjunta del juego en la grilla', fontsize=12)
    return base.save(fig, figures, '02_expected_evolution')


def plot_paths(games_paths, figures, p0, rho, periods):
    fig, axes = base.plt.subplots(3, 3, figsize=(15, 9), sharex=True, layout='constrained')
    time, state_time = np.arange(1, periods+1), np.arange(periods+1)
    for row, sellers in enumerate((1, 2, 3)):
        game, path, seed = games_paths[sellers]
        for seller, color in zip(range(sellers), SELLER_COLORS):
            label = f'Seller {"ABC"[seller]}'
            axes[row, 0].step(state_time, game.kernel.grid[path['states'][:, seller]],
                              where='post', color=color, lw=1, label=label)
            counts = np.cumsum(path['choices'] == seller+1)
            axes[row, 2].plot(time, counts, color=color, label=f'{label}: {counts[-1]}')
        outside = path['choices'] == 0
        axes[row, 1].scatter(time[outside], np.zeros(outside.sum()), c=OUTSIDE_COLOR, s=5, alpha=.55)
        for seller, color in zip(range(sellers), SELLER_COLORS):
            for product, marker in ((1, 'o'), (2, '^')):
                selected = (path['choices'] == seller+1) & (path['products'] == product)
                axes[row, 1].scatter(time[selected], np.full(selected.sum(), seller+1),
                                     c=color, s=14, marker=marker)
        name = 'Monopolio' if sellers == 1 else f'{sellers} sellers (OE marginal)'
        axes[row, 0].set(title=f'{name} | n = {game.kernel.n} | semilla {seed}',
                         ylabel='Reputación', ylim=(-.02, 1.02))
        axes[row, 1].set(title=f'Elección del comprador | outside: {outside.sum()}',
                         yticks=range(sellers+1), yticklabels=LABELS[:sellers+1], ylim=(-.4, sellers+.5))
        axes[row, 2].set(title='Ventas acumuladas de cada seller', ylabel='Número de ventas')
        axes[row, 2].legend(loc='upper left', frameon=False, fontsize=8)
        for ax in axes[row]:
            ax.set(xlim=(0, periods), xlabel='Período')
            base.style(ax)
            ax.tick_params(labelbottom=True)
    handles = [Patch(color=color, label=label) for color, label in zip(
        (*SELLER_COLORS, OUTSIDE_COLOR), (*LABELS[1:], LABELS[0]))]
    handles.extend([Line2D([], [], ls='', marker=marker, color='#444444', label=f'Producto {product}')
                    for product, marker in ((1, 'o'), (2, '^'))])
    fig.legend(handles=handles, loc='outside lower center', ncol=6, frameon=False)
    fig.suptitle(f'Una trayectoria aleatoria del juego | p₀ = {p0:g} | ρ = {rho:g}\n'
                 'Inicio: todas las reputaciones en 0.5. Solo actualiza el seller que vende; outside congela todo.',
                 fontsize=12)
    return base.save(fig, figures, '03_sample_path_p0_'+str(p0).replace('.', 'p'))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='high_p0_refinement_41')
    parser.add_argument('--name', default='choice_dynamics_41')
    parser.add_argument('--p0s', nargs='+', type=float, default=[.5, .8])
    parser.add_argument('--rho', type=float, default=.6)
    parser.add_argument('--periods', type=int, default=600)
    parser.add_argument('--seed', type=int, default=20261025)
    args = parser.parse_args(argv)
    base.require(args.periods > 0 and len(args.p0s) > 0, 'Use a positive horizon and at least one p0')
    for name in (args.source, args.name):
        base.require(Path(name).name == name and name not in ('.', '..'), 'Use a campaign folder name')
    data, figures = base.HOME/'data'/args.name, base.HOME/'figures'/args.name
    base.require(not data.exists() and not figures.exists(), 'Use a new output name; historical outputs are preserved')
    source = base.HOME/'data'/args.source
    manifest, rows, policies, inputs, p0s, rhos, models = base.read_inputs(source)
    base.require(manifest['status'] == 'complete', 'Source campaign is incomplete')
    report = json.loads((source/'report_manifest.json').read_text())
    base.require(report['verified'], 'Source report is not verified')
    base.verify_inputs(report['source_input_sha256'])
    base.verify_inputs(report['generated_data_sha256'])
    base.verify_inputs(report['reporting_source_sha256'])
    audit, audit_inputs = base.read_optional_audit(source, manifest)
    base.require(audit is not None, 'An independent source audit is required')
    inputs.update(audit_inputs)
    for filename in ('report_manifest.json', 'metric_retention.json'):
        inputs[base.location(source/filename)] = base.digest(source/filename)
    cases, comparisons = expanded.select_cases(rows, manifest, p0s, rhos)
    base.require(cases == report['case_grids'], 'Saved report grid decisions differ')
    for p0 in args.p0s:
        case = cases.get(expanded.case_key(p0, args.rho))
        base.require(case is not None and all(metric['passed'] for metric in case['metrics'].values()),
                     'Choose an illustrative case passing all existing grid filters')
    data.mkdir(parents=True)
    figures.mkdir(parents=True)
    records = stationary_share_rows(rows, cases)
    base.write_csv(data/'stationary_buyer_choices.csv', records)
    images = plot_stationary(records, figures, p0s, rhos)
    results, validation, path_counts, generated = {}, {}, [], [data/'stationary_buyer_choices.csv']
    for p0 in args.p0s:
        paths = {}
        for sellers in (1, 2, 3):
            n = cases[expanded.case_key(p0, args.rho)]['displayed_n']
            print(f'p0={p0:g}, rho={args.rho:g}, N={sellers}, n={n}: reconstructing saved game', flush=True)
            game, checks = load_saved_game(source, p0, args.rho, n, sellers)
            transient = exact_transient(game, args.periods)
            seed = args.seed + int(round(1000*p0))*10 + sellers
            path = simulate_path(game, args.periods, seed)
            paths[sellers] = game, path, seed
            results[p0, sellers] = game, transient
            stem = f'p0_{p0:g}_rho_{args.rho:g}_N{sellers}'.replace('.', 'p')
            raw = data/(stem+'.npz')
            np.savez_compressed(raw, **path, grid=game.kernel.grid,
                expected_choices=transient['choices'], expected_mean_reputation=transient['mean_reputation'],
                expected_cumulative_sales_per_seller=transient['cumulative_sales_per_seller'],
                final_law=transient['final_law'], seed=seed)
            generated.append(raw)
            series = []
            for t in range(args.periods):
                series.append(dict(period=t+1, choice=LABELS[path['choices'][t]],
                    product=int(path['products'][t]), success=int(path['successes'][t]),
                    **{f'reputation_{"ABC"[i]}_before': float(game.kernel.grid[path['states'][t, i]]) for i in range(sellers)},
                    **{f'reputation_{"ABC"[i]}_after': float(game.kernel.grid[path['states'][t+1, i]]) for i in range(sellers)},
                    **{f'expected_choice_{LABELS[i].replace(" ", "_")}': float(transient['choices'][t, i]) for i in range(sellers+1)},
                    expected_reputation=float(transient['mean_reputation'][t]),
                    expected_cumulative_sales_per_seller=float(transient['cumulative_sales_per_seller'][t])))
            csv_path = data/(stem+'.csv')
            base.write_csv(csv_path, series)
            generated.append(csv_path)
            counts = np.bincount(path['choices'], minlength=sellers+1)
            base.require(counts.sum() == args.periods, 'Path choices must partition the horizon')
            path_counts.append(dict(p0=p0, rho=args.rho, n=n, sellers=sellers, periods=args.periods, seed=seed,
                outside=int(counts[0]), seller_A=int(counts[1]), seller_B=int(counts[2]) if sellers > 1 else 0,
                seller_C=int(counts[3]) if sellers > 2 else 0,
                expected_sales_per_seller=float(transient['cumulative_sales_per_seller'][-1])))
            checks.update({key: transient[key] for key in ('maximum_mass_error', 'minimum_mass', 'row_sum_error', 'seller_symmetry_error')})
            validation[stem] = checks
            print(f'  finished; choices={counts.tolist()}, law mass error={transient["maximum_mass_error"]:.3g}', flush=True)
        images.extend(plot_paths(paths, figures, p0, args.rho, args.periods))
    images.extend(plot_transients(results, figures, args.p0s, args.rho, args.periods))
    count_file = data/'sample_path_counts.csv'
    base.write_csv(count_file, path_counts)
    generated.append(count_file)
    links = '\n'.join(f'- [{path.stem}]({path.name})' for path in sorted(images) if path.suffix == '.png')
    index = figures/'README.md'
    index.write_text(f'''# Elecciones del comprador y evolución del juego

Fuente: `{args.source}`. No se vuelve a resolver ningún equilibrio.

{links}

## Cómo leer las figuras

- **01:** porcentajes estacionarios de todos los períodos, incluyendo outside.
  Cada barra corresponde a un mercado N=1,2,3 y suma 100%. A/B/C son identidades
  simétricas: en promedio estacionario cada una tiene la misma frecuencia.
  Se conservan 23/24 pares p0/rho con el filtro de outside (máximo 5 puntos
  porcentuales entre las dos grillas comunes más finas). Se omite p0=.7,rho=.9.
- **02:** expectativas exactas desde todas las reputaciones en .5, propagando
  la ley conjunta completa; no son promedios de una sola trayectoria. El primer
  panel mide la probabilidad de elegir a **cada seller**, por período; las
  referencias discontinuas son estacionarias y no presuponen que se alcancen
  dentro del horizonte. El tercer panel cuenta ventas esperadas por seller.
- **03:** una realización aleatoria de {args.periods} períodos por modelo.
  Izquierda: reputaciones. Centro: a quién eligen y producto entregado
  (círculo=1, triángulo=2). Derecha: ventas acumuladas por identidad.
  Las diferencias entre A/B/C dentro de una realización son aleatorias.

Las políticas OE se mantienen fijas. El comprador elige según las demandas
del estado conjunto real, después el seller elegido sortea el producto según
su política y un éxito/fracaso según su producto. Solo cambia su reputación;
outside congela todo. La actualización sortea los dos vecinos de la grilla
con los mismos pesos de interpolación del kernel certificado, sin redondeo
determinista ni aproximación por reputaciones independientes.

Los p0=.5 y .8, rho=.6 pasan los cuatro filtros estacionarios existentes.
Sus transitorios se muestran en las grillas certificadas n=31 y n=41,
respectivamente; no se ha certificado estabilidad entre grillas del transitorio
ni convergencia al continuo. No hay adaptación o aprendizaje de las políticas.

Datos reproducibles, semillas, cuentas y controles: `../../data/{args.name}`.
Las figuras y checkpoints anteriores permanecen intactos.

```bash
MPLCONFIGDIR=/tmp/choice-dynamics-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B -m oblivious.oe_three_sellers_fixed_p0.code.choice_dynamics --name {args.name} --periods {args.periods} --seed {args.seed}
```

Usar otro `--name` para una ejecución adicional, preservando esta publicación.
''')
    generated.append(index)
    base.verify_inputs(inputs)
    output = dict(status='complete_verified', created_utc=datetime.now(timezone.utc).isoformat(),
        configuration=vars(args), source_input_sha256=inputs,
        reporting_source_sha256={base.location(__file__): base.digest(__file__),
            base.location(base.__file__): base.digest(base.__file__),
            base.location(expanded.__file__): base.digest(expanded.__file__)},
        scope='Exact finite-grid joint-law transients and reproducible sample paths under fixed saved OE policies',
        stationary_choice_cases_retained=sum(1 for case in cases.values() if case['metrics']['outside_rate']['passed']),
        stationary_choice_cases_total=len(cases), validation=validation,
        sample_path_counts=path_counts,
        generated_sha256={base.location(path): base.digest(path) for path in generated+images})
    base.write_json(data/'manifest.json', output)
    print(f'Complete: {len(images)//2} figures, {len(validation)} saved games; {figures}', flush=True)
    return output


if __name__ == '__main__':
    main()
