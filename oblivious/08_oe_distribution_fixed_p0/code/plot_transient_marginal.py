"""Propagate a point initial state under a saved marginal OE calendar kernel."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np
from scipy.sparse.csgraph import connected_components

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code.model import OutsideOptionGame
from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    HOME, ROOT, check_bundle,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p0', type=float, default=.8)
    parser.add_argument('--rho', type=float, default=.9)
    parser.add_argument('--n', type=int, default=201)
    parser.add_argument('--initial', type=float, default=.9,
                        help='Common initial reputation of both sellers')
    parser.add_argument('--periods', type=int, nargs='+', default=[100, 1000, 10000])
    args = parser.parse_args()
    periods = sorted(set(args.periods))
    if not periods or periods[0] < 1:
        parser.error('Periods must be positive integers')
    p0_tag = 'p0_' + str(float(args.p0)).replace('.', 'p')
    rho_tag = 'rho_' + str(float(args.rho)).replace('.', 'p')
    initial_tag = 'x0_' + str(float(args.initial)).replace('.', 'p')
    source = HOME / 'data/p0_sweep_201' / p0_tag / rho_tag
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    tag = f'{rho_tag}_n{args.n:03d}_marginal'
    if tag not in manifest['completed']:
        raise ValueError('Requested marginal checkpoint is not complete')
    case = check_bundle(source, tag, manifest)
    if (case['p0'], case['rho'], case['n']) != (args.p0, args.rho, args.n):
        raise ValueError('Checkpoint parameters differ from the requested plot')
    solution = case['competition']
    if not solution['verified'] or not case['demand_audit']['verified']:
        raise ValueError('Saved policy or demand audit is unverified')
    grid, policy, stationary, joint = (
        np.asarray(solution[key]) for key in ('grid', 'policy', 'mu', 'nu'))
    matches = np.flatnonzero(np.isclose(grid, args.initial, rtol=0, atol=1e-14))
    if len(matches) != 1:
        raise ValueError('Initial state must be an exact saved grid node')
    n = len(grid)
    game = OutsideOptionGame(grid, Parameters(**case['configuration']['parameters']),
                             p0=args.p0, order=case['configuration']['order'])
    alpha_a = np.broadcast_to(policy[:, None], (n, n))
    transition = game.transition(alpha_a, alpha_a.T)
    transition.eliminate_zeros()
    row_error = float(np.max(abs(np.asarray(transition.sum(axis=1)).ravel() - 1)))
    forward = transition.T.tocsr()
    residual = float(np.abs(forward @ joint.ravel() - joint.ravel()).sum())
    components = int(connected_components(transition, directed=True, connection='strong',
                                          return_labels=False))
    positive_diagonal = bool(np.all(transition.diagonal() > 0))
    if (row_error > 1e-12 or np.any(transition.data < 0)
            or residual > 1e-10
            or not np.allclose(joint.sum(axis=1), stationary, rtol=0, atol=1e-12)):
        raise ArithmeticError('Reconstructed kernel fails the saved stationary-law check')
    law = np.zeros(n * n)
    index = int(matches[0])
    law[index * n + index] = 1
    snapshots = {}
    summaries = []
    max_mass_error = 0.
    for date in range(1, periods[-1] + 1):
        law = forward @ law
        if date in periods:
            total = float(law.sum())
            max_mass_error = max(max_mass_error, abs(total - 1))
            joint_t = law.reshape(n, n)
            mu = joint_t.sum(axis=1)
            symmetry_error = float(np.abs(mu - joint_t.sum(axis=0)).sum())
            if abs(total - 1) > 1e-9 or law.min() < 0 or symmetry_error > 1e-10:
                raise ArithmeticError('Transient probability mass or symmetry check failed')
            snapshots[date] = mu.copy()
            summaries.append(dict(period=date, probability_sum=total,
                                  mean_reputation=float(grid @ mu),
                                  marginal_tv_to_stationary=float(.5 * abs(mu - stationary).sum()),
                                  seller_symmetry_l1=symmetry_error))

    figures = HOME / 'figures/marginal_distributions' / p0_tag / rho_tag
    exports = HOME / 'data/marginal_distributions' / p0_tag / rho_tag
    figures.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    time_tag = '_t_' + '_'.join(map(str, periods))
    stem = f'transient_marginal_n{n:03d}_{initial_tag}{time_tag}'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(10.2, 6), layout='constrained')
    colors = ['#56B4E9', '#009E73', '#0072B2', '#CC79A7']
    for k, (date, mu) in enumerate(snapshots.items()):
        ax.plot(grid, mu, color=colors[k % len(colors)], linewidth=1.9,
                label=f't = {date:,}')
    ax.plot(grid, stationary, color='#D55E00', linestyle='--', linewidth=2.3,
            label='Stationary')
    ax.set(xlim=(0, 1), ylim=(0, 1.12 * max(stationary.max(),
                                           *(mu.max() for mu in snapshots.values()))),
           xlabel=r'Reputation $x$', ylabel='Probability per grid node',
           title='Marginal distribution from a common initial reputation\n'
                 + rf'$x_{{A,0}}=x_{{B,0}}={args.initial:g}$   |   '
                 + rf'$p_0={args.p0:g}$   |   $\rho={args.rho:g}$   |   $n={n}$')
    ax.xaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis='y', color='#D5D9DF', linewidth=.7, alpha=.65)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', title='Calendar periods', frameon=False)
    fig.supxlabel('Same saved marginal OE policy at every date', fontsize=10, color='#4B5563')
    for extension in ('png', 'svg'):
        fig.savefig(figures / f'{stem}.{extension}', dpi=220, facecolor='white')
    plt.close(fig)
    csv_path = exports / f'{stem}.csv'
    np.savetxt(csv_path, np.column_stack((grid, *snapshots.values(), stationary)),
               delimiter=',', header=','.join(['reputation']
                   + [f'probability_t_{date}' for date in snapshots] + ['stationary_probability']),
               comments='', fmt='%.17g')
    report = dict(p0=args.p0, rho=args.rho, n=n,
                  initial_state=[args.initial, args.initial],
                  model='competition', closure='marginal', clock='calendar time',
                  method='Deterministic joint-law propagation: nu_t = (P.T)^t nu_0',
                  scope='Saved finite-grid policy held fixed; no new equilibrium solved',
                  source_manifest=str(manifest_path.relative_to(ROOT)),
                  source_manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
                  checkpoint_sha256={name: manifest['checkpoint_sha256'][name]
                                     for name in (tag + '.json', tag + '.npz')},
                  row_sum_error=row_error, stationary_joint_l1_residual=residual,
                  strongly_connected_components=components,
                  all_diagonal_entries_positive=positive_diagonal,
                  unique_stationary_and_converges=components == 1 and positive_diagonal,
                  max_snapshot_probability_sum_error=max_mass_error,
                  stationary_mean_reputation=float(grid @ stationary), snapshots=summaries,
                  script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  csv=str(csv_path.relative_to(ROOT)),
                  figures=[str((figures / f'{stem}.{ext}').relative_to(ROOT))
                           for ext in ('png', 'svg')])
    (exports / f'{stem}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
