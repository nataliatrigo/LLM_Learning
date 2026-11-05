"""Plot the saved calendar-time marginal of the fixed-p0 marginal OE."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter
import numpy as np

from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    HOME, ROOT, check_bundle,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p0', type=float, default=.8)
    parser.add_argument('--rho', type=float, default=.9)
    parser.add_argument('--n', type=int, default=201)
    args = parser.parse_args()
    p0_tag = 'p0_' + str(float(args.p0)).replace('.', 'p')
    rho_tag = 'rho_' + str(float(args.rho)).replace('.', 'p')
    source = HOME / 'data/p0_sweep_201' / p0_tag / rho_tag
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    tag = f'{rho_tag}_n{args.n:03d}_marginal'
    if tag not in manifest['completed']:
        raise ValueError(f'No completed marginal checkpoint: {tag}')
    case = check_bundle(source, tag, manifest)
    if (case['p0'], case['rho'], case['n']) != (args.p0, args.rho, args.n):
        raise ValueError('Checkpoint parameters differ from the requested plot')
    solution = case['competition']
    if not solution['verified'] or not case['demand_audit']['verified']:
        raise ValueError('Saved marginal OE or demand audit is not verified')
    x, mu, nu = (np.asarray(solution[key]) for key in ('grid', 'mu', 'nu'))
    if (x.shape != (args.n,) or mu.shape != x.shape
            or nu.shape != (args.n, args.n)
            or not np.array_equal(x, np.linspace(0, 1, args.n))
            or not np.isfinite(mu).all() or np.any(mu < 0)
            or not np.isclose(mu.sum(), 1, rtol=0, atol=1e-12)
            or not np.allclose(mu, nu.sum(axis=0), rtol=0, atol=1e-12)
            or not np.allclose(mu, nu.sum(axis=1), rtol=0, atol=1e-12)):
        raise ValueError('Invalid calendar-time marginal probability masses')

    figures = HOME / 'figures/marginal_distributions' / p0_tag / rho_tag
    exports = HOME / 'data/marginal_distributions' / p0_tag / rho_tag
    figures.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    stem = f'stationary_marginal_n{args.n:03d}'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(9.4, 5.6), layout='constrained')
    ax.fill_between(x, mu, color='#D55E00', alpha=.12)
    ax.plot(x, mu, color='#D55E00', linewidth=2.1, label='Marginal OE')
    ax.set(xlim=(0, 1), ylim=(0, mu.max() * 1.13),
           xlabel=r'Reputation $x$',
           ylabel='Stationary probability per grid node',
           title='Marginal stationary distribution\n'
                 + rf'$p_0 = {args.p0:g}$   |   $\rho = {args.rho:g}$   |   $n = {args.n}$')
    ax.xaxis.set_major_locator(MultipleLocator(.1))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
    ax.grid(axis='y', color='#D5D9DF', linewidth=.7, alpha=.65)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', frameon=False)
    ax.text(.98, .79, 'Calendar-time distribution\n'
            + r'$\mu_A(x)=\sum_y\nu(x,y)$',
            transform=ax.transAxes, ha='right', va='top', color='#4B5563',
            fontsize=11, linespacing=1.7)
    for extension in ('png', 'svg'):
        fig.savefig(figures / f'{stem}.{extension}', dpi=220, facecolor='white')
    plt.close(fig)

    csv_path = exports / f'{stem}.csv'
    np.savetxt(csv_path, np.column_stack((x, mu)), delimiter=',',
               header='reputation,stationary_probability', comments='', fmt='%.17g')
    provenance = dict(
        p0=args.p0, rho=args.rho, n=args.n, model='competition', closure='marginal',
        clock='calendar time', quantity='Probability mass at each saved grid node',
        normalization=float(mu.sum()), mean_reputation=float(x @ mu),
        source_manifest=str(manifest_path.relative_to(ROOT)),
        source_manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        checkpoint_sha256={name: manifest['checkpoint_sha256'][name]
                           for name in (tag + '.json', tag + '.npz')},
        verified=bool(solution['verified']),
        stationary_residual=solution['summary']['stationary_residual'],
        representation='Unsmoothed saved node masses joined by straight lines',
        csv=str(csv_path.relative_to(ROOT)),
        figures=[str((figures / f'{stem}.{ext}').relative_to(ROOT))
                 for ext in ('png', 'svg')])
    (exports / f'{stem}.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance, indent=2))


if __name__ == '__main__':
    main()
