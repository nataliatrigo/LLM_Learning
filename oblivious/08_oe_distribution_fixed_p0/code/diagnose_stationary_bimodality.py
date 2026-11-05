"""Compare stationary marginal laws at calendar dates and own-sale dates."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, PercentFormatter
import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_following_monopoly.code.event_chain import purchase_chain_statistics
from oblivious.oe_distribution_fixed_p0.code.model import outside_demand_matrix
from oblivious.oe_distribution_fixed_p0.code.run_experiment import HOME, ROOT, check_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=201)
    args = parser.parse_args()
    p0, rho = .8, .9
    source = HOME / 'data/p0_sweep_201/p0_0p8/rho_0p9'
    manifest_path = source / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    tag = f'rho_0p9_n{args.n:03d}_marginal'
    if tag not in manifest['completed']:
        raise ValueError('Requested checkpoint is not complete')
    case = check_bundle(source, tag, manifest)
    if (case['p0'], case['rho'], case['n']) != (p0, rho, args.n):
        raise ValueError('Checkpoint parameters differ')
    solution = case['competition']
    if not solution['verified'] or not case['demand_audit']['verified']:
        raise ValueError('Checkpoint policy or demand is not verified')
    x, alpha, mu, nu = (np.asarray(solution[key]) for key in ('grid', 'policy', 'mu', 'nu'))
    params = Parameters(**case['configuration']['parameters'])
    demand, _ = outside_demand_matrix(x, params, p0, case['configuration']['order'])
    sale_flow = (nu * demand).sum(axis=1)
    sale_rate = float(sale_flow.sum())
    sale_law = sale_flow / sale_rate
    effective_selection = sale_flow / mu
    embedded = purchase_chain_statistics(x, alpha, params)
    event_l1 = float(abs(sale_law - embedded['sale_stationary']).sum())
    if (event_l1 > 1e-9 or not np.isclose(mu.sum(), 1, atol=1e-12, rtol=0)
            or not np.allclose(mu, nu.sum(axis=1), atol=1e-12, rtol=0)):
        raise ArithmeticError('Stationary calendar/sale law identity failed')
    switch_indices = np.flatnonzero(np.diff(alpha) != 0)
    lower_switch = float(x[switch_indices[0] + 1])
    high = x >= lower_switch
    calendar_high = float(mu[high].sum())
    sale_high = float(sale_law[high].sum())
    drift = (1 - rho) * (params.p1 + (params.p2 - params.p1) * alpha - x)

    figures = HOME / 'figures/marginal_distributions/p0_0p8/rho_0p9'
    exports = HOME / 'data/marginal_distributions/p0_0p8/rho_0p9'
    figures.mkdir(parents=True, exist_ok=True)
    exports.mkdir(parents=True, exist_ok=True)
    stem = f'sale_vs_calendar_marginal_n{args.n:03d}'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none'})
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), layout='constrained')
    specs = [(sale_law, '#009E73', 'At own-sale dates', sale_high),
             (mu, '#D55E00', 'At calendar dates', calendar_high),
             (mu, '#D55E00', 'At calendar dates: log scale', calendar_high)]
    for ax, (law, color, title, high_mass) in zip(axes, specs):
        ax.plot(x, law, color=color, linewidth=1.8)
        ax.set(xlim=(0, 1), xlabel=r'Reputation $x$', title=title)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
        for position, label in ((params.p1, r'$p_1$'), (params.p2, r'$p_2$')):
            ax.axvline(position, color='#64748B', linestyle=':', linewidth=1)
            ax.text(position + .015, .96, label, transform=ax.get_xaxis_transform(),
                    va='top', color='#64748B')
        ax.set_ylabel('Probability per grid node')
        ax.text(.5, -.23, f'Mass above {lower_switch:g}: {100*high_mass:.4g}%',
                transform=ax.transAxes, ha='center', fontsize=10)
    axes[0].set_ylim(0, sale_law.max() * 1.15)
    axes[1].set_ylim(0, mu.max() * 1.15)
    axes[2].set_yscale('log')
    axes[2].set_ylim(1e-7, .1)
    axes[2].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{100*value:g}%'))
    i = int(np.flatnonzero(high)[np.argmax(mu[high])])
    axes[2].annotate('Small upper lobe', xy=(x[i], mu[i]), xytext=(.52, 1.5e-3),
                     arrowprops=dict(arrowstyle='->', color='#334155'),
                     color='#334155', ha='center', fontsize=10)
    fig.suptitle('The same stationary process, observed on two different clocks\n'
                 + rf'$p_0={p0}$   |   $\rho={rho}$   |   $n={args.n}$', fontsize=15)
    for extension in ('png', 'svg'):
        fig.savefig(figures / f'{stem}.{extension}', dpi=190, facecolor='white')
    plt.close(fig)
    csv_path = exports / f'{stem}.csv'
    np.savetxt(csv_path, np.column_stack((x, alpha, drift, mu, sale_law, effective_selection)),
               delimiter=',', comments='', fmt='%.17g',
               header='reputation,product2_probability,mean_drift_per_sale,'
                      'calendar_probability,sale_conditioned_probability,effective_selection')
    probes = []
    for target in (.1, .35, .8):
        index = int(np.argmin(abs(x - target)))
        probes.append(dict(reputation=float(x[index]),
                           effective_selection=float(effective_selection[index])))
    report = dict(
        p0=p0, rho=rho, n=args.n, model='competition', closure='marginal',
        source_manifest=str(manifest_path.relative_to(ROOT)),
        source_manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        checkpoint_sha256={name: manifest['checkpoint_sha256'][name]
                           for name in (tag + '.json', tag + '.npz')},
        policy_switch_brackets=[[float(x[i]), float(x[i+1])] for i in switch_indices],
        lower_high_region_boundary=lower_switch,
        calendar_high_region_mass=calendar_high, sale_high_region_mass=sale_high,
        calendar_mean=float(x @ mu), sale_mean=float(x @ sale_law), sale_rate=sale_rate,
        upper_lobe_maximum_grid_node=float(x[i]), upper_lobe_node_probability=float(mu[i]),
        independent_embedded_sale_law_l1=event_l1,
        effective_selection_probes=probes,
        identity='sale_law(x) = calendar_law(x) * E[D_A(x,Y) | X=x] / sale_rate',
        script_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        csv=str(csv_path.relative_to(ROOT)),
        figures=[str((figures / f'{stem}.{ext}').relative_to(ROOT)) for ext in ('png', 'svg')])
    (exports / f'{stem}.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
