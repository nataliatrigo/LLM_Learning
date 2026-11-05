"""Plot an original-map example whose optimal policy is 1 -> 2."""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from forgetting.Code.model import Primitives, solve_boundary

OUT = Path(__file__).resolve().parents[3] / 'forgetting/outputs/threshold_parameter_map'
P = Primitives(p0=.5, p1=.35, p2=.55, c1=.05, c2=.10, revenue=1., gamma=.98)
RHO = .95


def evaluate(nodes):
    solution = solve_boundary(P, RHO, central_nodes=nodes, tolerance=1e-12)
    grid = solution.coordinate
    knots = np.unique(np.r_[0., 1., grid/RHO, (grid-(1-RHO))/RHO])
    knots = knots[(knots >= 0) & (knots <= 1)]
    knots = knots[np.r_[True, np.diff(knots) > 1e-12]]
    knots[0], knots[-1] = 0., 1.
    a = -P.delta_c + P.gamma * P.delta_p * (
        np.interp(RHO*knots+1-RHO, grid, solution.value)
        - np.interp(RHO*knots, grid, solution.value))
    changes = np.flatnonzero(a[:-1]*a[1:] < 0)
    roots = [float(knots[j]-a[j]*(knots[j+1]-knots[j])/(a[j+1]-a[j])) for j in changes]
    assert len(roots) == 1 and a[0] < 0 and a[-1] > 0
    return knots, a, dict(nodes=nodes, root=roots[0], a0=float(a[0]), a1=float(a[-1]),
                         a_max=float(a.max()), bellman_residual=solution.bellman_residual)


def main():
    (OUT/'data').mkdir(exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    _, _, coarse = evaluate(8001)
    x, a, fine = evaluate(16001)
    movement = abs(coarse['root']-fine['root'])
    assert movement < 1e-6
    info = dict(rho=RHO, **P.as_dict(), delta_p=P.delta_p, delta_c=P.delta_c,
                coarse=coarse, fine=fine, root_refinement_difference=movement,
                scope='Numerical optimal boundary policy; not a certified continuous-state proof.')
    (OUT/'data/one_lower_case.json').write_text(json.dumps(info, indent=2)+'\n')
    pd.DataFrame({'x': x, 'a_rho': a}).to_csv(OUT/'data/one_lower_case.csv', index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':11,
                         'axes.spines.top':False, 'axes.spines.right':False,
                         'axes.labelsize':12, 'pdf.fonttype':42})
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2), sharex=True)
    root = fine['root']
    for ax in axes:
        ax.axhline(0, color='#657484', lw=1)
        ax.plot(x, a, color='#175d77', lw=2.1)
        ax.fill_between(x, 0, a, where=a>0, interpolate=True, color='#248b98', alpha=.19)
        ax.fill_between(x, 0, a, where=a<0, interpolate=True, color='#aab2be', alpha=.35)
        ax.axvline(root, color='#737d8a', ls=':', lw=1.2)
        ax.scatter([root], [0], s=35, color='#175d77', zorder=5)
        ax.set(xlim=(0, 1), xlabel=r'State $x$', ylabel=r'Advantage $a_\rho(x)$')
        ax.grid(axis='y', color='#e3e7eb', lw=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylim(-.12, fine['a_max']*1.09)
    axes[0].set_title('Full advantage curve', loc='left', pad=13)
    axes[0].text(.53, .80, 'Product 2 is optimal\nwhere the curve is positive',
                 color='#175d77', fontsize=11)
    axes[0].annotate(rf'Threshold $x\approx {root:.6f}$', xy=(root, 0),
                     xytext=(.32, -.085), arrowprops={'arrowstyle':'-', 'color':'#657484'},
                     fontsize=11)
    axes[1].set_ylim(-.065, .08)
    axes[1].set_title('Vertical zoom around zero', loc='left', pad=13)
    axes[1].scatter([0, 1], [a[0], a[-1]], color='#175d77', s=28, zorder=5, clip_on=False)
    axes[1].annotate(rf'$a_\rho(0)\approx {a[0]:.5f}$', xy=(0, a[0]),
                     xytext=(.23, -.052), arrowprops={'arrowstyle':'-', 'color':'#657484'})
    axes[1].annotate(rf'$a_\rho(1)\approx {a[-1]:.5f}>0$', xy=(1, a[-1]),
                     xytext=(.48, .050), arrowprops={'arrowstyle':'-', 'color':'#657484'})
    fig.suptitle(r'One lower threshold: the policy is $1\rightarrow2$',
                 x=.075, y=.98, ha='left', fontsize=20, fontweight='bold')
    fig.text(.075, .918,
             r'$\rho=0.95,\ \Delta p=0.20,\ \Delta c=0.05$'
             r'   |   $p_1=0.35,\ p_2=0.55,\ c_1=0.05,\ c_2=0.10$', fontsize=11.5)
    fig.text(.075, .866, r'$R=1,\ \gamma=0.98,\ p_0=0.50$', fontsize=11.5)
    fig.text(.075, .099,
             rf'Product 1: $0\leq x<{root:.6f}$     '
             rf'Product 2: ${root:.6f}<x\leq1$     Both products tie at the threshold.', fontsize=11)
    fig.text(.075, .042,
             r'$a_\rho(x)=-\Delta c+\gamma\Delta p[V_\rho(\rho x+1-\rho)-V_\rho(\rho x)]$'
             '   |   Numerical result; checked at 8,001 and 16,001 state nodes.',
             fontsize=10, color='#586575')
    fig.subplots_adjust(left=.075, right=.97, top=.77, bottom=.23, wspace=.24)
    for extension in ['png', 'pdf']:
        fig.savefig(OUT/f'figures/05_a_rho_one_lower_case.{extension}', dpi=190,
                    bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
