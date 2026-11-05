"""Compare upper-end advantages while changing only the product-2 cost."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from forgetting.Code.model import Primitives, solve_boundary, boundary_demand

OUT = Path(__file__).resolve().parents[3] / 'forgetting/outputs/threshold_parameter_map'
RHO = .95


def evaluate(delta_c, nodes):
    p = Primitives(p2=.55, c2=.05+delta_c)
    solution = solve_boundary(p, RHO, central_nodes=nodes, tolerance=1e-12)
    grid = solution.coordinate
    x = np.unique(np.r_[0., 1., grid/RHO, (grid-(1-RHO))/RHO])
    x = x[(x >= 0) & (x <= 1)]
    x = x[np.r_[True, np.diff(x) > 1e-12]]
    x[0], x[-1] = 0., 1.
    advantage = -p.delta_c + p.gamma*p.delta_p*(
        np.interp(RHO*x+1-RHO, grid, solution.value)
        - np.interp(RHO*x, grid, solution.value))
    crossings = np.flatnonzero(advantage[:-1]*advantage[1:] < 0)
    roots = [float(x[j]-advantage[j]*(x[j+1]-x[j])/(advantage[j+1]-advantage[j]))
             for j in crossings]
    v1 = float(solution.value[-1])
    vrho = float(np.interp(RHO, grid, solution.value))
    row = dict(delta_c=delta_c, c2=p.c2, nodes=nodes, roots=roots,
               a1=float(advantage[-1]), V1=v1, Vrho=vrho,
               continuation_gap_at_1=p.gamma*(v1-vrho),
               extra_continuation_benefit_at_1=p.gamma*p.delta_p*(v1-vrho),
               bellman_residual=solution.bellman_residual)
    return x, advantage, row


def main():
    curves, results = [], []
    for delta_c in [.05, .075, .10]:
        _, _, coarse = evaluate(delta_c, 8001)
        x, advantage, fine = evaluate(delta_c, 16001)
        assert len(coarse['roots']) == len(fine['roots'])
        movement = max(abs(a-b) for a,b in zip(coarse['roots'],fine['roots']))
        assert movement < 2e-6
        fine['root_refinement_change'] = movement
        assert len(fine['roots']) == (1 if delta_c == .05 else 2)
        curves.append((x,advantage))
        results.append(fine)
    (OUT/'data').mkdir(exist_ok=True)
    (OUT/'figures').mkdir(exist_ok=True)
    data = dict(rho=RHO, p0=.5, p1=.35, p2=.55, c1=.05, revenue=1., gamma=.98,
        demand_at_rho_and_1=boundary_demand([RHO,1.],RHO,.5).tolist(),
        cases=results,
        scope='Numerical optimal boundary policies, each solved afresh; not an analytical threshold proof.')
    (OUT/'data/upper_threshold_comparison.json').write_text(json.dumps(data,indent=2)+'\n')
    pd.concat([pd.DataFrame({'delta_c':row['delta_c'],'x':x,'a_rho':a})
               for (x,a),row in zip(curves,results)],ignore_index=True).to_csv(
                   OUT/'data/upper_threshold_comparison.csv',index=False)

    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,
        'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2,figsize=(12.8,5.8))
    colors = ['#147d92','#9856a8','#d27c27']
    for (x,a),row,color in zip(curves,results,colors):
        label=rf"$\Delta c={row['delta_c']:.3f}$"
        for ax in axes:
            ax.plot(x,a,color=color,lw=2,label=label)
        axes[1].scatter([1.],[a[-1]],color=color,s=28,clip_on=False,zorder=4)
        if len(row['roots'])==2:
            root=row['roots'][-1]
            axes[1].scatter([root],[0],color=color,s=38,zorder=4)
            axes[1].annotate(f'{root:.4f}',xy=(root,0),xytext=(root,-.022),
                ha='center',va='top',fontsize=10,color=color,
                arrowprops={'arrowstyle':'-','color':color,'lw':.8})
    for ax in axes:
        ax.axhline(0,color='#465365',lw=1)
        ax.set(xlabel=r'State $x$',ylabel=r'Advantage $a_\rho(x)$')
        ax.grid(axis='y',alpha=.18)
    axes[0].set(xlim=(0,1),ylim=(-.12,1.16),title='Full curves: one lower crossing in all three cases')
    axes[1].set(xlim=(.82,1.008),ylim=(-.038,.11),title='High states: increasing cost creates an upper crossing')
    axes[1].annotate(r'$a_\rho(1)=0.02493>0$',xy=(1,results[0]['a1']),
        xytext=(.837,.043),color=colors[0],fontsize=11,
        arrowprops={'arrowstyle':'-','color':colors[0]})
    fig.suptitle('Why the upper threshold disappears when the cost gap is small',fontsize=18,y=.985)
    fig.text(.5,.908,r'$\rho=0.95$, $\gamma=0.98$, $R=1$, $p_0=0.50$, '
             r'$p_1=0.35$, $p_2=0.55$, $c_1=0.05$; only $c_2$ changes.',ha='center',fontsize=11)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.057),ncol=3,frameon=False)
    fig.text(.5,.022,'Positive advantage: choose product 2. Negative advantage: choose product 1. '
             'Each curve uses its own reoptimized value function.',ha='center',fontsize=9,color='#526173')
    fig.subplots_adjust(left=.068,right=.978,bottom=.23,top=.79,wspace=.24)
    target=OUT/'figures/06_why_no_upper_threshold.png'
    fig.savefig(target,dpi=190,facecolor='white')
    plt.close(fig)
    print(json.dumps(data,indent=2))
    print(target)


if __name__ == '__main__':
    main()
