"""Reproduce the 2,000-interaction state plots for the verified 1 -> 2 case.

Run from the project root:
  .venv/bin/python -B forgetting/Code/threshold_parameter_map/simulate_one_lower_paths.py
"""
from pathlib import Path
import json
import sys

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd

from forgetting.Code.model import Primitives, solve_boundary
from forgetting.Code.experiments.states import replay_prefix, plot_states

ROOT = PROJECT / 'forgetting/outputs/policy_and_state_experiments/first_2000_one_lower'
REFERENCE = PROJECT / 'forgetting/outputs/policy_and_state_experiments/first_2000/data/p0_0.5_rho_0.95/records_manifest.json'
RHO, COUNT, PATHS = .95, 2000, 20


def main():
    reference = json.loads(REFERENCE.read_text())
    primitives = Primitives(p2=.55, c2=.10)
    solver = {**reference['policy_solver'], 'nodes': 8001, 'terminal_gap': 1e-8,
              'tolerance': 1e-12, 'action_tolerance': 1e-8}
    records = replay_prefix(primitives, RHO, solver, seed=reference['seed'],
                            original_paths=PATHS, paths=PATHS, count=COUNT)
    refined_solver = {**solver, 'nodes': 16001, 'terminal_gap': 1e-10}
    refined = replay_prefix(primitives, RHO, refined_solver, seed=reference['seed'],
                            original_paths=PATHS, paths=PATHS, count=COUNT)
    assert np.array_equal(records, refined), 'Refined simulation differs; inspect before delivery.'

    boundary = solve_boundary(primitives, RHO, central_nodes=16001, tolerance=1e-12)
    crossings = np.flatnonzero(boundary.advantage[:-1] * boundary.advantage[1:] < 0)
    assert len(crossings) == 1 and boundary.advantage[0] < 0 < boundary.advantage[-1]
    j = crossings[0]
    x, a = boundary.coordinate, boundary.advantage
    threshold = float(x[j] - a[j] * (x[j+1]-x[j]) / (a[j+1]-a[j]))

    # The second experiment uses the mature policy from an explicitly different
    # initial state: (x,y)=(0,1), rather than an empty history (0,0).
    mature_records = np.empty_like(records)
    rng = np.random.default_rng(reference['seed'])
    state = np.zeros(PATHS)
    for row in range(COUNT):
        product2 = np.interp(state, x, a) > solver['action_tolerance']
        success = rng.random(PATHS) < np.where(product2, primitives.p2, primitives.p1)
        mature_records['x_before'][row] = state
        mature_records['product2'][row] = product2
        mature_records['success'][row] = success
        state = RHO*state + (1-RHO)*success

    summaries = []
    for name, root, sample, initial_y in [
        ('empty_history', ROOT, records, 0),
        ('boundary_start', ROOT/'boundary_start', mature_records, 1),
    ]:
        directory = root/'data/p0_0.5_rho_0.95'
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory/'observations.npy', sample)
        meta = {
            'shape': list(sample.shape), 'burn_in': 0, 'seed': reference['seed'],
            'p0': .5, 'rho': RHO, 'seller': primitives.as_dict(), 'policy_solver': solver,
            'initial_state': {'x': 0, 'y': initial_y},
            'policy': 'Observation-layer policy, then mature approximation' if initial_y == 0 else 'Mature boundary policy throughout',
            'row_interaction': '1 + row_index', 'column_trajectory': '1 + column_index',
            'state_timing': 'before choice and outcome',
            'stored_state': 'x_before = (1-rho)*S before decision',
            'plotted_state': 'S = x_before/(1-rho)',
            'reference_settings': str(REFERENCE.relative_to(PROJECT)),
            'random_numbers': 'Same seed and 20-path uniform draw ordering as reference',
            'representative_selection': 'Customer 1 fixed before simulation; no path or seed search',
            'mature_threshold_x': threshold, 'mature_threshold_S': threshold/(1-RHO),
            'refinement': {'nodes': 16001, 'terminal_gap': 1e-10,
                           'empty_history_records_identical': True},
        }
        (directory/'records_manifest.json').write_text(json.dumps(meta, indent=2)+'\n')
        plot_states(root, .5, RHO, trajectory=1, start=1, count=COUNT)
        n = np.arange(1, COUNT+1)
        first = sample[:, 0]
        before = first['x_before']
        total = 1-RHO**(n-1) if initial_y == 0 else np.ones(COUNT)
        after = RHO*before + (1-RHO)*first['success']
        assert np.allclose(after[:-1], before[1:], atol=1e-14, rtol=0)
        assert np.all(before >= 0) and np.all(before <= total + 1e-14)
        frame = pd.DataFrame({'interaction': n, 'x_before': before,
            'y_before': total-before, 'S_before': before/(1-RHO),
            'product': np.where(first['product2'], 2, 1),
            'success': first['success'].astype(int), 'x_after': after})
        frame.to_csv(directory/'individual_state_trace.csv', index=False)
        for path in range(PATHS):
            action = sample['product2'][:, path]
            summaries.append({'experiment': name, 'trajectory': path+1,
                'product1_count': int((~action).sum()), 'product2_count': int(action.sum()),
                'temporal_switches': int(np.count_nonzero(action[1:] != action[:-1]))})

    figures = ROOT/'figures'
    state_threshold = threshold/(1-RHO)
    fig, axes = plt.subplots(2, 1, figsize=(13, 8.3), sharex=True, sharey=True)
    fig.subplots_adjust(left=.075, right=.985, top=.83, bottom=.15, hspace=.30)
    n = np.arange(1, COUNT+1)
    for ax, sample, title in zip(axes, [records, mature_records], [
        'Empty history: initial (x, y) = (0, 0); optimal transient policy',
        'Long-run boundary: initial (x, y) = (0, 1); boundary policy from the start',
    ]):
        first = sample[:, 0]
        s = first['x_before']/(1-RHO)
        ax.plot(n, s, color='#94a3b8', lw=.65, zorder=1)
        for mask, color in [(~first['product2'], '#2563eb'), (first['product2'], '#16a34a')]:
            ax.scatter(n[mask], s[mask], s=7, color=color, zorder=2)
        ax.axhline(state_threshold, color='#7c3aed', ls='--', lw=1.25, zorder=3)
        ax.set(title=title, ylabel=r'State $S$ before choice', ylim=(-.4, 17))
        ax.grid(alpha=.18)
        ax.text(.995, .96, f"Product 1: {(~first['product2']).sum():,} choices; "
                f"product 2: {first['product2'].sum():,}", transform=ax.transAxes,
                ha='right', va='top', fontsize=9,
                bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
    axes[1].set_xlabel('Interaction with A')
    axes[1].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker='o', ls='', color='#2563eb', label='Product 1'),
               Line2D([], [], marker='o', ls='', color='#16a34a', label='Product 2'),
               Line2D([], [], color='#7c3aed', ls='--', label=f'Long-run threshold: S = {state_threshold:.4f}')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5,.055), ncol=3, frameon=False)
    fig.suptitle('One state threshold: product 1 below it, product 2 above it', fontsize=17, y=.97)
    fig.text(.5,.915,r'$\rho=0.95$, $\gamma=0.98$, $p_0=0.50$, $p_1=0.35$, $p_2=0.55$, '
             r'$c_1=0.05$, $c_2=0.10$, $R=1$',ha='center',fontsize=11)
    fig.text(.5,.023,'Customer 1, selected before simulation. The dashed line describes the boundary policy; '
             'early empty-history choices use a different policy.',ha='center',fontsize=9,color='#475569')
    comparison = figures/'one_lower_initial_conditions.png'
    fig.savefig(comparison, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11,4.7), layout='constrained')
    first = mature_records[:100,0]
    n = np.arange(1,101)
    s = first['x_before']/(1-RHO)
    ax.plot(n,s,color='#94a3b8',lw=.8)
    for mask,color,label in [(~first['product2'],'#2563eb','Product 1'),
                             (first['product2'],'#16a34a','Product 2')]:
        ax.scatter(n[mask],s[mask],s=22,color=color,label=label,zorder=3)
    ax.axhline(state_threshold,color='#7c3aed',ls='--',label=f'Threshold S = {state_threshold:.4f}')
    ax.set(xlabel='Interaction with A',ylabel=r'State $S$ before choice',
           title='First 100 interactions on the long-run boundary\n'
                 r'Customer 1; initial $(x,y)=(0,1)$; $\rho=0.95$, $p_2=0.55$, $c_2=0.10$')
    ax.grid(alpha=.2)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.15),ncol=3,frameon=False)
    fig.savefig(figures/'one_lower_boundary_first_100.png',dpi=180,bbox_inches='tight')
    plt.close(fig)
    pd.DataFrame(summaries).to_csv(ROOT/'trajectory_summary.csv',index=False)
    report = {'threshold_x':threshold,'threshold_S':state_threshold,
              'empty_history_path1_product1_count':int((~records['product2'][:,0]).sum()),
              'boundary_start_path1_product1_count':int((~mature_records['product2'][:,0]).sum()),
              'boundary_start_path1_product1_interactions':(np.flatnonzero(~mature_records['product2'][:,0])+1).tolist(),
              'boundary_end_advantages':a[[0,-1]].tolist(),
              'boundary_bellman_residual':boundary.bellman_residual,
              'transient_refinement_records_identical':True}
    (ROOT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    print('Comparison:',comparison)


if __name__ == '__main__':
    main()
