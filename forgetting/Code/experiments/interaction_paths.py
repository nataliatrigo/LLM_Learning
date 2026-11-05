"""Individual forgetting histories indexed only by interactions with Seller A."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forgetting.Code.experiments import run_experiments as experiments
from forgetting.Code.layers import solve_observation_layers
from forgetting.Code.model import Primitives
from forgetting.Code.simulation import _policy_and_demand
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def simulate(primitives: Primitives, rho: float, *, paths: int = 100,
             interactions: int = 200_000, seed: int = 20260809,
             nodes: int = 2001, terminal_gap: float = 1e-5,
             tolerance: float = 1e-9, action_tolerance: float = 1e-6,
             max_iterations: int = 100_000,
             trace_window: tuple[int, int] = (50_000, 55_000)) -> dict:
    """Use the existing policy; every step is an A interaction, with no demand draw.

    At interaction n the policy uses n-1 observations and the pre-choice state.
    X starts at zero, Y = 1-rho**(n-1)-X, exactly as in calendar simulation.
    Vector entries are independent paths driven by separate IID random draws.
    """
    if paths < 1 or interactions < 1:
        raise ValueError('paths and interactions must be positive')
    if not 1 <= trace_window[0] <= trace_window[1] <= interactions:
        raise ValueError('trace_window must be within the simulated interactions')
    _, policy = solve_observation_layers(
        primitives, rho, nodes=nodes, terminal_gap=terminal_gap,
        tolerance=tolerance, action_tolerance=action_tolerance,
        max_iterations=max_iterations, return_policy=True)
    representatives = np.unique(np.linspace(0, paths-1, min(paths, 10), dtype=int))
    traced = representatives[:3]
    checkpoints = np.unique(np.r_[np.geomspace(1, interactions, 500).astype(int),
                                  np.linspace(1, interactions, 1000).astype(int)])
    cumulative = np.empty((len(checkpoints), paths))
    choices = np.empty((interactions, paths), dtype=bool)
    trace = []
    rng = np.random.default_rng(seed)
    x = np.zeros(paths)
    counts = np.zeros(paths, dtype=np.int64)
    observations = np.zeros(paths, dtype=np.int64)
    position = 0
    for n in range(1, interactions+1):
        product2, _ = _policy_and_demand(observations, x, rho=rho,
                                       policy=policy, action_tolerance=action_tolerance)
        success = rng.random(paths) < np.where(product2, primitives.p2, primitives.p1)
        next_x = rho*x + (1-rho)*success
        if trace_window[0] <= n <= trace_window[1]:
            for j in traced:
                trace.append({'trajectory': int(j+1), 'interaction': n,
                    'x_before': x[j], 'y_before': 1-rho**(n-1)-x[j],
                    'product2': int(product2[j]), 'success': int(success[j]),
                    'x_after': next_x[j], 'y_after': 1-rho**n-next_x[j]})
        choices[n-1] = product2
        counts += product2
        if n == checkpoints[position]:
            cumulative[position] = counts/n
            position += 1
        x = next_x
        observations += 1
        if n % 25_000 == 0:
            print(f'[interaction paths] {n:,}/{interactions:,}; {paths} trajectories', flush=True)
    return {'choices': choices, 'checkpoints': checkpoints, 'cumulative': cumulative,
            'representatives': representatives, 'trace': pd.DataFrame(trace),
            'final_x': x, 'final_y': 1-rho**interactions-x,
            'terminal_observation': int(policy['terminal_observation'])}


def distribution(values) -> dict:
    return {'mean': float(np.mean(values)), 'median': float(np.median(values)),
            'std': float(np.std(values, ddof=1)), 'min': float(np.min(values)),
            'max': float(np.max(values)), 'p05': float(np.quantile(values, .05)),
            'p95': float(np.quantile(values, .95))}


def diagnostics(result: dict, *, burn_in: int = 20_000, block_size: int = 10_000,
                lengths: tuple[int, ...] = (50_000, 100_000, 200_000)) -> dict:
    choices = result['choices']
    total, paths = choices.shape
    split = total//2
    if not 0 <= burn_in < split or block_size <= 0 or (total-burn_in) % block_size:
        raise ValueError('Require burn-in before midpoint and complete post-burn-in blocks')
    if not lengths or any(not burn_in < n <= total for n in lengths):
        raise ValueError('Each robustness length must exceed burn-in and fit the history')
    first = choices[burn_in:split].mean(axis=0)
    second = choices[split:].mean(axis=0)
    final = choices[burn_in:].mean(axis=0)
    blocks = choices[burn_in:].reshape(-1, block_size, paths).mean(axis=1)
    summary = pd.DataFrame({'trajectory': np.arange(1, paths+1),
        'post_burn_in_frequency': final, 'first_segment_frequency': first,
        'second_segment_frequency': second, 'absolute_segment_difference': abs(first-second),
        'min_block_frequency': blocks.min(axis=0), 'max_block_frequency': blocks.max(axis=0),
        'final_x': result['final_x'], 'final_y': result['final_y']})
    robustness = []
    for n in lengths:
        full = choices[:n].mean(axis=0)
        post = choices[burn_in:n].mean(axis=0)
        summary[f'F_{n}'] = full
        summary[f'post_burn_in_frequency_{n}'] = post
        robustness.append({'interactions': n,
            **{f'full_{k}': v for k,v in distribution(full).items()},
            **{f'post_burn_in_{k}': v for k,v in distribution(post).items()}})
    block_data = pd.DataFrame([{'trajectory': j+1, 'block': b+1,
        'start_interaction': burn_in+b*block_size+1,
        'end_interaction': burn_in+(b+1)*block_size, 'frequency': value}
        for b, values in enumerate(blocks) for j, value in enumerate(values)])
    stats = {**distribution(final),
             'mean_absolute_segment_difference': float(abs(first-second).mean()),
             'minimum_late_block_frequency': float(blocks.min()),
             'trajectories_with_zero_post_burn_in_frequency': int(np.sum(final == 0)),
             'burn_in': burn_in, 'block_size': block_size,
             'first_segment': [burn_in+1, split], 'second_segment': [split+1, total]}
    return {'summary': summary, 'blocks': block_data,
            'robustness': pd.DataFrame(robustness), 'statistics': stats}


def plot_frequency_distribution(summary: pd.DataFrame) -> plt.Figure:
    """Use round, explicitly labelled bin boundaries and integer counts."""
    values = summary.post_burn_in_frequency.to_numpy()
    target = max(float(np.ptp(values))/10, .001)
    scale = 10**np.floor(np.log10(target))
    width = next(step*scale for step in (1, 2, 5, 10) if step*scale >= target)
    lower = int(np.floor(values.min()/width))
    upper = max(lower+1, int(np.ceil(values.max()/width)))
    edges = np.arange(lower, upper+1)*width
    counts, _ = np.histogram(values, bins=edges)
    decimals = max(0, int(-np.floor(np.log10(width*100))))
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    _, _, bars = ax.hist(values, bins=edges, color='#2563eb', edgecolor='white',
                         linewidth=1.5, rwidth=.96)
    ax.set_xticks(edges, [f'{value*100:.{decimals}f}%' for value in edges],
                  rotation=45, ha='right')
    from matplotlib.ticker import MaxNLocator
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_axisbelow(True)
    ax.grid(axis='x', visible=True, linestyle=':', color='#94a3b8')
    ax.bar_label(bars, labels=[str(int(n)) if n else '' for n in counts], padding=3)
    ax.set_ylim(0, max(1, counts.max())*1.18)
    ax.set(xlabel=f'Product-2 frequency after burn-in — bin width: {width*100:g} percentage points',
           ylabel='Number of trajectories',
           title='Distribution of individual frequencies\n'
                 'Ticks mark bin boundaries; numbers above bars count trajectories')
    return fig


def save_outputs(result: dict, report: dict, output: Path, config: dict) -> None:
    data = output/'data'
    figures = output/'figures'
    data.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for key, filename in [('summary', 'trajectory_summary'), ('blocks', 'block_frequencies'),
                          ('robustness', 'length_robustness')]:
        report[key].to_csv(data/f'{filename}.csv', index=False)
    result['trace'].to_csv(data/'late_state_dynamics.csv', index=False)
    reps = result['representatives']
    cumulative = pd.DataFrame(result['cumulative'][:, reps],
        columns=[f'trajectory_{j+1}' for j in reps])
    cumulative.insert(0, 'interactions', result['checkpoints'])
    cumulative.to_csv(data/'representative_cumulative_frequencies.csv', index=False)
    (data/'diagnostic_statistics.json').write_text(json.dumps(report['statistics'], indent=2)+'\n')
    experiments.configure_style()
    colors = plt.colormaps['tab10'](np.arange(len(reps)))
    stems = []
    def save(fig, stem):
        fig.savefig(figures/f'{stem}.png', dpi=180, bbox_inches='tight')
        plt.close(fig)
        stems.append(stem)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for j, color in zip(reps, colors):
        ax.plot(result['checkpoints'], result['cumulative'][:,j], color=color,
                lw=1, label=f'Trajectory {j+1}')
    ax.set(xlabel='Interactions with A, N', ylabel='Cumulative product-2 frequency Fⱼ(N)',
           title='Individual cumulative frequencies — includes initial transient')
    ax.legend(ncols=2, fontsize=8)
    save(fig, 'individual_cumulative_frequencies')
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    for j, color in zip(reps, colors):
        b = report['blocks'].query('trajectory == @j + 1')
        ax.plot(b.end_interaction, b.frequency, 'o-', ms=3, lw=1,
                color=color, label=f'Trajectory {j+1}')
    ax.set(xlabel='Block end: interactions with A', ylabel='Product-2 frequency within block',
           title='Non-overlapping blocks after burn-in — individual trajectories')
    ax.legend(ncols=5, fontsize=8, loc='upper center', bbox_to_anchor=(.5, -.16))
    save(fig, 'individual_block_frequencies')
    fig = plot_frequency_distribution(report['summary'])
    save(fig, 'individual_frequency_distribution')
    fig, ax = plt.subplots(figsize=(6, 6), layout='constrained')
    s = report['summary']
    low = min(s.first_segment_frequency.min(), s.second_segment_frequency.min())-.002
    high = max(s.first_segment_frequency.max(), s.second_segment_frequency.max())+.002
    ax.scatter(s.first_segment_frequency, s.second_segment_frequency, s=16, alpha=.7)
    ax.plot([low,high], [low,high], '--', color='black', lw=1)
    a,b = report['statistics']['first_segment'], report['statistics']['second_segment']
    ax.set(xlim=(low,high), ylim=(low,high),
           xlabel=f'Frequency: interactions {a[0]:,}–{a[1]:,}',
           ylabel=f'Frequency: interactions {b[0]:,}–{b[1]:,}',
           title='Late-horizon stability — one point per trajectory')
    ax.set_aspect('equal')
    save(fig, 'individual_late_stability')
    trace = result['trace']
    ids = sorted(trace.trajectory.unique())
    fig, axes = plt.subplots(len(ids), 2, figsize=(13, 2.6*len(ids)), squeeze=False,
                             layout='constrained')
    for row,j in enumerate(ids):
        history = trace[trace.trajectory == j]
        for col, chunk in enumerate((history, history.tail(250))):
            ax = axes[row,col]
            ax.plot(chunk.interaction, chunk.x_before, lw=.65, color='#64748b')
            active = chunk[chunk.product2 == 1]
            ax.scatter(active.interaction, active.x_before, s=5, color='#2563eb',
                       label='Product 2 chosen')
            ax.set(xlabel='Interaction with A, n', ylabel='X before choice',
                   title=f'Trajectory {j} — '+('late window' if col == 0 else 'last 250: detail'))
    axes[0,0].legend(fontsize=8)
    save(fig, 'late_state_and_product_choices')
    fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
    r = report['robustness']
    ax.plot(r.interactions, r.full_std, 'o-', label='Fⱼ(N): all interactions')
    ax.plot(r.interactions, r.post_burn_in_std, 's-', label='Frequency after burn-in')
    ax.set(xlabel='Trajectory length N (same path prefixes)',
           ylabel='Across-trajectory sample standard deviation',
           title='Dispersion versus simulation length')
    ax.legend()
    save(fig, 'frequency_dispersion_by_length')
    manifest = {'config': config, 'initial_state': [0,0],
        'time_index': 'n is the 1-based interaction with A; states recorded before choice',
        'representatives': [int(j+1) for j in reps],
        'representative_selection': 'Ten evenly spaced path IDs selected before seeing results',
        'tie_behavior': 'Product 2 iff interpolated advantage > action_tolerance; else product 1',
        'terminal_observation': result['terminal_observation'],
        'policy': 'Existing solve_observation_layers and simulation._policy_and_demand',
        'figures': [f'figures/{stem}.png' for stem in stems]}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=experiments.CONFIG_DIR/'config_interaction.json')
    parser.add_argument('--output-dir', type=Path, default=experiments.ROOT/'outputs/policy_and_state_experiments/interaction_paths')
    args = parser.parse_args(argv)
    config = experiments.read_config(args.config)
    opts = config['experiment']
    result = simulate(Primitives.from_mapping(config['seller']), config['rho'],
        paths=opts['paths'], interactions=opts['interactions'], seed=opts['seed'],
        trace_window=tuple(opts['trace_window']), **config['solver'])
    report = diagnostics(result, burn_in=opts['burn_in'], block_size=opts['block_size'],
                         lengths=tuple(opts['lengths']))
    save_outputs(result, report, args.output_dir, config)
    print(json.dumps(report['statistics'], indent=2))
    print(report['robustness'].to_string(index=False))
    print(f'Results: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
