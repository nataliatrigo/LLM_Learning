"""State plots: saved histories, initial prefixes, distributions and reachable sets.

Run these through ``python -m forgetting <command> --help``. Numerical policies
and random draw ordering are shared with the existing simulation modules.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

from forgetting.Code.experiments.run_experiments import CONFIG_DIR, ROOT, configure_style, read_config
from forgetting.Code.experiments.interaction_paths import simulate
from forgetting.Code.layers import solve_observation_layers
from forgetting.Code.model import (
    Primitives, BoundarySolution, boundary_summary, solve_boundary, validate_rho,
)
from forgetting.Code.simulation import _policy_and_demand

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator, PercentFormatter
import numpy as np
import pandas as pd



# Reachable boundary and policy geometry

COLORS = {"product1": "#64748b", "product2": "#059669", "critical": "#d97706"}


def reachable_cover(rho: float, depth: int) -> np.ndarray:
    """Closed cylinder intervals containing K_rho, not finite-time states.

    For rho < 1/2, omitted gaps are exact exclusions; remaining intervals
    still contain unresolved smaller gaps. Each interval has width rho**depth.
    For rho >= 1/2, the full interval is the exact attractor.
    """
    validate_rho(rho)
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    if depth == 0 or rho >= .5:
        return np.array([[0., 1.]])
    intervals = np.array([[0., 1.]])
    for _ in range(depth):
        intervals = np.vstack((rho * intervals, rho * intervals + (1-rho)))
    return intervals


def policy_intervals(solution: BoundarySolution, tolerance: float) -> list[tuple[float, float, str]]:
    """Partition the full boundary by the interpolated action-value difference."""
    x, values = solution.coordinate, solution.advantage
    edges = [0., 1.]
    for level in (-tolerance, tolerance):
        shifted = values - level
        for i in np.flatnonzero(shifted[:-1] * shifted[1:] < 0):
            edges.append(float(x[i] - shifted[i] * (x[i+1]-x[i]) / (shifted[i+1]-shifted[i])))
        edges.extend(x[shifted == 0].tolist())
    edges = np.unique(edges)
    intervals = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        value = float(np.interp((lower+upper)/2, x, values))
        action = "product2" if value > tolerance else "product1" if value < -tolerance else "critical"
        intervals.append((float(lower), float(upper), action))
    return intervals


def intersect_intervals(cover: np.ndarray, lower: float, upper: float) -> np.ndarray:
    clipped = np.column_stack((np.maximum(cover[:, 0], lower), np.minimum(cover[:, 1], upper)))
    return clipped[clipped[:, 1] - clipped[:, 0] > 8 * np.finfo(float).eps]


def draw_intervals(ax, intervals: np.ndarray, row: int, color: str) -> None:
    if len(intervals):
        lines = np.stack((np.column_stack((intervals[:, 0], np.full(len(intervals), row))),
                          np.column_stack((intervals[:, 1], np.full(len(intervals), row)))), axis=1)
        ax.add_collection(LineCollection(lines, colors=color, linewidths=12,
                                        capstyle="butt", rasterized=True))
        # Keep tiny cylinders visible as location marks instead of letting
        # rasterization erase an entire nonempty reachable set.
        tiny = intervals[intervals[:, 1] - intervals[:, 0] < .001]
        if len(tiny):
            ax.plot(tiny[:, 0], np.full(len(tiny), row), "|", color=color,
                    markersize=12, markeredgewidth=.4, rasterized=True)


def save(fig, directory: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def reachable_main(argv=None) -> None:
    parser = argparse.ArgumentParser(description='Visualize the mature reachable set and distinguish geometry from policy gaps.')
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "config_paper.json")
    parser.add_argument("--p0", type=float, default=.9)
    parser.add_argument("--c2", type=float, help="override c2 without editing the configuration")
    parser.add_argument("--rhos", type=float, nargs="+", default=[.2, .35, .43, .45, .47, .49, .5, .7])
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--focus-rho", type=float, default=.45)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "reachable_states")
    args = parser.parse_args(argv)
    if not 1 <= args.depth <= 18:
        parser.error("--depth must lie between 1 and 18")
    config = read_config(args.config)
    primitives = Primitives.from_mapping({**config["seller"], "p0": args.p0,
        "gamma": config["rho_p0_comparative_statics"]["gamma"]})
    if args.c2 is not None:
        primitives = replace(primitives, c2=args.c2)
    rhos = sorted(set(args.rhos))
    for rho in [*rhos, args.focus_rho]:
        validate_rho(rho)
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, max(4, .48 * len(rhos))),
                             sharex=True, sharey=True, layout="constrained")
    summaries, cover_rows, policy_rows = [], [], []
    solver = config["solver"]
    for row, rho in enumerate(rhos):
        solution = solve_boundary(primitives, rho, central_nodes=int(solver["central_nodes"]),
            outer_nodes=int(solver["outer_nodes"]), half_widths=float(solver["half_widths"]),
            tolerance=float(solver["tolerance"]), max_iterations=int(solver["max_iterations"]))
        summary = boundary_summary(solution, action_tolerance=float(solver["action_tolerance"]),
                                   reachable_depth=int(solver["reachable_depth"]))
        summary["cover_interval_width"] = rho**args.depth if rho < .5 else 0.
        summaries.append(summary)
        cover = reachable_cover(rho, args.depth)
        cover_rows.extend({"rho": rho, "lower": a, "upper": b} for a,b in cover)
        draw_intervals(axes[0], cover, row, "#2563eb")
        for lower, upper, action in policy_intervals(solution, float(solver["action_tolerance"])):
            policy_rows.append({"rho": rho, "lower": lower, "upper": upper, "action": action})
            draw_intervals(axes[1], np.array([[lower, upper]]), row, COLORS[action])
            draw_intervals(axes[2], intersect_intervals(cover, lower, upper), row, COLORS[action])
    axes[0].set_yticks(range(len(rhos)), [f"{rho:g}" for rho in rhos])
    axes[0].set_ylabel(r"Factor de memoria $\rho$")
    titles = [r"(a) Cubierta de $\mathcal{K}_\rho$",
              "(b) Política en toda la frontera", "(c) Política sobre la cubierta"]
    for ax, title in zip(axes, titles):
        ax.set(xlim=(-.01, 1.01), ylim=(len(rhos)-.5, -.5), title=title,
               xlabel=r"Estado $x_s$ (con $x_f=1-x_s$)")
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.grid(axis="x", alpha=.3)
        ax.grid(axis="y", visible=False)
    handles = [Line2D([0], [0], color=COLORS[k], lw=5, label=v) for k,v in
               [("product1", "Producto 1"), ("product2", "Producto 2"), ("critical", "Cerca de indiferencia")]]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    fig.suptitle(f"Estados alcanzables y decisión: $p_0={primitives.p0:g}$, $c_2={primitives.c2:g}$, "
                 f"$\\gamma={primitives.gamma:g}$\nCubierta exterior de profundidad {args.depth}; "
                 "marcas finas para intervalos subpíxel; huecos más pequeños sin resolver",
                 fontsize=10)
    save(fig, directory, "reachable_states_and_policy")

    fig, ax = plt.subplots(figsize=(9, 4.3), layout="constrained")
    levels = range(min(args.depth, 6)+1)
    for level in levels:
        draw_intervals(ax, reachable_cover(args.focus_rho, level), level, "#2563eb")
    ax.set_yticks(list(levels), [f"d = {d}" for d in levels])
    ax.set(xlim=(-.01, 1.01), ylim=(len(levels)-.5, -.5), xlabel=r"Estado normalizado $x_s$",
           title=rf"Construcción de $\mathcal{{K}}_\rho$ con $\rho={args.focus_rho:g}$: cubiertas sucesivas")
    ax.grid(axis="y", visible=False)
    if args.focus_rho < .5:
        a, b = args.focus_rho, 1-args.focus_rho
        ax.axvspan(a, b, color="#fca5a5", alpha=.25, ymin=0, ymax=(len(levels)-1)/len(levels))
        ax.text((a+b)/2, (len(levels)-1)/2, f"Hueco\n({a:g}, {b:g})", ha="center", va="center", fontsize=9)
    save(fig, directory, "reachable_set_construction")
    pd.DataFrame(summaries).to_csv(directory / "diagnostics.csv", index=False)
    pd.DataFrame(cover_rows).to_csv(directory / "cover_intervals.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(directory / "full_boundary_policy_intervals.csv", index=False)
    (directory / "manifest.json").write_text(json.dumps({"config": config,
        "effective_primitives": primitives.as_dict(), "rhos": rhos, "depth": args.depth,
        "focus_rho": args.focus_rho, "note": "Outer cylinder cover, not exact finite-observation states."}, indent=2)+'\n')
    print(f"Figures and diagnostics: {directory}")


# Saved individual histories

def plot_states(root: Path, p0: float, rho: float, trajectory: int = 1,
                start: int | None = None, count: int = 1000) -> Path:
    directory = root / 'data' / f'p0_{p0}_rho_{rho}'
    meta = json.loads((directory / 'records_manifest.json').read_text())
    records = np.load(directory / 'observations.npy', mmap_mode='r')
    first = meta['burn_in'] + 1
    start = first if start is None else start
    offset = start - first
    if not 1 <= trajectory <= records.shape[1]:
        raise ValueError(f'Trajectory must be between 1 and {records.shape[1]}')
    if count < 1 or offset < 0 or offset + count > records.shape[0]:
        raise ValueError('Requested segment is outside the saved post-burn-in interactions')
    sample = records[offset:offset + count, trajectory - 1]
    n = np.arange(start, start + count)
    s = sample['x_before'] / (1-rho)
    product2 = sample['product2']
    fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
    ax.plot(n, s, color='#94a3b8', lw=.7, zorder=1)
    for mask, color, label in [
        (~product2, '#2563eb', 'Seller chose product 1'),
        (product2, '#16a34a', 'Seller chose product 2'),
    ]:
        ax.scatter(n[mask], s[mask], color=color, s=2 if count > 5000 else 12,
                   label=label, zorder=2)
    ax.set(xlabel='Interaction with A',
           ylabel='State S at interaction start',
           title=f'Customer {trajectory} · p₀ = {p0:g} · ρ = {rho:g} · '
                 f'Interactions {start:,}–{start + count - 1:,}')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
    ax.grid(alpha=.2)
    ax.margins(x=.01, y=.18)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.18), ncol=2, frameon=False)
    output = root / 'figures' / 'individual_states' / f'p0_{p0:g}'
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'individual_states_p0_{p0}_rho_{rho}_path_{trajectory}_start_{start}_n_{count}.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    return path


def individual_main(argv=None):
    parser = argparse.ArgumentParser(description="Plot an exact segment of one saved customer's states and policy choices.")
    parser.add_argument('--root', type=Path, default=Path('forgetting/outputs/policy_and_state_experiments/occupation_diagnostic'))
    parser.add_argument('--p0', type=float, default=.5)
    parser.add_argument('--rho', type=float, default=.95)
    parser.add_argument('--trajectory', type=int, default=1, help='One-based saved trajectory number')
    parser.add_argument('--start', type=int, help='First interaction; defaults to the first saved interaction')
    parser.add_argument('--count', type=int, default=1000)
    args = parser.parse_args(argv)
    print(plot_states(args.root, args.p0, args.rho, args.trajectory, args.start, args.count))


# Generate missing individual histories

def generate_main(argv=None):
    parser = argparse.ArgumentParser(description='Generate state plots for one customer using the existing interaction simulator.')
    parser.add_argument('--config', type=Path, default=CONFIG_DIR / 'config_occupation.json')
    parser.add_argument('--root', type=Path, default=Path('forgetting/outputs/policy_and_state_experiments/occupation_diagnostic'))
    parser.add_argument('--p0', type=float, default=.5)
    parser.add_argument('--rhos', type=float, nargs='+', default=[.3, .5, .7, .8, .9])
    parser.add_argument('--start', type=int, default=100001)
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=20260809)
    args = parser.parse_args(argv)
    if args.start < 1 or args.count < 1:
        parser.error('start and count must be positive')
    config = json.loads(args.config.read_text())
    primitives = replace(Primitives.from_mapping(config['seller']), p0=args.p0)
    for rho in args.rhos:
        directory = args.root / 'data' / f'p0_{args.p0}_rho_{rho}'
        if not (directory / 'observations.npy').exists():
            result = simulate(primitives, rho, paths=1,
                interactions=args.start + args.count - 1, seed=args.seed,
                trace_window=(args.start, args.start + args.count - 1),
                **config['policy_solver'])
            trace = result['trace']
            records = np.empty((len(trace), 1), dtype=[
                ('x_before', 'f8'), ('product2', '?'), ('success', '?')])
            for field in records.dtype.names:
                records[field][:, 0] = trace[field].to_numpy()
            directory.mkdir(parents=True, exist_ok=True)
            np.save(directory / 'observations.npy', records)
            trace.to_csv(directory / 'individual_state_trace.csv', index=False)
            (directory / 'records_manifest.json').write_text(json.dumps({
                'purpose': 'Single-customer state plot; not a full occupation experiment',
                'shape': list(records.shape), 'burn_in': args.start - 1,
                'row_interaction': 'burn_in + 1 + row_index',
                'column_trajectory': '1 + column_index',
                'state_timing': 'before choice and outcome at this interaction',
                'seed': args.seed, 'p0': args.p0, 'rho': rho,
                'seller': primitives.as_dict(), 'policy_solver': config['policy_solver'],
                'terminal_observation': result['terminal_observation'],
                'initial_state': {'x': 0, 'y': 0},
                'random_numbers': 'Same seed across rho; common outcome uniforms',
            }, indent=2) + '\n')
        print(plot_states(args.root, args.p0, rho, start=args.start, count=args.count), flush=True)


# Initial histories and visit counts

RECORD_DTYPE = [('x_before', 'f8'), ('product2', '?'), ('success', '?')]


def replay_prefix(primitives, rho, solver, *, seed, original_paths, paths, count):
    """Preserve original RNG draw ordering, even when adding more trajectories."""
    if not 1 <= original_paths <= paths or count < 1:
        raise ValueError('Require 1 <= original_paths <= paths and positive count')
    _, policy = solve_observation_layers(primitives, rho, **solver, return_policy=True)
    rng = np.random.default_rng(seed)
    additional_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
    records = np.empty((count, paths), dtype=RECORD_DTYPE)
    x = np.zeros(paths)
    observations = np.zeros(paths, dtype=np.int64)
    for row in range(count):
        product2, _ = _policy_and_demand(observations, x, rho=rho, policy=policy,
                                        action_tolerance=solver['action_tolerance'])
        uniforms = np.r_[rng.random(original_paths), additional_rng.random(paths-original_paths)]
        success = uniforms < np.where(product2, primitives.p2, primitives.p1)
        records['x_before'][row] = x
        records['product2'][row] = product2
        records['success'][row] = success
        x = rho*x + (1-rho)*success
        observations += 1
    return records


def histogram(ax, states, edges, title):
    counts, _ = np.histogram(states, bins=edges)
    if counts.sum() != states.size:
        raise ValueError('Histogram must account for every state visit')
    ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge',
           color='#2563eb', linewidth=0)
    ax.set(title=title, xlim=(edges[0], edges[-1]))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[['top', 'right']].set_visible(False)
    return counts


def plot_initial_states(source: Path, output: Path, count: int = 20_000, minimum_paths: int = 20,
        reuse_records: bool = False):
    manifests = sorted((source/'data').glob('*/records_manifest.json'))
    if not manifests:
        raise ValueError(f'No saved experiment manifests under {source}')
    for manifest in manifests:
        meta = json.loads(manifest.read_text())
        config = meta.get('config', meta)
        p0, rho = meta['p0'], meta['rho']
        edges = np.linspace(0, 1, 101) / (1-rho)
        primitives = replace(Primitives.from_mapping(config['seller']), p0=p0)
        original_paths = meta.get('original_paths', meta['shape'][1])
        paths = max(minimum_paths, meta['shape'][1])
        seed = int(meta['seed'])
        if reuse_records:
            saved = np.load(manifest.parent/'observations.npy', mmap_mode='r')
            if meta['burn_in'] != 0 or saved.shape[0] < count or saved.shape[1] != paths:
                raise ValueError(f'Saved records do not cover the requested prefix: {manifest}')
            records = saved[:count]
        else:
            print(f'Replaying p0={p0:g}, rho={rho:g}: {count:,} interactions, {paths} trajectories', flush=True)
            records = replay_prefix(primitives, rho, config['policy_solver'], seed=seed,
                                    original_paths=original_paths, paths=paths, count=count)
        directory = output/'data'/manifest.parent.name
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory/'observations.npy', records)
        (directory/'records_manifest.json').write_text(json.dumps({
            'shape': list(records.shape), 'burn_in': 0, 'seed': seed,
            'p0': p0, 'rho': rho, 'seller': primitives.as_dict(),
            'policy_solver': config['policy_solver'], 'source_manifest': str(manifest.resolve()),
            'original_paths': original_paths, 'initial_state': {'x': 0, 'y': 0},
            'row_interaction': '1 + row_index', 'column_trajectory': '1 + column_index',
            'state_timing': 'before choice and outcome at this interaction',
            'random_numbers': 'Original paths retain source RNG ordering; added paths use SeedSequence([seed, 1])',
            'stored_state': 'x_before = (1-rho)*S before decision',
            'plotted_state': 'S = x_before/(1-rho)',
            'histogram': {'bins': 100, 'state': 'S', 'range': [0, 1/(1-rho)], 'trajectory': 1, 'weighting': 'Raw visit counts for trajectory 1', 'total_visits': count},
        }, indent=2)+'\n')
        print(plot_states(output, p0, rho, start=1, count=count), flush=True)
    plot_state_distributions(output)


def extend_initial_states(root: Path, p0_values, count=2000):
    """Extend the saved p0=0.5 experiments using common outcome random draws."""
    baseline = []
    for manifest in sorted((root / 'data').glob('*/records_manifest.json')):
        meta = json.loads(manifest.read_text())
        if meta['p0'] == .5:
            baseline.append((manifest, meta))
    if not baseline:
        raise ValueError('Require saved p0=0.5 experiments as the baseline')
    values = sorted(set(p0_values))
    if not values or any(not 0 < value < 1 for value in values) or count < 1:
        raise ValueError('Require p0 values in (0, 1) and a positive count')
    for p0 in values:
        for _, meta in baseline:
            destination = root / 'data' / f'p0_{p0}_rho_{meta["rho"]}'
            if destination.exists():
                raise ValueError(f'Experiment already exists: {destination}')
    for p0 in values:
        for manifest, meta in baseline:
            rho = meta['rho']
            print(f'Simulating p0={p0:g}, rho={rho:g}: {count} interactions', flush=True)
            primitives = replace(Primitives.from_mapping(meta['seller']), p0=p0)
            records = replay_prefix(primitives, rho, meta['policy_solver'],
                seed=meta['seed'], original_paths=meta['original_paths'],
                paths=meta['shape'][1], count=count)
            directory = root / 'data' / f'p0_{p0}_rho_{rho}'
            directory.mkdir(parents=True)
            np.save(directory / 'observations.npy', records)
            updated = {**meta, 'p0': p0, 'seller': primitives.as_dict(),
                'shape': list(records.shape), 'burn_in': 0,
                'baseline_manifest': str(manifest.resolve()),
                'random_numbers': 'Same seed and draw ordering as the p0=0.5 baseline at this rho; common outcome uniforms across p0',
                'histogram': {**meta['histogram'], 'total_visits': count}}
            updated.pop('source_manifest', None)
            (directory / 'records_manifest.json').write_text(json.dumps(updated, indent=2)+'\n')
            plot_states(root, p0, rho, start=1, count=count)
    plot_state_distributions(root)


def shade_product2(ax, regions):
    """Regions use original S units and the mature simulation-policy tolerance."""
    for lower, upper in regions:
        ax.axvspan(lower, upper, color='#16a34a', alpha=.18, zorder=0)
    if regions:
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor='#16a34a', alpha=.18,
                  label='Product 2 · long-run policy')], frameon=False, fontsize=9)


def plot_state_distributions(root: Path):
    """Refresh saved-prefix histograms without changing histories or their seeds."""
    panels = []
    for manifest in sorted((root / 'data').glob('*/records_manifest.json')):
        meta = json.loads(manifest.read_text())
        config = meta.get('config', meta)
        p0, rho = meta['p0'], meta['rho']
        records = np.load(manifest.parent / 'observations.npy', mmap_mode='r')
        solver = config['policy_solver']
        primitives = replace(Primitives.from_mapping(config['seller']), p0=p0)
        solution = solve_boundary(primitives, rho,
            central_nodes=solver.get('nodes', 1001), outer_nodes=2, half_widths=1e6,
            tolerance=solver.get('tolerance', 1e-10),
            max_iterations=solver.get('max_iterations', 100000))
        tolerance = solver.get('action_tolerance', 1e-8)
        regions = [(lo/(1-rho), hi/(1-rho))
                   for lo, hi, action in policy_intervals(solution, tolerance)
                   if action == 'product2']
        edges = np.linspace(0, 1, 101) / (1-rho)
        first, count = meta['burn_in'] + 1, records.shape[0]
        fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
        counts = histogram(ax, records['x_before'][:, 0] / (1-rho), edges,
            f'Visited states · p₀ = {p0:g} · ρ = {rho:g}\n'
            f'Interactions {first:,}–{first+count-1:,} · Customer 1')
        shade_product2(ax, regions)
        ax.set(xlabel='State S at interaction start', ylabel='Number of visits')
        destination = root / 'figures' / 'state_distributions' / f'p0_{p0:g}'
        destination.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination / f'state_distribution_p0_{p0}_rho_{rho}.png', dpi=180)
        plt.close(fig)
        np.savetxt(manifest.parent / 'state_histogram.csv',
            np.column_stack((edges[:-1], edges[1:], counts)), delimiter=',',
            header='left,right,visits', comments='')
        (manifest.parent / 'histogram_policy_regions.json').write_text(json.dumps({
            'policy': 'long-run boundary policy; initial-period policy may differ',
            'state': 'S = x_before/(1-rho)', 'action_tolerance': tolerance,
            'product2_intervals': regions, 'policy_solver': solver,
        }, indent=2) + '\n')
        panels.append((p0, rho, counts, edges, regions, first, count))
    if not panels:
        raise ValueError(f'No saved histories under {root}')
    rows = (len(panels)+2)//3
    fig, axes = plt.subplots(rows, 3, figsize=(13, 3*rows),
                             layout='constrained', squeeze=False)
    for ax, (p0, rho, counts, edges, regions, first, count) in zip(axes.flat, panels):
        ax.bar(edges[:-1], counts, width=np.diff(edges), align='edge', color='#2563eb', linewidth=0)
        shade_product2(ax, regions)
        ax.set(title=f'p₀ = {p0:g} · ρ = {rho:g}', xlim=(edges[0], edges[-1]))
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.spines[['top', 'right']].set_visible(False)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    windows = {(panel[-2], panel[-1]) for panel in panels}
    title = 'Visited states · Customer 1'
    if len(windows) == 1:
        first, count = windows.pop()
        title += f' · Interactions {first:,}–{first+count-1:,}'
    fig.suptitle(title)
    fig.supxlabel('State S at interaction start')
    fig.supylabel('Number of visits')
    fig.savefig(root / 'figures' / 'state_distributions' / 'all_state_distributions.png', dpi=180)
    plt.close(fig)


def draw_kappa_histogram(ax, root: Path, p0=.5, rho=.3, depth=5):
    """Overlay a finite outer cover of Kappa across the full histogram height."""
    from matplotlib.patches import Patch

    if not 1 <= depth <= 12:
        raise ValueError('depth must be between 1 and 12')
    directory = root / 'data' / f'p0_{p0}_rho_{rho}'
    meta = json.loads((directory / 'records_manifest.json').read_text())
    policy = json.loads((directory / 'histogram_policy_regions.json').read_text())
    records = np.load(directory / 'observations.npy', mmap_mode='r')
    edges = np.linspace(0, 1, 101) / (1-rho)
    first, count = meta['burn_in'] + 1, records.shape[0]
    counts = histogram(ax, records['x_before'][:, 0] / (1-rho), edges,
        f'Visited states · p₀ = {p0:g} · ρ = {rho:g}\n'
        f'Interactions {first:,}–{first+count-1:,} · Customer 1')
    shade_product2(ax, policy['product2_intervals'])
    cover = reachable_cover(rho, depth) / (1-rho)
    # Axes-relative height keeps the set overlay spanning the entire histogram.
    ax.broken_barh([(lo, hi-lo) for lo, hi in cover], (0, 1),
                  transform=ax.get_xaxis_transform(), facecolors='#9333ea',
                  edgecolors='none', alpha=.30, zorder=3)
    handles = []
    if policy['product2_intervals']:
        handles.append(Patch(facecolor='#16a34a', alpha=.18,
                             label='Product 2 · long-run policy'))
    label = rf'$\mathcal{{K}}_\rho$ · depth-{depth} outer cover' if rho < .5 else r'$\mathcal{K}_\rho$'
    handles.append(Patch(facecolor='#9333ea', alpha=.30, label=label))
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1, .91), frameon=False, fontsize=9)
    ax.set(xlabel='State S at interaction start', ylabel='Number of visits',
           ylim=(0, counts.max()*1.18))


def plot_kappa_preview(root: Path, p0=.5, rho=.3, depth=5):
    fig, ax = plt.subplots(figsize=(10, 5.5), layout='constrained')
    draw_kappa_histogram(ax, root, p0, rho, depth)
    destination = root / 'figures' / 'state_distributions' / 'kappa' / f'p0_{p0:g}'
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f'state_distribution_p0_{p0}_rho_{rho}_kappa_preview.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_all_kappa_histograms(root: Path, depth=5):
    pairs = []
    for manifest in sorted((root / 'data').glob('*/records_manifest.json')):
        meta = json.loads(manifest.read_text())
        pairs.append((meta['p0'], meta['rho']))
    if not pairs:
        raise ValueError(f'No saved histories under {root}')
    for p0, rho in pairs:
        print(plot_kappa_preview(root, p0, rho, depth))
    rows = (len(pairs)+2)//3
    fig, axes = plt.subplots(rows, 3, figsize=(15, 4*rows),
                             layout='constrained', squeeze=False)
    for ax, (p0, rho) in zip(axes.flat, pairs):
        draw_kappa_histogram(ax, root, p0, rho, depth)
    for ax in list(axes.flat)[len(pairs):]:
        ax.set_visible(False)
    path = root / 'figures' / 'state_distributions' / 'kappa' / 'all_state_distributions_kappa.png'
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(path)


def distributions_main(argv=None):
    parser = argparse.ArgumentParser(description='Refresh saved state histograms with long-run product-2 regions.')
    parser.add_argument('--root', type=Path, default=Path('forgetting/outputs/policy_and_state_experiments/first_2000'))
    kappa = parser.add_mutually_exclusive_group()
    kappa.add_argument('--kappa-all', action='store_true', help='Save all reachable-set overlays under state_distributions/kappa')
    kappa.add_argument('--kappa-preview', action='store_true', help='Create a separate single-pair plot with a reachable-set overlay')
    parser.add_argument('--p0', type=float, default=.5)
    parser.add_argument('--rho', type=float, default=.3)
    parser.add_argument('--depth', type=int, default=5)
    args = parser.parse_args(argv)
    if args.kappa_all:
        plot_all_kappa_histograms(args.root, args.depth)
    elif args.kappa_preview:
        print(plot_kappa_preview(args.root, args.p0, args.rho, args.depth))
    else:
        plot_state_distributions(args.root)


def initial_main(argv=None):
    parser = argparse.ArgumentParser(description='Replay saved experiment prefixes and plot state histories and visit histograms.')
    parser.add_argument('--source', type=Path, default=Path('forgetting/outputs/policy_and_state_experiments/occupation_diagnostic'))
    parser.add_argument('--output', type=Path, default=Path('forgetting/outputs/policy_and_state_experiments/first_20000'))
    parser.add_argument('--count', type=int, default=20_000)
    parser.add_argument('--paths', type=int, default=20)
    parser.add_argument('--reuse-records', action='store_true',
                        help='Plot the prefix of saved records starting at interaction 1')
    args = parser.parse_args(argv)
    if args.count < 1 or args.paths < 1:
        parser.error('count and paths must be positive')
    if args.source.resolve() == args.output.resolve():
        parser.error('output must differ from source to preserve the original records')
    plot_initial_states(args.source, args.output, args.count, args.paths, args.reuse_records)


def extend_initial_main(argv=None):
    parser = argparse.ArgumentParser(description='Run new p0 values using the saved p0=0.5 initial-experiment settings.')
    parser.add_argument('--root', type=Path, default=Path('forgetting/outputs/policy_and_state_experiments/first_2000'))
    parser.add_argument('--p0-values', type=float, nargs='+', required=True)
    parser.add_argument('--count', type=int, default=2000)
    args = parser.parse_args(argv)
    extend_initial_states(args.root, args.p0_values, args.count)


# Observed states and mature policy

def plot_overlap(root: Path, p0=.5, rho=.95, trajectory=1):
    directory = root / 'data' / f'p0_{p0}_rho_{rho}'
    meta = json.loads((directory / 'records_manifest.json').read_text())
    records = np.load(directory / 'observations.npy', mmap_mode='r')
    if not 1 <= trajectory <= records.shape[1]:
        raise ValueError('Trajectory outside the saved range')
    records = records[:, trajectory - 1]
    x, action = records['x_before'], records['product2']
    curve = pd.read_csv(directory / 'simulation_mature_boundary.csv')
    tolerance = meta.get('policy_solver', meta.get('config', {}).get('policy_solver', {}))['action_tolerance']
    predicted = np.interp(x, curve.x, curve.advantage) > tolerance
    if not np.array_equal(predicted, action):
        raise ValueError('Saved choices do not match the mature policy; this plot requires mature records')
    solution = SimpleNamespace(coordinate=curve.x.to_numpy(), advantage=curve.advantage.to_numpy())
    regions = [(lo, hi) for lo, hi, a in policy_intervals(solution, tolerance) if a == 'product2']
    # Split bins at policy boundaries so color never extends across a cutoff.
    edges = np.unique(np.r_[np.linspace(0, 1, 201),
                            [cut for region in regions for cut in region]])
    count1, _ = np.histogram(x[~action], edges)
    count2, _ = np.histogram(x[action], edges)
    assert count1.sum() + count2.sum() == len(x)
    fig, (policy_ax, hist_ax) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={'height_ratios': [1, 3]}, layout='constrained')
    fig.suptitle(f'Estados visitados y decisión óptima · p₀ = {p0:g} · ρ = {rho:g}', fontsize=15)
    policy_ax.set_title('Dónde la política elige el producto 2, aunque el cliente no llegue allí', fontsize=11)
    policy_ax.broken_barh([(lo, hi-lo) for lo, hi in regions], (.25, .5), facecolors='#16a34a')
    for lo, hi in regions:
        policy_ax.text((lo+hi)/2, .5, 'Producto 2', ha='center', va='center', color='white')
        for cut in (lo, hi):
            policy_ax.text(cut, .87, f'{cut:.5f}', ha='center', fontsize=10)
            for ax in (policy_ax, hist_ax):
                ax.axvline(cut, color='#15803d', ls='--', lw=1)
    policy_ax.set(ylim=(0, 1.2), yticks=[])
    for spine in policy_ax.spines.values():
        spine.set_visible(False)
    hist_ax.bar(edges[:-1], count1 / len(x), width=np.diff(edges), align='edge',
        color='#2563eb', label='Visitas con producto 1', linewidth=0)
    hist_ax.bar(edges[:-1], count2 / len(x), bottom=count1 / len(x), width=np.diff(edges),
        align='edge', color='#16a34a', label='Visitas con producto 2', linewidth=0)
    first = meta['burn_in'] + 1
    hist_ax.set(title=f'Cliente {trajectory}: interacciones {first:,}–{first+len(x)-1:,}',
        xlabel='Estado X antes de la decisión (no es tiempo)',
        ylabel='Porcentaje de visitas por intervalo de X', xlim=(0, 1))
    hist_ax.yaxis.set_major_formatter(PercentFormatter(1))
    hist_ax.grid(axis='y', alpha=.2)
    hist_ax.legend(loc='upper left', frameon=False)
    hist_ax.text(.98, .95, f'Producto 2: {100*action.mean():.2f}% de las visitas\n'
        f'X observado: {x.min():.4f} a {x.max():.4f}\nIntervalos de 0,005, divididos en los cortes',
        transform=hist_ax.transAxes, ha='right', va='top', fontsize=10)
    output = root / 'figures' / 'individual_states' / f'p0_{p0:g}'
    output.mkdir(parents=True, exist_ok=True)
    path = output / f'visited_states_vs_product2_p0_{p0}_rho_{rho}_path_{trajectory}.png'
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    # A simpler view: policy above, actual observed states below, no frequencies.
    fig, ax = plt.subplots(figsize=(11, 3.8), layout='constrained')
    for lo, hi in regions:
        ax.plot([lo, hi], [1, 1], color='#16a34a', lw=13, solid_capstyle='butt')
        for cut in (lo, hi):
            ax.plot([cut, cut], [-.15, 1.2], color='#64748b', ls='--', lw=1)
            ax.text(cut, 1.28, f'{cut:.3f}', ha='center', fontsize=11)
    for mask, color in [(~action, '#2563eb'), (action, '#16a34a')]:
        ax.scatter(x[mask], np.zeros(int(mask.sum())), marker='|', s=110,
                   color=color, alpha=.25, linewidths=1, rasterized=True)
    ax.text(float(x.min()), -.3, f'Mínimo: {x.min():.3f}', ha='center', fontsize=10)
    ax.text(float(x.max()), -.3, f'Máximo: {x.max():.3f}', ha='center', fontsize=10)
    ax.set(xlim=(0, 1), ylim=(-.65, 1.65), yticks=[0, 1],
        yticklabels=['Estados que visitó\nel cliente', 'Estados donde conviene\nel producto 2'],
        xlabel='Estado X', title=f'¿Dónde coinciden? · p₀ = {p0:g} · ρ = {rho:g}')
    ax.tick_params(axis='y', length=0, pad=12)
    for name in ('top', 'left', 'right'):
        ax.spines[name].set_visible(False)
    fig.text(.5, -.15, 'Abajo: verde = visitó el estado y usó producto 2; azul = visitó el estado y usó producto 1.\n'
             f'Cliente {trajectory} · {len(x):,} estados registrados · las marcas se superponen.',
             ha='center', fontsize=10)
    simple_path = output / f'visited_states_simple_p0_{p0}_rho_{rho}_path_{trajectory}.png'
    fig.savefig(simple_path, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(simple_path)
    print(f'{len(x)} states checked; product 2: {int(action.sum())}; regions: {regions}')
    print(path)
    return path


def overlap_main(argv=None):
    parser = argparse.ArgumentParser(description="Compare one customer's saved state occupation with the implemented policy.")
    parser.add_argument('--root', type=Path, default=Path('forgetting/outputs/policy_and_state_experiments/occupation_diagnostic'))
    parser.add_argument('--p0', type=float, default=.5)
    parser.add_argument('--rho', type=float, default=.95)
    parser.add_argument('--trajectory', type=int, default=1)
    args = parser.parse_args(argv)
    plot_overlap(args.root, args.p0, args.rho, args.trajectory)
