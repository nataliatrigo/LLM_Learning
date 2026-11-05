"""Run forgetting Monte Carlo comparisons across rho without the window sweeps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from forgetting.Code.experiments import run_experiments as experiments

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from forgetting.Code.model import Primitives, validate_rho


def rho_values(values) -> list[float]:
    result = sorted(set(map(float, values)))
    if not result:
        raise ValueError("Specify at least one rho")
    for rho in result:
        validate_rho(rho)
    return result


def plot_rho_comparison(periods: pd.DataFrame) -> str:
    """Compare rho curves at fixed p0 using the same unconditional rates."""
    experiments.configure_style()
    p0_values = sorted(periods.p0.unique())
    rhos = sorted(periods.rho.unique())
    colors = plt.colormaps['viridis'](np.linspace(.05, .9, len(rhos)))
    fig, axes = plt.subplots(2, len(p0_values), figsize=(3.3*len(p0_values), 6.5),
                             sharex=True, sharey=True, squeeze=False)
    for column, p0 in enumerate(p0_values):
        for rho, color in zip(rhos, colors, strict=True):
            data = periods[(periods.p0 == p0) & (periods.rho == rho)].sort_values('period')
            for row, metric in enumerate(('a_market_share', 'product2_unconditional_rate')):
                axes[row, column].plot(data.period + 1, data[metric], color=color,
                                       linewidth=1.1, label=rf'$\rho={rho:g}$')
        axes[0, column].set_title(rf'$p_0={p0:g}$')
        axes[1, column].set_xlabel('Period')
    axes[0, 0].set_ylabel('Seller A market share')
    axes[1, 0].set_ylabel('Share of paths using product 2')
    for ax in axes.flat:
        ax.set_ylim(-.02, 1.02)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncols=min(6, len(rhos)))
    fig.suptitle('Forgetting Monte Carlo — comparison across memory persistence')
    fig.tight_layout(rect=(0, .08, 1, .95))
    stem = 'monte_carlo_forgetting_by_rho'
    experiments.save_figure(fig, stem, png_only=True)
    return stem


def run(config: dict, rhos: list[float], output: Path) -> None:
    rhos = rho_values(rhos)
    output = output.resolve()
    base = Primitives.from_mapping({**config['seller'],
        'gamma': config['rho_p0_comparative_statics']['gamma']})
    # An explicit Monte Carlo invocation enables simulation independently of
    # the full experiment runner's simulation toggle.
    effective = {**config, 'simulation': {**config['simulation'],
                                        'enabled': True, 'rhos': rhos}}
    collected = []
    stems = []
    for rho in rhos:
        scenario = {**effective, 'simulation': {**effective['simulation'], 'rho': rho}}
        directory = output / 'data' / f'rho_{rho!r}'
        data_dir = directory / 'data'
        data_dir.mkdir(parents=True, exist_ok=True)
        experiments.OUTPUTS = directory
        experiments.DATA = data_dir
        experiments.FIGURES = output / 'figures'
        results = experiments.run_monte_carlo_comparison(scenario, base)
        for name, frame in zip(('by_period', 'binned', 'path_summary'), results, strict=True):
            frame.to_csv(data_dir / f'monte_carlo_calendar_{name}.csv', index=False)
        stem = f'monte_carlo_forgetting_p0_panels_rho_{rho!r}'
        experiments.plot_monte_carlo_p0_panels(results[0], name=stem, png_only=True)
        (directory / 'manifest.json').write_text(json.dumps({
            'config': scenario, 'figures': [f'../../figures/{stem}.png']}, indent=2) + '\n')
        stems.append(stem)
        collected.append(results[0])
    experiments.OUTPUTS = output
    experiments.DATA = output / 'data'
    experiments.FIGURES = output / 'figures'
    experiments.DATA.mkdir(parents=True, exist_ok=True)
    periods = pd.concat(collected, ignore_index=True)
    periods.to_csv(experiments.DATA / 'monte_carlo_by_rho_by_period.csv', index=False)
    stems.insert(0, plot_rho_comparison(periods))
    (output / 'manifest.json').write_text(json.dumps({
        'config': effective, 'figures': [f'figures/{stem}.png' for stem in stems]},
        indent=2) + '\n')
    print(f'Results: {output}', flush=True)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=experiments.CONFIG_DIR / 'config_policy_comparisons.json')
    parser.add_argument('--rhos', type=float, nargs='+', help='override simulation.rhos')
    parser.add_argument('--output-dir', type=Path,
                        default=experiments.ROOT / 'outputs/policy_and_state_experiments/monte_carlo')
    args = parser.parse_args(argv)
    config = experiments.read_config(args.config)
    values = args.rhos if args.rhos is not None else config['simulation'].get(
        'rhos', [config['simulation']['rho']])
    try:
        rhos = rho_values(values)
    except ValueError as error:
        parser.error(str(error))
    run(config, rhos, args.output_dir)


if __name__ == '__main__':
    main()
