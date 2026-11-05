"""Add the product-2 time plot using existing raw paths, without new simulations."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from oblivious.oe_p0.code.run_experiment import load_bundle, save_bundle, write_csv
from oblivious.equilibrium_comparison.code.analysis import product_usage_evolution
from oblivious.equilibrium_comparison.code.reporting import MODES, plot_product_usage


ROOT = Path(__file__).resolve().parents[3]


def add_product_usage_plot(data, figures, *, periods=10000, smoothing_window=100,
                           early_periods=500):
    """Read all saved replications from t=0 and save a separate report supplement."""
    if periods < 1 or early_periods < 1:
        raise ValueError('Plot horizons must be positive.')
    data, figures = Path(data), Path(figures)
    manifest_path = data / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['status'] != 'complete':
        raise ValueError('Use a completed experiment with saved raw trajectories.')
    config = manifest['configuration']
    analysis_base = data / 'analysis'
    analysis = load_bundle(analysis_base)
    stationary_frequencies = {key: analysis['predicted'][key]['product2_market']
                              for key in ('oe', 'p0')}
    curves, sources = {}, {}
    initial = None
    for mode in MODES:
        if mode not in config['modes']:
            continue
        folder = data / 'raw' / f'n{config["grid"]:03d}_{mode}'
        metadata_path = folder / 'metadata.json'
        metadata = json.loads(metadata_path.read_text())['metadata']
        if (metadata['mode'] != mode or metadata['replications'] != config['replications']
                or metadata['assignments'] != {'oe': ['oe', 'oe'], 'p0': ['p0', 'p0']}):
            raise ValueError(f'Incompatible raw simulation metadata: {metadata_path}')
        if initial is None:
            initial = metadata['initial']
        if metadata['initial'] != initial:
            raise ValueError('All modes must use the same initial reputations.')
        curves[mode], sources[mode] = {}, {
            'metadata': str(metadata_path.resolve()),
            'metadata_sha256': sha256(metadata_path.read_bytes()).hexdigest(),
            'profiles': {},
        }
        for key in ('oe', 'p0'):
            path = folder / f'{key}.npz'
            with np.load(path, allow_pickle=False) as stored:
                product = stored['product']
            if (product.ndim != 2 or product.shape[0] != config['replications']
                    or product.shape[1] < periods):
                raise ValueError(f'Expected {config["replications"]} complete paths of at least {periods} periods: {path}')
            product = np.ascontiguousarray(product[:, :periods])
            curves[mode][key] = product_usage_evolution(product, smoothing_window=smoothing_window)
            sources[mode]['profiles'][key] = dict(
                path=str(path.resolve()), product_prefix_sha256=sha256(product.tobytes()).hexdigest(),
                shape=list(product.shape), dtype=str(product.dtype))
    if not curves:
        raise ValueError('No supported simulation modes in the saved configuration.')
    supplement = dict(
        product_usage=curves, initial=initial, periods=periods,
        smoothing_window=smoothing_window, early_periods=early_periods,
        layout='For each simulation mode: unsmoothed panel above the moving-average panel.',
        source_manifest=str(manifest_path.resolve()),
        source_manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
        raw_sources=sources, no_new_simulations=True, no_equilibrium_solved=True,
        stationary_reference=dict(
            frequencies=stationary_frequencies,
            definition='sum_ij nu[i,j] * (D[i,j] * policy[i] + (1-D[i,j]) * policy[j]); actual seller selection.',
            scope='Stationary for the original discrete reputation process; a reference benchmark for continuous extensions.',
            source_analysis={suffix: dict(path=str(analysis_base.with_suffix(suffix).resolve()),
                sha256=sha256(analysis_base.with_suffix(suffix).read_bytes()).hexdigest())
                for suffix in ('.json', '.npz')}),
        source_code_sha256={name: sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ('analysis.py', 'reporting.py', 'plot_product_usage.py')},
    )
    save_bundle(data / 'product2_usage_over_time', supplement)
    rows = []
    for mode, profiles in curves.items():
        for key, curve in profiles.items():
            for t in range(periods):
                rows.append(dict(
                    mode=mode, profile=key, period=t, q2=curve['q2'][t],
                    q2_smoothed=curve['smoothed'][t], smoothed_se=curve['se'][t],
                    smoothed_ci_low=curve['ci_low'][t], smoothed_ci_high=curve['ci_high'][t],
                    effective_window=curve['window_size'][t],
                    n_replications=curve['n_replications'],
                    original_stationary_q2=stationary_frequencies[key]))
    write_csv(data / 'product2_usage_over_time.csv', rows)
    plot_product_usage(curves, figures, stationary_frequencies=stationary_frequencies,
                       early_periods=early_periods, initial=initial)
    return supplement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='rho095_n51')
    parser.add_argument('--periods', type=int, default=10000)
    parser.add_argument('--smoothing-window', type=int, default=100)
    parser.add_argument('--early-periods', type=int, default=500)
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('--name must be an existing experiment folder name')
    if min(args.periods, args.smoothing_window, args.early_periods) < 1:
        parser.error('Horizons and smoothing window must be positive')
    data = ROOT / 'oblivious/06_equilibrium_comparison/data' / args.name
    figures = ROOT / 'oblivious/06_equilibrium_comparison/figures' / args.name
    supplement = add_product_usage_plot(data, figures, periods=args.periods,
        smoothing_window=args.smoothing_window, early_periods=args.early_periods)
    for mode, profiles in supplement['product_usage'].items():
        print(f'{mode}: {profiles["oe"]["n_replications"]} replications, {args.periods} periods from t=0')
    print(f'Figure saved: {figures / "14_product2_usage_over_time.png"}')


if __name__ == '__main__':
    main()
