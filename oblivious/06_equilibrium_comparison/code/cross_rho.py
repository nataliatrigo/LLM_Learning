"""Compare saved equilibrium policies across rho, without running any solver.

The primary comparison uses the original finite-grid process and actual Beta
draws for both sellers. All outputs are separate from the single-rho runs.
"""

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np

from oblivious.layout import resolve_saved_path
from oblivious.oe_p0.code.run_experiment import (
    case_tag, load_bundle, now, save_bundle, write_csv, write_json,
)
from oblivious.equilibrium_comparison.code.analysis import interval
from oblivious.equilibrium_comparison.code.inputs import load_pair
from oblivious.equilibrium_comparison.code.simulation import simulate_pair
from oblivious.equilibrium_comparison.code.validate_transients import validate_transients


ROOT = Path(__file__).resolve().parents[3]
RHOS = (.6, .7, .8, .9, .95, .97, .99)
METRICS = ('product2_market', 'seller_profit', 'success_rate')


def source_hashes():
    paths = [ROOT / 'oblivious/06_equilibrium_comparison/code' / name for name in
             ('cross_rho.py', 'simulation.py', 'inputs.py', 'analysis.py', 'validate_transients.py')]
    paths += list((ROOT / 'oblivious/02_oe_p0/code').glob('*.py'))
    paths += list((ROOT / 'oblivious/01_oe_distribution/code').glob('*.py'))
    return {str(p.relative_to(ROOT)): sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if not p.name.startswith('test_')}


def summarize_profile(grid, raw, *, start, length):
    """Pool sellers *within* a replication before constructing path-level CIs.

    Reputation observations are pre-selection calendar states, never weighted
    by purchases. Profit averages A and B, including their zero-profit periods.
    """
    end = start + length
    states = raw['before'][:, start:end]
    products = raw['product'][:, start:end]
    successes = raw['success'][:, start:end]
    profits = raw['profit'][:, start:end]
    if states.ndim != 3 or states.shape[1:] != (length, 2):
        raise ValueError('Requested window must contain both sellers in every calendar period.')
    if products.shape != states.shape[:2] or successes.shape != products.shape:
        raise ValueError('Product/success records must contain exactly one purchase per period.')
    if profits.shape != states.shape:
        raise ValueError('Expected calendar-period profits for both sellers.')
    if len(states) < 2 or not np.all((products == 1) | (products == 2)):
        raise ValueError('At least two replications with valid products are required.')
    edges = np.r_[0., (grid[:-1] + grid[1:]) / 2, 1.]
    joints, cdfs = [], []
    for x in states:
        # Binning is only used for the joint display, even in continuous modes.
        joints.append(np.histogram2d(x[:, 0], x[:, 1], bins=(edges, edges))[0] / length)
        cdfs.append(np.searchsorted(np.sort(x.ravel()), grid, side='right') / x.size)
    cdfs = np.asarray(cdfs)
    metrics = dict(product2_market=(products == 2).mean(axis=1),
                   seller_profit=profits.mean(axis=(1, 2)),
                   success_rate=successes.mean(axis=1))
    cdf_intervals = [interval(values) for values in cdfs.T]
    return dict(
        replications=[dict(replication=i, **{key: float(values[i]) for key, values in metrics.items()})
                      for i in range(len(states))],
        summary={key: interval(values) for key, values in metrics.items()},
        distribution=dict(
            grid=grid.copy(), edges=edges, joint=np.mean(joints, axis=0),
            cdf=np.array([r['mean'] for r in cdf_intervals]),
            cdf_low=np.clip([r['ci_low'] for r in cdf_intervals], 0, 1),
            cdf_high=np.clip([r['ci_high'] for r in cdf_intervals], 0, 1),
            cdf_replications=cdfs, joint_replications=np.asarray(joints),
        ),
    )


def reuse_raw(directory, pair, configuration):
    """Reuse a matching baseline path prefix only after provenance checks."""
    if directory is None:
        return None, None
    directory = Path(directory)
    manifest_path = directory / 'manifest.json'
    if not manifest_path.exists():
        return None, None
    manifest = json.loads(manifest_path.read_text())
    config = manifest.get('configuration', {})
    fields = ('rho', 'grid', 'periods', 'burn_in', 'replications', 'seed', 'oe_run', 'p0_run')
    expected = dict(configuration, rho=pair['params'].rho)
    if manifest.get('status') != 'complete' or any(config.get(k) != expected[k] for k in fields):
        return None, None
    if manifest.get('input_audits', {}).get(str(len(pair['grid']))) != pair['audit']:
        return None, None
    folder = directory / 'raw' / f"n{len(pair['grid']):03d}_{configuration['mode']}"
    if not (folder / 'metadata.json').exists():
        return None, None
    metadata = json.loads((folder / 'metadata.json').read_text())
    meta = metadata['metadata']
    if meta.get('initial') != [.5, .5] or meta.get('mode') != configuration['mode']:
        return None, None
    for field in ('periods', 'burn_in', 'replications', 'seed'):
        if meta.get(field) != configuration[field]:
            return None, None
    for path, digest in manifest.get('source_sha256', {}).items():
        if path in source_hashes() and source_hashes()[path] != digest:
            return None, None
    paths = [folder / (key + '.npz') for key in ('oe', 'p0')]
    if not all(p.exists() for p in paths):
        return None, None
    # Only fields required by this comparison are materialized.
    profiles = {}
    for key, path in zip(('oe', 'p0'), paths):
        with np.load(path, allow_pickle=False) as arrays:
            profiles[key] = {field: arrays[field] for field in ('before', 'product', 'success', 'profit')}
    provenance = dict(kind='reused_existing_raw', manifest=str(manifest_path),
                      manifest_sha256=sha256(manifest_path.read_bytes()).hexdigest(),
                      raw_files={str(p): sha256(p.read_bytes()).hexdigest() for p in paths},
                      metadata=metadata)
    return profiles, provenance


def analyze_case(pair, configuration, reuse_directory=None):
    started = time.monotonic()
    profiles, provenance = reuse_raw(reuse_directory, pair, configuration)
    if profiles is None:
        simulation = simulate_pair(
            pair, periods=configuration['periods'], burn_in=configuration['burn_in'],
            replications=configuration['replications'], seed=configuration['seed'],
            mode=configuration['mode'], short_periods=0, initial=(.5, .5),
            progress=lambda done, total: print(f'    {done:,}/{total:,} periods', flush=True),
        )
        profiles = simulation['profiles']
        provenance = dict(kind='existing_simulate_pair', metadata=simulation['metadata'],
                          replication_seeds=simulation['replication_seeds'])
    else:
        print('    Reusing matching saved raw simulations', flush=True)
    result = dict(rho=pair['params'].rho, verified=True, audit=pair['audit'],
                  parameters=asdict(pair['params']), grid=pair['grid'].copy(), profiles={},
                  simulation_provenance=provenance)
    for key, raw in profiles.items():
        profile = pair['profiles'][key]
        result['profiles'][key] = summarize_profile(
            pair['grid'], raw, start=configuration['burn_in'], length=configuration['periods'])
        predicted = profile['predicted']
        result['profiles'][key].update(
            policy=profile['policy'].copy(), stationary_joint=profile['nu'].copy(),
            stationary_cdf=np.cumsum((profile['nu'].sum(axis=0) + profile['nu'].sum(axis=1)) / 2),
            predicted=dict(product2_market=predicted['product2_market'],
                           seller_profit=predicted['industry_profit'] / 2,
                           success_rate=predicted['success_rate']),
            source=profile['source'],
        )
    # Free event-level arrays before deterministic propagation or the next rho.
    del profiles
    if 'simulation' in locals():
        del simulation
    print('    Checking finite-window expectations under the saved joint kernels', flush=True)
    if configuration['mode'] == 'grid':
        transient = validate_transients(
            pair, {'comparison': (configuration['burn_in'], configuration['periods'])},
            checkpoints=(0, configuration['burn_in'], configuration['burn_in'] + configuration['periods']))
        result['transient_validation'] = transient
        for key, item in result['profiles'].items():
            validation = transient['profiles'][key]
            expected = validation['windows']['comparison']['expected_metrics']
            item['finite_window_expected'] = dict(
                product2_market=expected['product2_market'],
                seller_profit=expected['industry_profit'] / 2,
                success_rate=expected['success_rate'])
            item['finite_window_joint_l1_from_stationary'] = validation['windows']['comparison']['l1_distance_to_stationary']
    result['elapsed_seconds'] = time.monotonic() - started
    return result


def write_tables(results, directory):
    summary, replications, distributions, status, differences = [], [], [], [], []
    for result in results:
        rho = result['rho']
        status.append(dict(rho=rho, verified=result['verified'], reason=result.get('reason', ''),
                           candidate_selection='saved_selected'))
        if not result['verified']:
            continue
        for key, item in result['profiles'].items():
            base = dict(rho=rho, profile=key, verified=True)
            for metric, estimate in item['summary'].items():
                finite = item.get('finite_window_expected', {}).get(metric)
                prediction = item['predicted'][metric]
                summary.append(dict(
                    **base, metric=metric, **estimate, stationary_prediction=prediction,
                    finite_window_expectation=finite,
                    initialization_bias=None if finite is None else finite - prediction,
                    finite_window_joint_l1_from_stationary=item.get('finite_window_joint_l1_from_stationary'),
                    ci_contains_stationary_prediction=estimate['ci_low'] <= prediction <= estimate['ci_high']))
            replications.extend(dict(**base, **r) for r in item['replications'])
            d = item['distribution']
            distributions.extend(dict(**base, reputation=float(x), simulated_cdf=float(d['cdf'][i]),
                                      ci_low=float(d['cdf_low'][i]), ci_high=float(d['cdf_high'][i]),
                                      stationary_cdf=float(item['stationary_cdf'][i]))
                                 for i, x in enumerate(result['grid']))
        for metric in METRICS:
            paired = np.array([r[metric] for r in result['profiles']['oe']['replications']])
            paired -= np.array([r[metric] for r in result['profiles']['p0']['replications']])
            differences.append(dict(rho=rho, metric=metric, contrast='OE distribution minus OE p0', **interval(paired)))
    write_csv(directory / 'summary.csv', summary)
    write_csv(directory / 'replications.csv', replications)
    write_csv(directory / 'reputation_cdfs.csv', distributions)
    write_csv(directory / 'verification_status.csv', status)
    write_csv(directory / 'paired_differences.csv', differences)


def write_readme(results, configuration, data, figures):
    lines = [
        'CROSS-RHO COMPARISON OF SAVED POLICIES', '',
        f"Rates: {', '.join(format(r, '.3g') for r in configuration['rhos'])}.",
        f"{configuration['replications']} independent paired paths; {configuration['periods']:,} retained calendar periods; "
        f"{configuration['burn_in']:,} burn-in periods; seed {configuration['seed']}; common initial state (0.5, 0.5).",
        f"Simulation mode: {configuration['mode']}; saved grid: {configuration['grid']} points.",
        'Both actual Beta posterior draws select the seller in every simulation. Only the chosen seller updates.',
        'Same path-level uniforms across models and rates; independent SeedSequence children across replications.',
        'No equilibrium was solved or changed. The original saved selected candidates were used without switching branches.',
        'Verified means passing the existing saved-solution numerical checks; it is not a uniqueness or exact-equilibrium proof.',
        '',
        'UNCERTAINTY AND STATIONARITY',
        '95% Student-t Monte Carlo intervals use independent path means, never individual time observations.',
        'CDF bands are pointwise. Sellers are pooled within each path before constructing CDF intervals.',
        'Average seller profit is (profit_A + profit_B)/2 per calendar period, including unselected periods.',
        'Success is the realized success fraction of all market purchases. There is one purchase per period.',
        'Circles/solid lines represent simulation estimates. Hollow diamonds/dashed lines represent saved stationary predictions.',
        'The rho axes show the discrete evaluated values with equal spacing; connecting lines guide the eye.',
        'Monte Carlo intervals describe finite-window simulation uncertainty and do not include initialization bias.',
        'summary.csv separates deterministic finite-window expectations, stationary predictions and initialization bias.',
        'A small stationary residual does not establish rapid mixing. Check the joint-law L1 distances below.',
        '',
        'FIGURES',
        '01: eight panels of saved own-reputation product-2 probabilities. Steps only guide the eye between saved nodes.',
        '02: product-2 share of all purchases, simulated with CIs and saved stationary predictions.',
        '03: average seller profit per calendar period and customer success rate, both with CIs and predictions.',
        '04: eight panels of pooled calendar-time reputation CDFs, simulated with pointwise CIs and saved predictions.',
        '05: 8x2 saved stationary joint laws, which are stationary under the verified finite-grid kernels.',
        'S01: 8x2 empirical joint laws from the common-start simulations; these can retain initialization bias.',
        'Both joint figures use identical axes, a shared viridis log probability-mass scale, and colored model headers.',
        'Green identifies OE distribution; orange identifies OE p0. Heatmaps use the same purple-green-yellow palette and sans-serif type as the original single-rho figures.',
        '',
        'VERIFICATION AND FINITE-WINDOW DIAGNOSTICS',
    ]
    for result in results:
        if not result['verified']:
            lines.append(f"rho={result['rho']}: UNVERIFIED / UNAVAILABLE. {result['reason']}")
            continue
        lines.append(f"rho={result['rho']}: both selected candidates verified.")
        for key, item in result['profiles'].items():
            distance = item.get('finite_window_joint_l1_from_stationary')
            if distance is not None:
                bias = item['finite_window_expected']['product2_market'] - item['predicted']['product2_market']
                lines.append(f'  {key}: expected window joint-law L1 distance {distance:.6g}; product-2 initialization bias {bias:+.6g}.')
    lines += ['', f'Data: {data}', f'Figures: {figures}',
              'Per-rho checkpoints preserve numeric data, source hashes, input audits and simulation provenance.',
              'Event-level records for new rates are reproducible from the seeds and existing simulate_pair; they are not duplicated here.',
              'Existing single-rho data and figures were left untouched.', '']
    (data / 'README.txt').write_text('\n'.join(lines))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='cross_rho_n51')
    parser.add_argument('--output', type=Path, default=ROOT / 'oblivious/06_equilibrium_comparison/data')
    parser.add_argument('--figures-output', type=Path, default=ROOT / 'oblivious/06_equilibrium_comparison/figures')
    parser.add_argument('--rhos', nargs='+', type=float, default=list(RHOS))
    parser.add_argument('--grid', type=int, default=51)
    parser.add_argument('--oe-run', default='rho_forgetting')
    parser.add_argument('--p0-run', default='rho_forgetting_mixed')
    parser.add_argument('--periods', type=int, default=100000)
    parser.add_argument('--burn-in', type=int, default=10000)
    parser.add_argument('--replications', type=int, default=20)
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--mode', choices=('grid', 'nearest', 'linear'), default='grid')
    parser.add_argument('--reuse-existing', type=Path, default=ROOT / 'oblivious/06_equilibrium_comparison/data/rho095_n51')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plots-only', action='store_true')
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in ('.', '..'):
        parser.error('--name must be a single folder name')
    if args.replications < 2 or args.periods < 1 or args.burn_in < 0 or args.grid < 3:
        parser.error('Require >=2 replications, positive periods, nonnegative burn-in, and >=3 grid nodes')
    if len(set(args.rhos)) != len(args.rhos) or any(not 0 < r < 1 for r in args.rhos):
        parser.error('Rates must be distinct and strictly between zero and one')
    args.rhos = sorted(args.rhos)
    return args


def main():
    args = arguments()
    data, figures = args.output / args.name, args.figures_output / args.name
    manifest_path = data / 'manifest.json'
    if args.plots_only or args.resume:
        if not manifest_path.exists():
            raise SystemExit('No saved cross-rho comparison at this destination')
        manifest = json.loads(manifest_path.read_text())
        configuration = manifest['configuration']
        figures = resolve_saved_path(configuration['figures_output'], project_root=ROOT) / configuration['name']
        if not args.plots_only and source_hashes() != manifest['source_sha256']:
            raise SystemExit('Simulation/analysis sources changed; choose a new name')
    else:
        if data.exists() or figures.exists():
            raise SystemExit('Existing outputs protected; use --resume, --plots-only, or a new --name')
        configuration = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                         if k not in ('resume', 'plots_only')}
        manifest = dict(status='running', started_utc=now(), configuration=configuration,
                        source_sha256=source_hashes(), completed=[])
        data.mkdir(parents=True)
        (data / 'checkpoints').mkdir()
        write_json(manifest_path, manifest)
    results = []
    try:
        for rho in configuration['rhos']:
            tag = case_tag(rho, configuration['grid'])
            base = data / 'checkpoints' / tag
            if tag in manifest['completed']:
                result = load_bundle(base)
                if not args.plots_only and result['verified']:
                    pair = load_pair(ROOT, rho=rho, n=configuration['grid'],
                                     oe_run=configuration['oe_run'], p0_run=configuration['p0_run'])
                    if pair['audit'] != result['audit']:
                        raise RuntimeError(f'Saved equilibrium inputs changed for rho={rho}')
                print(f'Loaded {tag}', flush=True)
            else:
                if args.plots_only:
                    raise RuntimeError(f'Comparison incomplete: missing {tag}; use --resume')
                print(f'Comparing {tag}', flush=True)
                try:
                    pair = load_pair(ROOT, rho=rho, n=configuration['grid'],
                                     oe_run=configuration['oe_run'], p0_run=configuration['p0_run'])
                except (FileNotFoundError, ValueError) as error:
                    result = dict(rho=rho, verified=False, reason=str(error), profiles={})
                    print(f'  UNVERIFIED / UNAVAILABLE: {error}', flush=True)
                else:
                    result = analyze_case(pair, configuration, configuration['reuse_existing'])
                    # Reject concurrent changes to a saved policy or its audit.
                    checked = load_pair(ROOT, rho=rho, n=configuration['grid'],
                                        oe_run=configuration['oe_run'], p0_run=configuration['p0_run'])
                    if checked['audit'] != result['audit']:
                        raise RuntimeError(f'Saved equilibrium inputs changed during rho={rho}')
                    print(f'  Completed in {result["elapsed_seconds"]:.1f}s', flush=True)
                save_bundle(base, result)
                manifest['completed'].append(tag)
                write_json(manifest_path, manifest)
            results.append(result)
        if not args.plots_only and source_hashes() != manifest['source_sha256']:
            raise RuntimeError('Simulation/analysis sources changed during the comparison')
        write_tables(results, data)
        write_readme(results, configuration, data, figures)
        from oblivious.equilibrium_comparison.code.cross_rho_figures import make_figures
        make_figures(results, configuration, figures)
        manifest.update(status='complete', finished_utc=now(), no_equilibrium_solved=True,
                        numerical_equilibria_unchanged=True,
                        all_rates_verified=all(r['verified'] for r in results),
                        figure_source_sha256=sha256((ROOT / 'oblivious/06_equilibrium_comparison/code/cross_rho_figures.py').read_bytes()).hexdigest())
        write_json(manifest_path, manifest)
        print(f'Cross-rho comparison complete: {figures}', flush=True)
    except BaseException as error:
        if not args.plots_only:
            manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                            last_error=str(error), stopped_utc=now())
            write_json(manifest_path, manifest)
        raise


if __name__ == '__main__':
    main()
