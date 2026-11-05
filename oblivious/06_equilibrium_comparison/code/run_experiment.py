"""Run a reproducible comparison without solving or modifying either equilibrium."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from oblivious.layout import resolve_saved_path
from oblivious.oe_p0.code.run_experiment import save_bundle, load_bundle, write_json, write_csv, now
from oblivious.equilibrium_comparison.code.inputs import load_pair
from oblivious.equilibrium_comparison.code.simulation import simulate_pair
from oblivious.equilibrium_comparison.code.analysis import summarize_window, policy_comparison, product_usage_evolution

ROOT = Path(__file__).resolve().parents[3]


def source_hashes():
    files = list((ROOT/'oblivious/06_equilibrium_comparison/code').glob('*.py'))
    files += list((ROOT/'oblivious/02_oe_p0/code').glob('*.py'))
    files += list((ROOT/'oblivious/01_oe_distribution/code').glob('*.py'))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files) if not p.name.startswith('test_')}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='rho095_n51')
    parser.add_argument('--output', type=Path, default=ROOT/'oblivious/06_equilibrium_comparison/data')
    parser.add_argument('--figures-output', type=Path, default=ROOT/'oblivious/06_equilibrium_comparison/figures')
    parser.add_argument('--rho', type=float, default=.95)
    parser.add_argument('--grid', type=int, default=51)
    parser.add_argument('--oe-run', default='rho_forgetting')
    parser.add_argument('--p0-run', default='rho_forgetting_mixed')
    parser.add_argument('--oe-candidate', type=int)
    parser.add_argument('--p0-candidate', type=int)
    parser.add_argument('--periods', type=int, default=100000)
    parser.add_argument('--burn-in', type=int, default=10000)
    parser.add_argument('--replications', type=int, default=20)
    parser.add_argument('--short-periods', type=int, default=1000)
    parser.add_argument('--usage-periods', type=int, default=10000)
    parser.add_argument('--usage-smoothing-window', type=int, default=100)
    parser.add_argument('--seed', type=int, default=20261017)
    parser.add_argument('--modes', nargs='+', choices=('grid','nearest','linear'), default=['grid','nearest','linear'])
    parser.add_argument('--refinement-grids', nargs='*', type=int, default=[21,31])
    parser.add_argument('--no-stability', action='store_true')
    parser.add_argument('--compress-raw', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name or args.name in ('.','..'):
        parser.error('--name must be a single new folder name')
    if min(args.periods, args.short_periods) < 1 or args.replications < 2 or args.burn_in < 0:
        parser.error('Positive simulation lengths, at least two replications, and nonnegative burn-in required')
    if args.short_periods > args.periods+args.burn_in:
        parser.error('Short-run horizon must fit in the simulated path')
    if args.usage_periods < 1 or args.usage_smoothing_window < 1:
        parser.error('Product-usage horizon and smoothing window must be positive')
    if len(set(args.modes)) != len(args.modes):
        parser.error('Repeated simulation modes')
    return args


def save_raw(folder, simulation, compress=False):
    folder.mkdir(parents=True, exist_ok=True)
    save = np.savez_compressed if compress else np.savez
    for name, arrays in simulation['profiles'].items():
        target = folder/(name+'.npz')
        with target.with_suffix('.npz.tmp').open('wb') as stream:
            save(stream, time=simulation['time'], **arrays)
        target.with_suffix('.npz.tmp').replace(target)
    with (folder/'common_random_numbers.npz.tmp').open('wb') as stream:
        save(stream, uniforms=simulation['uniforms'])
    (folder/'common_random_numbers.npz.tmp').replace(folder/'common_random_numbers.npz')
    write_json(folder/'metadata.json', dict(metadata=simulation['metadata'],
                                          replication_seeds=simulation['replication_seeds']))


def main():
    args = arguments()
    data = args.output/args.name
    manifest_path = data/'manifest.json'
    if args.resume:
        if not manifest_path.exists():
            raise SystemExit('No saved experiment to resume')
        manifest = json.loads(manifest_path.read_text())
        if manifest['status'] == 'complete':
            raise SystemExit('Experiment already complete; use a new --name')
        if source_hashes() != manifest['source_sha256']:
            raise SystemExit('Code changed since this comparison started; use a new --name')
        args = argparse.Namespace(**manifest['configuration'], resume=True)
        args.output = resolve_saved_path(args.output, project_root=ROOT)
        args.figures_output = resolve_saved_path(args.figures_output, project_root=ROOT)
    else:
        if data.exists() or (args.figures_output/args.name).exists():
            raise SystemExit('Existing outputs protected; choose a new --name or --resume')
        # Validate all sources before creating the new experiment's output.
        manifest = dict(status='running', started_utc=now(), configuration={
            key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items() if key != 'resume'},
            source_sha256=source_hashes(), completed=[])
    grids = list(dict.fromkeys([args.grid]+args.refinement_grids))
    pairs = {n: load_pair(ROOT, rho=args.rho, n=n, oe_run=args.oe_run, p0_run=args.p0_run,
                         oe_candidate=args.oe_candidate, p0_candidate=args.p0_candidate) for n in grids}
    pair = pairs[args.grid]
    source_audits = {str(n): value['audit'] for n,value in pairs.items()}
    if args.resume and manifest['input_audits'] != source_audits:
        raise SystemExit('Saved equilibrium inputs changed; use a new comparison name')
    manifest['input_audits'] = source_audits
    data.mkdir(parents=True, exist_ok=args.resume)
    (data/'checkpoints').mkdir(exist_ok=True)
    figures = args.figures_output/args.name
    figures.mkdir(parents=True, exist_ok=args.resume)
    write_json(manifest_path, manifest)
    write_json(data/'input_validation.json', source_audits)
    analysis = dict(summary=[], differences=[], replications=[], sensitivity=[], sensitivity_differences=[],
                    distributions={}, samples={}, running={}, product_usage={}, audit=pair['audit'],
                    configuration=manifest['configuration'], policy_comparison=policy_comparison(pair),
                    predicted={key: profile['predicted'] for key, profile in pair['profiles'].items()})
    try:
        for n in grids:
            modes = args.modes if n == args.grid else [m for m in ('grid','nearest') if m in args.modes]
            for mode in modes:
                tag = f'n{n:03d}_{mode}'
                checkpoint = data/'checkpoints'/tag
                if tag in manifest['completed']:
                    result = load_bundle(checkpoint)
                    print('Loaded completed '+tag, flush=True)
                else:
                    started = time.monotonic()
                    extended = n == args.grid and not args.no_stability
                    total = (2 if extended else 1)*(args.burn_in+args.periods)
                    print(f'Simulating {tag}: {args.replications} paired replications, {total} periods each', flush=True)
                    simulation = simulate_pair(pairs[n], periods=args.periods, burn_in=args.burn_in,
                        replications=args.replications, seed=args.seed, mode=mode,
                        short_periods=args.short_periods, total_periods=total,
                        progress=lambda done, total: print(f'  {tag}: {done}/{total} periods', flush=True))
                    save_raw(data/'raw'/tag, simulation, args.compress_raw)
                    windows = [('baseline', args.burn_in, args.periods)]
                    if extended:
                        windows += [('longer', args.burn_in, 2*args.periods),
                                    ('longer_burn', 2*args.burn_in, args.periods),
                                    ('longer_both', 2*args.burn_in, 2*args.periods)]
                    result = dict(summary=[], differences=[], replications=[], distributions={}, running={},
                        samples={key: {field: values[:min(3,args.replications), :args.short_periods].copy()
                                       for field,values in raw.items()}
                                 for key,raw in simulation['profiles'].items()})
                    if n == args.grid:
                        result['product_usage'] = {
                            key: product_usage_evolution(
                                raw['product'][:, :args.usage_periods],
                                smoothing_window=args.usage_smoothing_window)
                            for key, raw in simulation['profiles'].items()}
                    for window,start,length in windows:
                        print(f'Analyzing {tag}/{window}', flush=True)
                        part = summarize_window(pairs[n], simulation, mode=mode, window=window,
                            start=start, length=length, short_periods=args.short_periods,
                            include_distributions=window == 'baseline')
                        for key in ('summary','differences','replications'):
                            result[key].extend(part[key])
                        if window == 'baseline':
                            result['distributions'],result['running'] = part['distributions'],part['running']
                    result['elapsed_seconds'] = time.monotonic()-started
                    save_bundle(checkpoint, result)
                    del simulation
                    manifest['completed'].append(tag)
                    write_json(manifest_path, manifest)
                    print(f'Completed {tag} in {result["elapsed_seconds"]:.1f}s', flush=True)
                if n == args.grid:
                    for key in ('summary','differences','replications'):
                        analysis[key].extend(result[key])
                    for key in ('distributions','samples','running'):
                        analysis[key][mode] = result[key]
                    if 'product_usage' in result:
                        analysis['product_usage'][mode] = result['product_usage']
                analysis['sensitivity'].extend(dict(grid=n, **r) for r in result['summary'] if r['window']=='baseline')
                analysis['sensitivity_differences'].extend(dict(grid=n, **r) for r in result['differences'] if r['window']=='baseline')
        # Source files and equilibrium inputs must remain unchanged during the run.
        if source_hashes() != manifest['source_sha256']:
            raise RuntimeError('Source code changed during the comparison')
        for n in grids:
            reloaded = load_pair(ROOT, rho=args.rho, n=n, oe_run=args.oe_run, p0_run=args.p0_run,
                                oe_candidate=args.oe_candidate, p0_candidate=args.p0_candidate)
            if reloaded['audit'] != source_audits[str(n)]:
                raise RuntimeError('Equilibrium source changed during the comparison')
        save_bundle(data/'analysis', analysis)
        for key in ('summary','differences','replications','sensitivity','sensitivity_differences'):
            write_csv(data/(key+'.csv'), analysis[key])
        write_csv(data/'stationary_predictions.csv', [dict(grid=n, profile=key, **profile['predicted'])
            for n, value in pairs.items() for key, profile in value['profiles'].items()])
        from oblivious.equilibrium_comparison.code.reporting import make_figures, write_findings
        make_figures(pair, analysis, figures)
        write_findings(pair, analysis, data)
        from oblivious.equilibrium_comparison.code.validate_transients import write_validation
        windows = {'baseline': (args.burn_in, args.periods)}
        if not args.no_stability:
            windows.update(longer=(args.burn_in, 2*args.periods),
                           longer_burn=(2*args.burn_in, args.periods),
                           longer_both=(2*args.burn_in, 2*args.periods))
        horizon = max(start+length for start,length in windows.values())
        print('Validating finite-horizon expectations directly from the saved kernels', flush=True)
        write_validation(pair, analysis['summary'], data, windows,
            checkpoints=sorted({0, args.burn_in, horizon}))
        manifest.update(status='complete', finished_utc=now(),
            parameters=asdict(pair['params']), numerical_equilibria_unchanged=True,
            transient_validation='transient_validation.json',
            uncertainty='Student-t intervals across independent replications; model differences paired by seed.',
            discounted_tail_bound_per_seller=pair['params'].R*pair['params'].gamma**args.short_periods/(1-pair['params'].gamma))
        write_json(manifest_path, manifest)
        print('Comparison complete: '+str(data), flush=True)
    except BaseException as error:
        manifest.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                        last_error=str(error), stopped_utc=now())
        write_json(manifest_path, manifest)
        raise


if __name__ == '__main__':
    main()
