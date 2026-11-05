"""Checkpointed, isolated pilot comparing one, two and three sellers."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from datetime import datetime, timezone

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from .model import Experiment

HOME = Path(__file__).resolve().parents[1]
ROOT = HOME.parents[1]
MODELS = [(1, 'monopoly'), (2, 'marginal'), (2, 'conditional'), (3, 'marginal'), (3, 'conditional')]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
        json.dump(data, handle, indent=2, default=serial, allow_nan=False)
        handle.write('\n')
        temporary = handle.name
    os.replace(temporary, path)


def write_csv(path, rows):
    path = Path(path)
    if not rows:
        return
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False, newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
        temporary = handle.name
    os.replace(temporary, path)


def source_hashes():
    paths = [HOME/'code'/name for name in ['kernel.py', 'stationary.py', 'model.py', 'run_experiment.py']]
    paths += [ROOT/'oblivious/02_oe_p0/code'/name for name in
              ['primitives.py', 'policy.py', 'core.py', 'positive_stationary.py', 'positive_stationary.cpp']]
    paths += [ROOT/'oblivious/03_oe_following_monopoly/code/event_chain.py', ROOT/'forgetting/Code/model.py']
    return {str(p.relative_to(ROOT)): digest(p) for p in paths if p.exists()}


def check_hashes(hashes):
    for path, expected in hashes.items():
        if digest(ROOT/path) != expected:
            raise RuntimeError(f'Frozen experiment input changed: {path}')


def label(value):
    return f'{value:g}'.replace('.', 'p')


def key(p0, rho, n, sellers, closure):
    return f'p0_{label(p0)}/rho_{label(rho)}/n{n}_N{sellers}_{closure}'


def export(data, manifest):
    rows, policies = [], []
    for stem in manifest['completed']:
        saved = json.loads((data/(stem+'.json')).read_text())
        rows.append(saved['summary'])
        arrays = np.load(data/(stem+'.npz'), allow_pickle=False)
        meta = {name: saved['summary'][name] for name in ['p0', 'rho', 'n', 'sellers', 'closure']}
        policies.extend(dict(meta, reputation=float(x), product2_probability=float(a))
                        for x, a in zip(arrays['grid'], arrays['policy']))
    write_csv(data/'summary.csv', rows)
    write_csv(data/'policies.csv', policies)
    for name in ['summary.csv', 'policies.csv']:
        if (data/name).exists():
            manifest['input_sha256'][str((data/name).relative_to(ROOT))] = digest(data/name)


def load_seed(data, manifest, p0, rho, n, sellers, closure):
    candidates = []
    # First use the same equilibrium's preceding grid, then same-grid N2 or
    # the other closure. Seeds never substitute for a fresh N3 solve.
    for stem in manifest['completed']:
        saved = json.loads((data/(stem+'.json')).read_text())['summary']
        if saved['p0'] != p0 or saved['rho'] != rho:
            continue
        score = None
        if saved['sellers'] == sellers and saved['closure'] == closure and saved['n'] < n:
            score = (0, -saved['n'])
        elif saved['n'] == n and saved['sellers'] == sellers:
            score = (1, 0)
        elif saved['n'] == n and saved['sellers'] == 2:
            score = (2, 0 if saved['closure'] == closure else 1)
        elif saved['n'] == n and saved['sellers'] == 1:
            score = (3, 0)
        if score is not None:
            candidates.append((score, stem))
    if candidates:
        stem = sorted(candidates)[0][1]
        with np.load(data/(stem+'.npz'), allow_pickle=False) as arrays:
            return np.interp(np.linspace(0, 1, n), arrays['grid'], arrays['policy']), stem
    return None, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='pilot')
    parser.add_argument('--p0s', nargs='+', type=float, default=[.1, .5, .8])
    parser.add_argument('--rhos', nargs='+', type=float, default=[.6, .8])
    parser.add_argument('--grids', nargs='+', type=int, default=[21, 31])
    parser.add_argument('--order', type=int, default=256)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--extend-grids', action='store_true', help='With --resume, append finer grids without changing any other parameter')
    parser.add_argument('--max-evaluations', type=int, default=500)
    args = parser.parse_args(argv)
    if '/' in args.name or '..' in args.name or any(r >= .999 or r < 0 for r in args.rhos):
        parser.error('Use a simple campaign name and rhos in [0,.999).')
    if any(n < 3 or n % 2 != 1 for n in args.grids) or any(not 0 <= p < 1 for p in args.p0s):
        parser.error('Use odd grids >=3 and p0 in [0,1).')
    data = HOME/'data'/args.name
    data.mkdir(parents=True, exist_ok=True)
    configuration = dict(p0s=args.p0s, rhos=args.rhos, grids=sorted(set(args.grids)),
                         models=MODELS, order=args.order, tolerance=1e-7,
                         economics=dict(R=1., c1=.05, c2=.65, p1=.35, p2=.8, gamma=.98),
                         purpose='Compare 1, 2 and 3 strategic sellers against the same fixed outside benchmark')
    manifest_path = data/'manifest.json'
    if manifest_path.exists():
        if not args.resume:
            parser.error('Campaign exists; choose another name or use --resume.')
        manifest = json.loads(manifest_path.read_text())
        normalized = json.loads(json.dumps(configuration))
        if normalized != manifest['configuration']:
            previous = manifest['configuration']
            same_other = {k:v for k,v in normalized.items() if k != 'grids'} == {k:v for k,v in previous.items() if k != 'grids'}
            if not args.extend_grids or not same_other or not set(previous['grids']) < set(normalized['grids']):
                raise RuntimeError('Resume configuration differs; only an explicit grid extension is allowed')
            manifest.setdefault('configuration_history', []).append(previous)
            manifest['configuration'] = normalized
        check_hashes(manifest['numerical_sources'])
        check_hashes(manifest['input_sha256'])
    else:
        manifest = dict(configuration=configuration, created_utc=now(), completed=[], failures={},
                        numerical_sources=source_hashes(), input_sha256={}, status='created')
    manifest.update(status='running', started_utc=now(), current_case=None)
    write_json(manifest_path, manifest)
    try:
        for n in configuration['grids']:
            for p0 in args.p0s:
                for rho in args.rhos:
                    for sellers, closure in MODELS:
                        stem = key(p0, rho, n, sellers, closure)
                        if stem in manifest['completed']:
                            continue
                        manifest['current_case'] = stem
                        write_json(manifest_path, manifest)
                        print(f'{now()} START {stem}', flush=True)
                        started = time.monotonic()
                        initial, seed = load_seed(data, manifest, p0, rho, n, sellers, closure)
                        experiment = Experiment(n=n, sellers=sellers, closure=closure, p0=p0,
                                                params=Parameters(rho=rho), order=args.order)
                        result = experiment.solve(initial, args.max_evaluations,
                            progress=lambda message: print(f'{now()} {stem} {message}', flush=True))
                        if not result['verified']:
                            raise RuntimeError('Unverified equilibrium cannot be exported')
                        check_hashes(manifest['numerical_sources'])
                        target = data/stem
                        target.parent.mkdir(parents=True, exist_ok=True)
                        npz_path = Path(str(target)+'.npz')
                        with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                            np.savez_compressed(handle, **{name: result[name] for name in
                                ['grid', 'policy', 'joint', 'value', 'actual_value', 'demand', 'advantage']})
                            temporary = handle.name
                        os.replace(temporary, npz_path)
                        record = dict(configuration=experiment.configuration(), summary=result['summary'],
                                      diagnostics=result['diagnostics'], history=result.get('history', []),
                                      search_calls=result.get('search_calls', 0), seed=seed,
                                      checkpoint_sha256=digest(npz_path), completed_utc=now(),
                                      elapsed_seconds=time.monotonic()-started,
                                      demand_diagnostics=experiment.game.demand_diagnostics)
                        json_path = Path(str(target)+'.json')
                        write_json(json_path, record)
                        for path in [json_path, npz_path]:
                            manifest['input_sha256'][str(path.relative_to(ROOT))] = digest(path)
                        manifest['completed'].append(stem)
                        manifest['failures'].pop(stem, None)
                        export(data, manifest)
                        write_json(manifest_path, manifest)
                        print(f'{now()} DONE {stem} {result["summary"]} seconds={record["elapsed_seconds"]:.2f}', flush=True)
        manifest.update(status='complete', current_case=None, completed_utc=now())
        export(data, manifest)
        check_hashes(manifest['numerical_sources'])
        check_hashes(manifest['input_sha256'])
        write_json(manifest_path, manifest)
        print(f'{now()} COMPLETE {len(manifest["completed"])} certified model/grid cases', flush=True)
    except BaseException as exc:
        manifest['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        manifest['failures'][manifest.get('current_case') or 'controller'] = repr(exc)
        manifest['stopped_utc'] = now()
        export(data, manifest)
        write_json(manifest_path, manifest)
        raise


if __name__ == '__main__':
    main()
