"""Checkpointed N=4 low-p0 marginal OE pilot, retaining certified N=1..3.

The two small grids are numerical validation inputs; only the finest common
certified and metric-stable comparison is published. No old figure is revived.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time
import traceback

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from . import report as base
from . import run_experiment as io


NAME = 'four_sellers_low_p0'
P0S, RHO, GRIDS = (.1, .3), .6, (21, 31)
MODELS = ((1, 'monopoly'), (2, 'marginal'), (3, 'marginal'), (4, 'marginal'))
SOURCE = base.HOME/'data/high_p0_refinement_41'
NEW_SOURCES = ('four_seller_kernel.py', 'four_seller_model.py',
               'four_seller_audit.py', 'four_seller_experiment.py')


def sources():
    previous = json.loads((SOURCE/'manifest.json').read_text())['numerical_sources']
    result = dict(previous)
    for name in NEW_SOURCES:
        path = base.HOME/'code'/name
        result[base.location(path)] = base.digest(path)
    base.verify_inputs(result)
    return result


def expected_stems():
    return [io.key(p0, RHO, n, sellers, closure)
            for p0 in P0S for n in GRIDS for sellers, closure in MODELS]


def export(data, manifest):
    rows = [json.loads((data/(stem+'.json')).read_text())['summary']
            for stem in sorted(manifest['completed'])]
    base.write_csv(data/'summary.csv', rows)
    manifest['input_sha256'][base.location(data/'summary.csv')] = base.digest(data/'summary.csv')
    io.write_json(data/'manifest.json', manifest)


def prepare(data, resume=False):
    if (data/'manifest.json').exists():
        base.require(resume, 'Use --resume for an existing campaign')
        manifest = json.loads((data/'manifest.json').read_text())
        base.verify_inputs(manifest['numerical_sources'])
        base.verify_inputs(manifest['input_sha256'])
        base.require(manifest['numerical_sources'] == sources(), 'Frozen campaign sources differ')
        return manifest
    base.require(not resume, 'No existing campaign to resume')
    origin, *_ = base.read_inputs(SOURCE)
    base.require(origin['status'] == 'complete', 'Need a complete source campaign')
    audit, audit_inputs = base.read_optional_audit(SOURCE, origin)
    base.require(audit is not None, 'Need an independent baseline audit')
    data.mkdir(parents=True, exist_ok=True)
    provenance = data/'provenance'
    provenance.mkdir()
    inputs, completed = {}, []
    for name in ('manifest.json', 'independent_audit.json'):
        shutil.copyfile(SOURCE/name, provenance/('source_'+name))
        for path in (SOURCE/name, provenance/('source_'+name)):
            inputs[base.location(path)] = base.digest(path)
    for stem in expected_stems():
        if '_N4_' in stem:
            continue
        record = json.loads((SOURCE/(stem+'.json')).read_text())
        base.require(record['summary']['verified'], 'Uncertified baseline')
        for suffix in ('.json', '.npz'):
            original, target = SOURCE/(stem+suffix), data/(stem+suffix)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            base.require(base.digest(original) == base.digest(target), 'Baseline bytes changed')
            for path in (original, target):
                inputs[base.location(path)] = base.digest(path)
        completed.append(stem)
    manifest = dict(status='created', created_utc=io.now(), current_case=None,
        configuration=dict(p0s=list(P0S), rhos=[RHO], grids=list(GRIDS), models=MODELS,
            order=256, parameters=dict(R=1., c1=.05, c2=.65, p1=.35, p2=.8, gamma=.98, rho=RHO),
            source_campaign=base.location(SOURCE),
            scope='Marginal OE with exact joint distribution of three rivals; four sellers and fixed outside'),
        numerical_sources=sources(), input_sha256=inputs, completed=completed, failures={},
        expected_model_grid_cases=len(expected_stems()),
        import_provenance=dict(source_campaign=base.location(SOURCE),
            source_manifest_sha256=base.digest(SOURCE/'manifest.json'),
            source_audit_sha256=base.digest(SOURCE/'independent_audit.json'),
            imported_stems=completed.copy(), strategy='Byte-identical certified N=1,2,3 baselines'),
        joint_representation_N4='Exact permutation-quotient calendar law; never an independent-rival product')
    export(data, manifest)
    return manifest


def worker(task):
    from .four_seller_model import FourSellerExperiment
    data, p0 = Path(task['data']), task['p0']
    directory = data/io.key(p0, RHO, 21, 4, 'marginal').rsplit('/', 1)[0]
    directory.mkdir(parents=True, exist_ok=True)
    with (directory/'.worker.lock').open('a') as lock, (directory/'compute.log').open('a', buffering=1) as log:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with redirect_stdout(log), redirect_stderr(log):
            state_path = directory/'worker_state.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else dict(
                status='created', p0=p0, rho=RHO, completed=[], input_sha256={})
            try:
                base.verify_inputs(task['sources'])
                base.verify_inputs(state['input_sha256'])
                state['status'] = 'running'
                for n in GRIDS:
                    stem = io.key(p0, RHO, n, 4, 'marginal')
                    if stem in state['completed']:
                        continue
                    target = data/stem
                    base.require(not target.with_suffix('.npz').exists() and not target.with_suffix('.json').exists(),
                                 'Uncommitted checkpoint exists: '+stem)
                    state['current_case'] = stem
                    io.write_json(state_path, state)
                    print(io.now(), 'START', stem, flush=True)
                    started = time.monotonic()
                    if n == GRIDS[0]:
                        seed_path = data/(io.key(p0, RHO, n, 3, 'marginal')+'.npz')
                    else:
                        seed_path = data/(io.key(p0, RHO, GRIDS[0], 4, 'marginal')+'.npz')
                    with np.load(seed_path) as previous:
                        initial = np.interp(np.linspace(0, 1, n), previous['grid'], previous['policy'])
                    experiment = FourSellerExperiment(n=n, p0=p0, params=Parameters(rho=RHO), order=256)
                    result = experiment.solve(initial_policy=initial, max_evaluations=400,
                        progress=lambda message: print(io.now(), stem, message, flush=True))
                    base.require(result['summary']['verified'], 'Uncertified four-seller solution')
                    arrays = {name: result[name] for name in (
                        'grid', 'policy', 'joint', 'value', 'actual_value', 'demand', 'advantage')}
                    game = experiment.game
                    arrays['representatives'] = game.representatives
                    arrays['orbit_sizes'] = game.orbit_sizes
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                        np.savez_compressed(handle, **arrays)
                        temporary = handle.name
                    os.replace(temporary, target.with_suffix('.npz'))
                    record = dict(configuration=experiment.configuration(), summary=result['summary'],
                        diagnostics=result['diagnostics'], history=result.get('history', []),
                        search_calls=result.get('search_calls', 0), seed=base.location(seed_path),
                        checkpoint_sha256=base.digest(target.with_suffix('.npz')), completed_utc=io.now(),
                        elapsed_seconds=time.monotonic()-started,
                        demand_diagnostics=game.demand_diagnostics,
                        joint_representation='permutation quotient calendar law',
                        actual_value_representation='Mean seller value on quotient; exact per-seller value at symmetric initial state')
                    io.write_json(target.with_suffix('.json'), record)
                    base.verify_inputs(task['sources'])
                    for suffix in ('.json', '.npz'):
                        path = target.with_suffix(suffix)
                        state['input_sha256'][base.location(path)] = base.digest(path)
                    state['completed'].append(stem)
                    io.write_json(state_path, state)
                    print(io.now(), 'DONE', stem, result['summary'], flush=True)
                    del experiment, result
                state.update(status='complete', current_case=None, completed_utc=io.now())
            except BaseException as error:
                state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed',
                             error=repr(error), stopped_utc=io.now())
                traceback.print_exc()
            finally:
                io.write_json(state_path, state)
            return state


def publish(data, manifest):
    from .four_seller_audit import audit_campaign
    audit = audit_campaign(data)
    base.require(audit.get('verified') is True, 'Independent audit failed')
    rows = [json.loads((data/(stem+'.json')).read_text())['summary'] for stem in manifest['completed']]
    lookup = {(row['p0'], row['n'], row['sellers']): row for row in rows}
    diagnostics, retention = [], {}
    for p0 in P0S:
        for metric in ('product2_conditional', 'outside_rate', 'profit_calendar', 'value_initial'):
            changes = []
            for sellers, closure in MODELS:
                coarse, fine = [lookup[p0, n, sellers][metric] for n in GRIDS]
                change = abs(fine-coarse)
                if metric not in ('product2_conditional', 'outside_rate'):
                    change = change/max(abs(coarse), abs(fine)) if coarse or fine else 0.
                changes.append(change)
                diagnostics.append(dict(p0=p0, rho=RHO, sellers=sellers, metric=metric,
                    coarse_n=GRIDS[0], fine_n=GRIDS[-1], change=change, passed=change <= .05+1e-12))
            retention[p0, metric] = max(changes) <= .05+1e-12
    base.write_csv(data/'grid_stability.csv', diagnostics)
    figures = base.HOME/'figures'/data.name
    base.require(not figures.exists(), 'Refusing to overwrite an existing publication')
    figures.mkdir()
    paths = []
    colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
    with base.plt.rc_context({'font.size': 10}):
        fig, ax = base.plt.subplots(figsize=(8, 5), layout='constrained')
        for p0, color, marker in zip(P0S, colors, ('o', 's')):
            if not retention[p0, 'product2_conditional']:
                continue
            values = [100*lookup[p0, GRIDS[-1], sellers]['product2_conditional'] for sellers, _ in MODELS]
            ax.plot([1, 2, 3, 4], values, color=color, marker=marker, label=f'p₀ = {p0:g}')
            for x, value in enumerate(values, 1):
                ax.annotate(f'{value:.2f}%', (x, value), xytext=(0, 8), textcoords='offset points',
                            ha='center', color=color, fontsize=9)
        ax.set(title=f'¿Agregar un cuarto seller reduce el uso de producto 2?\nρ = {RHO:g} | n = 31 contrastado con n = 21',
               xlabel='Número de sellers (N=1: monopolio; N≥2: OE marginal)',
               ylabel='Producto 2 (% de ventas propias)', xticks=[1, 2, 3, 4], ylim=(-3, 100))
        base.style(ax)
        ax.legend(frameon=False)
        paths.extend(base.save(fig, figures, '01_product2_by_seller_count'))
        fig, axes = base.plt.subplots(1, 2, figsize=(12, 4), sharey=True, layout='constrained')
        for ax, p0 in zip(axes, P0S):
            for (sellers, closure), color in zip(MODELS, colors):
                stem = io.key(p0, RHO, GRIDS[-1], sellers, closure)
                with np.load(data/(stem+'.npz')) as arrays:
                    ax.plot(arrays['grid'], arrays['policy'], color=color,
                            label='Monopolio' if sellers == 1 else f'{sellers} sellers', lw=1.6)
            ax.set(title=f'p₀ = {p0:g}', xlabel='Reputación propia', ylabel='Pr(producto 2 | venta propia)',
                   ylim=(-.03, 1.03))
            base.style(ax)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside lower center', ncol=4, frameon=False)
        fig.suptitle('Políticas certificadas en la grilla de 31 puntos | ρ = 0.6\n'
                     'No se certifica estabilidad puntual de los umbrales', fontsize=11)
        paths.extend(base.save(fig, figures, '02_policy_comparison'))
    index = figures/'README.md'
    lines = ['# Cuatro sellers: p0 bajos, OE marginal', '',
        'Grilla publicada: 31, contrastada con 21. Rho=.6; p0=.1,.3.', '',
        '[Producto 2 por número de sellers](01_product2_by_seller_count.png) · '
        '[Políticas](02_policy_comparison.png)', '',
        'Producto 2 se mide entre ventas propias, incluyendo solo los pares que pasan '
        'el límite de 5 puntos porcentuales en los cuatro modelos. Los datos de grilla '
        'menor sirven como validación y no se publican como versiones alternativas.', '',
        '| p0 | Monopolio | 2 sellers | 3 sellers | 4 sellers | Pasa contraste de producto 2 |',
        '| --- | ---: | ---: | ---: | ---: | --- |']
    for p0 in P0S:
        values = [f'{100*lookup[p0,31,n]["product2_conditional"]:.4f}%' for n in (1, 2, 3, 4)]
        lines.append('| '+ ' | '.join([str(p0), *values, str(retention[p0, 'product2_conditional'])])+' |')
    lines += ['', 'La distribución conjunta de los tres rivales se conserva. '
        'La reducción por permutaciones es exacta; no se sustituye por marginales independientes.',
        'Las decisiones se toman tras conseguir la venta. Outside congela las reputaciones; '
        'solo se actualiza el seller seleccionado.', '',
        'El valor calculado en el cociente es el promedio entre sellers; en el estado inicial '
        'simétrico (.5,.5,.5,.5) coincide exactamente con el valor individual. '
        'No representa el valor de una identidad concreta en estados asimétricos.', '',
        '[Datos y certificado](../../data/'+data.name+'/manifest.json) · '
        '[Auditoría](../../data/'+data.name+'/independent_audit.json) · '
        '[Sensibilidad a la grilla](../../data/'+data.name+'/grid_stability.csv)', '',
        'Se certifica el juego numérico finito; el contraste no prueba convergencia al continuo ni unicidad.']
    index.write_text('\n'.join(lines)+'\n')
    report = dict(status='complete_verified', verified=True, created_utc=io.now(),
        source_manifest_sha256=base.digest(data/'manifest.json'),
        audit_sha256=base.digest(data/'independent_audit.json'),
        metric_retention={f'p0_{p0:g}/{metric}': valid for (p0, metric), valid in retention.items()},
        generated_sha256={base.location(p): base.digest(p) for p in [*paths, index, data/'grid_stability.csv']})
    io.write_json(data/'report_manifest.json', report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default=NAME)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    base.require(Path(args.name).name == args.name and args.name not in ('.', '..'), 'Use a campaign folder name')
    data = base.HOME/'data'/args.name
    data.mkdir(parents=True, exist_ok=True)
    with (data/'.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = prepare(data, args.resume)
        if manifest['status'] == 'complete' and (data/'report_manifest.json').exists():
            print('Campaign already complete and published', flush=True)
            return 0
        manifest.update(status='running', active_workers=2, started_utc=io.now())
        export(data, manifest)
        tasks = [dict(data=str(data), p0=p0, sources=manifest['numerical_sources']) for p0 in P0S]
        with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
            jobs = {pool.submit(worker, task): task['p0'] for task in tasks}
            for future in as_completed(jobs):
                state = future.result()
                base.verify_inputs(state['input_sha256'])
                manifest['completed'] = sorted(set(manifest['completed']) | set(state['completed']))
                manifest['input_sha256'].update(state['input_sha256'])
                manifest['active_workers'] -= 1
                if state['status'] != 'complete':
                    manifest['failures'][str(state['p0'])] = state.get('error', state['status'])
                else:
                    manifest['failures'].pop(str(state['p0']), None)
                export(data, manifest)
                print(io.now(), 'WORKER', state['p0'], state['status'], 'completed', len(manifest['completed']), flush=True)
        finished = set(manifest['completed']) == set(expected_stems()) and not manifest['failures']
        manifest.update(status='complete' if finished else 'partial_failed', active_workers=0, completed_utc=io.now())
        export(data, manifest)
        if not finished:
            print(json.dumps(manifest['failures']), flush=True)
            return 1
        report = publish(data, manifest)
        print(json.dumps({'status': report['status'], 'metric_retention': report['metric_retention']}), flush=True)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
