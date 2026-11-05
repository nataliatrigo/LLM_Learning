"""Parallel, checkpointed marginal OE sweep with explicit per-case grids.

The original pilot and all its sources are immutable. Each worker owns one
(p0,rho) case; only the parent updates campaign metadata and CSV exports.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import fcntl
import gc
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from . import run_experiment as io
from . import expanded_stationary
from .search_only import FastExperiment

HOME, ROOT = io.HOME, io.ROOT
MODELS = [(1, 'monopoly'), (2, 'marginal'), (3, 'marginal')]
P0S, RHOS = [.1, .3, .5, .7, .8, .9], [.6, .7, .8, .9]


def pair_key(p0, rho):
    return f'p0_{io.label(p0)}/rho_{io.label(rho)}'


def expected(config):
    return [io.key(p0,rho,n,sellers,closure)
            for p0 in config['p0s'] for rho in config['rhos']
            for n in config['case_grids'][pair_key(p0,rho)] for sellers,closure in MODELS]


def new_sources(previous):
    sources = dict(previous, **expanded_stationary.source_reference())
    for name in ['expanded_run.py', 'search_only.py']:
        path = HOME/'code'/name
        sources[str(path.relative_to(ROOT))] = io.digest(path)
    return sources


def prepare(data, config):
    manifest_path = data/'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        io.check_hashes(manifest['numerical_sources'])
        io.check_hashes(manifest['input_sha256'])
        before = manifest['configuration']
        if config != before:
            fixed = lambda c: {k:v for k,v in c.items() if k not in ('case_grids','grids')}
            if fixed(config) != fixed(before):
                raise RuntimeError('Resume changes the economics, p0/rho scope, or models')
            for tag, grids in before['case_grids'].items():
                if not set(grids) <= set(config['case_grids'][tag]):
                    raise RuntimeError('Existing grids cannot be removed')
            manifest.setdefault('configuration_history', []).append(before)
            manifest['configuration'] = config
        return manifest
    source = HOME/'data/marginal_pilot'
    origin_path = source/'manifest.json'
    origin = json.loads(origin_path.read_text())
    if origin['status'] != 'complete':
        raise RuntimeError('The import source must be a closed, complete pilot')
    io.check_hashes(origin['numerical_sources']); io.check_hashes(origin['input_sha256'])
    allowed = set(expected(config))
    imported, source_inputs, inputs = [], {}, {}
    for stem in origin['completed']:
        if stem not in allowed:
            continue
        record = json.loads((source/(stem+'.json')).read_text())
        if not record['summary']['verified']:
            raise RuntimeError('An uncertified source checkpoint cannot be imported')
        imported.append(stem)
        for suffix in ['.json','.npz']:
            original, target = source/(stem+suffix), data/(stem+suffix)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(original,target)
            if io.digest(original) != io.digest(target):
                raise RuntimeError('Imported bytes differ')
            source_inputs[str(original.relative_to(ROOT))] = io.digest(original)
            inputs[str(target.relative_to(ROOT))] = io.digest(target)
    snapshot = data/'provenance/source_manifest.json'
    snapshot.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(origin_path,snapshot)
    inputs.update(source_inputs)
    inputs[str(snapshot.relative_to(ROOT))] = io.digest(snapshot)
    sources = new_sources(origin['numerical_sources'])
    manifest = dict(configuration=config, created_utc=io.now(), status='created',
        current_case=None, completed=imported, failures={}, numerical_sources=sources,
        input_sha256=inputs,
        import_provenance=dict(source_manifest=str(origin_path.relative_to(ROOT)),
            source_manifest_sha256=io.digest(origin_path), source_manifest_snapshot=str(snapshot.relative_to(ROOT)),
            imported_stems=imported, source_input_sha256=source_inputs,
            scope='Byte-identical certified marginal pilot; expanded p0/rho scope is freshly solved'),
        execution_entrypoint_history=[dict(phase='import certified marginal pilot', completed_stems=imported,
                                          numerical_sources=origin['numerical_sources']),
            dict(phase='parallel expanded marginal sweep', numerical_sources=sources,
                 entrypoint=str(Path(__file__).relative_to(ROOT)),
                 search='Exact pinned sparse stationary solve, MMD ordering and lazy analytic Jacobian',
                 final_certificate='Unchanged equations/tolerances; positive dense GTH in both orders on exact seller quotient')])
    io.export(data,manifest);io.write_json(manifest_path,manifest)
    return manifest


def checkpoint(data, stem, result, experiment, seed, elapsed):
    target = data/stem
    target.parent.mkdir(parents=True,exist_ok=True)
    npz_path, json_path = Path(str(target)+'.npz'), Path(str(target)+'.json')
    if npz_path.exists() or json_path.exists():
        raise RuntimeError('Refusing to overwrite an existing checkpoint: '+stem)
    with tempfile.NamedTemporaryFile(dir=target.parent,delete=False) as handle:
        np.savez_compressed(handle,**{name:result[name] for name in
            ['grid','policy','joint','value','actual_value','demand','advantage']})
        temporary = handle.name
    os.replace(temporary,npz_path)
    record = dict(configuration=experiment.configuration(),summary=result['summary'],
        diagnostics=result['diagnostics'],history=result.get('history',[]),search_calls=result.get('search_calls',0),
        seed=seed,checkpoint_sha256=io.digest(npz_path),completed_utc=io.now(),elapsed_seconds=elapsed,
        demand_diagnostics=experiment.game.demand_diagnostics,
        execution_phase='parallel expanded marginal sweep')
    io.write_json(json_path,record)
    return {str(path.relative_to(ROOT)):io.digest(path) for path in (npz_path,json_path)}


def recover_checkpoint(data, stem, p0, rho, n, sellers, closure, state):
    """Recover a completed atomic bundle, preserving interrupted partial files."""
    paths=[data/(stem+suffix) for suffix in ('.json','.npz')]
    if not any(path.exists() for path in paths):return False
    if all(path.exists() for path in paths):
        record=json.loads(paths[0].read_text());summary=record['summary'];config=record['configuration']
        correct=all(summary[k]==v for k,v in dict(p0=p0,rho=rho,n=n,sellers=sellers,closure=closure).items())
        if (not correct or not summary['verified'] or config['parameters']['rho']!=rho
                or record['checkpoint_sha256']!=io.digest(paths[1])
                or record.get('execution_phase')!='parallel expanded marginal sweep'):
            raise RuntimeError('Existing orphan is not a valid matching checkpoint: '+stem)
        state['completed'].append(stem)
        state['input_sha256'].update({str(path.relative_to(ROOT)):io.digest(path) for path in paths})
        state.setdefault('recovered_atomic_bundles',[]).append(stem)
        return True
    destination=data/'interrupted_partial_bundles'/stem
    destination.parent.mkdir(parents=True,exist_ok=True)
    archived=[]
    for path in paths:
        if path.exists():
            target=Path(str(destination)+'.'+str(time.time_ns())+path.suffix)
            shutil.move(path,target)
            archived.append(dict(path=str(target.relative_to(ROOT)),sha256=io.digest(target)))
    state.setdefault('uncommitted_partial_files',[]).append(dict(stem=stem,files=archived))
    return False


def work_case(task):
    data = Path(task['data']);p0,rho = task['p0'],task['rho'];tag=pair_key(p0,rho)
    directory=data/tag;directory.mkdir(parents=True,exist_ok=True)
    state_path=directory/'worker_state.json'
    log_path=directory/'compute.log'
    with (directory/'.worker.lock').open('a') as lock, log_path.open('a',buffering=1) as log:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        old_stdout,old_stderr=sys.stdout,sys.stderr
        sys.stdout=sys.stderr=log
        restore=None
        state=dict(status='running',p0=p0,rho=rho,completed=[],input_sha256={},current_case=None)
        try:
            restore=expanded_stationary.install()
            if state_path.exists():
                state=json.loads(state_path.read_text());io.check_hashes(state['input_sha256'])
                state['status']='running'
            io.check_hashes(task['sources'])
            local_completed=list(task['completed'])
            for stem in state['completed']:
                if stem not in local_completed:local_completed.append(stem)
            for n in task['grids']:
                for sellers,closure in MODELS:
                    stem=io.key(p0,rho,n,sellers,closure)
                    if stem in local_completed:continue
                    if recover_checkpoint(data,stem,p0,rho,n,sellers,closure,state):
                        local_completed.append(stem);io.write_json(state_path,state)
                        continue
                    state['current_case']=stem;io.write_json(state_path,state)
                    print(io.now(),'START',stem,flush=True);started=time.monotonic()
                    initial,seed=io.load_seed(data,{'completed':local_completed},p0,rho,n,sellers,closure)
                    experiment=FastExperiment(n=n,sellers=sellers,closure=closure,p0=p0,
                                              params=Parameters(rho=rho),order=task['order'])
                    result=experiment.solve(initial,max_evaluations=task['max_evaluations'],
                        progress=lambda message:print(io.now(),stem,message,flush=True))
                    if not result['verified']:raise RuntimeError('Uncertified equilibrium')
                    io.check_hashes(task['sources'])
                    fingerprints=checkpoint(data,stem,result,experiment,seed,time.monotonic()-started)
                    state['completed'].append(stem);state['input_sha256'].update(fingerprints)
                    local_completed.append(stem)
                    io.write_json(state_path,state)
                    print(io.now(),'DONE',stem,result['summary'],flush=True)
                    del experiment,result
                    gc.collect()
            state.update(status='complete',current_case=None,completed_utc=io.now())
        except BaseException as exc:
            state.update(status='failed',error=repr(exc),stopped_utc=io.now())
            traceback.print_exc()
        finally:
            try:
                io.write_json(state_path,state)
            finally:
                try:
                    if restore is not None:restore()
                finally:
                    sys.stdout,sys.stderr=old_stdout,old_stderr
        return state


def merge_states(data,manifest):
    allowed=set(expected(manifest['configuration']))
    for path in data.glob('p0_*/rho_*/worker_state.json'):
        state=json.loads(path.read_text())
        io.check_hashes(state['input_sha256'])
        if not set(state['completed'])<=allowed:raise RuntimeError('Worker checkpoint outside current scope')
        own_prefix=pair_key(state['p0'],state['rho'])+'/'
        if any(not stem.startswith(own_prefix) for stem in state['completed']):
            raise RuntimeError('Worker claimed another p0/rho case')
        allowed_paths={str((data/(stem+suffix)).relative_to(ROOT))
                       for stem in state['completed'] for suffix in ('.json','.npz')}
        if set(state['input_sha256'])!=allowed_paths:
            raise RuntimeError('Worker fingerprint map differs from its checkpoint scope')
        for name,value in state['input_sha256'].items():
            if name in manifest['input_sha256'] and manifest['input_sha256'][name]!=value:
                raise RuntimeError('Worker attempted to replace an existing certified fingerprint')
        for stem in state['completed']:
            if stem not in manifest['completed']:manifest['completed'].append(stem)
        manifest['input_sha256'].update(state['input_sha256'])
    manifest['completed'].sort()
    io.export(data,manifest)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='rho_p0_sweep')
    parser.add_argument('--p0s',type=float,nargs='+',default=P0S)
    parser.add_argument('--rhos',type=float,nargs='+',default=RHOS)
    parser.add_argument('--grids',type=int,nargs='+',default=[21,31,41])
    parser.add_argument('--add-case-grid',action='append',default=[],metavar='P0:RHO:N')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--max-evaluations',type=int,default=800)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args(argv)
    if '/' in args.name or '..' in args.name or not 1<=args.workers<=3:parser.error('Use a simple name and 1..3 workers')
    if any(n<3 or n%2!=1 or n>51 for n in args.grids):parser.error('This dense-reference campaign supports odd grids 3..51')
    if any(not 0<p<1 for p in args.p0s) or any(not .5<r<.999 for r in args.rhos):parser.error('Invalid p0/rho')
    if len(set(args.p0s))!=len(args.p0s) or len(set(args.rhos))!=len(args.rhos):parser.error('Duplicate p0/rho values')
    data=HOME/'data'/args.name;data.mkdir(parents=True,exist_ok=True)
    with (data/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (data/'manifest.json').exists() and not args.resume:parser.error('Campaign exists; use --resume')
        case_grids={pair_key(p,r):sorted(set(args.grids)) for p in args.p0s for r in args.rhos}
        if args.resume and (data/'manifest.json').exists():
            saved=json.loads((data/'manifest.json').read_text())
            for tag,grids in saved['configuration']['case_grids'].items():
                if tag in case_grids:case_grids[tag]=sorted(set(case_grids[tag]+grids))
        for item in args.add_case_grid:
            p,r,n=map(float,item.split(':'));tag=pair_key(p,r)
            if tag not in case_grids or int(n)!=n or n<3 or int(n)%2!=1 or n>51:parser.error('Invalid case-grid extension')
            case_grids[tag]=sorted(set(case_grids[tag]+[int(n)]))
        config=dict(p0s=args.p0s,rhos=args.rhos,grids=sorted({n for ns in case_grids.values() for n in ns}),
                    case_grids=case_grids,models=[list(model) for model in MODELS],order=256,tolerance=1e-7,
                    economics=dict(R=1.,c1=.05,c2=.65,p1=.35,p2=.8,gamma=.98),
                    purpose='Compare 1, 2 and 3 strategic sellers with fixed p0 across persistence and outside benchmarks')
        manifest=prepare(data,config);merge_states(data,manifest)
        io.write_json(data/'manifest.json',manifest)
        if args.prepare_only:return
        tasks=[];done=set(manifest['completed'])
        for p0 in args.p0s:
            for rho in args.rhos:
                if all(io.key(p0,rho,n,s,c) in done for n in case_grids[pair_key(p0,rho)] for s,c in MODELS):continue
                tasks.append(dict(data=str(data),p0=p0,rho=rho,grids=case_grids[pair_key(p0,rho)],
                    completed=list(manifest['completed']),sources=manifest['numerical_sources'],order=256,
                    max_evaluations=args.max_evaluations))
        # Start cheaper cases first; the two difficult high-p0/rho cases have
        # their own files and cannot prevent other cases from being certified.
        tasks.sort(key=lambda t:(t['rho'],t['p0']))
        manifest.update(status='running',current_case=None,started_utc=io.now(),active_workers=args.workers)
        io.write_json(data/'manifest.json',manifest)
        try:
            pool=ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn'))
            try:
                futures={pool.submit(work_case,task):task for task in tasks}
                for future in as_completed(futures):
                    task=futures[future];tag=pair_key(task['p0'],task['rho'])
                    try:state=future.result()
                    except Exception as exc:state=dict(status='failed',error=repr(exc))
                    if state['status']!='complete':manifest['failures'][tag]=state.get('error','worker failed')
                    else:manifest['failures'].pop(tag,None)
                    merge_states(data,manifest);io.write_json(data/'manifest.json',manifest)
                    print(io.now(),tag,state['status'],len(manifest['completed']),'/',len(expected(config)),flush=True)
            except BaseException:
                for process in list(pool._processes.values()):process.terminate()
                pool.shutdown(wait=True,cancel_futures=True)
                raise
            else:
                pool.shutdown(wait=True)
            merge_states(data,manifest)
            full=set(manifest['completed'])==set(expected(config))
            manifest.update(status='complete' if full else 'partial_failed',current_case=None,
                            active_workers=0,completed_utc=io.now())
            io.check_hashes(manifest['numerical_sources']);io.check_hashes(manifest['input_sha256'])
            io.write_json(data/'manifest.json',manifest)
            print(io.now(),manifest['status'],len(manifest['completed']),flush=True)
            return 0 if full else 1
        except BaseException as exc:
            merge_states(data,manifest)
            manifest.update(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
                            active_workers=0,controller_error=repr(exc),stopped_utc=io.now())
            io.write_json(data/'manifest.json',manifest)
            raise


if __name__=='__main__':
    raise SystemExit(main())
