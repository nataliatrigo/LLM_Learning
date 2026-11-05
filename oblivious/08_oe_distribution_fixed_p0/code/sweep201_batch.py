"""Run the six-p0, four-rho 201/251 comparison in isolated resumable workers."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading

from oblivious.oe_distribution_fixed_p0.code import sweep201_run as worker
from oblivious.oe_distribution_fixed_p0.code.run_experiment import ROOT, now, require, write_json


def ordered_jobs(p0s,rhos):
    """Resolve the newly requested persistence and its hardest p0 cases first."""
    jobs = [(p0,rho) for p0 in p0s for rho in rhos]
    return sorted(jobs,key=lambda pair:(pair[1] != .9,
                  (0 if pair[0] == .8 else 1 if pair[0] == .9 else 2) if pair[1] == .9 else 0,
                  jobs.index(pair)))


def completion_counts(p0s, rhos, grids):
    """Count only committed checkpoints; a running/failed search adds nothing."""
    bundles = outcomes = complete_pairs = 0
    for p0 in p0s:
        for rho in rhos:
            path = worker.campaign_directory(p0, rho)/'manifest.json'
            if not path.exists():
                continue
            manifest = json.loads(path.read_text())
            completed = set(manifest['completed'])
            required = []
            for n in grids:
                for closure in ('marginal', 'conditional'):
                    tag = worker.runner.case_tag(rho, n)+'_'+closure
                    required.append(tag)
                    if tag in completed:
                        # These digests are also checked again by every resume.
                        case = worker.runner.check_bundle(path.parent, tag, manifest)
                        bundles += 1
                        outcomes += len(case['outcomes'])
            complete_pairs += int(all(tag in completed for tag in required))
    return dict(certified_checkpoint_bundles=bundles,
                certified_model_grid_solutions=outcomes, completed_p0_rho_pairs=complete_pairs,
                expected_checkpoint_bundles=len(p0s)*len(rhos)*len(grids)*2,
                expected_model_grid_solutions=len(p0s)*len(rhos)*len(grids)*3,
                expected_p0_rho_pairs=len(p0s)*len(rhos))


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p0s', type=float, nargs='+', default=list(worker.P0S))
    parser.add_argument('--rhos', type=float, nargs='+', default=list(worker.RHOS))
    parser.add_argument('--grids', type=int, nargs='+', default=list(worker.GRIDS))
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--max-evaluations', type=int, default=250)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    if (not 1 <= args.workers <= 3 or args.max_evaluations < 1
            or len(set(args.p0s)) != len(args.p0s) or any(x not in worker.P0S for x in args.p0s)
            or len(set(args.rhos)) != len(args.rhos) or any(x not in worker.RHOS for x in args.rhos)
            or len(set(args.grids)) != len(args.grids) or any(x not in worker.GRIDS for x in args.grids)):
        parser.error('Use requested p0/rho values, distinct 201/251 grids and 1–3 workers')
    return args


def main():
    args = arguments()
    campaign = worker.CAMPAIGN
    path = campaign/'sweep_manifest.json'
    source = sha256(Path(__file__).read_bytes()).hexdigest()
    config = dict(p0s=args.p0s, rhos=args.rhos, grids=sorted(args.grids),
                  order=256, tol=1e-7, dp_tol=1e-10)
    if args.resume:
        state = json.loads(path.read_text())
        require(state['configuration'] == config, 'Saved campaign scope/calibration differs')
        require(state['scheduler_sha256'] == source, 'Campaign scheduler source changed')
        require(state['worker_sha256'] == sha256(Path(worker.__file__).read_bytes()).hexdigest(),
                'Campaign worker wrapper changed')
        require(state['numerical_sources'] == worker.runner.source_fingerprints(),
                'Campaign numerical sources changed')
    else:
        require(not path.exists(), 'Existing campaign protected; use --resume')
        state = dict(created_utc=now(), configuration=config, scheduler_sha256=source,
                     worker_sha256=sha256(Path(worker.__file__).read_bytes()).hexdigest(),
                     numerical_sources=worker.runner.source_fingerprints(), jobs={}, attempts=[],
                     interpretation='Independently certified finite-grid solutions; no uniqueness claim',
                     historical_results_unchanged=True)
    campaign.mkdir(parents=True, exist_ok=True)
    logs = campaign/'logs'
    logs.mkdir(exist_ok=True)
    state['attempts'].append(dict(started_utc=now(), workers=args.workers,
                                  max_evaluations=args.max_evaluations))
    state.update(status='running', **completion_counts(args.p0s,args.rhos,args.grids))
    write_json(path, state)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MPLCONFIGDIR='/tmp/sweep201-mpl')
    stopping = threading.Event()
    children = {}
    lock = threading.Lock()

    def stop(signum, frame):
        stopping.set()
        with lock:
            for process in children.values():
                if process.poll() is None:
                    process.send_signal(signal.SIGINT)

    original_handlers = {s:signal.signal(s,stop) for s in (signal.SIGINT,signal.SIGTERM)}

    def run(job):
        p0,rho = job
        tag = f'p0_{worker.runner.number_tag(p0)}_rho_{worker.runner.number_tag(rho)}'
        if stopping.is_set():
            return tag,dict(p0=p0,rho=rho,status='not_started_after_interrupt',exit_code=None)
        directory = worker.campaign_directory(p0,rho)
        logfile = logs/(tag+'.log')
        command = [sys.executable,'-B','-m','oblivious.oe_distribution_fixed_p0.code.sweep201_run',
                   '--p0',str(p0),'--rho',str(rho),'--grids',*map(str,args.grids),
                   '--max-evaluations',str(args.max_evaluations)]
        if (directory/'manifest.json').exists():
            command.append('--resume')
        print(f'[{now()}] Starting p0={p0:g}, rho={rho:g}, grids={args.grids}',flush=True)
        with logfile.open('a') as stream:
            process = subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            with lock:
                children[tag] = process
                if stopping.is_set():
                    process.send_signal(signal.SIGINT)
            code = process.wait()
            with lock:
                children.pop(tag,None)
        return tag,dict(p0=p0,rho=rho,grids=args.grids,exit_code=code,finished_utc=now(),
                        status='finished' if code == 0 else 'interrupted' if stopping.is_set() else 'failed',
                        log=str(logfile.relative_to(ROOT)))

    try:
        # Full 201→251 validation is completed per calibration before advancing.
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run,job) for job in ordered_jobs(args.p0s,args.rhos)]
            for future in as_completed(futures):
                tag,result = future.result()
                state['jobs'][tag] = result
                state.update(updated_utc=now(),**completion_counts(args.p0s,args.rhos,args.grids))
                write_json(path,state)
                print(f'[{now()}] Finished {tag}: {result["status"]}; '
                      f'{state["certified_model_grid_solutions"]}/{state["expected_model_grid_solutions"]} certified model-grid solutions',flush=True)
    finally:
        for signum,handler in original_handlers.items():
            signal.signal(signum,handler)
        state.update(finished_utc=now(),**completion_counts(args.p0s,args.rhos,args.grids))
        state['status'] = ('interrupted_preserving_checkpoints' if stopping.is_set() else
                           'complete_certified' if state['completed_p0_rho_pairs'] == state['expected_p0_rho_pairs']
                           else 'complete_with_unresolved_cases')
        write_json(path,state)
    if stopping.is_set() or state['status'] != 'complete_certified':
        raise SystemExit(130 if stopping.is_set() else 1)


if __name__ == '__main__':
    main()
