"""Run the baseline fixed-p0 comparison in independently resumable workers.

The historical p0=.5 computation is read by the reporter, never rerun here.
Each other p0/rho gets fresh monopoly, marginal and conditional solutions.
Only cases whose 51/101 outcomes differ need an additional 151-point check.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys

from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    HOME, ROOT, check_bundle, now, write_json,
)
from oblivious.oe_distribution_fixed_p0.code.conditional_comparison_report import sensitivity_selection

P0S = [.1, .3, .5, .7, .8]
RHOS = [.6, .7, .8, .9, .95, .97]


def p0_name(value):
    return 'p0_' + str(float(value)).replace('.', 'p')


def needs_finer_check(directory, rho):
    """Require complete certified pairs before considering finer validation."""
    path = directory / 'manifest.json'
    if not path.exists():
        return False
    manifest = json.loads(path.read_text())
    required = [f'rho_{str(rho).replace(".", "p")}_n{n:03d}_{model}'
                for n in (51, 101) for model in ('marginal', 'conditional')]
    if not all(tag in manifest['completed'] for tag in required):
        return False
    rows = [row for tag in required for row in check_bundle(directory, tag, manifest)['outcomes']]
    return rho not in sensitivity_selection(rows, [rho], 51, 101)['included_rhos']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', choices=['p0_sweep'], default='p0_sweep')
    parser.add_argument('--p0s', type=float, nargs='+', default=P0S)
    parser.add_argument('--rhos', type=float, nargs='+', default=RHOS)
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--max-evaluations', type=int, default=250)
    parser.add_argument('--skip-refinement', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args()
    if (Path(args.name).name != args.name or args.name in ('', '.', '..')
            or not 1 <= args.workers <= 4 or args.max_evaluations < 1
            or len(set(args.p0s)) != len(args.p0s) or any(p0 not in P0S for p0 in args.p0s)
            or len(set(args.rhos)) != len(args.rhos) or any(rho not in RHOS for rho in args.rhos)):
        parser.error('Use a campaign basename, baseline p0/rho values and 1–4 workers')
    campaign = HOME / 'data' / args.name
    logs = campaign / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    baseline = ROOT / 'Fluid base model/config.json'
    state = dict(status='running', created_utc=now(), configuration=vars(args),
                 baseline_p0_values=json.loads(baseline.read_text())['p0_comparison']['values'],
                 p0_selection_note='User replaced baseline p0=0.9 with p0=0.8',
                 baseline_configuration_sha256=sha256(baseline.read_bytes()).hexdigest(),
                 scheduler_sha256=sha256(Path(__file__).read_bytes()).hexdigest(), jobs={})
    path = campaign / 'sweep_manifest.json'
    write_json(path, state)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
               MKL_NUM_THREADS='1', MPLCONFIGDIR='/tmp/fixed-p0-sweep-mpl')

    def run(job):
        p0, rho, grids = job
        directory = campaign / p0_name(p0) / ('rho_'+str(rho).replace('.', 'p'))
        tag = f'{p0_name(p0)}_rho_{str(rho).replace(".", "p")}_n{grids[-1]}'
        log = logs / (tag+'.log')
        command = [sys.executable, '-B', '-m',
                   'oblivious.oe_distribution_fixed_p0.code.multi_p0_endpoint_recovery',
                   '--p0', str(p0), '--rho', str(rho),
                   '--grids', *map(str, grids), '--max-evaluations', str(args.max_evaluations)]
        if (directory/'manifest.json').exists():
            command.append('--resume')
        print(f'[{now()}] Starting p0={p0:g}, rho={rho:g}, grids={grids}', flush=True)
        with log.open('a') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        return tag, dict(p0=p0, rho=rho, grids=grids, exit_code=result.returncode,
                         log=str(log.relative_to(ROOT)), finished_utc=now())

    def batch(jobs):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(run, job) for job in jobs]
            for future in as_completed(futures):
                tag, result = future.result()
                state['jobs'][tag] = result
                write_json(path, state)
                print(f'[{now()}] Finished {tag}: exit={result["exit_code"]}', flush=True)

    fresh = [(p0, rho, [51, 101]) for p0 in args.p0s if p0 != .5 for rho in args.rhos]
    batch(fresh)
    if not args.skip_refinement:
        extra = [(p0, rho, [151]) for p0 in args.p0s if p0 != .5 for rho in args.rhos
                 if needs_finer_check(campaign/p0_name(p0)/('rho_'+str(rho).replace('.', 'p')), rho)]
        state['finer_checks_requested'] = [dict(p0=p0, rho=rho, grids=grids) for p0,rho,grids in extra]
        write_json(path, state)
        batch(extra)
    state.update(status='complete' if all(row['exit_code'] == 0 for row in state['jobs'].values())
                 else 'complete_with_unresolved_cases', finished_utc=now())
    write_json(path, state)
    if args.report:
        command = [sys.executable, '-B', '-m',
                   'oblivious.oe_distribution_fixed_p0.code.multi_p0_report',
                   '--name', args.name, '--p0s', *map(str,args.p0s), '--rhos', *map(str,args.rhos)]
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
