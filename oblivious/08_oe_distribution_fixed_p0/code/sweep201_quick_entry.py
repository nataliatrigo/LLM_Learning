"""Explicit new execution phase for optional quick proposal screens and 1–6 jobs.

The original 201/251 worker and scheduler remain byte-identical. Only the
proposal-screen response is optionally replaced; every equilibrium certificate,
local solver and actual value calculation remains original.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys

from oblivious.oe_distribution_fixed_p0.code import sweep201_run as worker
from oblivious.oe_distribution_fixed_p0.code import sweep201_batch as batch
from oblivious.oe_distribution_fixed_p0.code import sweep201_quick_prescreen as quick

ENTRY_MODULE = 'oblivious.oe_distribution_fixed_p0.code.sweep201_quick_entry'
WORKER_MODULE = 'oblivious.oe_distribution_fixed_p0.code.sweep201_run'


def helper_reference():
    path = Path(__file__).resolve()
    return {str(path.relative_to(worker.runner.ROOT)):sha256(path.read_bytes()).hexdigest()}


def batch_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--workers',type=int,default=3)
    options,remainder = parser.parse_known_args(argv)
    if not 1 <= options.workers <= 6:
        parser.error('Use 1–6 parallel workers')
    # Reuse every original scope/calibration validator; only concurrency expands.
    args = batch.arguments(remainder+['--workers','3'])
    args.workers = options.workers
    return args


def child_command(command,enabled):
    command = list(command)
    if WORKER_MODULE in command:
        index = command.index(WORKER_MODULE)
        command[index] = ENTRY_MODULE
        if not enabled:
            command.insert(index+1,'--no-quick-prescreen')
    return command


def record_phase(enabled,started):
    path = worker.CAMPAIGN/'sweep_manifest.json'
    if not path.exists():
        return
    state = json.loads(path.read_text())
    record = dict(entry_module=ENTRY_MODULE,started_utc=started,
                  recorded_utc=worker.runner.now(),quick_prescreen_enabled=enabled,
                  source_input_sha256=dict(helper_reference(),**quick.helper_reference()),
                  immutable_scheduler_sha256=sha256(Path(batch.__file__).read_bytes()).hexdigest(),
                  immutable_worker_sha256=sha256(Path(worker.__file__).read_bytes()).hexdigest())
    state.setdefault('execution_entrypoint_history',[]).append(record)
    state['execution_entrypoint'] = record
    worker.runner.write_json(path,state)


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--batch',action='store_true')
    parser.add_argument('--no-quick-prescreen',action='store_true')
    options,remainder = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    enabled = not options.no_quick_prescreen
    if options.batch:
        args = batch_arguments(remainder)
        batch.arguments = lambda:args
        original_popen = batch.subprocess.Popen

        def new_entry_process(command,*args,**kwargs):
            return original_popen(child_command(command,enabled),*args,**kwargs)

        batch.subprocess.Popen = new_entry_process
        started = worker.runner.now()
        try:
            batch.main()
        finally:
            record_phase(enabled,started)
    else:
        args = worker.arguments(remainder)
        original_install = worker.install

        def install_phase(calibration):
            original_install(calibration)
            if enabled:
                quick.install()
            old = worker.runner.baseline_seed

            def seed_with_entrypoint(rho,n,closure):
                proposal,inputs = old(rho,n,closure)
                return proposal,dict(inputs,**helper_reference())

            worker.runner.baseline_seed = seed_with_entrypoint

        worker.install = install_phase
        worker.arguments = lambda:args
        worker.main()


if __name__ == '__main__':
    main()
