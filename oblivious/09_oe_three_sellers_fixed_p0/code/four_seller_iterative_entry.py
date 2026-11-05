"""Recorded continuation using iterative pinned solves for the N4 pilot.

All previously completed bundles and source snapshots are immutable. The new
linear-solver phase is recorded before the frozen controller resumes work.
"""

from contextlib import ExitStack
import argparse
import fcntl
import json
from pathlib import Path
import shutil

from . import four_seller_experiment as frozen
from . import report as base
from . import run_experiment as io


_ORIGINAL_WORKER = frozen.worker
ADDITIONAL_SOURCES = ('four_seller_iterative.py', 'four_seller_iterative_entry.py')


def phase_sources():
    result = frozen.sources()
    for filename in ADDITIONAL_SOURCES:
        path = base.HOME/'code'/filename
        result[base.location(path)] = base.digest(path)
    return result


def record_phase(data):
    """Reconcile only checkpointed work after the previous process has stopped."""
    data = Path(data)
    with ExitStack() as locks:
        paths = [data/'.controller.lock'] + sorted(data.glob('p0_*/rho_*/.worker.lock'))
        base.require(len(paths) == 3, 'Require both previous worker locks')
        for path in paths:
            handle = locks.enter_context(path.open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path = data/'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        base.verify_inputs(manifest['numerical_sources'])
        base.verify_inputs(manifest['input_sha256'])
        if manifest.get('active_solver_phase') == 'iterative_pinned_v1':
            base.require(manifest['numerical_sources'] == phase_sources(), 'Recorded iterative sources differ')
            return manifest
        base.require(manifest['status'] in ('running', 'interrupted', 'partial_failed'),
                     'Only an unfinished original phase may be continued')
        snapshots = data/'provenance/before_iterative_pinned_v1'
        base.require(not snapshots.exists(), 'Prior phase snapshots already exist without a committed phase')
        snapshots.mkdir()
        inputs = {}
        stop_receipt = data/'provenance/original_compute_stop.json'
        if stop_receipt.exists():
            inputs[base.location(stop_receipt)] = base.digest(stop_receipt)
        before = snapshots/'manifest.json'
        shutil.copyfile(manifest_path, before)
        inputs[base.location(before)] = base.digest(before)
        previous_summary = snapshots/'summary.csv'
        shutil.copyfile(data/'summary.csv', previous_summary)
        inputs[base.location(previous_summary)] = base.digest(previous_summary)
        previous_workers = []
        for state_path in sorted(data.glob('p0_*/rho_*/worker_state.json')):
            state = json.loads(state_path.read_text())
            base.verify_inputs(state['input_sha256'])
            snapshot = snapshots/(state_path.parent.parent.name+'_worker_state.json')
            shutil.copyfile(state_path, snapshot)
            inputs[base.location(snapshot)] = base.digest(snapshot)
            previous_workers.append(dict(path=base.location(snapshot), status=state['status'],
                                         completed=state['completed'].copy()))
            manifest['completed'] = sorted(set(manifest['completed']) | set(state['completed']))
            manifest['input_sha256'].update(state['input_sha256'])
            state.update(status='interrupted', current_case=None,
                         solver_phase='iterative_pinned_v1', continued_utc=io.now())
            state.pop('error', None)
            io.write_json(state_path, state)
        for path_string, digest in manifest['numerical_sources'].items():
            original = base.ROOT/path_string
            snapshot = snapshots/'sources'/original.relative_to(base.ROOT)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, snapshot)
            base.require(base.digest(snapshot) == digest, 'Previous source snapshot differs')
            inputs[base.location(snapshot)] = digest
        manifest['input_sha256'].update(inputs)
        manifest.setdefault('execution_phases', []).append(dict(
            phase='iterative_pinned_v1', prepared_utc=io.now(),
            previous_manifest=base.location(before), previous_worker_snapshots=previous_workers,
            previous_numerical_sources=manifest['numerical_sources'].copy(),
            unchanged_checkpoint_count=len(manifest['completed']),
            reason='Reduce sparse factorization fill; same pinned equations and independent certificate tolerances'))
        manifest.update(numerical_sources=phase_sources(), status='interrupted',
                        active_workers=0, current_case=None, failures={},
                        active_solver_phase='iterative_pinned_v1')
        frozen.export(data, manifest)
        return manifest


def worker(task):
    from . import four_seller_iterative as iterative
    restore = iterative.install()
    try:
        return _ORIGINAL_WORKER(task)
    finally:
        restore()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default=frozen.NAME)
    args = parser.parse_args(argv)
    base.require(Path(args.name).name == args.name and args.name not in ('.', '..'), 'Use a campaign name')
    data = base.HOME/'data'/args.name
    record_phase(data)
    previous_sources, previous_worker = frozen.sources, frozen.worker
    # Freeze the complete phase source map before replacing the source reader.
    recorded_sources = phase_sources()
    frozen.sources = lambda: dict(recorded_sources)
    frozen.worker = worker
    try:
        return frozen.main(['--name', args.name, '--resume'])
    finally:
        frozen.sources, frozen.worker = previous_sources, previous_worker


if __name__ == '__main__':
    raise SystemExit(main())
