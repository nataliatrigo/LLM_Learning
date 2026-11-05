"""Continue the marginal pilot with a longer independent validation budget.

Only the maximum work allowed to the existing positive stationary iteration
changes. Its initial laws, algorithm, convergence criteria, equilibrium checks,
and all economic equations remain unchanged. Earlier checkpoints and the failed
manifest are preserved, with explicit execution-phase provenance.
"""

import argparse
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import sys

from . import marginal_entry
from . import model
from . import run_experiment as runner

HOME, ROOT = runner.HOME, runner.ROOT
MAX_ITERATIONS = 1_000_000
TIME_LIMIT = 240
MODULE = 'oblivious.oe_three_sellers_fixed_p0.code.validation_entry'
SNAPSHOT = 'provenance/manifest_before_validation_extension.json'


def entry_sources():
    path = Path(__file__).resolve()
    return {str(path.relative_to(ROOT)): runner.digest(path)}


def install():
    """Increase work budgets in this process, preserving every tolerance."""
    original = model.checked_symmetric_stationary

    def longer_validation(*args, **kwargs):
        kwargs['positive_max_iterations'] = max(kwargs.get('positive_max_iterations', 100_000), MAX_ITERATIONS)
        previous_limit = kwargs.get('positive_time_limit', 120)
        kwargs['positive_time_limit'] = (None if previous_limit is None
                                         else max(previous_limit, TIME_LIMIT))
        return original(*args, **kwargs)

    model.checked_symmetric_stationary = longer_validation
    return original


def prepare(data):
    """Record an immutable before-state and the unchanged numerical sources."""
    data = Path(data)
    manifest_path = data/'manifest.json'
    before = manifest_path.read_bytes()
    saved = json.loads(before)
    if saved['status'] == 'running':
        raise RuntimeError('Close the active campaign before extending its validation budget')
    if saved['configuration']['models'] != [list(model) for model in marginal_entry.MODELS]:
        raise RuntimeError('Validation extension applies only to the marginal-only experiment')
    runner.check_hashes(saved['numerical_sources'])
    runner.check_hashes(saved['input_sha256'])
    additional_sources = entry_sources()
    source_name, source_hash = next(iter(additional_sources.items()))
    snapshot = data/SNAPSHOT
    snapshot_name = str(snapshot.relative_to(ROOT))
    # A repeated closed invocation validates the prior phase without creating
    # another snapshot or concealing its original failure.
    if source_name in saved['numerical_sources']:
        if saved['numerical_sources'][source_name] != source_hash:
            raise RuntimeError('Validation entry source differs from its recorded phase')
        if saved['input_sha256'].get(snapshot_name) != runner.digest(snapshot):
            raise RuntimeError('Validation before-state snapshot changed')
        return saved
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        if snapshot.read_bytes() != before:
            raise RuntimeError('An existing validation snapshot cannot be overwritten')
    else:
        with snapshot.open('xb') as handle:
            handle.write(before)
    numerical_sources = dict(saved['numerical_sources'], **additional_sources)
    prior_failure = dict(status=saved['status'], current_case=saved.get('current_case'),
                         failures=deepcopy(saved.get('failures', {})),
                         stopped_utc=saved.get('stopped_utc'),
                         source_manifest_snapshot=snapshot_name,
                         source_manifest_sha256=runner.digest(snapshot))
    extension = dict(prepared_utc=runner.now(), entrypoint=source_name,
                     completed_stems_at_start=list(saved['completed']),
                     positive_max_iterations=MAX_ITERATIONS, positive_time_limit=TIME_LIMIT,
                     algorithm='Existing positive lazy jump-chain iteration from two starts',
                     acceptance='All original tolerances, references and rowwise checks unchanged',
                     prior_failure=prior_failure)
    saved.setdefault('failure_history', []).append(prior_failure)
    saved.setdefault('validation_extensions', []).append(extension)
    saved.setdefault('execution_entrypoint_history', []).append(dict(
        phase='longer independent positive stationary validation',
        numerical_sources=numerical_sources, **extension))
    saved['numerical_sources'] = numerical_sources
    saved['input_sha256'][snapshot_name] = runner.digest(snapshot)
    runner.write_json(manifest_path, saved)
    # No checkpoint or preexisting numerical source may change in preparation.
    runner.check_hashes(json.loads(before)['input_sha256'])
    runner.check_hashes(json.loads(before)['numerical_sources'])
    return saved


def active_controllers():
    modules = {MODULE, 'oblivious.oe_three_sellers_fixed_p0.code.run_experiment',
               'oblivious.oe_three_sellers_fixed_p0.code.marginal_entry'}
    active = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            command = (path/'cmdline').read_bytes().split(b'\0')
            arguments = [part.decode(errors='replace') for part in command if part]
        except (OSError, ProcessLookupError):
            continue
        if not arguments or not Path(arguments[0]).name.startswith('python'):
            continue
        if any(argument in modules for argument in arguments):
            active.append(int(path.name))
    return active


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--name', default='marginal_pilot')
    parser.add_argument('--prepare-only', action='store_true')
    args, _ = parser.parse_known_args(argv)
    if '/' in args.name or '..' in args.name:
        raise ValueError('Use a simple campaign name')
    if '--help' in argv or '-h' in argv:
        marginal_entry.main(argv)
        return
    data = HOME/'data'/args.name
    with (data/'.validation_entry.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another validation continuation already holds the campaign lock') from exc
        active = active_controllers()
        if active:
            raise RuntimeError(f'Another experiment controller is still active: {active}')
        saved = prepare(data)
        print(f'Preserved {len(saved["completed"])} completed cases; positive validation budget '
              f'{MAX_ITERATIONS} iterations / {TIME_LIMIT} seconds; tolerances unchanged', flush=True)
        if args.prepare_only:
            return
        original = install()
        try:
            marginal_entry.main(argv)
        finally:
            model.checked_symmetric_stationary = original


if __name__ == '__main__':
    main()
