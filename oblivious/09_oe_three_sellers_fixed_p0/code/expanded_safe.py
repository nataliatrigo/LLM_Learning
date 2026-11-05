"""Repair import-list metadata and isolate future controller list mutations.

The frozen runner reused one list for current completions and two historical
import declarations. This adapter never changes numerical results or frozen
sources. A repair preserves the complete original manifest and verifies every
declared source/checkpoint fingerprint before correcting those declarations.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import tempfile

from . import expanded_run as campaign
from . import run_experiment as io

HOME, ROOT = io.HOME, io.ROOT


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def helper_reference():
    path = Path(__file__).resolve()
    return str(path.relative_to(ROOT)), io.digest(path)


def json_isolate(manifest):
    """JSON has no object aliases: each declaration receives its own list."""
    return json.loads(json.dumps(manifest, allow_nan=False))


def verified_imports(data, manifest):
    provenance = manifest['import_provenance']
    snapshot_name = provenance['source_manifest_snapshot']
    snapshot = ROOT/snapshot_name
    source_manifest = ROOT/provenance['source_manifest']
    expected_manifest_sha = provenance['source_manifest_sha256']
    require(manifest['input_sha256'].get(snapshot_name) == io.digest(snapshot)
            == expected_manifest_sha == io.digest(source_manifest), 'Import source manifest fingerprint differs')
    origin = json.loads(snapshot.read_bytes())
    require(origin['status'] == 'complete', 'Imported pilot is not complete')
    io.check_hashes(origin['numerical_sources'])
    io.check_hashes(origin['input_sha256'])
    declared = provenance['source_input_sha256']
    require(bool(declared), 'Import source fingerprint map is empty')
    io.check_hashes(declared)
    source_root = source_manifest.parent.resolve()
    pairs = {}
    for name, expected_sha in declared.items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(source_root), 'Declared import is outside the source pilot')
        relative = path.relative_to(source_root)
        require(relative.suffix in {'.json', '.npz'}, 'Declared import is not a checkpoint file')
        stem = relative.with_suffix('').as_posix()
        require(relative.suffix not in pairs.setdefault(stem, {}), 'Duplicate import checkpoint component')
        pairs[stem][relative.suffix] = name
        require(origin['input_sha256'].get(name) == expected_sha,
                'Declared import fingerprint differs from its source snapshot')
        destination = data/relative
        destination_name = str(destination.relative_to(ROOT))
        require(destination.is_file() and manifest['input_sha256'].get(destination_name) == expected_sha
                and io.digest(destination) == expected_sha and path.read_bytes() == destination.read_bytes(),
                'Imported source and destination checkpoint bytes differ: '+stem)
    allowed = set(campaign.expected(manifest['configuration']))
    correct = [stem for stem in origin['completed'] if stem in allowed]
    require(len(correct) == len(set(correct)) and set(pairs) == set(correct),
            'Declared imported files do not match the source pilot within this campaign scope')
    require(set(correct) <= set(manifest['completed']), 'An imported checkpoint is absent from current completions')
    for stem, components in pairs.items():
        require(set(components) == {'.json', '.npz'}, 'Imported checkpoint pair is incomplete: '+stem)
        record = json.loads((ROOT/components['.json']).read_bytes())
        require(record['summary']['verified'] is True
                and record['checkpoint_sha256'] == declared[components['.npz']],
                'Imported checkpoint was not certified against its stored arrays: '+stem)
    return correct


def repair_imports(data):
    data = Path(data).resolve()
    require(data.is_relative_to(ROOT), 'Campaign must reside inside the repository')
    path = data/'manifest.json'
    with (data/'.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before_bytes = path.read_bytes()
        manifest = json.loads(before_bytes)
        require(manifest['status'] not in {'created', 'running'}, 'Stop the controller before repairing metadata')
        io.check_hashes(manifest['numerical_sources'])
        io.check_hashes(manifest['input_sha256'])
        correct = verified_imports(data, manifest)
        phase = manifest['execution_entrypoint_history'][0]
        require(phase['phase'] == 'import certified marginal pilot', 'First execution phase is not the pilot import')
        old_imported = manifest['import_provenance']['imported_stems']
        old_phase = phase['completed_stems']
        same = lambda values: len(values) == len(correct) and set(values) == set(correct)
        if same(old_imported) and same(old_phase):
            return dict(status='already_correct', imported_checkpoints=len(correct),
                        completed_checkpoints=len(manifest['completed']), manifest_changed=False)
        snapshot = data/'provenance/manifest_before_import_scope_repair.json'
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            require(snapshot.read_bytes() == before_bytes, 'Existing before-repair snapshot differs; refusing to overwrite it')
        else:
            with tempfile.NamedTemporaryFile(dir=snapshot.parent, delete=False) as handle:
                handle.write(before_bytes)
                temporary = handle.name
            os.replace(temporary, snapshot)
        snapshot_name, snapshot_sha = str(snapshot.relative_to(ROOT)), io.digest(snapshot)
        helper_name, helper_sha = helper_reference()
        # These are the only historical declarations corrected. Current
        # completions, source maps, checkpoints, CSVs and current phase survive.
        manifest['import_provenance']['imported_stems'] = list(correct)
        phase['completed_stems'] = list(correct)
        for name, value in ((snapshot_name, snapshot_sha), (helper_name, helper_sha)):
            require(name not in manifest['input_sha256'] or manifest['input_sha256'][name] == value,
                    'Repair metadata fingerprint conflicts with an existing input')
            manifest['input_sha256'][name] = value
        runner_source = {name:value for name,value in manifest['numerical_sources'].items()
                         if name.endswith('/expanded_run.py') or name == 'expanded_run.py'}
        manifest.setdefault('metadata_repairs', []).append(dict(
            kind='import_scope_shared_list_alias', repaired_utc=io.now(),
            reason='Frozen controller appended new results to the list also used by the historical import declarations',
            fields=['import_provenance.imported_stems', 'execution_entrypoint_history[0].completed_stems'],
            old_imported_count=len(old_imported), old_phase_imported_count=len(old_phase),
            corrected_imported_count=len(correct), current_completed_count=len(manifest['completed']),
            before_manifest=snapshot_name, before_manifest_sha256=snapshot_sha,
            helper=helper_name, helper_sha256=helper_sha, frozen_runner_source_sha256=runner_source,
            numerical_sources_unchanged=True, checkpoint_and_csv_bytes_unchanged=True))
        io.check_hashes(manifest['numerical_sources'])
        io.check_hashes(manifest['input_sha256'])
        require(path.read_bytes() == before_bytes, 'Manifest changed during metadata repair')
        io.write_json(path, manifest)
        return dict(status='repaired', imported_checkpoints=len(correct),
                    completed_checkpoints=len(manifest['completed']), manifest_changed=True,
                    before_manifest=snapshot_name, before_manifest_sha256=snapshot_sha)


def install():
    """Wrap only controller metadata construction for future invocations."""
    previous_prepare, previous_sources = campaign.prepare, campaign.new_sources

    def safe_prepare(*args, **kwargs):
        return json_isolate(previous_prepare(*args, **kwargs))

    def safe_sources(previous):
        sources = dict(previous_sources(previous))
        name, digest = helper_reference()
        sources[name] = digest
        return sources

    campaign.prepare, campaign.new_sources = safe_prepare, safe_sources

    def restore():
        campaign.prepare, campaign.new_sources = previous_prepare, previous_sources

    return restore


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--repair-only', action='store_true')
    parser.add_argument('--name', default='rho_p0_sweep')
    args, remaining = parser.parse_known_args(argv)
    require('/' not in args.name and '..' not in args.name and args.name, 'Use a simple campaign name')
    if args.repair_only:
        require(not remaining, 'Unknown repair arguments')
        print(json.dumps(repair_imports(HOME/'data'/args.name), indent=2))
        return 0
    restore = install()
    try:
        return campaign.main(['--name', args.name] + remaining)
    finally:
        restore()


if __name__ == '__main__':
    raise SystemExit(main())
