"""Run the marginal-only pilot, preserving earlier conditional checkpoints.

The user narrowed the scope after the original pilot stopped at a conditional
row-accuracy check. Already certified monopoly/marginal bundles are imported
byte for byte into an isolated campaign; neither old files nor model equations
change. This module selects the three requested models in the frozen runner.
"""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys

from . import run_experiment as runner

MODELS = [(1, 'monopoly'), (2, 'marginal'), (3, 'marginal')]
HOME, ROOT = runner.HOME, runner.ROOT


def entry_sources():
    paths = [Path(__file__).resolve()]
    return {str(path.relative_to(ROOT)): runner.digest(path) for path in paths}


def prepare(target, source=None):
    target = Path(target)
    source = HOME/'data/pilot' if source is None else Path(source)
    manifest_path = target/'manifest.json'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved['configuration']['models'] != [list(model) for model in MODELS]:
            raise RuntimeError('Existing target is not the marginal-only experiment')
        runner.check_hashes(saved['numerical_sources'])
        runner.check_hashes(saved['input_sha256'])
        return saved
    origin = json.loads((source/'manifest.json').read_text())
    if origin['status'] == 'running':
        raise RuntimeError('Close the source campaign before importing immutable checkpoints')
    runner.check_hashes(origin['numerical_sources'])
    runner.check_hashes(origin['input_sha256'])
    selected, source_inputs = [], {}
    for stem in origin['completed']:
        saved = json.loads((source/(stem+'.json')).read_text())
        summary = saved['summary']
        if (summary['sellers'], summary['closure']) not in MODELS:
            continue
        if not summary['verified']:
            raise RuntimeError('Unverified source bundle cannot be imported')
        selected.append(stem)
        for suffix in ('.json', '.npz'):
            original = source/(stem+suffix)
            path = target/(stem+suffix)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, path)
            if runner.digest(original) != runner.digest(path):
                raise RuntimeError('Imported checkpoint bytes differ from the source')
            source_inputs[str(original.relative_to(ROOT))] = runner.digest(original)
    target.mkdir(parents=True, exist_ok=True)
    snapshot = target/'provenance/source_manifest.json'
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source/'manifest.json', snapshot)
    config = deepcopy(origin['configuration'])
    config['models'] = [list(model) for model in MODELS]
    numerical_sources = dict(origin['numerical_sources'], **entry_sources())
    inputs = dict(source_inputs)
    inputs[str(snapshot.relative_to(ROOT))] = runner.digest(snapshot)
    for stem in selected:
        for suffix in ('.json', '.npz'):
            path = target/(stem+suffix)
            inputs[str(path.relative_to(ROOT))] = runner.digest(path)
    manifest = dict(configuration=config, created_utc=runner.now(), completed=selected,
                    failures={}, numerical_sources=numerical_sources, input_sha256=inputs,
                    status='created', current_case=None,
                    scope_reason='Explicit user request: only marginal-distribution OE for now',
                    import_provenance=dict(source_manifest=str((source/'manifest.json').relative_to(ROOT)),
                        source_manifest_sha256=runner.digest(source/'manifest.json'),
                        source_manifest_snapshot=str(snapshot.relative_to(ROOT)),
                        imported_stems=selected, source_input_sha256=source_inputs,
                        scope='Certified monopoly and marginal checkpoints imported byte for byte; conditional results remain in the source campaign'),
                    execution_entrypoint_history=[dict(
                        phase='import certified monopoly/marginal bundles', completed_stems=selected,
                        numerical_sources=origin['numerical_sources']), dict(
                        phase='marginal-only continuation', entrypoint=str(Path(__file__).relative_to(ROOT)),
                        numerical_sources=numerical_sources)])
    runner.export(target, manifest)
    runner.write_json(manifest_path, manifest)
    return manifest


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--name', default='marginal_pilot')
    parser.add_argument('--prepare-only', action='store_true')
    args, _ = parser.parse_known_args(argv)
    if '/' in args.name or '..' in args.name:
        raise ValueError('Use a simple campaign name')
    if '--help' in argv or '-h' in argv:
        runner.main(argv)
        return
    manifest = prepare(HOME/'data'/args.name)
    print(f'Imported/preserved {len(manifest["completed"])} certified monopoly/marginal cases', flush=True)
    if args.prepare_only:
        return
    forwarded = [arg for arg in argv if arg != '--prepare-only']
    if not any(arg == '--name' or arg.startswith('--name=') for arg in forwarded):
        forwarded += ['--name', args.name]
    if '--resume' not in forwarded:
        forwarded.append('--resume')
    runner.MODELS = MODELS
    runner.main(forwarded)


if __name__ == '__main__':
    main()
