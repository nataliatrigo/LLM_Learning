"""Refine high outside benchmarks without modifying the completed sweep.

All certified 21/31 checkpoints are copied byte for byte from the closed source
campaign. Only p0=.7,.8,.9 gain grid 41 (and, if requested, 51), for the same
four rho values and the same three models. The frozen parallel worker, equations
and tolerances remain unchanged. At rho<=.7, large chains use the original
sparse/positive reference branch; at rho>=.8, the exact quotient uses positive
dense GTH in both orders. Both branches retain independent agreement checks.
"""

import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from . import expanded_run as campaign
from . import high_p0_worker as worker
from . import run_experiment as io

HOME, ROOT = io.HOME, io.ROOT
SOURCE = 'rho_p0_sweep_31'
HIGH_P0S = [.7, .8, .9]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def helper_reference():
    path = Path(__file__).resolve()
    return str(path.relative_to(ROOT)), io.digest(path)


def isolate(value):
    # A JSON roundtrip also separates aliases in nested historical lists.
    return json.loads(json.dumps(value, allow_nan=False))


def fixed_configuration(config):
    return {key:value for key,value in config.items() if key not in {'grids', 'case_grids'}}


def verify_refinement_config(origin, config):
    before = origin['configuration']
    require(fixed_configuration(before) == fixed_configuration(config),
            'Refinement changes the economics, p0/rho scope, models, order or tolerance')
    require(set(before['case_grids']) == set(config['case_grids']), 'Refinement changes parameter pairs')
    require(all(set(grids) <= set(config['case_grids'][tag])
                for tag,grids in before['case_grids'].items()), 'Refinement removes a source grid')


def copy_identical(source, destination):
    """Preserve existing files and publish a complete copied file atomically."""
    if destination.exists():
        require(destination.read_bytes() == source.read_bytes(), 'Existing import bytes differ: '+str(destination))
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        shutil.copyfile(source, temporary)
        require(io.digest(source) == io.digest(temporary), 'Imported checkpoint bytes differ')
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_refinement(data, config, source, source_builder):
    """Construct metadata under the controller's exclusive target lock."""
    data, source = Path(data).resolve(), Path(source).resolve()
    require(data.is_relative_to(ROOT) and source.is_relative_to(ROOT), 'Campaigns must reside inside the repository')
    require(data != source, 'Refinement must use a new campaign directory')
    require(not (data/'manifest.json').exists(), 'Refinement manifest already exists')
    origin_path = source/'manifest.json'
    origin_bytes = origin_path.read_bytes()
    origin = json.loads(origin_bytes)
    require(origin['status'] == 'complete' and not origin.get('failures'), 'Import source is not closed and complete')
    io.check_hashes(origin['numerical_sources'])
    io.check_hashes(origin['input_sha256'])
    verify_refinement_config(origin, config)
    imported = list(origin['completed'])
    require(len(imported) == len(set(imported)) and set(imported) == set(campaign.expected(origin['configuration'])),
            'Source completed checkpoints do not exactly cover its configured scope')
    require(set(imported) <= set(campaign.expected(config)), 'Refinement omits a source checkpoint')
    source_inputs = {}
    # Check every certificate and NPZ fingerprint before writing destination files.
    for stem in imported:
        record = json.loads((source/(stem+'.json')).read_bytes())
        summary = record['summary']
        require(summary['verified'] is True, 'An uncertified source checkpoint cannot be imported')
        require(io.key(**{key:summary[key] for key in ['p0','rho','n','sellers','closure']}) == stem,
                'Source certificate has a different case identity')
        for suffix in ('.json', '.npz'):
            path = source/(stem+suffix)
            name = str(path.relative_to(ROOT))
            expected = origin['input_sha256'].get(name)
            require(expected is not None and io.digest(path) == expected, 'Source checkpoint fingerprint is missing or changed')
            source_inputs[name] = expected
        require(record['checkpoint_sha256'] == source_inputs[str((source/(stem+'.npz')).relative_to(ROOT))],
                'Source certificate is not certified against its stored checkpoint')
    # Preserve all source input declarations, including earlier import metadata.
    inputs = dict(origin['input_sha256'])
    for stem in imported:
        for suffix in ('.json', '.npz'):
            original, target = source/(stem+suffix), data/(stem+suffix)
            copy_identical(original, target)
            inputs[str(target.relative_to(ROOT))] = source_inputs[str(original.relative_to(ROOT))]
    snapshot = data/'provenance/source_manifest.json'
    copy_identical(origin_path, snapshot)
    origin_sha = io.digest(origin_path)
    inputs[str(snapshot.relative_to(ROOT))] = origin_sha
    sources = source_builder(origin['numerical_sources'])
    io.check_hashes(sources)
    # These three lists are deliberately distinct; later worker merges append
    # only to current completions, never to historical import declarations.
    manifest = dict(configuration=isolate(config), created_utc=io.now(), status='created',
        current_case=None, completed=list(imported), failures={}, numerical_sources=dict(sources),
        input_sha256=inputs, stationary_certification=isolate(worker.CERTIFICATION),
        import_provenance=dict(source_manifest=str(origin_path.relative_to(ROOT)),
            source_manifest_sha256=origin_sha, source_manifest_snapshot=str(snapshot.relative_to(ROOT)),
            imported_stems=list(imported), source_input_sha256=dict(source_inputs),
            scope='All certified checkpoints copied byte-identically from the complete 21/31 sweep; only new grid cases are solved'),
        execution_entrypoint_history=[
            dict(phase='import complete certified 21/31 sweep', completed_stems=list(imported),
                 numerical_sources=dict(origin['numerical_sources'])),
            dict(phase='selective high-p0 grid refinement', numerical_sources=dict(sources),
                 entrypoint=helper_reference()[0],
                 search='Unchanged frozen parallel worker and exact sparse stationary search',
                 final_certificate='Unchanged 1e-11 invariant/1e-7 independent agreement tolerances; rho<=.7 uses two MMD-ordered sparse jump solves plus positive lazy iteration from two starts; rho>=.8 uses two sparse solves plus positive dense GTH in both orders on the exact quotient',
                 stationary_certification=isolate(worker.CERTIFICATION))])
    io.check_hashes(origin['input_sha256'])
    io.check_hashes(origin['numerical_sources'])
    require(origin_path.read_bytes() == origin_bytes, 'Source manifest changed during import')
    io.export(data, manifest)
    io.check_hashes(manifest['input_sha256'])
    io.write_json(data/'manifest.json', manifest)
    return manifest


def install(source):
    """Adapt only fresh-campaign import construction, then restore globals."""
    previous_prepare, previous_sources, previous_worker = campaign.prepare, campaign.new_sources, campaign.work_case

    def sources(previous):
        output = dict(previous_sources(previous))
        output.update(worker.source_reference())
        name, fingerprint = helper_reference()
        output[name] = fingerprint
        return output

    def prepare(data, config):
        if (data/'manifest.json').exists():
            saved = json.loads((data/'manifest.json').read_bytes())
            require(saved['import_provenance']['source_manifest'] == str((source/'manifest.json').relative_to(ROOT)),
                    'Resume import source differs from the original refinement source')
            require(io.digest(source/'manifest.json') == saved['import_provenance']['source_manifest_sha256'],
                    'Resume source manifest fingerprint differs from the recorded immutable source')
            return isolate(previous_prepare(data, config))
        return prepare_refinement(data, config, source, sources)

    campaign.prepare, campaign.new_sources, campaign.work_case = prepare, sources, worker.work_case

    def restore():
        campaign.prepare, campaign.new_sources, campaign.work_case = previous_prepare, previous_sources, previous_worker

    return restore


def controller_arguments(args):
    name = args.name or f'high_p0_refinement_{args.target_grid}'
    require(name and '/' not in name and '..' not in name, 'Use a simple campaign name')
    output = ['--name', name, '--grids', '21', '31', '--workers', str(args.workers),
              '--max-evaluations', str(args.max_evaluations)]
    for p0 in HIGH_P0S:
        for rho in campaign.RHOS:
            for grid in (41, 51):
                if grid <= args.target_grid:
                    output += ['--add-case-grid', f'{p0}:{rho}:{grid}']
    if args.resume:
        output += ['--resume']
    if args.prepare_only:
        output += ['--prepare-only']
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', help='New campaign name; defaults to high_p0_refinement_41')
    parser.add_argument('--source', default=SOURCE, help='Complete immutable 21/31 source campaign')
    parser.add_argument('--target-grid', type=int, choices=[41, 51], default=41)
    parser.add_argument('--workers', type=int, choices=[1, 2, 3], default=3)
    parser.add_argument('--max-evaluations', type=int, default=800)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    require(args.source and '/' not in args.source and '..' not in args.source, 'Use a simple source campaign name')
    require(args.max_evaluations > 0, 'max-evaluations must be positive')
    restore = install(HOME/'data'/args.source)
    try:
        return campaign.main(controller_arguments(args))
    finally:
        restore()


if __name__ == '__main__':
    raise SystemExit(main())
