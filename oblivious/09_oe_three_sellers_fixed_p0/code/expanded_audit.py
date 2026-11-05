"""Independent audit of the expanded, optionally adaptive marginal campaign.

The numerical checkpoint and import checks are the frozen pilot auditor's
unchanged checks.  This adapter validates per-(p0,rho) grid schedules and keeps
its own source provenance, without editing the pilot's audit or its reports.
"""

import argparse
from hashlib import sha256
from itertools import product
import json
import math
from pathlib import Path
import time

from . import audit as original

HOME, ROOT = original.HOME, original.ROOT
OUTPUT = 'independent_audit.json'
require = original.require


def source_map():
    references = original.source_map()
    path = Path(__file__).resolve()
    references[str(path.relative_to(ROOT))] = sha256(path.read_bytes()).hexdigest()
    return references


def pair_key(p0, rho):
    return '/'.join(original.case_stem(p0,rho,3,1,'monopoly').split('/')[:2])


def _grid_list(values, label):
    require(isinstance(values,list) and len(values) > 0, f'{label} must be a nonempty grid list')
    require(all(isinstance(n,int) and not isinstance(n,bool) and n >= 3 and n % 2 == 1 for n in values),
            f'{label} must contain odd integer grid sizes >=3')
    require(len(values) == len(set(values)), f'{label} contains duplicate grid sizes')
    return sorted(values)


def expected_scope(configuration):
    """Return the exact model/grid stems and the validated pair grid map."""
    p0s, rhos, models = configuration['p0s'], configuration['rhos'], configuration['models']
    require(isinstance(p0s,list) and p0s and isinstance(rhos,list) and rhos,
            'Configured p0s and rhos must be nonempty lists')
    require(all(isinstance(value,(int,float)) and not isinstance(value,bool)
                and math.isfinite(value) and 0 <= value < 1 for value in p0s), 'Invalid configured p0')
    require(all(isinstance(value,(int,float)) and not isinstance(value,bool)
                and math.isfinite(value) and .5 < value < 1 for value in rhos), 'Invalid configured rho')
    require(len(p0s) == len(set(p0s)) and len(rhos) == len(set(rhos)),
            'Configured p0s or rhos contain duplicates')
    require(isinstance(models,list) and models, 'Configured models must be a nonempty list')
    normalized_models = []
    for item in models:
        require(isinstance(item,(list,tuple)) and len(item) == 2, 'Invalid configured model')
        sellers,closure = item
        require(isinstance(sellers,int) and not isinstance(sellers,bool) and sellers in (1,2,3)
                and ((sellers == 1 and closure == 'monopoly')
                     or (sellers > 1 and closure in ('marginal','conditional'))),
                'Invalid configured seller count or belief closure')
        normalized_models.append((sellers,closure))
    require(len(normalized_models) == len(set(normalized_models)), 'Configured models contain duplicates')
    pairs = {(p0,rho):pair_key(p0,rho) for p0,rho in product(p0s,rhos)}
    require(len(set(pairs.values())) == len(pairs), 'Configured parameters collide in checkpoint pair labels')
    if 'case_grids' in configuration:
        case_grids = configuration['case_grids']
        require(isinstance(case_grids,dict), 'case_grids must be a mapping')
        require(set(case_grids) == set(pairs.values()),
                'case_grids must have exactly all configured p0/rho keys, with no missing or unknown pairs')
        grids = {key:_grid_list(values,f'case_grids[{key}]') for key,values in case_grids.items()}
    else:
        common = _grid_list(configuration['grids'],'configuration.grids')
        grids = {key:list(common) for key in pairs.values()}
    stems = {original.case_stem(p0,rho,n,sellers,closure)
             for (p0,rho),key in pairs.items() for n in grids[key]
             for sellers,closure in normalized_models}
    return dict(stems=stems,case_grids=grids,parameter_pairs=len(pairs),
                models=[list(item) for item in normalized_models])


def audit_pass(data, previous=None):
    data = Path(data)
    manifest_path = data/'manifest.json'
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    manifest_sha = sha256(manifest_bytes).hexdigest()
    scope = expected_scope(manifest['configuration'])
    expected_stems = scope['stems']
    completed_stems = manifest['completed']
    require(len(completed_stems) == len(set(completed_stems)), 'Duplicate completed checkpoint stems')
    require(set(completed_stems) <= expected_stems,
            'Completed checkpoint outside the configured experiment scope')
    audit_sources = source_map()
    original.verify_hashes(manifest['numerical_sources'])
    imported = original.verify_import_provenance(data,manifest)
    if previous:
        require(previous['audit_sources'] == audit_sources,
                'Independent expanded audit sources changed during one campaign')
        require(set(previous.get('cases',{})) <= set(completed_stems),
                'Previously audited checkpoints disappeared from the campaign')
    records = dict(previous.get('cases',{})) if previous else {}
    for stem in completed_stems:
        if stem in records:
            original.verify_hashes(records[stem]['input_sha256'])
        else:
            records[stem] = original.audit_checkpoint(data,stem,manifest['input_sha256'])
    complete = manifest['status'] == 'complete'
    if complete:
        require(set(completed_stems) == expected_stems,
                'Closed campaign has an incomplete or incorrect checkpoint scope')
        require(set(records) == expected_stems, 'Closed independent audit is missing checkpoint proofs')
        original.verify_hashes(manifest['input_sha256'])
    original.verify_hashes(manifest['numerical_sources'])
    original.verify_hashes(audit_sources)
    current_bytes = manifest_path.read_bytes()
    current = json.loads(current_bytes)
    if complete:
        require(sha256(current_bytes).hexdigest() == manifest_sha, 'Closed manifest changed during audit')
    partial_status = ('running_partial_verified' if manifest['status'] in {'created','running'}
                      else 'partial_verified_inputs_closed')
    return dict(status='complete_verified' if complete else partial_status,
                verified=complete,audited_utc=original.now(),
                manifest_status=manifest['status'],current_manifest_status=current['status'],
                manifest_snapshot_sha256=manifest_sha,
                expected_model_grid_cases=len(expected_stems),audited_model_grid_cases=len(records),
                expected_parameter_pairs=scope['parameter_pairs'],case_grids=scope['case_grids'],
                models=scope['models'],audit_sources=audit_sources,
                numerical_sources=manifest['numerical_sources'],
                campaign_input_sha256=manifest['input_sha256'] if complete else None,
                import_provenance=imported,mutable_csv_fingerprints_checked=complete,cases=records,
                scope='Stored finite-grid policies and independent equations/accounting audit; no grid stability, equilibrium uniqueness, or mixing-time guarantee')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data','--name',dest='name',default='rho_p0_sweep',
                        help='Campaign folder inside this experiment data directory')
    parser.add_argument('--watch',action='store_true')
    parser.add_argument('--interval',type=float,default=30)
    args = parser.parse_args(argv)
    require('/' not in args.name and '..' not in args.name,'Use a simple campaign name')
    require(1 <= args.interval <= 60,'Watch interval must be 1..60 seconds')
    data = HOME/'data'/args.name
    target = data/OUTPUT
    previous = json.loads(target.read_text()) if target.exists() else None
    while True:
        result = audit_pass(data,previous)
        original.write_json(target,result)
        print(f"{original.now()} {result['status']} {result['audited_model_grid_cases']}/{result['expected_model_grid_cases']}",flush=True)
        if result['verified'] or not args.watch:
            return 0
        previous = result
        # Finish immediately if the controller closed during this audit pass.
        if result['current_manifest_status'] == 'complete':
            continue
        if result['current_manifest_status'] not in {'created','running'}:
            return 1
        time.sleep(args.interval)


if __name__ == '__main__':
    raise SystemExit(main())
