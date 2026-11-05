"""Optional recorded continuation for the final high-p0 grid-41 case.

This entrypoint cannot stop or attach to an active controller. It accepts only
an interrupted/failed campaign with precisely the N3,p0=.8,rho=.8,n41 case
missing. Every existing checkpoint, import declaration and economic setting is
preserved. A before-manifest snapshot and a new source-fingerprinted execution
phase document the new exploratory search. Certification still uses the frozen
hybrid worker and the original complete independent checks.
"""

import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import tempfile

from . import active_face_search
from . import audit
from . import expanded_run as campaign
from . import high_p0_worker as hybrid
from . import run_experiment as io

HOME, ROOT = io.HOME, io.ROOT
DEFAULT_NAME = 'high_p0_refinement_41'
RECOVERY_CASE = dict(p0=.8, rho=.8, n=41, sellers=3, closure='marginal')
ACTIVE_COLUMNS = [13, 35]
_HYBRID_WORK_CASE = hybrid.work_case


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def isolate(value):
    return json.loads(json.dumps(value, allow_nan=False))


def source_reference():
    references = dict(hybrid.source_reference())
    references.update(active_face_search.source_reference())
    path = Path(__file__).resolve()
    references[str(path.relative_to(ROOT))] = io.digest(path)
    return references


def snapshot_bytes(path,content):
    """Publish exact before-state bytes without replacing an existing backup."""
    require(not path.exists() or path.read_bytes() == content,
            'Existing before-recovery snapshot differs; refusing to overwrite it')
    if path.exists():
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    try:
        os.replace(temporary,path)
    finally:
        temporary.unlink(missing_ok=True)
    require(path.read_bytes() == content,'Before-recovery snapshot bytes differ')


class RecoveryExperiment(active_face_search.ActiveFaceExperiment):
    """Adapt the exploratory face search to the frozen worker's solve API."""

    def solve(self, initial_policy=None, *, max_evaluations=800, progress=None):
        require((self.n, self.sellers, self.closure, self.p0, self.params.rho) == (41,3,'marginal',.8,.8),
                'Active-face recovery is restricted to the single declared missing case')
        require(initial_policy is not None, 'Recovery requires the recorded preceding-grid policy seed')
        return self.solve_active_face(initial_policy, list(ACTIVE_COLUMNS),
                                      max_evaluations=max_evaluations, progress=progress)


def work_case(task):
    """Picklable child wrapper; leave the complete hybrid certificate intact."""
    require(task['p0'] == .8 and task['rho'] == .8, 'Recovery worker received an unexpected parameter pair')
    previous = campaign.FastExperiment
    campaign.FastExperiment = RecoveryExperiment
    try:
        return _HYBRID_WORK_CASE(task)
    finally:
        campaign.FastExperiment = previous


def _prepare_under_controller_lock(data, config, previous_prepare):
    """Called exclusively by campaign.main while it owns .controller.lock."""
    path = data/'manifest.json'
    before_bytes = path.read_bytes()
    before = json.loads(before_bytes)
    require(before['status'] in {'interrupted','failed'},
            'Recovery requires an interrupted or failed controller; an active or complete campaign is never changed')
    require(config == before['configuration'], 'Recovery cannot change configuration, grids, economics or parameter scope')
    io.check_hashes(before['numerical_sources'])
    io.check_hashes(before['input_sha256'])
    declared = campaign.expected(before['configuration'])
    require(len(declared) == len(set(declared)) == 180, 'Recovery expects the original 180-case refinement scope')
    require(len(before['completed']) == len(set(before['completed'])) == 179,
            'Recovery requires all 179 previous checkpoints to be committed')
    missing = set(declared)-set(before['completed'])
    require(missing == {io.key(**RECOVERY_CASE)} and set(before['completed']) <= set(declared),
            'Recovery is restricted to the single declared missing case')
    require(len(before['import_provenance']['imported_stems']) == 144,
            'Recovery import scope differs from the original 144 checkpoints')
    audit.verify_import_provenance(data,before)
    # A stale worker_state can say "running" after an interrupted controller;
    # an actually live worker still owns its kernel lock and is rejected.
    with ExitStack() as locks:
        for worker_lock in sorted(data.glob('p0_*/rho_*/.worker.lock')):
            handle = locks.enter_context(worker_lock.open('a'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = isolate(previous_prepare(data,config))
        require(path.read_bytes() == before_bytes, 'Manifest changed while preparing recovery')
        require(manifest['configuration'] == before['configuration']
                and manifest['completed'] == before['completed']
                and manifest['import_provenance'] == before['import_provenance']
                and manifest['execution_entrypoint_history'] == before['execution_entrypoint_history'],
                'Recovery preparation changed existing configuration or provenance')
        sources = dict(before['numerical_sources'])
        for name,fingerprint in source_reference().items():
            require(name not in sources or sources[name] == fingerprint, 'Recovery would replace a frozen numerical source')
            sources[name] = fingerprint
        io.check_hashes(sources)
        last_phase = manifest['execution_entrypoint_history'][-1]
        if last_phase.get('continuation_protocol') == 'active_face_v1':
            require(last_phase['numerical_sources'] == sources == manifest['numerical_sources'],
                    'Recorded recovery source fingerprints differ')
            require(last_phase['missing_stems_before'] == sorted(missing),
                    'Recorded recovery has a different remaining checkpoint')
            for field in ('before_manifest','before_worker_state'):
                if field in last_phase:
                    snapshot = ROOT/last_phase[field]
                    require(io.digest(snapshot) == last_phase[field+'_sha256']
                            == manifest['input_sha256'].get(last_phase[field]),
                            'Recorded recovery snapshot fingerprint differs')
            return manifest
        phase_number = len(before['execution_entrypoint_history'])
        snapshot = data/f'provenance/manifest_before_active_face_recovery_{phase_number:02d}.json'
        snapshot_bytes(snapshot,before_bytes)
        snapshot_name = str(snapshot.relative_to(ROOT))
        snapshot_sha = io.digest(snapshot)
        require(snapshot_name not in manifest['input_sha256']
                or manifest['input_sha256'][snapshot_name] == snapshot_sha,
                'Before-recovery snapshot conflicts with an existing fingerprint')
        manifest['input_sha256'][snapshot_name] = snapshot_sha
        worker_before = {}
        worker_state = data/campaign.pair_key(.8,.8)/'worker_state.json'
        if worker_state.exists():
            worker_snapshot = data/f'provenance/worker_state_before_active_face_recovery_{phase_number:02d}.json'
            snapshot_bytes(worker_snapshot,worker_state.read_bytes())
            worker_name = str(worker_snapshot.relative_to(ROOT))
            worker_sha = io.digest(worker_snapshot)
            require(worker_name not in manifest['input_sha256']
                    or manifest['input_sha256'][worker_name] == worker_sha,
                    'Before-worker snapshot conflicts with an existing fingerprint')
            manifest['input_sha256'][worker_name] = worker_sha
            worker_before = dict(before_worker_state=worker_name,before_worker_state_sha256=worker_sha)
        manifest['numerical_sources'] = sources
        manifest['execution_entrypoint_history'].append(dict(
            phase='recorded active-face continuation for the final high-p0 grid case',
            continuation_protocol='active_face_v1',
            created_utc=io.now(), entrypoint=str(Path(__file__).resolve().relative_to(ROOT)),
            numerical_sources=dict(sources), completed_stems_before=list(before['completed']),
            missing_stems_before=sorted(missing), before_manifest=snapshot_name,
            before_manifest_sha256=snapshot_sha, initial_active_columns=list(ACTIVE_COLUMNS),
            search='Exact pinned exploratory policy evaluation with selected analytic Jacobian columns; adaptive face expansion checks full regret',
            final_certificate='Inherited full certificate with unchanged equations, 1e-11 invariant residual and 1e-7 independent agreement tolerances',
            stationary_certification=isolate(hybrid.CERTIFICATION),**worker_before))
        manifest.setdefault('recorded_continuations',[]).append(dict(
            kind='active_face_search',created_utc=io.now(),before_status=before['status'],
            before_manifest=snapshot_name,before_manifest_sha256=snapshot_sha,
            completed_before=len(before['completed']),remaining_case=sorted(missing)[0],
            imported_checkpoints_unchanged=144,existing_checkpoint_bytes_unchanged=True,
            configuration_unchanged=True,sources_added={name:value for name,value in sources.items()
                                                       if name not in before['numerical_sources']}))
        io.check_hashes(manifest['input_sha256'])
        io.check_hashes(manifest['numerical_sources'])
        require(path.read_bytes() == before_bytes, 'Master manifest changed before recovery registration')
        # The frozen controller writes this returned manifest under the same
        # exclusive lock, merges immutable worker bundles, then starts one task.
        return manifest


def controller_arguments(name,config,max_evaluations,prepare_only):
    common = sorted(set.intersection(*(set(grids) for grids in config['case_grids'].values())))
    require(common, 'Recovery configuration has no common seed grid')
    output = ['--name',name,'--resume','--workers','1','--max-evaluations',str(max_evaluations),
              '--p0s',*[str(value) for value in config['p0s']],
              '--rhos',*[str(value) for value in config['rhos']],
              '--grids',*[str(value) for value in common]]
    for p0 in config['p0s']:
        for rho in config['rhos']:
            for grid in config['case_grids'][campaign.pair_key(p0,rho)]:
                if grid not in common:
                    output += ['--add-case-grid',f'{p0}:{rho}:{grid}']
    if prepare_only:
        output += ['--prepare-only']
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default=DEFAULT_NAME)
    parser.add_argument('--workers',type=int,choices=[1],default=1)
    parser.add_argument('--max-evaluations',type=int,default=800)
    parser.add_argument('--resume',action='store_true',help='Continue the recorded recovery after --prepare-only')
    parser.add_argument('--prepare-only',action='store_true')
    args = parser.parse_args(argv)
    require(args.name and '/' not in args.name and '..' not in args.name,'Use a simple campaign name')
    require(args.max_evaluations > 0,'max-evaluations must be positive')
    data = HOME/'data'/args.name
    config = json.loads((data/'manifest.json').read_bytes())['configuration']
    previous_prepare,previous_worker = campaign.prepare,campaign.work_case
    campaign.prepare = lambda current,configured:_prepare_under_controller_lock(current,configured,previous_prepare)
    campaign.work_case = work_case
    try:
        return campaign.main(controller_arguments(args.name,config,args.max_evaluations,args.prepare_only))
    finally:
        campaign.prepare,campaign.work_case = previous_prepare,previous_worker


if __name__ == '__main__':
    raise SystemExit(main())
