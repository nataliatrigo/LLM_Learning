"""Optional second recorded search for the single remaining high-p0 case.

After the first active-face continuation fails, this continuation sets the
preceding-grid seed's coordinate 35 to product 2 and begins with coordinate 13
active. The unchanged active-face search still checks every outside inequality,
can expand the face, and invokes the complete original independent certificate.
No active controller, existing checkpoint or frozen numerical source is changed.
"""

import argparse
from contextlib import ExitStack
import fcntl
import json
from pathlib import Path

import numpy as np

from . import audit
from . import expanded_run as campaign
from . import high_p0_face_recovery as previous
from . import high_p0_worker as hybrid
from . import run_experiment as io

HOME, ROOT = io.HOME, io.ROOT
PROTOCOL = 'lower_face_v2'
_HYBRID_WORK_CASE = hybrid.work_case


def source_reference():
    references = dict(previous.source_reference())
    path = Path(__file__).resolve()
    references[str(path.relative_to(ROOT))] = io.digest(path)
    return references


class RecoveryExperiment(previous.RecoveryExperiment):
    """Select a different initial face without changing its acceptance checks."""

    def solve(self, initial_policy=None, *, max_evaluations=800, progress=None):
        previous.require((self.n,self.sellers,self.closure,self.p0,self.params.rho) == (41,3,'marginal',.8,.8),
                         'Lower-face recovery is restricted to the declared missing case')
        previous.require(initial_policy is not None,'Recovery requires a recorded preceding-grid policy seed')
        seed = np.array(initial_policy,dtype=float,copy=True)
        previous.require(seed.shape == (41,) and np.all(np.isfinite(seed)),'Recovery seed must be a finite grid-41 policy')
        seed[35] = 1.
        return self.solve_active_face(seed,[13],max_evaluations=max_evaluations,progress=progress)


def work_case(task):
    previous.require(task['p0'] == .8 and task['rho'] == .8,'Unexpected recovery parameter pair')
    original = campaign.FastExperiment
    campaign.FastExperiment = RecoveryExperiment
    try:
        return _HYBRID_WORK_CASE(task)
    finally:
        campaign.FastExperiment = original


def _prepare_under_controller_lock(data,config,original_prepare):
    path = data/'manifest.json'
    before_bytes = path.read_bytes()
    before = json.loads(before_bytes)
    previous.require(before['status'] in {'failed','interrupted','partial_failed'},
                     'Lower-face recovery requires an explicitly failed or interrupted controller')
    previous.require(config == before['configuration'],'Recovery cannot change the experiment configuration')
    declared = campaign.expected(config)
    completed = before['completed']
    previous.require(len(declared) == len(set(declared)) == 180
                     and len(completed) == len(set(completed)) == 179
                     and set(completed) <= set(declared)
                     and set(declared)-set(completed) == {io.key(**previous.RECOVERY_CASE)},
                     'Recovery requires the same 179 checkpoints and single declared missing case')
    last = before['execution_entrypoint_history'][-1]
    previous.require(last.get('continuation_protocol') in {'active_face_v1',PROTOCOL},
                     'Lower-face recovery requires the first active-face continuation to be recorded')
    previous.require(len(before['import_provenance']['imported_stems']) == 144,'Import scope changed')
    io.check_hashes(before['numerical_sources'])
    io.check_hashes(before['input_sha256'])
    audit.verify_import_provenance(data,before)
    with ExitStack() as locks:
        for worker_lock in sorted(data.glob('p0_*/rho_*/.worker.lock')):
            handle = locks.enter_context(worker_lock.open('a'))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = previous.isolate(original_prepare(data,config))
        previous.require(path.read_bytes() == before_bytes,'Manifest changed during recovery preparation')
        previous.require(manifest['configuration'] == before['configuration']
                         and manifest['completed'] == before['completed']
                         and manifest['import_provenance'] == before['import_provenance']
                         and manifest['execution_entrypoint_history'] == before['execution_entrypoint_history'],
                         'Recovery preparation changed existing provenance')
        sources = dict(before['numerical_sources'])
        for name,digest in source_reference().items():
            previous.require(name not in sources or sources[name] == digest,'Recovery would replace a frozen source')
            sources[name] = digest
        io.check_hashes(sources)
        if last.get('continuation_protocol') == PROTOCOL:
            previous.require(last['numerical_sources'] == sources == manifest['numerical_sources'],
                             'Recorded lower-face source fingerprints differ')
            for field in ('before_manifest','before_worker_state'):
                if field in last:
                    previous.require(io.digest(ROOT/last[field]) == last[field+'_sha256']
                                     == manifest['input_sha256'].get(last[field]),'Recovery snapshot fingerprint differs')
            return manifest
        phase_number = len(before['execution_entrypoint_history'])
        snapshot = data/f'provenance/manifest_before_lower_face_recovery_{phase_number:02d}.json'
        previous.snapshot_bytes(snapshot,before_bytes)
        before_fields = dict(before_manifest=str(snapshot.relative_to(ROOT)),before_manifest_sha256=io.digest(snapshot))
        worker = data/campaign.pair_key(.8,.8)/'worker_state.json'
        if worker.exists():
            saved = data/f'provenance/worker_state_before_lower_face_recovery_{phase_number:02d}.json'
            previous.snapshot_bytes(saved,worker.read_bytes())
            before_fields.update(before_worker_state=str(saved.relative_to(ROOT)),before_worker_state_sha256=io.digest(saved))
        for field in ('before_manifest','before_worker_state'):
            if field in before_fields:
                name,digest = before_fields[field],before_fields[field+'_sha256']
                previous.require(name not in manifest['input_sha256'] or manifest['input_sha256'][name] == digest,
                                 'Recovery snapshot conflicts with an existing fingerprint')
                manifest['input_sha256'][name] = digest
        manifest['numerical_sources'] = sources
        manifest['execution_entrypoint_history'].append(dict(
            phase='recorded lower-face continuation after the first active-face attempt',
            continuation_protocol=PROTOCOL,created_utc=io.now(),
            entrypoint=str(Path(__file__).resolve().relative_to(ROOT)),numerical_sources=dict(sources),
            completed_stems_before=list(completed),missing_stems_before=[io.key(**previous.RECOVERY_CASE)],
            initial_active_columns=[13],initial_seed_overrides={'35':1.},
            search='Exact pinned evaluation and selected analytic Jacobian; adaptive face expansion checks full outside regret',
            final_certificate='Unchanged inherited complete certificate; original equations and independent numerical tolerances',
            stationary_certification=previous.isolate(hybrid.CERTIFICATION),**before_fields))
        manifest.setdefault('recorded_continuations',[]).append(dict(
            kind=PROTOCOL,created_utc=io.now(),before_status=before['status'],
            completed_before=179,remaining_case=io.key(**previous.RECOVERY_CASE),
            imported_checkpoints_unchanged=144,existing_checkpoint_bytes_unchanged=True,configuration_unchanged=True,
            sources_added={name:value for name,value in sources.items() if name not in before['numerical_sources']},
            **before_fields))
        io.check_hashes(manifest['input_sha256'])
        io.check_hashes(manifest['numerical_sources'])
        previous.require(path.read_bytes() == before_bytes,'Master changed before recovery registration')
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default=previous.DEFAULT_NAME)
    parser.add_argument('--workers',type=int,choices=[1],default=1)
    parser.add_argument('--max-evaluations',type=int,default=800)
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(argv)
    previous.require(args.name and '/' not in args.name and '..' not in args.name,'Use a simple campaign name')
    previous.require(args.max_evaluations > 0,'max-evaluations must be positive')
    data = HOME/'data'/args.name
    config = json.loads((data/'manifest.json').read_bytes())['configuration']
    original_prepare,original_worker = campaign.prepare,campaign.work_case
    campaign.prepare = lambda current,configured:_prepare_under_controller_lock(current,configured,original_prepare)
    campaign.work_case = work_case
    try:
        return campaign.main(previous.controller_arguments(args.name,config,args.max_evaluations,args.prepare_only))
    finally:
        campaign.prepare,campaign.work_case = original_prepare,original_worker


if __name__ == '__main__':
    raise SystemExit(main())
