"""Independently certify a policy-only .3/.9/n201 backup in isolated staging.

The active sweep case is never written. The unchanged run_grid pipeline freshly
audits demand, solves monopoly, certifies the proposed marginal policy, evaluates
actual joint discounted values and writes its original atomic checkpoint format.
An uncertified proposal contributes only its policy, never laws or values.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import traceback

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import load_bundle
from oblivious.oe_distribution_fixed_p0.code import multi_p0_run as runner
from oblivious.oe_distribution_fixed_p0.code import sweep201_run as worker
from oblivious.oe_distribution_fixed_p0.code import sweep201_dense_gth as backend

STAGING = runner.HOME/'data/p0_sweep_201/recovery/p0_0p3/rho_0p9'
DEFAULT_PROPOSAL = Path('/tmp/sweep201-active-face-proposals/p03_single_face_recovery_n201')


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def helper_reference():
    path = Path(__file__).resolve()
    return dict(backend.helper_reference(),**{str(path.relative_to(runner.ROOT)):digest(path)})


def prepare_proposal(prefix,directory):
    prefix = Path(prefix).resolve()
    if prefix.suffix in ('.json','.npz'):
        prefix = prefix.with_suffix('')
    proposal = load_bundle(prefix)
    runner.require(proposal['source_fingerprints'] == runner.source_fingerprints(),
                   'Proposal numerical sources differ from the frozen target sources')
    configuration = proposal['configuration']
    runner.require(configuration['n'] == 201 and configuration['p0'] == .3
                   and configuration['parameters'] == asdict(Parameters(rho=.9)),
                   'Proposal target differs from .3/.9/n201 staging')
    alpha = np.asarray(proposal['competition']['policy'],dtype=float)
    grid = np.asarray(proposal['competition']['grid'],dtype=float)
    runner.require(alpha.shape == (201,) and np.isfinite(alpha).all()
                   and np.all((alpha >= 0)&(alpha <= 1))
                   and np.array_equal(grid,np.linspace(0,1,201)),
                   'Proposal policy/grid is malformed')
    inputs = dict(proposal.get('reference_input_sha256',{}))
    for source,expected in inputs.items():
        runner.require(digest(runner.ROOT/source) == expected,'Proposal reference changed: '+source)
    source_files = [prefix.with_suffix(suffix) for suffix in ('.json','.npz')]
    scripts = proposal['proposal_script_sha256']
    for source,expected in scripts.items():
        path = Path(source)
        if not path.is_absolute():
            path = runner.ROOT/path
        runner.require(path.is_file() and digest(path) == expected,
                       'Proposal script changed: '+str(path))
        source_files.append(path)
    archived = directory/'provenance/policy_only_backup'
    archived.mkdir(parents=True,exist_ok=True)
    seen = {}
    for source in source_files:
        expected = digest(source)
        target = archived/source.name
        runner.require(target.name not in seen or seen[target.name] == expected,
                       'Proposal provenance filename collision')
        seen[target.name] = expected
        if target.exists():
            runner.require(digest(target) == expected,'Protected archived proposal changed: '+str(target))
        else:
            pending = target.with_name('_pending_'+target.name)
            shutil.copyfile(source,pending)
            runner.require(digest(pending) == expected,'Proposal changed during archival')
            pending.replace(target)
        runner.require(digest(source) == expected and digest(target) == expected,
                       'Proposal provenance changed during archival')
        inputs[str(target.relative_to(runner.ROOT))] = expected
    inputs.update(helper_reference())
    return dict(configuration=configuration,policy=alpha.copy(),input_sha256=inputs,
                archived_proposal=str((archived/prefix.name).relative_to(runner.ROOT)))


def certify_policy_only(experiment,proposal):
    runner.require(experiment.configuration() == proposal['configuration'],
                   'Fresh target experiment differs from the archived policy proposal')
    for source,expected in proposal['input_sha256'].items():
        runner.require(digest(runner.ROOT/source) == expected,'Staging input changed: '+source)
    fresh = experiment.evaluate(proposal['policy'].copy())
    runner.require(fresh.get('verified',False)
                   and fresh.get('dp_diagnostics',{}).get('verified',False)
                   and fresh.get('stationary_diagnostics',{}).get('verified',False),
                   'Original full target certificate did not verify the staged policy')
    fresh['attempts'] = [dict(initialization='archived_uncertified_policy_only_backup',
                             stage='fresh_original_full_certificate',verified=True,
                             dp=fresh['dp_diagnostics'])]
    fresh['search_metadata'] = dict(
        method='Fresh original full certificate of policy-only backup in isolated staging',
        seed_is_proposal_only=True,imported_stationary_or_value_reuse=False,
        fresh_full_certificate=True,unique=False,exhaustive=False,
        archived_proposal=proposal['archived_proposal'],
        external_proposal_input_sha256=proposal['input_sha256'],staging_only=True)
    return fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--proposal',type=Path,default=DEFAULT_PROPOSAL)
    parser.add_argument('--resume',action='store_true')
    options = parser.parse_args()
    args = SimpleNamespace(p0=.3,rho=.9,grids=[201],order=256,tol=1e-7,dp_tol=1e-10,
                           max_evaluations=250,resume=options.resume)
    path = STAGING/'manifest.json'
    configuration = dict(p0=.3,rho=.9,order=256,tol=1e-7,dp_tol=1e-10,
        parameters={k:v for k,v in asdict(Parameters()).items() if k != 'rho'})
    if options.resume:
        manifest = json.loads(path.read_text())
        runner.require(manifest['configuration'] == configuration,'Staging calibration changed')
        runner.require(manifest['numerical_sources'] == runner.source_fingerprints(),
                       'Staging numerical sources changed')
        for source,expected in manifest['reference_input_sha256'].items():
            runner.require(digest(runner.ROOT/source) == expected,'Saved staging input changed: '+source)
    else:
        runner.require(not STAGING.exists(),'Protected staging directory exists; use --resume')
        (STAGING/'arrays').mkdir(parents=True)
        (STAGING/'provenance').mkdir()
        manifest = dict(created_utc=runner.now(),configuration=configuration,
            numerical_sources=runner.source_fingerprints(),requested_grids=[201],completed=[],
            checkpoint_sha256={},reference_input_sha256={},failures={},attempts=[],
            staging_only=True,staged_models=['monopoly','competition'],
            explanation='Isolated independently certified recovery, never an active-case write',
            current_grid=None,current_model=None)
    proposal = prepare_proposal(options.proposal,STAGING)
    worker.install(args)
    backend.install(threads=2)
    baseline = runner.baseline_seed

    def baseline_with_staging_sources(rho,n,closure):
        seed,inputs = baseline(rho,n,closure)
        return seed,dict(inputs,**proposal['input_sha256'])

    def solve_only_fresh_backup(experiment,proposals,max_evaluations,log):
        log('  fresh original certificate of archived POLICY ONLY backup; no imported laws/values')
        return certify_policy_only(experiment,proposal)

    runner.baseline_seed = baseline_with_staging_sources
    runner.solve_from_proposals = solve_only_fresh_backup
    manifest['reference_input_sha256'].update(proposal['input_sha256'])
    manifest['attempts'].append(dict(started_utc=runner.now(),grids=[201],
        models=['monopoly','competition'],entrypoint='sweep201_stage_p03_recovery',
        dense_gth_threads=2,dense_memory_budget_gib=24,imported_policy_only=True))
    manifest.update(status='staging_running',current_grid=None,current_model=None)
    runner.write_json(path,manifest)

    def log(message):
        with (STAGING/'provenance/solver.log').open('a') as stream:
            stream.write(message+'\n')
        print(message,flush=True)

    try:
        runner.run_grid(args,STAGING,manifest,201,'marginal',log)
        runner.export_completed(STAGING,manifest)
    except (Exception,KeyboardInterrupt) as error:
        log(traceback.format_exc())
        manifest['failures']['staged_rho_0p9_n201_marginal'] = dict(
            error=str(error),failed_utc=runner.now(),interrupted=isinstance(error,KeyboardInterrupt))
        manifest.update(status='staging_stopped_without_new_certificate',current_grid=None,
                        current_model=None,updated_utc=runner.now())
        runner.export_completed(STAGING,manifest)
        runner.write_json(path,manifest)
        raise
    manifest.update(status='staged_monopoly_and_marginal_certified',current_grid=None,
                    current_model=None,finished_utc=runner.now())
    runner.write_json(path,manifest)
    log('STAGED CERTIFIED: '+str(path))


if __name__ == '__main__':
    main()
