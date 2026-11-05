"""Attach a byte-identical fresh strict proposal after the local writer stops."""
from hashlib import sha256
import json
from pathlib import Path
import shutil
import time

import numpy as np

from oblivious.oe_distribution_fixed_p0.code.conditional_model import FixedOutsideConditionalExperiment
from oblivious.oe_distribution_fixed_p0.code.conditional_refine_case import (
    conditional_seed, source_fingerprints,
)
from oblivious.oe_distribution_fixed_p0.code.conditional_run import marginal_reference, stability
from oblivious.oe_distribution_fixed_p0.code.refine_case import atomic_bundle
from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
    HOME, ROOT, audit_demand, case_tag, economic_outcomes, now, require, write_csv, write_json,
)
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import load_bundle


started=time.monotonic();rho=.97;n=151
directory=HOME/'data/p0_0p5_conditional_refinement/rho_0p97'
path=directory/'manifest.json';manifest=json.loads(path.read_text())
require(manifest['current_grid'] is None
        and manifest['status']=='interrupted_for_positive_face_proposal',
        'Original writer must have stopped before attachment')
require(manifest['completed']==[], 'Existing checkpoints protected')
sources=source_fingerprints()
require(manifest['numerical_sources']==sources,'Frozen refinement sources changed')
seed,seed_inputs=conditional_seed(rho)
reference,monopoly_inputs=marginal_reference(rho,n)
inputs=dict(seed_inputs,**monopoly_inputs)
require(manifest['reference_input_sha256']==inputs,'Frozen reference inputs changed')
proposal_prefix=Path('/tmp/conditional_positive_scalar73_u105_0p97_n151')
proposal=load_bundle(proposal_prefix)
require(proposal['source_fingerprints']==sources,'Proposal numerical source fingerprints differ')
proposer=Path('/tmp/conditional_positive_scalar73_u105_097_n151.py')
require(proposal['proposer_sha256']==sha256(proposer.read_bytes()).hexdigest(),
        'Positive proposer source changed after certification')
e=FixedOutsideConditionalExperiment(n,Parameters(rho=rho),p0=.5,order=256,tol=1e-7,dp_tol=1e-10)
require(proposal['configuration']==e.configuration(),'Proposal configuration differs')
c=proposal['competition'];alpha=c['policy']
require(alpha.shape==e.grid.shape and np.array_equal(c['grid'],e.grid)
        and np.isfinite(alpha).all() and np.all((alpha>=0)&(alpha<=1)),
        'Invalid proposal grid or policy')
require(proposal['policy_sha256']==sha256(alpha.tobytes()).hexdigest(),
        'Policy differs from independently certified proposal')
require(c['verified'] and c['dp_diagnostics']['verified']
        and c['conditional_diagnostics']['verified']
        and c['conditional_diagnostics']['independent_order_verified']
        and c['stationary_diagnostics']['verified'],
        'Fresh full positive/reversed-order certificate missing')
print('Reusing exact fresh full strict certificate; policy/configuration/source hashes match',flush=True)
audit=audit_demand(e)
full=e.game.evaluate(np.broadcast_to(alpha[:,None],(n,n)),np.broadcast_to(alpha[None,:],(n,n)),tol=1e-10)
c['actual_values']=full['values']
c['actual_value_diagnostics']={key:value for key,value in full.items() if key!='values'}
c['certificate_provenance']=dict(
    fresh_strict_certificate_reused=True,policy_sha256=proposal['policy_sha256'],
    numerical_sources=sources,proposal_producer_sha256=proposal['proposer_sha256'],
    rule='No policy, configuration, demand, source, or certificate tolerance changes after the fresh full positive and reversed-order evaluation')
outcomes,policies=economic_outcomes(e.params,n,.5,reference['monopoly'],c,e.game)
outcomes=[dict(row,model='conditional_competition') for row in outcomes if row['model']=='competition']
policies=[dict(row,model='conditional_competition') for row in policies if row['model']=='competition']
case=dict(rho=rho,n=n,p0=.5,configuration=e.configuration(),
    monopoly=reference['monopoly'],competition=c,demand_audit=audit,outcomes=outcomes,
    policy_rows=policies,elapsed_seconds=proposal['elapsed_seconds']+time.monotonic()-started,
    reference_input_sha256=inputs,
    refinement_metadata=dict(seed_is_proposal_only=True,seed_grid=101,
        certificate='Fresh unchanged conditional OE certificate with positive joint law and independent reversed-order rowwise check; byte-identical certificate reuse on attachment',
        monopoly='Unmodified certified n151 historical reference',
        value='Fresh actual joint-environment discounted evaluation',
        external_positive_face_proposal=True,
        abandoned_attempt=manifest['interrupted_local_attempt']))
require(sources==source_fingerprints(),'Sources changed during attachment')
for source,expected in inputs.items():
    require(sha256((ROOT/source).read_bytes()).hexdigest()==expected,
            'Historical or conditional reference changed during attachment')
tag=case_tag(rho,n)
manifest['checkpoint_sha256'].update(atomic_bundle(directory,tag,case))
manifest['completed'].append(tag)
comparison=stability(seed,case)
write_csv(directory/'summary_all_grids.csv',[*seed['outcomes'],*case['outcomes']])
write_csv(directory/'summary.csv',case['outcomes'])
write_csv(directory/'policies.csv',case['policy_rows'])
write_csv(directory/'grid_stability.csv',[comparison])
provenance=directory/'provenance'
for artifact in (
    Path('/tmp/conditional_round_n151_097.py'),
    Path('/tmp/conditional_round_0p97_n151.json'),
    Path('/tmp/conditional_round_0p97_n151.npz'),
    Path('/tmp/fixed_conditional_rounded_rho097_n151.log'),
    proposer,proposal_prefix.with_suffix('.json'),proposal_prefix.with_suffix('.npz'),
    Path('/tmp/fixed_conditional_positive_scalar73_u105_rho097_n151.log'),
    Path('/tmp/fixed_conditional_rho097_n151_preflight.json'),
    HOME/'code/conditional_refine_case.py',HOME/'code/conditional_model.py',Path(__file__)):
    shutil.copyfile(artifact,provenance/artifact.name)
write_json(provenance/'verification_diagnostics.json',dict(
    archived_utc=now(),rho=rho,n=n,dp=c['dp_diagnostics'],
    conditional=c['conditional_diagnostics'],stationary=c['stationary_diagnostics'],
    demand_audit=audit,actual_value_diagnostics=c['actual_value_diagnostics'],
    certificate_provenance=c['certificate_provenance'],grid_stability=comparison))
manifest.update(status='grid_stable_under_declared_thresholds' if comparison['passed']
                else 'certified_but_grid_sensitive',current_grid=None,
                grid_stability=comparison,finished_utc=now(),
                positive_face_proposal_sha256={
                    str((provenance/filename).relative_to(directory)):
                    sha256((provenance/filename).read_bytes()).hexdigest()
                    for filename in (proposal_prefix.with_suffix('.json').name,
                                     proposal_prefix.with_suffix('.npz').name,proposer.name)})
manifest['failures'].pop(tag,None)
manifest['diagnostic_artifact_sha256']={str(item.relative_to(directory)):
    sha256(item.read_bytes()).hexdigest() for item in provenance.iterdir() if item.is_file()}
write_json(path,manifest)
print('ATTACHED',manifest['status'],json.dumps(comparison),json.dumps(outcomes),flush=True)
