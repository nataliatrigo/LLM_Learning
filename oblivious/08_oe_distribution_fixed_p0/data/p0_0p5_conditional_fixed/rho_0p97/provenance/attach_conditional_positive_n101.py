from pathlib import Path
from hashlib import sha256
import json, shutil,time
import numpy as np
from oblivious.oe_distribution_fixed_p0.code.conditional_run import source_fingerprints,marginal_reference
from oblivious.oe_distribution_fixed_p0.code.conditional_model import FixedOutsideConditionalExperiment
from oblivious.oe_distribution_fixed_p0.code.refine_case import atomic_bundle
from oblivious.oe_distribution_fixed_p0.code.run_experiment import HOME,ROOT,audit_demand,economic_outcomes,require,write_json,write_csv,now
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import load_bundle,case_tag
rho=.97;n=101;directory=HOME/'data/p0_0p5_conditional_fixed/rho_0p97';manifest_path=directory/'manifest.json';manifest=json.loads(manifest_path.read_text());tag=case_tag(rho,n)
require(manifest['status']=='stopped_without_certificate' and manifest['current_grid'] is None,'Original worker must have stopped before attachment')
require(tag not in manifest['completed'],'Never replace an already committed case')
require(manifest['numerical_sources']==source_fingerprints(),'Numerical sources changed')
proposal_base=Path('/tmp/conditional_positive_scalar_u70_0p97_n101');proposal=load_bundle(proposal_base);c=proposal['competition'];p=Parameters(rho=rho);experiment=FixedOutsideConditionalExperiment(n,p,p0=.5,order=256)
require(proposal['configuration']==experiment.configuration(),'Proposal calibration differs')
require(c['verified'] and c['dp_diagnostics']['verified'] and c['conditional_diagnostics']['verified'],'Proposal lacks strict fresh certificate')
reference,inputs=marginal_reference(rho,n);t=time.monotonic();demand_audit=audit_demand(experiment)
# Reuse the unchanged fresh positive and reversed-order certificate already computed
# in the proposer. The policy, configuration, source fingerprints, arrays and input
# files are fixed and hashed, rather than running a duplicate stationary elimination.
alpha=c['policy'];full=experiment.game.evaluate(np.broadcast_to(alpha[:,None],(n,n)),np.broadcast_to(alpha[None,:],(n,n)),tol=1e-10);c['actual_values']=full['values'];c['actual_value_diagnostics']={key:value for key,value in full.items() if key!='values'}
outcomes,policies=economic_outcomes(p,n,.5,reference['monopoly'],c,experiment.game);outcomes=[dict(r,model='conditional_competition') for r in outcomes if r['model']=='competition'];policies=[dict(r,model='conditional_competition') for r in policies if r['model']=='competition']
provenance=directory/'provenance';provenance.mkdir(exist_ok=True)
paths=[Path('/tmp/conditional_positive_scalar48_u70_097.py'),Path('/tmp/conditional_positive_scalar48_u70_097.log'),Path('/tmp/fixed_conditional_rho097_resume.log'),Path('/tmp/conditional_positive_two_face097.py'),Path('/tmp/conditional_positive_two_face097.log'),Path('/tmp/conditional_positive_complementarity_n101.py'),Path('/tmp/conditional_positive_097_n101_complementarity.log'),Path(__file__),proposal_base.with_suffix('.json'),proposal_base.with_suffix('.npz')]
hashes={}
for path in paths:
 target=provenance/path.name;shutil.copyfile(path,target);hashes[str(target.relative_to(ROOT))]=sha256(target.read_bytes()).hexdigest()
case=dict(rho=rho,n=n,p0=.5,configuration=experiment.configuration(),monopoly=reference['monopoly'],competition=c,demand_audit=demand_audit,outcomes=outcomes,policy_rows=policies,elapsed_seconds=proposal['elapsed_seconds']+time.monotonic()-t,reference_input_sha256=inputs,refinement_metadata=dict(seed_is_proposal_only=True,seed_grid=n,certificate='Fresh positive joint law, reversed-order rowwise conditionals and strict Bellman/complementarity at every node; byte-identical certificate reused from hashed proposer bundle',monopoly='Unmodified certified same-grid historical reference',value='Fresh actual joint-environment discounted evaluation',proposal_provenance_sha256=hashes,failed_original_search=manifest['failures'].get(tag)))
require(manifest['numerical_sources']==source_fingerprints(),'Sources changed during attachment')
for source,expected in inputs.items():require(sha256((ROOT/source).read_bytes()).hexdigest()==expected,'Reference changed during attachment')
old_cases=[load_bundle(directory/'arrays'/old_tag) for old_tag in manifest['completed']]
manifest['reference_input_sha256'].update(inputs);manifest['checkpoint_sha256'].update(atomic_bundle(directory,tag,case));manifest['completed'].append(tag);manifest['failures'].pop(tag,None);manifest.update(status='ready_to_resume',current_grid=None,updated_utc=now(),positive_proposal_sha256=hashes)
write_csv(directory/'summary_all_grids.csv',[row for old in old_cases for row in old['outcomes']]+outcomes);write_csv(directory/'summary.csv',outcomes);write_csv(directory/'policies.csv',policies);write_json(manifest_path,manifest)
print('ATTACHED',json.dumps(outcomes),flush=True)
