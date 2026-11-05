"""Complete and publish a signed-law search proposal with a fresh full positive certificate.

Usage: python attach_followup_proposal.py 151 [or 201]. The exact worker for
this rho must be stopped first. Reuse the contemporary e.evaluate certificate
only after immutable source/config checks, then audit demand, solve monopoly,
evaluate actual joint discounted values, and check every economic identity.
"""
from pathlib import Path
from hashlib import sha256
import json
import sys
import time
import numpy as np
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'oblivious/layout.py').exists())
sys.path.insert(0,str(ROOT))
from oblivious.oe_distribution_fixed_p0.code.refine_case import source_fingerprints, atomic_bundle
from oblivious.oe_distribution_fixed_p0.code.run_experiment import (
 audit_demand,economic_outcomes,numerical_sources,require,solve_monopoly,write_json,now,case_tag)
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import load_bundle
from oblivious.oe_p0.code.policy import evaluate_policy

def main():
 n=int(sys.argv[1]);directory=Path(__file__).resolve().parents[1];provenance=directory/'provenance'
 path=directory/'manifest.json';manifest=json.loads(path.read_text());tag=case_tag(.97,n)
 require(manifest['numerical_sources']==source_fingerprints(),'Sources changed')
 require(all(manifest['configuration'][k]==v for k,v in dict(rho=.97,p0=.5,order=256,tol=1e-7,dp_tol=1e-10).items()),'Configuration differs')
 require(tag not in manifest['completed'],'Already committed; refusing overwrite')
 proposal=load_bundle(provenance/f'rho097_positive_face_n{n}')
 require(proposal['source_fingerprints']==numerical_sources(),'Proposal sources changed')
 competition=proposal['competition'];require(competition['verified'],'No fresh full proposal certificate')
 started=time.monotonic();params=Parameters(rho=.97)
 experiment=FixedOutsideMarginalExperiment(n=n,params=params,p0=.5,order=256,tol=1e-7,dp_tol=1e-10)
 require(np.array_equal(competition['grid'],experiment.grid),'Proposal grid differs')
 require(competition['summary']['rho']==.97 and competition['summary']['n']==n,'Proposal parameters differ')
 alpha=competition['policy'];nu=competition['nu']
 kernel=experiment.game.transition(np.broadcast_to(alpha[:,None],(n,n)),np.broadcast_to(alpha[None,:],(n,n)))
 require(float(np.abs(kernel.T@nu.ravel()-nu.ravel()).sum())<=1e-11,'Imported joint law fails fresh kernel residual')
 demand=experiment.game.D@competition['mu_b']
 require(float(np.max(abs(demand-competition['demand'])))<=1e-12,'Proposal demand differs')
 checked=evaluate_policy(experiment.grid,demand,alpha,params)
 regret=float(np.max(np.maximum(checked['advantage'],0)-alpha*checked['advantage']))
 require(regret<=1e-10 and checked['residual']<=1e-10,'Imported policy fails fresh action inequalities')
 require(float(np.max(abs(checked['value']-competition['value'])))<=1e-10,'Imported perceived values differ')
 print('Fresh demand audit and monopoly',flush=True)
 demand_audit=audit_demand(experiment);monopoly=solve_monopoly(params,n,.5,1e-10)
 competition['history']=proposal['search_history']
 competition['positive_polish_history']=proposal['positive_polish_history']
 competition['search_metadata']=dict(method=proposal['method'],
  source=f'provenance/sparse_face_refine097_n{n}.py',exploratory_evaluations=len(proposal['search_history']),
  fresh_certifications=len(proposal['positive_polish_history']),exhaustive=False,unique=False,
  policy_restriction='None: all endpoint inequalities checked by final certificate')
 print('Actual joint discounted values',flush=True)
 full=experiment.game.evaluate(np.broadcast_to(alpha[:,None],(n,n)),np.broadcast_to(alpha[None,:],(n,n)),tol=1e-10)
 competition['actual_values']=full['values'];competition['actual_value_diagnostics']={k:v for k,v in full.items() if k!='values'}
 competition['attempts']=[dict(initialization='independent sparse signed-law proposal with full positive certificate',verified=True,
  diagnostics=competition['dp_diagnostics'],search=competition['search_metadata'])]
 outcomes,policies=economic_outcomes(params,n,.5,monopoly,competition,experiment.game)
 previous_n=max(int(t.rsplit('_n',1)[1]) for t in manifest['completed'])
 case=dict(rho=.97,n=n,p0=.5,configuration=experiment.configuration(),monopoly=monopoly,
  competition=competition,demand_audit=demand_audit,outcomes=outcomes,policy_rows=policies,
  elapsed_seconds=time.monotonic()-started,refinement_metadata=dict(seed_grid=previous_n,
   seed_is_proposal_only=True,elapsed_seconds_scope='Attachment stage only; positive search response times are recorded separately in competition.history',
   exploratory_search_response_seconds=sum(row['seconds'] for row in proposal['search_history']),
   exploratory_stationary_law='Sparse signed jump-chain solve; search proposal only',
   final_stationary_certificate='Unchanged full positive GTH certificate through e.evaluate',certificate='Fresh unchanged full e.evaluate in proposal; immutable source/config check; fresh demand/monopoly/actual values/flow checks at attachment',
   previous_completed_tag=case_tag(.97,previous_n),supplementary_search='Sparse signed-law active-set candidates with fresh full positive certificate',
   search_log=f'provenance/sparse_face_refine097_n{n}.log',
   prior_search_route='Certified n151 policy used as the proposal seed; no n201 continuation run'))
 require(manifest['numerical_sources']==source_fingerprints(),'Sources changed during attachment')
 manifest['checkpoint_sha256'].update(atomic_bundle(directory,tag,case));manifest['completed'].append(tag)
 manifest['failures'].pop(tag,None);manifest.update(status='running',current_grid=None,updated_utc=now())
 manifest['supplemental_search_provenance']={str(p.relative_to(directory)):sha256(p.read_bytes()).hexdigest()
  for p in sorted(provenance.iterdir()) if p.is_file()}
 write_json(path,manifest)
 print('CERTIFIED',tag,json.dumps(outcomes),'attachment_seconds',case['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
