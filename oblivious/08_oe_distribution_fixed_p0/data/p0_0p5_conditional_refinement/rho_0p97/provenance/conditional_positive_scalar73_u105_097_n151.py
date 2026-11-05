from hashlib import sha256
from pathlib import Path
import time
import numpy as np
from scipy.optimize import brentq
from oblivious.oe_distribution_fixed_p0.code.conditional_model import FixedOutsideConditionalExperiment,conditional_from_joint
from oblivious.oe_distribution_fixed_p0.code.conditional_refine_case import source_fingerprints
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.policy import evaluate_policy
from oblivious.oe_p0.code.core import checked_joint_stationary
from oblivious.oe_p0.code.run_experiment import load_bundle,save_bundle

t=time.monotonic(); sources=source_fingerprints()
n=151;e=FixedOutsideConditionalExperiment(n,Parameters(rho=.97),p0=.5,order=256)
base=np.zeros(n);base[74:106]=1
pure=load_bundle(Path('/tmp/conditional_round_0p97_n151'))
expected=base.copy();expected[73]=1
if not np.array_equal(pure['competition']['policy'],expected):raise RuntimeError('Pure endpoint policy mismatch')
if pure['configuration']!=e.configuration() or pure['source_fingerprints']!=sources:raise RuntimeError('Endpoint model mismatch')
cache={1.:float(pure['competition']['advantage'][73])};calls=0

def response(weight):
 global calls
 if float(weight) in cache:return cache[float(weight)]
 alpha=base.copy();alpha[73]=weight
 matrix=e.game.transition(np.broadcast_to(alpha[:,None],(n,n)),np.broadcast_to(alpha[None,:],(n,n)))
 stationary=checked_joint_stationary(e.game,matrix,force_positive=True)
 demand=conditional_from_joint(stationary['nu'],e.game.D)['conditional_demand']
 checked=evaluate_policy(e.grid,demand,alpha,e.params);calls+=1
 value=float(checked['advantage'][73]);cache[float(weight)]=value
 print('response',calls,'alpha73',weight,'a73',value,'a105',checked['advantage'][105],
       'regret',float((np.maximum(checked['advantage'],0)-alpha*checked['advantage']).max()),flush=True)
 return value

a,b=response(0.),response(1.)
print('bracket',a,b,flush=True)
if a*b>=0:raise RuntimeError('No interior scalar bracket; full endpoint certificate locates required face')
x=brentq(response,0.,1.,xtol=1e-12,rtol=1e-14)
alpha=base.copy();alpha[73]=x
r=e.evaluate(alpha)
regret=np.maximum(r['advantage'],0)-alpha*r['advantage'];bad=np.flatnonzero(regret>1e-10)
print('FULL',r['verified'],r['dp_diagnostics'],flush=True)
print('mixed alpha73',x,'bad',bad.tolist(),'alpha',alpha[bad].tolist(),'advantage',r['advantage'][bad].tolist(),flush=True)
if sources!=source_fingerprints():raise RuntimeError('Numerical source changed')
r['search_metadata']=dict(method='Scalar lower active face with unchanged positive joint stationary law; all nodes freshly checked with reversed-order conditional certification',positive_responses=calls,exhaustive=False,unique=False)
save_bundle(Path('/tmp/conditional_positive_scalar73_u105_0p97_n151'),dict(
 competition=r,configuration=e.configuration(),elapsed_seconds=time.monotonic()-t,
 source_fingerprints=sources,policy_sha256=sha256(alpha.tobytes()).hexdigest(),
 proposer_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
 positive_endpoint_source='/tmp/conditional_round_0p97_n151'))
print('SAVED elapsed',time.monotonic()-t,flush=True)
