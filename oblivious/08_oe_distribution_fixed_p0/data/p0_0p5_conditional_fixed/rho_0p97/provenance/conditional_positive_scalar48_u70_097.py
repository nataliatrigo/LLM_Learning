from pathlib import Path
import time
import numpy as np
from scipy.optimize import brentq
from oblivious.oe_distribution_fixed_p0.code.conditional_model import FixedOutsideConditionalExperiment,conditional_from_joint
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.policy import evaluate_policy
from oblivious.oe_p0.code.core import checked_joint_stationary
from oblivious.oe_p0.code.run_experiment import save_bundle
n=101;e=FixedOutsideConditionalExperiment(n,Parameters(rho=.97),p0=.5,order=256);calls=0;t=time.time()
base=np.zeros(n);base[49:71]=1

def response(weight):
 global calls
 alpha=base.copy();alpha[48]=weight
 P=e.game.transition(np.broadcast_to(alpha[:,None],(n,n)),np.broadcast_to(alpha[None,:],(n,n)))
 inv=checked_joint_stationary(e.game,P,force_positive=True)
 demand=conditional_from_joint(inv['nu'],e.game.D)['conditional_demand']
 r=evaluate_policy(e.grid,demand,alpha,e.params);calls+=1
 print('response',calls,'alpha48',weight,'a48',r['advantage'][48],'a69',r['advantage'][69],'regret',float((np.maximum(r['advantage'],0)-alpha*r['advantage']).max()),flush=True)
 return r['advantage'][48]
a,b=response(0),response(1)
if a*b>=0:raise RuntimeError('No scalar bracket')
x=brentq(response,0,1,xtol=1e-12,rtol=1e-14);alpha=base.copy();alpha[48]=x;r=e.evaluate(alpha)
print('FULL',r['verified'],r['dp_diagnostics'],flush=True)
if not r['verified']:
 bad=np.flatnonzero(np.maximum(r['advantage'],0)-alpha*r['advantage']>1e-10);print('bad',bad.tolist(),'a',alpha[bad].tolist(),'adv',r['advantage'][bad].tolist(),flush=True)
 raise RuntimeError('Full certificate failed')
r['search_metadata']={'method':'Scalar active-face positive joint stationary proposal; every grid node freshly certified','positive_responses':calls,'exhaustive':False,'unique':False}
save_bundle(Path('/tmp/conditional_positive_scalar_u70_0p97_n101'),dict(competition=r,configuration=e.configuration(),elapsed_seconds=time.time()-t))
