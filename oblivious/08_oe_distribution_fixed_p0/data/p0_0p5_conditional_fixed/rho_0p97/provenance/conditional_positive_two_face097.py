from pathlib import Path
import argparse,time,json
import numpy as np
from scipy.optimize import root
from oblivious.oe_distribution_fixed_p0.code.conditional_model import FixedOutsideConditionalExperiment,conditional_from_joint
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.policy import evaluate_policy
from oblivious.oe_p0.code.core import checked_joint_stationary
from oblivious.oe_p0.code.run_experiment import save_bundle
parser=argparse.ArgumentParser();parser.add_argument('--rho',type=float,required=True);parser.add_argument('--n',type=int,default=51);parser.add_argument('--lower',type=int,required=True);parser.add_argument('--upper',type=int,required=True);args=parser.parse_args()
e=FixedOutsideConditionalExperiment(args.n,Parameters(rho=args.rho),p0=.5,order=256)
n=args.n;t=time.time();calls=0

def response(alpha):
 global calls
 P=e.game.transition(np.broadcast_to(alpha[:,None],(n,n)),np.broadcast_to(alpha[None,:],(n,n)))
 inv=checked_joint_stationary(e.game,P,force_positive=True)
 demand=conditional_from_joint(inv['nu'],e.game.D)['conditional_demand']
 r=evaluate_policy(e.grid,demand,alpha,e.params);calls+=1
 if calls%5==0: print('call',calls,'a48',alpha[48],'a70',alpha[70],'adv48',r['advantage'][48],'adv70',r['advantage'][70],flush=True)
 return r['advantage']

alpha=np.zeros(n);alpha[49:70]=1;alpha[48]=.9585165727855545;alpha[70]=.25
for iteration in range(20):
 adv=response(alpha);regret=np.maximum(adv,0)-alpha*adv;bad=np.flatnonzero(regret>1e-10);mixed=np.flatnonzero((alpha>0)&(alpha<1))
 print('iteration',iteration,'regret',regret.max(),'bad',bad.tolist(),'mixed',mixed.tolist(),'weights',alpha[mixed].tolist(),flush=True)
 if regret.max()<=1e-10:
  result=e.evaluate(alpha)
  print('FULL',result['verified'],result['dp_diagnostics'],flush=True)
  if result['verified']:
   result['search_metadata']={'method':'Positive joint stationary active-face proposal; every node freshly certified','positive_responses':calls,'exhaustive':False,'unique':False}
   path=Path('/tmp/conditional_positive_two_face_'+str(args.rho).replace('.','p')+'_n'+str(n));save_bundle(path,dict(competition=result,configuration=e.configuration(),elapsed_seconds=time.time()-t));np.save(str(path)+'_policy.npy',alpha);break
 active=np.unique(np.r_[bad,mixed]);base=alpha.copy()
 def face(x):
  trial=base.copy();trial[active]=np.clip(x,0,1)
  return response(trial)[active]
 solved=root(face,np.clip(alpha[active],1e-4,1-1e-4),method='hybr',options={'xtol':1e-10,'eps':1e-7,'maxfev':80,'factor':.3})
 proposed=base.copy();proposed[active]=np.clip(solved.x,0,1)
 # Project root excursions to pure endpoints, then re-evaluate all nodes.
 proposed[proposed<1e-7]=0;proposed[proposed>1-1e-7]=1
 print('face',active.tolist(),'x',solved.x.tolist(),'residual',solved.fun.tolist(),'success',solved.success,flush=True)
 if np.max(abs(proposed-alpha))<1e-10 and len(bad):
  proposed[bad]=(adv[bad]>0).astype(float)
 alpha=proposed
else: raise RuntimeError('No verified face found')
