from pathlib import Path
import sys,time,json,hashlib
import numpy as np
from scipy.optimize import brentq, root
sys.path.insert(0,'/home/natalia/Projects/LLM_Learning')
from oblivious.oe_distribution_fixed_p0.code.multi_p0_endpoint_recovery import EndpointSafeConditionalExperiment
from oblivious.oe_distribution_fixed_p0.code.conditional_model import conditional_from_joint
from oblivious.oe_distribution_fixed_p0.code.multi_p0_run import source_fingerprints
from oblivious.oe_distribution_fixed_p0.code.run_experiment import save_bundle, numerical_sources
from oblivious.oe_p0.code.run_experiment import load_bundle
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.policy import evaluate_policy
from oblivious.oe_p0.code.core import checked_joint_stationary
n=101;e=EndpointSafeConditionalExperiment(n=n,params=Parameters(rho=.97),p0=.1,order=256)
calls=0;cache={};history=[]
def calc(a):
 global calls
 a=np.clip(a,0,1);key=a.tobytes()
 if key in cache:return cache[key]
 t=time.monotonic();P=e.game.transition(np.broadcast_to(a[:,None],(n,n)),np.broadcast_to(a[None,:],(n,n)))
 st=checked_joint_stationary(e.game,P,force_positive=True)
 ev=evaluate_policy(e.grid,conditional_from_joint(st['nu'],e.game.D)['conditional_demand'],a,e.params)
 adv=ev['advantage'];regret=np.maximum(adv,0)-a*adv;calls+=1
 bad=np.flatnonzero(regret>1e-10)
 row=dict(call=calls,seconds=time.monotonic()-t,regret=float(regret.max()),violations=[dict(i=int(i),alpha=float(a[i]),advantage=float(adv[i])) for i in bad])
 history.append(row);print(json.dumps(row),flush=True)
 r=dict(advantage=adv,regret=regret,stationary=st);cache[key]=r;return r
def certify(a,label):
 r=e.evaluate(a);print('CERT',label,r['verified'],r['dp_diagnostics'],flush=True)
 if r['verified']:
  save_bundle(Path('/tmp/p01_r097_conditional_candidate'),dict(competition=r,configuration=e.configuration(),source_fingerprints=source_fingerprints(),method='Positive-stationary exploratory active-face search; unchanged full certificate',search_history=history,proposal_script_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path('/home/natalia/Projects/LLM_Learning/oblivious/08_oe_distribution_fixed_p0/code/multi_p0_endpoint_recovery.py'))}))
  np.save('/tmp/p01_r097_conditional_policy.npy',a)
  raise SystemExit(0)
 return r

for upper in (68,69,67,70,66,65,64):
 for lower in (48,47,49):
  baseline=np.zeros(n);baseline[lower+1:upper+1]=1
  def expand(w):
   a=baseline.copy();a[lower]=w;return a
  f0=calc(expand(0))['advantage'][lower];f1=calc(expand(1))['advantage'][lower]
  print('BRACKET',lower,upper,float(f0),float(f1),flush=True)
  if f0*f1<0:
   w=brentq(lambda w:calc(expand(w))['advantage'][lower],0,1,xtol=1e-13,rtol=1e-13)
   a=expand(w);rr=calc(a);print('ROOT',lower,upper,w,float(rr['regret'].max()),flush=True)
   if rr['regret'].max()<1e-8:certify(a,'root lower'+str(lower)+'upper'+str(upper))
raise ArithmeticError('No certificate in finite conditional endpointface scan')
