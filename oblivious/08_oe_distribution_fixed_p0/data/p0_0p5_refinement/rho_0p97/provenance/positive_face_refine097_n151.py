"""Find candidate policies using positive stationarity; certify all states afresh."""
from pathlib import Path
import sys,time,json
import numpy as np
from scipy.optimize import brentq, root
sys.path.insert(0,'/home/natalia/Projects/LLM_Learning')
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment
from oblivious.oe_distribution_fixed_p0.code.run_experiment import numerical_sources,save_bundle
from oblivious.oe_p0.code.run_experiment import load_bundle
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.policy import evaluate_policy
from oblivious.oe_p0.code.core import checked_joint_stationary
n=int(sys.argv[1]);previous=int(sys.argv[2]);base=Path('/home/natalia/Projects/LLM_Learning/oblivious/08_oe_distribution_fixed_p0/data/p0_0p5_refinement/rho_0p97/arrays')
old=load_bundle(base/f'rho_0p97_n{previous:03d}')
e=FixedOutsideMarginalExperiment(n=n,params=Parameters(rho=.97),p0=.5,order=256);calls=0;cache={};history=[]
def calc(a):
 global calls
 a=np.clip(a,0,1);key=a.tobytes()
 if key in cache:return cache[key]
 started=time.monotonic();P=e.game.transition(np.broadcast_to(a[:,None],(n,n)),np.broadcast_to(a[None,:],(n,n)))
 st=checked_joint_stationary(e.game,P,force_positive=True);ev=evaluate_policy(e.grid,e.game.D@st['mu_b'],a,e.params)
 adv=ev['advantage'];regret=np.maximum(adv,0)-a*adv;calls+=1
 row=dict(call=calls,seconds=time.monotonic()-started,regret=float(regret.max()),violations=[dict(i=int(i),x=float(e.grid[i]),alpha=float(a[i]),advantage=float(adv[i])) for i in np.flatnonzero(regret>1e-9)])
 history.append(row);print(json.dumps(row),flush=True)
 result=dict(advantage=adv,regret=regret,st=st);cache[key]=result;return result

def certify(a,label):
 r=e.evaluate(a);print('CERT',label,r['verified'],r['dp_diagnostics'],flush=True)
 if r['verified']:
  save_bundle(Path(f'/tmp/rho097_positive_face_n{n}'),dict(competition=r,source_fingerprints=numerical_sources(),method='Positive-stationary exploratory active-set complementarity; final unchanged full certificate',search_history=history))
  raise SystemExit(0)
 return r
alpha=(np.interp(e.grid,old['competition']['grid'],old['competition']['policy'])>=.5).astype(float)
visited={};faces=[]
for iteration in range(8):
 result=calc(alpha);bad=np.flatnonzero(result['regret']>1e-9)
 if len(bad)==0:certify(alpha,'rounded/updated previous policy');break
 print('BAD',bad.tolist(),flush=True)
 # Every violating action is eligible; no threshold form is maintained.
 for i in bad:
  a0=alpha.copy();a0[i]=0;f0=calc(a0)['advantage'][i]
  a1=alpha.copy();a1[i]=1;f1=calc(a1)['advantage'][i]
  if f0*f1<0:
   baseline=alpha.copy()
   def expand(w):
    a=baseline.copy();a[i]=w;return a
   w=brentq(lambda w:calc(expand(w))['advantage'][i],0,1,xtol=1e-13,rtol=1e-13)
   alpha=expand(w);checked=certify(alpha,f'positive mixed root at {i}')
  else:alpha[i]=float(result['advantage'][i]>0)
 signature=alpha.tobytes()
 if signature in visited:break
 visited[signature]=iteration
raise ArithmeticError('No fully certified positive-face proposal in finite exploratory search')
