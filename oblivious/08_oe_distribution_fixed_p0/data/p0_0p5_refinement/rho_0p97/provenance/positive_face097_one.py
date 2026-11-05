from pathlib import Path
import sys,time,json
import numpy as np
from scipy.optimize import root,brentq
sys.path.insert(0,'/home/natalia/Projects/LLM_Learning')
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment
from oblivious.oe_distribution_fixed_p0.code.run_experiment import check_bundle, numerical_sources, save_bundle
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.policy import evaluate_policy
from oblivious.oe_p0.code.core import checked_joint_stationary
base=Path('/home/natalia/Projects/LLM_Learning/oblivious/08_oe_distribution_fixed_p0/data/p0_0p5');m=json.loads((base/'manifest.json').read_text());old=check_bundle(base,'rho_0p97_n051',m)
e=FixedOutsideMarginalExperiment(n=101,params=Parameters(rho=.97),p0=.5,order=256);n=e.n
calls=0;cache={};history=[]
def calc(a):
 global calls
 a=np.clip(np.asarray(a,dtype=float),0,1);key=a.tobytes()
 if key in cache:return cache[key]
 t=time.monotonic();P=e.game.transition(np.broadcast_to(a[:,None],(n,n)),np.broadcast_to(a[None,:],(n,n)))
 st=checked_joint_stationary(e.game,P,force_positive=True)
 ev=evaluate_policy(e.grid,e.game.D@st['mu_b'],a,e.params);calls+=1
 adv=ev['advantage'];regret=np.maximum(adv,0)-a*adv
 r=dict(advantage=adv,regret=regret,st=st,ev=ev);cache[key]=r
 print('CALL',calls,'secs',time.monotonic()-t,'regret',regret.max(),'violations',[(int(i),float(e.grid[i]),float(a[i]),float(adv[i])) for i in np.flatnonzero(regret>1e-9)],flush=True)
 return r

def certify(a,label):
 r=e.evaluate(a)
 print('CERT',label,r['verified'],r['dp_diagnostics'],flush=True)
 if r['verified']:
  save_bundle(Path('/tmp/rho097_positive_face_n101'),dict(competition=r,source_fingerprints=numerical_sources(),method='Positive-stationary exploratory active-set complementarity; final unchanged full certificate',search_history=history))
  np.save('/tmp/rho097_positive_face_n101.npy',a)
  raise SystemExit(0)
 return r


baseline=np.zeros(n);baseline[48:69]=1
def rule(w):
 a=baseline.copy();a[47]=w;return a
def f(w):return calc(rule(w))['advantage'][47]
w=brentq(f,0,1,xtol=1e-13,rtol=1e-13);a=rule(w);print('BRENTMIX',w,flush=True);certify(a,'one-dimensional positive face lower47')
