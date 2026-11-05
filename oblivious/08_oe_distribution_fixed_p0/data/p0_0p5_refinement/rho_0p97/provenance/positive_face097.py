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

seed=np.interp(e.grid,old['competition']['grid'],old['competition']['policy']);a=(seed>.5).astype(float)
seen={};rules=[]
for iteration in range(12):
 r=calc(a);adv=r['advantage'];regret=r['regret'];history.append(dict(stage='positive pure best-response update',iteration=iteration,max_regret=float(regret.max()),alpha=a.tolist()))
 if regret.max()<1e-10:certify(a,'pure')
 new=(adv>=0).astype(float);rules.append(a.copy());key=new.tobytes()
 if key in seen:
  start=seen[key];recent=rules[start:]+[new];active=np.flatnonzero(np.ptp(np.array(recent),axis=0)>0)
  print('CYCLE ACTIVE',active.tolist(),flush=True);baseline=new.copy();a=np.mean(recent,axis=0)
  for attempt in range(8):
   if not len(active):break
   def expand(w):
    x=baseline.copy();x[active]=np.clip(w,0,1);return x
   def fun(w):return calc(expand(w))['advantage'][active]
   solved=root(fun,a[active],method='hybr',options={'xtol':1e-10,'eps':1e-9,'maxfev':60})
   aa=expand(solved.x);data=calc(aa);print('FACE',active.tolist(),'raw',solved.x.tolist(),'adv',data['advantage'][active].tolist(),'success',solved.success,flush=True)
   if data['regret'].max()<1e-9:certify(aa,'positive face')
   # If a root lies outside [0,1], freeze that action and expand with all current pure violations.
   mixed=active[(solved.x>0)&(solved.x<1)]
   violations=np.flatnonzero(data['regret']>1e-8)
   active=np.unique(np.r_[mixed,violations]);baseline=aa.copy();a=aa.copy();
   print('NEXTACTIVE',active.tolist(),flush=True)
  break
 seen[a.tobytes()]=iteration;a=new
print('UNRESOLVED',flush=True)
# A near-pure candidate observed above has only the top boundary violating;
# free its nearest lower pure boundary as a second candidate variable.
baseline=np.zeros(n);baseline[48:69]=1;active=np.array([47,69]);a=baseline.copy();a[active]=.5
for attempt in range(10):
 def expand(w):
  x=baseline.copy();x[active]=np.clip(w,0,1);return x
 def fun(w):return calc(expand(w))['advantage'][active]
 solved=root(fun,a[active],method='hybr',options={'xtol':1e-10,'eps':1e-8,'maxfev':60})
 aa=expand(solved.x);data=calc(aa);print('NEARFACE',active.tolist(),'raw',solved.x.tolist(),'adv',data['advantage'][active].tolist(),'success',solved.success,flush=True)
 if data['regret'].max()<1e-9:certify(aa,'positive near face')
 mixed=active[(solved.x>0)&(solved.x<1)];violations=np.flatnonzero(data['regret']>1e-8)
 active=np.unique(np.r_[mixed,violations]);baseline=aa.copy();a=aa.copy();print('NEXTACTIVE',active.tolist(),flush=True)
 if not len(active):break
