"""Cheap signed-law proposals followed by unchanged positive OE certification.

Only the search uses sparse signed arithmetic. Saved policies require the
original all-node e.evaluate certificate, including positive stationary checks.
"""
from pathlib import Path
import sys,time,json
import numpy as np
from scipy.optimize import brentq
from scipy.sparse import csr_matrix
sys.path.insert(0,'/home/natalia/Projects/LLM_Learning')
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment,_OutsideResponse
from oblivious.oe_distribution_fixed_p0.code.run_experiment import numerical_sources,save_bundle
from oblivious.oe_p0.code.run_experiment import load_bundle
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.policy import evaluate_policy
from oblivious.oe_p0.code.core import _jump_system,_sparse_invariant
n=201;previous=151;base=Path('/home/natalia/Projects/LLM_Learning/oblivious/08_oe_distribution_fixed_p0/data/p0_0p5_refinement/rho_0p97/arrays')
old=load_bundle(base/f'rho_0p97_n{previous:03d}')
e=FixedOutsideMarginalExperiment(n=n,params=Parameters(rho=.97),p0=.5,order=256)
i,j=np.triu_indices(n);representatives=i*n+j;groups=np.empty((n,n),dtype=int);groups[i,j]=np.arange(len(i));groups[j,i]=np.arange(len(i))
membership=csr_matrix((np.ones(n*n),(np.arange(n*n),groups.ravel())),shape=(n*n,len(i)))
calls=0;cache={};history=[];strong_history=[];derivative_engine=None

def calc(a):
 global calls
 a=np.clip(a,0,1);key=a.tobytes()
 if key in cache:return cache[key]
 started=time.monotonic();P=e.game.transition(np.broadcast_to(a[:,None],(n,n)),np.broadcast_to(a[None,:],(n,n)))
 reduced=(P[representatives]@membership).tocsr();jump,rates=_jump_system(reduced)
 try:calendar=_sparse_invariant(jump,rates,len(rates)//2)
 except ArithmeticError:calendar=_sparse_invariant(jump,rates,0)
 nu=np.zeros((n,n));shares=calendar/np.where(i==j,1,2);nu[i,j]=shares;nu[j,i]=shares
 ev=evaluate_policy(e.grid,e.game.D@nu.sum(axis=0),a,e.params)
 adv=ev['advantage'];regret=np.maximum(adv,0)-a*adv;calls+=1
 row=dict(call=calls,kind='exploratory sparse response only',seconds=time.monotonic()-started,
  regret=float(regret.max()),violations=[dict(i=int(k),x=float(e.grid[k]),alpha=float(a[k]),advantage=float(adv[k])) for k in np.flatnonzero(regret>1e-8)])
 history.append(row);print(json.dumps(row),flush=True)
 result=dict(advantage=adv,regret=regret);cache[key]=result;return result

def certify(a,label):
 global derivative_engine
 a=a.copy()
 for polish in range(4):
  started=time.monotonic();r=e.evaluate(a)
  row=dict(label=label,polish=polish,seconds=time.monotonic()-started,verified=r['verified'],dp=r['dp_diagnostics'])
  strong_history.append(row);print('FULL POSITIVE CERT',json.dumps(row),flush=True)
  if r['verified']:
   save_bundle(Path('/tmp/rho097_positive_face_n201'),dict(competition=r,source_fingerprints=numerical_sources(),
    method='Sparse jump-stationarity active-set proposals; full unchanged positive e.evaluate certificate; analytic-Jacobian positive-residual Newton polish if needed',
    search_history=history,positive_polish_history=strong_history))
   raise SystemExit(0)
  mixed=np.flatnonzero((a>0)&(a<1));endpoint=(a==0)|(a==1)
  endpoint_regret=np.maximum(r['advantage'],0)-a*r['advantage']
  if not len(mixed) or float(np.max(endpoint_regret[endpoint],initial=0))>1e-7:return r
  if derivative_engine is None:derivative_engine=_OutsideResponse(e)
  started=time.monotonic();jac=derivative_engine.calculate(a)['jacobian'][np.ix_(mixed,mixed)]
  correction=np.linalg.solve(jac,r['advantage'][mixed]);trial=a.copy();trial[mixed]-=correction
  print('ANALYTIC NEWTON',json.dumps(dict(indices=mixed.tolist(),correction=correction.tolist(),seconds=time.monotonic()-started)),flush=True)
  if np.any((trial<0)|(trial>1)) or not np.all(np.isfinite(trial)):return r
  a=trial
 return r

alpha=(np.interp(e.grid,old['competition']['grid'],old['competition']['policy'])>=.5).astype(float)
visited={}
for iteration in range(10):
 result=calc(alpha);bad=np.flatnonzero(result['regret']>1e-8)
 if len(bad)==0:
  checked=certify(alpha,'rounded/updated previous policy');alpha=checked['policy'].copy()
  bad=np.flatnonzero(np.maximum(checked['advantage'],0)-alpha*checked['advantage']>1e-10)
  if not len(bad):break
 print('BAD',bad.tolist(),flush=True)
 for k in bad:
  a0=alpha.copy();a0[k]=0;f0=calc(a0)['advantage'][k]
  a1=alpha.copy();a1[k]=1;f1=calc(a1)['advantage'][k]
  if f0*f1<0:
   baseline=alpha.copy()
   def expand(w):
    a=baseline.copy();a[k]=w;return a
   w=brentq(lambda w:calc(expand(w))['advantage'][k],0,1,xtol=1e-10,rtol=1e-10)
   alpha=expand(w);checked=certify(alpha,f'sparse mixed root at {k}');alpha=checked['policy'].copy()
  else:alpha[k]=float(f0>0)
 signature=alpha.tobytes()
 if signature in visited:break
 visited[signature]=iteration
raise ArithmeticError('No fully certified proposal in finite exploratory search')
