from hashlib import sha256
import json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code import sweep201_run as worker
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment
from oblivious.oe_distribution_fixed_p0.code import sweep201_quick_prescreen as quick
from oblivious.oe_distribution_fixed_p0.code.sweep201_active_face_proposal import score
args=SimpleNamespace(p0=.3,rho=.9,order=256,tol=1e-7,dp_tol=1e-10)
seed,inputs=worker.checked_same_p0_seed(args,201,"marginal")
print("Constructing fresh n201/.3/.9 marginal game",flush=True)
e=FixedOutsideMarginalExperiment(n=201,params=Parameters(rho=.9),p0=.3,order=256)
engine=quick.QuickMarginalResponse(e)
def no_fallback(alpha,reason):raise RuntimeError("No full response/GTH allowed in diagnostic: "+reason)
engine.fallback=no_fallback
alpha=np.interp(e.grid,seed["grid"],seed["policy"])
history=[];best=None;started=time.monotonic()
def evaluate(a,label):
 global best
 if len(history)>=6:raise RuntimeError("Six-call cap")
 data=engine.calculate(a); metrics=score(a,data); objective=max(metrics.values())
 regret=np.maximum(data["advantage"],0)-a*data["advantage"]
 violations=np.flatnonzero(regret>1e-10)
 record=dict(label=label,call=len(history)+1,**metrics,violations=[dict(i=int(i),x=float(e.grid[i]),alpha=float(a[i]),advantage=float(data["advantage"][i]),regret=float(regret[i])) for i in violations],mixed=[int(i) for i in np.flatnonzero((a>0)&(a<1))],elapsed_seconds=time.monotonic()-started)
 history.append(record);print(json.dumps(record),flush=True)
 if best is None or objective<best[0]:best=(objective,a.copy(),data,metrics)
 return data,metrics
first,_=evaluate(alpha,"historical_interpolation")
pure=(first["advantage"]>=0).astype(float)
data,metrics=evaluate(pure,"full_grid_pure_best_response_proposal")
a=pure.copy();bracket=None
while len(history)<6 and max(metrics.values())>1e-10:
 regret=np.maximum(data["advantage"],0)-a*data["advantage"]
 k=int(np.argmax(regret)); old=a.copy();oldadv=float(data["advantage"][k]);new=a.copy()
 if bracket is not None and bracket[0]==k:
  _,lo,hi,alo,ahi=bracket
  new[k]=float(np.clip(lo-alo*(hi-lo)/(ahi-alo),0,1)) if ahi!=alo else .5*(lo+hi)
 else:new[k]=1. if oldadv>0 else 0.
 data,metrics=evaluate(new,"one_node_secant_or_flip_"+str(k))
 newadv=float(data["advantage"][k])
 if oldadv*newadv<0:
  bracket=(k,float(old[k]),float(new[k]),oldadv,newadv)
 elif bracket is not None and bracket[0]==k:
  _,lo,hi,alo,ahi=bracket
  if newadv*alo>0:bracket=(k,float(new[k]),hi,newadv,ahi)
  else:bracket=(k,lo,float(new[k]),alo,newadv)
 a=new
objective,a,data,metrics=best
prefix=Path("/tmp/sweep201-active-face-proposals/p03_single_face_recovery_n201")
paths=[Path(__file__).resolve(),Path(quick.__file__).resolve(),Path(worker.__file__).resolve()]
bundle=dict(configuration=e.configuration(),competition=dict(grid=e.grid.copy(),policy=a,verified=False,candidate_verified=False),quick_metrics=metrics,history=history,reference_input_sha256=inputs,source_fingerprints=worker.runner.source_fingerprints(),proposal_script_sha256={str(p):sha256(p.read_bytes()).hexdigest() for p in paths},explanation="UNCERTIFIED SEARCH PROPOSAL ONLY. No full certificate or values computed; original strict target evaluate and actual values mandatory.")
worker.runner.save_bundle(prefix,bundle)
print(json.dumps(dict(status="uncertified",prefix=str(prefix),calls=len(history),**metrics,full_certificate_performed=False)),flush=True)
