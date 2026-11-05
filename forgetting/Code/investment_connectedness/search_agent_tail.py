from search_agent import solve,valleys,OUT
import numpy as np,json,time
rng=np.random.default_rng(964725);start=time.time();best=-np.inf
for ix in range(4000):
 rho=rng.uniform(.5001,.85)
 p0=1-10**rng.uniform(-4,-.7)
 p1=10**rng.uniform(-3,-.4)
 p2=1-10**rng.uniform(-3,-.4)
 if p1>=p2:continue
 gamma=1-10**rng.uniform(-6,-2.2)
 frac=1-10**rng.uniform(-4,-.1)
 par=[rho,p0,p1,p2,.05,.05+.95*frac,gamma]
 x,v,a,it=solve(par,n=2001)
 m,p,va=valleys(a,x)
 if m>best:
  best=m; result=dict(par=par,margin=m,points=p,advantages=va,index=ix,n=2001,it=it)
  (OUT/'search_agent_tail_best.json').write_text(json.dumps(result,indent=2))
 if m>1e-7:
  print('CANDIDATE',json.dumps(result),flush=True)
  x,v,a,it=solve(par,n=16001);m,p,va=valleys(a,x)
  result=dict(par=par,margin=m,points=p,advantages=va,index=ix,n=16001,it=it)
  (OUT/f'search_agent_tail_candidate_{ix}.json').write_text(json.dumps(result,indent=2))
  print('REFINED',json.dumps(result),flush=True)
  if m>1e-6:break
 if ix%100==0:print('PROGRESS',ix,best,round(time.time()-start,1),flush=True)
