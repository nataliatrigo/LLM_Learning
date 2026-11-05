import json, time, sys
from pathlib import Path
import numpy as np
from scipy.special import betaincc
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
OUT=Path(__file__).resolve().parents[3] / 'forgetting/outputs/investment_connectedness'

def solve(par,n=2001,tol=1e-10):
 rho,p0,p1,p2,c1,c2,g=par
 x=np.linspace(0,1,n); h=1-rho
 w=betaincc(1+x/h,1+(1-x)/h,p0);w=w/(1-g+g*w)
 t1=rho*x*(n-1); t2=(rho*x+h)*(n-1)
 i1=np.minimum(t1.astype(int),n-2);i2=np.minimum(t2.astype(int),n-2)
 z1=t1-i1;z2=t2-i2
 rows=np.repeat(np.arange(n),5)
 cols=np.column_stack([np.arange(n),i1,i1+1,i2,i2+1]).ravel()
 policy=np.zeros(n,dtype=bool);v=np.zeros(n)
 for it in range(100):
  p=np.where(policy,p2,p1); r=np.where(policy,1-c2,1-c1)
  data=np.column_stack([np.ones(n),-g*w*(1-p)*(1-z1),-g*w*(1-p)*z1,-g*w*p*(1-z2),-g*w*p*z2]).ravel()
  v=spsolve(coo_matrix((data,(rows,cols)),shape=(n,n)).tocsc(),w*r)
  vf=v[i1]*(1-z1)+v[i1+1]*z1;vs=v[i2]*(1-z2)+v[i2+1]*z2
  a=g*(p2-p1)*(vs-vf)-(c2-c1)
  pol2=a>0
  if np.array_equal(policy,pol2):break
  policy=pol2
 return x,v,a,it

def valleys(a,x):
 # best +-+ margin with center negative, positives in each side
 left=np.maximum.accumulate(a);right=np.maximum.accumulate(a[::-1])[::-1]
 margin=np.minimum(np.minimum(left,right),-a)
 j=int(np.argmax(margin))
 i=int(np.argmax(a[:j+1]));k=j+int(np.argmax(a[j:]))
 return float(margin[j]),[float(x[i]),float(x[j]),float(x[k])],[float(a[i]),float(a[j]),float(a[k])]

def run(seed,total):
 rng=np.random.default_rng(seed);start=time.time();best=-1e9
 for ix in range(total):
  rho=rng.uniform(.50001,.96) if ix%4 else 1-10**rng.uniform(-2.6,-.35)
  p0=rng.uniform(.025,.975)
  p1,p2=np.sort(rng.uniform(.01,.99,2))
  if ix%3==0:p1=rng.uniform(.01,.25);p2=rng.uniform(.75,.99)
  c1=.05
  frac=rng.uniform(.01,.99)
  c2=c1+(1-c1)*frac
  g=1-10**rng.uniform(-2.8,-.35)
  par=[rho,p0,p1,p2,c1,c2,g]
  x,v,a,it=solve(par,n=1501)
  margin,points,vals=valleys(a,x)
  if margin>best:
   best=margin; result=dict(par=par,margin=margin,points=points,advantages=vals,index=ix,n=1501,it=it)
   (OUT/f'search_agent_best_{seed}.json').write_text(json.dumps(result,indent=2))
   if margin>0:print('CANDIDATE',json.dumps(result),flush=True)
  if ix%100==0:print('PROGRESS',seed,ix,'best',best,'elapsed',round(time.time()-start,1),flush=True)
  if margin>1e-4:
   x,v,a,it=solve(par,n=12001);m,p,va=valleys(a,x)
   refined=dict(par=par,margin=m,points=p,advantages=va,index=ix,n=12001,it=it)
   (OUT/f'search_agent_candidate_{seed}_{ix}.json').write_text(json.dumps(refined,indent=2))
   print('REFINED',json.dumps(refined),flush=True)
   if m>1e-4:return
if __name__=='__main__':run(int(sys.argv[1]) if len(sys.argv)>1 else 725,int(sys.argv[2]) if len(sys.argv)>2 else 3000)
