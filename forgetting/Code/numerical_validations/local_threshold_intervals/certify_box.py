"""Uniform value/slope interval certificate for exact Bellman policy geometry.

Demand and demand derivatives are enclosed separately using Arb. All Bellman
arithmetic is rounded outward. Values use monotonicity and floor/ceil bounds;
slopes use interval recurrences and strict action exclusions, never derivatives
of an interpolated numerical solution.
"""
from __future__ import annotations
import argparse,json,time
from fractions import Fraction
from pathlib import Path
import numpy as np
from flint import arb,ctx
from demand_bounds import demand_bounds

OUT=Path(__file__).resolve().parents[4] / 'forgetting/outputs/numerical_validations/local_threshold_intervals'

def dn(x):return np.nextafter(x,-np.inf)
def up(x):return np.nextafter(x,np.inf)
def ab(x):return float(dn(float(x.lower()))),float(up(float(x.upper())))
def iv(a,b=None):
 a=arb(str(a));b=a if b is None else arb(str(b));return a.union(b)
def add(a,b):return dn(a[0]+b[0]),up(a[1]+b[1])
def sub(a,b):return dn(a[0]-b[1]),up(a[1]-b[0])
def mul(a,b):
 pairs=np.broadcast_arrays(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1])
 return dn(np.minimum.reduce(pairs)),up(np.maximum.reduce(pairs))
def divpos(a,b):return dn(a[0]/b[1]),up(a[1]/b[0])
def exact(x):return x,x

def index_bounds(N,rho):
 rl,rh=map(Fraction,rho)
 assert max(N*rl.numerator,N*rl.denominator,N*rh.numerator,N*rh.denominator)<2**63
 j=np.arange(N+1,dtype=np.int64)
 fl=(j*rl.numerator)//rl.denominator
 fu=(j*rh.numerator+rh.denominator-1)//rh.denominator
 # Success map decreases in rho.
 sl=(j*rh.numerator+(rh.denominator-rh.numerator)*N)//rh.denominator
 su=(j*rl.numerator+(rl.denominator-rl.numerator)*N+rl.denominator-1)//rl.denominator
 return fl,fu,sl,su

def extrema(a,left,right,minimum):
 result=a[left].copy()
 op=np.minimum if minimum else np.maximum
 for offset in range(1,int(np.max(right-left))+1):
  result=op(result,a[np.minimum(left+offset,right)])
 return result

def expectation_node(lo,hi,p,fl,fu,sl,su):
 # V(success)>=V(failure): extrema in p occur at its endpoints.
 l=add(mul(exact(p[0]),exact(lo[sl])),mul(exact(dn(1-p[0])),exact(lo[fl])))[0]
 u=add(mul(exact(p[1]),exact(hi[su])),mul(exact(up(1-p[1])),exact(hi[fu])))[1]
 return l,u

def expectation_slope(ls,us,lf,uf,p):
 e=[]
 for pp in p:
  e.append(add(mul(exact(pp),(ls,us)),mul((dn(1-pp),up(1-pp)),(lf,uf))))
 return np.minimum(e[0][0],e[1][0]),np.maximum(e[0][1],e[1][1])

def certify(box,N=2048,iterations=3000,deriv_iterations=250,use_residual_seed=False):
 start=time.time();ctx.prec=100
 assert all(isinstance(v,str) for pair in box.values() for v in pair)
 rho=ab(iv(*box['rho']));p1=ab(iv(*box['p1']));p2=ab(iv(*box['p2']));gam=ab(iv(*box['gamma']));cost=ab(iv(*box['cost_ratio']))
 assert 0<rho[0]<rho[1]<1 and 0<p1[0]<p1[1]<p2[0]<p2[1]<1 and 0<gam[0]<gam[1]<1
 assert 0<cost[0]<cost[1]<1
 demand=demand_bounds(N,box['rho'],box['p0'])
 D=demand['nodes'];Dc=demand['cells'];Dp=demand['derivative']
 def waiting(d):
  # Monotone in demand and gamma; use exact endpoint combinations.
  dl=dn(d[0]/up(up(1-gam[0])+up(gam[0]*d[0])))
  du=up(d[1]/dn(dn(1-gam[1])+dn(gam[1]*d[1])))
  return np.maximum(0,dl),np.minimum(1,du)
 w=waiting(D);wc=waiting(Dc)
 # w'(x)=(1-gamma) D'/(1-gamma+gamma D)^2. Natural interval bound.
 delta=(dn(1-gam[1]),up(1-gam[0]))
 denom=add(delta,mul(gam,Dc));wprime=divpos(mul(delta,Dp),mul(denom,denom))
 fl,fu,sl,su=index_bounds(N,box['rho'])
 lo=np.zeros(N+1);hi=np.full(N+1,up(1/dn(1-gam[1])))
 seedmeta=None
 if use_residual_seed:
  from residual_seed import residual_seed
  slo,shi,seedmeta=residual_seed(box,N,demand,w,wc,wprime)
  lo=np.maximum(lo,slo);hi=np.minimum(hi,shi)
 r1=exact(1.);r2=(dn(1-cost[1]),up(1-cost[0]))
 for it in range(iterations):
  q1=add(r1,mul(gam,expectation_node(lo,hi,p1,fl,fu,sl,su)))
  q2=add(r2,mul(gam,expectation_node(lo,hi,p2,fl,fu,sl,su)))
  new=mul(w,(np.maximum(q1[0],q2[0]),np.maximum(q1[1],q2[1])))
  nl=np.maximum(lo,new[0]);nh=np.minimum(hi,new[1])
  change=max(float(np.max(nl-lo)),float(np.max(hi-nh)))
  lo=np.maximum.accumulate(nl);hi=np.minimum.accumulate(nh[::-1])[::-1]
  assert np.all(lo<=hi)
  if change<1e-10:break
 # Cell successor ranges and Q bounds.
 cflo=fl[:-1];cfhi=fu[1:];cslo=sl[:-1];cshi=su[1:]
 q=[]
 for p,r in [(p1,r1),(p2,r2)]:
  q.append(add(r,mul(gam,expectation_node(lo,hi,p,cflo,cfhi,cslo,cshi))))
 gap=(np.maximum(0,dn(lo[cslo]-hi[cfhi])),up(hi[cshi]-lo[cflo]))
 dp=(dn(p2[0]-p1[1]),up(p2[1]-p1[0]))
 A=sub(mul(mul(gam,dp),gap),cost)
 possible1=A[0]<=0;possible2=A[1]>=0
 assert np.all(possible1|possible2)
 # Cell index covers for derivative evaluation. Including both sides of a
 # mapped grid knot is conservative and handles nonsmooth points safely.
 fleft=np.maximum(0,cflo-1);fright=np.minimum(N-1,cfhi)
 sleft=np.maximum(0,cslo-1);sright=np.minimum(N-1,cshi)
 coeff=mul(mul(gam,rho),wc)
 M=up(1/dn(1-gam[1]));W1=float(np.max(wprime[1]))
 L=up(up(M*W1)/dn(1-up(gam[1]*rho[1])))
 ld=np.zeros(N);ud=np.full(N,L)
 for dit in range(deriv_iterations):
  lf=extrema(ld,fleft,fright,True);uf=extrema(ud,fleft,fright,False)
  ls=extrema(ld,sleft,sright,True);us=extrema(ud,sleft,sright,False)
  candidates=[]
  for p,qi in [(p1,q[0]),(p2,q[1])]:
   candidates.append(add(mul(wprime,qi),mul(coeff,expectation_slope(ls,us,lf,uf,p))))
  newl=np.minimum(np.where(possible1,candidates[0][0],np.inf),np.where(possible2,candidates[1][0],np.inf))
  newu=np.maximum(np.where(possible1,candidates[0][1],-np.inf),np.where(possible2,candidates[1][1],-np.inf))
  nl=np.maximum(ld,newl);nu=np.minimum(ud,newu)
  change=max(float(np.max(nl-ld)),float(np.max(ud-nu)))
  ld,ud=nl,nu
  assert np.all(ld<=ud)
  if change<1e-9:break
 lf=extrema(ld,fleft,fright,True);uf=extrema(ud,fleft,fright,False)
 ls=extrema(ld,sleft,sright,True);us=extrema(ud,sleft,sright,False)
 Ap=mul(mul(mul(gam,dp),rho),sub((ls,us),(lf,uf)))
 # One decreasing crossing, with investment extending to x=0.
 result=dict(box=box,N=N,value_iterations=it+1,derivative_iterations=dit+1,
  max_value_width=float(np.max(hi-lo)),max_derivative_width=float(np.max(ud-ld)),
  seconds=time.time()-start,certified=False,
  advantage_left=[float(A[0][0]),float(A[1][0])],
  advantage_right=[float(A[0][-1]),float(A[1][-1])],
  max_advantage_lower=float(np.max(A[0])))
 if A[0][0]>0 and A[1][-1]<0:
  first=int(np.flatnonzero(A[0]<=0)[0]);last=int(np.flatnonzero(A[1]>=0)[-1])
  assert first<=last
  upper=float(np.max(Ap[1][first:last+1]))
  result.update(type='one_threshold',bracket=[first/N,(last+1)/N],
    advantage_at_zero_lower=float(A[0][0]),advantage_at_one_upper=float(A[1][-1]),
    derivative_upper=upper,certified=bool(upper<0))
 # Two crossings: positive core and negative regions on both sides.
 elif A[1][0]<0 and A[1][-1]<0 and np.any(A[0]>0):
  mask=A[0]>0
  edges=np.diff(np.r_[False,mask,False].astype(int));starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)-1
  longest=int(np.argmax(ends-starts));pa=int(starts[longest]);pb=int(ends[longest])
  left=int(np.flatnonzero(A[1]>=0)[0]);right=int(np.flatnonzero(A[1]>=0)[-1])
  lower=float(np.min(Ap[0][left:pa+1]));upper=float(np.max(Ap[1][pb:right+1]))
  core=bool(np.all(A[0][pa:pb+1]>0))
  result.update(type='two_thresholds',brackets=[[left/N,(pa+1)/N],[pb/N,(right+1)/N]],
    left_derivative_lower=lower,right_derivative_upper=upper,positive_core=core,
    advantage_at_zero_upper=float(A[1][0]),advantage_at_one_upper=float(A[1][-1]),
    certified=bool(core and lower>0 and upper<0))
 result['seconds']=time.time()-start
 if seedmeta is not None:result['residual_seed']=seedmeta
 return result

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--box',required=True);parser.add_argument('--nodes',type=int,default=2048);parser.add_argument('--out');parser.add_argument('--residual-seed',action='store_true')
 args=parser.parse_args();box=json.loads(Path(args.box).read_text());result=certify(box,args.nodes,use_residual_seed=args.residual_seed)
 print(json.dumps(result,indent=2))
 if args.out:Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
