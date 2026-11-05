"""Independent characteristic integrations and a monotone-demand counterexample."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq
from scipy.special import betaln, digamma
from solve_fluid import P1,P2,C1,C2,R,DELTA,demand_at,solve,OUT

K=(C2-C1)/(P2-P1)

def beta_prime(x,rho,p0):
    m=1/(1-rho)
    alpha,beta=1+m*x,1+m*(1-x)
    center=digamma(alpha)-digamma(beta)
    def f(z):
        return math.exp((alpha-1)*math.log(z)+(beta-1)*math.log1p(-z)-betaln(alpha,beta))*(math.log(z)-math.log1p(-z)-center)
    if x <= p0:
        ans=quad(f,p0,1,epsabs=1e-12,epsrel=1e-10,limit=200)[0]
    else:
        ans=-quad(f,0,p0,epsabs=1e-12,epsrel=1e-10,limit=200)[0]
    return m*ans

def characterize(rho,p0=.5):
    lam=-math.log(rho)
    d=demand_at(rho,p0)
    def a(x): return R-C1+K*(P1-x)
    def g(x): return a(x)*beta_prime(x,rho,p0)-K*(d(x)+DELTA/lam)
    grid=np.linspace(0,1,301)
    gs=np.array([g(x) for x in grid])
    roots=[brentq(g,grid[i],grid[i+1],xtol=1e-12) for i in np.flatnonzero(gs[:-1]*gs[1:]<0)]
    peaks=[x for x in roots if g(max(0,x-1e-5))>0 and g(min(1,x+1e-5))<0]
    ans=dict(rho=rho,p0=p0,stationary_candidates=roots)
    if not peaks or not P1 < peaks[-1] < P2:
        return ans
    high=peaks[-1]
    vhigh=d(high)*a(high)/DELTA
    def rhs2(x,v): return [(DELTA*v[0]-d(x)*(R-C2))/(lam*d(x)*(P2-x))]
    to_p1=solve_ivp(rhs2,[high,P1],[vhigh],method='Radau',rtol=2e-10,atol=2e-11,dense_output=True)
    assert to_p1.success,to_p1.message
    u0=d(P1)*(R-C1)/DELTA
    du0=(R-C1)*beta_prime(P1,rho,p0)/(DELTA+lam*d(P1))
    def rhs1(x,v): return [(DELTA*v[0]-d(x)*(R-C1))/(lam*d(x)*(P1-x))]
    eps=1e-7
    u1=solve_ivp(rhs1,[P1+eps,high],[u0+du0*eps],method='Radau',rtol=2e-10,atol=2e-11,dense_output=True)
    assert u1.success,u1.message
    if to_p1.y[0,-1]<u0:
        lower=brentq(lambda x: to_p1.sol(x)[0]-u1.sol(x)[0],P1+eps,high,xtol=1e-11)
        slopes=[rhs1(lower,u1.sol(lower))[0],rhs2(lower,to_p1.sol(lower))[0]]
        ans.update(lower=lower,lower_type='Skiba value matching',upper=high,threshold_derivative=K/lam,one_sided_Vprime=slopes)
    else:
        def event(x,v): return DELTA*v[0]-d(x)*a(x)
        event.terminal=True
        event.direction=0
        left=solve_ivp(rhs2,[P1,0],[to_p1.y[0,-1]],method='Radau',rtol=2e-10,atol=2e-11,events=event)
        assert left.success,left.message
        ans.update(lower=float(left.t_events[0][0]) if len(left.t_events[0]) else None,lower_type='smooth tangency' if len(left.t_events[0]) else 'none in [0,1]',upper=high)
    return ans

def main():
    rows=[characterize(rho) for rho in (.5,.7,.8,.9,.95,.99,.999)]
    (OUT/'characteristic_checks.json').write_text(json.dumps(rows,indent=2))
    for row in rows: print(row,flush=True)
    def demand(x): return (.01+.0001*np.cos(20*np.pi*x)+.01*K*x)/(R-C1+K*(P1-x))
    x=np.linspace(0,1,20001)
    sol=solve(demand,1.,n=4001,delta=.01)
    finer=solve(demand,1.,n=8001,delta=.01,initial=sol)
    result=dict(formula='(.01 + .0001*cos(20*pi*x) + .01*(4/3)*x) / (.95 + (4/3)*(.35-x))',lambda_value=1,delta=.01,min_demand=float(demand(x).min()),max_demand=float(demand(x).max()),min_sampled_slope=float(np.min(np.diff(demand(x))/np.diff(x))),thresholds_4001=sol['roots'],thresholds_8001=finer['roots'],residual=finer['residual'],note='Strict monotonicity and multiple optimal steady targets are also verified analytically in the research note.')
    (OUT/'counterexample.json').write_text(json.dumps(result,indent=2))
    print('COUNTEREXAMPLE',result,flush=True)

if __name__=='__main__': main()
