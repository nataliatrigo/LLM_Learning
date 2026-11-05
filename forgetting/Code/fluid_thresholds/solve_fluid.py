"""Monotone upwind policy iteration for the fluid HJB; numerical evidence only."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from scipy.linalg import solve_banded
from scipy.special import betaincc, expit

OUT = Path(__file__).resolve().parents[3] / 'forgetting/outputs/fluid_thresholds'
P1, P2, C1, C2, R = .35, .80, .05, .65, 1.
DELTA = -math.log(.98)

def solve(demand, lam, n=4001, delta=DELTA, initial=None):
    x = np.linspace(0., 1., n)
    h = x[1] - x[0]
    d = demand(x)
    b = lam * d[None, :] * (np.array([P1, P2])[:, None] - x)
    right = np.maximum(b, 0.) / h
    left = np.maximum(-b, 0.) / h
    right[:, -1] = 0.
    left[:, 0] = 0.
    rewards = (R - np.array([C1, C2]))[:, None] * d
    policy = np.zeros(n, dtype=int) if initial is None else (np.interp(x, initial["x"], initial["policy"]) > .5).astype(int)
    j = np.arange(n)
    for iteration in range(20000):
        rr, ll = right[policy, j], left[policy, j]
        ab = np.zeros((3, n))
        ab[0, 1:] = -rr[:-1]
        ab[1] = delta + rr + ll
        ab[2, :-1] = -ll[1:]
        v = solve_banded((1, 1), ab, rewards[policy, j])
        forward = np.r_[np.diff(v)/h, 0.]
        backward = np.r_[0., np.diff(v)/h]
        slopes = np.where(np.array([P1, P2])[:, None] >= x, forward, backward)
        normalized_h = (R - np.array([C1, C2]))[:, None] + lam*(np.array([P1, P2])[:, None]-x)*slopes
        gap = normalized_h[1] - normalized_h[0]
        updated = np.where(gap > 1.e-8, 1, np.where(gap < -1.e-8, 0, policy))
        if np.array_equal(updated, policy):
            break
        policy = updated
    else:
        raise RuntimeError('Policy iteration did not converge')
    residual = np.max(np.abs(delta*v - d*np.max(normalized_h, axis=0)))
    changes = np.flatnonzero(policy[1:] != policy[:-1])
    roots = [float(x[i] - gap[i]*h/(gap[i+1]-gap[i])) for i in changes]
    starts = np.r_[0, changes+1]
    ends = np.r_[changes, n-1]
    bands = [[float(x[a]),float(x[z])] for a,z in zip(starts,ends) if policy[a] == 1]
    return dict(x=x,demand=d,value=v,gap=gap,policy=policy,roots=roots,bands=bands,
                residual=float(residual),iterations=iteration+1)

def demand_at(rho,p0,kind='beta'):
    if kind == 'beta':
        return lambda x: betaincc(1+x/(1-rho),1+(1-x)/(1-rho),p0)
    a = 2*np.sqrt(8/np.pi)*np.sqrt(1+3*(1-rho))/((1+2*(1-rho))*np.sqrt(1-rho))
    return lambda x: expit(a*(x-p0))

def main():
    rows=[]
    for kind in ('beta','logistic'):
        for p0 in ((.1,.3,.5,.7,.9) if kind=='beta' else (.5,)):
            for rho in (.5,.7,.8,.9,.95,.99,.999):
                previous=None
                for n in (2001,4001,8001):
                    sol=solve(demand_at(rho,p0,kind),-math.log(rho),n,initial=previous)
                    stable = None if previous is None or len(sol['roots']) != len(previous['roots']) else max([abs(a-b) for a,b in zip(sol['roots'],previous['roots'])],default=0.)
                    rows.append(dict(demand=kind,p0=p0,rho=rho,n=n,threshold_count=len(sol['roots']),bands=json.dumps(sol['bands']),thresholds=json.dumps(sol['roots']),residual=sol['residual'],iterations=sol['iterations'],root_change=stable))
                    previous=sol
                np.savez_compressed(OUT/f'{kind}_p0_{p0:g}_rho_{rho:g}.npz',**{k:sol[k] for k in ('x','demand','value','gap','policy')})
                print(kind,p0,rho,sol['roots'],flush=True)
    pd.DataFrame(rows).to_csv(OUT/'grid_results.csv',index=False)
    (OUT/'parameters.json').write_text(json.dumps(dict(p1=P1,p2=P2,c1=C1,c2=C2,revenue=R,gamma=.98,delta=DELTA,lambda_rule='-log(rho)',method='upwind Markov-chain approximation, Howard policy iteration',note='D_rho is held fixed within each continuous-time control problem; no proof of a stochastic-to-fluid limit.'),indent=2))

if __name__=='__main__':
    main()
