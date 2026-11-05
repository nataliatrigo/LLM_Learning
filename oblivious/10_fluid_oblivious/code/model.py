"""Symmetric fluid oblivious equilibria against a fixed rival reputation.

Demand is the original independent Thompson/Beta outside-option demand.  A
mixed stationary candidate must be an optimal *holding* mixture: a crossing
of upwind Hamiltonian differences alone does not establish this at a kink.
Searches and all error indicators below are numerical, not existence proofs.
"""
from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache
import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import betainc, betaincc, betaln, digamma, logsumexp, roots_jacobi, roots_legendre

from forgetting.Code.model import Primitives
from forgetting.Code.fluid_convergence.numerics import solve_fluid
from oblivious.oe_p0.code.primitives import Parameters, posterior
from oblivious.oe_distribution_fixed_p0.code.model import outside_demand_probability


@lru_cache(maxsize=16)
def _legendre(order):
    return roots_legendre(int(order))


def _row_quadrature(x, params, p0, order):
    """The same positive truncated quadrature as the stochastic source."""
    a, b = posterior(float(x), params)
    if params.rho >= .99:
        nodes, weights = _legendre(order)
        t = p0+(1-p0)*(nodes+1)/2
        logweight = np.log(weights*(1-p0)/2)
        logcoef = logweight+(a-1)*np.log(t)+(b-1)*np.log1p(-t)-betaln(a,b)
    else:
        nodes, weights = roots_jacobi(int(order), b-1, 0)
        t = p0+(1-p0)*(nodes+1)/2
        logcoef = (np.log(weights)+b*(np.log1p(-p0)-np.log(2))
                   -betaln(a,b)+(a-1)*np.log(t))
    return t, logcoef


def demand_at_states(x, y, params, p0, order=256):
    """Actual joint demands (A, B, outside), with cached positive quadrature.

    The smaller seller tail is integrated directly and the larger demand uses
    the exact event partition, as in outside_demand_matrix. This inexpensive
    Legendre variant is intended for trajectories and must be order-checked.
    The HJB uses FixedRivalDemand's original Jacobi/Legendre row quadrature.
    """
    if not 0 <= p0 < 1 or order < 8:
        raise ValueError("require 0 <= p0 < 1 and order >= 8")
    x,y = np.broadcast_arrays(np.asarray(x,dtype=float),np.asarray(y,dtype=float))
    a,b=posterior(x,params); c,d=posterior(y,params)
    fx,fy=betainc(a,b,p0),betainc(c,d,p0)
    sx,sy=betaincc(a,b,p0),betaincc(c,d,p0)
    market=(sx+fx*sy+sy+fy*sx)/2
    nodes,weights=_legendre(order)
    t=p0+(1-p0)*(nodes+1)/2
    smaller=x<=y
    own_a,own_b=np.where(smaller,a,c),np.where(smaller,b,d)
    rival_a,rival_b=np.where(smaller,c,a),np.where(smaller,d,b)
    logpdf=((own_a[...,None]-1)*np.log(t)
            +(own_b[...,None]-1)*np.log1p(-t)-betaln(own_a,own_b)[...,None])
    cdf=betainc(rival_a[...,None],rival_b[...,None],t)
    with np.errstate(divide="ignore"):
        tail=np.exp(logsumexp(logpdf+np.log(weights*(1-p0)/2)+np.log(cdf),axis=-1))
    tail=np.minimum(tail,market)
    da=np.where(smaller,tail,market-tail)
    db=np.where(smaller,market-tail,tail)
    equal=x==y
    da=np.where(equal,market/2,da); db=np.where(equal,market/2,db)
    if da.ndim==0:
        return float(da),float(db),float(fx*fy)
    return da,db,fx*fy


class FixedRivalDemand:
    """Prepared own-state grid: fast demand for arbitrary fixed conjectures."""

    def __init__(self,params,p0,grid,order=128):
        if not 0 <= p0 < 1 or int(order)!=order or order<8:
            raise ValueError("require 0 <= p0 < 1 and integer order >= 8")
        self.params,self.p0,self.grid,self.order=params,float(p0),np.asarray(grid),int(order)
        rows=[_row_quadrature(x,params,p0,order) for x in self.grid]
        self.nodes=np.asarray([row[0] for row in rows])
        self.logcoef=np.asarray([row[1] for row in rows])
        self.a,self.b=posterior(self.grid,params)
        self.cdf=betainc(self.a,self.b,p0)
        self.survival=betaincc(self.a,self.b,p0)

    def __call__(self,conjecture):
        c,d=posterior(float(conjecture),self.params)
        with np.errstate(divide="ignore"):
            raw=np.exp(logsumexp(self.logcoef+np.log(betainc(c,d,self.nodes)),axis=1))
        cf,sf=betainc(c,d,self.p0),betaincc(c,d,self.p0)
        market=(self.survival+self.cdf*sf+sf+cf*self.survival)/2
        larger=self.grid>conjecture
        if np.any(larger):
            nodes,coef=_row_quadrature(conjecture,self.params,self.p0,self.order)
            with np.errstate(divide="ignore"):
                tail=np.exp(logsumexp(coef[None,:]+np.log(betainc(
                    self.a[larger,None],self.b[larger,None],nodes)),axis=1))
            raw[larger]=market[larger]-tail
        equal=self.grid==conjecture
        raw[equal]=market[equal]/2
        if np.any(raw < -1e-12) or np.any(raw > self.survival+1e-10):
            raise ArithmeticError("fixed-rival demand violated probability bounds")
        return np.maximum(raw,0)


def symmetric_demand_derivative(params,p0,z,order=128):
    """D(z,z), partial D_A/partial x at (z,z), holding the rival fixed."""
    nodes,coef=_row_quadrature(z,params,p0,order)
    a,b=posterior(z,params)
    cdf=betainc(a,b,nodes)
    weight=np.exp(coef)*cdf
    score=(np.log(nodes)-np.log1p(-nodes)-digamma(a)+digamma(b))/(1-params.rho)
    survival=betaincc(a,b,p0)
    demand=survival*(1+betainc(a,b,p0))/2
    return float(demand),float(np.sum(weight*score))


def _primitives(params,p0):
    return Primitives(p0=max(np.nextafter(0.,1.),float(p0)),p1=params.p1,p2=params.p2,
                      c1=params.c1,c2=params.c2,revenue=params.R,gamma=params.gamma)


def _policy_description(params,response):
    roots=list(response["roots"])
    edges=[0.,*roots,1.]
    products=[2 if np.interp((a+b)/2,response["x"],response["gap"])>0 else 1
              for a,b in zip(edges,edges[1:])]
    kinds=[]; holding=[]
    for i,r in enumerate(roots):
        left=(params.p1 if products[i]==1 else params.p2)-r
        right=(params.p1 if products[i+1]==1 else params.p2)-r
        kind="attractive" if left>0 and right<0 else ("repelling" if left<0 and right>0 else "crossing")
        kinds.append(kind)
        if kind=="attractive":
            holding.append(dict(x=r,alpha=(r-params.p1)/(params.p2-params.p1)))
    return dict(roots=roots,products=products,root_kinds=kinds,
                intervals=[dict(lo=a,hi=b,product=p) for a,b,p in zip(edges,edges[1:],products)],
                holding_points=holding)


def solve_response(params,p0,conjecture,n=1001,order=128,*,prepared=None):
    grid=np.linspace(0,1,int(n))
    if prepared is None:
        prepared=FixedRivalDemand(params,p0,grid,order)
    demand=prepared(conjecture)
    response=solve_fluid(_primitives(params,p0),params.rho,int(n),demand_function=lambda x: demand)
    response.update(_policy_description(params,response))
    response["conjecture"]=float(conjecture)
    return response


def find_equilibria(params,p0,n=1001,order=128,scan_points=121):
    """Enumerate detected pure and mixed stationary symmetric candidates.

    Interior candidates solve the smooth holding first-order condition
        D_x r_bar - (dc/(lambda dp)) (delta + lambda D) = 0.
    This is only a candidate generator. Every candidate is then checked using
    the complete optimized HJB and its holding payoff. Attractive response-root
    consistency is refined with the complete HJB, and repelling candidates
    remain eligible for the payoff test instead of being accepted as mixtures.
    """
    if not 0 < params.gamma < 1:
        raise ValueError("fluid discount requires 0 < gamma < 1")
    dp,dc=params.p2-params.p1,params.c2-params.c1
    lam,delta=-math.log(params.rho),-math.log(params.gamma)
    k=dc/(lam*dp)
    grid=np.linspace(0,1,int(n)); h=1/(n-1)
    engine=FixedRivalDemand(params,p0,grid,order)
    cache={}
    def response(z):
        key=float(z)
        if key not in cache:
            cache[key]=solve_response(params,p0,key,n,order,prepared=engine)
        return cache[key]
    def foc(z):
        d,dx=symmetric_demand_derivative(params,p0,z,order)
        reward=params.R-params.c1-dc*(z-params.p1)/dp
        return dx*reward-k*(delta+lam*d)
    scan=np.linspace(params.p1,params.p2,int(scan_points))
    residual=np.asarray([foc(z) for z in scan])
    seeds=[]
    for a,b,fa,fb in zip(scan,scan[1:],residual,residual[1:]):
        if fa==0: seeds.append(float(a))
        if fa*fb<0: seeds.append(float(brentq(foc,a,b,xtol=2e-12)))
    # A repeated zero can have no sign change. Search sampled local minima of
    # |FOC| and keep only minima whose unscaled residual is close to zero.
    scale=max(1.,float(np.max(np.abs(residual))))
    for j in range(1,len(scan)-1):
        if abs(residual[j])<abs(residual[j-1]) and abs(residual[j])<abs(residual[j+1]):
            candidate=minimize_scalar(lambda z:abs(foc(z)),bounds=(scan[j-1],scan[j+1]),method="bounded",options={"xatol":1e-12})
            if abs(foc(candidate.x)) < 1e-9*scale:
                seeds.append(float(candidate.x))
    seeds=sorted(z for z in seeds if params.p1+1e-8<z<params.p2-1e-8)
    seeds=[z for j,z in enumerate(seeds) if j==0 or z-seeds[j-1]>1e-6]
    candidates=[(params.p1,"pure_product1"),(params.p2,"pure_product2")]+[(z,"mixed") for z in seeds]
    accepted=[];rejected=[]
    for seed,kind in candidates:
        z=float(seed)
        seed_response=response(z)
        attractive=[r for r,kd in zip(seed_response["roots"],seed_response["root_kinds"]) if kd=="attractive"]
        root_refined=False
        if kind=="mixed" and attractive:
            near=min(attractive,key=lambda r:abs(r-z))
            if abs(near-z)<max(.03,20*h):
                def consistency(v):
                    rr=response(v)
                    roots=[r for r,kd in zip(rr["roots"],rr["root_kinds"]) if kd=="attractive"]
                    if not roots: return float("nan")
                    return min(roots,key=lambda r:abs(r-near))-v
                radius=max(.006,8*h)
                lo,hi=max(params.p1+1e-9,z-radius),min(params.p2-1e-9,z+radius)
                fl,fh=consistency(lo),consistency(hi)
                if np.isfinite(fl) and np.isfinite(fh) and fl*fh<0:
                    z=float(brentq(consistency,lo,hi,xtol=2e-10))
                    root_refined=True
        rr=response(z)
        alpha=float((z-params.p1)/dp)
        demand,dx=symmetric_demand_derivative(params,p0,z,order)
        holding_value=demand*(params.R-params.c1-dc*alpha)/delta
        optimized_value=float(np.interp(z,rr["x"],rr["value"]))
        hold_gain=optimized_value-holding_value
        gap=float(np.interp(z,rr["x"],rr["gap"]))
        # Conditional Bellman advantage is checked separately from calendar
        # residual: tiny demand must not hide a poor holding decision.
        conditional_defect=delta*hold_gain/max(demand,np.finfo(float).tiny)
        value_tolerance=max(2e-6,50*h*h*(1+abs(k))*(1+abs(optimized_value)))
        conditional_tolerance=max(2e-5,10*h*h*(1+dc+lam*abs(k)))
        policy=_policy_description(params,rr)
        nearest=min((abs(r-z) for r in rr["roots"]),default=1.)
        if kind=="mixed":
            consistency_error=nearest
            action_valid=nearest<=5e-8 and abs(gap)<=max(2e-5,8*h*(1+abs(k))*lam)
        else:
            desired=1 if kind=="pure_product1" else 2
            action_valid=gap<=1e-8 if desired==1 else gap>=-1e-8
            consistency_error=0.
        # Explicit hold versus globally optimized departure comparison rejects
        # most repelling first-order stationary points. Spatial tolerances are
        # reported and must be assessed across the runner's grids.
        verified=bool(action_valid and abs(hold_gain)<=value_tolerance
                      and abs(conditional_defect)<=conditional_tolerance
                      and rr["residual"]<=1e-8)
        diag=dict(foc_seed=float(seed),foc_residual=float(foc(z)),
                  root_consistency_refined=root_refined,consistency_error=float(consistency_error),
                  holding_value=float(holding_value),optimized_value=optimized_value,
                  holding_value_gap=float(hold_gain),conditional_holding_defect=float(conditional_defect),
                  hamiltonian_gap=gap,holding_value_tolerance=value_tolerance,
                  conditional_tolerance=conditional_tolerance,hjb_residual=rr["residual"],
                  spatial_tolerance=4*h,action_valid=bool(action_valid))
        if kind=="mixed" and verified:
            # The scalar root is solved to tighter tolerance than plot precision;
            # set its representation identically to the conjecture for the ODE.
            idx=int(np.argmin(np.abs(np.asarray(policy["roots"])-z)))
            diag["policy_root_adjustment"]=float(z-policy["roots"][idx])
            policy["roots"][idx]=z
            edges=[0.,*policy["roots"],1.]
            policy["intervals"]=[dict(lo=a,hi=b,product=p) for a,b,p in zip(edges,edges[1:],policy["products"])]
            policy["holding_points"]=[dict(x=z if abs(p["x"]-z)<4*h else p["x"],
                                          alpha=alpha if abs(p["x"]-z)<4*h else p["alpha"])
                                      for p in policy["holding_points"]]
            if not any(abs(point["x"]-z)<1e-12 for point in policy["holding_points"]):
                policy["holding_points"].append(dict(x=z,alpha=alpha))
            diag["selected_root_kind"]=policy["root_kinds"][idx]
        record=dict(x_star=z,alpha=alpha,kind=kind,verified=verified,response=rr,
                    policy=policy,diagnostics=diag)
        if verified and not any(abs(z-old["x_star"])<max(1e-7,h/10) for old in accepted):
            accepted.append(record)
        elif not verified:
            rejected.append(dict(x_star=z,kind=kind,diagnostics=diag))
    search=dict(status="equilibrium_found" if accepted else "no_equilibrium_found",
                candidate_count=len(candidates),mixed_foc_candidate_count=len(seeds),
                rejected_candidates=rejected,scan_points=int(scan_points),
                foc_scan_states=scan,foc_scan_residuals=residual,
                hjb_solves=len(cache),grid_nodes=int(n),quadrature_order=int(order),
                scope="Numerically detected stationary equilibria; absence is not a nonexistence proof.")
    return dict(equilibria=sorted(accepted,key=lambda v:v["x_star"]),search=search)


def audit_response(params,p0,response,order=128):
    """Double quadrature on EVERY own-state node and audit selected entries.

    The adaptive audits reuse the stochastic model's independent scaled
    integration, including its tiny-tail treatment. Reported error indicators
    do not bound continuous-state discretization error.
    """
    x=np.asarray(response["x"]);z=float(response["conjecture"])
    base=np.asarray(response["demand"])
    doubled=FixedRivalDemand(params,p0,x,2*order)(z)
    difference=np.abs(base-doubled)
    relative=difference/np.maximum(doubled,np.finfo(float).tiny)
    samples=[]
    # Re-evaluate production demand at exact audit states, avoiding an
    # interpolation error being mistaken for quadrature disagreement.
    audit_states=np.unique([0.,params.p1,z,params.p2,1.])
    production=FixedRivalDemand(params,p0,audit_states,order)(z)
    for own,estimate in zip(audit_states,production):
        independent=outside_demand_probability(own,z,params,p0)
        error=abs(float(estimate)-independent)
        samples.append(dict(x=float(own),rival=z,production=float(estimate),
                            independent=float(independent),absolute_error=error,
                            relative_error=error/max(independent,np.finfo(float).tiny)))
    result=dict(order=int(order),doubled_order=int(2*order),own_state_nodes=len(x),
                order_doubling_max_abs=float(difference.max()),
                order_doubling_max_relative_tail=float(relative.max()),
                independent_max_abs=max(row["absolute_error"] for row in samples),
                independent_max_relative=max(row["relative_error"] for row in samples),
                independent_samples=samples)
    result["passed"]=bool(result["order_doubling_max_abs"]<2e-9
                          and result["order_doubling_max_relative_tail"]<2e-6
                          and result["independent_max_abs"]<2e-9
                          and result["independent_max_relative"]<2e-6)
    return result
