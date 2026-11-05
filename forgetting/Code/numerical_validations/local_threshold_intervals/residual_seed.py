"""A posteriori continuous Bellman residual enclosure for a candidate interpolant.

The numerical solve only constructs exact binary64 nodal values. Its accuracy
is not assumed. All subsequent arithmetic is rounded outward and certifies the
same candidate against every Bellman operator in the supplied parameter box.
"""
from __future__ import annotations
from fractions import Fraction
from pathlib import Path
import sys
import time
import numpy as np


def residual_seed(box, N, demand, w, wc, wprime):
    """Return uniform exact-value enclosures at j/N, plus residual metadata."""
    from certify_box import (ab, iv, dn, up, add, sub, mul, exact,
                             index_bounds, extrema, expectation_slope)
    root = Path(__file__).resolve().parents[4]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from forgetting.Code.model import Primitives, BoundaryGrid, solve_boundary
    started = time.monotonic()
    center = {key: float(sum(map(Fraction, pair))/2) for key, pair in box.items()}
    params = Primitives(p0=center['p0'], p1=center['p1'], p2=center['p2'],
                        c1=0., c2=center['cost_ratio'], revenue=1.,
                        gamma=center['gamma'])
    grid = BoundaryGrid(np.arange(N+1)/N, N+1, 0, 9.)
    sol = solve_boundary(params, center['rho'], grid=grid, tolerance=1e-12)
    # Defining the candidate this way enforces exact monotonicity, irrespective
    # of tiny floating solver errors. The values themselves are exact floats.
    v = np.maximum.accumulate(np.maximum(0., sol.value))
    assert len(v) == N+1 and np.all(np.isfinite(v)) and np.all(np.diff(v) >= 0)

    rho=ab(iv(*box['rho'])); p1=ab(iv(*box['p1'])); p2=ab(iv(*box['p2']))
    gam=ab(iv(*box['gamma'])); cost=ab(iv(*box['cost_ratio']))
    r1=exact(1.); r2=(dn(1-cost[1]), up(1-cost[0]))
    rl,rh=map(Fraction,box['rho'])
    j=np.arange(N+1,dtype=np.int64)

    def interp(numerator, denominator):
        """Exact interpolant at N*z=numerator/denominator, enclosed."""
        # These guards ensure integer operations and conversion in the
        # vectorized remainder division introduce no unaccounted error.
        assert 0 < denominator <= 2**53
        assert np.all(numerator >= 0) and np.max(numerator) <= 2**53
        k,rem=np.divmod(numerator,denominator)
        assert np.all(k <= N)
        k1=np.minimum(k+1,N)
        t=(np.maximum(0.,dn(rem/denominator)),np.minimum(1.,up(rem/denominator)))
        ans=add(exact(v[k]),mul(t,sub(exact(v[k1]),exact(v[k]))))
        at_knot=rem==0
        return np.where(at_knot,v[k],ans[0]),np.where(at_knot,v[k],ans[1])

    for rr in (rl,rh):
        assert max(N*rr.denominator,N*rr.numerator) < 2**53
    fl=interp(j*rl.numerator,rl.denominator)
    fu=interp(j*rh.numerator,rh.denominator)
    sl=interp(j*rh.numerator+(rh.denominator-rh.numerator)*N,rh.denominator)
    su=interp(j*rl.numerator+(rl.denominator-rl.numerator)*N,rl.denominator)
    vf=(fl[0],fu[1]); vs=(sl[0],su[1])
    qn=[]
    for p,r in ((p1,r1),(p2,r2)):
        qn.append(add(r,mul(gam,expectation_slope(vs[0],vs[1],vf[0],vf[1],p))))
    residual_nodes=sub(mul(w,(np.maximum(qn[0][0],qn[1][0]),
                              np.maximum(qn[0][1],qn[1][1]))),exact(v))

    # Candidate successor-value enclosures throughout each whole grid cell.
    vfc=(vf[0][:-1],vf[1][1:]); vsc=(vs[0][:-1],vs[1][1:])
    qc=[]
    for p,r in ((p1,r1),(p2,r2)):
        qc.append(add(r,mul(gam,expectation_slope(vsc[0],vsc[1],vfc[0],vfc[1],p))))
    dp=(dn(p2[0]-p1[1]),up(p2[1]-p1[0]))
    gap=sub(vsc,vfc)
    gap=(np.maximum(0.,gap[0]),gap[1])
    advantage=sub(mul(mul(gam,dp),gap),cost)
    possible1=advantage[0]<=0
    possible2=advantage[1]>=0
    assert np.all(possible1 | possible2)

    slopes=mul(exact(float(N)),sub(exact(v[1:]),exact(v[:-1])))
    slopes=(np.maximum(0.,slopes[0]),slopes[1])
    ifl,ifu,isl,isu=index_bounds(N,box['rho'])
    fleft=np.maximum(0,ifl[:-1]-1); fright=np.minimum(N-1,ifu[1:])
    sleft=np.maximum(0,isl[:-1]-1); sright=np.minimum(N-1,isu[1:])
    lf=extrema(slopes[0],fleft,fright,True)
    uf=extrema(slopes[1],fleft,fright,False)
    ls=extrema(slopes[0],sleft,sright,True)
    us=extrema(slopes[1],sleft,sright,False)
    coefficient=mul(mul(gam,rho),wc)
    derivatives=[]
    for p,q in ((p1,qc[0]),(p2,qc[1])):
        derivatives.append(sub(add(mul(wprime,q),mul(coefficient,
            expectation_slope(ls,us,lf,uf,p))),slopes))
    ep=(np.minimum(np.where(possible1,derivatives[0][0],np.inf),
                   np.where(possible2,derivatives[1][0],np.inf)),
        np.maximum(np.where(possible1,derivatives[0][1],-np.inf),
                   np.where(possible2,derivatives[1][1],-np.inf)))
    assert np.all(np.isfinite(ep[0])) and np.all(np.isfinite(ep[1]))

    # Each residual is absolutely continuous. Integrate its a.e. derivative
    # enclosure from either endpoint, retaining the sharper bound at each cell.
    from_left_upper=up(residual_nodes[1][:-1]+up(np.maximum(0.,ep[1])/N))
    from_right_upper=up(residual_nodes[1][1:]+up(np.maximum(0.,-ep[0])/N))
    from_left_lower=dn(residual_nodes[0][:-1]+dn(np.minimum(0.,ep[0])/N))
    from_right_lower=dn(residual_nodes[0][1:]-up(np.maximum(0.,ep[1])/N))
    cell_upper=np.minimum(from_left_upper,from_right_upper)
    cell_lower=np.maximum(from_left_lower,from_right_lower)
    assert np.all(cell_lower<=cell_upper)
    residual_lower=float(np.min(cell_lower))
    residual_upper=float(np.max(cell_upper))
    residual=max(abs(residual_lower),abs(residual_upper))
    error=float(up(residual/dn(1-gam[1])))
    # Spatial error majorants avoid transporting the worst residual everywhere.
    # If E_j bounds |V-v| on each closed cell, Bellman nonexpansiveness gives
    # the following monotone update; the scalar bound initializes it rigorously.
    local_residual=np.maximum(np.abs(cell_lower),np.abs(cell_upper))
    cell_error=np.full(N,error)
    discount_wait=mul(gam,wc)
    for eit in range(2500):
        es=extrema(cell_error,sleft,sright,False)
        ef=extrema(cell_error,fleft,fright,False)
        expected=expectation_slope(es,es,ef,ef,(p1[0],p2[1]))[1]
        propagated=mul(discount_wait,exact(expected))[1]
        new_error=np.minimum(cell_error,up(local_residual+propagated))
        reduction=float(np.max(cell_error-new_error))
        cell_error=new_error
        if reduction<1e-12:
            break
    node_error=np.r_[cell_error[0],np.maximum(cell_error[:-1],cell_error[1:]),cell_error[-1]]
    lo=np.maximum(0.,dn(v-node_error)); hi=up(v+node_error)
    meta=dict(method='continuous interpolant residual; outward interval arithmetic and a.e. derivative bounds',
              candidate_iterations=sol.iterations, candidate_tolerance=1e-12,
              residual_lower=residual_lower,residual_upper=residual_upper,
              endpoint_residual_lower=float(np.min(residual_nodes[0])),
              endpoint_residual_upper=float(np.max(residual_nodes[1])),
              residual_derivative_min=float(np.min(ep[0])),
              residual_derivative_max=float(np.max(ep[1])),
              scalar_value_error_upper=error,
              value_error_upper=float(np.max(node_error)),
              minimum_node_error_upper=float(np.min(node_error)),
              error_profile_iterations=eit+1,seconds=time.monotonic()-started)
    return lo,hi,meta
