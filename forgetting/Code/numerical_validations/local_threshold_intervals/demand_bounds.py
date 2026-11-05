"""Validated exact Beta demand and spatial derivative bounds on parameter boxes.

The caller supplies rational interval endpoints (strings, ints, or Fraction).
All special functions and error-bound arithmetic use Arb; conversion to NumPy
uses outward endpoint rounding. No sampled or floating derivative is certified.

Run a nonrigorous implementation sanity check with:
  uv run --with python-flint python -B forgetting/Code/numerical_validations/local_threshold_intervals/demand_bounds.py
"""
from __future__ import annotations
from fractions import Fraction
import time
import numpy as np
from flint import arb, ctx


def _fraction(value):
    if isinstance(value, float):
        raise TypeError('Pass exact rational strings/Fraction rather than floats.')
    return Fraction(value)


def _arb(value):
    q = _fraction(value)
    return arb(q.numerator) / q.denominator


def _interval(endpoints):
    lo, hi = map(_fraction, endpoints)
    if lo > hi:
        raise ValueError('Reversed interval.')
    return lo, hi, _arb(lo).union(_arb(hi))


def _outward(value):
    if not value.is_finite():
        raise ArithmeticError('Nonfinite Arb enclosure.')
    return (float(np.nextafter(float(value.lower()), -np.inf)),
            float(np.nextafter(float(value.upper()), np.inf)))


def _demand(x, rho, p0):
    a = 1 + x / (1-rho)
    b = 1 + (1-x) / (1-rho)
    # Complementary regularized Beta via lower tail at 1-p0, avoiding subtraction.
    return (1-p0).beta_lower(b, a, regularized=True)


def _trigamma(x):
    return arb(2).zeta(x)


def demand_bounds(n_cells, rho, p0, precision=100):
    """Return rigorous node/cell demand and a.e.-usable D_x cell enclosures.

    Keys ``nodes``, ``cells``, and ``derivative`` map to (lower, upper) NumPy
    arrays, respectively of lengths n_cells+1, n_cells, n_cells. The bounds
    hold simultaneously for every rho and p0 in the supplied rational boxes.
    ``metadata`` gives the method and uniform second derivative bound.
    """
    if not isinstance(n_cells, int) or n_cells < 2:
        raise ValueError('n_cells must be an integer >=2.')
    ctx.prec = precision
    rl, ru, rb = _interval(rho)
    pl, pu, pb = _interval(p0)
    if not (0 < rl <= ru < 1 and 0 < pl <= pu < 1):
        raise ValueError('rho and p0 intervals must lie inside (0,1).')
    rc = _arb((rl+ru)/2)
    pc = _arb((pl+pu)/2)
    rr = _arb((ru-rl)/2)
    pr = _arb((pu-pl)/2)
    sl = 1/(1-_arb(rl))
    su = 1/(1-_arb(ru))
    # For alpha,beta>=1: D_xx=Cov(1_{theta>p0},(logit(theta)-Elogit)^2)/h^2.
    # Its absolute value is <=Var(logit)/h^2 <= (pi^2/3)/h^2 <4/h^2.
    second = 4*su**2
    first_upper = _outward(su)[1]
    summax = 2+su
    tri_summax = _trigamma(summax)
    # D_xxx=s^3[Cov(I,Y^3)-3*kappa2*Cov(I,Y)], Y=logit(theta)-Elogit.
    # kappa2<=4, kappa4<=16 give E[Y^4]<=64, E|Y^3|<=16, and
    # |Cov(I,Y)|<=sqrt(kappa2)/2<=1. Thus |D_xxx|<=28*s^3<32*s^3.
    third = 32*su**3
    chord_error = third/_arb(8*n_cells*n_cells)
    stepq=min(Fraction(1,10**8),Fraction(1,4*n_cells))
    step=_arb(stepq)
    fd_error=(second*step/2).upper()
    fd_error_ball=(-fd_error).union(fd_error)
    centered_derivative_nodes=[]
    node_lo=np.empty(n_cells+1); node_hi=np.empty(n_cells+1)
    dx_lo=np.empty(n_cells); dx_hi=np.empty(n_cells)
    start=time.monotonic()
    for j in range(n_cells+1):
        x = _arb(Fraction(j,n_cells))
        raw_demand=_demand(x,rb,pb)
        # Arb can decline a wide shape-parameter ball; [0,1] remains a valid
        # starting enclosure, subsequently tightened by the independent bound.
        lo,hi = _outward(raw_demand) if raw_demand.is_finite() else (0.,1.)
        # An independent enclosure retains exact monotonicity in p0 and
        # controls rho by D_rho=s^2 Cov(I,H), H=x*log(theta)+(1-x)*log(1-theta).
        # Var(H)=x^2*psi1(a)+(1-x)^2*psi1(b)-psi1(a+b).
        # This avoids losing the opposite movement of a and b inside Arb.
        amin=1+x*sl
        bmin=1+(1-x)*sl
        h_variance=x*x*_trigamma(amin)+(1-x)**2*_trigamma(bmin)-tri_summax
        rho_error=rr*su**2*h_variance.sqrt()/2
        lower_monotone=_demand(x,rc,_arb(pu))-rho_error.upper()
        upper_monotone=_demand(x,rc,_arb(pl))+rho_error.upper()
        lo=max(lo,_outward(lower_monotone)[0])
        hi=min(hi,_outward(upper_monotone)[1])
        # Demand is a probability, so intersection with [0,1] is rigorous.
        node_lo[j]=max(0.,lo); node_hi[j]=min(1.,hi)
        if node_lo[j]>node_hi[j]:
            raise ArithmeticError('Empty demand enclosure.')
        # Centered differences in the interior, one-sided at endpoints. Both
        # average D_x and have error <=C2*step/2 at the evaluation node.
        if j==0:
            slope=(_demand(x+step,rc,pc)-_demand(x,rc,pc))/step
        elif j==n_cells:
            slope=(_demand(x,rc,pc)-_demand(x-step,rc,pc))/step
        else:
            slope=(_demand(x+step,rc,pc)-_demand(x-step,rc,pc))/(2*step)
        centered_derivative_nodes.append(slope+fd_error_ball)
    for j in range(n_cells):
        xlq=Fraction(j,n_cells); xuq=Fraction(j+1,n_cells)
        xl=_arb(xlq); xu=_arb(xuq)
        xb=xl.union(xu)
        # The linear interpolant of the two exact endpoint derivatives stays
        # in their interval hull; the C3 bound controls its error by C3*h^2/8.
        derivative_hull=centered_derivative_nodes[j].union(centered_derivative_nodes[j+1])
        error=chord_error

        if ru != rl:
            # D_xrho = s^2(D_a-D_b)+s^3[x D_aa+(1-2x)D_ab-(1-x)D_bb].
            # Cauchy-Schwarz and centered-score identities give
            # |D_a-D_b|<=sqrt(psi1(a)+psi1(b))/2,
            # |D_aa|<=Va, |D_bb|<=Vb, |D_ab|<=sqrt(Va*Vb).
            # psi1 decreases, so the following are uniform cell/box bounds.
            amin=1+xl*sl
            bmin=1+(1-xu)*sl
            ta=_trigamma(amin); tb=_trigamma(bmin)
            va=ta-tri_summax; vb=tb-tri_summax
            cl=(ta+tb).sqrt()/2
            rho_mixed=su**2*(cl+su*(xu*va+abs(1-2*xb)*(va*vb).sqrt()+(1-xl)*vb))
            error+=rr*rho_mixed.upper()

        if pu != pl:
            # d(D_x)/dp0 = -f(p0)[logit(p0)-psi(a)+psi(b)]/(1-rho).
            a=(1+xl*sl).union(1+xu*su)
            b=(1+(1-xu)*sl).union(1+(1-xl)*su)
            log_density=((a-1)*pb.log()+(b-1)*(1-pb).log()
                         -a.lgamma()-b.lgamma()+(a+b).lgamma())
            density=log_density.exp()
            p0_mixed=abs(density*((pb/(1-pb)).log()-a.digamma()+b.digamma())/(1-rb))
            error+=pr*p0_mixed.upper()

        radius=error.upper()
        enclosure=derivative_hull+(-radius).union(radius)
        # D_x=s*Cov(I,logit(theta)); Cauchy-Schwarz gives 0<=D_x<=s.
        # This also supplies a rigorous fallback if a very wide parameter ball
        # prevents Arb from resolving an intermediate special function.
        lo,hi=_outward(enclosure) if enclosure.is_finite() else (0.,first_upper)
        dx_lo[j]=max(0.,lo); dx_hi[j]=min(first_upper,hi)
        if dx_lo[j]>dx_hi[j]:
            raise ArithmeticError('Empty derivative enclosure.')

    # For each parameter choice demand increases in x; endpoint boxes suffice.
    nodes=(node_lo,node_hi)
    cells=(node_lo[:-1].copy(),node_hi[1:].copy())
    derivative=(dx_lo,dx_hi)
    return dict(nodes=nodes,cells=cells,derivative=derivative,
                D_nodes=nodes,D_cells=cells,Dx_cells=derivative,
                metadata=dict(n_cells=n_cells,rho=[str(rl),str(ru)],
                              p0=[str(pl),str(pu)],precision_bits=precision,
                              second_derivative_upper=_outward(second)[1],
                              third_derivative_upper=_outward(third)[1],
                              derivative_chord_error_upper=_outward(chord_error)[1],
                              elapsed_seconds=time.monotonic()-start,
                              method='Arb Beta tails; validated endpoint derivatives and third derivative chord bound; analytic mixed-parameter bounds'))


build_demand_bounds=demand_bounds

if __name__=='__main__':
    # This finite-difference comparison is a bug check, not part of the proof.
    from scipy.special import betaincc
    result=demand_bounds(128,('699/1000','701/1000'),('499/1000','501/1000'))
    n=128
    for r in [.699,.7,.701]:
        for p in [.499,.5,.501]:
            def d(x):
                return betaincc(1+x/(1-r),1+(1-x)/(1-r),p)
            nodes=np.arange(n+1)/n
            assert np.all(d(nodes)>=result['nodes'][0])
            assert np.all(d(nodes)<=result['nodes'][1])
            for offset in [.05,.5,.95]:
                xs=(np.arange(n)+offset)/n
                deriv=(d(xs+1e-6)-d(xs-1e-6))/(2e-6)
                assert np.all(deriv>=result['derivative'][0])
                assert np.all(deriv<=result['derivative'][1])
    print(result['metadata'])
    print('Nonrigorous implementation sanity checks passed.')
