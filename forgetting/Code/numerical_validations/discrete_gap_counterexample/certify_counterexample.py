"""Computer-assisted counterexample for the exact discrete Bellman equation.

Run from repository root:
  uv run --with python-flint python -B forgetting/Code/numerical_validations/discrete_gap_counterexample/certify_counterexample.py

Only the construction of the candidate uses ordinary floating-point numerics.
The certificate encloses Beta demand using Arb, all further endpoint arithmetic
using outward-rounded intervals, and the continuous residual using a proved
second-derivative bound. The stored nodal floats define an exact interpolant.
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path
from fractions import Fraction
import numpy as np
from flint import arb, ctx

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from forgetting.Code.model import Primitives, solve_boundary

OUT = Path(__file__).resolve().parents[4] / 'forgetting/outputs/numerical_validations/discrete_gap_counterexample'
N = 65536
ctx.prec = 100
NEG_INF, POS_INF = -np.inf, np.inf

def down(x): return np.nextafter(x, NEG_INF)
def up(x): return np.nextafter(x, POS_INF)
def exact(x):
    x = np.asarray(x, dtype=float)
    return x, x

def abounds(x):
    # Arb endpoints themselves round outward at ctx.prec; one nextafter
    # protects their final conversion to IEEE binary64.
    assert x.is_finite()
    return float(down(float(x.lower()))), float(up(float(x.upper())))

def rational(n, d=1): return abounds(arb(int(n)) / int(d))
def add(a, b): return down(a[0]+b[0]), up(a[1]+b[1])
def sub(a, b): return down(a[0]-b[1]), up(a[1]-b[0])
def mul_pos(a, b):
    assert np.all(a[0] >= 0) and np.all(b[0] >= 0)
    return down(a[0]*b[0]), up(a[1]*b[1])

def interp_rational(v, numerator, denominator):
    """v evaluated at z where N*z = numerator/denominator, exactly."""
    numerator = np.asarray(numerator, dtype=np.int64)
    j, rem = np.divmod(numerator, denominator)
    j1 = np.minimum(j+1, N)
    tl, th = down(rem / denominator), up(rem / denominator)
    sl, sh = down((denominator-rem) / denominator), up((denominator-rem) / denominator)
    # At endpoints, clamp valid interval lower bounds to zero.
    return add(mul_pos((np.maximum(0,sl),sh),exact(v[j])),
               mul_pos((np.maximum(0,tl),th),exact(v[j1])))

start = time.time()
p = Primitives(p0=.7,p1=.1,p2=.9,c1=0,c2=.925,revenue=1,gamma=.95)
s = solve_boundary(p,.7,central_nodes=N+1,tolerance=1e-13)
v = s.value
assert len(v)==N+1 and np.array_equal(s.coordinate, np.arange(N+1)/N)
assert np.all(v >= 0) and np.max(v) < 4
# Explicit independent upper bound on |v[j+1]-v[j]|/(1/N).
diffs = sub(exact(v[1:]), exact(v[:-1]))
lip_upper = float(np.max(up(np.maximum(np.abs(diffs[0]),np.abs(diffs[1]))*N)))
assert lip_upper < 5
np.savez_compressed(OUT/'counterexample_value.npz',x=s.coordinate,value=v)
print(f'Candidate constructed: {N+1} nodes, Lip(v)<5; {time.time()-start:.1f}s',flush=True)

# Partition at all knots of v(x), v(.7*x), and v(.7*x+.3).
# Every endpoint is m/(7*N), with integer m.
j = np.arange(N+1,dtype=np.int64)
m = np.unique(np.concatenate((7*j, 10*j, 10*j-3*N)))
m = m[(m>=0)&(m<=7*N)]
wl=np.empty(len(m)); wh=np.empty(len(m))
for k, numerator in enumerate(m):
    a=1+arb(10)*int(numerator)/(21*N)
    b=1+arb(10)*int(7*N-numerator)/(21*N)
    demand=(arb(3)/10).beta_lower(b,a,regularized=True)
    waiting=demand/(arb(1)/20+arb(19)/20*demand)
    wl[k], wh[k]=abounds(waiting)
    if k and k%30000==0:
        print(f'Arb demand endpoints: {k}/{len(m)}; {time.time()-start:.1f}s',flush=True)
assert np.all(wl>=0) and np.all(wh<=1)

v0=interp_rational(v,m,7)
vf=interp_rational(v,m,10)
vs=interp_rational(v,m+3*N,10)
p1=rational(1,10); p2=rational(9,10); gamma=rational(19,20)
q1=add(exact(1),mul_pos(gamma,add(mul_pos(p1,vs),mul_pos(p2,vf))))
q2=add(rational(3,40),mul_pos(gamma,add(mul_pos(p2,vs),mul_pos(p1,vf))))
e1=sub(mul_pos((wl,wh),q1),v0)
e2=sub(mul_pos((wl,wh),q2),v0)
endpoint_upper=np.maximum(e1[1],e2[1])
endpoint_lower=np.maximum(e1[0],e2[0])

# For each interval, the maximum of the two residual chords has its
# maximum at an endpoint and its minimum at an endpoint or intersection.
chord_min=np.minimum(endpoint_lower[:-1],endpoint_lower[1:]).copy()
win=e1[0]>=e2[0]
cross=np.flatnonzero(win[:-1]!=win[1:])
for k in cross:
    a,b,c,d=map(arb,[float(e1[0][k]),float(e2[0][k]),float(e1[0][k+1]),float(e2[0][k+1])])
    t=(b-a)/((c-a)-(d-b))
    assert t>=0 and t<=1
    intersection=a+t*(c-a)
    chord_min[k]=min(chord_min[k],abounds(intersection)[0])

# Proven elementary global constants, with k=1-rho=3/10:
# |D'|<=1/k=10/3, |D''|<=4/k^2=400/9.
# |w'|<=200/3, |w''|<=28000/3 (derivatives in x).
# Q_i<=1+gamma*max(v)<5 and |Q_i'|<=gamma*rho*Lip(v)<133/40.
# Thus |(w*Q_i-v)''| <= 5*(28000/3)+2*(133/40)*(200/3)=141330/3=47110.
C=arb(47110)
hmax=arb(int(np.max(np.diff(m))))/(7*N)
curvature=abounds(C*hmax*hmax/8)[1]
residual_upper=float(up(np.max(endpoint_upper)+curvature))
residual_lower=float(down(np.min(chord_min)-curvature))
residual=max(residual_upper,-residual_lower)
# Bellman contraction modulus <= gamma = 19/20, so value error <=20*residual.
value_error=float(up(20*residual))

rows=[]
for xstr in ['31/100','27/80','9/25']:
    x=Fraction(xstr)
    xf=Fraction(7,10)*x
    xs=xf+Fraction(3,10)
    vsx=interp_rational(v,np.array([xs.numerator*N]),xs.denominator)
    vfx=interp_rational(v,np.array([xf.numerator*N]),xf.denominator)
    gap=sub(vsx,vfx)
    err=float(up(2*value_error))
    lower=float(down(gap[0][0]-err)); upper=float(up(gap[1][0]+err))
    rows.append(dict(x=xstr,x_decimal=float(x),success=str(xs),failure=str(xf),
                     gap_approx=float((gap[0][0]+gap[1][0])/2),
                     gap_lower=lower,gap_upper=upper))
assert rows[0]['gap_lower']>rows[1]['gap_upper']
assert rows[2]['gap_lower']>rows[1]['gap_upper']
result=dict(parameters={'rho':'7/10','p0':'7/10','p1':'1/10','p2':'9/10',
                        'c1':'0','c2':'37/40','R':'1','gamma':'19/20'},
            grid_intervals=N,partition_intervals=len(m)-1,arb_precision_bits=ctx.prec,
            max_interpolant=float(np.max(v)),interpolant_lipschitz_upper=lip_upper,
            residual_curvature_bound=47110,maximum_partition_length=str(hmax),
            curvature_remainder_upper=curvature,
            endpoint_residual_upper=float(np.max(endpoint_upper)),
            chord_residual_lower=float(np.min(chord_min)),
            residual_lower=residual_lower,residual_upper=residual_upper,
            sup_value_error_upper=value_error,gap_error_upper=float(up(2*value_error)),
            points=rows,certified_not_unimodal=True,
            certified_valley_depth_lower=float(down(min(rows[0]['gap_lower'],rows[2]['gap_lower'])-rows[1]['gap_upper'])),
            runtime_seconds=time.time()-start)
(OUT/'certificate.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2),flush=True)
