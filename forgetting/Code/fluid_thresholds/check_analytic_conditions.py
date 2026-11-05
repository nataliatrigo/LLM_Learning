"""Numerical checks of analytical criteria; these are not interval certificates.

Run from the repository root:
    uv run python -B forgetting/Code/fluid_thresholds/check_analytic_conditions.py
"""
from pathlib import Path
import json
import math

from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from scipy.special import digamma, expit, polygamma

from check_characterization import beta_prime
from solve_fluid import C1, C2, DELTA, P1, P2, R, demand_at


OUT = Path(__file__).resolve().parents[3] / 'forgetting/outputs/fluid_thresholds'
K = (C2 - C1) / (P2 - P1)
Q = P1 + (R - C1) / K


def explicit_bound(rho):
    n = 1 / (1 - rho)
    mu0 = digamma(1) - digamma(1 + n)
    var0 = polygamma(1, 1) + polygamma(1, 1 + n)
    d0 = 1 / (n * Q)
    return float(expit(mu0 + d0 - math.sqrt(var0 + d0 * d0)))


def candidate_diagnostics(rho, p0=0.5, rtol=1e-9, atol=1e-12):
    demand = demand_at(rho, p0)
    lam = -math.log(rho)
    k = DELTA / lam

    def h(x):
        return (Q - x) * beta_prime(x, rho, p0) - demand(x) - k

    if not h(P1) > 0 > h(P2):
        raise ValueError('This diagnostic requires H(p1)>0>H(p2).')
    # This locates a candidate; uniqueness is a separate analytical hypothesis.
    z = brentq(h, P1, P2, xtol=2e-13)

    def rhs(x, g):
        return [DELTA * g[0] / (lam * demand(x) * (P2 - x)) - h(x)]

    sol = solve_ivp(rhs, [z, 0.0], [0.0], method='Radau',
                    rtol=rtol, atol=atol, dense_output=True)
    if not sol.success:
        raise RuntimeError(sol.message)
    c = float(sol.y[0, -1])
    lower = None
    if c < 0:
        lower = float(brentq(lambda x: sol.sol(x)[0], 0, P1,
                             xtol=2e-12))
    return dict(rho=rho, p0=p0, upper_candidate=z, C=c,
                lower_candidate=lower, H0=h(0), Hp1=h(P1), Hp2=h(P2))


def main():
    reference = json.loads((OUT / 'characteristic_checks.json').read_text())
    rows = []
    for rho in (0.7, 0.8, 0.9, 0.95):
        coarse = candidate_diagnostics(rho)
        fine = candidate_diagnostics(rho, rtol=2e-10, atol=2e-14)
        prior = next(row for row in reference if row['rho'] == rho)
        assert abs(fine['upper_candidate'] - prior['upper']) < 1e-8
        assert (fine['lower_candidate'] is None) == (prior['lower'] is None)
        if prior['lower'] is not None:
            assert abs(fine['lower_candidate'] - prior['lower']) < 2e-7
        assert (fine['C'] > 0) == (coarse['C'] > 0)
        fine['C_tolerance_change'] = abs(fine['C'] - coarse['C'])
        rows.append(fine)
        print(json.dumps(fine), flush=True)

    boundary = brentq(lambda rho: candidate_diagnostics(rho)['C'],
                      0.8, 0.9, xtol=2e-10)
    report = {
        'status': 'Floating-point checks only, not a proof of the shape hypothesis.',
        'primitives': dict(p1=P1, p2=P2, c1=C1, c2=C2, R=R,
                           delta=DELTA, K=K, q=Q),
        'candidate_checks': rows,
        'conditional_C_zero_at_p0_0_5': boundary,
        'explicit_sufficient_bounds': [
            dict(rho=r, p0_upper=explicit_bound(r))
            for r in (0.6, 0.7, 0.8, 0.9, 0.95, 0.99)
        ],
    }
    path = OUT / 'analytic_conditions_checks.json'
    path.write_text(json.dumps(report, indent=2) + '\n')
    print(f'Conditional C=0 boundary at p0=.5: rho={boundary:.10f}')
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
