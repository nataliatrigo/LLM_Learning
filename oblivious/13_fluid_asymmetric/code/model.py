"""Crossed fixed-rival optimal policies and their common-start consistency.

This tests the endpoint pair (p1,p2), not all asymmetric stationary pairs.
Optimality refers to the same frozen-rival HJB as experiments 10 and 11.
Actual demand along a joint path multiplies each own-state phase line by a
strictly positive factor. Consequently each phase-line limit survives joint
evolution, and common starts are exactly the intersection of the two basins.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import betainc, betaln, logsumexp, roots_legendre

from oblivious.fluid_oblivious.code.model import FixedRivalDemand, demand_at_states, solve_response
from oblivious.fluid_oblivious_initial_conditions.code.initial_conditions import (
    contains, initial_conditions_for_equilibrium, interval_notation,
)


def independent_transformed_demand(params, p0, conjecture, own, order=256, power=4):
    """Independent direct integral with a smooth endpoint transformation.

    t=1-(1-p0)*u**power transforms the Beta endpoint weight to
    power*(1-p0)**b*u**(power*b-1)*t**(a-1)/B(a,b).
    Since b>=1 and power>=4, the integrand has no low-order endpoint
    singularity. Plain Legendre quadrature in t can be inaccurate near x=1.
    No prepared Jacobi weights or event-partition repairs are used here.
    """
    own = np.asarray(own, dtype=float)
    if (not 0 <= p0 < 1 or not 0 <= conjecture <= 1
            or np.any(~np.isfinite(own)) or np.any((own < 0) | (own > 1))
            or order < 16 or power < 4):
        raise ValueError("Invalid states, quadrature order, or endpoint power")
    nodes, weights = roots_legendre(int(order))
    u = (nodes+1)/2
    t = 1-(1-p0)*u**power
    log_t = np.log1p(-(1-p0)*u**power)
    concentration = 1/(1-params.rho)
    a, b = 1+own*concentration, 1+(1-own)*concentration
    c, d = 1+conjecture*concentration, 1+(1-conjecture)*concentration
    with np.errstate(divide="ignore"):
        common = np.log(weights/2)+np.log(betainc(c, d, t))+math.log(power)
    logs = (common+(a[..., None]-1)*log_t+(power*b[..., None]-1)*np.log(u)
            +b[..., None]*math.log1p(-p0)-betaln(a, b)[..., None])
    result = np.exp(logsumexp(logs, axis=-1))
    return float(result) if result.ndim == 0 else result


def intersect_intervals(first, second):
    """Intersect interval unions, respecting open ends and singleton points."""
    pieces = []
    for a in first:
        for b in second:
            lo, hi = max(a["lo"], b["lo"]), min(a["hi"], b["hi"])
            if lo > hi:
                continue
            left = contains([a], lo) and contains([b], lo)
            right = contains([a], hi) and contains([b], hi)
            if lo < hi or left and right:
                pieces.append(dict(lo=lo, hi=hi, left_closed=left, right_closed=right))
    merged = []
    for part in sorted(pieces, key=lambda row: (row["lo"], row["hi"])):
        if not merged:
            merged.append(dict(part))
            continue
        previous = merged[-1]
        touching = previous["hi"] == part["lo"] and (
            previous["right_closed"] or part["left_closed"])
        if previous["hi"] > part["lo"] or touching:
            if previous["lo"] == part["lo"]:
                previous["left_closed"] |= part["left_closed"]
            if part["hi"] > previous["hi"]:
                previous.update(hi=part["hi"], right_closed=part["right_closed"])
            elif part["hi"] == previous["hi"]:
                previous["right_closed"] |= part["right_closed"]
        else:
            merged.append(dict(part))
    return merged


def analyse_response(params, p0, conjecture, target, n=4001, order=128, *, prepared=None):
    """Solve a fresh global HJB and test holding at the own endpoint target."""
    if target not in (params.p1, params.p2):
        raise ValueError("This experiment tests pure endpoint targets only")
    solution = solve_response(params, p0, conjecture, n=n, order=order, prepared=prepared)
    policy = {key: solution[key] for key in
              ("roots", "products", "root_kinds", "intervals", "holding_points")}
    basin = initial_conditions_for_equilibrium(params, policy, target)
    demand = float(np.interp(target, solution["x"], solution["demand"]))
    alpha = float(target == params.p2)
    reward = params.R-params.c1-(params.c2-params.c1)*alpha
    delta, lam = -math.log(params.gamma), -math.log(params.rho)
    holding_value = demand*reward/delta
    optimized_value = float(np.interp(target, solution["x"], solution["value"]))
    holding_gap = optimized_value-holding_value
    conditional_defect = delta*holding_gap/max(demand, np.finfo(float).tiny)
    gap = float(np.interp(target, solution["x"], solution["gap"]))
    action_valid = gap <= 1e-8 if alpha == 0 else gap >= -1e-8
    h = 1/(n-1)
    k = (params.c2-params.c1)/(lam*(params.p2-params.p1))
    value_tolerance = max(2e-6, 50*h*h*(1+abs(k))*(1+abs(optimized_value)))
    conditional_tolerance = max(2e-5, 10*h*h*(1+params.c2-params.c1+lam*abs(k)))
    target_optimal = bool(action_valid and basin["stationary"]
                          and abs(holding_gap) <= value_tolerance
                          and abs(conditional_defect) <= conditional_tolerance
                          and solution["residual"] <= 1e-8)
    return dict(conjecture=float(conjecture), target=float(target), grid_nodes=int(n),
                policy=policy, basin=basin,
                diagnostics=dict(hjb_residual=float(solution["residual"]),
                    conditional_regret=float(solution["conditional_regret"]),
                    holding_value=float(holding_value), optimized_value=optimized_value,
                    holding_value_gap=float(holding_gap), conditional_holding_defect=float(conditional_defect),
                    hamiltonian_gap=gap, action_valid=bool(action_valid), target_optimal=target_optimal,
                    holding_value_tolerance=float(value_tolerance),
                    conditional_tolerance=float(conditional_tolerance),
                    target_demand=demand, howard_iterations=int(solution["iterations"])))


def analyse_pair(params, p0, n=4001, order=128):
    """A optimizes against p2; B optimizes against p1. No saved policies used."""
    prepared = FixedRivalDemand(params, p0, np.linspace(0, 1, n), order)
    a = analyse_response(params, p0, params.p2, params.p1, n, order, prepared=prepared)
    b = analyse_response(params, p0, params.p1, params.p2, n, order, prepared=prepared)
    common = intersect_intervals(a["basin"]["intervals"], b["basin"]["intervals"])
    return dict(p0=float(p0), rho=float(params.rho), grid_nodes=int(n), response_a=a, response_b=b,
                common_initial_intervals=common, common_initial_notation=interval_notation(common),
                qualifies=bool(common and a["diagnostics"]["target_optimal"]
                               and b["diagnostics"]["target_optimal"]))


def compare_refinement(coarse, fine, tolerance):
    """Compare all roots, target tests and exact interval-union topology."""
    changes = []
    topology = coarse["qualifies"] == fine["qualifies"]
    for label in ("response_a", "response_b"):
        a, b = coarse[label], fine[label]
        topology &= (a["policy"]["products"] == b["policy"]["products"]
                     and a["policy"]["root_kinds"] == b["policy"]["root_kinds"]
                     and a["diagnostics"]["target_optimal"] == b["diagnostics"]["target_optimal"])
        if len(a["policy"]["roots"]) != len(b["policy"]["roots"]):
            topology = False
        else:
            changes.extend(abs(x-y) for x, y in zip(a["policy"]["roots"], b["policy"]["roots"]))
        x, y = a["basin"]["intervals"], b["basin"]["intervals"]
        if len(x) != len(y):
            topology = False
        else:
            for old, new in zip(x, y):
                topology &= (old["left_closed"] == new["left_closed"]
                             and old["right_closed"] == new["right_closed"]
                             and (old["lo"] == old["hi"]) == (new["lo"] == new["hi"]))
                changes.extend(abs(old[key]-new[key]) for key in ("lo", "hi"))
    change = max(changes, default=0.)
    return dict(coarse_n=coarse["grid_nodes"], fine_n=fine["grid_nodes"],
                same_topology=bool(topology), maximum_root_or_boundary_change=float(change),
                tolerance=float(tolerance), passed=bool(topology and change <= tolerance))


def simulate_pure_pair_log(params, p0, pair, x0, times, *, order=256,
                          rtol=1e-10, atol=1e-12, max_step=10000.):
    """Stable long paths for the constant actions actually selected on this orbit.

    For A's product 1 and B's product 2, write
      x_i=p_i+(x0-p_i)*exp(ell_i), ell_i(0)=0,
      ell_i'=-lambda*D_i(x_A,x_B).
    This exact coordinate transformation avoids roundoff/stiffness once B is
    close to p2. Both demands still use both realized states. It is valid only
    if neither own-state path crosses a policy switch; this is checked first.
    """
    times = np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("times must strictly increase from zero")
    if not params.p1 < x0 < params.p2:
        raise ValueError("Pure-pair log coordinates require p1 < x0 < p2")
    for label, target, product in (("a", params.p1, 1), ("b", params.p2, 2)):
        policy = pair["response_"+label]["policy"]
        lo, hi = sorted((x0, target))
        if any(lo < root < hi for root in policy["roots"]):
            raise ValueError("A policy switch lies on the proposed constant-action path")
        # Endpoints at repelling ties need explicit event semantics; excluded here.
        if x0 in policy["roots"]:
            raise ValueError("Use the event integrator at an exact policy threshold")
        region = int(np.searchsorted(policy["roots"], x0))
        if policy["products"][region] != product:
            raise ValueError("The selected policy does not use the required pure product")
    targets = np.asarray([params.p1, params.p2])
    distances = float(x0)-targets
    lam = -math.log(params.rho)

    def derivative(_time, log_distance):
        states = targets+distances*np.exp(log_distance)
        da, db, _outside = demand_at_states(*states, params, p0, order=order)
        return -lam*np.asarray([da, db])

    solution = solve_ivp(derivative, (0., float(times[-1])), [0., 0.],
                         method="DOP853", t_eval=times, rtol=rtol, atol=atol, max_step=max_step)
    if not solution.success:
        raise RuntimeError(solution.message)
    states = targets[:, None]+distances[:, None]*np.exp(solution.y)
    return dict(time=times, x_a=states[0], x_b=states[1], nfev=int(solution.nfev),
                method="Exact log-distance coordinate change; actual joint demand; DOP853")
