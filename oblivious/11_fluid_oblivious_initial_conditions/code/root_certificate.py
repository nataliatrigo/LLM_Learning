"""Independent numerical enumeration of necessary stationary candidates.

This uses ordinary Gauss--Legendre integration, not the production solver's
state-dependent Gauss--Jacobi rule. The diagonal derivative is analytic and
includes the rival-distribution derivative. Dense/refined scans and extrema
checks provide a numerical completeness audit, not interval-arithmetic bounds.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import betainc, betaincc, betaln, digamma, polygamma, roots_legendre


@lru_cache(maxsize=12)
def _rule(order):
    return roots_legendre(int(order))


def stationary_conditions(params, p0, z, *, order=512, chunk_size=256):
    """Return F, dF/dz and demand quantities at diagonal conjectures z.

    Write h=1-rho, s=partial_z log(f_z), and A=int_tail f_z*s.
    For J(z)=partial_x D(x,z)|x=z, differentiating along the diagonal gives
      J' = int_tail f_z*(s*s+s')*CDF_z - A*A/2.
    The second term follows by integration by parts because partial_z CDF_z
    has t-derivative f_z*s and is zero at t=1. This avoids finite differences
    of Beta CDF parameters and of the HJB value function.
    """
    if not 0 <= p0 < 1 or int(order) != order or order < 16:
        raise ValueError("Require 0 <= p0 < 1 and integer quadrature order >= 16")
    if not 0 < params.rho < 1 or not 0 < params.gamma < 1:
        raise ValueError("rho and gamma must lie strictly between zero and one")
    if not params.p1 < params.p2:
        raise ValueError("Require p1 < p2")
    values = np.asarray(z, dtype=float)
    scalar = values.ndim == 0
    shape = values.shape
    flat = values.reshape(-1)
    if not np.isfinite(flat).all() or np.any((flat < 0) | (flat > 1)):
        raise ValueError("Conjectures must be finite and lie in [0,1]")
    nodes, weights = _rule(order)
    t = p0+(1-p0)*(nodes+1)/2
    weight = weights*(1-p0)/2
    log_t, log_one_minus_t = np.log(t), np.log1p(-t)
    h = 1-params.rho
    lam, delta = -math.log(params.rho), -math.log(params.gamma)
    margin_slope = (params.c2-params.c1)/(params.p2-params.p1)
    coefficient = margin_slope/lam
    names = ("residual", "derivative", "demand", "demand_diagonal_derivative",
             "own_demand_derivative", "own_demand_diagonal_second_derivative", "reward",
             "own_demand_second_derivative", "holding_value_curvature")
    outputs = {name:np.empty(len(flat)) for name in names}
    for start in range(0, len(flat), chunk_size):
        stop = min(len(flat), start+chunk_size)
        part = flat[start:stop]
        a, b = 1+part/h, 1+(1-part)/h
        density = np.exp((a[:,None]-1)*log_t+(b[:,None]-1)*log_one_minus_t-betaln(a,b)[:,None])
        score = (log_t-log_one_minus_t-digamma(a)[:,None]+digamma(b)[:,None])/h
        score_derivative = -(polygamma(1,a)+polygamma(1,b))/(h*h)
        cdf = betainc(a[:,None],b[:,None],t)
        cdf_outside = betainc(a,b,p0)
        own = np.sum(weight*density*score*cdf,axis=1)
        tail_score = np.sum(weight*density*score,axis=1)
        own_second = np.sum(weight*density*(score*score+score_derivative[:,None])*cdf,axis=1)
        own_diagonal = own_second-.5*tail_score*tail_score
        demand = .5*betaincc(a,b,p0)*(1+cdf_outside)
        demand_diagonal = cdf_outside*tail_score
        reward = params.R-params.c1-margin_slope*(part-params.p1)
        results = (own*reward-coefficient*(delta+lam*demand),
                   own_diagonal*reward-margin_slope*(own+demand_diagonal),
                   demand,demand_diagonal,own,own_diagonal,reward,own_second,
                   (reward*own_second-2*margin_slope*own)/delta)
        for name,result in zip(names,results,strict=True):
            outputs[name][start:stop] = result
    return {name:float(result[0]) if scalar else result.reshape(shape)
            for name,result in outputs.items()}


def _isolate_sampled_roots(function, grid, samples, *, zero_tolerance=1e-10):
    """Sign brackets plus local |function| minima, retaining repeated zeros."""
    roots = []
    for index,(left,right,fl,fr) in enumerate(zip(grid,grid[1:],samples,samples[1:])):
        if fl == 0:
            roots.append(dict(z=float(left),bracket=[float(left),float(left)],method="sample_zero"))
        if fl*fr < 0:
            point = brentq(function,float(left),float(right),xtol=2e-13,rtol=1e-14)
            roots.append(dict(z=float(point),bracket=[float(left),float(right)],method="sign_change"))
    if samples[-1] == 0:
        roots.append(dict(z=float(grid[-1]),bracket=[float(grid[-1]),float(grid[-1])],method="sample_zero"))
    for index in range(1,len(grid)-1):
        if abs(samples[index]) < min(abs(samples[index-1]),abs(samples[index+1])):
            left,right = float(grid[index-1]),float(grid[index+1])
            result = minimize_scalar(lambda x:abs(function(x)),bounds=(left,right),
                                     method="bounded",options={"xatol":1e-14})
            if result.success and abs(function(result.x)) <= zero_tolerance:
                roots.append(dict(z=float(result.x),bracket=[left,right],method="absolute_minimum"))
    roots.sort(key=lambda row:row["z"])
    unique = []
    for row in roots:
        if not unique or row["z"]-unique[-1]["z"] > 2e-7:
            unique.append(row)
        elif row["method"] == "sign_change":
            unique[-1] = row
    return unique


def _analyse_scan(params,p0,grid,order,zero_tolerance):
    evaluated = stationary_conditions(params,p0,grid,order=order)
    function = lambda z:stationary_conditions(params,p0,z,order=order)["residual"]
    derivative = lambda z:stationary_conditions(params,p0,z,order=order)["derivative"]
    roots = _isolate_sampled_roots(function,grid,evaluated["residual"],zero_tolerance=zero_tolerance)
    extrema = _isolate_sampled_roots(derivative,grid,evaluated["derivative"],zero_tolerance=zero_tolerance)
    for point in extrema:
        residual = function(point["z"])
        point["residual"] = float(residual)
        if abs(residual) <= zero_tolerance:
            roots.append(dict(z=point["z"],bracket=point["bracket"],method="stationary_zero"))
    roots.sort(key=lambda row:row["z"])
    unique = []
    for row in roots:
        if params.p1+1e-10 < row["z"] < params.p2-1e-10:
            if not unique or row["z"]-unique[-1]["z"] > 2e-7:
                row["residual"] = function(row["z"])
                row["derivative"] = derivative(row["z"])
                unique.append(row)
    return evaluated,unique,extrema


def analytic_global_derivative_bounds(params):
    """Conservative analytic envelopes for |F'| and |F''| on [p1,p2].

    Beta score moments obey E[s]=0, E[s²]=I and E[s⁴]=kappa4+3I².
    For CDF parameter derivatives G,H, covariance with an indicator gives
    |G|<=sqrt(I)/2 and |H|<=sqrt(E[s⁴]-I²)/2. Polygamma monotonicity at the
    minimum admissible a,b supplies uniform moment envelopes. These analytic
    envelopes do not by themselves validate quadrature evaluations of F,F'.
    """
    h = 1-params.rho
    a_min,b_min = 1+params.p1/h,1+(1-params.p2)/h
    information = float((polygamma(1,a_min)+polygamma(1,b_min))/(h*h))
    third = float((-polygamma(2,a_min)-polygamma(2,b_min))/(h**3))
    fourth_cumulant = float((polygamma(3,a_min)+polygamma(3,b_min))/(h**4))
    fourth = fourth_cumulant+3*information*information
    square_variance = fourth_cumulant+2*information*information
    root_i,root_v = math.sqrt(information),math.sqrt(square_variance)
    own_first_bound = root_v+information/8
    own_second_bound = (math.sqrt(information*fourth)+3*information*root_i+third
                        +.75*root_i*root_v)
    demand_second_bound = information/4+.5*root_v
    margin_slope = abs((params.c2-params.c1)/(params.p2-params.p1))
    maximum_reward = max(abs(params.R-params.c1),abs(params.R-params.c2))
    first = maximum_reward*own_first_bound+margin_slope*root_i
    second = maximum_reward*own_second_bound+2*margin_slope*own_first_bound+margin_slope*demand_second_bound
    # Round outward by a margin much larger than binary64 rounding here.
    first = float(np.nextafter(first*(1+1e-12),np.inf))
    second = float(np.nextafter(second*(1+1e-12),np.inf))
    return dict(first_derivative_bound=first,second_derivative_bound=second,
                score_information_bound=information,score_abs_third_cumulant_bound=third,
                score_fourth_moment_bound=fourth,
                derivation="Beta-score moments, Cauchy-Schwarz and CDF-indicator covariance bounds")


def _conditional_continuous_exclusion(params,p0,grid,fine,orders,quadrature_changes):
    """Cover every cell using analytic derivative envelopes and error budgets.

    The implication is mathematical conditional on the stated numerical error
    budgets. Those budgets use conservative quadrature convergence and a floor;
    they are not certified quadrature-error bounds.
    """
    mid = (grid[:-1]+grid[1:])/2
    high = stationary_conditions(params,p0,mid,order=orders[1])
    low = stationary_conditions(params,p0,mid,order=orders[0])
    observed = {name:max(quadrature_changes[name],float(np.max(abs(high[name]-low[name]))))
                for name in ("residual","derivative")}
    errors = dict(residual=max(1e-9,100*observed["residual"]),
                  derivative=max(1e-7,100*observed["derivative"]))
    bounds = analytic_global_derivative_bounds(params)
    half_widths = np.diff(grid)/2
    value_margin = abs(high["residual"])-bounds["first_derivative_bound"]*half_widths-errors["residual"]
    slope_margin = abs(high["derivative"])-bounds["second_derivative_bound"]*half_widths-errors["derivative"]
    excluded = value_margin > 0
    monotone = slope_margin > 0
    uncovered = ~(excluded|monotone)
    blocks = []
    indices = np.flatnonzero(~excluded)
    if len(indices):
        groups = np.split(indices,np.where(np.diff(indices)>1)[0]+1)
        for group in groups:
            start,stop = int(group[0]),int(group[-1])
            slopes = high["derivative"][group]
            fixed_sign = bool(np.all(slopes>0) or np.all(slopes<0))
            increasing = bool(np.all(slopes>0))
            first,last = float(fine["residual"][start]),float(fine["residual"][stop+1])
            robust_endpoints = abs(first)>errors["residual"] and abs(last)>errors["residual"]
            strict = bool(np.all(monotone[group]) and fixed_sign)
            count = int(first*last<0) if strict and robust_endpoints else None
            blocks.append(dict(lo=float(grid[start]),hi=float(grid[stop+1]),cell_count=len(group),
                               strictly_monotone_conditional_on_error_budget=strict,
                               increasing=increasing,left_residual=first,right_residual=last,
                               minimum_derivative_sign_margin=float(np.min(slope_margin[group])),
                               root_count_conditional_on_error_budget=count))
    valid = not np.any(uncovered) and all(row["root_count_conditional_on_error_budget"] is not None for row in blocks)
    return dict(analytic_envelopes=bounds,evaluation_error_budgets=errors,
                error_budget_basis="max(floor, 100 times observed order-doubling discrepancy) at all nodes and midpoints",
                quadrature_error_budgets_rigorously_validated=False,
                cell_count=len(mid),cells_excluded_by_residual=int(np.sum(excluded)),
                remaining_cells_with_strict_derivative=int(np.sum(~excluded&monotone)),
                unresolved_cells=int(np.sum(uncovered)),root_blocks=blocks,
                all_cells_covered_conditional_on_error_budgets=bool(valid),
                total_interior_roots_conditional_on_error_budgets=sum(row["root_count_conditional_on_error_budget"] for row in blocks) if valid else None,
                statement="Continuous root count follows from the analytic envelopes if the stated numerical evaluation-error budgets hold.")


def enumerate_stationary_candidates(params,p0,*,scan_points=4097,orders=(256,512),
                                   zero_tolerance=1e-10,return_samples=True):
    """Re-enumerate endpoints and all numerically detected interior F zeros.

    The fine scan doubles the number of intervals and raises quadrature order.
    Quadrature differences are checked on every fine node. Extrema and minima
    of |F| are searched independently, including candidates without sign changes.
    Derivative and curvature diagnostics are sampled numerical evidence, not
    certified global bounds between samples.
    """
    if scan_points < 33 or len(orders) != 2 or orders[1] <= orders[0]:
        raise ValueError("Use >=33 scan nodes and two increasing quadrature orders")
    coarse_grid = np.linspace(params.p1,params.p2,int(scan_points))
    grid = np.linspace(params.p1,params.p2,2*int(scan_points)-1)
    coarse,coarse_roots,coarse_extrema = _analyse_scan(params,p0,coarse_grid,orders[0],zero_tolerance)
    fine,roots,extrema = _analyse_scan(params,p0,grid,orders[1],zero_tolerance)
    low_on_fine = stationary_conditions(params,p0,grid,order=orders[0])
    differences = {name:float(np.max(abs(fine[name]-low_on_fine[name]))) for name in ("residual","derivative")}
    root_changes = [abs(a["z"]-b["z"]) for a,b in zip(coarse_roots,roots)]
    same_roots = len(roots) == len(coarse_roots)
    same_extrema = len(extrema) == len(coarse_extrema)
    resolution = float(grid[1]-grid[0])
    sampled_curvature = np.gradient(fine["derivative"],grid,edge_order=2)
    segments = []
    splits = [params.p1,*[row["z"] for row in extrema],params.p2]
    for left,right in zip(splits,splits[1:]):
        selected = (grid > left+2*resolution)&(grid < right-2*resolution)
        slopes = fine["derivative"][selected]
        segments.append(dict(lo=float(left),hi=float(right),
                             left_residual=stationary_conditions(params,p0,left,order=orders[1])["residual"],
                             right_residual=stationary_conditions(params,p0,right,order=orders[1])["residual"],
                             sampled_derivative_sign_consistent=bool(len(slopes) and (np.all(slopes>0) or np.all(slopes<0))),
                             min_abs_sampled_derivative_away_extrema=float(np.min(abs(slopes))) if len(slopes) else None))
    away = np.ones(len(grid),dtype=bool)
    for row in roots:
        away &= abs(grid-row["z"]) > 2*resolution
    margin = float(np.min(abs(fine["residual"][away]))) if np.any(away) else None
    output = dict(p0=float(p0),rho=float(params.rho),domain=[float(params.p1),float(params.p2)],
                  method="Independent Gauss-Legendre Beta-score residual and analytic diagonal derivative",
                  certification="Converged numerical root-completeness audit; no interval-arithmetic existence proof",
                  coarse_scan_points=int(scan_points),fine_scan_points=len(grid),quadrature_orders=list(orders),
                  interior_roots=roots,critical_points=extrema,monotonic_segments=segments,
                  root_count_stable=same_roots,critical_count_stable=same_extrema,
                  max_root_change_under_refinement=max(root_changes,default=0.) if same_roots else None,
                  quadrature_max_change=differences,
                  smallest_sampled_abs_residual_away_root_neighborhoods=margin,
                  maximum_sampled_abs_curvature=float(np.max(abs(sampled_curvature))),
                  estimated_linear_interpolation_remainder=float(np.max(abs(sampled_curvature))*resolution*resolution/8),
                  curvature_is_rigorous_bound=False,
                  pure_candidates=[dict(z=float(params.p1),kind="pure_product1"),dict(z=float(params.p2),kind="pure_product2")],
                  numerical_passed=bool(same_roots and same_extrema
                                        and max(root_changes,default=0.)<1e-7
                                        and differences["residual"]<1e-7
                                        and differences["derivative"]<1e-5
                                        and all(row["sampled_derivative_sign_consistent"] for row in segments)))
    if return_samples:
        output["scan"] = dict(z=grid.tolist(),residual=fine["residual"].tolist(),
                              derivative=fine["derivative"].tolist())
    output["conditional_continuous_exclusion"] = _conditional_continuous_exclusion(
        params,p0,grid,fine,orders,differences)
    coverage = output["conditional_continuous_exclusion"]
    output["numerical_passed"] = bool(output["numerical_passed"]
                                      and coverage["all_cells_covered_conditional_on_error_budgets"]
                                      and coverage["total_interior_roots_conditional_on_error_budgets"] == len(roots))
    return output
