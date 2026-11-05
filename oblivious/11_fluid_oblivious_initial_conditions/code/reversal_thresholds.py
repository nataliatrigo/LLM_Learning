"""Matched-grid checks of monopoly-high/competition-low initial conditions.

This module does not alter the historical experiment 10 results.  It reuses
its upwind HJB and positive quadrature, but prepares only quadrature rows
that the fixed-rival demand calculation actually uses.  In particular, when
the own state exceeds the rival conjecture, one rival quadrature row replaces
all the otherwise unused own rows.  Memory is bounded by ``chunk_size*order``.

Thresholds are interpolated finite-grid policy switches.  Small HJB or
quadrature residuals do not certify their spatial discretization errors.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.linalg import solve_banded
from scipy.special import betainc, betaincc, logsumexp

from forgetting.Code.fluid_convergence.numerics import solve_fluid
from forgetting.Code.model import boundary_demand
from oblivious.fluid_oblivious.code.model import (
    _policy_description,
    _primitives,
    _row_quadrature,
)
from oblivious.oe_p0.code.primitives import posterior


def fixed_rival_demand_grid(params, p0, conjecture, grid, order=128, *, chunk_size=2048):
    """Reproduce ``FixedRivalDemand(...)(conjecture)`` without unused rows.

    The event partition is the same as in experiment 10: integrate the
    smaller seller's demand directly, then subtract it from total market
    demand to obtain the larger seller's demand.  This avoids subtracting
    almost equal numbers for the small seller's very small sales tail.
    """
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 1 or np.any((grid < 0) | (grid > 1)):
        raise ValueError("grid must be a one-dimensional array in [0,1]")
    if not 0 <= p0 < 1 or not 0 <= conjecture <= 1:
        raise ValueError("require p0 in [0,1) and conjecture in [0,1]")
    if int(order) != order or order < 8 or int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError("require integer order >= 8 and positive integer chunk_size")
    order, chunk_size = int(order), int(chunk_size)
    a, b = posterior(grid, params)
    c, d = posterior(float(conjecture), params)
    cdf, survival = betainc(a, b, p0), betaincc(a, b, p0)
    rival_cdf, rival_survival = betainc(c, d, p0), betaincc(c, d, p0)
    market = (survival + cdf*rival_survival + rival_survival + rival_cdf*survival)/2
    raw = np.empty_like(grid)

    smaller_indices = np.flatnonzero(grid < conjecture)
    for start in range(0, len(smaller_indices), chunk_size):
        indices = smaller_indices[start:start+chunk_size]
        rows = [_row_quadrature(x, params, p0, order) for x in grid[indices]]
        nodes = np.asarray([row[0] for row in rows])
        coefficients = np.asarray([row[1] for row in rows])
        with np.errstate(divide="ignore"):
            raw[indices] = np.exp(logsumexp(
                coefficients + np.log(betainc(c, d, nodes)), axis=1))

    larger_indices = np.flatnonzero(grid > conjecture)
    if len(larger_indices):
        nodes, coefficients = _row_quadrature(conjecture, params, p0, order)
        for start in range(0, len(larger_indices), chunk_size):
            indices = larger_indices[start:start+chunk_size]
            with np.errstate(divide="ignore"):
                smaller_demand = np.exp(logsumexp(
                    coefficients[None, :] + np.log(betainc(
                        a[indices, None], b[indices, None], nodes)), axis=1))
            raw[indices] = market[indices] - smaller_demand

    equal = grid == conjecture
    raw[equal] = market[equal]/2
    if np.any(raw < -1e-12) or np.any(raw > survival+1e-10):
        raise ArithmeticError("fixed-rival demand violated probability bounds")
    return np.maximum(raw, 0)


def repelling_threshold(params, response):
    """Return the unique separating switch between p1 and p2, or ``None``."""
    roots = [float(root) for root, kind in zip(response["roots"], response["root_kinds"])
             if kind == "repelling" and params.p1 < root < params.p2]
    if len(roots) > 1:
        raise ValueError("more than one separating threshold; inspect the full policy")
    return roots[0] if roots else None


def _solve_with_initial_policy(primitives, rho, grid, demand, previous, tol=1e-10):
    """The established upwind Howard equations, initialized from a prior grid.

    The operator, action tolerance, tridiagonal solve, stopping criterion and
    root interpolation are identical to ``solve_fluid``.  Only its all-product1
    initial policy is replaced by the interpolated previous Hamiltonian gap.
    This matters for large n because the default initial policy can require
    thousands of iterations to propagate the investment region across a grid.
    """
    n = len(grid)
    spacing = 1/(n-1)
    lam, delta = -math.log(rho), -math.log(primitives.gamma)
    rewards = np.asarray(primitives.net_rewards)[:, None]
    velocity = lam*(np.asarray([primitives.p1, primitives.p2])[:, None]-grid)
    drift = demand*velocity
    right, left = np.maximum(drift, 0)/spacing, np.maximum(-drift, 0)/spacing
    right[:, -1], left[:, 0] = 0, 0
    policy = (np.interp(grid, previous["x"], previous["gap"]) > tol*.5).astype(np.int8)
    indices = np.arange(n)
    for iteration in range(1, 20_001):
        rr, ll = right[policy, indices], left[policy, indices]
        banded = np.zeros((3, n))
        banded[0, 1:], banded[1], banded[2, :-1] = -rr[:-1], delta+rr+ll, -ll[1:]
        value = solve_banded((1, 1), banded, demand*rewards[policy, 0])
        forward = np.r_[np.diff(value)/spacing, 0.]
        backward = np.r_[0., np.diff(value)/spacing]
        slopes = np.where(velocity >= 0, forward, backward)
        hamiltonian = rewards+velocity*slopes
        gap = hamiltonian[1]-hamiltonian[0]
        residual = float(np.max(np.abs(delta*value-demand*hamiltonian.max(axis=0))))
        updated = np.where(gap > tol*.5, 1, np.where(gap < -tol*.5, 0, policy)).astype(np.int8)
        if np.array_equal(updated, policy):
            if residual > max(tol, 1e-11):
                raise RuntimeError(f"fluid Howard stopped with residual {residual:.3e}")
            break
        policy = updated
    else:
        raise RuntimeError("fluid Howard iteration did not converge")
    changes = np.flatnonzero(policy[1:] != policy[:-1])
    roots = []
    for index in changes:
        difference = gap[index+1]-gap[index]
        fraction = -gap[index]/difference if difference != 0 else .5
        roots.append(float(grid[index]+spacing*np.clip(fraction, 0, 1)))
    starts, ends = np.r_[0, changes+1], np.r_[changes, n-1]
    bands = [[float(grid[start]), float(grid[end])]
             for start, end in zip(starts, ends) if policy[start] == 1]
    conditional_regret = np.maximum(gap, 0)-policy*gap
    return dict(grid=grid, x=grid, value=value, gap=gap, policy=policy, demand=demand,
                roots=roots, bands=bands, residual=residual,
                value_error_bound=residual/delta,
                conditional_regret=float(np.max(conditional_regret)),
                iterations=iteration, **{"lambda": lam, "delta": delta})


def solve_threshold_pair(params, p0, n, order=128, *, check_order=False, chunk_size=2048,
                         initial_pair=None):
    """Solve monopoly and competition against p1 on exactly the same grid.

    Returns complete HJB responses in ``monopoly`` and ``competition``, their
    policy thresholds in ``threshold_monopoly`` and ``threshold_competition``,
    and numerical diagnostics.  The competition conjecture is ``params.p1``.
    If ``check_order`` is true, compare demand at ``order`` and ``2*order`` on
    257 states augmented with the two computed switches and p1,p2.  This is a
    quadrature check only; run multiple n values to assess spatial variation.
    An optional complete prior ``initial_pair`` initializes Howard from its
    policies, using the same finite-grid equations and stopping criterion.
    """
    if int(n) != n or n < 3:
        raise ValueError("n must be an integer >= 3")
    n = int(n)
    started = time.perf_counter()
    grid = np.linspace(0, 1, n)
    primitives = _primitives(params, p0)
    monopoly_demand = boundary_demand(grid, params.rho, p0)
    competition_demand = fixed_rival_demand_grid(
        params, p0, params.p1, grid, order, chunk_size=chunk_size)
    demand_seconds = time.perf_counter()-started
    if initial_pair is None:
        monopoly = solve_fluid(primitives, params.rho, n,
                               demand_function=lambda _: monopoly_demand)
        competition = solve_fluid(primitives, params.rho, n,
                                  demand_function=lambda _: competition_demand)
    else:
        monopoly = _solve_with_initial_policy(primitives, params.rho, grid, monopoly_demand,
                                              initial_pair["monopoly"])
        competition = _solve_with_initial_policy(primitives, params.rho, grid, competition_demand,
                                                 initial_pair["competition"])
    monopoly.update(_policy_description(params, monopoly))
    competition.update(_policy_description(params, competition))
    competition["conjecture"] = float(params.p1)
    threshold_monopoly = repelling_threshold(params, monopoly)
    threshold_competition = repelling_threshold(params, competition)
    diagnostics = dict(
        n=n, h=1/(n-1), order=int(order), conjecture=float(params.p1),
        hjb_residual_monopoly=float(monopoly["residual"]),
        hjb_residual_competition=float(competition["residual"]),
        demand_seconds=demand_seconds,
        max_demand_reduction=float(np.max(monopoly_demand-competition_demand)),
        quadrature_checked=False,
        warm_start=initial_pair is not None,
        iterations_monopoly=int(monopoly["iterations"]),
        iterations_competition=int(competition["iterations"]),
    )
    if threshold_monopoly is not None and threshold_competition is not None:
        diagnostics["interval_width"] = threshold_competition-threshold_monopoly
    if check_order:
        extra = [params.p1, params.p2]
        extra += [t for t in (threshold_monopoly, threshold_competition) if t is not None]
        check_grid = np.unique(np.r_[np.linspace(0, 1, 257), extra])
        original = fixed_rival_demand_grid(params, p0, params.p1, check_grid, order,
                                           chunk_size=chunk_size)
        refined = fixed_rival_demand_grid(params, p0, params.p1, check_grid, 2*order,
                                          chunk_size=chunk_size)
        error = np.abs(original-refined)
        diagnostics.update(
            quadrature_checked=True,
            quadrature_check_states=len(check_grid),
            quadrature_check_order=2*int(order),
            quadrature_max_absolute_difference=float(np.max(error)),
            quadrature_max_relative_difference=float(np.max(error/np.maximum(
                refined, np.finfo(float).tiny))),
        )
    diagnostics["elapsed_seconds"] = time.perf_counter()-started
    return dict(monopoly=monopoly, competition=competition,
                threshold_monopoly=threshold_monopoly,
                threshold_competition=threshold_competition,
                diagnostics=diagnostics)
