"""Feasible-deviation checks independent of the stationary HJB solver.

The rival is frozen at the optimization conjecture z. Starting at x=z, using
either pure product forever is an admissible departure from stationary
holding. A strictly higher discounted payoff rejects holding, regardless of
how the globally optimal HJB policy was represented at a threshold.

Demand is integrated directly with positive Gauss-Legendre quadrature. No
prepared Jacobi engine, policy checkpoint, FOC candidate generator or Howard
solver is used here. Truncation bounds are analytic; ODE/quadrature refinement
indicators are numerical and are not validated interval-arithmetic bounds.
"""
from __future__ import annotations

from functools import lru_cache
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import betainc, betaincc, betaln, logsumexp, roots_legendre


@lru_cache(maxsize=8)
def _legendre(order):
    return roots_legendre(int(order))


class FixedRivalLegendreDemand:
    """Direct evaluation of integral[p0,1] f_x(t) F_z(t) dt."""

    def __init__(self, params, p0, conjecture, order=128):
        if not 0 < params.rho < 1 or not 0 <= p0 < 1 or not 0 <= conjecture <= 1:
            raise ValueError("Invalid retention, outside option or conjecture")
        if int(order) != order or order < 16:
            raise ValueError("Use an integer quadrature order >=16")
        self.params, self.p0, self.conjecture = params, float(p0), float(conjecture)
        self.order = int(order)
        nodes, weights = _legendre(self.order)
        t = self.p0+(1-self.p0)*(nodes+1)/2
        concentration = 1/(1-params.rho)
        rival_a = 1+self.conjecture*concentration
        rival_b = 1+(1-self.conjecture)*concentration
        with np.errstate(divide="ignore"):
            self.log_weights = np.log(weights*(1-self.p0)/2)+np.log(betainc(rival_a,rival_b,t))
        self.log_t, self.log_one_minus_t = np.log(t), np.log1p(-t)

    def __call__(self, own):
        x = np.asarray(own, dtype=float)
        if np.any(~np.isfinite(x)) or np.any((x < -1e-10)|(x > 1+1e-10)):
            raise ValueError("Own reputation must lie in [0,1]")
        x = np.clip(x, 0., 1.)
        concentration = 1/(1-self.params.rho)
        a, b = 1+x*concentration, 1+(1-x)*concentration
        log_pdf = ((a[...,None]-1)*self.log_t+(b[...,None]-1)*self.log_one_minus_t
                   -betaln(a,b)[...,None])
        result = np.exp(logsumexp(self.log_weights+log_pdf,axis=-1))
        if np.any(~np.isfinite(result)) or np.any((result < 0)|(result > 1+1e-11)):
            raise ArithmeticError("Demand quadrature violated probability bounds")
        return float(result) if result.ndim == 0 else result


def stationary_holding_value(params, p0, conjecture):
    """Payoff of the exact zero-drift mixture at x=z against rival z."""
    z = float(conjecture)
    if not params.p1 <= z <= params.p2 or not 0 <= p0 < 1:
        raise ValueError("Stationary holding requires p1 <= conjecture <= p2")
    if not 0 < params.gamma < 1:
        raise ValueError("Discount factor must lie in (0,1)")
    alpha = (z-params.p1)/(params.p2-params.p1)
    a, b = 1+z/(1-params.rho), 1+(1-z)/(1-params.rho)
    cdf, survival = float(betainc(a,b,p0)), float(betaincc(a,b,p0))
    demand = .5*survival*(1+cdf)
    reward = params.R-params.c1-(params.c2-params.c1)*alpha
    return dict(alpha=alpha, demand=demand, reward=reward,
                value=demand*reward/(-math.log(params.gamma)))


def constant_action_deviation(params, p0, conjecture, product, *, order=128,
                              rtol=1e-10, atol=1e-12, max_step=10.,
                              tail_tolerance=1e-10, horizon=None,
                              demand_function=None):
    """Integrate a feasible pure-product policy from x(0)=conjecture.

    With nonnegative margins, the integrated finite-horizon payoff itself is
    a lower bound on the infinite-horizon payoff, up to numerical integration
    error. The analytic remaining tail is <= r_i exp(-delta*T)/delta because
    seller demand is <=1. ``demand_function`` permits independent exact tests.
    """
    if product not in (1,2):
        raise ValueError("product must be 1 or 2")
    if (not 0 < params.rho < 1 or not 0 < params.gamma < 1
            or rtol <= 0 or atol <= 0 or max_step <= 0 or tail_tolerance <= 0):
        raise ValueError("Invalid rates or integration settings")
    reward = float(params.R-(params.c1 if product == 1 else params.c2))
    if reward < 0:
        raise ValueError("This payoff lower-bound check requires nonnegative margins")
    target = float(params.p1 if product == 1 else params.p2)
    lam, delta = -math.log(params.rho), -math.log(params.gamma)
    z = float(conjecture)
    if not 0 <= z <= 1:
        raise ValueError("Conjecture must lie in [0,1]")
    if horizon is None:
        horizon = max(1., math.log(max(reward/delta,tail_tolerance)/tail_tolerance)/delta)
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("Horizon must be positive and finite")
    demand = FixedRivalLegendreDemand(params,p0,z,order) if demand_function is None else demand_function

    def derivative(time, state):
        d = float(demand(float(state[0])))
        if not math.isfinite(d) or not 0 <= d <= 1+1e-10:
            raise ArithmeticError("A seller's selection probability must lie in [0,1]")
        return [lam*d*(target-state[0]), math.exp(-delta*time)*d*reward]

    result = solve_ivp(derivative,(0.,float(horizon)),[z,0.],method="DOP853",
                       rtol=rtol,atol=atol,max_step=max_step)
    if not result.success:
        raise ArithmeticError(result.message)
    finite = float(result.y[1,-1])
    tail = reward*math.exp(-delta*horizon)/delta
    return dict(product=int(product), conjecture=z, target_reputation=target,
                finite_horizon_value=finite, terminal_reputation=float(result.y[0,-1]),
                horizon=float(horizon), tail_upper_bound=float(tail),
                infinite_value_interval_from_truncation=[finite,finite+tail],
                quadrature_order=int(order), ode_rtol=rtol, ode_atol=atol,
                function_evaluations=result.nfev,
                method="Independent direct Legendre demand and DOP853; product fixed forever")


def excursion_then_hold_deviation(params, p0, conjecture, target, *, order=128,
                                  rtol=1e-10, atol=1e-12, max_step=10.,
                                  tail_tolerance=1e-10, horizon=None,
                                  demand_function=None):
    """Move toward ``target`` with a pure action, then hold with a mixture.

    This is an explicit admissible policy with no HJB dependency. If the target
    is not reached by the horizon, it switches to holding at the reached state;
    its continuation value is still evaluated exactly for that chosen state.
    """
    z, target = float(conjecture), float(target)
    if not params.p1 <= min(z,target) <= max(z,target) <= params.p2:
        raise ValueError("An excursion with holding must remain inside [p1,p2]")
    if (not 0 < params.rho < 1 or not 0 < params.gamma < 1
            or rtol <= 0 or atol <= 0 or max_step <= 0 or tail_tolerance <= 0):
        raise ValueError("Invalid rates or integration settings")
    lam, delta = -math.log(params.rho), -math.log(params.gamma)
    if horizon is None:
        horizon = max(1.,math.log(max((params.R-params.c1)/delta,tail_tolerance)/tail_tolerance)/delta)
    if not math.isfinite(horizon) or horizon <= 0:
        raise ValueError("Horizon must be positive and finite")
    demand = FixedRivalLegendreDemand(params,p0,z,order) if demand_function is None else demand_function

    def holding(x):
        alpha = (x-params.p1)/(params.p2-params.p1)
        reward = params.R-params.c1-(params.c2-params.c1)*alpha
        return float(demand(x))*reward/delta

    if target == z:
        return dict(deviation_kind="excursion_then_hold",conjecture=z,requested_target=target,
                    reached_state=z,target_reached=True,travel_time=0.,finite_horizon_value=0.,
                    continuation_value=holding(z),value=holding(z),quadrature_order=int(order),
                    ode_rtol=rtol,ode_atol=atol,function_evaluations=0)
    product = 2 if target > z else 1
    quality = params.p2 if product == 2 else params.p1
    reward = params.R-(params.c2 if product == 2 else params.c1)

    def derivative(time,state):
        d = float(demand(float(state[0])))
        return [lam*d*(quality-state[0]),math.exp(-delta*time)*d*reward]

    def arrival(_time,state):
        return state[0]-target

    arrival.terminal = True
    arrival.direction = 1 if target > z else -1
    result=solve_ivp(derivative,(0.,float(horizon)),[z,0.],method="DOP853",events=arrival,
                     rtol=rtol,atol=atol,max_step=max_step)
    if not result.success:
        raise ArithmeticError(result.message)
    reached=float(np.clip(result.y[0,-1],params.p1,params.p2))
    time=float(result.t[-1])
    terminal=holding(reached)
    finite=float(result.y[1,-1])
    continuation=math.exp(-delta*time)*terminal
    return dict(deviation_kind="excursion_then_hold",product=product,conjecture=z,
                requested_target=target,reached_state=reached,target_reached=bool(len(result.t_events[0])),
                travel_time=time,finite_horizon_value=finite,continuation_value=continuation,
                value=finite+continuation,terminal_holding_value=terminal,
                quadrature_order=int(order),ode_rtol=rtol,ode_atol=atol,
                function_evaluations=result.nfev,
                method="Independent direct Legendre demand and DOP853 travel, then exact holding payoff")


def audit_stationary_candidate(params, p0, conjecture, *, order=128,
                               tail_tolerance=1e-10, horizon=None,
                               excursion_targets=None):
    """Check pure departures with quadrature and ODE refinement.

    A detected profitable departure rules out the holding candidate. Failure
    to detect one is not a proof that holding is globally optimal.
    """
    holding = stationary_holding_value(params,p0,conjecture)
    departures = []

    def record(coarse,fine,payoff_key):
        refinement=abs(fine[payoff_key]-coarse[payoff_key])
        lower_gain=min(coarse[payoff_key],fine[payoff_key])-holding["value"]
        safety_margin=max(1e-9,20*refinement)
        departures.append(dict(fine,holding_value=holding["value"],
                               finite_horizon_gain=fine[payoff_key]-holding["value"],
                               payoff_gain=fine[payoff_key]-holding["value"],
                               conservative_observed_gain=lower_gain-safety_margin,
                               refinement_change=refinement,numerical_safety_margin=safety_margin,
                               profitable_departure_detected=bool(lower_gain>safety_margin)))

    for product in (1,2):
        coarse = constant_action_deviation(params,p0,conjecture,product,order=order,
                                          rtol=1e-9,atol=1e-11,max_step=10.,
                                          tail_tolerance=tail_tolerance,horizon=horizon)
        fine = constant_action_deviation(params,p0,conjecture,product,order=2*order,
                                        rtol=1e-11,atol=1e-13,max_step=5.,
                                        tail_tolerance=tail_tolerance,horizon=coarse["horizon"])
        coarse["deviation_kind"]=fine["deviation_kind"]="constant_product"
        record(coarse,fine,"finite_horizon_value")
    if excursion_targets is None:
        z=float(conjecture)
        excursion_targets=sorted({float(np.clip(z+offset,params.p1,params.p2))
                                  for offset in (-.1,-.05,-.01,.01,.05,.1)}-{z})
    for target in excursion_targets:
        coarse=excursion_then_hold_deviation(params,p0,conjecture,target,order=order,
                                             rtol=1e-9,atol=1e-11,max_step=10.,
                                             tail_tolerance=tail_tolerance,horizon=horizon)
        fine=excursion_then_hold_deviation(params,p0,conjecture,target,order=2*order,
                                           rtol=1e-11,atol=1e-13,max_step=5.,
                                           tail_tolerance=tail_tolerance,horizon=horizon)
        record(coarse,fine,"value")
    return dict(p0=float(p0),rho=float(params.rho),conjecture=float(conjecture),
                holding=holding,departures=departures,
                rejected_by_feasible_departure=any(row["profitable_departure_detected"] for row in departures),
                scope="A positive refined departure gain rejects holding independently of the HJB; "
                      "nonpositive gains do not certify optimality, and refinement indicators are not interval bounds.")
