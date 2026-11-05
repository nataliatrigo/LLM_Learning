"""Global scalar searches for the constant-mixture average-profit game.

Only q(u), the Beta posterior, demand and margin enter these objectives.
Discounting, transition speeds, HJBs and initial conditions do not enter.
Finite global exploration plus local refinement is a numerical search, not an
analytic exhaustiveness certificate.
"""
from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import betainc, betaincc, betaln

from oblivious.oe_p0.code.primitives import posterior
from oblivious.fluid_oblivious.code.model import (
    _legendre, FixedRivalDemand, symmetric_demand_derivative,
)


def _unique_candidates(candidates, tolerance):
    """Cluster numerical copies, retaining the lowest residual, never averaging."""
    groups = []
    for candidate in sorted(candidates, key=lambda item: item["u"]):
        if groups and abs(candidate["u"]-groups[-1][-1]["u"]) <= tolerance:
            groups[-1].append(candidate)
        else:
            groups.append([candidate])
    return [min(group, key=lambda item: item.get("residual", -item.get("value", 0.)))
            for group in groups]


def global_maxima(function, points=501, profit_atol=1e-12, profit_rtol=1e-9,
                  u_tolerance=1e-6, optimizer_xatol=1e-12, sampled_values=None):
    """Scan [0,1], refine every sampled local peak, and compare both endpoints.

    Does not assume unimodality. Tied detected global peaks are all returned.
    A completely flat objective is flagged as an interval, never averaged.
    """
    grid = np.linspace(0., 1., int(points))
    if len(grid) < 5:
        raise ValueError("Global exploration requires at least five points")
    values = np.asarray(function(grid) if sampled_values is None else sampled_values, dtype=float)
    if values.shape != grid.shape or not np.isfinite(values).all():
        raise ArithmeticError("Nonfinite or incorrectly shaped profit scan")
    # Prepared scan values locate peaks only. Reevaluate all reported values
    # with the scalar production function so scan quadrature cannot inflate g.
    candidates = [dict(u=float(grid[i]), value=float(function(grid[i])), source="endpoint")
                  for i in (0, len(grid)-1)]
    peak = ((values[1:-1] >= values[:-2]) & (values[1:-1] >= values[2:]) &
            ((values[1:-1] > values[:-2]) | (values[1:-1] > values[2:])))
    for i in np.flatnonzero(peak)+1:
        fit = minimize_scalar(lambda u: -float(function(u)),
                              bounds=(grid[i-1], grid[i+1]), method="bounded",
                              options={"xatol": optimizer_xatol})
        # Keep the scan point if refinement fails to improve it.
        u, value = float(fit.x), float(-fit.fun)
        sample_value = float(function(grid[i]))
        if value < sample_value:
            u, value = float(grid[i]), sample_value
        candidates.append(dict(u=u, value=value, source="refined_local_peak"))
    if np.ptp(values) == 0:
        candidates[0]["flat_interval"] = [0., 1.]
    candidates = _unique_candidates(candidates, u_tolerance)
    maximum = max(row["value"] for row in candidates)
    scale = max(abs(maximum), float(np.max(np.abs(values))))
    tolerance = float(profit_atol+profit_rtol*scale)
    maxima = [row for row in candidates if maximum-row["value"] <= tolerance]
    return dict(maximizers=maxima, max_value=maximum, candidates=candidates,
                profit_scale=scale, profit_tolerance=tolerance, scan_points=len(grid))


class AverageProfitModel:
    def __init__(self, params, p0, order=128):
        if not 0 <= p0 < 1:
            raise ValueError("Outside option must be in [0,1)")
        self.params, self.p0, self.order = params, float(p0), int(order)
        self._scalar_engine = lru_cache(maxsize=256)(
            lambda q: FixedRivalDemand(self.params, self.p0, np.array([q]), self.order))

    def q(self, u):
        return self.params.p1+(self.params.p2-self.params.p1)*np.asarray(u)

    def margin(self, u):
        return self.params.R-self.params.c1-(self.params.c2-self.params.c1)*np.asarray(u)

    def monopoly_demand(self, u):
        a, b = posterior(self.q(u), self.params)
        return betaincc(a, b, self.p0)

    def competition_demand(self, u, rival_u):
        # Reuse the original endpoint-aware Jacobi integration. Ordinary
        # untransformed Legendre quadrature can create a tiny jump next to the
        # exact diagonal identity, which would contaminate unilateral gains.
        own, rival = np.broadcast_arrays(self.q(u), self.q(rival_u))
        if own.ndim == 0:
            return float(self._scalar_engine(float(own))(float(rival))[0])
        result = np.empty_like(own, dtype=float)
        for value in np.unique(rival):
            mask = rival == value
            selected = own[mask]
            equal = selected == value
            demand = np.empty_like(selected)
            if equal.any():
                a,b = posterior(value, self.params)
                demand[equal] = .5*betaincc(a,b,self.p0)*(1+betainc(a,b,self.p0))
            if (~equal).any():
                demand[~equal] = FixedRivalDemand(self.params, self.p0, selected[~equal], self.order)(float(value))
            result[mask] = demand
        return result

    def monopoly_profit(self, u):
        return self.monopoly_demand(u)*self.margin(u)

    def competition_profit(self, u, rival_u):
        return self.competition_demand(u, rival_u)*self.margin(u)

    def symmetric_foc(self, u):
        """Partial own derivative of profit at (u,u), keeping rival fixed."""
        d, dx = symmetric_demand_derivative(self.params, self.p0, float(self.q(u)), self.order)
        return float((self.params.p2-self.params.p1)*dx*self.margin(u)
                     -(self.params.c2-self.params.c1)*d)


class PreparedCompetition:
    """Endpoint-regularized positive quadrature for inexpensive full BR scans.

    Uses t=1-(1-p0)*s**4 in the existing Beta-demand integral, precomputing
    own PDFs and the Jacobian. This smooths the fractional endpoint power.
    Scalar peaks/endpoints use the existing Jacobi FixedRivalDemand directly.
    Independent integration audits all final candidates and deviations.
    """
    def __init__(self, model, points):
        self.model = model
        self.grid = np.linspace(0., 1., int(points))
        nodes, weights = _legendre(model.order)
        s = (nodes+1)/2
        tail = (1-model.p0)*s**4
        self.t = 1-tail
        a, b = posterior(model.q(self.grid), model.params)
        log_jacobian = np.log(4*(1-model.p0))+3*np.log(s)
        self.weights = np.exp((a[:, None]-1)*np.log1p(-tail)
                             +(b[:, None]-1)*np.log(tail)
                             -betaln(a, b)[:, None]+log_jacobian)*weights/2
        self.margin = model.margin(self.grid)

    def profits(self, rival_u):
        a, b = posterior(self.model.q(rival_u), self.model.params)
        return (self.weights @ betainc(a, b, self.t))*self.margin


def _settings(settings):
    keys = ("own_points", "search_points", "quadrature_order", "profit_atol",
            "profit_rtol", "u_tolerance", "optimizer_xatol")
    return {name: settings[name] for name in keys}


def find_competitive_equilibria(model, settings):
    """Explore g itself, including minima without sign changes and endpoints.

    Own-FOC roots supplement, rather than replace, global deviation tests.
    Every best response searches the entire own-action interval [0,1].
    """
    s = _settings(settings)
    prepared = PreparedCompetition(model, s["own_points"])
    kwargs = {key: s[key] for key in ("profit_atol", "profit_rtol", "u_tolerance", "optimizer_xatol")}
    cache = {}

    def response(u):
        u = float(u)
        if u not in cache:
            br = global_maxima(lambda v: model.competition_profit(v, u),
                               points=s["own_points"], sampled_values=prepared.profits(u), **kwargs)
            current = float(model.competition_profit(u, u))
            # Feasibility of v=u makes the true gain nonnegative. Include it
            # explicitly to avoid a small negative numerical search residual.
            br["max_value"] = max(br["max_value"], current)
            br.update(u=u, symmetric_profit=current,
                      deviation_gain=max(0., br["max_value"]-current))
            cache[u] = br
        return cache[u]

    def gain(u):
        return response(u)["deviation_gain"]

    scan = np.linspace(0., 1., s["search_points"])
    gains = np.array([gain(u) for u in scan])
    candidates = [dict(u=0., source="endpoint"), dict(u=1., source="endpoint")]
    minima = ((gains[1:-1] <= gains[:-2]) & (gains[1:-1] <= gains[2:]) &
              ((gains[1:-1] < gains[:-2]) | (gains[1:-1] < gains[2:])))
    for i in np.flatnonzero(minima)+1:
        fit = minimize_scalar(gain, bounds=(scan[i-1], scan[i+1]), method="bounded",
                              options={"xatol": s["optimizer_xatol"]})
        u = float(fit.x) if gain(fit.x) <= gains[i] else float(scan[i])
        candidates.append(dict(u=u, source="refined_gain_minimum"))
    # Retain the best point even if the scan is flat or the lowest residual
    # lies on an unresolved plateau. This also makes failure diagnostics honest.
    candidates.append(dict(u=float(scan[np.argmin(gains)]), source="scan_minimum"))
    foc = np.array([model.symmetric_foc(u) for u in scan])
    foc_seeds = []
    for lo, hi, left, right in zip(scan[:-1], scan[1:], foc[:-1], foc[1:]):
        if left == 0:
            foc_seeds.append(float(lo))
        if left*right < 0:
            foc_seeds.append(float(brentq(model.symmetric_foc, lo, hi, xtol=s["optimizer_xatol"])))
    for i in range(1, len(scan)-1):
        if abs(foc[i]) < abs(foc[i-1]) and abs(foc[i]) < abs(foc[i+1]):
            fit = minimize_scalar(lambda u: abs(model.symmetric_foc(u)),
                                  bounds=(scan[i-1], scan[i+1]), method="bounded",
                                  options={"xatol": s["optimizer_xatol"]})
            if abs(model.symmetric_foc(fit.x)) < 1e-10*max(1., np.max(abs(foc))):
                foc_seeds.append(float(fit.x))
    for u in foc_seeds:
        candidates.append(dict(u=u, source="own_foc_root"))
    # Polish numerical gain minima onto nearby own-FOC roots when available.
    for candidate in candidates:
        if candidate["source"] in {"refined_gain_minimum", "scan_minimum"} and foc_seeds:
            nearest = min(foc_seeds, key=lambda u: abs(u-candidate["u"]))
            if abs(nearest-candidate["u"]) < 2/(len(scan)-1):
                candidate["u"] = nearest
    audited = []
    for candidate in candidates:
        u = candidate["u"]
        br = response(u)
        audited.append(dict(**candidate, residual=br["deviation_gain"],
                            deviation_gain=br["deviation_gain"],
                            profit_tolerance=br["profit_tolerance"],
                            profit_scale=br["profit_scale"],
                            best_response_profit=br["max_value"],
                            symmetric_profit=br["symmetric_profit"],
                            best_responses=br["maximizers"],
                            verified=br["deviation_gain"] <= br["profit_tolerance"]))
    audited = _unique_candidates(audited, s["u_tolerance"])
    minimum = min(audited, key=lambda row: row["deviation_gain"])
    accepted = [row for row in audited if row["verified"]]
    return dict(equilibria=accepted, candidates=audited,
                minimum_residual=minimum["deviation_gain"],
                minimum_residual_u=minimum["u"],
                status="verified" if accepted else "no_verified_equilibrium_detected",
                scan=[dict(u=float(u), deviation_gain=float(g), own_foc=float(f))
                      for u, g, f in zip(scan, gains, foc)],
                best_response_searches=len(cache))


def solution_row(model, market, u, index, gain, tolerance, scale, best_responses):
    demand = float(model.monopoly_demand(u) if market == "monopoly"
                   else model.competition_demand(u, u))
    sellers = 1 if market == "monopoly" else 2
    return dict(model=market, p0=model.p0, rho=model.params.rho,
                solution_id=f"{market}_p0_{model.p0:g}_rho_{model.params.rho:g}_s{index:02d}",
                solution_index=index, u=float(u), reputation=float(model.q(u)),
                demand=demand, average_profit=float(demand*model.margin(u)),
                product2_per_seller=float(demand*u), product2_market=float(sellers*demand*u),
                seller_count=sellers, outside_demand=float(1-sellers*demand),
                max_deviation_gain=float(gain), profit_tolerance=float(tolerance),
                profit_scale=float(scale), u_tolerance=1e-6, verified=True,
                best_response_us=[float(row["u"]) for row in best_responses])


def solve_case(params, p0, settings):
    model = AverageProfitModel(params, p0, settings["quadrature_order"])
    kwargs = {key: settings[key] for key in ("profit_atol", "profit_rtol", "u_tolerance", "optimizer_xatol")}
    monopoly = global_maxima(model.monopoly_profit, points=settings["own_points"], **kwargs)
    competition = find_competitive_equilibria(model, settings)
    solutions = []
    for i, row in enumerate(monopoly["maximizers"]):
        solutions.append(solution_row(model, "monopoly", row["u"], i,
                                      monopoly["max_value"]-row["value"],
                                      monopoly["profit_tolerance"], monopoly["profit_scale"],
                                      monopoly["maximizers"]))
    for i, row in enumerate(competition["equilibria"]):
        solutions.append(solution_row(model, "competition", row["u"], i,
                                      row["deviation_gain"], row["profit_tolerance"],
                                      row["profit_scale"], row["best_responses"]))
    for row in solutions:
        row["u_tolerance"] = settings["u_tolerance"]
    return dict(p0=float(p0), rho=params.rho, solutions=solutions,
                monopoly=monopoly, competition=competition, settings=_settings(settings))
