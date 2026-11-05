"""Marginal-distribution oblivious equilibrium with an external fixed p0.

There are TWO endogenous sellers and a known outside option. A consumer buys
from A iff its independent posterior draw exceeds both B's draw and p0.
Selecting the outside option produces zero focal profit and no observation:
both reputations stay fixed. Forgetting is indexed by observations, as in the
existing experiments. The numerical OE closes the unconditional rival marginal
with the invariant law of this new joint chain; it is not an MPE.
"""

from dataclasses import asdict

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar, root
from scipy.special import betainc, betaincc, betaln, expit, logsumexp, roots_jacobi, roots_legendre
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu

from oblivious.oe_distribution.code.model import JointMarginalExperiment, _JointResponse
from oblivious.oe_p0.code.core import checked_joint_stationary
from oblivious.oe_p0.code.game import Game
from oblivious.oe_p0.code.policy import evaluate_policy, solve_dp
from oblivious.oe_p0.code.primitives import (
    Parameters, check_grid, demand_matrix, posterior,
)


def _threshold(p0):
    p0 = float(p0)
    if not np.isfinite(p0) or not 0 <= p0 <= 1:
        raise ValueError("The external p0 must be a finite number in [0,1].")
    return p0


def _raw_outside_demand(grid, params, p0, order):
    """Direct positive quadrature; no event partition or symmetry repair.

    Map t=p0+(1-p0)(z+1)/2 and absorb (1-z)^(b-1) in the
    Gauss--Jacobi weight. The remaining integrand is smooth even for
    noninteger Beta shapes at t=1. Accumulate in logs so the representable
    ~1e-302 tails at rho=.999 survive normalization of the quadrature weights.
    """
    a, b = posterior(grid, params)
    raw = np.empty((len(grid), len(grid)))
    if params.rho >= .99:
        # At very concentrated posteriors the single-endpoint Jacobi weight
        # can place too few nodes near a rare pairwise integrand's interior
        # mode. Uniform Legendre nodes resolve all these narrow interior
        # peaks; order doubling still checks every rare representable tail.
        nodes, weights = roots_legendre(order)
        t = p0+(1-p0)*(nodes+1)/2
        with np.errstate(divide="ignore"):
            log_cdf = np.log(betainc(a[:, None], b[:, None], t[None, :]))
        log_weights = np.log(weights*(1-p0)/2)
        for i in range(len(grid)):
            log_pdf = ((a[i]-1)*np.log(t)+(b[i]-1)*np.log1p(-t)-betaln(a[i], b[i]))
            raw[i] = np.exp(logsumexp(log_weights[None, :]+log_pdf[None, :]+log_cdf, axis=1))
        return raw
    for i in range(len(grid)):
        nodes, weights = roots_jacobi(order, b[i]-1, 0)
        t = p0+(1-p0)*(nodes+1)/2
        with np.errstate(divide="ignore"):
            log_cdf = np.log(betainc(a[:, None], b[:, None], t[None, :]))
            log_weights = np.log(weights)
        constant = b[i]*(np.log1p(-p0)-np.log(2))-betaln(a[i], b[i])
        raw[i] = np.exp(constant+logsumexp(
            log_weights[None, :]+(a[i]-1)*np.log(t)[None, :]+log_cdf, axis=1))
    return raw


def outside_demand_matrix(grid, params, p0=.5, order=128):
    """Return D[x,y]=P(theta_x >= max(theta_y,p0)) and raw diagnostics.

    Small selection tails are integrated directly. The larger seller demand is
    recovered from the exact union mass; diagonal entries use its exact half.
    The union is S_x + F_x S_y, using beta survival probabilities directly,
    rather than subtracting F_x F_y from one. This preserves extremely rare
    sales when the outside option wins almost every period. Diagnostics record
    errors BEFORE these event identities are enforced and include order doubling.
    """
    grid, p0 = check_grid(grid), _threshold(p0)
    if int(order) != order or order < 8:
        raise ValueError("Use an integer quadrature order >=8.")
    order = int(order)
    a, b = posterior(grid, params)
    cdf, survival = betainc(a, b, p0), betaincc(a, b, p0)
    outside = np.outer(cdf, cdf)
    # Symmetrize two cancellation-free forms to retain the event symmetry.
    market = (survival[:, None]+cdf[:, None]*survival[None, :]
              +survival[None, :]+cdf[None, :]*survival[:, None])/2
    if p0 == 0:
        result, original = demand_matrix(grid, params, order)
        doubled, _ = demand_matrix(grid, params, 2*order)
        diagnostics = dict(original, p0=p0, quadrature="Original untruncated Gauss--Jacobi",
                           raw_partition_error=original["raw_symmetry_error"],
                           raw_relative_selected_mass_error=original["raw_symmetry_error"],
                           order_doubling_max_abs=float(np.max(abs(result-doubled))),
                           order_doubling_max_relative_tail=0.,
                           positive_tail_underflow_entries=int(np.sum(result == 0)))
        return result, diagnostics
    if p0 == 1:
        return np.zeros_like(outside), dict(
            order=order, p0=p0, quadrature="Exact endpoint", raw_partition_error=0.,
            raw_relative_selected_mass_error=0., order_doubling_max_abs=0.,
            order_doubling_max_relative_tail=0., positive_tail_underflow_entries=0)

    raw = _raw_outside_demand(grid, params, p0, order)
    doubled = _raw_outside_demand(grid, params, p0, 2*order)
    tiny = np.finfo(float).tiny
    raw_mass_error = raw+raw.T-market
    relative_mass_error = float(np.max(abs(raw_mass_error)/np.maximum(market, tiny)))
    doubling_absolute = float(np.max(abs(raw-doubled)))
    doubling_relative = float(np.max(abs(raw-doubled)/np.maximum(doubled, tiny)))
    diagnostics = dict(
        order=order, p0=p0,
        quadrature=("Truncated positive log-domain Gauss--Legendre" if params.rho >= .99
                    else "Truncated positive log-domain Gauss--Jacobi"),
        raw_partition_error=float(np.max(abs(raw+raw.T+outside-1))),
        raw_relative_selected_mass_error=relative_mass_error,
        raw_diagonal_error=float(np.max(abs(raw.diagonal()-market.diagonal()/2))),
        raw_monopoly_bound_violation=float(np.max(raw-survival[:, None], initial=0)),
        raw_min=float(raw.min()), raw_max=float(raw.max()),
        order_doubling_max_abs=doubling_absolute,
        order_doubling_max_relative_tail=doubling_relative,
        positive_tail_underflow_entries=int(np.sum(doubled == 0)),
        min_monopoly_sale_probability=float(survival.min()),
    )
    if (not np.all(np.isfinite(raw)) or raw.min() < 0 or raw.max() > 1+1e-9
            or relative_mass_error > 2e-7 or doubling_absolute > 2e-9
            or doubling_relative > 2e-6):
        raise ArithmeticError(f"Outside-option demand quadrature failed; increase order: {diagnostics}")
    result = np.empty_like(raw)
    upper = np.triu_indices(len(grid), 1)
    result[upper] = raw[upper]
    result[(upper[1], upper[0])] = market[upper]-raw[upper]
    np.fill_diagonal(result, market.diagonal()/2)
    if np.any(result < 0) or np.any(result > survival[:, None]+1e-10):
        raise ArithmeticError("Invalid outside-option selection probability after event partition.")
    diagnostics["partition_error"] = float(np.max(abs(result+result.T+outside-1)))
    diagnostics["min_positive_demand"] = float(result[result > 0].min(initial=1.))
    return result, diagnostics


def outside_demand_probability(x, y, params, p0=.5):
    """Independent scaled adaptive integration for auditing individual entries.

    Scaling at the log-integrand's mode prevents an absolute integration
    tolerance from treating representable rare sales as zero. The scaling and
    interval subdivision are independent of the production Jacobi quadrature.
    """
    p0 = _threshold(p0)
    a, b = posterior(float(x), params)
    c, d = posterior(float(y), params)
    if p0 == 1:
        return 0.

    def log_integrand(t):
        if not 0 < t < 1:
            return -np.inf
        cdf = betainc(c, d, t)
        if cdf == 0:
            return -np.inf
        return ((a-1)*np.log(t)+(b-1)*np.log1p(-t)-betaln(a, b)+np.log(cdf))

    optimum = minimize_scalar(lambda t: -log_integrand(t), bounds=(p0, 1),
                              method="bounded", options={"xatol": 1e-15})
    candidates = [float(optimum.x), max(np.nextafter(0., 1.), p0),
                  np.nextafter(1., 0.)]
    mode = max(candidates, key=log_integrand)
    scale = log_integrand(mode)
    if not np.isfinite(scale):
        return 0.
    points = [mode] if p0 < mode < 1 else None
    integral, error = quad(lambda t: np.exp(log_integrand(t)-scale), p0, 1,
                           epsabs=2e-12, epsrel=2e-11, limit=400, points=points)
    if error > max(1e-9*integral, 2e-11) or integral < 0:
        raise ArithmeticError(f"Independent outside demand audit failed: error={error:g}")
    return float(np.exp(scale+np.log(integral))) if integral > 0 else 0.


class OutsideOptionGame(Game):
    """Two endogenous sellers plus a fixed known outside option."""

    def __init__(self, grid, params, p0=.5, order=128):
        self.p0 = _threshold(p0)
        super().__init__(grid, params, order)
        self.D, self.demand_diagnostics = outside_demand_matrix(
            self.grid, params, self.p0, order)
        a, b = posterior(self.grid, params)
        self.outside = np.outer(betainc(a, b, self.p0), betainc(a, b, self.p0))
        self.selection = np.stack((self.D, self.D.T))

    def transition(self, sigma_a, sigma_b):
        return (super().transition(sigma_a, sigma_b)+diags(self.outside.ravel())).tocsr()

    def best_response(self, opponent, seller=0, tol=1e-10, initial=None):
        """Full-state diagnostic best response, including outside holding time."""
        opponent = self._policy(opponent)
        if seller not in (0, 1) or tol <= 0:
            raise ValueError("Seller must be 0 or 1, with positive tolerance")
        if seller == 1:
            result = self.best_response(opponent.T, seller=0, tol=tol,
                                        initial=None if initial is None else np.asarray(initial).T)
            return {**result, **{key: result[key].T for key in ("value", "sigma", "advantage")}}
        if initial is None:
            value = np.zeros(self.n**2)
        else:
            initial = np.asarray(initial, dtype=float)
            if initial.shape != (self.n, self.n) or np.any(~np.isfinite(initial)):
                raise ValueError("Initial best-response value must have shape (n,n)")
            value = initial.ravel().copy()
        p, demand = self.params, self.D.ravel()
        rival_probability = p.p1+(p.p2-p.p1)*opponent.ravel()
        rival_demand = self.selection[1].ravel()
        holding_and_rival = (diags(self.outside.ravel())
                            +diags(rival_demand*rival_probability)@self.rival_success
                            +diags(rival_demand*(1-rival_probability))@self.rival_failure)
        for iterations in range(1, 100001):
            vs, vf = self.own_success@value, self.own_failure@value
            advantage = -(p.c2-p.c1)+p.gamma*(p.p2-p.p1)*(vs-vf)
            q1 = p.R-p.c1+p.gamma*(p.p1*vs+(1-p.p1)*vf)
            updated = demand*(q1+np.maximum(advantage, 0))+p.gamma*(holding_and_rival@value)
            residual = float(np.max(abs(updated-value)))
            if residual <= tol:
                return dict(value=value.reshape(self.n, self.n),
                            sigma=(advantage >= 0).astype(float).reshape(self.n, self.n),
                            advantage=advantage.reshape(self.n, self.n),
                            residual=residual, iterations=iterations)
            value = updated
        raise RuntimeError(f"Outside-option best response did not converge; residual={residual:g}")


class _OutsideResponse(_JointResponse):
    """Joint response with ratio-first sensitivities for exceptionally rare sales.

    Outside demand can make calendar exit rates ~1e-304. Squaring such rates
    underflows even when the final derivative is representable. The algebra
    below never squares a rate and computes relative rate changes before
    multiplying by the jump-chain invariant mass. Existing sources stay intact.
    """

    def calculate(self, alpha):
        alpha = np.clip(np.asarray(alpha, dtype=float), 0, 1)
        if self.last_alpha is not None and np.array_equal(alpha, self.last_alpha):
            return self.last_result
        e, n, p = self.experiment, self.experiment.n, self.experiment.params
        self.calls += 1
        full = e.game.transition(np.broadcast_to(alpha[:, None], (n, n)),
                                 np.broadcast_to(alpha[None, :], (n, n)))
        reduced = (full[self.representatives]@self.membership).tocsr()
        off = reduced.copy(); off.setdiag(0); off.eliminate_zeros()
        rates = np.asarray(off.sum(axis=1)).ravel()
        if np.any(rates <= 0):
            raise ArithmeticError("Nonpositive joint exit rate during outside-option search")
        jump = (diags(1/rates)@off).tocsr()
        anchor = len(rates)//2
        linear = (jump.T-eye(len(rates), format="csr")).tolil()
        linear[anchor, :] = np.ones(len(rates))
        lu = splu(linear.tocsc())
        rhs = np.zeros(len(rates)); rhs[anchor] = 1.
        w = lu.solve(rhs)
        calendar = w/rates
        calendar /= calendar.sum()
        residual = float(np.sum(abs(reduced.T@calendar-calendar)))
        if (np.any(~np.isfinite(calendar)) or calendar.min() < -1e-10 or residual > 1e-10):
            invariant = checked_joint_stationary(e.game, full, force_positive=False)
            nu = invariant["nu"]
            calendar = nu[self.i, self.j]*np.where(self.i == self.j, 1, 2)
            w = calendar*rates; w /= w.sum()
        else:
            calendar = np.maximum(calendar, 0); calendar /= calendar.sum()
            w = calendar*rates; w /= w.sum()
        mu = self.marginal@calendar
        demand = e.game.D@mu
        evaluation = evaluate_policy(e.grid, demand, alpha, p)
        value, advantage = evaluation["value"], evaluation["advantage"]
        relative_rate_derivatives = self.rate_derivatives/rates[:, None]
        dw_rhs = np.empty((len(rates), n))
        for k, delta in enumerate(self.derivatives):
            dw_rhs[:, k] = -(delta.T@(w/rates)
                             -jump.T@(w*relative_rate_derivatives[:, k]))
        dw_rhs[anchor, :] = 0
        dw = lu.solve(dw_rhs)
        unnormalized = w/rates
        normalizer = unnormalized.sum()
        dt = dw/rates[:, None]-unnormalized[:, None]*relative_rate_derivatives
        dc = dt/normalizer-calendar[:, None]*(dt.sum(axis=0)/normalizer)
        dmu = self.marginal@dc
        dd = e.game.D@dmu
        success = p.p1+(p.p2-p.p1)*alpha
        selected = diags(success)@self.success+diags(1-success)@self.failure
        own_transition = diags(1-demand)+diags(demand)@selected
        own_lu = splu((eye(n, format="csr")-p.gamma*own_transition).tocsc())
        flow = p.R-p.c1-(p.c2-p.c1)*alpha+p.gamma*(selected@value-value)
        dv = own_lu.solve(flow[:, None]*dd+np.diag(demand*advantage))
        da = p.gamma*(p.p2-p.p1)*((self.success-self.failure)@dv)
        result = dict(advantage=advantage, jacobian=da, mu=mu, policy=alpha,
                      value=value, demand=demand, stationary_residual=residual)
        self.last_alpha, self.last_result = alpha.copy(), result
        return result


class FixedOutsideMarginalExperiment(JointMarginalExperiment):
    """New marginal OE whose p0 is external and never a consistency scalar."""

    def __init__(self, n=21, params=Parameters(), p0=.5, order=128, tol=1e-7, dp_tol=1e-10):
        if int(n) != n or n < 3 or tol <= 0 or dp_tol <= 0:
            raise ValueError("Require at least three nodes and positive tolerances")
        self.p0 = _threshold(p0)
        if self.p0 == 1:
            raise ValueError("p0=1 has no observations and no unique stationary comparison; use p0<1.")
        self.n, self.params, self.order = int(n), params, int(order)
        self.tol, self.dp_tol = float(tol), float(dp_tol)
        self.grid = np.linspace(0, 1, self.n)
        self.game = OutsideOptionGame(self.grid, params, self.p0, order=order)

    def configuration(self):
        return dict(n=self.n, parameters=asdict(self.params), p0=self.p0, order=self.order,
                    tolerance=self.tol, dp_tolerance=self.dp_tol,
                    stationary_tolerance=1e-11, stationary_stability_tolerance=self.tol,
                    closure="Full marginal of actual joint calendar-time invariant law",
                    strategy="Symmetric mixed own-reputation policy",
                    market="Two endogenous sellers and one external fixed known outside option",
                    outside_event="Zero focal profit, no observation, both reputations freeze",
                    forgetting_clock="Seller observations only; unselected reputation freezes",
                    p0_role="External common benchmark; not an endogenous equilibrium object")

    def solve(self, initial_policy=None, max_evaluations=1500, progress=None, prefer_local=False):
        """Use the established search with the rare-event response engine.

        Only the response engine changes. The continuation, face polish, and
        fresh positive stationary certification retain the original criteria.
        """
        if max_evaluations < 1:
            raise ValueError("Positive evaluation budget required")
        initial = np.full(self.n, .5) if initial_policy is None else np.asarray(initial_policy, dtype=float)
        if (initial.shape != self.grid.shape or np.any(~np.isfinite(initial))
                or np.any((initial < 0) | (initial > 1))):
            raise ValueError("Initial policy must be a finite probability vector")
        if prefer_local and initial_policy is not None:
            return self._solve_prefer_local(
                initial, _OutsideResponse, max_evaluations, progress,
                lambda: self._continuation(initial, max_evaluations, progress))
        return self._continuation(initial, max_evaluations, progress)

    def _continuation(self, initial, max_evaluations, progress):
        """Continuation and final certificates, isolated from prior experiments."""
        engine = _OutsideResponse(self)
        history, candidates = [], []
        strong_evaluations = 0
        schedules = [[.3, .1, .03, .01, .003, .001, .0003, .0001, 0.],
                     [.3, .1, .03, .02, .01, .007, .003, .002, .001, .0005, .0002, .0001, 0.]]
        for attempt, (start, schedule) in enumerate(zip([initial, np.full(self.n, .5)], schedules)):
            alpha = start.copy()
            for temperature in schedule:
                if engine.calls >= max_evaluations:
                    break

                def equations(raw):
                    data = engine.calculate(raw)
                    if temperature:
                        return raw-expit(data["advantage"]/temperature)
                    return raw-np.clip(raw+data["advantage"], 0, 1)

                def jacobian(raw):
                    data = engine.calculate(raw)
                    da = data["jacobian"]*((raw >= 0) & (raw <= 1))[None, :]
                    if temperature:
                        target = expit(data["advantage"]/temperature)
                        return np.eye(self.n)-(target*(1-target)/temperature)[:, None]*da
                    inside = (raw+data["advantage"] > 0) & (raw+data["advantage"] < 1)
                    return np.eye(self.n)-inside[:, None]*(np.eye(self.n)+da)

                before = engine.calls
                solved = root(equations, alpha, jac=jacobian, method="hybr",
                              options={"xtol": 1e-10,
                                       "maxfev": min(150, max_evaluations-engine.calls), "factor": .3})
                error = float(np.max(abs(equations(solved.x))))
                if np.all(np.isfinite(solved.x)):
                    alpha = np.clip(solved.x, 0, 1)
                history.append(dict(attempt=attempt, temperature=temperature,
                                    equation_error=error, evaluations=engine.calls-before,
                                    solver_success=bool(solved.success), solver_message=str(solved.message)))
                if progress is not None:
                    progress(f"continuation {attempt+1}, temperature={temperature:g}, residual={error:.3g}")
            for _ in range(5):
                alpha[alpha < 1e-7] = 0
                alpha[alpha > 1-1e-7] = 1
                mixed = np.flatnonzero((alpha > 0) & (alpha < 1))
                if not len(mixed) or engine.calls >= max_evaluations:
                    break
                baseline = alpha.copy()

                def expand(weights):
                    rule = baseline.copy(); rule[mixed] = weights
                    return rule

                def face(weights):
                    return engine.calculate(expand(weights))["advantage"][mixed]

                def face_jac(weights):
                    return engine.calculate(expand(weights))["jacobian"][np.ix_(mixed, mixed)]

                polished = root(face, alpha[mixed], jac=face_jac, method="hybr",
                                options={"xtol": 1e-11, "maxfev": min(100, max_evaluations-engine.calls)})
                if not np.all(np.isfinite(polished.x)):
                    break
                raw = expand(polished.x)
                alpha = np.clip(raw, 0, 1)
                if np.all((raw >= 0) & (raw <= 1)):
                    break
            candidate = self.evaluate(alpha)
            if (not candidate["verified"] and candidate["mixed_indices"]
                    and candidate["dp_diagnostics"]["complementarity_residual"] < 1e-5):
                mixed = np.asarray(candidate["mixed_indices"], dtype=int)
                baseline = alpha.copy()
                strong_calls = [0]

                def strong_face(weights):
                    rule = baseline.copy(); rule[mixed] = np.clip(weights, 0, 1)
                    matrix = self.game.transition(np.broadcast_to(rule[:, None], (self.n, self.n)),
                                                  np.broadcast_to(rule[None, :], (self.n, self.n)))
                    stationary = checked_joint_stationary(self.game, matrix, tol=1e-11,
                                                           stability_tol=self.tol, force_positive=True)
                    checked = evaluate_policy(self.grid, self.game.D@stationary["mu_b"], rule, self.params)
                    strong_calls[0] += 1
                    return checked["advantage"][mixed]

                polished = root(strong_face, alpha[mixed], method="hybr",
                                options={"xtol": 1e-10, "eps": 1e-10, "maxfev": 60})
                strong_evaluations += strong_calls[0]
                if (np.all(np.isfinite(polished.x))
                        and np.all((polished.x >= 0) & (polished.x <= 1))):
                    trial = baseline.copy(); trial[mixed] = polished.x
                    fresh = self.evaluate(trial)
                    if (fresh["verified"] or fresh["dp_diagnostics"]["conditional_regret"]
                            <= candidate["dp_diagnostics"]["conditional_regret"]):
                        candidate = fresh
                history.append(dict(attempt=attempt, stage="positive stationary mixed-face polish",
                                    evaluations=strong_calls[0], solver_success=bool(polished.success),
                                    equation_error=float(np.max(abs(polished.fun))),
                                    solver_message=str(polished.message)))
            candidates.append(candidate)
            if candidate["verified"]:
                break
        best = min(candidates, key=lambda c: (not c["verified"], c["dp_diagnostics"]["conditional_regret"]))
        best.update(history=history, search_metadata=dict(
            evaluations=engine.calls+strong_evaluations,
            exploratory_evaluations=engine.calls, strong_polish_evaluations=strong_evaluations,
            max_evaluations=max_evaluations, attempts=len(candidates),
            method="Outside-option joint-marginal complementarity with ratio-first sensitivities",
            exhaustive=False, unique=False,
            scope="Finite grid and numerical demand matrix; no continuous-state or MPE claim"))
        return best
