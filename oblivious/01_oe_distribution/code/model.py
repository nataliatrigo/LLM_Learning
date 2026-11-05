"""Mixed oblivious optimization closed with the actual joint-game marginal.

The seller optimizes against the unconditional rival marginal.  Only the
stationary closure changes relative to OE: it comes from the two-player kernel.
This is not a full-state Markov perfect equilibrium.
"""

from dataclasses import asdict

import numpy as np
from scipy.optimize import root
from scipy.special import expit
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import splu

from oblivious.oe_p0.code.game import Game
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.policy import evaluate_policy, solve_dp, interpolation_matrix
from oblivious.oe_p0.code.core import checked_joint_stationary, posterior_mean


class _JointResponse:
    """Exact symmetric lumping and analytic policy sensitivities for search.

    Final certificates use a separate, independently checked invariant solver.
    The jump-chain formulation avoids subtracting a tiny exit rate from one.
    """

    def __init__(self, experiment):
        self.experiment = experiment
        game, n = experiment.game, experiment.n
        i, j = np.triu_indices(n)
        self.i, self.j = i, j
        self.representatives = i*n+j
        groups = np.empty((n, n), dtype=int)
        groups[i, j] = np.arange(len(i))
        groups[j, i] = np.arange(len(i))
        self.membership = csr_matrix((np.ones(n*n), (np.arange(n*n), groups.ravel())),
                                     shape=(n*n, len(i)))
        columns = np.arange(len(i))
        self.marginal = csr_matrix((np.full(2*len(i), .5),
                                    (np.concatenate((i, j)), np.tile(columns, 2))),
                                   shape=(n, len(i)))
        # Store derivatives of OFF-diagonal rates, preserving tiny TS tails.
        self.derivatives = []
        self.rate_derivatives = []
        dp = game.params.p2-game.params.p1
        ds = game.own_success-game.own_failure
        dr = game.rival_success-game.rival_failure
        for k in range(n):
            a = np.zeros((n, n)); a[k, :] = game.D[k, :]
            b = np.zeros((n, n)); b[:, k] = game.D.T[:, k]
            delta = dp*(diags(a.ravel())@ds + diags(b.ravel())@dr)
            delta = (delta[self.representatives]@self.membership).tocsr()
            delta.setdiag(0); delta.eliminate_zeros()
            self.derivatives.append(delta)
            self.rate_derivatives.append(np.asarray(delta.sum(axis=1)).ravel())
        self.rate_derivatives = np.column_stack(self.rate_derivatives)
        self.success = interpolation_matrix(experiment.grid, game.params.rho*experiment.grid+1-game.params.rho)
        self.failure = interpolation_matrix(experiment.grid, game.params.rho*experiment.grid)
        self.last_alpha = None
        self.calls = 0

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
            raise ArithmeticError("Nonpositive joint exit rate during search")
        jump = (diags(1/rates)@off).tocsr()
        anchor = len(rates)//2
        linear = (jump.T-eye(len(rates), format="csr")).tolil()
        linear[anchor, :] = np.ones(len(rates))
        lu = splu(linear.tocsc())
        rhs = np.zeros(len(rates)); rhs[anchor] = 1.
        w = lu.solve(rhs)
        calendar = w/rates
        calendar /= calendar.sum()
        residual = float(np.sum(np.abs(reduced.T@calendar-calendar)))
        if (np.any(~np.isfinite(calendar)) or calendar.min() < -1e-10
                or residual > 1e-10):
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
        # Differentiate the jump law, then restore calendar holding times.
        dw_rhs = np.empty((len(rates), n))
        for k, delta in enumerate(self.derivatives):
            dw_rhs[:, k] = -(delta.T@(w/rates)
                             - jump.T@(w*self.rate_derivatives[:, k]/rates))
        dw_rhs[anchor, :] = 0
        dw = lu.solve(dw_rhs)
        normalizer = np.sum(w/rates)
        dt = dw/rates[:, None] - w[:, None]*self.rate_derivatives/rates[:, None]**2
        dc = dt/normalizer-calendar[:, None]*(dt.sum(axis=0)/normalizer)
        dmu = self.marginal@dc
        dd = e.game.D@dmu
        success = p.p1+(p.p2-p.p1)*alpha
        selected = diags(success)@self.success+diags(1-success)@self.failure
        own_transition = diags(1-demand)+diags(demand)@selected
        own_lu = splu((eye(n, format="csr")-p.gamma*own_transition).tocsc())
        flow = p.R-p.c1-(p.c2-p.c1)*alpha+p.gamma*(selected@value-value)
        dv_rhs = flow[:, None]*dd+np.diag(demand*advantage)
        dv = own_lu.solve(dv_rhs)
        da = p.gamma*(p.p2-p.p1)*((self.success-self.failure)@dv)
        result = dict(advantage=advantage, jacobian=da, mu=mu,
                      policy=alpha, value=value, demand=demand,
                      stationary_residual=residual)
        self.last_alpha, self.last_result = alpha.copy(), result
        return result


class JointMarginalExperiment:
    def __init__(self, n=21, params=Parameters(), order=128, tol=1e-7, dp_tol=1e-10):
        if int(n) != n or n < 3 or tol <= 0 or dp_tol <= 0:
            raise ValueError("Require at least three nodes and positive tolerances")
        self.n, self.params, self.order = int(n), params, int(order)
        self.tol, self.dp_tol = float(tol), float(dp_tol)
        self.grid = np.linspace(0, 1, self.n)
        self.game = Game(self.grid, params, order=order)

    def configuration(self):
        return dict(n=self.n, parameters=asdict(self.params), order=self.order,
                    tolerance=self.tol, dp_tolerance=self.dp_tol,
                    stationary_tolerance=1e-11, stationary_stability_tolerance=self.tol,
                    closure="Full marginal of actual joint calendar-time invariant law",
                    strategy="Symmetric mixed own-reputation policy")

    def evaluate(self, alpha, conjectured_mu=None):
        """Fresh certificate; passing a conjecture explicitly tests its closure."""
        alpha = np.asarray(alpha, dtype=float)
        if (alpha.shape != self.grid.shape or np.any(~np.isfinite(alpha))
                or np.any((alpha < 0) | (alpha > 1))):
            raise ValueError("Policy must be a finite probability on every grid node")
        n, p = self.n, self.params
        matrix = self.game.transition(np.broadcast_to(alpha[:, None], (n, n)),
                                      np.broadcast_to(alpha[None, :], (n, n)))
        stationary = checked_joint_stationary(self.game, matrix, tol=1e-11,
                                               stability_tol=self.tol, force_positive=True)
        mu_b, mu_a = stationary["mu_b"], stationary["mu_a"]
        mu = mu_b.copy() if conjectured_mu is None else np.asarray(conjectured_mu, dtype=float)
        if (mu.shape != self.grid.shape or np.any(~np.isfinite(mu)) or np.any(mu < 0)
                or abs(mu.sum()-1) > 1e-12):
            raise ValueError("Conjectured marginal must be a normalized probability vector")
        demand = self.game.D@mu
        checked = evaluate_policy(self.grid, demand, alpha, p)
        response = solve_dp(self.grid, demand, p, tol=self.dp_tol)
        advantage, value = checked["advantage"], checked["value"]
        regret = np.maximum(advantage, 0)-alpha*advantage
        complementarity = float(np.max(np.abs(alpha-np.clip(alpha+advantage, 0, 1))))
        interior = np.flatnonzero((alpha > 0) & (alpha < 1))
        indifference = float(np.max(np.abs(advantage[interior]), initial=0))
        bellman = checked["residual"]+float(np.max(demand*regret))
        numerical = (checked["residual"]+response["residual"])/(1-p.gamma)
        gain = max(0., float(np.max(response["value"]-value)))+numerical
        marginal_l1 = float(np.sum(np.abs(mu-mu_b)))
        dp_verified = bool(checked["residual"] <= self.dp_tol and bellman <= self.dp_tol
                           and np.max(regret) <= self.dp_tol
                           and complementarity <= self.dp_tol
                           and indifference <= self.dp_tol and gain <= self.tol)
        verified = bool(dp_verified and marginal_l1 <= self.tol
                        and stationary["summary"]["verified"]
                        and all(np.all(np.isfinite(a)) for a in (value, advantage, mu, demand)))
        dp = dict(verified=dp_verified, bellman_residual=bellman,
                  policy_evaluation_residual=checked["residual"],
                  conditional_regret=float(np.max(regret)),
                  complementarity_residual=complementarity,
                  mixed_indifference_residual=indifference,
                  best_response_gain_upper_bound=gain,
                  best_response_numerical_error_bound=numerical,
                  iterations=int(response["iterations"]))
        summary = dict(rho=p.rho, n=n, verified=verified, candidate_verified=verified,
                       marginal_l1=marginal_l1, marginal_residual=marginal_l1,
                       stationary_residual=stationary["summary"]["residual"],
                       mixed_nodes=len(interior), mixed_indices=interior.tolist(),
                       mixed_reputations=self.grid[interior].tolist(),
                       mixed_probabilities=alpha[interior].tolist(),
                       mean_reputation=float(mu_b@self.grid),
                       mean_posterior=float(mu_b@posterior_mean(self.grid, p)),
                       dp=dp, stationary=stationary["summary"])
        return dict(grid=self.grid.copy(), policy=alpha.copy(), value=value,
                    advantage=advantage, nu=stationary["nu"], mu_a=mu_a, mu_b=mu_b,
                    mu=mu.copy(), demand=demand, marginal_residual=marginal_l1,
                    verified=verified, candidate_verified=verified, summary=summary,
                    dp_diagnostics=dp, stationary_diagnostics=stationary["summary"],
                    mixed_nodes=len(interior), mixed_indices=interior.tolist(),
                    mixed_probabilities=alpha[interior].tolist())

    def _positive_stationary_demand(self, stationary):
        return self.game.D@stationary["mu_b"]

    def _solve_prefer_local(self, initial, response_type, max_evaluations, progress, fallback):
        """Try an existing policy without reheating it, retaining full certificates.

        This only changes search order. A failed local attempt falls back to the
        original search with its entire exploratory budget. No active set,
        threshold form, or uniqueness of equilibrium is assumed.
        """
        engine = response_type(self)
        alpha = initial.copy()
        history, certificates = [], {}
        strong_evaluations = 0
        local_budget = min(150, max_evaluations)

        def certify(rule):
            # Reuse a fresh certificate only for a byte-identical policy in
            # this solve, with the same experiment and certification settings.
            key = np.ascontiguousarray(rule, dtype=float).tobytes()
            if key not in certificates:
                certificates[key] = self.evaluate(rule)
            return certificates[key]

        def equations(raw):
            data = engine.calculate(raw)
            return raw-np.clip(raw+data["advantage"], 0, 1)

        def jacobian(raw):
            data = engine.calculate(raw)
            da = data["jacobian"]*((raw >= 0) & (raw <= 1))[None, :]
            inside = (raw+data["advantage"] > 0) & (raw+data["advantage"] < 1)
            return np.eye(self.n)-inside[:, None]*(np.eye(self.n)+da)

        candidate = None
        try:
            # A coarse-grid interpolation may create artificial mixed nodes
            # next to a pure action region. A few fixed-demand policy updates
            # remove them cheaply when they improve the *joint-closure*
            # complementarity residual. Reject non-improving updates rather
            # than assuming this best-response map is a contraction.
            for iteration in range(3):
                data = engine.calculate(alpha)
                old_error = float(np.max(abs(alpha-np.clip(alpha+data["advantage"], 0, 1))))
                if old_error <= self.dp_tol or engine.calls >= local_budget:
                    break
                proposal = solve_dp(self.grid, data["demand"], self.params,
                                    tol=self.dp_tol)["policy"]
                probe = engine.calculate(proposal)
                new_error = float(np.max(abs(proposal-np.clip(proposal+probe["advantage"], 0, 1))))
                accepted = new_error < .9*old_error
                history.append(dict(stage="local fixed-demand best response", iteration=iteration,
                                    equation_error=new_error, previous_error=old_error,
                                    accepted=accepted))
                if not accepted:
                    break
                alpha = proposal.copy()
            before = engine.calls
            initial_error = float(np.max(abs(equations(alpha))))
            if initial_error > self.dp_tol:
                result = root(equations, alpha, jac=jacobian, method="hybr",
                              options={"xtol": 1e-10, "maxfev": local_budget, "factor": .3})
                if np.all(np.isfinite(result.x)):
                    alpha = np.clip(result.x, 0, 1)
                success, message = bool(result.success), str(result.message)
            else:
                success, message = True, "Initial policy already satisfies exploratory complementarity"
            error = float(np.max(abs(equations(alpha))))
            history.append(dict(stage="local exact complementarity", temperature=0.,
                                equation_error=error, evaluations=engine.calls-before,
                                solver_success=success, solver_message=message))
            if progress is not None:
                progress(f"local exact complementarity, residual={error:.3g}")

            if error <= self.dp_tol:
                candidate = certify(alpha)

            # The full complementarity step can move any endpoint. Face polish
            # only refines the resulting interior nodes; certification checks
            # all endpoint inequalities again after every snap.
            for _ in range(0 if candidate is not None and candidate["verified"] else 5):
                alpha[alpha < 1e-7] = 0
                alpha[alpha > 1-1e-7] = 1
                mixed = np.flatnonzero((alpha > 0) & (alpha < 1))
                if not len(mixed) or engine.calls >= local_budget:
                    break
                baseline = alpha.copy()

                def expand(weights):
                    rule = baseline.copy(); rule[mixed] = weights
                    return rule

                def face(weights):
                    return engine.calculate(expand(weights))["advantage"][mixed]

                def face_jac(weights):
                    return engine.calculate(expand(weights))["jacobian"][np.ix_(mixed, mixed)]

                before = engine.calls
                polished = root(face, alpha[mixed], jac=face_jac, method="hybr",
                                options={"xtol": 1e-11,
                                         "maxfev": min(100, local_budget-engine.calls)})
                if not np.all(np.isfinite(polished.x)):
                    break
                raw = expand(polished.x)
                alpha = np.clip(raw, 0, 1)
                history.append(dict(stage="local mixed-face polish", evaluations=engine.calls-before,
                                    solver_success=bool(polished.success),
                                    equation_error=float(np.max(abs(polished.fun))),
                                    solver_message=str(polished.message)))
                if np.all((raw >= 0) & (raw <= 1)):
                    break

            # A poor local root is sent directly to the original continuation;
            # costly positive stationary certificates are reserved for nearby
            # candidates, and never replaced by this exploratory residual.
            if ((candidate is None or not candidate["verified"])
                    and float(np.max(abs(equations(alpha)))) < 1e-5):
                candidate = certify(alpha)
                if not candidate["verified"] and candidate["mixed_indices"]:
                    mixed = np.asarray(candidate["mixed_indices"], dtype=int)
                    baseline = alpha.copy()
                    strong_calls = [0]

                    def strong_face(weights):
                        rule = baseline.copy(); rule[mixed] = np.clip(weights, 0, 1)
                        matrix = self.game.transition(
                            np.broadcast_to(rule[:, None], (self.n, self.n)),
                            np.broadcast_to(rule[None, :], (self.n, self.n)))
                        stationary = checked_joint_stationary(
                            self.game, matrix, tol=1e-11, stability_tol=self.tol, force_positive=True)
                        checked = evaluate_policy(self.grid, self._positive_stationary_demand(stationary),
                                                  rule, self.params)
                        strong_calls[0] += 1
                        return checked["advantage"][mixed]

                    polished = root(strong_face, alpha[mixed], method="hybr",
                                    options={"xtol": 1e-10, "eps": 1e-10, "maxfev": 60})
                    strong_evaluations += strong_calls[0]
                    if (np.all(np.isfinite(polished.x))
                            and np.all((polished.x >= 0) & (polished.x <= 1))):
                        trial = baseline.copy(); trial[mixed] = polished.x
                        fresh = certify(trial)
                        if (fresh["verified"] or fresh["dp_diagnostics"]["conditional_regret"]
                                <= candidate["dp_diagnostics"]["conditional_regret"]):
                            candidate = fresh
                    history.append(dict(stage="local positive stationary mixed-face polish",
                                        evaluations=strong_calls[0], solver_success=bool(polished.success),
                                        equation_error=float(np.max(abs(polished.fun))),
                                        solver_message=str(polished.message)))
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            # A numerical failure in the optional shortcut must not remove
            # the established search's opportunity to find a certified root.
            history.append(dict(stage="local numerical failure", solver_success=False,
                                solver_message=str(exc)))

        local_verified = bool(candidate is not None and candidate["verified"])
        if local_verified:
            best = candidate
            best.update(history=history, search_metadata=dict(
                evaluations=engine.calls+strong_evaluations,
                exploratory_evaluations=engine.calls, strong_polish_evaluations=strong_evaluations,
                max_evaluations=max_evaluations, attempts=1,
                method="Local exact complementarity from supplied policy with fresh certification",
                exhaustive=False, unique=False,
                scope="Finite grid and numerical demand matrix; no continuous-state or MPE claim"))
        else:
            best = fallback()
            best["history"] = history+best["history"]
            best["search_metadata"]["attempts"] += 1
            for key, extra in (("evaluations", engine.calls+strong_evaluations),
                               ("exploratory_evaluations", engine.calls),
                               ("strong_polish_evaluations", strong_evaluations)):
                best["search_metadata"][key] += extra
        metadata = best["search_metadata"]
        metadata.update(prefer_local=True, local_attempted=True, local_verified=local_verified,
                        fallback_used=not local_verified, local_exploratory_evaluations=engine.calls,
                        local_strong_polish_evaluations=strong_evaluations,
                        local_certifications=len(certificates), local_max_evaluations=local_budget,
                        budget_interpretation="Soft optional local exploratory budget plus unchanged original budget on fallback; final certification and strong polish are additional")
        if hasattr(engine, "signed_roundoff_search_calls"):
            metadata["local_signed_roundoff_calls"] = engine.signed_roundoff_search_calls
            metadata["local_positive_fallback_calls"] = engine.positive_fallback_search_calls
        return best

    def solve(self, initial_policy=None, max_evaluations=1500, progress=None, prefer_local=False):
        """Continuation to exact complementarity, followed by fresh certification.

        A small residual of a regularized problem is never an equilibrium claim.
        Multiple starts improve search; neither uniqueness nor exhaustive root
        discovery is asserted. Pure endpoints remain valid mixed strategies.
        With prefer_local and a supplied initial policy, first try the exact
        problem directly; a failed certificate retains the original search.
        """
        if max_evaluations < 1:
            raise ValueError("Positive evaluation budget required")
        initial = np.full(self.n, .5) if initial_policy is None else np.asarray(initial_policy, dtype=float)
        if (initial.shape != self.grid.shape or np.any(~np.isfinite(initial))
                or np.any((initial < 0) | (initial > 1))):
            raise ValueError("Initial policy must be a finite probability vector")
        if prefer_local and initial_policy is not None:
            return self._solve_prefer_local(
                initial, _JointResponse, max_evaluations, progress,
                lambda: self.solve(initial, max_evaluations, progress, prefer_local=False))
        engine = _JointResponse(self)
        history, candidates = [], []
        strong_evaluations = 0
        schedules = [[.3, .1, .03, .01, .003, .001, .0003, .0001, 0.],
                     [.3, .1, .03, .02, .01, .007, .003, .002, .001, .0005, .0002, .0001, 0.]]
        starts = [initial, np.full(self.n, .5)]
        for attempt, (start, schedule) in enumerate(zip(starts, schedules)):
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
                result = root(equations, alpha, jac=jacobian, method="hybr",
                              options={"xtol": 1e-10,
                                       "maxfev": min(150, max_evaluations-engine.calls), "factor": .3})
                error = float(np.max(np.abs(equations(result.x))))
                if np.all(np.isfinite(result.x)):
                    alpha = np.clip(result.x, 0, 1)
                history.append(dict(attempt=attempt, temperature=temperature,
                                    equation_error=error, evaluations=engine.calls-before,
                                    solver_success=bool(result.success), solver_message=str(result.message)))
                if progress is not None:
                    progress(f"continuation {attempt+1}, temperature={temperature:g}, residual={error:.3g}")
            # Solve only interior action-indifference equations; endpoint choices
            # remain fixed. Snapped probabilities are always checked afresh.
            for _ in range(5):
                data = engine.calculate(alpha)
                boundary = 1e-7
                alpha[alpha < boundary] = 0
                alpha[alpha > 1-boundary] = 1
                mixed = np.flatnonzero((alpha > 0) & (alpha < 1))
                if not len(mixed) or engine.calls >= max_evaluations:
                    break
                def expand(weights):
                    rule = alpha.copy(); rule[mixed] = weights
                    return rule
                def face(weights):
                    rule = expand(weights)
                    return engine.calculate(rule)["advantage"][mixed]
                def face_jac(weights):
                    return engine.calculate(expand(weights))["jacobian"][np.ix_(mixed, mixed)]
                result = root(face, alpha[mixed], jac=face_jac, method="hybr",
                              options={"xtol": 1e-11, "maxfev": min(100, max_evaluations-engine.calls)})
                raw = expand(result.x)
                alpha = np.clip(raw, 0, 1)
                if np.all((raw >= 0) & (raw <= 1)):
                    break
            candidate = self.evaluate(alpha)
            if (not candidate["verified"] and len(candidate["mixed_indices"])
                    and candidate["dp_diagnostics"]["complementarity_residual"] < 1e-5):
                # A sparse invariant approximation can be adequate for search
                # but shift a tiny indifference residual past its tighter DP
                # tolerance. Re-solve the mixed face with subtraction-free
                # stationary laws; do not loosen the final certificate.
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
                if np.all(np.isfinite(polished.x)) and np.all((polished.x >= 0) & (polished.x <= 1)):
                    trial = baseline.copy(); trial[mixed] = polished.x
                    fresh = self.evaluate(trial)
                    if (fresh["verified"] or fresh["dp_diagnostics"]["conditional_regret"]
                            <= candidate["dp_diagnostics"]["conditional_regret"]):
                        candidate = fresh
                history.append(dict(attempt=attempt, stage="positive stationary mixed-face polish",
                                    evaluations=strong_calls[0], solver_success=bool(polished.success),
                                    equation_error=float(np.max(np.abs(polished.fun))),
                                    solver_message=str(polished.message)))
            candidates.append(candidate)
            if candidate["verified"]:
                break
        best = min(candidates, key=lambda c: (not c["verified"],
                                              c["dp_diagnostics"]["conditional_regret"]))
        best.update(history=history, search_metadata=dict(
            evaluations=engine.calls+strong_evaluations,
            exploratory_evaluations=engine.calls, strong_polish_evaluations=strong_evaluations,
            max_evaluations=max_evaluations, attempts=len(candidates),
            method="Joint-marginal reduced complementarity with analytic sensitivities and continuation",
            exhaustive=False, unique=False,
            scope="Finite grid and numerical demand matrix; no continuous-state or MPE claim"))
        return best
