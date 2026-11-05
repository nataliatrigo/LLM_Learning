"""Conditional OE with symmetric own-reputation policies and actual joint law.

At every own state x_i with strictly positive stationary mass, the conjectured
rival law is nu[i,:]/mu_A[i]. The fixed-demand Bellman problem uses its actual
conditional selection probability sum_j D[i,j]*nu[i,j]/mu_A[i]. This changes
the closure from Distribution OE, not the own-state optimization problem.
This is not a full-state Markov-perfect equilibrium certificate.

The search adapts JointMarginalExperiment's analytic complementarity method;
its final stationary law is independently certified with positive elimination.
No original experiment source or output is modified.
"""

import numpy as np
from scipy.optimize import root
from scipy.special import expit
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import splu

from oblivious.oe_p0.code.policy import evaluate_policy, solve_dp
from oblivious.oe_p0.code.core import checked_joint_stationary, posterior_mean, _jump_system, _positive_invariant
from oblivious.oe_distribution.code.model import JointMarginalExperiment, _JointResponse


def conditional_from_joint(nu, demand_matrix, off_path=None):
    """Return completed conditional laws, retaining every positive-mass row.

    nu must be a normalized nonnegative joint law; symmetry is not required.
    Exactly zero rows have no probabilistically defined conditional law. Their
    explicit completion is the unconditional rival marginal, or a supplied
    normalized vector/matrix. No probability floor or positive-mass truncation
    is used. All zero rows are reported in the returned metadata.
    """
    nu = np.asarray(nu, dtype=float)
    D = np.asarray(demand_matrix, dtype=float)
    if (nu.ndim != 2 or nu.shape[0] != nu.shape[1] or len(nu) < 1
            or not np.isfinite(nu).all() or np.any(nu < 0)
            or abs(float(nu.sum())-1) > 1e-12):
        raise ValueError("nu must be a finite normalized nonnegative square joint law")
    if D.shape != nu.shape or not np.isfinite(D).all() or np.any((D < 0) | (D > 1)):
        raise ValueError("Demand matrix must match nu and contain probabilities")
    mu_a, mu_b = nu.sum(axis=1), nu.sum(axis=0)
    positive = mu_a > 0
    zero = np.flatnonzero(~positive)
    if off_path is None:
        completion = np.broadcast_to(mu_b, nu.shape)
        convention = "Unconditional rival marginal on exactly zero own-mass rows"
    else:
        supplied = np.asarray(off_path, dtype=float)
        if supplied.shape == (len(nu),):
            supplied = np.broadcast_to(supplied, nu.shape)
        if (supplied.shape != nu.shape or not np.isfinite(supplied).all()
                or np.any(supplied < 0)
                or np.max(abs(supplied.sum(axis=1)-1)) > 1e-12):
            raise ValueError("off_path must be a normalized rival vector or rowwise conditional matrix")
        completion = supplied
        convention = "Explicit supplied conditional completion on exactly zero own-mass rows"
    conditional = np.array(completion, copy=True)
    conditional[positive] = nu[positive]/mu_a[positive, None]
    demand = np.sum(conditional*D, axis=1)
    return dict(conditional_mu=conditional, conditional_demand=demand,
                mu_a=mu_a, mu_b=mu_b, positive_rows=np.flatnonzero(positive).tolist(),
                conditioning_zero_rows=zero.tolist(), zero_rows=zero.tolist(),
                off_path_convention=convention,
                row_normalization_error=float(np.max(abs(conditional.sum(axis=1)-1))))


def _independent_conditional_stability(game, matrix, primary):
    """Check relative conditional laws via independent reversed positive GTH.

    Global joint-law L1 accuracy alone does not control conditioning on tiny
    own-state probabilities. Every strictly positive row is checked directly.
    """
    n = game.n
    i, j = np.triu_indices(n)
    groups = np.empty((n,n), dtype=int)
    groups[i,j] = np.arange(len(i)); groups[j,i] = np.arange(len(i))
    membership = csr_matrix((np.ones(n*n), (np.arange(n*n),groups.ravel())),
                             shape=(n*n,len(i)))
    reduced = (matrix[i*n+j]@membership).tocsr()
    jump, rates = _jump_system(reduced)
    reverse = _positive_invariant(jump,rates,np.arange(len(rates))[::-1])
    nu = np.zeros((n,n))
    shares = reverse/np.where(i == j,1,2)
    nu[i,j] = shares; nu[j,i] = shares
    checked = conditional_from_joint(nu, game.D)
    same_support = bool(np.array_equal(primary["mu_a"] > 0,checked["mu_a"] > 0))
    common = (primary["mu_a"] > 0)&(checked["mu_a"] > 0)
    row_error = np.sum(abs(primary["conditional_mu"]-checked["conditional_mu"]),axis=1)
    return dict(independent_order_supported_row_l1_max=float(np.max(row_error[common],initial=0)),
                independent_order_demand_max=float(np.max(abs(primary["conditional_demand"]-checked["conditional_demand"]))),
                independent_order_support_agrees=same_support,
                independent_order_joint_l1=float(np.sum(abs(nu-primary["mu_a"][:,None]*primary["conditional_mu"]))),
                independent_order_method="Reversed symmetric joint jump-chain positive GTH; conditional comparison on every positive row")


class _ConditionalResponse(_JointResponse):
    """Conditional-demand version of the exact joint stationary sensitivities."""

    def __init__(self, experiment):
        super().__init__(experiment)
        self.signed_roundoff_search_calls = 0
        self.positive_fallback_search_calls = 0
        # For unordered mass c: mu_A=M c; own selected mass z=B c.
        columns = np.arange(len(self.i))
        self.sale_marginal = csr_matrix((
            np.concatenate((.5*experiment.game.D[self.i, self.j],
                            .5*experiment.game.D[self.j, self.i])),
            (np.concatenate((self.i, self.j)), np.tile(columns, 2))),
            shape=self.marginal.shape)

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
            raise ArithmeticError("Nonpositive joint exit rate during conditional search")
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
        # Search may use signed roundoff dust, but never clips/floors rare mass.
        # If it prevents valid rowwise conditioning, recompute positively.
        # Fresh final certification always uses positive laws in both orders.
        exploratory_mu = self.marginal@calendar
        exploratory_num = self.sale_marginal@calendar
        if (np.any(~np.isfinite(calendar)) or calendar.min() < -1e-10 or residual > 1e-10
                or np.any(exploratory_mu <= 0) or np.any(exploratory_num < 0)
                or np.any(exploratory_num > exploratory_mu*(1+1e-13))):
            self.positive_fallback_search_calls += 1
            invariant = checked_joint_stationary(e.game, full, force_positive=True)
            nu = invariant["nu"]
            calendar = nu[self.i, self.j]*np.where(self.i == self.j, 1, 2)
            w = calendar*rates; w /= w.sum()
            residual = invariant["summary"]["residual"]
        else:
            self.signed_roundoff_search_calls += int(calendar.min() < 0)
            calendar /= calendar.sum()
            w = calendar*rates; w /= w.sum()
        mu = self.marginal@calendar
        numerator = self.sale_marginal@calendar
        positive = mu > 0
        demand = e.game.D@mu  # Explicit off-path completion only on zero rows.
        demand[positive] = numerator[positive]/mu[positive]
        demand = np.clip(demand,0,1)  # Remove only floating-point probability dust.
        evaluation = evaluate_policy(e.grid, demand, alpha, p)
        value, advantage = evaluation["value"], evaluation["advantage"]
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
        dnumerator = self.sale_marginal@dc
        dd = e.game.D@dmu
        dd[positive] = (dnumerator[positive]-demand[positive,None]*dmu[positive])/mu[positive,None]
        success = p.p1+(p.p2-p.p1)*alpha
        selected = diags(success)@self.success+diags(1-success)@self.failure
        own_transition = diags(1-demand)+diags(demand)@selected
        own_lu = splu((eye(n, format="csr")-p.gamma*own_transition).tocsc())
        flow = p.R-p.c1-(p.c2-p.c1)*alpha+p.gamma*(selected@value-value)
        dv_rhs = flow[:, None]*dd+np.diag(demand*advantage)
        dv = own_lu.solve(dv_rhs)
        da = p.gamma*(p.p2-p.p1)*((self.success-self.failure)@dv)
        if not np.all(np.isfinite(da)):
            raise ArithmeticError("Nonfinite conditional-demand policy sensitivity")
        result = dict(advantage=advantage, jacobian=da, mu=mu, policy=alpha,
                      value=value, demand=demand, conditional_demand=demand,
                      stationary_residual=residual,
                      conditioning_zero_rows=np.flatnonzero(~positive).tolist())
        self.last_alpha, self.last_result = alpha.copy(), result
        return result


class ConditionalExperiment(JointMarginalExperiment):
    """Symmetric Conditional OE with the original own-state Bellman problem."""

    def configuration(self):
        result = super().configuration()
        result.update(model="Conditional OE",
                      closure="Conditional rival distribution from actual joint calendar-time invariant law",
                      conditional_formula="mu_i(j)=nu(i,j)/mu_A(i); d_i=sum_j D(i,j)*mu_i(j)",
                      positive_mass_floor=0.,
                      zero_mass_convention="Unconditional rival marginal on exactly zero own-mass rows")
        return result

    def evaluate(self, alpha, conjectured_mu=None):
        """Fresh certificate; optional conjectured_mu is an (n,n) conditional law.

        Passing a conjecture checks it against the independently recomputed
        actual joint law, including positive rows however small their mass.
        Zero-row conjectures must respect the declared off-path completion.
        """
        alpha = np.asarray(alpha, dtype=float)
        if (alpha.shape != self.grid.shape or not np.isfinite(alpha).all()
                or np.any((alpha < 0) | (alpha > 1))):
            raise ValueError("Policy must be a finite probability on every grid node")
        n, p = self.n, self.params
        matrix = self.game.transition(np.broadcast_to(alpha[:, None], (n,n)),
                                      np.broadcast_to(alpha[None,:], (n,n)))
        stationary = checked_joint_stationary(self.game, matrix, tol=1e-11,
                                               stability_tol=self.tol, force_positive=True)
        actual = conditional_from_joint(stationary["nu"], self.game.D)
        order_stability = _independent_conditional_stability(self.game,matrix,actual)
        mu_a, mu_b = actual["mu_a"], actual["mu_b"]
        conditional = actual["conditional_mu"].copy() if conjectured_mu is None else np.asarray(conjectured_mu, dtype=float)
        if (conditional.shape != (n,n) or not np.isfinite(conditional).all()
                or np.any(conditional < 0) or np.max(abs(conditional.sum(axis=1)-1)) > 1e-12):
            raise ValueError("Conjectured conditional law must be an n-by-n normalized nonnegative matrix")
        demand = np.sum(self.game.D*conditional, axis=1)
        checked = evaluate_policy(self.grid, demand, alpha, p)
        response = solve_dp(self.grid, demand, p, tol=self.dp_tol)
        advantage, value = checked["advantage"], checked["value"]
        regret = np.maximum(advantage, 0)-alpha*advantage
        complementarity = float(np.max(np.abs(alpha-np.clip(alpha+advantage,0,1))))
        interior = np.flatnonzero((alpha > 0) & (alpha < 1))
        indifference = float(np.max(abs(advantage[interior]), initial=0))
        bellman = checked["residual"]+float(np.max(demand*regret))
        numerical = (checked["residual"]+response["residual"])/(1-p.gamma)
        gain = max(0.,float(np.max(response["value"]-value)))+numerical
        discrepancy = np.sum(abs(conditional-actual["conditional_mu"]), axis=1)
        supported = mu_a > 0
        weighted = float(mu_a@discrepancy)
        sup_supported = float(np.max(discrepancy[supported], initial=0))
        sup_zero = float(np.max(discrepancy[~supported], initial=0))
        marginal_l1 = float(np.sum(abs(mu_a@conditional-mu_b)))
        demand_error = float(np.max(abs(demand-actual["conditional_demand"])))
        order_verified = bool(order_stability["independent_order_support_agrees"]
                              and order_stability["independent_order_supported_row_l1_max"] <= self.tol
                              and order_stability["independent_order_demand_max"] <= self.tol)
        conditional_verified = bool(weighted <= self.tol and sup_supported <= self.tol
                                    and sup_zero <= self.tol and demand_error <= self.tol and order_verified)
        dp_verified = bool(checked["residual"] <= self.dp_tol and bellman <= self.dp_tol
                           and np.max(regret) <= self.dp_tol and complementarity <= self.dp_tol
                           and indifference <= self.dp_tol and gain <= self.tol)
        verified = bool(dp_verified and conditional_verified and stationary["summary"]["verified"]
                        and all(np.isfinite(a).all() for a in (value,advantage,conditional,demand)))
        dp = dict(verified=dp_verified, bellman_residual=bellman,
                  policy_evaluation_residual=checked["residual"], conditional_regret=float(regret.max()),
                  complementarity_residual=complementarity, mixed_indifference_residual=indifference,
                  best_response_gain_upper_bound=gain, best_response_numerical_error_bound=numerical,
                  iterations=int(response["iterations"]))
        conditional_diag = dict(verified=conditional_verified, closure_weighted_l1=weighted,
            closure_supported_row_l1_max=sup_supported, closure_unsupported_row_l1_max=sup_zero,
            row_normalization_error=float(np.max(abs(conditional.sum(axis=1)-1))),
            integrated_rival_marginal_l1=marginal_l1, demand_consistency_max=demand_error,
            positive_rows=actual["positive_rows"], zero_rows=actual["conditioning_zero_rows"],
            minimum_positive_own_mass=float(mu_a[supported].min()) if np.any(supported) else None,
            off_path_convention=actual["off_path_convention"], positive_mass_floor=0.,
            supplied_conjecture=conjectured_mu is not None,
            independent_order_verified=order_verified,**order_stability,
            certification="Independent actual joint invariant law; rowwise conditional closure on every positive-mass row")
        failures = []
        if not dp_verified:
            failures.append("Own-state Bellman/complementarity certificate failed")
        if not conditional_verified:
            failures.append("Conditional distribution or off-path completion consistency failed")
        if not order_verified:
            failures.append("Independent stationary-order conditional stability failed")
        if not stationary["summary"]["verified"]:
            failures.append("Actual joint stationary law certificate failed")
        summary = dict(rho=p.rho,n=n,model="Conditional OE",verified=verified,candidate_verified=verified,
            marginal_l1=marginal_l1,marginal_residual=marginal_l1,
            conditional_residual=max(sup_supported,sup_zero),conditional_weighted_l1=weighted,
            stationary_residual=stationary["summary"]["residual"],mixed_nodes=len(interior),
            mixed_indices=interior.tolist(),mixed_reputations=self.grid[interior].tolist(),
            mixed_probabilities=alpha[interior].tolist(),mean_reputation=float(mu_b@self.grid),
            mean_posterior=float(mu_b@posterior_mean(self.grid,p)),dp=dp,
            stationary=stationary["summary"],conditional=conditional_diag)
        return dict(grid=self.grid.copy(),policy=alpha.copy(),value=value,advantage=advantage,
            nu=stationary["nu"],mu_a=mu_a,mu_b=mu_b,mu=mu_b.copy(),demand=demand,
            conditional_mu=conditional.copy(),actual_conditional_mu=actual["conditional_mu"],
            conditional_demand=demand.copy(),actual_conditional_demand=actual["conditional_demand"],
            conditioning_zero_rows=actual["conditioning_zero_rows"],conditional_diagnostics=conditional_diag,
            conditional_residual=max(sup_supported,sup_zero),marginal_residual=marginal_l1,
            verified=verified,candidate_verified=verified,status="verified" if verified else "unresolved",
            failure_reasons=failures,summary=summary,dp_diagnostics=dp,
            stationary_diagnostics=stationary["summary"],mixed_nodes=len(interior),
            mixed_indices=interior.tolist(),mixed_probabilities=alpha[interior].tolist())

    def _positive_stationary_demand(self, stationary):
        return conditional_from_joint(stationary["nu"], self.game.D)["conditional_demand"]

    def solve(self, initial_policy=None, max_evaluations=1500, progress=None,
              damping=.35, picard_iterations=20, prefer_local=False):
        """Continuation to exact complementarity, followed by fresh certification.

        Damped policy/stationary/conditional-demand iterations initialize the search.
        A small residual of a regularized problem is never an equilibrium claim.
        Multiple starts improve search; neither uniqueness nor exhaustive root
        discovery is asserted. Pure endpoints remain valid mixed strategies.
        prefer_local first corrects a supplied policy on the exact problem,
        bypassing damping and smoothing only if the fresh certificate passes.
        """
        if max_evaluations < 1:
            raise ValueError("Positive evaluation budget required")
        initial = np.full(self.n, .5) if initial_policy is None else np.asarray(initial_policy, dtype=float)
        if (initial.shape != self.grid.shape or np.any(~np.isfinite(initial))
                or np.any((initial < 0) | (initial > 1))):
            raise ValueError("Initial policy must be a finite probability vector")
        if not 0 < damping <= 1 or int(picard_iterations) != picard_iterations or picard_iterations < 0:
            raise ValueError("Damping must lie in (0,1], and picard_iterations be a nonnegative integer")
        if prefer_local and initial_policy is not None:
            best = self._solve_prefer_local(
                initial, _ConditionalResponse, max_evaluations, progress,
                lambda: self.solve(initial, max_evaluations, progress,
                                   damping=damping, picard_iterations=picard_iterations,
                                   prefer_local=False))
            best["search_metadata"].update(damping=float(damping), picard_iterations=int(picard_iterations))
            best["converged"] = bool(best["verified"])
            return best
        engine = _ConditionalResponse(self)
        history, candidates = [], []
        alpha = initial.copy()
        for iteration in range(min(int(picard_iterations), max_evaluations)):
            before = engine.calls
            data = engine.calculate(alpha)
            response = solve_dp(self.grid, data["demand"], self.params, tol=self.dp_tol)
            regret = float(np.max(np.maximum(data["advantage"], 0)-alpha*data["advantage"]))
            proposal = (1-damping)*alpha+damping*response["policy"]
            history.append(dict(stage="damped conditional best response", iteration=iteration,
                damping=float(damping), conditional_regret=regret,
                policy_change_max=float(np.max(abs(proposal-alpha))),
                stationary_residual=data["stationary_residual"], evaluations=engine.calls-before))
            if progress is not None:
                progress(f"conditional iteration {iteration+1}, regret={regret:.3g}")
            if regret <= self.dp_tol:
                break
            alpha = proposal
        initial = alpha.copy()
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
                    conditional = conditional_from_joint(stationary["nu"], self.game.D)
                    checked = evaluate_policy(self.grid, conditional["conditional_demand"], rule, self.params)
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
            exploratory_signed_roundoff_calls=engine.signed_roundoff_search_calls,
            exploratory_positive_fallback_calls=engine.positive_fallback_search_calls,
            exploratory_stationary_note="Signed roundoff may be used only during root search when row conditioning is valid; final certificates use positive stationary laws and independent conditional-order checks",
            budget_interpretation="Soft exploratory-call budget; root callbacks, final certification and up to 60 strong-polish evaluations may exceed it",
            method="Damped conditional policy iteration followed by analytic conditional complementarity continuation",
            damping=float(damping), picard_iterations=int(picard_iterations),
            exhaustive=False, unique=False,
            scope="Finite grid and numerical demand matrix; no continuous-state or MPE claim"))
        best["converged"] = bool(best["verified"])
        return best
