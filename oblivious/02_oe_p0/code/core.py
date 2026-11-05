"""Scalar consistency, evaluated in the existing two-seller TS game.

The one-dimensional optimization uses a fixed benchmark. Mixtures are allowed
only at conditional action-value indifference. Evaluation uses the actual
two-dimensional TS demand and calendar transition kernel. A verified scalar
fixed point is not a Nash equilibrium of that full-state game.
"""

from dataclasses import asdict
import hashlib
import warnings
from itertools import product

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.stats import beta
from scipy.optimize import brentq

from oblivious.oe_p0.code.game import Game
from oblivious.oe_p0.code.primitives import Parameters, posterior, demand_matrix
from oblivious.oe_p0.code.policy import demand_probability, evaluate_policy, solve_dp
from oblivious.oe_p0.code.positive_stationary import positive_invariant as _positive_invariant


def posterior_mean(x, params):
    """Long-run TS means average this quantity, not realized product quality."""
    x = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("Reputations must be finite and in [0,1].")
    return (1 - params.rho + x) / (3 - 2 * params.rho)


def fixed_demand(grid, p0, params):
    """Upper Beta tail; no posterior-mean threshold approximation."""
    if not np.isfinite(p0) or not 0 <= p0 <= 1:
        raise ValueError("p0 must be finite and in [0,1].")
    a, b = posterior(grid, params)
    return beta.sf(p0, a, b)


def _jump_system(matrix):
    off = matrix.copy().tocsr()
    off.setdiag(0)
    off.eliminate_zeros()
    rates = np.asarray(off.sum(axis=1)).ravel()
    if np.any(rates <= 0):
        raise ArithmeticError("Zero exit rate: no unique full-support law can be certified.")
    components, _ = connected_components(off, directed=True, connection="strong")
    if components != 1:
        raise ArithmeticError(f"Kernel has {components} communicating components; stationary selection is unresolved.")
    return (diags(1 / rates) @ off).tocsr(), rates


def _sparse_invariant(jump, rates, anchor):
    n = len(rates)
    linear = (jump.T - eye(n, format="csr")).tolil()
    linear[anchor, :] = np.ones(n)
    rhs = np.zeros(n)
    rhs[anchor] = 1
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        law = spsolve(linear.tocsr(), rhs)
    if np.any(~np.isfinite(law)) or law.min() < -1e-10:
        raise ArithmeticError("Sparse invariant solve produced invalid mass.")
    law = np.maximum(law, 0) / rates
    if not np.isfinite(law.sum()) or law.sum() <= 0:
        raise ArithmeticError("Invalid calendar-time normalization.")
    return law / law.sum()


def checked_joint_stationary(game, P, tol=1e-11, stability_tol=1e-7,
                             force_positive=None):
    """Check a joint invariant law with independent numerical orderings.

    Symmetric kernels can be lumped exactly onto unordered state pairs. This
    removes the slow antisymmetric mode without changing the symmetric law.
    The returned law is checked against the original, unmodified full kernel.
    State-graph connectivity and ordering agreement supplement the residual;
    neither a mixing-time bound nor a continuous-state guarantee is claimed.
    """
    if tol <= 0 or stability_tol <= 0:
        raise ValueError("Positive stationarity and stability tolerances required.")
    matrix = csr_matrix(P)
    n, size = game.n, game.n ** 2
    if (matrix.shape != (size, size) or np.any(~np.isfinite(matrix.data))
            or matrix.data.min(initial=0) < 0):
        raise ValueError("Joint kernel must be finite, nonnegative, and (n*n,n*n).")
    row_error = float(np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel() - 1)))
    if row_error > 1e-12:
        raise ValueError(f"Joint kernel is not row-stochastic: {row_error:g}.")

    i, j = np.triu_indices(n)
    representatives, swapped = i * n + j, j * n + i
    groups = np.empty((n, n), dtype=int)
    groups[i, j] = np.arange(len(i))
    groups[j, i] = np.arange(len(i))
    membership = csr_matrix((np.ones(size), (np.arange(size), groups.ravel())),
                            shape=(size, len(i)))
    lumped = (matrix[representatives] @ membership).tocsr()
    difference = lumped - matrix[swapped] @ membership
    lump_error = float(np.max(np.abs(difference.data), initial=0))
    symmetric = lump_error <= 1e-13
    reduced = lumped if symmetric else matrix
    jump, rates = _jump_system(reduced)
    full_components, _ = connected_components(matrix, directed=True, connection="strong")
    methods = []
    failures = []
    # Independent normalization rows and reversed state numbering expose
    # conditioning problems that a small invariant residual can conceal.
    try:
        first = _sparse_invariant(jump, rates, len(rates) // 2)
        methods.append("sparse jump-chain, middle normalization")
        perm = np.arange(len(rates))[::-1]
        reverse = _sparse_invariant(jump[perm, :][:, perm], rates[perm], 0)
        second = np.empty_like(reverse)
        second[perm] = reverse
        discrepancy = float(np.sum(np.abs(first - second)))
        methods.append("sparse jump-chain, reversed order and edge normalization")
        residual = float(np.sum(np.abs(reduced.T @ first - first)))
        if discrepancy > stability_tol or residual > tol:
            raise ArithmeticError(f"Sparse stability discrepancy={discrepancy:g}, residual={residual:g}.")
        # At high persistence even two signed solves can agree after losing
        # rare flows. Always compare against subtraction-free elimination.
        # Elsewhere use it if the signed answers approach the error budget.
        require_positive = game.params.rho >= .99 if force_positive is None else force_positive
        if require_positive or (force_positive is not False
                                and discrepancy > min(1e-9, stability_tol / 100)):
            positive = _positive_invariant(jump, rates)
            discrepancy = max(discrepancy, float(np.sum(np.abs(first - positive))))
            methods.append("independent positive GTH elimination")
            if discrepancy > stability_tol:
                raise ArithmeticError(f"Sparse/GTH stationary discrepancy={discrepancy:g}.")
            first = positive
    except (ArithmeticError, RuntimeError, MatrixRankWarning) as exc:
        failures.append(str(exc))
        first = _positive_invariant(jump, rates)
        second = _positive_invariant(jump, rates, np.arange(len(rates))[::-1])
        methods += ["positive GTH elimination", "positive GTH elimination, reversed order"]
        discrepancy = float(np.sum(np.abs(first - second)))
    if discrepancy > stability_tol:
        raise ArithmeticError(f"Stationary law is numerically unstable: independent-order L1={discrepancy:g}.")
    if symmetric:
        nu = np.zeros((n, n))
        shares = first / np.where(i == j, 1, 2)
        nu[i, j] = shares
        nu[j, i] = shares
    else:
        nu = first.reshape(n, n)
    residual = float(np.sum(np.abs(matrix.T @ nu.ravel() - nu.ravel())))
    if residual > tol or abs(nu.sum() - 1) > tol or nu.min() < 0:
        raise ArithmeticError(f"Full joint invariant-law verification failed; residual={residual:g}.")
    summary = dict(verified=True, residual=residual, row_sum_error=row_error,
                   probability_sum=float(nu.sum()), minimum_probability=float(nu.min()),
                   stability_l1=discrepancy, stability_tolerance=stability_tol,
                   methods=methods, preliminary_failures=failures,
                   symmetric_lumping=symmetric, lumping_error=lump_error,
                   solved_states=len(rates), full_states=size,
                   selection="symmetric invariant law" if symmetric else "full joint invariant law",
                   communicating_components=int(full_components),
                   reduced_communicating_components=1,
                   minimum_exit_rate=float(rates.min()),
                   exploratory=force_positive is False,
                   marginal_symmetry_l1=float(np.sum(np.abs(nu.sum(axis=1) - nu.sum(axis=0)))),
                   scope="Finite numerical kernel; no mixing-time or continuous-state guarantee")
    return dict(nu=nu, mu_a=nu.sum(axis=1), mu_b=nu.sum(axis=0), summary=summary)


stationary_joint = checked_joint_stationary


def _signature(evaluation):
    policy = np.asarray(evaluation.get("policy", evaluation.get("policy0", [])), dtype=float)
    return hashlib.sha256(policy.tobytes()).hexdigest()


def find_fixed_points(evaluator, bounds, p0_points=41, consistency_tol=1e-7,
                      max_iterations=60):
    """Search piecewise policy plateaus; explicitly reject sign-change jumps.

    Every reported candidate is freshly evaluated at its own proposed p0.
    Multiple starts and every discovered policy-generated F value are tried.
    This is a finite numerical search, not an exhaustive existence theorem.
    """
    lo, hi = map(float, bounds)
    if not 0 <= lo < hi <= 1 or p0_points < 3 or consistency_tol <= 0:
        raise ValueError("Require valid bounds, >=3 scan points, and positive tolerance.")
    cache, failures, search_log = {}, [], []

    def evaluate(p0):
        p0 = float(np.clip(p0, lo, hi))
        key = float(p0)
        if key not in cache:
            try:
                result = evaluator(p0)
                result = {**result, "p0": p0, "F": float(result["F"])}
                result["G"] = result["F"] - p0
                if not result.get("verified", True) or not np.isfinite(result["F"]):
                    raise ArithmeticError("Evaluation did not verify.")
                result["candidate_verified"] = abs(result["G"]) <= consistency_tol
                cache[key] = result
            except (ArithmeticError, RuntimeError, ValueError) as exc:
                failures.append(dict(p0=p0, error=str(exc)))
                cache[key] = None
        return cache[key]

    scan = [evaluate(x) for x in np.linspace(lo, hi, p0_points)]
    # On a pure-policy plateau F is constant. Evaluate its generated value
    # directly, then iterate if the resulting policy differs.
    seen_policies = set()

    def complete_discovered_plateaus():
        for _ in range(max_iterations):
            fresh = [v for v in list(cache.values()) if v is not None
                     and _signature(v) not in seen_policies]
            if not fresh:
                break
            for result in fresh:
                signature = _signature(result)
                if signature in seen_policies:
                    continue
                seen_policies.add(signature)
                generated = evaluate(result["F"])
                search_log.append(dict(method="policy-plateau generated value", start=result["p0"],
                                       proposed_p0=result["F"],
                                       verified=bool(generated and generated["candidate_verified"])))

    complete_discovered_plateaus()

    for start in np.linspace(lo, hi, 7):
        p0 = float(start)
        history = []
        for iteration in range(min(max_iterations, 30)):
            result = evaluate(p0)
            if result is None or result["candidate_verified"]:
                break
            updated = .5 * (p0 + result["F"])
            history.append(float(abs(result["G"])))
            if abs(updated - p0) < consistency_tol / 10:
                break
            p0 = updated
        search_log.append(dict(method="damped fixed-point iteration", start=float(start),
                               iterations=len(history), final_p0=p0,
                               final_residual=None if result is None else abs(result["G"])))

    # Bisection is only a localization device. At a jump, its small interval
    # is not a root certificate. Check G directly and record both one-sided
    # values when the interval becomes smaller than numerical resolution.
    brackets = []
    for left, right in zip(scan[:-1], scan[1:]):
        if left is None or right is None or left["G"] * right["G"] >= 0:
            continue
        initial = [left["p0"], right["p0"]]
        found = False
        for iteration in range(max_iterations):
            middle = evaluate(.5 * (left["p0"] + right["p0"]))
            if middle is None:
                break
            if middle["candidate_verified"]:
                found = True
                break
            if left["G"] * middle["G"] < 0:
                right = middle
            else:
                left = middle
            if right["p0"] - left["p0"] <= max(1e-12, consistency_tol * .01):
                break
        brackets.append(dict(initial_bracket=initial, final_bracket=[left["p0"], right["p0"]],
                             left_G=left["G"], right_G=right["G"],
                             left_F=left["F"], right_F=right["F"],
                             candidate_detected=found,
                             policy_jump=_signature(left) != _signature(right),
                             status="candidate verified" if found else "sign change without verified root"))

    # Refinement can reveal narrow policy plateaus absent from the coarse scan.
    # Try every newly discovered plateau's own generated value as well.
    complete_discovered_plateaus()
    evaluations = sorted((v for v in cache.values() if v is not None), key=lambda r: r["p0"])
    candidates = []
    # Exact plateau roots are preferable to nearby approximate bisection hits.
    for result in sorted((v for v in evaluations if v["candidate_verified"]), key=lambda r: abs(r["G"])):
        if not any(abs(result["p0"] - old["p0"]) <= 5 * consistency_tol for old in candidates):
            candidates.append(result)
    candidates.sort(key=lambda r: r["p0"])
    best = min(evaluations, key=lambda r: abs(r["G"])) if evaluations else None
    return dict(evaluations=evaluations, candidates=candidates, best_diagnostic=best,
                failures=failures, search_log=search_log, brackets=brackets,
                summary=dict(verified_candidates=len(candidates), evaluations=len(evaluations),
                             failed_evaluations=len(failures), detected_policies=len(seen_policies),
                             minimum_consistency_residual=None if best is None else abs(best["G"]),
                             consistency_tolerance=consistency_tol, p0_bounds=[lo, hi],
                             scan_points=p0_points,
                             completeness="Detected numerical candidates only; uniqueness and exhaustive coverage are not proved"))


class ScalarExperiment:
    """One rho/grid experiment with mixed own-state product policies."""

    def __init__(self, n=21, params=None, order=128, dp_tol=1e-10,
                 stationary_tol=1e-11, stability_tol=1e-7, near_tie_tol=1e-7):
        self.params = Parameters() if params is None else params
        if n < 3 or int(n) != n or min(dp_tol, stationary_tol, stability_tol, near_tie_tol) <= 0:
            raise ValueError("Require integer n>=3 and positive tolerances.")
        self.grid = np.linspace(0, 1, int(n))
        self.n = int(n)
        self.game = Game(self.grid, self.params, order=order)
        self.dp_tol, self.stationary_tol = dp_tol, stationary_tol
        self.stability_tol, self.near_tie_tol = stability_tol, near_tie_tol
        self.p0_bounds = (float(posterior_mean(0, self.params)), float(posterior_mean(1, self.params)))
        self._policy_cache = {}
        self._evaluations = {}
        self._demand_audit = None
        self._optimal_cache = {}

    def demand_audit(self, tolerance=1e-6):
        """Audit the reused actual-game demand once, independently of symmetry enforcement."""
        if tolerance <= 0:
            raise ValueError("Demand audit tolerance must be positive.")
        if self._demand_audit is not None:
            if self._demand_audit["maximum_error"] > tolerance:
                raise ArithmeticError("Cached demand audit exceeds requested tolerance.")
            return self._demand_audit
        doubled, raw = demand_matrix(self.grid, self.params, order=2 * self.game.order)
        doubling = float(np.max(np.abs(doubled - self.game.D)))
        pairs = [(0, self.n - 1), (round(.25 * (self.n - 1)), round(.75 * (self.n - 1))),
                 (max(0, self.n // 2 - 1), min(self.n - 1, self.n // 2 + 1)),
                 (round(.8 * (self.n - 1)), round(.4 * (self.n - 1))),
                 (self.n // 2, self.n // 2)]
        checks = []
        for i, j in pairs:
            x, y = float(self.grid[i]), float(self.grid[j])
            direct = demand_probability(x, y, self.params)
            reverse = demand_probability(y, x, self.params)
            checks.append(dict(x=x, y=y, matrix_probability=float(self.game.D[i, j]),
                               direct_probability=direct, reverse_probability=reverse,
                               matrix_error=abs(direct - self.game.D[i, j]),
                               independent_symmetry_error=abs(direct + reverse - 1),
                               independent_diagonal_error=abs(direct - .5) if i == j else None))
        maximum = max(doubling, self.game.demand_diagnostics["raw_diagonal_error"],
                      self.game.demand_diagnostics["raw_symmetry_error"],
                      raw["raw_diagonal_error"], raw["raw_symmetry_error"],
                      max(check["matrix_error"] for check in checks),
                      max(check["independent_symmetry_error"] for check in checks))
        self._demand_audit = dict(verified=maximum <= tolerance, tolerance=tolerance,
                                  maximum_error=maximum, doubled_order=2 * self.game.order,
                                  doubled_order_max_difference=doubling,
                                  original_raw=self.game.demand_diagnostics, doubled_raw=raw,
                                  independent_checks=checks,
                                  independent_method="Adaptive quadrature with stable scipy.stats.beta.pdf")
        if maximum > tolerance:
            raise ArithmeticError(f"Actual TS demand failed independent audit: {maximum:g} > {tolerance:g}.")
        return self._demand_audit

    def evaluate(self, p0):
        p0 = float(p0)
        if p0 in self._evaluations:
            return self._evaluations[p0]
        demand = fixed_demand(self.grid, p0, self.params)
        solution = solve_dp(self.grid, demand, self.params, tol=self.dp_tol)
        policy = solution["policy"]
        checked = evaluate_policy(self.grid, demand, policy, self.params)
        advantage, value = checked["advantage"], checked["value"]
        regret = np.maximum(advantage, 0) - policy * advantage
        bellman = checked["residual"] + float(np.max(demand * regret))
        conditional_regret = float(np.max(regret))
        if bellman > self.dp_tol or conditional_regret > self.dp_tol:
            raise ArithmeticError(f"Fixed-benchmark DP failed independent verification: Bellman={bellman:g}, conditional regret={conditional_regret:g}.")
        key = policy.astype(np.uint8).tobytes()
        cached = key in self._policy_cache
        if not cached:
            sigma_a = np.broadcast_to(policy[:, None], (self.n, self.n)).copy()
            sigma_b = np.broadcast_to(policy[None, :], (self.n, self.n)).copy()
            matrix = self.game.transition(sigma_a, sigma_b)
            invariant = checked_joint_stationary(self.game, matrix, self.stationary_tol, self.stability_tol)
            self._policy_cache[key] = invariant
        invariant = self._policy_cache[key]
        mu_a, mu_b = invariant["mu_a"], invariant["mu_b"]
        generated = self._generated_scalar(policy, invariant)
        near_ties = np.flatnonzero(np.abs(advantage) <= self.near_tie_tol)
        dp_summary = dict(verified=True, bellman_residual=bellman,
                          conditional_regret=conditional_regret,
                          policy_evaluation_residual=checked["residual"],
                          iterations=int(solution["iterations"]),
                          tie_breaking="Product 2 on exact advantage >= 0; no threshold imposed",
                          near_tie_tolerance=self.near_tie_tol,
                          near_tie_indices=near_ties.tolist(),
                          near_tie_reputations=self.grid[near_ties].tolist())
        result = dict(p0=p0, F=generated, G=generated - p0, verified=True,
                      grid=self.grid, value=value, policy=policy, policy0=policy,
                      product_policy=(policy + 1).astype(int), advantage=advantage,
                      d_fixed=demand, nu=invariant["nu"], mu_a=mu_a, mu_b=mu_b,
                      mu_A=mu_a, mu_B=mu_b, dp_diagnostics=dp_summary,
                      stationary_diagnostics=invariant["summary"], stationary_cached=cached,
                      summary=dict(p0=p0, F=generated, G=generated - p0,
                                   product2_nodes=int(policy.sum()),
                                   average_reputation_a=float(mu_a @ self.grid),
                                   average_reputation_b=float(mu_b @ self.grid),
                                   dp=dp_summary, stationary=invariant["summary"]))
        self._evaluations[p0] = result
        return result

    def search_pure(self, p0_points=41, consistency_tol=1e-7):
        """Deterministic endpoint search used to discover optimal-policy ties."""
        return find_fixed_points(self.evaluate, self.p0_bounds, p0_points, consistency_tol)

    def _generated_scalar(self, policy, invariant):
        """Original mean-TS closure; separate experiments may supply another closure."""
        return float(invariant["mu_b"] @ posterior_mean(self.grid, self.params))

    def _exploratory_scalar(self, policy):
        """Evaluate a tie-face proposal without promoting it to a certificate."""
        sigma_a = np.broadcast_to(policy[:, None], (self.n, self.n))
        sigma_b = np.broadcast_to(policy[None, :], (self.n, self.n))
        invariant = checked_joint_stationary(
            self.game, self.game.transition(sigma_a, sigma_b),
            self.stationary_tol, self.stability_tol, force_positive=False)
        return self._generated_scalar(policy, invariant), invariant["summary"]

    def _optimal(self, p0):
        key = float(p0)
        if key not in self._optimal_cache:
            demand = fixed_demand(self.grid, key, self.params)
            result = solve_dp(self.grid, demand, self.params, tol=min(self.dp_tol, 1e-12))
            checked = evaluate_policy(self.grid, demand, result["policy"], self.params)
            self._optimal_cache[key] = {**result, **checked}
        return self._optimal_cache[key]

    def evaluate_mixed(self, p0, alpha, consistency_tol=1e-7):
        """Fresh certificate for any proposed scalar benchmark and mixed policy.

        ``verified`` checks DP optimality and the actual-game invariant law.
        ``candidate_verified`` additionally requires scalar consistency. A
        fractional policy is not valid merely because its consistency is small:
        every fractional node must independently satisfy action indifference.
        """
        p0 = float(p0)
        alpha = np.asarray(alpha, dtype=float)
        if (alpha.shape != self.grid.shape or np.any(~np.isfinite(alpha))
                or np.any(alpha < 0) or np.any(alpha > 1) or consistency_tol <= 0):
            raise ValueError("alpha must be a finite probability on each grid node; tolerance must be positive.")
        demand = fixed_demand(self.grid, p0, self.params)
        checked = evaluate_policy(self.grid, demand, alpha, self.params)
        response = solve_dp(self.grid, demand, self.params, tol=self.dp_tol)
        advantage, value = checked["advantage"], checked["value"]
        regret = np.maximum(advantage, 0) - alpha * advantage
        conditional = float(np.max(regret))
        complementarity = float(np.max(np.abs(alpha - np.clip(alpha + advantage, 0, 1))))
        interior = np.flatnonzero((alpha > 0) & (alpha < 1))
        indifference = float(np.max(np.abs(advantage[interior]), initial=0))
        bellman = checked["residual"] + float(np.max(demand * regret))
        gain = response["value"] - value
        numerical_bound = (checked["residual"] + response["residual"]) / (1 - self.params.gamma)
        gain_bound = max(0., float(np.max(gain))) + numerical_bound
        dp_verified = (checked["residual"] <= self.dp_tol and bellman <= self.dp_tol
                       and conditional <= self.dp_tol and complementarity <= self.dp_tol
                       and indifference <= self.dp_tol and gain_bound <= consistency_tol)
        sigma_a = np.broadcast_to(alpha[:, None], (self.n, self.n))
        sigma_b = np.broadcast_to(alpha[None, :], (self.n, self.n))
        invariant = checked_joint_stationary(self.game, self.game.transition(sigma_a, sigma_b),
                                             self.stationary_tol, self.stability_tol)
        mu_a, mu_b = invariant["mu_a"], invariant["mu_b"]
        generated = self._generated_scalar(alpha, invariant)
        near_ties = np.flatnonzero(np.abs(advantage) <= self.near_tie_tol)
        dp_summary = dict(verified=bool(dp_verified), bellman_residual=bellman,
                          conditional_regret=conditional, complementarity_residual=complementarity,
                          mixed_indifference_residual=indifference,
                          policy_evaluation_residual=checked["residual"],
                          best_response_gain_upper_bound=gain_bound,
                          best_response_numerical_error_bound=numerical_bound,
                          near_tie_tolerance=self.near_tie_tol,
                          near_tie_indices=near_ties.tolist(),
                          near_tie_reputations=self.grid[near_ties].tolist(),
                          tie_breaking="Mixtures allowed only at verified conditional indifference",
                          iterations=int(response["iterations"]))
        result = dict(p0=p0, F=generated, G=generated - p0, verified=bool(dp_verified),
                      candidate_verified=bool(dp_verified and abs(generated - p0) <= consistency_tol),
                      strategy_mode="mixed" if len(interior) else "pure endpoint",
                      grid=self.grid, value=value, policy=alpha.copy(), policy0=alpha.copy(),
                      product_policy=None if len(interior) else (1 + alpha).astype(int),
                      expected_product_index=1 + alpha,
                      advantage=advantage, d_fixed=demand,
                      nu=invariant["nu"], mu_a=mu_a, mu_b=mu_b, mu_A=mu_a, mu_B=mu_b,
                      mixed_nodes=len(interior), mixed_indices=interior.tolist(),
                      mixed_probabilities=alpha[interior].tolist(),
                      dp_diagnostics=dp_summary, stationary_diagnostics=invariant["summary"],
                      stationary_cached=False,
                      summary=dict(p0=p0, F=generated, G=generated - p0,
                                   mixed_nodes=len(interior), product2_nodes=int(np.sum(alpha == 1)),
                                   average_reputation_a=float(mu_a @ self.grid),
                                   average_reputation_b=float(mu_b @ self.grid),
                                   dp=dp_summary, stationary=invariant["summary"]))
        return result

    def search(self, p0_points=41, consistency_tol=1e-7, pure_seed=None, progress=None):
        """Find pure endpoints and mixtures on discovered optimal-action tie faces.

        Pure DP scans expose policy switches. At each located switch, p0 is
        solved from an optimal action advantage of zero; only independently
        tied nodes are randomized. Five points per face path detect sign
        changes, then scalar refinement calibrates the actual joint TS law.
        Full certification is separate from exploratory invariant calculations.
        This finite branch search does not establish exhaustive coverage.
        """
        def report(message):
            if progress is not None:
                progress(message)

        if pure_seed is None:
            pure = self.search_pure(p0_points, consistency_tol)
            rows = pure["evaluations"]
            prior_candidates = pure["candidates"]
            best = pure["best_diagnostic"]
        else:
            configuration = pure_seed.get("configuration", {})
            if (configuration.get("parameters") != asdict(self.params)
                    or configuration.get("n") != self.n
                    or configuration.get("order") != self.game.order):
                raise ValueError("Pure-search seed parameters/grid/quadrature do not match this experiment.")
            pure = dict(summary=pure_seed["search_summary"], brackets=pure_seed["brackets"],
                        search_log=pure_seed.get("search_log", []), failures=pure_seed.get("failures", []))
            rows = pure_seed["map_rows"]
            prior_candidates = pure_seed.get("candidates", [])
            best = pure_seed.get("selected")
        failures = list(pure.get("failures", []))
        history = list(pure.get("search_log", []))
        evaluations = [{**row, "strategy_mode": "pure DP scan"} for row in rows]
        candidates, branches = [], []
        if best is not None:
            best = {**best, "mixed_nodes": 0, "mixed_indices": [], "mixed_probabilities": [],
                    "strategy_mode": "pure DP diagnostic", "candidate_verified": False}

        def retain(candidate):
            nonlocal best
            if best is None or (candidate["verified"] and abs(candidate["G"]) < abs(best["G"])):
                best = candidate
            if candidate["candidate_verified"] and not any(
                    abs(candidate["p0"] - old["p0"]) <= 5 * consistency_tol
                    and np.max(np.abs(candidate["policy"] - old["policy"])) <= 1e-6
                    for old in candidates):
                candidates.append(candidate)

        for old in prior_candidates:
            try:
                fresh = self.evaluate_mixed(old["p0"], old["policy"], consistency_tol)
                retain(fresh)
                evaluations.append(fresh)
            except (ArithmeticError, RuntimeError, ValueError) as exc:
                failures.append(dict(stage="pure endpoint recertification", p0=old["p0"], error=str(exc)))

        # No joint laws are recomputed while reconstructing the old pure scan:
        # only its cheap one-dimensional optimal policy is needed for switches.
        points = sorted(set(float(row["p0"]) for row in rows))
        optimal = {}
        for point in points:
            try:
                optimal[point] = self._optimal(point)
            except (ArithmeticError, RuntimeError, ValueError) as exc:
                failures.append(dict(stage="optimal-policy reconstruction", p0=point, error=str(exc)))
        report(f"Reconstructed optimal policies at {len(optimal)} scalar values")
        tie_points = []
        for left, right in zip(points[:-1], points[1:]):
            if left not in optimal or right not in optimal:
                continue
            changed = np.flatnonzero(optimal[left]["policy"] != optimal[right]["policy"])
            for node in changed:
                try:
                    def advantage_at(point):
                        return float(self._optimal(point)["advantage"][node])
                    a, b = advantage_at(left), advantage_at(right)
                    if a * b > 0:
                        continue
                    tie = float(brentq(advantage_at, left, right, xtol=5e-15, rtol=1e-14))
                    if not any(abs(tie - old) <= 2e-11 for old in tie_points):
                        tie_points.append(tie)
                except (ArithmeticError, RuntimeError, ValueError) as exc:
                    failures.append(dict(stage="action-tie localization", bracket=[left, right],
                                         node=int(node), error=str(exc)))
        tie_tolerance = min(self.dp_tol / 5, 1e-11)
        report(f"Located {len(tie_points)} action-tie points; searching optimal mixing faces")
        for tie in sorted(tie_points):
            solution = self._optimal(tie)
            tied = np.flatnonzero(np.abs(solution["advantage"]) <= tie_tolerance)
            if not len(tied):
                failures.append(dict(stage="action-tie verification", p0=tie,
                                     error="No node met the strict action-indifference tolerance"))
                continue
            base = solution["policy"].copy()
            k = len(tied)
            # Enumerate all edges for small tied faces; larger faces use their
            # main diagonal and axes. Every fractional node is a genuine tie.
            paths = [(np.zeros(k), np.ones(k))]
            if k > 1 and k <= 3:
                for vertex in product((0., 1.), repeat=k):
                    for axis in range(k):
                        if vertex[axis] == 0:
                            target = np.array(vertex)
                            target[axis] = 1
                            paths.append((np.array(vertex), target))
            elif k > 3:
                for axis in range(k):
                    target = np.zeros(k)
                    target[axis] = 1
                    paths.append((np.zeros(k), target))
            for start, end in paths:
                branch = dict(p0=tie, tied_indices=tied.tolist(), tied_reputations=self.grid[tied].tolist(),
                              start_probabilities=start.tolist(), end_probabilities=end.tolist(),
                              trials=[], candidates=[], exploratory=True,
                              scope="Five-point scan along a feasible optimal-policy face path; not exhaustive")
                branches.append(branch)
                if len(branches) == 1 or len(branches) % 10 == 0:
                    report(f"Searching mixing branch {len(branches)}; {len(candidates)} candidates certified so far")
                cache = {}

                def trial(weight, certify=False):
                    weight = float(weight)
                    alpha = base.copy()
                    alpha[tied] = (1 - weight) * start + weight * end
                    if certify:
                        return self.evaluate_mixed(tie, alpha, consistency_tol)
                    if weight not in cache:
                        generated, stationary_summary = self._exploratory_scalar(alpha)
                        cache[weight] = generated - tie
                        branch["trials"].append(dict(weight=weight, p0=tie, F=generated, G=generated - tie,
                                                      mixing_probabilities=alpha[tied].tolist(),
                                                      exploratory=True,
                                                      stationary=stationary_summary))
                    return cache[weight]

                try:
                    weights = np.linspace(0, 1, 5)
                    residuals = [trial(weight) for weight in weights]
                    proposals = [float(w) for w, g in zip(weights, residuals) if abs(g) <= consistency_tol]
                    for w0, w1, g0, g1 in zip(weights[:-1], weights[1:], residuals[:-1], residuals[1:]):
                        if g0 * g1 < 0:
                            proposals.append(float(brentq(trial, w0, w1, xtol=1e-11, rtol=1e-11)))
                    for weight in proposals:
                        candidate = trial(weight, certify=True)
                        # If strong stationarity shifts the preliminary root,
                        # refine with strong certificates rather than promoting
                        # an exploratory result or silently relaxing tolerance.
                        if candidate["verified"] and not candidate["candidate_verified"]:
                            strong_cache = {weight: candidate}

                            def strong_residual(w):
                                if w not in strong_cache:
                                    strong_cache[w] = trial(w, certify=True)
                                return strong_cache[w]["G"]
                            radius = 1e-4
                            for _ in range(5):
                                w0, w1 = max(0., weight - radius), min(1., weight + radius)
                                if strong_residual(w0) * strong_residual(w1) <= 0:
                                    weight = float(brentq(strong_residual, w0, w1, xtol=1e-11, rtol=1e-11))
                                    candidate = strong_cache.get(weight) or trial(weight, certify=True)
                                    break
                                radius *= 10
                        candidate["mixture_search_weight"] = weight
                        candidate["tie_face_dimension"] = k
                        retain(candidate)
                        evaluations.append(candidate)
                        branch["candidates"].append(dict(weight=weight, verified=candidate["candidate_verified"],
                                                          G=candidate["G"],
                                                          mixed_probabilities=candidate["mixed_probabilities"]))
                        if not candidate["candidate_verified"]:
                            failures.append(dict(stage="mixed candidate certification", p0=tie,
                                                 G=candidate["G"], diagnostics=candidate["dp_diagnostics"]))
                except (ArithmeticError, RuntimeError, ValueError) as exc:
                    branch["failure"] = str(exc)
                    failures.append(dict(stage="mixed-face search", p0=tie,
                                         tied_indices=tied.tolist(), error=str(exc)))
        candidates.sort(key=lambda item: (item["p0"], tuple(item["policy"])))
        if candidates:
            best = min(candidates, key=lambda item: abs(item["G"]))
        report(f"Certified {len(candidates)} candidates, {sum(c['mixed_nodes'] > 0 for c in candidates)} with interior mixing")
        history.append(dict(method="optimal tie-face mixed search", localized_ties=len(tie_points),
                            branches=len(branches), reused_pure_seed=pure_seed is not None,
                            exploratory_stationarity="Independent sparse orderings with GTH fallback",
                            final_stationarity="Fresh strong certificate, mandatory high-rho GTH"))
        return dict(evaluations=sorted(evaluations, key=lambda e:e["p0"]), candidates=candidates,
                    best_diagnostic=best, failures=failures, search_log=history,
                    brackets=pure["brackets"], mixing_branches=branches,
                    summary=dict(verified_candidates=len(candidates),
                                 verified_mixed_candidates=sum(c["mixed_nodes"] > 0 for c in candidates),
                                 evaluations=len(evaluations), failed_evaluations=len(failures),
                                 minimum_consistency_residual=None if best is None else abs(best["G"]),
                                 consistency_tolerance=consistency_tol, p0_bounds=list(self.p0_bounds),
                                 scan_points=p0_points, localized_ties=len(tie_points),
                                 mixing_branches=len(branches), reused_pure_seed=pure_seed is not None,
                                 strategy_space="Mixed actions conditional on own reputation",
                                 completeness="Detected candidates only; finite policy scan and tie-face paths do not prove uniqueness or exhaustive coverage"))

    def configuration(self):
        return dict(parameters=asdict(self.params), n=self.n, order=self.game.order,
                    dp_tolerance=self.dp_tol, stationary_tolerance=self.stationary_tol,
                    stationary_stability_tolerance=self.stability_tol,
                    near_tie_tolerance=self.near_tie_tol)
