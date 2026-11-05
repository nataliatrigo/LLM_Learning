"""Shared demand, mixed-policy evaluation and stationary-law helpers.

Both active experiments use these numerical routines for their own demand
specification, without importing other experiment packages.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import spsolve
from scipy.stats import beta


def check_grid(grid):
    grid = np.asarray(grid, dtype=float)
    if (grid.ndim != 1 or len(grid) < 3 or grid[0] != 0 or grid[-1] != 1
            or not np.all(np.diff(grid) > 0)):
        raise ValueError("Grid must increase strictly from 0 to 1, with >=3 nodes.")
    return grid


def posterior(x, params):
    x = np.asarray(x)
    if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("Reputations must lie in [0,1].")
    return 1 + x / (1 - params.rho), 1 + (1 - x) / (1 - params.rho)


def demand_probability(x, y, params):
    """Independent adaptive integration, used to audit the faster matrix."""
    a, b = posterior(float(x), params)
    c, d = posterior(float(y), params)

    def integrand(t):
        if t == 0 or t == 1:
            return 0.0  # Endpoint values do not affect the integral.
        # SciPy's beta density uses Boost's stable implementation. Expanding the
        # log density by hand loses normalization precision for concentrated
        # posteriors (rho close to one), even when quad's own error is tiny.
        pdf = beta.pdf(t, a, b)
        return pdf * betainc(c, d, t)

    result, error = quad(integrand, 0, 1, epsabs=2e-12, epsrel=2e-12, limit=300)
    if error > 1e-9 or not -1e-12 <= result <= 1 + 1e-12:
        raise ArithmeticError(f"Demand quadrature failed: error={error:g}.")
    return float(np.clip(result, 0, 1))


def interpolation_matrix(grid, targets):
    """Rows allocate off-grid targets between their adjacent grid nodes."""
    grid = check_grid(grid)
    targets = np.asarray(targets, dtype=float).ravel()
    if np.any(~np.isfinite(targets)) or np.any((targets < -1e-14) | (targets > 1+1e-14)):
        raise ValueError("Interpolation targets must lie in [0,1].")
    targets = np.clip(targets, 0, 1)
    lo = np.clip(np.searchsorted(grid, targets, side="right") - 1, 0, len(grid) - 2)
    w = (targets - grid[lo]) / (grid[lo + 1] - grid[lo])
    rows = np.repeat(np.arange(len(targets)), 2)
    return csr_matrix((np.column_stack((1-w, w)).ravel(),
                       (rows, np.column_stack((lo, lo+1)).ravel())),
                      shape=(len(targets), len(grid)))


def _check_probabilities(vector, grid, name):
    vector = np.asarray(vector, dtype=float)
    if (vector.shape != grid.shape or np.any(~np.isfinite(vector))
            or np.any((vector < -1e-14) | (vector > 1+1e-14))):
        raise ValueError(f"{name} must be a probability vector on the grid.")
    # Dot products of probabilities (D @ mu) can exceed one by roundoff even
    # when mu comes from a normalized softmax. Only remove that numerical dust;
    # substantive invalid probabilities remain errors.
    return np.clip(vector, 0, 1)


def transition_matrix(grid, demand, policy, params):
    """Calendar kernel of a mixed rule, including unselected self-loops."""
    grid = check_grid(grid)
    demand = _check_probabilities(demand, grid, "Demand")
    policy = _check_probabilities(policy, grid, "Policy")
    success = params.p1 + policy * (params.p2 - params.p1)
    S = interpolation_matrix(grid, params.rho * grid + 1 - params.rho)
    F = interpolation_matrix(grid, params.rho * grid)
    return (diags(1-demand) + diags(demand*success) @ S
            + diags(demand*(1-success)) @ F).tocsr()


def evaluate_policy(grid, demand, policy, params):
    """Fresh sparse evaluation of an arbitrary feasible mixed policy."""
    grid = check_grid(grid)
    demand = _check_probabilities(demand, grid, "Demand")
    policy = _check_probabilities(policy, grid, "Policy")
    P = transition_matrix(grid, demand, policy, params)
    reward = demand * (params.R - params.c1 - policy*(params.c2-params.c1))
    value = spsolve(eye(len(grid), format="csr") - params.gamma*P, reward)
    S = interpolation_matrix(grid, params.rho*grid + 1-params.rho)
    F = interpolation_matrix(grid, params.rho*grid)
    advantage = -(params.c2-params.c1) + params.gamma*(params.p2-params.p1)*((S-F) @ value)
    residual = float(np.max(np.abs(value-reward-params.gamma*(P @ value))))
    return {"value": value, "advantage": advantage, "residual": residual}


def solve_dp(grid, demand, params, tol=1e-10, max_iterations=200):
    """Pure best response to FIXED demand; allowing mixing cannot improve it.

    Optimal values are attained by pure actions. This best-response helper uses
    product 2 on ties; the equilibrium solver itself has no forced tie rule.
    Conditional action gains are checked even when demand is extremely small.
    """
    grid = check_grid(grid)
    demand = _check_probabilities(demand, grid, "Demand")
    if tol <= 0 or max_iterations < 1:
        raise ValueError("Positive tolerance and iteration limit required.")
    policy = np.zeros(len(grid))
    for iteration in range(1, max_iterations+1):
        evaluation = evaluate_policy(grid, demand, policy, params)
        value, advantage = evaluation["value"], evaluation["advantage"]
        new_policy = (advantage >= 0).astype(float)
        regret = np.maximum(advantage, 0)-policy*advantage
        residual = float(np.max(np.abs(demand*regret))) + evaluation["residual"]
        if residual <= tol and float(np.max(regret)) <= tol:
            return {**evaluation, "policy": new_policy, "residual": residual,
                    "iterations": iteration}
        policy = new_policy
    raise RuntimeError(f"DP failed: Bellman residual {residual:g}.")


def stationary_distribution(P, tol=1e-11):
    """Solve an embedded jump chain, then restore calendar-time holding times.

    Exit rates are sums of off-diagonal entries, never ``1-P[i,i]``: very rare
    selection makes that subtraction lose all precision. The invariant law of
    the jump chain is divided by exit rates. If tails are small or holding times
    disparate, subtraction-free positive elimination avoids amplifying signed
    linear-solve roundoff. It also handles slowly mixing chains without waiting
    for a long simulation or power iteration. True absorbing states are reported
    rather than assuming a unique law.
    """
    n = P.shape[0]
    if P.shape != (n,n) or P.data.min(initial=0) < -1e-14:
        raise ValueError("Transition matrix must be nonnegative and square.")
    if np.max(np.abs(np.asarray(P.sum(axis=1)).ravel()-1)) > 1e-12:
        raise ValueError("Transition matrix is not row-stochastic.")
    off = P - diags(P.diagonal())
    rates = np.asarray(off.sum(axis=1)).ravel()
    if np.any(rates <= 0):
        raise ArithmeticError("Zero exit rate: cannot assume a unique stationary law.")
    jump = diags(1/rates) @ off
    A = (jump.T-eye(n, format="csr")).tolil()
    A[-1, :] = np.ones(n)
    rhs = np.zeros(n)
    rhs[-1] = 1
    nu = spsolve(A.tocsr(), rhs)
    iterations = 1
    if rates.min()/rates.max() < 1e-10 or nu.min() < 1e-10*nu.max():
        # Subtraction-free elimination (GTH) keeps tiny invariant probabilities
        # positive without waiting for a potentially metastable chain to mix.
        # Eliminate state k by summing its outgoing rates to remaining states,
        # then add the flow through k to the retained transition entries.
        reduced = jump.toarray()
        for k in range(n-1, 0, -1):
            denominator = reduced[k, :k].sum()
            if denominator <= 0:
                raise ArithmeticError("Reducible jump chain: cannot certify a unique full-support invariant law.")
            reduced[:k, k] /= denominator
            reduced[:k, :k] += np.outer(reduced[:k, k], reduced[k, :k])
        nu = np.zeros(n)
        nu[0] = 1.
        for k in range(1, n):
            nu[k] = nu[:k] @ reduced[:k, k]
            if nu[k] > 1e100:
                nu[:k+1] /= nu[k]
        nu /= nu.sum()
        iterations = n-1
    if np.any(~np.isfinite(nu)) or nu.min() < -1e-9:
        raise ArithmeticError("Stationary solve is singular or produced invalid mass.")
    mu = np.maximum(nu, 0)/rates
    mu /= mu.sum()
    residual = float(np.sum(np.abs(P.T @ mu-mu)))
    if residual > tol:
        raise ArithmeticError(f"Stationary residual {residual:g} exceeds {tol:g}.")
    return {"mu": mu, "residual": residual, "iterations": iterations}
