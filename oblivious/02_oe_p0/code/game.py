"""The two-seller discounted game on a joint reputation grid.

All policy arrays are probabilities of product 2 in GLOBAL (x_A, x_B) axes.
Thus a symmetric profile has sigma_b = sigma_a.T, not sigma_b = sigma_a.
Values include the current calendar period and are measured before selection.
"""

import numpy as np
from scipy.sparse import csr_matrix, diags, eye, kron
from scipy.sparse.linalg import bicgstab

from .primitives import (
    Parameters, check_grid, demand_matrix, demand_probability, interpolation_matrix,
    probability_array,
)


class Game:
    """Precomputed transitions, values, best responses, and regret checks."""

    def __init__(self, grid, params, order=128):
        self.grid = check_grid(grid).copy()
        self.params = params
        self.n = len(self.grid)
        self.D, self.demand_diagnostics = demand_matrix(self.grid, params, order)
        self.order = order
        success = interpolation_matrix(self.grid, params.rho * self.grid + 1 - params.rho)
        failure = interpolation_matrix(self.grid, params.rho * self.grid)
        identity = eye(self.n, format="csr")
        self.own_success = kron(success, identity, format="csr")
        self.own_failure = kron(failure, identity, format="csr")
        self.rival_success = kron(identity, success, format="csr")
        self.rival_failure = kron(identity, failure, format="csr")
        self.success_operators = (self.own_success, self.rival_success)
        self.failure_operators = (self.own_failure, self.rival_failure)
        # D.T equals 1-D mathematically, while preserving the directly integrated
        # tiny lower tail when subtracting a rounded probability of 1 would fail.
        self.selection = np.stack((self.D, self.D.T))

    def _policy(self, sigma):
        sigma = np.asarray(sigma, dtype=float)
        if sigma.ndim == 0:
            sigma = np.full((self.n, self.n), sigma)
        if sigma.shape != (self.n, self.n) or np.any(~np.isfinite(sigma)):
            raise ValueError("Policy must be a finite (n,n) array in global (A,B) axes")
        return probability_array(sigma)

    def transition(self, sigma_a, sigma_b):
        """Joint calendar-time kernel, flattened in C order (B index fastest)."""
        sigmas = (self._policy(sigma_a), self._policy(sigma_b))
        matrix = csr_matrix((self.n ** 2, self.n ** 2))
        for j, sigma in enumerate(sigmas):
            p = self.params.p1 + (self.params.p2 - self.params.p1) * sigma.ravel()
            demand = self.selection[j].ravel()
            matrix += diags(demand * p) @ self.success_operators[j]
            matrix += diags(demand * (1 - p)) @ self.failure_operators[j]
        return matrix.tocsr()

    def rewards(self, sigma_a, sigma_b):
        """Expected current profits, averaging product costs under mixtures."""
        sigmas = np.stack((self._policy(sigma_a), self._policy(sigma_b)))
        costs = self.params.c1 + (self.params.c2 - self.params.c1) * sigmas
        return self.selection * (self.params.R - costs)

    def evaluate(self, sigma_a, sigma_b, tol=1e-10, initial=None):
        """Evaluate both players by checked BiCGSTAB, with contraction fallback.

        This avoids the severe sparse-LU fill-in of the 2D transition graph.
        The returned residuals are checked at the returned values themselves.
        """
        if tol <= 0:
            raise ValueError("Positive policy-evaluation tolerance required")
        matrix = self.transition(sigma_a, sigma_b)
        rewards = self.rewards(sigma_a, sigma_b).reshape(2, -1).T
        if initial is None:
            values = np.zeros_like(rewards)
        else:
            initial = np.asarray(initial, dtype=float)
            if initial.shape != (2, self.n, self.n) or np.any(~np.isfinite(initial)):
                raise ValueError("Initial values must have shape (2,n,n) and be finite")
            values = initial.reshape(2, -1).T.copy()
        residuals = np.max(np.abs(rewards + self.params.gamma * (matrix @ values) - values), axis=0)
        if residuals.max() <= tol:
            return dict(values=values.T.reshape(2, self.n, self.n), residuals=residuals,
                        iterations=0, method="initial residual check")
        linear = eye(self.n ** 2, format="csr") - self.params.gamma * matrix
        counts, infos = [], []
        for j in range(2):
            count = [0]

            def callback(_):
                count[0] += 1

            candidate, info = bicgstab(linear, rewards[:, j], x0=values[:, j],
                                       rtol=0., atol=tol / 10, maxiter=2000, callback=callback)
            counts.append(count[0])
            infos.append(int(info))
            if np.all(np.isfinite(candidate)):
                candidate_residual = float(np.max(np.abs(
                    rewards[:, j] + self.params.gamma * (matrix @ candidate) - candidate)))
                # A failed Krylov attempt is still a useful warm start only if
                # it improves the original Bellman residual, not its own flag.
                if candidate_residual < residuals[j]:
                    values[:, j] = candidate
                    residuals[j] = candidate_residual
        if residuals.max() <= tol:
            return dict(values=values.T.reshape(2, self.n, self.n), residuals=residuals,
                        iterations=max(1, max(counts)), method="checked BiCGSTAB",
                        krylov_callback_iterations=counts, krylov_status=infos)
        # Unsuccessful or nonfinite Krylov solves never count as convergence.
        # The original positive contraction remains the reliable fallback.
        for iterations in range(1, 100001):
            updated = rewards + self.params.gamma * (matrix @ values)
            residuals = np.max(np.abs(updated - values), axis=0)
            if residuals.max() <= tol:
                return dict(values=values.T.reshape(2, self.n, self.n),
                            residuals=residuals, iterations=iterations,
                            method="contraction fallback", krylov_callback_iterations=counts,
                            krylov_status=infos)
            values = updated
        raise RuntimeError(f"Policy evaluation did not converge; residuals={residuals}")

    def advantages(self, values):
        """Conditional product-2 minus product-1 payoffs for each seller."""
        values = np.asarray(values, dtype=float)
        if values.shape != (2, self.n, self.n) or np.any(~np.isfinite(values)):
            raise ValueError("Values must have shape (2,n,n) and be finite")
        p = self.params
        advantages = []
        for j in range(2):
            v = values[j].ravel()
            difference = self.success_operators[j] @ v - self.failure_operators[j] @ v
            advantages.append(-(p.c2 - p.c1) + p.gamma * (p.p2 - p.p1) * difference)
        return np.array(advantages).reshape(2, self.n, self.n)

    def best_response(self, opponent, seller=0, tol=1e-10, initial=None):
        """Solve one player's unregularized MDP with the rival policy fixed.

        Opponent and outputs use GLOBAL axes, including when seller=1 (B).
        Exact ties choose product 2. No policy shape is imposed.
        """
        opponent = self._policy(opponent)
        if seller not in (0, 1) or tol <= 0:
            raise ValueError("Seller must be 0 or 1, with positive tolerance")
        if seller == 1:
            result = self.best_response(opponent.T, seller=0, tol=tol,
                                        initial=None if initial is None else np.asarray(initial).T)
            return {**result, **{key: result[key].T for key in ("value", "sigma", "advantage")}}
        if initial is None:
            value = np.zeros(self.n ** 2)
        else:
            initial = np.asarray(initial, dtype=float)
            if initial.shape != (self.n, self.n) or np.any(~np.isfinite(initial)):
                raise ValueError("Initial best-response value must have shape (n,n)")
            value = initial.ravel().copy()
        p = self.params
        demand = self.D.ravel()
        rival_demand = self.selection[1].ravel()
        rival_p = p.p1 + (p.p2 - p.p1) * opponent.ravel()
        rival_transition = (diags(rival_demand * rival_p) @ self.rival_success
                            + diags(rival_demand * (1 - rival_p)) @ self.rival_failure)
        for iterations in range(1, 100001):
            vs, vf = self.own_success @ value, self.own_failure @ value
            advantage = -(p.c2 - p.c1) + p.gamma * (p.p2 - p.p1) * (vs - vf)
            q1 = p.R - p.c1 + p.gamma * (p.p1 * vs + (1 - p.p1) * vf)
            updated = demand * (q1 + np.maximum(advantage, 0)) + p.gamma * (rival_transition @ value)
            residual = float(np.max(np.abs(updated - value)))
            if residual <= tol:
                return dict(value=value.reshape(self.n, self.n),
                            sigma=(advantage >= 0).astype(float).reshape(self.n, self.n),
                            advantage=advantage.reshape(self.n, self.n),
                            residual=residual, iterations=iterations)
            value = updated
        raise RuntimeError(f"Best response did not converge; residual={residual:g}")

    def verify(self, sigma_a, sigma_b, tol=1e-10, weights=None):
        """Independently solve both deviations against the original game.

        Regret bounds concern the finite MDP with its fixed numerical demand
        matrix only. They exclude continuous-state and demand-integration error.
        No stationarity or independence assumption is used in this verification.
        """
        evaluation = self.evaluate(sigma_a, sigma_b, tol=tol)
        values = evaluation["values"]
        responses = [self.best_response(sigma_b, seller=0, tol=tol, initial=values[0]),
                     self.best_response(sigma_a, seller=1, tol=tol, initial=values[1])]
        br_values = np.stack([response["value"] for response in responses])
        br_residuals = np.array([response["residual"] for response in responses])
        regrets = br_values - values
        bounds = (evaluation["residuals"] + br_residuals) / (1 - self.params.gamma)
        weighted = None
        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            if weights.shape not in ((self.n, self.n), (self.n ** 2,)):
                raise ValueError("Regret weights must have shape (n,n) or (n*n,)")
            if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("Regret weights must be finite, nonnegative, and have positive mass")
            weights = weights.reshape(self.n, self.n) / weights.sum()
            weighted = np.sum(regrets * weights, axis=(1, 2)).tolist()
        maximum = regrets.max(axis=(1, 2))
        summary = dict(
            max_regret=maximum.tolist(), mean_regret=regrets.mean(axis=(1, 2)).tolist(),
            weighted_regret=weighted, min_regret=regrets.min(axis=(1, 2)).tolist(),
            policy_residuals=evaluation["residuals"].tolist(),
            br_residuals=br_residuals.tolist(), regret_error_bounds=bounds.tolist(),
            max_regret_upper_bounds=(maximum + bounds).tolist(),
            policy_evaluation_iterations=evaluation["iterations"],
            policy_evaluation_method=evaluation["method"],
            br_iterations=[response["iterations"] for response in responses],
            tolerance=tol,
            residual_bound_scope="Finite numerical MDP only; excludes grid and quadrature error",
        )
        return dict(values=values, br_values=br_values,
                    br_policies=np.stack([response["sigma"] for response in responses]),
                    regrets=regrets, advantages=self.advantages(values), summary=summary)

    def stationary(self, P, tol=1e-11, max_iterations=100000, initial=None):
        """Positive lazy power iteration for a joint calendar-time invariant law.

        Default initialization is uniform. The returned law can depend on that
        initialization in a reducible chain. Convergence verifies invariance,
        not uniqueness or a mixing-time bound. Failure is returned explicitly.
        """
        matrix = csr_matrix(P)
        if matrix.shape != (self.n ** 2, self.n ** 2) or matrix.data.min(initial=0) < 0:
            raise ValueError("Joint transition matrix must be nonnegative with shape (n*n,n*n)")
        if np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel() - 1)) > 1e-12:
            raise ValueError("Joint transition matrix must be row-stochastic")
        if tol <= 0 or max_iterations < 1:
            raise ValueError("Positive stationarity tolerance and iteration limit required")
        mu = (np.ones(self.n ** 2) if initial is None
              else np.array(initial, dtype=float, copy=True).ravel())
        if mu.size != self.n ** 2 or np.any(~np.isfinite(mu)) or np.any(mu < 0) or mu.sum() <= 0:
            raise ValueError("Initial distribution must be finite and nonnegative on the joint grid")
        mu /= mu.sum()
        transpose = matrix.T.tocsr()
        for iterations in range(1, max_iterations + 1):
            moved = transpose @ mu
            residual = float(np.sum(np.abs(moved - mu)))
            if residual <= tol or iterations == max_iterations:
                return dict(mu=mu.reshape(self.n, self.n), residual=residual,
                            iterations=iterations, converged=bool(residual <= tol))
            mu = .5 * (mu + moved)  # Laziness removes periodic oscillations.
            mu /= mu.sum()
