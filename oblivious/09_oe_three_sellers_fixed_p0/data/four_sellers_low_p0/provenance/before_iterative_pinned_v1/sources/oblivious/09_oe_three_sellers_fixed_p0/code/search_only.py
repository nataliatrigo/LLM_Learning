"""Faster exploratory search with the original independent certificate.

This module changes neither the economic model nor ``Experiment.certify``.
Pinning one jump-invariant coordinate avoids a dense normalization row in the
sparse factorization. The analytic derivative differentiates that pinned system
and then normalizes the calendar law. A Jacobian is computed only when requested
by the inherited root solver; pure-policy improvements need only advantages.
"""

import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import splu

from oblivious.oe_p0.code.policy import evaluate_policy
from .model import Experiment as FrozenExperiment, regret
from .stationary import checked_stationary, jump_system


class _LazyJacobian(dict):
    """A search result whose expensive Jacobian is materialized on indexing."""

    def __init__(self, values, callback, release):
        super().__init__(values)
        self._jacobian_callback = callback
        self.release_factorization = release

    def __missing__(self, key):
        if key != "jacobian":
            raise KeyError(key)
        result = self._jacobian_callback()
        self["jacobian"] = result
        self._jacobian_callback = None
        self.release_factorization()
        return result


class FastExperiment(FrozenExperiment):
    """Original experiment with an exact pinned exploratory linear solve.

    All results returned by ``solve`` still pass the inherited fresh certificate.
    The lazy ``calculate`` result supports ``result['jacobian']``; a plain dict
    copy made before that access contains only the already evaluated fields.
    """

    def calculate(self, alpha):
        alpha = np.clip(np.asarray(alpha, dtype=float), 0, 1)
        if self.last_alpha is not None and np.array_equal(alpha, self.last_alpha):
            return self.last_result
        if alpha.shape != (self.n,) or np.any(~np.isfinite(alpha)):
            raise ValueError("Search policy must be a finite vector matching the grid.")
        if self.sellers == 1:
            raise ValueError("Monopoly uses the inherited direct DP, not joint search.")
        if isinstance(self.last_result, _LazyJacobian):
            # Pure improvements do not request Jacobians. Release their old LU
            # before constructing the next one, keeping only one large factor.
            # Accessing that old result's Jacobian later refactors its matrix.
            self.last_result.release_factorization()
        self.calls += 1
        full = self.game.transition(alpha)
        reduced = (full[self.representatives] @ self.membership).tocsr()
        jump, rates = jump_system(reduced)
        size = len(rates)
        anchor = size // 2
        keep = np.arange(size) != anchor
        transpose = jump.T.tocsr()
        linear = (eye(size, format="csr") - transpose)[keep, :][:, keep].tocsc()
        # The quotient is a multidimensional reputation graph. A symmetric
        # minimum-degree ordering reduces fill substantially here; SuperLU's
        # default numerical pivoting and the actual equations are unchanged.
        lu = splu(linear, permc_spec="MMD_AT_PLUS_A")
        rhs = np.asarray(transpose[keep, anchor].toarray()).ravel()
        w = np.empty(size)
        w[anchor] = 1.
        w[keep] = lu.solve(rhs)
        if np.any(~np.isfinite(w)) or np.any(w <= 0):
            # Search conditioning never uses a signed or floored invariant law.
            # The certificate is still a completely separate fresh computation.
            checked = checked_stationary(jump, rates)
            calendar = checked['law']
            log_w = np.log(calendar) + np.log(rates)
            w = np.exp(log_w - log_w.max())
        w /= w.sum()
        if np.any(w <= 0) or np.any(~np.isfinite(w)):
            raise ArithmeticError("Pinned search invariant lost a positive coordinate.")
        # q is w/r, scaled by a common constant before exponentiation. That
        # scale cancels in both the calendar law and its analytic derivative.
        log_q = np.log(w) - np.log(rates)
        q = np.exp(log_q - log_q.max())
        q_sum = q.sum()
        calendar = q / q_sum
        if np.any(calendar <= 0) or np.any(~np.isfinite(calendar)):
            raise ArithmeticError("Pinned search calendar law lost positive mass.")
        mu = np.asarray(self.own_marginal @ calendar).ravel()
        demand = self.marginal_demand @ calendar
        if self.closure == 'conditional':
            demand = np.asarray(self.sale_marginal @ calendar).ravel() / mu
        if np.any(~np.isfinite(demand)) or np.any(demand < -1e-12) or np.any(demand > 1 + 1e-12):
            raise ArithmeticError("Invalid pinned exploratory belief demand.")
        demand = np.clip(demand, 0, 1)
        evaluation = evaluate_policy(self.grid, demand, alpha, self.params)
        value, advantage = evaluation['value'], evaluation['advantage']
        factorization = dict(lu=lu)

        def jacobian():
            # For the pinned equation (I-J.T)w=0, dw[anchor]=0 and
            # (I-J.T)[keep,keep] dw[keep]=(dJ.T w)[keep]. We solve for
            # dw scaled by the same constant as q, avoiding w/r overflow.
            relative_dr = self.rate_derivatives / rates[:, None]
            derivative_rhs = np.empty((size, self.n))
            for k, delta in enumerate(self.derivatives):
                derivative_rhs[:, k] = delta.T @ q - jump.T @ (q * self.rate_derivatives[:, k])
            dw_scaled = np.zeros((size, self.n))
            factor = factorization['lu']
            if factor is None:
                factor = splu(linear, permc_spec="MMD_AT_PLUS_A")
            dw_scaled[keep] = factor.solve(derivative_rhs[keep])
            dc_raw = dw_scaled / rates[:, None] - q[:, None] * relative_dr
            dc = dc_raw / q_sum - calendar[:, None] * (dc_raw.sum(axis=0) / q_sum)
            dd = self.marginal_demand @ dc
            if self.closure == 'conditional':
                dd = (self.sale_marginal @ dc - demand[:, None] * (self.own_marginal @ dc)) / mu[:, None]
            p = self.params
            probability = p.p1 + (p.p2 - p.p1) * alpha
            selected = diags(probability) @ self.S + diags(1 - probability) @ self.F
            own = diags(1 - demand) + diags(demand) @ selected
            own_lu = splu((eye(self.n, format='csr') - p.gamma * own).tocsc())
            flow = p.R - p.c1 - (p.c2 - p.c1) * alpha + p.gamma * (selected @ value - value)
            dv = own_lu.solve(flow[:, None] * dd + np.diag(demand * advantage))
            result = p.gamma * (p.p2 - p.p1) * ((self.S - self.F) @ dv)
            if np.any(~np.isfinite(result)):
                raise ArithmeticError("Pinned search Jacobian is nonfinite.")
            return result

        result = _LazyJacobian(dict(policy=alpha, advantage=advantage, demand=demand,
                                    value=value, reduced_law=calendar,
                                    regret=regret(alpha, advantage)), jacobian,
                               lambda: factorization.update(lu=None))
        self.last_alpha, self.last_result = alpha.copy(), result
        return result


__all__ = ["FastExperiment"]
