"""Positive demand and calendar transitions for one to three sellers.

The buyer draws independent Beta valuations conditional on the *joint* state,
then buys from the largest valuation if it exceeds the fixed outside benchmark.
Only that seller receives an observation and updates its reputation. Policies
depend on own reputation; no independent-marginal approximation is used here.
"""

from itertools import permutations

import numpy as np
from scipy.special import betainc, betaincc, betaln, roots_jacobi
from scipy.sparse import csr_matrix

from oblivious.oe_p0.code.primitives import (
    check_grid, interpolation_matrix, posterior, probability_array,
)


def _raw_focal_demand(grid, params, p0, sellers, order):
    """Positive quadrature, before enforcing any event identity or symmetry."""
    a, b = posterior(grid, params)
    n = len(grid)
    result = np.empty((n,) * sellers)
    if p0 == 1:
        return np.zeros_like(result)
    for own in range(n):
        if p0 == 0:
            nodes, weights = roots_jacobi(order, b[own] - 1, a[own] - 1)
            t = (nodes + 1) / 2
            positive_weights = weights / weights.sum()
        else:
            # Absorb the endpoint factor (1-t)^(b-1) into Jacobi weights.
            nodes, weights = roots_jacobi(order, b[own] - 1, 0)
            t = p0 + (1 - p0) * (nodes + 1) / 2
            constant = b[own] * (np.log1p(-p0) - np.log(2)) - betaln(a[own], b[own])
            with np.errstate(divide="ignore"):
                positive_weights = np.exp(
                    constant + np.log(weights) + (a[own] - 1) * np.log(t))
        if sellers == 1:
            result[own] = positive_weights.sum()
            continue
        cdf = betainc(a[:, None], b[:, None], t[None, :])
        if sellers == 2:
            result[own] = cdf @ positive_weights
        else:
            # Each entry uses both realized rival coordinates. This is a
            # positive weighted Gram product, not the square of a marginal CDF.
            result[own] = (cdf * positive_weights[None, :]) @ cdf.T
    return result


def fixed_outside_demand(grid, params, p0=.5, sellers=3, order=128):
    """Return ``(demand, outside, diagnostics)`` on a joint reputation grid.

    ``demand[i, *state]`` is integral f_i(t) prod_{j!=i} F_j(t) dt,
    over [p0,1]. Raw quadrature and doubled-order diagnostics precede the
    symmetric proportional event-partition correction. The correction preserves
    positive small seller probabilities; no tail is obtained by subtraction.
    """
    grid = check_grid(grid)
    if int(sellers) != sellers or sellers not in (1, 2, 3):
        raise ValueError("Use one, two, or three endogenous sellers.")
    if not np.isfinite(p0) or not 0 <= p0 <= 1:
        raise ValueError("p0 must be finite and in [0,1].")
    if int(order) != order or order < 8:
        raise ValueError("Use an integer quadrature order >=8.")
    sellers, order, p0 = int(sellers), int(order), float(p0)
    a, b = posterior(grid, params)
    cdf, survival = betainc(a, b, p0), betaincc(a, b, p0)
    shape = (len(grid),) * sellers
    outside = np.ones(shape)
    log_outside = np.zeros(shape)
    for seller in range(sellers):
        view = [1] * sellers
        view[seller] = len(grid)
        outside *= cdf.reshape(view)
        with np.errstate(divide="ignore"):
            log_outside += np.log1p(-survival).reshape(view)
    # The survival form retains rare market selection when outside rounds to1.
    market = -np.expm1(log_outside)
    raw_focal = _raw_focal_demand(grid, params, p0, sellers, order)
    doubled_focal = _raw_focal_demand(grid, params, p0, sellers, 2 * order)
    raw = np.stack([np.moveaxis(raw_focal, 0, seller) for seller in range(sellers)])
    doubled = np.stack([np.moveaxis(doubled_focal, 0, seller) for seller in range(sellers)])
    raw_sum = raw.sum(axis=0)
    tiny = np.finfo(float).tiny
    relative_mass_error = float(np.max(abs(raw_sum - market) / np.maximum(market, tiny)))
    absolute_doubling = float(np.max(abs(raw - doubled)))
    relative_doubling = float(np.max(abs(raw - doubled) / np.maximum(doubled, tiny)))
    permutation_error = max(
        float(np.max(abs(raw[seller].transpose(axes) - raw[axes.index(seller)])))
        for axes in permutations(range(sellers)) for seller in range(sellers))
    diagonal = (np.arange(len(grid)),) * sellers
    diagnostics = dict(
        sellers=sellers, p0=p0, order=order,
        quadrature="Positive truncated Gauss-Jacobi with doubled-order check",
        raw_partition_error=float(np.max(abs(raw_sum + outside - 1))),
        raw_relative_selected_mass_error=relative_mass_error,
        raw_permutation_error=permutation_error,
        raw_diagonal_error=float(np.max(abs(raw[(slice(None),) + diagonal] - market[diagonal] / sellers))),
        raw_min=float(raw.min()), raw_max=float(raw.max()),
        order_doubling_max_abs=absolute_doubling,
        order_doubling_max_relative_tail=relative_doubling,
        positive_tail_underflow_entries=int(np.sum(doubled == 0)) if p0 < 1 else 0,
    )
    if (not np.all(np.isfinite(raw)) or np.any(raw < 0) or np.any(raw > 1 + 1e-10)
            or relative_mass_error > 2e-7 or absolute_doubling > 2e-9
            or relative_doubling > 2e-6 or permutation_error > 2e-12
            or (p0 < 1 and np.any(doubled == 0))):
        raise ArithmeticError(f"Demand quadrature failed; increase order: {diagnostics}")
    if p0 == 1:
        demand = raw
    else:
        # Common scaling at a state respects seller symmetry and preserves
        # positive directly integrated tails even when another seller dominates.
        demand = raw * (market / raw_sum)[None, ...]
    diagnostics.update(
        partition_error=float(np.max(abs(demand.sum(axis=0) + outside - 1))),
        relative_event_correction=float(np.max(abs(demand - raw) / np.maximum(raw, tiny))),
        min_positive_demand=float(demand[demand > 0].min(initial=1.)),
    )
    return demand, outside, diagnostics


class FixedOutsideKernel:
    """Joint calendar kernel, with C-order state flattening.

    ``alpha`` is either one own-reputation policy of shape (n,), shared by all
    sellers, or seller-specific own-state policies of shape (sellers,n).
    """

    def __init__(self, grid, params, p0=.5, sellers=3, order=128):
        self.grid = check_grid(grid).copy()
        self.params, self.p0 = params, float(p0)
        self.sellers, self.n, self.order = int(sellers), len(self.grid), int(order)
        self.demand, self.outside, self.demand_diagnostics = fixed_outside_demand(
            self.grid, params, p0=p0, sellers=sellers, order=order)
        self.selection = self.demand
        self.shape = (self.n,) * self.sellers
        self.size = self.n ** self.sellers
        self.success = interpolation_matrix(self.grid, params.rho * self.grid + 1 - params.rho)
        self.failure = interpolation_matrix(self.grid, params.rho * self.grid)
        self._rows = np.arange(self.size, dtype=np.int64)
        self._strides = tuple(self.n ** (self.sellers - seller - 1) for seller in range(self.sellers))
        self._own_indices = tuple((self._rows // stride) % self.n for stride in self._strides)

    def policies(self, alpha):
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape == (self.n,):
            alpha = np.broadcast_to(alpha, (self.sellers, self.n))
        if alpha.shape != (self.sellers, self.n):
            raise ValueError("Own-state policies must have shape (n,) or (sellers,n).")
        return probability_array(alpha)

    def _probabilities(self, alpha):
        return self.params.p1 + (self.params.p2 - self.params.p1) * self.policies(alpha)

    def coordinate_apply(self, values, seller, success=True):
        """Expected next value if just this seller succeeds or fails."""
        values = np.asarray(values, dtype=float)
        if values.shape == (self.size,):
            values = values.reshape(self.shape)
        if values.shape != self.shape or not np.all(np.isfinite(values)):
            raise ValueError("Values must have joint shape or be its finite flat array.")
        if not 0 <= seller < self.sellers:
            raise ValueError("Seller index out of range.")
        operator = self.success if success else self.failure
        moved = np.moveaxis(values, seller, 0)
        result = (operator @ moved.reshape(self.n, -1)).reshape(moved.shape)
        return np.moveaxis(result, 0, seller)

    def apply(self, values, alpha):
        """Matrix-free expectation of a value under the calendar transition."""
        values = np.asarray(values, dtype=float).reshape(self.shape)
        probabilities = self._probabilities(alpha)
        result = self.outside * values
        for seller in range(self.sellers):
            view = [1] * self.sellers
            view[seller] = self.n
            q = probabilities[seller].reshape(view)
            result = result + self.demand[seller] * (
                q * self.coordinate_apply(values, seller, True)
                + (1 - q) * self.coordinate_apply(values, seller, False))
        return result

    def _flow_matrix(self, alpha, off_diagonal_only=False):
        probabilities = self._probabilities(alpha)
        row_parts, column_parts, weight_parts = [], [], []
        for seller in range(self.sellers):
            own = self._own_indices[seller]
            q = probabilities[seller, own]
            selection = self.demand[seller].ravel()
            for operator, probability in ((self.success, q), (self.failure, 1 - q)):
                # interpolation_matrix has exactly two stored neighbors per row.
                destination = operator.indices.reshape(self.n, 2)
                weight = operator.data.reshape(self.n, 2)
                for neighbor in range(2):
                    columns = self._rows + (destination[own, neighbor] - own) * self._strides[seller]
                    weights = selection * probability * weight[own, neighbor]
                    keep = weights > 0
                    if off_diagonal_only:
                        keep &= columns != self._rows
                    row_parts.append(self._rows[keep])
                    column_parts.append(columns[keep])
                    weight_parts.append(weights[keep])
        if not off_diagonal_only:
            keep = self.outside.ravel() > 0
            row_parts.append(self._rows[keep])
            column_parts.append(self._rows[keep])
            weight_parts.append(self.outside.ravel()[keep])
        result = csr_matrix((np.concatenate(weight_parts),
                            (np.concatenate(row_parts), np.concatenate(column_parts))),
                           shape=(self.size, self.size))
        result.eliminate_zeros()
        return result

    def transition(self, alpha):
        """Full sparse calendar transition, including outside holding time."""
        return self._flow_matrix(alpha)

    def jump(self, alpha):
        """Return (zero-diagonal jump CSR, flat positive exit rates).

        Exit rates sum directly constructed off-diagonal positive flows. They
        are never calculated as 1 minus a rounded holding probability.
        """
        off_diagonal = self._flow_matrix(alpha, off_diagonal_only=True)
        rates = np.asarray(off_diagonal.sum(axis=1)).ravel()
        if np.any(rates <= 0) or np.any(~np.isfinite(rates)):
            raise ArithmeticError("Positive-exit jump chain undefined at an absorbing state.")
        jump = off_diagonal.copy()
        jump.data /= np.repeat(rates, np.diff(jump.indptr))
        return jump, rates


__all__ = ["FixedOutsideKernel", "fixed_outside_demand"]
