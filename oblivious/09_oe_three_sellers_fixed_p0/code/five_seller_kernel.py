"""Exact symmetric five-seller kernel without an ``n**5`` state allocation.

Rows are permutation orbits, represented by sorted reputation-index quintuples.
An orbit's probability is its total labelled-state probability. All five buyer
choices and all own-observation transitions are retained; equal coordinates
contribute once per seller and therefore have the correct multiplicity.

The marginal-OE demand operator averages over the *joint* four-rival law.
It never replaces this law by a product of one-seller marginals.
"""

from itertools import combinations_with_replacement
from math import comb, factorial

import numpy as np
from scipy.special import betainc, betaincc, betaln, roots_jacobi
from scipy.sparse import csr_matrix

from oblivious.oe_p0.code.primitives import (
    check_grid, interpolation_matrix, posterior, probability_array,
)


def _multisets(n, width):
    return np.asarray(list(combinations_with_replacement(range(n), width)),
                      dtype=np.int64)


def _colex_rank(coordinates, choose):
    """Bijective rank of sorted multisets; supports leading batch dimensions."""
    coordinates = np.asarray(coordinates, dtype=np.int64)
    result = np.zeros(coordinates.shape[:-1], dtype=np.int64)
    for slot in range(coordinates.shape[-1]):
        result += choose[coordinates[..., slot] + slot, slot + 1]
    return result


def _raw_lookup(grid, params, p0, rivals, order, chunk_size=2048):
    """Positive focal-demand quadrature indexed by own node and rival quartet."""
    a, b = posterior(grid, params)
    result = np.zeros((len(grid), len(rivals)))
    if p0 == 1:
        return result
    for own in range(len(grid)):
        if p0 == 0:
            nodes, weights = roots_jacobi(order, b[own]-1, a[own]-1)
            t = (nodes+1)/2
            positive_weights = weights/weights.sum()
        else:
            nodes, weights = roots_jacobi(order, b[own]-1, 0)
            t = p0+(1-p0)*(nodes+1)/2
            constant = b[own]*(np.log1p(-p0)-np.log(2))-betaln(a[own], b[own])
            with np.errstate(divide="ignore"):
                positive_weights = np.exp(
                    constant+np.log(weights)+(a[own]-1)*np.log(t))
        cdf = betainc(a[:, None], b[:, None], t[None, :])
        # Bound working memory even for doubled-order production quadrature.
        # This is a direct four-CDF product, without independent-rival beliefs.
        for start in range(0, len(rivals), chunk_size):
            stop = min(start+chunk_size, len(rivals))
            chunk = rivals[start:stop]
            products = cdf[chunk[:, 0]].copy()
            for slot in range(1, 4):
                products *= cdf[chunk[:, slot]]
            result[own, start:stop] = products @ positive_weights
    return result


class FiveSellerKernel:
    """Exact calendar/jump permutation quotient for a symmetric own policy.

    ``demand[row, slot]`` is the selection probability of the seller occupying
    that sorted coordinate. ``marginal_demand @ orbit_law`` is perceived demand
    at every candidate own reputation under the actual joint rival law.
    """

    def __init__(self, grid, params, p0=.1, order=256):
        self.grid = check_grid(grid).copy()
        if not np.isfinite(p0) or not 0 <= p0 <= 1:
            raise ValueError("p0 must be finite and in [0,1].")
        if int(order) != order or order < 8:
            raise ValueError("Use an integer quadrature order >=8.")
        self.params, self.p0 = params, float(p0)
        self.n, self.sellers, self.order = len(self.grid), 5, int(order)
        self.representatives = _multisets(self.n, 5)
        self.rival_representatives = _multisets(self.n, 4)
        self.size = len(self.representatives)
        self.shape, self.full_size = (self.size,), self.n**5
        self._choose = np.asarray([
            [comb(i, k) if i >= k else 0 for k in range(6)]
            for i in range(self.n+5)], dtype=np.int64)
        self._orbit_rank_to_index = np.empty(self.size, dtype=np.int64)
        self._orbit_rank_to_index[_colex_rank(self.representatives, self._choose)] = np.arange(self.size)
        self._rival_rank_to_index = np.empty(len(self.rival_representatives), dtype=np.int64)
        self._rival_rank_to_index[_colex_rank(self.rival_representatives, self._choose)] = np.arange(len(self.rival_representatives))
        denominators = np.ones(self.size, dtype=np.int64)
        factorials = np.array([factorial(k) for k in range(6)], dtype=np.int64)
        for slot in range(5):
            counts = np.sum(self.representatives == self.representatives[:, slot, None], axis=1)
            first = (np.ones(self.size, dtype=bool) if slot == 0 else
                     self.representatives[:, slot] != self.representatives[:, slot-1])
            denominators *= np.where(first, factorials[counts], 1)
        self.orbit_sizes = factorial(5)//denominators
        # Compatibility names explicitly mean orbit sizes, not state weights.
        self.counts = self.multiplicities = self.orbit_sizes
        self._index_dtype = np.int32 if self.size <= np.iinfo(np.int32).max else np.int64
        self._rows = np.arange(self.size, dtype=self._index_dtype)
        self._excluded_rival_ids = np.column_stack([
            self.rival_index(np.delete(self.representatives, slot, axis=1))
            for slot in range(5)])
        self._build_demand()
        self.success = interpolation_matrix(self.grid, params.rho*self.grid+1-params.rho)
        self.failure = interpolation_matrix(self.grid, params.rho*self.grid)
        rows = self.representatives.ravel()
        cols = np.repeat(self._rows, 5)
        self.own_marginal = csr_matrix((np.full(5*self.size, .2), (rows, cols)),
                                      shape=(self.n, self.size))
        self.sale_marginal = csr_matrix((self.demand.ravel()/5, (rows, cols)),
                                       shape=(self.n, self.size))
        self._edges = self._transition_edges()

    def orbit_index(self, coordinates):
        """Map sorted or unsorted index quintuples to quotient row indices."""
        coordinates = np.asarray(coordinates)
        if coordinates.shape[-1:] != (5,) or np.any(coordinates < 0) or np.any(coordinates >= self.n):
            raise ValueError("Require reputation-index quintuples in range.")
        if np.any(coordinates != np.asarray(coordinates, dtype=np.int64)):
            raise ValueError("Reputation indices must be integers.")
        return self._orbit_rank_to_index[_colex_rank(np.sort(coordinates, axis=-1), self._choose)]

    def rival_index(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=np.int64)
        if coordinates.shape[-1:] != (4,) or np.any(coordinates < 0) or np.any(coordinates >= self.n):
            raise ValueError("Require reputation-index quartets in range.")
        return self._rival_rank_to_index[_colex_rank(np.sort(coordinates, axis=-1), self._choose)]

    def _build_demand(self):
        a, b = posterior(self.grid, self.params)
        cdf, survival = betainc(a, b, self.p0), betaincc(a, b, self.p0)
        self.outside = np.prod(cdf[self.representatives], axis=1)
        with np.errstate(divide="ignore"):
            market = -np.expm1(np.log1p(-survival[self.representatives]).sum(axis=1))
        raw_lookup = _raw_lookup(self.grid, self.params, self.p0,
                                 self.rival_representatives, self.order)
        doubled_lookup = _raw_lookup(self.grid, self.params, self.p0,
                                     self.rival_representatives, 2*self.order)
        raw = raw_lookup[self.representatives, self._excluded_rival_ids]
        doubled = doubled_lookup[self.representatives, self._excluded_rival_ids]
        raw_sum, tiny = raw.sum(axis=1), np.finfo(float).tiny
        relative_mass_error = float(np.max(abs(raw_sum-market)/np.maximum(market, tiny)))
        absolute_doubling = float(np.max(abs(raw_lookup-doubled_lookup)))
        relative_doubling = float(np.max(abs(raw_lookup-doubled_lookup)/np.maximum(doubled_lookup, tiny)))
        diagonal = self.orbit_index(np.repeat(np.arange(self.n)[:, None], 5, axis=1))
        self.demand_diagnostics = dict(
            sellers=5, p0=self.p0, order=self.order,
            representation="Exact permutation quotient; all five labelled choices retained",
            quadrature="Positive truncated Gauss-Jacobi with doubled-order check",
            full_states=self.full_size, quotient_states=self.size,
            raw_partition_error=float(np.max(abs(raw_sum+self.outside-1))),
            raw_relative_selected_mass_error=relative_mass_error,
            raw_permutation_error=0.,
            raw_diagonal_error=float(np.max(abs(raw[diagonal]-market[diagonal, None]/5))),
            raw_min=float(raw.min()), raw_max=float(raw.max()),
            order_doubling_max_abs=absolute_doubling,
            order_doubling_max_relative_tail=relative_doubling,
            positive_tail_underflow_entries=int(np.sum(doubled == 0)) if self.p0 < 1 else 0,
        )
        if (not np.all(np.isfinite(raw_lookup)) or np.any(raw_lookup < 0)
                or np.any(raw_lookup > 1+1e-10) or relative_mass_error > 2e-7
                or absolute_doubling > 2e-9 or relative_doubling > 2e-6
                or (self.p0 < 1 and np.any(doubled_lookup == 0))):
            raise ArithmeticError(f"Demand quadrature failed; increase order: {self.demand_diagnostics}")
        factors = np.ones(self.size) if self.p0 == 1 else market/raw_sum
        self.demand = raw*factors[:, None]
        # Apply exactly the same state-specific event correction to each
        # hypothetical focal-state/rival-quartet demand used by the OE beliefs.
        self.focal_demand = np.empty_like(raw_lookup)
        for own in range(self.n):
            states = np.column_stack((np.full(len(self.rival_representatives), own),
                                      self.rival_representatives))
            self.focal_demand[own] = raw_lookup[own]*factors[self.orbit_index(states)]
        self.marginal_demand = np.zeros((self.n, self.size))
        for start in range(0, self.size, 8192):
            stop = min(start+8192, self.size)
            for slot in range(5):
                self.marginal_demand[:, start:stop] += (
                    self.focal_demand[:, self._excluded_rival_ids[start:stop, slot]]/5)
        self.demand_diagnostics.update(
            partition_error=float(np.max(abs(self.demand.sum(axis=1)+self.outside-1))),
            relative_event_correction=float(np.max(abs(self.demand-raw)/np.maximum(raw, tiny))),
            min_positive_demand=float(self.demand[self.demand > 0].min(initial=1.)),
        )

    def policies(self, alpha):
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (self.n,):
            raise ValueError("The exact symmetric quotient requires one shared policy of shape (n,).")
        return probability_array(alpha)

    def _transition_edges(self):
        """Precompute interpolation destinations without a full labelled grid."""
        edges = []
        for slot in range(5):
            own = self.representatives[:, slot]
            for success, operator in ((True, self.success), (False, self.failure)):
                destination = operator.indices.reshape(self.n, 2)
                weights = operator.data.reshape(self.n, 2)
                for neighbor in range(2):
                    states = self.representatives.copy()
                    states[:, slot] = destination[own, neighbor]
                    columns = self.orbit_index(states)
                    base = self.demand[:, slot]*weights[own, neighbor]
                    keep = base > 0
                    edges.append((self._rows[keep], columns[keep].astype(self._index_dtype),
                                  own[keep].astype(np.min_scalar_type(self.n-1)),
                                  base[keep], success))
        return tuple(edges)

    def _flow_matrix(self, alpha, off_diagonal_only=False):
        alpha = self.policies(alpha)
        probability = self.params.p1+(self.params.p2-self.params.p1)*alpha
        rows, columns, weights = [], [], []
        for rr, cc, own, base, success in self._edges:
            keep = cc != rr if off_diagonal_only else np.ones(len(rr), dtype=bool)
            selected_probability = probability[own] if success else 1-probability[own]
            rows.append(rr[keep]); columns.append(cc[keep])
            weights.append((base*selected_probability)[keep])
        if not off_diagonal_only:
            rows.append(self._rows); columns.append(self._rows); weights.append(self.outside)
        result = csr_matrix((np.concatenate(weights),
                             (np.concatenate(rows), np.concatenate(columns))),
                            shape=(self.size, self.size))
        result.eliminate_zeros()
        return result

    def transition(self, alpha):
        """Calendar transition of total permutation-orbit masses."""
        return self._flow_matrix(alpha)

    def jump(self, alpha):
        """Positive off-diagonal jump probabilities and direct exit rates."""
        off_diagonal = self._flow_matrix(alpha, off_diagonal_only=True)
        rates = np.asarray(off_diagonal.sum(axis=1)).ravel()
        if np.any(rates <= 0) or np.any(~np.isfinite(rates)):
            raise ArithmeticError("Positive-exit jump chain undefined at an absorbing state.")
        jump = off_diagonal.copy()
        jump.data /= np.repeat(rates, np.diff(jump.indptr))
        return jump, rates

    def transition_derivative(self, k):
        """Exact full calendar derivative dQ/d alpha[k], including diagonal."""
        if int(k) != k or not 0 <= k < self.n:
            raise ValueError("Policy-node index out of range.")
        rows, columns, weights = [], [], []
        delta_p = self.params.p2-self.params.p1
        for rr, cc, own, base, success in self._edges:
            keep = own == k
            rows.append(rr[keep]); columns.append(cc[keep])
            weights.append((delta_p if success else -delta_p)*base[keep])
        result = csr_matrix((np.concatenate(weights),
                             (np.concatenate(rows), np.concatenate(columns))),
                            shape=(self.size, self.size))
        result.eliminate_zeros()
        return result

    def transition_derivatives(self):
        return [self.transition_derivative(k) for k in range(self.n)]


__all__ = ["FiveSellerKernel"]
