"""Exact count-state competition on the seller-permutation quotient.

A mature memory contains ``capacity`` binary observations. Conditional on a
purchase, the new outcome replaces an independently, uniformly selected old
observation. Only the chosen seller updates. Sorted seller-count tuples are an
exact symmetry reduction of the labelled game, including correlated rivals.
"""

from itertools import combinations_with_replacement
from math import comb, factorial

import numpy as np
from scipy.special import betainc, betaincc, betaln, roots_jacobi
from scipy.sparse import csr_matrix, diags


def _multisets(n, width):
    return np.asarray(list(combinations_with_replacement(range(n), width)),
                      dtype=np.int64).reshape(-1, width) if width else np.empty((1, 0), dtype=np.int64)


def _colex_rank(coordinates, choose):
    result = np.zeros(coordinates.shape[:-1], dtype=np.int64)
    for slot in range(coordinates.shape[-1]):
        result += choose[coordinates[..., slot] + slot, slot + 1]
    return result


def _raw_lookup(capacity, p0, rivals, order, chunk_size=2048):
    """Positive integrals of a focal Beta density times all rival CDFs."""
    n = capacity + 1
    a, b = 1 + np.arange(n), 1 + capacity - np.arange(n)
    result = np.empty((n, len(rivals)))
    for own in range(n):
        if p0 == 0:
            nodes, weights = roots_jacobi(order, b[own] - 1, a[own] - 1)
            t = (nodes + 1) / 2
            positive_weights = weights / weights.sum()
        else:
            nodes, weights = roots_jacobi(order, b[own] - 1, 0)
            t = p0 + (1 - p0) * (nodes + 1) / 2
            constant = b[own] * (np.log1p(-p0) - np.log(2)) - betaln(a[own], b[own])
            with np.errstate(divide="ignore"):
                positive_weights = np.exp(constant + np.log(weights) + (a[own] - 1) * np.log(t))
        if not rivals.shape[1]:
            result[own] = positive_weights.sum()
            continue
        cdf = betainc(a[:, None], b[:, None], t[None, :])
        for start in range(0, len(rivals), chunk_size):
            stop = min(start + chunk_size, len(rivals))
            products = np.ones((stop - start, order))
            for slot in range(rivals.shape[1]):
                products *= cdf[rivals[start:stop, slot]]
            result[own, start:stop] = products @ positive_weights
    return result


class CountKernel:
    """Exact symmetric calendar chain and marginal-belief demand operator.

    ``states`` contains sorted tuples, and a probability assigned to a row is
    the total probability of its labelled permutation orbit. ``selection[r,j]``
    is the selection probability of the seller in sorted slot ``j``. Equal
    counts still contribute once per seller, retaining the exact multiplicity.
    """

    def __init__(self, capacity, sellers, p0, order=128):
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if isinstance(sellers, bool) or int(sellers) != sellers or sellers not in range(1, 6):
            raise ValueError("sellers must be an integer from one through five")
        if not np.isfinite(p0) or not 0 <= p0 < 1:
            raise ValueError("p0 must be finite and lie in [0,1)")
        if isinstance(order, bool) or int(order) != order or order < 8:
            raise ValueError("quadrature order must be an integer of at least eight")
        self.capacity, self.sellers, self.p0, self.order = int(capacity), int(sellers), float(p0), int(order)
        self.n = self.capacity + 1
        self.states = self.representatives = _multisets(self.n, self.sellers)
        self.rivals = _multisets(self.n, self.sellers - 1)
        self.size = len(self.states)
        self.full_size = self.n ** self.sellers
        self._choose = np.asarray([[comb(i, k) if i >= k else 0
                                   for k in range(self.sellers + 1)]
                                  for i in range(self.n + self.sellers)], dtype=np.int64)
        self._orbit_map = np.empty(self.size, dtype=np.int64)
        self._orbit_map[_colex_rank(self.states, self._choose)] = np.arange(self.size)
        self._rival_map = np.empty(len(self.rivals), dtype=np.int64)
        self._rival_map[_colex_rank(self.rivals, self._choose)] = np.arange(len(self.rivals))
        self._rows = np.arange(self.size)
        self._rival_ids = np.column_stack([
            self._rival_map[_colex_rank(np.delete(self.states, slot, axis=1), self._choose)]
            for slot in range(self.sellers)])
        self.orbit_sizes = np.array([
            factorial(self.sellers) // np.prod([factorial(int(c)) for c in np.unique(row, return_counts=True)[1]])
            for row in self.states], dtype=np.int64)
        self._build_demand()
        s = np.arange(self.n)
        fraction = s / self.capacity
        self.success = csr_matrix((np.r_[fraction, 1 - fraction[:-1]],
                                   (np.r_[s, s[:-1]], np.r_[s, s[1:]])), shape=(self.n, self.n))
        self.failure = csr_matrix((np.r_[1 - fraction, fraction[1:]],
                                   (np.r_[s, s[1:]], np.r_[s, s[:-1]])), shape=(self.n, self.n))
        self.success.eliminate_zeros()
        self.failure.eliminate_zeros()
        rows = self.states.ravel()
        columns = np.repeat(self._rows, self.sellers)
        self.own_marginal = csr_matrix((np.full(len(rows), 1 / self.sellers), (rows, columns)),
                                       shape=(self.n, self.size))
        self.sale_marginal = csr_matrix((self.selection.ravel() / self.sellers, (rows, columns)),
                                        shape=(self.n, self.size))
        self._edges = []
        for slot in range(self.sellers):
            own = self.states[:, slot]
            for direction in (-1, 1):
                keep = own > 0 if direction == -1 else own < self.capacity
                changed = self.states[keep].copy()
                changed[:, slot] += direction
                columns = self.orbit_index(changed)
                fraction = own[keep] / self.capacity
                replacement = fraction if direction == -1 else 1 - fraction
                self._edges.append((self._rows[keep], columns, own[keep],
                                    self.selection[keep, slot] * replacement, direction))

    def orbit_index(self, coordinates):
        """Index any sorted or unsorted integer seller-count tuple."""
        coordinates = np.asarray(coordinates)
        if (coordinates.shape[-1:] != (self.sellers,) or np.any(~np.isfinite(coordinates))
                or np.any(coordinates != np.floor(coordinates))
                or np.any(coordinates < 0) or np.any(coordinates >= self.n)):
            raise ValueError("seller counts must be integer tuples in [0,capacity]")
        ordered = np.sort(coordinates.astype(np.int64), axis=-1)
        return self._orbit_map[_colex_rank(ordered, self._choose)]

    def _build_demand(self):
        s = np.arange(self.n)
        a, b = 1 + s, 1 + self.capacity - s
        cdf, survival = betainc(a, b, self.p0), betaincc(a, b, self.p0)
        self.outside = np.prod(cdf[self.states], axis=1)
        with np.errstate(divide="ignore"):
            market = -np.expm1(np.log1p(-survival[self.states]).sum(axis=1))
        raw_lookup = _raw_lookup(self.capacity, self.p0, self.rivals, self.order)
        doubled_lookup = _raw_lookup(self.capacity, self.p0, self.rivals, 2 * self.order)
        raw = raw_lookup[self.states, self._rival_ids]
        total = raw.sum(axis=1)
        tiny = np.finfo(float).tiny
        relative_partition = float(np.max(abs(total - market) / np.maximum(market, tiny)))
        doubling_abs = float(np.max(abs(raw_lookup - doubled_lookup)))
        doubling_relative = float(np.max(abs(raw_lookup - doubled_lookup) / np.maximum(doubled_lookup, tiny)))
        self.demand_diagnostics = dict(
            capacity=self.capacity, sellers=self.sellers, p0=self.p0, order=self.order,
            full_states=self.full_size, quotient_states=self.size,
            representation="Exact seller-permutation quotient with correlated joint rival beliefs",
            quadrature="Positive truncated Gauss-Jacobi; independently doubled order",
            raw_partition_error=float(np.max(abs(total + self.outside - 1))),
            raw_relative_selected_mass_error=relative_partition,
            order_doubling_max_abs=doubling_abs,
            order_doubling_max_relative_tail=doubling_relative,
            positive_tail_underflow_entries=int(np.sum(doubled_lookup == 0)))
        if (np.any(~np.isfinite(raw_lookup)) or np.any(raw_lookup <= 0)
                or np.any(raw_lookup > 1 + 1e-10) or relative_partition > 2e-7
                or doubling_abs > 2e-9 or doubling_relative > 2e-6):
            raise ArithmeticError(f"Count-state demand quadrature failed: {self.demand_diagnostics}")
        correction = market / total
        self.selection = self.demand = raw * correction[:, None]
        self.focal_demand = np.empty_like(raw_lookup)
        for own in range(self.n):
            joint_states = np.column_stack((np.full(len(self.rivals), own), self.rivals))
            self.focal_demand[own] = raw_lookup[own] * correction[self.orbit_index(joint_states)]
        self.marginal_demand = sum(
            self.focal_demand[:, self._rival_ids[:, slot]] / self.sellers
            for slot in range(self.sellers))
        self.demand_diagnostics.update(
            partition_error=float(np.max(abs(self.selection.sum(axis=1) + self.outside - 1))),
            relative_event_correction=float(np.max(abs(self.selection - raw) / np.maximum(raw, tiny))),
            min_positive_demand=float(self.selection.min()))

    def policies(self, alpha):
        alpha = np.asarray(alpha, dtype=float)
        if alpha.shape != (self.n,) or np.any(~np.isfinite(alpha)) or np.any((alpha < 0) | (alpha > 1)):
            raise ValueError("alpha must be one shared finite probability vector with capacity+1 entries")
        return alpha

    def off_diagonal(self, alpha, p1=.35, p2=.8):
        """Direct positive state-change flows, retaining arbitrarily rare exits."""
        alpha = self.policies(alpha)
        if not 0 < p1 <= p2 < 1:
            raise ValueError("success probabilities must satisfy 0<p1<=p2<1")
        q = p1 + (p2 - p1) * alpha
        rows, columns, weights = [], [], []
        for rr, cc, own, base, direction in self._edges:
            probability = q[own] if direction == 1 else 1 - q[own]
            rows.append(rr)
            columns.append(cc)
            weights.append(base * probability)
        return csr_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
                          shape=(self.size, self.size))

    def transition(self, alpha, p1=.35, p2=.8):
        """Full quotient calendar transition, including all holding events."""
        off = self.off_diagonal(alpha, p1, p2)
        rates = np.asarray(off.sum(axis=1)).ravel()
        return (off + diags(1 - rates)).tocsr()

    def jump(self, alpha, p1=.35, p2=.8):
        """Zero-diagonal jump chain and direct positive calendar exit rates."""
        jump = self.off_diagonal(alpha, p1, p2)
        rates = np.asarray(jump.sum(axis=1)).ravel()
        if np.any(~np.isfinite(rates)) or np.any(rates <= 0):
            raise ArithmeticError("Count-state jump chain lost a positive exit rate")
        jump.data /= np.repeat(rates, np.diff(jump.indptr))
        return jump, rates

    def transition_derivative(self, k, p1=.35, p2=.8, off_diagonal_only=False):
        """Analytic derivative of the quotient transition with respect to alpha[k]."""
        if int(k) != k or not 0 <= k < self.n:
            raise ValueError("policy coordinate must lie in [0,capacity]")
        rows, columns, weights = [], [], []
        for rr, cc, own, base, direction in self._edges:
            keep = own == k
            rows.append(rr[keep])
            columns.append(cc[keep])
            weights.append(direction * (p2 - p1) * base[keep])
        derivative = csr_matrix((np.concatenate(weights), (np.concatenate(rows), np.concatenate(columns))),
                                shape=(self.size, self.size))
        if not off_diagonal_only:
            derivative = derivative - diags(np.asarray(derivative.sum(axis=1)).ravel())
        return derivative.tocsr()
