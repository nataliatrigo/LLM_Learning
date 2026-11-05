"""Exact correlated count models with more than five symmetric sellers.

The historical model and kernel are left unchanged. This adapter removes
implementation limits on seller count while preserving their economic
equations, permutation quotient, numerical tolerances and independent checks.
"""
from __future__ import annotations

from math import comb, factorial, lgamma

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.sparse import csr_matrix

from forgetting.Code.model import Primitives
from .kernel import CountKernel, _multisets, _colex_rank
from .large_count_model import LargeCountExperiment, DIRECT_STATE_LIMIT


class ExtendedCountKernel(CountKernel):
    """The same sorted-tuple quotient, with overflow-safe large-N indexing.

    Only combinatorial table entries within the used diagonal band are built.
    Thus large unrelated binomial coefficients never overflow int64 when the
    actual quotient still fits the explicit state budget. Orbit cardinalities
    are also retained in logarithmic form; their floating-point display values
    are not used to compute any demand, transition, marginal or probability.
    """

    def __init__(self, capacity, sellers, p0, order=128, max_states=250_000):
        if isinstance(capacity, bool) or int(capacity) != capacity or capacity < 1:
            raise ValueError("capacity must be a positive integer")
        if isinstance(sellers, bool) or int(sellers) != sellers or sellers < 1:
            raise ValueError("sellers must be a positive integer")
        if not np.isfinite(p0) or not 0 <= p0 < 1:
            raise ValueError("p0 must be finite and lie in [0,1)")
        if isinstance(order, bool) or int(order) != order or order < 8:
            raise ValueError("quadrature order must be an integer of at least eight")
        if isinstance(max_states, bool) or int(max_states) != max_states or max_states < 2:
            raise ValueError("max_states must be an integer of at least two")
        self.capacity, self.sellers, self.p0 = int(capacity), int(sellers), float(p0)
        size = comb(self.capacity+self.sellers, self.sellers)
        if size > max_states:
            raise ValueError(f"Exact seller quotient needs {size:,} states, exceeding the {max_states:,} state budget")
        self.max_states = int(max_states)
        # Integer Beta posteriors give polynomial integrands after the Jacobi
        # endpoint factor is absorbed. This lower bound covers the worst own
        # state and every rival, and the inherited doubled-order audit remains.
        degree = self.capacity+(self.capacity+1)*(self.sellers-1)
        self.order = max(int(order), (degree+2)//2)
        self.n = self.capacity+1
        self.states = self.representatives = _multisets(self.n, self.sellers)
        self.rivals = _multisets(self.n, self.sellers-1)
        self.size, self.full_size = len(self.states), self.n**self.sellers
        self._choose = np.zeros((self.n+self.sellers, self.sellers+1), dtype=np.int64)
        for column in range(1, self.sellers+1):
            for difference in range(self.capacity):
                row = column+difference
                self._choose[row, column] = comb(row, column)
        self._orbit_map = np.empty(self.size, dtype=np.int64)
        self._orbit_map[_colex_rank(self.states, self._choose)] = np.arange(self.size)
        self._rival_map = np.empty(len(self.rivals), dtype=np.int64)
        self._rival_map[_colex_rank(self.rivals, self._choose)] = np.arange(len(self.rivals))
        self._rows = np.arange(self.size)
        self._rival_ids = np.column_stack([
            self._rival_map[_colex_rank(np.delete(self.states, slot, axis=1), self._choose)]
            for slot in range(self.sellers)])
        self.histograms = np.zeros((self.size, self.n), dtype=np.int64)
        for count in range(self.n):
            self.histograms[:, count] = np.sum(self.states == count, axis=1)
        self.log_orbit_sizes = lgamma(self.sellers+1)-gammaln(self.histograms+1).sum(axis=1)
        self.orbit_sizes = np.exp(self.log_orbit_sizes)
        if self.sellers <= 20:
            # Preserve exact small cardinalities used by historical reference
            # tests, avoiding gamma roundoff when the integers fit int64.
            denominator = [np.prod([factorial(int(count)) for count in row], dtype=object)
                           for row in self.histograms]
            self.orbit_sizes = np.asarray([factorial(self.sellers)//value for value in denominator],
                                         dtype=np.int64)
        self._build_demand()
        s = np.arange(self.n)
        fraction = s/self.capacity
        self.success = csr_matrix((np.r_[fraction, 1-fraction[:-1]],
            (np.r_[s, s[:-1]], np.r_[s, s[1:]])), shape=(self.n, self.n))
        self.failure = csr_matrix((np.r_[1-fraction, fraction[1:]],
            (np.r_[s, s[1:]], np.r_[s, s[:-1]])), shape=(self.n, self.n))
        self.success.eliminate_zeros()
        self.failure.eliminate_zeros()
        rows = self.states.ravel()
        columns = np.repeat(self._rows, self.sellers)
        self.own_marginal = csr_matrix((np.full(len(rows), 1/self.sellers), (rows, columns)),
                                      shape=(self.n, self.size))
        self.sale_marginal = csr_matrix((self.selection.ravel()/self.sellers, (rows, columns)),
                                       shape=(self.n, self.size))
        self._edges = []
        for slot in range(self.sellers):
            own = self.states[:, slot]
            for direction in (-1, 1):
                keep = own > 0 if direction == -1 else own < self.capacity
                changed = self.states[keep].copy()
                changed[:, slot] += direction
                columns = self.orbit_index(changed)
                fraction = own[keep]/self.capacity
                replacement = fraction if direction == -1 else 1-fraction
                self._edges.append((self._rows[keep], columns, own[keep],
                    self.selection[keep, slot]*replacement, direction))
        self.demand_diagnostics.update(
            requested_order=int(order), polynomial_degree_bound=degree,
            state_budget=self.max_states,
            indexing="Exact colex rank; only the used bounded combinatorial band is stored",
            orbit_cardinality="Log cardinalities; displayed float sizes never enter economic calculations")


class ExtendedCountExperiment(LargeCountExperiment):
    """CountExperiment-compatible solver for any seller count within its budget."""

    def __init__(self, capacity, sellers=2, closure="marginal", p0=.5,
                 primitives=None, order=128, tolerance=1e-7,
                 max_states=250_000, direct_state_limit=DIRECT_STATE_LIMIT):
        if closure not in ("monopoly", "marginal", "conditional"):
            raise ValueError("Unsupported oblivious belief closure")
        if (sellers == 1) != (closure == "monopoly"):
            raise ValueError("One seller requires the monopoly closure")
        self.game = ExtendedCountKernel(capacity, sellers, p0, order=order, max_states=max_states)
        self.kernel = self.game
        self.capacity, self.sellers = self.game.capacity, self.game.sellers
        self.closure, self.p0, self.tolerance = closure, float(p0), float(tolerance)
        self.primitives = Primitives(p0=p0) if primitives is None else primitives
        if abs(self.primitives.p0-p0) > 1e-15:
            raise ValueError("Primitives and experiment must use the same outside option")
        if not 0 < self.tolerance <= 1e-7:
            raise ValueError("Use the original or a stricter equilibrium tolerance")
        self.n, self.grid = self.capacity+1, np.arange(self.capacity+1)/self.capacity
        self.S, self.F = self.game.success, self.game.failure
        self.calls, self.last_alpha, self.last_result = 0, None, None
        self.derivatives = None
        self.direct_state_limit = int(direct_state_limit)
        if closure == "conditional" and self.game.size > self.direct_state_limit:
            raise ValueError("Large seller-count extensions currently support marginal OE only")
        self._search_law, self._progress = None, None

    def configuration(self):
        result = super().configuration()
        result.update(seller_limit="No fixed seller-count limit; explicit exact-quotient state budget",
                      state_budget=self.game.max_states,
                      initial_evaluation="Exact binomial mixture over permutation orbits, O(N)")
        return result

    def _initial(self, values):
        """Exact symmetric mature initialization without enumerating 2**N labels."""
        values = np.asarray(values)
        low, high = self.capacity//2, (self.capacity+1)//2
        if values.shape == (self.n,):
            return float((values[low]+values[high])/2)
        if low == high:
            state = np.full((1, self.sellers), low, dtype=int)
            return float(values[self.game.orbit_index(state)[0]])
        high_counts = np.arange(self.sellers+1)
        log_weights = (gammaln(self.sellers+1)-gammaln(high_counts+1)
                       -gammaln(self.sellers-high_counts+1)-self.sellers*np.log(2))
        weights = np.exp(log_weights-logsumexp(log_weights))
        initial_states = np.where(np.arange(self.sellers)[None, :]
                                  < self.sellers-high_counts[:, None], low, high)
        return float(weights@values[self.game.orbit_index(initial_states)])


__all__ = ["ExtendedCountKernel", "ExtendedCountExperiment"]
