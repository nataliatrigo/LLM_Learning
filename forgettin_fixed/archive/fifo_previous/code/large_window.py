"""Numerical FIFO policies for windows too large for exhaustive enumeration.

The planner retains the oldest ``k`` outcomes in order and the success count
in the remaining tail.  Its planning closure treats that tail as exchangeable.
Execution uses the *actual* next tail bit when greedily evaluating the solved
value function; the consumer's queue is always a genuine FIFO queue.

The lifted value function's Bellman residual on the full, ordered FIFO state
space is computed exhaustively: a projected successor depends on only one
additional bit.  Thus the reported uniform value-error bound is rigorous
(apart from floating-point roundoff), although it can be conservative.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import betaincc

from forgetting.Code.model import Primitives


@lru_cache(maxsize=32)
def _popcounts(k: int) -> np.ndarray:
    return np.fromiter((i.bit_count() for i in range(1 << k)), dtype=np.int16)


def _waiting(demand: np.ndarray, gamma: float) -> np.ndarray:
    return demand / (1.0 - gamma * (1.0 - demand))


def _tail_offsets(n: int, k: int) -> np.ndarray:
    lengths = np.arange(n - k + 1, dtype=np.int64)
    return lengths * (lengths + 1) // 2


def _demand_table(n: int, p0: float) -> np.ndarray:
    out = np.empty((n + 1) * (n + 2) // 2, dtype=float)
    for m in range(n + 1):
        s = np.arange(m + 1)
        start = m * (m + 1) // 2
        out[start:start + m + 1] = betaincc(1 + s, 1 + m - s, p0)
    return out


@dataclass
class LargeWindowPolicy:
    n: int
    primitives: Primitives
    prefix_bits: int
    mature_value: np.ndarray
    transient_advantage: np.ndarray
    small_advantage: np.ndarray
    demand_table: np.ndarray
    diagnostics: dict
    action_tolerance: float = 1e-6
    method: str = "projected_fifo"

    def evaluate_state(self, lengths, successes, prefix, next_bit=None):
        """Evaluate scalar/vector states; prefix is oldest-first, MSB first.

        ``lengths`` is capped at n; ``prefix`` contains min(k,lengths) bits.
        For a mature queue, ``next_bit`` is its (k+1)-th oldest actual outcome.
        """
        lengths, successes, prefix = np.broadcast_arrays(
            np.asarray(lengths, dtype=np.int64),
            np.asarray(successes, dtype=np.int64),
            np.asarray(prefix, dtype=np.int64),
        )
        if np.any((lengths < 0) | (lengths > self.n)):
            raise ValueError("remembered lengths must lie in [0,n]")
        if np.any((successes < 0) | (successes > lengths)):
            raise ValueError("invalid success count")
        shape = lengths.shape
        ms, ss, ps = lengths.ravel(), successes.ravel(), prefix.ravel()
        gap = np.empty(ms.size, dtype=float)
        k = self.prefix_bits
        small = ms < k
        if np.any(small):
            gap[small] = self.small_advantage[(1 << ms[small]) - 1 + ps[small]]
        growing = (ms >= k) & (ms < self.n)
        pc = _popcounts(k)
        if np.any(growing):
            tails = ms[growing] - k
            offsets = tails * (tails + 1) // 2
            j = ss[growing] - pc[ps[growing]]
            gap[growing] = self.transient_advantage[ps[growing], offsets + j]
        mature = ms == self.n
        if np.any(mature):
            if next_bit is None:
                raise ValueError("actual next tail bit is required for mature FIFO states")
            bs = np.broadcast_to(np.asarray(next_bit, dtype=np.int64), shape).ravel()[mature]
            if np.any((bs < 0) | (bs > 1)):
                raise ValueError("next_bit must be binary")
            j = ss[mature] - pc[ps[mature]]
            tail_length = self.n - k
            if np.any((j < bs) | (j > tail_length - 1 + bs)):
                raise ValueError("next bit is inconsistent with the tail success count")
            nprefix = ((ps[mature] << 1) & ((1 << k) - 1)) | bs
            vf = self.mature_value[nprefix, j - bs]
            vs = self.mature_value[nprefix, j - bs + 1]
            gap[mature] = -self.primitives.delta_c + self.primitives.gamma * self.primitives.delta_p * (vs - vf)
        demand = self.demand_table[ms * (ms + 1) // 2 + ss]
        gap = gap.reshape(shape)
        return gap > self.action_tolerance, demand.reshape(shape), gap

    def evaluate_queue(self, queue, lengths):
        """Return (product2,demand) for oldest-first queue shape (paths,n)."""
        queue = np.asarray(queue, dtype=np.uint8)
        if queue.ndim == 1:
            queue = queue[None, :]
        lengths = np.broadcast_to(np.asarray(lengths, dtype=np.int64), (queue.shape[0],))
        successes = np.sum(queue * (np.arange(queue.shape[1])[None, :] < lengths[:, None]), axis=1)
        prefix = np.zeros(queue.shape[0], dtype=np.int64)
        for age in range(min(self.prefix_bits, queue.shape[1])):
            prefix = np.where(lengths > age, 2 * prefix + queue[:, age], prefix)
        next_bit = queue[:, self.prefix_bits] if queue.shape[1] > self.prefix_bits else 0
        action, demand, _ = self.evaluate_state(lengths, successes, prefix, next_bit)
        return action, demand

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"format": "projected-fifo-v1", "n": self.n, "prefix_bits": self.prefix_bits,
                    "primitives": self.primitives.as_dict(),
                    "diagnostics": self.diagnostics,
                    "action_tolerance": self.action_tolerance, "method": self.method}
        np.savez_compressed(path, metadata=json.dumps(metadata),
                            mature_value=self.mature_value,
                            transient_advantage=self.transient_advantage,
                            small_advantage=self.small_advantage,
                            demand_table=self.demand_table)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            meta.pop("format", None)
            meta["primitives"] = Primitives.from_mapping(meta["primitives"])
            return cls(**meta, mature_value=data["mature_value"],
                       transient_advantage=data["transient_advantage"],
                       small_advantage=data["small_advantage"],
                       demand_table=data["demand_table"])

    def summary_by_count(self):
        """Mature action ranges over ordered histories, not count-only policy."""
        k, n, p = self.prefix_bits, self.n, self.primitives
        pc = _popcounts(k)
        prefixes = np.arange(1 << k)
        base = (prefixes << 1) & ((1 << k) - 1)
        j = np.arange(n - k + 1)
        counts = pc[:, None] + j
        amin = np.full(n + 1, np.inf)
        amax = np.full(n + 1, -np.inf)
        selected = np.zeros(n + 1, dtype=np.int64)
        states = np.zeros(n + 1, dtype=np.int64)
        for b in (0, 1):
            valid = (j >= b) & (j <= n - k - 1 + b)
            js = j[valid] - b
            vf = self.mature_value[(base + b)[:, None], js]
            vs = self.mature_value[(base + b)[:, None], js + 1]
            gap = -p.delta_c + p.gamma * p.delta_p * (vs - vf)
            cs = counts[:, valid].ravel()
            np.minimum.at(amin, cs, gap.ravel())
            np.maximum.at(amax, cs, gap.ravel())
            np.add.at(selected, cs, (gap > self.action_tolerance).ravel())
            np.add.at(states, cs, 1)
        return pd.DataFrame({"n": n, "p0": p.p0, "successes": np.arange(n + 1),
                             "posterior_mean": (1 + np.arange(n + 1)) / (n + 2),
                             "advantage_min": amin, "advantage_max": amax,
                             "projected_contexts": states,
                             "product2_contexts": selected,
                             "product2_context_share": selected / states,
                             "method": self.method, "prefix_bits": k})


def solve_large_window(n: int, primitives: Primitives, *, prefix_bits=8,
                       tolerance=1e-9, max_iterations=100_000,
                       action_tolerance=1e-6) -> LargeWindowPolicy:
    """Solve the exchangeable-tail planner and certify its lifted FIFO residual."""
    started = time.monotonic()
    n, k = int(n), int(prefix_bits)
    if not 1 <= k < n:
        raise ValueError("require 1 <= prefix_bits < n")
    p = primitives
    prefixes = np.arange(1 << k)
    pc = _popcounts(k)
    tail_length = n - k
    j = np.arange(tail_length + 1)
    base = (prefixes << 1) & ((1 << k) - 1)
    demand_table = _demand_table(n, p.p0)
    counts = pc[:, None] + j
    d = demand_table[n * (n + 1) // 2 + counts]
    lam = _waiting(d, p.gamma)
    probs = j / tail_length
    value = np.zeros_like(lam)

    def expected_successors(v):
        vf = np.zeros_like(v)
        vs = np.zeros_like(v)
        for b, weight in ((0, 1 - probs), (1, probs)):
            js = np.clip(j - b, 0, tail_length)
            success_js = np.clip(j - b + 1, 0, tail_length)
            vf += weight * v[(base + b)[:, None], js]
            vs += weight * v[(base + b)[:, None], success_js]
        return vf, vs

    r1, r2 = p.net_rewards
    for iteration in range(1, max_iterations + 1):
        vf, vs = expected_successors(value)
        q1 = r1 + p.gamma * (p.p1 * vs + (1 - p.p1) * vf)
        q2 = r2 + p.gamma * (p.p2 * vs + (1 - p.p2) * vf)
        updated = lam * np.maximum(q1, q2)
        residual = float(np.max(np.abs(updated - value)))
        value = updated
        if residual <= tolerance:
            break
    else:
        raise RuntimeError(f"projected FIFO iteration failed: n={n}, k={k}")

    # Exhaustive full-history residual: only the first hidden tail bit matters.
    mature_residual = 0.0
    signed_residual_min, signed_residual_max = 0.0, 0.0
    residual_values = []
    for b in (0, 1):
        valid = (j >= b) & (j <= tail_length - 1 + b)
        js = j[valid] - b
        vf = value[(base + b)[:, None], js]
        vs = value[(base + b)[:, None], js + 1]
        q1 = r1 + p.gamma * (p.p1 * vs + (1 - p.p1) * vf)
        q2 = r2 + p.gamma * (p.p2 * vs + (1 - p.p2) * vf)
        signed_errors = lam[:, valid] * np.maximum(q1, q2) - value[:, valid]
        errors = np.abs(signed_errors)
        signed_residual_min = min(signed_residual_min, float(np.min(signed_errors)))
        signed_residual_max = max(signed_residual_max, float(np.max(signed_errors)))
        mature_residual = max(mature_residual, float(np.max(errors)))
        residual_values.append(errors.ravel())

    # Grow the queue without eviction.  These transitions are exact for the
    # lifted terminal value, including every ordered prefix and success count.
    offsets = _tail_offsets(n, k)
    transient = np.empty((1 << k, int(offsets[-1])), dtype=np.float32)
    small = np.empty((1 << k) - 1, dtype=np.float32)
    next_value = value
    rounding_residual = 0.0
    for m in range(n - 1, k - 1, -1):
        tail = m - k
        vf, vs = next_value[:, :-1], next_value[:, 1:]
        advantage = -p.delta_c + p.gamma * p.delta_p * (vs - vf)
        packed = advantage.astype(np.float32)
        rounding_residual = max(rounding_residual, float(np.max(np.abs(packed - advantage))))
        transient[:, offsets[tail]:offsets[tail + 1]] = packed
        s = pc[:, None] + np.arange(tail + 1)
        demand = demand_table[m * (m + 1) // 2 + s]
        q1 = r1 + p.gamma * (p.p1 * vs + (1 - p.p1) * vf)
        q2 = r2 + p.gamma * (p.p2 * vs + (1 - p.p2) * vf)
        next_value = _waiting(demand, p.gamma) * np.maximum(q1, q2)
    next_value = next_value[:, 0]
    for m in range(k - 1, -1, -1):
        vf, vs = next_value[0::2], next_value[1::2]
        advantage = -p.delta_c + p.gamma * p.delta_p * (vs - vf)
        packed = advantage.astype(np.float32)
        rounding_residual = max(rounding_residual, float(np.max(np.abs(packed - advantage))))
        small[(1 << m) - 1:(1 << (m + 1)) - 1] = packed
        success_counts = _popcounts(m)
        demand = demand_table[m * (m + 1) // 2 + success_counts]
        q1 = r1 + p.gamma * (p.p1 * vs + (1 - p.p1) * vf)
        q2 = r2 + p.gamma * (p.p2 * vs + (1 - p.p2) * vf)
        next_value = _waiting(demand, p.gamma) * np.maximum(q1, q2)

    all_errors = np.concatenate(residual_values)
    diagnostics = {"method": "projected_fifo", "prefix_bits": k,
                   "projected_mature_states": int(value.size),
                   "exact_mature_state_count": str(2 ** n),
                   "iterations": iteration, "projected_iteration_residual": residual,
                   "fifo_bellman_residual_sup": mature_residual,
                   "fifo_bellman_residual_min": signed_residual_min,
                   "fifo_bellman_residual_max": signed_residual_max,
                   "transient_bellman_residual_sup": 0.0,
                   "transient_advantage_float32_max_error": rounding_residual,
                   "uniform_value_error_bound": mature_residual / (1 - p.gamma),
                   "uniform_greedy_policy_loss_bound": min(r1 / (1 - p.gamma), (signed_residual_max - signed_residual_min + rounding_residual + action_tolerance) / (1 - p.gamma)),
                   "empty_initial_value_error_bound": p.gamma ** n * mature_residual / (1 - p.gamma),
                   "empty_initial_greedy_policy_loss_bound": min(r1 / (1 - p.gamma), (p.gamma ** n * (signed_residual_max - signed_residual_min) + rounding_residual + action_tolerance) / (1 - p.gamma)),
                   "empty_initial_optimal_value_lower_bound": float(next_value[0]) + p.gamma ** n * signed_residual_min / (1 - p.gamma),
                   "empty_initial_optimal_value_upper_bound": float(next_value[0]) + p.gamma ** n * signed_residual_max / (1 - p.gamma),
                   "residual_context_median": float(np.median(all_errors)),
                   "residual_context_q95": float(np.quantile(all_errors, .95)),
                   "initial_value_projected_terminal": float(next_value[0]),
                   "solve_seconds": time.monotonic() - started,
                   "approximation": "oldest-k ordered bits plus exchangeable tail count; greedy execution uses actual next FIFO bit",
                   "certificate_note": "global sup residual over all ordered FIFO histories; bounds exclude floating-point roundoff; context quantiles are unweighted projected contexts, not a history distribution"}
    return LargeWindowPolicy(n, p, k, value, transient, small, demand_table,
                             diagnostics, action_tolerance)


# Consistent entry points for the experiment runner.
ProjectedSolution = LargeWindowPolicy


def solve_projected(primitives: Primitives, n: int, *, k=8, **kwargs):
    return solve_large_window(n, primitives, prefix_bits=k, **kwargs)
