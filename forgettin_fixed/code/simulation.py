"""Count-only Monte Carlo for opposite-outcome replacement memory.

Calendar periods include the outside option.  Interaction histories condition
on selecting A, while retaining the optimal calendar-time policy.  Before the
capacity is reached, observations accumulate.  Afterwards a new success
replaces a failure, and a new failure replaces a success.  Counts reflect at
zero and n; choosing the outside option does not modify them.
"""
from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

RECORD_DTYPE = np.dtype([
    ('S_before', '<u4'), ('m_before', '<u4'), ('product2', '?'),
    ('success', '?'), ('demand', '<f4'),
])


def _window(solution) -> int:
    n = int(solution.n)
    if n < 1:
        raise ValueError('window length must be positive')
    return n


def policy_method(solution) -> str:
    return str(getattr(solution, 'method', 'exact_opposite_replacement'))


class CountState:
    """Vectorized retained length and success count; no ordered history."""
    def __init__(self, solution, paths):
        self.solution = solution
        self.n = _window(solution)
        if paths < 1:
            raise ValueError('number of paths must be positive')
        self.lengths = np.zeros(paths, dtype=np.int64)
        self.counts = np.zeros(paths, dtype=np.int64)

    def evaluate(self):
        result = self.solution.evaluate(self.lengths, self.counts)
        action, demand = result[:2]
        action = np.broadcast_to(np.asarray(action, dtype=bool), self.lengths.shape)
        demand = np.broadcast_to(np.asarray(demand, dtype=float), self.lengths.shape)
        if not np.all(np.isfinite(demand)) or np.any(demand < 0) or np.any(demand > 1):
            raise ValueError('policy returned an invalid purchase probability')
        return action, demand

    def advance(self, buyers, success):
        buyers = np.broadcast_to(np.asarray(buyers, dtype=bool), self.lengths.shape)
        success = np.broadcast_to(np.asarray(success, dtype=bool), self.lengths.shape)
        ids = np.flatnonzero(buyers)
        if not len(ids):
            return
        full = self.lengths[ids] == self.n
        growing_ids = ids[~full]
        mature_ids = ids[full]
        self.counts[growing_ids] += success[growing_ids]
        self.lengths[growing_ids] += 1
        self.counts[mature_ids] = np.clip(
            self.counts[mature_ids] + 2 * success[mature_ids].astype(np.int64) - 1,
            0, self.n,
        )


def _mean_interval(values, confidence_level=.95):
    values = np.asarray(values, dtype=float)
    estimate = float(values.mean())
    if len(values) < 2:
        return estimate, np.nan, np.nan, np.nan
    se = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else np.nan
    half = NormalDist().inv_cdf(.5 + confidence_level / 2) * se
    return estimate, se, max(0., estimate-half), min(1., estimate+half)


def _ratio_interval(numerator, denominator, confidence_level=.95):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    if denominator.sum() == 0:
        return (np.nan,) * 4
    estimate = float(numerator.sum() / denominator.sum())
    if len(numerator) < 2:
        return estimate, np.nan, np.nan, np.nan
    residual = numerator - estimate * denominator
    se = float(residual.std(ddof=1) / (np.sqrt(len(residual))*denominator.mean())) if len(residual)>1 else np.nan
    half = NormalDist().inv_cdf(.5 + confidence_level / 2) * se
    return estimate, se, max(0., estimate-half), min(1., estimate+half)


def _rates(a, p2, confidence_level):
    result = {}
    for metric, values in [('a_market_share', a), ('product2_unconditional_rate', p2)]:
        for suffix, value in zip(('', '_se', '_ci_lower', '_ci_upper'), _mean_interval(values, confidence_level)):
            result[metric + suffix] = value
    for suffix, value in zip(('', '_se', '_ci_lower', '_ci_upper'), _ratio_interval(p2, a, confidence_level)):
        result['product2_given_a_rate' + suffix] = value
    return result


def simulate_calendar(solution, *, n_paths=20_000, seed=20260809,
                      max_periods=3000, bin_width=25, confidence_level=.95):
    """Return per-period, path-clustered bin, and per-path summaries.

    Two independent uniform vectors are drawn each period, including for paths
    choosing the outside option, allowing common random numbers across n.
    Bin intervals use paths as independent clusters, including the conditional
    product-2 ratio (delta method).  These intervals describe simulation noise.
    """
    if n_paths < 1 or max_periods < 1 or bin_width < 1:
        raise ValueError('paths, periods and bin width must be positive')
    if not 0 < confidence_level < 1:
        raise ValueError('confidence level must lie in (0,1)')
    primitives = solution.primitives
    rng = np.random.default_rng(seed)
    state = CountState(solution, n_paths)
    bins = (max_periods + bin_width - 1) // bin_width
    a_bins = np.zeros((bins, n_paths), dtype=np.int32)
    p2_bins = np.zeros_like(a_bins)
    n_observed = np.zeros(n_paths, dtype=np.int32)
    successes = np.zeros(n_paths, dtype=np.int32)
    period_rows = []
    bin_prescribed = np.zeros(bins, dtype=np.int64)
    bin_count_sum = np.zeros(bins, dtype=np.int64)
    last_p2 = np.full(n_paths, -1, dtype=np.int32)
    for period in range(max_periods):
        product2, demand = state.evaluate()
        buyers = rng.random(n_paths) < demand
        success = rng.random(n_paths) < np.where(product2, primitives.p2, primitives.p1)
        p2 = buyers & product2
        b = period // bin_width
        a_bins[b] += buyers
        p2_bins[b] += p2
        bin_prescribed[b] += int(product2.sum())
        bin_count_sum[b] += int(state.counts.sum())
        period_rows.append({
            'period': period, 'mean_observation_n': float(n_observed.mean()),
            'mean_remembered_successes': float(state.counts.mean()),
            'mean_window_length': float(state.lengths.mean()),
            'product2_prescribed_share': float(product2.mean()),
            'a_chosen_count': int(buyers.sum()),
            'product2_chosen_count': int(p2.sum()),
            **_rates(buyers, p2, confidence_level),
        })
        last_p2[p2] = period
        n_observed += buyers
        successes += buyers & success
        state.advance(buyers, success)
    bin_rows = []
    for b in range(bins):
        start, stop = b*bin_width, min((b+1)*bin_width, max_periods)
        size = stop-start
        bin_rows.append({
            'bin_start': start, 'bin_stop_exclusive': stop,
            'bin_center': .5*(start+stop-1), 'periods_in_bin': size,
            'product2_prescribed_share': bin_prescribed[b]/(n_paths*size),
            'mean_remembered_successes': bin_count_sum[b]/(n_paths*size),
            'a_chosen_count': int(a_bins[b].sum()),
            'product2_chosen_count': int(p2_bins[b].sum()),
            'paths_with_a_count': int(np.count_nonzero(a_bins[b])),
            **_rates(a_bins[b]/size, p2_bins[b]/size, confidence_level),
        })
    path_summary = pd.DataFrame({
        'path': np.arange(1, n_paths+1), 'a_chosen_count': a_bins.sum(axis=0),
        'product2_count': p2_bins.sum(axis=0), 'final_observation_n': n_observed,
        'final_successes': successes, 'final_failures': n_observed-successes,
        'final_remembered_successes': state.counts, 'final_window_length': state.lengths,
        'last_product2_period_within_window': np.where(last_p2>=0, last_p2.astype(float), np.nan),
    })
    frames = [pd.DataFrame(period_rows), pd.DataFrame(bin_rows), path_summary]
    for frame in frames:
        frame['policy_method'] = policy_method(solution)
    return tuple(frames)


def simulate_interactions(solution, *, paths=20, interactions=20_000,
                          seed=20260809, burn_in=0):
    """Return exact pre-choice records and one summary per independent path."""
    if paths < 1 or interactions < 1 or not 0 <= burn_in < interactions:
        raise ValueError('invalid paths, interactions or burn-in')
    primitives = solution.primitives
    state = CountState(solution, paths)
    rng = np.random.default_rng(seed)
    records = np.empty((interactions, paths), dtype=RECORD_DTYPE)
    buyers = np.ones(paths, dtype=bool)
    for time in range(interactions):
        product2, demand = state.evaluate()
        success = rng.random(paths) < np.where(product2, primitives.p2, primitives.p1)
        row = records[time]
        row['S_before'] = state.counts
        row['m_before'] = state.lengths
        row['product2'] = product2
        row['success'] = success
        row['demand'] = demand
        state.advance(buyers, success)
    action = records['product2']
    summary = pd.DataFrame({
        'trajectory': np.arange(1, paths+1),
        'full_product2_frequency': action.mean(axis=0),
        'post_burn_in_frequency': action[burn_in:].mean(axis=0),
        'mean_remembered_successes': records['S_before'][burn_in:].mean(axis=0),
        'success_frequency': records['success'][burn_in:].mean(axis=0),
        'final_remembered_successes': state.counts, 'final_window_length': state.lengths,
        'policy_method': policy_method(solution),
    })
    return records, summary


def interaction_diagnostics(records, *, burn_in=20_000, block_size=10_000):
    """Additional cluster-level statistics for the long interaction experiment."""
    action = records['product2']
    total, paths = action.shape
    if not 0 <= burn_in < total or block_size < 1:
        raise ValueError('invalid burn-in or block size')
    rows = []
    for start in range(burn_in, total, block_size):
        stop = min(start+block_size, total)
        frequency = action[start:stop].mean(axis=0)
        rows.extend({'trajectory': p+1, 'start_interaction': start+1,
                     'end_interaction': stop, 'frequency': float(frequency[p])} for p in range(paths))
    checkpoints = np.unique(np.r_[np.geomspace(1, total, 100).astype(int), total])
    accumulated = np.cumsum(action, axis=0, dtype=np.int32)
    cumulative = pd.DataFrame(accumulated[checkpoints-1]/checkpoints[:, None],
                             columns=[f'path_{p+1}' for p in range(paths)])
    cumulative.insert(0, 'interactions', checkpoints)
    split = (burn_in+total)//2
    first, second = action[burn_in:split].mean(axis=0), action[split:].mean(axis=0)
    stability = pd.DataFrame({'trajectory': np.arange(1,paths+1),
                             'first_segment_frequency': first,
                             'second_segment_frequency': second,
                             'absolute_segment_difference': abs(first-second)})
    return pd.DataFrame(rows), cumulative, stability
