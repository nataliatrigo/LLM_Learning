"""Actual one-seller TS against a fixed outside score, on the solver grid.

An outside purchase yields zero profit and leaves reputation unchanged. Four
independent uniforms sample demand, product, success, and stochastic projection.
Demand Bernoulli(beta.sf(p0)) is distributionally identical to drawing the
seller's Beta score and testing score > p0, without an expensive inverse CDF.
"""

import numpy as np
from scipy.stats import t


def simulate(grid, policy, demand, params, *, periods=100000, burn_in=10000,
             replications=20, seed=20261017, initial=.5):
    grid, policy, demand = map(np.asarray, (grid, policy, demand))
    if not np.all((policy >= 0) & (policy <= 1)) or not np.all((demand >= 0) & (demand <= 1)):
        raise ValueError("Invalid action or demand probability")
    if not np.any(grid == initial):
        raise ValueError("Initial reputation must be a grid node")
    total = periods + burn_in
    seeds = np.random.SeedSequence(seed).spawn(replications)
    uniforms = np.stack([np.random.default_rng(s).random((total, 4)) for s in seeds], axis=1)
    targets = np.stack((params.rho * grid, params.rho * grid + 1-params.rho))
    lower = np.clip(np.searchsorted(grid, targets, side="right")-1, 0, len(grid)-2)
    weight = (targets-grid[lower])/(grid[lower+1]-grid[lower])
    index_dtype = np.min_scalar_type(len(grid)-1)
    before = np.zeros((replications, total), dtype=index_dtype)
    sold = np.zeros((replications, total), dtype=bool)
    product2 = np.zeros_like(sold)
    success = np.zeros_like(sold)
    states = np.full(replications, np.flatnonzero(grid == initial)[0], dtype=int)
    for step in range(total):
        before[:, step] = states
        u = uniforms[step]
        sale = u[:, 0] < demand[states]
        high = (u[:, 1] < policy[states]) & sale
        good = (u[:, 2] < params.p1 + (params.p2-params.p1)*high) & sale
        outcome = good.astype(int)
        target = lower[outcome, states] + (u[:, 3] < weight[outcome, states])
        states = np.where(sale, target, states)
        sold[:, step], product2[:, step], success[:, step] = sale, high, good
    rows = []
    for rep in range(replications):
        count = int(sold[rep, burn_in:].sum())
        count2 = int(product2[rep, burn_in:].sum())
        profit = (params.R-params.c1)*count - (params.c2-params.c1)*count2
        rows.append(dict(replication=rep, sales=count, product2_sales=count2,
                         product2_conditional=count2/count if count else np.nan,
                         product2_calendar=count2/periods, sale_rate=count/periods,
                         seller_profit=profit/periods))
    return dict(before_index=before, sold=sold, product2=product2, success=success,
                final_index=states, replications=rows,
                metadata=dict(periods=periods, burn_in=burn_in, replications=replications,
                              seed=seed, initial=initial, mode="grid",
                              replication_seeds=[dict(entropy=s.entropy, spawn_key=list(s.spawn_key)) for s in seeds],
                              uniform_channels=["sale", "product", "success", "projection"],
                              demand="Bernoulli(Beta.sf(p0)); exact marginal TS selection law",
                              outside_purchase="No sale, zero seller profit, no reputation update"))


def ratio_interval(rows):
    """Pooled sales ratio with independent-path cluster delta-method uncertainty.

Replications with zero sales remain in the cluster sample; an entirely empty
pooled denominator is unavailable. This is not the mean of per-path ratios.
"""
    numerator = np.array([r["product2_sales"] for r in rows], dtype=float)
    denominator = np.array([r["sales"] for r in rows], dtype=float)
    n = len(rows)
    if denominator.sum() == 0:
        return dict(mean=None, se=None, ci_low=None, ci_high=None, n_valid=0)
    ratio = numerator.sum()/denominator.sum()
    residual = numerator-ratio*denominator
    se = residual.std(ddof=1)/(np.sqrt(n)*denominator.mean()) if n > 1 else None
    half = t.ppf(.975, n-1)*se if n > 1 else None
    return dict(mean=float(ratio), se=None if se is None else float(se),
                ci_low=None if half is None else float(ratio-half),
                ci_high=None if half is None else float(ratio+half), n_valid=n)


def power_and_sum(matrix, length):
    """Return P**length and I+P+...+P**(length-1) in O(log length)."""
    size = len(matrix)
    power, series = np.eye(size), np.zeros((size, size))
    block_power, block_series = matrix.copy(), np.eye(size)
    while length:
        if length & 1:
            series = series + power @ block_series
            power = power @ block_power
        block_series = block_series + block_power @ block_series
        block_power = block_power @ block_power
        length >>= 1
    return power, series


def finite_window(matrix, grid, demand, policy, params, *, periods, burn_in, initial=.5):
    """Exact grid-kernel expected occupation; ratios of expected sale counts."""
    p = matrix.toarray() if hasattr(matrix, "toarray") else np.asarray(matrix)
    initial_law = np.zeros(len(grid))
    initial_law[np.flatnonzero(grid == initial)[0]] = 1
    at_start = initial_law @ power_and_sum(p, burn_in)[0]
    law = at_start @ power_and_sum(p, periods)[1] / periods
    sale = float(law @ demand)
    count2 = float(law @ (demand*policy))
    return dict(occupation=law, at_window_start=at_start,
                outcomes=dict(sale_rate=sale, product2_calendar=count2,
                              product2_conditional=count2/sale if sale else None,
                              seller_profit=(params.R-params.c1)*sale-(params.c2-params.c1)*count2),
                probability_sum_error=float(abs(law.sum()-1)))
