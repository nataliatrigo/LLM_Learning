"""Replication-based uncertainty and comparisons against saved stationary laws."""

import numpy as np
from scipy.stats import wasserstein_distance

from oblivious.oe_p0.code.diagnostics import _interval, posterior_mean
from oblivious.equilibrium_comparison.code.simulation import policy_probability


def interval(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(mean=None, se=None, ci_low=None, ci_high=None, n_valid=0)
    estimate = _interval(values)
    ci = estimate['ci95']
    return dict(mean=estimate['mean'], se=estimate['se'],
                ci_low=ci[0] if ci else None, ci_high=ci[1] if ci else None,
                n_valid=len(values))


def product_usage_evolution(product, *, smoothing_window=100):
    """Estimate q2(t) across paths from period zero, without discarding burn-in.

    ``product[r, t]`` records the selected seller's product in replication r.
    Smoothing is a trailing mean within each independent path, with shorter
    windows at the beginning. Confidence intervals use those path-level means;
    periods within a path are never counted as independent replications.
    """
    product = np.asarray(product)
    if (product.ndim != 2 or product.shape[0] < 2 or product.shape[1] < 1
            or not np.all((product == 1) | (product == 2))):
        raise ValueError('Expected products 1/2 in at least two nonempty replication rows.')
    if (isinstance(smoothing_window, bool) or int(smoothing_window) != smoothing_window
            or smoothing_window < 1):
        raise ValueError('The smoothing window must be a positive integer.')
    smoothing_window = int(smoothing_window)
    product2 = product == 2
    periods = product.shape[1]
    stops = np.arange(1, periods + 1)
    starts = np.maximum(0, stops - smoothing_window)
    sizes = stops - starts
    prefix = np.pad(np.cumsum(product2, axis=1), ((0, 0), (1, 0)))
    path_means = (prefix[:, stops] - prefix[:, starts]) / sizes
    estimate = _interval(path_means)
    bounds = np.clip(np.asarray(estimate['ci95']), 0, 1)
    return dict(
        time=np.arange(periods), q2=product2.mean(axis=0),
        smoothed=np.asarray(estimate['mean']), se=np.asarray(estimate['se']),
        ci_low=bounds[:, 0], ci_high=bounds[:, 1], window_size=sizes,
        n_replications=product.shape[0], smoothing_window=smoothing_window,
        definition='q2(t) = mean_r 1{product[r,t] == 2}; selected seller, one purchase per calendar period.',
        smoothing='Trailing mean over up to smoothing_window periods, including t; no future observations.',
        uncertainty='Pointwise 95% Student-t intervals across independent path-level moving averages; clipped to [0,1].',
        start_period=0, burn_in_discarded=False,
    )


def cdf_distance(states, grid, weights):
    """Exact empirical-versus-discrete KS distance, including both jump limits."""
    ordered = np.sort(states)
    support = np.unique(np.r_[ordered, grid])
    predicted = np.r_[0., np.cumsum(weights)]
    errors = [np.max(np.abs(np.searchsorted(ordered, support, side=side) / len(ordered)
                           - predicted[np.searchsorted(grid, support, side=side)]))
              for side in ('left', 'right')]
    return float(max(errors))


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else np.nan


def summarize_window(pair, simulation, *, mode, window, start, length, short_periods=1000,
                     include_distributions=False):
    """Use path-level means; paired differences preserve common-random-number pairing."""
    p, grid = pair['params'], pair['grid']
    edges = np.r_[-np.finfo(float).eps, (grid[:-1]+grid[1:])/2, 1+np.finfo(float).eps]
    summaries, differences, rows, distributions, running = [], [], [], {}, {}
    grouped = {}
    for name, raw in simulation['profiles'].items():
        profile = pair['profiles'][name]
        predicted_mu = np.stack((profile['nu'].sum(axis=1), profile['nu'].sum(axis=0)))
        predicted_mean = predicted_mu @ grid
        predicted_var = predicted_mu @ (grid**2) - predicted_mean**2
        states = raw['before'][:, start:start+length]
        if states.shape[1] != length:
            raise ValueError('Requested analysis window exceeds stored simulation')
        rep_rows, cdfs, masses, joints = [], [], [], []
        for rep, x in enumerate(states):
            sel = raw['selected'][rep, start:start+length]
            prod = raw['product'][rep, start:start+length]
            success = raw['success'][rep, start:start+length]
            profit = raw['profit'][rep, start:start+length]
            theta = raw['theta'][rep, start:start+length]
            product2 = prod == 2
            delta = np.abs(policy_probability(grid, pair['profiles']['oe']['policy'], x, mode)
                           - policy_probability(grid, pair['profiles']['p0']['policy'], x, mode))
            row = dict(mode=mode, window=window, profile=name, replication=rep,
                       product2_market=float(product2.mean()), industry_profit=float(profit.sum(axis=1).mean()),
                       success_rate=float(success.mean()), failure_rate=float(1-success.mean()),
                       expected_success_probability=float(p.p1+(p.p2-p.p1)*product2.mean()),
                       disagreement_market=float(delta[np.arange(length), sel].mean()),
                       disagreement_occupancy=float(delta.mean()))
            variance = x.var(axis=0)
            row['correlation'] = (float(np.mean((x[:, 0]-x[:, 0].mean())*(x[:, 1]-x[:, 1].mean()))
                                        / np.sqrt(np.prod(variance))) if np.all(variance > 0) else np.nan)
            for j, seller in enumerate(('A', 'B')):
                selected = sel == j
                count = selected.sum()
                post = posterior_mean(x[:, j], p)
                row.update({
                    'market_share_'+seller: float(selected.mean()),
                    'product2_conditional_'+seller: _ratio(np.sum(selected & product2), count),
                    'product2_calendar_'+seller: float(np.mean(selected & product2)),
                    'profit_'+seller: float(profit[:, j].mean()),
                    'mean_reputation_'+seller: float(x[:, j].mean()),
                    'variance_reputation_'+seller: float(variance[j]),
                    'mean_posterior_'+seller: float(post.mean()),
                    'mean_ts_'+seller: float(theta[:, j].mean()),
                    'mean_error_'+seller: float(x[:, j].mean()-predicted_mean[j]),
                    'variance_error_'+seller: float(variance[j]-predicted_var[j]),
                    'wasserstein_'+seller: float(wasserstein_distance(x[:, j], grid, v_weights=predicted_mu[j])),
                    'cdf_max_error_'+seller: cdf_distance(x[:, j], grid, predicted_mu[j]),
                    'disagreement_occupancy_'+seller: float(delta[:, j].mean()),
                    'disagreement_selected_'+seller: _ratio(delta[selected, j].sum(), count),
                    'discounted_profit_'+seller: float(p.gamma**np.arange(short_periods)
                                                       @ raw['profit'][rep, :short_periods, j]),
                })
            row['discounted_industry_profit'] = row['discounted_profit_A']+row['discounted_profit_B']
            if name == 'p0':
                row['p0_ts_error'] = row['mean_ts_B']-profile['p0']
                row['p0_mean_error'] = row['mean_posterior_B']-profile['p0']
            histogram = np.histogram2d(x[:, 0], x[:, 1], bins=(edges, edges))[0]/length
            if mode == 'grid':
                alpha = profile['policy']
                # Constructed once by the loader and reused here.
                matrix = profile['transition']
                row['empirical_stationarity_residual'] = float(np.abs(histogram.ravel() @ matrix-histogram.ravel()).sum())
            if include_distributions:
                joints.append(histogram)
                masses.append(np.stack((histogram.sum(axis=1), histogram.sum(axis=0))))
                cdfs.append(np.stack([np.searchsorted(np.sort(x[:, j]), grid, side='right')/length
                                      for j in (0, 1)]))
            rep_rows.append(row)
        metric_names = [k for k in rep_rows[0] if k not in ('mode','window','profile','replication')]
        grouped[name] = {metric: np.array([r[metric] for r in rep_rows]) for metric in metric_names}
        summaries.extend(dict(mode=mode, window=window, profile=name, metric=metric,
                              **interval(values)) for metric, values in grouped[name].items())
        rows.extend(rep_rows)
        if include_distributions:
            bands = _interval(np.array(cdfs))
            ci = np.array(bands['ci95'])
            distributions[name] = dict(grid=grid.copy(), cdf=np.array(bands['mean']),
                cdf_low=np.clip(ci[..., 0], 0, 1), cdf_high=np.clip(ci[..., 1], 0, 1),
                mass=np.mean(masses, axis=0), joint=np.mean(joints, axis=0), predicted_mu=predicted_mu,
                histogram_edges=edges,
                histogram_interpretation='Nearest-node cells for display; CDF/Wasserstein use actual unrounded states.')
            times = np.unique(np.r_[1, np.geomspace(1, length, min(500, length)).astype(int), length])
            curves = {'time': times}
            for label, observations in (
                ('ts_B', raw['theta'][:, start:start+length, 1]),
                ('posterior_B', posterior_mean(states[:, :, 1], p)),
            ):
                path_means = np.cumsum(observations, axis=1)[:, times-1]/times
                band = _interval(path_means)
                bounds = np.array(band['ci95'])
                curves.update({label: np.array(band['mean']), label+'_low': bounds[:, 0], label+'_high': bounds[:, 1]})
            running[name] = curves
    if set(grouped) == {'oe', 'p0'}:
        for metric in sorted(set(grouped['oe']) & set(grouped['p0'])):
            differences.append(dict(mode=mode, window=window, metric=metric,
                **interval(grouped['oe'][metric]-grouped['p0'][metric])))
    return dict(summary=summaries, differences=differences, replications=rows,
                distributions=distributions, running=running)


def policy_comparison(pair):
    grid = np.linspace(0, 1, 10001)
    oe = policy_probability(pair['grid'], pair['profiles']['oe']['policy'], grid, 'nearest')
    p0 = policy_probability(pair['grid'], pair['profiles']['p0']['policy'], grid, 'nearest')
    difference = np.abs(oe-p0)
    return dict(grid_dense=grid, oe=oe, p0=p0, absolute_difference=difference,
                uniform_mean_absolute_difference=float(np.abs(pair['profiles']['oe']['policy']-pair['profiles']['p0']['policy'])
                    @ np.diff(np.r_[0., (pair['grid'][:-1]+pair['grid'][1:])/2, 1.])),
                definition='Absolute probability difference, integrated exactly over nearest-node cells on [0,1].')
