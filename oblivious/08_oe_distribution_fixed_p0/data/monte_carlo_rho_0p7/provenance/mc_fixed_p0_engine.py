"""Streaming calendar Monte Carlo for frozen one- or two-seller grid policies.

Selection probabilities have axes (outside/A[/B], x_A[, x_B]) and encode the
actual Thompson-selection law at the realized state, including the fixed
outside benchmark. The selected seller's next reputation uses the original
solver interpolation lottery; all other reputations remain unchanged.

Four uniforms per path and date, in order selection/action/outcome/projection,
match ``choice_dynamics.simulate_step``. No path-by-date event array is kept.
"""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from oblivious.oe_p0.code.primitives import (
    check_grid, interpolation_matrix, probability_array,
)


_Z95 = NormalDist().inv_cdf(.975)
_COUNTS = ("own_sale_count", "own_product2_count", "b_sale_count",
           "b_product2_count", "outside_count", "customer_product2_count")


def _integer(value, name, *, minimum=1):
    if isinstance(value, (bool, np.bool_)) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _interval(values):
    values = np.asarray(values, dtype=float)
    estimate = float(values.mean())
    se = float(values.std(ddof=1)/np.sqrt(len(values))) if len(values) > 1 else np.nan
    return estimate, se, max(0., estimate-_Z95*se) if np.isfinite(se) else np.nan, min(1., estimate+_Z95*se) if np.isfinite(se) else np.nan


def _ratio_interval(numerator, denominator):
    total = int(denominator.sum())
    if not total:
        return np.nan, np.nan, np.nan, np.nan
    estimate = float(numerator.sum()/total)
    if len(numerator) > 1:
        residual = numerator-estimate*denominator
        se = float(residual.std(ddof=1)/(np.sqrt(len(numerator))*denominator.mean()))
    else:
        se = np.nan
    return estimate, se, max(0., estimate-_Z95*se) if np.isfinite(se) else np.nan, min(1., estimate+_Z95*se) if np.isfinite(se) else np.nan


def _window_statistics(counts, length):
    """All paths are independent clusters; ratios pool their sale counts."""
    own, own2, b, b2, outside, customer2 = counts
    estimates = {
        "own_sale_rate": _interval(own/length),
        "own_product2_rate": _interval(own2/length),
        "product2_given_sale": _ratio_interval(own2, own),
        "customer_product2_rate": _interval(customer2/length),
        "b_sale_rate": _interval(b/length),
        "b_product2_rate": _interval(b2/length),
        "outside_rate": _interval(outside/length),
    }
    result = {}
    for name, values in estimates.items():
        for suffix, value in zip(("", "_se", "_ci_lower", "_ci_upper"), values):
            result[name+suffix] = value
    return result


def simulate_calendar_paths(*, grid, policy, selection, rho, p1, p2,
                            n_paths, periods, seed, bin_width=25,
                            initial_index=None):
    """Return by-period, binned, per-path, and diagnostic streaming summaries.

    ``policy`` is either one common (n,) own-state probability vector or a
    (sellers,n) table. ``selection`` has shape (sellers+1,)+(n,)*sellers,
    with outside first; one or two sellers are supported. Default initial
    reputation is the stored grid node 0.5 for every seller. An explicit
    initial_index may be one integer or a vector of one index per seller.

    Dates and bin endpoints are zero indexed; bin_stop is exclusive. Binned
    intervals are nominal 95% normal intervals treating each independent path
    as one cluster. Product-2 conditional shares pool counts within the bin,
    and are NaN if the bin contains no own sales. All calendar denominators
    include outside purchases and rival purchases.
    """
    grid = check_grid(grid)
    n = len(grid)
    n_paths = _integer(n_paths, "n_paths")
    periods = _integer(periods, "periods")
    bin_width = _integer(bin_width, "bin_width")
    if (not np.isfinite([rho, p1, p2]).all() or not 0 < rho < 1
            or not 0 < p1 < p2 < 1):
        raise ValueError("Require 0 < rho < 1 and 0 < p1 < p2 < 1")
    probabilities = np.asarray(selection, dtype=float)
    if probabilities.ndim not in (2, 3):
        raise ValueError("Selection needs one or two reputation axes")
    sellers = probabilities.ndim-1
    expected_shape = (sellers+1,)+(n,)*sellers
    if probabilities.shape != expected_shape or not np.isfinite(probabilities).all():
        raise ValueError(f"Selection must be finite with shape {expected_shape}")
    if np.any(probabilities < -1e-12) or np.any(probabilities > 1+1e-12):
        raise ValueError("Selection entries must be probabilities")
    mass_error = float(np.max(abs(probabilities.sum(axis=0)-1)))
    if mass_error > 1e-10:
        raise ValueError("Selection probabilities must sum to one at every state")
    # Correct endpoint roundoff only. Preserve all interior selection values.
    probabilities = np.clip(probabilities, 0., 1.)
    cumulative = np.cumsum(probabilities.reshape(sellers+1, -1), axis=0)
    cumulative[-1] = 1.
    action = probability_array(policy)
    if action.shape == (n,):
        action = np.broadcast_to(action, (sellers, n))
    if action.shape != (sellers, n):
        raise ValueError("Policy must have shape (n,) or (sellers,n)")
    if initial_index is None:
        matches = np.flatnonzero(abs(grid-.5) <= 8*np.finfo(float).eps)
        if len(matches) != 1:
            raise ValueError("Supply initial_index if the grid has no unique 0.5 node")
        initial = np.full(sellers, matches[0], dtype=np.intp)
    else:
        initial = np.asarray(initial_index)
        if initial.ndim == 0:
            initial = np.full(sellers, initial.item())
        if (initial.shape != (sellers,) or not np.issubdtype(initial.dtype, np.integer)
                or np.any(initial < 0) or np.any(initial >= n)):
            raise ValueError("Initial indices must be one valid integer per seller")
        initial = initial.astype(np.intp)
    states = np.broadcast_to(initial[:, None], (sellers, n_paths)).copy()
    lower, upper, lower_weight = [], [], []
    for target in (rho*grid, rho*grid+1-rho):
        operator = interpolation_matrix(grid, target)
        # The primitive stores both neighbors, including zero weights.
        if not np.all(np.diff(operator.indptr) == 2):
            raise ValueError("Interpolation operator must retain both neighbors")
        destinations = operator.indices.reshape(n, 2)
        weights = operator.data.reshape(n, 2)
        lower.append(destinations[:, 0]); upper.append(destinations[:, 1])
        lower_weight.append(weights[:, 0])
    lower, upper, lower_weight = map(np.asarray, (lower, upper, lower_weight))
    rng = np.random.default_rng(seed)
    totals = np.zeros((len(_COUNTS), n_paths), dtype=np.int64)
    bin_counts = np.zeros_like(totals)
    success_counts = np.zeros((sellers, n_paths), dtype=np.int64)
    period_rows, bin_rows = [], []
    own_product2_total = b_product2_total = total_own_sales = total_b_sales = 0
    total_outside = 0
    for date in range(periods):
        uniforms = rng.random((n_paths, 4))
        flattened = states[0] if sellers == 1 else states[0]*n+states[1]
        choice = (uniforms[:, 0] >= cumulative[0, flattened]).astype(np.int8)
        if sellers == 2:
            choice += uniforms[:, 0] >= cumulative[1, flattened]
        own_sale = choice == 1
        b_sale = choice == 2
        outside = choice == 0
        own2 = np.zeros(n_paths, dtype=bool)
        b2 = np.zeros(n_paths, dtype=bool)
        for seller in range(sellers):
            selected = np.flatnonzero(choice == seller+1)
            own_state = states[seller, selected]
            product2 = uniforms[selected, 1] < action[seller, own_state]
            successful = uniforms[selected, 2] < np.where(product2, p2, p1)
            outcome = successful.astype(np.int8)
            use_upper = uniforms[selected, 3] >= lower_weight[outcome, own_state]
            states[seller, selected] = np.where(use_upper, upper[outcome, own_state], lower[outcome, own_state])
            success_counts[seller, selected] += successful
            (own2 if seller == 0 else b2)[selected] = product2
        customer2 = own2 | b2
        events = np.asarray((own_sale, own2, b_sale, b2, outside, customer2))
        totals += events
        bin_counts += events
        current = events.sum(axis=1)
        own_count, own2_count, b_count, b2_count, outside_count, customer2_count = map(int, current)
        total_own_sales += own_count; total_b_sales += b_count
        own_product2_total += own2_count; b_product2_total += b2_count
        total_outside += outside_count
        period_rows.append(dict(period=date, n_paths=n_paths,
            **dict(zip(_COUNTS, map(int, current))),
            own_sale_rate=own_count/n_paths, own_product2_rate=own2_count/n_paths,
            b_sale_rate=b_count/n_paths, b_product2_rate=b2_count/n_paths,
            outside_rate=outside_count/n_paths, customer_product2_rate=customer2_count/n_paths,
            product2_given_sale=own2_count/own_count if own_count else np.nan))
        if (date+1) % bin_width == 0 or date+1 == periods:
            start = date-date % bin_width
            length = date-start+1
            bin_rows.append(dict(bin_start=start, bin_stop=date+1,
                bin_stop_exclusive=date+1, bin_center=.5*(start+date),
                periods_in_bin=length, n_paths=n_paths,
                **dict(zip(_COUNTS, map(int, bin_counts.sum(axis=1)))),
                **_window_statistics(bin_counts, length)))
            bin_counts.fill(0)
    if (not np.all(totals[0]+totals[2]+totals[4] == periods)
            or not np.all(totals[1]+totals[3] == totals[5])
            or np.any(totals[1] > totals[0]) or np.any(totals[3] > totals[2])):
        raise AssertionError("Calendar event accounting failed")
    path_rows = {"path": np.arange(n_paths), **dict(zip(_COUNTS, totals))}
    for seller, label in enumerate(("own", "b")[:sellers]):
        path_rows[label+"_success_count"] = success_counts[seller]
        path_rows["final_"+label+"_index"] = states[seller]
        path_rows["final_"+label+"_reputation"] = grid[states[seller]]
    by_period = pd.DataFrame(period_rows)
    binned = pd.DataFrame(bin_rows)
    paths = pd.DataFrame(path_rows)
    diagnostics = dict(sellers=sellers, n=n, n_paths=n_paths, periods=periods,
        seed=seed, bin_width=bin_width, initial_indices=initial.tolist(),
        initial_reputations=grid[initial].tolist(),
        selection_mass_max_error=mass_error,
        uniform_channels=["selection", "action", "outcome", "projection"],
        selection="Categorical actual-state selection law: outside, A, B; no OE perceived demand",
        transition="Original interpolation_matrix lottery after a sale; only selected seller updates",
        initial_state="Mature normalized-boundary reputations; not zero evidence",
        metric="Own product2_given_sale is pooled own product-2 sales / own sales",
        uncertainty="Nominal 95% normal intervals with independent path clusters; conditional zero-sale bins are NaN",
        stored_event_history=False,
        event_totals=dict(own_sale_count=total_own_sales, own_product2_count=own_product2_total,
                          b_sale_count=total_b_sales, b_product2_count=b_product2_total,
                          outside_count=total_outside,
                          customer_product2_count=own_product2_total+b_product2_total),
        full_window=_window_statistics(totals, periods))
    return dict(by_period=by_period, binned=binned, paths=paths, diagnostics=diagnostics)
