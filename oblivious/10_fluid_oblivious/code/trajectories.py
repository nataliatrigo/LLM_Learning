"""Deterministic actual-state paths under frozen fluid oblivious policies.

Each policy was optimized against a fixed conjectured rival reputation. During
the realized trajectory, selection uses BOTH actual reputations. The state is
on the mature normalized boundary, matching the experiment-08 Monte Carlo
initial condition. Counters integrate D and D*alpha separately, so the share
among own sales is never confused with a calendar-time product-2 rate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.special import betainc, betaincc

from oblivious.oe_p0.code.primitives import Parameters, posterior


@dataclass
class _Policy:
    params: Parameters
    roots: list[float]
    products: list[int]
    holding_points: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if len(self.products) != len(self.roots) + 1:
            raise ValueError("One product is required for each policy interval")
        if (any(not 0 < root < 1 for root in self.roots)
                or any(a >= b for a, b in zip(self.roots, self.roots[1:]))):
            raise ValueError("Policy roots must increase strictly inside (0,1)")
        if any(product not in (1, 2) for product in self.products):
            raise ValueError("Policy interval products must be 1 or 2")
        for point in self.holding_points:
            x, alpha = float(point["x"]), float(point["alpha"])
            if (not any(abs(x-root) <= 1e-12 for root in self.roots)
                    or not 0 <= alpha <= 1
                    or abs(self.params.p1+(self.params.p2-self.params.p1)*alpha-x) > 1e-9):
                raise ValueError("Explicit holding mixture must have zero drift at a policy root")

    def probability(self, product):
        return self.params.p1 if product == 1 else self.params.p2

    def kind(self, index):
        root = self.roots[index]
        left = self.probability(self.products[index]) - root
        right = self.probability(self.products[index + 1]) - root
        if left > 0 and right < 0:
            return "attractive"
        if left < 0 and right > 0:
            return "repelling"
        return "crossing"

    def holding_alpha(self, root):
        explicit = next((point for point in self.holding_points if abs(point["x"]-root) <= 1e-12), None)
        if explicit is not None:
            return float(explicit["alpha"])
        return (root - self.params.p1)/(self.params.p2 - self.params.p1)

    def initial(self, x):
        if any(abs(x-point["x"]) <= 1e-13 for point in self.holding_points):
            # Only explicitly supplied holding choices extend beyond attractive
            # roots. The equilibrium solver validates their global optimality.
            return int(np.searchsorted(self.roots, x)), True
        for index, root in enumerate(self.roots):
            if abs(x-root) > 1e-13:
                continue
            kind = self.kind(index)
            if kind == "attractive":
                return index, True
            if kind == "repelling":
                return (index if self.products[index] == 1 else index+1), False
            if self.probability(self.products[index+1]) > root:
                return index+1, False
            return index, False
        return int(np.searchsorted(self.roots, x, side="right")), False


def _policy(params, response):
    """Accept the solver's policy mapping, whole equilibrium, or HJB output."""
    if isinstance(response, _Policy):
        return response
    if not isinstance(response, Mapping):
        raise TypeError("response must be an equilibrium or policy mapping")
    if isinstance(response.get("policy"), Mapping):
        response = response["policy"]
    roots = [float(x) for x in response.get("roots", [])]
    if "products" in response:
        products = list(map(int, response["products"]))
    elif "intervals" in response:
        products = [int(interval["product"]) for interval in response["intervals"]]
    else:
        grid = np.asarray(response.get("x", response.get("grid")), dtype=float)
        gap = np.asarray(response["gap"], dtype=float)
        edges = [0., *roots, 1.]
        products = [2 if np.interp((a+b)/2, grid, gap) > 0 else 1
                    for a, b in zip(edges, edges[1:])]
    return _Policy(params, roots, products, list(response.get("holding_points", [])))


def actual_selection(x_a, x_b, params, p0, order=256):
    """Actual joint TS selection, with the exact symmetry identity on diagonal.

    The diagonal expression is the same cancellation-free event partition as
    outside_demand_matrix: a seller gets half the probability that at least
    one posterior draw exceeds the fixed outside option. Off the diagonal,
    reuse the new core's prepared positive quadrature for the same integral.
    """
    x_a, x_b = float(x_a), float(x_b)
    if (not np.isfinite([x_a, x_b, p0]).all()
            or min(x_a, x_b) < -1e-10 or max(x_a, x_b) > 1+1e-10
            or not 0 <= p0 <= 1):
        raise ValueError("Actual reputations and p0 must lie in [0,1]")
    # Correct only solver boundary roundoff before the shared posterior helper.
    x_a, x_b = np.clip([x_a, x_b], 0, 1)
    if x_a == x_b:
        a, b = posterior(x_a, params)
        cdf, survival = float(betainc(a, b, p0)), float(betaincc(a, b, p0))
        own = .5*survival*(1+cdf)
        return np.asarray([own, own, cdf*cdf])
    from oblivious.fluid_oblivious.code.model import demand_at_states
    result = demand_at_states(x_a, x_b, params, p0, order=order)
    if isinstance(result, Mapping):
        return np.asarray([result["demand_a"], result["demand_b"], result["outside"]], dtype=float)
    return np.asarray(result, dtype=float)


def _rates(selection, alpha):
    d_a, d_b, outside = map(float, selection)
    a, b = alpha
    return np.asarray([d_a, d_a*a, d_b, d_b*b]), outside


def _divide(numerator, denominator):
    numerator, denominator = np.broadcast_arrays(numerator, denominator)
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan, dtype=float),
                     where=denominator > 0)


def simulate_pair(
    params, p0, response, *, times=None, initial=(.5, .5), bin_width=25,
    rtol=1e-9, atol=1e-11, max_step=10., demand_order=256,
    demand_function=None, response_b=None,
):
    """Integrate two realized reputations plus sale and product-2 counters.

    ``response`` is the common own-state policy optimized against its fixed
    rival conjecture. ``response_b`` is optional for asymmetric validation.
    ``demand_function(x_a,x_b)`` may provide ``(D_A,D_B,outside)`` for analytical
    tests. Default times cover [0,3000] at integer dates, matching the mature
    (.5,.5), no-burn-in stochastic simulation. State time is continuous.

    Product-2 fractions at attractive roots hold x exactly. The solver stops
    on threshold events before changing actions. Calendar counters remain
    active during holding; no time discretization induced chatter is added.
    """
    times = np.arange(3001, dtype=float) if times is None else np.asarray(times, dtype=float)
    if (times.ndim != 1 or len(times) < 2 or times[0] != 0
            or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)):
        raise ValueError("times must be finite and increase strictly from zero")
    initial = np.asarray(initial, dtype=float)
    if initial.shape != (2,) or not np.isfinite(initial).all() or np.any((initial < 0)|(initial > 1)):
        raise ValueError("initial must contain two mature reputations in [0,1]")
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("bin_width must be finite and positive")
    if not 0 < params.rho < 1 or not 0 < params.gamma < 1 or rtol <= 0 or atol <= 0 or max_step <= 0:
        raise ValueError("Invalid retention or integration tolerances")
    p = [_policy(params, response), _policy(params, response if response_b is None else response_b)]
    lam = -math.log(params.rho)
    delta = -math.log(params.gamma)
    demand = (lambda a, b: actual_selection(a, b, params, p0, order=demand_order)) if demand_function is None else demand_function
    horizon = float(times[-1])
    # Counters are accumulated at exact bin boundaries by the same integration,
    # so binned ratios do not average instantaneous conditional percentages.
    bins = np.r_[np.arange(0, horizon, bin_width), horizon]
    evaluation_times = np.unique(np.r_[times, bins])
    samples = np.full((len(evaluation_times), 8), np.nan)
    sample_alpha = np.full((len(evaluation_times), 2), np.nan)
    y = np.r_[initial, np.zeros(6)]
    region, holding = [], []
    for j in range(2):
        r, h = p[j].initial(float(y[j]))
        region.append(r); holding.append(h)
    alpha = np.zeros(2)
    events = []
    nfev = 0
    t = 0.
    for segment in range(sum(len(policy.roots) for policy in p)+4):
        targets = [None, None]
        event_functions, event_sellers = [], []
        probabilities = np.empty(2)
        for j in range(2):
            if holding[j]:
                alpha[j] = p[j].holding_alpha(float(y[j]))
                probabilities[j] = y[j]
                continue
            product = p[j].products[region[j]]
            alpha[j] = float(product == 2)
            probabilities[j] = p[j].probability(product)
            difference = probabilities[j]-y[j]
            if abs(difference) <= 1e-14:
                continue
            direction = 1 if difference > 0 else -1
            index = region[j] if direction > 0 else region[j]-1
            if not 0 <= index < len(p[j].roots):
                continue
            root = p[j].roots[index]
            if direction*(root-y[j]) <= 1e-13 or direction*(probabilities[j]-root) <= 1e-13:
                continue
            targets[j] = (root, index, direction)

            def event(_t, state, seller=j, target=root):
                return state[seller]-target

            event.terminal = True
            event.direction = direction
            event_functions.append(event)
            event_sellers.append(j)

        def derivative(_t, state):
            selection = np.asarray(demand(float(state[0]), float(state[1])), dtype=float)
            if (selection.shape != (3,) or not np.isfinite(selection).all()
                    or np.any(selection < -1e-12) or abs(selection.sum()-1) > 1e-8):
                raise ArithmeticError("Actual-state selection probabilities do not partition events")
            counters, _outside = _rates(selection, alpha)
            drift = lam*selection[:2]*(probabilities-state[:2])
            drift[np.asarray(holding)] = 0.
            profit = (math.exp(-delta*_t)*selection[:2]
                      *(params.R-params.c1-(params.c2-params.c1)*alpha))
            return np.r_[drift, counters, profit]

        solution = solve_ivp(derivative, (t, horizon), y, method="DOP853", dense_output=True,
                             events=event_functions or None, rtol=rtol, atol=atol, max_step=max_step)
        if not solution.success:
            raise RuntimeError(solution.message)
        nfev += solution.nfev
        end = float(solution.t[-1])
        mask = (evaluation_times >= t) & (evaluation_times <= end)
        samples[mask] = solution.sol(evaluation_times[mask]).T
        sample_alpha[mask] = alpha
        if end >= horizon:
            break
        y = solution.y[:, -1].copy()
        hit = []
        for j, target in enumerate(targets):
            if target is not None and abs(y[j]-target[0]) <= max(1e-10, 10*atol):
                hit.append(j)
        if not hit:
            raise RuntimeError("Threshold integration stopped without a detected event")
        for j in hit:
            root, index, direction = targets[j]
            kind = p[j].kind(index)
            y[j] = root
            record = dict(time=end, seller="A" if j == 0 else "B", x=root, kind=kind,
                          product_before=p[j].products[region[j]])
            if kind == "attractive":
                holding[j] = True
                record["alpha_after"] = p[j].holding_alpha(root)
            else:
                region[j] += direction
                record["product_after"] = p[j].products[region[j]]
            events.append(record)
        t = end
    else:
        raise RuntimeError("Too many threshold events for a one-dimensional stationary policy")
    if np.any(~np.isfinite(samples)) or np.any(~np.isfinite(sample_alpha)):
        raise RuntimeError("Some path samples were not filled")
    indices = np.searchsorted(evaluation_times, times)
    states, alphas = samples[indices], sample_alpha[indices]
    selections = np.asarray([demand(float(a), float(b)) for a, b in states[:, :2]])
    rate_a = selections[:, 0]*alphas[:, 0]
    rate_b = selections[:, 1]*alphas[:, 1]
    frame = pd.DataFrame(dict(
        time=times, x_a=states[:, 0], x_b=states[:, 1],
        alpha_a=alphas[:, 0], alpha_b=alphas[:, 1],
        demand_a=selections[:, 0], demand_b=selections[:, 1],
        own_sale_rate=selections[:, 0], own_product2_rate=rate_a,
        product2_given_sale=np.where(selections[:, 0] > 0, alphas[:, 0], np.nan),
        b_sale_rate=selections[:, 1], b_product2_rate=rate_b,
        outside_rate=selections[:, 2], customer_product2_rate=rate_a+rate_b,
        cumulative_own_sales=states[:, 2], cumulative_own_product2=states[:, 3],
        cumulative_b_sales=states[:, 4], cumulative_b_product2=states[:, 5],
        cumulative_product2_given_sale=_divide(states[:, 3], states[:, 2]),
        cumulative_own_product2_rate=_divide(states[:, 3], times),
        cumulative_customer_product2_rate=_divide(states[:, 3]+states[:, 5], times),
        cumulative_discounted_profit_a=states[:, 6], cumulative_discounted_profit_b=states[:, 7],
    ))
    bin_indices = np.searchsorted(evaluation_times, bins)
    counts = np.diff(samples[bin_indices, 2:6], axis=0)
    durations = np.diff(bins)
    binned = pd.DataFrame(dict(
        bin_start=bins[:-1], bin_stop=bins[1:], bin_center=(bins[:-1]+bins[1:])/2,
        periods_in_bin=durations,
        own_sale_rate=counts[:, 0]/durations,
        own_product2_rate=counts[:, 1]/durations,
        product2_given_sale=_divide(counts[:, 1], counts[:, 0]),
        b_sale_rate=counts[:, 2]/durations,
        b_product2_rate=counts[:, 3]/durations,
        outside_rate=1-(counts[:, 0]+counts[:, 2])/durations,
        customer_product2_rate=(counts[:, 1]+counts[:, 3])/durations,
    ))
    invariant_error = float(max(0., -np.min(states[:, :2]), np.max(states[:, :2])-1))
    if invariant_error > 1e-8 or np.any(states[:, 3] > states[:, 2]+1e-7) or np.any(states[:, 5] > states[:, 4]+1e-7):
        raise AssertionError("Path state or sale accounting invariant failed")
    symmetric = np.array_equal(initial[:1], initial[1:]) and response_b is None and demand_function is None
    symmetry_error = float(np.max(abs(states[:, 0]-states[:, 1]))) if symmetric else None
    if symmetry_error is not None and symmetry_error > 1e-8:
        raise AssertionError("Symmetric policies and initial states generated asymmetric paths")
    diagnostics = dict(
        initial_reputations=initial.tolist(), initial_state="Mature normalized boundary, not empty history",
        horizon=horizon, samples=len(times), bin_width=float(bin_width), nfev=nfev,
        selection="Actual joint Thompson demand at (x_A(t),x_B(t)); fixed conjecture affects policy only",
        calibration="lambda=-log(rho); continuous calendar time", rho=float(params.rho), p0=float(p0),
        invariant_interval_error=invariant_error, symmetry_error=symmetry_error,
        maximum_selection_partition_error=float(np.max(abs(selections.sum(axis=1)-1))),
        own_product2_given_sale=float(_divide(states[-1, 3], states[-1, 2])),
        own_product2_per_calendar_time=float(states[-1, 3]/horizon),
        customer_product2_per_calendar_time=float((states[-1, 3]+states[-1, 5])/horizon),
        value_initial_a=float(states[-1, 6]), value_initial_b=float(states[-1, 7]),
        discounted_profit_tail_bound=float(params.R/delta*math.exp(-delta*horizon)),
        events=len(events),
        metrics="Conditional mix: integral(D_A*alpha_A)/integral(D_A). Calendar rate: integral(D_A*alpha_A)/T.",
    )
    return dict(by_time=frame, binned=binned, events=events, diagnostics=diagnostics)
