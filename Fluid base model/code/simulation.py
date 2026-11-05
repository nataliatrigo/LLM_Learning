"""Deterministic trajectories for a stationary policy in the fluid model.

The control is evaluated at both stages of an explicit Heun step.  If the
stages prescribe different products, their average is a numerical mixture;
it can approximate rapid switching at a policy boundary.  The output records
that mixture as ``product2_fraction`` rather than suggesting that an entire
output interval used one product.  Decrease ``dt`` to assess switching and
trajectory accuracy independently of the HJB mesh size.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import betaincc


POLICIES = ("optimal", "product1", "product2")


def _demand(S: float, F: float, p0: float) -> float:
    return float(np.clip(betaincc(S + 1.0, F + 1.0, p0), 0.0, 1.0))


def _stage(solution: Any, S: float, F: float, policy: str) -> tuple[int, float, float, float]:
    """Return product, demand, S speed, and profit speed at a valid state."""

    params = solution.params
    product = int(solution.action_at(S, F)) if policy == "optimal" else int(policy[-1])
    if product not in (1, 2):
        raise ValueError("action_at must return product 1 or product 2, resolving ties")
    demand = _demand(S, F, params.p0)
    probability = params.p1 if product == 1 else params.p2
    cost = params.c1 if product == 1 else params.c2
    return product, demand, demand * probability, demand * (params.revenue - cost)


def _discount_integral(delta: float, t: float, step: float) -> float:
    """Exact integral of exp(-delta*u) over [t,t+step]."""

    if delta == 0.0:
        return step
    return float(np.exp(-delta * t) * (-np.expm1(-delta * step)) / delta)


def simulate_paths(
    solution: Any,
    initial_states: Iterable[tuple[float, float]],
    horizon: float,
    dt: float,
) -> pd.DataFrame:
    """Simulate optimal and two fixed-product ODE paths from every state.

    Parameters
    ----------
    solution:
        Fluid solution exposing ``params``, ``report_n``, and ``action_at``.
        Its action interpolation must be defined throughout the report region.
    initial_states:
        Pairs ``(S,F)`` with nonnegative finite coordinates in that region.
    horizon, dt:
        Final calendar time and maximum Heun step, both strictly positive.

    Returns
    -------
    pandas.DataFrame
        One row at each integration output, including both the initial and
        terminal state.  ``product`` is the action prescribed at that state;
        it is zero at termination.  ``product2_fraction`` describes the
        *preceding* interval (zero on the initial row), and
        ``product2_time`` is its cumulative time integral.  Demand-weighted
        Heun stages advance S and F and integrate profit.  The discount
        integral within each step is exact for a constant profit speed.

        ``termination_reason`` is empty except at the terminal row, which
        says ``horizon`` or ``report_boundary``.  Every policy stops at
        ``report_n``: no policy or value is silently extrapolated.

    Notes
    -----
    These are deterministic mean trajectories, not stochastic sample paths.
    The finite-horizon discounted profit does not include a terminal value.
    Mixed-stage shares are numerical approximations to occupation time, so
    interpreting switching requires checking smaller time steps.
    """

    if not np.isfinite(horizon) or horizon <= 0.0:
        raise ValueError("horizon must be finite and strictly positive")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and strictly positive")
    report_n = float(solution.report_n)
    if not np.isfinite(report_n) or report_n < 0.0:
        raise ValueError("solution.report_n must be finite and nonnegative")
    delta = float(solution.params.delta)
    if not np.isfinite(delta) or delta < 0.0:
        raise ValueError("the continuous discount rate must be finite and nonnegative")
    states = list(initial_states)
    if not states:
        raise ValueError("initial_states must contain at least one state")
    checked_states: list[tuple[float, float]] = []
    for state in states:
        if len(state) != 2:
            raise ValueError("each initial state must be a pair (S,F)")
        S, F = map(float, state)
        if not np.isfinite(S + F) or S < 0.0 or F < 0.0:
            raise ValueError("initial states must have finite nonnegative coordinates")
        if S + F > report_n:
            raise ValueError("an initial state lies outside solution.report_n")
        checked_states.append((S, F))

    rows: list[dict[str, float | str | int]] = []
    for state_index, (initial_S, initial_F) in enumerate(checked_states):
        path_id = f"{state_index + 1}: S0={initial_S:g},F0={initial_F:g}"
        for policy in POLICIES:
            S, F, t = initial_S, initial_F, 0.0
            cumulative_profit = 0.0
            product2_time = 0.0
            previous_fraction = 0.0
            previous_step = 0.0
            while True:
                n = S + F
                distance = max(0.0, report_n - n)
                boundary_epsilon = 32.0 * np.finfo(float).eps * max(1.0, report_n)
                time_epsilon = 32.0 * np.finfo(float).eps * max(1.0, horizon)
                if distance <= boundary_epsilon:
                    termination = "report_boundary"
                elif horizon - t <= time_epsilon:
                    termination = "horizon"
                else:
                    termination = ""
                product, D0, S0_speed, profit0 = _stage(solution, S, F, policy)
                rows.append(
                    {
                        "path_id": path_id,
                        "state_index": state_index,
                        "policy": policy,
                        "t": t,
                        "S": S,
                        "F": F,
                        "n": n,
                        "x": (S + 1.0) / (n + 2.0),
                        "demand": D0,
                        "product": 0 if termination else product,
                        "product2_fraction": previous_fraction,
                        "product2_time": product2_time,
                        "step": previous_step,
                        "discounted_profit": cumulative_profit,
                        "termination_reason": termination,
                    }
                )
                if termination:
                    break

                # Restrict the predictor as well as the accepted endpoint to
                # the reported triangle.  This keeps both policy lookups valid.
                step = min(dt, horizon - t)
                if D0 > 0.0:
                    step = min(step, distance / D0)

                def endpoint_stage(candidate_step: float) -> tuple[int, float, float, float]:
                    predicted_S = S + candidate_step * S0_speed
                    predicted_F = F + candidate_step * (D0 - S0_speed)
                    # Correct only roundoff at the exact predictor boundary.
                    predicted_n = predicted_S + predicted_F
                    if predicted_n > report_n:
                        predicted_S *= report_n / predicted_n
                        predicted_F *= report_n / predicted_n
                    return _stage(solution, predicted_S, predicted_F, policy)

                endpoint = endpoint_stage(step)
                D1 = endpoint[1]
                if 0.5 * step * (D0 + D1) > distance:
                    # Heun's endpoint can cross even when its predictor does
                    # not.  Locate that event within this numerical step.
                    step = float(
                        brentq(
                            lambda candidate: 0.5
                            * candidate
                            * (D0 + endpoint_stage(candidate)[1])
                            - distance,
                            0.0,
                            step,
                            xtol=np.finfo(float).tiny,
                            rtol=4.0 * np.finfo(float).eps,
                        )
                    )
                    endpoint = endpoint_stage(step)
                product1, D1, S1_speed, profit1 = endpoint
                increment_n = 0.5 * step * (D0 + D1)
                increment_S = 0.5 * step * (S0_speed + S1_speed)
                if increment_n > distance:
                    # At most a few ulps after root finding, with no lookup
                    # outside the domain and no negative state components.
                    increment_S *= distance / increment_n
                    increment_n = distance
                S += increment_S
                F += increment_n - increment_S
                cumulative_profit += _discount_integral(delta, t, step) * 0.5 * (
                    profit0 + profit1
                )
                previous_fraction = 0.5 * (int(product == 2) + int(product1 == 2))
                product2_time += step * previous_fraction
                previous_step = step
                new_time = t + step
                if new_time == t:
                    raise FloatingPointError("time step underflow before reaching the domain boundary")
                t = new_time

    return pd.DataFrame.from_records(rows)


def summarize_paths(paths: pd.DataFrame) -> pd.DataFrame:
    """Summarize each deterministic path without adding an uncomputed tail."""

    if paths.empty:
        raise ValueError("paths must contain simulated trajectories")
    required = {
        "state_index", "path_id", "policy", "t", "S", "F", "n", "x", "demand",
        "discounted_profit", "product2_time", "termination_reason",
    }
    missing = required.difference(paths.columns)
    if missing:
        raise ValueError(f"missing path columns: {', '.join(sorted(missing))}")
    rows = []
    for (state_index, path_id, policy), group in paths.groupby(
        ["state_index", "path_id", "policy"], sort=False
    ):
        first, last = group.iloc[0], group.iloc[-1]
        duration = float(last["t"])
        rows.append(
            {
                "state_index": int(state_index),
                "path_id": path_id,
                "policy": policy,
                "initial_S": float(first["S"]),
                "initial_F": float(first["F"]),
                "final_time": duration,
                "final_S": float(last["S"]),
                "final_F": float(last["F"]),
                "final_n": float(last["n"]),
                "final_x": float(last["x"]),
                "final_demand": float(last["demand"]),
                "discounted_profit": float(last["discounted_profit"]),
                "product2_time_share": float(last["product2_time"]) / duration if duration else 0.0,
                "termination_reason": str(last["termination_reason"]),
            }
        )
    return pd.DataFrame.from_records(rows)
