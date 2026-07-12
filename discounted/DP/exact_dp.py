"""Exact large-horizon discounted DP with rolling value arrays.

The state at the start of period ``t`` is the user's cumulative history
``(S, F)`` with Seller A.  The seller's hidden product choice affects only the
success probability conditional on Seller A being chosen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import betaincc


@dataclass(frozen=True)
class ModelParams:
    p1: float = 0.35
    p2: float = 0.80
    c1: float = 0.05
    c2: float = 0.65
    revenue: float = 1.0
    gamma: float = 0.98
    tol: float = 1e-10
    demand_floor: float = 1e-14


@dataclass(frozen=True)
class StateSpace:
    horizon: int
    S: np.ndarray
    F: np.ndarray
    total: np.ndarray
    success_index: np.ndarray
    failure_index: np.ndarray


def state_count_through_total(total: int) -> int:
    return (total + 1) * (total + 2) // 2


def state_count_for_period(period: int) -> int:
    """Number of states reachable at the start of one-indexed period t."""
    return period * (period + 1) // 2


def state_index(successes: np.ndarray, failures: np.ndarray) -> np.ndarray:
    total = successes + failures
    return total * (total + 1) // 2 + successes


def make_state_space(horizon: int) -> StateSpace:
    """Build every count state needed by a horizon-T backward induction."""
    offsets = np.arange(horizon + 1, dtype=np.int64)
    offsets = offsets * (offsets + 1) // 2
    total = np.repeat(
        np.arange(horizon + 1, dtype=np.int32),
        np.arange(1, horizon + 2, dtype=np.int32),
    )
    flat_index = np.arange(len(total), dtype=np.int64)
    successes = (flat_index - offsets[total]).astype(np.int32)
    failures = total - successes

    # States with total=T form the zero-value terminal layer. Their transition
    # indices intentionally remain -1: the Bellman recursion never transitions
    # out of that layer, so there is no absorbing boundary.
    success_index = np.full(len(total), -1, dtype=np.int64)
    failure_index = np.full(len(total), -1, dtype=np.int64)
    interior = total < horizon
    next_total = total[interior].astype(np.int64) + 1
    next_offset = next_total * (next_total + 1) // 2
    success_index[interior] = next_offset + successes[interior] + 1
    failure_index[interior] = next_offset + successes[interior]

    return StateSpace(
        horizon=horizon,
        S=successes,
        F=failures,
        total=total,
        success_index=success_index,
        failure_index=failure_index,
    )


def beta_ccdf(
    p0: float,
    successes: np.ndarray,
    failures: np.ndarray,
) -> np.ndarray:
    return np.clip(
        betaincc(successes + 1.0, failures + 1.0, p0),
        0.0,
        1.0,
    )


def required_horizon(gamma: float, tail_tolerance: float = 1e-6) -> int:
    """Smallest T such that gamma**T < tail_tolerance."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("required_horizon requires 0 < gamma < 1.")
    if not 0.0 < tail_tolerance < 1.0:
        raise ValueError("tail_tolerance must be in (0, 1).")
    return int(np.floor(np.log(tail_tolerance) / np.log(gamma))) + 1


def product2_gap_threshold(params: ModelParams) -> float:
    """Threshold for the discounted continuation value gap M_t."""
    return (params.c2 - params.c1) / (params.p2 - params.p1)


def solve_discounted_finite_horizon(
    p0: float,
    params: ModelParams,
    horizon: int,
    stored_policy_periods: set[int],
    snapshot_periods: set[int] | None = None,
) -> dict:
    """Solve the exact discounted finite-horizon DP with rolling value arrays."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if not 0.0 < params.gamma <= 1.0:
        raise ValueError("gamma must lie in (0, 1].")

    state_space = make_state_space(horizon)
    n_states = len(state_space.S)
    next_value = np.zeros(n_states, dtype=np.float64)
    current_value = np.zeros(n_states, dtype=np.float64)
    rho_all = beta_ccdf(p0, state_space.S, state_space.F)

    stored_periods = {
        period for period in stored_policy_periods if 1 <= period <= horizon
    } | {1}
    snapshots_requested = {
        period
        for period in (snapshot_periods or set())
        if 1 <= period <= horizon
    }
    stored_periods |= snapshots_requested

    policy_by_period: dict[int, np.ndarray] = {}
    raw_gap_by_period: dict[int, np.ndarray] = {}
    discounted_gap_by_period: dict[int, np.ndarray] = {}
    snapshots: dict[int, dict[str, np.ndarray]] = {}
    initial_conditional_q_gap = np.nan
    initial_raw_gap = np.nan
    initial_discounted_gap = np.nan

    incremental_cost = params.c2 - params.c1
    success_lift = params.p2 - params.p1

    for period in range(horizon, 0, -1):
        count = state_count_for_period(period)
        state_slice = slice(0, count)
        rho = rho_all[state_slice]
        success_idx = state_space.success_index[state_slice]
        failure_idx = state_space.failure_index[state_slice]

        continuation_same = next_value[state_slice]
        continuation_success = next_value[success_idx]
        continuation_failure = next_value[failure_idx]
        raw_gap = continuation_success - continuation_failure
        discounted_gap = params.gamma * raw_gap
        conditional_q_gap = -incremental_cost + success_lift * discounted_gap

        common_no_choice = params.gamma * (1.0 - rho) * continuation_same
        q1 = common_no_choice + rho * (
            params.revenue
            - params.c1
            + params.gamma
            * (
                params.p1 * continuation_success
                + (1.0 - params.p1) * continuation_failure
            )
        )
        q2 = common_no_choice + rho * (
            params.revenue
            - params.c2
            + params.gamma
            * (
                params.p2 * continuation_success
                + (1.0 - params.p2) * continuation_failure
            )
        )
        use_product2 = (
            (conditional_q_gap > params.tol) & (rho > params.demand_floor)
        )
        action = np.where(use_product2, 2, 1).astype(np.int8)
        current_value[state_slice] = np.maximum(q1, q2)

        if period == 1:
            initial_conditional_q_gap = float(conditional_q_gap[0])
            initial_raw_gap = float(raw_gap[0])
            initial_discounted_gap = float(discounted_gap[0])

        if period in stored_periods:
            policy_by_period[period] = action.copy()
            raw_gap_by_period[period] = raw_gap.astype(np.float32)
            discounted_gap_by_period[period] = discounted_gap.astype(np.float32)

        if period in snapshots_requested:
            snapshots[period] = {
                "S": state_space.S[state_slice].copy(),
                "F": state_space.F[state_slice].copy(),
                "rho": rho.copy(),
                "value": current_value[state_slice].copy(),
                "action": action.copy(),
                "raw_continuation_gap": raw_gap.copy(),
                "discounted_continuation_gap": discounted_gap.copy(),
                "conditional_q_gap_product2_minus_product1": (
                    conditional_q_gap.copy()
                ),
            }

        next_value, current_value = current_value, next_value

    return {
        "p0": float(p0),
        "gamma": float(params.gamma),
        "T": int(horizon),
        "gamma_to_T": float(params.gamma**horizon),
        "params": params,
        "state_space": state_space,
        "policy_by_period": policy_by_period,
        "raw_gap_by_period": raw_gap_by_period,
        "discounted_gap_by_period": discounted_gap_by_period,
        "snapshots": snapshots,
        "initial_value": float(next_value[0]),
        "initial_action": int(policy_by_period[1][0]),
        "initial_conditional_q_gap_product2_minus_product1": (
            initial_conditional_q_gap
        ),
        "initial_raw_continuation_gap": initial_raw_gap,
        "initial_discounted_continuation_gap": initial_discounted_gap,
        "product2_discounted_gap_threshold": product2_gap_threshold(params),
    }


def compare_policy_slices(
    first: dict,
    second: dict,
    period: int,
) -> dict:
    first_policy = first["policy_by_period"][period]
    second_policy = second["policy_by_period"][period]
    if len(first_policy) != len(second_policy):
        raise ValueError("Policy slices cover different state sets.")
    disagreements = first_policy != second_policy
    return {
        "period": int(period),
        "state_count": int(len(first_policy)),
        "disagreement_count": int(disagreements.sum()),
        "disagreement_fraction": float(disagreements.mean()),
        "first_product2_share": float(np.mean(first_policy == 2)),
        "second_product2_share": float(np.mean(second_policy == 2)),
    }


def simulate_policy(
    solution: dict,
    periods: int,
    n_rep: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Vectorized simulation of the stored early-period policy."""
    if periods <= 0 or periods > solution["T"]:
        raise ValueError("periods must lie in 1..T.")
    missing = set(range(1, periods + 1)) - set(solution["policy_by_period"])
    if missing:
        raise ValueError("The solution did not store every simulated period.")

    params: ModelParams = solution["params"]
    p0 = float(solution["p0"])
    rng = np.random.default_rng(seed)
    successes = np.zeros(n_rep, dtype=np.int32)
    failures = np.zeros(n_rep, dtype=np.int32)
    discounted_profit = np.zeros(n_rep, dtype=float)
    cumulative_profit = np.zeros(n_rep, dtype=float)
    chosen_count = np.zeros(n_rep, dtype=np.int32)
    product2_count = np.zeros(n_rep, dtype=np.int32)
    success_count = np.zeros(n_rep, dtype=np.int32)
    time_rows = []

    for period in range(1, periods + 1):
        rho = beta_ccdf(p0, successes, failures)
        alpha = successes + 1.0
        beta = failures + 1.0
        posterior_precision = alpha + beta
        posterior_mean = alpha / posterior_precision
        posterior_std = np.sqrt(
            alpha
            * beta
            / (posterior_precision**2 * (posterior_precision + 1.0))
        )
        indices = state_index(successes, failures)
        action = solution["policy_by_period"][period][indices]
        discounted_gap = solution["discounted_gap_by_period"][period][indices]

        chosen = rng.random(n_rep) < rho
        success_probability = np.where(action == 2, params.p2, params.p1)
        success = rng.random(n_rep) < success_probability
        product2_chosen = chosen & (action == 2)
        realized_success = chosen & success
        realized_failure = chosen & ~success
        profit = chosen * np.where(
            action == 2,
            params.revenue - params.c2,
            params.revenue - params.c1,
        )

        discounted_profit += params.gamma ** (period - 1) * profit
        cumulative_profit += profit
        chosen_count += chosen
        product2_count += product2_chosen
        success_count += realized_success

        chosen_total = int(chosen.sum())
        time_rows.append(
            {
                "p0": p0,
                "gamma": params.gamma,
                "T": solution["T"],
                "gamma_to_T": solution["gamma_to_T"],
                "t": period,
                "A_market_share": float(chosen.mean()),
                "mean_demand_probability": float(rho.mean()),
                "mean_posterior_mean": float(posterior_mean.mean()),
                "mean_posterior_std": float(posterior_std.mean()),
                "policy_product2_share": float(np.mean(action == 2)),
                "product2_rate_when_A_chosen": (
                    float(product2_chosen.sum() / chosen_total)
                    if chosen_total
                    else np.nan
                ),
                "mean_discounted_continuation_gap": float(
                    discounted_gap.mean()
                ),
                "mean_observations": float((successes + failures).mean()),
                "avg_profit_per_period": float(profit.mean()),
            }
        )

        successes += realized_success
        failures += realized_failure

    replications = pd.DataFrame(
        {
            "p0": p0,
            "gamma": params.gamma,
            "T": solution["T"],
            "rep": np.arange(n_rep),
            "A_market_share": chosen_count / periods,
            "product2_rate_when_A_chosen": np.divide(
                product2_count,
                chosen_count,
                out=np.full(n_rep, np.nan, dtype=float),
                where=chosen_count > 0,
            ),
            "A_success_rate_when_chosen": np.divide(
                success_count,
                chosen_count,
                out=np.full(n_rep, np.nan, dtype=float),
                where=chosen_count > 0,
            ),
            "avg_profit_per_period": cumulative_profit / periods,
            "discounted_profit": discounted_profit,
            "final_successes": successes,
            "final_failures": failures,
            "final_posterior_mean": (
                (successes + 1.0) / (successes + failures + 2.0)
            ),
        }
    )
    return replications, pd.DataFrame.from_records(time_rows)


def simulate_count_paths(
    solution: dict,
    periods: int,
    n_paths: int,
    seed: int,
) -> pd.DataFrame:
    """Simulate a modest number of full count paths for path diagnostics."""
    params: ModelParams = solution["params"]
    p0 = float(solution["p0"])
    rng = np.random.default_rng(seed)
    successes = np.zeros(n_paths, dtype=np.int32)
    failures = np.zeros(n_paths, dtype=np.int32)
    rows = []

    for period in range(1, periods + 1):
        rho = beta_ccdf(p0, successes, failures)
        indices = state_index(successes, failures)
        action = solution["policy_by_period"][period][indices]
        discounted_gap = solution["discounted_gap_by_period"][period][indices]
        posterior_mean = (successes + 1.0) / (successes + failures + 2.0)
        chosen = rng.random(n_paths) < rho
        success_probability = np.where(action == 2, params.p2, params.p1)
        success = rng.random(n_paths) < success_probability

        rows.append(
            pd.DataFrame(
                {
                    "p0": p0,
                    "gamma": params.gamma,
                    "T": solution["T"],
                    "rep": np.arange(n_paths),
                    "t": period,
                    "S": successes.copy(),
                    "F": failures.copy(),
                    "posterior_mean": posterior_mean,
                    "demand_probability": rho,
                    "discounted_continuation_gap": discounted_gap,
                    "policy_product": action,
                    "chosen_A": chosen.astype(np.int8),
                    "product_used": np.where(chosen, action, np.nan),
                    "success": np.where(chosen, success.astype(float), np.nan),
                }
            )
        )
        successes += chosen & success
        failures += chosen & ~success

    return pd.concat(rows, ignore_index=True)


def late_window_summary(
    time_series: pd.DataFrame,
    start_period: int = 201,
) -> dict:
    late = time_series[time_series["t"] >= start_period]
    if late.empty:
        raise ValueError("Late-window start lies beyond the simulation.")
    return {
        "late_start_period": int(start_period),
        "late_end_period": int(late["t"].max()),
        "mean_A_market_share": float(late["A_market_share"].mean()),
        "mean_demand_probability": float(
            late["mean_demand_probability"].mean()
        ),
        "mean_policy_product2_share": float(
            late["policy_product2_share"].mean()
        ),
        "mean_product2_rate_when_A_chosen": float(
            late["product2_rate_when_A_chosen"].mean()
        ),
        "mean_posterior_mean": float(late["mean_posterior_mean"].mean()),
        "mean_observations": float(late["mean_observations"].mean()),
    }
