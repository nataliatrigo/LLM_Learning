"""
Finite-horizon T=1000 study for the Bernoulli/Beta reputation model.

This script is intentionally separate from
other_experiments/average_cost/main.py. It solves the exact finite-horizon
dynamic program with rolling value arrays, so T=1000 is feasible without
writing a full T-by-state policy table. Outputs focus on the early calendar
periods, where the user starts from the prior.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaln, betaincc

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "llm_learning_matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


FINITE_HORIZON_METHOD = "finite_horizon"
FINITE_HORIZON_UCB_METHOD = "finite_horizon_ucb"
FINITE_HORIZON_POSTERIOR_MEAN_METHOD = "finite_horizon_posterior_mean"
THOMPSON_USER_POLICY = "thompson"
UCB_USER_POLICY = "ucb"
POSTERIOR_MEAN_USER_POLICY = "posterior_mean"
DEFAULT_P0_GRID = "0.1,0.3,0.5,0.7,0.9"
DEFAULT_SNAPSHOT_PERIODS = "1,2,5,10,25,50,100,200"
POSTERIOR_BIN_COUNT = 20


@dataclass(frozen=True)
class ModelParams:
    p1: float = 0.35
    p2: float = 0.80
    c1: float = 0.05
    c2: float = 0.65
    revenue: float = 1.0
    max_observations: int = 1000
    tol: float = 1e-8


@dataclass(frozen=True)
class StateSpace:
    max_observations: int
    S: np.ndarray
    F: np.ndarray
    total: np.ndarray
    state_index: np.ndarray
    success_index: np.ndarray
    success_probability: np.ndarray
    success_stay_probability: np.ndarray
    failure_index: np.ndarray
    failure_probability: np.ndarray
    failure_stay_probability: np.ndarray


def make_state_space(
    max_observations: int,
    method: str = "rolling_window",
) -> StateSpace:
    if method != "rolling_window":
        raise ValueError("finite_horizon/main.py only supports rolling_window states.")

    states: list[tuple[int, int]] = []
    state_index = -np.ones((max_observations + 1, max_observations + 1), dtype=int)
    for total in range(max_observations + 1):
        for successes in range(total + 1):
            failures = total - successes
            state_index[successes, failures] = len(states)
            states.append((successes, failures))

    successes_array = np.array([state[0] for state in states], dtype=int)
    failures_array = np.array([state[1] for state in states], dtype=int)
    total_array = successes_array + failures_array
    success_index = np.empty(len(states), dtype=int)
    success_probability = np.empty(len(states), dtype=float)
    success_stay_probability = np.empty(len(states), dtype=float)
    failure_index = np.empty(len(states), dtype=int)
    failure_probability = np.empty(len(states), dtype=float)
    failure_stay_probability = np.empty(len(states), dtype=float)

    for idx, (successes, failures) in enumerate(states):
        if successes + failures < max_observations:
            success_index[idx] = state_index[successes + 1, failures]
            success_probability[idx] = 1.0
            success_stay_probability[idx] = 0.0
            failure_index[idx] = state_index[successes, failures + 1]
            failure_probability[idx] = 1.0
            failure_stay_probability[idx] = 0.0
            continue

        success_successes = min(successes + 1, max_observations)
        success_failures = max_observations - success_successes
        failure_successes = max(successes - 1, 0)
        failure_failures = max_observations - failure_successes
        success_index[idx] = state_index[success_successes, success_failures]
        success_probability[idx] = failures / max_observations
        success_stay_probability[idx] = successes / max_observations
        failure_index[idx] = state_index[failure_successes, failure_failures]
        failure_probability[idx] = successes / max_observations
        failure_stay_probability[idx] = failures / max_observations

    return StateSpace(
        max_observations=max_observations,
        S=successes_array,
        F=failures_array,
        total=total_array,
        state_index=state_index,
        success_index=success_index,
        success_probability=success_probability,
        success_stay_probability=success_stay_probability,
        failure_index=failure_index,
        failure_probability=failure_probability,
        failure_stay_probability=failure_stay_probability,
    )


def beta_ccdf(p0: float, successes: np.ndarray, failures: np.ndarray) -> np.ndarray:
    """Pr(Beta(1 + S, 1 + F) >= p0)."""
    return np.clip(betaincc(successes + 1.0, failures + 1.0, p0), 0.0, 1.0)


def beta_ccdf_scalar(p0: float, successes: float, failures: float) -> float:
    return float(beta_ccdf(p0, np.array([successes]), np.array([failures]))[0])


def finite_horizon_method_name(user_policy: str) -> str:
    if user_policy == THOMPSON_USER_POLICY:
        return FINITE_HORIZON_METHOD
    if user_policy == UCB_USER_POLICY:
        return FINITE_HORIZON_UCB_METHOD
    if user_policy == POSTERIOR_MEAN_USER_POLICY:
        return FINITE_HORIZON_POSTERIOR_MEAN_METHOD
    raise ValueError(f"Unknown user policy: {user_policy}")


def user_policy_label(user_policy: str) -> str:
    if user_policy == THOMPSON_USER_POLICY:
        return "Thompson sampling"
    if user_policy == UCB_USER_POLICY:
        return "UCB"
    if user_policy == POSTERIOR_MEAN_USER_POLICY:
        return "posterior-mean myopic Bayesian"
    raise ValueError(f"Unknown user policy: {user_policy}")


def ucb_index(
    period: int,
    successes: np.ndarray,
    failures: np.ndarray,
    alpha: float,
) -> np.ndarray:
    posterior_mean_value = posterior_mean(successes, failures)
    effective_observations = successes + failures + 2.0
    exploration_bonus = np.sqrt(
        alpha * np.log(float(max(period + 1, 2))) / effective_observations
    )
    return posterior_mean_value + exploration_bonus


def ucb_index_scalar(
    period: int,
    successes: float,
    failures: float,
    alpha: float,
) -> float:
    return float(
        ucb_index(
            period,
            np.array([successes], dtype=float),
            np.array([failures], dtype=float),
            alpha,
        )[0]
    )


def user_demand_probability(
    p0: float,
    successes: np.ndarray,
    failures: np.ndarray,
    period: int,
    user_policy: str,
    ucb_alpha: float,
) -> np.ndarray:
    if user_policy == THOMPSON_USER_POLICY:
        return beta_ccdf(p0, successes, failures)
    if user_policy == UCB_USER_POLICY:
        return (ucb_index(period, successes, failures, ucb_alpha) >= p0).astype(float)
    if user_policy == POSTERIOR_MEAN_USER_POLICY:
        return (posterior_mean(successes, failures) >= p0).astype(float)
    raise ValueError(f"Unknown user policy: {user_policy}")


def user_demand_probability_scalar(
    p0: float,
    successes: float,
    failures: float,
    period: int,
    user_policy: str,
    ucb_alpha: float,
) -> float:
    if user_policy == THOMPSON_USER_POLICY:
        return beta_ccdf_scalar(p0, successes, failures)
    if user_policy == UCB_USER_POLICY:
        return float(ucb_index_scalar(period, successes, failures, ucb_alpha) >= p0)
    if user_policy == POSTERIOR_MEAN_USER_POLICY:
        return float(posterior_mean_scalar(successes, failures) >= p0)
    raise ValueError(f"Unknown user policy: {user_policy}")


def posterior_mean(successes: np.ndarray, failures: np.ndarray) -> np.ndarray:
    return (successes + 1.0) / (successes + failures + 2.0)


def posterior_std(successes: np.ndarray, failures: np.ndarray) -> np.ndarray:
    alpha = successes + 1.0
    beta = failures + 1.0
    posterior_precision = alpha + beta
    variance = alpha * beta / (
        posterior_precision**2 * (posterior_precision + 1.0)
    )
    return np.sqrt(variance)


def posterior_mean_scalar(successes: int, failures: int) -> float:
    return float((successes + 1.0) / (successes + failures + 2.0))


def posterior_density_from_counts(
    theta_grid: np.ndarray,
    successes: float,
    failures: float,
) -> np.ndarray:
    alpha = successes + 1.0
    beta = failures + 1.0
    log_density = (
        (alpha - 1.0) * np.log(theta_grid)
        + (beta - 1.0) * np.log1p(-theta_grid)
        - betaln(alpha, beta)
    )
    return np.exp(log_density)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#111827",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "font.size": 10,
            "savefig.facecolor": "white",
        }
    )


def prettify_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(True, axis=grid_axis)


def save_figure(fig: plt.Figure, path: Path) -> None:
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_float_grid(value: str) -> np.ndarray:
    if not value.strip():
        return np.array([], dtype=float)
    values = np.array([float(item.strip()) for item in value.split(",") if item.strip()])
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("All p0 values must be strictly between 0 and 1.")
    return np.unique(np.round(values, 10))


def parse_int_grid(value: str) -> list[int]:
    if not value.strip():
        return []
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if any(item <= 0 for item in values):
        raise ValueError("Integer grids must contain only positive values.")
    return values


def state_count_through_total(total: int) -> int:
    """Number of triangular states with S + F <= total."""
    return (total + 1) * (total + 2) // 2


def state_count_for_period(period: int) -> int:
    """States reachable at the start of a one-indexed finite-horizon period."""
    return state_count_through_total(period - 1)


def product2_continuation_threshold(params: ModelParams) -> float:
    return (params.c2 - params.c1) / (params.p2 - params.p1)


def asymptotic_product2_mix_bound(p0: float, params: ModelParams) -> float:
    mix = (p0 - params.p1) / (params.p2 - params.p1)
    return float(np.clip(mix, 0.0, 1.0))


def add_posterior_bins(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    bins = np.linspace(0.0, 1.0, POSTERIOR_BIN_COUNT + 1)
    working = frame.copy()
    working["posterior_bin"] = pd.cut(
        working["posterior_mean"],
        bins=bins,
        labels=False,
        include_lowest=True,
    )
    grouped = (
        working.dropna(subset=["posterior_bin"])
        .groupby(["p0", "T", "t", "posterior_bin"], observed=True)
        .agg(
            product_2_share=("use_product_2", "mean"),
            state_count=("use_product_2", "size"),
        )
        .reset_index()
    )
    grouped["posterior_mean"] = (
        (grouped["posterior_bin"].astype(float) + 0.5) / POSTERIOR_BIN_COUNT
    )
    return grouped


def solve_finite_horizon_early(
    p0: float,
    params: ModelParams,
    horizon: int,
    snapshot_periods: list[int],
    stored_policy_periods: set[int],
    user_policy: str,
    ucb_alpha: float,
) -> dict:
    """Solve exact finite-horizon DP and keep only early policy slices."""
    state_space = make_state_space(horizon, method="rolling_window")
    n_states = len(state_space.S)
    next_value = np.zeros(n_states, dtype=float)
    current_value = np.zeros(n_states, dtype=float)
    rho_all = (
        beta_ccdf(p0, state_space.S, state_space.F)
        if user_policy == THOMPSON_USER_POLICY
        else None
    )
    method_name = finite_horizon_method_name(user_policy)

    snapshot_periods = sorted(period for period in snapshot_periods if period <= horizon)
    snapshot_set = set(snapshot_periods)
    stored_policy_periods = {
        period for period in stored_policy_periods if 1 <= period <= horizon
    } | snapshot_set | {1}

    policy_by_period: dict[int, np.ndarray] = {}
    continuation_gap_by_period: dict[int, np.ndarray] = {}
    snapshot_frames: list[pd.DataFrame] = []
    usage_rows = []
    initial_q_gap = np.nan
    initial_continuation_gap = np.nan

    for period in range(horizon, 0, -1):
        count = state_count_for_period(period)
        state_slice = slice(0, count)
        if rho_all is None:
            rho = user_demand_probability(
                p0,
                state_space.S[state_slice],
                state_space.F[state_slice],
                period,
                user_policy,
                ucb_alpha,
            )
        else:
            rho = rho_all[state_slice]
        success_idx = state_space.success_index[state_slice]
        failure_idx = state_space.failure_index[state_slice]

        continuation_same = next_value[state_slice]
        continuation_success = next_value[success_idx]
        continuation_failure = next_value[failure_idx]
        continuation_gap = continuation_success - continuation_failure

        q1 = (
            rho * (params.revenue - params.c1)
            + (1.0 - rho) * continuation_same
            + rho
            * (
                params.p1 * continuation_success
                + (1.0 - params.p1) * continuation_failure
            )
        )
        q2 = (
            rho * (params.revenue - params.c2)
            + (1.0 - rho) * continuation_same
            + rho
            * (
                params.p2 * continuation_success
                + (1.0 - params.p2) * continuation_failure
            )
        )
        use_product2 = q2 > q1 + params.tol
        action = np.where(use_product2, 2, 1).astype(np.int8)
        current_value[state_slice] = np.maximum(q1, q2)

        usage_rows.append(
            {
                "method": method_name,
                "user_policy": user_policy,
                "ucb_alpha": ucb_alpha if user_policy == UCB_USER_POLICY else np.nan,
                "p0": p0,
                "T": horizon,
                "t": period,
                "time_remaining": horizon - period + 1,
                "state_count": count,
                "product_2_share": float(np.mean(use_product2)),
            }
        )

        if period == 1:
            initial_q_gap = float(q2[0] - q1[0])
            initial_continuation_gap = float(continuation_gap[0])

        if period in stored_policy_periods:
            policy_by_period[period] = action.copy()
            continuation_gap_by_period[period] = continuation_gap.astype(np.float32)

        if period in snapshot_set:
            idx = np.arange(count)
            snapshot_frames.append(
                pd.DataFrame(
                    {
                        "method": method_name,
                        "user_policy": user_policy,
                        "ucb_alpha": (
                            ucb_alpha if user_policy == UCB_USER_POLICY else np.nan
                        ),
                        "p0": p0,
                        "T": horizon,
                        "t": period,
                        "time_remaining": horizon - period + 1,
                        "S": state_space.S[idx],
                        "F": state_space.F[idx],
                        "total_count": state_space.total[idx],
                        "posterior_mean": posterior_mean(
                            state_space.S[idx],
                            state_space.F[idx],
                        ),
                        "posterior_std": posterior_std(
                            state_space.S[idx],
                            state_space.F[idx],
                        ),
                        "rho": rho,
                        "action": action.astype(int),
                        "use_product_2": use_product2.astype(int),
                        "V_t": current_value[idx],
                        "continuation_gap_success_minus_failure": continuation_gap,
                        "q_gap_product2_minus_product1": q2 - q1,
                    }
                )
            )

        next_value, current_value = current_value, next_value

    initial_idx = int(state_space.state_index[0, 0])
    initial_value = float(next_value[initial_idx])
    initial_action = int(policy_by_period[1][initial_idx])
    snapshots = (
        pd.concat(snapshot_frames, ignore_index=True)
        if snapshot_frames
        else pd.DataFrame()
    )
    return {
        "method": method_name,
        "user_policy": user_policy,
        "ucb_alpha": ucb_alpha if user_policy == UCB_USER_POLICY else np.nan,
        "p0": p0,
        "T": horizon,
        "state_space": state_space,
        "policy_by_period": policy_by_period,
        "continuation_gap_by_period": continuation_gap_by_period,
        "usage_by_time": pd.DataFrame.from_records(usage_rows),
        "policy_snapshots": snapshots,
        "initial_value": initial_value,
        "avg_value_T": initial_value / horizon,
        "initial_action": initial_action,
        "initial_q_gap_product2_minus_product1": initial_q_gap,
        "initial_continuation_gap_success_minus_failure": initial_continuation_gap,
    }


def simulate_early_policy(
    solution: dict,
    params: ModelParams,
    n_rep: int,
    periods: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if periods <= 0 or n_rep <= 0:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    method_name = str(solution["method"])
    user_policy = str(solution.get("user_policy", THOMPSON_USER_POLICY))
    ucb_alpha = float(solution.get("ucb_alpha", np.nan))
    p0 = float(solution["p0"])
    horizon = int(solution["T"])
    state_space: StateSpace = solution["state_space"]
    policy_by_period: dict[int, np.ndarray] = solution["policy_by_period"]
    continuation_gap_by_period: dict[int, np.ndarray] = solution[
        "continuation_gap_by_period"
    ]
    product2_incremental_cost = params.c2 - params.c1
    product2_success_lift = params.p2 - params.p1
    records = []

    for rep in range(n_rep):
        successes = 0
        failures = 0
        cumulative_profit = 0.0
        for period in range(1, periods + 1):
            current_successes = successes
            current_failures = failures
            demand_prob = user_demand_probability_scalar(
                p0,
                successes,
                failures,
                period,
                user_policy,
                ucb_alpha,
            )
            user_choice_index = (
                ucb_index_scalar(period, successes, failures, ucb_alpha)
                if user_policy == UCB_USER_POLICY
                else np.nan
            )
            if user_policy == POSTERIOR_MEAN_USER_POLICY:
                user_choice_index = posterior_mean_scalar(successes, failures)
            posterior_mean_t = posterior_mean_scalar(successes, failures)
            state_idx = int(state_space.state_index[successes, failures])
            action = int(policy_by_period[period][state_idx])
            marginal_reputation_value = float(
                continuation_gap_by_period[period][state_idx]
            )
            product2_reputational_benefit = (
                product2_success_lift * marginal_reputation_value
            )
            product2_net_benefit = (
                product2_reputational_benefit - product2_incremental_cost
            )
            q_gap = demand_prob * product2_net_benefit
            user_chose_A = rng.random() < demand_prob
            success_value = np.nan
            profit = 0.0

            if user_chose_A:
                success_probability = params.p2 if action == 2 else params.p1
                success_value = rng.random() < success_probability
                if success_value:
                    successes += 1
                else:
                    failures += 1
                profit = params.revenue - (params.c2 if action == 2 else params.c1)
                cumulative_profit += profit

            records.append(
                {
                    "method": method_name,
                    "user_policy": user_policy,
                    "ucb_alpha": ucb_alpha if user_policy == UCB_USER_POLICY else np.nan,
                    "p0": p0,
                    "T": horizon,
                    "N_or_T": horizon,
                    "rep": rep,
                    "t": period,
                    "time_remaining": horizon - period + 1,
                    "true_S": current_successes,
                    "true_F": current_failures,
                    "total_count": current_successes + current_failures,
                    "posterior_mean": posterior_mean_t,
                    "rho": demand_prob,
                    "user_choice_index": user_choice_index,
                    "marginal_reputation_value": marginal_reputation_value,
                    "continuation_gap_success_minus_failure": (
                        marginal_reputation_value
                    ),
                    "product2_reputational_benefit": product2_reputational_benefit,
                    "product2_incremental_cost": product2_incremental_cost,
                    "product2_net_benefit": product2_net_benefit,
                    "q_gap_product2_minus_product1": q_gap,
                    "policy_recommends_product_2": int(action == 2),
                    "user_chose_A": int(user_chose_A),
                    "action_if_A": action,
                    "success": float(success_value) if user_chose_A else np.nan,
                    "profit": profit,
                    "cumulative_profit": cumulative_profit,
                    "average_profit_to_date": cumulative_profit / period,
                }
            )

    paths = pd.DataFrame.from_records(records)
    paths["product_2_when_A_chosen"] = np.where(
        paths["user_chose_A"] == 1,
        (paths["action_if_A"] == 2).astype(float),
        np.nan,
    )
    time_summary = (
        paths.groupby(["method", "p0", "T", "N_or_T", "t"], observed=True)
        .agg(
            user_policy=("user_policy", "first"),
            ucb_alpha=("ucb_alpha", "first"),
            average_profit_to_date=("average_profit_to_date", "mean"),
            posterior_mean=("posterior_mean", "mean"),
            total_count=("total_count", "mean"),
            rho=("rho", "mean"),
            user_choice_index=("user_choice_index", "mean"),
            marginal_reputation_value=("marginal_reputation_value", "mean"),
            product2_reputational_benefit=("product2_reputational_benefit", "mean"),
            product2_net_benefit=("product2_net_benefit", "mean"),
            policy_recommends_product_2=("policy_recommends_product_2", "mean"),
            A_market_share=("user_chose_A", "mean"),
            product_2_frequency=("product_2_when_A_chosen", "mean"),
            simulated_avg_profit=("profit", "mean"),
        )
        .reset_index()
    )
    time_summary["time_remaining"] = horizon - time_summary["t"] + 1
    return paths, time_summary


def summarize_solution(solution: dict, snapshot_period: int, params: ModelParams) -> dict:
    snapshots = solution["policy_snapshots"]
    at_snapshot = snapshots[snapshots["t"] == snapshot_period]
    p0 = float(solution["p0"])
    return {
        "method": solution.get("method", FINITE_HORIZON_METHOD),
        "user_policy": solution.get("user_policy", THOMPSON_USER_POLICY),
        "ucb_alpha": solution.get("ucb_alpha", np.nan),
        "p0": p0,
        "T": solution["T"],
        "initial_total_value": solution["initial_value"],
        "avg_value_T": solution["avg_value_T"],
        "initial_action": solution["initial_action"],
        "initial_q_gap_product2_minus_product1": solution[
            "initial_q_gap_product2_minus_product1"
        ],
        "initial_continuation_gap_success_minus_failure": solution[
            "initial_continuation_gap_success_minus_failure"
        ],
        "product2_continuation_gap_threshold": product2_continuation_threshold(params),
        "asymptotic_product2_mix_bound": asymptotic_product2_mix_bound(p0, params),
        "snapshot_period": snapshot_period,
        "snapshot_product_2_share": (
            float(at_snapshot["use_product_2"].mean())
            if not at_snapshot.empty
            else np.nan
        ),
        "snapshot_state_count": int(len(at_snapshot)),
    }


def plot_summary(summary: pd.DataFrame, outputs_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharex=True)
    ax_value, ax_action, ax_gap, ax_share = axes.ravel()

    ax_value.plot(summary["p0"], summary["avg_value_T"], marker="o", linewidth=2.0)
    ax_value.set_title("Initial value per horizon period", loc="left")
    ax_value.set_ylabel("V_1(0,0) / T")
    prettify_axes(ax_value)

    ax_action.scatter(
        summary["p0"],
        summary["initial_action"],
        s=70,
        color="#0f766e",
        zorder=3,
    )
    ax_action.set_title("Initial action at grid p0", loc="left")
    ax_action.set_yticks([1, 2])
    ax_action.set_yticklabels(["Product 1", "Product 2"])
    ax_action.set_ylim(0.75, 2.25)
    prettify_axes(ax_action)

    threshold = float(summary["product2_continuation_gap_threshold"].iloc[0])
    ax_gap.axhline(
        threshold,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label=f"threshold={threshold:.2f}",
    )
    ax_gap.plot(
        summary["p0"],
        summary["initial_continuation_gap_success_minus_failure"],
        marker="o",
        linewidth=2.0,
        color="#2563eb",
        label="initial gap",
    )
    ax_gap.set_title("Initial continuation gap", loc="left")
    ax_gap.set_ylabel("V(S+1,F) - V(S,F+1)")
    prettify_axes(ax_gap)
    ax_gap.legend()

    ax_share.plot(
        summary["p0"],
        summary["snapshot_product_2_share"],
        marker="o",
        linewidth=2.0,
        color="#dc2626",
    )
    period = int(summary["snapshot_period"].iloc[0])
    ax_share.set_title(f"Product 2 share at period {period}", loc="left")
    ax_share.set_ylabel("Share of reachable states")
    ax_share.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_share.set_ylim(-0.03, 1.03)
    prettify_axes(ax_share)

    for ax in axes[-1]:
        ax.set_xlabel("p0")
    fig.suptitle(
        f"Finite-horizon T={int(summary['T'].iloc[0])}: early-period summary",
        x=0.01,
        ha="left",
    )
    save_figure(fig, outputs_dir / "finite_horizon_initial_summary.png")


def plot_product2_usage(
    usage_by_time: pd.DataFrame,
    usage_by_mean: pd.DataFrame,
    outputs_dir: Path,
    early_periods: int,
    posterior_snapshot_period: int,
) -> None:
    if usage_by_time.empty:
        return
    horizon = int(usage_by_time["T"].iloc[0])
    p0_values = np.array(sorted(usage_by_time["p0"].unique()))
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(p0_values)))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    ax_time, ax_mean = axes

    for p0, color in zip(p0_values, colors, strict=True):
        time_data = usage_by_time[
            np.isclose(usage_by_time["p0"], p0) & (usage_by_time["t"] <= early_periods)
        ].sort_values("t")
        label = f"p0={p0:.2f}"
        ax_time.plot(
            time_data["t"],
            time_data["product_2_share"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=2.8,
            label=label,
        )

        mean_data = usage_by_mean[
            np.isclose(usage_by_mean["p0"], p0)
            & (usage_by_mean["t"] == posterior_snapshot_period)
        ].sort_values("posterior_mean")
        if not mean_data.empty:
            ax_mean.plot(
                mean_data["posterior_mean"],
                mean_data["product_2_share"],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=2.8,
                label=label,
            )

    ax_time.set_title("Product 2 use over early periods", loc="left")
    ax_time.set_xlabel("Calendar period t")
    ax_time.set_ylabel("Share of reachable states")
    ax_time.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_time.set_ylim(-0.03, 1.03)
    prettify_axes(ax_time)
    ax_time.legend(ncols=min(3, len(p0_values)))

    ax_mean.set_title(
        f"Product 2 use by posterior mean at t={posterior_snapshot_period}",
        loc="left",
    )
    ax_mean.set_xlabel("Posterior mean E[theta | S,F]")
    ax_mean.set_ylabel("Share of reachable states")
    ax_mean.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_mean.set_xlim(-0.02, 1.02)
    ax_mean.set_ylim(-0.03, 1.03)
    prettify_axes(ax_mean)

    fig.suptitle(
        f"Finite-horizon product 2 summaries, T={horizon}",
        x=0.01,
        ha="left",
    )
    save_figure(fig, outputs_dir / "finite_horizon_product2_usage_all_p0.png")


def plot_product2_policy_share(
    usage_by_time: pd.DataFrame,
    outputs_dir: Path,
    plot_periods: int,
) -> None:
    if usage_by_time.empty:
        return

    horizon = int(usage_by_time["T"].iloc[0])
    last_period = min(plot_periods, horizon)
    p0_values = np.array(sorted(usage_by_time["p0"].unique()))
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(p0_values)))
    fig, ax = plt.subplots(figsize=(12.4, 5.8))

    for p0, color in zip(p0_values, colors, strict=True):
        data = usage_by_time[
            np.isclose(usage_by_time["p0"], p0)
            & (usage_by_time["t"] <= last_period)
        ].sort_values("t")
        ax.plot(
            data["t"],
            data["product_2_share"],
            color=color,
            linewidth=2.0,
            label=f"p0={p0:.2f}",
        )

    ax.set_title(
        f"Finite-horizon policy: Product 2 share through period {last_period}",
        loc="left",
    )
    ax.set_xlabel("Calendar period t")
    ax.set_ylabel("Share of reachable states choosing Product 2")
    ax.set_xlim(1, last_period)
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    prettify_axes(ax)
    ax.legend(ncols=min(3, len(p0_values)))
    save_figure(
        fig,
        outputs_dir
        / f"finite_horizon_product2_policy_share_first{last_period}.png",
    )


def plot_policy_heatmaps(
    snapshots: pd.DataFrame,
    outputs_dir: Path,
    snapshot_periods: list[int],
) -> None:
    if snapshots.empty:
        return
    p0_values = np.array(sorted(snapshots["p0"].unique()))
    horizon = int(snapshots["T"].iloc[0])
    available_periods = sorted(
        [period for period in snapshot_periods if period in set(snapshots["t"])]
    )
    if not available_periods:
        return
    informative_periods = [period for period in available_periods if period >= 5]
    if not informative_periods:
        informative_periods = available_periods
    if len(informative_periods) > 5:
        selected_indices = sorted(
            set(
                np.linspace(0, len(informative_periods) - 1, 5)
                .round()
                .astype(int)
            )
        )
        periods = [informative_periods[idx] for idx in selected_indices]
    else:
        periods = informative_periods

    cmap = ListedColormap(["#f2c94c", "#0f766e"])
    cmap.set_bad("#f8fafc")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(
        len(periods),
        len(p0_values),
        figsize=(2.65 * len(p0_values), 2.45 * len(periods)),
        sharex=False,
        sharey=False,
        constrained_layout=True,
        squeeze=False,
    )
    for row_idx, period in enumerate(periods):
        period_data = snapshots[snapshots["t"] == period]
        if period_data.empty:
            continue
        max_total = int(period_data["total_count"].max())
        if max_total <= 4:
            ticks = sorted(set([0, max_total]))
        else:
            ticks = [0, max_total // 2, max_total]
        for col_idx, p0 in enumerate(p0_values):
            ax = axes[row_idx, col_idx]
            subset = period_data[np.isclose(period_data["p0"], p0)]
            matrix = np.full((max_total + 1, max_total + 1), np.nan)
            matrix[subset["F"].astype(int), subset["S"].astype(int)] = subset[
                "use_product_2"
            ].to_numpy()
            ax.imshow(
                matrix,
                origin="lower",
                extent=[-0.5, max_total + 0.5, -0.5, max_total + 0.5],
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )
            product2_share = float(subset["use_product_2"].mean())
            ax.text(
                0.96,
                0.05,
                f"{product2_share:.0%}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color="#111827",
                bbox={
                    "boxstyle": "round,pad=0.18",
                    "facecolor": "white",
                    "edgecolor": "#cbd5e1",
                    "alpha": 0.86,
                },
            )
            if row_idx == 0:
                ax.set_title(f"p0={p0:.2f}", loc="left")
            if col_idx == 0:
                ax.set_ylabel(f"t={period}\nFailures F")
            if row_idx == len(periods) - 1:
                ax.set_xlabel("Successes S")
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.tick_params(labelsize=8)
            ax.set_facecolor("#f8fafc")
            prettify_axes(ax, grid_axis="both")

    handles = [
        Patch(facecolor="#f2c94c", edgecolor="none", label="Product 1"),
        Patch(facecolor="#0f766e", edgecolor="none", label="Product 2"),
        Patch(facecolor="white", edgecolor="#cbd5e1", label="panel label = product 2 share"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.015),
        ncols=3,
        frameon=False,
    )
    fig.suptitle(
        f"Finite-horizon policy on reachable states, T={horizon}",
        x=0.01,
        ha="left",
    )
    save_figure(fig, outputs_dir / "finite_horizon_policy_heatmaps_extended.png")


def plot_policy_posterior_space(
    snapshots: pd.DataFrame,
    outputs_dir: Path,
    period: int,
) -> None:
    data = snapshots[snapshots["t"] == period]
    if data.empty:
        return
    p0_values = np.array(sorted(data["p0"].unique()))
    ncols = min(3, len(p0_values))
    nrows = int(np.ceil(len(p0_values) / ncols))
    cmap = ListedColormap(["#f2c94c", "#0f766e"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 4.2 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    image = None
    for ax, p0 in zip(axes, p0_values, strict=False):
        subset = data[np.isclose(data["p0"], p0)]
        image = ax.scatter(
            subset["posterior_mean"],
            subset["posterior_std"],
            c=subset["use_product_2"],
            cmap=cmap,
            norm=norm,
            s=14,
            marker="s",
            linewidths=0.0,
            alpha=0.9,
            rasterized=True,
        )
        ax.axvline(p0, color="#111827", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_title(f"p0={p0:.2f}", loc="left")
        ax.set_xlabel("Posterior mean")
        ax.set_ylabel("Posterior standard deviation")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.005, np.sqrt(1.0 / 12.0) * 1.04)
        prettify_axes(ax, grid_axis="both")
    for ax in axes[len(p0_values) :]:
        ax.set_visible(False)
    cbar = fig.colorbar(image, ax=axes[: len(p0_values)], ticks=[0, 1], shrink=0.82)
    cbar.ax.set_yticklabels(["Product 1", "Product 2"])
    fig.suptitle(
        f"Finite-horizon policy over posterior states at t={period}",
        x=0.01,
        ha="left",
    )
    save_figure(fig, outputs_dir / "finite_horizon_policy_posterior_state_space.png")


def simulation_time_summary_from_paths(paths: pd.DataFrame) -> pd.DataFrame:
    return (
        paths.groupby(["p0", "t"], observed=True)
        .agg(
            rep_count=("rep", "nunique"),
            average_profit_to_date=("average_profit_to_date", "mean"),
            avg_profit_per_period=("profit", "mean"),
            posterior_mean=("posterior_mean", "mean"),
            total_count=("total_count", "mean"),
            total_count_q10=("total_count", lambda x: x.quantile(0.10)),
            total_count_q90=("total_count", lambda x: x.quantile(0.90)),
            rho=("rho", "mean"),
            marginal_reputation_value=("marginal_reputation_value", "mean"),
            marginal_reputation_value_q10=(
                "marginal_reputation_value",
                lambda x: x.quantile(0.10),
            ),
            marginal_reputation_value_q90=(
                "marginal_reputation_value",
                lambda x: x.quantile(0.90),
            ),
            product2_net_benefit=("product2_net_benefit", "mean"),
            policy_recommends_product_2=("policy_recommends_product_2", "mean"),
            A_market_share=("user_chose_A", "mean"),
            product_2_frequency=("product_2_when_A_chosen", "mean"),
        )
        .reset_index()
    )


def selected_p0_values(frame: pd.DataFrame) -> np.ndarray:
    p0_values = np.array(sorted(frame["p0"].unique()))
    selected_count = min(5, len(p0_values))
    selected_indices = sorted(
        set(np.linspace(0, len(p0_values) - 1, selected_count).round().astype(int))
    )
    return p0_values[selected_indices]


def plot_simulation_comparison(paths: pd.DataFrame, outputs_dir: Path) -> None:
    if paths.empty:
        return
    time_summary = simulation_time_summary_from_paths(paths)
    selected_p0 = selected_p0_values(time_summary)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    ax_share, ax_product, ax_profit, ax_mean = axes.ravel()
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(selected_p0)))

    for p0, color in zip(selected_p0, colors, strict=True):
        data = time_summary[np.isclose(time_summary["p0"], p0)]
        label = f"p_0={p0:.2f}"
        ax_share.plot(
            data["t"],
            data["A_market_share"],
            color=color,
            linewidth=2.0,
            label=label,
        )
        ax_product.plot(
            data["t"],
            data["product_2_frequency"],
            color=color,
            linewidth=2.0,
            label=label,
        )
        ax_profit.plot(
            data["t"],
            data["avg_profit_per_period"],
            color=color,
            linewidth=2.0,
            label=label,
        )
        ax_mean.plot(
            data["t"],
            data["posterior_mean"],
            color=color,
            linewidth=2.0,
            label=label,
        )

    ax_share.set_title("Simulated demand path", loc="left")
    ax_share.set_ylabel("A market share")
    ax_share.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_share.set_ylim(-0.03, 1.03)
    prettify_axes(ax_share)
    ax_share.legend(ncols=min(3, len(selected_p0)))

    ax_product.set_title("Product 2 use when A is chosen", loc="left")
    ax_product.set_ylabel("Product 2 rate")
    ax_product.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_product.set_ylim(-0.03, 1.03)
    prettify_axes(ax_product)

    ax_profit.set_title("Per-period profit", loc="left")
    ax_profit.set_xlabel("Period")
    ax_profit.set_ylabel("Average profit")
    prettify_axes(ax_profit)

    ax_mean.set_title("User posterior mean for Seller A", loc="left")
    ax_mean.set_xlabel("Period")
    ax_mean.set_ylabel("E[theta | S,F]")
    ax_mean.set_ylim(-0.03, 1.03)
    prettify_axes(ax_mean)

    fig.suptitle("Finite-horizon early simulated paths", x=0.01, ha="left")
    save_figure(fig, outputs_dir / "finite_horizon_simulation_by_period.png")


def plot_product2_paths_with_asymptotic(
    paths: pd.DataFrame,
    outputs_dir: Path,
    params: ModelParams,
) -> None:
    if paths.empty:
        return
    time_summary = simulation_time_summary_from_paths(paths)
    selected_p0 = selected_p0_values(time_summary)
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(selected_p0)))

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for idx, (p0, color) in enumerate(zip(selected_p0, colors, strict=True)):
        data = time_summary[np.isclose(time_summary["p0"], p0)]
        ax.plot(
            data["t"],
            data["product_2_frequency"],
            color=color,
            linewidth=2.0,
            label=f"p_0={p0:.2f}",
        )
        ax.axhline(
            asymptotic_product2_mix_bound(float(p0), params),
            color=color,
            linewidth=1.2,
            linestyle="--",
            alpha=0.75,
            label="asymptotic mix bound" if idx == 0 else "_nolegend_",
        )

    ax.set_title("Product 2 paths with asymptotic mix bounds", loc="left")
    ax.set_xlabel("Period")
    ax.set_ylabel("Product 2 rate when A is chosen")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_ylim(-0.03, 1.03)
    prettify_axes(ax)
    ax.legend(ncols=min(3, len(selected_p0) + 1))
    fig.suptitle("Finite-horizon simulated product 2 use", x=0.01, ha="left")
    save_figure(fig, outputs_dir / "finite_horizon_product2_paths_with_asymptotic.png")


def plot_reputation_diagnostic_paths(
    paths: pd.DataFrame,
    outputs_dir: Path,
    params: ModelParams,
) -> None:
    required_columns = {
        "p0",
        "rep",
        "t",
        "rho",
        "posterior_mean",
        "total_count",
        "marginal_reputation_value",
        "policy_recommends_product_2",
        "product_2_when_A_chosen",
    }
    if paths.empty or not required_columns.issubset(paths.columns):
        return

    time_summary = simulation_time_summary_from_paths(paths)
    selected_p0 = selected_p0_values(time_summary)
    if len(selected_p0) == 0:
        return

    nrows = len(selected_p0)
    fig, axes = plt.subplots(
        nrows,
        4,
        figsize=(16.6, 3.0 * nrows),
        sharex="col",
        squeeze=False,
    )
    reputation_threshold = product2_continuation_threshold(params)
    panel_titles = [
        "Marginal reputation value",
        "Demand and belief",
        "Belief rigidity",
        "Product 2 use",
    ]

    for col_idx, title in enumerate(panel_titles):
        axes[0, col_idx].set_title(title, loc="left")

    for row_idx, p0 in enumerate(selected_p0):
        data = time_summary[np.isclose(time_summary["p0"], p0)].sort_values("t")
        if data.empty:
            continue
        rep_count = int(data["rep_count"].max())
        x = data["t"].to_numpy()
        ax_gap, ax_belief, ax_count, ax_product = axes[row_idx]

        ax_gap.fill_between(
            x,
            data["marginal_reputation_value_q10"].to_numpy(),
            data["marginal_reputation_value_q90"].to_numpy(),
            color="#bfdbfe",
            alpha=0.55,
            linewidth=0.0,
            label="10-90% band",
        )

        ax_gap.plot(
            x,
            data["marginal_reputation_value"].to_numpy(),
            color="#2563eb",
            linewidth=1.9,
            label=r"mean $M_t(S_t,F_t)$",
        )
        ax_gap.axhline(
            reputation_threshold,
            color="#111827",
            linestyle="--",
            linewidth=1.1,
            label="product 2 threshold",
        )
        ax_gap.set_ylabel(f"p0={p0:.2f}\nmean, n={rep_count}")
        prettify_axes(ax_gap)

        ax_belief.plot(
            x,
            data["rho"].to_numpy(),
            color="#7c3aed",
            linewidth=1.8,
            label=r"$D(S_t,F_t)$",
        )
        ax_belief.plot(
            x,
            data["posterior_mean"].to_numpy(),
            color="#0f766e",
            linewidth=1.8,
            linestyle="--",
            label="posterior mean",
        )
        ax_belief.set_ylim(-0.03, 1.03)
        ax_belief.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        prettify_axes(ax_belief)

        ax_count.fill_between(
            x,
            data["total_count_q10"].to_numpy(),
            data["total_count_q90"].to_numpy(),
            color="#fed7aa",
            alpha=0.45,
            linewidth=0.0,
            label="10-90% band",
        )
        ax_count.plot(
            x,
            data["total_count"].to_numpy(),
            color="#7c2d12",
            linewidth=1.9,
            label=r"mean $S_t+F_t$",
        )
        ax_count.set_ylabel("S_t + F_t")
        ax_count.set_ylim(bottom=-0.5)
        prettify_axes(ax_count)

        ax_product.plot(
            x,
            data["policy_recommends_product_2"].to_numpy(),
            color="#0f766e",
            linewidth=2.1,
            label="policy share",
        )
        ax_product.plot(
            x,
            data["product_2_frequency"].fillna(0.0).to_numpy(),
            color="#2563eb",
            linewidth=1.5,
            linestyle="--",
            label="observed when A chosen",
        )
        ax_product.set_ylim(-0.03, 1.03)
        ax_product.set_ylabel("Product 2 share")
        ax_product.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        prettify_axes(ax_product)

        if row_idx == 0:
            for ax in (ax_gap, ax_belief, ax_count, ax_product):
                ax.legend(fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("Period")

    horizon = int(paths["T"].iloc[0]) if "T" in paths else np.nan
    fig.suptitle(
        f"Finite-horizon reputation diagnostics averaged over simulated paths, T={horizon}",
        x=0.01,
        ha="left",
    )
    save_figure(fig, outputs_dir / "finite_horizon_reputation_diagnostic_paths.png")


def posterior_density_snapshot_periods(max_period: int) -> list[int]:
    candidate_periods = [1, 2, 5, 10, 25, 50, 100, 200, max_period]
    return sorted({period for period in candidate_periods if 1 <= period <= max_period})


def representative_posterior_rep(paths: pd.DataFrame) -> int:
    final_period = int(paths["t"].max())
    final_rows = paths[paths["t"] == final_period].dropna(subset=["posterior_mean"])
    if final_rows.empty:
        return int(paths["rep"].iloc[0])
    median_mean = float(final_rows["posterior_mean"].median())
    closest_idx = (final_rows["posterior_mean"] - median_mean).abs().idxmin()
    return int(final_rows.loc[closest_idx, "rep"])


def plot_posterior_density_panels(
    paths: pd.DataFrame,
    outputs_dir: Path,
    groups: list[dict],
    success_column: str,
    failure_column: str,
    filename: str,
    title: str,
) -> None:
    if paths.empty or not groups:
        return

    ncols = min(3, len(groups))
    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, 3.65 * nrows),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    theta_grid = np.linspace(0.001, 0.999, 700)

    for ax_idx, (ax, group) in enumerate(zip(axes, groups, strict=False)):
        subset = paths[np.isclose(paths["p0"], group["p0"])]
        if subset.empty:
            ax.set_visible(False)
            continue

        target_rep = representative_posterior_rep(subset)
        rep_data = subset[subset["rep"] == target_rep].sort_values("t")
        periods = posterior_density_snapshot_periods(int(rep_data["t"].max()))
        colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(periods)))

        for period, color in zip(periods, colors, strict=True):
            period_rows = rep_data[rep_data["t"] == period]
            if period_rows.empty:
                continue
            row = period_rows.iloc[0]
            density = posterior_density_from_counts(
                theta_grid,
                float(row[success_column]),
                float(row[failure_column]),
            )
            ax.plot(
                theta_grid,
                density,
                color=color,
                linewidth=1.8,
                label=f"t={period}",
            )

        ax.axvline(
            group["p0"],
            color="#111827",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.set_title(f"{group['label']} (rep={target_rep})", loc="left")
        ax.set_xlabel("theta")
        ax.set_ylabel("Posterior density")
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(bottom=0.0)
        prettify_axes(ax)
        if ax_idx == 0:
            ax.legend(ncols=2, fontsize=8)

    for ax in axes[len(groups) :]:
        ax.set_visible(False)

    fig.suptitle(title, x=0.01, ha="left")
    save_figure(fig, outputs_dir / filename)


def plot_posterior_density_evolution(
    paths: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    if paths.empty:
        return
    groups = [
        {"p0": float(p0), "label": f"p0={p0:.2f}"}
        for p0 in selected_p0_values(paths)
    ]
    plot_posterior_density_panels(
        paths=paths,
        outputs_dir=outputs_dir,
        groups=groups,
        success_column="true_S",
        failure_column="true_F",
        filename="finite_horizon_posterior_density_evolution.png",
        title="Finite-horizon user posterior density evolution",
    )


def parse_args() -> argparse.Namespace:
    defaults = ModelParams()
    parser = argparse.ArgumentParser(
        description="Solve exact finite-horizon T=1000 early-period diagnostics."
    )
    parser.add_argument("--T", "--horizon", dest="horizon", type=int, default=1000)
    parser.add_argument("--p0-grid", default=DEFAULT_P0_GRID)
    parser.add_argument("--p1", type=float, default=defaults.p1)
    parser.add_argument("--p2", type=float, default=defaults.p2)
    parser.add_argument("--c1", type=float, default=defaults.c1)
    parser.add_argument("--c2", type=float, default=defaults.c2)
    parser.add_argument("--revenue", "-R", type=float, default=defaults.revenue)
    parser.add_argument("--tol", type=float, default=defaults.tol)
    parser.add_argument(
        "--user-policy",
        choices=[THOMPSON_USER_POLICY, UCB_USER_POLICY, POSTERIOR_MEAN_USER_POLICY],
        default=THOMPSON_USER_POLICY,
        help=(
            "User choice rule: Thompson sampling, deterministic UCB index, or "
            "myopic posterior-mean choice."
        ),
    )
    parser.add_argument(
        "--ucb-alpha",
        type=float,
        default=2.0,
        help=(
            "Exploration scale for --user-policy ucb. The user chooses A when "
            "posterior_mean + sqrt(alpha * log(t+1)/(S+F+2)) >= p0."
        ),
    )
    parser.add_argument(
        "--snapshot-periods",
        default=DEFAULT_SNAPSHOT_PERIODS,
        help="Comma-separated calendar periods to save policy snapshots for.",
    )
    parser.add_argument(
        "--early-periods",
        type=int,
        default=200,
        help="Number of initial policy periods to summarize and snapshot.",
    )
    parser.add_argument(
        "--simulation-periods",
        type=int,
        default=None,
        help=(
            "Number of initial periods to simulate. Defaults to --early-periods, "
            "but may be longer when only the simulated-path plots need extending."
        ),
    )
    parser.add_argument(
        "--policy-plot-periods",
        type=int,
        default=None,
        help=(
            "Also save a standalone Product 2 policy-share plot through this "
            "calendar period. This does not extend the simulation window."
        ),
    )
    parser.add_argument("--n-rep", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260619)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help=(
            "Defaults to finite_horizon/outputs for Thompson sampling and "
            "separate policy-specific folders for UCB and posterior-mean users."
        ),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.horizon <= 0:
        raise ValueError("T must be positive.")
    if args.early_periods <= 0:
        raise ValueError("early-periods must be positive.")
    if args.early_periods > args.horizon:
        raise ValueError("early-periods cannot exceed T.")
    if args.simulation_periods is not None and args.simulation_periods <= 0:
        raise ValueError("simulation-periods must be positive.")
    if (
        args.simulation_periods is not None
        and args.simulation_periods > args.horizon
    ):
        raise ValueError("simulation-periods cannot exceed T.")
    if args.policy_plot_periods is not None and args.policy_plot_periods <= 0:
        raise ValueError("policy-plot-periods must be positive.")
    if args.n_rep <= 0 and not args.skip_simulation:
        raise ValueError("n-rep must be positive unless simulation is skipped.")
    if args.user_policy == UCB_USER_POLICY and args.ucb_alpha <= 0.0:
        raise ValueError("ucb-alpha must be positive for UCB.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_plot_style()

    p0_grid = parse_float_grid(args.p0_grid)
    simulation_periods = (
        args.early_periods
        if args.simulation_periods is None
        else args.simulation_periods
    )
    snapshot_periods = parse_int_grid(args.snapshot_periods)
    snapshot_periods = sorted(
        {period for period in snapshot_periods if period <= args.early_periods}
        | {1, args.early_periods}
    )
    stored_policy_periods = (
        set(range(1, simulation_periods + 1))
        if not args.skip_simulation
        else {1}
    ) | set(snapshot_periods)
    params = ModelParams(
        p1=args.p1,
        p2=args.p2,
        c1=args.c1,
        c2=args.c2,
        revenue=args.revenue,
        max_observations=args.horizon,
        tol=args.tol,
    )

    project_dir = Path(__file__).resolve().parent
    default_outputs_by_policy = {
        THOMPSON_USER_POLICY: "outputs",
        UCB_USER_POLICY: "outputs_ucb",
        POSTERIOR_MEAN_USER_POLICY: "outputs_posterior_mean",
    }
    default_outputs_name = default_outputs_by_policy[args.user_policy]
    outputs_dir = (
        Path(args.outputs_dir)
        if args.outputs_dir
        else project_dir / default_outputs_name
    )
    data_dir = outputs_dir / "data"
    plots_dir = outputs_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    usage_frames = []
    snapshot_frames = []
    simulation_path_frames = []
    simulation_time_frames = []
    posterior_snapshot_period = max(snapshot_periods)

    print(
        f"Solving exact finite-horizon problem with T={args.horizon} "
        f"for {len(p0_grid)} p0 values..."
    )
    print(f"User policy: {user_policy_label(args.user_policy)}")
    if args.user_policy == UCB_USER_POLICY:
        print(f"  UCB alpha={args.ucb_alpha:.6g}")
    print(
        "Saving early policy snapshots at periods: "
        + ", ".join(str(period) for period in snapshot_periods)
    )
    if not args.skip_simulation:
        print(f"Simulating periods 1-{simulation_periods}.")

    for p0_idx, p0 in enumerate(p0_grid):
        print(f"  p0={p0:.3f}: solving backward induction...")
        solution = solve_finite_horizon_early(
            p0=float(p0),
            params=params,
            horizon=args.horizon,
            snapshot_periods=snapshot_periods,
            stored_policy_periods=stored_policy_periods,
            user_policy=args.user_policy,
            ucb_alpha=args.ucb_alpha,
        )
        summary_rows.append(
            summarize_solution(solution, posterior_snapshot_period, params)
        )
        usage_frames.append(solution["usage_by_time"])
        snapshot_frames.append(solution["policy_snapshots"])
        print(
            f"    initial action={solution['initial_action']}, "
            f"initial Q2-Q1={solution['initial_q_gap_product2_minus_product1']:.6g}, "
            "initial continuation gap="
            f"{solution['initial_continuation_gap_success_minus_failure']:.6g}, "
            f"V1/T={solution['avg_value_T']:.6g}"
        )

        if not args.skip_simulation:
            paths, time_summary = simulate_early_policy(
                solution=solution,
                params=params,
                n_rep=args.n_rep,
                periods=simulation_periods,
                seed=args.seed + 10_000 * p0_idx + args.horizon,
            )
            simulation_path_frames.append(paths)
            simulation_time_frames.append(time_summary)
        del solution

    summary = pd.DataFrame.from_records(summary_rows)
    usage_by_time = pd.concat(usage_frames, ignore_index=True)
    snapshots = pd.concat(snapshot_frames, ignore_index=True)
    usage_by_mean = add_posterior_bins(snapshots)

    summary.to_csv(data_dir / "finite_horizon_summary.csv", index=False)
    usage_by_time.to_csv(data_dir / "finite_horizon_product2_by_time.csv", index=False)
    usage_by_mean.to_csv(
        data_dir / "finite_horizon_product2_by_posterior_mean.csv",
        index=False,
    )
    snapshots.to_csv(data_dir / "finite_horizon_policy_snapshots.csv", index=False)

    plot_summary(summary, plots_dir)
    plot_product2_usage(
        usage_by_time,
        usage_by_mean,
        plots_dir,
        early_periods=args.early_periods,
        posterior_snapshot_period=posterior_snapshot_period,
    )
    if args.policy_plot_periods is not None:
        plot_product2_policy_share(
            usage_by_time,
            plots_dir,
            plot_periods=args.policy_plot_periods,
        )
    plot_policy_heatmaps(snapshots, plots_dir, snapshot_periods)
    plot_policy_posterior_space(snapshots, plots_dir, posterior_snapshot_period)

    if simulation_path_frames:
        simulation_paths = pd.concat(simulation_path_frames, ignore_index=True)
        simulation_times = pd.concat(simulation_time_frames, ignore_index=True)
        simulation_paths.to_csv(data_dir / "simulation_paths_early.csv", index=False)
        simulation_times.to_csv(data_dir / "simulation_timeseries_early.csv", index=False)
        plot_simulation_comparison(simulation_paths, plots_dir)
        plot_product2_paths_with_asymptotic(simulation_paths, plots_dir, params)
        plot_reputation_diagnostic_paths(simulation_paths, plots_dir, params)
        plot_posterior_density_evolution(simulation_paths, plots_dir)

    display_columns = [
        "p0",
        "avg_value_T",
        "initial_action",
        "initial_q_gap_product2_minus_product1",
        "initial_continuation_gap_success_minus_failure",
        "asymptotic_product2_mix_bound",
        "snapshot_product_2_share",
    ]
    print("\nSummary")
    print(summary[display_columns].round(6).to_string(index=False))
    print(f"\nSaved data to: {data_dir}")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
