"""
Seller A best-response simulation for the Bernoulli/Beta model in Paper/main.tex.

The paper's active model has a Thompson-sampling user who treats Seller A as a
single unknown Bernoulli arm. Seller A observes the user's sufficient statistics
(S, F), chooses between two hidden products when selected, and trades off current
cost savings against the effect of success/failure on future demand.

This script solves Seller A's stationary best response on a truncated state
space and then simulates histories under that policy for a grid of p_0 values.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betainc

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "llm_learning_matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter


@dataclass(frozen=True)
class ModelParams:
    """Numerical calibration and approximation controls."""

    p1: float = 0.35
    p2: float = 0.75
    c1: float = 0.05
    c2: float = 0.65
    revenue: float = 1.0
    gamma: float = 0.95
    max_observations: int = 100
    max_iter: int = 2_500
    tol: float = 1e-9
    demand_floor: float = 1e-12


@dataclass(frozen=True)
class StateSpace:
    """Triangular grid of feasible sufficient statistics S + F <= N."""

    S: np.ndarray
    F: np.ndarray
    total: np.ndarray
    state_index: np.ndarray
    success_index: np.ndarray
    failure_index: np.ndarray


def make_state_space(max_observations: int) -> StateSpace:
    states: list[tuple[int, int]] = []
    state_index = -np.ones((max_observations + 1, max_observations + 1), dtype=int)

    for total in range(max_observations + 1):
        for successes in range(total + 1):
            failures = total - successes
            state_index[successes, failures] = len(states)
            states.append((successes, failures))

    S = np.array([state[0] for state in states], dtype=int)
    F = np.array([state[1] for state in states], dtype=int)
    total = S + F
    success_index = np.empty(len(states), dtype=int)
    failure_index = np.empty(len(states), dtype=int)

    for idx, (successes, failures) in enumerate(states):
        if successes + failures < max_observations:
            success_index[idx] = state_index[successes + 1, failures]
            failure_index[idx] = state_index[successes, failures + 1]
        else:
            success_index[idx] = idx
            failure_index[idx] = idx

    return StateSpace(
        S=S,
        F=F,
        total=total,
        state_index=state_index,
        success_index=success_index,
        failure_index=failure_index,
    )


def beta_ccdf(p0: float, S: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Pr(Beta(1 + S, 1 + F) >= p0)."""
    cdf = betainc(S + 1.0, F + 1.0, p0)
    return np.clip(1.0 - cdf, 0.0, 1.0)


def beta_ccdf_scalar(p0: float, successes: int, failures: int) -> float:
    return float(beta_ccdf(p0, np.array([successes]), np.array([failures]))[0])


def posterior_mean(S: np.ndarray, F: np.ndarray) -> np.ndarray:
    return (S + 1.0) / (S + F + 2.0)


def posterior_std(S: np.ndarray, F: np.ndarray) -> np.ndarray:
    alpha = S + 1.0
    beta = F + 1.0
    posterior_precision = alpha + beta
    variance = alpha * beta / (
        posterior_precision**2 * (posterior_precision + 1.0)
    )
    return np.sqrt(variance)


def product2_continuation_gap_threshold(params: ModelParams) -> float:
    """Product 2 is optimal when V(S+1,F) - V(S,F+1) exceeds this cutoff."""
    return (params.c2 - params.c1) / (params.gamma * (params.p2 - params.p1))


def default_lookahead_demand_value(params: ModelParams) -> float:
    """Dollar value assigned to one extra unit of future demand in the heuristic."""
    return (params.revenue - params.c1) / (1.0 - params.gamma)


def solve_one_step_lookahead_policy(
    p0: float,
    params: ModelParams,
    state_space: StateSpace,
    lookahead_demand_value: float,
) -> dict:
    """Approximate Seller A's policy using only the immediate demand response."""
    rho = beta_ccdf(p0, state_space.S, state_space.F)
    rho_after_success = beta_ccdf(p0, state_space.S + 1, state_space.F)
    rho_after_failure = beta_ccdf(p0, state_space.S, state_space.F + 1)
    demand_gain_success_vs_failure = rho_after_success - rho_after_failure
    lookahead_benefit = (
        params.gamma
        * (params.p2 - params.p1)
        * demand_gain_success_vs_failure
        * lookahead_demand_value
    )
    net_benefit = lookahead_benefit - (params.c2 - params.c1)
    policy_product = np.where(
        (net_benefit >= 0.0) & (rho > params.demand_floor),
        2,
        1,
    )

    return {
        "p0": p0,
        "rho": rho,
        "rho_after_success": rho_after_success,
        "rho_after_failure": rho_after_failure,
        "demand_gain_success_vs_failure": demand_gain_success_vs_failure,
        "lookahead_demand_value": lookahead_demand_value,
        "lookahead_benefit": lookahead_benefit,
        "lookahead_net_benefit": net_benefit,
        "policy_product": policy_product,
    }


def solve_best_response(
    p0: float,
    params: ModelParams,
    state_space: StateSpace,
) -> dict:
    """Solve Seller A's approximate infinite-horizon best response."""
    rho = beta_ccdf(p0, state_space.S, state_space.F)
    scale = rho / (1.0 - params.gamma + params.gamma * rho)
    value = np.zeros(len(state_space.S), dtype=float)
    convergence_records = []

    for iteration in range(1, params.max_iter + 1):
        value_success = value[state_space.success_index]
        value_failure = value[state_space.failure_index]

        q1 = (
            params.revenue
            - params.c1
            + params.gamma
            * (params.p1 * value_success + (1.0 - params.p1) * value_failure)
        )
        q2 = (
            params.revenue
            - params.c2
            + params.gamma
            * (params.p2 * value_success + (1.0 - params.p2) * value_failure)
        )
        new_value = scale * np.maximum(q1, q2)
        residual = float(np.max(np.abs(new_value - value)))

        if iteration == 1 or iteration % 25 == 0 or residual < params.tol:
            convergence_records.append(
                {
                    "p0": p0,
                    "iteration": iteration,
                    "residual": residual,
                }
            )

        value = new_value
        if residual < params.tol:
            break

    value_success = value[state_space.success_index]
    value_failure = value[state_space.failure_index]
    continuation_gap = value_success - value_failure
    q1 = (
        params.revenue
        - params.c1
        + params.gamma
        * (params.p1 * value_success + (1.0 - params.p1) * value_failure)
    )
    q2 = (
        params.revenue
        - params.c2
        + params.gamma
        * (params.p2 * value_success + (1.0 - params.p2) * value_failure)
    )
    policy_product = np.where((q2 > q1) & (rho > params.demand_floor), 2, 1)

    return {
        "p0": p0,
        "value": value,
        "rho": rho,
        "q1": q1,
        "q2": q2,
        "q_gap": q2 - q1,
        "continuation_gap": continuation_gap,
        "policy_product": policy_product,
        "iterations": iteration,
        "residual": residual,
        "convergence": pd.DataFrame.from_records(convergence_records),
    }


def projected_state_index(
    successes: int,
    failures: int,
    state_space: StateSpace,
) -> int:
    """Map actual counts to the solved triangular grid."""
    max_observations = state_space.state_index.shape[0] - 1
    total = successes + failures
    if total <= max_observations:
        return int(state_space.state_index[successes, failures])

    projected_successes = int(round(max_observations * successes / total))
    projected_successes = int(np.clip(projected_successes, 0, max_observations))
    projected_failures = max_observations - projected_successes
    return int(state_space.state_index[projected_successes, projected_failures])


def simulate_solution(
    solution: dict,
    params: ModelParams,
    state_space: StateSpace,
    n_rep: int,
    horizon: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate Thompson-sampling demand under Seller A's solved best response."""
    rng = np.random.default_rng(seed)
    p0 = float(solution["p0"])
    policy_product = solution["policy_product"]
    rep_records = []

    chosen_by_t = np.zeros(horizon, dtype=float)
    product2_by_t = np.zeros(horizon, dtype=float)
    product2_den_by_t = np.zeros(horizon, dtype=float)
    profit_by_t = np.zeros(horizon, dtype=float)

    for rep in range(n_rep):
        successes = 0
        failures = 0
        chosen_count = 0
        product2_count = 0
        success_count = 0
        profit_sum = 0.0
        discounted_profit = 0.0
        discount = 1.0

        for t in range(horizon):
            demand_prob = beta_ccdf_scalar(p0, successes, failures)
            chosen_A = rng.random() < demand_prob
            profit = 0.0

            if chosen_A:
                state_idx = projected_state_index(successes, failures, state_space)
                product = int(policy_product[state_idx])
                success_probability = params.p2 if product == 2 else params.p1
                success = rng.random() < success_probability

                if success:
                    successes += 1
                    success_count += 1
                else:
                    failures += 1

                chosen_count += 1
                product2_count += int(product == 2)
                profit = params.revenue - (params.c2 if product == 2 else params.c1)
                product2_by_t[t] += int(product == 2)
                product2_den_by_t[t] += 1.0

            chosen_by_t[t] += float(chosen_A)
            profit_by_t[t] += profit
            profit_sum += profit
            discounted_profit += discount * profit
            discount *= params.gamma

        total_A_observations = successes + failures
        rep_records.append(
            {
                "p0": p0,
                "rep": rep,
                "A_market_share": chosen_count / horizon,
                "product2_rate_when_A_chosen": (
                    product2_count / chosen_count if chosen_count else np.nan
                ),
                "A_success_rate_when_chosen": (
                    success_count / chosen_count if chosen_count else np.nan
                ),
                "avg_profit_per_period": profit_sum / horizon,
                "discounted_profit": discounted_profit,
                "final_successes": successes,
                "final_failures": failures,
                "final_posterior_mean": (successes + 1.0)
                / (total_A_observations + 2.0),
            }
        )

    time_records = []
    for t in range(horizon):
        time_records.append(
            {
                "p0": p0,
                "t": t + 1,
                "A_market_share": chosen_by_t[t] / n_rep,
                "product2_rate_when_A_chosen": (
                    product2_by_t[t] / product2_den_by_t[t]
                    if product2_den_by_t[t] > 0
                    else np.nan
                ),
                "avg_profit_per_period": profit_by_t[t] / n_rep,
            }
        )

    return pd.DataFrame.from_records(rep_records), pd.DataFrame.from_records(time_records)


def summarize_solution(
    solution: dict,
    params: ModelParams,
    state_space: StateSpace,
    simulation_reps: pd.DataFrame,
) -> dict:
    product2 = solution["policy_product"] == 2
    rho = solution["rho"]
    demand_weight_sum = float(np.sum(rho))
    initial_idx = int(state_space.state_index[0, 0])
    product2_reps = simulation_reps["product2_rate_when_A_chosen"].dropna()

    return {
        "p0": solution["p0"],
        "initial_demand_probability": rho[initial_idx],
        "initial_best_response_product": int(solution["policy_product"][initial_idx]),
        "initial_uses_product2": float(solution["policy_product"][initial_idx] == 2),
        "initial_q_gap_product2_minus_product1": solution["q_gap"][initial_idx],
        "initial_value": solution["value"][initial_idx],
        "share_states_product2": float(np.mean(product2)),
        "demand_weighted_share_states_product2": (
            float(np.sum(product2 * rho) / demand_weight_sum)
            if demand_weight_sum > 0
            else np.nan
        ),
        "mean_A_market_share_sim": simulation_reps["A_market_share"].mean(),
        "mean_product2_rate_when_A_chosen_sim": product2_reps.mean(),
        "mean_A_success_rate_when_chosen_sim": simulation_reps[
            "A_success_rate_when_chosen"
        ].mean(),
        "mean_profit_per_period_sim": simulation_reps["avg_profit_per_period"].mean(),
        "mean_discounted_profit_sim": simulation_reps["discounted_profit"].mean(),
        "mean_final_posterior_mean_sim": simulation_reps["final_posterior_mean"].mean(),
        "value_iteration_iterations": solution["iterations"],
        "value_iteration_residual": solution["residual"],
        "p1": params.p1,
        "p2": params.p2,
        "c1": params.c1,
        "c2": params.c2,
        "revenue": params.revenue,
        "gamma": params.gamma,
        "max_observations": params.max_observations,
    }


def build_policy_state_table(
    solutions: list[dict],
    state_space: StateSpace,
    params: ModelParams,
    heuristic_solutions: list[dict] | None = None,
) -> pd.DataFrame:
    frames = []
    for solution_idx, solution in enumerate(solutions):
        data = {
            "p0": solution["p0"],
            "S": state_space.S,
            "F": state_space.F,
            "observations": state_space.total,
            "posterior_mean_A": posterior_mean(state_space.S, state_space.F),
            "posterior_std_A": posterior_std(state_space.S, state_space.F),
            "demand_probability": solution["rho"],
            "best_response_product": solution["policy_product"],
            "uses_product2": (solution["policy_product"] == 2).astype(int),
            "q_gap_product2_minus_product1": solution["q_gap"],
            "continuation_gap_success_minus_failure": solution["continuation_gap"],
            "product2_continuation_gap_threshold": (
                product2_continuation_gap_threshold(params)
            ),
            "continuation_gap_minus_product2_threshold": (
                solution["continuation_gap"]
                - product2_continuation_gap_threshold(params)
            ),
            "value": solution["value"],
        }

        if heuristic_solutions is not None:
            heuristic = heuristic_solutions[solution_idx]
            heuristic_matches = heuristic["policy_product"] == solution["policy_product"]
            data.update(
                {
                    "one_step_heuristic_product": heuristic["policy_product"],
                    "one_step_uses_product2": (
                        heuristic["policy_product"] == 2
                    ).astype(int),
                    "one_step_demand_gain_success_vs_failure": heuristic[
                        "demand_gain_success_vs_failure"
                    ],
                    "one_step_lookahead_benefit": heuristic["lookahead_benefit"],
                    "one_step_net_benefit": heuristic["lookahead_net_benefit"],
                    "one_step_lookahead_demand_value": heuristic[
                        "lookahead_demand_value"
                    ],
                    "one_step_matches_best_response": heuristic_matches.astype(int),
                }
            )

        frames.append(pd.DataFrame(data))
    return pd.concat(frames, ignore_index=True)


def demand_weighted_rate(indicator: np.ndarray, demand_weight: np.ndarray) -> float:
    weight_sum = float(np.sum(demand_weight))
    if weight_sum <= 0.0:
        return np.nan
    return float(np.sum(indicator * demand_weight) / weight_sum)


def compare_heuristic_to_best_response(
    solutions: list[dict],
    heuristic_solutions: list[dict],
    state_space: StateSpace,
    safe_cutoff: int,
) -> pd.DataFrame:
    max_observations = int(state_space.state_index.shape[0] - 1)
    cutoffs = [
        ("all_states", max_observations),
        ("safe_interior", safe_cutoff),
    ]
    rows = []

    for solution, heuristic in zip(solutions, heuristic_solutions, strict=True):
        for cutoff_label, cutoff in cutoffs:
            mask = state_space.total <= cutoff
            best_response_policy = solution["policy_product"][mask]
            heuristic_policy = heuristic["policy_product"][mask]
            demand_weight = solution["rho"][mask]

            best_response_uses_product2 = best_response_policy == 2
            heuristic_uses_product2 = heuristic_policy == 2
            disagreements = best_response_policy != heuristic_policy
            heuristic_only_product2 = (
                heuristic_uses_product2 & ~best_response_uses_product2
            )
            best_response_only_product2 = (
                best_response_uses_product2 & ~heuristic_uses_product2
            )
            initial_idx = int(state_space.state_index[0, 0])

            rows.append(
                {
                    "p0": solution["p0"],
                    "cutoff_label": cutoff_label,
                    "cutoff_observations": cutoff,
                    "states_compared": int(mask.sum()),
                    "agreement_rate": float(1.0 - disagreements.mean()),
                    "disagreement_rate": float(disagreements.mean()),
                    "demand_weighted_disagreement_rate": demand_weighted_rate(
                        disagreements.astype(float),
                        demand_weight,
                    ),
                    "heuristic_only_product2_rate": float(
                        heuristic_only_product2.mean()
                    ),
                    "best_response_only_product2_rate": float(
                        best_response_only_product2.mean()
                    ),
                    "best_response_product2_share": float(
                        best_response_uses_product2.mean()
                    ),
                    "heuristic_product2_share": float(heuristic_uses_product2.mean()),
                    "best_response_demand_weighted_product2_share": (
                        demand_weighted_rate(
                            best_response_uses_product2.astype(float),
                            demand_weight,
                        )
                    ),
                    "heuristic_demand_weighted_product2_share": (
                        demand_weighted_rate(
                            heuristic_uses_product2.astype(float),
                            demand_weight,
                        )
                    ),
                    "initial_best_response_product": int(
                        solution["policy_product"][initial_idx]
                    ),
                    "initial_heuristic_product": int(
                        heuristic["policy_product"][initial_idx]
                    ),
                    "initial_q_gap_product2_minus_product1": float(
                        solution["q_gap"][initial_idx]
                    ),
                    "initial_one_step_demand_gain_success_vs_failure": float(
                        heuristic["demand_gain_success_vs_failure"][initial_idx]
                    ),
                    "initial_one_step_lookahead_benefit": float(
                        heuristic["lookahead_benefit"][initial_idx]
                    ),
                    "initial_one_step_net_benefit": float(
                        heuristic["lookahead_net_benefit"][initial_idx]
                    ),
                    "one_step_lookahead_demand_value": float(
                        heuristic["lookahead_demand_value"]
                    ),
                }
            )

    return pd.DataFrame.from_records(rows)


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


def plot_p0_summary(summary: pd.DataFrame, outputs_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    ax_q, ax_market, ax_p2, ax_profit = axes.ravel()

    ax_q.axhline(0.0, color="#6b7280", linewidth=1.1, linestyle=":")
    ax_q.plot(
        summary["p0"],
        summary["initial_q_gap_product2_minus_product1"],
        color="#0f766e",
        marker="o",
        linewidth=2.0,
    )
    ax_q.set_title("Initial-state incentive for product 2", loc="left")
    ax_q.set_ylabel("Q2 - Q1 at (S,F) = (0,0)")
    prettify_axes(ax_q)

    ax_market.plot(
        summary["p0"],
        summary["initial_demand_probability"],
        color="#64748b",
        marker="o",
        linewidth=2.0,
        label="Initial demand probability",
    )
    ax_market.plot(
        summary["p0"],
        summary["mean_A_market_share_sim"],
        color="#2563eb",
        marker="o",
        linewidth=2.0,
        label="Simulated A market share",
    )
    ax_market.set_title("Demand for Seller A", loc="left")
    ax_market.set_ylabel("Probability/share")
    ax_market.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_market.set_ylim(-0.03, 1.03)
    prettify_axes(ax_market)
    ax_market.legend()

    ax_p2.plot(
        summary["p0"],
        summary["demand_weighted_share_states_product2"],
        color="#475569",
        marker="o",
        linewidth=2.0,
        label="Demand-weighted policy states",
    )
    ax_p2.plot(
        summary["p0"],
        summary["mean_product2_rate_when_A_chosen_sim"],
        color="#dc2626",
        marker="o",
        linewidth=2.0,
        label="Simulated when A is chosen",
    )
    ax_p2.step(
        summary["p0"],
        summary["initial_uses_product2"],
        where="mid",
        color="#111827",
        linewidth=1.4,
        label="Initial state uses product 2",
    )
    ax_p2.set_title("High-quality product use", loc="left")
    ax_p2.set_xlabel("Known competitor success probability p_0")
    ax_p2.set_ylabel("Share using product 2")
    ax_p2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_p2.set_ylim(-0.03, 1.03)
    prettify_axes(ax_p2)
    ax_p2.legend()

    ax_profit.plot(
        summary["p0"],
        summary["mean_profit_per_period_sim"],
        color="#7c3aed",
        marker="o",
        linewidth=2.0,
    )
    ax_profit.set_title("Seller A simulated profit", loc="left")
    ax_profit.set_xlabel("Known competitor success probability p_0")
    ax_profit.set_ylabel("Average profit per period")
    prettify_axes(ax_profit)

    save_figure(fig, outputs_dir / "best_response_by_p0.png")


def plot_policy_heatmaps(
    solutions: list[dict],
    state_space: StateSpace,
    outputs_dir: Path,
) -> None:
    selected_count = min(6, len(solutions))
    selected_indices = sorted(
        set(np.linspace(0, len(solutions) - 1, selected_count).round().astype(int))
    )
    selected_solutions = [solutions[idx] for idx in selected_indices]

    cmap = ListedColormap(["#f2c94c", "#0f766e"])
    cmap.set_bad("#e5e7eb")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    n_panels = len(selected_solutions)
    ncols = 3 if n_panels > 3 else n_panels
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.9 * ncols, 4.5 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    max_obs = state_space.state_index.shape[0] - 1
    image = None
    for ax, solution in zip(axes, selected_solutions, strict=False):
        matrix = np.full((max_obs + 1, max_obs + 1), np.nan)
        uses_product2 = (solution["policy_product"] == 2).astype(float)
        matrix[state_space.F, state_space.S] = uses_product2

        image = ax.imshow(
            matrix,
            origin="lower",
            extent=[-0.5, max_obs + 0.5, -0.5, max_obs + 0.5],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(f"p_0 = {solution['p0']:.2f}", loc="left")
        ax.set_xlabel("Successes S")
        ax.set_ylabel("Failures F")
        ax.plot([0], [0], marker="o", color="#111827", markersize=3)
        prettify_axes(ax, grid_axis="both")

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    cbar = fig.colorbar(image, ax=axes[:n_panels], ticks=[0, 1], shrink=0.82)
    cbar.ax.set_yticklabels(["Product 1: cheap", "Product 2: quality"])
    fig.suptitle("Seller A best-response policy over belief states", x=0.01, ha="left")
    save_figure(fig, outputs_dir / "best_response_policy_heatmaps.png")


def plot_policy_posterior_state_space(
    solutions: list[dict],
    state_space: StateSpace,
    outputs_dir: Path,
    max_total: int | None = None,
    filename: str = "best_response_policy_posterior_state_space.png",
    title: str = "Seller A best-response policy over posterior state space",
) -> None:
    selected_count = min(6, len(solutions))
    selected_indices = sorted(
        set(np.linspace(0, len(solutions) - 1, selected_count).round().astype(int))
    )
    selected_solutions = [solutions[idx] for idx in selected_indices]

    cmap = ListedColormap(["#f2c94c", "#0f766e"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    state_mask = np.ones(len(state_space.S), dtype=bool)
    if max_total is not None:
        state_mask = state_space.total <= max_total
    posterior_m = posterior_mean(state_space.S, state_space.F)[state_mask]
    posterior_s = posterior_std(state_space.S, state_space.F)[state_mask]

    n_panels = len(selected_solutions)
    ncols = 3 if n_panels > 3 else n_panels
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.9 * ncols, 4.5 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    image = None
    for ax, solution in zip(axes, selected_solutions, strict=False):
        uses_product2 = (solution["policy_product"][state_mask] == 2).astype(float)
        image = ax.scatter(
            posterior_m,
            posterior_s,
            c=uses_product2,
            cmap=cmap,
            norm=norm,
            s=9,
            marker="s",
            linewidths=0.0,
            alpha=0.9,
            rasterized=True,
        )
        ax.axvline(
            solution["p0"],
            color="#111827",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.plot(
            [0.5],
            [np.sqrt(1.0 / 12.0)],
            marker="o",
            color="#111827",
            markersize=3.2,
        )
        ax.set_title(f"p_0 = {solution['p0']:.2f}", loc="left")
        ax.set_xlabel("Posterior mean E[theta | S,F]")
        ax.set_ylabel("Posterior standard deviation")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.005, np.sqrt(1.0 / 12.0) * 1.04)
        prettify_axes(ax, grid_axis="both")

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    cbar = fig.colorbar(image, ax=axes[:n_panels], ticks=[0, 1], shrink=0.82)
    cbar.ax.set_yticklabels(["Product 1: cheap", "Product 2: quality"])
    fig.suptitle(title, x=0.01, ha="left")
    save_figure(fig, outputs_dir / filename)


def plot_value_difference_posterior_state_space(
    solutions: list[dict],
    state_space: StateSpace,
    params: ModelParams,
    outputs_dir: Path,
    max_total: int | None = None,
    filename: str = "best_response_value_difference_posterior_state_space.png",
    title: str = "Value difference over posterior state space",
) -> None:
    selected_count = min(6, len(solutions))
    selected_indices = sorted(
        set(np.linspace(0, len(solutions) - 1, selected_count).round().astype(int))
    )
    selected_solutions = [solutions[idx] for idx in selected_indices]

    state_mask = np.ones(len(state_space.S), dtype=bool)
    if max_total is not None:
        state_mask = state_space.total <= max_total
    posterior_m = posterior_mean(state_space.S, state_space.F)[state_mask]
    posterior_s = posterior_std(state_space.S, state_space.F)[state_mask]
    threshold = product2_continuation_gap_threshold(params)
    centered_values = [
        solution["continuation_gap"][state_mask] - threshold
        for solution in selected_solutions
    ]
    all_centered_values = np.concatenate(centered_values)
    vlim = float(np.nanpercentile(np.abs(all_centered_values), 98))
    if not np.isfinite(vlim) or vlim <= 0.0:
        vlim = 1.0
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    n_panels = len(selected_solutions)
    ncols = 3 if n_panels > 3 else n_panels
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.9 * ncols, 4.5 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    image = None
    for ax, solution, centered in zip(
        axes,
        selected_solutions,
        centered_values,
        strict=False,
    ):
        image = ax.scatter(
            posterior_m,
            posterior_s,
            c=centered,
            cmap="RdBu_r",
            norm=norm,
            s=9,
            marker="s",
            linewidths=0.0,
            alpha=0.9,
            rasterized=True,
        )
        ax.axvline(
            solution["p0"],
            color="#111827",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.plot(
            [0.5],
            [np.sqrt(1.0 / 12.0)],
            marker="o",
            color="#111827",
            markersize=3.2,
        )
        ax.set_title(f"p_0 = {solution['p0']:.2f}", loc="left")
        ax.set_xlabel("Posterior mean E[theta | S,F]")
        ax.set_ylabel("Posterior standard deviation")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.005, np.sqrt(1.0 / 12.0) * 1.04)
        prettify_axes(ax, grid_axis="both")

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    cbar = fig.colorbar(image, ax=axes[:n_panels], shrink=0.82)
    cbar.set_label("D(S,F) - product 2 threshold")
    fig.suptitle(title, x=0.01, ha="left")
    save_figure(fig, outputs_dir / filename)


def plot_one_step_heuristic_comparison(
    heuristic_comparison: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    all_states = (
        heuristic_comparison[heuristic_comparison["cutoff_label"] == "all_states"]
        .sort_values("p0")
        .copy()
    )
    safe_interior = (
        heuristic_comparison[
            heuristic_comparison["cutoff_label"] == "safe_interior"
        ]
        .sort_values("p0")
        .copy()
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    ax_agreement, ax_p2, ax_misses, ax_initial = axes.ravel()

    ax_agreement.plot(
        all_states["p0"],
        all_states["agreement_rate"],
        color="#334155",
        marker="o",
        linewidth=2.0,
        label="All states",
    )
    ax_agreement.plot(
        safe_interior["p0"],
        safe_interior["agreement_rate"],
        color="#0f766e",
        marker="o",
        linewidth=2.0,
        label="Safe interior",
    )
    ax_agreement.plot(
        all_states["p0"],
        1.0 - all_states["demand_weighted_disagreement_rate"],
        color="#2563eb",
        marker="o",
        linewidth=2.0,
        label="Demand-weighted",
    )
    ax_agreement.set_title("Agreement with DP policy", loc="left")
    ax_agreement.set_ylabel("Share of states")
    ax_agreement.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_agreement.set_ylim(-0.03, 1.03)
    prettify_axes(ax_agreement)
    ax_agreement.legend()

    ax_p2.plot(
        all_states["p0"],
        all_states["best_response_demand_weighted_product2_share"],
        color="#0f766e",
        marker="o",
        linewidth=2.0,
        label="DP best response",
    )
    ax_p2.plot(
        all_states["p0"],
        all_states["heuristic_demand_weighted_product2_share"],
        color="#dc2626",
        marker="o",
        linewidth=2.0,
        label="One-step heuristic",
    )
    ax_p2.set_title("Demand-weighted product 2 use", loc="left")
    ax_p2.set_ylabel("Share using product 2")
    ax_p2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_p2.set_ylim(-0.03, 1.03)
    prettify_axes(ax_p2)
    ax_p2.legend()

    ax_misses.plot(
        safe_interior["p0"],
        safe_interior["heuristic_only_product2_rate"],
        color="#dc2626",
        marker="o",
        linewidth=2.0,
        label="Heuristic uses product 2 only",
    )
    ax_misses.plot(
        safe_interior["p0"],
        safe_interior["best_response_only_product2_rate"],
        color="#2563eb",
        marker="o",
        linewidth=2.0,
        label="DP uses product 2 only",
    )
    ax_misses.set_title("Direction of disagreement", loc="left")
    ax_misses.set_xlabel("Known competitor success probability p_0")
    ax_misses.set_ylabel("Safe-interior state share")
    ax_misses.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_misses.set_ylim(-0.03, 1.03)
    prettify_axes(ax_misses)
    ax_misses.legend()

    ax_initial.axhline(0.0, color="#6b7280", linewidth=1.1, linestyle=":")
    ax_initial.plot(
        all_states["p0"],
        all_states["initial_q_gap_product2_minus_product1"],
        color="#0f766e",
        marker="o",
        linewidth=2.0,
        label="DP Q2 - Q1",
    )
    ax_initial.plot(
        all_states["p0"],
        all_states["initial_one_step_net_benefit"],
        color="#dc2626",
        marker="o",
        linewidth=2.0,
        label="Heuristic net benefit",
    )
    ax_initial.set_title("Initial-state product 2 incentive", loc="left")
    ax_initial.set_xlabel("Known competitor success probability p_0")
    ax_initial.set_ylabel("Product 2 advantage")
    prettify_axes(ax_initial)
    ax_initial.legend()

    save_figure(fig, outputs_dir / "one_step_heuristic_comparison_by_p0.png")


def plot_one_step_heuristic_agreement_heatmaps(
    solutions: list[dict],
    heuristic_solutions: list[dict],
    state_space: StateSpace,
    outputs_dir: Path,
    max_total: int | None = None,
    filename: str = "one_step_heuristic_agreement_heatmaps.png",
    title: str = "One-step heuristic agreement with DP best response",
) -> None:
    selected_count = min(6, len(solutions))
    selected_indices = sorted(
        set(np.linspace(0, len(solutions) - 1, selected_count).round().astype(int))
    )
    selected_pairs = [
        (solutions[idx], heuristic_solutions[idx]) for idx in selected_indices
    ]

    cmap = ListedColormap(["#f2c94c", "#0f766e", "#dc2626", "#2563eb"])
    cmap.set_bad("#e5e7eb")
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)

    n_panels = len(selected_pairs)
    ncols = 3 if n_panels > 3 else n_panels
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.9 * ncols, 4.5 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    max_obs = state_space.state_index.shape[0] - 1
    state_mask = np.ones(len(state_space.S), dtype=bool)
    if max_total is not None:
        state_mask = state_space.total <= max_total

    image = None
    for ax, (solution, heuristic) in zip(axes, selected_pairs, strict=False):
        best_response_uses_product2 = solution["policy_product"] == 2
        heuristic_uses_product2 = heuristic["policy_product"] == 2
        category = np.full(len(state_space.S), np.nan)
        category[
            state_mask & ~best_response_uses_product2 & ~heuristic_uses_product2
        ] = 0
        category[
            state_mask & best_response_uses_product2 & heuristic_uses_product2
        ] = 1
        category[
            state_mask & ~best_response_uses_product2 & heuristic_uses_product2
        ] = 2
        category[
            state_mask & best_response_uses_product2 & ~heuristic_uses_product2
        ] = 3

        matrix = np.full((max_obs + 1, max_obs + 1), np.nan)
        matrix[state_space.F, state_space.S] = category
        image = ax.imshow(
            matrix,
            origin="lower",
            extent=[-0.5, max_obs + 0.5, -0.5, max_obs + 0.5],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(f"p_0 = {solution['p0']:.2f}", loc="left")
        ax.set_xlabel("Successes S")
        ax.set_ylabel("Failures F")
        ax.plot([0], [0], marker="o", color="#111827", markersize=3)
        prettify_axes(ax, grid_axis="both")

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    cbar = fig.colorbar(image, ax=axes[:n_panels], ticks=[0, 1, 2, 3], shrink=0.82)
    cbar.ax.set_yticklabels(
        [
            "Agree: product 1",
            "Agree: product 2",
            "Heuristic only product 2",
            "DP only product 2",
        ]
    )
    fig.suptitle(title, x=0.01, ha="left")
    save_figure(fig, outputs_dir / filename)


def plot_simulation_timeseries(
    time_series: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    selected_count = min(5, time_series["p0"].nunique())
    p0_values = np.array(sorted(time_series["p0"].unique()))
    selected_indices = sorted(
        set(np.linspace(0, len(p0_values) - 1, selected_count).round().astype(int))
    )
    selected_p0 = p0_values[selected_indices]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True)
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(selected_p0)))

    for p0, color in zip(selected_p0, colors, strict=True):
        data = time_series[np.isclose(time_series["p0"], p0)]
        axes[0].plot(
            data["t"],
            data["A_market_share"],
            color=color,
            linewidth=2.0,
            label=f"p_0={p0:.2f}",
        )
        axes[1].plot(
            data["t"],
            data["product2_rate_when_A_chosen"],
            color=color,
            linewidth=2.0,
            label=f"p_0={p0:.2f}",
        )

    axes[0].set_title("Simulated demand path", loc="left")
    axes[0].set_ylabel("A market share")
    axes[0].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[0].set_ylim(-0.03, 1.03)
    prettify_axes(axes[0])
    axes[0].legend(ncols=min(3, len(selected_p0)))

    axes[1].set_title("Simulated product 2 use when A is chosen", loc="left")
    axes[1].set_xlabel("Period")
    axes[1].set_ylabel("Product 2 rate")
    axes[1].yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    axes[1].set_ylim(-0.03, 1.03)
    prettify_axes(axes[1])

    save_figure(fig, outputs_dir / "simulation_paths_by_p0.png")


def parse_p0_grid(args: argparse.Namespace) -> np.ndarray:
    if args.p0_grid:
        values = np.array([float(item.strip()) for item in args.p0_grid.split(",")])
    else:
        values = np.linspace(args.p0_min, args.p0_max, args.p0_count)

    values = np.unique(np.round(values, 10))
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("All p_0 values must be strictly between 0 and 1.")
    return values


def parse_int_grid(value: str) -> list[int]:
    if not value.strip():
        return []
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def compare_policy_to_larger_truncation(
    base_solution: dict,
    base_state_space: StateSpace,
    comparison_solution: dict,
    comparison_state_space: StateSpace,
    cutoff: int,
    cutoff_label: str,
) -> dict:
    mask = base_state_space.total <= cutoff
    comparison_indices = comparison_state_space.state_index[
        base_state_space.S[mask],
        base_state_space.F[mask],
    ]
    base_policy = base_solution["policy_product"][mask]
    comparison_policy = comparison_solution["policy_product"][comparison_indices]
    base_q_gap = base_solution["q_gap"][mask]
    comparison_q_gap = comparison_solution["q_gap"][comparison_indices]
    base_d = base_solution["continuation_gap"][mask]
    comparison_d = comparison_solution["continuation_gap"][comparison_indices]
    disagreements = base_policy != comparison_policy
    initial_base = int(base_state_space.state_index[0, 0])
    initial_comparison = int(comparison_state_space.state_index[0, 0])

    return {
        "p0": base_solution["p0"],
        "base_max_observations": int(base_state_space.state_index.shape[0] - 1),
        "comparison_max_observations": int(
            comparison_state_space.state_index.shape[0] - 1
        ),
        "cutoff_label": cutoff_label,
        "cutoff_observations": cutoff,
        "states_compared": int(mask.sum()),
        "policy_disagreement_count": int(disagreements.sum()),
        "policy_disagreement_rate": float(disagreements.mean()),
        "product2_share_base": float(np.mean(base_policy == 2)),
        "product2_share_comparison": float(np.mean(comparison_policy == 2)),
        "mean_abs_q_gap_diff": float(np.mean(np.abs(base_q_gap - comparison_q_gap))),
        "max_abs_q_gap_diff": float(np.max(np.abs(base_q_gap - comparison_q_gap))),
        "mean_abs_D_diff": float(np.mean(np.abs(base_d - comparison_d))),
        "max_abs_D_diff": float(np.max(np.abs(base_d - comparison_d))),
        "initial_product_base": int(base_solution["policy_product"][initial_base]),
        "initial_product_comparison": int(
            comparison_solution["policy_product"][initial_comparison]
        ),
        "initial_q_gap_base": float(base_solution["q_gap"][initial_base]),
        "initial_q_gap_comparison": float(
            comparison_solution["q_gap"][initial_comparison]
        ),
    }


def run_truncation_robustness_checks(
    base_solutions: list[dict],
    base_state_space: StateSpace,
    params: ModelParams,
    comparison_max_observations: list[int],
    safe_margin: int,
) -> pd.DataFrame:
    base_max_observations = int(base_state_space.state_index.shape[0] - 1)
    comparison_max_observations = [
        value for value in comparison_max_observations if value > base_max_observations
    ]
    if not comparison_max_observations:
        return pd.DataFrame()

    rows = []
    p0_values = [float(solution["p0"]) for solution in base_solutions]
    cutoffs = [
        ("all_common_states", base_max_observations),
        ("safe_interior", max(0, base_max_observations - safe_margin)),
    ]

    for max_observations in comparison_max_observations:
        comparison_params = replace(params, max_observations=max_observations)
        comparison_state_space = make_state_space(max_observations)
        print(f"\nChecking truncation robustness at N={max_observations}...")
        for base_solution, p0 in zip(base_solutions, p0_values, strict=True):
            comparison_solution = solve_best_response(
                p0,
                comparison_params,
                comparison_state_space,
            )
            for cutoff_label, cutoff in cutoffs:
                rows.append(
                    compare_policy_to_larger_truncation(
                        base_solution,
                        base_state_space,
                        comparison_solution,
                        comparison_state_space,
                        cutoff,
                        cutoff_label,
                    )
                )

    return pd.DataFrame.from_records(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve and simulate Seller A's best response in the Bernoulli/Beta "
            "model from Paper/main.tex."
        )
    )
    parser.add_argument("--p0-grid", default=None, help="Comma-separated p_0 values.")
    parser.add_argument("--p0-min", type=float, default=0.10)
    parser.add_argument("--p0-max", type=float, default=0.90)
    parser.add_argument("--p0-count", type=int, default=17)
    parser.add_argument("--p1", type=float, default=0.35)
    parser.add_argument("--p2", type=float, default=0.75)
    parser.add_argument("--c1", type=float, default=0.05)
    parser.add_argument("--c2", type=float, default=0.65)
    parser.add_argument("--revenue", "-R", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--max-observations", type=int, default=100)
    parser.add_argument("--safe-margin", type=int, default=10)
    parser.add_argument(
        "--lookahead-demand-value",
        type=float,
        default=None,
        help=(
            "Value assigned to one extra unit of future demand in the one-step "
            "heuristic. Defaults to (R-c1)/(1-gamma)."
        ),
    )
    parser.add_argument(
        "--robustness-max-observations",
        default="150,200",
        help="Comma-separated larger truncation levels used for robustness checks.",
    )
    parser.add_argument("--skip-robustness-checks", action="store_true")
    parser.add_argument("--max-iter", type=int, default=2_500)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--T", "--horizon", dest="horizon", type=int, default=250)
    parser.add_argument("--n-rep", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260615)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--skip-simulation", action="store_true")
    return parser.parse_args()


def validate_params(params: ModelParams) -> None:
    if not 0.0 < params.p1 < params.p2 < 1.0:
        raise ValueError("Require 0 < p1 < p2 < 1.")
    if not params.c1 < params.c2:
        raise ValueError("Require c1 < c2.")
    if not 0.0 < params.gamma < 1.0:
        raise ValueError("Require 0 < gamma < 1.")
    if params.revenue <= params.c2:
        raise ValueError("Revenue should exceed c2 so product 2 can be profitable.")
    if params.max_observations < 2:
        raise ValueError("max_observations must be at least 2.")


def main() -> None:
    args = parse_args()
    if args.safe_margin < 0:
        raise ValueError("safe-margin must be nonnegative.")
    p0_grid = parse_p0_grid(args)
    robustness_max_observations = parse_int_grid(args.robustness_max_observations)
    params = ModelParams(
        p1=args.p1,
        p2=args.p2,
        c1=args.c1,
        c2=args.c2,
        revenue=args.revenue,
        gamma=args.gamma,
        max_observations=args.max_observations,
        max_iter=args.max_iter,
        tol=args.tol,
    )
    validate_params(params)
    lookahead_demand_value = (
        default_lookahead_demand_value(params)
        if args.lookahead_demand_value is None
        else args.lookahead_demand_value
    )
    if lookahead_demand_value <= 0.0:
        raise ValueError("lookahead-demand-value must be positive.")

    outputs_dir = Path(args.outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    state_space = make_state_space(params.max_observations)
    configure_plot_style()

    solutions = []
    heuristic_solutions = []
    simulation_rep_frames = []
    simulation_time_frames = []
    summary_records = []

    print("Solving Seller A best responses across p_0 values...")
    for p0_idx, p0 in enumerate(p0_grid):
        solution = solve_best_response(float(p0), params, state_space)
        heuristic_solution = solve_one_step_lookahead_policy(
            float(p0),
            params,
            state_space,
            lookahead_demand_value,
        )
        solutions.append(solution)
        heuristic_solutions.append(heuristic_solution)

        if args.skip_simulation:
            simulation_reps = pd.DataFrame(
                {
                    "p0": [p0],
                    "rep": [0],
                    "A_market_share": [np.nan],
                    "product2_rate_when_A_chosen": [np.nan],
                    "A_success_rate_when_chosen": [np.nan],
                    "avg_profit_per_period": [np.nan],
                    "discounted_profit": [np.nan],
                    "final_posterior_mean": [np.nan],
                }
            )
        else:
            simulation_reps, simulation_time = simulate_solution(
                solution=solution,
                params=params,
                state_space=state_space,
                n_rep=args.n_rep,
                horizon=args.horizon,
                seed=args.seed + 10_000 * p0_idx,
            )
            simulation_rep_frames.append(simulation_reps)
            simulation_time_frames.append(simulation_time)

        summary_records.append(
            summarize_solution(solution, params, state_space, simulation_reps)
        )
        print(
            f"  p_0={p0:.3f}: initial product "
            f"{int(solution['policy_product'][0])}, "
            f"initial Q2-Q1={solution['q_gap'][0]:.4f}, "
            f"iterations={solution['iterations']}, "
            f"residual={solution['residual']:.2e}"
        )

    summary = pd.DataFrame.from_records(summary_records)
    safe_cutoff = max(0, params.max_observations - args.safe_margin)
    policy_states = build_policy_state_table(
        solutions,
        state_space,
        params,
        heuristic_solutions=heuristic_solutions,
    )
    heuristic_comparison = compare_heuristic_to_best_response(
        solutions,
        heuristic_solutions,
        state_space,
        safe_cutoff,
    )
    convergence = pd.concat(
        [solution["convergence"] for solution in solutions],
        ignore_index=True,
    )

    summary.to_csv(outputs_dir / "best_response_summary.csv", index=False)
    policy_states.to_csv(outputs_dir / "best_response_policy_by_state.csv", index=False)
    heuristic_comparison.to_csv(
        outputs_dir / "one_step_heuristic_comparison.csv",
        index=False,
    )
    convergence.to_csv(outputs_dir / "value_iteration_convergence.csv", index=False)

    if simulation_rep_frames:
        simulation_reps = pd.concat(simulation_rep_frames, ignore_index=True)
        simulation_times = pd.concat(simulation_time_frames, ignore_index=True)
        simulation_reps.to_csv(outputs_dir / "simulation_replications.csv", index=False)
        simulation_times.to_csv(outputs_dir / "simulation_timeseries.csv", index=False)
        plot_simulation_timeseries(simulation_times, outputs_dir)

    plot_p0_summary(summary, outputs_dir)
    plot_policy_heatmaps(solutions, state_space, outputs_dir)
    plot_policy_posterior_state_space(solutions, state_space, outputs_dir)
    plot_policy_posterior_state_space(
        solutions,
        state_space,
        outputs_dir,
        max_total=safe_cutoff,
        filename="best_response_policy_posterior_state_space_interior.png",
        title=(
            "Seller A best-response policy over posterior state space "
            f"(S+F <= {safe_cutoff})"
        ),
    )
    plot_value_difference_posterior_state_space(
        solutions,
        state_space,
        params,
        outputs_dir,
    )
    plot_value_difference_posterior_state_space(
        solutions,
        state_space,
        params,
        outputs_dir,
        max_total=safe_cutoff,
        filename="best_response_value_difference_posterior_state_space_interior.png",
        title=(
            "Value difference over posterior state space "
            f"(S+F <= {safe_cutoff})"
        ),
    )
    plot_one_step_heuristic_comparison(heuristic_comparison, outputs_dir)
    plot_policy_posterior_state_space(
        heuristic_solutions,
        state_space,
        outputs_dir,
        filename="one_step_heuristic_policy_posterior_state_space.png",
        title="One-step lookahead heuristic policy over posterior state space",
    )
    plot_one_step_heuristic_agreement_heatmaps(
        solutions,
        heuristic_solutions,
        state_space,
        outputs_dir,
    )
    plot_one_step_heuristic_agreement_heatmaps(
        solutions,
        heuristic_solutions,
        state_space,
        outputs_dir,
        max_total=safe_cutoff,
        filename="one_step_heuristic_agreement_heatmaps_interior.png",
        title=(
            "One-step heuristic agreement with DP best response "
            f"(S+F <= {safe_cutoff})"
        ),
    )

    robustness = pd.DataFrame()
    if not args.skip_robustness_checks:
        robustness = run_truncation_robustness_checks(
            solutions,
            state_space,
            params,
            robustness_max_observations,
            args.safe_margin,
        )
        if not robustness.empty:
            robustness.to_csv(outputs_dir / "truncation_robustness.csv", index=False)

    display_columns = [
        "p0",
        "initial_best_response_product",
        "initial_q_gap_product2_minus_product1",
        "mean_A_market_share_sim",
        "mean_product2_rate_when_A_chosen_sim",
        "mean_profit_per_period_sim",
    ]
    print("\nSummary")
    print(summary[display_columns].round(4).to_string(index=False))

    heuristic_display_columns = [
        "p0",
        "agreement_rate",
        "demand_weighted_disagreement_rate",
        "best_response_demand_weighted_product2_share",
        "heuristic_demand_weighted_product2_share",
        "initial_heuristic_product",
        "initial_one_step_net_benefit",
    ]
    heuristic_display = (
        heuristic_comparison[heuristic_comparison["cutoff_label"] == "all_states"]
        .sort_values("p0")
        .copy()
    )
    print(
        "\nOne-step heuristic comparison "
        f"(demand value={lookahead_demand_value:.4f})"
    )
    print(
        heuristic_display[heuristic_display_columns].round(4).to_string(index=False)
    )
    if not robustness.empty:
        robustness_summary = (
            robustness.groupby(
                ["comparison_max_observations", "cutoff_label"],
                as_index=False,
            )
            .agg(
                max_policy_disagreement_rate=("policy_disagreement_rate", "max"),
                mean_policy_disagreement_rate=("policy_disagreement_rate", "mean"),
                max_abs_q_gap_diff=("max_abs_q_gap_diff", "max"),
                max_abs_D_diff=("max_abs_D_diff", "max"),
            )
            .sort_values(["comparison_max_observations", "cutoff_label"])
        )
        print("\nTruncation robustness summary")
        print(robustness_summary.round(6).to_string(index=False))
    print(f"\nSaved CSVs and plots to: {outputs_dir}")


if __name__ == "__main__":
    main()
