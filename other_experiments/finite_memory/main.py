"""
Finite-horizon experiment with a bounded-memory Bayesian user.

The user forms a Beta posterior from at most N remembered observations. When the
memory is full and Seller A is observed again, one remembered observation is
forgotten uniformly at random and the new outcome is added. This is an exact
finite-state dynamic program for that memory model.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaincc

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "llm_learning_matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import PercentFormatter


METHOD = "finite_memory_random_forgetting"
DEFAULT_P0_GRID = "0.1,0.3,0.5,0.7,0.9"
DEFAULT_SNAPSHOT_PERIODS = "1,2,5,10,25,50,100,200,500,1000"


@dataclass(frozen=True)
class ModelParams:
    p1: float = 0.35
    p2: float = 0.80
    c1: float = 0.05
    c2: float = 0.65
    revenue: float = 1.0
    tol: float = 1e-8


@dataclass(frozen=True)
class StateSpace:
    memory_size: int
    S: np.ndarray
    F: np.ndarray
    total: np.ndarray
    state_index: np.ndarray
    success_index: np.ndarray
    success_move_probability: np.ndarray
    success_stay_probability: np.ndarray
    failure_index: np.ndarray
    failure_move_probability: np.ndarray
    failure_stay_probability: np.ndarray


def make_state_space(memory_size: int) -> StateSpace:
    states: list[tuple[int, int]] = []
    state_index = -np.ones((memory_size + 1, memory_size + 1), dtype=int)
    for total in range(memory_size + 1):
        for successes in range(total + 1):
            failures = total - successes
            state_index[successes, failures] = len(states)
            states.append((successes, failures))

    S = np.array([state[0] for state in states], dtype=int)
    F = np.array([state[1] for state in states], dtype=int)
    total = S + F
    success_index = np.empty(len(states), dtype=int)
    success_move_probability = np.empty(len(states), dtype=float)
    success_stay_probability = np.empty(len(states), dtype=float)
    failure_index = np.empty(len(states), dtype=int)
    failure_move_probability = np.empty(len(states), dtype=float)
    failure_stay_probability = np.empty(len(states), dtype=float)

    for idx, (successes, failures) in enumerate(states):
        if successes + failures < memory_size:
            success_index[idx] = state_index[successes + 1, failures]
            success_move_probability[idx] = 1.0
            success_stay_probability[idx] = 0.0
            failure_index[idx] = state_index[successes, failures + 1]
            failure_move_probability[idx] = 1.0
            failure_stay_probability[idx] = 0.0
            continue

        success_index[idx] = state_index[min(successes + 1, memory_size), memory_size - min(successes + 1, memory_size)]
        success_move_probability[idx] = failures / memory_size
        success_stay_probability[idx] = successes / memory_size
        failure_successes = max(successes - 1, 0)
        failure_index[idx] = state_index[failure_successes, memory_size - failure_successes]
        failure_move_probability[idx] = successes / memory_size
        failure_stay_probability[idx] = failures / memory_size

    return StateSpace(
        memory_size=memory_size,
        S=S,
        F=F,
        total=total,
        state_index=state_index,
        success_index=success_index,
        success_move_probability=success_move_probability,
        success_stay_probability=success_stay_probability,
        failure_index=failure_index,
        failure_move_probability=failure_move_probability,
        failure_stay_probability=failure_stay_probability,
    )


def beta_ccdf(p0: float, successes: np.ndarray, failures: np.ndarray) -> np.ndarray:
    return np.clip(betaincc(successes + 1.0, failures + 1.0, p0), 0.0, 1.0)


def beta_ccdf_scalar(p0: float, successes: int, failures: int) -> float:
    return float(beta_ccdf(p0, np.array([successes]), np.array([failures]))[0])


def posterior_mean(successes: np.ndarray, failures: np.ndarray) -> np.ndarray:
    return (successes + 1.0) / (successes + failures + 2.0)


def posterior_mean_scalar(successes: int, failures: int) -> float:
    return float((successes + 1.0) / (successes + failures + 2.0))


def state_count_through_total(total: int) -> int:
    return (total + 1) * (total + 2) // 2


def product2_threshold(params: ModelParams) -> float:
    return (params.c2 - params.c1) / (params.p2 - params.p1)


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
    values = np.array([float(item.strip()) for item in value.split(",") if item.strip()])
    if len(values) == 0:
        raise ValueError("p0-grid must contain at least one value.")
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("All p0 values must be strictly between 0 and 1.")
    return np.unique(np.round(values, 10))


def parse_int_grid(value: str) -> list[int]:
    if not value.strip():
        return []
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if any(item <= 0 for item in values):
        raise ValueError("snapshot periods must be positive.")
    return values


def solve_finite_memory(
    p0: float,
    params: ModelParams,
    horizon: int,
    memory_size: int,
    stored_policy_periods: set[int],
    snapshot_periods: set[int],
) -> dict:
    state_space = make_state_space(memory_size)
    n_states = len(state_space.S)
    next_value = np.zeros(n_states, dtype=float)
    current_value = np.zeros(n_states, dtype=float)
    demand = beta_ccdf(p0, state_space.S, state_space.F)
    threshold = product2_threshold(params)

    policy_by_period: dict[int, np.ndarray] = {}
    marginal_by_period: dict[int, np.ndarray] = {}
    q_gap_by_period: dict[int, np.ndarray] = {}
    time_rows = []
    policy_rows = []
    initial_q_gap = np.nan
    initial_marginal = np.nan

    for period in range(horizon, 0, -1):
        max_reachable_total = min(period - 1, memory_size)
        reachable_count = state_count_through_total(max_reachable_total)
        reachable_slice = slice(0, reachable_count)

        continuation_same = next_value
        continuation_success = (
            state_space.success_move_probability * next_value[state_space.success_index]
            + state_space.success_stay_probability * next_value
        )
        continuation_failure = (
            state_space.failure_move_probability * next_value[state_space.failure_index]
            + state_space.failure_stay_probability * next_value
        )
        marginal = continuation_success - continuation_failure

        q1 = (
            demand * (params.revenue - params.c1)
            + (1.0 - demand) * continuation_same
            + demand
            * (
                params.p1 * continuation_success
                + (1.0 - params.p1) * continuation_failure
            )
        )
        q2 = (
            demand * (params.revenue - params.c2)
            + (1.0 - demand) * continuation_same
            + demand
            * (
                params.p2 * continuation_success
                + (1.0 - params.p2) * continuation_failure
            )
        )
        q_gap = q2 - q1
        use_product2 = q_gap > params.tol
        action = np.where(use_product2, 2, 1).astype(np.int8)
        current_value[:] = np.maximum(q1, q2)

        reachable_policy = use_product2[reachable_slice]
        reachable_demand = demand[reachable_slice]
        demand_weight_sum = float(np.sum(reachable_demand))
        time_rows.append(
            {
                "method": METHOD,
                "p0": p0,
                "T": horizon,
                "memory_size": memory_size,
                "t": period,
                "time_remaining": horizon - period + 1,
                "state_count": reachable_count,
                "product_2_share": float(np.mean(reachable_policy)),
                "demand_weighted_product_2_share": (
                    float(np.sum(reachable_policy * reachable_demand) / demand_weight_sum)
                    if demand_weight_sum > 0.0
                    else np.nan
                ),
            }
        )

        if period == 1:
            initial_q_gap = float(q_gap[0])
            initial_marginal = float(marginal[0])

        if period in stored_policy_periods:
            policy_by_period[period] = action.copy()
            marginal_by_period[period] = marginal.copy()
            q_gap_by_period[period] = q_gap.copy()

        if period in snapshot_periods:
            idx = np.arange(reachable_count)
            policy_rows.append(
                pd.DataFrame(
                    {
                        "method": METHOD,
                        "p0": p0,
                        "T": horizon,
                        "memory_size": memory_size,
                        "t": period,
                        "time_remaining": horizon - period + 1,
                        "S": state_space.S[idx],
                        "F": state_space.F[idx],
                        "observations_in_memory": state_space.total[idx],
                        "posterior_mean": posterior_mean(state_space.S[idx], state_space.F[idx]),
                        "rho": demand[idx],
                        "action": action[idx].astype(int),
                        "use_product_2": use_product2[idx].astype(int),
                        "V_t": current_value[idx],
                        "marginal_reputation_value": marginal[idx],
                        "product2_threshold": threshold,
                        "q_gap_product2_minus_product1": q_gap[idx],
                    }
                )
            )

        next_value, current_value = current_value, next_value

    return {
        "method": METHOD,
        "p0": p0,
        "T": horizon,
        "memory_size": memory_size,
        "state_space": state_space,
        "policy_by_period": policy_by_period,
        "marginal_by_period": marginal_by_period,
        "q_gap_by_period": q_gap_by_period,
        "usage_by_time": pd.DataFrame.from_records(time_rows),
        "policy_snapshots": (
            pd.concat(policy_rows, ignore_index=True) if policy_rows else pd.DataFrame()
        ),
        "initial_value": float(next_value[0]),
        "avg_value_T": float(next_value[0] / horizon),
        "initial_action": int(policy_by_period[1][0]),
        "initial_q_gap_product2_minus_product1": initial_q_gap,
        "initial_marginal_reputation_value": initial_marginal,
    }


def simulate_policy(
    solution: dict,
    params: ModelParams,
    n_rep: int,
    periods: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    p0 = float(solution["p0"])
    horizon = int(solution["T"])
    memory_size = int(solution["memory_size"])
    state_space: StateSpace = solution["state_space"]
    policy_by_period: dict[int, np.ndarray] = solution["policy_by_period"]
    marginal_by_period: dict[int, np.ndarray] = solution["marginal_by_period"]
    q_gap_by_period: dict[int, np.ndarray] = solution["q_gap_by_period"]
    records = []

    for rep in range(n_rep):
        successes = 0
        failures = 0
        cumulative_profit = 0.0
        for period in range(1, periods + 1):
            state_idx = int(state_space.state_index[successes, failures])
            demand_prob = beta_ccdf_scalar(p0, successes, failures)
            action = int(policy_by_period[period][state_idx])
            marginal = float(marginal_by_period[period][state_idx])
            q_gap = float(q_gap_by_period[period][state_idx])
            user_chose_A = rng.random() < demand_prob
            success_value = np.nan
            profit = 0.0

            current_successes = successes
            current_failures = failures
            if user_chose_A:
                success_probability = params.p2 if action == 2 else params.p1
                success_value = rng.random() < success_probability
                profit = params.revenue - (params.c2 if action == 2 else params.c1)
                cumulative_profit += profit

                if successes + failures < memory_size:
                    if success_value:
                        successes += 1
                    else:
                        failures += 1
                elif success_value:
                    if rng.random() < failures / memory_size:
                        successes += 1
                        failures -= 1
                else:
                    if rng.random() < successes / memory_size:
                        successes -= 1
                        failures += 1

            records.append(
                {
                    "method": METHOD,
                    "p0": p0,
                    "T": horizon,
                    "memory_size": memory_size,
                    "rep": rep,
                    "t": period,
                    "time_remaining": horizon - period + 1,
                    "memory_S": current_successes,
                    "memory_F": current_failures,
                    "observations_in_memory": current_successes + current_failures,
                    "posterior_mean": posterior_mean_scalar(
                        current_successes,
                        current_failures,
                    ),
                    "rho": demand_prob,
                    "marginal_reputation_value": marginal,
                    "product2_net_benefit": q_gap,
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
        paths.groupby(["method", "p0", "T", "memory_size", "t"], observed=True)
        .agg(
            average_profit_to_date=("average_profit_to_date", "mean"),
            avg_profit_per_period=("profit", "mean"),
            posterior_mean=("posterior_mean", "mean"),
            observations_in_memory=("observations_in_memory", "mean"),
            rho=("rho", "mean"),
            marginal_reputation_value=("marginal_reputation_value", "mean"),
            product2_net_benefit=("product2_net_benefit", "mean"),
            policy_recommends_product_2=("policy_recommends_product_2", "mean"),
            A_market_share=("user_chose_A", "mean"),
            product_2_frequency=("product_2_when_A_chosen", "mean"),
        )
        .reset_index()
    )
    time_summary["time_remaining"] = horizon - time_summary["t"] + 1
    return paths, time_summary


def summarize_solution(solution: dict, params: ModelParams) -> dict:
    usage = solution["usage_by_time"]
    first_period = usage[usage["t"] == 1].iloc[0]
    return {
        "method": METHOD,
        "p0": solution["p0"],
        "T": solution["T"],
        "memory_size": solution["memory_size"],
        "state_count": len(solution["state_space"].S),
        "initial_total_value": solution["initial_value"],
        "avg_value_T": solution["avg_value_T"],
        "initial_action": solution["initial_action"],
        "initial_q_gap_product2_minus_product1": solution[
            "initial_q_gap_product2_minus_product1"
        ],
        "initial_marginal_reputation_value": solution[
            "initial_marginal_reputation_value"
        ],
        "product2_threshold": product2_threshold(params),
        "initial_product_2_share": first_period["product_2_share"],
    }


def selected_p0_values(frame: pd.DataFrame) -> np.ndarray:
    p0_values = np.array(sorted(frame["p0"].unique()))
    selected_count = min(5, len(p0_values))
    selected_indices = sorted(
        set(np.linspace(0, len(p0_values) - 1, selected_count).round().astype(int))
    )
    return p0_values[selected_indices]


def plot_policy_by_time(usage: pd.DataFrame, outputs_dir: Path) -> None:
    p0_values = selected_p0_values(usage)
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(p0_values)))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8), sharex=True)
    ax_share, ax_weighted = axes
    for p0, color in zip(p0_values, colors, strict=True):
        data = usage[np.isclose(usage["p0"], p0)].sort_values("t")
        ax_share.plot(
            data["t"],
            data["product_2_share"],
            color=color,
            linewidth=1.8,
            label=f"p0={p0:.2f}",
        )
        ax_weighted.plot(
            data["t"],
            data["demand_weighted_product_2_share"],
            color=color,
            linewidth=1.8,
            label=f"p0={p0:.2f}",
        )

    ax_share.set_title("Product 2 share over finite-memory states", loc="left")
    ax_share.set_ylabel("Share of reachable states")
    ax_weighted.set_title("Demand-weighted product 2 share", loc="left")
    ax_weighted.set_ylabel("Demand-weighted share")
    for ax in axes:
        ax.set_xlabel("Calendar period")
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylim(-0.03, 1.03)
        prettify_axes(ax)
    ax_share.legend(ncols=min(3, len(p0_values)))
    save_figure(fig, outputs_dir / "finite_memory_product2_by_time.png")


def plot_simulation(time_summary: pd.DataFrame, outputs_dir: Path) -> None:
    if time_summary.empty:
        return
    p0_values = selected_p0_values(time_summary)
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(p0_values)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    ax_share, ax_product, ax_profit, ax_mean = axes.ravel()

    for p0, color in zip(p0_values, colors, strict=True):
        data = time_summary[np.isclose(time_summary["p0"], p0)].sort_values("t")
        label = f"p0={p0:.2f}"
        ax_share.plot(data["t"], data["A_market_share"], color=color, linewidth=1.8, label=label)
        ax_product.plot(data["t"], data["product_2_frequency"], color=color, linewidth=1.8, label=label)
        ax_profit.plot(data["t"], data["avg_profit_per_period"], color=color, linewidth=1.8, label=label)
        ax_mean.plot(data["t"], data["posterior_mean"], color=color, linewidth=1.8, label=label)

    ax_share.set_title("Simulated demand path", loc="left")
    ax_product.set_title("Product 2 use when A is chosen", loc="left")
    ax_profit.set_title("Per-period profit", loc="left")
    ax_mean.set_title("Mean posterior over remembered observations", loc="left")
    ax_share.set_ylabel("A market share")
    ax_product.set_ylabel("Product 2 rate")
    ax_profit.set_ylabel("Average profit")
    ax_mean.set_ylabel("E[theta | memory]")
    for ax in (ax_share, ax_product):
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylim(-0.03, 1.03)
    ax_mean.set_ylim(-0.03, 1.03)
    for ax in axes[-1]:
        ax.set_xlabel("Period")
    for ax in axes.ravel():
        prettify_axes(ax)
    ax_share.legend(ncols=min(3, len(p0_values)))
    save_figure(fig, outputs_dir / "finite_memory_simulation_by_period.png")


def plot_policy_heatmaps(
    snapshots: pd.DataFrame,
    outputs_dir: Path,
    snapshot_periods: list[int],
) -> None:
    if snapshots.empty:
        return
    p0_values = selected_p0_values(snapshots)
    periods = [period for period in snapshot_periods if period in set(snapshots["t"])]
    if not periods:
        return
    memory_size = int(snapshots["memory_size"].iloc[0])
    periods = periods[: min(4, len(periods))]
    cmap = ListedColormap(["#f2c94c", "#0f766e"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(
        len(periods),
        len(p0_values),
        figsize=(3.0 * len(p0_values), 2.75 * len(periods)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    image = None
    for row_idx, period in enumerate(periods):
        for col_idx, p0 in enumerate(p0_values):
            ax = axes[row_idx, col_idx]
            subset = snapshots[
                snapshots["t"].eq(period) & np.isclose(snapshots["p0"], p0)
            ]
            matrix = np.full((memory_size + 1, memory_size + 1), np.nan)
            matrix[subset["F"].astype(int), subset["S"].astype(int)] = subset[
                "use_product_2"
            ].to_numpy()
            image = ax.imshow(
                matrix,
                origin="lower",
                extent=[-0.5, memory_size + 0.5, -0.5, memory_size + 0.5],
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )
            if row_idx == 0:
                ax.set_title(f"p0={p0:.2f}", loc="left")
            if col_idx == 0:
                ax.set_ylabel(f"t={period}\nFailures F")
            if row_idx == len(periods) - 1:
                ax.set_xlabel("Successes S")
            prettify_axes(ax, grid_axis="both")

    cbar = fig.colorbar(image, ax=axes.ravel(), ticks=[0, 1], shrink=0.82)
    cbar.ax.set_yticklabels(["Product 1", "Product 2"])
    fig.suptitle("Finite-memory policy heatmaps", x=0.01, ha="left")
    save_figure(fig, outputs_dir / "finite_memory_policy_heatmaps.png")


def parse_args() -> argparse.Namespace:
    defaults = ModelParams()
    parser = argparse.ArgumentParser(
        description="Finite-horizon exact DP with bounded user memory."
    )
    parser.add_argument("--T", "--horizon", dest="horizon", type=int, default=1000)
    parser.add_argument("--memory-size", type=int, default=20)
    parser.add_argument("--p0-grid", default=DEFAULT_P0_GRID)
    parser.add_argument("--p1", type=float, default=defaults.p1)
    parser.add_argument("--p2", type=float, default=defaults.p2)
    parser.add_argument("--c1", type=float, default=defaults.c1)
    parser.add_argument("--c2", type=float, default=defaults.c2)
    parser.add_argument("--revenue", "-R", type=float, default=defaults.revenue)
    parser.add_argument("--tol", type=float, default=defaults.tol)
    parser.add_argument("--early-periods", type=int, default=200)
    parser.add_argument("--n-rep", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--snapshot-periods", default=DEFAULT_SNAPSHOT_PERIODS)
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Defaults to other_experiments/finite_memory/outputs next to this script.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.horizon <= 0:
        raise ValueError("T must be positive.")
    if args.memory_size <= 0:
        raise ValueError("memory-size must be positive.")
    if args.early_periods <= 0:
        raise ValueError("early-periods must be positive.")
    if args.early_periods > args.horizon:
        raise ValueError("early-periods cannot exceed T.")
    if args.n_rep <= 0 and not args.skip_simulation:
        raise ValueError("n-rep must be positive unless simulation is skipped.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_plot_style()

    p0_grid = parse_float_grid(args.p0_grid)
    snapshot_periods = sorted(
        {period for period in parse_int_grid(args.snapshot_periods) if period <= args.horizon}
        | {1, min(args.early_periods, args.horizon), args.horizon}
    )
    stored_policy_periods = set(range(1, args.early_periods + 1)) | set(snapshot_periods)
    params = ModelParams(
        p1=args.p1,
        p2=args.p2,
        c1=args.c1,
        c2=args.c2,
        revenue=args.revenue,
        tol=args.tol,
    )

    project_dir = Path(__file__).resolve().parent
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else project_dir / "outputs"
    data_dir = outputs_dir / "data"
    plots_dir = outputs_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Solving finite-memory problem with T={args.horizon}, "
        f"memory_size={args.memory_size}, p0 values={len(p0_grid)}"
    )
    print(
        "Memory model: unordered remembered observations with uniform random "
        "replacement when memory is full."
    )

    summary_rows = []
    usage_frames = []
    snapshot_frames = []
    simulation_path_frames = []
    simulation_time_frames = []

    for p0_idx, p0 in enumerate(p0_grid):
        print(f"  p0={p0:.3f}: solving backward induction...")
        solution = solve_finite_memory(
            p0=float(p0),
            params=params,
            horizon=args.horizon,
            memory_size=args.memory_size,
            stored_policy_periods=stored_policy_periods,
            snapshot_periods=set(snapshot_periods),
        )
        summary_rows.append(summarize_solution(solution, params))
        usage_frames.append(solution["usage_by_time"])
        snapshot_frames.append(solution["policy_snapshots"])
        print(
            f"    initial action={solution['initial_action']}, "
            f"Q2-Q1={solution['initial_q_gap_product2_minus_product1']:.6g}, "
            f"M={solution['initial_marginal_reputation_value']:.6g}, "
            f"V1/T={solution['avg_value_T']:.6g}"
        )

        if not args.skip_simulation:
            paths, time_summary = simulate_policy(
                solution=solution,
                params=params,
                n_rep=args.n_rep,
                periods=args.early_periods,
                seed=args.seed + 10_000 * p0_idx + args.horizon + args.memory_size,
            )
            simulation_path_frames.append(paths)
            simulation_time_frames.append(time_summary)

    summary = pd.DataFrame.from_records(summary_rows)
    usage = pd.concat(usage_frames, ignore_index=True)
    snapshots = pd.concat(snapshot_frames, ignore_index=True)

    summary.to_csv(data_dir / "finite_memory_summary.csv", index=False)
    usage.to_csv(data_dir / "finite_memory_product2_by_time.csv", index=False)
    snapshots.to_csv(data_dir / "finite_memory_policy_snapshots.csv", index=False)
    plot_policy_by_time(usage, plots_dir)
    plot_policy_heatmaps(snapshots, plots_dir, snapshot_periods)

    if simulation_path_frames:
        simulation_paths = pd.concat(simulation_path_frames, ignore_index=True)
        simulation_times = pd.concat(simulation_time_frames, ignore_index=True)
        simulation_paths.to_csv(data_dir / "simulation_paths.csv", index=False)
        simulation_times.to_csv(data_dir / "simulation_timeseries.csv", index=False)
        plot_simulation(simulation_times, plots_dir)

    display_cols = [
        "p0",
        "initial_total_value",
        "avg_value_T",
        "initial_action",
        "initial_q_gap_product2_minus_product1",
        "initial_marginal_reputation_value",
    ]
    print("\nSummary")
    print(summary[display_cols].round(6).to_string(index=False))
    print(f"\nSaved data to: {data_dir}")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
