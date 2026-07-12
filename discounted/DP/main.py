"""Exact large-horizon discounted reputation study.

Unlike the previous stationary truncation, this script uses backward induction
on the true count state space through a horizon large enough to make the
discounted tail negligible.  Only the early policy slices needed for plots and
simulation are retained.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import betaln

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "llm_learning_matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter

try:
    from discounted.DP.exact_dp import (
        ModelParams,
        compare_policy_slices,
        late_window_summary,
        product2_gap_threshold,
        required_horizon,
        simulate_count_paths,
        simulate_policy,
        solve_discounted_finite_horizon,
    )
except ModuleNotFoundError:
    from exact_dp import (
        ModelParams,
        compare_policy_slices,
        late_window_summary,
        product2_gap_threshold,
        required_horizon,
        simulate_count_paths,
        simulate_policy,
        solve_discounted_finite_horizon,
    )


DEFAULT_P0_GRID = "0.1,0.3,0.5,0.7,0.9"
POLICY_PRODUCT1_COLOR = "#f2c94c"
POLICY_PRODUCT2_COLOR = "#0f766e"
POLICY_UNREACHABLE_COLOR = "#f8fafc"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_float_grid(value: str) -> np.ndarray:
    values = np.array(
        [float(item.strip()) for item in value.split(",") if item.strip()]
    )
    if len(values) == 0 or np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("p0-grid must contain values strictly between 0 and 1.")
    return np.unique(np.round(values, 10))


def selected_p0_values(frame: pd.DataFrame) -> np.ndarray:
    values = np.array(sorted(frame["p0"].unique()))
    count = min(5, len(values))
    indices = sorted(set(np.linspace(0, len(values) - 1, count).round().astype(int)))
    return values[indices]


def quality_maintenance_mix_benchmark(p0: float, params: ModelParams) -> float:
    return float(np.clip((p0 - params.p1) / (params.p2 - params.p1), 0.0, 1.0))


def posterior_mean(successes: np.ndarray, failures: np.ndarray) -> np.ndarray:
    return (successes + 1.0) / (successes + failures + 2.0)


def posterior_std(successes: np.ndarray, failures: np.ndarray) -> np.ndarray:
    alpha = successes + 1.0
    beta = failures + 1.0
    precision = alpha + beta
    return np.sqrt(alpha * beta / (precision**2 * (precision + 1.0)))


def posterior_density(
    theta_grid: np.ndarray,
    successes: float,
    failures: float,
) -> np.ndarray:
    alpha = successes + 1.0
    beta = failures + 1.0
    return np.exp(
        (alpha - 1.0) * np.log(theta_grid)
        + (beta - 1.0) * np.log1p(-theta_grid)
        - betaln(alpha, beta)
    )


def snapshot_frame(solution: dict, period: int) -> pd.DataFrame:
    snapshot = solution["snapshots"][period]
    return pd.DataFrame(
        {
            "p0": solution["p0"],
            "gamma": solution["gamma"],
            "T": solution["T"],
            "t": period,
            "S": snapshot["S"],
            "F": snapshot["F"],
            "total_count": snapshot["S"] + snapshot["F"],
            "posterior_mean": posterior_mean(snapshot["S"], snapshot["F"]),
            "posterior_std": posterior_std(snapshot["S"], snapshot["F"]),
            "rho": snapshot["rho"],
            "value": snapshot["value"],
            "action": snapshot["action"],
            "use_product2": (snapshot["action"] == 2).astype(np.int8),
            "raw_continuation_gap": snapshot["raw_continuation_gap"],
            "discounted_continuation_gap": snapshot[
                "discounted_continuation_gap"
            ],
            "conditional_q_gap_product2_minus_product1": snapshot[
                "conditional_q_gap_product2_minus_product1"
            ],
        }
    )


def solution_summary(
    solution: dict,
    policy_snapshot: pd.DataFrame,
    simulation_reps: pd.DataFrame,
    late_summary: dict | None,
) -> dict:
    return {
        "p0": solution["p0"],
        "gamma": solution["gamma"],
        "T": solution["T"],
        "gamma_to_T": solution["gamma_to_T"],
        "initial_value": solution["initial_value"],
        "initial_annuity_equivalent": (
            (1.0 - solution["gamma"]) * solution["initial_value"]
        ),
        "initial_action": solution["initial_action"],
        "initial_conditional_q_gap_product2_minus_product1": solution[
            "initial_conditional_q_gap_product2_minus_product1"
        ],
        "initial_raw_continuation_gap": solution[
            "initial_raw_continuation_gap"
        ],
        "initial_discounted_continuation_gap": solution[
            "initial_discounted_continuation_gap"
        ],
        "product2_discounted_gap_threshold": solution[
            "product2_discounted_gap_threshold"
        ],
        "snapshot_period": int(policy_snapshot["t"].iloc[0]),
        "snapshot_product2_share": float(policy_snapshot["use_product2"].mean()),
        "mean_A_market_share_sim": float(simulation_reps["A_market_share"].mean())
        if not simulation_reps.empty
        else np.nan,
        "mean_product2_rate_when_A_chosen_sim": float(
            simulation_reps["product2_rate_when_A_chosen"].mean()
        )
        if not simulation_reps.empty
        else np.nan,
        **(
            {
                f"late_{key}": value
                for key, value in late_summary.items()
            }
            if late_summary
            else {}
        ),
    }


def plot_initial_summary(summary: pd.DataFrame, outputs_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), sharex=True)
    ax_gap, ax_m, ax_share, ax_value = axes.ravel()

    ax_gap.plot(
        summary["p0"],
        summary["initial_conditional_q_gap_product2_minus_product1"],
        color="#2563eb",
        marker="o",
        linewidth=2.0,
    )
    ax_gap.axhline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    ax_gap.set_title("Initial product-2 action gap", loc="left")
    ax_gap.set_ylabel("Q2 - Q1 conditional on A chosen")
    prettify_axes(ax_gap)

    ax_m.plot(
        summary["p0"],
        summary["initial_discounted_continuation_gap"],
        color="#7c3aed",
        marker="o",
        linewidth=2.0,
    )
    ax_m.plot(
        summary["p0"],
        summary["product2_discounted_gap_threshold"],
        color="#111827",
        linestyle="--",
        linewidth=1.1,
        label="Delta c / Delta p",
    )
    ax_m.set_title("Discounted continuation value gap", loc="left")
    ax_m.set_ylabel("M_t = gamma [V(S+1,F)-V(S,F+1)]")
    ax_m.legend()
    prettify_axes(ax_m)

    ax_share.plot(
        summary["p0"],
        summary["snapshot_product2_share"],
        color="#0f766e",
        marker="o",
        linewidth=2.0,
    )
    ax_share.set_title(
        f"Policy share at t={int(summary['snapshot_period'].iloc[0])}",
        loc="left",
    )
    ax_share.set_xlabel("p0")
    ax_share.set_ylabel("States using product 2")
    ax_share.set_ylim(-0.03, 1.03)
    ax_share.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    prettify_axes(ax_share)

    ax_value.plot(
        summary["p0"],
        summary["initial_annuity_equivalent"],
        color="#b45309",
        marker="o",
        linewidth=2.0,
    )
    ax_value.set_title("Initial value, annuity equivalent", loc="left")
    ax_value.set_xlabel("p0")
    ax_value.set_ylabel("(1-gamma) V1(0,0)")
    prettify_axes(ax_value)

    fig.suptitle(
        f"Exact discounted DP, gamma={summary['gamma'].iloc[0]:.3f}, "
        f"T={int(summary['T'].iloc[0])}",
        x=0.01,
        ha="left",
    )
    save_figure(fig, outputs_dir / "discounted_initial_summary.png")


def plot_policy_heatmaps(
    snapshots: pd.DataFrame,
    outputs_dir: Path,
    period: int,
) -> None:
    p0_values = selected_p0_values(snapshots)
    ncols = min(3, len(p0_values))
    nrows = int(np.ceil(len(p0_values) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, 4.2 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    cmap = ListedColormap([POLICY_PRODUCT1_COLOR, POLICY_PRODUCT2_COLOR])
    cmap.set_bad(POLICY_UNREACHABLE_COLOR)
    norm = BoundaryNorm([0.5, 1.5, 2.5], cmap.N)

    for ax, p0 in zip(axes, p0_values, strict=False):
        data = snapshots[np.isclose(snapshots["p0"], p0)]
        grid = np.full((period, period), np.nan)
        grid[data["F"].astype(int), data["S"].astype(int)] = data["action"]
        ax.imshow(
            grid,
            origin="lower",
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(f"p0={p0:.2f}", loc="left")
        ax.set_xlabel("Successes S")
        ax.set_ylabel("Failures F")

    for ax in axes[len(p0_values) :]:
        ax.set_visible(False)
    fig.legend(
        handles=[
            Patch(facecolor=POLICY_PRODUCT1_COLOR, label="Product 1"),
            Patch(facecolor=POLICY_PRODUCT2_COLOR, label="Product 2"),
        ],
        loc="lower center",
        ncols=2,
    )
    fig.suptitle(f"Exact discounted policy at t={period}", x=0.01, ha="left")
    save_figure(fig, outputs_dir / "best_response_policy_heatmaps.png")


def plot_posterior_state_space(
    snapshots: pd.DataFrame,
    outputs_dir: Path,
    period: int,
) -> None:
    p0_values = selected_p0_values(snapshots)
    ncols = min(3, len(p0_values))
    nrows = int(np.ceil(len(p0_values) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 4.3 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    cmap = ListedColormap([POLICY_PRODUCT1_COLOR, POLICY_PRODUCT2_COLOR])
    norm = BoundaryNorm([0.5, 1.5, 2.5], cmap.N)

    for ax, p0 in zip(axes, p0_values, strict=False):
        data = snapshots[np.isclose(snapshots["p0"], p0)]
        ax.scatter(
            data["posterior_mean"],
            data["posterior_std"],
            c=data["action"],
            cmap=cmap,
            norm=norm,
            s=15,
            marker="s",
            linewidths=0.0,
            rasterized=True,
        )
        ax.axvline(p0, color="#111827", linestyle="--", linewidth=1.0)
        ax.set_title(f"p0={p0:.2f}", loc="left")
        ax.set_xlabel("Posterior mean")
        ax.set_ylabel("Posterior standard deviation")
        ax.set_xlim(-0.02, 1.02)
        prettify_axes(ax, grid_axis="both")

    for ax in axes[len(p0_values) :]:
        ax.set_visible(False)
    fig.suptitle(
        f"Exact discounted policy over posterior states at t={period}",
        x=0.01,
        ha="left",
    )
    save_figure(
        fig,
        outputs_dir / "best_response_policy_posterior_state_space.png",
    )


def plot_gap_state_space(
    snapshots: pd.DataFrame,
    outputs_dir: Path,
    period: int,
    threshold: float,
) -> None:
    p0_values = selected_p0_values(snapshots)
    centered = snapshots["discounted_continuation_gap"] - threshold
    vlim = float(np.nanpercentile(np.abs(centered), 98))
    if not np.isfinite(vlim) or vlim <= 0.0:
        vlim = 1.0
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)
    ncols = min(3, len(p0_values))
    nrows = int(np.ceil(len(p0_values) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 4.3 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    image = None

    for ax, p0 in zip(axes, p0_values, strict=False):
        data = snapshots[np.isclose(snapshots["p0"], p0)]
        image = ax.scatter(
            data["posterior_mean"],
            data["posterior_std"],
            c=data["discounted_continuation_gap"] - threshold,
            cmap="RdBu_r",
            norm=norm,
            s=15,
            marker="s",
            linewidths=0.0,
            rasterized=True,
        )
        ax.axvline(p0, color="#111827", linestyle="--", linewidth=1.0)
        ax.set_title(f"p0={p0:.2f}", loc="left")
        ax.set_xlabel("Posterior mean")
        ax.set_ylabel("Posterior standard deviation")
        ax.set_xlim(-0.02, 1.02)
        prettify_axes(ax, grid_axis="both")

    for ax in axes[len(p0_values) :]:
        ax.set_visible(False)
    cbar = fig.colorbar(image, ax=axes[: len(p0_values)], shrink=0.82)
    cbar.set_label("M_t - Delta c / Delta p")
    fig.suptitle(
        f"Discounted continuation value gap at t={period}",
        x=0.01,
        ha="left",
    )
    save_figure(
        fig,
        outputs_dir / "best_response_value_difference_posterior_state_space.png",
    )


def plot_simulation_timeseries(
    time_series: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    p0_values = selected_p0_values(time_series)
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(p0_values)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    ax_share, ax_product, ax_profit, ax_mean = axes.ravel()

    for p0, color in zip(p0_values, colors, strict=True):
        data = time_series[np.isclose(time_series["p0"], p0)]
        label = f"p0={p0:.2f}"
        ax_share.plot(data["t"], data["A_market_share"], color=color, label=label)
        ax_product.plot(
            data["t"],
            data["product2_rate_when_A_chosen"],
            color=color,
            label=label,
        )
        ax_profit.plot(
            data["t"], data["avg_profit_per_period"], color=color, label=label
        )
        ax_mean.plot(
            data["t"], data["mean_posterior_mean"], color=color, label=label
        )

    for ax in (ax_share, ax_product, ax_mean):
        ax.set_ylim(-0.03, 1.03)
    ax_share.set_title("Simulated demand path", loc="left")
    ax_share.set_ylabel("A market share")
    ax_share.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_share.legend(ncols=min(3, len(p0_values)))
    ax_product.set_title("Product 2 use when A is chosen", loc="left")
    ax_product.set_ylabel("Product 2 rate")
    ax_product.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_profit.set_title("Per-period profit", loc="left")
    ax_profit.set_xlabel("Period")
    ax_profit.set_ylabel("Average profit")
    ax_mean.set_title("User posterior mean for Seller A", loc="left")
    ax_mean.set_xlabel("Period")
    ax_mean.set_ylabel("E[theta | S,F]")
    for ax in axes.ravel():
        prettify_axes(ax)
    fig.suptitle("Exact discounted simulated paths", x=0.01, ha="left")
    save_figure(fig, outputs_dir / "discounted_simulation_by_period.png")


def plot_user_belief_timeseries(
    time_series: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    p0_values = selected_p0_values(time_series)
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(p0_values)))
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    ax_mean, ax_demand, ax_std, ax_obs = axes.ravel()

    for p0, color in zip(p0_values, colors, strict=True):
        data = time_series[np.isclose(time_series["p0"], p0)]
        label = f"p0={p0:.2f}"
        ax_mean.plot(data["t"], data["mean_posterior_mean"], color=color, label=label)
        ax_demand.plot(
            data["t"], data["mean_demand_probability"], color=color, label=label
        )
        ax_std.plot(data["t"], data["mean_posterior_std"], color=color, label=label)
        ax_obs.plot(data["t"], data["mean_observations"], color=color, label=label)

    ax_mean.set_title("Posterior mean", loc="left")
    ax_mean.set_ylabel("E[theta | S,F]")
    ax_mean.set_ylim(-0.03, 1.03)
    ax_mean.legend(ncols=min(3, len(p0_values)))
    ax_demand.set_title("Demand probability for Seller A", loc="left")
    ax_demand.set_ylabel("rho(S,F)")
    ax_demand.set_ylim(-0.03, 1.03)
    ax_demand.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_std.set_title("Posterior uncertainty", loc="left")
    ax_std.set_xlabel("Period")
    ax_std.set_ylabel("Posterior standard deviation")
    ax_obs.set_title("Observations of Seller A", loc="left")
    ax_obs.set_xlabel("Period")
    ax_obs.set_ylabel("Average S + F")
    for ax in axes.ravel():
        prettify_axes(ax)
    fig.suptitle("Exact discounted user-belief paths", x=0.01, ha="left")
    save_figure(fig, outputs_dir / "discounted_user_belief_by_period.png")


def plot_product2_benchmark(
    time_series: pd.DataFrame,
    outputs_dir: Path,
    params: ModelParams,
) -> None:
    p0_values = selected_p0_values(time_series)
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(p0_values)))
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    for idx, (p0, color) in enumerate(zip(p0_values, colors, strict=True)):
        data = time_series[np.isclose(time_series["p0"], p0)]
        ax.plot(
            data["t"],
            data["product2_rate_when_A_chosen"],
            color=color,
            linewidth=1.8,
            label=f"p0={p0:.2f}",
        )
        ax.axhline(
            quality_maintenance_mix_benchmark(float(p0), params),
            color=color,
            linestyle="--",
            linewidth=1.1,
            alpha=0.75,
            label="quality-maintenance benchmark" if idx == 0 else "_nolegend_",
        )
    ax.set_title("Product 2 paths with quality-maintenance benchmarks", loc="left")
    ax.set_xlabel("Period")
    ax.set_ylabel("Product 2 rate when A is chosen")
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.legend(ncols=3)
    prettify_axes(ax)
    save_figure(
        fig,
        outputs_dir / "discounted_product2_paths_with_quality_benchmark.png",
    )


def plot_reputation_diagnostics(
    time_series: pd.DataFrame,
    outputs_dir: Path,
    threshold: float,
) -> None:
    p0_values = selected_p0_values(time_series)
    fig, axes = plt.subplots(
        len(p0_values),
        4,
        figsize=(16.5, 2.8 * len(p0_values)),
        sharex="col",
        squeeze=False,
    )
    titles = [
        "Discounted continuation gap",
        "Demand and belief",
        "Belief rigidity",
        "Product 2 use",
    ]
    for idx, title in enumerate(titles):
        axes[0, idx].set_title(title, loc="left")

    for row, p0 in enumerate(p0_values):
        data = time_series[np.isclose(time_series["p0"], p0)]
        ax_gap, ax_belief, ax_obs, ax_p2 = axes[row]
        ax_gap.plot(
            data["t"],
            data["mean_discounted_continuation_gap"],
            color="#2563eb",
        )
        ax_gap.axhline(
            threshold,
            color="#111827",
            linestyle="--",
            linewidth=1.0,
        )
        ax_gap.set_ylabel(f"p0={p0:.2f}")
        ax_belief.plot(
            data["t"],
            data["mean_demand_probability"],
            color="#7c3aed",
            label="demand",
        )
        ax_belief.plot(
            data["t"],
            data["mean_posterior_mean"],
            color="#0f766e",
            linestyle="--",
            label="posterior mean",
        )
        ax_belief.set_ylim(-0.03, 1.03)
        ax_obs.plot(data["t"], data["mean_observations"], color="#7c2d12")
        ax_p2.plot(
            data["t"],
            data["policy_product2_share"],
            color="#0f766e",
            label="policy",
        )
        ax_p2.plot(
            data["t"],
            data["product2_rate_when_A_chosen"],
            color="#2563eb",
            linestyle="--",
            label="realized",
        )
        ax_p2.set_ylim(-0.03, 1.03)
        if row == 0:
            ax_belief.legend(fontsize=8)
            ax_p2.legend(fontsize=8)
        for ax in axes[row]:
            prettify_axes(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Period")
    fig.suptitle(
        "Exact discounted reputation diagnostics averaged over simulations",
        x=0.01,
        ha="left",
    )
    save_figure(
        fig,
        outputs_dir / "discounted_reputation_diagnostic_paths.png",
    )


def representative_path(paths: pd.DataFrame) -> pd.DataFrame:
    final_t = int(paths["t"].max())
    final = paths[paths["t"] == final_t]
    median = float(final["posterior_mean"].median())
    idx = (final["posterior_mean"] - median).abs().idxmin()
    rep = int(final.loc[idx, "rep"])
    return paths[paths["rep"] == rep].sort_values("t")


def plot_posterior_density_evolution(
    paths: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    p0_values = selected_p0_values(paths)
    ncols = min(3, len(p0_values))
    nrows = int(np.ceil(len(p0_values) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, 3.7 * nrows),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    theta = np.linspace(0.001, 0.999, 700)

    for panel, (ax, p0) in enumerate(zip(axes, p0_values, strict=False)):
        path = representative_path(paths[np.isclose(paths["p0"], p0)])
        periods = sorted(
            {
                value
                for value in [1, 2, 5, 10, 25, 50, 100, 200, int(path["t"].max())]
                if value <= path["t"].max()
            }
        )
        colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(periods)))
        for period, color in zip(periods, colors, strict=True):
            row = path[path["t"] == period].iloc[0]
            ax.plot(
                theta,
                posterior_density(theta, row["S"], row["F"]),
                color=color,
                linewidth=1.6,
                label=f"t={period}",
            )
        ax.axvline(p0, color="#111827", linestyle="--", linewidth=1.0)
        ax.set_title(f"p0={p0:.2f}, rep={int(path['rep'].iloc[0])}", loc="left")
        ax.set_xlabel("theta")
        ax.set_ylabel("Posterior density")
        ax.set_xlim(0.0, 1.0)
        prettify_axes(ax)
        if panel == 0:
            ax.legend(ncols=2, fontsize=8)
    for ax in axes[len(p0_values) :]:
        ax.set_visible(False)
    fig.suptitle(
        "Exact discounted user posterior density evolution",
        x=0.01,
        ha="left",
    )
    save_figure(
        fig,
        outputs_dir / "discounted_posterior_density_evolution.png",
    )


def plot_horizon_convergence(
    convergence: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(convergence))
    values = convergence["disagreement_fraction"].to_numpy()
    ax.plot(
        x,
        values,
        color="#2563eb",
        marker="o",
        markersize=7,
        linewidth=1.8,
    )
    for xpos, row in zip(x, convergence.itertuples(), strict=True):
        ax.annotate(
            f"{row.disagreement_count}/{row.state_count}",
            (xpos, row.disagreement_fraction),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )
    ax.set_xticks(x, [f"{p0:.1f}" for p0 in convergence["p0"]])
    ax.set_ylim(-0.0005, max(0.01, float(values.max()) * 1.25))
    ax.set_title("Policy disagreement across discounted horizons", loc="left")
    ax.set_xlabel("p0")
    ax.set_ylabel("Fraction differing at comparison period")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    prettify_axes(ax)
    save_figure(fig, outputs_dir / "discounted_horizon_convergence_t50.png")


def write_continuation_gap_note(path: Path, params: ModelParams) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_threshold = product2_gap_threshold(params) / params.gamma
    path.write_text(
        "\n".join(
            [
                "# Continuation value gap",
                "",
                "Define the raw gap",
                "D_t(S,F) = V_{t+1}(S+1,F) - V_{t+1}(S,F+1).",
                "",
                "The reported discounted continuation value gap is",
                "M_t(S,F) = gamma * D_t(S,F).",
                "",
                "Conditional on Seller A being chosen,",
                "Q2 - Q1 = -(c2-c1) + (p2-p1) * M_t(S,F).",
                "",
                "Therefore product 2 is optimal iff",
                f"M_t(S,F) > (c2-c1)/(p2-p1) = "
                f"{product2_gap_threshold(params):.12g}.",
                "",
                "If the raw gap D_t is plotted instead, its threshold is",
                f"(c2-c1)/(gamma*(p2-p1)) = {raw_threshold:.12g}.",
                "",
                "The unconditional Bellman action gap multiplies this conditional",
                "gap by rho(S,F), which does not change its sign when rho>0.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    defaults = ModelParams()
    parser = argparse.ArgumentParser(
        description="Solve the exact large-horizon discounted reputation DP."
    )
    parser.add_argument("--T", "--horizon", dest="horizon", type=int, default=700)
    parser.add_argument("--simulation-periods", type=int, default=250)
    parser.add_argument("--policy-period", type=int, default=50)
    parser.add_argument("--comparison-horizon", type=int, default=850)
    parser.add_argument("--tail-tolerance", type=float, default=1e-6)
    parser.add_argument("--p0-grid", default=DEFAULT_P0_GRID)
    parser.add_argument("--diagnostic-p0", type=float, default=0.5)
    parser.add_argument("--p1", type=float, default=defaults.p1)
    parser.add_argument("--p2", type=float, default=defaults.p2)
    parser.add_argument("--c1", type=float, default=defaults.c1)
    parser.add_argument("--c2", type=float, default=defaults.c2)
    parser.add_argument("--revenue", "-R", type=float, default=defaults.revenue)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--tol", type=float, default=defaults.tol)
    parser.add_argument("--n-rep", type=int, default=400)
    parser.add_argument("--density-paths", type=int, default=101)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--skip-horizon-check", action="store_true")
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Defaults to discounted/DP/outputs next to this script.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.gamma < 1.0:
        raise ValueError("This discounted study requires gamma in (0,1).")
    if args.horizon <= 0:
        raise ValueError("T must be positive.")
    if not 1 <= args.simulation_periods <= args.horizon:
        raise ValueError("simulation-periods must lie in 1..T.")
    if not 1 <= args.policy_period <= args.horizon:
        raise ValueError("policy-period must lie in 1..T.")
    if (
        not args.skip_horizon_check
        and args.comparison_horizon < args.horizon
    ):
        raise ValueError("comparison-horizon must be at least T.")
    if args.n_rep <= 0 and not args.skip_simulation:
        raise ValueError("n-rep must be positive.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    configure_plot_style()
    p0_grid = parse_float_grid(args.p0_grid)
    if not np.any(np.isclose(p0_grid, args.diagnostic_p0)):
        p0_grid = np.sort(np.append(p0_grid, args.diagnostic_p0))

    params = ModelParams(
        p1=args.p1,
        p2=args.p2,
        c1=args.c1,
        c2=args.c2,
        revenue=args.revenue,
        gamma=args.gamma,
        tol=args.tol,
    )
    minimum_horizon = required_horizon(args.gamma, args.tail_tolerance)
    if args.horizon < minimum_horizon:
        print(
            f"WARNING: T={args.horizon} gives gamma^T={args.gamma**args.horizon:.3e}; "
            f"T>={minimum_horizon} is required for tail tolerance "
            f"{args.tail_tolerance:.1e}."
        )

    project_dir = Path(__file__).resolve().parent
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else project_dir / "outputs"
    data_dir = outputs_dir / "data"
    plots_dir = outputs_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    stored_periods = set(range(1, args.simulation_periods + 1))
    stored_periods.add(args.policy_period)
    solutions: dict[float, dict] = {}
    snapshot_frames = []
    summary_rows = []
    rep_frames = []
    time_frames = []
    path_frames = []

    print(
        f"Solving exact discounted DP with gamma={args.gamma}, T={args.horizon}, "
        f"gamma^T={args.gamma**args.horizon:.3e}..."
    )
    for p0_idx, p0 in enumerate(p0_grid):
        print(f"  p0={p0:.3f}")
        solution = solve_discounted_finite_horizon(
            p0=float(p0),
            params=params,
            horizon=args.horizon,
            stored_policy_periods=stored_periods,
            snapshot_periods={args.policy_period},
        )
        solutions[float(p0)] = solution
        snapshot = snapshot_frame(solution, args.policy_period)
        snapshot_frames.append(snapshot)

        simulation_reps = pd.DataFrame()
        late = None
        if not args.skip_simulation:
            simulation_reps, time_series = simulate_policy(
                solution,
                periods=args.simulation_periods,
                n_rep=args.n_rep,
                seed=args.seed + 10_000 * p0_idx,
            )
            paths = simulate_count_paths(
                solution,
                periods=args.simulation_periods,
                n_paths=args.density_paths,
                seed=args.seed + 10_000 * p0_idx + 777,
            )
            rep_frames.append(simulation_reps)
            time_frames.append(time_series)
            path_frames.append(paths)
            late_start = max(1, args.simulation_periods - 49)
            late = late_window_summary(time_series, late_start)

        summary_rows.append(
            solution_summary(solution, snapshot, simulation_reps, late)
        )

    summary = pd.DataFrame.from_records(summary_rows)
    snapshots = pd.concat(snapshot_frames, ignore_index=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(data_dir / "discounted_summary.csv", index=False)
    snapshots.to_csv(
        data_dir / f"discounted_policy_t{args.policy_period}.csv",
        index=False,
    )
    write_continuation_gap_note(
        outputs_dir / "continuation_value_gap.md",
        params,
    )

    plot_initial_summary(summary, plots_dir)
    plot_policy_heatmaps(snapshots, plots_dir, args.policy_period)
    plot_posterior_state_space(snapshots, plots_dir, args.policy_period)
    plot_gap_state_space(
        snapshots,
        plots_dir,
        args.policy_period,
        product2_gap_threshold(params),
    )

    if time_frames:
        replications = pd.concat(rep_frames, ignore_index=True)
        time_series = pd.concat(time_frames, ignore_index=True)
        paths = pd.concat(path_frames, ignore_index=True)
        data_dir.mkdir(parents=True, exist_ok=True)
        replications.to_csv(
            data_dir / "discounted_simulation_replications.csv",
            index=False,
        )
        time_series.to_csv(
            data_dir / "discounted_simulation_timeseries.csv",
            index=False,
        )
        plot_simulation_timeseries(time_series, plots_dir)
        plot_user_belief_timeseries(time_series, plots_dir)
        plot_product2_benchmark(time_series, plots_dir, params)
        plot_reputation_diagnostics(
            time_series,
            plots_dir,
            product2_gap_threshold(params),
        )
        plot_posterior_density_evolution(paths, plots_dir)

    if not args.skip_horizon_check:
        comparison_rows = []
        for p0 in p0_grid:
            comparison = solve_discounted_finite_horizon(
                p0=float(p0),
                params=params,
                horizon=args.comparison_horizon,
                stored_policy_periods={args.policy_period},
                snapshot_periods=set(),
            )
            row = compare_policy_slices(
                solutions[float(p0)],
                comparison,
                args.policy_period,
            )
            row.update(
                {
                    "p0": float(p0),
                    "gamma": args.gamma,
                    "base_T": args.horizon,
                    "comparison_T": args.comparison_horizon,
                    "gamma_to_base_T": args.gamma**args.horizon,
                    "gamma_to_comparison_T": (
                        args.gamma**args.comparison_horizon
                    ),
                    "initial_value_base": solutions[float(p0)]["initial_value"],
                    "initial_value_comparison": comparison["initial_value"],
                    "initial_value_difference": (
                        comparison["initial_value"]
                        - solutions[float(p0)]["initial_value"]
                    ),
                }
            )
            comparison_rows.append(row)
        convergence = pd.DataFrame.from_records(comparison_rows)
        convergence.to_csv(
            data_dir / "discounted_horizon_convergence.csv",
            index=False,
        )
        plot_horizon_convergence(convergence, plots_dir)
        print("\nHorizon convergence at t=" f"{args.policy_period}")
        print(
            convergence[
                [
                    "p0",
                    "state_count",
                    "disagreement_count",
                    "disagreement_fraction",
                    "initial_value_difference",
                ]
            ].to_string(index=False)
        )

    display = [
        "p0",
        "initial_action",
        "initial_discounted_continuation_gap",
        "snapshot_product2_share",
    ]
    late_columns = [
        "late_mean_A_market_share",
        "late_mean_product2_rate_when_A_chosen",
    ]
    display.extend(column for column in late_columns if column in summary)
    print("\nSummary")
    print(summary[display].round(6).to_string(index=False))
    print(f"\nSaved data to: {data_dir}")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
