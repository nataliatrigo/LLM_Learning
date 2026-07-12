"""Solve and plot the discounted fluid model in the original (s, f) state.

The stationary HJB is solved by a monotone semi-Lagrangian recursion.  At each
step the total fluid count increases by ``grid_step`` and the next state lies
between the adjacent success/failure grid points.  The terminal count is chosen
so that discounting bounds its influence on the plotted state region by the
requested tail tolerance.
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
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


DEFAULT_P0_GRID = "0.1,0.3,0.5,0.7,0.9"
PRODUCT1_COLOR = "#f2c94c"
PRODUCT2_COLOR = "#0f766e"


@dataclass(frozen=True)
class FluidParams:
    p1: float = 0.35
    p2: float = 0.80
    c1: float = 0.05
    c2: float = 0.65
    revenue: float = 1.0
    gamma: float = 0.999

    @property
    def discount_rate(self) -> float:
        return -float(np.log(self.gamma))


@dataclass
class FluidSolution:
    p0: float
    params: FluidParams
    grid_step: float
    plot_max_count: float
    terminal_count: float
    tail_discount_bound: float
    values: list[np.ndarray]
    actions: list[np.ndarray]


def validate_inputs(
    p0: float,
    params: FluidParams,
    grid_step: float,
    plot_max_count: float,
    tail_tolerance: float,
) -> None:
    if not 0.0 < p0 < 1.0:
        raise ValueError("p0 must lie in (0, 1).")
    if not 0.0 < params.p1 < params.p2 <= 1.0:
        raise ValueError("Require 0 < p1 < p2 <= 1.")
    if not 0.0 <= params.c1 < params.c2:
        raise ValueError("Require 0 <= c1 < c2.")
    if not 0.0 < params.gamma < 1.0:
        raise ValueError("gamma must lie in (0, 1).")
    if grid_step <= 0.0 or plot_max_count <= 0.0:
        raise ValueError("grid-step and max-count must be positive.")
    if not 0.0 < tail_tolerance < 1.0:
        raise ValueError("tail-tolerance must lie in (0, 1).")
    plot_steps = plot_max_count / grid_step
    if not np.isclose(plot_steps, round(plot_steps)):
        raise ValueError("max-count must be an integer multiple of grid-step.")


def fluid_demand(
    p0: float,
    successes: np.ndarray | float,
    failures: np.ndarray | float,
) -> np.ndarray:
    """Thompson-sampling demand extended to nonnegative real fluid counts."""
    return np.clip(
        betaincc(
            np.asarray(successes, dtype=float) + 1.0,
            np.asarray(failures, dtype=float) + 1.0,
            p0,
        ),
        0.0,
        1.0,
    )


def solve_fluid_hjb(
    p0: float,
    params: FluidParams,
    grid_step: float = 1.0,
    plot_max_count: float = 80.0,
    tail_tolerance: float = 1e-4,
) -> FluidSolution:
    """Solve the infinite-horizon fluid HJB on the plotted triangular region."""
    validate_inputs(p0, params, grid_step, plot_max_count, tail_tolerance)

    discount_rate = params.discount_rate
    plot_steps = int(round(plot_max_count / grid_step))
    tail_steps = int(
        np.ceil(-np.log(tail_tolerance) / (discount_rate * grid_step))
    )
    terminal_steps = plot_steps + tail_steps
    terminal_count = terminal_steps * grid_step

    # Zero terminal value is harmless on the plotted region up to the reported
    # bound: D <= 1 means reaching the terminal count takes at least the count
    # difference in calendar time.
    next_value = np.zeros(terminal_steps + 1, dtype=float)
    stored_values: list[np.ndarray | None] = [None] * (plot_steps + 1)
    stored_actions: list[np.ndarray | None] = [None] * (plot_steps + 1)

    for layer in range(terminal_steps - 1, -1, -1):
        successes = grid_step * np.arange(layer + 1, dtype=float)
        failures = layer * grid_step - successes
        demand = fluid_demand(p0, successes, failures)

        # One count step takes approximately h / D units of calendar time.
        # The limiting expressions below remain well behaved when D underflows.
        positive_demand = demand > 0.0
        discount_exponent = np.full_like(demand, np.inf)
        with np.errstate(over="ignore"):
            discount_exponent[positive_demand] = (
                discount_rate * grid_step / demand[positive_demand]
            )
        discount = np.exp(-discount_exponent)
        one_minus_discount = -np.expm1(-discount_exponent)
        reward_factor = np.zeros_like(demand)
        reward_factor[positive_demand] = (
            demand[positive_demand]
            * one_minus_discount[positive_demand]
            / discount_rate
        )

        next_failure = next_value[:-1]
        next_success = next_value[1:]
        continuation_1 = (
            (1.0 - params.p1) * next_failure + params.p1 * next_success
        )
        continuation_2 = (
            (1.0 - params.p2) * next_failure + params.p2 * next_success
        )
        value_1 = (
            reward_factor * (params.revenue - params.c1)
            + discount * continuation_1
        )
        value_2 = (
            reward_factor * (params.revenue - params.c2)
            + discount * continuation_2
        )

        action = np.where(value_2 > value_1, 2, 1).astype(np.int8)
        current_value = np.maximum(value_1, value_2)
        if layer <= plot_steps:
            stored_values[layer] = current_value.copy()
            stored_actions[layer] = action.copy()
        next_value = current_value

    values = [value for value in stored_values if value is not None]
    actions = [action for action in stored_actions if action is not None]
    tail_discount_bound = float(
        np.exp(-discount_rate * (terminal_count - plot_max_count))
    )
    return FluidSolution(
        p0=p0,
        params=params,
        grid_step=grid_step,
        plot_max_count=plot_max_count,
        terminal_count=terminal_count,
        tail_discount_bound=tail_discount_bound,
        values=values,
        actions=actions,
    )


def solution_frame(solution: FluidSolution) -> pd.DataFrame:
    """Return the stored HJB value and policy in the original state."""
    rows: list[pd.DataFrame] = []
    r = solution.params.discount_rate
    h = solution.grid_step
    for layer, (value, action) in enumerate(
        zip(solution.values, solution.actions, strict=True)
    ):
        successes = h * np.arange(layer + 1, dtype=float)
        failures = layer * h - successes
        rows.append(
            pd.DataFrame(
                {
                    "s": successes,
                    "f": failures,
                    "posterior_mean": (successes + 1.0)
                    / (successes + failures + 2.0),
                    "demand": fluid_demand(
                        solution.p0,
                        successes,
                        failures,
                    ),
                    "value": value,
                    "annuity_value": r * value,
                    "product": action,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def action_at_state(solution: FluidSolution, s: float, f: float) -> int:
    """Nearest-grid feedback action for a continuous fluid state."""
    h = solution.grid_step
    layer = int(np.clip(np.rint((s + f) / h), 0, len(solution.actions) - 1))
    success_index = int(np.clip(np.rint(s / h), 0, layer))
    return int(solution.actions[layer][success_index])


def trace_optimal_path(
    solution: FluidSolution,
    path_max_count: float = 60.0,
    path_step: float = 0.05,
) -> pd.DataFrame:
    """Trace the deterministic fluid dynamics from the uniform-prior state."""
    if path_max_count > solution.plot_max_count:
        raise ValueError("path-max-count cannot exceed max-count.")
    if path_step <= 0.0:
        raise ValueError("path-step must be positive.")

    params = solution.params
    s = 0.0
    f = 0.0
    calendar_time = 0.0
    rows: list[dict[str, float | int]] = []

    while s + f <= path_max_count + 1e-12:
        action = action_at_state(solution, s, f)
        quality = params.p2 if action == 2 else params.p1
        demand = float(fluid_demand(solution.p0, s, f))
        rows.append(
            {
                "calendar_time": calendar_time,
                "s": s,
                "f": f,
                "s_plus_f": s + f,
                "posterior_mean": (s + 1.0) / (s + f + 2.0),
                "demand": demand,
                "product": action,
                "quality": quality,
            }
        )

        remaining = path_max_count - (s + f)
        if remaining <= 1e-12:
            break
        increment = min(path_step, remaining)
        if demand <= np.finfo(float).tiny:
            break
        calendar_time += increment / demand
        s += quality * increment
        f += (1.0 - quality) * increment

    return pd.DataFrame(rows)


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


def parse_float_grid(value: str) -> np.ndarray:
    values = np.array(
        [float(item.strip()) for item in value.split(",") if item.strip()]
    )
    if len(values) == 0 or np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("p0-grid must contain values strictly between 0 and 1.")
    return np.unique(np.round(values, 10))


def panel_layout(count: int) -> tuple[int, int]:
    columns = min(3, count)
    rows = int(np.ceil(count / columns))
    return rows, columns


def posterior_cutoff_curve(
    p0: float,
    max_count: float,
) -> tuple[np.ndarray, np.ndarray]:
    successes = np.linspace(0.0, max_count, 400)
    failures = ((1.0 - p0) * successes + 1.0 - 2.0 * p0) / p0
    valid = (failures >= 0.0) & (successes + failures <= max_count)
    return successes[valid], failures[valid]


def plot_fluid_policies_by_p0(
    grids: dict[float, pd.DataFrame],
    paths: dict[float, pd.DataFrame],
    solutions: dict[float, FluidSolution],
    output_path: Path,
) -> None:
    """Plot fluid policies and optimal paths in comparable p0 panels."""
    configure_plot_style()
    p0_values = np.array(sorted(solutions))
    rows, columns = panel_layout(len(p0_values))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.7 * columns, 4.35 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    policy_cmap = ListedColormap([PRODUCT1_COLOR, PRODUCT2_COLOR])
    policy_norm = BoundaryNorm([0.5, 1.5, 2.5], policy_cmap.N)

    for ax, p0 in zip(axes, p0_values, strict=False):
        solution = solutions[float(p0)]
        grid = grids[float(p0)]
        path = paths[float(p0)]
        marker_size = max(
            4.0,
            (290.0 * solution.grid_step / solution.plot_max_count) ** 2,
        )
        ax.scatter(
            grid["s"],
            grid["f"],
            c=grid["product"],
            s=marker_size,
            marker="s",
            cmap=policy_cmap,
            norm=policy_norm,
            linewidths=0,
            rasterized=True,
        )
        ax.plot(path["s"], path["f"], color="#111827", linewidth=1.8)
        ax.scatter([0.0], [0.0], color="#111827", s=20, zorder=4)
        cutoff_s, cutoff_f = posterior_cutoff_curve(
            float(p0),
            solution.plot_max_count,
        )
        ax.plot(
            cutoff_s,
            cutoff_f,
            color="#475569",
            linestyle="--",
            linewidth=1.1,
        )
        ax.set_title(f"$p_0={p0:.2f}$", loc="left")
        ax.set_xlabel("Fluid successes, $s$")
        ax.set_ylabel("Fluid failures, $f$")
        ax.set_xlim(-2.0, solution.plot_max_count + 2.0)
        ax.set_ylim(-2.0, solution.plot_max_count + 2.0)
        ax.set_aspect("equal", adjustable="box")

    legend_handles = [
        Patch(facecolor=PRODUCT1_COLOR, label="Product 1"),
        Patch(facecolor=PRODUCT2_COLOR, label="Product 2"),
        plt.Line2D([0], [0], color="#111827", linewidth=2, label="Path"),
        plt.Line2D(
            [0],
            [0],
            color="#475569",
            linestyle="--",
            linewidth=1.2,
            label="Posterior mean $=p_0$",
        ),
    ]
    extra_axes = axes[len(p0_values) :]
    if len(extra_axes) > 0:
        extra_axes[0].set_axis_off()
        extra_axes[0].legend(
            handles=legend_handles,
            loc="center",
            fontsize=11,
        )
        for ax in extra_axes[1:]:
            ax.set_visible(False)
    else:
        fig.legend(handles=legend_handles, loc="lower center", ncols=4)
    gamma = next(iter(solutions.values())).params.gamma
    fig.suptitle(
        f"Discounted fluid policy in the original state, $\\gamma={gamma:.3f}$",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fluid_values_by_p0(
    grids: dict[float, pd.DataFrame],
    paths: dict[float, pd.DataFrame],
    solutions: dict[float, FluidSolution],
    output_path: Path,
) -> None:
    """Plot annuity-equivalent fluid values with one common color scale."""
    configure_plot_style()
    p0_values = np.array(sorted(solutions))
    rows, columns = panel_layout(len(p0_values))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.7 * columns, 4.35 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes.ravel()
    maximum = max(float(grid["annuity_value"].max()) for grid in grids.values())
    norm = Normalize(vmin=0.0, vmax=maximum)
    image = None

    for ax, p0 in zip(axes, p0_values, strict=False):
        solution = solutions[float(p0)]
        grid = grids[float(p0)]
        path = paths[float(p0)]
        marker_size = max(
            4.0,
            (290.0 * solution.grid_step / solution.plot_max_count) ** 2,
        )
        image = ax.scatter(
            grid["s"],
            grid["f"],
            c=grid["annuity_value"],
            s=marker_size,
            marker="s",
            cmap="viridis",
            norm=norm,
            linewidths=0,
            rasterized=True,
        )
        ax.plot(path["s"], path["f"], color="white", linewidth=1.7)
        ax.set_title(f"$p_0={p0:.2f}$", loc="left")
        ax.set_xlabel("Fluid successes, $s$")
        ax.set_ylabel("Fluid failures, $f$")
        ax.set_xlim(-2.0, solution.plot_max_count + 2.0)
        ax.set_ylim(-2.0, solution.plot_max_count + 2.0)
        ax.set_aspect("equal", adjustable="box")

    extra_axes = axes[len(p0_values) :]
    if len(extra_axes) > 0:
        extra_axes[0].set_axis_off()
        colorbar_axis = extra_axes[0].inset_axes([0.18, 0.10, 0.10, 0.80])
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        for ax in extra_axes[1:]:
            ax.set_visible(False)
    else:
        colorbar = fig.colorbar(
            image,
            ax=axes[: len(p0_values)],
            shrink=0.82,
            pad=0.02,
        )
    colorbar.set_label("Annuity-equivalent value $r v(s,f)$")
    gamma = next(iter(solutions.values())).params.gamma
    fig.suptitle(
        f"Discounted fluid value in the original state, $\\gamma={gamma:.3f}$",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fluid_trajectory_panels(
    paths: dict[float, pd.DataFrame],
    gamma: float,
    output_path: Path,
    x_column: str,
    x_label: str,
    title: str,
    share_x: bool,
) -> None:
    """Plot posterior means and product-2 regions along every optimal path."""
    configure_plot_style()
    p0_values = np.array(sorted(paths))
    rows, columns = panel_layout(len(p0_values))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.9 * columns, 3.6 * rows),
        constrained_layout=True,
        squeeze=False,
        sharex=share_x,
        sharey=True,
    )
    axes = axes.ravel()

    for ax, p0 in zip(axes, p0_values, strict=False):
        path = paths[float(p0)]
        ax.fill_between(
            path[x_column],
            0.0,
            1.0,
            where=path["product"].eq(2),
            step="post",
            color=PRODUCT2_COLOR,
            alpha=0.18,
        )
        ax.plot(
            path[x_column],
            path["posterior_mean"],
            color="#2563eb",
            linewidth=2.0,
        )
        ax.axhline(
            p0,
            color="#111827",
            linestyle="--",
            linewidth=1.1,
        )
        ax.set_title(f"$p_0={p0:.2f}$", loc="left")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Posterior mean")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y")
    if share_x:
        maximum_x = max(float(path[x_column].max()) for path in paths.values())
        for ax in axes[: len(p0_values)]:
            ax.set_xlim(0.0, maximum_x)

    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            color="#2563eb",
            linewidth=2,
            label="Posterior mean",
        ),
        plt.Line2D(
            [0],
            [0],
            color="#111827",
            linestyle="--",
            linewidth=1.2,
            label="$p_0$",
        ),
        Patch(
            facecolor=PRODUCT2_COLOR,
            alpha=0.18,
            label="Product 2",
        ),
    ]
    extra_axes = axes[len(p0_values) :]
    if len(extra_axes) > 0:
        extra_axes[0].set_axis_off()
        extra_axes[0].legend(
            handles=legend_handles,
            loc="center",
            fontsize=11,
        )
        for ax in extra_axes[1:]:
            ax.set_visible(False)
    else:
        fig.legend(handles=legend_handles, loc="lower center", ncols=3)
    fig.suptitle(
        f"{title}, $\\gamma={gamma:.3f}$",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fluid_trajectories_by_p0(
    paths: dict[float, pd.DataFrame],
    gamma: float,
    output_path: Path,
) -> None:
    """Plot every fluid trajectory against endogenous calendar time."""
    plot_fluid_trajectory_panels(
        paths=paths,
        gamma=gamma,
        output_path=output_path,
        x_column="calendar_time",
        x_label="Calendar time, $\\tau$",
        title="Optimal discounted fluid trajectories",
        share_x=False,
    )


def plot_fluid_trajectories_by_observations(
    paths: dict[float, pd.DataFrame],
    gamma: float,
    output_path: Path,
) -> None:
    """Plot every fluid trajectory against the common count s+f."""
    plot_fluid_trajectory_panels(
        paths=paths,
        gamma=gamma,
        output_path=output_path,
        x_column="s_plus_f",
        x_label="Fluid observations, $s+f$",
        title="Optimal fluid trajectories by observations",
        share_x=True,
    )


def rolling_observation_average(
    path: pd.DataFrame,
    values: np.ndarray,
    window_observations: float,
) -> np.ndarray:
    """Smooth values over a centered window measured in fluid observations."""
    increments = np.diff(path["s_plus_f"].to_numpy(dtype=float))
    positive = increments[increments > 0.0]
    step = float(np.median(positive)) if len(positive) > 0 else 1.0
    window = max(1, int(round(window_observations / step)))
    return (
        pd.Series(values)
        .rolling(window=window, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )


def plot_fluid_diagnostics_by_observations(
    paths: dict[float, pd.DataFrame],
    params: FluidParams,
    output_path: Path,
    window_observations: float = 5.0,
) -> None:
    """Plot DP-style fluid diagnostics against the common state count s+f."""
    configure_plot_style()
    p0_values = np.array(sorted(paths))
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(p0_values)))
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.2, 8.6),
        constrained_layout=True,
        sharex=True,
    )
    demand_axis, product_axis, profit_axis, belief_axis = axes.ravel()

    for p0, color in zip(p0_values, colors, strict=True):
        path = paths[float(p0)]
        observations = path["s_plus_f"].to_numpy(dtype=float)
        demand = path["demand"].to_numpy(dtype=float)
        product2 = path["product"].eq(2).to_numpy(dtype=float)
        cost = np.where(product2 > 0.5, params.c2, params.c1)
        profit_rate = demand * (params.revenue - cost)
        product2_intensity = rolling_observation_average(
            path,
            product2,
            window_observations,
        )
        smoothed_profit = rolling_observation_average(
            path,
            profit_rate,
            window_observations,
        )

        label = f"$p_0={p0:.2f}$"
        demand_axis.plot(observations, demand, color=color, label=label)
        product_axis.plot(
            observations,
            product2_intensity,
            color=color,
        )
        profit_axis.plot(observations, smoothed_profit, color=color)
        belief_axis.plot(
            observations,
            path["posterior_mean"],
            color=color,
        )

    demand_axis.set_title("Fluid demand path", loc="left")
    product_axis.set_title("Product 2 intensity", loc="left")
    profit_axis.set_title("Expected profit rate", loc="left")
    belief_axis.set_title("User posterior mean for Seller A", loc="left")
    demand_axis.set_ylabel("Demand")
    product_axis.set_ylabel("Product 2 intensity")
    profit_axis.set_ylabel("Expected profit rate")
    belief_axis.set_ylabel("Posterior mean")
    for ax in axes[-1]:
        ax.set_xlabel("Fluid observations, $s+f$")
    demand_axis.set_ylim(-0.03, 1.03)
    product_axis.set_ylim(-0.03, 1.03)
    belief_axis.set_ylim(0.0, 1.0)
    demand_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    product_axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    demand_axis.legend(ncols=3, fontsize=9)
    for ax in axes.ravel():
        ax.grid(True, axis="y")

    fig.suptitle(
        f"Optimal discounted fluid diagnostics, $\\gamma={params.gamma:.3f}$",
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fluid_solution(
    grid: pd.DataFrame,
    path: pd.DataFrame,
    solution: FluidSolution,
    output_path: Path,
) -> None:
    """Plot the optimal policy, value, and trajectory in the original state."""
    configure_plot_style()
    params = solution.params
    marker_size = max(
        4.0,
        (290.0 * solution.grid_step / solution.plot_max_count) ** 2,
    )
    fig = plt.figure(figsize=(11.5, 8.5), constrained_layout=True)
    grid_spec = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.82])
    axes = [
        fig.add_subplot(grid_spec[0, 0]),
        fig.add_subplot(grid_spec[0, 1]),
        fig.add_subplot(grid_spec[1, :]),
    ]

    policy_cmap = ListedColormap([PRODUCT1_COLOR, PRODUCT2_COLOR])
    policy_norm = BoundaryNorm([0.5, 1.5, 2.5], policy_cmap.N)
    axes[0].scatter(
        grid["s"],
        grid["f"],
        c=grid["product"],
        s=marker_size,
        marker="s",
        cmap=policy_cmap,
        norm=policy_norm,
        linewidths=0,
        rasterized=True,
    )
    axes[0].plot(path["s"], path["f"], color="#111827", linewidth=2.0)
    axes[0].scatter(
        [path["s"].iloc[0]],
        [path["f"].iloc[0]],
        color="#111827",
        s=25,
        zorder=4,
    )
    s_line = np.linspace(0.0, solution.plot_max_count, 400)
    f_line = (
        (1.0 - solution.p0) * s_line + 1.0 - 2.0 * solution.p0
    ) / solution.p0
    valid = (
        (f_line >= 0.0)
        & (s_line + f_line <= solution.plot_max_count)
    )
    axes[0].plot(
        s_line[valid],
        f_line[valid],
        color="#475569",
        linestyle="--",
        linewidth=1.3,
    )
    axes[0].set_title("(a) Optimal product and path", loc="left")
    axes[0].set_xlabel("Fluid successes, $s$")
    axes[0].set_ylabel("Fluid failures, $f$")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(
        handles=[
            Patch(facecolor=PRODUCT1_COLOR, label="Product 1"),
            Patch(facecolor=PRODUCT2_COLOR, label="Product 2"),
            plt.Line2D([0], [0], color="#111827", linewidth=2, label="Path"),
            plt.Line2D(
                [0],
                [0],
                color="#475569",
                linestyle="--",
                linewidth=1.3,
                label="Posterior mean $=p_0$",
            ),
        ],
        loc="upper right",
        fontsize=8,
    )

    value_plot = axes[1].scatter(
        grid["s"],
        grid["f"],
        c=grid["annuity_value"],
        s=marker_size,
        marker="s",
        cmap="viridis",
        linewidths=0,
        rasterized=True,
    )
    axes[1].plot(path["s"], path["f"], color="white", linewidth=1.8)
    axes[1].set_title("(b) Annuity-equivalent value $r v(s,f)$", loc="left")
    axes[1].set_xlabel("Fluid successes, $s$")
    axes[1].set_ylabel("Fluid failures, $f$")
    axes[1].set_aspect("equal", adjustable="box")
    fig.colorbar(value_plot, ax=axes[1], shrink=0.82, pad=0.02)

    axes[2].plot(
        path["calendar_time"],
        path["posterior_mean"],
        color="#2563eb",
        linewidth=2.0,
        label="Posterior mean",
    )
    axes[2].axhline(
        solution.p0,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label="$p_0$",
    )
    axes[2].set_xlabel("Calendar time, $\\tau$")
    axes[2].set_ylabel("Posterior mean")
    axes[2].set_ylim(0.0, 1.0)
    axes[2].grid(True, axis="y")
    action_axis = axes[2].twinx()
    action_axis.step(
        path["calendar_time"],
        (path["product"] == 2).astype(float),
        where="post",
        color=PRODUCT2_COLOR,
        alpha=0.55,
        linewidth=1.2,
        label="Product",
    )
    action_axis.set_ylim(-0.08, 1.08)
    action_axis.set_yticks([0.0, 1.0], ["Product 1", "Product 2"])
    action_axis.tick_params(axis="y", colors="#0f766e")
    axes[2].set_title("(c) Optimal trajectory", loc="left")
    handles, labels = axes[2].get_legend_handles_labels()
    axes[2].legend(handles, labels, loc="lower right", fontsize=8)

    fig.suptitle(
        (
            f"Discounted fluid solution: $p_0={solution.p0:.2f}$, "
            f"$\\gamma={params.gamma:.3f}$"
        ),
        x=0.01,
        ha="left",
        fontsize=13,
        fontweight="bold",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    defaults = FluidParams()
    parser = argparse.ArgumentParser(
        description="Solve and plot the discounted fluid HJB in (s, f)."
    )
    parser.add_argument("--p0-grid", default=DEFAULT_P0_GRID)
    parser.add_argument(
        "--p0",
        type=float,
        help="Optional single-p0 override for the three-panel plot.",
    )
    parser.add_argument("--p1", type=float, default=defaults.p1)
    parser.add_argument("--p2", type=float, default=defaults.p2)
    parser.add_argument("--c1", type=float, default=defaults.c1)
    parser.add_argument("--c2", type=float, default=defaults.c2)
    parser.add_argument("--revenue", type=float, default=defaults.revenue)
    parser.add_argument("--gamma", type=float, default=defaults.gamma)
    parser.add_argument("--grid-step", type=float, default=1.0)
    parser.add_argument("--max-count", type=float, default=80.0)
    parser.add_argument("--path-max-count", type=float, default=60.0)
    parser.add_argument("--path-step", type=float, default=0.05)
    parser.add_argument("--tail-tolerance", type=float, default=1e-4)
    parser.add_argument("--outputs-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = FluidParams(
        p1=args.p1,
        p2=args.p2,
        c1=args.c1,
        c2=args.c2,
        revenue=args.revenue,
        gamma=args.gamma,
    )
    project_dir = Path(__file__).resolve().parent
    if args.outputs_dir is not None:
        outputs_dir = args.outputs_dir
    elif np.isclose(args.gamma, 0.999):
        outputs_dir = project_dir / "outputs_gamma_0999"
    else:
        outputs_dir = project_dir / "outputs"

    p0_values = (
        np.array([args.p0], dtype=float)
        if args.p0 is not None
        else parse_float_grid(args.p0_grid)
    )
    solutions: dict[float, FluidSolution] = {}
    grids: dict[float, pd.DataFrame] = {}
    paths: dict[float, pd.DataFrame] = {}

    for p0 in p0_values:
        p0 = float(p0)
        print(
            f"Solving fluid HJB for p0={p0:.3f}, gamma={args.gamma:.4f}, "
            f"grid step={args.grid_step:g}...",
            flush=True,
        )
        solution = solve_fluid_hjb(
            p0=p0,
            params=params,
            grid_step=args.grid_step,
            plot_max_count=args.max_count,
            tail_tolerance=args.tail_tolerance,
        )
        grid = solution_frame(solution)
        path = trace_optimal_path(
            solution,
            path_max_count=args.path_max_count,
            path_step=args.path_step,
        )
        solutions[p0] = solution
        grids[p0] = grid
        paths[p0] = path

        p0_tag = f"{round(100 * p0):03d}"
        grid_path = (
            outputs_dir / "data" / f"fluid_solution_grid_p0_{p0_tag}.csv"
        )
        path_path = (
            outputs_dir / "data" / f"fluid_solution_path_p0_{p0_tag}.csv"
        )
        grid_path.parent.mkdir(parents=True, exist_ok=True)
        grid.to_csv(grid_path, index=False)
        path.to_csv(path_path, index=False)
        print(
            f"  terminal count={solution.terminal_count:g}; "
            f"tail bound={solution.tail_discount_bound:.3e}; "
            f"v(0,0)={solution.values[0][0]:.6f}.",
            flush=True,
        )

    plots_dir = outputs_dir / "plots"
    if len(p0_values) == 1:
        p0 = float(p0_values[0])
        p0_tag = f"{round(100 * p0):03d}"
        plot_path = plots_dir / f"fluid_solution_p0_{p0_tag}.png"
        plot_fluid_solution(grids[p0], paths[p0], solutions[p0], plot_path)
        print(f"Saved plot to: {plot_path}")
        return

    policy_path = plots_dir / "fluid_policy_by_p0.png"
    value_path = plots_dir / "fluid_value_by_p0.png"
    trajectory_path = plots_dir / "fluid_trajectories_by_p0.png"
    observation_path = (
        plots_dir / "fluid_trajectories_by_observations.png"
    )
    diagnostic_path = plots_dir / "fluid_diagnostics_by_observations.png"
    plot_fluid_policies_by_p0(grids, paths, solutions, policy_path)
    plot_fluid_values_by_p0(grids, paths, solutions, value_path)
    plot_fluid_trajectories_by_p0(paths, params.gamma, trajectory_path)
    plot_fluid_trajectories_by_observations(
        paths,
        params.gamma,
        observation_path,
    )
    plot_fluid_diagnostics_by_observations(
        paths,
        params,
        diagnostic_path,
    )
    print(f"Saved policy plot to: {policy_path}")
    print(f"Saved value plot to: {value_path}")
    print(f"Saved trajectory plot to: {trajectory_path}")
    print(f"Saved observation-path plot to: {observation_path}")
    print(f"Saved fluid diagnostic plot to: {diagnostic_path}")


if __name__ == "__main__":
    main()
