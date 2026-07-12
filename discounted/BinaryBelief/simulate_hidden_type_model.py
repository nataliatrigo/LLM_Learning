#!/usr/bin/env python3
"""Simplified hidden-type belief model for hidden product reputation.

This script deliberately uses the scalar belief

    mu = P(Seller A is the high type / product 2 | history)

as the only state.  It does not use the old Beta posterior or the old
(S, F) state.  For each outside option p0 and demand mode, it solves the
seller Bellman equation on a mu-grid, extracts the product policy, simulates
forward paths, and writes plots plus a comparative-statics CSV.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
FIGURE_DIR = BASE_DIR / "figures_hidden_type"
RESULTS_PATH = BASE_DIR / "hidden_type_results.csv"
DIAGNOSTICS_PATH = BASE_DIR / "hidden_type_diagnostics.txt"

DEMAND_MODES = ("mean_threshold", "ts_type")

# The pasted request contains two selected-p0 lists: one in the setup text and
# one in the task list.  The default figures use their union so both are covered.
P0_VALUES_FROM_SETUP = (0.30, 0.50, 0.70, 0.90)
P0_VALUES_FROM_TASKS = (0.20, 0.45, 0.65, 0.90)
P0_VALUES_TO_PLOT = tuple(sorted(set(P0_VALUES_FROM_SETUP + P0_VALUES_FROM_TASKS)))
P0_GRID = np.linspace(0.05, 0.95, 19)

BOUNDARY_TOL = 1e-12
RESULT_COLUMNS = [
    "p0",
    "demand_mode",
    "regime",
    "iterations",
    "bellman_residual",
    "V_mu_init",
    "product2_grid_fraction",
    "num_switches",
    "switch_points",
    "lowest_mu_product2",
    "highest_mu_product2",
    "average_demand_simulated",
    "average_product2_usage_simulated",
    "average_discounted_profit_simulated",
]


@dataclass(frozen=True)
class Params:
    p1: float = 0.35
    p2: float = 0.80
    p0: float = 0.50
    c1: float = 0.05
    c2: float = 0.65
    R: float = 1.0
    gamma: float = 0.98
    mu_init: float = 0.50
    grid_size: int = 1000
    epsilon: float = 1e-5
    max_iter: int = 10000
    tol: float = 1e-9

    def __post_init__(self) -> None:
        if not (0.0 < self.p1 < self.p2 < 1.0):
            raise ValueError("Require 0 < p1 < p2 < 1.")
        if not (0.0 <= self.p0 <= 1.0):
            raise ValueError("Require p0 in [0, 1].")
        if not (self.c1 < self.c2):
            raise ValueError("Require c1 < c2.")
        if not (0.0 < self.gamma < 1.0):
            raise ValueError("Require gamma in (0, 1).")
        if not (0.0 < self.epsilon < 0.5):
            raise ValueError("Require epsilon in (0, 0.5).")
        if not (self.epsilon <= self.mu_init <= 1.0 - self.epsilon):
            raise ValueError("Require mu_init inside the numerical grid.")
        if self.grid_size < 10:
            raise ValueError("grid_size must be at least 10.")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive.")
        if self.tol <= 0.0:
            raise ValueError("tol must be positive.")

    @property
    def dp(self) -> float:
        return self.p2 - self.p1

    @property
    def dc(self) -> float:
        return self.c2 - self.c1

    @property
    def product2_threshold(self) -> float:
        return self.dc / self.dp


@dataclass(frozen=True)
class SolveResult:
    grid: np.ndarray
    V: np.ndarray
    policy: np.ndarray
    continuation_gap: np.ndarray
    demand: np.ndarray
    iterations: int
    bellman_residual: float
    converged: bool


@dataclass(frozen=True)
class PolicyRegion:
    region_type: str
    product2_grid_fraction: float
    num_switches: int
    switch_points: list[float]
    lowest_mu_product2: float
    highest_mu_product2: float
    components: list[tuple[float, float]]


def _validate_mode(mode: str) -> None:
    if mode not in DEMAND_MODES:
        raise ValueError(f"Unknown demand mode {mode!r}; use one of {DEMAND_MODES}.")


def _maybe_scalar(values: np.ndarray, scalar_input: bool):
    return float(values) if scalar_input else values


def pbar(mu, params: Params):
    """User's perceived success probability for Seller A at belief mu."""
    scalar_input = np.isscalar(mu)
    mu_arr = np.asarray(mu, dtype=float)
    out = params.p1 + mu_arr * params.dp
    return _maybe_scalar(out, scalar_input)


def mu_threshold(params: Params) -> float:
    """Belief where the perceived quality equals the outside option p0."""
    return (params.p0 - params.p1) / params.dp


def mu_success(mu, params: Params):
    """Bayes update after an observed success from Seller A."""
    scalar_input = np.isscalar(mu)
    mu_arr = np.asarray(mu, dtype=float)
    denominator = mu_arr * params.p2 + (1.0 - mu_arr) * params.p1
    out = np.divide(
        mu_arr * params.p2,
        denominator,
        out=np.zeros_like(mu_arr, dtype=float),
        where=denominator > 0.0,
    )
    return _maybe_scalar(out, scalar_input)


def mu_failure(mu, params: Params):
    """Bayes update after an observed failure from Seller A."""
    scalar_input = np.isscalar(mu)
    mu_arr = np.asarray(mu, dtype=float)
    denominator = mu_arr * (1.0 - params.p2) + (1.0 - mu_arr) * (1.0 - params.p1)
    out = np.divide(
        mu_arr * (1.0 - params.p2),
        denominator,
        out=np.zeros_like(mu_arr, dtype=float),
        where=denominator > 0.0,
    )
    return _maybe_scalar(out, scalar_input)


def demand(mu, params: Params, mode: str):
    """Expected user demand for Seller A under the selected demand mode.

    The boundary regimes are explicit.  If the outside option is no better than
    product 1, Seller A is always chosen.  If it is strictly better than product
    2, Seller A is never chosen.  Only the interior regime depends on mu.
    """
    _validate_mode(mode)
    scalar_input = np.isscalar(mu)
    mu_arr = np.asarray(mu, dtype=float)

    if params.p0 <= params.p1 + BOUNDARY_TOL:
        out = np.ones_like(mu_arr, dtype=float)
    elif params.p0 > params.p2 + BOUNDARY_TOL:
        out = np.zeros_like(mu_arr, dtype=float)
    elif mode == "mean_threshold":
        out = (mu_arr >= mu_threshold(params)).astype(float)
    else:
        out = np.clip(mu_arr, 0.0, 1.0)

    return _maybe_scalar(out, scalar_input)


def regime_label(params: Params) -> str:
    if params.p0 < params.p1 - BOUNDARY_TOL:
        return "below_p1"
    if params.p0 > params.p2 + BOUNDARY_TOL:
        return "above_p2"
    return "between_p1_p2"


def make_grid(params: Params) -> np.ndarray:
    return np.linspace(params.epsilon, 1.0 - params.epsilon, params.grid_size)


def _interp_on_grid(grid: np.ndarray, values: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.clip(points, grid[0], grid[-1])
    return np.interp(points, grid, values)


def _bellman_update_precomputed(
    V: np.ndarray,
    grid: np.ndarray,
    params: Params,
    D: np.ndarray,
    mu_s: np.ndarray,
    mu_f: np.ndarray,
) -> np.ndarray:
    """One Bellman update on the original equation.

    With probability 1-D(mu), no user arrives and the belief stays fixed.  With
    probability D(mu), the seller earns current profit and reputation moves by
    the success/failure Bayes update induced by the product actually chosen.
    """
    VS = _interp_on_grid(grid, V, mu_s)
    VF = _interp_on_grid(grid, V, mu_f)

    product1_value = (
        params.R
        - params.c1
        + params.gamma * (params.p1 * VS + (1.0 - params.p1) * VF)
    )
    product2_value = (
        params.R
        - params.c2
        + params.gamma * (params.p2 * VS + (1.0 - params.p2) * VF)
    )
    engaged_value = np.maximum(product1_value, product2_value)
    return params.gamma * (1.0 - D) * V + D * engaged_value


def extract_policy(grid: np.ndarray, V: np.ndarray, params: Params, mode: str):
    """Extract policy, continuation gap, and demand from a solved value function.

    The product-2 decision compares the reputation value of a success versus a
    failure to the cost-per-success threshold dc/dp.
    """
    _validate_mode(mode)
    mu_s = np.asarray(mu_success(grid, params), dtype=float)
    mu_f = np.asarray(mu_failure(grid, params), dtype=float)
    VS = _interp_on_grid(grid, V, mu_s)
    VF = _interp_on_grid(grid, V, mu_f)
    continuation_gap = params.gamma * (VS - VF)
    policy = np.where(continuation_gap >= params.product2_threshold, 2, 1)
    D = np.asarray(demand(grid, params, mode), dtype=float)
    return policy.astype(int), continuation_gap, D


def solve_value_iteration(params: Params, mode: str) -> SolveResult:
    """Solve the Bellman equation by value iteration on a uniform mu-grid."""
    _validate_mode(mode)
    grid = make_grid(params)
    D = np.asarray(demand(grid, params, mode), dtype=float)
    mu_s = np.asarray(mu_success(grid, params), dtype=float)
    mu_f = np.asarray(mu_failure(grid, params), dtype=float)

    V = np.zeros_like(grid)
    bellman_residual = np.inf
    iterations = 0
    for iterations in range(1, params.max_iter + 1):
        V_next = _bellman_update_precomputed(V, grid, params, D, mu_s, mu_f)
        bellman_residual = float(np.max(np.abs(V_next - V)))
        V = V_next
        if bellman_residual < params.tol:
            break

    final_update = _bellman_update_precomputed(V, grid, params, D, mu_s, mu_f)
    final_residual = float(np.max(np.abs(final_update - V)))
    policy, continuation_gap, D = extract_policy(grid, V, params, mode)
    return SolveResult(
        grid=grid,
        V=V,
        policy=policy,
        continuation_gap=continuation_gap,
        demand=D,
        iterations=iterations,
        bellman_residual=final_residual,
        converged=final_residual < params.tol,
    )


def classify_policy_region(grid: np.ndarray, policy: np.ndarray) -> PolicyRegion:
    """Summarize where product 2 is optimal on the belief grid."""
    policy = np.asarray(policy, dtype=int)
    uses_product2 = policy == 2
    switch_idx = np.flatnonzero(policy[1:] != policy[:-1])
    switch_points = [float(0.5 * (grid[i] + grid[i + 1])) for i in switch_idx]
    fraction = float(np.mean(uses_product2))

    if not np.any(uses_product2):
        return PolicyRegion(
            region_type="empty",
            product2_grid_fraction=fraction,
            num_switches=len(switch_points),
            switch_points=switch_points,
            lowest_mu_product2=np.nan,
            highest_mu_product2=np.nan,
            components=[],
        )

    idx = np.flatnonzero(uses_product2)
    starts = [int(idx[0])]
    ends: list[int] = []
    for left, right in zip(idx[:-1], idx[1:]):
        if right > left + 1:
            ends.append(int(left))
            starts.append(int(right))
    ends.append(int(idx[-1]))
    components = [(float(grid[start]), float(grid[end])) for start, end in zip(starts, ends)]

    if len(components) > 1:
        region_type = "disconnected"
    elif len(switch_points) == 0:
        region_type = "interval"
    elif len(switch_points) == 1:
        region_type = "threshold"
    elif len(switch_points) == 2:
        region_type = "band"
    else:
        region_type = "interval"

    return PolicyRegion(
        region_type=region_type,
        product2_grid_fraction=fraction,
        num_switches=len(switch_points),
        switch_points=switch_points,
        lowest_mu_product2=float(grid[idx[0]]),
        highest_mu_product2=float(grid[idx[-1]]),
        components=components,
    )


def _format_switch_points(points: list[float]) -> str:
    if not points:
        return ""
    return ";".join(f"{point:.6f}" for point in points)


def _policy_at_mu(mu: float, result: SolveResult, params: Params) -> int:
    gap = float(np.interp(mu, result.grid, result.continuation_gap))
    return 2 if gap >= params.product2_threshold else 1


def _user_chooses_A(
    mu: float,
    params: Params,
    mode: str,
    rng: np.random.Generator,
    simulate_user_randomness: bool,
) -> bool:
    D = float(demand(mu, params, mode))
    if mode == "mean_threshold":
        return D >= 1.0 - BOUNDARY_TOL

    if simulate_user_randomness:
        if params.p0 <= params.p1 + BOUNDARY_TOL:
            return True
        if params.p0 > params.p2 + BOUNDARY_TOL:
            return False
        sampled_high_type = rng.random() < mu
        sampled_success_probability = params.p2 if sampled_high_type else params.p1
        return sampled_success_probability >= params.p0 - BOUNDARY_TOL

    return rng.random() < D


def simulate_path(
    params: Params,
    mode: str,
    T: int,
    seed: int | None = None,
    solve_result: SolveResult | None = None,
    simulate_user_randomness: bool = True,
) -> pd.DataFrame:
    """Simulate one forward path under the optimal policy.

    If a user chooses Seller A, the seller uses the policy from the solved
    Bellman equation, pays the chosen product's cost, generates a Bernoulli
    outcome, and the user's hidden-type belief is updated by Bayes rule.
    Periods with no demand leave the belief unchanged and generate zero profit.
    """
    _validate_mode(mode)
    if T <= 0:
        raise ValueError("T must be positive.")
    result = solve_value_iteration(params, mode) if solve_result is None else solve_result
    rng = np.random.default_rng(seed)
    mu = float(np.clip(params.mu_init, params.epsilon, 1.0 - params.epsilon))
    records: list[dict[str, float | int]] = []
    cumulative_discounted_profit = 0.0

    for t in range(T):
        mu_before = mu
        D = float(demand(mu_before, params, mode))
        user_chose_A = _user_chooses_A(
            mu_before, params, mode, rng, simulate_user_randomness
        )

        chosen_product = 0
        outcome = np.nan
        profit = 0.0
        if user_chose_A:
            chosen_product = _policy_at_mu(mu_before, result, params)
            success_probability = params.p2 if chosen_product == 2 else params.p1
            outcome = int(rng.random() < success_probability)
            product_cost = params.c2 if chosen_product == 2 else params.c1
            profit = params.R - product_cost
            mu = (
                float(mu_success(mu_before, params))
                if outcome == 1
                else float(mu_failure(mu_before, params))
            )
            mu = float(np.clip(mu, params.epsilon, 1.0 - params.epsilon))

        discounted_profit = (params.gamma**t) * profit
        cumulative_discounted_profit += discounted_profit
        records.append(
            {
                "t": t,
                "mu": mu_before,
                "demand": D,
                "user_chose_A": int(user_chose_A),
                "chosen_product": chosen_product,
                "outcome": outcome,
                "profit": profit,
                "discounted_profit": discounted_profit,
                "cumulative_discounted_profit": cumulative_discounted_profit,
                "mu_next": mu,
            }
        )

    return pd.DataFrame.from_records(records)


def run_p0_sweep(
    base_params: Params,
    p0_grid,
    mode: str,
    simulation_T: int = 500,
    simulation_paths: int = 50,
    seed: int = 12345,
    simulate_user_randomness: bool = True,
) -> pd.DataFrame:
    """Solve and simulate the model over a p0-grid for one demand mode."""
    _validate_mode(mode)
    rows: list[dict[str, float | int | str]] = []
    diagnostics: list[dict[str, object]] = []

    for p0_index, p0 in enumerate(p0_grid):
        params = replace(base_params, p0=float(p0))
        result = solve_value_iteration(params, mode)
        region = classify_policy_region(result.grid, result.policy)
        V_mu_init = float(np.interp(params.mu_init, result.grid, result.V))

        average_demands: list[float] = []
        average_product2_usage: list[float] = []
        discounted_profits: list[float] = []
        for path_index in range(simulation_paths):
            path_seed = seed + 100_000 * p0_index + path_index
            path = simulate_path(
                params,
                mode,
                simulation_T,
                seed=path_seed,
                solve_result=result,
                simulate_user_randomness=simulate_user_randomness,
            )
            average_demands.append(float(path["user_chose_A"].mean()))
            average_product2_usage.append(float((path["chosen_product"] == 2).mean()))
            discounted_profits.append(float(path["discounted_profit"].sum()))

        row = {
            "p0": float(p0),
            "demand_mode": mode,
            "regime": regime_label(params),
            "iterations": result.iterations,
            "bellman_residual": result.bellman_residual,
            "V_mu_init": V_mu_init,
            "product2_grid_fraction": region.product2_grid_fraction,
            "num_switches": region.num_switches,
            "switch_points": _format_switch_points(region.switch_points),
            "lowest_mu_product2": region.lowest_mu_product2,
            "highest_mu_product2": region.highest_mu_product2,
            "average_demand_simulated": float(np.mean(average_demands)),
            "average_product2_usage_simulated": float(np.mean(average_product2_usage)),
            "average_discounted_profit_simulated": float(np.mean(discounted_profits)),
        }
        rows.append(row)
        diagnostics.append(
            {
                "p0": float(p0),
                "demand_mode": mode,
                "regime": regime_label(params),
                "iterations": result.iterations,
                "bellman_residual": result.bellman_residual,
                "converged": result.converged,
                "policy_region_type": region.region_type,
                "switch_points": region.switch_points,
                "components": region.components,
            }
        )

    df = pd.DataFrame(rows, columns=RESULT_COLUMNS)
    df.attrs["diagnostics"] = diagnostics
    return df


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#b8b8b8",
            "axes.grid": True,
            "grid.color": "#e5e5e5",
            "grid.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.frameon": False,
            "figure.dpi": 110,
            "savefig.dpi": 180,
            "savefig.bbox": "tight",
        }
    )


def _p0_tag(p0: float) -> str:
    return f"{p0:.2f}".replace(".", "p")


def _mode_color(mode: str) -> str:
    return {"mean_threshold": "#1f77b4", "ts_type": "#2ca02c"}[mode]


def _mark_mu_threshold(ax, params: Params) -> None:
    threshold = mu_threshold(params)
    if 0.0 <= threshold <= 1.0:
        ax.axvline(
            threshold,
            color="#6f6f6f",
            linestyle=":",
            linewidth=1.2,
            label=f"mu0={threshold:.3f}",
        )


def plot_solution_panel(
    params: Params,
    mode: str,
    result: SolveResult,
    outdir: Path,
) -> Path:
    """Four-panel plot for V, policy, continuation gap, and demand."""
    color = _mode_color(mode)
    region = classify_policy_region(result.grid, result.policy)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    fig.suptitle(
        f"{mode}, p0={params.p0:.2f}, regime={regime_label(params)}, "
        f"product-2 region={region.region_type}"
    )

    ax = axes[0, 0]
    ax.plot(result.grid, result.V, color=color)
    _mark_mu_threshold(ax, params)
    ax.set_title("Seller value")
    ax.set_xlabel("belief mu")
    ax.set_ylabel("V(mu)")
    ax.set_xlim(0.0, 1.0)

    ax = axes[0, 1]
    ax.step(result.grid, result.policy, where="mid", color=color)
    _mark_mu_threshold(ax, params)
    ax.set_title("Optimal product")
    ax.set_xlabel("belief mu")
    ax.set_ylabel("x*(mu)")
    ax.set_yticks([1, 2], ["1 (low cost)", "2 (high quality)"])
    ax.set_ylim(0.8, 2.2)
    ax.set_xlim(0.0, 1.0)

    ax = axes[1, 0]
    ax.plot(result.grid, result.continuation_gap, color="#7a3db8", label="G(mu)")
    ax.axhline(
        params.product2_threshold,
        color="#4d4d4d",
        linestyle="--",
        linewidth=1.2,
        label=f"dc/dp={params.product2_threshold:.3f}",
    )
    _mark_mu_threshold(ax, params)
    ax.set_title("Continuation gap")
    ax.set_xlabel("belief mu")
    ax.set_ylabel("gamma [V(muS) - V(muF)]")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="best")

    ax = axes[1, 1]
    if mode == "mean_threshold":
        ax.step(result.grid, result.demand, where="mid", color=color)
    else:
        ax.plot(result.grid, result.demand, color=color)
    _mark_mu_threshold(ax, params)
    ax.set_title("Demand")
    ax.set_xlabel("belief mu")
    ax.set_ylabel("D(mu; p0)")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(0.0, 1.0)

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"solution_{mode}_p0_{_p0_tag(params.p0)}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_comparative_statics(
    results: pd.DataFrame,
    base_params: Params,
    outdir: Path,
) -> Path:
    metrics = [
        ("product2_grid_fraction", "fraction grid product 2"),
        ("lowest_mu_product2", "lowest mu product 2"),
        ("highest_mu_product2", "highest mu product 2"),
        ("num_switches", "number of switches"),
        ("V_mu_init", "V(mu_init)"),
        ("average_demand_simulated", "average demand"),
        ("average_product2_usage_simulated", "average product-2 usage"),
        ("average_discounted_profit_simulated", "average discounted profit"),
    ]

    fig, axes = plt.subplots(4, 2, figsize=(12, 12), constrained_layout=True)
    axes = axes.ravel()
    for ax, (column, label) in zip(axes, metrics):
        for mode in DEMAND_MODES:
            sub = results.loc[results["demand_mode"] == mode].sort_values("p0")
            ax.plot(
                sub["p0"],
                sub[column],
                marker="o",
                linewidth=1.8,
                markersize=4,
                color=_mode_color(mode),
                label=mode,
            )
        ax.axvline(base_params.p1, color="#666666", linestyle=":", linewidth=1.0)
        ax.axvline(base_params.p2, color="#666666", linestyle=":", linewidth=1.0)
        ax.set_xlabel("outside option p0")
        ax.set_ylabel(label)
        ax.set_title(label)
    axes[0].legend(loc="best")
    fig.suptitle("Comparative statics across outside-option regimes")

    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "comparative_statics.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_sample_path(
    params: Params,
    mode: str,
    result: SolveResult,
    outdir: Path,
    T: int,
    seed: int,
    simulate_user_randomness: bool,
) -> Path:
    path_df = simulate_path(
        params,
        mode,
        T,
        seed=seed,
        solve_result=result,
        simulate_user_randomness=simulate_user_randomness,
    )
    t = path_df["t"].to_numpy()
    color = _mode_color(mode)

    fig, axes = plt.subplots(5, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
    fig.suptitle(f"Sample path: {mode}, p0={params.p0:.2f}, regime={regime_label(params)}")

    axes[0].plot(t, path_df["mu"], color=color)
    axes[0].set_ylabel("mu_t")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("Belief")

    axes[1].step(t, path_df["chosen_product"], where="post", color=color)
    axes[1].set_yticks([0, 1, 2], ["none", "1", "2"])
    axes[1].set_ylim(-0.2, 2.2)
    axes[1].set_ylabel("product")
    axes[1].set_title("Chosen product")

    axes[2].plot(t, path_df["demand"], color=color, label="expected demand")
    axes[2].step(
        t,
        path_df["user_chose_A"],
        where="post",
        color="#444444",
        linestyle="--",
        linewidth=1.0,
        label="realized choice",
    )
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("demand")
    axes[2].set_title("Demand")
    axes[2].legend(loc="best")

    observed = path_df["outcome"].notna()
    axes[3].scatter(
        path_df.loc[observed, "t"],
        path_df.loc[observed, "outcome"],
        s=14,
        color=color,
    )
    axes[3].set_yticks([0, 1], ["failure", "success"])
    axes[3].set_ylim(-0.2, 1.2)
    axes[3].set_ylabel("outcome")
    axes[3].set_title("Realized outcomes when Seller A is chosen")

    axes[4].plot(t, path_df["cumulative_discounted_profit"], color=color)
    axes[4].set_ylabel("discounted profit")
    axes[4].set_xlabel("period t")
    axes[4].set_title("Cumulative discounted profit")

    outdir.mkdir(parents=True, exist_ok=True)
    file_path = outdir / f"sample_path_{mode}_p0_{_p0_tag(params.p0)}.png"
    fig.savefig(file_path)
    plt.close(fig)
    return file_path


def _trend_text(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=float)
    diffs = np.diff(values)
    tol = 1e-8
    if np.all(diffs >= -tol):
        return "weakly expands"
    if np.all(diffs <= tol):
        return "weakly shrinks"
    if values[-1] > values[0] + tol:
        return "expands overall but is nonmonotone"
    if values[-1] < values[0] - tol:
        return "shrinks overall but is nonmonotone"
    return "is nonmonotone with little net change"


def write_diagnostics(
    results: pd.DataFrame,
    diagnostics: list[dict[str, object]],
    path: Path,
) -> None:
    lines: list[str] = []
    lines.append("Hidden-type scalar-belief model diagnostics")
    lines.append("")
    lines.append("Per-p0 policy diagnostics")
    for item in diagnostics:
        switches = item["switch_points"]
        if switches:
            switch_text = "; ".join(f"{float(point):.6f}" for point in switches)
        else:
            switch_text = "none"
        components = item["components"]
        if components:
            component_text = "; ".join(
                f"[{float(lo):.6f}, {float(hi):.6f}]" for lo, hi in components
            )
        else:
            component_text = "none"
        lines.append(
            "  "
            f"mode={item['demand_mode']}, p0={float(item['p0']):.2f}, "
            f"regime={item['regime']}, converged={item['converged']}, "
            f"iterations={int(item['iterations'])}, "
            f"residual={float(item['bellman_residual']):.3e}, "
            f"product2_region={item['policy_region_type']}, "
            f"switches={switch_text}, components={component_text}"
        )

    lines.append("")
    lines.append("Product-2 region trend as p0 increases")
    for mode in DEMAND_MODES:
        sub = results.loc[results["demand_mode"] == mode].sort_values("p0")
        trend = _trend_text(sub["product2_grid_fraction"].to_numpy())
        lines.append(f"  {mode}: product-2 grid fraction {trend}.")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve and simulate the simplified hidden-type belief model."
    )
    parser.add_argument(
        "--simulation-T",
        type=int,
        default=500,
        help="Periods per path used in comparative-statics simulation summaries.",
    )
    parser.add_argument(
        "--simulation-paths",
        type=int,
        default=50,
        help="Number of paths averaged for each p0 in the sweep.",
    )
    parser.add_argument(
        "--sample-T",
        type=int,
        default=200,
        help="Periods shown in each sample-path figure.",
    )
    parser.add_argument(
        "--expected-ts-demand",
        action="store_true",
        help="Use expected TS demand as an engagement probability instead of simulating the TS type draw.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only write the CSV and diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.simulation_T <= 0:
        raise ValueError("--simulation-T must be positive.")
    if args.simulation_paths <= 0:
        raise ValueError("--simulation-paths must be positive.")
    if args.sample_T <= 0:
        raise ValueError("--sample-T must be positive.")

    setup_style()
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    base_params = Params()
    simulate_user_randomness = not args.expected_ts_demand

    frames: list[pd.DataFrame] = []
    diagnostics: list[dict[str, object]] = []
    for mode in DEMAND_MODES:
        print(f"Solving p0 sweep for {mode}...")
        df = run_p0_sweep(
            base_params,
            P0_GRID,
            mode,
            simulation_T=args.simulation_T,
            simulation_paths=args.simulation_paths,
            simulate_user_randomness=simulate_user_randomness,
        )
        diagnostics.extend(df.attrs.get("diagnostics", []))
        frames.append(df)

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(RESULTS_PATH, index=False)
    write_diagnostics(results, diagnostics, DIAGNOSTICS_PATH)
    print(f"Wrote {RESULTS_PATH.relative_to(BASE_DIR)}")
    print(f"Wrote {DIAGNOSTICS_PATH.relative_to(BASE_DIR)}")

    if not args.skip_plots:
        for mode in DEMAND_MODES:
            for p0_index, p0 in enumerate(P0_VALUES_TO_PLOT):
                params = replace(base_params, p0=float(p0))
                result = solve_value_iteration(params, mode)
                solution_path = plot_solution_panel(params, mode, result, FIGURE_DIR)
                sample_path = plot_sample_path(
                    params,
                    mode,
                    result,
                    FIGURE_DIR,
                    T=args.sample_T,
                    seed=10_000 + 1_000 * p0_index,
                    simulate_user_randomness=simulate_user_randomness,
                )
                print(f"Wrote {solution_path.relative_to(BASE_DIR)}")
                print(f"Wrote {sample_path.relative_to(BASE_DIR)}")
        comp_path = plot_comparative_statics(results, base_params, FIGURE_DIR)
        print(f"Wrote {comp_path.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
