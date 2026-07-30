"""Diagnostic figures built only from saved, unsmoothed experiment data."""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/modular_dp_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter


COLORS = ["#3b1b73", "#355c9a", "#148a8a", "#39b977", "#a8df13", "#d97706", "#be123c"]

METHOD_LABELS = {
    "standard_ts": "Standard TS",
    "scaled_updates": "Scaled updates",
    "mean_preserving_temperature": "Mean-preserving temperature",
    "epsilon_greedy": "Epsilon-greedy",
    "observation_forgetting_ts": "Observation-time forgetting",
    "calendar_forgetting_ts": "Calendar-time forgetting",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _label(row: pd.Series) -> str:
    method = str(row["method"])
    name = METHOD_LABELS.get(method, method.replace("_", " "))
    parameter = row.get("parameter_name", "")
    value = row.get("parameter_value", np.nan)
    if parameter and pd.notna(value):
        symbol = {
            "eta": r"$\eta$",
            "temperature": r"$T$",
            "epsilon": r"$\varepsilon$",
            "rho": r"$\rho$",
        }.get(str(parameter), str(parameter))
        # The surrounding figure title already names the method; compact
        # parameter-only panel titles avoid collisions in multi-column grids.
        if method in {
            "scaled_updates",
            "mean_preserving_temperature",
            "epsilon_greedy",
        }:
            return f"{symbol}={float(value):g}"
        return f"{name}: {symbol}={float(value):g}"
    return name


def _sensitivity_label(row: pd.Series, slope: float | None = None) -> str:
    method = str(row["method"])
    parameter = row.get("parameter_name", "")
    value = row.get("parameter_value", np.nan)
    if method == "standard_ts":
        result = "Standard TS"
    elif parameter and pd.notna(value):
        symbol = {"eta": r"$\eta$", "temperature": r"$T$", "epsilon": r"$\varepsilon$"}.get(str(parameter), str(parameter))
        result = f"{symbol}={float(value):g}"
    else:
        result = METHOD_LABELS.get(method, method)
    if slope is not None:
        result += f"; slope={slope:.3f}" if np.isfinite(slope) else "; slope=NA"
    return result


def plot_demand_sensitivity(
    sensitivity: pd.DataFrame,
    regressions: pd.DataFrame,
    figures: Path,
) -> None:
    """Figures 1--2: raw and log-log maximal local demand sensitivity."""
    if sensitivity.empty:
        return
    configure_style()
    keys = ["method", "parameter_name", "parameter_value"]
    panel_specs = [
        (["standard_ts", "scaled_updates"], "Standard TS and scaled updates"),
        (["standard_ts", "mean_preserving_temperature"], "Mean-preserving temperature"),
        (["epsilon_greedy"], "Uniform epsilon-greedy"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), sharex=True)
    for ax, (methods, title) in zip(axes, panel_specs, strict=True):
        panel = sensitivity[sensitivity.method.isin(methods)]
        groups = list(panel.groupby(keys, dropna=False, sort=False))
        for color, (_, data) in zip(COLORS, groups, strict=False):
            data = data.sort_values("n")
            if str(data.method.iloc[0]) == "epsilon_greedy":
                ax.plot(
                    data.n,
                    data.g_n,
                    linestyle="none",
                    marker=".",
                    markersize=1.8,
                    color=color,
                    alpha=0.55,
                    label=_sensitivity_label(data.iloc[0]),
                )
            else:
                ax.plot(data.n, data.g_n, lw=1.15, color=color, alpha=0.9, label=_sensitivity_label(data.iloc[0]))
        ax.set(xlabel=r"Observation count $n$", title=title)
        ax.legend(fontsize=7.5)
    axes[0].set_ylabel(r"$g_n=\max_S q(S,n-S)$")
    fig.suptitle("Maximum one-observation demand sensitivity", x=0.01, ha="left")
    _save(fig, figures / "demand" / "01_demand_sensitivity")

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.7), sharex=True)
    for ax, (methods, title) in zip(axes, panel_specs, strict=True):
        panel = sensitivity[sensitivity.method.isin(methods)]
        groups = list(panel.groupby(keys, dropna=False, sort=False))
        for color, (key, data) in zip(COLORS, groups, strict=False):
            positive = data[(data.n > 0) & (data.g_n > 0)].sort_values("n")
            if positive.empty:
                continue
            match = regressions
            for column, value in zip(keys, key, strict=True):
                match = match[
                    match[column].fillna("__nan__")
                    == ("__nan__" if pd.isna(value) else value)
                ]
            slope = match.estimated_sensitivity_slope.iloc[0] if len(match) else np.nan
            # Markers reveal epsilon-greedy's even/odd nonvanishing sequence
            # instead of visually filling the panel with a dense zigzag.
            if str(data.method.iloc[0]) == "epsilon_greedy":
                ax.loglog(
                    positive.n,
                    positive.g_n,
                    linestyle="none",
                    marker=".",
                    markersize=1.8,
                    color=color,
                    alpha=0.55,
                    label=_sensitivity_label(data.iloc[0], slope),
                )
            else:
                ax.loglog(positive.n, positive.g_n, lw=1.15, color=color, alpha=0.9, label=_sensitivity_label(data.iloc[0], slope))
        ax.set(xlabel=r"Observation count $n$", title=title)
        ax.legend(fontsize=7.2)
        if methods == ["epsilon_greedy"] and np.any(
            np.isclose(
                pd.to_numeric(panel.parameter_value, errors="coerce"),
                1.0,
                equal_nan=False,
            )
        ):
            ax.text(
                0.98,
                0.04,
                r"$\varepsilon=1$: $g_n=0$ (not on log scale)",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=7.5,
                color="#475569",
            )
    axes[0].set_ylabel(r"$g_n$")
    fig.suptitle("Demand sensitivity on log-log axes", x=0.01, ha="left")
    _save(fig, figures / "demand" / "02_demand_sensitivity_loglog")


def _policy_panel(data: pd.DataFrame, title: str, stem: Path) -> None:
    if data.empty:
        return
    groups = list(data.groupby(["method", "parameter_name", "parameter_value"], dropna=False, sort=False))
    ncols = min(3, len(groups))
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.3 * nrows), squeeze=False, sharex=True, sharey=True)
    for ax, (_, group) in zip(axes.ravel(), groups, strict=False):
        valid = group[group.n.notna() & group.posterior_mean.notna()]
        if "robust_action2" in valid:
            valid = valid[valid.robust_action2.fillna(False).astype(bool)]
        ax.scatter(valid.n, valid.posterior_mean, s=2.0, color="#0f766e", alpha=0.75, rasterized=True)
        ax.axhline(float(group.p0.iloc[0]), color="#475569", lw=0.8, ls="--")
        ax.set_title(_label(group.iloc[0]), loc="left")
        ax.set(xlabel="n", ylabel="posterior mean", ylim=(0, 1))
        if valid.empty:
            ax.text(
                0.5,
                0.5,
                "No product-2 states",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color="#475569",
            )
    for ax in axes.ravel()[len(groups):]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left")
    _save(fig, stem)


def plot_discrete_policy_regions(
    policy_points: pd.DataFrame,
    figures: Path,
    summary: pd.DataFrame | None = None,
) -> None:
    """Figures 3--5. Only product-2 points are shown; no state-share plot."""
    if policy_points.empty:
        return
    if summary is not None and not summary.empty:
        discrete = summary[
            summary.solver_type == "discrete_triangular_truncation"
        ]
        present = set(policy_points.method_key.unique())
        placeholders = []
        for row in discrete.itertuples(index=False):
            if row.method_key in present:
                continue
            placeholders.append(
                {
                    "method": row.method,
                    "parameter_name": row.parameter_name,
                    "parameter_value": row.parameter_value,
                    "method_key": row.method_key,
                    "p0": row.p0,
                    "n": np.nan,
                    "posterior_mean": np.nan,
                }
            )
        if placeholders:
            policy_points = pd.concat(
                [policy_points, pd.DataFrame(placeholders)], ignore_index=True
            )
    standard_scaled = policy_points[policy_points.method.isin(["standard_ts", "scaled_updates"])]
    _policy_panel(standard_scaled, "Product-2 policy regions: standard TS and scaled updates", figures / "policy" / "03_policy_standard_scaled")
    _policy_panel(
        policy_points[policy_points.method == "mean_preserving_temperature"],
        "Product-2 policy regions: mean-preserving temperature",
        figures / "policy" / "04_policy_temperature",
    )
    _policy_panel(
        policy_points[policy_points.method == "epsilon_greedy"],
        "Product-2 policy regions: uniform epsilon-greedy",
        figures / "policy" / "05_policy_epsilon_greedy",
    )


def plot_last_active(summary: pd.DataFrame, figures: Path) -> None:
    """Figures 6--7: empirical extinction location for the two TS variants."""
    specs = [
        ("scaled_updates", "eta", "06_last_active_vs_eta", r"Scaled update $\eta$"),
        ("mean_preserving_temperature", "temperature", "07_last_active_vs_temperature", r"Temperature $T$"),
    ]
    for method, parameter, stem, xlabel in specs:
        data = summary[summary.method == method].copy()
        if "simulation_policy_last_active_diagonal" in data:
            extended = pd.to_numeric(
                data["simulation_policy_last_active_diagonal"], errors="coerce"
            )
            data["plotted_last_active"] = extended.fillna(data.last_active_diagonal)
            if "simulation_policy_last_active_censored" in data:
                data["plotted_censored"] = data[
                    "simulation_policy_last_active_censored"
                ].fillna(data.last_active_censored)
            else:
                data["plotted_censored"] = data.last_active_censored
        else:
            data["plotted_last_active"] = data.last_active_diagonal
            data["plotted_censored"] = data.last_active_censored
        data = data[data.plotted_last_active.notna()].sort_values("parameter_value")
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(6.5, 4.1))
        ax.plot(data.parameter_value, data.plotted_last_active, marker="o", lw=1.6, color="#2563eb")
        censored = data.plotted_censored.astype(bool)
        if censored.any():
            ax.scatter(data.loc[censored, "parameter_value"], data.loc[censored, "plotted_last_active"], marker="^", color="#be123c", label="censored")
            ax.legend()
        ax.set(xlabel=xlabel, ylabel="Last active observation diagonal", title=f"Empirical product-2 boundary: {parameter}")
        _save(fig, figures / "policy" / stem)


def plot_forgetting_policy_maps(policy: pd.DataFrame, figures: Path) -> None:
    """Figure 8: product-2 sets in normalized and belief coordinates."""
    if policy.empty:
        return
    if "robust_action2" in policy:
        policy = policy[policy.robust_action2.fillna(False).astype(bool)].copy()
    if policy.empty:
        return
    # Export every rho in the tidy table, but keep this diagnostic figure
    # readable by showing the low, middle, and high rho for each clock.
    selected_parts = []
    for _, method_data in policy.groupby("method", sort=False):
        rho_values = np.sort(method_data.parameter_value.unique())
        positions = sorted({0, len(rho_values) // 2, len(rho_values) - 1})
        selected_rho = rho_values[positions]
        selected_parts.append(
            method_data[method_data.parameter_value.isin(selected_rho)]
        )
    policy = pd.concat(selected_parts, ignore_index=True)
    keys = list(policy.groupby(["method", "parameter_value"], sort=False))
    ncols = min(3, len(keys))
    nrows = math.ceil(len(keys) / ncols)
    fig, axes = plt.subplots(
        2 * nrows,
        ncols,
        figsize=(4.3 * ncols, 7.2 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    for panel, ((method, rho), data) in enumerate(keys):
        row = panel // ncols
        col = panel % ncols
        ax_xy = axes[2 * row, col]
        ax_mc = axes[2 * row + 1, col]
        ax_xy.scatter(data.x, data.y, s=2, color="#0f766e", rasterized=True)
        ax_xy.plot([0, 1], [1, 0], color="#94a3b8", lw=0.8)
        clock = METHOD_LABELS.get(method, method)
        ax_xy.set(xlabel="x", ylabel="y", title=f"{clock}, rho={rho:g}", xlim=(0, 1), ylim=(0, 1))
        ax_mc.scatter(data.posterior_mean, data.effective_concentration, s=2, color="#7c3aed", rasterized=True)
        ax_mc.set(xlabel="posterior mean", ylabel=r"Beta concentration $a+b$", xlim=(0, 1))
    used = len(keys)
    for panel in range(used, nrows * ncols):
        row, col = divmod(panel, ncols)
        axes[2 * row, col].set_visible(False)
        axes[2 * row + 1, col].set_visible(False)
    fig.suptitle("Forgetting: product-2 policy regions", x=0.01, ha="left")
    _save(fig, figures / "policy" / "08_forgetting_policy_maps")


def plot_observation_forgetting_policy_regions(
    policy: pd.DataFrame,
    figures: Path,
) -> None:
    """Figure 24: observation forgetting in the discrete-policy coordinates.

    The horizontal coordinate is bounded effective sample size rather than
    calendar time or the unbounded observation count used by discrete models.
    Only robust product-2 states are shown, matching Figures 3--5.
    """
    if policy.empty or "method" not in policy:
        return
    data = policy[policy.method == "observation_forgetting_ts"].copy()
    if "robust_action2" in data:
        data = data[data.robust_action2.fillna(False).astype(bool)]
    if data.empty:
        return

    configure_style()
    groups = list(data.groupby("parameter_value", sort=True))
    ncols = min(3, len(groups))
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.2 * ncols, 3.3 * nrows),
        squeeze=False,
        sharey=True,
        constrained_layout=True,
    )
    for ax, (rho, group) in zip(axes.ravel(), groups, strict=False):
        rho = float(rho)
        maximum_effective_sample = 1.0 / (1.0 - rho)
        effective_axis = np.linspace(0.0, maximum_effective_sample, 500)
        feasible_lower = 1.0 / (effective_axis + 2.0)
        feasible_upper = 1.0 - feasible_lower
        ax.fill_between(
            effective_axis,
            feasible_lower,
            feasible_upper,
            color="#e2e8f0",
            alpha=0.65,
            linewidth=0.0,
            zorder=0,
        )
        ax.scatter(
            group.effective_sample_size,
            group.posterior_mean,
            s=1.6,
            color="#0f766e",
            alpha=0.65,
            rasterized=True,
            zorder=2,
        )
        ax.axhline(
            float(group.p0.iloc[0]),
            color="#475569",
            linewidth=0.8,
            linestyle="--",
            zorder=3,
        )
        ax.set(
            title=rf"$\rho$={rho:g}; max $n_{{\rm eff}}$={maximum_effective_sample:g}",
            xlabel=r"Effective sample size $n_{\rm eff}=a+b-2$",
            ylabel="Posterior mean",
            xlim=(-0.02 * maximum_effective_sample, 1.02 * maximum_effective_sample),
            ylim=(0.0, 1.0),
        )
    for ax in axes.ravel()[len(groups):]:
        ax.set_visible(False)
    fig.suptitle(
        "Product-2 policy regions on the continuous grid: observation-time forgetting",
        x=0.01,
        ha="left",
    )
    _save(fig, figures / "policy" / "24_policy_observation_forgetting")


def _late_share_column(summary: pd.DataFrame) -> str:
    for candidate in ["late_product2_share_conditional_on_A", "product2_share_conditional_on_A"]:
        if candidate in summary:
            return candidate
    raise KeyError("Simulation summary has no product-2 share column")


def plot_long_run_shares(summary: pd.DataFrame, figures: Path) -> None:
    """Figures 9--10: realized late use, not share of policy-grid states."""
    if summary.empty:
        return
    column = _late_share_column(summary)
    forgetting = summary[summary.method.isin(["observation_forgetting_ts", "calendar_forgetting_ts"])]
    if not forgetting.empty:
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        for color, (method, data) in zip(COLORS, forgetting.groupby("method")):
            data = data.sort_values("parameter_value")
            ax.plot(data.parameter_value, data[column], marker="o", lw=1.6, color=color, label=METHOD_LABELS.get(method, method))
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set(xlabel=r"Forgetting factor $\rho$", ylabel="Late product-2 use among A interactions", title="Recurrent investment under forgetting")
        ax.legend()
        _save(fig, figures / "simulation" / "09_long_run_product2_vs_rho")

    epsilon = summary[summary.method == "epsilon_greedy"].sort_values("parameter_value")
    if not epsilon.empty:
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(epsilon.parameter_value, epsilon[column], marker="o", lw=1.6, color="#0f766e")
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set(xlabel=r"Exploration probability $\varepsilon$", ylabel="Late product-2 use among A interactions", title="Recurrent investment under uniform epsilon-greedy")
        _save(fig, figures / "simulation" / "10_long_run_product2_vs_epsilon")


def plot_representative_trajectories(trajectories: pd.DataFrame, figures: Path) -> None:
    """Figure 11: exact simulated paths for a small reproducible subset."""
    if trajectories.empty:
        return
    available = trajectories[["method_key", "method"]].drop_duplicates()
    preferred_methods = ["standard_ts", "epsilon_greedy", "observation_forgetting_ts", "calendar_forgetting_ts"]
    selected_keys: list[str] = []
    for method in preferred_methods:
        rows = available[available.method == method]
        if len(rows):
            selected_keys.append(str(rows.method_key.iloc[min(len(rows) // 2, len(rows) - 1)]))
    selected_keys = selected_keys[:5]
    path_id = int(trajectories.representative_path.min())
    data = trajectories[(trajectories.method_key.isin(selected_keys)) & (trajectories.representative_path == path_id)]
    fig, axes = plt.subplots(5, 1, figsize=(10.2, 11.7), sharex=True)
    for color, (key, path) in zip(COLORS, data.groupby("method_key", sort=False)):
        label = str(path.method_label.iloc[0]) if "method_label" in path else key
        axes[0].plot(path.period, path.demand, lw=1.0, color=color, label=label)
        axes[1].plot(path.period, path.posterior_mean, lw=1.0, color=color)
        axes[2].plot(path.period, path.effective_sample_size, lw=1.0, color=color)
        profit = path.profit.to_numpy(dtype=float)
        axes[3].plot(
            path.period,
            pd.Series(profit).rolling(51, min_periods=1).mean(),
            lw=1.0,
            color=color,
        )
        used = path.product2_used.to_numpy(dtype=float)
        window = min(51, max(1, len(used)))
        rolling = pd.Series(used).rolling(window, min_periods=1).mean()
        axes[4].plot(path.period, rolling, lw=1.0, color=color)
    axes[0].set(ylabel="Demand")
    axes[1].set(ylabel="Posterior mean")
    axes[2].set(ylabel="Effective\nsample size")
    axes[2].set_yscale("symlog", linthresh=1.0)
    axes[3].set(ylabel="Profit\n(51-period mean)")
    axes[4].set(xlabel="Calendar period", ylabel="Product 2 use\n(rolling rate)")
    for ax in axes:
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle(f"Representative simulated path {path_id}", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    _save(fig, figures / "simulation" / "11_representative_trajectories")


def make_all_figures(tables: dict[str, pd.DataFrame], figures: Path) -> None:
    plot_demand_sensitivity(tables.get("demand_sensitivity", pd.DataFrame()), tables.get("sensitivity_regressions", pd.DataFrame()), figures)
    plot_discrete_policy_regions(
        tables.get("discrete_policy_points", pd.DataFrame()),
        figures,
        tables.get("summary", pd.DataFrame()),
    )
    plot_last_active(tables.get("summary", pd.DataFrame()), figures)
    plot_forgetting_policy_maps(tables.get("continuous_policy_points", pd.DataFrame()), figures)
    plot_observation_forgetting_policy_regions(
        tables.get("continuous_policy_points", pd.DataFrame()), figures
    )
    plot_long_run_shares(tables.get("simulation_summary", pd.DataFrame()), figures)
    plot_representative_trajectories(tables.get("representative_trajectories", pd.DataFrame()), figures)
