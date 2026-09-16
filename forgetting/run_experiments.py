"""Reproduce the three observation-forgetting figures in the paper."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import replace
from pathlib import Path
import shutil

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/paper_matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DATA = OUTPUTS / "data"
FIGURES = OUTPUTS / "figures"
PAPER_FIGURES = ROOT.parent / "discounted" / "OverleafPaper" / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUTS)
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument("--sync-paper-assets", action="store_true",
                        help="copy this run's PDFs into the local manuscript")
    return parser.parse_args()


def finish_run(config: dict, stems: list[str], sync: bool) -> None:
    """Record the effective configuration and only the figures from this run."""
    (OUTPUTS / "manifest.json").write_text(
        json.dumps({"config": config, "figures": [f"figures/{s}.pdf" for s in stems]},
                   indent=2) + "\n", encoding="utf-8")
    if sync:
        PAPER_FIGURES.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            shutil.copy2(FIGURES / f"{stem}.pdf", PAPER_FIGURES / f"{stem}.pdf")
    print(f"Results: {OUTPUTS}", flush=True)


def clear_simulation_outputs(stem: str) -> None:
    for name in ("by_period", "binned", "path_summary"):
        (DATA / f"monte_carlo_calendar_{name}.csv").unlink(missing_ok=True)
    for suffix in ("pdf", "png"):
        (FIGURES / f"{stem}.{suffix}").unlink(missing_ok=True)


from typing import Callable
from forgetting.model import Primitives, boundary_summary, solve_boundary
from forgetting.simulation import simulate_calendar_paths

RAW_ADVANTAGE_MAX_LABEL = "largest gain from choosing product 2 over product 1"


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def rho_grid(config: dict) -> np.ndarray:
    linear = config["rho_linear"]
    values = np.concatenate(
        (
            np.linspace(
                float(linear["start"]),
                float(linear["stop"]),
                int(linear["points"]),
            ),
            np.asarray(config["rho_tail"], dtype=float),
        )
    )
    return np.unique(values)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.65,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    png = FIGURES / f"{name}.png"
    pdf = FIGURES / f"{name}.pdf"
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=220, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)


def configured_p0_values(config: dict, baseline_p0: float) -> list[float]:
    """Return the extinction-experiment p0 comparison grid."""

    values = sorted(
        set(float(value) for value in config["p0_comparison"]["values"])
    )
    if not values or not all(0.0 < value < 1.0 for value in values):
        raise ValueError("p0_comparison values must lie strictly inside (0,1)")
    if not any(np.isclose(value, baseline_p0) for value in values):
        values.append(float(baseline_p0))
        values.sort()
    return values


def bisect_root(
    function: Callable[[float], float],
    lower: float,
    upper: float,
    *,
    rho_tolerance: float,
    max_iterations: int,
) -> tuple[float, float, float]:
    if lower == upper:
        return lower, lower, upper
    lower_value = float(function(lower))
    upper_value = float(function(upper))
    if not lower_value * upper_value < 0.0:
        raise RuntimeError(
            f"root bracket [{lower:g},{upper:g}] has values "
            f"{lower_value:+.3e}, {upper_value:+.3e}"
        )
    for _ in range(max_iterations):
        midpoint = 0.5 * (lower + upper)
        midpoint_value = float(function(midpoint))
        if (midpoint_value > 0.0) == (lower_value > 0.0):
            lower, lower_value = midpoint, midpoint_value
        else:
            upper, upper_value = midpoint, midpoint_value
        if upper - lower <= rho_tolerance:
            break
    return 0.5 * (lower + upper), lower, upper


def run_validation(config: dict, base: Primitives) -> pd.DataFrame:
    solver = config["solver"]
    rows = []
    for rho, gamma in config["validation_pairs"]:
        primitives = replace(base, gamma=float(gamma))
        coarse = solve_boundary(
            primitives,
            float(rho),
            central_nodes=int(solver["central_nodes"]),
            outer_nodes=int(solver["outer_nodes"]),
            half_widths=float(solver["half_widths"]),
            tolerance=float(solver["tolerance"]),
        )
        fine = solve_boundary(
            primitives,
            float(rho),
            central_nodes=2 * int(solver["central_nodes"]) - 1,
            outer_nodes=2 * int(solver["outer_nodes"]),
            half_widths=float(solver["half_widths"]),
            tolerance=0.25 * float(solver["tolerance"]),
        )
        coarse_max = float(np.max(coarse.advantage))
        fine_max = float(np.max(fine.advantage))
        rows.append(
            {
                "rho": float(rho),
                "gamma": float(gamma),
                "coarse_maximum_advantage": coarse_max,
                "fine_maximum_advantage": fine_max,
                "absolute_maximum_difference": abs(fine_max - coarse_max),
                "coarse_bellman_residual": coarse.bellman_residual,
                "fine_bellman_residual": fine.bellman_residual,
                "coarse_nodes": len(coarse.coordinate),
                "fine_nodes": len(fine.coordinate),
            }
        )
    return pd.DataFrame(rows)


def plot_p0_advantage_slices(
    phase: pd.DataFrame,
    p0_values: list[float],
) -> None:
    """Plot raw action-value maxima over the mature reachable state set."""

    configure_style()
    fig, ax = plt.subplots(figsize=(8.0, 4.4), constrained_layout=True)
    palette = plt.cm.viridis(np.linspace(0.12, 0.88, len(p0_values)))
    for color, p0 in zip(palette, p0_values, strict=True):
        group = phase[np.isclose(phase.p0, p0)].sort_values("rho")
        ax.plot(
            group.rho,
            group.reachable_maximum_advantage,
            marker="o",
            markersize=2.5,
            lw=1.4,
            color=color,
            label=rf"$p_0={p0:g}$",
        )
    ax.axhline(0.0, color="#334155", ls="--", lw=0.9)
    ax.set(
        xlabel=r"Forgetting rate $\rho$",
        ylabel=RAW_ADVANTAGE_MAX_LABEL,
    )
    ax.legend(loc="upper left")
    save_figure(fig, "investment_incentives_across_forgetting_rates")


def run_monte_carlo_comparison(
    config: dict,
    base: Primitives,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate the forgetting policy separately for every configured ``p0``."""

    simulation = config["simulation"]
    p0_values = configured_p0_values(config, base.p0)
    period_frames: list[pd.DataFrame] = []
    binned_frames: list[pd.DataFrame] = []
    path_frames: list[pd.DataFrame] = []
    for index, p0 in enumerate(p0_values):
        seed = int(simulation["seed"]) + 10_000 * index
        print(
            f"[simulate forgetting] p0={p0:.2f}, "
            f"rho={float(simulation['rho']):g}, "
            f"{int(simulation['paths'])} paths",
            flush=True,
        )
        period, binned, paths = simulate_calendar_paths(
            replace(base, p0=p0),
            float(simulation["rho"]),
            n_paths=int(simulation["paths"]),
            seed=seed,
            max_periods=int(simulation["periods"]),
            bin_width=int(simulation["bin_width"]),
            confidence_level=float(simulation["confidence_level"]),
            policy_nodes=int(simulation["policy_nodes"]),
            terminal_gap=float(simulation["terminal_gap"]),
            solver_tolerance=float(simulation["solver_tolerance"]),
            action_tolerance=float(config["solver"]["action_tolerance"]),
            max_iterations=int(config["solver"]["max_iterations"]),
        )
        for frame in (period, binned, paths):
            frame.insert(0, "p0", p0)
            frame.insert(1, "rho", float(simulation["rho"]))
            frame.insert(2, "seed", seed)
        period_frames.append(period)
        binned_frames.append(binned)
        path_frames.append(paths)
    return (
        pd.concat(period_frames, ignore_index=True),
        pd.concat(binned_frames, ignore_index=True),
        pd.concat(path_frames, ignore_index=True),
    )


def plot_monte_carlo_p0_panels(period_summary: pd.DataFrame) -> None:
    """Match the paper's demand and unconditional-product-2 panels."""

    configure_style()
    p0_values = sorted(float(value) for value in period_summary.p0.unique())
    colors = plt.colormaps["viridis"](np.linspace(0.08, 0.88, len(p0_values)))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for p0, color in zip(p0_values, colors, strict=True):
        data = period_summary[np.isclose(period_summary.p0, p0)]
        label = rf"$p_0={p0:.2f}$"
        axes[0].plot(
            data.period + 1,
            data.a_market_share,
            color=color,
            linewidth=1.45,
            label=label,
        )
        axes[1].plot(
            data.period + 1,
            data.product2_unconditional_rate,
            color=color,
            linewidth=1.45,
        )
    axes[0].set(
        ylim=(-0.02, 1.02),
        xlabel=r"Period $t$",
        ylabel="Seller A demand (market share)",
    )
    axes[1].set(
        ylim=(-0.02, 1.02),
        xlabel=r"Period $t$",
        ylabel="Share of paths using product 2",
    )
    for axis, panel_label in zip(axes, ("(a)", "(b)"), strict=True):
        axis.text(
            0.0,
            1.015,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
        )
    for axis in axes:
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncols=len(p0_values),
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save_figure(fig, "monte_carlo_forgetting_p0_panels")

def p0_grid(config: dict) -> np.ndarray:
    comparison = config["rho_p0_comparative_statics"]
    return np.linspace(
        float(comparison["p0_start"]),
        float(comparison["p0_stop"]),
        int(comparison["p0_points"]),
    )


def solve_summary(
    config: dict,
    primitives: Primitives,
    rho: float,
) -> dict[str, float | int | bool | None]:
    solver = config["solver"]
    solution = solve_boundary(
        primitives,
        float(rho),
        central_nodes=int(solver["central_nodes"]),
        outer_nodes=int(solver["outer_nodes"]),
        half_widths=float(solver["half_widths"]),
        tolerance=float(solver["tolerance"]),
        max_iterations=int(solver["max_iterations"]),
    )
    return boundary_summary(
        solution,
        action_tolerance=float(solver["action_tolerance"]),
        reachable_depth=int(solver["reachable_depth"]),
    )


def run_map(config: dict, base: Primitives) -> pd.DataFrame:
    rhos = rho_grid(config)
    p0_values = list(p0_grid(config))
    for value in configured_p0_values(config, base.p0):
        if not any(np.isclose(value, existing, rtol=0.0, atol=1e-14) for existing in p0_values):
            p0_values.append(value)
    p0_values.sort()
    total = len(rhos) * len(p0_values)
    position = 0
    rows: list[dict] = []
    for p0 in p0_values:
        primitives = replace(base, p0=float(p0))
        for rho in rhos:
            position += 1
            if position == 1 or position % 100 == 0 or position == total:
                print(
                    f"[rho-p0 map {position}/{total}] p0={p0:g}, rho={rho:g}",
                    flush=True,
                )
            summary = solve_summary(config, primitives, float(rho))
            summary["p0"] = float(p0)
            rows.append(summary)
    return pd.DataFrame(rows).sort_values(["p0", "rho"]).reset_index(drop=True)


def refine_positive_windows(
    map_data: pd.DataFrame,
    config: dict,
    base: Primitives,
) -> pd.DataFrame:
    """Refine all detected positive intervals within the sampled rho range."""

    root = config["root_refinement"]
    rows: list[dict[str, float | int | None]] = []
    for p0, group in map_data.groupby("p0", sort=True):
        group = group.sort_values("rho").reset_index(drop=True)
        rhos = group.rho.to_numpy(dtype=float)
        values = group.reachable_maximum_advantage.to_numpy(dtype=float)
        brackets: list[tuple[float, float]] = []
        for index in range(len(rhos) - 1):
            if values[index] == 0.0:
                brackets.append((float(rhos[index]), float(rhos[index])))
            elif values[index] * values[index + 1] < 0.0:
                brackets.append((float(rhos[index]), float(rhos[index + 1])))

        if values[-1] == 0.0:
            brackets.append((float(rhos[-1]), float(rhos[-1])))

        primitives = replace(base, p0=float(p0))
        cache = {
            float(rho): float(value)
            for rho, value in zip(rhos, values, strict=True)
        }

        def objective(rho: float) -> float:
            rho = float(rho)
            if rho not in cache:
                cache[rho] = float(
                    solve_summary(config, primitives, rho)[
                        "reachable_maximum_advantage"
                    ]
                )
            return cache[rho]

        roots = [
            bisect_root(
                objective,
                lower,
                upper,
                rho_tolerance=float(root["rho_tolerance"]),
                max_iterations=int(root["max_bisection_iterations"]),
            )
            for lower, upper in brackets
        ]
        # Keep every connected positive interval, rather than filling between
        # the first and last crossing when the reachable set creates a gap.
        edges = [(float(rhos[0]), None), *[(r[0], r[2] - r[1]) for r in roots],
                 (float(rhos[-1]), None)]
        intervals = []
        for (lower, lower_width), (upper, upper_width) in zip(edges[:-1], edges[1:]):
            if upper > lower and objective(0.5 * (lower + upper)) > 0.0:
                intervals.append({
                    "rho_lower": lower,
                    "rho_upper": upper,
                    "window_width": upper - lower,
                    "lower_bracket_width": lower_width,
                    "upper_bracket_width": upper_width,
                    "lower_censored": lower_width is None,
                    "upper_censored": upper_width is None,
                })
        common = {"p0": float(p0), "crossing_count": len(roots),
                  "window_count": len(intervals)}
        if intervals:
            for index, interval in enumerate(intervals, start=1):
                rows.append({**common, "window_index": index, **interval})
        else:
            rows.append({**common, "window_index": None, "rho_lower": None,
                         "rho_upper": None, "window_width": None,
                         "lower_bracket_width": None, "upper_bracket_width": None,
                         "lower_censored": False, "upper_censored": False})
        print(f"[positive windows] p0={p0:g}, crossings={len(roots)}, "
              f"intervals={[(r['rho_lower'], r['rho_upper']) for r in intervals]}",
              flush=True)

    return pd.DataFrame(rows)


def plot_positive_windows(windows: pd.DataFrame) -> None:
    """Plot the rho interval on which some state has A(rho,p0)>0."""

    configure_style()
    valid = windows.dropna(subset=["rho_lower", "rho_upper"])
    fig, ax = plt.subplots(figsize=(7.6, 4.8), constrained_layout=True)
    if (windows.groupby("p0").size() > 1).any():
        # With changing topology, joining outer bounds would invent activity
        # inside gaps. Show the computed intervals at each sampled p0 instead.
        ax.vlines(valid.p0, valid.rho_lower, valid.rho_upper,
                  color="#2563eb", linewidth=5, label="Positive investment intervals")
        ax.scatter(valid.p0, valid.rho_lower, color="#2563eb", s=18)
        ax.scatter(valid.p0, valid.rho_upper, color="#2563eb", s=18)
        if windows.p0.nunique() <= 10:
            ax.set_xticks(sorted(windows.p0.unique()))
    else:
        # Retain NaN rows so curves cannot bridge an inactive p0 sample.
        ordered = windows.sort_values("p0")
        ax.plot(ordered.p0, ordered.rho_lower, "o-", color="#2563eb",
                label=r"$\rho_L(p_0)$: lower threshold")
        ax.plot(ordered.p0, ordered.rho_upper, "o-", color="#f97316",
                label=r"$\rho_H(p_0)$: upper threshold")
        ax.fill_between(ordered.p0.to_numpy(dtype=float),
                        ordered.rho_lower.to_numpy(dtype=float),
                        ordered.rho_upper.to_numpy(dtype=float),
                        color="#ccfbf1", alpha=0.8)
    if valid.empty:
        ax.text(0.5, 0.5, "No positive interval detected on the sampled grid",
                transform=ax.transAxes, ha="center")
    ax.set(
        xlabel=r"Outside option quality $p_0$",
        ylabel=r"Forgetting rate $\rho$",
        ylim=(-0.02, 1.01),
    )
    ax.legend(loc="lower left")
    save_figure(fig, "positive_investment_window_by_p0")


def main() -> None:
    global OUTPUTS, DATA, FIGURES
    args = parse_args()
    OUTPUTS = args.output_dir.resolve()
    DATA, FIGURES = OUTPUTS / "data", OUTPUTS / "figures"
    DATA.mkdir(parents=True, exist_ok=True)
    config = read_config(args.config)
    base = Primitives.from_mapping({**config["seller"], "gamma": config["rho_p0_comparative_statics"]["gamma"]})
    # Include the five manuscript curves in the dense p0 map so they are solved once.
    map_data = run_map(config, base)
    map_data.to_csv(DATA / "rho_p0_phase.csv", index=False)
    plot_p0_advantage_slices(map_data, configured_p0_values(config, base.p0))
    windows = refine_positive_windows(map_data, config, base)
    windows.to_csv(DATA / "positive_window_by_p0.csv", index=False)
    plot_positive_windows(windows)
    run_validation(config, base).to_csv(DATA / "validation.csv", index=False)
    stems = ["investment_incentives_across_forgetting_rates", "positive_investment_window_by_p0"]
    mc_stem = "monte_carlo_forgetting_p0_panels"
    if not args.skip_simulation and config["simulation"]["enabled"]:
        results = run_monte_carlo_comparison(config, base)
        for name, frame in zip(("by_period", "binned", "path_summary"), results, strict=True):
            frame.to_csv(DATA / f"monte_carlo_calendar_{name}.csv", index=False)
        plot_monte_carlo_p0_panels(results[0])
        stems.append(mc_stem)
    else:
        clear_simulation_outputs(mc_stem)
    finish_run(config, stems, args.sync_paper_assets)


if __name__ == "__main__":
    main()
