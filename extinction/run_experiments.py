"""Reproduce the cumulative-learning figures and cutoff table in the paper."""

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


import matplotlib
from extinction.model import Parameters, POLICY_PRODUCT2, action_tolerance, solve_stationary_policy_truncation
from extinction.simulation import simulate_calendar_paths


def series_colors(count: int) -> np.ndarray:
    """Return the shared viridis palette used by every multi-line figure."""

    return matplotlib.colormaps["viridis"](np.linspace(0.08, 0.88, count))


def read_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def configured_parameters(config: dict) -> Parameters:
    return Parameters(**config["primitives"])


def configured_tolerance(config: dict):
    solver = config["solver"]

    def tolerance(values: np.ndarray) -> float:
        return action_tolerance(
            values,
            absolute=float(solver["absolute_action_tolerance"]),
            relative=float(solver["relative_action_tolerance"]),
        )

    return tolerance


def configured_p0_values(config: dict, baseline_p0: float) -> list[float]:
    """Return the validated deterministic comparative-static grid."""

    values = sorted(
        set(float(value) for value in config["p0_comparison"]["values"])
    )
    if not values or any(not 0.0 < value < 1.0 for value in values):
        raise ValueError("p0_comparison values must lie strictly inside (0,1)")
    if not any(np.isclose(value, baseline_p0) for value in values):
        values.append(float(baseline_p0))
        values.sort()
    return values


def compare_compact_policies(
    smaller: dict,
    larger: dict,
    *,
    minimum_inactive_tail: int,
) -> dict[str, float | int | bool]:
    """Compare two compact policy solves on their common reported lattice."""

    report_diagonal = min(
        int(smaller["report_diagonal"]),
        int(larger["report_diagonal"]),
    )
    disagreements = 0
    state_count = 0
    for n in range(report_diagonal + 1):
        first = np.asarray(smaller["actions"][n], dtype=np.int8)
        second = np.asarray(larger["actions"][n], dtype=np.int8)
        disagreements += int(np.count_nonzero(first != second))
        state_count += n + 1

    smaller_diagonals = smaller["diagonals"]
    larger_diagonals = larger["diagonals"]
    smaller_active = smaller_diagonals.loc[
        smaller_diagonals.n <= report_diagonal
    ]
    smaller_active = smaller_active.loc[smaller_active.active, "n"]
    larger_active = larger_diagonals.loc[
        larger_diagonals.n <= report_diagonal
    ]
    larger_active = larger_active.loc[larger_active.active, "n"]
    smaller_last_active = int(smaller_active.max()) if len(smaller_active) else -1
    last_active = int(larger_active.max()) if len(larger_active) else -1
    tail = larger_diagonals[
        (larger_diagonals.n > last_active)
        & (larger_diagonals.n <= report_diagonal)
    ]
    inactive_tail_length = report_diagonal - last_active
    stable = bool(
        disagreements == 0 and smaller_last_active == last_active
    )
    tail_is_strict_product1 = bool(
        not tail.active.any() and not tail.ambiguous.any()
    )
    cutoff_resolved = bool(
        stable
        and tail_is_strict_product1
        and inactive_tail_length >= minimum_inactive_tail
    )
    return {
        "p0": float(larger["parameters"].p0),
        "smaller_outer": int(smaller["outer_diagonal"]),
        "larger_outer": int(larger["outer_diagonal"]),
        "report_diagonal": report_diagonal,
        "smaller_last_active_diagonal": smaller_last_active,
        "last_active_diagonal": last_active,
        "first_inactive_diagonal_after_last_active": last_active + 1,
        "inactive_tail_length": inactive_tail_length,
        "cutoff_resolved": cutoff_resolved,
        "active_at_report_diagonal": bool(
            larger_diagonals.loc[
                larger_diagonals.n == report_diagonal,
                "active",
            ].iloc[0]
        ),
        "no_reactivation_through_report_diagonal": bool(not tail.active.any()),
        "strict_product1_tail": tail_is_strict_product1,
        "policy_disagreements": disagreements,
        "policy_disagreement_fraction": disagreements / state_count,
        "numerically_ambiguous_diagonals": int(
            larger_diagonals.loc[
                larger_diagonals.n <= report_diagonal,
                "ambiguous",
            ].sum()
        ),
        "robust_interval_violations": int(
            larger_diagonals.loc[
                larger_diagonals.n <= report_diagonal,
                "robust_interval_violation",
            ].sum()
        ),
        "distance_from_last_active_to_larger_outer": int(
            larger["outer_diagonal"]
        )
        - last_active,
        "maximum_bellman_residual": float(
            larger["maximum_bellman_residual"]
        ),
    }


def solve_p0_policy_comparison(
    config: dict,
    params: Parameters,
    tolerance_function,
) -> tuple[dict[float, dict], pd.DataFrame]:
    """Compute compact statewise policies for every configured ``p0``."""

    comparison = config["p0_comparison"]
    outer_diagonals = [int(value) for value in comparison["outer_diagonals"]]
    report_diagonal = int(comparison["report_diagonal"])
    minimum_inactive_tail = int(comparison["minimum_inactive_tail"])
    if sorted(outer_diagonals) != outer_diagonals or len(outer_diagonals) != 2:
        raise ValueError(
            "p0_comparison outer_diagonals must contain two increasing values"
        )
    if report_diagonal >= outer_diagonals[0]:
        raise ValueError(
            "p0_comparison report_diagonal must be below both outer diagonals"
        )
    if minimum_inactive_tail < 1:
        raise ValueError("minimum_inactive_tail must be positive")

    largest_policies: dict[float, dict] = {}
    rows: list[dict[str, float | int | bool]] = []
    for p0 in configured_p0_values(config, params.p0):
        policy_params = replace(params, p0=p0)
        compact_solutions = []
        for outer in outer_diagonals:
            print(
                f"[solve p0 policy] p0={p0:.2f}, outer diagonal {outer}",
                flush=True,
            )
            compact_solutions.append(
                solve_stationary_policy_truncation(
                    policy_params,
                    outer,
                    report_diagonal,
                    tolerance_function=tolerance_function,
                )
            )
        rows.append(
            compare_compact_policies(
                compact_solutions[0],
                compact_solutions[1],
                minimum_inactive_tail=minimum_inactive_tail,
            )
        )
        largest_policies[p0] = compact_solutions[1]

    return largest_policies, pd.DataFrame(rows).sort_values(
        "p0",
        ignore_index=True,
    )


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#94a3b8",
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
            "savefig.dpi": 300,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    png_path = FIGURES / f"{stem}.png"
    pdf_path = FIGURES / f"{stem}.pdf"
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)


def product2_policy_points(
    policy_solution: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(n, posterior mean)`` for compact-policy product-2 states."""

    observation_chunks: list[np.ndarray] = []
    posterior_chunks: list[np.ndarray] = []
    for n in range(int(policy_solution["report_diagonal"]) + 1):
        actions = np.asarray(policy_solution["actions"][n], dtype=np.int8)
        successes = np.flatnonzero(actions == POLICY_PRODUCT2)
        if not len(successes):
            continue
        observation_chunks.append(np.full(len(successes), n, dtype=int))
        posterior_chunks.append((successes + 1.0) / (n + 2.0))
    if not observation_chunks:
        return np.array([], dtype=int), np.array([], dtype=float)
    return np.concatenate(observation_chunks), np.concatenate(posterior_chunks)


def plot_cumulative_learning_p0_panels(
    period_summary: pd.DataFrame,
) -> None:
    """Plot demand and unconditional product-2 use with one shared legend."""

    p0_values = sorted(float(value) for value in period_summary.p0.unique())
    colors = series_colors(len(p0_values))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8))
    for p0, color in zip(p0_values, colors, strict=True):
        data = period_summary[
            np.isclose(period_summary.p0, p0) & (period_summary.period < 500)
        ]
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
    for ax in axes:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))

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
    save_figure(fig, "monte_carlo_cumulative_learning_p0_panels")

def plot_overlay(
    points: dict[float, tuple[np.ndarray, np.ndarray]],
) -> plt.Figure:
    """Final paper figure: all five state spaces using the default cycle."""

    fig, axis = plt.subplots(figsize=(7.2, 3.35), constrained_layout=True)
    for p0 in sorted(points):
        observations, posterior_means = points[p0]
        artist = axis.scatter(
            observations,
            posterior_means,
            s=0.75,
            alpha=0.42,
            linewidths=0,
            rasterized=True,
            label=rf"$p_0={p0:.2f}$",
        )
        color = artist.get_facecolors()[0]
        axis.axhline(
            p0,
            color=color,
            linewidth=0.8,
            linestyle="--",
            alpha=0.75,
        )

    largest_observation = max(
        int(observations.max()) for observations, _ in points.values()
    )
    horizontal_maximum = int(np.ceil(1.04 * largest_observation / 100.0) * 100)
    axis.set(
        xlim=(0, horizontal_maximum),
        ylim=(0, 1),
        xlabel=r"Number of Seller A observations $n=S+F$",
        ylabel=r"Posterior mean $(S+1)/(S+F+2)$",
    )
    axis.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        ncols=5,
        fontsize=7.5,
        markerscale=4,
        columnspacing=1.1,
        handletextpad=0.3,
        borderaxespad=0.15,
    )
    axis.set_axisbelow(True)
    return fig


def main() -> None:
    global OUTPUTS, DATA, FIGURES
    args = parse_args()
    OUTPUTS = args.output_dir.resolve()
    DATA, FIGURES = OUTPUTS / "data", OUTPUTS / "figures"
    DATA.mkdir(parents=True, exist_ok=True)
    config = read_config(args.config)
    params = configured_parameters(config)
    configure_style()
    policies, diagnostics = solve_p0_policy_comparison(config, params, configured_tolerance(config))
    diagnostics.to_csv(DATA / "p0_policy_diagnostics.csv", index=False)
    if not diagnostics.cutoff_resolved.all():
        raise RuntimeError("Cutoffs unresolved: increase the outer/report grids or inspect diagnostics.")
    if diagnostics.numerically_ambiguous_diagonals.any():
        raise RuntimeError("Numerically ambiguous policy states: inspect diagnostics.")
    diagnostics[["p0", "last_active_diagonal"]].to_csv(DATA / "paper_cutoffs.csv", index=False)
    points = {p0: product2_policy_points(policy) for p0, policy in policies.items()}
    save_figure(plot_overlay(points), "cumulative_learning_overlay")
    stems = ["cumulative_learning_overlay"]
    simulation = config["simulation"]
    calendar = simulation["calendar"]
    mc_stem = "monte_carlo_cumulative_learning_p0_panels"
    if not args.skip_simulation and simulation["enabled"] and calendar["enabled"]:
        frames = [[], [], []]
        for index, (p0, policy) in enumerate(sorted(policies.items())):
            seed = int(calendar["seed"]) + (0 if np.isclose(p0, params.p0) else 10_000 * (index + 1))
            print(f"[simulate calendar] p0={p0:.2f}", flush=True)
            results = simulate_calendar_paths(
                policy, n_paths=int(calendar["paths"]), seed=seed,
                max_periods=int(calendar["periods"]), bin_width=int(calendar["bin_width"]),
                confidence_level=float(simulation["confidence_level"]),
                absolute_action_tolerance=float(config["solver"]["absolute_action_tolerance"]),
                relative_action_tolerance=float(config["solver"]["relative_action_tolerance"]))
            for target, frame in zip(frames, results, strict=True):
                frame.insert(0, "p0", p0)
                frame.insert(1, "seed", seed)
                target.append(frame)
        combined = [pd.concat(parts, ignore_index=True) for parts in frames]
        for name, frame in zip(("by_period", "binned", "path_summary"), combined, strict=True):
            frame.to_csv(DATA / f"monte_carlo_calendar_{name}.csv", index=False)
        plot_cumulative_learning_p0_panels(combined[0])
        stems.append(mc_stem)
    else:
        clear_simulation_outputs(mc_stem)
    finish_run(config, stems, args.sync_paper_assets)


if __name__ == "__main__":
    main()
