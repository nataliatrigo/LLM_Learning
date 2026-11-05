"""Compare the optimal boundary advantage across p0 and rho.

Run from the repository root:
    uv run python -B forgetting/Code/visualizations/plot_advantage_by_p0_rho.py
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_advantage_by_rho import (
    FORMULA, RHOS, ROOT, Primitives, make_figures, save_figure,
    solve_cases, style_axis, write_csv,
)


DEST = ROOT / "forgetting/outputs/advantage_by_p0_and_rho"
RHO_COLORS = ("#68737d", "#2a72ad", "#54a1c4", "#24845b", "#dc9225", "#ce513d", "#8d66ad", "#573b75")
P0_COLORS = ("#2675b0", "#df8526", "#249061", "#b44955", "#8155a7")
CALIBRATION = r"$c_1=0.05,\ c_2=0.65,\ p_1=0.35,\ p_2=0.80,\ \gamma=0.98,\ R=1$"


def common_limits(groups):
    minimum = min(float(c["a"].min()) for cases in groups.values() for c in cases)
    maximum = max(float(c["a"].max()) for cases in groups.values() for c in cases)
    padding = 0.07 * (maximum - minimum)
    return minimum - padding, maximum + padding


def decorate(fig, title, subtitle):
    fig.suptitle(title, y=0.975, fontsize=21)
    fig.text(0.52, 0.919, FORMULA, ha="center", fontsize=17)
    fig.text(0.52, 0.868, CALIBRATION, ha="center", fontsize=12)
    fig.text(0.52, 0.823, subtitle, ha="center", fontsize=12, color="#44505d")
    fig.supxlabel(r"Normalized success state $x=(1-\rho)S$, along $z=(x,1-x)$", y=0.11, fontsize=13)
    fig.supylabel(r"Advantage $a_\rho(z)=Q_2-Q_1$", x=0.012, fontsize=13)
    fig.text(0.52, 0.060, "Above zero: product 2 is preferred. Below zero: product 1 is preferred. All panels use the same scale.", ha="center", fontsize=11)
    fig.text(0.52, 0.026, r"Curves cover the full mature boundary; for $\rho=0.3$, the limiting reachable set is a Cantor subset.", ha="center", fontsize=10, color="#52606d")


def comparison_by_p0(groups, destination):
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.98, top=0.755, bottom=0.18, hspace=0.30, wspace=0.13)
    for ax, (p0, cases) in zip(axes.flat, groups.items()):
        style_axis(ax)
        ax.set_ylim(*common_limits(groups))
        ax.set_title(rf"$p_0={p0:g}$", pad=10)
        for case, color in zip(cases, RHO_COLORS):
            rho = case["solution"].rho
            ax.plot(case["x"], case["a"], color=color, lw=1.8, ls="--" if rho < 0.5 else "-")
        ax.tick_params(labelbottom=True)
    legend_ax = axes.flat[-1]
    legend_ax.set_axis_off()
    handles = [Line2D([0], [0], color=color, lw=2.5, ls="--" if rho < 0.5 else "-", label=rf"$\rho={rho:g}$") for rho, color in zip(RHOS, RHO_COLORS)]
    legend_ax.legend(handles=handles, ncols=2, frameon=False, loc="center", title="Forgetting parameter", fontsize=13, title_fontsize=14)
    decorate(fig, "Product-2 advantage across outside-option qualities", "Each panel fixes p₀; colored curves compare the same eight values of ρ.")
    save_figure(fig, "advantage_comparison_by_p0", destination)


def comparison_by_rho(groups, destination):
    fig, axes = plt.subplots(2, 4, figsize=(16, 9.5), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.98, top=0.705, bottom=0.18, hspace=0.36, wspace=0.13)
    for index, (ax, rho) in enumerate(zip(axes.flat, RHOS)):
        style_axis(ax)
        ax.set_ylim(*common_limits(groups))
        ax.set_title(rf"$\rho={rho:g}$", pad=10)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=["0", ".25", ".5", ".75", "1"])
        for (p0, cases), color in zip(groups.items(), P0_COLORS):
            case = cases[index]
            ax.plot(case["x"], case["a"], color=color, lw=1.9, ls="--" if rho < 0.5 else "-", label=rf"$p_0={p0:g}$")
    handles = [Line2D([0], [0], color=color, lw=2, label=rf"$p_0={p0:g}$") for p0, color in zip(groups, P0_COLORS)]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.52, 0.777), ncols=len(groups), frameon=False, fontsize=12)
    decorate(fig, "How outside-option quality changes product-2 advantage", "Each panel fixes ρ; colored curves compare p₀.")
    save_figure(fig, "advantage_comparison_by_rho", destination)


def main():
    config = json.loads((ROOT / "forgetting/Code/experiments/config_paper.json").read_text())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p0-values", type=float, nargs="+", default=config["p0_comparison"]["values"])
    parser.add_argument("--output-dir", type=Path, default=DEST)
    args = parser.parse_args()
    p0_values = sorted(set(args.p0_values))
    if len(p0_values) != 5:
        parser.error("The comparison layout requires five distinct p0 values.")
    if not all(0 < p0 < 1 for p0 in p0_values):
        parser.error("Each p0 must be strictly between zero and one.")
    destination = args.output_dir
    for name in ("figures", "data"):
        (destination / name).mkdir(parents=True, exist_ok=True)
    baseline = Primitives.from_mapping({**config["seller"], "gamma": config["rho_p0_comparative_statics"]["gamma"]})
    groups = {}
    for p0 in p0_values:
        case_destination = destination / f"p0_{p0:g}"
        for name in ("figures", "data"):
            (case_destination / name).mkdir(parents=True, exist_ok=True)
        groups[p0] = solve_cases(replace(baseline, p0=p0), case_destination)
    summaries = [case["summary"] for cases in groups.values() for case in cases]
    write_csv(destination / "data/summary.csv", summaries)
    for p0, cases in groups.items():
        make_figures(cases, destination / f"p0_{p0:g}")
    comparison_by_p0(groups, destination)
    comparison_by_rho(groups, destination)
    manifest = dict(
        baseline_primitives=baseline.as_dict(), p0_values=p0_values, rhos=RHOS,
        domain="Mature boundary z=(x,1-x), x=(1-rho)*S in [0,1]",
        expression="-delta_c + gamma*delta_p*(V_rho(T_success z)-V_rho(T_failure z))",
        value_function="Optimal Bellman value, independently solved for every (p0,rho) pair",
        solver="16001 uniform nodes; linear interpolation; iteration tolerance 1e-12",
        validation="8001-to-16001-node comparison at the union of all advantage interpolation knots",
        max_curve_change=max(row["max_curve_change_8001_to_16001"] for row in summaries),
        max_bellman_residual=max(row["bellman_residual"] for row in summaries),
        reachable_set="Full interval for rho>=0.5; depth-16 endpoint approximation otherwise",
        note="Numerical grid stability checks are not rigorous discretization error bounds.",
    )
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (destination / "README.md").write_text(
        "# Product-2 advantage across p0 and rho\n\n"
        "The two combined figures group the same results by p0 and by rho. "
        "All combined panels share the same x and y scales. "
        "Each p0 folder contains an eight-panel figure, an overlay, and CSV curve data. "
        "All figures are available as PNG and vector PDF.\n\n"
        "The horizontal axis follows the long-run boundary z=(x,1-x), with x=(1-rho)*S. "
        "Positive advantage favors product 2. For rho<0.5, only a Cantor subset "
        "of the plotted boundary is limiting-reachable; individual panels mark its endpoints. "
        "summary.csv reports numerical reachable-set maxima and full-boundary zero crossings.\n\n"
        "Calibration and convergence checks are recorded in manifest.json. "
        "The optimal value is recomputed for every parameter pair.\n\n"
        "Reproduce from the repository root:\n\n"
        "```bash\nuv run python -B forgetting/Code/visualizations/plot_advantage_by_p0_rho.py\n```\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Saved plots and data to {destination}", flush=True)


if __name__ == "__main__":
    main()
