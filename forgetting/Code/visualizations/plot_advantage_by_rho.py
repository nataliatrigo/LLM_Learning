"""Plot the original optimal-value action advantage on the mature boundary.

Run from the repository root:
    uv run python -B forgetting/Code/visualizations/plot_advantage_by_rho.py
"""

from pathlib import Path
import csv
import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from forgetting.Code.model import Primitives, reachable_boundary_points, solve_boundary


DEST = ROOT / "forgetting/outputs/advantage_by_rho_p0_0.5"
RHOS = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999)
FORMULA = r"$a_\rho(z)=-\Delta c+\gamma\Delta p\,[V_\rho(T_\rho^s z)-V_\rho(T_\rho^f z)]$"


def advantage(solution, x):
    """Evaluate the requested expression directly using interpolated V."""
    p, rho = solution.primitives, solution.rho
    success = np.interp(rho * x + 1 - rho, solution.coordinate, solution.value)
    failure = np.interp(rho * x, solution.coordinate, solution.value)
    return -p.delta_c + p.gamma * p.delta_p * (success - failure)


def knots(solution):
    """All breakpoints of the piecewise-linear numerical advantage."""
    rho, x = solution.rho, solution.coordinate
    return np.unique(np.clip(np.concatenate((x / rho, (x - 1 + rho) / rho, [0, 1])), 0, 1))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def solve_cases(primitives, destination=DEST, *, optimize=True):
    cases, summaries, curve_rows = [], [], []
    for rho in RHOS:
        settings = dict(outer_nodes=2, half_widths=1e6, tolerance=1e-12, optimize=optimize)
        coarse = solve_boundary(primitives, rho, central_nodes=8001, **settings)
        fine = solve_boundary(primitives, rho, central_nodes=16001, **settings)
        np.testing.assert_allclose(
            advantage(fine, fine.coordinate), fine.advantage, atol=1e-12, rtol=0
        )
        x = knots(fine)
        a = advantage(fine, x)
        crossing = np.flatnonzero(a[:-1] * a[1:] < 0)
        roots = x[crossing] - a[crossing] * np.diff(x)[crossing] / np.diff(a)[crossing]
        roots = np.unique(np.concatenate((roots, x[a == 0])))
        assert np.all(np.abs(advantage(fine, roots)) < 1e-10)
        evaluation = reachable_boundary_points(rho, depth=16) if rho < 0.5 else x
        reachable_a = advantage(fine, evaluation)
        index = int(np.argmax(reachable_a))
        check_x = np.union1d(x, knots(coarse))
        difference = float(np.max(np.abs(advantage(fine, check_x) - advantage(coarse, check_x))))
        assert fine.bellman_residual < 1.1e-12
        assert np.all(np.isfinite(a))
        summary = dict(
            rho=rho,
            p0=primitives.p0,
            max_reachable_advantage=float(reachable_a[index]),
            argmax_reachable_x=float(evaluation[index]),
            max_full_boundary_advantage=float(a.max()),
            min_full_boundary_advantage=float(a.min()),
            zero_crossings_x=json.dumps(roots.tolist()),
            bellman_residual=fine.bellman_residual,
            max_curve_change_8001_to_16001=difference,
            reachable_set_approximation_radius=rho**16 if rho < 0.5 else 0.0,
        )
        summaries.append(summary)
        cases.append(dict(solution=fine, x=x, a=a, roots=roots, summary=summary))
        for xx, aa in zip(x, a):
            curve_rows.append(dict(rho=rho, p0=primitives.p0, x=xx, S=xx / (1 - rho), advantage=aa))
        print(f"p0={primitives.p0:g}, rho={rho:g}: max reachable={reachable_a[index]:.8f}, roots={roots.tolist()}, grid change={difference:.2e}", flush=True)
    write_csv(destination / "data/summary.csv", summaries)
    write_csv(destination / "data/curves.csv", curve_rows)
    return cases


def style_axis(ax):
    ax.set_xlim(0, 1)
    ax.axhline(0, color="#323b46", linewidth=1, linestyle="--", zorder=2)
    ax.grid(axis="y", color="#e3e7ec", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b3bbc4")


def save_figure(fig, stem, destination=DEST):
    for extension in ("png", "pdf"):
        fig.savefig(destination / "figures" / f"{stem}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def make_figures(cases, destination=DEST):
    p0 = cases[0]["solution"].primitives.p0
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12})
    blue, green = "#2468a0", "#24835b"
    low = min(float(case["a"].min()) for case in cases)
    high = max(float(case["a"].max()) for case in cases)
    span = high - low
    y_limits = (low - 0.05 * span, high + 0.23 * span)
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.77, bottom=0.18, hspace=0.33, wspace=0.13)
    for ax, case in zip(axes.flat, cases):
        sol, x, a = case["solution"], case["x"], case["a"]
        style_axis(ax)
        ax.set_ylim(*y_limits)
        if sol.rho < 0.5:
            ax.plot(x, a, color="#929ca8", linestyle="--", linewidth=1.5)
            reachable = reachable_boundary_points(sol.rho, depth=8)
            ax.scatter(reachable, advantage(sol, reachable), s=5, color=blue, zorder=3)
        else:
            ax.plot(x, a, color=blue, linewidth=2)
            ax.fill_between(x, 0, a, where=a > 0, interpolate=True, color=green, alpha=0.18)
        ax.scatter(case["roots"], np.zeros_like(case["roots"]), s=24, facecolor="white", edgecolor=blue, zorder=4)
        ax.set_title(rf"$\rho={sol.rho:g}$", pad=10)
        ax.text(0.96, 0.93, rf"$\max_{{K_\rho}} a_\rho={case['summary']['max_reachable_advantage']:.3f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=10)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=["0", ".25", ".5", ".75", "1"])
    fig.suptitle(rf"Product-2 advantage across forgetting rates  |  $p_0={p0:g}$", y=0.97, fontsize=20)
    fig.text(0.52, 0.902, FORMULA, ha="center", fontsize=17)
    fig.text(0.52, 0.845, r"$c_1=0.05,\ c_2=0.65,\ p_1=0.35,\ p_2=0.80,\ \gamma=0.98,\ R=1$", ha="center", fontsize=12)
    fig.supxlabel(r"Normalized success state $x=(1-\rho)S$, along $z=(x,1-x)$", y=0.108, fontsize=13)
    fig.supylabel(r"Advantage $a_\rho(z)=Q_2-Q_1$", x=0.012, fontsize=13)
    fig.text(0.52, 0.055, "Above zero: product 2 is preferred. Green shading marks positive advantage. All panels use the same scale.", ha="center", fontsize=11)
    fig.text(0.52, 0.02, r"For $\rho=0.3$, blue dots approximate the limiting reachable set; gray dashes show the full boundary.", ha="center", fontsize=10, color="#52606d")
    save_figure(fig, "advantage_by_rho_panels", destination)

    fig, ax = plt.subplots(figsize=(10.5, 7))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.75, bottom=0.20)
    colors = ("#68737d", "#2a72ad", "#54a1c4", "#24845b", "#dc9225", "#ce513d", "#8d66ad", "#573b75")
    style_axis(ax)
    ax.set_ylim(*y_limits)
    for case, color in zip(cases, colors):
        rho = case["solution"].rho
        ax.plot(case["x"], case["a"], color=color, linewidth=2, linestyle="--" if rho < 0.5 else "-", label=rf"$\rho={rho:g}$")
    ax.legend(ncols=4, frameon=False, loc="upper right", fontsize=10)
    ax.set_xlabel(r"Normalized success state $x=(1-\rho)S$", labelpad=10)
    ax.set_ylabel(r"Advantage $a_\rho(z)=Q_2-Q_1$")
    fig.suptitle(rf"Product-2 advantage across forgetting rates  |  $p_0={p0:g}$", y=0.97, fontsize=18)
    fig.text(0.53, 0.895, FORMULA, ha="center", fontsize=15)
    fig.text(0.53, 0.827, r"$z=(x,1-x)$;  $c_1=0.05,\ c_2=0.65,\ p_1=0.35,\ p_2=0.80,\ \gamma=0.98,\ R=1$", ha="center", fontsize=11)
    fig.text(0.53, 0.078, "Above the dashed zero line, product 2 is preferred; below it, product 1 is preferred.", ha="center", fontsize=11)
    fig.text(0.53, 0.035, r"Curves cover the full mature boundary. For $\rho<0.5$, only a Cantor subset is limiting-reachable.", ha="center", fontsize=10, color="#52606d")
    save_figure(fig, "advantage_by_rho_overlay", destination)


def main():
    (DEST / "figures").mkdir(parents=True, exist_ok=True)
    (DEST / "data").mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "forgetting/Code/experiments/config_paper.json").read_text())
    primitives = Primitives.from_mapping({**config["seller"], "p0": 0.5, "gamma": config["rho_p0_comparative_statics"]["gamma"]})
    cases = solve_cases(primitives)
    make_figures(cases)
    manifest = dict(
        primitives=primitives.as_dict(), rhos=RHOS,
        expression="-delta_c + gamma*delta_p*(V_rho(T_success z)-V_rho(T_failure z))",
        domain="Mature boundary z=(x,1-x), x in [0,1]; x=(1-rho)*S",
        value_function="Optimal Bellman value, recomputed separately for each rho",
        solver="16001 uniform nodes, linear interpolation, iteration tolerance 1e-12",
        validation="Compared with 8001 nodes at all breakpoints of both interpolated advantage functions",
        reachable_set="Full interval for rho >= 0.5; depth-16 endpoint approximation otherwise",
        maximum_curve_change=max(c["summary"]["max_curve_change_8001_to_16001"] for c in cases),
        note="Grid comparison is a numerical stability check, not a rigorous discretization error bound.",
    )
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (DEST / "README.md").write_text(
        "# Product-2 advantage at p0=0.5\n\n"
        "The figures evaluate -delta_c + gamma*delta_p*[V_rho(T_success z)-V_rho(T_failure z)] "
        "on the long-run boundary z=(x,1-x). The horizontal coordinate is x=(1-rho)*S. "
        "These are boundary-state curves, not finite-interaction trajectories or the full two-dimensional state space.\n\n"
        "Positive values favor product 2; negative values favor product 1; zero is indifference. "
        "For rho<0.5 the limiting reachable set is a Cantor subset; the panel plot marks "
        "reachable endpoints and the summary reports a reachable-set maximum.\n\n"
        "Parameters and validation settings are in manifest.json. CSV data are in data/. "
        "Figures are provided as PNG and vector PDF.\n\n"
        "Reproduce from the repository root:\n\n"
        "```bash\nuv run python -B forgetting/Code/visualizations/plot_advantage_by_rho.py\n```\n"
    )
    print(f"Saved figures and data to {DEST}", flush=True)


if __name__ == "__main__":
    main()
