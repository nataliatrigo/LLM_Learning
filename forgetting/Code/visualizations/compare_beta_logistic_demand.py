"""Isolate the effect of logistic demand in the mature-boundary Bellman model.

Run: uv run python -B forgetting/Code/visualizations/compare_beta_logistic_demand.py
Only the demand vector changes between the two calls to the shared Bellman kernel.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json
import platform
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import scipy
from scipy.special import expit
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from forgetting.Code.model import Primitives, _bellman_update, boundary_demand

RHOS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999)
P = Primitives(p0=0.5, p1=0.35, p2=0.8, c1=0.05, c2=0.65, revenue=1, gamma=0.98)
DEST = ROOT / "forgetting/outputs/logistic_demand_comparison_p0_0.5"
TOLERANCE = 1e-12
CURVE_STABILITY = 5e-5
ROOT_STABILITY = 2e-5
BLUE, ORANGE = "#2468a0", "#d47a24"
RHO_COLORS = ("#2a72ad", "#54a1c4", "#24845b", "#dc9225", "#ce513d", "#8d66ad", "#573b75")


def slope(rho):
    delta = 1 - rho
    return 2 * np.sqrt(8 / np.pi) * np.sqrt(1 + 3 * delta) / ((1 + 2 * delta) * np.sqrt(delta))


def demands(x, rho):
    delta = 1 - rho
    exact = beta.sf(P.p0, 1 + x / delta, 1 + (1 - x) / delta)
    logistic = expit(slope(rho) * (x - 0.5))
    return exact, logistic


@dataclass
class Solution:
    rho: float
    model: str
    x: np.ndarray
    demand: np.ndarray
    value: np.ndarray
    iterations: int
    iteration_residual: float
    bellman_residual: float


def solve(x, demand, rho, model):
    """Same initialization, implementation, and tolerance for both demands."""
    value = np.zeros_like(x)
    for iteration in range(1, 100_001):
        updated, _ = _bellman_update(value, x, demand, P, rho, optimize=True)
        residual = float(np.max(np.abs(updated - value)))
        value = updated
        if residual <= TOLERANCE:
            break
    else:
        raise RuntimeError(f"Value iteration did not converge: {rho=}, {model=}")
    checked, gap = _bellman_update(value, x, demand, P, rho, optimize=True)
    solution = Solution(rho, model, x, demand, value, iteration, residual, float(np.max(np.abs(checked - value))))
    np.testing.assert_allclose(advantage(solution, x), gap, atol=2e-12, rtol=0)
    assert solution.bellman_residual <= 1.01 * TOLERANCE
    assert np.all(np.isfinite(value))
    return solution


def advantage(solution, x):
    success = np.interp(solution.rho * x + (1 - solution.rho), solution.x, solution.value)
    failure = np.interp(solution.rho * x, solution.x, solution.value)
    return -P.delta_c + P.gamma * P.delta_p * (success - failure)


def knots(solution):
    """All knots of a(x) induced by the linearly interpolated numerical value."""
    rho = solution.rho
    shifted = np.concatenate((solution.x / rho, (solution.x - (1 - rho)) / rho, [0, 1]))
    return np.unique(np.clip(shifted, 0, 1))


def detect_regions(x, a):
    """Find all zero points, zero segments, and strict positive components.

    Operates on any continuous piecewise-linear curve, allowing multiple
    disjoint regions, isolated zero touches, and intervals identically zero.
    No anticipated policy topology is imposed.
    """
    x, a = np.asarray(x), np.asarray(a)
    if len(x) < 2 or len(x) != len(a) or np.any(np.diff(x) <= 0):
        raise ValueError("Expected equal-length arrays and strictly increasing x")
    crossing = np.flatnonzero(((a[:-1] < 0) & (a[1:] > 0)) | ((a[:-1] > 0) & (a[1:] < 0)))
    roots = x[crossing] - a[crossing] * np.diff(x)[crossing] / np.diff(a)[crossing]
    roots = np.unique(np.concatenate((roots, x[a == 0])))
    edges = np.unique(np.concatenate(([x[0]], roots, [x[-1]])))
    intervals = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.interp((lo + hi) / 2, x, a) > 0:
            intervals.append(dict(lower=float(lo), upper=float(hi),
                lower_included=bool(lo == x[0] and a[0] > 0),
                upper_included=bool(hi == x[-1] and a[-1] > 0)))
    zero_intervals = []
    for index in np.flatnonzero((a[:-1] == 0) & (a[1:] == 0)):
        lo, hi = float(x[index]), float(x[index + 1])
        if zero_intervals and zero_intervals[-1][1] == lo:
            zero_intervals[-1][1] = hi
        else:
            zero_intervals.append([lo, hi])
    return roots.tolist(), intervals, zero_intervals


def describe(solution):
    x = knots(solution)
    a = advantage(solution, x)
    roots, intervals, zero_intervals = detect_regions(x, a)
    if roots:
        assert np.max(np.abs(advantage(solution, np.array(roots)))) < 1e-10
    best = int(np.argmax(a))
    return dict(max_advantage=float(a[best]), argmax=float(x[best]),
                zero_crossings=roots, positive_intervals=intervals,
                positive_region_count=len(intervals), zero_intervals=zero_intervals,
                structure=classify(intervals))


def classify(intervals):
    if not intervals:
        return "empty"
    if len(intervals) > 1:
        return "multiple"
    region = intervals[0]
    if region["lower_included"] and region["upper_included"]:
        return "full"
    if region["lower_included"]:
        return "left-anchored"
    if region["upper_included"]:
        return "right-anchored"
    return "interior"


def interval_text(intervals, digits=8):
    return " U ".join(
        f"{'[' if q['lower_included'] else '('}{q['lower']:.{digits}f}, {q['upper']:.{digits}f}{']' if q['upper_included'] else ')'}"
        for q in intervals) or "empty"


def solve_pair(rho, nodes):
    x = np.linspace(0, 1, nodes)
    exact, logistic = demands(x, rho)
    np.testing.assert_allclose(exact, boundary_demand(x, rho, P.p0), atol=1e-12, rtol=0)
    for demand in (exact, logistic):
        assert np.all(np.isfinite(demand)) and np.all((demand >= 0) & (demand <= 1))
        assert np.min(np.diff(demand)) >= -1e-14
        np.testing.assert_allclose(demand + demand[::-1], 1, atol=2e-12, rtol=0)
    return {name: solve(x, demand, rho, name) for name, demand in zip(("exact", "logistic"), (exact, logistic))}


def refinement_row(coarse, fine):
    old, new = describe(coarse), describe(fine)
    union = np.union1d(knots(coarse), knots(fine))
    curve_change = float(np.max(np.abs(advantage(coarse, union) - advantage(fine, union))))
    same_topology = old["structure"] == new["structure"] and old["positive_region_count"] == new["positive_region_count"]
    same_root_count = len(old["zero_crossings"]) == len(new["zero_crossings"])
    root_change = (max((abs(a - b) for a, b in zip(old["zero_crossings"], new["zero_crossings"])), default=0.0)
                   if same_root_count else None)
    stable = same_topology and same_root_count and curve_change <= CURVE_STABILITY and root_change <= ROOT_STABILITY
    return dict(rho=fine.rho, model=fine.model, coarse_nodes=len(coarse.x), fine_nodes=len(fine.x),
                advantage_sup_change=curve_change, max_advantage_change=abs(new["max_advantage"] - old["max_advantage"]),
                argmax_change=abs(new["argmax"] - old["argmax"]), max_root_change=root_change,
                same_region_count_and_type=same_topology, same_root_count=same_root_count, stable=stable,
                iterations=fine.iterations, bellman_residual=fine.bellman_residual)


def refined_pair(rho, initial_nodes, convergence):
    coarse = solve_pair(rho, initial_nodes)
    nodes = initial_nodes
    while nodes < 128_001:
        nodes = 2 * nodes - 1
        fine = solve_pair(rho, nodes)
        checks = [refinement_row(coarse[name], fine[name]) for name in ("exact", "logistic")]
        convergence.extend(checks)
        print(f"rho={rho:g}, nodes={nodes}: " + "; ".join(
            f"{name} max={describe(fine[name])['max_advantage']:.7f}, {describe(fine[name])['structure']}, grid change={check['advantage_sup_change']:.2e}"
            for name, check in zip(("exact", "logistic"), checks)), flush=True)
        if all(check["stable"] for check in checks) and (rho < 0.99 or nodes >= 32_001):
            return fine
        coarse = fine
    raise RuntimeError(f"Grid stability not established at {rho=}; inspect grid_convergence.csv")


def error_metrics(error):
    return dict(max=float(np.max(np.abs(error))), MAE=float(np.mean(np.abs(error))), RMSE=float(np.sqrt(np.mean(error**2))))


def summarize(pair, evaluation_x, destination):
    exact, logistic = pair["exact"], pair["logistic"]
    rho = exact.rho
    d_exact, d_logistic = demands(evaluation_x, rho)
    a_exact, a_logistic = advantage(exact, evaluation_x), advantage(logistic, evaluation_x)
    d_error, a_error = error_metrics(d_exact - d_logistic), error_metrics(a_exact - a_logistic)
    # The advantage difference is piecewise linear: its exact numerical supremum
    # lies at the union of both models' shifted interpolation knots.
    union = np.union1d(knots(exact), knots(logistic))
    a_error["max"] = float(np.max(np.abs(advantage(exact, union) - advantage(logistic, union))))
    descriptions = {name: describe(sol) for name, sol in pair.items()}
    e, l = descriptions["exact"], descriptions["logistic"]
    row = dict(rho=rho, demand_max_error=d_error["max"], demand_MAE=d_error["MAE"], demand_RMSE=d_error["RMSE"],
        exact_max_advantage=e["max_advantage"], logistic_max_advantage=l["max_advantage"],
        exact_argmax=e["argmax"], logistic_argmax=l["argmax"],
        exact_zero_crossings=json.dumps(e["zero_crossings"]), logistic_zero_crossings=json.dumps(l["zero_crossings"]),
        exact_positive_intervals=interval_text(e["positive_intervals"]), logistic_positive_intervals=interval_text(l["positive_intervals"]),
        advantage_max_error=a_error["max"], advantage_RMSE=a_error["RMSE"],
        exact_positive_region_count=e["positive_region_count"], logistic_positive_region_count=l["positive_region_count"],
        exact_structure=e["structure"], logistic_structure=l["structure"],
        grid_nodes=len(exact.x), logistic_lambda=float(slope(rho)))
    pd.DataFrame(dict(x=evaluation_x, exact_demand=d_exact, logistic_demand=d_logistic,
        demand_error=d_exact-d_logistic, exact_advantage=a_exact, logistic_advantage=a_logistic,
        advantage_error=a_exact-a_logistic)).to_csv(destination / "data" / f"curves_rho_{rho:g}.csv", index=False)
    np.savez_compressed(destination / "data" / f"solutions_rho_{rho:g}.npz", x=exact.x,
        exact_demand=exact.demand, logistic_demand=logistic.demand, exact_value=exact.value, logistic_value=logistic.value)
    for name, sol in pair.items():
        descriptions[name].update(iterations=sol.iterations, iteration_residual=sol.iteration_residual,
            bellman_residual=sol.bellman_residual, grid_nodes=len(sol.x))
    (destination / "data" / f"policy_rho_{rho:g}.json").write_text(json.dumps(descriptions, indent=2) + "\n")
    return row


def style(ax, zero=True):
    ax.set_xlim(0, 1)
    ax.grid(axis="y", color="#e3e7ec", lw=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#b3bbc4")
    if zero:
        ax.axhline(0, color="#47515c", lw=1, ls=":")


def save(fig, destination, stem):
    for suffix in ("png", "pdf"):
        fig.savefig(destination / "figures" / f"{stem}.{suffix}", dpi=170, facecolor="white")
    plt.close(fig)


def model_legend(ax):
    ax.set_axis_off()
    handles = [Line2D([0], [0], color=BLUE, lw=2, label="Exact Beta"),
               Line2D([0], [0], color=ORANGE, lw=2, ls="--", label="Logistic")]
    ax.legend(handles=handles, loc="center", frameon=False, fontsize=14)


def make_figures(pairs, summary, evaluation_x, destination):
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12})
    rows = summary.set_index("rho")
    for kind in ("demand", "demand_error", "advantage"):
        fig, axes = plt.subplots(2, 4, figsize=(15, 8), sharex=True, sharey=True)
        fig.subplots_adjust(left=0.07, right=0.98, top=0.82, bottom=0.16, hspace=0.34, wspace=0.16)
        for ax, rho in zip(axes.flat, RHOS):
            pair = pairs[rho]
            exact_d, log_d = demands(evaluation_x, rho)
            style(ax, zero=kind != "demand")
            ax.set_title(rf"$\rho={rho:g}$", pad=10)
            ax.set_xticks([0, 0.5, 1])
            if kind == "demand":
                ax.plot(evaluation_x, exact_d, color=BLUE, lw=2)
                ax.plot(evaluation_x, log_d, color=ORANGE, lw=2, ls="--")
                ax.set_ylim(-0.04, 1.04)
            elif kind == "demand_error":
                ax.plot(evaluation_x, exact_d-log_d, color=BLUE, lw=2)
            else:
                for name, color, ls in (("exact", BLUE, "-"), ("logistic", ORANGE, "--")):
                    sol = pair[name]
                    x = knots(sol)
                    ax.plot(x, advantage(sol, x), color=color, lw=2, ls=ls)
                    roots = describe(sol)["zero_crossings"]
                    ax.scatter(roots, np.zeros(len(roots)), s=25, facecolor="white", edgecolor=color, zorder=4)
        if kind == "demand_error":
            axes.flat[-1].set_axis_off()
            axes.flat[-1].text(0.5, 0.5, "Error = Exact Beta − Logistic\nSame scale in every panel", transform=axes.flat[-1].transAxes, ha="center", va="center", fontsize=13)
        else:
            model_legend(axes.flat[-1])
        title = {"demand": "Demand: exact Beta and logistic approximation", "demand_error": "Demand approximation error", "advantage": "Product-2 advantage: exact Beta versus logistic demand"}[kind]
        ylabel = {"demand": r"Demand $D_\rho(x)$", "demand_error": r"$D_\rho(x)-\widetilde D_\rho(x)$", "advantage": r"Advantage $a_\rho(x)$"}[kind]
        fig.suptitle(title + r"  |  $p_0=0.5$", y=0.965, fontsize=20)
        fig.text(0.52, 0.90, r"$R=1,\ c_1=0.05,\ c_2=0.65,\ p_1=0.35,\ p_2=0.80,\ \gamma=0.98$", ha="center", fontsize=12)
        fig.supxlabel(r"Normalized state $x=(1-\rho)S$, on $z=(x,1-x)$", y=0.074, fontsize=13)
        fig.supylabel(ylabel, x=0.012, fontsize=13)
        fig.text(0.52, 0.024, "Only the demand function changes; both models use the same Bellman equation and numerical method.", ha="center", fontsize=11)
        save(fig, destination, f"{kind}_comparison_panels")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.065, right=0.98, top=0.75, bottom=0.18, wspace=0.12)
    for ax, name, title in zip(axes, ("exact", "logistic"), ("Exact Beta demand", "Logistic demand approximation")):
        style(ax)
        ax.set_title(title, pad=12)
        for rho, color in zip(RHOS, RHO_COLORS):
            sol = pairs[rho][name]
            x = knots(sol)
            ax.plot(x, advantage(sol, x), color=color, lw=2, label=rf"$\rho={rho:g}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.52, 0.83), ncols=7, frameon=False, fontsize=11)
    fig.suptitle(r"Product-2 advantage across forgetting rates  |  $p_0=0.5$", y=0.965, fontsize=21)
    fig.supxlabel(r"Normalized state $x=(1-\rho)S$", y=0.085, fontsize=13)
    fig.supylabel(r"Advantage $a_\rho(x)$", x=0.009, fontsize=13)
    fig.text(0.52, 0.025, "Identical axes. Product 2 is strictly optimal above zero; zero crossings are detected without imposing a policy shape.", ha="center", fontsize=11)
    save(fig, destination, "advantage_by_model")

    for rho in RHOS:
        fig, ax = plt.subplots(figsize=(10, 6.5))
        fig.subplots_adjust(left=0.11, right=0.97, top=0.75, bottom=0.20)
        style(ax)
        for name, color, ls, title in (("exact", BLUE, "-", "Exact Beta"), ("logistic", ORANGE, "--", "Logistic")):
            sol = pairs[rho][name]
            x = knots(sol)
            ax.plot(x, advantage(sol, x), color=color, lw=2.3, ls=ls, label=title)
            roots = describe(sol)["zero_crossings"]
            ax.scatter(roots, np.zeros(len(roots)), s=35, facecolor="white", edgecolor=color, zorder=4)
        ax.set_xlabel(r"Normalized state $x=(1-\rho)S$", labelpad=10)
        ax.set_ylabel(r"Advantage $a_\rho(x)$")
        fig.suptitle(rf"Product-2 advantage: exact Beta vs logistic  |  $\rho={rho:g}$", y=0.96, fontsize=18)
        fig.text(0.54, 0.872, rf"$p_0=0.5$   |   max curve difference = {rows.loc[rho, 'advantage_max_error']:.5f}", ha="center", fontsize=12)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.54, 0.811), ncols=2, frameon=False)
        fig.text(0.54, 0.075, "Circles mark zero crossings. Positive regions are detected from the solved curves.", ha="center", fontsize=11)
        fig.text(0.54, 0.032, "Demand is the only model input changed. Both curves use the same grid and convergence tolerance.", ha="center", fontsize=10, color="#52606d")
        save(fig, destination, f"advantage_overlay_rho_{rho:g}")


def write_report(summary, convergence, destination, manifest):
    lines = ["# Exact Beta versus logistic demand, p0=0.5", "",
        "Both versions use the original shared `_bellman_update` function with optimization over both products. "
        "Only its demand vector changes. The value function is initialized to zero and computed independently for each model.", "",
        "Parameters: R=1, c1=0.05, c2=0.65, p1=0.35, p2=0.80, gamma=0.98, p0=0.5. "
        "The state is the mature boundary z=(x,1-x), x=(1-rho)S in [0,1]. All tested rho are at least 0.5, so this whole interval is limiting-reachable.", "",
        "Exact demand uses scipy.stats.beta.sf; logistic demand uses scipy.special.expit and the supplied lambda formula.", "",
        "## Demand errors", "", "| rho | Maximum absolute error | MAE | RMSE |", "|---:|---:|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.rho:g} | {row.demand_max_error:.8f} | {row.demand_MAE:.8f} | {row.demand_RMSE:.8f} |")
    lines += ["", "## Detected policy structure", "", "| rho | Exact positive intervals | Logistic positive intervals | Exact maximum | Logistic maximum |", "|---:|---|---|---:|---:|"]
    for row in summary.itertuples():
        lines.append(f"| {row.rho:g} | {row.exact_positive_intervals} | {row.logistic_positive_intervals} | {row.exact_max_advantage:.7f} | {row.logistic_max_advantage:.7f} |")
    lines += ["", "Parentheses exclude zero-advantage endpoints; square brackets include endpoints with strictly positive advantage. "
        "Every numerical zero and every connected positive region was detected from the piecewise-linear advantage, including all successor-shifted interpolation knots.", ""]
    differing = summary[summary.exact_structure != summary.logistic_structure]
    lines += ["Exact sequence on sampled rho: " + ", ".join(summary.exact_structure) + ".", "",
              "Logistic sequence on sampled rho: " + ", ".join(summary.logistic_structure) + ".", ""]
    if len(differing):
        lines.append("The region type differs at rho=" + ", ".join(f"{rho:g}" for rho in differing.rho) + ". Thus the approximation does not preserve the policy structure at every tested rho, even if it reproduces the broad sequence.")
    else:
        lines.append("The detected region type agrees at every sampled rho.")
    last = summary.iloc[-1]
    lines += ["", f"At the largest tested rho={last.rho:g}, the exact region is {last.exact_structure} and the logistic region is {last.logistic_structure}. "
        "These samples do not identify the precise transition rho or establish a claim for all rho approaching one.", "",
        "The two models generally retain similar curve shapes, but their heights, maximizers, zero crossings, and interval endpoints need not agree. "
        f"The largest observed absolute advantage difference is {summary.advantage_max_error.max():.8f}.", "",
        "## Numerical checks", "",
        f"Initial grid: {manifest['initial_nodes']} uniform points; both models are refined together. "
        f"Final grids: {', '.join(f'{r.rho:g}: {r.grid_nodes}' for r in summary.itertuples())}. "
        f"Value iteration tolerance: {TOLERANCE:g}. Same interpolation (np.interp), initialization, and implementation for both demands.", "",
        f"Refinement requires unchanged region type/count and root count, advantage sup change <= {CURVE_STABILITY:g}, "
        f"and root movement <= {ROOT_STABILITY:g}. Rho=0.99 and 0.999 require at least 32001 points. "
        "The full history is in data/grid_convergence.csv. Grid checks are empirical stability checks, not rigorous discretization-error bounds.", "",
        f"Demand metrics and advantage RMSE use the same {manifest['evaluation_nodes']}-point uniform evaluation grid, including endpoints. "
        "Advantage sup errors and maxima use all numerical interpolation breakpoints, not only the plotting grid. "
        "Argmax reports the first maximizer in increasing x order when numerical ties occur.", "",
        "Exact demand is cross-checked against the repository's scipy.special.betaincc demand. "
        "Bellman residuals, finite outputs, demand symmetry, and the algebraic advantage identity are checked for both models.", "",
        "## Files", "", "- data/summary.csv: all requested columns, plus region counts, classifications, final grid sizes, and lambda.",
        "- data/grid_convergence.csv: coarse/fine changes for both models.",
        "- data/curves_rho_*.csv: demand and advantage curves and signed errors on a common grid.",
        "- data/solutions_rho_*.npz: final solver grid, demands, and independently solved value functions.",
        "- data/policy_rho_*.json: full precision policy geometry, boundary inclusion, zero segments, and solver diagnostics.",
        "- figures/: demand overlays, demand errors, advantage overlays, and the side-by-side model figure; PNG and vector PDF.",
        "", "Reproduce:", "", "```bash", "uv run python -B forgetting/Code/visualizations/compare_beta_logistic_demand.py", "```", ""]
    (destination / "report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEST)
    parser.add_argument("--initial-nodes", type=int, default=8001)
    parser.add_argument("--evaluation-nodes", type=int, default=40001)
    args = parser.parse_args()
    if args.initial_nodes < 5001 or args.evaluation_nodes < 5001:
        parser.error("Use at least 5001 nodes for both grids")
    destination = args.output_dir
    for folder in ("data", "figures"):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    pairs, rows, convergence = {}, [], []
    evaluation_x = np.linspace(0, 1, args.evaluation_nodes)
    for rho in RHOS:
        pairs[rho] = refined_pair(rho, args.initial_nodes, convergence)
        rows.append(summarize(pairs[rho], evaluation_x, destination))
        pd.DataFrame(rows).to_csv(destination / "data/summary.csv", index=False)
        pd.DataFrame(convergence).to_csv(destination / "data/grid_convergence.csv", index=False)
    summary = pd.DataFrame(rows)
    manifest = dict(primitives=P.as_dict(), rhos=RHOS, initial_nodes=args.initial_nodes,
        evaluation_nodes=args.evaluation_nodes, iteration_tolerance=TOLERANCE,
        curve_stability_threshold=CURVE_STABILITY, root_stability_threshold=ROOT_STABILITY,
        logistic_lambda="2*sqrt(8/pi)*sqrt(1+3*(1-rho))/((1+2*(1-rho))*sqrt(1-rho))",
        exact_demand="scipy.stats.beta.sf(p0, 1+x/(1-rho), 1+(1-x)/(1-rho))",
        changed_model_input="demand only", value_initialization="zero",
        interpolation="numpy.interp (linear)", shared_bellman_kernel="forgetting.Code.model._bellman_update, optimize=True",
        bellman_source_sha256=hashlib.sha256((ROOT / "forgetting/Code/model.py").read_bytes()).hexdigest(),
        final_nodes={str(r.rho):int(r.grid_nodes) for r in summary.itertuples()},
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, matplotlib=matplotlib.__version__),
        source_file=str(Path(__file__).resolve()))
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    make_figures(pairs, summary, evaluation_x, destination)
    write_report(summary, convergence, destination, manifest)
    print(summary.to_string(index=False), flush=True)
    print(f"Saved experiment to {destination}", flush=True)


if __name__ == "__main__":
    main()
