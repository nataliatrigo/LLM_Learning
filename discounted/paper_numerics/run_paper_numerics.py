"""Reproduce the baseline numerical illustration used in the paper."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/llm_learning_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stationary_solver import Parameters, solve_stationary_truncation


HERE = Path(__file__).resolve().parent
OUTPUTS = HERE / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
OUTER_GRIDS = (800, 1200, 1600)
N_REPORT = 600


def tolerance(values: np.ndarray) -> float:
    return max(1e-10, 1e-8 * max(1.0, float(np.max(np.abs(values)))))


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    cuts = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(indices) - 1]
    return [(int(indices[i]), int(indices[j])) for i, j in zip(starts, ends)]


def classify(solution: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    params = solution["parameters"]
    states: list[dict] = []
    diagonals: list[dict] = []
    for n, layer in solution["layers"].items():
        advantage = layer["advantage"]
        tol = tolerance(advantage)
        robust2 = advantage > tol
        robust1 = advantage < -tol
        tied = ~(robust2 | robust1)
        components = runs(robust2)
        robust_violation = False
        ambiguous = False
        if len(components) > 1:
            for (_, left_end), (right_start, _) in zip(components, components[1:]):
                if robust1[left_end + 1 : right_start].any():
                    robust_violation = True
                elif tied[left_end + 1 : right_start].any():
                    ambiguous = True
        for s in range(n + 1):
            states.append(
                {
                    "n": n,
                    "S": s,
                    "F": n - s,
                    "m": (s + 1) / (n + 2),
                    "value": layer["value"][s],
                    "G": layer["gap"][s],
                    "advantage": advantage[s],
                    "classification": "product2" if robust2[s] else ("product1" if robust1[s] else "tie"),
                }
            )
        active = np.flatnonzero(robust2)
        diagonals.append(
            {
                "n": n,
                "active": bool(len(active)),
                "robust_product2_states": int(len(active)),
                "components": int(len(components)),
                "ambiguous": bool(ambiguous or tied.any()),
                "robust_interval_violation": bool(robust_violation),
                "lower_S": int(active[0]) if len(active) else np.nan,
                "upper_S": int(active[-1]) if len(active) else np.nan,
                "lower_m": (active[0] + 1) / (n + 2) if len(active) else np.nan,
                "upper_m": (active[-1] + 1) / (n + 2) if len(active) else np.nan,
                "maximum_G": float(np.max(layer["gap"])),
                "tolerance": tol,
            }
        )
    return pd.DataFrame(states), pd.DataFrame(diagonals).sort_values("n")


def convergence_table(solutions: list[dict]) -> pd.DataFrame:
    rows = []
    for smaller, larger in zip(solutions[:-1], solutions[1:]):
        max_v = max(
            float(np.max(np.abs(smaller["layers"][n]["value"] - larger["layers"][n]["value"])))
            for n in range(N_REPORT + 1)
        )
        max_g = max(
            float(np.max(np.abs(smaller["layers"][n]["gap"] - larger["layers"][n]["gap"])))
            for n in range(N_REPORT + 1)
        )
        disagreements = 0
        states = 0
        for n in range(N_REPORT + 1):
            first = smaller["layers"][n]["advantage"] > tolerance(smaller["layers"][n]["advantage"])
            second = larger["layers"][n]["advantage"] > tolerance(larger["layers"][n]["advantage"])
            disagreements += int(np.sum(first != second))
            states += n + 1
        _, smaller_diagonal = classify(smaller)
        _, diagonal = classify(larger)
        active = diagonal.loc[diagonal.active, "n"]
        last_active = int(active.max()) if len(active) else -1
        endpoint_comparison = smaller_diagonal.merge(diagonal, on="n", suffixes=("_small", "_large"))
        endpoint_comparison = endpoint_comparison[endpoint_comparison.active_small | endpoint_comparison.active_large]
        lower_change = float(np.nanmax(np.abs(endpoint_comparison.lower_S_small - endpoint_comparison.lower_S_large)))
        upper_change = float(np.nanmax(np.abs(endpoint_comparison.upper_S_small - endpoint_comparison.upper_S_large)))
        rows.append(
            {
                "smaller_outer": smaller["outer_diagonal"],
                "larger_outer": larger["outer_diagonal"],
                "N_report": N_REPORT,
                "maximum_abs_value_difference": max_v,
                "maximum_abs_gap_difference": max_g,
                "policy_disagreements": disagreements,
                "policy_disagreement_fraction": disagreements / states,
                "maximum_bellman_residual": larger["maximum_bellman_residual"],
                "last_active_diagonal": last_active,
                "maximum_lower_endpoint_change": lower_change,
                "maximum_upper_endpoint_change": upper_change,
                "distance_to_outer_boundary": larger["outer_diagonal"] - last_active,
            }
        )
    return pd.DataFrame(rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
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


def plot_policy(states: pd.DataFrame, diagonals: pd.DataFrame, params: Parameters) -> None:
    active = states[states.classification == "product2"]
    last_active = int(diagonals.loc[diagonals.active, "n"].max())
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.scatter(active.n, active.m, s=3, color="#0891b2", alpha=0.8, rasterized=True, label="Product 2 state")
    n = np.arange(N_REPORT + 1)
    sd = np.sqrt(params.p0 * (1 - params.p0) / (n + 2))
    ax.plot(n, params.p0 + sd, color="#f59e0b", lw=1, ls="--", label=r"Statistical reference: $p_0\pm\sigma_n$")
    ax.plot(n, params.p0 - sd, color="#f59e0b", lw=1, ls="--")
    ax.axhline(params.p0, color="#334155", lw=1.1, label=r"Outside-option quality $p_0$")
    ax.axvline(last_active, color="#be123c", lw=1, ls=":", label=f"Last active diagonal: {last_active}")
    ax.set(xlabel=r"History length $n=S+F$", ylabel=r"Posterior mean $m=(S+1)/(n+2)$", xlim=(0, N_REPORT), ylim=(0, 1))
    ax.legend(loc="upper right", fontsize=8)
    fig.savefig(FIGURES / "baseline_policy_nm.pdf")
    fig.savefig(FIGURES / "baseline_policy_nm.png")
    plt.close(fig)


def plot_gap(solution: dict, diagonals: pd.DataFrame, params: Parameters) -> list[int]:
    active = diagonals[diagonals.active]
    last = int(active.n.max())
    widest = int(active.loc[active.robust_product2_states.idxmax(), "n"])
    selected = sorted(set([25, 100, 250, widest, max(0, last - 25), last, min(N_REPORT, last + 25), N_REPORT]))
    fig, axes = plt.subplots(2, 4, figsize=(10.5, 5.4), sharey=True)
    for ax, n in zip(axes.flat, selected):
        gap = solution["layers"][n]["gap"]
        ax.plot(np.arange(n + 1), gap, color="#0369a1", lw=1.25)
        ax.axhline(params.threshold, color="#be123c", lw=1, ls="--")
        ax.set_title(f"$n={n}$")
        ax.set_xlabel("Successes $S$")
    axes[0, 0].set_ylabel("Continuation gap $G$")
    axes[1, 0].set_ylabel("Continuation gap $G$")
    handles = [
        plt.Line2D([], [], color="#0369a1", label=r"$G(S,n-S)$"),
        plt.Line2D([], [], color="#be123c", ls="--", label=r"Threshold $h=\Delta c/\Delta p$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURES / "continuation_gap_diagonals.pdf")
    fig.savefig(FIGURES / "continuation_gap_diagonals.png")
    plt.close(fig)
    return selected


def plot_boundaries(diagonals: pd.DataFrame, params: Parameters) -> None:
    active = diagonals[diagonals.active & ~diagonals.robust_interval_violation]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(active.n, active.lower_m, color="#2563eb", lw=1.4, label="Empirical lower boundary")
    ax.plot(active.n, active.upper_m, color="#dc2626", lw=1.4, label="Empirical upper boundary")
    ax.axhline(params.p0, color="#334155", lw=1, ls="--", label=r"$p_0$")
    ax.set(xlabel=r"History length $n=S+F$", ylabel="Posterior mean", ylim=(0, 1))
    ax.legend()
    fig.savefig(FIGURES / "empirical_boundaries.pdf")
    fig.savefig(FIGURES / "empirical_boundaries.png")
    plt.close(fig)


def plot_extinction_margin(solution: dict, diagonals: pd.DataFrame, params: Parameters) -> None:
    """Plot the diagonal extinction margin and centered gaps near extinction."""
    selected = [410, 434, 435, 436, 460]
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(7.6, 7.0))

    margin = diagonals.maximum_G - params.threshold
    window = diagonals.n.between(350, 500)
    top.plot(
        diagonals.loc[window, "n"],
        margin.loc[window],
        color="#0369a1",
        lw=1.5,
        label=r"Maximum margin $M_n$",
    )
    top.axhline(0.0, color="#be123c", lw=1.1, ls="--", label="Investment threshold")
    top.axvline(435, color="#475569", lw=1.0, ls=":", label="Last active diagonal: 435")
    top.scatter(
        selected,
        margin.iloc[selected],
        color="#0f766e",
        s=14,
        zorder=3,
        label="Selected diagonals",
    )
    top.set(
        xlim=(350, 500),
        xlabel=r"History length $n=S+F$",
        ylabel=r"$M_n=\max_{S+F=n}G(S,F)-h$",
    )
    top.legend(loc="upper right", fontsize=8)

    colors = plt.cm.viridis(np.linspace(0.08, 0.9, len(selected)))
    for color, n in zip(colors, selected):
        gap = solution["layers"][n]["gap"] - params.threshold
        successes = np.arange(n + 1)
        posterior_mean = (successes + 1) / (n + 2)
        bottom.plot(posterior_mean, gap, color=color, lw=1.35, label=f"$n={n}$")
    bottom.axhline(0.0, color="#be123c", lw=1.1, ls="--")
    bottom.axvline(params.p0, color="#475569", lw=1.0, ls=":", label=r"$p_0=0.5$")
    bottom.set(
        xlim=(0.44, 0.56),
        ylim=(-0.02, 0.04),
        xlabel=r"Posterior mean $m=(S+1)/(n+2)$",
        ylabel=r"Centered gap $G(S,n-S)-h$",
    )
    bottom.legend(loc="upper right", ncol=2, fontsize=8)

    fig.tight_layout()
    fig.savefig(FIGURES / "extinction_margin_diagnostic.pdf")
    fig.savefig(FIGURES / "extinction_margin_diagnostic.png")
    plt.close(fig)


def markdown_table(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    rows = []
    for row in frame.itertuples(index=False, name=None):
        rows.append([f"{value:.6g}" if isinstance(value, float) else str(value) for value in row])
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def write_report(convergence: pd.DataFrame, diagonals: pd.DataFrame, selected: list[int], params: Parameters) -> None:
    active = diagonals[diagonals.active]
    report = f"""# Paper Numerical Illustration

## Numerical inventory and method

The baseline paper computation uses backward recursion on the rearranged stationary Bellman equation, with zero terminal value on diagonal `N_outer+1`. It is a truncated-state approximation, not an exact stationary solution. The older `discounted/DP/exact_dp.py` instead performs finite-calendar-horizon backward induction and is not used for the main figures.

Baseline: `p0={params.p0}`, `p1={params.p1}`, `p2={params.p2}`, `c1={params.c1}`, `c2={params.c2}`, `R={params.revenue}`, `gamma={params.gamma}`. These values agree with the current repository defaults.

## Convergence

{markdown_table(convergence)}

The reliable reported interior is `n<=600`. The largest solve has outer diagonal `1600`. The empirical last active diagonal is `{int(active.n.max())}`, leaving `{1600-int(active.n.max())}` diagonals to the terminal boundary.

## Interval diagnostics

- Tested diagonals: `{len(diagonals)}`.
- Active diagonals: `{len(active)}`.
- Active diagonals with one robust interval: `{int((active.components == 1).sum())}`.
- Diagonals containing numerical ties: `{int(diagonals.ambiguous.sum())}`.
- Robust interval violations: `{int(diagonals.robust_interval_violation.sum())}`.
- Active diagonals containing gaps: `{int((active.components > 1).sum())}`.
- Empirical last active diagonal: `{int(active.n.max())}`.

The action tolerance is `max(1e-10, 1e-8 max(1, max|advantage|))` separately on each diagonal. A violation requires two robust product-2 components separated by a robust product-1 state.

The continuation-gap panels use diagonals `{selected}` and plot the unsmoothed computed sequences.

## Existing robustness evidence

The prior reproducible search in `discounted/DP/analysis/test_gap_unimodality.py` examined 1175 configurations across all three quality regimes and found no robust within-diagonal interval violation. However, 78 first-pass cases had policy changes between the small outer grids, so the broad search is supporting evidence rather than a claim that all 1175 policies were fully converged. The baseline and every candidate violation received stronger outer-grid checks.

## Interpretation

- **Analytical:** the paper proves localization in an outer shrinking collar and eventual extinction.
- **Numerical approximation:** the stationary truncated-state recursion produces the plotted baseline policy and continuation gaps.
- **Empirical regularity:** every stable active baseline diagonal is a single product-2 interval.
- **Open:** a general analytical proof of the interval property is not provided.
"""
    (HERE / "REPORT_paper_numerics.md").write_text(report)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    configure_style()
    params = Parameters()
    solutions = [solve_stationary_truncation(params, outer, N_REPORT) for outer in OUTER_GRIDS]
    convergence = convergence_table(solutions)
    states, diagonals = classify(solutions[-1])
    convergence.to_csv(TABLES / "convergence.csv", index=False)
    diagonals.to_csv(TABLES / "interval_diagnostics.csv", index=False)
    states.to_csv(TABLES / "baseline_states.csv", index=False)
    plot_policy(states, diagonals, params)
    selected = plot_gap(solutions[-1], diagonals, params)
    plot_boundaries(diagonals, params)
    plot_extinction_margin(solutions[-1], diagonals, params)
    write_report(convergence, diagonals, selected, params)
    print((HERE / "REPORT_paper_numerics.md").read_text())


if __name__ == "__main__":
    main()
