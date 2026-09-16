"""Visualize the mature reachable set and distinguish geometry from policy gaps."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from forgetting.run_experiments import configure_style, read_config
from forgetting.model import (
    Primitives, BoundarySolution, boundary_summary,
    solve_boundary, validate_rho,
)

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
COLORS = {"product1": "#64748b", "product2": "#059669", "critical": "#d97706"}


def reachable_cover(rho: float, depth: int) -> np.ndarray:
    """Closed cylinder intervals containing K_rho, not finite-time states.

    For rho < 1/2, omitted gaps are exact exclusions; remaining intervals
    still contain unresolved smaller gaps. Each interval has width rho**depth.
    For rho >= 1/2, the full interval is the exact attractor.
    """
    validate_rho(rho)
    if depth < 0:
        raise ValueError("depth must be nonnegative")
    if depth == 0 or rho >= .5:
        return np.array([[0., 1.]])
    intervals = np.array([[0., 1.]])
    for _ in range(depth):
        intervals = np.vstack((rho * intervals, rho * intervals + (1-rho)))
    return intervals


def policy_intervals(solution: BoundarySolution, tolerance: float) -> list[tuple[float, float, str]]:
    """Partition the full boundary by the interpolated action-value difference."""
    x, values = solution.coordinate, solution.advantage
    edges = [0., 1.]
    for level in (-tolerance, tolerance):
        shifted = values - level
        for i in np.flatnonzero(shifted[:-1] * shifted[1:] < 0):
            edges.append(float(x[i] - shifted[i] * (x[i+1]-x[i]) / (shifted[i+1]-shifted[i])))
        edges.extend(x[shifted == 0].tolist())
    edges = np.unique(edges)
    intervals = []
    for lower, upper in zip(edges[:-1], edges[1:]):
        value = float(np.interp((lower+upper)/2, x, values))
        action = "product2" if value > tolerance else "product1" if value < -tolerance else "critical"
        intervals.append((float(lower), float(upper), action))
    return intervals


def intersect_intervals(cover: np.ndarray, lower: float, upper: float) -> np.ndarray:
    clipped = np.column_stack((np.maximum(cover[:, 0], lower), np.minimum(cover[:, 1], upper)))
    return clipped[clipped[:, 1] - clipped[:, 0] > 8 * np.finfo(float).eps]


def draw_intervals(ax, intervals: np.ndarray, row: int, color: str) -> None:
    if len(intervals):
        lines = np.stack((np.column_stack((intervals[:, 0], np.full(len(intervals), row))),
                          np.column_stack((intervals[:, 1], np.full(len(intervals), row)))), axis=1)
        ax.add_collection(LineCollection(lines, colors=color, linewidths=12,
                                        capstyle="butt", rasterized=True))
        # Keep tiny cylinders visible as location marks instead of letting
        # rasterization erase an entire nonempty reachable set.
        tiny = intervals[intervals[:, 1] - intervals[:, 0] < .001]
        if len(tiny):
            ax.plot(tiny[:, 0], np.full(len(tiny), row), "|", color=color,
                    markersize=12, markeredgewidth=.4, rasterized=True)


def save(fig, directory: Path, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{stem}.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--p0", type=float, default=.9)
    parser.add_argument("--c2", type=float, help="override c2 without editing the configuration")
    parser.add_argument("--rhos", type=float, nargs="+", default=[.2, .35, .43, .45, .47, .49, .5, .7])
    parser.add_argument("--depth", type=int, default=10)
    parser.add_argument("--focus-rho", type=float, default=.45)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "reachable_states")
    args = parser.parse_args()
    if not 1 <= args.depth <= 18:
        parser.error("--depth must lie between 1 and 18")
    config = read_config(args.config)
    primitives = Primitives.from_mapping({**config["seller"], "p0": args.p0,
        "gamma": config["rho_p0_comparative_statics"]["gamma"]})
    if args.c2 is not None:
        primitives = replace(primitives, c2=args.c2)
    rhos = sorted(set(args.rhos))
    for rho in [*rhos, args.focus_rho]:
        validate_rho(rho)
    directory = args.output_dir
    directory.mkdir(parents=True, exist_ok=True)
    configure_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, max(4, .48 * len(rhos))),
                             sharex=True, sharey=True, layout="constrained")
    summaries, cover_rows, policy_rows = [], [], []
    solver = config["solver"]
    for row, rho in enumerate(rhos):
        solution = solve_boundary(primitives, rho, central_nodes=int(solver["central_nodes"]),
            outer_nodes=int(solver["outer_nodes"]), half_widths=float(solver["half_widths"]),
            tolerance=float(solver["tolerance"]), max_iterations=int(solver["max_iterations"]))
        summary = boundary_summary(solution, action_tolerance=float(solver["action_tolerance"]),
                                   reachable_depth=int(solver["reachable_depth"]))
        summary["cover_interval_width"] = rho**args.depth if rho < .5 else 0.
        summaries.append(summary)
        cover = reachable_cover(rho, args.depth)
        cover_rows.extend({"rho": rho, "lower": a, "upper": b} for a,b in cover)
        draw_intervals(axes[0], cover, row, "#2563eb")
        for lower, upper, action in policy_intervals(solution, float(solver["action_tolerance"])):
            policy_rows.append({"rho": rho, "lower": lower, "upper": upper, "action": action})
            draw_intervals(axes[1], np.array([[lower, upper]]), row, COLORS[action])
            draw_intervals(axes[2], intersect_intervals(cover, lower, upper), row, COLORS[action])
    axes[0].set_yticks(range(len(rhos)), [f"{rho:g}" for rho in rhos])
    axes[0].set_ylabel(r"Factor de memoria $\rho$")
    titles = [r"(a) Cubierta de $\mathcal{K}_\rho$",
              "(b) Política en toda la frontera", "(c) Política sobre la cubierta"]
    for ax, title in zip(axes, titles):
        ax.set(xlim=(-.01, 1.01), ylim=(len(rhos)-.5, -.5), title=title,
               xlabel=r"Estado $x_s$ (con $x_f=1-x_s$)")
        ax.set_xticks([0, .25, .5, .75, 1])
        ax.grid(axis="x", alpha=.3)
        ax.grid(axis="y", visible=False)
    handles = [Line2D([0], [0], color=COLORS[k], lw=5, label=v) for k,v in
               [("product1", "Producto 1"), ("product2", "Producto 2"), ("critical", "Cerca de indiferencia")]]
    fig.legend(handles=handles, loc="outside lower center", ncols=3)
    fig.suptitle(f"Estados alcanzables y decisión: $p_0={primitives.p0:g}$, $c_2={primitives.c2:g}$, "
                 f"$\\gamma={primitives.gamma:g}$\nCubierta exterior de profundidad {args.depth}; "
                 "marcas finas para intervalos subpíxel; huecos más pequeños sin resolver",
                 fontsize=10)
    save(fig, directory, "reachable_states_and_policy")

    fig, ax = plt.subplots(figsize=(9, 4.3), layout="constrained")
    levels = range(min(args.depth, 6)+1)
    for level in levels:
        draw_intervals(ax, reachable_cover(args.focus_rho, level), level, "#2563eb")
    ax.set_yticks(list(levels), [f"d = {d}" for d in levels])
    ax.set(xlim=(-.01, 1.01), ylim=(len(levels)-.5, -.5), xlabel=r"Estado normalizado $x_s$",
           title=rf"Construcción de $\mathcal{{K}}_\rho$ con $\rho={args.focus_rho:g}$: cubiertas sucesivas")
    ax.grid(axis="y", visible=False)
    if args.focus_rho < .5:
        a, b = args.focus_rho, 1-args.focus_rho
        ax.axvspan(a, b, color="#fca5a5", alpha=.25, ymin=0, ymax=(len(levels)-1)/len(levels))
        ax.text((a+b)/2, (len(levels)-1)/2, f"Hueco\n({a:g}, {b:g})", ha="center", va="center", fontsize=9)
    save(fig, directory, "reachable_set_construction")
    pd.DataFrame(summaries).to_csv(directory / "diagnostics.csv", index=False)
    pd.DataFrame(cover_rows).to_csv(directory / "cover_intervals.csv", index=False)
    pd.DataFrame(policy_rows).to_csv(directory / "full_boundary_policy_intervals.csv", index=False)
    (directory / "manifest.json").write_text(json.dumps({"config": config,
        "effective_primitives": primitives.as_dict(), "rhos": rhos, "depth": args.depth,
        "focus_rho": args.focus_rho, "note": "Outer cylinder cover, not exact finite-observation states."}, indent=2)+'\n')
    print(f"Figures and diagnostics: {directory}")


if __name__ == "__main__":
    main()
