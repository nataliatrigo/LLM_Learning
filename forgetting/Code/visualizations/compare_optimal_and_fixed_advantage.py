"""Compare saved optimal and fixed-policy curves on identical axes."""

import json

from plot_advantage_by_rho import ROOT, RHOS, save_figure, style_axis
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


OPTIMAL = ROOT / "forgetting/outputs/advantage_by_rho_p0_0.5"
FIXED = ROOT / "forgetting/outputs/fixed_policy_advantage_by_rho_p0_0.5"
DEST = ROOT / "forgetting/outputs/optimal_vs_fixed_advantage_p0_0.5"


def positive_intervals(curve, roots):
    edges = [0.0, *roots, 1.0]
    return [(lo, hi) for lo, hi in zip(edges[:-1], edges[1:])
            if np.interp((lo + hi) / 2, curve.x, curve.advantage) > 0]


def main():
    optimal_manifest = json.loads((OPTIMAL / "manifest.json").read_text())
    fixed_manifest = json.loads((FIXED / "manifest.json").read_text())
    assert optimal_manifest["primitives"] == fixed_manifest["primitives"]
    assert optimal_manifest["rhos"] == fixed_manifest["rhos"]
    assert fixed_manifest["optimize"] is False
    for name in ("figures", "data"):
        (DEST / name).mkdir(parents=True, exist_ok=True)
    curves = [pd.read_csv(folder / "data/curves.csv") for folder in (OPTIMAL, FIXED)]
    summaries = [pd.read_csv(folder / "data/summary.csv").set_index("rho") for folder in (OPTIMAL, FIXED)]
    rows = []
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14})
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.76, bottom=0.18, hspace=0.34, wspace=0.13)
    colors = ("#2468a0", "#d47a24")
    low = min(curve.advantage.min() for curve in curves)
    high = max(curve.advantage.max() for curve in curves)
    span = high - low
    for ax, rho in zip(axes.flat, RHOS):
        style_axis(ax)
        ax.set_ylim(low - 0.05 * span, high + 0.12 * span)
        ax.set_title(rf"$\rho={rho:g}$", pad=10)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=["0", ".25", ".5", ".75", "1"])
        row = {"rho": rho, "p0": 0.5}
        for name, data, summary, color in zip(("optimal", "fixed"), curves, summaries, colors):
            curve = data[data.rho == rho]
            roots = json.loads(summary.loc[rho, "zero_crossings_x"])
            regions = positive_intervals(curve, roots)
            ax.plot(curve.x, curve.advantage, color=color, lw=2, ls="-" if name == "optimal" else "--")
            ax.scatter(roots, np.zeros(len(roots)), s=24, facecolor="white", edgecolor=color, zorder=4)
            row[f"A_{name}"] = float(summary.loc[rho, "max_reachable_advantage"])
            row[f"positive_intervals_full_boundary_{name}"] = json.dumps(regions)
            row[f"positive_width_full_boundary_{name}"] = sum(hi - lo for lo, hi in regions)
        row["same_positivity_of_maxima"] = (row["A_optimal"] > 0) == (row["A_fixed"] > 0)
        row["width_difference_fixed_minus_optimal"] = row["positive_width_full_boundary_fixed"] - row["positive_width_full_boundary_optimal"]
        assert row["same_positivity_of_maxima"]
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(DEST / "data/comparison.csv", index=False)
    print(result.to_string(index=False), flush=True)
    fig.suptitle(r"Política óptima y política fija: comparación directa  |  $p_0=0.5$", y=0.97, fontsize=20)
    fig.text(0.52, 0.904, r"$a_\rho(z)=-\Delta c+\gamma\Delta p[V_\rho(T_\rho^s z)-V_\rho(T_\rho^f z)]$", ha="center", fontsize=15)
    fig.text(0.52, 0.855, r"$a_\rho^1(z)=-\Delta c+\gamma\Delta p[U_\rho^1(T_\rho^s z)-U_\rho^1(T_\rho^f z)]$", ha="center", fontsize=15)
    fig.legend(handles=[Line2D([0], [0], color=colors[0], lw=2, label=r"$a_\rho$: continuación óptima"), Line2D([0], [0], color=colors[1], lw=2, ls="--", label=r"$a_\rho^1$: siempre producto 1 después")], loc="center", bbox_to_anchor=(0.52, 0.807), ncols=2, frameon=False, fontsize=12)
    fig.supxlabel(r"Estado normalizado $x=(1-\rho)S$, sobre $z=(x,1-x)$", y=0.11, fontsize=13)
    fig.supylabel("Ventaja de elegir producto 2", x=0.012, fontsize=13)
    fig.text(0.52, 0.063, r"La equivalencia $\mathcal{A}_\rho>0\ \Longleftrightarrow\ \mathcal{A}_\rho^1>0$ se verifica en los 8 valores calculados.", ha="center", fontsize=12)
    fig.text(0.52, 0.025, r"Los intervalos de estados positivos pueden diferir. Para $\rho=0.3$, sólo un subconjunto de Cantor de la frontera es límite alcanzable.", ha="center", fontsize=10, color="#52606d")
    save_figure(fig, "optimal_vs_fixed_advantage", DEST)
    (DEST / "manifest.json").write_text(json.dumps({
        "primitives": optimal_manifest["primitives"], "rhos": RHOS,
        "sources": [str(OPTIMAL), str(FIXED)],
        "same_positivity_of_maxima_all_pairs": bool(result.same_positivity_of_maxima.all()),
        "intervals": "Full-boundary positive intervals; endpoints at roots are excluded. At x=0 or x=1, include the endpoint only if its advantage is strictly positive.",
        "cantor_note": "For rho<0.5 intersect the reported intervals with the limiting reachable set; full-boundary interval lengths are not lengths of positive reachable states.",
        "scope": "Checks eight computed rho values at p0=0.5; not a numerical verification of every rho.",
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
