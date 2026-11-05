"""Static, directly viewable companion to the threshold explorer."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

DEST = Path(__file__).resolve().parents[3] / "forgetting/outputs/threshold_visualizations"
CASES = {c["id"]: c for c in json.loads((DEST / "threshold-data.json").read_text())["cases"]}


def advantage(c, x):
    p = c["primitives"]
    grid = np.linspace(0, 1, len(c["value"]))
    return -(p["c2"]-p["c1"]) + p["gamma"]*(p["p2"]-p["p1"]) * (
        np.interp(c["rho"]*x+1-c["rho"], grid, c["value"])
        - np.interp(c["rho"]*x, grid, c["value"])
    )


def make_plot(case_id, filename):
    c = CASES[case_id]
    gap = case_id == "gap"
    x = np.linspace(.435, .565, 3001) if gap else np.linspace(0, 1, 8001)
    a = advantage(c, x)
    plt.rcParams.update({"font.size": 14, "axes.titlesize": 17, "axes.labelsize": 14})
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    fig.subplots_adjust(left=.16, right=.97, top=.80, bottom=.20)
    blue = "#246bb2"
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e1e5e9", linewidth=.8)
    if gap:
        ax.axvspan(.45, .55, color="#ececf0", label="Hueco de Cρ")
    ax.fill_between(x, 0, a, where=a>0, color=blue, alpha=.16, interpolate=True)
    ax.plot(x, a, color=blue, linewidth=2.3, label="Ventaja del producto 2")
    ax.axhline(0, color="#565b62", linestyle="--", linewidth=1)
    ax.scatter(c["roots"], [0]*len(c["roots"]), s=60, facecolor="white", edgecolor=blue, linewidth=1.8, zorder=4)
    for root in c["roots"]:
        ax.annotate(f"x ≈ {root:.4f}", (root, 0), xytext=(0,-28), textcoords="offset points", ha="center", fontsize=12)
    ax.scatter([c["xMaxReachable"]], [c["maxReachable"]], s=58, marker="D", color="#24272b", zorder=5)
    if gap:
        ax.text(.5, -.00118, "Hueco del conjunto alcanzable\n0.45 < x < 0.55", ha="center", va="center", fontsize=13)
        ax.annotate(f"Aρ ≈ {c['maxReachable']:.6f}", (c["xMaxReachable"], c["maxReachable"]), xytext=(-9,-47), textcoords="offset points", ha="right", fontsize=12,
                    arrowprops={"arrowstyle":"-", "color":"#555b62", "lw":.9})
        ax.set_ylim(-.0016,.00082)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.4f}"))
        title="ρ = 0.45: la zona positiva queda dentro de un hueco"
        subtitle="p₀ = 0.9 · c₂ = 0.6665 · detalle de x cerca de los cruces"
        footer="La región positiva no intersecta Cρ; el máximo sobre Cρ es negativo."
    else:
        ax.annotate(f"Aρ ≈ {c['maxReachable']:.3f}", (c["xMaxReachable"], c["maxReachable"]), xytext=(35,8), textcoords="offset points", ha="left", fontsize=12)
        ax.set_ylim(-.85,2.25)
        title="ρ = 0.95: dos umbrales numéricos"
        subtitle="p₀ = 0.5 · c₂ = 0.65 · conjunto alcanzable Cρ = [0, 1]"
        footer="Entre los dos cruces, la ventaja es positiva y se prefiere el producto 2."
    ax.set_xlim(x[0],x[-1])
    ax.set_xlabel("Estado x", labelpad=8)
    ax.set_ylabel("Ventaja aρ(x)", labelpad=10)
    for spine in ax.spines.values():
        spine.set_color("#bcc2c8")
    fig.suptitle(title, x=.52, y=.975)
    fig.text(.52,.90,subtitle,ha="center",fontsize=12,color="#444a52")
    fig.text(.52,.075,footer,ha="center",fontsize=12)
    fig.text(.52,.025,"Modelo original · aproximación numérica de Bellman · 8 001 nodos",ha="center",fontsize=10,color="#555b62")
    fig.savefig(DEST/filename, facecolor="white")
    plt.close(fig)
    print(DEST/filename)


make_plot("two", "umbrales-rho-095.png")
make_plot("gap", "hueco-rho-045.png")
