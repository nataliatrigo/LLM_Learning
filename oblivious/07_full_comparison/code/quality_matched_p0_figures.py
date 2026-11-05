"""Plot monopoly at rival sales quality against saved endogenous-p0 policies."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np


MONOPOLY_COLOR, P0_COLOR = "#0072B2", "#D55E00"
P0_LABEL = "Endogenous p0 (OE p0)"


def make_figures(pairs, data, n, directory, save) -> list[Path]:
    """Render the same two comparisons using an injected save callback."""
    data = data.sort_values("rho")
    pairs = sorted(pairs, key=lambda pair: pair["rho"])
    if data.empty or not pairs:
        raise ValueError("Need nonempty endogenous-p0 comparison data and policy pairs")
    directory = Path(directory)
    rhos = data.rho.to_numpy()
    x = np.arange(len(rhos))
    fig, (ax, ref_ax) = plt.subplots(
        2, 1, figsize=(11, 8.5), sharex=True,
        gridspec_kw=dict(height_ratios=[2.5, 1]),
    )
    fig.subplots_adjust(left=.09, right=.98, top=.78, bottom=.19, hspace=.25)
    ax.plot(x, data.monopoly_product2_conditional, color=MONOPOLY_COLOR,
            marker="o", lw=2, ms=5,
            label=r"Monopolio con $p_0=p_0^{ref}(\rho)$")
    ax.plot(x, data.p0_product2_conditional, color=P0_COLOR,
            marker="D", mfc="white", lw=2, ms=5, ls="--", label=P0_LABEL)
    for xi, row in zip(x, data.itertuples()):
        monopoly_above = row.monopoly_product2_conditional >= row.p0_product2_conditional
        ax.annotate(f"{100*row.monopoly_product2_conditional:.1f}%",
                    (xi, row.monopoly_product2_conditional),
                    xytext=(0, 10 if monopoly_above else -17),
                    textcoords="offset points", ha="center", fontsize=9,
                    color=MONOPOLY_COLOR)
        ax.annotate(f"{100*row.p0_product2_conditional:.1f}%",
                    (xi, row.p0_product2_conditional),
                    xytext=(0, -17 if monopoly_above else 10),
                    textcoords="offset points", ha="center", fontsize=9, color=P0_COLOR)
    ax.set(ylabel="Producto 2 / ventas propias", ylim=(-.025, 1.025))
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ref_ax.plot(x, data.p0_ref, color=".3", marker="o", lw=1.6, ms=4)
    for xi, p0 in zip(x, data.p0_ref):
        ref_ax.annotate(f"{p0:.4f}", (xi, p0), xytext=(0, 9),
                        textcoords="offset points", ha="center", fontsize=9)
    ref_ax.set(ylabel=r"Calidad de referencia $p_0^{ref}$",
               xlabel=r"Factor de olvido $\rho$ (valores discretos)",
               xlim=(-.2, len(rhos)-.8),
               ylim=(data.p0_ref.min()-.045, data.p0_ref.max()+.065))
    ref_ax.set_xticks(x, [f"{rho:g}" for rho in rhos])
    for panel in (ax, ref_ax):
        panel.grid(axis="y", alpha=.2)
        panel.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Outside option igual a la calidad media del rival endogenous p0",
                 fontsize=15, y=.985)
    fig.text(.53, .925,
             r"$p_0^{ref}(\rho)=p_1+(p_2-p_1)\,u_j^{\mathrm{OE}\ p_0}(\rho)$"
             + "     ·     calidad entre las ventas del rival", ha="center", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, .89),
               ncol=2, frameon=False, fontsize=10)
    fig.text(.53, .105, f"Predicciones estacionarias · grilla común n={n} · "
             "todos los ρ > 0.5 disponibles · sin filtro sobre p₀.", ha="center", fontsize=9)
    fig.text(.53, .060,
             "La referencia de calidad p₀ref se calcula por separado del escalar endógeno del equilibrio.",
             ha="center", fontsize=8.5)
    fig.text(.53, .025,
             "Los valores de ρ están equiespaciados; las líneas orientan la lectura.",
             ha="center", fontsize=8.5)
    paths = list(save(fig, directory, "01_product2_usage_quality_matched"))

    count = len(pairs)
    rows = (count+1)//2
    fig, axes = plt.subplots(rows, 2, figsize=(11.8, 3.1*rows+.8), squeeze=False)
    fig.subplots_adjust(left=.08, right=.98, bottom=.09, top=.90,
                        hspace=.43, wspace=.23)
    for pair, panel in zip(pairs, axes.flat):
        monopoly = pair["monopoly"]
        p0_grid = np.asarray(pair["p0_grid"])
        p0_policy = np.asarray(pair["p0_policy"])
        panel.plot(monopoly["grid"], monopoly["policy"], color=MONOPOLY_COLOR, lw=1.8)
        panel.plot(p0_grid, p0_policy, color=P0_COLOR, lw=1.7, ls="--")
        mixed = (p0_policy > 1e-9) & (p0_policy < 1-1e-9)
        panel.scatter(p0_grid[mixed], p0_policy[mixed], color=P0_COLOR,
                      marker="D", s=27, edgecolors="white", linewidths=.5, zorder=4)
        panel.text(.02, .91, rf"$p_0^{{ref}}={pair['p0_ref']:.4f}$",
                   transform=panel.transAxes, va="top", fontsize=10,
                   bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=2))
        panel.set(title=rf"$\rho={pair['rho']:g}$", xlabel="Reputación propia",
                  ylabel="Probabilidad de producto 2", xlim=(-.02, 1.02), ylim=(-.04, 1.04),
                  xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
        panel.grid(alpha=.2)
        panel.spines[["top", "right"]].set_visible(False)
    for panel in list(axes.flat)[count:]:
        panel.set_visible(False)
    fig.suptitle("Políticas con referencia igual a la calidad media del rival endogenous p0",
                 fontsize=14, y=.99)
    fig.legend(handles=[
        Line2D([], [], color=MONOPOLY_COLOR, lw=2,
               label=r"Monopolio con $p_0=p_0^{ref}(\rho)$"),
        Line2D([], [], color=P0_COLOR, lw=2, ls="--", label=P0_LABEL),
        Line2D([], [], color=P0_COLOR, marker="D", ls="none", ms=5,
               label="Mezcla en un nodo OE p0")],
        loc="upper center", bbox_to_anchor=(.53, .965), ncol=3, frameon=False)
    fig.text(.53, .04, f"Grilla común n={n} · selector puro original del monopolio · "
             "líneas unen nodos resueltos · calidad calibrada desde OE p0, sin nuevo cierre.",
             ha="center", fontsize=8.3)
    fig.text(.53, .018,
             "p₀ref denota calidad entre ventas; el escalar endógeno del equilibrio conserva su definición.",
             ha="center", fontsize=8.3)
    return paths + list(save(fig, directory, "02_policies_quality_matched"))
