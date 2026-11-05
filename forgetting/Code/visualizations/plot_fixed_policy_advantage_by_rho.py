"""Statewise one-shot advantage and its reachable maximum under product 1.

Run from the repository root:
    uv run python -B forgetting/Code/visualizations/plot_fixed_policy_advantage_by_rho.py
"""

import csv
import json

from plot_advantage_by_rho import (
    ROOT, RHOS, Primitives, advantage, reachable_boundary_points,
    save_figure, solve_cases, style_axis, write_csv,
)
import matplotlib.pyplot as plt
import numpy as np


DEST = ROOT / "forgetting/outputs/fixed_policy_advantage_by_rho_p0_0.5"
FORMULA = r"$a_\rho^1(z)=-\Delta c+\gamma\Delta p\,[U_\rho^1(T_\rho^s z)-U_\rho^1(T_\rho^f z)]$"
COLORS = ("#68737d", "#2a72ad", "#54a1c4", "#24845b", "#dc9225", "#ce513d", "#8d66ad", "#573b75")


def calibration(p):
    return rf"$p_0={p.p0:g},\ c_1={p.c1:g},\ c_2={p.c2:g},\ p_1={p.p1:g},\ p_2={p.p2:g},\ \gamma={p.gamma:g},\ R={p.revenue:g}$"


def make_figures(cases):
    p = cases[0]["solution"].primitives
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12})
    low = min(float(case["a"].min()) for case in cases)
    high = max(float(case["a"].max()) for case in cases)
    span = high - low
    limits = (low - 0.05 * span, high + 0.23 * span)
    fig, axes = plt.subplots(2, 4, figsize=(15, 8.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.77, bottom=0.20, hspace=0.33, wspace=0.13)
    for ax, case in zip(axes.flat, cases):
        sol, x, a = case["solution"], case["x"], case["a"]
        style_axis(ax)
        ax.set_ylim(*limits)
        if sol.rho < 0.5:
            ax.plot(x, a, color="#929ca8", ls="--", lw=1.5)
            reachable = reachable_boundary_points(sol.rho, depth=8)
            ax.scatter(reachable, advantage(sol, reachable), s=5, color="#2468a0", zorder=3)
        else:
            ax.plot(x, a, color="#2468a0", lw=2)
            ax.fill_between(x, 0, a, where=a > 0, interpolate=True, color="#24835b", alpha=0.18)
        ax.scatter(case["roots"], np.zeros_like(case["roots"]), s=24, facecolor="white", edgecolor="#2468a0", zorder=4)
        summary = case["summary"]
        ax.scatter([summary["argmax_reachable_x"]], [summary["max_reachable_advantage"]], s=26, marker="D", color="#303947", zorder=5, clip_on=False)
        ax.set_title(rf"$\rho={sol.rho:g}$", pad=10)
        ax.text(0.96, 0.93, rf"$\mathcal{{A}}_\rho^1={summary['max_reachable_advantage']:.3f}$", transform=ax.transAxes, ha="right", va="top", fontsize=11)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], labels=["0", ".25", ".5", ".75", "1"])
    fig.suptitle(r"Ventaja de una desviación respecto a la política de producto 1  |  $p_0=0.5$", y=0.97, fontsize=19)
    fig.text(0.52, 0.906, FORMULA, ha="center", fontsize=17)
    fig.text(0.52, 0.855, calibration(p), ha="center", fontsize=12)
    fig.supxlabel(r"Estado de éxitos normalizado $x=(1-\rho)S$, sobre $z=(x,1-x)$", y=0.13, fontsize=13)
    fig.supylabel(r"Ventaja $a_\rho^1(z)$", x=0.018, fontsize=14)
    fig.text(0.52, 0.084, "Positivo: conviene elegir producto 2 ahora y producto 1 después, frente a elegir siempre producto 1.", ha="center", fontsize=11)
    fig.text(0.52, 0.049, r"Rombo: $\mathcal{A}_\rho^1=\max_{z\in\mathcal{K}_\rho}a_\rho^1(z)$. Misma escala en todos los paneles.", ha="center", fontsize=11)
    fig.text(0.52, 0.018, r"Para $\rho=0.3$, los puntos azules aproximan el conjunto límite alcanzable; los guiones muestran toda la frontera.", ha="center", fontsize=10, color="#52606d")
    save_figure(fig, "fixed_policy_advantage_panels", DEST)

    fig, ax = plt.subplots(figsize=(11.5, 7.8))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.74, bottom=0.20)
    style_axis(ax)
    ax.set_ylim(*limits)
    for case, color in zip(cases, COLORS):
        rho = case["solution"].rho
        ax.plot(case["x"], case["a"], color=color, lw=2, ls="--" if rho < 0.5 else "-", label=rf"$\rho={rho:g}$")
    ax.legend(ncols=4, frameon=False, loc="upper right", fontsize=10)
    ax.set_xlabel(r"Estado de éxitos normalizado $x=(1-\rho)S$", labelpad=10)
    ax.set_ylabel(r"Ventaja $a_\rho^1(z)$")
    fig.suptitle(r"Valor de continuación: siempre producto 1  |  $p_0=0.5$", y=0.97, fontsize=19)
    fig.text(0.53, 0.898, FORMULA, ha="center", fontsize=16)
    fig.text(0.53, 0.835, calibration(p), ha="center", fontsize=12)
    fig.text(0.53, 0.081, "Por encima de cero: mejora al elegir producto 2 una vez y después siempre producto 1.", ha="center", fontsize=11)
    fig.text(0.53, 0.039, r"Frontera de largo plazo $z=(x,1-x)$; para $\rho=0.3$, sólo un subconjunto de Cantor es límite alcanzable.", ha="center", fontsize=10, color="#52606d")
    save_figure(fig, "fixed_policy_advantage_overlay", DEST)

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    fig.subplots_adjust(left=0.12, right=0.97, top=0.73, bottom=0.23)
    style_axis(ax)
    rhos = [case["solution"].rho for case in cases]
    maxima = [case["summary"]["max_reachable_advantage"] for case in cases]
    ax.scatter(rhos, maxima, c=COLORS, s=65, edgecolor="white", linewidth=0.7, zorder=3)
    ax.set_xlim(0.25, 1.02)
    ax.set_xlabel(r"Parámetro $\rho$", labelpad=10)
    ax.set_ylabel(r"Máximo $\mathcal{A}_\rho^1$")
    fig.suptitle(r"Máxima ventaja de una desviación  |  $p_0=0.5$", y=0.96, fontsize=19)
    fig.text(0.54, 0.875, r"$\mathcal{A}_\rho^1=\max_{z\in\mathcal{K}_\rho}\{-\Delta c+\gamma\Delta p[U_\rho^1(T_\rho^s z)-U_\rho^1(T_\rho^f z)]\}$", ha="center", fontsize=15)
    fig.text(0.54, 0.794, calibration(p), ha="center", fontsize=11)
    fig.text(0.54, 0.11, "Cada punto corresponde a uno de los ocho valores calculados de ρ; no se interpola entre ellos.", ha="center", fontsize=10)
    fig.text(0.54, 0.05, r"El máximo se toma sobre el conjunto límite alcanzable $\mathcal{K}_\rho$.", ha="center", fontsize=11)
    save_figure(fig, "fixed_policy_maximum_by_rho", DEST)


def main():
    for directory in ("figures", "data"):
        (DEST / directory).mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "forgetting/Code/experiments/config_paper.json").read_text())
    p = Primitives.from_mapping({**config["seller"], "p0": 0.5, "gamma": config["rho_p0_comparative_statics"]["gamma"]})
    cases = solve_cases(p, DEST, optimize=False)
    assert all(not case["solution"].optimized for case in cases)
    # The old optimal-value curves provide a useful independent saved benchmark:
    # with rho=.999, product 1 is already optimal on the whole boundary.
    previous = ROOT / "forgetting/outputs/advantage_by_rho_p0_0.5/data/curves.csv"
    if previous.exists() and p == Primitives():
        with previous.open(newline="") as handle:
            old = [(float(row["x"]), float(row["advantage"])) for row in csv.DictReader(handle) if float(row["rho"]) == 0.999]
        if old:
            coordinates, values = np.array(old).T
            np.testing.assert_allclose(advantage(cases[-1]["solution"], coordinates), values, atol=1e-10, rtol=0)
    write_csv(DEST / "data/maxima.csv", [dict(rho=case["solution"].rho, A_rho_1=case["summary"]["max_reachable_advantage"], argmax_x=case["summary"]["argmax_reachable_x"]) for case in cases])
    make_figures(cases)
    manifest = dict(
        primitives=p.as_dict(), rhos=RHOS, optimize=False,
        value_function="U_rho^1: discounted value of always choosing product 1",
        statewise_expression="a_rho^1(z)=-delta_c+gamma*delta_p*(U_rho^1(T_success z)-U_rho^1(T_failure z))",
        scalar_definition="A_rho^1=max over K_rho of a_rho^1(z)",
        interpretation="Positive a_rho^1 favors choosing product 2 once and product 1 forever afterward over product 1 forever; it is not the optimal-policy action gap.",
        domain="Mature boundary z=(x,1-x); x=(1-rho)*S in [0,1]",
        solver="16001 uniform nodes, linear interpolation, iteration tolerance 1e-12",
        grid_check="Compared against 8001 nodes at all breakpoints of both interpolated curves",
        maximum_curve_change=max(c["summary"]["max_curve_change_8001_to_16001"] for c in cases),
        maximum_bellman_residual=max(c["summary"]["bellman_residual"] for c in cases),
        reachable_set="Full interval for rho>=0.5; depth-16 endpoint approximation for rho=0.3",
        note="Grid comparison is a numerical stability check, not a rigorous discretization error bound.",
    )
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (DEST / "README.md").write_text(
        "# Ventaja respecto a la política fija de producto 1, p0=0.5\n\n"
        "U_rho^1 es el valor de elegir siempre el producto 1. Las curvas muestran "
        "a_rho^1(z)=-delta_c+gamma*delta_p*[U_rho^1(T_success z)-U_rho^1(T_failure z)]. "
        "Un valor positivo favorece elegir producto 2 una vez y después producto 1 "
        "para siempre, frente a elegir siempre producto 1.\n\n"
        "A_rho^1 es el máximo de esa curva sobre K_rho, el conjunto límite alcanzable. "
        "Se indica en cada panel, con un rombo en el estado del máximo, "
        "y se representa por separado frente a los ocho valores de rho. "
        "Las curvas por estado cubren la frontera z=(x,1-x), x=(1-rho)*S. "
        "Para rho<0.5, sólo un subconjunto de Cantor es límite alcanzable.\n\n"
        "PNG y PDF en figures/. Curvas, máximos y verificaciones numéricas en data/ y manifest.json.\n\n"
        "```bash\nuv run python -B forgetting/Code/visualizations/plot_fixed_policy_advantage_by_rho.py\n```\n"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"Saved figures and data to {DEST}", flush=True)


if __name__ == "__main__":
    main()
