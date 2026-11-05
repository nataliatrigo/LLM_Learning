"""Render and summarize the isolated high-p0 monopoly experiment."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/high-p0-monte-carlo-mpl")

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    config = json.loads((output / "configuration.json").read_text())
    stationary = pd.read_csv(output / "stationary/summary.csv")
    fine = stationary[stationary.n == stationary.n.max()].sort_values("p0")
    sale = pd.read_csv(output / "sale_time/summary.csv")
    sale = sale[sale.nodes == sale.nodes.max()].sort_values("p0")
    calendar = pd.read_csv(output / "data/calendar_summary.csv")
    first500 = calendar[(calendar.rho == config["primary_rho"]) &
                        (calendar.window_stop == config["paper_plot_periods"])]
    assert len(sale) == len(config["p0_values"])
    assert fine.verified.all()
    assert sale.half_window_difference_within_3se.all()
    assert sale.mean_identity_within_3se.all()

    comparison = sale[["p0", "rho", "product2_given_sale_estimate",
                       "product2_given_sale_ci_lower", "product2_given_sale_ci_upper"]].merge(
        first500[["p0", "sale_rate", "product2_calendar", "product2_conditional"]], on="p0")
    comparison = comparison.rename(columns={
        "product2_given_sale_estimate": "mature_sale_MC_product2_given_sale",
        "product2_given_sale_ci_lower": "mature_sale_MC_ci_lower",
        "product2_given_sale_ci_upper": "mature_sale_MC_ci_upper",
        "sale_rate": "first500_calendar_MC_sale_rate",
        "product2_calendar": "first500_calendar_MC_product2_per_period",
        "product2_conditional": "first500_calendar_MC_product2_given_sale",
    }).merge(fine[fine.requested_case][["p0", "product2_conditional", "product2_calendar", "sale_rate"]]
             .rename(columns={c: "stationary_grid801_" + c for c in
                              ["product2_conditional", "product2_calendar", "sale_rate"]}), on="p0")
    comparison.to_csv(output / "data/comparison.csv", index=False)

    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .2, "legend.frameon": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), layout="constrained")
    for ax in axes:
        ax.plot(fine.p0, fine.product2_conditional, color="#0072B2", lw=1.6,
                marker=".", ms=3.5, label="Stationary calculation: 801-node grid")
        ax.errorbar(sale.p0, sale.product2_given_sale_estimate,
                    yerr=np.vstack((sale.product2_given_sale_estimate-sale.product2_given_sale_ci_lower,
                                    sale.product2_given_sale_ci_upper-sale.product2_given_sale_estimate)),
                    fmt="o", ms=5, capsize=3, color="#D55E00", zorder=5,
                    label="Sale-time Monte Carlo: 95% intervals")
        ax.set(xlabel=r"Outside option $p_0$", ylabel="Product 2 / own sales")
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set(title="(a) Full tested range", xlim=(.778, .922), ylim=(0, .91))
    axes[0].set_xticks(np.arange(.78, .921, .02))
    axes[0].legend(loc="lower left", fontsize=9)
    axes[1].set(title="(b) Zoom: the local increase", xlim=(.795, .857), ylim=(.823, .847))
    axes[1].set_xticks([.8, .81039, .835, .85], ["0.8", "0.81039", "0.835", "0.85"])
    for row in sale[sale.p0 < .9].itertuples():
        axes[1].annotate(f"{100*row.product2_given_sale_estimate:.2f}%", xy=(row.p0, row.product2_given_sale_estimate),
                         xytext=(0, -19), textcoords="offset points", ha="center", color="#A54300", fontsize=9)
    fig.suptitle(
        rf"Monopoly product mix | $\rho={config['primary_rho']:g}$"
        "\nMonte Carlo: 1,000 paths, 1,000 burn-in sales + 10,000 retained sales each; 4,001-node policy",
        fontsize=11,
    )
    fig.savefig(output / "figures/03_stationary_usage_vs_p0.png", dpi=190, bbox_inches="tight")
    plt.close(fig)

    lines = [
        "MONOPOLIO: COMPARACION DE p0 ALTOS",
        "",
        "Calibracion: rho=0.7, p1=0.35, p2=0.8, c1=0.05, c2=0.65, gamma=0.98, ingreso=1.",
        "",
        "RESULTADO PRINCIPAL",
        "La proporcion de producto 2 entre ventas aumenta localmente de 0.8 a 0.81039 y a 0.85,",
        "pero cae fuertemente en 0.9. No es monotona en el rango estudiado.",
        "Con esta calibracion no se confirma que p0=0.9 use mas producto 2 que p0=0.8.",
        "La cantidad de ventas de producto 2 por periodo disminuye en los cuatro casos.",
        "",
        "MEDIDAS DISTINTAS (todos los valores siguientes son porcentajes)",
        "p0       P2/venta maduro MC [IC95%]          P2/periodo, primeras 500 fechas MC",
    ]
    for row in comparison.itertuples():
        lines.append(f"{row.p0:<8g} {100*row.mature_sale_MC_product2_given_sale:8.4f} "
                     f"[{100*row.mature_sale_MC_ci_lower:.4f}, {100*row.mature_sale_MC_ci_upper:.4f}]"
                     f"                  {100*row.first500_calendar_MC_product2_per_period:.4f}")
    peak = fine.loc[fine.product2_conditional.idxmax()]
    paired = pd.read_csv(output / "sale_time/paired_p0_differences.csv")
    delta = paired[(paired.nodes == 4001) & np.isclose(paired.p0, .81039)].iloc[0]
    lines += [
        "",
        f"Aumento pareado .8 -> .81039: {100*delta.product2_difference_estimate:.4f} puntos porcentuales;",
        f"IC95% [{100*delta.product2_difference_ci_lower:.4f}, {100*delta.product2_difference_ci_upper:.4f}].",
        f"El maximo muestreado del barrido estacionario esta en p0={peak.p0:g} "
        f"({100*peak.product2_conditional:.4f}% en la grilla de 801 nodos).",
        "El barrido cubre .78 a .92; este maximo muestreado no prueba un maximo continuo ni unimodalidad.",
        "",
        "INTERPRETACION",
        "P2/periodo = ventas/periodo * P2/venta para una misma ventana o distribucion.",
        "El denominador por si solo no explica el aumento local: si mantuvieramos fija la politica,",
        "la dinamica entre ventas no dependeria de la demanda. Aqui cambia la politica optima y,",
        "con ella, la distribucion de reputaciones observadas al vender.",
        "La politica tiene dos umbrales; sus cambios no son anidados al aumentar p0.",
        "",
        "DISENO Y LIMITES",
        "Calendar MC: 20,000 trayectorias desde Beta(1,1), politicas transientes; 3,000 fechas para rho=.7.",
        "El plot del paper muestra las primeras 500 fechas; no supone estacionariedad.",
        "Referencia adicional del paper: rho=.95, 20,000 trayectorias y 500 fechas; figura separada.",
        "Para rho=.95, P2/venta en esas primeras 500 fechas disminuye en los cuatro p0 solicitados.",
        "Sale-time MC: 1,000 trayectorias independientes del proceso maduro continuo, inicial x=.5,",
        "1,000 ventas de calentamiento y 10,000 ventas retenidas por trayectoria; politica de 4,001 nodos.",
        "Los IC95% usan trayectorias como clusters; numeros aleatorios comunes entre valores de p0.",
        "Las dos mitades retenidas coinciden dentro de 3 errores estandar; esto es un diagnostico,",
        "no una demostracion de ausencia de sesgo de calentamiento.",
        "El contraste de politicas 2,001 vs 4,001 nodos cambia P2/venta como maximo 0.00011 pp.",
        "El barrido estacionario usa 201, 401 y 801 nodos (58 p0 en cada grilla), con verificaciones",
        "de Bellman y distribuciones invariantes. Se grafica la grilla 801; sus diferencias frente",
        "al MC continuo son discretizacion numerica, y no se interpretan como ruido Monte Carlo.",
        "En p0=.9 la demanda estacionaria es muy baja: 3,000 fechas conservan sesgo de inicializacion.",
        "Por ello el ratio de las primeras 500 o de todas las 3,000 fechas no se llama estacionario.",
        "",
        "ARCHIVOS",
        "figures/01_paper_style_rho_0p7.pdf: demanda y producto 2 por periodo, formato del paper.",
        "figures/01_paper_style_rho_0p95.pdf: referencia con persistencia del paper.",
        "figures/02_extended_calendar_comparison.png: 3,000 fechas en bloques de 25, tres medidas.",
        "figures/03_stationary_usage_vs_p0.png: barrido y MC entre ventas, con zoom local.",
        "data/comparison.csv: tabla de medidas y ventanas explicitamente separadas.",
        "data/, sale_time/, stationary/: datos detallados, configuraciones y manifiestos.",
        "No se modificaron el paper ni resultados historicos.",
    ]
    (output / "RESULTADOS.txt").write_text("\n".join(lines) + "\n")

    manifest_path = output / "calendar_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    snapshot = output / "provenance/monopoly_high_p0_monte_carlo_simulated.py"
    original_hash = manifest["source_sha256"]["forgetting/Code/experiments/monopoly_high_p0_monte_carlo.py"]
    assert sha256(snapshot.read_bytes()).hexdigest() == original_hash
    manifest["simulation_source_snapshot"] = str(snapshot.relative_to(output))
    manifest["plotting_only_revision"] = "Raised y-axis limit to .36 to include all observations; simplified interval-column selection. Simulation data unchanged."
    manifest["current_render_sha256"] = sha256(Path(__file__).with_name("monopoly_high_p0_monte_carlo.py").read_bytes()).hexdigest()
    manifest["summary_source_sha256"] = sha256(Path(__file__).read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
