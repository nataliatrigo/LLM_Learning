"""Seller-level product 2 per calendar period from audited experiment 08 data."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/seller-product2-mpl")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from oblivious.oe_distribution_fixed_p0.code.customer_product2_report import (
    GRID, HOME, MODELS, P0S, RHOS, ROOT, digest, load_sources, verify_inputs,
)


def seller_flows(frame):
    frame = frame.copy()
    frame["seller_product2_calendar"] = frame.product2_calendar
    frame["seller_product1_calendar"] = frame.sale_rate - frame.product2_calendar
    frame["seller_no_sale_calendar"] = 1 - frame.sale_rate
    frame["seller_product2_identity_residual"] = abs(frame.product2_calendar - frame.sale_rate*frame.product2_conditional)
    if (frame.seller_product2_identity_residual.max() > 1e-12
            or (frame.product2_calendar < -1e-12).any()
            or (frame.product2_calendar > frame.sale_rate+1e-12).any()):
        raise ValueError("Seller calendar accounting failed")
    return frame


def validation(data, all_grids, selections):
    comparisons = []
    for row in data.itertuples():
        pair = next(p for p in selections[str(row.p0)]["selected_validation_pairs"]
                    if p["rho"] == row.rho and p["model"] == row.model)
        values = []
        for n in [pair["coarse_n"], pair["fine_n"]]:
            found = all_grids[(all_grids.p0 == row.p0) & (all_grids.rho == row.rho) &
                             (all_grids.model == row.model) & (all_grids.n == n)]
            if len(found) != 1 or not bool(found.iloc[0].verified):
                raise ValueError("Missing verified validation grid for seller measure")
            values.append(float(found.iloc[0].product2_calendar))
        change = abs(values[1]-values[0])
        comparisons.append(dict(p0=row.p0, rho=row.rho, model=row.model,
                                displayed_n=GRID, coarse_n=pair["coarse_n"], fine_n=pair["fine_n"],
                                seller_product2_coarse=values[0], seller_product2_fine=values[1],
                                absolute_change=change, change_pp=100*change, passed=change <= .05+1e-12))
    result = pd.DataFrame(comparisons)
    if not result.passed.all():
        raise ValueError("Seller measure fails the 5-percentage-point grid screen")
    return result


def draw(ax, frame, horizontal, *, positions=None):
    for model, label, color, linestyle, marker in MODELS:
        selected = frame[frame.model == model].sort_values(horizontal)
        x = selected[horizontal].to_numpy() if positions is None else positions
        ax.plot(x, selected.seller_product2_calendar, color=color, ls=linestyle,
                marker=marker, ms=5.5, lw=1.7,
                markerfacecolor=color if model == "monopoly" else "none", label=label)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)


def finish(fig, axes, title, *, local_scales=False):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    subtitle = "Stationary calendar probability | n = 101 | verified grid solutions"
    if local_scales:
        subtitle += " | panel-specific vertical scales"
    note = "One seller in each market: probability of making a sale AND supplying product 2."
    fig.suptitle(title + "\n" + subtitle + "\n" + note, fontsize=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HOME / "seller_product2_per_period")
    args = parser.parse_args()
    data, all_grids, selections, inputs, reports, economics = load_sources()
    data = seller_flows(data).sort_values(["p0", "rho", "model"])
    checks = validation(data, all_grids, selections)
    paper = ROOT / "OverleafPaper/main.tex"
    simulator = ROOT / "forgetting/Code/simulation.py"
    paper_plotter = ROOT / "forgetting/Code/experiments/run_experiments.py"
    paper_inputs = {str(path.relative_to(ROOT)): digest(path) for path in [paper, simulator, paper_plotter]}
    inputs.update(paper_inputs)
    root = args.output.resolve()
    figures, tables = root / "figures", root / "data"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(exist_ok=True)
    data.to_csv(tables / "seller_product2_summary.csv", index=False)
    checks.to_csv(tables / "seller_product2_grid_stability.csv", index=False)
    differences = []
    for (p0, rho), group in data.groupby(["p0", "rho"]):
        monopoly = float(group[group.model == "monopoly"].iloc[0].seller_product2_calendar)
        for row in group[group.model != "monopoly"].itertuples():
            differences.append(dict(p0=p0, rho=rho, model=row.model,
                                    monopoly_seller_product2=monopoly,
                                    competition_seller_product2=row.seller_product2_calendar,
                                    competition_minus_monopoly_pp=100*(row.seller_product2_calendar-monopoly)))
    pd.DataFrame(differences).to_csv(tables / "competition_vs_monopoly.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    paths = []
    pdf_path = figures / "seller_product2_comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        def save(fig, name):
            path = figures / (name + ".png")
            fig.savefig(path, dpi=190, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            paths.append(path)
            plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(12.5, 8), layout="constrained")
        for ax, p0 in zip(axes.flat, P0S, strict=True):
            group = data[data.p0 == p0]
            draw(ax, group, "rho", positions=np.arange(len(RHOS)))
            maximum = float(group.seller_product2_calendar.max())
            ax.set(title=rf"$p_0={p0:g}$", xlabel=r"Persistence $\rho$", ylim=(0, maximum*1.14 if maximum else .01),
                   ylabel="Product 2 supplied by one seller\n(% of calendar periods)")
            ax.set_xticks(np.arange(len(RHOS)), [f"{rho:g}" for rho in RHOS])
        finish(fig, list(axes.flat), "Product 2 per seller and period: monopoly vs competition", local_scales=True)
        save(fig, "01_seller_product2_by_rho")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True, layout="constrained")
        maximum = float(data.seller_product2_calendar.max())
        for ax, rho in zip(axes, RHOS, strict=True):
            draw(ax, data[data.rho == rho], "p0")
            ax.set(title=rf"$\rho={rho:g}$", xlabel=r"Outside option $p_0$", ylim=(0, maximum*1.1))
            ax.set_xticks(P0S, [f"{p0:g}" for p0 in P0S])
        axes[0].set_ylabel("Product 2 supplied by one seller\n(% of calendar periods)")
        finish(fig, list(axes), "Product 2 per seller and period as the outside option changes")
        save(fig, "02_seller_product2_by_p0")

        for p0 in P0S:
            group = data[data.p0 == p0]
            fig, ax = plt.subplots(figsize=(7.8, 5.1), layout="constrained")
            draw(ax, group, "rho", positions=np.arange(len(RHOS)))
            maximum = float(group.seller_product2_calendar.max())
            ax.set(xlabel=r"Persistence $\rho$ (equally spaced cases)", ylim=(0, maximum*1.14),
                   ylabel="Product 2 supplied by one seller (% of calendar periods)")
            ax.set_xticks(np.arange(len(RHOS)), [f"{rho:g}" for rho in RHOS])
            finish(fig, [ax], rf"Product 2 per seller and period | $p_0={p0:g}$")
            save(fig, "p0_" + str(p0).replace(".", "p") + "_seller_product2")
    paths.append(pdf_path)
    verify_inputs(inputs)
    source = Path(__file__).resolve()
    helper = Path(__file__).with_name("customer_product2_report.py").resolve()
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), status="complete", p0s=P0S, rhos=RHOS,
        displayed_n=GRID, source_reports=reports, shared_economic_configuration=economics,
        source_input_sha256=inputs,
        reporting_source_sha256={str(path.relative_to(ROOT)): digest(path) for path in [source, helper]},
        generated_file_sha256={str(p.relative_to(root)): digest(p) for p in [*paths, *tables.glob("*.csv")]},
        measure="Probability one seller is selected and supplies product 2 in a calendar period",
        formula="product2_calendar = sale_rate * product2_conditional",
        competition_aggregation="One symmetric seller; do not sum the two sellers",
        no_sale="All calendar periods remain in the denominator; zero when own seller is not selected",
        paper_comparison=dict(same_event=True,
            paper_measure="product2_unconditional_rate: seller A selected AND product 2 prescribed",
            paper_reference="OverleafPaper/main.tex, numerical sections: cumulative learning and forgetting simulations",
            difference="Paper plots the date-specific transient probability from the initial state; this report plots its stationary counterpart"),
        uncertainty="Deterministic stationary finite-grid flows, not Monte Carlo",
        grid_validation=dict(all_passed=bool(checks.passed.all()), probability_change_limit_pp=5,
                             max_seller_product2_change_pp=float(checks.change_pp.max())),
        accounting_max_residual=float(data.seller_product2_identity_residual.max()),
        preservation="Source files verified before and after rendering; original seller plots, customer plots and paused n201 campaign unchanged",
        scope="Finite-grid results and sensitivity checks; no continuous-state convergence or monotonicity theorem",
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Producto 2 por seller y período", "",
        "Comparación separada del experimento 08: monopolio, OE marginal y OE condicional.", "",
        "La medida es la probabilidad de que un seller sea seleccionado y entregue producto 2 en un período calendario.",
        "En competencia se cuenta un seller simétrico, sin sumar al rival. Se incluyen todos los períodos en el denominador.", "",
        "```text", "seller_product2_calendar = product2_calendar",
        "                        = sale_rate * product2_conditional", "```", "",
        "## Relación con el paper", "",
        "Sí: es el mismo evento que mide el panel de producto 2 de las simulaciones Monte Carlo del paper.",
        "El paper lo define como la fracción de trayectorias en que el seller A es seleccionado y elige producto 2 en esa fecha.",
        "El simulador lo guarda como `product2_unconditional_rate`, calculado con `seller_a_chosen & product2_prescribed`.",
        "La diferencia es temporal: el paper muestra esa probabilidad fecha a fecha desde el estado inicial; aquí se muestra su nivel estacionario.",
        "Se revisaron `OverleafPaper/main.tex`, `forgetting/Code/simulation.py` y `forgetting/Code/experiments/run_experiments.py`.", "",
        "## Datos y validación", "",
        "Se usan las mismas soluciones estacionarias auditadas de 101 nodos que en la comparación por customer: seis p0 y tres rho comunes.",
        "Son flujos deterministas, sin Monte Carlo ni intervalos de simulación.",
        f"El cambio máximo entre las grillas de validación de esta medida es {checks.change_pp.max():.4f} puntos porcentuales; límite: 5.",
        "La campaña de 201/251 nodos permanece pausada; se preservan los gráficos anteriores y los del customer.", "",
        "## Figuras", "",
        "- [Todos los gráficos en un PDF](figures/seller_product2_comparison.pdf).",
        "- [Frente a rho, un panel por p0](figures/01_seller_product2_by_rho.png).",
        "- [Frente a p0, un panel por rho](figures/02_seller_product2_by_p0.png).", "",
    ]
    for p0 in P0S:
        name = "p0_" + str(p0).replace(".", "p") + "_seller_product2.png"
        lines.append(f"- [p0 = {p0:g}](figures/{name}).")
    lines += ["", "- `data/seller_product2_summary.csv`: flujos por seller y período.",
              "- `data/competition_vs_monopoly.csv`: diferencias por seller en puntos porcentuales.",
              "- `data/seller_product2_grid_stability.csv`: sensibilidad de la medida entre grillas.",
              "- `manifest.json`: procedencia y verificaciones.", "", "## Reproducción", "", "```bash",
              "MPLCONFIGDIR=/tmp/seller-product2-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
              "  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.seller_product2_report", "```", ""]
    (root / "README.md").write_text("\n".join(lines))
    print(json.dumps(dict(output=str(root), rows=len(data), pdf_pages=8,
                         max_grid_change_pp=float(checks.change_pp.max()),
                         seller_identity_residual=float(data.seller_product2_identity_residual.max())), indent=2))


if __name__ == "__main__":
    main()
