"""Replot the original experiment 08 metric: product 2 among own sales."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/seller-product2-given-sale-mpl")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from oblivious.oe_distribution_fixed_p0.code.customer_product2_report import (
    GRID, HOME, MODELS, P0S, RHOS, ROOT, digest, load_sources, verify_inputs,
)


def draw(ax, data, horizontal, *, positions=None):
    for model, label, color, linestyle, marker in MODELS:
        selected = data[data.model == model].sort_values(horizontal)
        x = selected[horizontal].to_numpy() if positions is None else positions
        ax.plot(x, selected.product2_conditional, color=color, ls=linestyle,
                marker=marker, ms=5.5, lw=1.7,
                markerfacecolor=color if model == "monopoly" else "none", label=label)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.grid(alpha=.22)
    ax.spines[["top", "right"]].set_visible(False)


def finish(fig, axes, title):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(title + "\n" +
                 r"Original metric: product mix when the seller sells | $c_2=0.65$ | n = 101" +
                 "\nProduct 2 as a percentage of own sales; stationary verified grid solutions.", fontsize=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HOME / "seller_product2_given_sale")
    args = parser.parse_args()
    data, all_grids, selections, inputs, reports, economics = load_sources()
    data = data.sort_values(["p0", "rho", "model"])
    if economics["parameters"]["c2"] != .65:
        raise ValueError("Keep the agreed c2=0.65 calibration")
    residual = abs(data.product2_calendar-data.sale_rate*data.product2_conditional)
    if (residual.max() > 1e-12 or (data.sale_rate <= 0).any()
            or not data.product2_conditional.between(0, 1).all()):
        raise ValueError("Original conditional metric accounting failed")
    comparisons = []
    for row in data.itertuples():
        pair = next(p for p in selections[str(row.p0)]["selected_validation_pairs"]
                    if p["rho"] == row.rho and p["model"] == row.model)
        values = []
        for n in [pair["coarse_n"], pair["fine_n"]]:
            found = all_grids[(all_grids.p0 == row.p0) & (all_grids.rho == row.rho) &
                             (all_grids.model == row.model) & (all_grids.n == n)]
            if len(found) != 1 or not bool(found.iloc[0].verified):
                raise ValueError("Missing verified original-metric validation grid")
            values.append(float(found.iloc[0].product2_conditional))
        change = abs(values[1]-values[0])
        comparisons.append(dict(p0=row.p0, rho=row.rho, model=row.model,
                                coarse_n=pair["coarse_n"], fine_n=pair["fine_n"],
                                absolute_change=change, change_pp=100*change, passed=change <= .05+1e-12))
    checks = pd.DataFrame(comparisons)
    if not checks.passed.all():
        raise ValueError("Original metric no longer passes its grid screen")
    root = args.output.resolve()
    figures, tables = root / "figures", root / "data"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(exist_ok=True)
    data.to_csv(tables / "product2_given_sale_summary.csv", index=False)
    checks.to_csv(tables / "grid_stability.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    generated = []
    pdf_path = figures / "original_metric_comparison.pdf"
    with PdfPages(pdf_path) as pdf:
        def save(fig, name):
            path = figures / (name+".png")
            fig.savefig(path, dpi=190, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            generated.append(path)
            plt.close(fig)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True, layout="constrained")
        for ax, rho in zip(axes, RHOS, strict=True):
            draw(ax, data[data.rho == rho], "p0")
            ax.set(title=rf"$\rho={rho:g}$", xlabel=r"Outside option $p_0$")
            ax.set_xticks(P0S, [f"{p0:g}" for p0 in P0S])
        axes[0].set_ylabel("Product 2 (% of own sales)")
        finish(fig, list(axes), "Product 2 among own sales as the outside option changes")
        save(fig, "01_product2_given_sale_by_p0")

        fig, axes = plt.subplots(2, 3, figsize=(12.5, 8), sharey=True, layout="constrained")
        for ax, p0 in zip(axes.flat, P0S, strict=True):
            draw(ax, data[data.p0 == p0], "rho", positions=np.arange(len(RHOS)))
            ax.set(title=rf"$p_0={p0:g}$", xlabel=r"Persistence $\rho$", ylabel="Product 2 (% of own sales)")
            ax.set_xticks(np.arange(len(RHOS)), [f"{rho:g}" for rho in RHOS])
        finish(fig, list(axes.flat), "Product 2 among own sales: monopoly vs competition")
        save(fig, "02_product2_given_sale_by_rho")
    generated.append(pdf_path)
    verify_inputs(inputs)
    source = Path(__file__).resolve()
    helper = source.with_name("customer_product2_report.py")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="complete",
        p0s=P0S, rhos=RHOS, displayed_n=GRID, c2=.65,
        metric="product2_conditional", formula="product2_calendar / sale_rate",
        interpretation="Probability of product 2 conditional on an own sale, averaged over actual own-sale states",
        aggregation="One seller in each model; no summing competitors",
        time_measure="Stationary finite-grid product mix, not transient Monte Carlo or calendar-period delivery probability",
        common_vertical_axis=[0, 1], source_reports=reports, shared_economic_configuration=economics,
        source_input_sha256=inputs,
        reporting_source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [source, helper]},
        generated_file_sha256={str(p.relative_to(root)): digest(p) for p in [*generated, *tables.glob("*.csv")]},
        accounting_max_residual=float(residual.max()),
        grid_validation=dict(all_passed=True, max_change_pp=float(checks.change_pp.max()), limit_pp=5),
        preservation="Historical data, other plot folders, and paused n201 campaign unchanged")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (root / "README.md").write_text("\n".join([
        "# Métrica original: producto 2 entre las ventas propias", "",
        "Ambos formatos solicitados usan `product2_conditional`, la métrica original del experimento 08.",
        "Es la probabilidad de producto 2 condicionada en una venta propia, promediada sobre las reputaciones de las ventas reales.",
        "Se calcula como `product2_calendar / sale_rate`. No es la probabilidad conjunta de vender y entregar producto 2 por período.",
        "Se conserva c2=0.65 y la misma selección auditada de 101 nodos: seis p0 y rho=0.6,0.7,0.8.",
        "Todas las figuras tienen eje vertical común de 0% a 100%.", "",
        "- [Tres paneles frente a p0](figures/01_product2_given_sale_by_p0.png).",
        "- [Seis paneles frente a rho](figures/02_product2_given_sale_by_rho.png).",
        "- [Ambos gráficos en PDF](figures/original_metric_comparison.pdf).", "",
        "`data/` conserva los 54 resultados originales y su contraste de grilla; `manifest.json` registra procedencia y huellas.",
        "El contraste de grilla no demuestra convergencia al modelo continuo.", "", "```bash",
        "MPLCONFIGDIR=/tmp/seller-product2-given-sale-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
        "  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.seller_product2_given_sale_report", "```", ""]))
    print(json.dumps(dict(output=str(root), rows=len(data), pdf_pages=2,
                         max_grid_change_pp=float(checks.change_pp.max())), indent=2))


if __name__ == "__main__":
    main()
