"""Add customer purchases from p0, A and B to the existing customer report."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/customer-purchase-sources-mpl")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from oblivious.oe_distribution_fixed_p0.code.customer_product2_report import (
    GRID, HOME, MODELS, P0S, RHOS, ROOT, digest, load_sources, verify_inputs,
)


SOURCES = (
    ("purchase_from_p0", "Outside option p₀", "#C6CBD0"),
    ("purchase_from_A", "Seller A", "#0072B2"),
    ("purchase_from_B", "Seller B", "#D55E00"),
)


def shares(row):
    return np.array([row.outside_rate, row.sale_rate, 0 if row.model == "monopoly" else row.sale_rate])


def style(ax):
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=.2)


def stack(ax, frame, labels, *, annotate=False):
    positions = np.arange(len(frame))
    bottom = np.zeros(len(frame))
    for metric, label, color in SOURCES:
        values = frame[metric].to_numpy()
        ax.bar(positions, values, bottom=bottom, width=.68, color=color,
               edgecolor="white", linewidth=.5, label=label)
        if annotate:
            for x, value, base in zip(positions, values, bottom, strict=True):
                if value > .025:
                    ax.text(x, base+value/2, f"{100*value:.2f}%", ha="center", va="center",
                            color="#222222" if metric == "purchase_from_p0" else "white", fontsize=11)
        bottom += values
    ax.set_xticks(positions, labels)
    style(ax)


def finish(fig, axes, title):
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(title + "\n" + r"All products included | stationary shares | $c_2=0.65$ | n = 101" +
                 "\nEach bar sums to 100% of calendar periods. Seller B is absent in monopoly.", fontsize=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HOME / "customer_product2_per_period")
    args = parser.parse_args()
    root = args.output.resolve()
    original_manifest = root / "manifest.json"
    original = json.loads(original_manifest.read_text())
    protected = {str(root/rel): expected for rel, expected in original["generated_file_sha256"].items()}
    protected[str(original_manifest)] = digest(original_manifest)
    verify_inputs(protected)
    raw, all_grids, selections, inputs, reports, economics = load_sources()
    if economics["parameters"]["c2"] != .65:
        raise ValueError("Keep the agreed c2=0.65 calibration")
    data = raw[["p0", "rho", "n", "model", "verified", "seller_sales_symmetry_l1"]].copy()
    for index, (metric, _, _) in enumerate(SOURCES):
        data[metric] = [shares(row)[index] for row in raw.itertuples()]
    data["sum_purchase_probabilities"] = data[[metric for metric, *_ in SOURCES]].sum(axis=1)
    if (abs(data.sum_purchase_probabilities-1).max() > 1e-9
            or (data.seller_sales_symmetry_l1.abs() > 1e-9).any()
            or not np.isfinite(data[[metric for metric, *_ in SOURCES]]).all().all()
            or (data[[metric for metric, *_ in SOURCES]].to_numpy() < -1e-12).any()):
        raise ValueError("Purchase-source accounting or symmetry failed")
    checks = []
    for row in raw.itertuples():
        pair = next(p for p in selections[str(row.p0)]["selected_validation_pairs"]
                    if p["rho"] == row.rho and p["model"] == row.model)
        values = []
        for n in [pair["coarse_n"], pair["fine_n"]]:
            selected = all_grids[(all_grids.p0 == row.p0) & (all_grids.rho == row.rho) &
                                 (all_grids.model == row.model) & (all_grids.n == n)]
            if len(selected) != 1 or not bool(selected.iloc[0].verified):
                raise ValueError("Missing verified grid comparison")
            values.append(shares(selected.iloc[0]))
        changes = abs(values[1]-values[0])
        checks.append(dict(p0=row.p0, rho=row.rho, model=row.model,
                           coarse_n=pair["coarse_n"], fine_n=pair["fine_n"],
                           **{metric+"_change_pp": 100*change for (metric, *_), change in zip(SOURCES, changes, strict=True)},
                           max_change_pp=100*changes.max(), passed=bool((changes <= .05+1e-12).all())))
    checks = pd.DataFrame(checks)
    if not checks.passed.all():
        raise ValueError("Purchase-source grid screen failed")
    figures, tables = root / "figures", root / "data"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(exist_ok=True)
    data = data.sort_values(["p0", "rho", "model"])
    table_path = tables / "customer_purchase_sources.csv"
    checks_path = tables / "customer_purchase_sources_grid_stability.csv"
    data.to_csv(table_path, index=False)
    checks.to_csv(checks_path, index=False)
    generated = [table_path, checks_path]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 10})
    pdf_path = figures / "customer_purchase_sources.pdf"
    with PdfPages(pdf_path) as pdf:
        def save(fig, name):
            path = figures / (name+".png")
            fig.savefig(path, dpi=190, bbox_inches="tight")
            pdf.savefig(fig, bbox_inches="tight")
            generated.append(path)
            plt.close(fig)

        fig, axes = plt.subplots(3, 3, figsize=(12.8, 10.2), sharey=True, layout="constrained")
        for row_index, rho in enumerate(RHOS):
            for column, (model, label, *_) in enumerate(MODELS):
                ax = axes[row_index, column]
                selected = data[(data.model == model) & (data.rho == rho)].sort_values("p0")
                stack(ax, selected, [f"{p0:g}" for p0 in selected.p0])
                ax.set(title=label+rf" | $\rho={rho:g}$", xlabel=r"Outside-option quality $p_0$")
                if column == 0:
                    ax.set_ylabel("Purchases (% of calendar periods)")
        finish(fig, list(axes.flat), "Where does the customer buy? Outside option, seller A and seller B")
        save(fig, "03_customer_purchase_sources_by_p0")

        selected = data[(data.p0 == .8) & (data.rho == .6)].copy()
        order = [model for model, *_ in MODELS]
        selected["order"] = selected.model.map({model: i for i, model in enumerate(order)})
        selected = selected.sort_values("order")
        fig, ax = plt.subplots(figsize=(8.7, 5.7), layout="constrained")
        stack(ax, selected, [label for _, label, *_ in MODELS], annotate=True)
        ax.set_ylabel("Purchases (% of calendar periods)")
        finish(fig, [ax], r"Customer purchase sources | $p_0=0.8$, $\rho=0.6$")
        save(fig, "04_customer_purchase_sources_p0_0p8_rho_0p6")
    generated.append(pdf_path)
    verify_inputs(inputs)
    verify_inputs(protected)
    source = Path(__file__).resolve()
    helper = source.with_name("customer_product2_report.py")
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="complete",
        p0s=P0S, rhos=RHOS, displayed_n=GRID, c2=.65, source_reports=reports,
        shared_economic_configuration=economics, source_input_sha256=inputs,
        reporting_source_sha256={str(p.relative_to(ROOT)): digest(p) for p in [source, helper]},
        preserved_customer_artifact_sha256=protected,
        generated_file_sha256={str(p.relative_to(root)): digest(p) for p in generated},
        metric="Stationary calendar probability of purchase from each source, all products included",
        overview_layout=dict(rows_rho=RHOS, columns_model=[model for model, *_ in MODELS]),
        formulas=dict(outside="outside_rate", A="sale_rate", B="0 in monopoly; sale_rate in symmetric competition"),
        accounting_max_residual=float(abs(data.sum_purchase_probabilities-1).max()),
        maximum_seller_symmetry_residual=float(data.seller_sales_symmetry_l1.abs().max()),
        grid_validation=dict(all_passed=True, limit_pp=5, max_change_pp=float(checks.max_change_pp.max())),
        scope="Audited finite-grid stationary probabilities, not Monte Carlo; paused n201 campaign unchanged")
    (root / "purchase_sources_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    readme = root / "README.md"
    heading = "## Compras a la opción externa, A y B"
    text = readme.read_text()
    section = "\n".join([
        heading, "",
        "Estos gráficos incluyen compras de cualquier producto, midiendo el porcentaje de períodos en que el customer compra a p0, A o B.",
        "En la comparación general, las filas corresponden a rho=0.6, 0.7 y 0.8; las columnas a monopolio, OE marginal y OE condicional.",
        "Cada barra suma 100%: `outside_rate + sale_rate = 1` en monopolio y `outside_rate + 2*sale_rate = 1` en competencia simétrica.",
        "B no existe en monopolio. En competencia, A y B tienen la misma probabilidad estacionaria de vender.", "",
        "- [Comparación de los tres modelos, tres rho y seis p0](figures/03_customer_purchase_sources_by_p0.png).",
        "- [Detalle de p0=0.8 y rho=0.6](figures/04_customer_purchase_sources_p0_0p8_rho_0p6.png).",
        "- [Ambos gráficos de compras en PDF](figures/customer_purchase_sources.pdf).",
        "- `data/customer_purchase_sources.csv`: probabilidades por origen de la compra.",
        "- `data/customer_purchase_sources_grid_stability.csv`: contraste entre grillas.",
        "- `purchase_sources_manifest.json`: procedencia y comprobación de los gráficos anteriores sin cambios.", "", "```bash",
        "MPLCONFIGDIR=/tmp/customer-purchase-sources-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
        "  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.customer_purchase_sources_report", "```", ""])
    if heading not in text:
        readme.write_text(text.rstrip()+"\n\n"+section)
    print(json.dumps(dict(output=str(root), rows=len(data), pdf_pages=2,
                         max_grid_change_pp=float(checks.max_change_pp.max()),
                         purchase_accounting_residual=float(abs(data.sum_purchase_probabilities-1).max())), indent=2))


if __name__ == "__main__":
    main()
