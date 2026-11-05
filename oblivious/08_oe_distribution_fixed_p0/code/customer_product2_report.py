"""Customer-level product 2 flows from the preserved, audited n101 campaign.

This report does not solve an equilibrium or resume the paused n201 campaign.
It sums both symmetric sellers' calendar flows and writes a separate report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/customer-product2-mpl")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from oblivious.oe_distribution_fixed_p0.code.run_experiment import HOME, ROOT
from oblivious.oe_distribution_fixed_p0.code.conditional_comparison_report import MODELS, digest
from oblivious.oe_distribution_fixed_p0.code.multi_p0_report import verify_inputs


P0S = [.1, .3, .5, .7, .8, .9]
RHOS = [.6, .7, .8]
GRID = 101


def load_sources():
    frames, coarse_frames, selections, inputs, reports = [], [], {}, {}, []
    baseline = None
    for folder in [HOME / "data/p0_sweep", HOME / "data/p0_sweep/p0_0p9"]:
        report_path = folder / "comparison_report_manifest.json"
        report = json.loads(report_path.read_text())
        if report.get("displayed_n") != GRID or report.get("independent_audit", {}).get("verified") is not True:
            raise ValueError("The preserved n101 comparison must have a verified independent audit")
        retained = report.get("common_retained_rhos", report.get("retained_rhos"))
        if retained != RHOS:
            raise ValueError("Source report has a different common rho selection")
        economics = report["shared_economic_configuration"]
        if baseline is not None and baseline != economics:
            raise ValueError("Economic configuration differs between source reports")
        baseline = economics
        for field in ["source_input_sha256", "reporting_source_sha256", "figure_files", "generated_data_sha256"]:
            declared = report.get(field, {})
            if not declared:
                raise ValueError("Missing preserved report provenance: " + field)
            verify_inputs(declared)
            inputs.update(declared)
        inputs[str(report_path.relative_to(ROOT))] = digest(report_path)
        audit_path = ROOT / report["independent_audit"]["path"]
        audit = json.loads(audit_path.read_text())
        if audit.get("verified") is not True:
            raise ValueError("Source independent review is unverified")
        verify_inputs(audit["audited_input_sha256"])
        inputs.update(audit["audited_input_sha256"])
        inputs[str(audit_path.relative_to(ROOT))] = digest(audit_path)
        frames.append(pd.read_csv(folder / "comparison_summary.csv"))
        coarse_frames.append(pd.read_csv(folder / "comparison_summary_all_grids.csv"))
        selections.update(report.get("selections", {"0.9": report.get("selection")}))
        reports.append(dict(path=str(report_path.relative_to(ROOT)), sha256=digest(report_path),
                            p0s=report["requested_p0s"], rhos=retained, displayed_n=GRID))
    data = pd.concat(frames, ignore_index=True)
    all_grids = pd.concat(coarse_frames, ignore_index=True)
    expected = {(p0, rho, model) for p0 in P0S for rho in RHOS for model, *_ in MODELS}
    actual = list(data[["p0", "rho", "model"]].itertuples(index=False, name=None))
    if len(actual) != len(expected) or set(actual) != expected or not data.verified.all() or not (data.n == GRID).all():
        raise ValueError("Incomplete, duplicate, or unverified published comparison")
    return data, all_grids, selections, inputs, reports, baseline


def customer_flows(data):
    data = data.copy()
    data["seller_count"] = np.where(data.model == "monopoly", 1, 2)
    data["customer_product2_calendar"] = data.seller_count * data.product2_calendar
    data["customer_product1_calendar"] = data.seller_count * (data.sale_rate - data.product2_calendar)
    data["customer_purchase_calendar"] = data.seller_count * data.sale_rate
    cross_check = (1-data.outside_rate) * data.product2_conditional
    data["customer_product2_identity_residual"] = abs(data.customer_product2_calendar-cross_check)
    accounting = abs(data.customer_product1_calendar + data.customer_product2_calendar + data.outside_rate - 1)
    if (data.customer_product2_identity_residual.max() > 1e-9 or accounting.max() > 1e-9
            or (data.customer_product2_calendar < -1e-12).any()
            or (data.customer_product2_calendar > data.customer_purchase_calendar+1e-12).any()):
        raise ValueError("Customer-level calendar accounting failed")
    return data


def validation(data, all_grids, selections):
    comparisons = []
    for row in data.itertuples():
        pairs = selections[str(row.p0)]["selected_validation_pairs"]
        pair = next(p for p in pairs if p["rho"] == row.rho and p["model"] == row.model)
        values = []
        for n in [pair["coarse_n"], pair["fine_n"]]:
            found = all_grids[(all_grids.p0 == row.p0) & (all_grids.rho == row.rho) &
                             (all_grids.model == row.model) & (all_grids.n == n)]
            if len(found) != 1 or not bool(found.iloc[0].verified):
                raise ValueError("Missing verified validation grid for customer measure")
            values.append(float(found.iloc[0].product2_calendar) * row.seller_count)
        change = abs(values[1]-values[0])
        comparisons.append(dict(p0=row.p0, rho=row.rho, model=row.model,
                                displayed_n=GRID, coarse_n=pair["coarse_n"], fine_n=pair["fine_n"],
                                customer_product2_coarse=values[0], customer_product2_fine=values[1],
                                absolute_change=change, change_pp=100*change, passed=change <= .05+1e-12))
    result = pd.DataFrame(comparisons)
    if not result.passed.all():
        raise ValueError("Aggregated customer measure fails the 5-percentage-point grid screen")
    return result


def draw(ax, frame, horizontal, *, positions=None):
    for model, label, color, linestyle, marker in MODELS:
        selected = frame[frame.model == model].sort_values(horizontal)
        x = selected[horizontal].to_numpy() if positions is None else positions
        ax.plot(x, selected.customer_product2_calendar, color=color, ls=linestyle,
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
    note = "Competition sums both sellers; outside-option purchases do not count as product 2."
    fig.suptitle(title + "\n" + subtitle + "\n" + note, fontsize=11)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HOME / "customer_product2_per_period")
    args = parser.parse_args()
    data, all_grids, selections, inputs, reports, baseline = load_sources()
    data = customer_flows(data).sort_values(["p0", "rho", "model"])
    checks = validation(data, all_grids, selections)
    root = args.output.resolve()
    figures, tables = root / "figures", root / "data"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(exist_ok=True)
    data.to_csv(tables / "customer_product2_summary.csv", index=False)
    checks.to_csv(tables / "customer_product2_grid_stability.csv", index=False)
    differences = []
    for (p0, rho), group in data.groupby(["p0", "rho"]):
        monopoly = float(group[group.model == "monopoly"].iloc[0].customer_product2_calendar)
        for row in group[group.model != "monopoly"].itertuples():
            differences.append(dict(p0=p0, rho=rho, model=row.model,
                                    monopoly_customer_product2=monopoly,
                                    competition_customer_product2=row.customer_product2_calendar,
                                    competition_minus_monopoly_pp=100*(row.customer_product2_calendar-monopoly)))
    pd.DataFrame(differences).to_csv(tables / "competition_vs_monopoly.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})
    paths = []
    pdf_path = figures / "customer_product2_comparison.pdf"
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
            maximum = float(group.customer_product2_calendar.max())
            ax.set(title=rf"$p_0={p0:g}$", xlabel=r"Persistence $\rho$", ylim=(0, maximum*1.14 if maximum else .01),
                   ylabel="Customer receives product 2\n(% of calendar periods)")
            ax.set_xticks(np.arange(len(RHOS)), [f"{rho:g}" for rho in RHOS])
        finish(fig, list(axes.flat), "Product 2 received by the customer: monopoly vs competition", local_scales=True)
        save(fig, "01_customer_product2_by_rho")

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), sharey=True, layout="constrained")
        maximum = float(data.customer_product2_calendar.max())
        for ax, rho in zip(axes, RHOS, strict=True):
            draw(ax, data[data.rho == rho], "p0")
            ax.set(title=rf"$\rho={rho:g}$", xlabel=r"Outside option $p_0$", ylim=(0, maximum*1.1))
            ax.set_xticks(P0S, [f"{p0:g}" for p0 in P0S])
        axes[0].set_ylabel("Customer receives product 2\n(% of calendar periods)")
        finish(fig, list(axes), "Product 2 received per period as the outside option changes")
        save(fig, "02_customer_product2_by_p0")

        for p0 in P0S:
            group = data[data.p0 == p0]
            fig, ax = plt.subplots(figsize=(7.8, 5.1), layout="constrained")
            draw(ax, group, "rho", positions=np.arange(len(RHOS)))
            maximum = float(group.customer_product2_calendar.max())
            ax.set(xlabel=r"Persistence $\rho$ (equally spaced cases)", ylim=(0, maximum*1.14),
                   ylabel="Customer receives product 2 (% of calendar periods)")
            ax.set_xticks(np.arange(len(RHOS)), [f"{rho:g}" for rho in RHOS])
            finish(fig, [ax], rf"Product 2 received by the customer | $p_0={p0:g}$")
            save(fig, "p0_" + str(p0).replace(".", "p") + "_customer_product2")
    paths.append(pdf_path)
    verify_inputs(inputs)
    source = Path(__file__).resolve()
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), status="complete", p0s=P0S, rhos=RHOS,
        displayed_n=GRID, source_reports=reports, shared_economic_configuration=baseline,
        source_input_sha256=inputs, reporting_source_sha256={str(source.relative_to(ROOT)): digest(source)},
        generated_file_sha256={str(p.relative_to(root)): digest(p) for p in [*paths, *tables.glob("*.csv")]},
        measure="Probability customer receives product 2 from any strategic seller in a calendar period",
        formula="seller_count * product2_calendar = (1-outside_rate) * product2_conditional",
        seller_count={"monopoly": 1, "competition": 2, "conditional_competition": 2},
        outside_option="Remains in calendar denominator and contributes zero to product 2 receipt",
        uncertainty="Deterministic stationary finite-grid flows, not Monte Carlo",
        grid_validation=dict(all_passed=bool(checks.passed.all()), probability_change_limit_pp=5,
                             max_customer_product2_change_pp=float(checks.change_pp.max())),
        accounting_max_residual=float(data.customer_product2_identity_residual.max()),
        preservation="Source files verified before and after rendering; historical outputs and paused n201 campaign unchanged",
        scope="Finite-grid results and sensitivity checks; no continuous-state convergence or monotonicity theorem",
    )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    lines = [
        "# Producto 2 recibido por el customer por período", "",
        "Comparación separada del experimento 08: monopolio, OE marginal y OE condicional.", "",
        "La medida es la probabilidad de que el comprador reciba producto 2 de algún seller en un período calendario.",
        "En competencia se suman los dos sellers simétricos; en monopolio se cuenta el único seller.",
        "La opción externa permanece en el denominador de períodos y no cuenta como producto 2.",
        "No se condiciona en que el customer compre a un seller ni se multiplica por la probabilidad de éxito p2.", "",
        "```text", "Monopolio:   customer_product2_calendar = product2_calendar",
        "Competencia: customer_product2_calendar = 2 * product2_calendar",
        "Ambos:       customer_product2_calendar = (1-outside_rate) * product2_conditional", "```", "",
        "Se usan las soluciones estacionarias auditadas de 101 nodos, con los seis p0 y los tres rho comunes publicados.",
        "Son flujos estacionarios deterministas, sin Monte Carlo ni intervalos de simulación.",
        f"La nueva medida agregada también pasa el filtro de sensibilidad: cambio máximo {checks.change_pp.max():.4f} puntos porcentuales entre las grillas de validación, frente al límite de 5.",
        "El contraste de grillas no prueba convergencia al modelo continuo.",
        "La campaña de 201/251 nodos permanece pausada y no se usa en estas figuras.", "",
        "## Figuras", "",
        "- [Todos los gráficos en un PDF](figures/customer_product2_comparison.pdf).",
        "- [Comparación frente a rho, un panel por p0](figures/01_customer_product2_by_rho.png).",
        "- [Comparación frente a p0, un panel por rho](figures/02_customer_product2_by_p0.png).", "",
    ]
    for p0 in P0S:
        name = "p0_" + str(p0).replace(".", "p") + "_customer_product2.png"
        lines.append(f"- [p0 = {p0:g}](figures/{name}).")
    lines += ["", "## Datos y reproducción", "",
              "- `data/customer_product2_summary.csv`: flujos por seller y por customer, con sus identidades contables.",
              "- `data/competition_vs_monopoly.csv`: diferencias de competencia menos monopolio en puntos porcentuales.",
              "- `data/customer_product2_grid_stability.csv`: contraste de la medida agregada entre grillas.",
              "- `manifest.json`: procedencia, huellas y verificaciones.", "", "```bash",
              "MPLCONFIGDIR=/tmp/customer-product2-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
              "  .venv/bin/python -B -m oblivious.oe_distribution_fixed_p0.code.customer_product2_report", "```", ""]
    (root / "README.md").write_text("\n".join(lines))
    print(json.dumps(dict(output=str(root), rows=len(data), pdf_pages=8,
                         max_grid_change_pp=float(checks.change_pp.max()),
                         customer_identity_residual=float(data.customer_product2_identity_residual.max())), indent=2))


if __name__ == "__main__":
    main()
