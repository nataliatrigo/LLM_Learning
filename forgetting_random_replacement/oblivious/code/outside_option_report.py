"""Publish the two reference three-panel comparisons against the outside option.

Reuses verified random-replacement equilibria; does not solve new cases.
Run from the repository root with python -m
forgetting_random_replacement.oblivious.code.outside_option_report.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from forgetting_random_replacement.code.model import evidence_matched_n
from .run_experiment import (CONFIG, HOME, collect, now, numerical_sources,
                             save_source_manifest, write_csv, write_json)

REFERENCE_RHOS = (.6, .7, .8)
MODELS = ((1, "monopoly", "Monopoly", "#0072B2", "-", "o"),
          (2, "marginal", "Marginal OE", "#D55E00", "--", "s"),
          (2, "conditional", "Conditional OE", "#009E73", "-.", "^"))
# Retain full numerical accounting while showing the requested model comparisons.
PLOT_MODELS = tuple(model for model in MODELS if model[1] != "conditional")


def publish_comparisons(rows, validation=None):
    raise RuntimeError(
        "Legacy rho-matched figure publication is retired. Use python -m "
        "forgetting_random_replacement.oblivious.code.capacity_report."
    )
    if validation is None:
        from .figures import validated_campaign
        validation = validated_campaign(rows)
    lookup = {(row["capacity"], row["p0"], row["sellers"], row["closure"]): row for row in rows}
    plotted, maximum_error = [], 0.
    for rho in REFERENCE_RHOS:
        capacity = evidence_matched_n(rho)
        for p0 in CONFIG["outside_options"]:
            for sellers, closure, *_ in MODELS:
                source = lookup[capacity, p0, sellers, closure]
                own = source["product2_conditional"]
                customer = sellers * source["product2_calendar"]
                purchases = sellers * source["sale_rate"]
                errors = (abs(source["product2_calendar"] - source["sale_rate"] * own),
                          abs(customer - (1 - source["outside_rate"]) * own),
                          abs(purchases + source["outside_rate"] - 1))
                maximum_error = max(maximum_error, *errors)
                if (not source["verified"] or max(errors) > 1e-10
                        or not 0 <= own <= 1 or not 0 <= customer <= purchases + 1e-12):
                    raise ArithmeticError(f"Outside-option comparison accounting failed: {source['case']}")
                plotted.append(dict(source, reference_rho=rho,
                    customer_product2_calendar=customer,
                    customer_product1_calendar=purchases-customer,
                    customer_purchase_calendar=purchases))
    table = HOME / "data/outside_option_comparisons.csv"
    write_csv(table, plotted)
    lookup = {(r["reference_rho"], r["p0"], r["sellers"], r["closure"]): r for r in plotted}
    specifications = (
        ("product2_conditional", "01_product2_given_sale_by_p0",
         "Product 2 among own sales as the outside option changes",
         "Product 2 (% of own sales)",
         "Product mix when the seller sells; one seller in each model.", 1.),
        ("customer_product2_calendar", "02_customer_product2_by_p0",
         "Product 2 received per period as the outside option changes",
         "Customer receives product 2\n(% of calendar periods)",
         "Competition sums both sellers; outside-option purchases do not count as product 2.",
         min(1., max(.1, math.ceil(11 * max(r["customer_product2_calendar"] for r in plotted))/10))))
    paths = []
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11,
                         "figure.titlesize": 12, "pdf.fonttype": 42}):
        for metric, name, title, ylabel, note, upper in specifications:
            fig, axes = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True, layout="constrained")
            for ax, rho in zip(axes, REFERENCE_RHOS, strict=True):
                for sellers, closure, label, color, linestyle, marker in PLOT_MODELS:
                    values = [lookup[rho, p0, sellers, closure][metric] for p0 in CONFIG["outside_options"]]
                    ax.plot(CONFIG["outside_options"], values, color=color, linestyle=linestyle,
                            marker=marker, markersize=5.5, linewidth=1.7, label=label,
                            markerfacecolor=color if sellers == 1 else "none")
                capacity = evidence_matched_n(rho)
                ax.set_title(f"Reference ρ = {rho:g}\nC = {capacity}; effective ρ = {1-1/capacity:.4g}")
                ax.set_xlabel("Outside option p₀")
                ax.set_xticks(CONFIG["outside_options"], [f"{p0:g}" for p0 in CONFIG["outside_options"]])
                ax.set_ylim(0, upper)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
                if upper == 1:
                    ax.set_yticks(np.arange(0, 1.01, .2))
                ax.grid(alpha=.22)
                ax.spines[["top", "right"]].set_visible(False)
            axes[0].set_ylabel(ylabel)
            handles, labels = axes[0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="outside lower center", ncol=len(PLOT_MODELS), frameon=False)
            fig.suptitle(title + "\nRandom replacement | c₂ = 0.65 | exact count states\n" + note)
            target = HOME / "figures/experiment8/by_outside_option" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            for suffix in (".png", ".pdf"):
                path = target.with_suffix(suffix)
                fig.savefig(path, dpi=190, facecolor="white", bbox_inches="tight")
                paths.append(path)
            plt.close(fig)
    used_cases = {r["case"] for r in plotted}
    write_json(HOME / "data/outside_option_comparison_manifest.json", {
        "created_utc": now(), "status": "complete", "language": "English",
        "reference_rhos": REFERENCE_RHOS, "outside_options": CONFIG["outside_options"],
        "rows": len(plotted), "unique_checkpoints": len(used_cases),
        "displayed_models": [model[2] for model in PLOT_MODELS],
        "conditional_results_note": "Conditional OE results remain in the saved cases and tables; they are omitted from the figures.",
        "own_sales_formula": "product2_conditional = product2_calendar / sale_rate",
        "customer_calendar_formula": "sellers * product2_calendar = (1-outside_rate) * product2_conditional",
        "maximum_accounting_residual": maximum_error,
        "capacity_note": "Reference rho 0.6 and 0.7 both map to capacity 3 and identical models.",
        "validated_sources": validation["sources"],
        "source_checkpoints": {case["case"]: case["checkpoint_sha256"] for case in validation["cases"]
                               if case["case"] in used_cases},
        "reporting_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "generated_file_sha256": {str(path.relative_to(HOME)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [*paths, table]}})
    return paths


def main():
    raise RuntimeError(
        "Legacy rho-matched figure publication is retired. Use python -m "
        "forgetting_random_replacement.oblivious.code.capacity_report."
    )
    from .figures import validated_campaign
    from .report import summarize

    rows = collect(numerical_sources())
    validation = validated_campaign(rows)
    paths = publish_comparisons(rows, validation)
    manifest_path = HOME / "figures/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = {entry["path"]: entry for entry in manifest["figures"]}
    entries.update({str(path.relative_to(HOME)): {"path": str(path.relative_to(HOME)),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths})
    manifest["figures"] = list(entries.values())
    write_json(manifest_path, manifest)
    summarize(rows, validation)
    save_source_manifest()
    print(f"Published {len(paths)} files for the two outside-option comparisons; {len(entries)} figure files in total.")


if __name__ == "__main__":
    main()
