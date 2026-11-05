"""English publication figures for the independently checked count-state campaign."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .run_experiment import CONFIG, HOME, numerical_sources, write_json
from forgetting_random_replacement.code.model import evidence_matched_n

MODELS = ((1, "monopoly", "Monopoly", "#0072B2", "-"),
          (2, "marginal", "Marginal OE", "#D55E00", "--"),
          (2, "conditional", "Conditional OE", "#009E73", ":"))
# Keep all saved models in the campaign; only the requested comparisons are drawn.
PLOT_MODELS = tuple(model for model in MODELS if model[1] != "conditional")


def validated_campaign(rows):
    """Bind publication inputs to the current complete campaign audit."""
    audit_path = HOME / "data/validation.json"
    if not audit_path.exists() or not json.loads(audit_path.read_text()).get("verified"):
        raise ArithmeticError("Publication requires a complete passing campaign audit")
    validation = json.loads(audit_path.read_text())
    if validation["sources"] != numerical_sources() or len(rows) != validation["passed_case_count"]:
        raise ArithmeticError("Publication inputs differ from the validated campaign")
    for case in validation["cases"]:
        stem = HOME / "data/cases" / case["case"]
        for suffix, key in ((".npz", "checkpoint_sha256"), (".json", "record_sha256")):
            if hashlib.sha256(stem.with_suffix(suffix).read_bytes()).hexdigest() != case[key]:
                raise ArithmeticError(f"Publication input changed after audit: {case['case']}")
    return validation


def publish(rows):
    raise RuntimeError(
        "Legacy rho-matched figure publication is retired. Use python -m "
        "forgetting_random_replacement.oblivious.code.capacity_report."
    )
    validation = validated_campaign(rows)
    lookup = {(r["capacity"], r["p0"], r["sellers"], r["closure"]): r for r in rows}
    capacities, rhos = CONFIG["capacities"], CONFIG["reference_rhos"]
    paths = []

    def save(fig, relative):
        target = HOME / "figures" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            path = target.with_suffix("." + extension)
            fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    def style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.18)

    def retention_axis(ax):
        ax.set_xticks(range(len(rhos)), [f"{rho:g}\nC={evidence_matched_n(rho)}" for rho in rhos])
        ax.set_xlabel("Reference retention ρ and matched memory capacity C")
        style(ax)

    def metric_lines(ax, p0, metric, scale=1):
        for sellers, closure, label, color, linestyle in PLOT_MODELS:
            values = [scale * lookup[evidence_matched_n(rho), p0, sellers, closure][metric] for rho in rhos]
            ax.plot(range(len(rhos)), values, label=label, color=color,
                    linestyle=linestyle, marker="o", markersize=4)
        retention_axis(ax)

    with plt.rc_context({"font.size": 10, "axes.titlesize": 11,
                         "figure.titlesize": 13, "pdf.fonttype": 42}):
        for p0 in CONFIG["outside_options"]:
            directory = Path("experiment8") / f"p0_{str(p0).replace('.', 'p')}"
            fig, axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained", sharex=True, sharey=True)
            distribution, dist_axes = plt.subplots(2, 3, figsize=(12, 7), layout="constrained", sharex=True)
            for capacity, ax, dist_ax in zip(capacities, axes.flat, dist_axes.flat):
                for sellers, closure, label, color, linestyle in PLOT_MODELS:
                    row = lookup[capacity, p0, sellers, closure]
                    with np.load(HOME / "data/cases" / (row["case"] + ".npz"), allow_pickle=False) as saved:
                        x = np.arange(capacity + 1) / capacity
                        ax.plot(x, 100 * saved["policy"], color=color, linestyle=linestyle,
                                marker="o", markersize=3, label=label)
                        states, joint = saved["states"], saved["joint"]
                        own = np.bincount(states.ravel(), weights=np.repeat(joint / sellers, sellers),
                                          minlength=capacity + 1)
                        dist_ax.plot(x, own, color=color, linestyle=linestyle,
                                     marker="o", markersize=3, label=label)
                title = f"C = {capacity}; effective ρ = {1 - 1 / capacity:.4g}"
                for panel in (ax, dist_ax):
                    panel.set_title(title)
                    panel.set_xlabel("Retained success fraction S/C")
                    style(panel)
                ax.set_ylim(-3, 103)
                ax.set_ylabel("Product 2 probability (% of own sales)")
                dist_ax.set_ylabel("Calendar-time probability")
            axes.flat[0].legend(fontsize=9)
            dist_axes.flat[0].legend(fontsize=9)
            fig.suptitle(f"Random replacement: optimal own-count policies | p₀ = {p0:g}")
            distribution.suptitle(f"Random replacement: stationary own-count distributions | p₀ = {p0:g}")
            save(fig, directory / "01_policy_comparison")
            save(distribution, directory / "05_stationary_count_distributions")

            fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
            metric_lines(ax, p0, "product2_conditional", 100)
            ax.set_ylim(-2, 102)
            ax.set_ylabel("Product 2 (% of own sales)")
            ax.set_title(f"Random replacement: product 2 use | p₀ = {p0:g}")
            ax.legend()
            save(fig, directory / "02_product2_usage_comparison")

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), layout="constrained")
            metric_lines(axes[0], p0, "profit_calendar")
            metric_lines(axes[1], p0, "value_initial")
            axes[0].set_ylabel("Profit per seller and calendar period")
            axes[1].set_ylabel("Expected discounted profit per seller")
            axes[0].set_title("Stationary profit flow")
            axes[1].set_title("Actual value from symmetric mature initialization")
            axes[0].legend()
            fig.suptitle(f"Random replacement: realized profits | p₀ = {p0:g}")
            save(fig, directory / "03_profit_comparison")

            fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
            metric_lines(ax, p0, "outside_rate", 100)
            ax.set_ylim(-2, 102)
            ax.set_ylabel("Outside option (% of market calendar periods)")
            ax.set_title(f"Random replacement: outside-option choice | p₀ = {p0:g}")
            ax.legend()
            save(fig, directory / "04_outside_option_comparison")

        fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained", sharey=True)
        for p0, ax in zip(CONFIG["outside_options"], axes.flat):
            metric_lines(ax, p0, "product2_conditional", 100)
            ax.set_title(f"p₀ = {p0:g}")
            ax.set_ylim(-2, 102)
            ax.set_ylabel("Product 2 (% of own sales)")
        axes.flat[0].legend(fontsize=9)
        fig.suptitle("Experiment 8 with random replacement: product 2 use")
        save(fig, Path("experiment8/00_product2_usage_overview"))

        fig, ax = plt.subplots(figsize=(8.5, 5.5), layout="constrained")
        for p0, color, marker, offset in ((.1, "#0072B2", "o", -15), (.3, "#D55E00", "s", 8)):
            values = [100 * lookup[3, p0, sellers, "monopoly" if sellers == 1 else "marginal"]
                      ["product2_conditional"] for sellers in range(1, 6)]
            ax.plot(range(1, 6), values, color=color, marker=marker, label=f"p₀ = {p0:g}")
            for x, y in enumerate(values, 1):
                ax.annotate(f"{y:.2f}%", (x, y), xytext=(0, offset), textcoords="offset points",
                            ha="center", color=color, fontsize=9)
        ax.set_title("Product 2 use as the number of sellers increases\n"
                     "Random replacement | reference ρ = 0.6 | capacity C = 3 | effective ρ = 2/3")
        ax.set_xticks(range(1, 6))
        ax.set_ylim(-6, 104)
        ax.set_xlim(.8, 5.2)
        ax.set_xlabel("Number of sellers (N = 1: monopoly; N ≥ 2: marginal OE)")
        ax.set_ylabel("Product 2 (% of own sales)")
        ax.legend()
        style(ax)
        save(fig, Path("seller_count/01_product2_by_seller_count"))
    from .outside_option_report import publish_comparisons
    paths.extend(publish_comparisons(rows, validation))
    write_json(HOME / "figures/manifest.json", {
        "language": "English", "figures": [
            {"path": str(path.relative_to(HOME)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for path in paths], "validated_cases": len(rows),
        "displayed_models": [model[2] for model in PLOT_MODELS],
        "conditional_results_note": "Conditional OE results remain in the saved cases and tables; they are omitted from the figures.",
        "reference_rho_note": "Reference rho is mapped to integer capacity; effective rho is 1-1/C.",
        "grid_note": "Exact count states; no reputation-grid interpolation or 21/31-grid comparison."})
    from .report import summarize
    summarize(rows, validation)
    print(f"Published {len(paths)} figure files", flush=True)
