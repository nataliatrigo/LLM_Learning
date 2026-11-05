"""English publication for the complete C=1,...,30 capacity campaign.

The report reads certified case summaries and binds every input to the saved
campaign audit. It does not solve equilibria or change historical outputs.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from .run_experiment import (ROOT, now,
                             numerical_sources, write_csv, write_json)
from .capacity_sweep import CAMPAIGN, CONFIG as CAMPAIGN_CONFIG

CAPACITIES = tuple(range(1, 31))
P0S = tuple(CAMPAIGN_CONFIG["outside_options"])
EXTENSION_P0S = (.1, .3)
EXTENSION_CAPACITIES = tuple(CAMPAIGN_CONFIG.get("extension_capacities", CAPACITIES))
P0_PANEL_CAPACITIES = tuple(CAMPAIGN_CONFIG["outside_option_panel_capacities"])
SELLER_PANEL_CAPACITIES = ((3, 10, 20, 30) if all(c in EXTENSION_CAPACITIES for c in (3, 10, 20, 30))
    else tuple(EXTENSION_CAPACITIES[index] for index in
               np.unique(np.linspace(0, len(EXTENSION_CAPACITIES)-1, min(4, len(EXTENSION_CAPACITIES))).astype(int))))
MODELS = ((1, "monopoly", "Monopoly", "#0072B2", "-", "o"),
          (2, "marginal", "Marginal OE", "#D55E00", "--", "s"),
          (2, "conditional", "Conditional OE", "#009E73", "-.", "^"))
PLOT_MODELS = tuple(model for model in MODELS if model[1] != "conditional")
SELLER_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6A3D9A")
PUBLISHED_PLOT_IDS = tuple(f"{number:02d}" for number in (1, 2, 3, 4, 6, 7, 9, 10, 11))


def digest(path):
    """Hash large joint-law checkpoints without loading them into memory."""
    checksum = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def expected_keys():
    keys = {(c, p0, n, closure) for c in CAPACITIES for p0 in P0S
            for n, closure, *_ in MODELS}
    keys.update((c, p0, n, "marginal") for c in EXTENSION_CAPACITIES
                for p0 in EXTENSION_P0S for n in (3, 4, 5))
    return keys


def _typed_rows(rows):
    converted = []
    integers = {"capacity", "sellers", "count_states", "joint_states"}
    numbers = {"p0", "effective_rho", "product2_conditional", "sale_rate", "outside_rate",
               "product2_calendar", "profit_calendar", "value_initial", "perceived_value_initial"}
    for raw in rows:
        row = dict(raw)
        for name in integers & row.keys():
            row[name] = int(row[name])
        for name in numbers & row.keys():
            row[name] = float(row[name])
        if isinstance(row.get("verified"), str):
            row["verified"] = row["verified"].lower() == "true"
        converted.append(row)
    return converted


def validated_inputs(rows, validation=None):
    """Reject incomplete, changed, or unverified publication inputs."""
    if validation is None:
        path = CAMPAIGN / "data/validation.json"
        if not path.exists():
            raise ArithmeticError("Capacity publication requires a saved complete campaign audit")
        validation = json.loads(path.read_text())
    if validation.get("verified") is not True:
        raise ArithmeticError("Capacity publication requires a passing complete campaign audit")
    if validation.get("sources") != numerical_sources():
        raise ArithmeticError("Capacity numerical sources changed after the saved audit")
    rows = _typed_rows(rows)
    actual = [(r["capacity"], r["p0"], r["sellers"], r["closure"]) for r in rows]
    if len(actual) != len(set(actual)) or set(actual) != expected_keys():
        raise ArithmeticError(f"Capacity publication requires all {len(expected_keys())} distinct configured cases")
    if not all(r.get("verified") is True for r in rows):
        raise ArithmeticError("An unverified case cannot be published")
    audited = {case["case"]: case for case in validation["cases"]}
    if len(audited) != len(rows) or {r["case"] for r in rows} != set(audited):
        raise ArithmeticError("Capacity publication case list differs from the audit")
    for row in rows:
        stem = CAMPAIGN / "data/cases" / row["case"]
        evidence = audited[row["case"]]
        for suffix, key in ((".npz", "checkpoint_sha256"), (".json", "record_sha256")):
            if digest(stem.with_suffix(suffix)) != evidence[key]:
                raise ArithmeticError(f"Capacity input changed after audit: {row['case']}")
        record = json.loads(stem.with_suffix(".json").read_text())
        for filename, checksum in record.get("solver_sources", {}).items():
            path = Path(filename)
            path = path if path.is_absolute() else ROOT/path
            if digest(path) != checksum:
                raise ArithmeticError(f"Capacity solver source changed after audit: {filename}")
        for metric in ("product2_conditional", "sale_rate", "outside_rate", "product2_calendar",
                       "profit_calendar", "value_initial", "perceived_value_initial"):
            if row[metric] != record["summary"][metric]:
                raise ArithmeticError(f"Capacity summary changed after audit: {row['case']}, {metric}")
    return rows, validation


def _figure_rows(rows):
    exported, maximum_error = [], 0.
    for row in rows:
        n = row["sellers"]
        customer_product2 = n * row["product2_calendar"]
        purchase_rate = n * row["sale_rate"]
        errors = (abs(row["product2_calendar"] - row["sale_rate"] * row["product2_conditional"]),
                  abs(customer_product2 - (1-row["outside_rate"]) * row["product2_conditional"]),
                  abs(purchase_rate + row["outside_rate"] - 1))
        maximum_error = max(maximum_error, *errors)
        if (not np.all(np.isfinite(errors)) or max(errors) > 1e-9
                or not 0 <= row["product2_conditional"] <= 1
                or not 0 <= customer_product2 <= purchase_rate + 1e-12):
            raise ArithmeticError(f"Capacity figure accounting failed: {row['case']}")
        exported.append(dict(row, customer_product2_calendar=customer_product2,
                             customer_product1_calendar=purchase_rate-customer_product2,
                             customer_purchase_calendar=purchase_rate,
                             expected_replacement_fraction=1/row["capacity"]))
    return exported, maximum_error


def _style(ax):
    ax.grid(alpha=.2)
    ax.spines[["top", "right"]].set_visible(False)


def _capacity_axis(ax, capacities=CAPACITIES):
    ax.set_xlim(min(capacities)-.5, max(capacities)+.5)
    ticks = ((1, 5, 10, 15, 20, 25, 30) if tuple(capacities) == CAPACITIES
             else tuple(capacities) if len(capacities) <= 10
             else tuple(capacities[index] for index in
                        np.unique(np.linspace(0, len(capacities)-1, 7).astype(int))))
    ax.set_xticks(ticks)
    ax.set_xlabel("Memory capacity C (integer observations)")
    _style(ax)


def _percent_axis(ax):
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.yaxis.set_major_formatter(PercentFormatter(1))


def _legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=min(3, len(handles)), frameon=False)


def publish(rows, validation=None, *, plot_ids=None):
    """Publish outcome and policy figures from all configured certified cases."""
    if plot_ids is not None and not set(plot_ids) <= set(PUBLISHED_PLOT_IDS):
        raise ValueError("Requested plot is not in the current publication set")
    rows, validation = validated_inputs(rows, validation)
    plotted, accounting_error = _figure_rows(rows)
    lookup = {(r["capacity"], r["p0"], r["sellers"], r["closure"]): r for r in plotted}
    paths = []
    policy_figures = None
    old_manifest_path = CAMPAIGN / "figures/manifest.json"
    if old_manifest_path.exists():
        previous = json.loads(old_manifest_path.read_text())
        policy_figures = previous.get("policy_figures")

    def save(fig, name):
        if plot_ids is not None and name[:2] not in plot_ids:
            plt.close(fig)
            return
        target = CAMPAIGN / "figures" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        for extension in ("png", "pdf"):
            path = target.with_suffix("." + extension)
            fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    comparisons = (
        ("product2_conditional", "01_product2_own_sales_by_capacity",
         "Product 2 among own sales", "Product 2 (% of own sales)", True),
        ("customer_product2_calendar", "02_customer_product2_by_capacity",
         "Product 2 received per calendar period", "Customer receives product 2\n(% of calendar periods)", True),
        ("profit_calendar", "03_profit_calendar_by_capacity",
         "Stationary profit flow", "Profit per seller and calendar period", False),
        ("value_initial", "04_actual_value_by_capacity",
         "Actual discounted profit", "Expected discounted profit per seller", False))
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11,
                         "figure.titlesize": 13, "pdf.fonttype": 42}):
        overview_columns = min(3, len(P0S))
        overview_rows = (len(P0S)+overview_columns-1)//overview_columns
        for metric, name, title, ylabel, percent in comparisons:
            fig, axes = plt.subplots(overview_rows, overview_columns,
                figsize=(14, 3.7*overview_rows+1), layout="constrained",
                sharex=True, sharey=True, squeeze=False)
            for ax, p0 in zip(axes.flat, P0S):
                for n, closure, label, color, linestyle, marker in PLOT_MODELS:
                    values = [lookup[c, p0, n, closure][metric] for c in CAPACITIES]
                    ax.plot(CAPACITIES, values, color=color, linestyle=linestyle,
                            marker=marker, markersize=3.5, linewidth=1.5,
                            markerfacecolor=color if n == 1 else "none", label=label)
                ax.set_title(f"p₀ = {p0:g}")
                ax.set_ylabel(ylabel)
                _capacity_axis(ax)
                if percent:
                    _percent_axis(ax)
            for ax in list(axes.flat)[len(P0S):]:
                ax.set_visible(False)
            _legend(fig, axes.flat[0])
            fig.suptitle(f"Random replacement: {title.lower()}\nExact integer capacities C = 1, …, 30")
            save(fig, name)

        for metric, name, title, ylabel in (
            ("product2_conditional", "06_product2_given_sale_by_p0",
             "Product 2 among own sales as the outside option changes", "Product 2 (% of own sales)"),
            ("customer_product2_calendar", "07_customer_product2_by_p0",
             "Product 2 received per period as the outside option changes",
             "Customer receives product 2\n(% of calendar periods)")):
            columns = 2 if len(P0_PANEL_CAPACITIES) > 6 else min(3, len(P0_PANEL_CAPACITIES))
            panel_rows = (len(P0_PANEL_CAPACITIES)+columns-1)//columns
            fig, axes = plt.subplots(panel_rows, columns,
                figsize=(4.2*columns, 3.6*panel_rows+1.3),
                layout="constrained", sharey=True, squeeze=False)
            for ax, capacity in zip(axes.flat, P0_PANEL_CAPACITIES):
                for n, closure, label, color, linestyle, marker in PLOT_MODELS:
                    values = [lookup[capacity, p0, n, closure][metric] for p0 in P0S]
                    ax.plot(P0S, values, color=color, linestyle=linestyle, marker=marker,
                            markersize=5.5, linewidth=1.7,
                            markerfacecolor=color if n == 1 else "none", label=label)
                ax.set_title(f"C = {capacity}; effective ρ = {1-1/capacity:.4g}")
                ax.set_xlabel("Outside option p₀")
                ax.set_xticks(P0S, [f"{p0:g}" for p0 in P0S])
                ax.tick_params(axis="x", labelsize=9)
                _percent_axis(ax)
                _style(ax)
            for ax in list(axes.flat)[len(P0_PANEL_CAPACITIES):]:
                ax.set_visible(False)
            for ax in axes[:, 0]:
                ax.set_ylabel(ylabel)
            _legend(fig, axes.flat[0])
            note = ("One seller's product mix conditional on an own sale."
                    if metric == "product2_conditional" else
                    "Competition sums both sellers; outside-option selection contributes zero.")
            fig.suptitle(title + "\nRandom replacement | exact count states\n" + note, fontsize=12)
            save(fig, name)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), layout="constrained", sharey=True)
        for ax, p0 in zip(axes, EXTENSION_P0S, strict=True):
            for n, color in zip(range(1, 6), SELLER_COLORS, strict=True):
                closure = "monopoly" if n == 1 else "marginal"
                values = [lookup[c, p0, n, closure]["product2_conditional"] for c in EXTENSION_CAPACITIES]
                label = "N = 1 (monopoly)" if n == 1 else f"N = {n} (marginal OE)"
                ax.plot(EXTENSION_CAPACITIES, values, color=color, marker="o", markersize=3.5,
                        linewidth=1.5, label=label)
            ax.set_title(f"p₀ = {p0:g}")
            ax.set_ylabel("Product 2 (% of own sales)")
            _percent_axis(ax)
            _capacity_axis(ax, EXTENSION_CAPACITIES)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
        fig.suptitle("Random replacement: memory capacity and the number of sellers\n"
                     f"Exact integer capacities C = {min(EXTENSION_CAPACITIES)}, …, {max(EXTENSION_CAPACITIES)}")
        save(fig, "09_product2_by_capacity_and_seller_count")

    policy_ids = {"10", "11"} if plot_ids is None else set(plot_ids) & {"10", "11"}
    if policy_ids:
        from .capacity_policy_report import publish as publish_policies
        paths.extend(publish_policies(rows, validation, figure_ids=policy_ids))
        policy_figures = dict(path="data/policy_figure_manifest.json",
                             sha256=digest(CAMPAIGN / "data/policy_figure_manifest.json"))

    table = CAMPAIGN / "data/figure_data.csv"
    write_csv(table, plotted)
    entries = {}
    if plot_ids is not None:
        previous = json.loads((CAMPAIGN / "figures/manifest.json").read_text())
        entries.update({item["path"]: item for item in previous["figures"]
                        if (item["path"].startswith("figures/policies/")
                            or Path(item["path"]).name[:2] in PUBLISHED_PLOT_IDS)
                        and not ({"10", "11"} <= policy_ids
                                and item["path"].startswith("figures/policies/"))})
    entries.update({str(path.relative_to(CAMPAIGN)):
                    dict(path=str(path.relative_to(CAMPAIGN)), sha256=digest(path)) for path in paths})
    manifest = dict(created_utc=now(), status="complete", language="English",
        validated_cases=len(rows), capacities=CAPACITIES, outside_options=P0S,
        published_plot_ids=PUBLISHED_PLOT_IDS,
        displayed_comparison_models=[model[1] for model in PLOT_MODELS],
        conditional_results="Retained in numerical data, tables and audits; omitted from comparison figures",
        extension_capacities=EXTENSION_CAPACITIES,
        p0_comparison_capacities=P0_PANEL_CAPACITIES,
        seller_comparison_capacities=SELLER_PANEL_CAPACITIES,
        own_sales_formula="product2_conditional = product2_calendar / sale_rate",
        customer_calendar_formula="sellers * product2_calendar = (1-outside_rate) * product2_conditional",
        maximum_accounting_residual=accounting_error,
        reporting_source_sha256=digest(__file__),
        validation_path="data/validation.json", validation_sha256=digest(CAMPAIGN / "data/validation.json"),
        figure_data_path="data/figure_data.csv", figure_data_sha256=digest(table),
        figures=list(entries.values()),
        policy_figures=policy_figures,
        interpretation="C is the economic memory capacity, not a numerical interpolation grid. "
                       "Markers represent every solved integer capacity; connecting lines guide the eye.")
    write_json(CAMPAIGN / "figures/manifest.json", manifest)
    write_notes(plotted, validation, manifest)
    print(f"Published {len(paths)} capacity-campaign figure files from {len(rows)} validated cases.", flush=True)
    return manifest


def write_notes(rows, validation, manifest):
    lookup = {(r["capacity"], r["p0"], r["sellers"], r["closure"]): r for r in rows}
    base_count = len(CAPACITIES)*len(P0S)*len(MODELS)
    extension_count = len(EXTENSION_CAPACITIES)*len(EXTENSION_P0S)*3
    extension_range = f"C={min(EXTENSION_CAPACITIES)},...,{max(EXTENSION_CAPACITIES)}"
    panel_capacities_text = ", ".join(f"C{c}" for c in P0_PANEL_CAPACITIES)
    outside_options_text = ", ".join(f"{p0:g}" for p0 in P0S)
    policy_count = len(CAPACITIES)*len(P0S)*len(PLOT_MODELS)
    policy_state_count = sum(c+1 for c in CAPACITIES)*len(P0S)*len(PLOT_MODELS)
    has_count_views = (CAMPAIGN/"figures/policies/10_monopoly_policy_by_capacity_success_counts.pdf").exists()
    scope_note = ("The five-seller extension is limited to the certified range "
                  f"`{extension_range}`. Solver stress tests at larger capacities did not pass "
                  "the current independent stationary-convergence checks. Those exploratory "
                  "calculations are excluded from the published results; this does not establish "
                  "that an equilibrium fails to exist. Details are retained in "
                  "[exploratory solver checks](data/exploratory_solver_checks.json)."
                  if max(EXTENSION_CAPACITIES) < max(CAPACITIES) else "")
    readme = ["# Random-replacement forgetting: integer capacities 1 through 30", "",
        "This campaign uses every integer memory capacity `C=1,...,30` directly. "
        f"It compares monopoly and two-seller marginal/conditional OE at {len(P0S)} outside "
        f"options (`{outside_options_text}`), and extends marginal OE through five sellers "
        f"at `p0=.1,.3` over `{extension_range}`.", "",
        f"The published campaign contains **{len(rows)} distinct certified cases**: {base_count} Experiment-8 "
        f"cases plus {extension_count} additional three-, four-, and five-seller cases. The original "
        "rho-matched numerical data remain in the parent folder's `data/` directory; "
        "its older figure exports have been retired.", "",
        "Current comparison figures display Monopoly and Marginal OE. Conditional OE "
        "solutions remain in the saved data, tables, and numerical audits.", "",
        scope_note, "",
        "## Open the results", "",
        "- [Results and selected tables](RESULTS.md).",
        "- [Product 2 among own sales across C](figures/01_product2_own_sales_by_capacity.png).",
        "- [Customer product 2 per period across C](figures/02_customer_product2_by_capacity.png).",
        "- [Product 2 among own sales across the outside option](figures/06_product2_given_sale_by_p0.png).",
        "- [Customer product 2 per period across the outside option](figures/07_customer_product2_by_p0.png).",
        "- [All five seller counts across C](figures/09_product2_by_capacity_and_seller_count.png).",
        *( ["- [Monopoly policies at all capacities](figures/10_monopoly_policy_by_capacity.png).",
            "- [Marginal OE policies at all capacities](figures/11_oblivious_policy_by_capacity.png).",
            "- [Side-by-side policies at p0=.5](figures/policies/01_policy_comparison_p0_0p5.pdf).",
            "- [Adjacent C5, C6, and C7 policies](figures/policies/02_adjacent_capacities_C5_C6_C7_p0_0p5.pdf)."]
           if manifest.get("policy_figures") else [] ),
        *( ["- Policy maps using success counts: [Monopoly](figures/policies/10_monopoly_policy_by_capacity_success_counts.pdf) "
            "and [Marginal OE](figures/policies/11_oblivious_policy_by_capacity_success_counts.pdf).",
            "- [Policy comparison using success counts at p0=.1](figures/policies/01_policy_comparison_p0_0p1_success_counts.pdf)."]
           if has_count_views else [] ),
        "- [Complete case summary](data/summary.csv), [figure data](data/figure_data.csv), "
        "[numerical validation](data/validation.json), and [figure manifest](figures/manifest.json).", "",
        f"All {sum(item['path'].endswith('.pdf') for item in manifest['figures'])} figures are available in PDF. "
        "Policy comparisons are stored directly in `figures/policies/`, without subfolders or PNG copies. "
        "Calendar profit and actual discounted profit are also plotted against C.", "",
        "## What C means", "",
        "A mature seller remembers C binary observations. Its retained success count "
        "is `S=0,...,C`, with posterior `Beta(1+S,1+C-S)`. After a purchase, one old "
        "observation is chosen uniformly, independently of the new outcome, and replaced. "
        "Only the selected seller updates; outside-option selection freezes every memory.", "",
        "The expected fraction retained per own purchase is `rho_effective=1-1/C`: "
        "it ranges from zero at C1 to 29/30 at C30. C changes the memory model, "
        "rather than refining a numerical approximation. Integer capacities can produce "
        "changes in optimal policy and nonmonotone outcome curves; no smoothing is applied.", "",
        "The outside-option and policy comparison panels use these capacities:", "",
        "| Capacity C | Effective retention rho |", "|---:|---:|",
        *(f"| {c} | {1-1/c:.6f} |" for c in P0_PANEL_CAPACITIES), "",
        "The sharp peaks arise because adding one memory slot changes the available "
        "integer success counts, their posteriors, the optimal product choices, and the "
        "stationary probability of selling at each count. Product choices often switch "
        "between zero and one at an integer boundary. Straight lines connect separately "
        "solved capacities and make those discrete changes look pointed. See the "
        "adjacent-capacity example in [RESULTS.md](RESULTS.md).", "",
        "The actual joint law is retained on the exact seller-permutation quotient, "
        "including the correlated law of all four rivals with five sellers. Independent "
        "rival marginals do not replace the joint distribution.", "",
        "## Metrics and initialization", "",
        "Primary product-2 use is `product2_conditional`, the percentage of own "
        "sales using product 2. Customer receipt per calendar period sums the actual "
        "product-2 flow across every seller: `N * product2_calendar`. Outside-option "
        "selection stays in the calendar denominator and contributes zero receipt. "
        "Profit flow and discounted payoff are per seller. Actual discounted payoff "
        "is evaluated in the real joint chain, separately from perceived OE value.", "",
        "Discounted payoff uses the same symmetric mature initialization as the "
        "parent campaign: exact C/2 when C is even, and an independent equal "
        "floor(C/2)/ceil(C/2) mixture for each seller when C is odd. Both have "
        "initial mean posterior one half. The economic parameters remain "
        "`p1=.35,p2=.8,c1=.05,c2=.65,revenue=1,gamma=.98`.", "",
        "## Validation and reproduction", "",
        "Publication requires every configured case to pass the campaign audit. "
        "The figure generator verifies saved record/checkpoint hashes and the "
        "accounting identities for own-sales and customer-calendar product 2. "
        "`data/validation.json` records the numerical checks for each case. "
        "These are floating-point checks of finite exact-count models; they do not "
        "prove equilibrium uniqueness or fast convergence to the invariant law.", "",
        "Run the campaign from the repository root:", "", "```bash",
        "MPLCONFIGDIR=/tmp/rr-capacity-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
        "  .venv/bin/python -B -m forgetting_random_replacement.oblivious.code.capacity_sweep --stage all",
        "```", "", "To rebuild only these figures and English notes from the validated cases:", "", "```bash",
        "MPLCONFIGDIR=/tmp/rr-capacity-mpl OPENBLAS_NUM_THREADS=1 \\",
        "  .venv/bin/python -B -m forgetting_random_replacement.oblivious.code.capacity_report",
        "```", ""]
    (CAMPAIGN / "README.md").write_text("\n".join(readme))
    notes = ["# Integer-capacity campaign results", "",
        f"All **{len(rows)} distinct cases** passed the saved campaign audit. "
        "The primary horizontal variable is the actual integer memory capacity C, "
        "covering every value from 1 through 30 without duplicate rho mappings.", "",
        scope_note, "",
        "## Experiment 8", "",
        "![Product 2 among own sales](figures/01_product2_own_sales_by_capacity.png)", "",
        "![Customer product 2 per calendar period](figures/02_customer_product2_by_capacity.png)", "",
        "The figures compare Monopoly and two-seller Marginal OE at outside options "
        f"`{outside_options_text}`; Conditional OE remains in the numerical tables. Every marker is a separately "
        "certified integer-capacity model. Connecting lines guide the eye.", "",
        "### Why the curves are jagged", "",
        "Each capacity C is a different finite-memory model, with exact states "
        "`S=0,...,C` and posterior `Beta(1+S,1+C-S)`. Changing C changes the "
        "state space and replacement probabilities. The product-2 decision is usually "
        "pure: an individual count state can enter or leave the product-2 region. "
        "The reported share is the sum of product-2 probabilities weighted by the "
        "stationary distribution at own sales; both the policy and those weights change.", "",
        "For monopoly at `p0=.5`, the saved policies give:", "",
        "| Capacity C | Counts choosing product 2 | Product 2 among own sales |",
        "|---:|---|---:|"]
    for c in (5, 6, 7, 8, 9):
        row = lookup[c, .5, 1, "monopoly"]
        with np.load(CAMPAIGN/"data/cases"/(row["case"]+".npz"), allow_pickle=False) as saved:
            active = np.flatnonzero(saved["policy"] > .5)
        counts = f"S = {active.min()}–{active.max()}" if len(active) else "None"
        notes.append(f"| {c} | {counts} | {100*row['product2_conditional']:.4f}% |")
    notes += ["", "C5 and C6 use product 2 at the same counts, S0–S3, but their "
        "sale-time weights differ. At C7, count S4 enters the product-2 region. "
        "This produces a fall and rebound without any interpolation approximation. "
        "These monopoly cases also pass an independent count-MDP comparison; their "
        "Bellman residuals are of order 1e-15, far smaller than the changes shown. "
        "No smoothness, monotonicity, or concavity in C is imposed or proved.", "",
        "For the baseline outside option `p0=.5`, selected own-sales product-2 shares are:", "",
        "| Capacity C | Effective retention | Monopoly | Marginal OE | Conditional OE |",
        "|---:|---:|---:|---:|---:|"]
    for c in sorted(set(P0_PANEL_CAPACITIES) | {15, 20, 25}):
        values = [lookup[c, .5, n, closure]["product2_conditional"] for n, closure, *_ in MODELS]
        notes.append(f"| {c} | {1-1/c:.6f} | " + " | ".join(f"{100*v:.4f}%" for v in values) + " |")
    if manifest.get("policy_figures"):
        notes += ["", "## Optimal product policies", "",
            "![Monopoly policies](figures/10_monopoly_policy_by_capacity.png)", "",
            "![Marginal OE policies](figures/11_oblivious_policy_by_capacity.png)", "",
            f"Each map shows all C1 through C30 and all {len(P0S)} outside options. Color is "
            "the probability of choosing product 2 conditional on being selected, "
            "at the seller's own retained success count S. Zero and one are pure "
            "product choices; intermediate colors represent mixed choices. Each cell "
            "represents an exact count state, with edges halfway between neighboring "
            "S/C values. This is a policy map, not a stationary frequency plot.", "",
            *( ["Success-count versions: [Monopoly](figures/policies/10_monopoly_policy_by_capacity_success_counts.pdf) "
                "and [Marginal OE](figures/policies/11_oblivious_policy_by_capacity_success_counts.pdf). "
                "These maps use integer successes S horizontally. Each unit cell is one state; "
                "gray S > C is infeasible. Both views display the same saved action probabilities.", ""]
               if has_count_views else [] ),
            "[Side-by-side policies at p0=.5](figures/policies/01_policy_comparison_p0_0p5.pdf)", "",
            f"The comparison panels use {panel_capacities_text}. Every marker is "
            "an exact count state; connecting lines guide the eye. Both models are "
            "shown together, with Conditional OE excluded. Comparisons at each "
            "outside option:", ""]
        for p0 in P0S:
            suffix = f"p0_{str(p0).replace('.', 'p')}"
            count_link = (f" | [success count S](figures/policies/01_policy_comparison_{suffix}_success_counts.pdf)"
                          if has_count_views else "")
            notes.append(f"- p0 = {p0:g}: [fraction S/C](figures/policies/01_policy_comparison_{suffix}.pdf){count_link}.")
        notes += ["", "[Adjacent-capacity policies at p0=.5]"
            "(figures/policies/02_adjacent_capacities_C5_C6_C7_p0_0p5.pdf)", "",
            *( ["[Adjacent-capacity policies using success counts]"
                "(figures/policies/02_adjacent_capacities_C5_C6_C7_p0_0p5_success_counts.pdf)", "",
                "Success-count panels use separate integer axes S = 0,...,C, with failures F = C-S. "
                "Fraction panels retain the shared S/C scale. The ten-capacity comparisons use "
                "five rows and two columns in both representations.", ""] if has_count_views else [] ),
            "In monopoly, C5 and C6 choose product 2 at S0–S3; C7 also chooses it "
            "at S4. The stationary weights still differ between C5 and C6 even "
            f"though these sets of counts coincide. All {policy_count:,} policies and {policy_state_count:,} "
            "count-state actions are exported in [policy figure data]"
            "(data/policy_figure_data.csv), with checkpoint hashes in the "
            "[policy figure manifest](data/policy_figure_manifest.json).", ""]
    notes += ["", "## Outside-option comparisons", "",
        "![Product 2 among own sales across p0](figures/06_product2_given_sale_by_p0.png)", "",
        "![Customer receives product 2 across p0](figures/07_customer_product2_by_p0.png)", "",
        f"The {len(P0_PANEL_CAPACITIES)} panels use {panel_capacities_text}. Own-sales product mix and customer "
        "calendar receipt have different denominators. The latter sums both sellers "
        "in competition and counts outside-option selection as zero receipt.", "",
        "## Seller-count comparisons", "",
        "![Product 2 by capacity and seller count](figures/09_product2_by_capacity_and_seller_count.png)", "",
        "The capacity-sweep seller extension uses marginal OE for N2 through N5 and the monopoly "
        f"solution for N1, at both low outside options. Every capacity in `{extension_range}` is "
        "solved for all five seller counts. The following table reports own-sales "
        "product-2 shares at the capacities highlighted in the seller-count figure:", "",
        "| Capacity | Outside option | 1 seller | 2 sellers | 3 sellers | 4 sellers | 5 sellers |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for c in SELLER_PANEL_CAPACITIES:
        for p0 in EXTENSION_P0S:
            values = [lookup[c, p0, n, "monopoly" if n == 1 else "marginal"]
                      ["product2_conditional"] for n in range(1, 6)]
            notes.append(f"| {c} | {p0:g} | " + " | ".join(f"{100*v:.4f}%" for v in values) + " |")
    notes += ["", "Calendar sales, profit, and outside-option choice can differ even "
        "when own-sales product mix is equal. The complete correlated joint law "
        "and each model's actual discounted payoff remain available in its NPZ checkpoint.", "",
        "## Validation", "",
        f"The audit accepted {validation.get('passed_case_count', len(validation['cases']))}/"
        f"{validation.get('expected_case_count', len(rows))} cases. The maximum "
        f"additional figure-accounting residual is {manifest['maximum_accounting_residual']:.3e}.", "",
        "| Reconstructed check | Worst error |", "|---|---:|"]
    for name, error in validation.get("worst_errors", {}).items():
        notes.append(f"| {name.replace('_', ' ')} | {float(error):.3e} |")
    notes += ["", "[Full validation](data/validation.json), [all summary rows](data/summary.csv), "
        "[plotted values](data/figure_data.csv), and [publication hashes](figures/manifest.json). "
        "See [README.md](README.md) for definitions, initialization, and reproduction.", ""]
    (CAMPAIGN / "RESULTS.md").write_text("\n".join(notes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots", nargs="+", choices=PUBLISHED_PLOT_IDS,
                        help="Rebuild only these figure numbers and update the shared notes and manifest")
    args = parser.parse_args()
    with (CAMPAIGN / "data/summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    publish(rows, plot_ids=args.plots)


if __name__ == "__main__":
    main()
