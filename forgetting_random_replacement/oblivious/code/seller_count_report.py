"""Publish the audited finite-N extension and distinguish observations from bounds.

Uncomputed or rejected cases remain gaps. The displayed metric is the share
of a seller's own sales using product 2, not its calendar flow per period.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from .run_experiment import ROOT, case_key, now, numerical_sources, write_csv, write_json
from .seller_count_sweep import CAMPAIGN, CONFIG, collect, expected_cases, extension_sources


P0_STYLES = ((.1, "#0072B2", "o"), (.3, "#D55E00", "s"),
             (.5, "#009E73", "^"), (.7, "#CC79A7", "D"),
             (.8, "#E69F00", "v"), (.9, "#56B4E9", "P"))
FIGURE_NAME = "08_product2_by_seller_count"


def digest(path):
    checksum = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def _typed_rows(rows):
    converted = []
    for raw in rows:
        row = dict(raw)
        for key in ("capacity", "sellers", "count_states", "joint_states"):
            if key in row:
                row[key] = int(row[key])
        for key in ("p0", "product2_conditional", "product2_calendar", "sale_rate",
                    "outside_rate", "profit_calendar", "value_initial"):
            if key in row:
                row[key] = float(row[key])
        if isinstance(row.get("verified"), str):
            row["verified"] = row["verified"].lower() == "true"
        converted.append(row)
    return converted


def validated_inputs(rows, validation=None):
    """Bind every available plotted case to the saved numerical audit."""
    if validation is None:
        validation = json.loads((CAMPAIGN/"data/validation.json").read_text())
    if validation.get("verified") is not True:
        raise ArithmeticError("Seller-count publication requires a passing audit of the available cases")
    if validation.get("sources") != numerical_sources():
        raise ArithmeticError("Seller-count numerical sources changed after audit")
    current_extra = extension_sources()
    if validation.get("extension_sources", current_extra) != current_extra:
        raise ArithmeticError("Generic seller-count sources changed after audit")
    rows = _typed_rows(rows)
    expected = {case_key(*case) for case in expected_cases()}
    audited = {case["case"]: case for case in validation["cases"]}
    names = [row["case"] for row in rows]
    if len(names) != len(set(names)) or set(names) != set(audited) or not set(names) <= expected:
        raise ArithmeticError("Seller-count summary and audited case sets differ")
    for row in rows:
        if row.get("verified") is not True:
            raise ArithmeticError(f"Unverified seller-count case cannot be plotted: {row['case']}")
        target = CAMPAIGN/"data/cases"/row["case"]
        for suffix, key in ((".npz", "checkpoint_sha256"), (".json", "record_sha256")):
            if digest(target.with_suffix(suffix)) != audited[row["case"]][key]:
                raise ArithmeticError(f"Seller-count checkpoint changed after audit: {row['case']}")
        record = json.loads(target.with_suffix(".json").read_text())
        if record.get("sources") != numerical_sources():
            raise ArithmeticError(f"Seller-count case uses different numerical sources: {row['case']}")
        for filename, checksum in record.get("extension_sources", {}).items():
            path = Path(filename)
            path = path if path.is_absolute() else ROOT/path
            if digest(path) != checksum:
                raise ArithmeticError(f"Seller-count dependency changed after audit: {filename}")
        for key in ("capacity", "p0", "sellers", "closure", "product2_conditional",
                    "product2_calendar", "sale_rate", "outside_rate"):
            if row[key] != record["summary"][key]:
                raise ArithmeticError(f"Seller-count summary differs from saved case: {row['case']}, {key}")
        own = row["product2_conditional"]
        if not np.isfinite(own) or not -1e-12 <= own <= 1+1e-12:
            raise ArithmeticError("Own-sales product mix is outside its probability range")
        if (row["sale_rate"] <= 0 or
                abs(row["product2_calendar"]-row["sale_rate"]*own) > 1e-10 or
                abs(row["outside_rate"]+row["sellers"]*row["sale_rate"]-1) > 1e-10):
            raise ArithmeticError(f"Seller-count flow accounting failed: {row['case']}")
    return rows, validation


def near_zero_summary(rows):
    """Summarize finite verified values without converting missing points to zero."""
    lookup = {(row["capacity"], row["p0"], row["sellers"]): row for row in rows}
    threshold = float(CONFIG.get("near_zero_threshold", .01))
    summaries = []
    scenarios = [(capacity, p0, CONFIG["seller_counts_by_capacity"][str(capacity)], False)
                 for capacity in CONFIG["capacities"] for p0 in CONFIG["outside_options"]]
    example = CONFIG.get("illustrative_example")
    if example:
        scenarios.append((example["capacity"], example["p0"], example["seller_counts"], True))
    for capacity, p0, counts, illustrative in scenarios:
            values = [(n, lookup[capacity, p0, n]["product2_conditional"])
                      for n in counts if (capacity, p0, n) in lookup]
            positive = [(n, value) for n, value in values if value > 0]
            first_after_positive = None
            first_after_above_band = None
            seen_positive = seen_above_band = False
            for n, value in values:
                if value <= threshold:
                    if seen_positive and first_after_positive is None:
                        first_after_positive = n
                    if seen_above_band and first_after_above_band is None:
                        first_after_above_band = n
                seen_positive |= value > 0
                seen_above_band |= value > threshold
            suffix = []
            for n in reversed(counts):
                row = lookup.get((capacity, p0, n))
                if row is None or row["product2_conditional"] > threshold:
                    break
                suffix.append(n)
            last_n, last_value = values[-1] if values else (None, None)
            summaries.append(dict(
                capacity=capacity, p0=p0, illustrative_example=illustrative, requested_max_N=max(counts),
                verified_points=len(values), requested_points=len(counts),
                last_verified_N=last_n, last_verified_product2_conditional=last_value,
                last_verified_product2_percent=None if last_value is None else 100*last_value,
                minimum_positive_share=None if not positive else min(value for _, value in positive),
                minimum_positive_percent=None if not positive else 100*min(value for _, value in positive),
                all_verified_shares_exactly_zero=bool(values) and all(value == 0 for _, value in values),
                first_verified_near_zero_N_after_positive=first_after_positive,
                first_verified_near_zero_N_after_above_threshold=first_after_above_band,
                terminal_near_zero_run_starts_N=min(suffix) if suffix else None,
                near_zero_through_requested_max=bool(suffix), near_zero_threshold=threshold,
                c1_zero_for_every_N_proven_for_marginal_OE=capacity == 1))
    return summaries


def _illustrative_rows(rows):
    example = CONFIG.get("illustrative_example")
    if not example:
        return []
    lookup = {(row["capacity"], row["p0"], row["sellers"]): row for row in rows}
    selected = []
    for n in example["seller_counts"]:
        key = example["capacity"], example["p0"], n
        if key not in lookup:
            raise ArithmeticError("The illustrative decline requires every selected finite-N case to be audited")
        selected.append(lookup[key])
    return selected


def _publish_illustrative(rows, threshold):
    """Keep the chosen example and its actual calendar demand in separate panels."""
    if not rows:
        return []
    example = CONFIG["illustrative_example"]
    counts = [row["sellers"] for row in rows]
    shares = [row["product2_conditional"] for row in rows]
    demand = [row["sale_rate"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), layout="constrained", sharex=True)
    axes[0].axhspan(0, threshold, color="#707070", alpha=.17)
    axes[0].plot(counts, shares, color="#111111", marker="o", linewidth=2)
    axes[0].set_ylabel("Product 2 (% of own sales)")
    axes[0].set_title("Own-sales product mix")
    axes[0].set_ylim(0, max(.05, max(shares)*1.18))
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    label_indices = sorted({0, 2, 3, 4, 5, len(rows)-1})
    for index in label_indices:
        if index < len(rows):
            axes[0].annotate(f"{100*shares[index]:.2f}%", (counts[index], shares[index]),
                             xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    axes[1].plot(counts, demand, color="#0072B2", marker="s", linewidth=2)
    axes[1].set_title("Actual calendar demand per seller")
    axes[1].set_ylabel("Own sale (% of calendar periods)")
    axes[1].set_ylim(0, max(demand)*1.15)
    axes[1].yaxis.set_major_formatter(PercentFormatter(1, decimals=3))
    for ax in axes:
        ax.set_xticks(counts)
        ax.set_xlim(min(counts)-.3, max(counts)+.3)
        ax.set_xlabel("Number of sellers N")
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Illustrative decline | C = {example['capacity']}, p₀ = {example['p0']:g}\n"
                 "Audited marginal OE for N ≥ 2; monopoly at N = 1\n"
                 "Parameters selected by exploratory search near the investment cutoff", fontsize=12)
    paths = []
    for suffix in (".png", ".pdf"):
        path = CAMPAIGN/"figures"/("08b_illustrative_decline_and_demand"+suffix)
        fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    write_csv(CAMPAIGN/"data/illustrative_decline.csv", rows)
    return paths


def _theory_text():
    return """# Eventual product-1 dominance: a conservative theoretical bound

The numerical plots display only audited finite seller counts. The following
argument is separate from the numerical experiment. It proves eventual zero
product-2 use for a fixed finite memory capacity, without predicting where
the finite-N equilibrium curve first approaches zero.

## A uniform selection bound

Set `m=C+1`. The most favorable own posterior is `Beta(m,1)` and the least
favorable rival posterior is `Beta(1,m)`. First-order stochastic dominance
therefore gives, for any own count and any joint rival state,

```text
d <= integral[p0,1] m*t^(m-1) * [1-(1-t)^m]^(N-1) dt
  <= B(1/m,N).
```

For the second inequality, substitute `u=1-t`, bound `(1-u)^(m-1)` by one,
extend the integration interval to `[0,1]`, and substitute `z=u^m`. The factor
m cancels the Jacobian factor 1/m. Rival correlations cannot violate this
pointwise bound; averaging over any rival belief preserves it.

Every calendar-period reward is nonnegative and at most `(revenue-c1)*d`.
Hence every continuation value is bounded by

```text
Vmax <= (revenue-c1) * B(1/m,N) / (1-gamma).
```

The seller chooses a product after selection. Its conditional advantage from
product 2 is at most `-(c2-c1) + gamma*(p2-p1)*Vmax`. There is no extra current
selection factor in that conditional comparison. Thus product 1 strictly
dominates at every own state whenever

```text
B(1/m,N) < (c2-c1)*(1-gamma) / [gamma*(p2-p1)*(revenue-c1)]
         = 80/2793
         = approximately 0.02864303616.
```

Since the beta-function bound tends to zero as N increases, eventual
product-1 dominance follows for each fixed finite C. This argument applies
to marginal and conditional beliefs, and does not require independent rival
reputations.

## Safe sufficient seller counts

For `N>1`, the inequality `(1-z)^(N-1) <= exp(-(N-1)*z)` implies

```text
B(1/m,N) <= Gamma(1/m)/(N-1)^(1/m) <= m/(N-1)^(1/m).
```

The last step uses `Gamma(1+1/m)<=1`. Consequently
`N > 1 + (2793*m/80)^m` is a transparent sufficient condition.

| Capacity C | A safe sufficient N from this generic bound |
|---:|---:|
| 1 | 5,000 |
| 3 | 400,000,000 |
| 5 | 90,000,000,000,000 |
| 8 | 30,000,000,000,000,000,000,000 |

These deliberately conservative numbers are **not observed crossing points,
estimates of a minimal threshold, or forecasts of the plotted curves**. They
show why a moderate finite-N experiment need not reach the guaranteed regime.

## A stronger result for C=1 and the plotted marginal closure

For C=1, a purchase entirely replaces the single observation. Let d0 and d1
be selection probabilities at own counts zero and one against the same
unconditional rival law. If T is the maximum of the outside option and the
rival samples, then

```text
d0 = E[(1-T)^2]
d1 = E[1-T^2] <= 2*sqrt(d0)-d0.
```

The inequality is Jensen's inequality. Evaluating the always-product-1 policy
gives the continuation-value difference

```text
V1-V0 = .95*(d1-d0) / [.02 + .98*(.35*d0+.65*d1)].
```

Its product-2 advantage is nonpositive if
`.98*(.0375*d1-.6375*d0) <= .012`. Writing `x=sqrt(d0)` bounds its left side by

```text
.98*(.075*x-.675*x^2) <= .98/480 < .012.
```

The inequality is strict, so product 1 is the unique best response to every
common marginal rival law under this calibration. Therefore every C=1
marginal oblivious equilibrium, including monopoly, has **zero product-2 use
for every N**. This stronger C=1 argument does not automatically apply to
conditional beliefs, whose rival law may change with the own count.

## What the graph can establish

Product 2 among own sales is `product2_calendar/sale_rate`. A symmetric
seller's calendar product-2 flow is at most `1/N`, which declines mechanically
with N. That observation does not prove its conditional product mix declines.
Finite audited points can show a decline, plateau, or entry into the 1% band;
they do not by themselves establish an asymptotic limit or monotonicity.
"""


def publish(rows=None, validation=None):
    if rows is None:
        rows = collect()
    rows, validation = validated_inputs(rows, validation)
    lookup = {(row["capacity"], row["p0"], row["sellers"]): row for row in rows}
    illustrative = _illustrative_rows(rows)
    example = CONFIG.get("illustrative_example")
    threshold = float(CONFIG.get("near_zero_threshold", .01))
    summaries = near_zero_summary(rows)
    figure_rows = []
    with plt.rc_context({"font.size": 10, "axes.titlesize": 11,
                         "figure.titlesize": 13, "pdf.fonttype": 42}):
        fig, axes = plt.subplots(2, 2, figsize=(13.5, 9), layout="constrained", sharey=True)
        for ax, capacity in zip(axes.flat, CONFIG["capacities"], strict=True):
            counts = CONFIG["seller_counts_by_capacity"][str(capacity)]
            panel_cases = [row for row in rows if row["capacity"] == capacity]
            ax.axhspan(0, threshold, color="#707070", alpha=.17, zorder=0)
            ax.axhline(threshold, color="#777777", linewidth=.7, linestyle=":", zorder=0)
            for p0, color, marker in P0_STYLES:
                if p0 not in CONFIG["outside_options"]:
                    continue
                values = []
                for n in counts:
                    row = lookup.get((capacity, p0, n))
                    value = np.nan if row is None else row["product2_conditional"]
                    values.append(value)
                    figure_rows.append(dict(capacity=capacity, p0=p0, sellers=n,
                        verified=row is not None, product2_conditional=None if row is None else value,
                        product2_percent=None if row is None else 100*value,
                        sample_type="Pre-specified broad grid",
                        case=case_key(capacity, p0, n, "monopoly" if n == 1 else "marginal"),
                        status="Audited finite-N result" if row is not None else "Unverified or uncomputed; gap"))
                ax.plot(counts, values, color=color, marker=marker, markersize=3.2,
                        linewidth=1.5, label=f"p₀ = {p0:g}")
            requested = len(counts)*len(CONFIG["outside_options"])
            if illustrative and capacity == example["capacity"]:
                selected_counts = [row["sellers"] for row in illustrative]
                selected_values = [row["product2_conditional"] for row in illustrative]
                requested += len(illustrative)
                ax.plot(selected_counts, selected_values, color="#111111", marker="X",
                        markersize=5, linewidth=2.2, zorder=10,
                        label=f"p₀ = {example['p0']:g} (C = {capacity}, illustrative)")
                zeros = [row["sellers"] for row in illustrative if row["product2_conditional"] == 0]
                if zeros:
                    ax.annotate(f"Illustrative p₀ = {example['p0']:g}\n"
                                f"0% audited at N = {min(zeros)}–{max(zeros)}",
                        xy=(min(zeros), .005), xycoords="data", xytext=(.48, .16),
                        textcoords="axes fraction", fontsize=9,
                        arrowprops=dict(arrowstyle="->", color="#222222", linewidth=.8),
                        bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
                for row in illustrative:
                    figure_rows.append(dict(capacity=capacity, p0=row["p0"], sellers=row["sellers"],
                        verified=True, product2_conditional=row["product2_conditional"],
                        product2_percent=100*row["product2_conditional"], case=row["case"],
                        sample_type="Illustrative cutoff-focused example",
                        status="Audited finite-N result; selected by exploratory search"))
            ax.set_title(f"C = {capacity} | requested N = {min(counts)}–{max(counts)}\n"
                         f"Audited cases: {len(panel_cases)}/{requested}")
            ax.set_xlabel("Number of sellers N")
            ax.set_ylabel("Product 2 (% of own sales)")
            ax.set_xlim(min(counts)-.5, max(counts)+.5)
            if len(counts) <= 12:
                ticks = counts
            else:
                candidates = (1, 5, 10, 20, 30, 40, 50) if max(counts) <= 50 else (1, 20, 40, 60, 80, 100)
                ticks = sorted({min(counts), max(counts), *[n for n in candidates if min(counts) <= n <= max(counts)]})
            ax.set_xticks(ticks)
            ax.set_ylim(0, 1)
            ax.set_yticks(np.arange(0, 1.01, .2))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
            if capacity == 1:
                ax.text(.03, .93, "C = 1: product 1 is strictly optimal for every N\n"
                        "under the plotted marginal closure; see THEORY.md.",
                        transform=ax.transAxes, va="top", fontsize=9)
        unique_labels = {}
        for ax in axes.flat:
            handles, labels = ax.get_legend_handles_labels()
            unique_labels.update(zip(labels, handles))
        fig.legend(list(unique_labels.values()), list(unique_labels), loc="outside lower center",
                   ncol=3, frameon=False, fontsize=9.5)
        fig.suptitle("Random replacement: product 2 as the number of sellers increases\n"
                     "N = 1: monopoly; N ≥ 2: marginal OE | shaded band: ≤1% of own sales\n"
                     "Panel-specific N ranges; missing or rejected cases remain gaps", fontsize=12)
        directory = CAMPAIGN/"figures"
        directory.mkdir(parents=True, exist_ok=True)
        paths = []
        for suffix in (".png", ".pdf"):
            path = directory/(FIGURE_NAME+suffix)
            fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
            paths.append(path)
        plt.close(fig)
        paths.extend(_publish_illustrative(illustrative, threshold))
    write_csv(CAMPAIGN/"data/figure_data.csv", figure_rows)
    write_csv(CAMPAIGN/"data/near_zero_summary.csv", summaries)
    theory = CAMPAIGN/"THEORY.md"
    theory.write_text(_theory_text())
    manifest = dict(created_utc=now(), language="English", metric="product2_conditional",
        formula="Own product-2 calendar flow divided by own sale rate",
        validated_case_count=len(rows), requested_case_count=len(expected_cases()),
        coverage_complete=len(rows) == len(expected_cases()),
        requested_sellers_by_capacity=CONFIG["seller_counts_by_capacity"],
        outside_options=CONFIG["outside_options"], near_zero_threshold=threshold,
        illustrative_example=example,
        missing_value_policy="NaN gaps; no missing case is plotted as zero and no line crosses a missing grid point",
        theoretical_bounds="Sufficient product-1 dominance bounds, not observed crossing points",
        validation_path="data/validation.json", validation_sha256=digest(CAMPAIGN/"data/validation.json"),
        reporting_source_sha256=digest(__file__), theory_sha256=digest(theory),
        figures=[dict(path=str(path.relative_to(CAMPAIGN)), sha256=digest(path)) for path in paths],
        generated_data_sha256={str(path.relative_to(CAMPAIGN)): digest(path) for path in
            [CAMPAIGN/"data/figure_data.csv", CAMPAIGN/"data/near_zero_summary.csv",
             *([CAMPAIGN/"data/illustrative_decline.csv"] if illustrative else [])]})
    write_json(CAMPAIGN/"figures/manifest.json", manifest)
    write_notes(summaries, validation, manifest, illustrative)
    print(f"Published extended plot 08 from {len(rows)}/{len(expected_cases())} audited finite-N cases.", flush=True)
    return manifest


def write_notes(summaries, validation, manifest, illustrative=None):
    verified = manifest["validated_case_count"]
    requested = manifest["requested_case_count"]
    incomplete = requested-verified
    ranges = ", ".join(f"C={c}: N={min(ns)}–{max(ns)}" for c in CONFIG["capacities"]
                       for ns in [CONFIG["seller_counts_by_capacity"][str(c)]])
    coverage_note = (f"All {requested} requested cases are audited." if incomplete == 0 else
        f"The available-case audit passed for {verified}/{requested} requested cases. "
        f"The remaining {incomplete} cases are not published as results; see data/validation.json "
        "for missing or failed cases. A passing available-case audit is not complete requested coverage.")
    readme = ["# More sellers and outside options: extended plot 08", "",
        "This separate campaign extends the own-sales product-2 comparison while preserving "
        "the exact correlated rival distribution and the numerical acceptance criteria.", "",
        coverage_note, "", f"Requested finite ranges: {ranges}.", "",
        "Outside options: `.1, .3, .5, .7, .8, .9`. Monopoly is N=1; N≥2 uses marginal OE.", "",
        "- [Extended plot 08](figures/08_product2_by_seller_count.png) "
        "([PDF](figures/08_product2_by_seller_count.pdf)).",
        "- [Finite observations](RESULTS.md) and [near-zero summary](data/near_zero_summary.csv).",
        "- [A conservative eventual-zero proof](THEORY.md).",
        "- [Numerical audit and coverage](data/validation.json), "
        "[plotted grid including gaps](data/figure_data.csv), and [figure hashes](figures/manifest.json).", "",
        "The gray band denotes product 2 in at most 1% of own sales. Missing or rejected "
        "points remain gaps. N ranges differ between capacity panels because the exact "
        "quotient contains binomial(N+C,C) states.", "",
        "Markers identify the seller counts actually solved; connecting lines guide the eye. "
        "For C3 the grid is dense through N30, followed by the additional checks at N35, N40, and N50.", "",
        "This is not a per-seller calendar-flow graph. The latter is bounded by 1/N "
        "mechanically and cannot answer whether the product mix among own sales vanishes.", "",
        "## Reproduction", "", "From the repository root:", "", "```bash",
        "MPLCONFIGDIR=/tmp/rr-seller-count-mpl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\",
        "  .venv/bin/python -B -m forgetting_random_replacement.oblivious.code.seller_count_sweep --stage solve",
        "OPENBLAS_NUM_THREADS=1 .venv/bin/python -B -m forgetting_random_replacement.oblivious.code.seller_count_audit",
        "MPLCONFIGDIR=/tmp/rr-seller-count-mpl OPENBLAS_NUM_THREADS=1 \\",
        "  .venv/bin/python -B -m forgetting_random_replacement.oblivious.code.seller_count_report",
        "```", ""]
    if illustrative:
        example = CONFIG["illustrative_example"]
        insertion = ["## A selected positive-to-zero example", "",
            f"An exploratory search near the investment cutoff selected `C={example['capacity']}, "
            f"p0={example['p0']:g}`. All ten cases N1–10 are separately audited. This extra "
            "outside option is highlighted in black only in the C3 panel; it was not part "
            "of the pre-specified six-value outside-option grid.", "",
            "- [Own-sales decline and actual calendar demand](figures/08b_illustrative_decline_and_demand.png) "
            "([PDF](figures/08b_illustrative_decline_and_demand.pdf)).",
            "- [Selected-example data](data/illustrative_decline.csv).", "",
            "It illustrates a possible decline, not a result for every outside option, "
            "a controlled causal comparison, or a claim of equilibrium uniqueness.", ""]
        location = readme.index("## Reproduction")
        readme[location:location] = insertion
    (CAMPAIGN/"README.md").write_text("\n".join(readme))
    notes = ["# Extended seller-count observations", "", coverage_note, "",
        "![Product 2 among own sales by number of sellers](figures/08_product2_by_seller_count.png)", "",
        "The table reports the last audited value, rather than substituting zero for "
        "unavailable endpoints. A crossing is the first audited point at or below 1% "
        "after an earlier audited point above 1%; it does not imply all subsequent "
        "seller counts stay in that band. The complete CSV also records the beginning "
        "of any verified terminal run in the near-zero band.", "",
        "| C | p₀ | Requested max N | Last audited N | Product 2 at last N | Smallest positive share | First ≤1% after >1% |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summaries:
        last = "—" if row["last_verified_N"] is None else str(row["last_verified_N"])
        share = "—" if row["last_verified_product2_percent"] is None else f"{row['last_verified_product2_percent']:.4f}%"
        minimum = "—" if row["minimum_positive_percent"] is None else f"{row['minimum_positive_percent']:.4g}%"
        crossing = row["first_verified_near_zero_N_after_above_threshold"]
        notes.append(f"| {row['capacity']} | {row['p0']:g} | {row['requested_max_N']} | {last} | "
                     f"{share} | {minimum} | {'—' if crossing is None else crossing} |")
    if illustrative:
        example = CONFIG["illustrative_example"]
        notes += ["", "## An audited positive-to-zero example", "",
            "![Illustrative own-sales decline and actual demand](figures/08b_illustrative_decline_and_demand.png)", "",
            f"The additional `C={example['capacity']}, p0={example['p0']:g}` curve was selected "
            "after an exploratory search near the investment cutoff. It is an illustrative "
            "example, not a pre-specified broad-grid finding or a general claim across p0. "
            "It displays a verified equilibrium branch; uniqueness is not established.", "",
            "| Sellers N | Product 2 among own sales | Actual own sale per calendar period |",
            "|---:|---:|---:|"]
        notes += [f"| {row['sellers']} | {100*row['product2_conditional']:.4f}% | {100*row['sale_rate']:.5f}% |"
                  for row in illustrative]
        notes += ["", "Product-2 use is positive initially and is exactly zero in the saved "
            "policy for N6–10. Actual demand is `sale_rate`: the probability that one seller "
            "makes a sale in a calendar period, evaluated under the actual correlated joint "
            "law. It differs from own-sales product mix and from the perceived OE demand "
            "at a particular reputation. Both demand and policy change in this equilibrium "
            "comparison; the right panel is not a controlled causal estimate.", ""]
    notes += ["", "C=1 is separate: under the plotted marginal closure and this calibration, "
        "product 1 is strictly optimal for every seller count. Its zero line is not "
        "evidence of a decline from a positive share.", "",
        "For each fixed C, a uniform selection-probability bound proves eventual "
        "product-1 dominance for sufficiently large N. The generic sufficient bounds "
        "are very conservative: 5,000 for C1, 400 million for C3, 90 trillion for C5, "
        "and 3×10²² for C8. These are not predicted or observed crossing points. "
        "See [THEORY.md](THEORY.md) for the derivation and the stronger C1 result.", "",
        "Finite points do not establish monotonicity or identify an asymptotic limit. "
        "Own-sales product mix, per-seller calendar flow, and total market receipt are "
        "different quantities.", "", "## Numerical verification", "",
        f"Audit of available cases: {verified} accepted. Requested coverage: {verified}/{requested}.", ""]
    if validation.get("worst_errors"):
        notes += ["| Reconstructed check | Worst error |", "|---|---:|"]
        notes += [f"| {name.replace('_', ' ')} | {value:.3e} |"
                  for name, value in sorted(validation["worst_errors"].items())]
    notes += ["", "[All near-zero summaries](data/near_zero_summary.csv), "
        "[plotted values and gaps](data/figure_data.csv), "
        "[case audit](data/validation.json), and [reproduction](README.md).", ""]
    (CAMPAIGN/"RESULTS.md").write_text("\n".join(notes))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    publish()


if __name__ == "__main__":
    main()
