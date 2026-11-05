"""Publish exact Monopoly and Marginal OE policies for the integer-C campaign.

All actions come from audited checkpoints. Heatmap cells correspond to exact
integer count states; no intermediate policy states are solved or interpolated.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.collections import PolyCollection
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import PercentFormatter
import numpy as np

from .capacity_sweep import CAMPAIGN, CONFIG
from .run_experiment import now, write_csv, write_json

MODELS = ((1, "monopoly", "Monopoly", "#0072B2", "-", "o", "10_monopoly_policy_by_capacity"),
          (2, "marginal", "Marginal OE", "#D55E00", "--", "s", "11_oblivious_policy_by_capacity"))


def publish(rows, validation=None, *, figure_ids=("10", "11"), axis_views=("fraction", "count")):
    from .capacity_report import validated_inputs, digest

    if not axis_views or not set(axis_views) <= {"fraction", "count"}:
        raise ValueError("Policy axes must be fraction, count, or both")
    rows, validation = validated_inputs(rows, validation)
    lookup = {(row["capacity"], row["p0"], row["sellers"], row["closure"]): row for row in rows}
    p0s, capacities = CONFIG["outside_options"], CONFIG["capacities"]
    panels = CONFIG["outside_option_panel_capacities"]
    policies, evidence, exported = {}, {}, []
    for capacity in capacities:
        for p0 in p0s:
            for sellers, closure, label, *_ in MODELS:
                row = lookup[capacity, p0, sellers, closure]
                target = CAMPAIGN/"data/cases"/(row["case"]+".npz")
                with np.load(target, allow_pickle=False) as saved:
                    policy = saved["policy"].copy()
                    advantage = saved["advantage"].copy()
                    grid = saved["grid"].copy()
                if (policy.shape != (capacity+1,) or advantage.shape != policy.shape
                        or np.any(~np.isfinite(advantage)) or np.any(~np.isfinite(policy))
                        or np.any((policy < 0)|(policy > 1))
                        or not np.array_equal(grid, np.arange(capacity+1)/capacity)):
                    raise ArithmeticError(f"Invalid exact-state policy: {row['case']}")
                policies[capacity, p0, closure] = policy
                evidence[row["case"]] = digest(target)
                for s, probability in enumerate(policy):
                    exported.append(dict(case=row["case"], model=label, sellers=sellers, closure=closure,
                        capacity=capacity, p0=p0, S=s, F=capacity-s,
                        retained_success_fraction=s/capacity, posterior_mean=(s+1)/(capacity+2),
                        product2_probability=float(probability), advantage=float(advantage[s])))
    paths = []

    def save(fig, relative):
        relative = Path(relative)
        target = CAMPAIGN/"figures"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        suffixes = (".pdf",) if relative.parent == Path("policies") else (".png", ".pdf")
        for suffix in suffixes:
            path = target.with_suffix(suffix)
            fig.savefig(path, dpi=180, facecolor="white", bbox_inches="tight")
            paths.append(path)
        plt.close(fig)

    def comparison(p0, selected_capacities, name, *, count_axis=False):
        columns = 2 if len(selected_capacities) > 6 else min(3, len(selected_capacities))
        panel_rows = (len(selected_capacities)+columns-1)//columns
        fig, axes = plt.subplots(panel_rows, columns,
            figsize=(4.2*columns, 3.6*panel_rows+1.3),
            sharex=not count_axis, sharey=True, squeeze=False, layout="constrained")
        for ax, capacity in zip(axes.flat, selected_capacities):
            for sellers, closure, label, color, linestyle, marker, _ in MODELS:
                states = np.arange(capacity+1)
                ax.plot(states if count_axis else states/capacity, policies[capacity, p0, closure],
                    color=color, linestyle=linestyle, linewidth=1.5, marker=marker,
                    markersize=5.5 if sellers == 1 else 4.2, markerfacecolor="none",
                    label=f"{label} (N = {sellers})")
            ax.set_title(f"C = {capacity}; effective ρ = {1-1/capacity:.4g}")
            ax.set_xlabel("Remembered successes S (integer count)" if count_axis
                          else "Remembered success fraction S/C")
            if count_axis:
                ax.set_xlim(-.3, capacity+.3)
                ax.set_xticks(np.arange(capacity+1) if capacity <= 10
                              else sorted({0, capacity, *range(5, capacity, 5)}))
            else:
                ax.set_xlim(-.02, 1.02)
            ax.set_ylim(-.04, 1.04)
            ax.set_yticks((0, .25, .5, .75, 1))
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.2)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in list(axes.flat)[len(selected_capacities):]:
            ax.set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("Probability of product 2 when selected")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, frameon=False)
        coordinate_note = "Success counts S = 0, …, C; failures F = C − S\n" if count_axis else ""
        fig.suptitle(f"Random replacement: Monopoly and Marginal OE policies | p₀ = {p0:g}\n"
                     + coordinate_note +
                     "Markers are exact count states; connecting lines guide the eye", fontsize=12)
        save(fig, name)

    with plt.rc_context({"font.size": 10, "axes.titlesize": 11,
                         "figure.titlesize": 13, "pdf.fonttype": 42}):
        norm, cmap = Normalize(0, 1), plt.get_cmap("viridis")
        for axis_view in axis_views:
            count_axis = axis_view == "count"
            for sellers, closure, label, _, _, _, name in MODELS:
                if name[:2] not in figure_ids:
                    continue
                p0_columns = min(3, len(p0s))
                p0_rows = (len(p0s)+p0_columns-1)//p0_columns
                fig, axes = plt.subplots(p0_rows, p0_columns,
                    figsize=(14, 4*p0_rows+.5), sharex=True, sharey=True,
                    squeeze=False, layout="constrained")
                for ax, p0 in zip(axes.flat, p0s):
                    rectangles, values = [], []
                    for capacity in capacities:
                        nodes = np.arange(capacity+1)/capacity
                        edges = (np.arange(capacity+2)-.5 if count_axis
                                 else np.r_[0., (nodes[:-1]+nodes[1:])/2, 1.])
                        for s, probability in enumerate(policies[capacity, p0, closure]):
                            left, right = edges[s:s+2]
                            rectangles.append([(left, capacity-.5), (right, capacity-.5),
                                               (right, capacity+.5), (left, capacity+.5)])
                            values.append(probability)
                    collection = PolyCollection(rectangles, array=np.asarray(values), cmap=cmap,
                        norm=norm, edgecolors="white", linewidths=.15)
                    ax.add_collection(collection)
                    ax.set_title(f"p₀ = {p0:g}")
                    ax.set_xlim((-.5, max(capacities)+.5) if count_axis else (0, 1))
                    ax.set_ylim(min(capacities)-.5, max(capacities)+.5)
                    ax.set_xticks((0, 5, 10, 15, 20, 25, 30) if count_axis
                                  else (0, .25, .5, .75, 1))
                    ax.set_yticks((1, 5, 10, 15, 20, 25, 30))
                    ax.set_xlabel("Remembered successes S (integer count)" if count_axis
                                  else "Remembered success fraction S/C")
                    ax.set_ylabel("Memory capacity C")
                    if count_axis:
                        ax.set_facecolor("#e8e8e8")
                    ax.spines[["top", "right"]].set_visible(False)
                for ax in list(axes.flat)[len(p0s):]:
                    ax.set_visible(False)
                bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=list(axes.flat),
                                   ticks=(0, .25, .5, .75, 1), fraction=.025, pad=.02)
                bar.ax.yaxis.set_major_formatter(PercentFormatter(1))
                bar.set_label("Probability of choosing product 2 when selected")
                note = ("one cell per integer state | gray: infeasible S > C" if count_axis
                        else "each colored cell represents one exact success-count state")
                fig.suptitle(f"Random replacement: {label} policy (N = {sellers})\n"
                             f"C = 1, …, 30 | {note}")
                save(fig, Path("policies")/(name+"_success_counts") if count_axis else name)
        if {"10", "11"} <= set(figure_ids):
            for axis_view in axis_views:
                count_axis = axis_view == "count"
                suffix = "_success_counts" if count_axis else ""
                for p0 in p0s:
                    comparison(p0, panels,
                        Path("policies")/f"01_policy_comparison_p0_{str(p0).replace('.', 'p')}{suffix}",
                        count_axis=count_axis)
                comparison(.5, (5, 6, 7),
                    Path("policies")/("02_adjacent_capacities_C5_C6_C7_p0_0p5"+suffix),
                    count_axis=count_axis)
    table = CAMPAIGN/"data/policy_figure_data.csv"
    write_csv(table, exported)
    manifest_path = CAMPAIGN/"data/policy_figure_manifest.json"
    entries = {}
    if manifest_path.exists():
        entries.update({entry["path"]: entry for entry in json.loads(manifest_path.read_text())["figures"]})
    entries.update({str(path.relative_to(CAMPAIGN)):dict(path=str(path.relative_to(CAMPAIGN)),
                    sha256=digest(path)) for path in paths})
    manifest = dict(created_utc=now(), language="English", displayed_models=[model[2] for model in MODELS],
        case_count=len(evidence), state_count=len(exported), capacities=capacities, outside_options=p0s,
        selected_comparison_capacities=panels, conditional_OE_displayed=False,
        selected_comparison_effective_rho=[1-1/c for c in panels],
        policy_definition="alpha_C(S) = Pr(product 2 | selected, own retained success count S)",
        axis_views=[view for view, token in (("fraction", False), ("count", True))
                    if any(("_success_counts" in entry["path"]) == token for entry in entries.values())],
        state_geometry=dict(
            fraction="Exact S at its C, with edges midway between S/C nodes. No policy interpolation.",
            count="Unit cells centered at integer (S,C), 0<=S<=C. Gray S>C is infeasible, not zero action probability."),
        mixed_state_count=sum(1 for row in exported if 1e-8 < row["product2_probability"] < 1-1e-8),
        source_checkpoints=evidence, validation_sha256=digest(CAMPAIGN/"data/validation.json"),
        reporting_source_sha256=digest(__file__),
        policy_data_path=str(table.relative_to(CAMPAIGN)), policy_data_sha256=digest(table),
        figures=list(entries.values()))
    write_json(manifest_path, manifest)
    source_path = CAMPAIGN/"data/source_manifest.json"
    sources = json.loads(source_path.read_text())
    for name in ("capacity_sweep.py", "capacity_report.py", "capacity_policy_report.py"):
        sources["campaign_sources"][name] = digest(Path(__file__).with_name(name))
    sources["campaign_entrypoint_sha256"] = sources["campaign_sources"]["capacity_sweep.py"]
    sources["updated_utc"] = now()
    write_json(source_path, sources)
    config_path = CAMPAIGN/"config.json"
    configuration = json.loads(config_path.read_text())
    configuration["outside_option_panel_capacities"] = list(panels)
    write_json(config_path, configuration)
    return paths


def main():
    from .capacity_report import validated_inputs, digest, write_notes

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axes", nargs="+", choices=("fraction", "count"), default=("fraction", "count"),
                        help="Rebuild these horizontal-axis representations; preserve other saved figures")
    args = parser.parse_args()
    with (CAMPAIGN/"data/summary.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows, validation = validated_inputs(rows)
    paths = publish(rows, validation, axis_views=args.axes)
    target = CAMPAIGN/"figures/manifest.json"
    manifest = json.loads(target.read_text())
    entries = {entry["path"]: entry for entry in manifest["figures"]}
    entries.update({str(path.relative_to(CAMPAIGN)):dict(path=str(path.relative_to(CAMPAIGN)),
                    sha256=digest(path)) for path in paths})
    manifest["figures"] = list(entries.values())
    manifest["policy_figures"] = dict(path="data/policy_figure_manifest.json",
                                       sha256=digest(CAMPAIGN/"data/policy_figure_manifest.json"))
    manifest["created_utc"] = now()
    manifest["reporting_source_sha256"] = digest(Path(__file__).with_name("capacity_report.py"))
    write_json(target, manifest)
    write_notes(rows, validation, manifest)
    policy_count = len(CONFIG["capacities"])*len(CONFIG["outside_options"])*len(MODELS)
    print(f"Published {len(paths)} policy figure files from {policy_count} audited policies.", flush=True)


if __name__ == "__main__":
    main()
