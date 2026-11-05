"""Render only the three requested figures from this folder's data snapshot."""

import argparse
from hashlib import sha256
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/full-comparison-mpl")
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "Figures" / "n051"
STYLE = {
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 9, "axes.linewidth": .8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def save(fig, name):
    for extension in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{extension}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def policy_figure(config, policies):
    rhos, models = config["rhos"], config["models"]
    rows = (len(rhos) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 3.6 * rows), squeeze=False)
    fig.subplots_adjust(left=.075, right=.99, top=.925, bottom=.05,
                        wspace=.18, hspace=.40)
    for ax, rho in zip(axes.flat, rhos):
        panel_policies = []
        for key, model in models.items():
            data = policies[(policies["rho"] == rho) & (policies["model"] == key)]
            data = data.sort_values("reputation")
            x, alpha = data["reputation"].to_numpy(), data["product2_probability"].to_numpy()
            panel_policies.append(alpha)
            ax.plot(x, alpha, color=model["color"], ls="-",
                    lw=1.7, marker=".", ms=3, label=model["label"], zorder=3)
            mixed = (alpha > 1e-9) & (alpha < 1 - 1e-9)
            ax.scatter(x[mixed], alpha[mixed], marker="D", s=28,
                       color=model["color"], edgecolors="white", linewidths=.4, zorder=5)
        if np.all(np.asarray(panel_policies) == 0):
            ax.text(.5, .5, "All policies choose product 1",
                    transform=ax.transAxes, ha="center", color=".35", fontsize=9)
        ax.set(title=rf"$\rho={rho:g}$, n={config['grid']} | verified on grid",
               xlabel="Own reputation", ylabel="Probability of product 2",
               xlim=(-.035, 1.035), ylim=(-.04, 1.04),
               xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
        ax.grid(alpha=.22)
    for ax in axes.flat[len(rhos):]:
        ax.set_visible(False)
    fig.suptitle("Optimal policies across models",
                 fontsize=13, y=.99)
    handles = [Line2D([], [], color=model["color"], ls="-", lw=1.7,
                      marker=".", ms=3, label=model["label"])
               for model in models.values()]
    handles.append(Line2D([], [], color=".35", marker="D", ls="none", ms=4.5,
                          label="Mixing at a grid point"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .971),
               ncol=len(handles), frameon=False, fontsize=9)
    save(fig, "01_optimal_policies_across_rho")


def outcome_figure(config, outcomes, metric, title, ylabel, name):
    rhos, models = config["rhos"], config["models"]
    fig, ax = plt.subplots(figsize=(10, 5.3))
    fig.subplots_adjust(left=.105, right=.98, top=.73, bottom=.27)
    offsets = np.linspace(-.055, .055, len(models))
    for (key, model), offset in zip(models.items(), offsets):
        data = outcomes[(outcomes["metric"] == metric) & (outcomes["model"] == key)]
        data = data.set_index("rho").loc[rhos]
        x = np.arange(len(rhos)) + offset
        mean = data["simulation_mean"].to_numpy()
        low, high = data["simulation_ci_low"].to_numpy(), data["simulation_ci_high"].to_numpy()
        ax.plot(x, data["stationary"], color=model["color"], ls="--",
                marker="D", mfc="white", ms=6, lw=1.3, zorder=3)
        ax.errorbar(x, mean, yerr=[mean - low, high - mean], color=model["color"],
                    marker="o", ms=4, lw=1.5, capsize=3.5, elinewidth=1.1, zorder=4)
    ax.set_xticks(range(len(rhos)), [f"{rho:g}" for rho in rhos])
    ax.set(xlim=(-.3, len(rhos) - .7), xlabel=r"Forgetting factor $\rho$ (discrete values)",
           ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=".9", lw=.7)
    if metric == "product2_market":
        ax.set_ylim(-.025, 1.025)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    else:
        ax.set_ylim(0, .5)
    fig.suptitle(title, y=.98, fontsize=14)
    fig.legend(handles=[
        Line2D([], [], color=model["color"], lw=2, label=model["label"])
        for model in models.values()
    ], loc="upper center", bbox_to_anchor=(.53, .92), ncol=len(models), frameon=False)
    fig.legend(handles=[
        Line2D([], [], color=".3", marker="o", ms=4, lw=1.4,
               label="Simulation (95% MC CI)"),
        Line2D([], [], color=".3", marker="D", mfc="white", ms=5, ls="--", lw=1.3,
               label="Stationary prediction"),
    ], loc="upper center", bbox_to_anchor=(.53, .85), ncol=2, frameon=False)
    sim = config["simulation"]
    fig.text(.53, .14,
             f"{sim['replications']} independent paths per model; {sim['periods']:,} retained periods; "
             f"{sim['burn_in']:,} burn-in; start (0.5, 0.5).",
             ha="center", fontsize=8.5)
    fig.text(.53, .095, f"Actual two-seller TS; common {config['grid']}-point grid. "
             "MC intervals exclude initialization bias.", ha="center", fontsize=8.5)
    note = ("One purchase per calendar period."
            if metric == "product2_market" else "Profit averages both sellers, including unselected periods.")
    fig.text(.53, .05, note + " Rho values are equally spaced; lines guide the eye.",
             ha="center", fontsize=8.2)
    save(fig, name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--policies-only", action="store_true",
                       help="Update only the policy figure; preserve the saved usage and profit figures")
    group.add_argument("--outcomes-only", action="store_true",
                       help="Update usage and profit figures; preserve the saved policy figure")
    args = parser.parse_args()
    config = json.loads((DATA / "comparison.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, digest in manifest["data_files_sha256"].items():
        if sha256((DATA / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Data snapshot changed: {name}; rerun prepare_data.py.")
    policies, outcomes = pd.read_csv(DATA / "policies.csv"), pd.read_csv(DATA / "outcomes.csv")
    policy_config = dict(config, models=dict(config["models"]))
    outcome_config = dict(config, models=dict(config["models"]))
    conditional_path = DATA / "conditional_policy_manifest.json"
    conditional_hash = None
    if conditional_path.exists():
        conditional = json.loads(conditional_path.read_text())
        if (conditional["status"] != "complete" or conditional["rhos"] != config["rhos"]
                or conditional["grid"] != config["grid"] or conditional["parameters"] != config["parameters"]
                or conditional["comparison_manifest_sha256"] != sha256((DATA / "manifest.json").read_bytes()).hexdigest()):
            raise ValueError("Conditional policy snapshot is incompatible; rerun prepare_conditional_policy.py.")
        for name, digest in conditional["data_files_sha256"].items():
            if sha256((DATA / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Conditional policy snapshot changed: {name}")
        model = dict(conditional["model"])
        policy_config["models"][model.pop("key")] = model
        policies = pd.concat([policies, pd.read_csv(DATA / "conditional_policies.csv")], ignore_index=True)
        conditional_hash = sha256(conditional_path.read_bytes()).hexdigest()
    outcome_path = DATA / "conditional_outcomes_manifest.json"
    outcome_hash = None
    if not args.policies_only and outcome_path.exists():
        extension = json.loads(outcome_path.read_text())
        if (extension["status"] != "complete" or extension["rhos"] != config["rhos"]
                or extension["grid"] != config["grid"] or extension["parameters"] != config["parameters"]
                or extension["simulation"] != config["simulation"] or extension["initial"] != config["initial"]
                or extension["comparison_manifest_sha256"] != sha256((DATA / "manifest.json").read_bytes()).hexdigest()
                or conditional_hash is None or extension["conditional_policy_manifest_sha256"] != conditional_hash):
            raise ValueError("Conditional outcomes are incomplete or incompatible; run prepare_conditional_outcomes.py.")
        for name, digest in extension["data_files_sha256"].items():
            if sha256((DATA / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Conditional outcome snapshot changed: {name}")
        model = dict(extension["model"])
        outcome_config["models"][model.pop("key")] = model
        outcomes = pd.concat([outcomes, pd.read_csv(DATA / "conditional_outcomes.csv")], ignore_index=True)
        outcome_hash = sha256(outcome_path.read_bytes()).hexdigest()
    FIGURES.mkdir(parents=True, exist_ok=True)
    previous_path = DATA / "figure_manifest.json"
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    with plt.rc_context(STYLE):
        if not args.outcomes_only:
            policy_figure(policy_config, policies)
        if not args.policies_only:
            outcome_figure(outcome_config, outcomes, "product2_market", "Product 2 usage",
                           "Product 2 share of all purchases", "02_product2_usage_across_rho")
            outcome_figure(outcome_config, outcomes, "seller_profit", "Seller profit",
                           "Mean seller profit per calendar period", "03_profit_across_rho")
    figure_paths = sorted(FIGURES.glob("*.png")) + sorted(FIGURES.glob("*.pdf"))
    plot_hash = sha256(Path(__file__).read_bytes()).hexdigest()
    data_hash = sha256((DATA / "manifest.json").read_bytes()).hexdigest()
    provenance = dict(previous.get("figure_provenance", {}))
    for path in figure_paths:
        is_policy = path.stem == "01_optimal_policies_across_rho"
        if (is_policy and not args.outcomes_only) or (not is_policy and not args.policies_only):
            provenance[path.name] = dict(plot_code_sha256=plot_hash, data_manifest_sha256=data_hash)
            if is_policy and conditional_hash is not None:
                provenance[path.name]["conditional_policy_manifest_sha256"] = conditional_hash
            if not is_policy and outcome_hash is not None:
                provenance[path.name]["conditional_outcomes_manifest_sha256"] = outcome_hash
        elif path.name not in provenance:
            provenance[path.name] = {key:previous.get(key) for key in
                                     ("plot_code_sha256", "data_manifest_sha256")}
    (DATA / "figure_manifest.json").write_text(json.dumps({
        "figure_directory": FIGURES.relative_to(ROOT).as_posix(),
        "grid": config["grid"],
        "figures": {path.name: sha256(path.read_bytes()).hexdigest() for path in figure_paths},
        "plot_code_sha256": plot_hash,
        "data_manifest_sha256": data_hash,
        "figure_provenance": provenance,
        "formats": ["pdf", "png"], "png_dpi": 300,
    }, indent=2) + "\n")
    rendered = "policy plot" if args.policies_only else "usage and profit plots" if args.outcomes_only else "three plots"
    print(f"Saved {rendered}, each as PDF and PNG, in {FIGURES}")


if __name__ == "__main__":
    main()
