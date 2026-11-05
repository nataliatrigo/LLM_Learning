"""Plot the forgetting monopoly at saved OE p0 benchmarks versus competition."""

from hashlib import sha256
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/monopoly-comparison-mpl")
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA, FIGURES = ROOT / "data", ROOT / "Figures"
MODELS = {
    "p0": {"label": "Competition (OE p0)", "color": "#D55E00", "ls": "-"},
    "monopoly": {"label": "Monopoly (forgetting)", "color": "#0072B2", "ls": "--"},
}
STYLE = {
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
    "legend.fontsize": 9, "axes.linewidth": .8, "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def save(fig, name):
    for suffix in ("png", "pdf"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=300,
                    bbox_inches="tight", facecolor="white")
    plt.close(fig)


def model_handles(policy=False):
    return [Line2D([], [], color=item["color"], lw=2,
                   ls=item["ls"] if policy else "-", label=item["label"])
            for item in MODELS.values()]


def policy_figure(config, policies):
    rhos = config["rhos"]
    rows = (len(rhos) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(11, 3.5 * rows), squeeze=False)
    fig.subplots_adjust(left=.075, right=.99, top=.92, bottom=.095,
                        hspace=.4, wspace=.18)
    for ax, rho in zip(axes.flat, rhos):
        curves = []
        for key, item in MODELS.items():
            frame = policies[(policies["rho"] == rho) & (policies["model"] == key)]
            frame = frame.sort_values("reputation")
            x = frame["reputation"].to_numpy()
            alpha = frame["product2_probability"].to_numpy()
            curves.append(alpha)
            ax.plot(x, alpha, color=item["color"], ls=item["ls"],
                    lw=2 if key == "p0" else 1.5, marker=".", ms=3, zorder=3)
            mixed = (alpha > 1e-9) & (alpha < 1 - 1e-9)
            ax.scatter(x[mixed], alpha[mixed], color=item["color"],
                       marker="D", s=28, zorder=5, edgecolors="white", linewidths=.4)
        p0 = config["p0_by_rho"][str(rho)]
        ax.set(title=rf"$\rho={rho:g}$, $p_0^{{OE}}={p0:.6f}$",
               xlabel="Own reputation", ylabel="Probability of product 2",
               xlim=(-.035, 1.035), ylim=(-.04, 1.04),
               xticks=np.linspace(0, 1, 6), yticks=np.linspace(0, 1, 6))
        ax.grid(alpha=.22)
        if np.all(np.asarray(curves) == 0):
            ax.text(.5, .5, "Both policies choose product 1", transform=ax.transAxes,
                    ha="center", fontsize=9, color=".4")
    for ax in axes.flat[len(rhos):]:
        ax.set_visible(False)
    fig.suptitle("Optimal policies: monopoly at the OE p0 equilibrium benchmark",
                 fontsize=13, y=.99)
    handles = model_handles(policy=True)
    handles.append(Line2D([], [], color=".35", ls="none", marker="D", ms=5,
                          label="OE p0 mixing at a grid point"))
    fig.legend(handles=handles, ncol=3, loc="upper center",
               bbox_to_anchor=(.53, .97), frameon=False)
    fig.text(.53, .025, "Same fixed benchmark and Bellman equation; common 51-point mature-state grid.\n"
             "Forgetting uses its original pure choice rule (product 1 at numerical indifference); OE p0 retains its saved mixtures.",
             ha="center", fontsize=8.5, linespacing=1.6)
    save(fig, "01_optimal_policies_across_rho")


def metric_panel(ax, config, outcomes, metric):
    rhos = config["rhos"]
    for (key, item), offset in zip(MODELS.items(), (-.035, .035)):
        rows = outcomes[(outcomes["model"] == key) & (outcomes["metric"] == metric)]
        rows = rows.set_index("rho").loc[rhos]
        x = np.arange(len(rhos)) + offset
        mean = rows["simulation_mean"].to_numpy()
        low = rows["simulation_ci_low"].to_numpy()
        high = rows["simulation_ci_high"].to_numpy()
        ax.plot(x, rows["stationary"], color=item["color"], ls="--", lw=1.4,
                marker="D", ms=6, mfc="white", zorder=3)
        ax.plot(x, rows["finite_window_expected"], color=item["color"],
                ls="-", lw=1.7, zorder=3)
        ax.errorbar(x, mean, yerr=[mean - low, high - mean],
                    color=item["color"], ls="none", lw=1.5, marker="o", ms=4,
                    capsize=3.5, elinewidth=1, zorder=4)
    ax.set_xticks(range(len(rhos)), [f"{rho:g}" for rho in rhos])
    ax.set(xlim=(-.3, len(rhos) - .7), xlabel=r"Forgetting factor $\rho$ (discrete values)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=".9", lw=.7)


def outcome_decoration(fig, config, title):
    fig.suptitle(title, fontsize=14, y=.98)
    fig.legend(handles=model_handles(), ncol=2, loc="upper center",
               bbox_to_anchor=(.53, .92), frameon=False)
    handles = [
        Line2D([], [], color=".3", lw=1.7, label="Exact finite-window expectation"),
        Line2D([], [], color=".3", marker="o", ls="none", ms=4,
               label="Simulation (95% MC CI)"),
        Line2D([], [], color=".3", marker="D", mfc="white", ls="--", lw=1.3,
               label="Stationary prediction"),
    ]
    fig.legend(handles=handles, ncol=3, loc="upper center",
               bbox_to_anchor=(.53, .86), frameon=False)
    sim = config["simulation"]
    fig.text(.53, .165,
             r"Monopoly: one seller versus fixed $p_0^{OE}(\rho)$. Competition: two sellers following OE p0.",
             ha="center", fontsize=8.5)
    fig.text(.53, .12,
             f"{sim['replications']} paths; {sim['periods']:,} retained periods; "
             f"{sim['burn_in']:,} burn-in; initial reputation 0.5; common 51-point grid.",
             ha="center", fontsize=8.5)
    fig.text(.53, .075,
             "At high rho, finite-window and stationary predictions differ; 20-path MC intervals can also miss rare events.",
             ha="center", fontsize=8.2)
    fig.text(.53, .03,
             "Rho values are equally spaced; lines guide the eye. Stationary results refer to this finite grid.",
             ha="center", fontsize=8.2)


def usage_figure(config, outcomes):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6))
    fig.subplots_adjust(left=.065, right=.99, top=.74, bottom=.30, wspace=.23)
    for ax, metric, title, label in zip(axes,
            ("product2_conditional", "product2_calendar"),
            ("(a) Among the seller's sales", "(b) Per seller and calendar period"),
            ("Product 2 share of the seller's sales", "Probability the seller sells product 2")):
        metric_panel(ax, config, outcomes, metric)
        ax.set(title=title, ylabel=label, ylim=(-.025, 1.025))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    outcome_decoration(fig, config, "Product 2 usage: monopoly versus competition")
    save(fig, "02_product2_usage_across_rho")


def profit_figure(config, outcomes):
    fig, ax = plt.subplots(figsize=(11.7, 6))
    fig.subplots_adjust(left=.095, right=.985, top=.74, bottom=.30)
    metric_panel(ax, config, outcomes, "seller_profit")
    ax.set(ylabel="Mean seller profit per calendar period", ylim=(-.015, 1))
    outcome_decoration(fig, config, "Seller profit: monopoly versus competition")
    save(fig, "03_profit_across_rho")


def main():
    config = json.loads((DATA / "comparison.json").read_text())
    manifest = json.loads((DATA / "manifest.json").read_text())
    for name, digest in manifest.get("data_files_sha256", {}).items():
        if sha256((DATA / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Data changed since preparation: {name}")
    policies = pd.read_csv(DATA / "policies.csv")
    outcomes = pd.read_csv(DATA / "outcomes.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        policy_figure(config, policies)
        usage_figure(config, outcomes)
        profit_figure(config, outcomes)
    paths = sorted(FIGURES.glob("*.png")) + sorted(FIGURES.glob("*.pdf"))
    (DATA / "figure_manifest.json").write_text(json.dumps({
        "figures": {path.name: sha256(path.read_bytes()).hexdigest() for path in paths},
        "plot_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "data_manifest_sha256": sha256((DATA / "manifest.json").read_bytes()).hexdigest(),
        "formats": ["png", "pdf"], "png_dpi": 300,
    }, indent=2) + "\n")
    print(f"Saved three plots as PNG and PDF in {FIGURES}")


if __name__ == "__main__":
    main()
