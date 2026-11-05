"""Saved-result figures for Experiment 12's average-profit mixture game.

Every detected verified solution is drawn.  Where a parameter case has more
than one solution, the figure displays its full finite set and interrupts the
connecting line rather than inventing a branch continuation.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluid-average-profit-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from oblivious.oe_distribution_fixed_p0.code.reporting import _rho_axis, _style


SERIES = (
    ("monopoly", "Monopoly", "#0072B2", "-", "o"),
    ("competition", "Competition (per seller)", "#D55E00", "--", "D"),
)
METRICS = (
    ("01_product2_fraction_by_rho", "u",
     "Fraction of sales using product 2", "Product 2 / own sales, u"),
    ("02_product2_per_seller_by_rho", "product2_per_seller",
     "Product-2 use per seller and per period", "Product-2 units / seller / period, D × u"),
    ("03_product2_market_by_rho", "product2_market",
     "Total product-2 use in the market per period", "Product-2 units / market / period, N × D × u"),
)


def _read_solutions(path):
    data = pd.read_csv(path)
    required = {"model", "p0", "rho", "solution_id", "u", "demand",
                "product2_per_seller", "product2_market", "verified"}
    if required - set(data):
        raise ValueError(f"Solution table is missing {sorted(required-set(data))}")
    if not data.model.isin([entry[0] for entry in SERIES]).all():
        raise ValueError("Solution table contains an unsupported model")
    verified = data.verified.astype(str).str.lower().isin(["true", "1"])
    data = data[verified].copy()
    if len(data):
        seller_count = data.model.map({"monopoly": 1, "competition": 2})
        errors = (
            abs(data.product2_per_seller-data.demand*data.u),
            abs(data.product2_market-seller_count*data.product2_per_seller),
        )
        if any(error.max() > 2e-10 for error in errors):
            raise ValueError("Product-2 flow identities do not hold")
        if (seller_count*data.demand > 1+1e-10).any():
            raise ValueError("Total seller demand exceeds the normalized market")
        for name in ("u", "demand", "product2_per_seller", "product2_market"):
            if not data[name].between(-1e-10, 1+1e-10).all():
                raise ValueError(f"Invalid {name} in verified solution table")
    return data


def _legend(fig, axes, *, columns=2):
    handles = {}
    for ax in np.asarray(axes, dtype=object).flat:
        artists, labels = ax.get_legend_handles_labels()
        for artist, label in zip(artists, labels, strict=True):
            handles.setdefault(label, artist)
    if handles:
        fig.legend(handles.values(), handles.keys(), loc="outside lower center",
                   ncol=columns, frameon=False, fontsize=9)


def _metric_figure(data, config, metric, title, ylabel):
    p0s, rhos = config["p0s"], config["rhos"]
    columns = min(3, len(p0s))
    rows = math.ceil(len(p0s)/columns)
    fig, all_axes = plt.subplots(rows, columns,
        figsize=(4.35*columns, 3.1*rows+1.15), sharey=True,
        squeeze=False, layout="constrained")
    axes = list(all_axes.flat)
    for ax in axes[len(p0s):]:
        ax.set_visible(False)
    for ax, p0 in zip(axes, p0s):
        chosen = data[np.isclose(data.p0, p0, rtol=0, atol=1e-12)]
        for model, label, color, linestyle, marker in SERIES:
            if metric == "product2_market" and model == "competition":
                label = "Competition (both sellers)"
            selected = chosen[chosen.model.eq(model)]
            groups = [selected[np.isclose(selected.rho, rho, rtol=0, atol=1e-12)]
                      for rho in rhos]
            values = [float(part.iloc[0][metric]) if len(part) == 1 else np.nan
                      for part in groups]
            ax.plot(np.arange(len(rhos)), values, color=color, ls=linestyle,
                    lw=1.6, marker=marker, ms=5, markerfacecolor="none", label=label)
            for index, part in enumerate(groups):
                if len(part) > 1:
                    ax.scatter(np.full(len(part), index), part[metric],
                               edgecolors=color, facecolors="none", marker=marker,
                               s=38, linewidths=1.3, zorder=4)
                elif len(part) == 0:
                    ax.text(index, .02, "none found", ha="center", va="bottom",
                            color=color, fontsize=7, rotation=90,
                            transform=ax.get_xaxis_transform())
        ax.set_title(rf"$p_0={p0:g}$", loc="left")
        ax.set_ylabel(ylabel)
        _rho_axis(ax, rhos)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylim(-.025, 1.025)
        ax.set_yticks(np.arange(0, 1.01, .2))
        _style(ax)
    _legend(fig, axes[:len(p0s)])
    fig.suptitle(
        "Experiment 12 — "+title+"\n"
        "Long-run average profit; every detected verified solution is shown.\n"
        "Multiple solutions appear as sets of points; connecting lines stop at those cases.",
        fontsize=11.5)
    return fig


def _threshold(metadata):
    """Accept the documented top-level threshold and common audit formats."""
    for key in ("deviation_tolerance", "profit_tolerance", "gain_tolerance"):
        value = metadata.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    values = []
    for row in metadata.get("equilibria", []):
        for key in ("deviation_tolerance", "profit_tolerance", "gain_tolerance"):
            if isinstance(row.get(key), (int, float)):
                values.append(float(row[key]))
    if values:
        return max(values)
    return None


def _diagnostic_figure(directory):
    curve_path = directory/"diagnostic_curves.csv"
    metadata_path = directory/"diagnostic.json"
    if not curve_path.exists() or not metadata_path.exists():
        return None
    curves = pd.read_csv(curve_path).sort_values("u")
    metadata = json.loads(metadata_path.read_text())
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), layout="constrained")
    monopoly_color, competition_color = SERIES[0][2], SERIES[1][2]
    ax = axes[0, 0]
    ax.plot(curves.u, curves.monopoly_profit, color=monopoly_color,
            label="Monopoly profit ΠM(u)")
    ax.plot(curves.u, curves.symmetric_competition_profit, color=competition_color,
            ls="--", label="Symmetric profit ΠC(u,u) — reference")
    ax.plot(curves.u, curves.best_response_profit, color="#009E73", ls=":",
            label="Best-response value against rival u")
    for row in metadata.get("monopoly", []):
        ax.scatter(row["u"], row["average_profit"], color=monopoly_color,
                   marker="o", s=45, zorder=4)
    for row in metadata.get("equilibria", []):
        ax.scatter(row["u"], row["average_profit"], color=competition_color,
                   marker="D", s=45, zorder=4)
    ax.set(title="Average profits and best-response values",
           xlabel="Candidate mixture u", ylabel="Average profit / seller / period")

    ax = axes[0, 1]
    payoffs_path = directory/"diagnostic_payoffs.csv"
    if payoffs_path.exists():
        payoffs = pd.read_csv(payoffs_path)
        palette = [competition_color, "#009E73", "#CC79A7", "#E69F00", monopoly_color]
        for index, (rival, part) in enumerate(payoffs.groupby("rival_u", sort=True)):
            part = part.sort_values("u")
            ax.plot(part.u, part.profit, color=palette[index % len(palette)],
                    ls=["-", "--", ":", "-."][index % 4],
                    label=f"Rival fixed at u={rival:.6g}")
    ax.set(title="Unilateral deviation profit: rival held fixed",
           xlabel="Own mixture v", ylabel="ΠC(v, fixed rival u)")

    ax = axes[1, 0]
    br_path = directory/"diagnostic_best_responses.csv"
    if br_path.exists():
        best_responses = pd.read_csv(br_path)
        ax.scatter(best_responses.rival_u, best_responses.best_response_u,
                   color=competition_color, s=8, alpha=.8,
                   label="All detected global best responses")
    else:
        ax.scatter(curves.u, curves.best_response_u_low, color=competition_color,
                   s=8, label="Lowest/highest detected best response")
        ax.scatter(curves.u, curves.best_response_u_high,
                   color=competition_color, s=8)
    ax.plot([0, 1], [0, 1], color=".4", lw=1, ls=":", label="Symmetric mixture v=u")
    for row in metadata.get("equilibria", []):
        ax.scatter(row["u"], row["u"], color="#009E73", marker="D", s=45, zorder=4,
                   label="Verified symmetric equilibrium")
    ax.set(title="Best-response correspondence against fixed rival",
           xlabel="Rival mixture u", ylabel="Own best-response mixture v",
           xlim=(-.02, 1.02), ylim=(-.02, 1.02))

    ax = axes[1, 1]
    ax.plot(curves.u, np.maximum(0, curves.deviation_gain), color=competition_color,
            label="g(u) = maxv ΠC(v,u) − ΠC(u,u)")
    tolerance = _threshold(metadata)
    if tolerance is not None:
        ax.axhline(tolerance, color="#009E73", ls="--", lw=1,
                   label=f"Largest candidate tolerance {tolerance:.2g}")
    for row in metadata.get("equilibria", []):
        gain = row.get("max_deviation_gain", row.get("deviation_gain", 0))
        ax.scatter(row["u"], max(0, gain), marker="D", color="#009E73",
                   s=45, zorder=4, clip_on=False)
        candidate_tolerance = row.get("profit_tolerance", row.get("deviation_tolerance"))
        if candidate_tolerance is not None and tolerance is not None and not np.isclose(
                candidate_tolerance, tolerance, rtol=1e-8, atol=0):
            ax.plot([row["u"]], [candidate_tolerance], marker="_", ms=12,
                    color="#009E73", label="Equilibrium-specific tolerance")
    linear_threshold = max(tolerance or 1e-10, 1e-13)
    ax.set_yscale("symlog", linthresh=linear_threshold)
    ax.set_xlim(-.02, 1.02)
    ax.set_ylim(bottom=-.25*linear_threshold)
    ax.set(title="Maximum unilateral deviation gain (nonnegative)",
           xlabel="Candidate symmetric mixture u", ylabel="Profit gain / seller / period")
    for ax in axes.flat:
        _style(ax)
        artists, labels = ax.get_legend_handles_labels()
        distinct = dict(zip(labels, artists))
        ax.legend(distinct.values(), distinct.keys(), loc="best", fontsize=8, frameon=False)
    figure.suptitle(
        rf"Experiment 12 — Fixed-rival diagnostic: $p_0={metadata['p0']:g}$, "
        rf"$\rho={metadata['rho']:g}$"+"\n"
        "Equilibria pass the global unilateral-deviation test; the diagonal profit curve is a reference.",
        fontsize=12)
    return figure


def make_figures(output_dir, config):
    """Render all requested comparison figures and the available diagnostic."""
    output_dir = Path(output_dir)
    data_dir, figure_dir = output_dir/"data", output_dir/"figures"/"overview"
    figure_dir.mkdir(parents=True, exist_ok=True)
    solutions = _read_solutions(data_dir/"solutions.csv")
    paths = []
    combined = figure_dir/"average_profit_monopoly_vs_competition.pdf"
    with PdfPages(combined) as pdf:
        figures = [(stem, _metric_figure(solutions, config, metric, title, ylabel))
                   for stem, metric, title, ylabel in METRICS]
        diagnostic = _diagnostic_figure(data_dir)
        if diagnostic is not None:
            figures.append(("04_fixed_rival_diagnostic_p0_0p8_rho_0p9", diagnostic))
        for stem, figure in figures:
            for suffix in ("png", "pdf"):
                path = figure_dir/f"{stem}.{suffix}"
                figure.savefig(path, dpi=180, bbox_inches="tight")
                paths.append(str(path))
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
    paths.append(str(combined))
    manifest = {
        "experiment": 12,
        "objective": "long_run_average_profit",
        "solution_table_sha256": sha256((data_dir/"solutions.csv").read_bytes()).hexdigest(),
        "multiplicity": "All solution points shown; lines interrupted at multiple-solution cases",
        "metrics": {metric: title for _, metric, title, _ in METRICS},
        "figures": paths,
    }
    (output_dir/"figures"/"report_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return paths
