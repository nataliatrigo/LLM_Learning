#!/usr/bin/env python3
"""Plot the full-gap sweep without changing its numerical classifications."""
from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3] / 'forgetting/outputs/full_delta_map'
COLORS = {
    "all1": "#e2e5e9",
    "all2": "#eab959",
    "one_upper": "#ae9bcb",
    "two": "#248b98",
    "one_lower": "#d791a4",
    "multiple": "#d86e48",
    "uncertain": "#424955",
}
LABELS = {
    "all1": "1 everywhere",
    "all2": "2 everywhere",
    "one_upper": "2 → 1",
    "two": "1 → 2 → 1  (two thresholds)",
    "one_lower": "1 → 2",
    "multiple": "Multiple product-2 intervals",
    "uncertain": "Unresolved on the numerical grid",
}
REGIMES = list(COLORS)
PARAMETERS = r"$R=1$, $\gamma=0.98$, $p_0=0.50$"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 14, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#939aa3", "axes.labelcolor": "#273545",
    "xtick.color": "#455163", "ytick.color": "#455163",
    "text.color": "#233344", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})


def subset(data, **kwargs):
    mask = np.ones(len(data), dtype=bool)
    for key, value in kwargs.items():
        mask &= np.isclose(data[key].to_numpy(float), value, atol=1e-8, rtol=0)
    return data.loc[mask].copy()


def edges(values):
    values = np.asarray(values, float)
    if len(values) < 2:
        raise ValueError("Need at least two sampled gaps on each axis.")
    return np.r_[values[0] - (values[1] - values[0]) / 2,
                 (values[:-1] + values[1:]) / 2,
                 values[-1] + (values[-1] - values[-2]) / 2]


def pivot(data, value):
    table = data.pivot(index="delta_c", columns="delta_p", values=value)
    table = table.sort_index().sort_index(axis=1)
    if table.isna().any().any():
        raise ValueError(f"Missing plotted entries in {value}.")
    return table


def mesh(ax, table, cmap, norm):
    return ax.pcolormesh(edges(table.columns), edges(table.index), table.to_numpy(),
                         cmap=cmap, norm=norm, rasterized=True, shading="flat")


def map_axes(ax):
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.set_xticks(np.arange(0, 1.01, .2))
    ax.set_yticks(np.arange(0, 1.01, .2))
    ax.tick_params(length=3.2)


def save(fig, folder, name):
    for ext in ("png", "pdf"):
        path = folder / f"{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(path, flush=True)
    plt.close(fig)


def count_maps(data, folder):
    rhos = np.sort(data.rho.unique())
    if len(rhos) != 4:
        raise ValueError(f"Expected four rho values; received {rhos}.")
    if not data.n_total.eq(9).all():
        raise ValueError("Every plotted gap pair must have nine tested choices.")
    ramp = LinearSegmentedColormap.from_list(
        "teal_count", ["#f0f2f4", "#d8ebe5", "#8ec8b9", "#338f92", "#124f68"])
    cmap = ListedColormap(ramp(np.linspace(0, 1, 10)))
    norm = BoundaryNorm(np.arange(-.5, 10.5), cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(12.3, 11.35), sharex=True, sharey=True)
    for ax, rho in zip(axes.flat, rhos):
        frame = subset(data, rho=rho)
        table = pivot(frame, "n_two")
        artist = mesh(ax, table, cmap, norm)
        map_axes(ax)
        ax.set_title(rf"$\rho={rho:.2f}$", loc="left", fontweight="bold", pad=9)
        uncertain = frame[frame.n_uncertain.gt(0)]
        if len(uncertain):
            ax.scatter(uncertain.delta_p, uncertain.delta_c, marker="o", s=8,
                       c="#233344", edgecolor="white", linewidth=.3, zorder=5)
        multiple = frame[frame.n_multiple.gt(0)]
        if len(multiple):
            ax.scatter(multiple.delta_p, multiple.delta_c, marker="x", s=22,
                       c="#ca5131", linewidth=1, zorder=6)
    for ax in axes[-1]:
        ax.set_xlabel(r"Success gap $\Delta p=p_2-p_1$", labelpad=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(r"Cost gap $\Delta c=c_2-c_1$", labelpad=8)
    fig.suptitle("How many tested choices produce two thresholds?", x=.075, y=.98,
                 ha="left", fontsize=21, fontweight="bold")
    fig.text(.075, .941, r"$p_1$ and $c_1$ vary: 9 feasible pairs per cell.  "
             + PARAMETERS, fontsize=12.4)
    fig.subplots_adjust(left=.075, right=.86, bottom=.205, top=.885,
                        wspace=.16, hspace=.25)
    cax = fig.add_axes([.895, .253, .024, .535])
    cb = fig.colorbar(artist, cax=cax, ticks=range(10), spacing="uniform")
    cb.set_label("Tested choices with 1 → 2 → 1, out of 9", labelpad=11)
    handles = []
    if data.n_uncertain.gt(0).any():
        handles.append(Line2D([], [], marker="o", markersize=4, linestyle="none",
                              color="#233344", label="At least one unresolved choice"))
    if data.n_multiple.gt(0).any():
        handles.append(Line2D([], [], marker="x", markersize=6, linestyle="none",
                              color="#ca5131", label="At least one multiple-interval choice"))
    if handles:
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.07, .113),
                   frameon=False, ncol=2, fontsize=10.5)
    fig.text(.075, .090,
             r"At each gap pair: $p_1=f(1-\Delta p)$ and $c_1=g(1-\Delta c)$, "
             r"with $f,g\in\{0.1,0.5,0.9\}$.", fontsize=11)
    fig.text(.075, .057,
             r"Then $p_2=p_1+\Delta p$, $c_2=c_1+\Delta c$. "
             "All four levels lie strictly between 0 and 1.", fontsize=11)
    fig.text(.075, .024,
             "Numerical samples: gaps 0.025–0.975, step 0.025. Counts are not probabilities "
             "or a statement about every feasible choice.", fontsize=10, color="#596777")
    save(fig, folder, "01_two_threshold_counts")


def baseline_fraction_maps(data, folder, rho=.95):
    frame = subset(data, rho=rho)
    fs = np.sort(frame.p1_fraction.unique())
    gs = np.sort(frame.c1_fraction.unique())
    if len(fs) != 3 or len(gs) != 3:
        raise ValueError("Expected three probability and three cost fractions.")
    unknown = set(frame.regime) - set(REGIMES)
    if unknown:
        raise ValueError(f"Unknown regime labels: {unknown}.")
    cmap = ListedColormap([COLORS[k] for k in REGIMES])
    norm = BoundaryNorm(np.arange(-.5, len(REGIMES) + .5), cmap.N)
    fig, axes = plt.subplots(3, 3, figsize=(13.6, 13.0), sharex=True, sharey=True)
    for i, g in enumerate(gs):
        for j, f in enumerate(fs):
            ax = axes[i, j]
            part = subset(frame, p1_fraction=f, c1_fraction=g)
            part["code"] = part.regime.map({k: n for n, k in enumerate(REGIMES)})
            mesh(ax, pivot(part, "code"), cmap, norm)
            map_axes(ax)
            if i == 0:
                ax.set_title(rf"$p_1={f:.1f}(1-\Delta p)$", pad=14,
                             fontsize=15, fontweight="bold")
            if j == 0:
                ax.set_ylabel(r"Cost gap $\Delta c$", labelpad=8)
            if i == 2:
                ax.set_xlabel(r"Success gap $\Delta p$", labelpad=8)
    fig.suptitle(r"The underlying levels also affect the policy  ·  $\rho=0.95$",
                 x=.085, y=.987, ha="left", fontsize=20, fontweight="bold")
    fig.text(.085, .950, PARAMETERS + r".  Product sequence is read as state $x$ increases.",
             fontsize=12.3)
    fig.subplots_adjust(left=.14, right=.98, bottom=.19, top=.879,
                        wspace=.12, hspace=.16)
    for i, g in enumerate(gs):
        pos = axes[i, 0].get_position()
        fig.text(.035, (pos.y0 + pos.y1) / 2,
                 rf"$c_1={g:.1f}(1-\Delta c)$", rotation=90,
                 ha="center", va="center", fontsize=15, fontweight="bold")
    present = [k for k in REGIMES if k in set(frame.regime)]
    handles = [Patch(facecolor=COLORS[k], label=LABELS[k]) for k in present]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.535, .091),
               ncol=3 if len(present) <= 6 else 4, frameon=False, fontsize=11, columnspacing=2.8,
               handlelength=1.6, handleheight=1.2)
    fig.text(.085, .059,
             r"Each panel follows a different feasible choice of levels: "
             r"$p_2=p_1+\Delta p$ and $c_2=c_1+\Delta c$.", fontsize=11)
    fig.text(.085, .026,
             "Numerical samples: gaps 0.025–0.975, step 0.025. Teal indicates "
             "one interior interval where product 2 is optimal.", fontsize=10.5,
             color="#596777")
    save(fig, folder, "02_baseline_fraction_maps")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate = pd.read_csv(args.data_dir / "aggregated_map.csv")
    details = pd.read_csv(args.data_dir / "parameter_map.csv")
    count_maps(aggregate, args.output_dir)
    baseline_fraction_maps(details, args.output_dir)


if __name__ == "__main__":
    main()
