#!/usr/bin/env python3
"""Render the finite-grid policy sweep. Does not solve or modify sweep data."""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.patheffects as pe


ROOT = Path(__file__).resolve().parents[3] / 'forgetting/outputs/threshold_parameter_map'
BASE_DP, BASE_DC, BASE_RHO = 0.45, 0.60, 0.95
PARAMETERS = r"$R=1$, $\gamma=0.98$, $c_1=0.05$, $p_1=0.35$, $p_0=0.50$"
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
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 13, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#929aa5", "axes.labelcolor": "#273545",
    "xtick.color": "#455163", "ytick.color": "#455163",
    "text.color": "#233344", "figure.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42,
})


def edges(values):
    values = np.asarray(values, float)
    if len(values) == 1:
        return np.array([values[0] - 0.005, values[0] + 0.005])
    d = np.diff(values)
    return np.r_[values[0] - d[0] / 2,
                 (values[:-1] + values[1:]) / 2,
                 values[-1] + d[-1] / 2]


def subset(data, **kwargs):
    mask = np.ones(len(data), dtype=bool)
    for key, value in kwargs.items():
        mask &= np.isclose(data[key].to_numpy(float), value, atol=1e-8, rtol=0)
    return data.loc[mask].copy()


def pivot(data, value):
    out = data.pivot(index="delta_c", columns="delta_p", values=value)
    return out.sort_index().sort_index(axis=1)


def save(fig, folder, name):
    for ext in ("png", "pdf"):
        fig.savefig(folder / f"{name}.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def base_star(ax, x=BASE_DP, y=BASE_DC, size=155):
    return ax.scatter(x, y, marker="*", s=size, c="#ffd260",
                      edgecolor="#263644", linewidth=1.1, zorder=10)


def map_axes(ax, frame):
    x = np.sort(frame.delta_p.unique())
    y = np.sort(frame.delta_c.unique())
    ax.set(xlim=(edges(x)[0], edges(x)[-1]),
           ylim=(edges(y)[0], edges(y)[-1]),
           xlabel=r"Success gap $\Delta p=p_2-p_1$",
           ylabel=r"Cost gap $\Delta c=c_2-c_1$")
    ax.set_xticks([v for v in [0.15, 0.25, 0.35, 0.45, 0.55, 0.64]
                   if x.min() - 1e-6 <= v <= x.max() + 1e-6])
    ax.set_yticks([v for v in [0.1, 0.3, 0.5, 0.7, 0.9]
                   if y.min() - 1e-6 <= v <= y.max() + 1e-6])


def phase_maps(data, folder, frontier):
    rhos = np.sort(data.rho.unique())
    ncols = min(3, len(rhos))
    nrows = int(np.ceil(len(rhos) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.2, 4.05 * nrows + 1.9),
                             sharex=True, sharey=True, squeeze=False)
    cmap = ListedColormap([COLORS[k] for k in REGIMES])
    norm = BoundaryNorm(np.arange(-0.5, len(REGIMES) + 0.5), cmap.N)
    any_frontier = False
    for ax, rho in zip(axes.flat, rhos):
        frame = subset(data, rho=rho)
        frame["code"] = frame.regime.map({name: i for i, name in enumerate(REGIMES)})
        table = pivot(frame, "code")
        ax.pcolormesh(edges(table.columns), edges(table.index), table.to_numpy(),
                      cmap=cmap, norm=norm, rasterized=True, shading="flat")
        map_axes(ax, frame)
        title = rf"$\rho={rho:.2f}$"
        if np.isclose(rho, BASE_RHO):
            title += "  ·  base case"
        ax.set_title(title, loc="left", pad=10, fontweight="bold")
        base_star(ax)
        if frontier is not None:
            row = subset(frontier, rho=rho)
            if len(row):
                slope = row.K_rho.iloc[0]
                xx = np.linspace(table.columns.min(), table.columns.max(), 250)
                yy = slope * xx
                keep = (yy >= ax.get_ylim()[0]) & (yy <= ax.get_ylim()[1])
                ax.plot(xx[keep], yy[keep], color="#283441", linestyle=(0, (4, 3)),
                        lw=1.15, zorder=5,
                        path_effects=[pe.Stroke(linewidth=2.8, foreground="white"), pe.Normal()])
                any_frontier = True
    for ax in list(axes.flat)[len(rhos):]:
        ax.set_visible(False)
    for ax in axes[:-1].flat:
        ax.set_xlabel("")
    for ax in axes[:, 1:].flat:
        ax.set_ylabel("")
    present = [k for k in REGIMES if k in set(data.regime)]
    handles = [Patch(facecolor=COLORS[k], label=LABELS[k]) for k in present]
    handles += [Line2D([], [], marker="*", color="none", markerfacecolor="#ffd260",
                       markeredgecolor="#263644", markersize=12,
                       label=r"Base gaps: $\Delta p=0.45$, $\Delta c=0.60$")]
    if any_frontier:
        handles += [Line2D([], [], color="#283441", linestyle="--", lw=1.2,
                           label=r"Existence boundary from $U_\rho^1$ (numerical)")]
    fig.suptitle("Where does the two-threshold policy appear?", x=0.075, y=0.99,
                 ha="left", fontsize=21, fontweight="bold")
    fig.text(0.075, 0.952, PARAMETERS + "  |  Product sequence is read as state $x$ increases.",
             fontsize=11.8)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.515, 0.040),
               ncol=3, frameon=False, fontsize=10.5, handlelength=1.9,
               columnspacing=2.3)
    fig.text(0.075, 0.018,
             "Each tile represents a solved parameter pair; finite-grid numerical evidence. "
             r"The domain satisfies $p_0\leq p_2$.", fontsize=10, color="#616c79")
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.19, top=0.90,
                        wspace=0.12, hspace=0.24)
    save(fig, folder, "01_policy_phase_maps")


def threshold_maps(data, folder):
    frame = subset(data, rho=BASE_RHO)
    baseline = subset(frame, delta_p=BASE_DP, delta_c=BASE_DC)
    if frame.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 7.2), sharex=True, sharey=True)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#e5e8ec")
    for ax, name, label in zip(axes, ["x_lower", "x_upper"],
                               ["Lower threshold", "Upper threshold"]):
        frame["display"] = frame[name].where(frame.regime.eq("two"))
        table = pivot(frame, "display")
        mesh = ax.pcolormesh(edges(table.columns), edges(table.index),
                             np.ma.masked_invalid(table.to_numpy()),
                             cmap=cmap, vmin=0, vmax=1, rasterized=True)
        map_axes(ax, frame)
        label_detail = ""
        if len(baseline) and baseline.regime.iloc[0] == "two":
            label_detail = f"  ·  base value {baseline[name].iloc[0]:.4f}"
        ax.set_title(label + label_detail, loc="left", pad=12,
                     fontsize=12.5, fontweight="bold")
        base_star(ax)
    axes[1].set_ylabel("")
    fig.subplots_adjust(left=0.075, right=0.90, bottom=0.19, top=0.77, wspace=0.10)
    cax = fig.add_axes([0.924, 0.19, 0.018, 0.58])
    cb = fig.colorbar(mesh, cax=cax)
    cb.set_label("State $x$ at switching point", labelpad=9)
    fig.suptitle(r"How the two thresholds move at $\rho=0.95$", x=0.075,
                 y=0.97, ha="left", fontsize=20, fontweight="bold")
    fig.text(0.075, 0.915, PARAMETERS, fontsize=12)
    fig.text(0.075, 0.095,
             "Color is shown only where product 2 occupies one interior interval. "
             "Gray indicates another policy pattern.", fontsize=10.6)
    fig.text(0.075, 0.055,
             r"★ marks $(\Delta p,\Delta c)=(0.45,0.60)$. "
             "Thresholds are interpolated from numerical state grids; tiles show sampled parameter pairs.",
             fontsize=10.2, color="#616c79")
    save(fig, folder, "02_threshold_locations")


def threshold_slices(data, folder):
    frames = [subset(data, rho=BASE_RHO, delta_p=BASE_DP).sort_values("delta_c"),
              subset(data, rho=BASE_RHO, delta_c=BASE_DC).sort_values("delta_p")]
    if any(frame.empty for frame in frames):
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.7), sharey=True)
    unknown = False
    for ax, frame, param, fixed, base in zip(
            axes, frames, ["delta_c", "delta_p"],
            [r"$\Delta p=0.45$", r"$\Delta c=0.60$"], [BASE_DC, BASE_DP]):
        xx = frame[param].to_numpy(float)
        accepted = frame.regime.isin(["all2", "one_upper", "two", "one_lower"])
        lower = frame.x_lower.where(accepted).to_numpy(float)
        upper = frame.x_upper.where(accepted).to_numpy(float)
        ax.set_facecolor("#f5f6f8")
        ax.fill_between(xx, lower, upper, color=COLORS["two"], alpha=.24)
        ax.plot(xx, lower, color="#145b74", lw=1.8, label="Lower boundary")
        ax.plot(xx, upper, color="#247f89", lw=1.8, linestyle="--", label="Upper boundary")
        uncertain = frame.regime.isin(["multiple", "uncertain"])
        e = edges(xx)
        for i in np.flatnonzero(uncertain):
            ax.axvspan(e[i], e[i+1], facecolor="#bdc4ce", hatch="///",
                       edgecolor="#8793a1", alpha=.6)
            unknown = True
        ax.axvline(base, color="#9e6920", linestyle=":", lw=1.5, zorder=4)
        b = frame.loc[np.isclose(xx, base)]
        if len(b) and b.regime.iloc[0] in ["all2", "one_upper", "two", "one_lower"]:
            ax.scatter([base, base], [b.x_lower.iloc[0], b.x_upper.iloc[0]],
                        s=38, color="#ffd260", edgecolor="#263644", zorder=6)
        ax.set(xlim=(xx.min(), xx.max()), ylim=(0, 1),
               xlabel=(r"Cost gap $\Delta c$" if param == "delta_c"
                       else r"Success gap $\Delta p$"))
        ax.set_title("Vary " + ("cost" if param == "delta_c" else "success")
                      + " gap, hold " + fixed, loc="left", pad=12, fontweight="bold")
        ax.set_yticks(np.arange(0, 1.01, .2))
        ax.grid(axis="y", alpha=.3, linewidth=.65, color="white")
    axes[0].set_ylabel("State $x$")
    handles = [Patch(facecolor=COLORS["two"], alpha=.24, label="Product 2"),
               Patch(facecolor="#f5f6f8", edgecolor="#d6dbe2", label="Product 1"),
               Line2D([], [], color="#145b74", lw=1.8, label="Lower boundary"),
               Line2D([], [], color="#247f89", lw=1.8, linestyle="--", label="Upper boundary"),
               Line2D([], [], color="#9e6920", lw=1.5, linestyle=":", label="Base gap")]
    if unknown:
        handles.append(Patch(facecolor="#bdc4ce", hatch="///", label="Multiple / unresolved"))
    fig.suptitle(r"Two slices through the base case, $\rho=0.95$", x=.075, y=.98,
                 ha="left", fontsize=20, fontweight="bold")
    fig.text(.075, .925, PARAMETERS, fontsize=12)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.51, .070),
               ncol=3, frameon=False, columnspacing=2.6)
    fig.text(.075, .025,
             "Curves connect neighboring sampled parameters. Boundaries at 0 or 1 indicate an interval reaching a state endpoint.",
             fontsize=10.1, color="#616c79")
    fig.subplots_adjust(left=.075, right=.975, bottom=.25, top=.795, wspace=.14)
    save(fig, folder, "03_threshold_slices")


def advantage_curves(data, curves, folder):
    rhos = np.sort(curves.rho.unique())
    ncols = min(3, len(rhos))
    nrows = int(np.ceil(len(rhos) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.2, 3.5 * nrows + 1.3),
                             sharex=True, squeeze=False)
    for ax, rho in zip(axes.flat, rhos):
        frame = subset(curves, rho=rho).sort_values("x")
        xx = frame.x.to_numpy(float)
        aa = frame.advantage.to_numpy(float)
        ax.axhline(0, color="#77808c", lw=.9)
        ax.plot(xx, aa, color="#145b74", lw=1.8)
        ax.fill_between(xx, 0, aa, where=aa > 0, color=COLORS["two"], alpha=.22,
                         interpolate=True)
        baseline = subset(data, rho=rho, delta_p=BASE_DP, delta_c=BASE_DC)
        if len(baseline) and baseline.regime.iloc[0] == "two":
            lower, upper = baseline[["x_lower", "x_upper"]].iloc[0]
            ax.scatter([lower, upper], [0, 0], color="#145b74", s=18, zorder=5)
            ax.text(.03, .93, f"Thresholds: {lower:.3f}, {upper:.3f}",
                     transform=ax.transAxes, va="top", fontsize=10)
        elif len(baseline):
            ax.text(.03, .93, LABELS[baseline.regime.iloc[0]], transform=ax.transAxes,
                     va="top", fontsize=10)
        ax.set_title(rf"$\rho={rho:.2f}$", loc="left", fontweight="bold")
        ax.set(xlim=(0, 1), xlabel="State $x$", ylabel="Advantage of product 2")
        ax.grid(axis="y", alpha=.2, linewidth=.65)
        lim = ax.get_ylim()
        ax.set_ylim(lim[0], lim[1] + .14 * (lim[1] - lim[0]))
    for ax in axes[:-1].flat:
        ax.set_xlabel("")
    for ax in axes[:, 1:].flat:
        ax.set_ylabel("")
    fig.suptitle("The base gaps can produce different policy patterns", x=.075,
                 y=.985, ha="left", fontsize=20, fontweight="bold")
    fig.text(.075, .94, PARAMETERS + r", $\Delta p=0.45$, $\Delta c=0.60$", fontsize=11.6)
    fig.text(.075, .032,
             r"Product 2 is optimal where $-\Delta c+\Delta p\,G_\rho(x)>0$ (shaded). "
             "Each curve uses a numerically optimized continuation value.",
             fontsize=10.5, color="#616c79")
    fig.subplots_adjust(left=.075, right=.985, top=.865, bottom=.13,
                        wspace=.20, hspace=.28)
    save(fig, folder, "04_baseline_advantage_curves")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data" / "parameter_map.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "figures")
    args = parser.parse_args()
    data = pd.read_csv(args.data)
    if data.duplicated(["rho", "delta_p", "delta_c"]).any():
        raise ValueError("Duplicate sweep parameter pairs.")
    unexpected = set(data.regime) - set(REGIMES)
    if unexpected:
        raise ValueError(f"Unknown regimes: {unexpected}")
    args.out.mkdir(parents=True, exist_ok=True)
    fp = args.data.parent / "fixed_policy_frontier.csv"
    frontier = pd.read_csv(fp) if fp.exists() else None
    phase_maps(data, args.out, frontier)
    threshold_maps(data, args.out)
    threshold_slices(data, args.out)
    curves_path = args.data.parent / "baseline_curves.csv"
    if curves_path.exists():
        advantage_curves(data, pd.read_csv(curves_path), args.out)
    summary = {
        "parameter_pairs": len(data),
        "regime_counts": data.regime.value_counts().to_dict(),
        "baseline": subset(data, rho=BASE_RHO, delta_p=BASE_DP,
                           delta_c=BASE_DC).to_dict(orient="records"),
        "figures": sorted(p.name for p in args.out.glob("*.png")),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
