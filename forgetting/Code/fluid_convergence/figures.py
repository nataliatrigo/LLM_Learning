"""Render saved spatial comparisons for the original stochastic model."""

from pathlib import Path
import math
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/forgetting-fluid-convergence-mpl")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd


STYLE = {
    "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "legend.fontsize": 9, "axes.linewidth": .8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}
BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#626a73"
SERIES = {
    "loss_x0": (BLUE, "-", r"Pérdida en $x_0=0.5$"),
    "loss_sup": (ORANGE, "--", r"Pérdida máxima en $x$"),
    "loss_grid_change_sup": (GREY, ":", "Cambio espacial de la pérdida"),
    "value_star_change_sup": (BLUE, "-", r"Cambio de $V^*$"),
    "value_policy_change_sup": (GREEN, "--", r"Cambio de $V^{\pi_{\rm fluid}}$"),
    "loss_change_sup": (ORANGE, ":", "Cambio de la pérdida"),
}


def _read(output, name):
    path = output / name
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _require(data, fields, name):
    missing = set(fields) - set(data.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")


def _facets(count, title, *, columns=3):
    columns = min(columns, count)
    rows = math.ceil(count / columns)
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(5.0*columns, 2.9*rows+1.2))
    fig.subplots_adjust(left=.08, right=.985, bottom=1.25/fig.get_figheight(),
                        top=.73 if rows == 1 else .85, hspace=.5, wspace=.27)
    fig.suptitle(title, y=.985, fontsize=15)
    visible = list(axes.flat)[:count]
    for ax in visible:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.2)
        ax.set_axisbelow(True)
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    return fig, visible


def _legend(fig, handles, *, rows=1):
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(.53, .93 if rows == 1 else .955),
               ncol=min(4, len(handles)), frameon=False)


def _series_handles(fields):
    return [Line2D([], [], color=SERIES[field][0], ls=SERIES[field][1],
                   lw=1.7, marker="o", ms=3.5, label=SERIES[field][2])
            for field in fields]


def _save(fig, destination, name, caption):
    fig.text(.53, .12/fig.get_figheight(), caption, ha="center", va="bottom", fontsize=8.3)
    paths = []
    for extension in ("png", "pdf"):
        path = destination / f"{name}.{extension}"
        metadata = ({"Software": "forgetting fluid_convergence"}
                    if extension == "png" else
                    {"Creator": "forgetting fluid_convergence", "CreationDate": None,
                     "ModDate": None})
        fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white", metadata=metadata)
        paths.append(path)
    plt.close(fig)
    return paths


def _symlog(ax, data, fields):
    """Keep signed values and exact zeros; the axis alone changes scale."""
    values = np.concatenate([data[field].to_numpy(dtype=float) for field in fields])
    finite = values[np.isfinite(values)]
    scale = max(float(np.max(np.abs(finite))) if len(finite) else 0., 1e-12)
    threshold = max(scale*1e-5, 1e-12)
    ax.set_yscale("symlog", linthresh=threshold, linscale=.7)
    plotted = np.concatenate([np.asarray(line.get_ydata(), dtype=float) for line in ax.lines])
    plotted = plotted[np.isfinite(plotted)]
    if not len(plotted):
        return
    lower, upper = float(plotted.min()), float(plotted.max())
    if lower == upper == 0:
        ax.set_ylim(-threshold*.2, threshold)
        ax.set_yticks([0], ["0"])
        ax.axhline(0, color=GREY, lw=.7, alpha=.6)
        return
    bottom = lower*1.4 if lower < 0 else (lower/1.4 if lower > 0 else -threshold*.15)
    top = upper*1.4 if upper > 0 else (upper/1.4 if upper < 0 else threshold*.15)
    ax.set_ylim(bottom, top)
    # Do not crowd 0 with logarithmic ticks inside the linear neighborhood.
    ticks = [(0., "0")] if bottom <= 0 <= top else []
    for sign, bound in ((1, top), (-1, -bottom)):
        if bound <= 0:
            continue
        near = bottom if sign > 0 else -top
        narrow = near > 0 and bound/near < 10
        start = (math.floor(math.log10(near)) if narrow else
                 math.ceil(math.log10(max(threshold*10, 1e-300))))
        stop = math.floor(math.log10(bound))
        exponents = list(range(start, stop+1))
        if len(exponents) > 6:
            exponents = exponents[::math.ceil(len(exponents)/6)]
        for exponent in exponents:
            for multiplier in ((1, 2, 5) if narrow else (1,)):
                tick = sign*multiplier*10.**exponent
                if bottom <= tick <= top:
                    prefix = "-" if sign < 0 else ""
                    coefficient = "" if multiplier == 1 else rf"{multiplier}\times "
                    ticks.append((tick, rf"${prefix}{coefficient}10^{{{exponent}}}$"))
    if not ticks or (len(ticks) == 1 and ticks[0][0] == 0):
        edge = upper if upper > 0 else lower
        if edge != 0:
            ticks.append((edge, f"{edge:.1g}"))
    ticks.sort()
    ax.set_yticks([value for value, _ in ticks], [label for _, label in ticks])
    if bottom <= 0 <= top:
        ax.axhline(0, color=GREY, lw=.7, alpha=.6)


def _plot_fields(ax, x, frame, fields):
    for field in fields:
        color, style, _ = SERIES[field]
        ax.plot(x, frame[field], color=color, ls=style, lw=1.7, marker="o", ms=3.5)


def _rho_axis(ax, frame):
    ax.set_xticks(np.arange(len(frame)), [f"{rho:g}" for rho in frame.rho])
    ax.set_xlabel(r"$\rho$ (muestras discretas)")


def _closest_p0(frame, desired=.5):
    values = np.asarray(sorted(frame.p0.unique()), dtype=float)
    chosen = float(values[np.argmin(abs(values-desired))])
    return frame[np.isclose(frame.p0, chosen)], chosen


def _original_summary(data, destination):
    _require(data, ("p0", "rho", "loss_x0", "loss_sup", "relative_loss_x0",
                    "loss_grid_change_sup"), "original_summary.csv")
    p0s = sorted(data.p0.unique())
    fields = ("loss_x0", "loss_sup", "loss_grid_change_sup")
    fig, axes = _facets(len(p0s), "Modelo estocástico original: pérdida de la política fluida")
    for p0, ax in zip(p0s, axes):
        frame = data[np.isclose(data.p0, p0)].sort_values("rho")
        _plot_fields(ax, np.arange(len(frame)), frame, fields)
        _symlog(ax, data, fields)
        _rho_axis(ax, frame)
        ax.set(title=rf"$p_0={p0:g}$", ylabel="Diferencia de valor")
    _legend(fig, _series_handles(fields), rows=math.ceil(len(p0s)/3))
    paths = _save(fig, destination, "01_original_policy_loss",
                  r"Pérdida $V^*-V^{\pi_{\rm fluid}}$ en el modelo original; grillas más finas. "
                  "Eje y simlog: lineal cerca de cero.\n"
                  "Las diferencias entre mallas son diagnósticos, no cotas de error; las líneas unen muestras.")

    fig, axes = _facets(len(p0s), "Modelo estocástico original: pérdida relativa en x₀ = 0.5")
    for p0, ax in zip(p0s, axes):
        frame = data[np.isclose(data.p0, p0)].sort_values("rho")
        ax.plot(np.arange(len(frame)), frame.relative_loss_x0, color=BLUE, lw=1.8,
                marker="o", ms=4)
        ax.axhline(0, color=GREY, lw=.7)
        _rho_axis(ax, frame)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set(title=rf"$p_0={p0:g}$", ylabel="Pérdida relativa al valor óptimo")
    paths += _save(fig, destination, "02_original_relative_policy_loss",
                   r"$(V^*(x_0)-V^{\pi_{\rm fluid}}(x_0))/V^*(x_0)$; "
                   "dinámica original sin cambios. Las líneas unen muestras de ρ.")
    return paths


def _original_profiles(data, destination):
    _require(data, ("p0", "rho", "x", "loss"), "original_curves.csv")
    selected, p0 = _closest_p0(data)
    rhos = sorted(selected.rho.unique())
    fig, axes = _facets(len(rhos), rf"Modelo estocástico original: pérdida por reputación · $p_0={p0:g}$",
                        columns=2)
    for rho, ax in zip(rhos, axes):
        frame = selected[np.isclose(selected.rho, rho)].sort_values("x")
        ax.plot(frame.x, frame.loss, color=ORANGE, lw=1.8)
        ax.axhline(0, color=GREY, lw=.7)
        ax.set(title=rf"$\rho={rho:g}$", xlabel="Reputación normalizada x",
               ylabel=r"$V^*(x)-V^{\pi_{\rm fluid}}(x)$", xlim=(0, 1))
    return _save(fig, destination, "03_original_loss_profiles",
                 "La política fluida se evalúa en el modelo estocástico original. "
                 "Sólo se refina la malla espacial.")


def _original_grid(data, destination):
    _require(data, ("p0", "rho", "n_fine", "value_star_change_sup",
                    "value_policy_change_sup", "loss_change_sup"), "grid_refinement.csv")
    original = data
    if original.empty:
        return []
    selected, p0 = _closest_p0(original)
    rhos = sorted(selected.rho.unique())
    fields = ("value_star_change_sup", "value_policy_change_sup", "loss_change_sup")
    fig, axes = _facets(len(rhos), rf"Refinamiento espacial del modelo estocástico original · $p_0={p0:g}$",
                        columns=2)
    for rho, ax in zip(rhos, axes):
        frame = selected[np.isclose(selected.rho, rho)].sort_values("n_fine")
        _plot_fields(ax, frame.n_fine, frame, fields)
        _symlog(ax, selected, fields)
        ax.set_xscale("log", base=2)
        ax.set_xticks(frame.n_fine, [str(int(n)) for n in frame.n_fine])
        ax.set(title=rf"$\rho={rho:g}$", xlabel="Nodos de la malla fina",
               ylabel="Cambio supremo entre mallas")
    _legend(fig, _series_handles(fields), rows=math.ceil(len(rhos)/2))
    return _save(fig, destination, "04_original_spatial_refinement",
                 "Se mantiene fija la política fluida de referencia al cambiar la malla estocástica.\n"
                 "Eje y simlog, lineal cerca de cero. Los cambios espaciales no son cotas de error.")


def _fluid_grid(data, destination):
    fields = ("value_change_sup", "policy_value_change_sup", "threshold_change_max")
    _require(data, ("p0", "rho", "n_fine", *fields), "fluid_refinement.csv")
    selected, p0 = _closest_p0(data)
    rhos = sorted(selected.rho.unique())
    colors = plt.get_cmap("viridis")(np.linspace(.08, .85, len(rhos)))
    labels = (r"Cambio de $V_{\rm fluid}$",
              "Cambio de valor al refinar la política fluida", "Cambio máximo de umbrales")
    fig, axes = _facets(3, rf"Control espacial de la referencia fluida · $p_0={p0:g}$")
    for ax, field, label in zip(axes, fields, labels):
        for rho, color in zip(rhos, colors):
            frame = selected[np.isclose(selected.rho, rho)].sort_values("n_fine")
            ax.plot(frame.n_fine, frame[field], color=color, marker="o", ms=3.5, lw=1.5)
        _symlog(ax, selected, (field,))
        ax.set_xscale("log", base=2)
        ticks = sorted(selected.n_fine.unique())
        ax.set_xticks(ticks, [str(int(n)) for n in ticks], rotation=20)
        ax.set(title=label, xlabel="Nodos de la malla fina", ylabel="Cambio entre mallas")
    _legend(fig, [Line2D([], [], color=color, marker="o", lw=1.5, ms=3.5,
                         label=rf"$\rho={rho:g}$") for rho, color in zip(rhos, colors)])
    return _save(fig, destination, "05_fluid_spatial_refinement",
                 "Eje y simlog, lineal cerca de cero. Cambios entre mallas de la referencia fluida; "
                 "no son cotas del error frente al modelo estocástico.")


def make_figures(output_dir):
    """Read available result CSVs and return generated PNG/PDF Paths.

    No values are rounded, clipped, or replaced to accommodate logarithmic axes.
    """
    output = Path(output_dir)
    destination = output / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    jobs = (
        ("original_summary.csv", _original_summary),
        ("original_curves.csv", _original_profiles),
        ("grid_refinement.csv", _original_grid),
        ("fluid_refinement.csv", _fluid_grid),
    )
    with plt.rc_context(STYLE):
        for filename, render in jobs:
            data = _read(output, filename)
            if not data.empty:
                paths.extend(render(data, destination))
    if not paths:
        raise ValueError(f"No result rows available to plot in {output}")
    return paths
