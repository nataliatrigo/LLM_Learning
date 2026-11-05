"""Spanish figures for the two-dimensional fluid control problem.

All panels use the saved solution and trajectories.  Plotting never re-solves
the optimization problem, and categorical policies are not interpolated.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["MPLCONFIGDIR"] = "/tmp/fluid_base_matplotlib"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
from scipy.special import betaincc


COLORS = {0: "#D8DEE4", 1: "#26828E", 2: "#DB7857"}
POLICY_LABELS = {
    "optimal": "Política óptima",
    "product1": "Siempre producto 1",
    "product2": "Siempre producto 2",
}
POLICY_STYLES = {"optimal": "-", "product1": "--", "product2": ":"}
STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#72808B",
    "text.color": "#263642",
    "axes.labelcolor": "#263642",
    "xtick.color": "#52616B",
    "ytick.color": "#52616B",
    "figure.facecolor": "white",
    "axes.facecolor": "#F7F9FB",
    "savefig.facecolor": "white",
    "legend.frameon": False,
    "legend.fontsize": 8,
}

# Only the rendered panels are thinned.  The optimization, saved solution,
# trajectories and numerical diagnostics continue to use every solved node.
SF_MAX_POINTS = 1001
NX_MAX_N_POINTS = 1001
NX_MAX_X_POINTS = 2001


def _sample_indices(count: int, max_points: int) -> np.ndarray:
    """Regular exact-node subsampling, retaining the final domain endpoint."""
    if count < 1 or max_points < 2:
        raise ValueError("A display grid needs source nodes and max_points >= 2")
    stride = max(1, int(np.ceil((count - 1) / (max_points - 1))))
    indices = np.arange(0, count, stride)
    if indices[-1] != count - 1:
        indices = np.r_[indices, count - 1]
    return indices


def _cell_edges(coordinates, *, fallback: float) -> np.ndarray:
    """Midpoint edges also handle a shorter final subsampling interval."""
    coordinates = np.asarray(coordinates, dtype=float)
    if len(coordinates) == 1:
        return coordinates[0] + np.array([-.5, .5]) * fallback
    midpoints = (coordinates[:-1] + coordinates[1:]) / 2
    return np.r_[coordinates[0] - (coordinates[1] - coordinates[0]) / 2,
                 midpoints,
                 coordinates[-1] + (coordinates[-1] - coordinates[-2]) / 2]


def _save(fig, output_dir: Path, stem: str) -> Path:
    destination = output_dir / f"{stem}.png"
    fig.savefig(destination, dpi=185, bbox_inches="tight")
    plt.close(fig)
    return destination


def _note(solution) -> str:
    p = solution.params
    return (
        f"p₀={p.p0:g} · p₁={p.p1:g} · p₂={p.p2:g} · Δc={p.delta_c:g} · "
        f"δ={p.delta:g} · h de cálculo={solution.h:g} · "
        f"corte de cálculo N={solution.cutoff_n:g}"
        + ("\nMapa dibujado con una selección de nodos; datos guardados a resolución completa."
           if len(solution.actions) > SF_MAX_POINTS else "")
    )


def _square(rows, *, max_points: int = SF_MAX_POINTS) -> np.ma.MaskedArray:
    """Sample exact Cartesian nodes (F,S), without building a full square.

    Policy categories are copied, never averaged or linearly interpolated.
    Source indices, rather than array dimensions, determine plot coordinates.
    """
    indices = _sample_indices(len(rows), max_points)
    result = np.full((len(indices), len(indices)), np.nan)
    for i, failures in enumerate(indices):
        successes = indices[indices + failures < len(rows)]
        result[i, :len(successes)] = np.fromiter(
            (rows[int(failures + j)][int(j)] for j in successes),
            dtype=float, count=len(successes),
        )
    return np.ma.masked_invalid(result)


def _sf(ax, solution, data, *, cmap, norm=None, vmin=None, vmax=None,
        max_points: int = SF_MAX_POINTS):
    indices = _sample_indices(len(solution.actions), max_points)
    if data.shape != (len(indices), len(indices)):
        raise ValueError("Cartesian display data and sampled source coordinates differ")
    extent = (len(solution.actions) - 1) * solution.h
    coordinates = indices * solution.h
    edges = _cell_edges(coordinates, fallback=solution.h)
    mesh = ax.pcolormesh(
        edges, edges, data, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
        shading="flat", rasterized=True,
    )
    # Clip the outer half-cells exactly at S+F=N instead of smearing the
    # categorical policy into states outside the reported triangle.
    if extent > 0:
        triangle = Polygon(((0, 0), (extent, 0), (0, extent)), transform=ax.transData)
        mesh.set_clip_path(triangle)
        ax.plot([0, extent], [extent, 0], color="#8B97A1", linewidth=.65)
        ax.set(xlim=(0, extent), ylim=(0, extent))
    else:
        ax.set(xlim=(-solution.h / 2, solution.h / 2),
               ylim=(-solution.h / 2, solution.h / 2))
    ax.set(xlabel="Éxitos acumulados, S", ylabel="Fracasos acumulados, F")
    ax.set_aspect("equal", adjustable="box")
    return mesh


def _policy_cmap():
    return ListedColormap([COLORS[i] for i in (0, 1, 2)]), BoundaryNorm([-.5, .5, 1.5, 2.5], 3)


def _policy_legend(actions, include_paths=False):
    present = set(np.unique(actions.compressed()).astype(int))
    labels = {0: "Indiferencia / acción no resuelta", 1: "Producto 1", 2: "Producto 2"}
    handles = [Patch(facecolor=COLORS[i], label=labels[i]) for i in (1, 2, 0) if i in present]
    if include_paths:
        handles.append(Line2D([], [], color="#243644", linewidth=1.4,
                              label="Trayectorias óptimas"))
    return handles


def _optimal_paths(ax, paths, solution, *, nx=False):
    if paths.empty:
        return False
    selected = paths.loc[paths["policy"].eq("optimal")]
    for _, group in selected.groupby("path_id", sort=False):
        group = group.sort_values("t")
        group = group.loc[group["n"].le(solution.report_n + solution.h * .01)]
        if group.empty:
            continue
        horizontal, vertical = ("x", "n") if nx else ("S", "F")
        ax.plot(group[horizontal], group[vertical], color="white", linewidth=2.8, alpha=.85)
        ax.plot(group[horizontal], group[vertical], color="#243644", linewidth=1.15, alpha=.95)
        ax.scatter(group[horizontal].iloc[0], group[vertical].iloc[0], s=22,
                   facecolor="white", edgecolor="#243644", linewidth=.9, zorder=6)
    return not selected.empty


def _policy_sf(solution, paths, directory, actions):
    fig, ax = plt.subplots(figsize=(8, 6.6), layout="constrained")
    cmap, norm = _policy_cmap()
    _sf(ax, solution, actions, cmap=cmap, norm=norm)
    plotted_paths = _optimal_paths(ax, paths, solution)
    ax.set_title("¿Qué producto conviene en cada estado?")
    ax.legend(handles=_policy_legend(actions, plotted_paths), loc="upper right")
    fig.suptitle("Política óptima del modelo fluido", fontsize=15, fontweight="semibold")
    fig.supxlabel(_note(solution), fontsize=8, color="#52616B")
    return _save(fig, directory, "policy_sf")


def _value_demand(solution, directory, values, demands):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    for ax, data, cmap, title, label in (
        (axes[0], values, "viridis", "Beneficio descontado óptimo", "V(S,F)"),
        (axes[1], demands, "YlGnBu", "Demanda en cada estado", "D(S,F)"),
    ):
        mesh = _sf(ax, solution, data, cmap=cmap,
                   vmin=0 if label == "D(S,F)" else None,
                   vmax=1 if label == "D(S,F)" else None)
        ax.set_title(title)
        fig.colorbar(mesh, ax=ax, shrink=.8, pad=.025, label=label)
    fig.suptitle("Valor y demanda", fontsize=15, fontweight="semibold")
    fig.supxlabel(_note(solution), fontsize=8, color="#52616B")
    return _save(fig, directory, "value_and_demand")


def _nx_array(solution, *, max_n_points: int = NX_MAX_N_POINTS,
              max_x_points: int = NX_MAX_X_POINTS):
    """Nearest categorical values along each exact n level; no fitted boundary."""
    count = len(solution.actions)
    # An odd number includes x=1/2, including for a one-level smoke grid.
    x_count = min(2 * max(count - 1, 1) + 1, max_x_points)
    if x_count % 2 == 0:
        x_count -= 1
    x = np.linspace(0, 1, max(x_count, 3))
    levels = _sample_indices(count, max_n_points)
    n = levels * solution.h
    result = np.full((len(levels), len(x)), np.nan)
    for i, k in enumerate(levels):
        successes = x * (n[i] + 2) - 1
        valid = (successes >= -solution.h / 2) & (successes <= n[i] + solution.h / 2)
        indices = np.clip(np.rint(successes[valid] / solution.h).astype(int), 0, k)
        result[i, valid] = np.asarray(solution.actions[k])[indices]
    return x, n, np.ma.masked_invalid(result)


def _policy_nx(solution, paths, directory, actions):
    fig, ax = plt.subplots(figsize=(8.8, 6), layout="constrained")
    x, n, data = _nx_array(solution)
    cmap, norm = _policy_cmap()
    x_edges = _cell_edges(x, fallback=1.)
    n_edges = _cell_edges(n, fallback=solution.h)
    mesh = ax.pcolormesh(x_edges, n_edges, data, cmap=cmap, norm=norm,
                         shading="flat", rasterized=True)
    fine_n = np.linspace(0, max(float(n[-1]), solution.h), 500)
    if n[-1] > 0:
        feasible = np.column_stack((
            np.r_[1 / (fine_n + 2), ((fine_n + 1) / (fine_n + 2))[::-1]],
            np.r_[fine_n, fine_n[::-1]],
        ))
        mesh.set_clip_path(Polygon(feasible, transform=ax.transData))
    ax.plot(1 / (fine_n + 2), fine_n, color="#74828D", linewidth=.8)
    ax.plot((fine_n + 1) / (fine_n + 2), fine_n, color="#74828D", linewidth=.8)
    ax.set(xlim=(0, 1), ylim=(0, n[-1] if n[-1] > 0 else solution.h / 2),
           xlabel="Media posterior, x = (S+1)/(S+F+2)",
           ylabel="Información acumulada, n = S+F",
           title="Misma media, distinta cantidad de información")
    plotted_paths = _optimal_paths(ax, paths, solution, nx=True)
    ax.legend(handles=_policy_legend(actions, plotted_paths), loc="upper right")
    fig.suptitle("Política en coordenadas de creencias", fontsize=15, fontweight="semibold")
    fig.supxlabel("Colores del estado de malla más próximo; no se impone una forma de umbral.\n" +
                  _note(solution), fontsize=8, color="#52616B")
    return _save(fig, directory, "policy_nx")


def _trajectories(solution, paths, directory):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), layout="constrained")
    metrics = (
        ("n", "Información acumulada", "n = S+F"),
        ("x", "Media posterior", "x = (S+1)/(n+2)"),
        ("demand", "Demanda", "D(S,F)"),
        ("discounted_profit", "Beneficio descontado acumulado", "Beneficio hasta t"),
    )
    initial_ids = list(paths["path_id"].drop_duplicates()) if not paths.empty else []
    palette = plt.get_cmap("tab10")
    initial_colors = {name: palette(k % 10) for k, name in enumerate(initial_ids)}
    for ax, (column, title, ylabel) in zip(axes.flat, metrics):
        for (name, policy), group in paths.groupby(["path_id", "policy"], sort=False):
            group = group.sort_values("t")
            ax.plot(group["t"], group[column], color=initial_colors[name],
                    linestyle=POLICY_STYLES.get(policy, "-"),
                    linewidth=1.9 if policy == "optimal" else 1.15,
                    alpha=1 if policy == "optimal" else .72)
        ax.set(title=title, xlabel="Tiempo", ylabel=ylabel)
        ax.grid(alpha=.17, linewidth=.6)
        if column in ("x", "demand"):
            ax.set_ylim(-.025, 1.025)
        if paths.empty:
            ax.text(.5, .5, "No hay trayectorias guardadas", transform=ax.transAxes,
                    ha="center", va="center", color="#72808B")
    handles = [Line2D([], [], color=initial_colors[name], linewidth=2, label=str(name))
               for name in initial_ids]
    present_policies = set(paths["policy"]) if not paths.empty else set()
    handles += [Line2D([], [], color="#263642", linestyle=POLICY_STYLES[policy],
                       label=POLICY_LABELS[policy])
                for policy in POLICY_LABELS if policy in present_policies]
    if handles:
        fig.legend(handles=handles, loc="outside lower center", ncol=min(4, len(handles)))
    fig.suptitle("Trayectorias fluidas: política óptima y productos fijos\n"
                 f"p₀ = {solution.params.p0:g}",
                 fontsize=15, fontweight="semibold")
    return _save(fig, directory, "trajectories")


def _advantage(solution, directory, advantage):
    fig, ax = plt.subplots(figsize=(8.1, 6.6), layout="constrained")
    finite = advantage.compressed()
    maximum = max(float(np.max(np.abs(finite))) if len(finite) else 0, 1e-12)
    mesh = _sf(ax, solution, advantage, cmap="PuOr_r",
               norm=TwoSlopeNorm(vmin=-maximum, vcenter=0, vmax=maximum))
    # Draw a zero contour only if both signs occur, avoiding Matplotlib's
    # warning for examples that use one product throughout the triangle.
    if advantage.shape[0] > 1 and len(finite) and finite.min() < 0 < finite.max():
        grid = _sample_indices(len(solution.actions), SF_MAX_POINTS) * solution.h
        ax.contour(grid, grid, advantage, levels=[0], colors=["#263642"], linewidths=.9)
    fig.colorbar(mesh, ax=ax, shrink=.8, pad=.025, label="Ventaja de producto 2 frente a producto 1")
    ax.set_title("Signo positivo: producto 2 · negativo: producto 1")
    fig.suptitle("Ventaja local de cambiar de producto", fontsize=15, fontweight="semibold")
    fig.supxlabel("La cercanía a cero señala decisiones sensibles a la precisión numérica.\n" +
                  _note(solution), fontsize=8, color="#52616B")
    return _save(fig, directory, "switching_advantage")


def make_figures(solution, paths, output_dir) -> list[Path]:
    """Write the five PNG diagnostics and return their paths.

    ``solution`` is the solved triangular fluid grid and ``paths`` is the
    long-form DataFrame returned by the deterministic trajectory simulator.
    The triangle displayed is the reported interior, not the truncation edge.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    values = _square(solution.values)
    actions = _square(solution.actions)
    advantages = _square(solution.advantages)
    if getattr(solution, "demand_values", None) is not None:
        demand_rows = solution.demand_values
    else:
        demand_rows = []
        for k in range(len(solution.values)):
            successes = np.arange(k + 1) * solution.h
            failures = (k - np.arange(k + 1)) * solution.h
            demand_rows.append(betaincc(successes + 1, failures + 1, solution.params.p0))
    demands = _square(demand_rows)
    with plt.rc_context(STYLE):
        return [
            _policy_sf(solution, paths, directory, actions),
            _value_demand(solution, directory, values, demands),
            _policy_nx(solution, paths, directory, actions),
            _trajectories(solution, paths, directory),
            _advantage(solution, directory, advantages),
        ]
