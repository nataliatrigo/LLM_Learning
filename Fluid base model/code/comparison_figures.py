"""Comparable Spanish figures for a sweep of the demand threshold p0.

Each case is read from the standard experiment layout.  The policy panels
show categorical actions on the saved grid, without fitting a threshold.
Only one saved solution is kept in memory at a time.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics import load_solution
# figures configures the writable Matplotlib cache and the Agg backend.
from figures import (COLORS, STYLE, _cell_edges, _nx_array, _policy_cmap,
                     _save, _sf, _square)
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon


# Separate p0 colors from the two categorical product colors.
P0_COLORS = ("#3B528B", "#21918C", "#73AD32", "#D58B19", "#A84772")
PATH_COLOR = "#243644"
_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
COMPARISON_MAX_POINTS = 601
COMPARISON_MAX_X_POINTS = 1201


def _case_info(directory: Path) -> dict:
    """Read small metadata first so panels are sorted numerically by p0."""
    with (directory / "diagnostics.json").open(encoding="utf-8") as stream:
        diagnostics = json.load(stream)
    p0 = float(diagnostics["parameters"]["p0"])
    paths = pd.read_csv(directory / "data" / "trajectories.csv")
    optimal = paths.loc[paths["policy"].eq("optimal")].copy()
    if optimal.empty:
        raise ValueError(f"No optimal trajectories saved in {directory}")
    return {"directory": directory, "p0": p0, "paths": optimal}


def _path_groups(paths: pd.DataFrame):
    """Return deterministic ordering of starts even if CSV ordering changes."""
    grouping = "state_index" if "state_index" in paths else "path_id"
    return [group.sort_values("t") for _, group in paths.groupby(grouping, sort=True)]


def _time_label(paths: pd.DataFrame) -> str:
    endings = np.array([float(group["t"].iloc[-1]) for group in _path_groups(paths)])
    if len(endings) and np.allclose(endings, endings[0], rtol=0, atol=1e-8):
        return f"□ final de cada trayectoria: t={endings[0]:g}"
    return "□ finales: tiempo indicado junto al marcador"


def _draw_paths(ax, paths: pd.DataFrame, *, nx: bool, report_n: float):
    """Distinguish starts and finite-time endpoints explicitly on policy maps."""
    for index, group in enumerate(_path_groups(paths)):
        group = group.loc[group["n"].le(report_n + 1e-8)]
        if group.empty:
            continue
        horizontal, vertical = ("x", "n") if nx else ("S", "F")
        x, y = group[horizontal], group[vertical]
        ax.plot(x, y, color="white", linewidth=2.8, alpha=.9, zorder=4)
        ax.plot(x, y, color=PATH_COLOR, linewidth=1.05, alpha=.95, zorder=5)
        ax.scatter(x.iloc[0], y.iloc[0], s=22, marker="o", facecolor="white",
                   edgecolor=PATH_COLOR, linewidth=.9, zorder=7, clip_on=False)
        ax.scatter(x.iloc[-1], y.iloc[-1], s=22, marker="s", facecolor="white",
                   edgecolor=PATH_COLOR, linewidth=.9, zorder=7, clip_on=False)
        label = _LETTERS[index] if index < len(_LETTERS) else str(index + 1)
        # The origin and the common initial mean are close to the lower edge.
        offset = (5, 6) if nx else (5, 4)
        if nx and index % 2:
            offset = (-11, 5)
        ax.annotate(label, (x.iloc[0], y.iloc[0]), xytext=offset,
                    textcoords="offset points", fontsize=8, fontweight="bold",
                    color=PATH_COLOR, zorder=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1})
        # Label the other end as well, so converging paths remain identifiable.
        # Offset opposite to the start; alternating sides limits collisions.
        end_offset = (5, -11) if index % 2 == 0 else (-11, -11)
        if float(np.hypot(x.iloc[-1] - x.iloc[0], y.iloc[-1] - y.iloc[0])) > .025:
            ax.annotate(label, (x.iloc[-1], y.iloc[-1]), xytext=end_offset,
                        textcoords="offset points", fontsize=7, fontweight="bold",
                        color=PATH_COLOR, zorder=8,
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1})
    endings = [float(group["t"].iloc[-1]) for group in _path_groups(paths)]
    if endings and not np.allclose(endings, endings[0], rtol=0, atol=1e-8):
        for group in _path_groups(paths):
            last = group.iloc[-1]
            horizontal, vertical = ("x", "n") if nx else ("S", "F")
            ax.annotate(f"t={last['t']:g}", (last[horizontal], last[vertical]),
                        xytext=(5, 4), textcoords="offset points", fontsize=7,
                        color=PATH_COLOR, zorder=8)


def _explain_panel(ax, paths: pd.DataFrame, *, nx: bool):
    ax.set_axis_off()
    ax.set_title("Cómo leer estos mapas", loc="left", pad=14)
    handles = [
        Patch(facecolor=COLORS[1], label="Producto 1"),
        Patch(facecolor=COLORS[2], label="Producto 2"),
        Patch(facecolor=COLORS[0], label="Indiferencia / acción no resuelta"),
        Line2D([], [], color=PATH_COLOR, linewidth=1.2, label="Trayectoria óptima"),
        Line2D([], [], color=PATH_COLOR, linestyle="none", marker="o",
               markerfacecolor="white", label="Inicio (letra A–D)"),
        Line2D([], [], color=PATH_COLOR, linestyle="none", marker="s",
               markerfacecolor="white", label="Final de la simulación (misma letra)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=9,
              labelspacing=.65, handlelength=2.1, borderaxespad=0)
    starts = []
    for index, group in enumerate(_path_groups(paths)):
        first = group.iloc[0]
        label = _LETTERS[index] if index < len(_LETTERS) else str(index + 1)
        starts.append(f"{label}: (S₀,F₀)=({first['S']:g},{first['F']:g})")
    text = "Estados iniciales\n" + " · ".join(starts[:2])
    if len(starts) > 2:
        text += "\n" + " · ".join(starts[2:])
    text += "\n\nLos cuadrados marcan el tiempo final;\nno son estados absorbentes."
    text += "\nColores: acción del nodo más próximo;\nno se presupone una frontera de umbral."
    if nx:
        text += "\nLas curvas grises delimitan los estados factibles."
    ax.text(0, .43, text, transform=ax.transAxes, va="top", fontsize=9,
            color="#52616B", linespacing=1.45)


def _draw_nx(ax, solution):
    x, n, data = _nx_array(solution, max_n_points=COMPARISON_MAX_POINTS,
                           max_x_points=COMPARISON_MAX_X_POINTS)
    cmap, norm = _policy_cmap()
    x_edges = _cell_edges(x, fallback=1.)
    n_edges = _cell_edges(n, fallback=solution.h)
    mesh = ax.pcolormesh(x_edges, n_edges, data, cmap=cmap, norm=norm,
                         shading="flat", rasterized=True)
    fine_n = np.linspace(0, solution.report_n, 500)
    left, right = 1 / (fine_n + 2), (fine_n + 1) / (fine_n + 2)
    if solution.report_n > 0:
        feasible = np.column_stack((np.r_[left, right[::-1]], np.r_[fine_n, fine_n[::-1]]))
        mesh.set_clip_path(Polygon(feasible, transform=ax.transData))
        ax.plot(left, fine_n, color="#74828D", linewidth=.65)
        ax.plot(right, fine_n, color="#74828D", linewidth=.65)
    ax.set(xlim=(0, 1), ylim=(0, solution.report_n if solution.report_n > 0 else solution.h / 2),
           xlabel="Media posterior, x=(S+1)/(n+2)",
           ylabel="Información acumulada, n=S+F")
    ax.set_xticks(np.arange(0, 1.01, .2))


def _common_origin(paths: pd.DataFrame, directory: Path) -> pd.DataFrame:
    for group in _path_groups(paths):
        first = group.iloc[0]
        if np.isclose(first["S"], 0, atol=1e-10) and np.isclose(first["F"], 0, atol=1e-10):
            return group
    raise ValueError(f"The comparison needs an optimal path starting at (0,0): {directory}")


def _comparison_trajectories(cases: list[dict], directory: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), layout="constrained")
    metrics = (
        ("n", "Información acumulada", "n = S+F"),
        ("x", "Media posterior", "x = (S+1)/(n+2)"),
        ("demand", "Demanda", "D(S,F)"),
        ("discounted_profit", "Beneficio descontado hasta el tiempo t", "Beneficio acumulado"),
    )
    end_times = []
    handles = []
    for index, case in enumerate(cases):
        group = _common_origin(case["paths"], case["directory"])
        color = P0_COLORS[index % len(P0_COLORS)]
        end_times.append(float(group["t"].iloc[-1]))
        handles.append(Line2D([], [], color=color, linewidth=2.2, label=f"p₀ = {case['p0']:g}"))
        for ax, (metric, _, _) in zip(axes.flat, metrics):
            ax.plot(group["t"], group[metric], color=color, linewidth=2)
            ax.scatter(group["t"].iloc[0], group[metric].iloc[0], marker="o", s=23,
                       color=color, edgecolor="white", linewidth=.65, zorder=5, clip_on=False)
            ax.scatter(group["t"].iloc[-1], group[metric].iloc[-1], marker="s", s=27,
                       color=color, edgecolor="white", linewidth=.65, zorder=5, clip_on=False)
    for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
        ax.set(title=title, xlabel="Tiempo, t", ylabel=ylabel)
        ax.grid(alpha=.18, linewidth=.6)
        ax.set_xlim(left=0, right=max(end_times) * 1.025)
        if metric in ("x", "demand"):
            ax.set_ylim(-.025, 1.025)
    # Keep the shared p0 legend inside the ample upper-left space of n(t),
    # leaving the footer independent of constrained-layout legend placement.
    axes[0, 0].legend(handles=handles, loc="upper left", ncol=1, fontsize=9,
                      title="Umbral de demanda", title_fontsize=9)
    common_end = np.allclose(end_times, end_times[0], rtol=0, atol=1e-8)
    ending = f"Todas terminan en t={end_times[0]:g}." if common_end else "El cuadrado marca el final guardado de cada caso."
    fig.suptitle("Efecto de p₀ sobre la trayectoria óptima desde (S₀,F₀)=(0,0)\n"
                 f"○ inicio · □ final. {ending}", fontsize=15, fontweight="semibold")
    fig.supxlabel("La altura n es información acumulada, no tiempo. El beneficio mostrado no incluye la continuación después del final.",
                  fontsize=9, color="#52616B")
    return _save(fig, directory, "trajectories_p0_comparison")


def _initial_value_and_policy(rows: list[dict], directory: Path) -> Path:
    frame = pd.DataFrame(rows).sort_values("p0")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), layout="constrained")
    p0 = frame["p0"].to_numpy()
    value = frame["value"].to_numpy()
    axes[0].plot(p0, value, color="#72808B", linewidth=1.4, zorder=2)
    for row in frame.to_dict("records"):
        color = COLORS[row["origin_action"]]
        axes[0].scatter(row["p0"], row["value"], s=75, facecolor=color,
                        edgecolor="white", linewidth=1.1, zorder=3)
        axes[0].annotate(f"{row['value']:.2f}", (row["p0"], row["value"]),
                         xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
    axes[0].set(title="Valor óptimo desde (S,F)=(0,0)", xlabel="Umbral de demanda, p₀",
                ylabel="V(0,0): beneficio descontado de horizonte infinito")
    axes[0].margins(y=.18)
    present = set(frame["origin_action"].astype(int))
    labels = {0: "Acción no resuelta", 1: "Inicio con producto 1", 2: "Inicio con producto 2"}
    axes[0].legend(handles=[Line2D([], [], marker="o", linestyle="none", color=COLORS[i],
                                 label=labels[i]) for i in (1, 2, 0) if i in present],
                   loc="upper right", fontsize=9)
    fractions = 100 * frame["product2_grid_fraction"].to_numpy()
    axes[1].bar(p0, fractions, width=.11, color=COLORS[2], edgecolor="white", linewidth=.8)
    for x, fraction in zip(p0, fractions):
        axes[1].annotate(f"{fraction:.1f}%", (x, fraction), xytext=(0, 5),
                         textcoords="offset points", ha="center", fontsize=9)
    axes[1].set(title="Nodos de la malla que eligen producto 2", xlabel="Umbral de demanda, p₀",
                ylabel="Fracción de nodos de (S,F), en %", ylim=(0, max(5, max(fractions) * 1.2)))
    for ax in axes:
        ax.set_xticks(p0)
        ax.set_xlim(float(min(p0)) - .1, float(max(p0)) + .1)
        ax.grid(axis="y", alpha=.18, linewidth=.6)
        ax.set_axisbelow(True)
    h = rows[0]["h"]
    report_n = rows[0]["report_n"]
    fig.suptitle("Cómo cambian el valor y la elección del producto al variar p₀",
                 fontsize=15, fontweight="semibold")
    fig.supxlabel(f"Mismo interior S+F≤{report_n:g} y paso h={h:g}. La fracción de nodos no es una probabilidad ni una frecuencia de uso.\n"
                  "El valor conserva el horizonte infinito; las trayectorias tienen un tiempo final finito.",
                  fontsize=9, color="#52616B")
    return _save(fig, directory, "initial_value_and_policy")


def make_comparison_figures(case_directories: list[Path], output_dir: Path) -> list[Path]:
    """Compare up to five p0 cases, each in the standard experiment layout.

    The intended sweep is ``[.1, .3, .5, .7, .9]``.  The common initial state
    ``(0,0)`` must be present.  Equal reported domains and mesh steps are
    required so the displayed policy areas have the same interpretation.
    Returns paths to the four written PNGs.
    """
    if not case_directories or len(case_directories) > 5:
        raise ValueError("Provide between one and five case directories")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    cases = sorted((_case_info(Path(path)) for path in case_directories), key=lambda case: case["p0"])
    if len({case["p0"] for case in cases}) != len(cases):
        raise ValueError("Each case must use a distinct p0")
    results = []
    rows = []
    common_grid = None
    with plt.rc_context(STYLE):
        nx_fig, nx_axes = plt.subplots(2, 3, figsize=(16.3, 9.9), layout="constrained")
        sf_fig, sf_axes = plt.subplots(2, 3, figsize=(16.3, 10.2), layout="constrained")
        try:
            for index, case in enumerate(cases):
                solution = load_solution(case["directory"] / "data" / "solution.npz")
                grid = (solution.h, solution.report_n)
                if common_grid is not None and not np.allclose(grid, common_grid, rtol=0, atol=1e-9):
                    raise ValueError("Policy comparisons require equal h and report_n")
                common_grid = grid
                nx_ax, sf_ax = nx_axes.flat[index], sf_axes.flat[index]
                _draw_nx(nx_ax, solution)
                cmap, norm = _policy_cmap()
                _sf(sf_ax, solution, _square(solution.actions, max_points=COMPARISON_MAX_POINTS),
                    cmap=cmap, norm=norm, max_points=COMPARISON_MAX_POINTS)
                for ax, nx in ((nx_ax, True), (sf_ax, False)):
                    _draw_paths(ax, case["paths"], nx=nx, report_n=solution.report_n)
                    ax.set_title(f"p₀ = {case['p0']:g}\n{_time_label(case['paths'])}", fontsize=11)
                states = sum(len(row) for row in solution.actions)
                product2_states = sum(int(np.count_nonzero(row == 2)) for row in solution.actions)
                rows.append({"p0": case["p0"], "value": float(solution.values[0][0]),
                             "origin_action": int(solution.actions[0][0]),
                             "product2_grid_fraction": product2_states / states,
                             "h": solution.h, "report_n": solution.report_n})
                if index == 0:
                    p = solution.params
                    primitives = f"p₁={p.p1:g} · p₂={p.p2:g} · c₁={p.c1:g} · c₂={p.c2:g} · R={p.revenue:g} · δ={p.delta:.6g}"
                del solution
            for index in range(len(cases), 5):
                nx_axes.flat[index].set_axis_off()
                sf_axes.flat[index].set_axis_off()
            _explain_panel(nx_axes.flat[5], cases[0]["paths"], nx=True)
            _explain_panel(sf_axes.flat[5], cases[0]["paths"], nx=False)
            nx_fig.suptitle("Política óptima al cambiar el umbral de demanda p₀\n"
                            "Misma media posterior, distinta información acumulada", fontsize=16, fontweight="semibold")
            sf_fig.suptitle("Producto óptimo para cada estado (S,F) y cada p₀", fontsize=16, fontweight="semibold")
            graphical_sampling = (
                "Visualización submuestreada de nodos originales; "
                if round(common_grid[1] / common_grid[0]) + 1 > COMPARISON_MAX_POINTS
                else ""
            )
            for fig in (nx_fig, sf_fig):
                fig.supxlabel(f"{primitives} · h de cálculo={common_grid[0]:g} · interior n≤{common_grid[1]:g}\n"
                              f"{graphical_sampling}cada panel usa su propio p₀ y las mismas escalas.",
                              fontsize=9, color="#52616B")
            results.append(_save(nx_fig, directory, "policy_nx_comparison"))
            results.append(_save(sf_fig, directory, "policy_sf_comparison"))
        finally:
            plt.close(nx_fig)
            plt.close(sf_fig)
        results.append(_initial_value_and_policy(rows, directory))
        results.append(_comparison_trajectories(cases, directory))
    return results
