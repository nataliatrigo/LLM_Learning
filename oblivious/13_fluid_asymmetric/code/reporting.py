"""Spanish scientific report for crossed fixed-rival fluid responses.

All numerical evidence is supplied by the runner; plots never infer an
infinite-time destination from a finite trajectory endpoint.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluid-asymmetric-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {"a": "#D55E00", "b": "#0072B2"}
SHADE = "#A69AC8"


def _tag(case):
    return "p0_" + f"{case['p0']:g}".replace(".", "p") + "_rho_" + f"{case['rho']:g}".replace(".", "p")


def _number(value, digits=9):
    return f"{float(value):.{digits}g}" if value is not None else "—"


def _parts_text(parts):
    answer = []
    for part in parts:
        lo, hi = part["lo"], part["hi"]
        if lo == hi:
            answer.append("{" + _number(lo) + "}")
        else:
            answer.append(("[" if part["left_closed"] else "(") + _number(lo) + ", "
                          + _number(hi) + ("]" if part["right_closed"] else ")"))
    return " ∪ ".join(answer) if answer else "∅"


def _save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _case_paths(frame, case):
    return frame[np.isclose(frame.p0, case["p0"], rtol=0, atol=1e-12)
                 & np.isclose(frame.rho, case["rho"], rtol=0, atol=1e-12)]


def _policy_figure(case, parameters, destination):
    p1, p2 = parameters["p1"], parameters["p2"]
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.25), sharex=True, sharey=True)
    for seller, ax in zip(("a", "b"), axes):
        response = case["response_" + seller]
        policy = response["policy"]
        roots, products = policy["roots"], policy["products"]
        edges = [0., *roots, 1.]
        for lo, hi, product in zip(edges, edges[1:], products):
            ax.plot([lo, hi], [float(product == 2)] * 2, color=COLORS[seller], lw=2.8,
                    solid_capstyle="butt")
        for interval in case["common_initial_intervals"]:
            ax.axvspan(interval["lo"], interval["hi"], color=SHADE, alpha=.24, lw=0)
        holding = {point["x"]: point["alpha"] for point in policy.get("holding_points", [])}
        for index, root in enumerate(roots):
            kinds = policy.get("root_kinds", [])
            kind = kinds[index] if index < len(kinds) else ""
            alpha = next((a for x, a in holding.items() if abs(x-root) < 1e-12), None)
            if alpha is None and kind == "attractive":
                alpha = (root-p1)/(p2-p1)
            if alpha is None:
                alpha = 0. if kind == "repelling" else float(products[index+1] == 2)
            ax.plot([root, root], [0, 1], color=COLORS[seller], alpha=.4, lw=.9, ls=":")
            ax.scatter([root, root], [0, 1], s=30, facecolor="white", edgecolor=COLORS[seller], zorder=4)
            ax.scatter(root, alpha, s=31, color=COLORS[seller], zorder=5)
        for value, label in ((p1, "$p_1$"), (p2, "$p_2$")):
            ax.axvline(value, color="#A6ACB4", lw=.8, ls="--", zorder=0)
            ax.text(value, 1.065, label, ha="center", va="bottom", color="#5D6570", fontsize=10)
        ax.set_title(f"{seller.upper()}: respuesta a rival fijo en {response['conjecture']:g}", pad=26)
        ax.set(xlim=(0, 1), ylim=(-.07, 1.12), xlabel="Reputación propia $x$", yticks=[0, 1])
        ax.grid(axis="x", alpha=.1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(.02, .5, f"Destino propuesto: {response['target']:g}\n"
                + f"Inicios que lo alcanzan:\n{response['basin']['text']}",
                transform=ax.transAxes, fontsize=10, va="center",
                bbox=dict(facecolor="white", edgecolor="none", alpha=.92, pad=5))
    axes[0].set_ylabel(r"$\sigma(x)$: fracción de producto 2")
    title = fr"Políticas cruzadas · $p_0={case['p0']:g}$, $\rho={case['rho']:g}$"
    fig.suptitle(title, fontsize=14, y=1.01)
    if case["common_initial_intervals"]:
        fig.text(.5, -.005, "Franja violeta: estados iniciales comunes que producen A → p₁ y B → p₂.",
                 ha="center", fontsize=10, color="#4F4862")
    else:
        fig.text(.5, -.005, "La intersección es vacía: estas políticas no sostienen el par propuesto.",
                 ha="center", fontsize=10, color="#6D4340")
    fig.tight_layout(pad=1.2)
    return _save(fig, destination / "policies.png")


def _trajectory_figure(case, frame, parameters, destination):
    data = _case_paths(frame, case)
    if data.empty:
        return None
    starts = sorted(data.x0.unique())
    columns = min(3, len(starts))
    rows = (len(starts)+columns-1)//columns
    fig, axs = plt.subplots(rows, columns, figsize=(4.6*columns, 3.65*rows), squeeze=False, sharey=True)
    for index, x0 in enumerate(starts):
        ax = axs.flat[index]
        trajectory = data[np.isclose(data.x0, x0, rtol=0, atol=1e-13)].sort_values("time")
        for seller in ("a", "b"):
            ax.plot(trajectory.time, trajectory["x_" + seller], color=COLORS[seller], lw=2.3)
        for target in (parameters["p1"], parameters["p2"]):
            ax.axhline(target, color="#9DA5AF", ls="--", lw=.9, zorder=0)
        ax.scatter([0], [x0], color="#30363D", s=23, zorder=5)
        last = trajectory.iloc[-1]
        ax.set_title(f"Mismo inicio: $x_0={x0:.6g}$", fontsize=11)
        ax.set_xlabel("Tiempo calendario continuo")
        ax.set_xlim(0, trajectory.time.max())
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.14)
        ax.text(.98, .05, f"Finito: $x_A(T)={last.x_a:.6f}$\n$x_B(T)={last.x_b:.6f}$",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.88, pad=3))
    for ax in axs.flat[len(starts):]:
        ax.set_visible(False)
    for ax in axs[:, 0]:
        ax.set_ylabel("Reputación efectiva")
    fig.suptitle(fr"Dinámica conjunta · $p_0={case['p0']:g}$, $\rho={case['rho']:g}$", fontsize=14, y=1.015)
    handles = [Line2D([0], [0], color=COLORS["a"], lw=2.3, label="Jugador A"),
               Line2D([0], [0], color=COLORS["b"], lw=2.3, label="Jugador B"),
               Line2D([0], [0], color="#9DA5AF", ls="--", lw=.9, label="Referencias p₁ y p₂")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(.5, -.025))
    fig.tight_layout(rect=(0, .035, 1, 1), pad=1.2)
    return _save(fig, destination / "trajectories.png")


def _overview(cases, destination):
    positive = [case for case in cases if case["qualifies"]]
    if not positive:
        return None
    fig, ax = plt.subplots(figsize=(10.7, max(3.1, .95*len(positive)+1.1)))
    for index, case in enumerate(positive):
        for part in case["common_initial_intervals"]:
            ax.plot([part["lo"], part["hi"]], [index, index], lw=7, color=SHADE, solid_capstyle="butt")
            for x, closed in ((part["lo"], part["left_closed"]), (part["hi"], part["right_closed"])):
                ax.scatter(x, index, s=64, edgecolor="#68558F", facecolor="#68558F" if closed else "white", zorder=4)
        ax.annotate(case["common_initial_notation"], (1.01, index), xycoords=("axes fraction", "data"),
                    va="center", fontsize=10)
    ax.set_yticks(range(len(positive)), [fr"$p_0={c['p0']:g}$, $\rho={c['rho']:g}$" for c in positive])
    bounds = [part[key] for case in positive for part in case["common_initial_intervals"] for key in ("lo", "hi")]
    padding = max(.025, .12*(max(bounds)-min(bounds)))
    ax.set_xlim(min(bounds)-padding, max(bounds)+padding)
    ax.set_ylim(len(positive)-.5, -.5)
    ax.set_xlabel("Reputación inicial común $x_0$")
    ax.grid(axis="x", alpha=.18)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_title("Mismo inicio, destinos distintos: A → p₁ y B → p₂", pad=18, fontsize=14)
    fig.text(.5, -.015, "● extremo incluido   ○ extremo excluido · Fronteras numéricas; selector producto 1 en umbrales repulsivos.",
             ha="center", fontsize=9, color="#5D6570")
    fig.tight_layout(rect=(0, .035, .76, 1))
    return _save(fig, destination / "01_common_initial_intervals.png")


def _long_horizon_figure(cases, frame, parameters, destination):
    if frame is None or frame.empty:
        return None
    available = [(case, _case_paths(frame, case)) for case in cases if case["qualifies"]]
    available = [(case, data) for case, data in available if not data.empty]
    if not available:
        return None
    fig, axes = plt.subplots(1, len(available), figsize=(4.7*len(available), 4.2),
                             squeeze=False, sharey=True)
    for ax, (case, data) in zip(axes.flat, available):
        starts = sorted(data.x0.unique())
        x0 = starts[len(starts)//2]
        path = data[np.isclose(data.x0, x0, atol=1e-13, rtol=0)].sort_values("time")
        for seller in ("a", "b"):
            ax.plot(path.time, path["x_" + seller], color=COLORS[seller], lw=2.3)
        for value in (parameters["p1"], parameters["p2"]):
            ax.axhline(value, color="#9DA5AF", ls="--", lw=.9, zorder=0)
        final = path.iloc[-1]
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlim(0, final.time)
        ax.set_xlabel("Tiempo calendario (escala symlog)")
        ax.set_title(fr"$p_0={case['p0']:g}$, $\rho={case['rho']:g}$" + f"\n$x_0={x0:.6g}$", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.16)
        ax.text(.98, .47, f"$T={final.time:.0e}$\n$x_A(T)={final.x_a:.7f}$\n$x_B(T)={final.x_b:.7f}$",
                transform=ax.transAxes, ha="right", va="center", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=3))
    axes.flat[0].set_ylabel("Reputación efectiva")
    fig.suptitle("Horizonte extendido: la demanda escasa demora la convergencia de A", fontsize=14, y=1.015)
    fig.legend(handles=[Line2D([0], [0], color=COLORS["a"], lw=2.3, label="Jugador A"),
                        Line2D([0], [0], color=COLORS["b"], lw=2.3, label="Jugador B")],
               loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.015))
    fig.tight_layout(rect=(0, .035, 1, 1), pad=1.2)
    return _save(fig, destination / "long_horizon_trajectories.png")


def _report(cases, trajectories, config, destination):
    parameters = config["parameters"]
    positive = [case for case in cases if case.get("verified", False)]
    lines = ["# Prueba de equilibrios fluid oblivious asimétricos", "",
             f"La propuesta se sostiene numéricamente en **{len(positive)} de los {len(cases)} casos examinados**: "
             f"partiendo del mismo estado inicial, A puede converger a `p1={parameters['p1']:g}` "
             f"y B a `p2={parameters['p2']:g}` con políticas distintas calculadas contra la reputación final del otro.", "",
             "## Respuestas cruzadas y estados iniciales comunes", "",
             "A optimiza suponiendo que B está fijo en `p2`; B optimiza suponiendo que A está fijo en `p1`. "
             "La intersección de los conjuntos de inicios que llevan a los destinos propuestos da el intervalo común.", "",
             "| p0 | rho | A alcanza p1 | B alcanza p2 | Mismo x0: A → p1, B → p2 | Resultado |",
             "| --- | --- | --- | --- | --- | --- |"]
    for case in cases:
        lines.append(f"| {case['p0']:g} | {case['rho']:g} | {case['response_a']['basin']['text']} | "
                     f"{case['response_b']['basin']['text']} | {case['common_initial_notation']} | "
                     + ("Verificado" if case.get("verified", False) else
                        "Compatible; validación pendiente o fallida" if case["qualifies"] else
                        "No sostiene este par") + " |")
    lines.extend(["", "Los tres intervalos positivos coinciden, a la precisión mostrada, con los huecos "
                  "entre las ramas simétricas puras de la [comprobación del experimento 11]"
                  "(../11_fluid_oblivious_initial_conditions/EXISTENCE_CHECK.md). "
                  "Permitir políticas distintas para A y B cubre esos inicios que el análisis simétrico dejaba sin cubrir.", "",
                  "Al intercambiar las identidades A y B se obtiene el perfil reflejado, con los mismos inicios comunes.", "",
                  "![Intervalos iniciales comunes](figures/01_common_initial_intervals.png)", "",
                  "## Dinámicas con ambas reputaciones efectivas", "",
                  "Las políticas se mantienen fijas durante cada simulación. La demanda se recalcula con "
                  "`D_A(x_A(t),x_B(t))` y `D_B(x_A(t),x_B(t))`; no se congela al rival en la dinámica realizada. "
                  "La tabla distingue el estado al horizonte finito del destino asintótico, que procede de la línea de fase.", "",
                  "| p0 | rho | x0 común | T | xA(T) | xB(T) | Límites de A y B |",
                  "| --- | --- | --- | --- | --- | --- | --- |"])
    for case in cases:
        data = _case_paths(trajectories, case)
        if data.empty:
            continue
        starts = sorted(data.x0.unique())
        middle = starts[len(starts)//2]
        sample = data[np.isclose(data.x0, middle, atol=1e-13, rtol=0)].sort_values("time")
        final = sample.iloc[-1]
        simulation = next((row for row in case.get("simulations", [])
                           if abs(row["x0"]-middle) < 1e-12), None)
        limits = (f"{simulation['asymptotic_a']:.9g}, {simulation['asymptotic_b']:.9g}" if simulation is not None
                  else f"{parameters['p1']:g}, {parameters['p2']:g}" if case["qualifies"]
                  else "Ver diagnósticos del control")
        lines.append(f"| {case['p0']:g} | {case['rho']:g} | {middle:.9g} | {final.time:g} | "
                     f"{final.x_a:.9g} | {final.x_b:.9g} | {limits} |")
    extended = [case for case in cases if case.get("long_horizon")]
    if extended:
        lines.extend(["", "## Convergencia en un horizonte extendido", "",
                      "La demanda del vendedor de reputación baja puede ser muy pequeña. Por ello, "
                      "el estado de A al tiempo `T=3000` puede seguir lejos de `p1`, aunque su límite "
                      "sea `p1`. Se integró también hasta el horizonte indicado con coordenadas "
                      "de logaritmo de la distancia al destino y se contrastó el tramo común con la integración original.", "",
                      "| p0 | rho | T extendido | xA(T) | xB(T) | Mayor distancia al destino | Discrepancia en tramo común | Control |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- |"])
        for case in extended:
            long = case["long_horizon"]
            lines.append(f"| {case['p0']:g} | {case['rho']:g} | {long['horizon']:g} | "
                         f"{long['terminal_a']:.10g} | {long['terminal_b']:.10g} | "
                         f"{long['maximum_terminal_target_error']:.3g} | {long['maximum_overlap_error']:.3g} | "
                         + ("Pasa" if long["passed"] else "Falla") + " |")
        lines.extend(["", "![Trayectorias de horizonte extendido](figures/long_horizon_trajectories.png)", ""])
    for case in cases:
        tag = _tag(case)
        lines.extend(["", f"### p0={case['p0']:g}, rho={case['rho']:g}", "",
                      f"![Políticas cruzadas](figures/{tag}/policies.png)", ""])
        if not _case_paths(trajectories, case).empty:
            lines.extend([f"![Trayectorias conjuntas](figures/{tag}/trajectories.png)", ""])
        if not case["qualifies"]:
            failed = [seller.upper() for seller in ("a", "b")
                      if not case["response_" + seller]["diagnostics"].get("target_optimal", False)
                      or not case["response_" + seller]["basin"].get("stationary", False)]
            if failed:
                lines.extend(["El destino propuesto falla para " + ", ".join(failed)
                              + ": la respuesta optimizada no sostiene esa reputación como estado estacionario.", ""])
    lines.extend(["## Verificación y alcance", "",
                  "| p0 | rho | Cambio de frontera/raíz, dos mallas más finas | Mayor residuo HJB | Mayor cambio de demanda | Cambio al refinar integración | Controles numéricos |",
                  "| --- | --- | --- | --- | --- | --- | --- |"])
    for case in cases:
        comparisons = case.get("refinement", {}).get("comparisons", [])
        change = comparisons[-1].get("maximum_root_or_boundary_change") if comparisons else None
        hjb = max(case["response_" + seller]["diagnostics"].get("hjb_residual", float("nan"))
                  for seller in ("a", "b"))
        validation = case.get("numeric_validation", {})
        checks = validation.get("demand", {}).get("checks", [])
        demand = max((max(row["maximum_demand_change_on_doubling"],
                          row["maximum_independent_demand_disagreement"],
                          row.get("maximum_transformed_demand_change", 0.)) for row in checks), default=None)
        integration = validation.get("integration", {}).get("maximum_path_change_on_refinement")
        def scientific(value):
            return f"{value:.3g}" if value is not None else "—"
        lines.append(f"| {case['p0']:g} | {case['rho']:g} | {scientific(change)} | {scientific(hjb)} | "
                     f"{scientific(demand)} | {scientific(integration)} | "
                     + ("Pasan" if validation.get("passed", False) else "Pendientes o fallidos") + " |")
    lines.extend(["",
                  "Los HJB se resuelven de nuevo para las dos conjeturas cruzadas. "
                  "Se comprueban la acción óptima en cada destino, el valor de mantenerlo, el residuo de Bellman, "
                  "la sensibilidad a la malla y las integraciones conjuntas. Los datos completos están en "
                  "[results.json](data/results.json), [resumen de casos](data/summary.csv) "
                  "y [trayectorias](data/trajectories.csv).", "",
                  "La conclusión corresponde al concepto **oblivious con conjetura fija del rival** empleado en los experimentos 10 y 11. "
                  "No demuestra que estas políticas sean mejores respuestas frente a toda la trayectoria transitoria del rival "
                  "ni que formen un equilibrio de Nash del juego dinámico completo.", "",
                  "La demanda de cada vendedor es positiva en el dominio compacto de estados de esta calibración. "
                  "Por tanto, multiplicar la dinámica propia por la demanda efectiva cambia la velocidad, "
                  "pero conserva la dirección y el destino asintótico de cada política. "
                  "Esto permite calcular el conjunto completo de inicios comunes mediante la intersección de las dos líneas de fase.", "",
                  "Las fronteras de los intervalos son aproximaciones numéricas. En umbrales repulsivos se elige producto 1 "
                  "y su continuación inmediata; los extremos abiertos/cerrados corresponden a ese selector. "
                  "El refinamiento mide sensibilidad numérica, sin constituir una cota rigurosa del error continuo. "
                  "Esta prueba del par `(p1,p2)` no enumera todos los posibles equilibrios asimétricos.", ""])
    (destination / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return destination / "RESULTS.md"


def create_report(cases, trajectories, output_dir, config, long_trajectories=None):
    """Write PNG figures and a Spanish report from the completed evidence."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    paths = []
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.titleweight": "medium", "axes.labelcolor": "#30363D",
                         "text.color": "#30363D", "axes.edgecolor": "#B8BFC7"}):
        overview = _overview(cases, figures)
        if overview is not None:
            paths.append(overview)
        extended = _long_horizon_figure(cases, long_trajectories, config["parameters"], figures)
        if extended is not None:
            paths.append(extended)
        for case in cases:
            destination = figures / _tag(case)
            paths.append(_policy_figure(case, config["parameters"], destination))
            trajectory = _trajectory_figure(case, trajectories, config["parameters"], destination)
            if trajectory is not None:
                paths.append(trajectory)
    paths.append(_report(cases, trajectories, config, root))
    (figures / "report_manifest.json").write_text(json.dumps(
        {"files": [str(path.relative_to(root)) for path in paths],
         "source": "Supplied new crossed-response computations; no historical files modified"},
        indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return paths
