"""Build a Spanish results note directly from saved simulation outputs."""
from __future__ import annotations

import json
from statistics import NormalDist

import numpy as np
import pandas as pd

from .run_experiment import ROOT
from .simulation import _mean_interval, _ratio_interval


def main():
    summary = []
    for path in sorted((ROOT / "data" / "calendar").glob("*/path_summary.csv")):
        frame = pd.read_csv(path)
        a, p2 = frame.a_chosen_count.to_numpy(), frame.product2_count.to_numpy()
        row = {"n": int(frame.n.iloc[0]), "p0": float(frame.p0.iloc[0]),
               "method": frame.method.iloc[0]}
        for name, values in (("a_market_share", a / 3000), ("product2_unconditional_rate", p2 / 3000)):
            for suffix, value in zip(("", "_se", "_ci95_lower", "_ci95_upper"), _mean_interval(values)):
                row[name + suffix] = value
        for suffix, value in zip(("", "_se", "_ci95_lower", "_ci95_upper"), _ratio_interval(p2, a)):
            row["product2_given_a_rate" + suffix] = value
        summary.append(row)
    table = pd.DataFrame(summary).sort_values(["p0", "n"])
    table.to_csv(ROOT / "data" / "calendar_summary.csv", index=False)
    states = pd.read_csv(ROOT / "data" / "policy_by_count.csv")
    exact = states[(states.p0 == .5) & (states.n == 20)]
    ref = pd.read_csv(ROOT / "data" / "refinement" / "state_comparison.csv")
    refinement = ref.groupby(["p0", "n"], as_index=False).agg(
        maximum_sampled_disagreement=("action_disagreement_share", "max"))
    refinement.to_csv(ROOT / "data" / "refinement" / "summary.csv", index=False)
    long = pd.read_csv(ROOT / "data" / "long_interactions" / "path_summary.csv")
    mean, se, lower, upper = _mean_interval(long.post_burn_in_frequency.to_numpy())
    diagnostics = pd.read_csv(ROOT / "data" / "solver_diagnostics.csv")
    n1000 = diagnostics[(diagnostics.n == 1000) & (diagnostics.p0 == .5)].iloc[0]
    paired = json.loads((ROOT / "data" / "refinement" / "paired_calendar_summary.json").read_text())
    rows = ["# Resultados: monopolio con ventana fija", "",
        "Se ejecutaron 35 escenarios de 20.000 consumidores × 3.000 períodos calendario, "
        "11 escenarios de 20 historias × 20.000 interacciones y el diagnóstico de 100 historias × "
        "200.000 interacciones. En todos los casos c₂=0.65 y n es el entero más cercano a 1/(1−ρ).", "",
        "**Las políticas son soluciones del MDP FIFO completo para n≤20. Las de n=33,100,1000 son "
        "aproximadas: las cotas globales en madurez no permiten certificarlas como cercanas a la óptima.** "
        "Todos los consumidores simulados, también en esos casos, conservan y expulsan resultados mediante FIFO real.", "",
        "## Caso principal: p₀=0.5, n=20", "",
        "Este es el equivalente al gráfico adjunto con ρ=0.95. El solver enumera 1.048.576 historias "
        "maduras, además de las historias transitorias. En las cinco calibraciones de p₀, todas las "
        "acciones exactas se distinguen de la indiferencia por más que la cota numérica de error.", "",
        f"En el experimento largo, después de las primeras 20.000 interacciones, la frecuencia del "
        f"producto 2 es **{100*mean:.3f}%** (IC 95% entre consumidores: "
        f"{100*lower:.3f}%–{100*upper:.3f}%). El promedio de éxitos recordados es "
        f"{long.mean_remembered_successes.mean():.4f}. El intervalo describe incertidumbre Monte Carlo; "
        "no es una afirmación de estacionariedad o tiempo de mezcla.", "",
        "La decisión madura no se resume con un único umbral en el conteo S:", "",
        "| Éxitos S en los 20 resultados | Política exacta |", "|---|---|"]
    classes = []
    for _, row in exact.sort_values("S").iterrows():
        label = "Producto 2 en todas las historias" if row.action_min == 1 else (
            "Producto 1 en todas las historias" if row.action_max == 0 else "Depende del orden de los resultados")
        classes.append((int(row.S), label))
    start, end, previous = classes[0][0], classes[0][0], classes[0][1]
    for s, label in classes[1:] + [(None, None)]:
        if label != previous:
            rows.append(f"| {start if start == end else f'{start}–{end}'} | {previous} |")
            start, previous = s, label
        end = s
    rows.extend(["", "Las tablas por S proyectan sobre todos los órdenes posibles, sin ponderarlos por "
        "su frecuencia de visita. El mapa de historias completas y las tablas NPZ permiten consultar la acción de cada historia.", "",
        "![Trayectoria FIFO equivalente al adjunto](figures/individual_states/individual_states_p0_0.5_n_20.png)", "",
        "![Referencia geométrica frente a FIFO](figures/geometric_vs_fixed_p0_0.5_rho_0.95_n_20.png)", "",
        "Los dos paneles son una comparación descriptiva con la referencia original guardada. "
        "Las trayectorias de esa figura no están emparejadas por sus números aleatorios.", "",
        "## Monte Carlo calendario", "",
        "Promedios de los 3.000 períodos para p₀=0.5. La última columna condiciona en comprar A; "
        "la columna anterior divide por todos los períodos de todos los consumidores.", "",
        "| n | Demanda de A | Producto 2, incondicional | Producto 2 dado A | Política |",
        "|---|---:|---:|---:|---|"])
    for _, row in table[table.p0 == .5].iterrows():
        method = "Exacta" if row.method == "exact_fifo" else "Aproximada"
        rows.append(f"| {int(row.n)} | {100*row.a_market_share:.2f}% | "
                    f"{100*row.product2_unconditional_rate:.2f}% | {100*row.product2_given_a_rate:.2f}% | {method} |")
    rows.extend(["", "[Tabla completa con intervalos al 95% para los cinco p₀](data/calendar_summary.csv).", "",
        "![Monte Carlo de todos los escenarios](figures/monte_carlo_by_window.png)", "",
        "## Precisión de las ventanas grandes", "",
        "Se compararon dos profundidades del planificador sobre historias aleatorias y también "
        "historias ordenadas deliberadamente por éxitos antiguos o recientes. El máximo desacuerdo "
        "entre resoluciones fue **" + f"{100*ref.action_disagreement_share.max():.2f}%** en alguna de "
        "las muestras. Esto no acota el error respecto de la política óptima: dos aproximaciones pueden coincidir y compartir sesgo.", "",
        "Para n=100, p₀=0.5, se ejecutó además un Monte Carlo emparejado de 2.000 consumidores "
        "× 1.000 períodos, con memoria FIFO real en ambas resoluciones:", "",
        "| Diferencia k=12 menos k=10 | Estimación | IC 95% emparejado |", "|---|---:|---:|"])
    for metric, label in (("discounted_profit", "Beneficio descontado"),
                          ("a_market_share", "Cuota de A, en proporción"),
                          ("product2_unconditional_rate", "Producto 2 incondicional, en proporción")):
        record = paired[metric]
        rows.append(f"| {label} | {record['paired_difference']:.6f} | "
                    f"[{record['paired_ci95_lower']:.6f}, {record['paired_ci95_upper']:.6f}] |")
    rows.extend(["", "El beneficio no presenta una diferencia estadísticamente distinguible de cero "
        "en este contraste. Sí se detectan pequeños cambios en las frecuencias: no se afirma igualdad de las políticas.", "",
        "Para n=1000, el descuento γ=0.98 hace que los estados maduros reciban muy poco peso al "
        "comenzar desde memoria vacía. La pérdida de beneficio descontado desde ese estado inicial "
        f"queda acotada por **{n1000.empty_initial_greedy_policy_loss_bound:.3g}**. Esta garantía "
        "inicial no certifica la política después de llenar la ventana.", "",
        "[Diagnósticos y cotas por escenario](data/solver_diagnostics.csv) · "
        "[Historias de validación y refinamientos](data/refinement/state_comparison.csv) · "
        "[Auditoría de resultados](data/validation.json)", "",
        "## Reproducción", "", "Las instrucciones, ecuaciones, semillas y formatos están en "
        "[README.md](README.md). Código en `code/`, datos en `data/` y gráficos en `figures/`. "
        "El experimento original se conserva sin modificaciones."])
    (ROOT / "RESULTADOS.md").write_text("\n".join(rows) + "\n")
    print(f"Wrote {ROOT / 'RESULTADOS.md'}")


if __name__ == "__main__":
    main()
