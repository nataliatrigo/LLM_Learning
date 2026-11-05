"""Summarize the independent uniform-random-replacement experiment."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .run_experiment import ROOT, CONFIG, load_solution, slug, initial_state
from .simulation import _mean_interval, _ratio_interval


def raw_interval(values):
    values = np.asarray(values,dtype=float)
    mean = float(values.mean())
    se = float(values.std(ddof=1)/np.sqrt(len(values)))
    return mean,se,mean-1.959963984540054*se,mean+1.959963984540054*se


def main():
    config = json.loads(CONFIG.read_text())
    rows = []
    for path in sorted((ROOT / "data" / "calendar").glob("*/path_summary.csv")):
        frame = pd.read_csv(path)
        p0,n = float(frame.p0.iloc[0]),int(frame.n.iloc[0])
        a,p2 = frame.a_chosen_count.to_numpy(),frame.product2_count.to_numpy()
        row = {"n":n,"p0":p0,"method":frame.method.iloc[0]}
        for name,values in (("a_market_share",a/config["calendar"]["max_periods"]),
                            ("product2_unconditional_rate",p2/config["calendar"]["max_periods"])):
            for suffix,value in zip(("","_se","_ci95_lower","_ci95_upper"),_mean_interval(values)):
                row[name+suffix] = value
        for suffix,value in zip(("","_se","_ci95_lower","_ci95_upper"),_ratio_interval(p2,a)):
            row["product2_given_a_rate"+suffix] = value
        mean,se,lower,upper = raw_interval(frame.discounted_profit)
        row.update(discounted_profit=mean,discounted_profit_se=se,
                   discounted_profit_ci95_lower=lower,discounted_profit_ci95_upper=upper)
        solution = load_solution(ROOT / "data" / "policies" / f"{slug(p0,n)}.npz")
        state = initial_state(config,n)
        expected = float(solution.values[state["m"]][state["S"]])
        row.update(bellman_initial_value=expected,profit_difference_from_bellman=mean-expected,
                   profit_z_score=(mean-expected)/se if se else 0.0)
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["p0","n"])
    table.to_csv(ROOT / "data" / "calendar_summary.csv",index=False)
    intervals = pd.read_csv(ROOT / "data" / "policy_intervals.csv")
    diagnostics = pd.read_csv(ROOT / "data" / "solver_diagnostics.csv")
    long = pd.read_csv(ROOT / "data" / "long_interactions" / "path_summary.csv")
    mean,se,lower,upper = _mean_interval(long.post_burn_in_frequency)
    case = table[(table.p0 == .5)&(table.n == 20)].iloc[0]
    initialization = "vacía, acumulando hasta n" if config["initialization"] == "empty" else "llena con S=⌊n/2⌋ y F=n−S"
    lines = ["# Experimento nuevo: reemplazo aleatorio uniforme", "",
        "Este experimento independiente está en `forgetting_random_replacement/`. "
        "Los resultados del reemplazo opuesto permanecen en `forgettin_fixed/`.", "",
        "## Regla", "",
        "Con la memoria llena se elige uniformemente una de sus n observaciones y se sustituye "
        "por el resultado nuevo. El tipo de dato eliminado se sortea independientemente del resultado nuevo:", "",
        r"\[R\sim\operatorname{Bernoulli}(S/n),\quad Y\sim\operatorname{Bernoulli}(p_i),\quad S'=S+Y-R.\]", "",
        "Por tanto, con n=20 y S=18, un fracaso nuevo deja S=17 con probabilidad 18/20 y "
        "S=18 con probabilidad 2/20, exactamente como en el ejemplo solicitado. También puede "
        "eliminarse un éxito al llegar otro éxito: S entonces no cambia.", "",
        f"La simulación inicia con memoria **{initialization}**. Si comienza vacía, se agregan "
        "los primeros n resultados sin eliminar ninguno; después la memoria conserva exactamente n "
        "observaciones. La opción externa no modifica el estado.", "",
        "La regla mantiene conteos enteros y no conserva el orden cronológico. En madurez, "
        "E[S′|S,i]=(1−1/n)S+pᵢ. Así, para n=20, la retención esperada es 0.95. Esta igualdad "
        "de medias no implica las mismas políticas que el forgetting geométrico, pues la demanda "
        "y el valor de continuación son no lineales.", "",
        "## Políticas óptimas", "",
        "Todas las capacidades se resuelven directamente como MDP finitos de conteos. "
        "Se integra el riesgo de reemplazo al calcular Q₁ y Q₂, antes de elegir el producto. "
        "No se impone que la política sea una banda; se detectan sus intervalos después de resolver.", "",
        "Para p₀=0.5, los intervalos inclusivos de S donde se elige producto 2 son:", "",
        "| n | Producto 2 en madurez |", "|---|---|"]
    for n in sorted(intervals[intervals.p0 == .5].n.unique()):
        subset = intervals[(intervals.p0 == .5)&(intervals.n == n)]
        bands = [f"{int(r.lower_S)}–{int(r.upper_S)}" for _,r in subset.iterrows() if r.product2_active]
        lines.append(f"| {int(n)} | {', '.join(bands) if bands else 'Ningún estado'} |")
    lines.extend(["", "Para **n=20, p₀=0.5**, la banda es **S=5–12**. "
        "El experimento de reemplazo opuesto daba S=4–14: son modelos diferentes.", "",
        "![Política y ventaja, n=20](figures/exact_policy_detail_p0_0.5_n_20.png)", "",
        "![Políticas de todas las capacidades](figures/optimal_policies_by_count.png)", "",
        "## Simulaciones", "",
        "Se recalcularon 35 escenarios calendario: cinco p₀ y siete capacidades, con 20.000 "
        "consumidores × 3.000 períodos cada uno. Se guardaron 11 escenarios de 20 historias × "
        "20.000 interacciones y un diagnóstico largo de 100 historias × 200.000 interacciones. "
        "Se mantienen c₂=0.65, c₁=0.05, p₁=0.35, p₂=0.8, R=1 y γ=0.98.", "",
        f"Para n=20 y p₀=0.5, después del burn-in de 20.000 interacciones, la frecuencia "
        f"media de producto 2 es **{100*mean:.3f}%** (IC 95% entre consumidores: "
        f"{100*lower:.3f}%–{100*upper:.3f}%). El promedio de éxitos recordados es "
        f"{long.mean_remembered_successes.mean():.4f}.", "",
        "![Trayectoria individual](figures/individual_states/individual_states_p0_0.5_n_20.png)", "",
        "![Monte Carlo por capacidad](figures/monte_carlo_by_window.png)", "",
        "Cada histograma utiliza al consumidor 1 e incluye el transiente. Los registros guardan "
        "el estado anterior a elegir y el tipo de observación eliminada, de modo que es posible "
        "reconstruir todos los estados. Los intervalos Monte Carlo usan al consumidor como clúster "
        "y no describen incertidumbre de parámetros ni prueban estacionariedad.", "",
        "## Comprobación del beneficio descontado", "",
        f"Para n=20,p₀=0.5, el valor de Bellman del estado inicial es "
        f"**{case.bellman_initial_value:.5f}**. El beneficio Monte Carlo medio es "
        f"**{case.discounted_profit:.5f}**, con IC 95% "
        f"[{case.discounted_profit_ci95_lower:.5f}, {case.discounted_profit_ci95_upper:.5f}]. "
        "La tabla completa permite comprobar esta comparación para cada escenario.", "",
        "[Resumen calendario con intervalos y beneficios](data/calendar_summary.csv) · "
        "[Transiciones condicionadas al resultado nuevo](data/conditional_transitions.csv) · "
        "[Diagnósticos](data/solver_diagnostics.csv) · [Auditoría](data/validation.json)", "",
        "## Comparación descriptiva con los otros modelos", "",
        "![Reemplazo opuesto frente a aleatorio](figures/opposite_vs_random_replacement_p0_0.5_n_20.png)", "",
        "![Referencia geométrica frente a aleatorio](figures/geometric_vs_random_replacement_p0_0.5_rho_0.95_n_20.png)", "",
        "Las trayectorias de modelos distintos no están emparejadas: el modelo aleatorio requiere "
        "un sorteo adicional para eliminar la observación. Dentro del experimento nuevo, las "
        "capacidades comparten uniformes a igual p₀.", "",
        "## Reproducción", "", f"Se resolvieron {len(diagnostics)} calibraciones exactas por conteos. "
        "El método, la inicialización y los comandos están documentados en [README.md](README.md)."])
    (ROOT / "RESULTADOS.md").write_text("\n".join(lines)+"\n")
    print(f"Wrote {ROOT / 'RESULTADOS.md'}",flush=True)


if __name__ == "__main__":
    main()
