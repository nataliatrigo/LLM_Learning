"""Build result tables and a Spanish note for opposite-outcome replacement."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .run_experiment import ROOT
from .simulation import _mean_interval, _ratio_interval


def main():
    rows = []
    for path in sorted((ROOT / "data" / "calendar").glob("*/path_summary.csv")):
        frame = pd.read_csv(path)
        a, p2 = frame.a_chosen_count.to_numpy(), frame.product2_count.to_numpy()
        row = {"n": int(frame.n.iloc[0]), "p0": float(frame.p0.iloc[0]),
               "method": frame.method.iloc[0]}
        for name, values in (("a_market_share",a/3000),("product2_unconditional_rate",p2/3000)):
            for suffix,value in zip(("","_se","_ci95_lower","_ci95_upper"),_mean_interval(values)):
                row[name+suffix] = value
        for suffix,value in zip(("","_se","_ci95_lower","_ci95_upper"),_ratio_interval(p2,a)):
            row["product2_given_a_rate"+suffix] = value
        rows.append(row)
    table = pd.DataFrame(rows).sort_values(["p0","n"])
    table.to_csv(ROOT / "data" / "calendar_summary.csv",index=False)
    intervals = pd.read_csv(ROOT / "data" / "policy_intervals.csv")
    diagnostics = pd.read_csv(ROOT / "data" / "solver_diagnostics.csv")
    long = pd.read_csv(ROOT / "data" / "long_interactions" / "path_summary.csv")
    mean,se,lower,upper = _mean_interval(long.post_burn_in_frequency.to_numpy())
    lines = ["# Resultados: reemplazo del resultado opuesto", "",
        "Los gráficos activos de `figures/` y los datos de `data/` corresponden al nuevo experimento. "
        "La versión FIFO anterior se conserva en `archive/fifo_previous/`.", "",
        "**Todas las capacidades se resuelven directamente como MDP de conteos, incluida n=1000. "
        "No se usan aproximaciones de historias ni interpolación.** Los certificados numéricos "
        "y sus tolerancias están en `data/solver_diagnostics.csv`.", "",
        "## Regla de actualización", "",
        "El estado es (m,S), con F=m−S. Antes de completar la capacidad, m aumenta en uno y S "
        "aumenta sólo si el resultado es éxito. Al completar n resultados:", "",
        r"\[S'=\min(n,S+1)\text{ ante éxito},\qquad S'=\max(0,S-1)\text{ ante fracaso}.\]", "",
        "El resultado nuevo reemplaza uno del tipo opuesto; si no existe uno, S queda saturado "
        "en el borde. Elegir la opción externa no actualiza el estado. n es una capacidad de "
        "memoria, no una ventana de los últimos n resultados cronológicos.", "",
        "Se mantienen c₂=0.65, p₁=0.35, p₂=0.8, c₁=0.05, R=1 y γ=0.98. La equivalencia "
        "n≈1/(1−ρ) conserva la capacidad de evidencia y usa redondeo al entero más cercano, con empates hacia arriba.", "",
        "## Política óptima en madurez", "",
        "Para p₀=0.5, los intervalos inclusivos de éxitos donde se elige producto 2 son:", "",
        "| Capacidad n | S donde se elige producto 2 |", "|---|---|"]
    for n in sorted(intervals[intervals.p0 == .5].n.unique()):
        subset = intervals[(intervals.p0 == .5)&(intervals.n == n)]
        bands = [f"{int(r.lower_S)}–{int(r.upper_S)}" for _,r in subset.iterrows() if r.product2_active]
        lines.append(f"| {int(n)} | {', '.join(bands) if bands else 'Ninguno'} |")
    lines.extend(["", "En particular, para **n=20** se elige producto 1 con S=0–3, "
        "producto 2 con **S=4–14** y producto 1 con S=15–20. La banda aparece numéricamente "
        "en esta calibración; no se impone al solver ni constituye una demostración para todos los parámetros.", "",
        "Las políticas de los niveles iniciales m<n se resuelven por inducción hacia atrás. "
        "Los gráficos de trayectorias e histogramas incluyen esos niveles: un color fuera de la banda "
        "madura durante el inicio no indica una inconsistencia.", "",
        "![Políticas exactas de conteos](figures/optimal_policies_by_count.png)", "",
        "![Ventaja y política para n=20](figures/exact_policy_detail_p0_0.5_n_20.png)", "",
        "## Trayectorias y estabilidad", "",
        "Se simularon 20 consumidores × 20.000 interacciones en cada uno de 11 escenarios. "
        "Cada gráfico e histograma usa al consumidor 1, sin promediar estados. Se registra el "
        "estado antes de la decisión y el resultado.", "",
        f"En el experimento largo de n=20 y p₀=0.5, con 100 consumidores × 200.000 interacciones "
        f"y burn-in de 20.000, la frecuencia media del producto 2 es **{100*mean:.3f}%** "
        f"(IC 95% entre consumidores: {100*lower:.3f}%–{100*upper:.3f}%). El promedio de "
        f"éxitos recordados después del burn-in es {long.mean_remembered_successes.mean():.4f}. "
        "El intervalo mide incertidumbre Monte Carlo; no certifica estacionariedad ni tiempo de mezcla.", "",
        "![Gráfico equivalente al adjunto](figures/individual_states/individual_states_p0_0.5_n_20.png)", "",
        "![Comparación descriptiva con la referencia geométrica](figures/geometric_vs_replacement_p0_0.5_rho_0.95_n_20.png)", "",
        "Los paneles de la referencia geométrica y el nuevo modelo no usan trayectorias emparejadas. "
        "La referencia original no se modifica.", "",
        "## Monte Carlo calendario", "",
        "Se recalcularon **35 escenarios de 20.000 consumidores × 3.000 períodos**, siete capacidades "
        "y cinco valores de p₀. En los bloques de 25 períodos, los intervalos al 95% usan al consumidor "
        "como clúster. Los mismos uniformes se comparten entre capacidades a igual p₀.", "",
        "Promedios de los 3.000 períodos para p₀=0.5:", "",
        "| n | Cuota de A | Producto 2, incondicional | Producto 2 dado A |", "|---|---:|---:|---:|"])
    for _,r in table[table.p0 == .5].iterrows():
        lines.append(f"| {int(r.n)} | {100*r.a_market_share:.2f}% | "
                     f"{100*r.product2_unconditional_rate:.2f}% | {100*r.product2_given_a_rate:.2f}% |")
    lines.extend(["", "![Monte Carlo con reemplazo opuesto](figures/monte_carlo_by_window.png)", "",
        "[Tabla completa con intervalos](data/calendar_summary.csv) · "
        "[Bandas detectadas, sin imponer umbrales](data/policy_intervals.csv) · "
        "[Diagnósticos numéricos](data/solver_diagnostics.csv) · "
        "[Auditoría](data/validation.json)", "",
        "## Reproducción", "", "Código y comandos en [README.md](README.md). "
        f"Se resolvieron {len(diagnostics)} calibraciones exactas; las simulaciones se generaron "
        "de nuevo con esta dinámica, no se reclasificaron ni reutilizaron las trayectorias FIFO."])
    (ROOT / "RESULTADOS.md").write_text("\n".join(lines)+"\n")
    print(f"Wrote {ROOT / 'RESULTADOS.md'}",flush=True)


if __name__ == "__main__":
    main()
