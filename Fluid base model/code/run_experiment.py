"""Reproduce the base fluid experiment, convergence checks and PNG figures."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import pandas as pd

from diagnostics import (compare_meshes, load_solution, save_solution, solution_summary,
                         switching_table, trajectory_diagnostics)
from model import Parameters, choose_cutoff, solve
from simulation import simulate_paths, summarize_paths


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quick", action="store_true", help="Smaller smoke experiment, saved separately")
    parser.add_argument("--reuse-hjb", action="store_true", help="Reuse matching saved grids; recompute any missing ones")
    return parser.parse_args()


def compare_time_steps(coarse: pd.DataFrame, fine: pd.DataFrame) -> list[dict]:
    results = []
    for key, group in coarse.groupby(["state_index", "path_id", "policy"], sort=False):
        other = fine.loc[fine.state_index.eq(key[0]) & fine.policy.eq(key[2])]
        a, b = group.iloc[-1], other.iloc[-1]
        results.append({"state_index": int(key[0]), "path_id": key[1], "policy": key[2],
                        "coarse_terminal_t": float(a.t), "fine_terminal_t": float(b.t),
                        "terminal_S_change": float(abs(a.S - b.S)),
                        "terminal_F_change": float(abs(a.F - b.F)),
                        "terminal_x_change": float(abs(a.x - b.x)),
                        "discounted_profit_change": float(abs(a.discounted_profit - b.discounted_profit)),
                        "product2_time_share_change": float(abs(a.product2_time / a.t - b.product2_time / b.t)) if a.t > 0 and b.t > 0 else 0.0,
                        "both_reach_same_horizon": bool(a.termination_reason == "horizon" and b.termination_reason == "horizon" and abs(a.t - b.t) < 1e-8)})
    return results


def write_summary(path: Path, report: dict, trajectory_summary: pd.DataFrame,
                  trajectory_checks: pd.DataFrame):
    finest = report["meshes"][-1]
    comparisons = report["mesh_comparisons"]
    lines = ["# Resultados del modelo fluido base", "",
             "Experimento determinista sin olvido; horizonte infinito en la HJB, dominio espacial truncado.", "",
             "## Parámetros y precisión", "",
             f"- Parámetros: `{json.dumps(report['parameters'], ensure_ascii=False)}`.",
             f"- Tasa de descuento continuo: δ = {report['delta']:.10g} = −log(γ).",
             f"- Malla final: h = {finest['h']:g}; interior mostrado S+F ≤ {finest['report_n']:g}; frontera de cálculo N = {finest['cutoff_n']:g}.",
             f"- Cota teórica del efecto del truncamiento en el valor de la malla: {finest['theoretical_truncation_bound']:.3g}.",
             f"- Máxima anchura observada del intervalo superior/inferior: {finest['maximum_truncation_interval_width']:.3g} (puede redondear a cero en doble precisión).",
             f"- Residuo algebraico de la HJB discretizada: {finest['maximum_hjb_residual']:.3g}.",
             f"- V_h(0,0) = {finest['value_at_origin_lower']:.8f}; ventaja del producto 2 en el origen = {finest['advantage_at_origin']:.8f}.", "",
             "La cota anterior controla solamente el corte espacial del problema de malla. El residuo es una comprobación algebraica; ninguno certifica el error de discretización de la HJB continua. Las cotas se calculan en aritmética de doble precisión, sin aritmética de intervalos.", "",
             "## Comparación de mallas", "",
             "| h anterior → h nuevo | Máximo cambio de V | Fracción de acciones distintas |",
             "|---|---:|---:|"]
    for row in comparisons:
        lines.append(f"| {row['coarse_h']:g} → {row['fine_h']:g} | {row['maximum_value_change']:.6g} | {row['policy_disagreement_fraction']:.3%} |")
    lines += ["", "Estas diferencias son evidencia numérica y no una demostración de convergencia. Cerca de la indiferencia, cambios pequeños del valor pueden cambiar el producto elegido.", "",
              "## Trayectorias", "",
              "Los beneficios de esta tabla son integrales hasta T; no incluyen el beneficio posterior. Todas las políticas se comparan con el mismo horizonte si ninguna alcanza antes la frontera del interior.", "",
              "| Estado inicial | Política | Tiempo final | Beneficio hasta T | Tiempo con producto 2 |",
              "|---|---|---:|---:|---:|"]
    for row in trajectory_summary.to_dict("records"):
        lines.append(f"| {row['path_id']} | {row['policy']} | {row['final_time']:.4g} | {row['discounted_profit']:.6f} | {row['product2_time_share']:.2%} |")
    if report["time_refinement"]:
        maximum = max(r["discounted_profit_change"] for r in report["time_refinement"])
        lines += ["", f"Al dividir dt por dos, el máximo cambio del beneficio acumulado es {maximum:.6g}. Se conserva la trayectoria con el paso más fino."]
    optimal = trajectory_checks.loc[trajectory_checks.policy.eq("optimal")]
    gap = float(optimal.continuation_identity_gap.abs().max())
    lines += ["", f"Comprobación adicional: máximo |beneficio hasta T + e^(−δT)V_h(S_T,F_T) − V_h(S_0,F_0)| para la política óptima interpolada = {gap:.6g}. Incluye errores de malla, interpolación y tiempo; no se presenta como error de truncamiento.", "",
              "## Archivos", "",
              "- `figures/policy_sf.png`: producto por estado (S,F) y trayectorias óptimas.",
              "- `figures/policy_nx.png`: producto por media posterior e información acumulada.",
              "- `figures/value_and_demand.png`: valor óptimo y demanda.",
              "- `figures/trajectories.png`: estados, creencias, demanda y beneficios frente a productos fijos.",
              "- `figures/switching_advantage.png`: ventaja local y frontera de cambio de producto.",
              "- `data/solution.npz`: valores, intervalos, demanda y acciones del interior completo.",
              "- `data/switching_regions.csv`: todas las componentes de producto 2 por diagonal, sin imponer un solo intervalo.",
              "- `data/trajectories.csv`, `data/trajectory_summary.csv`, `data/trajectory_diagnostics.csv`: trayectorias y comprobaciones.",
              "- `diagnostics.json`: configuración efectiva y diagnósticos de malla, frontera y tiempo.", "",
              "La clasificación gris indica empate o incertidumbre por truncamiento. La ausencia de gris no elimina el error de malla. No se ha demostrado aquí la aproximación del modelo estocástico original por el fluido.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.quick:
        config["solver"].update(mesh_steps=[1.0, 0.5], report_n=40.0, truncation_tolerance=1e-3)
        config["simulation"].update(horizon=20.0, dt=0.2)
    output = args.output_dir or ROOT / "outputs" / ("quick" if args.quick else "")
    output = output.resolve()
    data_dir, figure_dir = output / "data", output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    params = Parameters(**config["primitives"])
    settings = config["solver"]
    meshes = list(map(float, settings["mesh_steps"]))
    if not meshes or any(not math.isfinite(h) or h <= 0 for h in meshes):
        raise ValueError("mesh_steps must be nonempty, positive and finite")
    if any(b >= a or not np.isclose(a / b, round(a / b)) for a, b in zip(meshes, meshes[1:])):
        raise ValueError("mesh_steps must be strictly decreasing nested meshes")
    report = {"parameters": asdict(params), "delta": params.delta, "effective_config": config,
              "method": "monotone forward-neighbor HJB, backward diagonal sweep, simultaneous boundary brackets",
              "meshes": [], "mesh_comparisons": [], "time_refinement": []}
    previous = None
    started = time.monotonic()
    for h in meshes:
        cutoff = choose_cutoff(params, h, settings["report_n"], settings["truncation_tolerance"])
        print(f"Solving h={h:g}, N={cutoff:g}, reported n<={settings['report_n']:g}", flush=True)
        before = time.monotonic()
        cache_path = data_dir / ("solution.npz" if h == meshes[-1] else f"mesh_h_{h:g}.npz")
        reused = False
        if args.reuse_hjb and cache_path.exists():
            current = load_solution(cache_path)
            reused = (current.params == params and np.isclose(current.h, h)
                      and np.isclose(current.report_n, settings["report_n"])
                      and np.isclose(current.cutoff_n, cutoff)
                      and current.action_tolerance == settings.get("action_tolerance", 1e-8))
        if not reused:
            current = solve(params, h, cutoff, settings["report_n"],
                            action_tolerance=settings.get("action_tolerance", 1e-8))
            save_solution(current, cache_path)
        summary = solution_summary(current)
        summary["solve_seconds"] = time.monotonic() - before
        summary["reused_saved_grid"] = reused
        report["meshes"].append(summary)
        print(f"  V(0,0)={current.values[0][0]:.8f}; theoretical boundary bound={current.theoretical_error_bound:.3g}; {summary['solve_seconds']:.2f}s", flush=True)
        if previous is not None:
            report["mesh_comparisons"].append(compare_meshes(previous, current))
        previous = current
    solution = previous
    simulation = config["simulation"]
    print("Simulating deterministic optimal and fixed-product paths", flush=True)
    paths = simulate_paths(solution, simulation["initial_states"], simulation["horizon"], simulation["dt"])
    if simulation.get("check_time_refinement", True):
        refined = simulate_paths(solution, simulation["initial_states"], simulation["horizon"], simulation["dt"] / 2)
        report["time_refinement"] = compare_time_steps(paths, refined)
        paths = refined
    summaries = summarize_paths(paths)
    checks = trajectory_diagnostics(paths, solution)
    report["all_paths_reach_requested_horizon"] = bool(summaries.termination_reason.eq("horizon").all())
    report["saved_trajectory_dt"] = simulation["dt"] / (2 if simulation.get("check_time_refinement", True) else 1)
    save_solution(solution, data_dir / "solution.npz")
    switching_table(solution).to_csv(data_dir / "switching_regions.csv", index=False)
    paths.to_csv(data_dir / "trajectories.csv", index=False)
    summaries.to_csv(data_dir / "trajectory_summary.csv", index=False)
    checks.to_csv(data_dir / "trajectory_diagnostics.csv", index=False)
    print("Rendering figures", flush=True)
    from figures import make_figures
    figures = make_figures(solution, paths, figure_dir)
    report["figures"] = [str(path.relative_to(output)) for path in figures]
    report["elapsed_seconds"] = time.monotonic() - started
    (output / "diagnostics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary(output / "summary.md", report, summaries, checks)
    print(f"Results: {output}\nElapsed: {report['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
