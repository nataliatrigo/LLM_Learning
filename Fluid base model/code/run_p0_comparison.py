"""Repeat the fluid experiment across demand thresholds, preserving the base run."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from model import Parameters, choose_cutoff


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "p0_comparison")
    parser.add_argument("--workers", type=int, default=2, help="Concurrent scenarios, from 1 to 5")
    parser.add_argument("--reuse-hjb", action="store_true", help="Reuse matching saved HJB grids")
    parser.add_argument("--quick", action="store_true", help="Smoke run in a separate quick/ subfolder")
    return parser.parse_args()


def validate_config(config: dict) -> list[float]:
    """Check the comparison thresholds before creating any scenario files."""
    raw = config.get("p0_comparison", {}).get("values")
    if not isinstance(raw, list) or not raw:
        raise ValueError("config.p0_comparison.values must be a nonempty list")
    if len(raw) > 5:
        raise ValueError("The comparison figure layout supports at most five p0 thresholds")
    if any(isinstance(value, bool) for value in raw):
        raise ValueError("p0 thresholds must be numbers, not booleans")
    values = [float(value) for value in raw]
    if any(not math.isfinite(value) or not 0 < value < 1 for value in values):
        raise ValueError("Every p0 threshold must be finite and strictly between 0 and 1")
    if len(set(values)) != len(values):
        raise ValueError("p0 comparison values must be distinct")
    if len({case_name(value) for value in values}) != len(values):
        raise ValueError("p0 thresholds are too close to have distinct scenario folder names")
    states = config.get("simulation", {}).get("initial_states", [])
    if not any(isinstance(state, (list, tuple)) and len(state) == 2
               and state[0] == 0 and state[1] == 0 for state in states):
        raise ValueError("p0 comparison requires the initial state (0,0) for the origin comparison figures")
    for value in values:
        Parameters(**{**config["primitives"], "p0": value})
    return values


def build_case_config(config: dict, p0: float, quick: bool = False) -> dict:
    """All economic and numerical settings stay fixed except the threshold."""
    case = deepcopy(config)
    case["primitives"]["p0"] = float(p0)
    if quick:
        # Match run_experiment.py's smoke settings, without passing --quick
        # downstream: effective_config.json then records the actual settings.
        case["solver"].update(mesh_steps=[1.0, 0.5], report_n=40.0, truncation_tolerance=1e-3)
        case["simulation"].update(horizon=20.0, dt=0.2)
    return case


def case_name(p0: float) -> str:
    return f"p0_{p0:g}"


def seed_matching_base_grids(case: dict, case_dir: Path) -> list[str]:
    """Copy only base grids matching every parameter and solver field.

    The child runner independently validates every copied or local cache.
    Existing scenario caches are never replaced here.
    """
    params = Parameters(**case["primitives"])
    settings = case["solver"]
    steps = list(map(float, settings["mesh_steps"]))
    copied = []
    for h in steps:
        name = "solution.npz" if h == steps[-1] else f"mesh_h_{h:g}.npz"
        source = ROOT / "outputs" / "data" / name
        target = case_dir / "data" / name
        if target.exists() or not source.exists():
            continue
        cutoff = choose_cutoff(params, h, settings["report_n"], settings["truncation_tolerance"])
        try:
            with np.load(source, allow_pickle=False) as saved:
                metadata = json.loads(str(saved["metadata"]))
            matches = (
                metadata["parameters"] == asdict(params)
                and metadata["h"] == h
                and metadata["report_n"] == settings["report_n"]
                and metadata["cutoff_n"] == cutoff
                and metadata.get("action_tolerance", 1e-8) == settings.get("action_tolerance", 1e-8)
            )
        except (ValueError, OSError, KeyError, TypeError):
            matches = False
        if matches:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(name)
    return copied


def run_case(config: dict, p0: float, output: Path, reuse_hjb: bool = False,
             quick: bool = False) -> Path:
    case_dir = output / case_name(p0)
    case_dir.mkdir(parents=True, exist_ok=True)
    case = build_case_config(config, p0, quick)
    config_path = case_dir / "effective_config.json"
    config_path.write_text(json.dumps(case, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    seeded = seed_matching_base_grids(case, case_dir) if reuse_hjb else []
    command = [sys.executable, str(ROOT / "code" / "run_experiment.py"),
               "--config", str(config_path), "--output-dir", str(case_dir)]
    if reuse_hjb:
        command.append("--reuse-hjb")
    log_path = case_dir / "run.log"
    print(f"Starting p0={p0:g}; log: {log_path}", flush=True)
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"Seeded matching base grids: {', '.join(seeded) or 'none'}\n")
        log.flush()
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError(f"p0={p0:g} failed (exit {result.returncode}); see {log_path}")
    print(f"Completed p0={p0:g} in {time.monotonic() - started:.1f}s", flush=True)
    return case_dir


def summarize_case(case_dir: Path) -> dict:
    """Extract scalar evidence without interpreting grid counts as probabilities."""
    report = json.loads((case_dir / "diagnostics.json").read_text(encoding="utf-8"))
    finest = report["meshes"][-1]
    counts = finest["actions"]
    total = sum(int(count) for count in counts.values())
    with np.load(case_dir / "data" / "solution.npz", allow_pickle=False) as saved:
        action = int(saved["actions"][0])
    latest = report["mesh_comparisons"][-1] if report["mesh_comparisons"] else {}
    return {
        "p0": report["parameters"]["p0"],
        "h": finest["h"], "cutoff_n": finest["cutoff_n"],
        "originV": finest["value_at_origin_lower"],
        "origin_action": action,
        "origin_advantage": finest["advantage_at_origin"],
        "product2_grid_fraction": int(counts.get("2", 0)) / total,
        "uncertainty_fraction": int(counts.get("0", 0)) / total,
        "trunc_bound": finest["theoretical_truncation_bound"],
        "max_hjb_residual": finest["maximum_hjb_residual"],
        "last_mesh_value_change": latest.get("maximum_value_change", float("nan")),
        "last_mesh_policy_disagreement_fraction": latest.get("policy_disagreement_fraction", float("nan")),
        "max_time_profit_change": max((row["discounted_profit_change"] for row in report["time_refinement"]), default=float("nan")),
        "all_paths_reach_horizon": report["all_paths_reach_requested_horizon"],
    }


def check_value_monotonicity(cases: list[Path], tolerance: float = 1e-10) -> list[dict]:
    """For nonnegative margins, harder demand thresholds cannot improve value.

    Compare every retained node on identical grids. This is a numerical
    structural check, not an estimate of the continuous HJB's mesh error.
    """
    ordered = []
    for case in cases:
        with np.load(case / "data" / "solution.npz", allow_pickle=False) as saved:
            meta = json.loads(str(saved["metadata"]))
        ordered.append((float(meta["parameters"]["p0"]), case, meta))
    ordered.sort(key=lambda item: item[0])
    checks = []
    previous_values = None
    previous_meta = None
    previous_p0 = None
    for p0, case, meta in ordered:
        with np.load(case / "data" / "solution.npz", allow_pickle=False) as saved:
            values = saved["values"]
        if previous_values is not None:
            same_grid = all(meta[key] == previous_meta[key] for key in ("h", "report_n", "cutoff_n"))
            same_primitives = {key: value for key, value in meta["parameters"].items() if key != "p0"} == {
                key: value for key, value in previous_meta["parameters"].items() if key != "p0"}
            if not same_grid or not same_primitives or len(values) != len(previous_values):
                raise ValueError("Value monotonicity check requires matching grids and other parameters")
            difference = values - previous_values
            checks.append({"lower_p0": previous_p0, "higher_p0": p0, "retained_states": len(values),
                           "maximum_positive_value_increase": max(0.0, float(difference.max())),
                           "violations_above_tolerance": int(np.count_nonzero(difference > tolerance)),
                           "tolerance": tolerance})
        previous_values, previous_meta, previous_p0 = values, meta, p0
    return checks


def write_summary(output: Path, config: dict, cases: list[Path],
                  summary: pd.DataFrame, trajectories: pd.DataFrame,
                  figures: list[Path], quick: bool, monotonicity: list[dict] | None = None):
    effective = build_case_config(config, float(summary.iloc[0].p0), quick)
    simulation, settings = effective["simulation"], effective["solver"]
    horizon = float(simulation["horizon"])
    action_text = {0: "indiferencia/incertidumbre", 1: "producto 1", 2: "producto 2"}
    lines = ["# Comparación del modelo fluido para distintos p₀", "",
             f"Se resuelve de nuevo la HJB para p₀ ∈ {{{', '.join(f'{p:g}' for p in summary.p0)}}}. "
             "Cada escenario conserva los demás parámetros económicos, las mallas, el dominio de interés y los estados iniciales.", "",
             f"La HJB tiene horizonte infinito y descuento δ = −log(γ). Las trayectorias se integran hasta el tiempo calendario T = {horizon:g}; "
             "su beneficio acumulado hasta T no incluye la continuación posterior. El eje n = S+F mide información acumulada, no tiempo.", "",
             f"Los {len(simulation['initial_states'])} estados iniciales comunes (S₀,F₀) son "
             + ", ".join(f"({S:g},{F:g})" for S, F in simulation["initial_states"]) + ".",
             f"Mallas h = {settings['mesh_steps']}; interior S+F ≤ {settings['report_n']:g}. "
             "La frontera de cálculo queda más lejos y se elige mediante una cota de truncamiento.", "",
             "## Política y valor en el origen", "",
             "| p₀ | Vₕ(0,0) | Producto en (0,0) | Ventaja producto 2 | Fracción de malla con producto 2 |",
             "|---:|---:|---|---:|---:|"]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['p0']:g} | {row['originV']:.6f} | {action_text[row['origin_action']]} | "
                     f"{row['origin_advantage']:.6g} | {row['product2_grid_fraction']:.2%} |")
    lines += ["", "La fracción anterior cuenta nodos en la malla uniforme de (S,F). No es una probabilidad de visitar estados, "
              "ni una fracción de tiempo con producto 2; depende del dominio y de las coordenadas de la malla."]
    if monotonicity:
        violations = sum(row["violations_above_tolerance"] for row in monotonicity)
        maximum = max(row["maximum_positive_value_increase"] for row in monotonicity)
        lines += ["", f"Comprobación de estructura: al aumentar p₀, el valor no debe aumentar (con márgenes no negativos). "
                  f"Se compararon todos los nodos entre escenarios consecutivos: {violations} aumentos superiores a 10⁻¹⁰; "
                  f"máximo aumento positivo observado = {maximum:.3g}. El detalle se guarda en `manifest.json`."]
    lines += ["",
              "## Trayectoria óptima desde (0,0)", "",
              f"| p₀ | Tiempo final | n final | Media x final | Demanda final | Beneficio hasta T={horizon:g} | Tiempo con producto 2 |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    origin = trajectories.loc[trajectories.initial_S.eq(0) & trajectories.initial_F.eq(0)
                              & trajectories.policy.eq("optimal")]
    for row in origin.to_dict("records"):
        lines.append(f"| {row['p0']:g} | {row['final_time']:.4g} | {row['final_n']:.6g} | {row['final_x']:.6g} | "
                     f"{row['final_demand']:.6g} | {row['discounted_profit']:.6f} | {row['product2_time_share']:.2%} |")
    if origin.empty:
        lines += ["", "La configuración no incluye el estado inicial (0,0)."]
    if bool(summary.all_paths_reach_horizon.all()):
        lines += ["", f"Todas las trayectorias óptimas y de productos fijos alcanzan T = {horizon:g} sin salir del interior calculado."]
    else:
        lines += ["", "Algunas trayectorias terminan al alcanzar S+F = report_n. Sus beneficios tienen horizontes efectivos distintos; "
                  "se deben revisar las terminaciones en `trajectory_summary.csv` antes de compararlos."]
    lines += ["", "## Precisión numérica", "",
              "| p₀ | Cota truncamiento | Máximo cambio de V, últimas mallas | Acciones distintas, últimas mallas | Cambio de beneficio al dividir dt por 2 |",
              "|---:|---:|---:|---:|---:|"]
    for row in summary.to_dict("records"):
        lines.append(f"| {row['p0']:g} | {row['trunc_bound']:.3g} | {row['last_mesh_value_change']:.6g} | "
                     f"{row['last_mesh_policy_disagreement_fraction']:.3%} | {row['max_time_profit_change']:.6g} |")
    lines += ["", "La cota controla el corte espacial de la HJB discretizada. Las diferencias entre mallas y pasos de tiempo son "
              "comprobaciones de sensibilidad, no cotas del error continuo. El residuo algebraico de la HJB se guarda en `case_summary.csv`. "
              "La clasificación de indiferencia/incertidumbre no incorpora todo el error de malla. No se presupone que toda trayectoria llegue "
              "a una frontera entre productos ni se ha probado la aproximación al modelo estocástico.", "",
              "## Figuras comparativas", ""]
    lines += [f"- [{figure.stem}]({figure.relative_to(output).as_posix()})" for figure in figures]
    lines += ["", "## Detalle y datos", ""]
    for case in cases:
        lines.append(f"- [{case.name}: resumen y figuras]({case.name}/summary.md); "
                     f"[configuración]({case.name}/effective_config.json); [registro]({case.name}/run.log).")
    lines += ["- [Tabla de escenarios](case_summary.csv).",
              "- [Resumen de todas las trayectorias](trajectory_summary.csv).",
              "- [Extremos de las trayectorias óptimas desde el origen](origin_path_endpoints.csv).",
              "- [Configuración y archivos del barrido](manifest.json).", "",
              "Los resultados base en `outputs/` se conservan; este barrido guarda sus escenarios y comparaciones en una carpeta separada.", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    if not 1 <= args.workers <= 5:
        raise ValueError("--workers must be between 1 and 5")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    values = validate_config(config)
    output = args.output_dir.resolve()
    if args.quick:
        output /= "quick"
    if output in {ROOT, ROOT / "outputs", ROOT / "outputs" / "data", ROOT / "outputs" / "figures"}:
        raise ValueError("Use a dedicated comparison folder to preserve the base experiment")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "running", "source_config": str(args.config.resolve()), "config": config,
                "quick": args.quick, "workers": args.workers, "reuse_hjb": args.reuse_hjb,
                "scenario_directories": [case_name(p0) for p0 in values]}
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    started = time.monotonic()
    failures = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_case, config, p0, output, args.reuse_hjb, args.quick): p0 for p0 in values}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:
                failures.append(str(error))
                print(f"FAILED: {error}", file=sys.stderr, flush=True)
    if failures:
        manifest.update(status="failed", errors=failures, elapsed_seconds=time.monotonic() - started)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        raise RuntimeError("Comparison was not aggregated because one or more scenarios failed; inspect the scenario logs")

    cases = [output / case_name(p0) for p0 in values]
    summary = pd.DataFrame([summarize_case(case) for case in cases])
    monotonicity = check_value_monotonicity(cases)
    trajectory_tables = []
    for p0, case in zip(values, cases):
        table = pd.read_csv(case / "data" / "trajectory_summary.csv")
        table.insert(0, "p0", p0)
        trajectory_tables.append(table)
    trajectories = pd.concat(trajectory_tables, ignore_index=True)
    print("Rendering comparison figures and writing aggregate tables", flush=True)
    from comparison_figures import make_comparison_figures
    figure_dir = output / "figures"
    result = make_comparison_figures(cases, figure_dir)
    figures = list(result) if result is not None else sorted(figure_dir.glob("*.png"))
    summary.to_csv(output / "case_summary.csv", index=False)
    trajectories.to_csv(output / "trajectory_summary.csv", index=False)
    trajectories.loc[trajectories.initial_S.eq(0) & trajectories.initial_F.eq(0)
                     & trajectories.policy.eq("optimal")].to_csv(output / "origin_path_endpoints.csv", index=False)
    write_summary(output, config, cases, summary, trajectories, figures, args.quick, monotonicity)
    manifest.update(status="complete", elapsed_seconds=time.monotonic() - started,
                    value_monotonicity_checks=monotonicity,
                    files=[str(path.relative_to(output)) for path in sorted(output.rglob("*")) if path.is_file()])
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Comparison complete: {output / 'summary.md'} ({manifest['elapsed_seconds']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
