"""Compute fluid oblivious equilibrium triples (policy, conjecture, initial_conditions).

Verified stationary responses can be reused because the additional condition
changes the admissible initial states, not the frozen-rival HJB.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluid-oblivious-initial_conditions-mpl")

import numpy as np
import pandas as pd

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.fluid_oblivious.code.run_experiment import (
    case_tag, match_equilibria, numerical_sources, refinement_checks,
    solve_case, write_json,
)
from oblivious.fluid_oblivious.code.trajectories import simulate_pair
from oblivious.fluid_oblivious_initial_conditions.code.initial_conditions import (
    initial_conditions_for_equilibrium, contains, interval_notation, phase_limit,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def load_case(path, config, p0, rho):
    """Reject incompatible or stale stationary computations before reuse."""
    with gzip.open(path, "rt") as handle:
        saved = json.load(handle)
    expected = dict(parameters=config["parameters"], p0=p0, rho=rho,
                    grids=config["grids"], order=config["quadrature_order"],
                    search_points=config["search_points"],
                    search_refinement_points=config["search_refinement_points"],
                    sources=numerical_sources())
    if saved.get("inputs") != expected or not saved.get("completed"):
        raise ValueError(f"Stale or incompatible checkpoint {path}; use --recompute")
    if [row["n"] for row in saved["grids"]] != config["grids"]:
        raise ValueError(f"Grid mismatch in {path}")
    if any(row["error"] is not None or row["result"] is None for row in saved["grids"]):
        raise ValueError(f"Incomplete stationary search in {path}")
    return saved


def correspondence(equilibria):
    """Exact atomic partition by all initial_conditions endpoints, including isolated points."""
    edges = sorted({0., 1., *(edge for eq in equilibria if eq["verified"]
                            for part in eq["initial_conditions"] for edge in (part["lo"], part["hi"]))})
    parts = []
    for index, edge in enumerate(edges):
        parts.append(dict(lo=edge, hi=edge, left_closed=True, right_closed=True))
        if index + 1 < len(edges):
            parts.append(dict(lo=edge, hi=edges[index+1], left_closed=False, right_closed=False))
    for part in parts:
        midpoint = (part["lo"] + part["hi"])/2
        part["equilibrium_ids"] = [eq["equilibrium_id"] for eq in equilibria
                                   if eq["verified"] and contains(eq["initial_conditions"], midpoint)]
        part["equilibrium_count"] = len(part["equilibrium_ids"])
    merged = []
    for part in parts:
        if (merged and merged[-1]["equilibrium_ids"] == part["equilibrium_ids"]
                and merged[-1]["hi"] == part["lo"]
                and (merged[-1]["right_closed"] or part["left_closed"])):
            merged[-1].update(hi=part["hi"], right_closed=part["right_closed"])
        else:
            merged.append(dict(part))
    return merged


def analyse_case(saved, config):
    params = Parameters(**config["parameters"], rho=saved["rho"])
    refinements = refinement_checks(saved, config["refinement"])
    fine, coarse = saved["grids"][-1], saved["grids"][-2]
    equilibria = fine["result"]["equilibria"]
    previous = coarse["result"]["equilibria"]
    matching = match_equilibria(equilibria, previous)
    entries = []
    for index, equilibrium in enumerate(equilibria):
        initial_conditions = initial_conditions_for_equilibrium(params, equilibrium["policy"], equilibrium["x_star"])
        tests = [row for row in refinements if row["fine_n"] == fine["n"]
                 and row["equilibrium_id"] == index]
        stationary_refined = len(tests) == 1 and tests[0]["passed"]
        comparison = dict(passed=False, boundary_change=None, same_topology=False)
        if index in matching:
            old = previous[matching[index]]
            old_initial_conditions = initial_conditions_for_equilibrium(params, old["policy"], old["x_star"])
            a, b = initial_conditions["intervals"], old_initial_conditions["intervals"]
            topology = len(a) == len(b) and all(
                first["left_closed"] == second["left_closed"]
                and first["right_closed"] == second["right_closed"]
                and (first["lo"] == first["hi"]) == (second["lo"] == second["hi"])
                for first, second in zip(a, b))
            change = max((abs(first[key]-second[key]) for first, second in zip(a, b)
                          for key in ("lo", "hi")), default=0.) if topology else None
            comparison = dict(coarse_n=coarse["n"], fine_n=fine["n"],
                              coarse_initial_conditions=old_initial_conditions["intervals"], same_topology=topology,
                              boundary_change=change,
                              passed=bool(topology and change <= config["initial_conditions_refinement_absolute"]))
        valid = bool(equilibrium["verified"] and stationary_refined
                     and initial_conditions["stationary"] and comparison["passed"])
        entries.append(dict(
            equilibrium_id=index, x_bar=equilibrium["x_star"], alpha=equilibrium["alpha"],
            kind=equilibrium["kind"], verified=valid, policy=equilibrium["policy"],
            initial_conditions=initial_conditions["intervals"], initial_conditions_notation=initial_conditions["text"],
            initial_conditions_measure=initial_conditions["measure"], stationarity_defect=initial_conditions["stationarity_defect"],
            phase_partition=initial_conditions["partition"], root_actions=initial_conditions["root_actions"],
            tie_convention=initial_conditions["tie_convention"],
            stationary_diagnostics=equilibrium["diagnostics"],
            demand_audit=equilibrium.get("demand_audit"), initial_conditions_refinement=comparison,
        ))
    partition = correspondence(entries)
    case = dict(p0=saved["p0"], rho=saved["rho"], grid_nodes=fine["n"],
                status="verified" if entries and all(eq["verified"] for eq in entries)
                else "no_equilibrium_detected" if not entries else "verification_failed",
                equilibria=entries, initial_state_correspondence=partition,
                uncovered_initial_states=[part for part in partition if part["equilibrium_count"] == 0],
                search=fine["result"]["search"])
    return case, refinements


def save_tables(cases, refinements, destination):
    summaries, intervals, policies, regions, case_rows = [], [], [], [], []
    for case in cases:
        identity = dict(p0=case["p0"], rho=case["rho"])
        case_rows.append(dict(identity, status=case["status"], equilibrium_count=len(case["equilibria"]),
                              equilibria_at_x0_0p5=sum(eq["verified"] and contains(eq["initial_conditions"], .5)
                                                       for eq in case["equilibria"]),
                              maximum_equilibria_for_one_x0=max(part["equilibrium_count"]
                                                               for part in case["initial_state_correspondence"]),
                              uncovered_x0=interval_notation(case["uncovered_initial_states"]),
                              uncovered_measure=sum(part["hi"]-part["lo"]
                                                    for part in case["uncovered_initial_states"])))
        for eq in case["equilibria"]:
            base = dict(identity, equilibrium_id=eq["equilibrium_id"], x_bar=eq["x_bar"])
            summaries.append(dict(base, alpha=eq["alpha"], kind=eq["kind"], verified=eq["verified"],
                                  initial_conditions=eq["initial_conditions_notation"], initial_conditions_measure=eq["initial_conditions_measure"],
                                  stationarity_defect=eq["stationarity_defect"],
                                  initial_conditions_boundary_refinement=eq["initial_conditions_refinement"]["boundary_change"],
                                  hjb_residual=eq["stationary_diagnostics"]["hjb_residual"]))
            intervals.extend(dict(base, **part) for part in eq["initial_conditions"])
            for part in eq["policy"]["intervals"]:
                policies.append(dict(base, type="interval", lo=part["lo"], hi=part["hi"],
                                     left_closed=part["lo"] == 0., right_closed=part["hi"] == 1.,
                                     sigma=float(part["product"] == 2)))
            for point in eq["root_actions"]:
                policies.append(dict(base, type="point", lo=point["x"], hi=point["x"],
                                     left_closed=True, right_closed=True, sigma=point["alpha"]))
        regions.extend(dict(identity, **part,
                            notation=interval_notation([part]))
                       for part in case["initial_state_correspondence"])
    for name, rows in (("equilibria", summaries), ("initial_conditions_intervals", intervals),
                       ("policies", policies), ("initial_state_correspondence", regions),
                       ("case_summary", case_rows), ("stationary_refinement", refinements)):
        pd.DataFrame(rows).to_csv(destination/(name+".csv"), index=False)
    write_json(destination/"equilibria.json", dict(cases=cases))


def validate(cases, config, source, destination):
    """Compare saved long-run predictions and freshly integrated actual paths."""
    baseline_rows, live_rows, trajectories = [], [], []
    settings = config["validation"]
    times = np.linspace(0., settings["horizon"], settings["samples"])
    baseline_sources = {}
    for case in cases:
        params = Parameters(**config["parameters"], rho=case["rho"])
        tag = lambda x: str(float(x)).replace(".", "p")
        baseline_path = source/f'data/initial_conditions_sweep/p0_{tag(case["p0"])}/rho_{tag(case["rho"])}/summary.csv'
        if baseline_path.exists():
            baseline = pd.read_csv(baseline_path)
            baseline_sources[str(baseline_path.relative_to(PROJECT))] = fingerprint(baseline_path)
        else:
            baseline = pd.DataFrame()
        for eq in case["equilibria"]:
            identity = dict(p0=case["p0"], rho=case["rho"], equilibrium_id=eq["equilibrium_id"], x_bar=eq["x_bar"])
            for x0 in settings["initial_states"]:
                limit = phase_limit(params, eq["policy"], x0)
                member = contains(eq["initial_conditions"], x0)
                assert member == (abs(limit-eq["x_bar"]) < 1e-10)
                row = dict(identity, x0=x0, long_run_limit=limit, equilibrium_from_x0=member,
                           saved_limit_change=None)
                if not baseline.empty:
                    old = baseline[(abs(baseline.rival_conjecture-eq["x_bar"]) < 1e-10)
                                   & (abs(baseline.x0-x0) < 1e-12)]
                    assert len(old) == 1
                    row["saved_limit_change"] = abs(limit-float(old.iloc[0].long_run_limit))
                    assert row["saved_limit_change"] < 1e-10
                baseline_rows.append(row)
            starts = {0.5, eq["x_bar"]}
            for point in eq["root_actions"]:
                if point["kind"] == "repelling":
                    starts.update([max(0., point["x"]-1e-4), min(1., point["x"]+1e-4)])
            for x0 in sorted(starts):
                args = dict(initial=(x0, x0), times=times, bin_width=10.,
                            rtol=settings["rtol"], atol=settings["atol"], max_step=settings["max_step"])
                result = simulate_pair(params, case["p0"], eq["policy"], **args)
                args.update(rtol=settings["rtol"]/10, atol=settings["atol"]/10,
                            max_step=settings["max_step"]/2)
                checked = simulate_pair(params, case["p0"], eq["policy"], **args)
                frame, refined = result["by_time"], checked["by_time"]
                change = float(np.max(abs(frame[["x_a", "x_b"]].to_numpy()
                                          -refined[["x_a", "x_b"]].to_numpy())))
                limit = phase_limit(params, eq["policy"], x0)
                symmetry = float(np.max(abs(frame.x_a-frame.x_b)))
                motion = np.diff(frame.x_a)
                assert change < 2e-8 and symmetry < 1e-11
                assert not (np.any(motion > 2e-10) and np.any(motion < -2e-10))
                assert np.all(frame.x_a >= min(x0, limit)-2e-9)
                assert np.all(frame.x_a <= max(x0, limit)+2e-9)
                if x0 == eq["x_bar"]:
                    assert np.max(abs(frame.x_a-x0)) < 1e-12
                live_rows.append(dict(identity, x0=x0, phase_limit=limit, terminal_x=float(frame.x_a.iloc[-1]),
                                      maximum_refinement_change=change, maximum_asymmetry=symmetry,
                                      passed=True))
                subset = frame[["time", "x_a", "x_b", "alpha_a", "demand_a"]].copy()
                for key, value in dict(identity, x0=x0).items():
                    subset[key] = value
                trajectories.append(subset)
        print(f'Validated p0={case["p0"]:g}, rho={case["rho"]:g}: {len(case["equilibria"])} equilibrium triples', flush=True)
    pd.DataFrame(baseline_rows).to_csv(destination/"initial_state_checks.csv", index=False)
    pd.DataFrame(live_rows).to_csv(destination/"ode_checks.csv", index=False)
    if trajectories:
        pd.concat(trajectories, ignore_index=True).to_csv(destination/"validation_trajectories.csv", index=False)
    report = dict(passed=True, initial_state_checks=len(baseline_rows), live_ode_checks=len(live_rows),
                  saved_phase_limit_checks=sum(row["saved_limit_change"] is not None for row in baseline_rows),
                  max_ode_refinement_change=max((row["maximum_refinement_change"] for row in live_rows), default=0.),
                  baseline_source_sha256=baseline_sources,
                  interpretation="Finite-time ODE checks verify the dynamics; limits and complete initial_conditions come from the phase line.")
    write_json(destination/"validation.json", report)
    return report


def write_results(cases, validation, output):
    count = sum(len(case["equilibria"]) for case in cases)
    lines = ["# Resultados: equilibrio desde una reputación inicial común", "",
             f"Se detectaron **{count} ternas verificadas** en **{len(cases)} combinaciones** de parámetros.", "",
             "Cada fila identifica una política óptima calculada contra el rival fijo ",
             "`x_bar`, su reputación estacionaria y el conjunto completo de inicios comunes `x0`.", "",
             "| p0 | rho | política | x_bar | sigma(x_bar) | condiciones iniciales x0 |",
             "| --- | --- | --- | --- | --- | --- |"]
    for case in cases:
        for eq in case["equilibria"]:
            lines.append(f'| {case["p0"]:g} | {case["rho"]:g} | {eq["equilibrium_id"]} | '
                         f'{eq["x_bar"]:.9f} | {eq["alpha"]:.6f} | {eq["initial_conditions_notation"]} |')
    lines.extend(["", "## Condiciones iniciales sin equilibrio detectado", "",
                  "Estos conjuntos se refieren a las políticas numéricamente detectadas y al selector declarado.", "",
                  "| p0 | rho | x0 sin equilibrio detectado |", "| --- | --- | --- |"])
    for case in cases:
        if case["uncovered_initial_states"]:
            lines.append(f'| {case["p0"]:g} | {case["rho"]:g} | '
                         f'{interval_notation(case["uncovered_initial_states"])} |')
    maximum = max(part["equilibrium_count"] for case in cases for part in case["initial_state_correspondence"])
    lines.extend(["", f"Máximo de equilibrios para un mismo x0: **{maximum}**.", "",
                  "## Verificación e interpretación", "",
                  f'- {validation["initial_state_checks"]} condiciones iniciales contrastadas; '
                  f'{validation["saved_phase_limit_checks"]} coinciden con las predicciones guardadas del experimento 10.',
                  f'- {validation["live_ode_checks"]} integraciones nuevas con demanda efectiva D(x,x), '
                  f'con cambio máximo {validation["max_ode_refinement_change"]:.3g} al refinar el integrador.',
                  '- Todas las políticas mantienen x_bar; se comprobaron las mallas 1001, 2001 y 4001 y '
                  'la sensibilidad de las fronteras de los conjuntos de condiciones iniciales en la pareja más fina.',
                  '- En umbrales repulsivos se selecciona producto 1 y se toma su continuación inmediata '
                  'hacia la izquierda (convención de flujo por eventos). La ecuación diferencial discontinua '
                  'interpretada sólo casi en todas partes no fija por sí sola esa continuación. Los puntos '
                  'de holding verificados usan la mezcla que mantiene su reputación. Los extremos abiertos/cerrados '
                  'corresponden a esta convención; los interiores de los conjuntos de condiciones iniciales no dependen de ella.',
                  '- Los conjuntos de condiciones iniciales son intervalos de la política numérica: sus umbrales tienen error de discretización. '
                  'Las búsquedas no son una prueba de exhaustividad ni se enumeran todos los selectores óptimos en indiferencias.',
                  '- La convergencia usa la línea de fase, no la cercanía de x(T) a x_bar en un horizonte finito.', "",
                  'Ver [políticas y conjuntos de condiciones iniciales completas](data/equilibria.json), [tabla de equilibrios](data/equilibria.csv), '
                  '[correspondencia por condición inicial](data/initial_state_correspondence.csv) y '
                  '[figuras](figures/README.md).', ""])
    (output/"RESULTS.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--recompute", action="store_true", help="Re-solve stationary HJBs in the new experiment")
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_dir.resolve()
    data = output/"data"
    data.mkdir(parents=True, exist_ok=True)
    from oblivious.fluid_oblivious_initial_conditions.code.reporting import make_figures
    if args.figures_only:
        cases = json.loads((data/"equilibria.json").read_text())["cases"]
        _, report = make_figures(cases, output/"figures", p0s=config["p0s"], rhos=config["rhos"], parameters=config["parameters"])
        manifest_path = data/"manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            manifest.setdefault("computation_source_sha256_at_generation", manifest.get("source_sha256", {}))
            manifest["source_sha256"] = {str(p.relative_to(PROJECT)):fingerprint(p)
                                         for p in sorted((ROOT/"code").glob("*.py"))}
            manifest["generated_data_sha256"] = {
                str(p.relative_to(output)):fingerprint(p) for p in sorted(data.glob("*"))
                if p.is_file() and p.name not in {"manifest.json", "incomplete_search.json"}}
            manifest["presentation"] = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                                             format_reference=report["format_reference"],
                                             figure_count=report["figure_count"],
                                             wording="sets of initial conditions",
                                             numerical_results="unchanged; field and artifact names updated")
            write_json(manifest_path, manifest)
        return
    source = PROJECT/config["source_experiment"]
    sources = {}
    pairs = [(p0, rho) for p0 in config["p0s"] for rho in config["rhos"]]
    if args.recompute:
        with ProcessPoolExecutor(max_workers=args.workers or config["workers"]) as pool:
            futures = [pool.submit(solve_case, config, p0, rho, data, numerical_sources(), True)
                       for p0, rho in pairs]
            for future in futures:
                future.result()
    snapshots = data/"checkpoints"
    snapshots.mkdir(exist_ok=True)
    cases, refinements = [], []
    for p0, rho in pairs:
        name = case_tag(p0, rho)+".json.gz"
        checkpoint = snapshots/name if args.recompute else source/"data/checkpoints"/name
        saved = load_case(checkpoint, config, p0, rho)
        sources[str(checkpoint.relative_to(PROJECT)) if checkpoint.is_relative_to(PROJECT) else str(checkpoint)] = fingerprint(checkpoint)
        if not args.recompute:
            shutil.copyfile(checkpoint, snapshots/name)
        case, checks = analyse_case(saved, config)
        cases.append(case)
        refinements.extend(checks)
    if any(case["status"] != "verified" for case in cases):
        write_json(data/"incomplete_search.json", dict(cases=cases))
        raise ArithmeticError("Some candidates failed stationary or initial_conditions verification; see incomplete_search.json")
    save_tables(cases, refinements, data)
    validation = validate(cases, config, source, data)
    make_figures(cases, output/"figures", p0s=config["p0s"], rhos=config["rhos"], parameters=config["parameters"])
    write_results(cases, validation, output)
    manifest = dict(status="complete", created_utc=datetime.now(timezone.utc).isoformat(), config=config,
                    case_count=len(cases), equilibrium_count=sum(len(case["equilibria"]) for case in cases),
                    stationary_source="recomputed" if args.recompute else "hash-verified experiment-10 checkpoint snapshots",
                    checkpoint_source_sha256=sources, numerical_source_sha256=numerical_sources(),
                    checkpoint_snapshot_sha256={str(p.relative_to(output)):fingerprint(p) for p in sorted(snapshots.glob("*.gz"))},
                    source_sha256={str(p.relative_to(PROJECT)):fingerprint(p) for p in sorted((ROOT/"code").glob("*.py"))},
                    validation=validation,
                    generated_data_sha256={str(p.relative_to(output)):fingerprint(p) for p in sorted(data.glob("*"))
                                           if p.is_file() and p.name not in {"manifest.json", "incomplete_search.json"}},
                    scope="Numerically detected optimal policy selectors, with exact phase-line initial_conditions for their computed thresholds.")
    write_json(data/"manifest.json", manifest)
    print(f'Complete: {manifest["equilibrium_count"]} equilibrium triples in {output}', flush=True)


if __name__ == "__main__":
    main()
