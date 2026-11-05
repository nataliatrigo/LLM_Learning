"""Spatial refinement of fluid-policy loss in the original forgetting monopoly."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import threading

for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/forgetting-fluid-convergence-mpl")
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from forgetting.Code.model import Primitives
from forgetting.Code.fluid_convergence import numerics as N

DEFAULT_CONFIG = Path(__file__).with_name("config.json")
DEFAULT_OUTPUT = ROOT/"forgetting/outputs/fluid_convergence"
CANCEL = threading.Event()
TABLES = ("original_summary", "original_curves", "grid_refinement", "fluid_refinement",
          "solver_diagnostics")


def plain(value):
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(plain(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return sha256(json.dumps(plain(value), sort_keys=True).encode()).hexdigest()


def tag(value):
    return f"{value:.12g}".replace(".", "p")


def validate_config(config):
    Primitives(p0=.5, **config["parameters"])
    if not 0 < config["parameters"]["gamma"] < 1:
        raise ValueError("The fluid study requires 0<gamma<1")
    if (not config["p0_values"] or not config["rhos"]
            or len(set(config["p0_values"])) != len(config["p0_values"])
            or len(set(config["rhos"])) != len(config["rhos"])
            or any(not 0 < p < 1 for p in config["p0_values"])
            or any(not .5 < r < 1 for r in config["rhos"])):
        raise ValueError("Require p0 in (0,1) and rho in (.5,1)")
    intervals = config["grid_intervals"]
    if (len(intervals) < 3 or intervals != sorted(set(intervals))
            or any(not isinstance(n, int) or n < 4 or n % 2 for n in intervals)
            or any(b != 2*a for a, b in zip(intervals[:-1], intervals[1:]))):
        raise ValueError("Require >=3 nested dyadic grids with even interval counts")
    if config["report_nodes"] < 3 or not 0 <= config["initial_x"] <= 1:
        raise ValueError("Invalid reporting grid or initial state")
    for collection in (config["solver"], config["assessment"]):
        if any(not math.isfinite(v) or v <= 0 for v in collection.values()):
            raise ValueError("Require positive finite solver/assessment tolerances")
    if config["policy_action_tolerance"] < 0 or config["workers"] < 1:
        raise ValueError("Invalid policy tolerance or worker count")


class Checkpoints:
    def __init__(self, folder, numerical_sources, recompute=False):
        self.folder, self.sources, self.recompute = Path(folder), numerical_sources, recompute
        self.folder.mkdir(parents=True, exist_ok=True)

    def get(self, name, settings, compute):
        if CANCEL.is_set():
            raise InterruptedError("Campaign interrupted; complete checkpoints remain reusable")
        base = self.folder/name
        key = identity(dict(settings=settings, numerical_sources=self.sources))
        metadata_path, arrays_path = base.with_suffix(".json"), base.with_suffix(".npz")
        if not self.recompute and metadata_path.exists():
            saved = json.loads(metadata_path.read_text())
            if saved["identity"] != key:
                raise ValueError(f"Changed numerical inputs: {base}; choose a new output or --recompute")
            if digest(arrays_path) != saved["npz_sha256"]:
                raise ValueError(f"Changed checkpoint array: {arrays_path}")
            with np.load(arrays_path, allow_pickle=False) as arrays:
                result = dict(saved["scalars"], **{k: arrays[k].copy() for k in saved["arrays"]})
            return result
        result = compute()
        arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
        scalars = {k: v for k, v in result.items() if k not in arrays}
        temporary = arrays_path.with_name(arrays_path.name+".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(arrays_path)
        atomic_json(metadata_path, dict(identity=key, settings=settings, arrays=list(arrays),
                                        scalars=scalars, npz_sha256=digest(arrays_path)))
        return result


def common_change(coarse, fine, field):
    # Compare values at common states, not action fractions or scalar maxima.
    return float(np.max(abs(coarse[field]-np.interp(coarse["grid"], fine["grid"], fine[field]))))


def comparison_metrics(grid, difference, x0):
    """Unweighted state-space diagnostics; L1 is not an occupation average."""
    absolute = np.abs(difference)
    return dict(sup=float(np.max(absolute)), x0=float(np.interp(x0, grid, absolute)),
                l1=float(np.trapezoid(absolute, grid)))


def case_worker(config, output, p0, rho, numerical_sources, recompute):
    parameters = Primitives(p0=p0, **config["parameters"])
    levels = [i+1 for i in config["grid_intervals"]]
    folder = output/"checkpoints"/f"p0_{tag(p0)}"/f"rho_{tag(rho)}"
    store = Checkpoints(folder, numerical_sources, recompute)
    tables = {name: [] for name in TABLES}
    common = dict(p0=p0, rho=rho)
    x0 = config["initial_x"]
    target = config["solver"]
    tol = target["conditional_tolerance"]
    numeric_settings = dict(parameters=asdict(parameters), rho=rho)

    def checked(result, kind, n, fluid_n=None):
        if (not np.isfinite(result["value"]).all()
                or result["value_error_bound"] > target["maximum_value_error_bound"]
                or (kind != "evaluation" and result["conditional_regret"] > target["maximum_conditional_regret"])):
            raise ValueError(f"Numerical certificate failed: {common}, {kind}, n={n}")
        tables["solver_diagnostics"].append(dict(**common, kind=kind,
            n_stochastic=n if kind != "fluid" else None, n_fluid=fluid_n if kind != "fluid" else n,
            residual=result["residual"], value_error_bound=result["value_error_bound"],
            conditional_regret=result["conditional_regret"], iterations=result["iterations"]))
        return result

    fluids = {}
    for n in levels:
        f = store.get(f"fluid_n{n}", dict(numeric_settings, kind="fluid", n=n, tol=tol),
                      lambda n=n: N.solve_fluid(parameters, rho, n, tol))
        fluids[n] = checked(f, "fluid", n)
    fluid_n = levels[-1]
    fluid = fluids[fluid_n]
    fluid_value_change = common_change(fluids[levels[-2]], fluid, "value")
    evaluations, stochastic = {}, {}

    def evaluate(n, f_n):
        key = (n, f_n)
        if key not in evaluations:
            grid = np.linspace(0, 1, n)
            alpha = N.transfer_policy(fluids[f_n], grid, config["policy_action_tolerance"])
            policy_hash = sha256(alpha.tobytes()).hexdigest()
            result = store.get(f"evaluate_s{n}_f{f_n}",
                dict(numeric_settings, kind="evaluation", n=n, fluid_n=f_n, policy_sha256=policy_hash),
                lambda: N.evaluate_stochastic_policy(parameters, rho, grid, alpha))
            evaluations[key] = checked(result, "evaluation", n, f_n)
        return evaluations[key]

    # Freeze the finest fluid candidate while refining the original kernel.
    for n in levels:
        opt = store.get(f"optimal_n{n}", dict(numeric_settings, kind="optimal", n=n, tol=tol),
                        lambda n=n: N.solve_stochastic(parameters, rho, n, tol))
        opt = checked(opt, "optimal", n)
        fixed = evaluate(n, fluid_n)
        loss = opt["value"]-fixed["value"]
        allowance = opt["value_error_bound"]+fixed["value_error_bound"]+1e-10
        if loss.min() < -allowance:
            raise ValueError(f"Negative policy loss beyond solver error: {common}, n={n}")
        stochastic[n] = dict(grid=opt["grid"], value_star=opt["value"],
                             value_policy=fixed["value"], loss=loss, optimal=opt, fixed=fixed)

    original_n = levels[-1]
    original_optimum = stochastic[original_n]["value_star"]
    # Freeze ONE original stochastic kernel while refining the fluid candidate.
    for nc, nf in zip(levels[:-1], levels[1:]):
        fc, ff = fluids[nc], fluids[nf]
        pc, pf = evaluate(original_n, nc), evaluate(original_n, nf)
        policy_changes = comparison_metrics(pc["grid"], pc["value"]-pf["value"], x0)
        maximum_loss_change = abs(float(np.max(abs(original_optimum-pc["value"])))
                                  -float(np.max(abs(original_optimum-pf["value"]))))
        root_change = (max((abs(a-b) for a, b in zip(fc["roots"], ff["roots"])), default=0.)
                       if len(fc["roots"]) == len(ff["roots"]) else None)
        tables["fluid_refinement"].append(dict(**common, n_coarse=nc, n_fine=nf,
            value_change_sup=common_change(fc, ff, "value"),
            **{f"policy_value_change_{key}": value for key, value in policy_changes.items()},
            maximum_loss_change=maximum_loss_change, threshold_change_max=root_change,
            threshold_count_coarse=len(fc["roots"]), threshold_count_fine=len(ff["roots"])))

    differences = []
    for nc, nf in zip(levels[:-1], levels[1:]):
        coarse, fine = stochastic[nc], stochastic[nf]
        loss_changes = comparison_metrics(coarse["grid"],
            coarse["loss"]-np.interp(coarse["grid"], fine["grid"], fine["loss"]), x0)
        row = dict(**common, n_coarse=nc, n_fine=nf,
            value_star_change_sup=common_change(coarse, fine, "value_star"),
            value_policy_change_sup=common_change(coarse, fine, "value_policy"),
            loss_change_sup=loss_changes["sup"], loss_change_x0=loss_changes["x0"],
            loss_change_l1=loss_changes["l1"],
            maximum_loss_change=abs(float(np.max(abs(coarse["loss"])))
                                    -float(np.max(abs(fine["loss"])))))
        differences.append(row)
        tables["grid_refinement"].append(row)

    solved = stochastic[original_n]
    x, val, policy_val, loss = (solved[key] for key in ("grid", "value_star", "value_policy", "loss"))
    previous_policy = evaluate(original_n, levels[-2])
    candidate_changes = comparison_metrics(x, previous_policy["value"]-policy_val, x0)
    v0 = float(np.interp(x0, x, val))
    summary = dict(**common, n_stochastic=original_n, n_fluid=fluid_n,
        loss_sup=float(np.max(abs(loss))), loss_min=float(loss.min()),
        loss_l1=float(np.trapezoid(abs(loss), x)), loss_x0=float(np.interp(x0, x, loss)),
        value_star_x0=v0, value_policy_x0=float(np.interp(x0, x, policy_val)),
        solver_value_error_bound=solved["optimal"]["value_error_bound"]+solved["fixed"]["value_error_bound"],
        stochastic_grid_change_sup=differences[-1]["value_star_change_sup"],
        policy_grid_change_sup=differences[-1]["value_policy_change_sup"],
        loss_grid_change_sup=differences[-1]["loss_change_sup"],
        loss_grid_change_max_last_two=max(d["loss_change_sup"] for d in differences[-2:]),
        loss_grid_change_x0=differences[-1]["loss_change_x0"],
        loss_grid_change_l1=differences[-1]["loss_change_l1"],
        loss_grid_change_x0_max_last_two=max(d["loss_change_x0"] for d in differences[-2:]),
        loss_grid_change_l1_max_last_two=max(d["loss_change_l1"] for d in differences[-2:]),
        maximum_loss_grid_change_max_last_two=max(d["maximum_loss_change"] for d in differences[-2:]),
        fluid_value_grid_change_sup=fluid_value_change,
        fluid_policy_value_change_sup=candidate_changes["sup"],
        fluid_policy_value_change_x0=candidate_changes["x0"],
        fluid_policy_value_change_l1=candidate_changes["l1"],
        fluid_candidate_maximum_loss_change=abs(float(np.max(abs(loss)))
            -float(np.max(abs(val-previous_policy["value"])))))
    summary["relative_loss_x0"] = summary["loss_x0"]/v0 if v0 > 1e-12 else None
    tables["original_summary"].append(summary)
    report_grid = np.linspace(0, 1, config["report_nodes"])
    columns = dict(value_star=np.interp(report_grid, x, val),
                   value_policy=np.interp(report_grid, x, policy_val),
                   loss=np.interp(report_grid, x, loss))
    transfer = N.transfer_policy(fluid, report_grid, config["policy_action_tolerance"])
    nearest = np.rint(report_grid*(original_n-1)).astype(int)
    tables["original_curves"] = [dict(**common, x=float(state),
        **{key: float(values[i]) for key, values in columns.items()},
        fluid_policy=int(transfer[i]), stochastic_policy=int(solved["optimal"]["policy"][nearest[i]]))
        for i, state in enumerate(report_grid)]
    atomic_json(folder/"status.json", dict(status="complete", p0=p0, rho=rho,
        fluid_grids=levels, stochastic_grids=levels, original=summary))
    return tables


def assessments(config, tables):
    threshold = config["assessment"]
    original_rows = []
    for row in tables["original_summary"]:
        def tolerance(scale):
            return max(threshold["absolute_sensitivity_floor"],
                       threshold["original_sensitivity_fraction"]*abs(scale))
        limit, point_limit, l1_limit = (tolerance(row[key]) for key in ("loss_sup", "loss_x0", "loss_l1"))
        original_rows.append(dict(p0=row["p0"], rho=row["rho"], loss_sup=row["loss_sup"],
            final_loss_grid_change=row["loss_grid_change_sup"], last_two_loss_grid_changes=row["loss_grid_change_max_last_two"],
            fluid_policy_value_change=row["fluid_policy_value_change_sup"], comparison_tolerance=limit,
            loss_stable_at_initial_state=row["loss_grid_change_x0_max_last_two"] <= point_limit
                and row["fluid_policy_value_change_x0"] <= point_limit,
            loss_stable_in_l1=row["loss_grid_change_l1_max_last_two"] <= l1_limit
                and row["fluid_policy_value_change_l1"] <= l1_limit,
            maximum_loss_stable=row["maximum_loss_grid_change_max_last_two"] <= limit
                and row["fluid_candidate_maximum_loss_change"] <= limit,
            loss_stable_at_tested_grids=row["loss_grid_change_max_last_two"] <= limit
                and row["fluid_policy_value_change_sup"] <= limit))
    return dict(original=original_rows, interpretation=
        "Spatial sensitivity of fluid-policy loss in the original stochastic model only. "
        "Refinement differences are diagnostics, not rigorous spatial error bounds. "
        "A stable positive policy loss need not vanish with grid refinement.")


def write_summary(output, config, report):
    original = pd.read_csv(output/"original_summary.csv")
    rows = report["assessments"]["original"]
    lines = ["# Política fluida en el monopolio estocástico con forgetting", "",
        "Experimento en la frontera madura x∈[0,1], con horizonte infinito. "
        "Se mantienen fijos rho, p0, gamma, las transiciones y el beneficio por período.", "",
        "Se calcula L(x)=V*(x)−V^pi_fluida(x), evaluando ambos valores en el modelo estocástico original.", "",
        f"Se calcularon {len(original)} combinaciones de parámetros, con grillas "
        + ", ".join(str(n+1) for n in config["grid_intervals"])+" nodos.", "",
        f"Pérdida máxima entre los casos y estados: {original.loss_sup.max():.8g}. "
        f"Mayor cambio de la pérdida entre las dos últimas grillas estocásticas: {original.loss_grid_change_sup.max():.8g}.", "",
        "## Estabilidad espacial observada", "",
        f"- Uniforme sobre los nodos comparados: {sum(r['loss_stable_at_tested_grids'] for r in rows)}/{len(rows)} casos.",
        f"- En el estado inicial x₀={config['initial_x']:g}: {sum(r['loss_stable_at_initial_state'] for r in rows)}/{len(rows)} casos.",
        f"- En L1 respecto de x: {sum(r['loss_stable_in_l1'] for r in rows)}/{len(rows)} casos.",
        f"- Del escalar de pérdida máxima: {sum(r['maximum_loss_stable'] for r in rows)}/{len(rows)} casos.", "",
        "Los controles refinan por separado la grilla estocástica con una candidata fluida congelada y la grilla fluida sobre un kernel estocástico fijo. "
        "La candidata binaria es discontinua en sus umbrales. Desplazar un umbral puede mantener una diferencia grande en norma supremo aunque la región afectada se estreche; por eso se informan también x₀ y L1.", "",
        "Una pérdida positiva que se estabiliza representa el coste de usar la política fluida en este problema. "
        "Las diferencias entre grillas son diagnósticos de sensibilidad y no cotas rigurosas del error espacial. Los casos que no cumplen el criterio quedan sin declarar estables.", "",
        "## Archivos", "",
        "- `original_summary.csv`, `original_curves.csv`: pérdidas y valores del problema original.",
        "- `grid_refinement.csv`, `fluid_refinement.csv`: controles separados de ambas grillas.",
        "- `solver_diagnostics.csv`: residuos, certificados e iteraciones.",
        "- `figures/`: cinco figuras PNG/PDF; `checkpoints/`: soluciones y evaluaciones reutilizables.",
        "- `manifest.json`: configuración, criterios, huellas y evaluaciones por caso.", ""]
    (output/"summary.md").write_text("\n".join(lines))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--figures-only", action="store_true")
    args = parser.parse_args(argv)
    if args.figures_only and args.recompute:
        parser.error("Choose --figures-only or --recompute")
    CANCEL.clear()
    config = json.loads(args.config.read_text())
    if args.quick:
        config.update(p0_values=[.5], rhos=[.7, .9], grid_intervals=[250, 500, 1000], report_nodes=501)
    if args.workers is not None:
        config["workers"] = args.workers
    validate_config(config)
    output = (args.output_dir or (DEFAULT_OUTPUT/"quick" if args.quick else DEFAULT_OUTPUT)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    numerical_paths = [Path(N.__file__), ROOT/"forgetting/Code/model.py"]
    numerical_sources = {str(p.relative_to(ROOT)): digest(p) for p in numerical_paths}
    science = {k: v for k, v in config.items() if k != "workers"}
    fingerprint = identity(dict(configuration=science, numerical_sources=numerical_sources))
    manifest_path = output/"manifest.json"
    if manifest_path.exists() and not args.recompute:
        previous = json.loads(manifest_path.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError("Scientific configuration/code changed; use a separate output or --recompute")
    report = dict(configuration=config, fingerprint=fingerprint, numerical_sources=numerical_sources,
                  started_utc=datetime.now(timezone.utc).isoformat(), status="running")
    if args.figures_only:
        report = json.loads(manifest_path.read_text())
        if report["status"] != "complete":
            raise ValueError("Need a complete saved campaign to regenerate figures")
        for name, expected in report["data_sha256"].items():
            if digest(output/name) != expected:
                raise ValueError(f"Saved data changed: {name}")
    else:
        report["data_generator_sha256"] = {str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))}
        atomic_json(manifest_path, report)
        tables = {name: [] for name in TABLES}
        tasks = [(p0, rho) for p0 in config["p0_values"] for rho in config["rhos"]]
        executor = ThreadPoolExecutor(max_workers=config["workers"])
        futures = {executor.submit(case_worker, config, output, p0, rho, numerical_sources, args.recompute): (p0, rho)
                   for p0, rho in tasks}
        try:
            for count, future in enumerate(as_completed(futures), 1):
                result = future.result()
                for name in TABLES:
                    tables[name].extend(result[name])
                p0, rho = futures[future]
                print(f"[{count}/{len(tasks)}] verified p0={p0:g}, rho={rho:g}", flush=True)
        except BaseException:
            CANCEL.set()
            executor.shutdown(wait=True, cancel_futures=True)
            report["status"] = "interrupted_or_failed"
            atomic_json(manifest_path, report)
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        for name, rows in tables.items():
            df = pd.DataFrame(rows)
            sort_keys = [key for key in ("p0", "rho", "n_stochastic", "n_coarse", "n_fine", "x") if key in df]
            df.sort_values(sort_keys, na_position="last").to_csv(output/f"{name}.csv", index=False)
        report["assessments"] = assessments(config, tables)
        report["data_sha256"] = {f"{name}.csv": digest(output/f"{name}.csv") for name in TABLES}
        report["status"] = "complete"
        report["completed_utc"] = datetime.now(timezone.utc).isoformat()
        write_summary(output, config, report)
    from forgetting.Code.fluid_convergence.figures import make_figures
    figures = make_figures(output)
    code_paths = [Path(__file__), Path(N.__file__), Path(__file__).with_name("figures.py")]
    report["source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in code_paths}
    report["plotting_source_sha256"] = {str(Path(__file__).with_name("figures.py").relative_to(ROOT)): digest(Path(__file__).with_name("figures.py"))}
    report["input_sha256"] = {str(args.config.resolve()): digest(args.config)}
    report["figures"] = {str(p.relative_to(output)): digest(p) for p in figures}
    report["summary_sha256"] = digest(output/"summary.md")
    atomic_json(manifest_path, report)
    print(f"Saved study and {len(figures)//2} PNG/PDF figures: {output}")


if __name__ == "__main__":
    main()
