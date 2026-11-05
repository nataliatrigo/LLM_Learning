"""Dense comparison of exact-Beta and logistic mature-boundary policies.

uv run python -B forgetting/Code/visualizations/compare_policy_geometry.py --workers 4
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

from compare_beta_logistic_demand import (
    ROOT, P, TOLERANCE, solve_pair, describe, advantage, interval_text,
)

DEST = ROOT / "forgetting/outputs/beta_logistic_policy_geometry_p0_0.5"
INITIAL_NODES = 5001
ENDPOINT_TOL = 2e-5
CRITICAL_RHO_TOL = 5e-8
ROOT_XTOL = 2e-10
MODELS = ("exact", "logistic")


def rho_grid():
    # Integer construction avoids float duplicates at the overlap of both grids.
    return np.unique(np.concatenate((np.arange(5000, 9991, 10), np.arange(9800, 9996)))) / 10000


def settings():
    return dict(version=1, primitives=P.as_dict(), initial_nodes=INITIAL_NODES,
        iteration_tolerance=TOLERANCE, endpoint_tolerance=ENDPOINT_TOL,
        critical_rho_grid_tolerance=CRITICAL_RHO_TOL, root_xtol=ROOT_XTOL,
        helper_sha256=hashlib.sha256(Path(__file__).with_name("compare_beta_logistic_demand.py").read_bytes()).hexdigest(),
        bellman_sha256=hashlib.sha256((ROOT / "forgetting/Code/model.py").read_bytes()).hexdigest())


def summarize_pair(pair):
    output = {}
    for model, sol in pair.items():
        data = describe(sol)
        data.update(advantage_at_zero=float(advantage(sol, np.array([0.0]))[0]),
            grid_nodes=len(sol.x), iterations=sol.iterations, bellman_residual=sol.bellman_residual)
        data["touches_zero"] = data["advantage_at_zero"] > 0
        intervals = data["positive_intervals"]
        # xL/xH only describe a single component; never bridge disconnected sets.
        data["xL"] = intervals[0]["lower"] if len(intervals) == 1 else None
        data["xH"] = intervals[0]["upper"] if len(intervals) == 1 else None
        output[model] = data
    return output


def geometry_check(old, new):
    roots_old, roots_new = old["zero_crossings"], new["zero_crossings"]
    same_count = len(roots_old) == len(roots_new)
    movement = max((abs(a-b) for a, b in zip(roots_old, roots_new)), default=0.0) if same_count else None
    same_geometry = (same_count and old["positive_region_count"] == new["positive_region_count"]
        and old["structure"] == new["structure"] and old["touches_zero"] == new["touches_zero"]
        and ((old["max_advantage"] > 0) == (new["max_advantage"] > 0)))
    return dict(coarse_nodes=old["grid_nodes"], fine_nodes=new["grid_nodes"],
        same_geometry=same_geometry, root_movement=movement,
        max_advantage_change=abs(old["max_advantage"]-new["max_advantage"]),
        advantage_at_zero_change=abs(old["advantage_at_zero"]-new["advantage_at_zero"]),
        argmax_change=abs(old["argmax"]-new["argmax"]),
        stable=bool(same_geometry and movement <= ENDPOINT_TOL))


def near_transition(summary):
    for data in summary.values():
        if abs(data["max_advantage"]) < 0.015 or abs(data["advantage_at_zero"]) < 0.015:
            return True
        if data["positive_region_count"] > 1 or any(0 < x < 0.015 for x in data["zero_crossings"]):
            return True
    return False


def run_dense_case(rho):
    nodes = INITIAL_NODES
    coarse = summarize_pair(solve_pair(rho, nodes))
    history = []
    while nodes < 160001:
        nodes = 2 * nodes - 1
        fine = summarize_pair(solve_pair(rho, nodes))
        checks = {model: geometry_check(coarse[model], fine[model]) for model in MODELS}
        history.append(checks)
        minimum_nodes = 20001 if near_transition(fine) or near_transition(coarse) else 10001
        if all(row["stable"] for row in checks.values()) and nodes >= minimum_nodes:
            return dict(rho=float(rho), models=fine, grid_history=history)
        coarse = fine
    raise RuntimeError(f"Policy geometry has not stabilized at rho={rho:g}")


def flatten(case):
    row = {"rho":case["rho"]}
    aliases = {"num_positive_components":"positive_region_count", "xL":"xL", "xH":"xH",
               "max_advantage":"max_advantage", "argmax":"argmax", "advantage_at_zero":"advantage_at_zero",
               "touches_zero":"touches_zero", "structure":"structure", "grid_nodes":"grid_nodes",
               "bellman_residual":"bellman_residual"}
    for model in MODELS:
        data = case["models"][model]
        for label, field in aliases.items():
            row[f"{model}_{label}"] = data[field]
        row[f"{model}_zero_crossings"] = json.dumps(data["zero_crossings"])
        row[f"{model}_positive_intervals"] = interval_text(data["positive_intervals"], digits=10)
        row[f"{model}_all_components"] = json.dumps(data["positive_intervals"])
    return row


def make_table(cases):
    table = pd.DataFrame([flatten(case) for case in cases]).sort_values("rho").reset_index(drop=True)
    requested = ["rho", "exact_num_positive_components", "logistic_num_positive_components", "exact_xL", "exact_xH",
        "logistic_xL", "logistic_xH", "exact_max_advantage", "logistic_max_advantage", "exact_argmax",
        "logistic_argmax", "exact_advantage_at_zero", "logistic_advantage_at_zero"]
    return table[requested + [name for name in table if name not in requested]]


def brackets_for(table):
    events = []
    for model in MODELS:
        for kind, field in (("B", "advantage_at_zero"), ("H", "max_advantage")):
            values = table[f"{model}_{field}"].to_numpy()
            rhos = table.rho.to_numpy()
            for i in range(len(rhos)-1):
                if values[i] * values[i+1] < 0:
                    events.append(dict(model=model, kind=kind, field=field,
                        lower=float(rhos[i]), upper=float(rhos[i+1]),
                        direction="positive_to_negative" if values[i] > 0 else "negative_to_positive"))
                elif values[i] == 0:
                    j = max(i-1, 0)
                    events.append(dict(model=model, kind=kind, field=field, lower=float(rhos[j]),
                        upper=float(rhos[i+1]), direction="sampled_zero"))
    return events


def refine_event(event):
    """Refine every detected sign-change event, checking x-grid stability too."""
    traces, grid_results = [], []
    lower, upper = event["lower"], event["upper"]
    model, field = event["model"], event["field"]
    previous = None
    for nodes in (20001, 40001, 80001, 160001):
        cache = {}

        def evaluate(rho):
            key = float(rho)
            if key not in cache:
                summary = summarize_pair(solve_pair(key, nodes))
                cache[key] = summary
                traces.append(dict(rho=key, grid_nodes=nodes, models=summary))
            return cache[key][model][field]

        f_lower, f_upper = evaluate(lower), evaluate(upper)
        if f_lower * f_upper > 0:
            raise RuntimeError(f"Critical event no longer bracketed on finer x grid: {event}, nodes={nodes}")
        root = float(brentq(evaluate, lower, upper, xtol=ROOT_XTOL, rtol=1e-13))
        residual = float(evaluate(root))
        # Preserve a verified sign bracket around the estimated numerical root.
        radius = 2e-9
        for _ in range(12):
            lo, hi = max(lower, root-radius), min(upper, root+radius)
            flo, fhi = evaluate(lo), evaluate(hi)
            if flo * fhi <= 0:
                break
            radius *= 2
        else:
            raise RuntimeError("Could not verify the refined root bracket")
        movement = abs(root-previous) if previous is not None else None
        grid_results.append(dict(grid_nodes=nodes, rho=root, function_residual=residual,
            root_grid_change=movement, bracket_lower=lo, bracket_upper=hi,
            value_at_lower=float(flo), value_at_upper=float(fhi)))
        if previous is not None and movement <= CRITICAL_RHO_TOL:
            return {**event, "rho":root, "grid_nodes":nodes, "root_grid_change":movement,
                "grid_results":grid_results, "evaluations":traces}
        previous = root
    raise RuntimeError(f"Critical rho not grid-stable: {event}")


def threshold_table(events):
    rows = []
    for model in MODELS:
        row = {"model":model}
        for kind in ("B", "H"):
            selected = [event for event in events if event["model"] == model and event["kind"] == kind
                        and event["direction"] in ("positive_to_negative", "sampled_zero")]
            row[f"rho_{kind}"] = selected[0]["rho"] if len(selected) == 1 else None
            row[f"rho_{kind}_event_count"] = len(selected)
            row[f"rho_{kind}_all_events"] = json.dumps([event["rho"] for event in selected])
            if len(selected) == 1:
                event = selected[0]
                last = event["grid_results"][-1]
                row[f"rho_{kind}_grid_change"] = event["root_grid_change"]
                row[f"rho_{kind}_bracket_lower"] = last["bracket_lower"]
                row[f"rho_{kind}_bracket_upper"] = last["bracket_upper"]
                row[f"rho_{kind}_grid_nodes"] = event["grid_nodes"]
        rows.append(row)
    return pd.DataFrame(rows)


def collect_dense(rhos, destination, workers, config):
    cache_dir = destination / "cache"
    cache_dir.mkdir(exist_ok=True)
    cases, missing = [], []
    for rho in rhos:
        path = cache_dir / f"rho_{rho:.10f}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["settings"] == config:
                cases.append(saved["result"])
                continue
        missing.append(float(rho))
    print(f"Dense scan: {len(rhos)} rho values, {len(cases)} cached, {len(missing)} to solve", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(run_dense_case, rho):rho for rho in missing}
        for count, job in enumerate(as_completed(jobs), 1):
            rho = jobs[job]
            case = job.result()
            cases.append(case)
            (cache_dir / f"rho_{rho:.10f}.json").write_text(json.dumps(dict(settings=config, result=case)) + "\n")
            if count % 25 == 0 or count == len(missing):
                print(f"Completed {count}/{len(missing)} new rho values; latest rho={rho:g}", flush=True)
    return sorted(cases, key=lambda case:case["rho"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=DEST)
    parser.add_argument("--compute-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("Use between 1 and 8 workers")
    destination = args.output_dir
    for folder in ("data", "figures"):
        (destination / folder).mkdir(parents=True, exist_ok=True)
    config = settings()
    cases = collect_dense(rho_grid(), destination, args.workers, config)
    dense = make_table(cases)
    dense.to_csv(destination / "data/dense_summary.csv", index=False)
    history = [{"rho":case["rho"], "model":model, **row[model]}
               for case in cases for row in case["grid_history"] for model in MODELS]
    pd.DataFrame(history).to_csv(destination / "data/grid_convergence.csv", index=False)
    events = brackets_for(dense)
    (destination / "data/coarse_event_brackets.json").write_text(json.dumps(events, indent=2) + "\n")
    print("Detected event brackets: " + json.dumps(events), flush=True)
    refined = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(refine_event, event) for event in events]
        for job in as_completed(jobs):
            event = job.result()
            refined.append(event)
            print(f"Refined {event['model']} rho_{event['kind']}={event['rho']:.10f}, grid change={event['root_grid_change']:.2e}", flush=True)
            (destination / "data/critical_events.json").write_text(json.dumps(refined, indent=2) + "\n")
    thresholds = threshold_table(refined)
    thresholds.to_csv(destination / "data/critical_rates.csv", index=False)
    # Add samples between the models' distinct thresholds, so short policy
    # disagreements cannot be hidden by the uniform rho grid.
    extra_rhos = []
    for kind in ("B", "H"):
        roots = [event["rho"] for event in refined if event["kind"] == kind]
        for root in roots:
            extra_rhos.extend(root + step for step in (-1e-5, -1e-6, 1e-6, 1e-5))
        if len(roots) > 1:
            extra_rhos.extend(np.linspace(min(roots), max(roots), 17)[1:-1])
    extra_rhos = sorted(set(float(rho) for rho in extra_rhos if 0.5 <= rho <= 0.9995) - set(dense.rho))
    extras = collect_dense(extra_rhos, destination, args.workers, config)
    all_cases = cases + extras
    combined = make_table(all_cases)
    combined.to_csv(destination / "data/policy_summary.csv", index=False)
    all_history = [{"rho":case["rho"], "model":model, **row[model]}
                   for case in all_cases for row in case["grid_history"] for model in MODELS]
    pd.DataFrame(all_history).to_csv(destination / "data/grid_convergence.csv", index=False)
    (destination / "data/all_policy_regions.json").write_text(json.dumps([
        {"rho":case["rho"], "models":case["models"]} for case in all_cases], indent=2) + "\n")
    manifest = {**config, "base_rho_count":len(dense), "additional_rho_count":len(extras),
        "total_rho_count":len(combined), "rho_domain":[0.5,0.9995], "base_rho_step":0.001,
        "tail_rho_step":0.0001, "tail_start":0.98, "workers":args.workers,
        "demand_only_model_change":True, "initial_value":"zero", "interpolation":"linear numpy.interp",
        "root_method":"Brent bracketed root finding, with an explicit verified sign bracket and x-grid refinement",
        "endpoint_convention":"Strict positivity: exclude zero crossings; touches_zero means a(0)>0. xL/xH are null for empty or multiple-component regions; all components are preserved separately.",
        "numerical_scope":"Finite sampled rho range and numerical root refinement; not a proof for every rho >= 0.5.",
        "runner_source":str(Path(__file__).resolve())}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(thresholds.to_string(index=False), flush=True)
    print(f"Saved {len(combined)} parameter pairs to {destination}", flush=True)
    if not args.compute_only:
        from plot_policy_geometry_comparison import generate
        generate(destination)


if __name__ == "__main__":
    main()
