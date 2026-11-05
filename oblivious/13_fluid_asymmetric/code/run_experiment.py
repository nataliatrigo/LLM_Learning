"""Fresh crossed HJB solves, spatial/demand checks and realized joint paths."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluid-asymmetric-mpl")

import numpy as np
import pandas as pd

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.fluid_oblivious.code.model import FixedRivalDemand
from oblivious.fluid_oblivious.code.trajectories import simulate_pair
from oblivious.fluid_oblivious_initial_conditions.code.initial_conditions import phase_limit
from oblivious.fluid_asymmetric.code.model import (
    analyse_pair, compare_refinement, independent_transformed_demand, simulate_pure_pair_log,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parents[1]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def tag(p0, rho):
    return f"p0_{p0:g}_rho_{rho:g}".replace(".", "p")


def demand_validation(params, p0, case, config):
    """Full-grid Jacobi doubling and independent direct Legendre integration."""
    grid = np.linspace(0, 1, case["grid_nodes"])
    low = FixedRivalDemand(params, p0, grid, config["quadrature_order"])
    high = FixedRivalDemand(params, p0, grid, config["validation_quadrature_order"])
    rows = []
    for label in ("response_a", "response_b"):
        z = case[label]["conjecture"]
        first, second = low(z), high(z)
        independent = independent_transformed_demand(params, p0, z, grid,
            order=config["validation_quadrature_order"])
        transformed_doubled = independent_transformed_demand(params, p0, z, grid,
            order=2*config["validation_quadrature_order"])
        doubling = float(np.max(abs(first-second)))
        disagreement = float(np.max(abs(second-independent)))
        transformed_change = float(np.max(abs(independent-transformed_doubled)))
        rows.append(dict(seller=label[-1].upper(), conjecture=z,
                         maximum_demand_change_on_doubling=doubling,
                         maximum_independent_demand_disagreement=disagreement,
                         maximum_transformed_demand_change=transformed_change,
                         passed=bool(max(doubling, disagreement, transformed_change)
                                     <= config["quadrature_demand_absolute"])))
    return dict(method="Full state grid; original positive Jacobi quadrature doubled; "
                       "independent direct Legendre integral after t=1-(1-p0)*u^4; "
                       "transformed Legendre quadrature also doubled", checks=rows,
                passed=all(row["passed"] for row in rows))


def run_path(params, p0, pair, x0, config, *, refined=False, swapped=False):
    settings = config["trajectories"]
    a, b = pair["response_a"], pair["response_b"]
    if swapped:
        a, b = b, a
    return simulate_pair(params, p0, a, response_b=b, initial=(x0, x0),
        times=np.linspace(0, settings["horizon"], settings["samples"]),
        bin_width=settings["bin_width"],
        rtol=settings["rtol"]/(10 if refined else 1),
        atol=settings["atol"]/(10 if refined else 1),
        max_step=settings["max_step"]/(2 if refined else 1),
        demand_order=config["validation_quadrature_order"]*(2 if refined else 1))


def simulate_case(params, p0, pair, config):
    intervals = pair["common_initial_intervals"]
    if intervals:
        interval = max(intervals, key=lambda row: row["hi"]-row["lo"])
        starts = [interval["lo"]+fraction*(interval["hi"]-interval["lo"])
                  for fraction in (.5, .25, .75)]
    else:
        # Control in the fourth symmetric uncovered interval, away from ties.
        starts = [.4]
    frames, bins, summaries = [], [], []
    for x0 in starts:
        result = run_path(params, p0, pair, x0, config)
        frame = result["by_time"]
        limit_a = phase_limit(params, pair["response_a"], x0)
        limit_b = phase_limit(params, pair["response_b"], x0)
        summary = dict(x0=float(x0), **result["diagnostics"],
            terminal_a=float(frame.x_a.iloc[-1]), terminal_b=float(frame.x_b.iloc[-1]),
            asymptotic_a=float(limit_a), asymptotic_b=float(limit_b),
            terminal_error_a=float(abs(frame.x_a.iloc[-1]-limit_a)),
            terminal_error_b=float(abs(frame.x_b.iloc[-1]-limit_b)))
        monotonic = True
        no_overshoot = True
        for label, limit in (("a", limit_a), ("b", limit_b)):
            path = frame[f"x_{label}"].to_numpy()
            direction = np.sign(limit-x0)
            monotonic &= bool(np.min(direction*np.diff(path)) >= -1e-9)
            no_overshoot &= bool(np.min(path) >= min(x0, limit)-1e-9
                                 and np.max(path) <= max(x0, limit)+1e-9)
        # Each positive case uses one pure product throughout the path.
        # Its independent demand-clock identity is x(t)=p_i+(x0-p_i)e^-lambda*sales.
        clock_error = None
        if pair["qualifies"]:
            lam = -math.log(params.rho)
            exact_a = params.p1+(x0-params.p1)*np.exp(-lam*frame.cumulative_own_sales)
            exact_b = params.p2+(x0-params.p2)*np.exp(-lam*frame.cumulative_b_sales)
            clock_error = float(max(np.max(abs(frame.x_a-exact_a)), np.max(abs(frame.x_b-exact_b))))
        summary.update(monotonic=monotonic, no_overshoot=no_overshoot,
                       demand_clock_identity_error=clock_error,
                       passed=bool(monotonic and no_overshoot
                                   and (clock_error is None or clock_error <= 1e-7)))
        summaries.append(summary)
        frames.append(frame.assign(p0=p0, rho=params.rho, x0=x0))
        bins.append(result["binned"].assign(p0=p0, rho=params.rho, x0=x0))
    baseline = frames[0]
    refined = run_path(params, p0, pair, starts[0], config, refined=True)["by_time"]
    error = float(np.max(abs(baseline[["x_a", "x_b"]].to_numpy()-refined[["x_a", "x_b"]].to_numpy())))
    swapped = run_path(params, p0, pair, starts[0], config, swapped=True)["by_time"]
    swap_error = float(np.max(abs(baseline[["x_a", "x_b"]].to_numpy()-swapped[["x_b", "x_a"]].to_numpy())))
    validation = dict(maximum_path_change_on_refinement=error,
                      maximum_label_swap_error=swap_error,
                      refined_rtol=config["trajectories"]["rtol"]/10,
                      refined_max_step=config["trajectories"]["max_step"]/2,
                      refined_demand_order=config["validation_quadrature_order"]*2,
                      passed=bool(error <= 1e-7 and swap_error <= 1e-8
                                  and all(row["passed"] for row in summaries)))
    return summaries, validation, frames, bins


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"config.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--figures-only", action="store_true")
    parser.add_argument("--recheck-demand", action="store_true",
        help="Recompute only the independent demand audit for existing compatible results")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    output = args.output_dir.resolve()
    data = output/"data"
    data.mkdir(parents=True, exist_ok=True)
    if args.figures_only or args.recheck_demand:
        saved = json.loads((data/"results.json").read_text())
        if saved["config"] != config:
            raise ValueError("Saved scientific configuration differs from --config")
        cases = saved["cases"]
        trajectories = pd.read_csv(data/"trajectories.csv")
        long_trajectories = pd.read_csv(data/"long_horizon_trajectories.csv")
        if args.recheck_demand:
            manifest_path = data/"manifest.json"
            manifest = json.loads(manifest_path.read_text())
            for name, digest in manifest["source_sha256"].items():
                if (not name.startswith("oblivious/13_fluid_asymmetric/")
                        and sha256((PROJECT/name).read_bytes()).hexdigest() != digest):
                    raise ValueError(f"The original numerical source changed: {name}; recompute the experiment")
            before = [dict(p0=c["p0"], rho=c["rho"], demand=c["numeric_validation"]["demand"]) for c in cases]
            archive = data/"demand_audit_before_endpoint_transform.json"
            if not archive.exists():
                write_json(archive, before)
            with ProcessPoolExecutor(max_workers=min(2, len(cases))) as pool:
                futures = [pool.submit(demand_validation,
                    Parameters(**config["parameters"], rho=case["rho"]), case["p0"], case, config)
                    for case in cases]
                for case, future in zip(cases, futures):
                    validation = case["numeric_validation"]
                    validation["demand"] = future.result()
                    validation["passed"] = bool(validation["demand"]["passed"] and validation["integration"]["passed"]
                        and case["refinement"]["passed"] and case.get("long_horizon", {}).get("passed", True))
                    case["verified"] = bool(case["qualifies"] and validation["passed"])
                    print(f"Rechecked demand: p0={case['p0']}, rho={case['rho']}, "
                          f"passed={validation['demand']['passed']}", flush=True)
            saved["all_numeric_checks_passed"] = all(c["numeric_validation"]["passed"] for c in cases)
            write_json(data/"results.json", saved)
            summary = pd.read_csv(data/"summary.csv")
            summary["verified"] = [c["verified"] for c in cases]
            summary.to_csv(data/"summary.csv", index=False)
            manifest["demand_rechecked_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["config_sha256"] = sha256(args.config.read_bytes()).hexdigest()
            manifest["source_sha256"] = {name: sha256((PROJECT/name).read_bytes()).hexdigest()
                                         for name in manifest["source_sha256"]}
            manifest["results_sha256"] = sha256((data/"results.json").read_bytes()).hexdigest()
            write_json(manifest_path, manifest)
    else:
        cases, trajectories, all_bins, grid_records = [], [], [], []
        long_frames = []
        for spec in config["cases"]:
            p0, rho = spec["p0"], spec["rho"]
            params = Parameters(**config["parameters"], rho=rho)
            levels = []
            for n in config["grids"]:
                print(f"Solving crossed responses: p0={p0:g}, rho={rho:g}, n={n}", flush=True)
                pair = analyse_pair(params, p0, n=n, order=config["quadrature_order"])
                levels.append(pair)
                grid_records.append(pair)
                write_json(data/"grids"/(tag(p0, rho)+f"_n{n}.json"), pair)
            fine = levels[-1]
            comparisons = [compare_refinement(a, b, config["boundary_refinement_absolute"])
                           for a, b in zip(levels, levels[1:])]
            fine["refinement"] = dict(comparisons=comparisons, passed=all(row["passed"] for row in comparisons))
            print(f"  Common starts: {fine['common_initial_notation']}; checking demand and joint paths", flush=True)
            demand = demand_validation(params, p0, fine, config)
            simulations, integration, paths, bins = simulate_case(params, p0, fine, config)
            fine["simulations"] = simulations
            fine["numeric_validation"] = dict(demand=demand, integration=integration,
                passed=bool(demand["passed"] and integration["passed"] and fine["refinement"]["passed"]))
            if fine["qualifies"]:
                initial = simulations[0]["x0"]
                times = np.unique(np.r_[paths[0].time, np.geomspace(1., config["long_horizon"], 501)])
                long = simulate_pure_pair_log(params, p0, fine, initial, times,
                    order=config["validation_quadrature_order"])
                long_frame = pd.DataFrame({key: long[key] for key in ("time", "x_a", "x_b")})
                overlap = long_frame.iloc[np.searchsorted(times, paths[0].time)]
                overlap_error = float(np.max(abs(overlap[["x_a", "x_b"]].to_numpy()
                                                -paths[0][["x_a", "x_b"]].to_numpy())))
                terminal_error = float(max(abs(long["x_a"][-1]-params.p1), abs(long["x_b"][-1]-params.p2)))
                fine["long_horizon"] = dict(horizon=float(times[-1]),
                    terminal_a=float(long["x_a"][-1]), terminal_b=float(long["x_b"][-1]),
                    maximum_terminal_target_error=terminal_error, maximum_overlap_error=overlap_error,
                    nfev=long["nfev"], method=long["method"],
                    passed=bool(terminal_error <= 1e-7 and overlap_error <= 1e-7))
                fine["numeric_validation"]["passed"] &= fine["long_horizon"]["passed"]
                long_frames.append(long_frame.assign(p0=p0, rho=rho, x0=initial))
            fine["verified"] = bool(fine["qualifies"] and fine["numeric_validation"]["passed"])
            cases.append(fine)
            trajectories.extend(paths)
            all_bins.extend(bins)
            print(f"  qualifies={fine['qualifies']}, numerical checks={fine['numeric_validation']['passed']}", flush=True)
        trajectories = pd.concat(trajectories, ignore_index=True)
        trajectories.to_csv(data/"trajectories.csv", index=False)
        pd.concat(all_bins, ignore_index=True).to_csv(data/"trajectory_bins.csv", index=False)
        long_trajectories = (pd.concat(long_frames, ignore_index=True) if long_frames else
                            pd.DataFrame(columns=["time", "x_a", "x_b", "p0", "rho", "x0"]))
        long_trajectories.to_csv(data/"long_horizon_trajectories.csv", index=False)
        rows = []
        for case in cases:
            s = case["simulations"][0]
            rows.append(dict(p0=case["p0"], rho=case["rho"], qualifies=case["qualifies"],
                             verified=case["verified"], common_initial_states=case["common_initial_notation"],
                             x0=s["x0"], terminal_a=s["terminal_a"], terminal_b=s["terminal_b"],
                             asymptotic_a=s["asymptotic_a"], asymptotic_b=s["asymptotic_b"],
                             **{label+"_"+key: case["response_"+label]["diagnostics"][key]
                                for label in ("a", "b") for key in
                                ("hjb_residual", "holding_value_gap", "hamiltonian_gap", "target_optimal")}))
        pd.DataFrame(rows).to_csv(data/"summary.csv", index=False)
        write_json(data/"results.json", dict(config=config, cases=cases,
            interpretation="Numerical fixed-rival oblivious optimality plus joint-path consistency; "
                           "not a Nash verification against a moving rival during transition.",
            all_numeric_checks_passed=all(case["numeric_validation"]["passed"] for case in cases)))
        sources = [Path(__file__), ROOT/"code/model.py", ROOT/"code/reporting.py", ROOT/"config.json",
                   PROJECT/"oblivious/10_fluid_oblivious/code/model.py",
                   PROJECT/"oblivious/10_fluid_oblivious/code/trajectories.py",
                   PROJECT/"oblivious/11_fluid_oblivious_initial_conditions/code/initial_conditions.py",
                   PROJECT/"oblivious/11_fluid_oblivious_initial_conditions/code/independent_response.py",
                   PROJECT/"forgetting/Code/fluid_convergence/numerics.py",
                   PROJECT/"forgetting/Code/model.py",
                   PROJECT/"oblivious/02_oe_p0/code/primitives.py"]
        write_json(data/"manifest.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
            config=config, python_version=__import__("sys").version,
            versions={name: __import__(name).__version__ for name in ("numpy", "scipy", "pandas", "matplotlib")},
            source_sha256={str(path.relative_to(PROJECT)): sha256(path.read_bytes()).hexdigest()
                           for path in sources},
            config_path=str(args.config.resolve()),
            config_sha256=sha256(args.config.read_bytes()).hexdigest(),
            results_sha256=sha256((data/"results.json").read_bytes()).hexdigest(),
            historical_outputs_modified=False, new_hjb_solves=len(grid_records)*2))
    from oblivious.fluid_asymmetric.code.reporting import create_report
    artifacts = create_report(cases, trajectories, output, config, long_trajectories=long_trajectories)
    print(f"Saved {len(cases)} cases to {output}; {len(artifacts)} report artifacts", flush=True)
    if not all(case["numeric_validation"]["passed"] for case in cases):
        raise RuntimeError("Some numerical checks failed; inspect data/results.json")


if __name__ == "__main__":
    main()
