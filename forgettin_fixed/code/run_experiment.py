"""Reproduce exact count-state opposite-outcome replacement experiments.

Run from the repository root:
    .venv/bin/python -B -m forgettin_fixed.code.run_experiment --workers 3
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd

from forgetting.Code.model import Primitives
from .model import CountSolution, evidence_matched_n, solve_counts

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_name("config.json")
MODEL = "opposite_outcome_replacement"


def slug(p0: float, n: int) -> str:
    return f"p0_{p0:g}_n_{n}"


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_solution(path):
    return CountSolution.load(path)


def source_hashes():
    paths = [*sorted((ROOT / "code").glob("*.py")), CONFIG,
             ROOT.parent / "forgetting" / "Code" / "model.py"]
    return {os.path.relpath(p, ROOT): sha256(p) for p in paths}


def solve_case(task):
    p0, n, config, force = task
    path = ROOT / "data" / "policies" / f"{slug(p0,n)}.npz"
    primitives = Primitives.from_mapping({**config["seller"], "p0": p0})
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
        if (meta.get("format") == "opposite-replacement-v1" and
            meta["primitives"] == primitives.as_dict() and
            meta["action_tolerance"] == config["solver"]["action_tolerance"] and
            meta["diagnostics"].get("tolerance") == config["solver"]["tolerance"]):
            return {"case": slug(p0,n), "cached": True}
    started = time.monotonic()
    solution = solve_counts(primitives, n, **config["solver"])
    solution.save(path)
    dump(path.with_suffix(".json"), solution.diagnostics)
    return {"case": slug(p0,n), "seconds": round(time.monotonic()-started, 3),
            "mature_states": n+1}


def cache_matches(directory, signature, required, force):
    path = directory / "manifest.json"
    if force or not path.exists() or any(not (directory / name).exists() for name in required):
        return False
    return json.loads(path.read_text()).get("signature") == signature


def calendar_case(task):
    from .simulation import simulate_calendar
    p0, n, config, force = task
    label = slug(p0,n)
    path = ROOT / "data" / "policies" / f"{label}.npz"
    directory = ROOT / "data" / "calendar" / label
    args = {**config["calendar"], "seed": config["calendar"]["seed"] +
            10000 * config["p0_values"].index(p0)}
    signature = {"model": MODEL, "simulation": args, "policy_sha256": sha256(path)}
    if cache_matches(directory, signature, ["by_period.csv", "binned.csv", "path_summary.csv"], force):
        return {"calendar": label, "cached": True}
    started = time.monotonic()
    solution = load_solution(path)
    frames = simulate_calendar(solution, **args)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(("by_period", "binned", "path_summary"), frames, strict=True):
        frame.insert(0, "method", solution.method)
        frame.insert(0, "p0", p0)
        frame.insert(0, "n", n)
        frame.to_csv(directory / f"{name}.csv", index=False)
    dump(directory / "manifest.json", {"signature": signature, "seller": solution.primitives.as_dict(),
         "initial_state": "empty Beta(1,1)", "memory_clock": "A purchases only",
         "replacement": "new success replaces failure; new failure replaces success; saturated boundaries",
         "random_numbers": "common uniforms across capacities at fixed p0",
         "seconds": time.monotonic()-started})
    return {"calendar": label, "seconds": round(time.monotonic()-started, 2)}


def interaction_case(task):
    from .simulation import simulate_interactions
    from .figures import plot_interactions
    p0, n, config, force = task
    label = slug(p0,n)
    path = ROOT / "data" / "policies" / f"{label}.npz"
    directory = ROOT / "data" / "interactions" / label
    args = {k:v for k,v in config["interaction"].items() if k != "cases"}
    signature = {"model": MODEL, "simulation": args, "policy_sha256": sha256(path)}
    solution = load_solution(path)
    directory.mkdir(parents=True, exist_ok=True)
    if cache_matches(directory, signature, ["observations.npy", "path_summary.csv", "customer_1.csv", "customer_1_histogram.csv"], force):
        records = np.load(directory / "observations.npy", mmap_mode="r")
    else:
        records, summary = simulate_interactions(solution, **args)
        np.save(directory / "observations.npy", records)
        summary.to_csv(directory / "path_summary.csv", index=False)
        customer = pd.DataFrame.from_records(records[:,0])
        customer.insert(0, "interaction", np.arange(1, len(records)+1))
        customer.to_csv(directory / "customer_1.csv", index=False)
        s, a = records["S_before"][:,0], records["product2"][:,0]
        pd.DataFrame({"S": np.arange(n+1),
            "visits_product1": np.bincount(s[~a].astype(int), minlength=n+1),
            "visits_product2": np.bincount(s[a].astype(int), minlength=n+1)
        }).to_csv(directory / "customer_1_histogram.csv", index=False)
        dump(directory / "manifest.json", {"signature": signature,
            "seller": solution.primitives.as_dict(), "state_timing": "before choice and outcome",
            "initial_state": "empty", "row_interaction": "row+1", "column_customer": "column+1",
            "histogram_customer": 1, "histogram_total": len(records)})
    plot_interactions(records, solution, ROOT / "figures" / "individual_states")
    return {"interactions": label, "count": len(records)}


def run_long_interactions(config, force):
    from .simulation import simulate_interactions, interaction_diagnostics
    from .figures import plot_long_interaction_diagnostics
    settings = config["long_interaction"]
    path = ROOT / "data" / "policies" / f"{slug(settings['p0'],settings['n'])}.npz"
    solution = load_solution(path)
    directory = ROOT / "data" / "long_interactions"
    directory.mkdir(parents=True, exist_ok=True)
    args = {k:settings[k] for k in ("paths", "interactions", "seed", "burn_in")}
    signature = {"model": MODEL, "simulation": args, "policy_sha256": sha256(path)}
    if cache_matches(directory, signature, ["observations.npy", "path_summary.csv"], force):
        records = np.load(directory / "observations.npy", mmap_mode="r")
    else:
        records, summary = simulate_interactions(solution, **args)
        np.save(directory / "observations.npy", records)
        summary.to_csv(directory / "path_summary.csv", index=False)
    blocks, cumulative, stability = interaction_diagnostics(records,
        burn_in=settings["burn_in"], block_size=settings["block_size"])
    for name, frame in [("blocks",blocks), ("cumulative",cumulative), ("stability",stability)]:
        frame.to_csv(directory / f"{name}.csv", index=False)
    rows = []
    for length in (50000,100000,200000):
        if length <= len(records):
            frequencies = records["product2"][settings["burn_in"]:length].mean(axis=0)
            rows.append({"interactions": length, "mean": float(frequencies.mean()),
                "sd_between_paths": float(frequencies.std(ddof=1)),
                "min": float(frequencies.min()), "max": float(frequencies.max())})
    pd.DataFrame(rows).to_csv(directory / "length_robustness.csv", index=False)
    dump(directory / "manifest.json", {"signature": signature, **settings,
         "method": solution.method, "state_timing": "before action", "initial_state": "empty"})
    plot_long_interaction_diagnostics(blocks,cumulative,stability,ROOT / "figures")
    print(json.dumps({"long_interaction": len(records), "paths": records.shape[1]}), flush=True)


def execute(function, tasks, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(function,task) for task in tasks]):
            print(json.dumps(future.result()), flush=True)


def export_policy_tables(cases):
    from .figures import plot_policy_summary, plot_policy_detail
    rows, diagnostics, intervals = [], [], []
    for p0,n in cases:
        solution = load_solution(ROOT / "data" / "policies" / f"{slug(p0,n)}.npz")
        rows.extend(solution.summary_by_count())
        diagnostics.append({**solution.diagnostics, "n": n, "p0": p0, "method": solution.method})
        active = np.flatnonzero(solution.advantages[n] > solution.action_tolerance)
        groups = np.split(active, np.flatnonzero(np.diff(active) > 1)+1)
        bands = [(int(g[0]),int(g[-1])) for g in groups if len(g)]
        if bands:
            for index,(lower,upper) in enumerate(bands,1):
                intervals.append({"n": n, "p0": p0, "band": index,
                    "lower_S": lower, "upper_S": upper, "number_of_bands": len(bands),
                    "product2_active": True, "method": solution.method})
        else:
            intervals.append({"n": n, "p0": p0, "band": 0, "lower_S": None,
                "upper_S": None, "number_of_bands": 0, "product2_active": False, "method": solution.method})
    table = pd.DataFrame(rows)
    table["policy_method"] = "exact_opposite_replacement"
    table.to_csv(ROOT / "data" / "policy_by_count.csv", index=False)
    pd.DataFrame(intervals).to_csv(ROOT / "data" / "policy_intervals.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(ROOT / "data" / "solver_diagnostics.csv", index=False)
    dump(ROOT / "data" / "solver_diagnostics.json", diagnostics)
    plot_policy_summary(table,ROOT / "figures")
    for n in (10,20,100):
        plot_policy_detail(load_solution(ROOT / "data" / "policies" / f"{slug(.5,n)}.npz"), ROOT / "figures")


def export_reference_comparison():
    from .figures import geometric_reference_comparison
    snapshot = ROOT / "data" / "geometric_reference.npy"
    source = ROOT / "archive" / "fifo_previous" / "data" / "geometric_reference.npy"
    if not snapshot.exists() and source.exists():
        np.save(snapshot,np.load(source))
        dump(ROOT / "data" / "geometric_reference_manifest.json", {
            "source": os.path.relpath(source,ROOT), "source_sha256": sha256(source),
            "rho": .95, "p0": .5, "c2": .65, "customer": 1,
            "comparison": "descriptive, not paired random streams"})
    path = ROOT / "data" / "interactions" / "p0_0.5_n_20" / "observations.npy"
    if snapshot.exists() and path.exists():
        geometric_reference_comparison(np.load(snapshot),np.load(path,mmap_mode="r"),ROOT / "figures")


def export_calendar_figures():
    from .figures import plot_monte_carlo
    paths = sorted((ROOT / "data" / "calendar").glob("*/by_period.csv"))
    if not paths:
        return
    periods = pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    binned = pd.concat([pd.read_csv(p.with_name("binned.csv")) for p in paths],ignore_index=True)
    periods.to_csv(ROOT / "data" / "monte_carlo_by_period.csv",index=False)
    binned.to_csv(ROOT / "data" / "monte_carlo_binned.csv",index=False)
    plot_monte_carlo(periods,binned,ROOT / "figures")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage",choices=["all","solve","calendar","interactions","figures","audit"],default="all")
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--force",action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(CONFIG.read_text())
    assert config["model"] == MODEL
    calendar_ns = sorted({evidence_matched_n(rho) for rho in config["rho_values"]})
    calendar_cases = [(p0,n) for p0 in config["p0_values"] for n in calendar_ns]
    cases = sorted(set(calendar_cases) | {tuple(c) for c in config["interaction"]["cases"]})
    mapping = [{"rho": rho, "geometric_evidence": 1/(1-rho), "n": evidence_matched_n(rho),
                "rounding": "half up", "interpretation": "memory evidence capacity"}
        for rho in sorted(set(config["rho_values"]+config["additional_interaction_rhos"]))]
    pd.DataFrame(mapping).to_csv(ROOT / "data" / "rho_to_n.csv",index=False)
    dump(ROOT / "data" / "effective_config.json",config)
    if args.stage in ("all","solve"):
        execute(solve_case,[(p0,n,config,args.force) for p0,n in cases],args.workers)
    if args.stage in ("all","solve","figures"):
        export_policy_tables(cases)
    if args.stage in ("all","calendar"):
        execute(calendar_case,[(p0,n,config,args.force) for p0,n in calendar_cases],args.workers)
    if args.stage in ("all","interactions"):
        execute(interaction_case,[(p0,n,config,args.force) for p0,n in config["interaction"]["cases"]],args.workers)
        run_long_interactions(config,args.force)
    if args.stage in ("all","calendar","figures"):
        export_calendar_figures()
    if args.stage in ("all","interactions","figures"):
        export_reference_comparison()
    if args.stage in ("all","figures"):
        from .summarize import main as summarize
        summarize()
    dump(ROOT / "data" / "run_manifest.json",{"model": MODEL,"configuration": config,
         "stage": args.stage,"python": sys.version,"numpy": np.__version__,
         "platform": platform.platform(),"source_sha256": source_hashes(),
         "policy_cases": cases,"calendar_cases": calendar_cases,
         "all_capacities": "exact count-state Bellman solution, including n=1000"})
    if args.stage in ("all","audit"):
        from .audit_outputs import main as audit
        audit()


if __name__ == "__main__":
    main()
