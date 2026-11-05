"""Reproduce the fixed-window experiments from the repository root.

    .venv/bin/python -B -m forgettin_fixed.code.run_experiment

Existing policy checkpoints and completed simulations are reused.  Use
--force to recompute them.  Run manifests record the effective configuration.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
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
from .model import WindowSolution, evidence_matched_n, solve_window

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path(__file__).with_name("config.json")


def slug(p0: float, n: int) -> str:
    return f"p0_{p0:g}_n_{n}"


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_solution(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["metadata"]))
    if meta["format"] == "fifo-window-v1":
        return WindowSolution.load(path)
    from .large_window import ProjectedSolution
    return ProjectedSolution.load(path)


def solve_case(task):
    p0, n, config, force = task
    label = slug(p0, n)
    path = ROOT / "data" / "policies" / f"{label}.npz"
    primitives = Primitives.from_mapping({**config["seller"], "p0": p0})
    settings = config["solver"]
    if path.exists() and not force:
        with np.load(path, allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
        consistent = (meta["primitives"] == primitives.as_dict() and
                      meta["action_tolerance"] == settings["action_tolerance"])
        if n > settings["exact_max_n"]:
            consistent &= meta.get("prefix_bits") == settings["large_prefix"][str(n)]
        if consistent:
            return {"case": label, "cached": True}
    started = time.monotonic()
    if n <= settings["exact_max_n"]:
        solution = solve_window(primitives, n, tolerance=settings["tolerance"],
                                action_tolerance=settings["action_tolerance"])
    else:
        from .large_window import solve_projected
        solution = solve_projected(primitives, n,
                                   k=settings["large_prefix"][str(n)],
                                   tolerance=settings["tolerance"],
                                   action_tolerance=settings["action_tolerance"])
    solution.save(path)
    dump(path.with_suffix(".json"), solution.diagnostics)
    return {"case": label, "seconds": round(time.monotonic() - started, 2),
            "method": solution.method}


def calendar_case(task):
    p0, n, config, force = task
    label = slug(p0, n)
    directory = ROOT / "data" / "calendar" / label
    index = config["p0_values"].index(p0)
    args = {**config["calendar"], "seed": config["calendar"]["seed"] + 10000 * index}
    if (directory / "manifest.json").exists() and not force:
        manifest = json.loads((directory / "manifest.json").read_text())
        if (manifest["simulation"] == args and
            manifest["seller"] == {**config["seller"], "p0": p0}):
            policy_path = ROOT / "data" / "policies" / f"{label}.npz"
            if (directory / "manifest.json").stat().st_mtime >= policy_path.stat().st_mtime:
                return {"calendar": label, "cached": True}
    from .simulation import simulate_calendar
    started = time.monotonic()
    solution = load_solution(ROOT / "data" / "policies" / f"{label}.npz")
    frames = simulate_calendar(solution, **args)
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in zip(("by_period", "binned", "path_summary"), frames, strict=True):
        frame.insert(0, "method", solution.method)
        frame.insert(0, "p0", p0)
        frame.insert(0, "n", n)
        frame.to_csv(directory / f"{name}.csv", index=False)
    dump(directory / "manifest.json", {"case": label, "simulation": args,
         "seller": solution.primitives.as_dict(), "method": solution.method,
         "initial_state": "empty Beta(1,1)",
         "memory_clock": "A purchases only; outside choices leave FIFO unchanged",
         "policy": f"../../policies/{label}.npz",
         "random_numbers": "same seed across n at fixed p0; common uniforms",
         "seconds": time.monotonic() - started})
    return {"calendar": label, "seconds": round(time.monotonic() - started, 2)}


def interaction_case(task):
    p0, n, config, force = task
    from .simulation import simulate_interactions
    from .figures import plot_interactions
    label = slug(p0, n)
    directory = ROOT / "data" / "interactions" / label
    path = directory / "observations.npy"
    solution = load_solution(ROOT / "data" / "policies" / f"{label}.npz")
    directory.mkdir(parents=True, exist_ok=True)
    if path.exists() and (directory / "manifest.json").exists() and not force:
        records = np.load(path)
    else:
        records, summary = simulate_interactions(solution, **config["interaction"])
        np.save(path, records)
        summary.to_csv(directory / "path_summary.csv", index=False)
        sample = pd.DataFrame.from_records(records[:, 0])
        sample.insert(0, "interaction", np.arange(1, len(records) + 1))
        sample.to_csv(directory / "customer_1.csv", index=False)
        s = records["S_before"][:, 0]
        a = records["product2"][:, 0]
        hist = pd.DataFrame({"S": np.arange(n + 1),
            "visits_product1": np.bincount(s[~a].astype(int), minlength=n + 1),
            "visits_product2": np.bincount(s[a].astype(int), minlength=n + 1)})
        hist.to_csv(directory / "customer_1_histogram.csv", index=False)
        dump(directory / "manifest.json", {"case": label,
            "seller": solution.primitives.as_dict(), "method": solution.method,
            "paths": config["interaction"]["paths"],
            "interactions": config["interaction"]["interactions"],
            "seed": config["interaction"]["seed"], "initial_state": "empty",
            "state_timing": "before choice, outcome, and FIFO eviction",
            "row_interaction": "row+1", "column_customer": "column+1",
            "histogram_customer": 1, "histogram_total": int(len(records))})
    plot_interactions(records, solution, ROOT / "figures" / "individual_states")
    return {"interactions": label, "count": len(records)}


def run_long_interactions(config, force=False):
    from .simulation import simulate_interactions, interaction_diagnostics
    from .figures import plot_long_interaction_diagnostics
    settings = config["long_interaction"]
    p0, n = settings["p0"], settings["n"]
    solution = load_solution(ROOT / "data" / "policies" / f"{slug(p0,n)}.npz")
    directory = ROOT / "data" / "long_interactions"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "observations.npy"
    if path.exists() and not force:
        records = np.load(path, mmap_mode="r")
    else:
        records, summary = simulate_interactions(solution, **{k:settings[k] for k in
            ("paths", "interactions", "seed", "burn_in")})
        np.save(path, records)
        summary.to_csv(directory / "path_summary.csv", index=False)
    blocks, cumulative, stability = interaction_diagnostics(records,
        burn_in=settings["burn_in"], block_size=settings["block_size"])
    for name, frame in [("blocks", blocks), ("cumulative", cumulative), ("stability", stability)]:
        frame.to_csv(directory / f"{name}.csv", index=False)
    robustness = []
    for length in (50000, 100000, 200000):
        if length > len(records):
            continue
        frequencies = records["product2"][settings["burn_in"]:length].mean(axis=0)
        robustness.append({"interactions": length, "mean": float(frequencies.mean()),
            "sd_between_paths": float(frequencies.std(ddof=1)),
            "min": float(frequencies.min()), "max": float(frequencies.max())})
    pd.DataFrame(robustness).to_csv(directory / "length_robustness.csv", index=False)
    dump(directory / "manifest.json", {**settings, "method": solution.method,
        "state_timing": "before action", "initial_state": "empty"})
    plot_long_interaction_diagnostics(blocks, cumulative, stability, ROOT / "figures")
    print(json.dumps({"long_interaction": slug(p0,n), "interactions": len(records)}), flush=True)


def execute(function, tasks, workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(function, task) for task in tasks]
        for result in as_completed(futures):
            print(json.dumps(result.result()), flush=True)


def export_policy_tables(cases):
    from .figures import plot_policy_summary, plot_exact_policy_heatmap
    rows, diagnostics = [], []
    for p0, n in cases:
        solution = load_solution(ROOT / "data" / "policies" / f"{slug(p0,n)}.npz")
        records = solution.summary_by_count()
        frame = pd.DataFrame(records)
        if "S" not in frame:
            frame = frame.rename(columns={"successes": "S"})
        if "action_min" not in frame:
            frame["action_min"] = (frame.advantage_min > solution.action_tolerance).astype(int)
            frame["action_max"] = (frame.advantage_max > solution.action_tolerance).astype(int)
        frame["method"] = solution.method
        frame["policy_method"] = solution.method
        rows.append(frame)
        diagnostics.append({**solution.diagnostics, "n": n, "p0": p0,
                            "solver_method": solution.diagnostics["method"],
                            "method": solution.method})
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(ROOT / "data" / "policy_by_count.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(ROOT / "data" / "solver_diagnostics.csv", index=False)
    dump(ROOT / "data" / "solver_diagnostics.json", diagnostics)
    plot_policy_summary(table, ROOT / "figures")
    for n in (10, 20):
        path = ROOT / "data" / "policies" / f"{slug(.5,n)}.npz"
        if path.exists():
            plot_exact_policy_heatmap(load_solution(path), ROOT / "figures")


def export_reference_comparison():
    from .figures import geometric_reference_comparison
    path = ROOT / "data" / "interactions" / "p0_0.5_n_20" / "observations.npy"
    snapshot = ROOT / "data" / "geometric_reference.npy"
    source = ROOT.parent / "forgetting" / "outputs" / "policy_and_state_experiments" / "first_20000" / "data" / "p0_0.5_rho_0.95" / "observations.npy"
    if not snapshot.exists() and source.exists():
        old = np.load(source, mmap_mode="r")[:20000, 0]
        selected = np.empty(len(old), dtype=[("x_before", "f8"), ("product2", "?")])
        for name in selected.dtype.names:
            selected[name] = old[name]
        np.save(snapshot, selected)
        dump(ROOT / "data" / "geometric_reference_manifest.json", {
            "source": os.path.relpath(source, ROOT), "customer": 1,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "rho": .95, "c2": .65, "interactions": len(selected),
            "comparison": "descriptive; simulation streams are not paired"})
    if path.exists() and snapshot.exists():
        geometric_reference_comparison(np.load(snapshot), np.load(path, mmap_mode="r"), ROOT / "figures")


def source_hashes():
    paths = [*sorted((ROOT / "code").glob("*.py")),
             ROOT.parent / "forgetting" / "Code" / "model.py", CONFIG]
    return {os.path.relpath(p, ROOT): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["solve", "calendar", "interactions", "figures", "all"], default="all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config = json.loads(CONFIG.read_text())
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "figures").mkdir(exist_ok=True)
    calendar_ns = sorted({evidence_matched_n(rho) for rho in config["rho_values"]})
    calendar_cases = [(p0, n) for p0 in config["p0_values"] for n in calendar_ns]
    cases = sorted(set(calendar_cases) | {tuple(x) for x in config["interaction"]["cases"]})
    mapping = [{"rho": rho, "geometric_evidence": 1 / (1 - rho),
                "n": evidence_matched_n(rho), "rounding": "half up"}
               for rho in sorted(set(config["rho_values"] + config["additional_interaction_rhos"]))]
    pd.DataFrame(mapping).to_csv(ROOT / "data" / "rho_to_n.csv", index=False)
    dump(ROOT / "data" / "effective_config.json", config)
    if args.stage in ("all", "solve"):
        execute(solve_case, [(p0, n, config, args.force) for p0,n in cases], args.workers)
        export_policy_tables(cases)
    elif args.stage == "figures":
        export_policy_tables(cases)
    if args.stage in ("all", "calendar"):
        execute(calendar_case, [(p0, n, config, args.force) for p0,n in calendar_cases], args.workers)
    if args.stage in ("all", "interactions"):
        # The 'cases' key is orchestration, not a simulator parameter.
        sim_config = {**config, "interaction": {k:v for k,v in config["interaction"].items() if k != "cases"}}
        execute(interaction_case, [(p0, n, sim_config, args.force)
                                   for p0,n in config["interaction"]["cases"]], args.workers)
        run_long_interactions(config, args.force)
    if args.stage in ("all", "figures", "calendar"):
        from .figures import plot_monte_carlo
        paths = sorted((ROOT / "data" / "calendar").glob("*/by_period.csv"))
        if paths:
            periods = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
            binned = pd.concat([pd.read_csv(p.with_name("binned.csv")) for p in paths], ignore_index=True)
            periods.to_csv(ROOT / "data" / "monte_carlo_by_period.csv", index=False)
            binned.to_csv(ROOT / "data" / "monte_carlo_binned.csv", index=False)
            plot_monte_carlo(periods, binned, ROOT / "figures")
    if args.stage in ("all", "figures", "interactions"):
        export_reference_comparison()
    if args.stage in ("all", "figures") and (ROOT / "data" / "refinement" / "paired_calendar_summary.json").exists():
        from .summarize import main as summarize_results
        summarize_results()
    dump(ROOT / "data" / "run_manifest.json", {"configuration": config,
         "stage": args.stage, "python": sys.version, "numpy": np.__version__,
         "platform": platform.platform(), "source_sha256": source_hashes(),
         "policy_cases": cases, "calendar_cases": calendar_cases,
         "note": "n<=20 exhaustive FIFO; larger n approximate planning with actual FIFO execution and explicit error diagnostics"})


if __name__ == "__main__":
    main()
