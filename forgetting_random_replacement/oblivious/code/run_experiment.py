"""Reproduce Experiment 8 and the one-to-five-seller plot with random replacement.

Run from the repository root with python -m
forgetting_random_replacement.oblivious.code.run_experiment --stage all.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from forgetting.Code.model import Primitives
from forgetting_random_replacement.code.model import evidence_matched_n

HOME = Path(__file__).resolve().parents[1]
ROOT = HOME.parents[1]
CONFIG = json.loads((HOME / "code/config.json").read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=plain, allow_nan=False) + "\n")
    temporary.replace(path)


def write_csv(path, rows):
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def case_key(capacity, p0, sellers, closure):
    return f"capacity_{capacity:03d}_p0_{str(p0).replace('.', 'p')}_N{sellers}_{closure}"


def numerical_sources():
    paths = [HOME / "code" / name for name in ("model.py", "kernel.py", "config.json")]
    paths += [ROOT / "forgetting_random_replacement/code/model.py",
              ROOT / "forgetting/Code/model.py",
              ROOT / "oblivious/02_oe_p0/code/positive_stationary.py",
              ROOT / "oblivious/02_oe_p0/code/positive_stationary.cpp"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def save_case_result(experiment, result, sources, *, elapsed_seconds=0., search_notes=None):
    """Persist only a certified result, with immutable numerical input hashes."""
    summary = result["summary"]
    if not summary.get("verified"):
        raise ArithmeticError("Refusing to save an uncertified equilibrium")
    if numerical_sources() != sources:
        raise ArithmeticError("Numerical sources changed during the solve")
    stem = case_key(summary["capacity"], summary["p0"], summary["sellers"], summary["closure"])
    target = HOME / "data/cases" / stem
    arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
    arrays["states"] = experiment.kernel.states
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(target.with_suffix(".npz"))
    record = {"summary": summary, "diagnostics": result["diagnostics"],
              "configuration": result.get("configuration", experiment.configuration()),
              "search_history": result.get("history", []), "search_notes": search_notes,
              "search_calls": result.get("search_calls", 0), "sources": sources,
              "completed_utc": now(), "elapsed_seconds": elapsed_seconds,
              "checkpoint_sha256": hashlib.sha256(target.with_suffix(".npz").read_bytes()).hexdigest()}
    write_json(target.with_suffix(".json"), record)
    failed_path = HOME / "data/failures" / (stem + ".json")
    if failed_path.exists():
        previous_failure = json.loads(failed_path.read_text())
        previous_failure.update(resolved=True, resolution="A new candidate passed the unchanged full certificate",
                                resolved_checkpoint_sha256=record["checkpoint_sha256"])
        write_json(failed_path, previous_failure)
    return stem


def solve_group(capacity, p0, sources, force=False, extension=False):
    from .model import CountExperiment

    completed, failed, initial = [], [], None
    models = [(1, "monopoly"), (2, "marginal"), (2, "conditional")]
    if extension:
        models += [(sellers, "marginal") for sellers in (3, 4, 5)]
    for sellers, closure in models:
        stem = case_key(capacity, p0, sellers, closure)
        target = HOME / "data/cases" / stem
        started = time.monotonic()
        try:
            if target.with_suffix(".json").exists() and target.with_suffix(".npz").exists() and not force:
                record = json.loads(target.with_suffix(".json").read_text())
                checksum = hashlib.sha256(target.with_suffix(".npz").read_bytes()).hexdigest()
                if (record.get("sources") == sources and record["summary"].get("verified")
                        and record.get("checkpoint_sha256") == checksum):
                    with np.load(target.with_suffix(".npz"), allow_pickle=False) as saved:
                        initial = saved["policy"].copy()
                    completed.append(stem)
                    print(now(), "CACHED", stem, flush=True)
                    continue
            print(now(), "START", stem, flush=True)
            experiment = CountExperiment(capacity=capacity, sellers=sellers, closure=closure,
                                         p0=p0, primitives=Primitives(p0=p0, **CONFIG["economic_parameters"]),
                                         order=CONFIG["quadrature_order"])
            search_notes = None
            try:
                result = experiment.solve(initial_policy=initial)
            except (RuntimeError, ArithmeticError) as first_error:
                if not (capacity == 100 and p0 == .1 and sellers == 2 and closure == "marginal"):
                    raise
                from .recovery import solve_marginal_continuation
                seed_target = HOME / "data/cases" / (case_key(capacity, .3, 2, "marginal") + ".npz")
                if seed_target.exists():
                    with np.load(seed_target, allow_pickle=False) as seed_file:
                        seed = seed_file["policy"].copy()
                else:
                    bridge = CountExperiment(capacity, 2, "marginal", .3,
                        primitives=Primitives(p0=.3, **CONFIG["economic_parameters"]),
                        order=CONFIG["quadrature_order"])
                    seed = bridge.solve()["policy"]
                result = solve_marginal_continuation(experiment, seed)
                search_notes = {"initial_search_failure": str(first_error),
                    "recovery": "Natural-map continuation in outside-option value from 0.3 to 0.1",
                    "recovery_source_sha256": hashlib.sha256((HOME/"code/recovery.py").read_bytes()).hexdigest()}
            if not result["summary"].get("verified"):
                raise ArithmeticError("Equilibrium did not pass its numerical certificate")
            initial = result["policy"].copy()
            save_case_result(experiment, result, sources,
                             elapsed_seconds=time.monotonic()-started, search_notes=search_notes)
            completed.append(stem)
            print(now(), "DONE", stem, result["summary"]["product2_conditional"], flush=True)
        except Exception as error:
            failure = {"case": stem, "error": repr(error), "elapsed_seconds": time.monotonic() - started}
            failed.append(failure)
            write_json(HOME / "data/failures" / f"{stem}.json", failure)
            print(now(), "FAILED", stem, repr(error), flush=True)
    return {"completed": completed, "failed": failed}


def collect(sources=None):
    rows = []
    for path in sorted((HOME / "data/cases").glob("*.json")):
        record = json.loads(path.read_text())
        if sources is not None and record.get("sources") != sources:
            continue
        rows.append(dict(record["summary"], case=path.stem))
    write_csv(HOME / "data/summary.csv", rows)
    expanded = []
    for rho in CONFIG["reference_rhos"]:
        capacity = evidence_matched_n(rho)
        expanded.extend(dict(row, reference_rho=rho, effective_rho=1 - 1 / capacity)
                        for row in rows if row["capacity"] == capacity and row["sellers"] <= 2)
    write_csv(HOME / "data/experiment8_by_reference_rho.csv", expanded)
    write_csv(HOME / "data/rho_to_capacity.csv", [
        {"reference_rho": rho, "capacity": evidence_matched_n(rho),
         "effective_rho": 1 - 1 / evidence_matched_n(rho)} for rho in CONFIG["reference_rhos"]])
    policies = []
    for row in rows:
        with np.load(HOME / "data/cases" / (row["case"] + ".npz"), allow_pickle=False) as saved:
            for s, alpha in enumerate(saved["policy"]):
                policies.append({"case": row["case"], "p0": row["p0"], "capacity": row["capacity"],
                                 "sellers": row["sellers"], "closure": row["closure"], "S": s,
                                 "retained_success_fraction": s / row["capacity"],
                                 "product2_probability": float(alpha), "belief_demand": float(saved["demand"][s]),
                                 "advantage": float(saved["advantage"][s])})
    write_csv(HOME / "data/policies.csv", policies)
    return rows


def save_source_manifest():
    code_files = sorted((HOME / "code").glob("*.py")) + sorted((HOME / "code").glob("*.json"))
    write_json(HOME / "data/provenance/source_manifest.json", {
        "created_utc": now(), "numerical_sources": numerical_sources(),
        "experiment_code": {str(path.relative_to(HOME)): hashlib.sha256(path.read_bytes()).hexdigest()
                            for path in code_files},
        "software_versions": {name: version(name) for name in ("numpy", "scipy", "matplotlib")}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("solve", "figures", "audit", "all"), default="all")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--capacities", type=int, nargs="+", default=CONFIG["capacities"])
    parser.add_argument("--p0s", type=float, nargs="+", default=CONFIG["outside_options"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 3 or min(args.capacities) < 1:
        parser.error("Use one to three workers and positive capacities")
    sources = numerical_sources()
    if args.stage in ("solve", "all"):
        manifest = {"started_utc": now(), "status": "running", "configuration": CONFIG,
                    "software_versions": {name: version(name) for name in ("numpy", "scipy", "matplotlib")},
                    "requested_capacities": args.capacities, "requested_outside_options": args.p0s,
                    "sources": sources, "completed": [], "failed": []}
        write_json(HOME / "data/manifest.json", manifest)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            jobs = [executor.submit(solve_group, capacity, p0, sources, args.force,
                                    capacity == 3 and p0 in (.1, .3))
                    for capacity in args.capacities for p0 in args.p0s]
            for future in as_completed(jobs):
                result = future.result()
                manifest["completed"].extend(result["completed"])
                manifest["failed"].extend(result["failed"])
                write_json(HOME / "data/manifest.json", manifest)
        manifest.update(status="complete" if not manifest["failed"] else "incomplete",
                        completed_utc=now())
        write_json(HOME / "data/manifest.json", manifest)
    rows = collect(sources)
    if args.stage in ("audit", "all"):
        from .audit import audit_campaign
        audit_campaign()
    if args.stage in ("figures", "all"):
        from .figures import publish
        publish(rows)
    save_source_manifest()


if __name__ == "__main__":
    main()
