"""Dense integer-memory experiment, preserving the historical rho-matched run.

Run from the repository root with python -m
forgetting_random_replacement.oblivious.code.capacity_sweep --stage solve.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from importlib.metadata import version
import json
from math import comb
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.optimize import root

from forgetting.Code.model import Primitives
from .run_experiment import (CONFIG as ORIGINAL_CONFIG, HOME as ORIGINAL_HOME, ROOT,
                             case_key, now, numerical_sources, write_csv, write_json)

CAMPAIGN = ORIGINAL_HOME / "figures"
CONFIG = {
    "capacities": list(range(1, 31)),
    "outside_options": [.1, .2, .3, .4, .5, .6, .7, .8, .9],
    "models": [[1, "monopoly"], [2, "marginal"], [2, "conditional"]],
    "extension_p0s": [.1, .3],
    "extension_capacities": list(range(1, 9)),
    "max_sellers": 5,
    "economic_parameters": ORIGINAL_CONFIG["economic_parameters"],
    "quadrature_order": ORIGINAL_CONFIG["quadrature_order"],
    "outside_option_panel_capacities": [1, 2, 3, 4, 5, 6, 7, 8, 10, 30],
    "large_chain_threshold": 2000,
    "memory_parameter": "Every integer capacity C=1,...,30; no rho-to-capacity rounding",
    "retention_identity": "effective_rho=1-1/C",
}


def expected_cases(capacities=None, p0s=None, max_sellers=5):
    capacities = CONFIG["capacities"] if capacities is None else capacities
    p0s = CONFIG["outside_options"] if p0s is None else p0s
    result = []
    for capacity in capacities:
        for p0 in p0s:
            result.extend((capacity, p0, sellers, closure) for sellers, closure in CONFIG["models"])
            if p0 in CONFIG["extension_p0s"] and capacity in CONFIG["extension_capacities"]:
                result.extend((capacity, p0, sellers, "marginal") for sellers in range(3, max_sellers+1))
    return result


def large_solver_sources():
    """Bind scalable-solver imports separately from the unchanged count model."""
    paths = [ORIGINAL_HOME / "code/large_count_model.py"]
    dependency_directory = ROOT / "oblivious/09_oe_three_sellers_fixed_p0/code"
    paths += [dependency_directory / name for name in
              ("five_seller_model.py", "four_seller_iterative.py", "stationary.py",
               "model.py", "search_only.py")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths if path.exists()}


def _checkpoint(stem):
    return CAMPAIGN / "data/cases" / stem


def _load_cached(stem, sources):
    target = _checkpoint(stem)
    if not target.with_suffix(".json").exists() or not target.with_suffix(".npz").exists():
        return None
    record = json.loads(target.with_suffix(".json").read_text())
    if (record.get("sources") != sources or not record["summary"].get("verified")
            or record.get("checkpoint_sha256") != hashlib.sha256(target.with_suffix(".npz").read_bytes()).hexdigest()):
        return None
    if record.get("solver_sources") and record["solver_sources"] != large_solver_sources():
        return None
    with np.load(target.with_suffix(".npz"), allow_pickle=False) as saved:
        return saved["policy"].copy(), record


def _copy_historical(stem, sources):
    original = ORIGINAL_HOME / "data/cases" / stem
    if not original.with_suffix(".json").exists() or not original.with_suffix(".npz").exists():
        return None
    record_bytes = original.with_suffix(".json").read_bytes()
    record = json.loads(record_bytes)
    checksum = hashlib.sha256(original.with_suffix(".npz").read_bytes()).hexdigest()
    if (record.get("sources") != sources or not record["summary"].get("verified")
            or record.get("checkpoint_sha256") != checksum):
        return None
    destination = _checkpoint(stem)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp.npz")
    shutil.copyfile(original.with_suffix(".npz"), temporary)
    temporary.replace(destination.with_suffix(".npz"))
    record["reused_from"] = {"path": str(original.relative_to(ROOT)),
                             "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
                             "checkpoint_sha256": checksum,
                             "reason": "Identical economic parameters and exact count model"}
    record["campaign"] = "capacity_sweep_1_to_30"
    write_json(destination.with_suffix(".json"), record)
    with np.load(destination.with_suffix(".npz"), allow_pickle=False) as saved:
        return saved["policy"].copy(), record


def save_result(experiment, result, sources, elapsed_seconds, *, solver_sources=None, seed=None):
    summary = result["summary"]
    if not summary.get("verified") or numerical_sources() != sources:
        raise ArithmeticError("Only certified solutions from unchanged numerical sources may be saved")
    if solver_sources and large_solver_sources() != solver_sources:
        raise ArithmeticError("Scalable solver sources changed during this solve")
    stem = case_key(summary["capacity"], summary["p0"], summary["sellers"], summary["closure"])
    target = _checkpoint(stem)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: value for name, value in result.items() if isinstance(value, np.ndarray)}
    arrays["states"] = experiment.kernel.states
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(target.with_suffix(".npz"))
    record = {"campaign": "capacity_sweep_1_to_30", "summary": summary,
              "configuration": result.get("configuration", experiment.configuration()),
              "diagnostics": result["diagnostics"], "search_history": result.get("history", []),
              "search_calls": result.get("search_calls", 0), "seed": seed,
              "sources": sources, "solver_sources": solver_sources or {},
              "elapsed_seconds": elapsed_seconds, "completed_utc": now(),
              "checkpoint_sha256": hashlib.sha256(target.with_suffix(".npz").read_bytes()).hexdigest()}
    write_json(target.with_suffix(".json"), record)
    failure_path = CAMPAIGN / "data/failures" / (stem+".json")
    if failure_path.exists():
        failure = json.loads(failure_path.read_text())
        failure.update(resolved=True, resolved_utc=now(),
                       resolved_checkpoint_sha256=record["checkpoint_sha256"])
        write_json(failure_path, failure)
    return stem


def direct_warm_search(experiment, initial_policy, *, seed=None, original_failure=None,
                       max_evaluations=300):
    """Search directly from a nearby certified policy, then apply the full certificate.

    Pure-policy improvements can move a useful warm start into another search
    basin. This complementary search solves the same natural-map equations
    immediately; it does not restrict the active counts or policy shape.
    """
    initial = np.asarray(initial_policy, dtype=float).copy()
    if (initial.shape != (experiment.n,) or np.any(~np.isfinite(initial))
            or np.any((initial < 0) | (initial > 1))):
        raise ValueError("Warm search requires a probability at every exact count state")
    before = experiment.calls

    def equations(raw):
        data = experiment.calculate(raw)
        return raw-np.clip(raw+data["advantage"], 0, 1)

    def jacobian(raw):
        data = experiment.calculate(raw)
        derivative = data["jacobian"]*((raw >= 0)&(raw <= 1))[None, :]
        interior = (raw+data["advantage"] > 0)&(raw+data["advantage"] < 1)
        return np.eye(experiment.n)-interior[:, None]*(np.eye(experiment.n)+derivative)

    solved = root(equations, initial, jac=jacobian, method="hybr",
                  options={"xtol": 1e-10, "maxfev": max_evaluations, "factor": .3})
    policy = np.clip(solved.x, 0, 1)
    policy[policy < 1e-9] = 0
    policy[policy > 1-1e-9] = 1
    data = experiment.calculate(policy)
    if data["regret"] > experiment.tolerance:
        raise ArithmeticError(f"Alternative warm search did not certify; regret={data['regret']:g}")
    result = experiment.certify(policy)
    if not result["verified"]:
        raise ArithmeticError("Alternative warm candidate failed the unchanged full certificate")
    history = [{"stage": "Direct natural-map search from an alternative certified outside-option policy",
                "seed": seed, "original_search_failure": original_failure,
                "regret": data["regret"], "calls": experiment.calls-before,
                "root_success": bool(solved.success), "root_message": str(solved.message)}]
    return dict(result, history=history, search_calls=experiment.calls)


def recover_from_alternative_p0(experiment, sources, original_failure):
    """Try available same-capacity/seller/closure checkpoints at other p0 values."""
    failures = []
    for p0 in CONFIG["outside_options"]:
        if p0 == experiment.p0:
            continue
        stem = case_key(experiment.capacity, p0, experiment.sellers, experiment.closure)
        cached = _load_cached(stem, sources)
        if cached is None:
            continue
        policy, record = cached
        seed = {"case": stem, "checkpoint_sha256": record["checkpoint_sha256"],
                "method": "Direct natural-map root from another verified outside-option policy",
                "original_search_failure": str(original_failure)}
        try:
            result = direct_warm_search(experiment, policy, seed=seed,
                                        original_failure=str(original_failure))
            return result, seed
        except (RuntimeError, ArithmeticError, ValueError) as error:
            failures.append({"seed": stem, "error": repr(error)})
    raise RuntimeError("No alternative outside-option seed certified: "+json.dumps(failures)) from original_failure


def solve_sequence(p0, capacities, max_sellers, sources, force=False):
    """One ascending-capacity sequence, retaining useful own-policy warm starts."""
    from .model import CountExperiment
    previous, completed, failed = {}, [], []
    for capacity in sorted(capacities):
        current = {}
        models = list(CONFIG["models"])
        if p0 in CONFIG["extension_p0s"] and capacity in CONFIG["extension_capacities"]:
            models += [(sellers, "marginal") for sellers in range(3, max_sellers+1)]
        for sellers, closure in models:
            stem = case_key(capacity, p0, sellers, closure)
            started = time.monotonic()
            try:
                cached = None if force else (_load_cached(stem, sources) or _copy_historical(stem, sources))
                if cached is not None:
                    policy, _ = cached
                    previous[sellers, closure] = capacity, policy, stem
                    current[sellers, closure] = policy
                    completed.append(stem)
                    print(now(), "CACHED", stem, flush=True)
                    continue
                kwargs = dict(capacity=capacity, sellers=sellers, closure=closure, p0=p0,
                              primitives=Primitives(p0=p0, **CONFIG["economic_parameters"]),
                              order=CONFIG["quadrature_order"])
                joint_states = comb(capacity+sellers, sellers)
                scalable = sellers >= 3 and joint_states > CONFIG["large_chain_threshold"]
                extra_sources = large_solver_sources() if scalable else {}
                if scalable:
                    from .large_count_model import LargeCountExperiment
                    experiment = LargeCountExperiment(**kwargs)
                else:
                    experiment = CountExperiment(**kwargs)
                initial, seed = None, None
                if sellers >= 3 and (sellers-1, "marginal") in current:
                    initial = current[sellers-1, "marginal"]
                    seed = {"case": case_key(capacity, p0, sellers-1, "marginal"),
                            "method": "Current-capacity policy from one fewer seller"}
                elif (sellers, closure) in previous:
                    old_capacity, old_policy, old_stem = previous[sellers, closure]
                    initial = np.interp(np.arange(capacity+1)/capacity,
                                        np.arange(old_capacity+1)/old_capacity, old_policy)
                    seed = {"case": old_stem, "method": "Linear interpolation of previous capacity policy for search only"}
                elif (sellers-1, "marginal") in current:
                    initial = current[sellers-1, "marginal"]
                    seed = {"method": "Current-capacity policy from one fewer seller"}
                elif (1, "monopoly") in current:
                    initial = current[1, "monopoly"]
                    seed = {"method": "Current-capacity monopoly policy"}
                print(now(), "START", stem, f"states={joint_states}", flush=True)
                try:
                    if sellers >= 3 and initial is not None:
                        try:
                            result = direct_warm_search(experiment, initial, seed=seed,
                                                        max_evaluations=100)
                        except (RuntimeError, ArithmeticError, ValueError) as warm_error:
                            seed = dict(seed or {}, direct_search_failure=repr(warm_error))
                            result = experiment.solve(initial_policy=initial)
                    else:
                        result = experiment.solve(initial_policy=initial)
                except (RuntimeError, ArithmeticError) as error:
                    result, seed = recover_from_alternative_p0(experiment, sources, error)
                save_result(experiment, result, sources, time.monotonic()-started,
                            solver_sources=extra_sources, seed=seed)
                policy = result["policy"].copy()
                previous[sellers, closure] = capacity, policy, stem
                current[sellers, closure] = policy
                completed.append(stem)
                print(now(), "DONE", stem, result["summary"]["product2_conditional"], flush=True)
                del experiment, result
            except Exception as error:
                failure = {"case": stem, "error": repr(error),
                           "elapsed_seconds": time.monotonic()-started, "failed_utc": now(), "resolved": False}
                failed.append(failure)
                write_json(CAMPAIGN / "data/failures" / (stem+".json"), failure)
                print(now(), "FAILED", stem, repr(error), flush=True)
    return {"completed": completed, "failed": failed}


def collect(sources=None):
    sources = numerical_sources() if sources is None else sources
    rows, policies = [], []
    for path in sorted((CAMPAIGN / "data/cases").glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("sources") != sources:
            continue
        rows.append(dict(record["summary"], case=path.stem))
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
            for count, alpha in enumerate(saved["policy"]):
                summary = record["summary"]
                policies.append({"case": path.stem, "capacity": summary["capacity"],
                    "p0": summary["p0"], "sellers": summary["sellers"], "closure": summary["closure"],
                    "S": count, "retained_success_fraction": count/summary["capacity"],
                    "product2_probability": float(alpha), "belief_demand": float(saved["demand"][count]),
                    "advantage": float(saved["advantage"][count])})
    write_csv(CAMPAIGN / "data/summary.csv", rows)
    write_csv(CAMPAIGN / "data/policies.csv", policies)
    write_csv(CAMPAIGN / "data/capacities.csv", [
        {"capacity": capacity, "count_states": capacity+1, "effective_rho": 1-1/capacity,
         **{f"joint_states_N{sellers}": comb(capacity+sellers, sellers) for sellers in range(1, 6)}}
        for capacity in CONFIG["capacities"]])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("solve", "audit", "figures", "all"), default="all")
    parser.add_argument("--capacities", type=int, nargs="+", default=CONFIG["capacities"])
    parser.add_argument("--p0s", type=float, nargs="+", default=CONFIG["outside_options"])
    parser.add_argument("--max-sellers", type=int, choices=(2, 3, 4, 5), default=5)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if (not 1 <= args.workers <= 3 or any(c not in CONFIG["capacities"] for c in args.capacities)
            or len(set(args.capacities)) != len(args.capacities)
            or any(p0 not in CONFIG["outside_options"] for p0 in args.p0s) or len(set(args.p0s)) != len(args.p0s)):
        parser.error("Use one to three workers, distinct capacities1..30 and configured outside options")
    sources = numerical_sources()
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    write_json(CAMPAIGN / "config.json", CONFIG)
    if args.stage in ("solve", "all"):
        requested = expected_cases(args.capacities, args.p0s, args.max_sellers)
        manifest = {"created_utc": now(), "status": "running", "configuration": CONFIG,
                    "requested_capacities": args.capacities, "requested_p0s": args.p0s,
                    "requested_max_sellers": args.max_sellers, "expected_case_count": len(requested),
                    "sources": sources, "completed": [], "failed": [],
                    "software_versions": {name: version(name) for name in ("numpy", "scipy", "matplotlib")}}
        write_json(CAMPAIGN / "data/manifest.json", manifest)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            jobs = [executor.submit(solve_sequence, p0, args.capacities, args.max_sellers, sources, args.force)
                    for p0 in args.p0s]
            for future in as_completed(jobs):
                result = future.result()
                manifest["completed"].extend(result["completed"])
                manifest["failed"].extend(result["failed"])
                write_json(CAMPAIGN / "data/manifest.json", manifest)
        manifest.update(status="complete" if not manifest["failed"] else "incomplete", completed_utc=now())
        write_json(CAMPAIGN / "data/manifest.json", manifest)
    rows = collect(sources)
    if args.stage in ("audit", "all"):
        from .capacity_audit import audit_campaign
        audit_campaign()
    if args.stage in ("figures", "all"):
        from .capacity_report import publish
        publish(rows)
    source_path = CAMPAIGN / "data/source_manifest.json"
    source_manifest = json.loads(source_path.read_text()) if source_path.exists() else {"created_utc": now()}
    source_manifest.update(updated_utc=now(), numerical_sources=sources,
        large_solver_sources=large_solver_sources(),
        campaign_entrypoint_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    campaign_sources = source_manifest.setdefault("campaign_sources", {})
    for name in ("capacity_sweep.py", "capacity_audit.py", "capacity_report.py",
                 "capacity_policy_report.py", "large_count_model.py", "test_large_count_model.py"):
        campaign_sources[name] = hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
    write_json(source_path, source_manifest)


if __name__ == "__main__":
    main()
