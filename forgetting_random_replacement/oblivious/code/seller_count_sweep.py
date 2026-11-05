"""Extend the exact random-replacement experiment across seller counts.

The separate campaign preserves every checkpoint and figure in the existing
integer-capacity experiment. No unverified or state-budget-limited case is
filled with zero. Run from the repository root with python -m
forgetting_random_replacement.oblivious.code.seller_count_sweep --stage solve.
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
from .capacity_sweep import CAMPAIGN as CAPACITY_CAMPAIGN, large_solver_sources
from .run_experiment import (CONFIG as BASE_CONFIG, HOME as BASE, ROOT,
                             case_key, now, numerical_sources, write_csv, write_json)

CAMPAIGN = CAPACITY_CAMPAIGN / "seller_count_extension"
CONFIG = {
    "capacities": [1, 3, 5, 8],
    "outside_options": BASE_CONFIG["outside_options"],
    "seller_counts_by_capacity": {
        "1": list(range(1, 101)), "3": list(range(1, 31))+[35, 40, 50],
        "5": list(range(1, 11)), "8": list(range(1, 11))},
    "economic_parameters": BASE_CONFIG["economic_parameters"],
    "quadrature_order": BASE_CONFIG["quadrature_order"],
    "maximum_joint_states": 250_000,
    "direct_state_limit": 2000,
    "strategy": "Monopoly at N=1; symmetric marginal OE at N>=2",
    "joint_representation": "Exact correlated seller-permutation quotient",
    "near_zero_threshold": .01,
    "illustrative_example": {"capacity": 3, "p0": .8805, "seller_counts": list(range(1, 11)),
        "selection_note": "Exploratory example selected near the investment cutoff; not a pre-specified broad-grid result"},
    "scope_note": "Finite verified seller counts do not establish the N-to-infinity limit.",
}


def digest(path):
    """Stream a checkpoint hash without creating another large array copy."""
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def extension_sources():
    """The generic-N implementation and its unchanged scalable dependencies."""
    model = BASE / "code/seller_count_model.py"
    if not model.exists():
        raise FileNotFoundError(f"Generic seller-count model is not ready: {model}")
    return {str(model.relative_to(ROOT)): digest(model), **large_solver_sources()}


def seller_counts(capacity, requested=None):
    values = CONFIG["seller_counts_by_capacity"][str(capacity)] if requested is None else requested
    return sorted(set(int(value) for value in values))


def expected_cases(capacities=None, p0s=None, sellers=None, *, include_illustrative=True):
    capacities = CONFIG["capacities"] if capacities is None else capacities
    p0s = CONFIG["outside_options"] if p0s is None else p0s
    result = [(capacity, p0, count, "monopoly" if count == 1 else "marginal")
              for capacity in capacities for p0 in p0s for count in seller_counts(capacity, sellers)]
    example = CONFIG.get("illustrative_example")
    if include_illustrative and example and example["capacity"] in capacities:
        counts = example["seller_counts"] if sellers is None else sorted(set(sellers)&set(example["seller_counts"]))
        result.extend((example["capacity"], example["p0"], n, "monopoly" if n == 1 else "marginal") for n in counts)
    return result


def _checkpoint(stem):
    return CAMPAIGN / "data/cases" / stem


def _record_matches(record, target, sources, extra_sources):
    return (record.get("sources") == sources
            and record.get("summary", {}).get("verified") is True
            and record.get("checkpoint_sha256") == digest(target.with_suffix(".npz"))
            and (not record.get("extension_sources")
                 or record["extension_sources"] == extra_sources))


def load_cached(stem, sources, extra_sources):
    target = _checkpoint(stem)
    if not target.with_suffix(".json").exists() or not target.with_suffix(".npz").exists():
        return None
    record = json.loads(target.with_suffix(".json").read_text())
    if not _record_matches(record, target, sources, extra_sources):
        return None
    with np.load(target.with_suffix(".npz"), allow_pickle=False) as saved:
        return saved["policy"].copy(), record


def copy_historical(stem, sources):
    """Copy a compatible, immutable existing result into this campaign."""
    for directory in (CAPACITY_CAMPAIGN / "data/cases", BASE / "data/cases"):
        original = directory / stem
        if not original.with_suffix(".json").exists() or not original.with_suffix(".npz").exists():
            continue
        original_bytes = original.with_suffix(".json").read_bytes()
        record = json.loads(original_bytes)
        if (record.get("sources") != sources or not record.get("summary", {}).get("verified")
                or record.get("checkpoint_sha256") != digest(original.with_suffix(".npz"))):
            continue
        if record.get("solver_sources") and record["solver_sources"] != large_solver_sources():
            continue
        destination = _checkpoint(stem)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp.npz")
        shutil.copyfile(original.with_suffix(".npz"), temporary)
        temporary.replace(destination.with_suffix(".npz"))
        record.update(campaign="seller_count_extension", extension_sources={}, reused_from={
            "path": str(original.relative_to(ROOT)),
            "record_sha256": hashlib.sha256(original_bytes).hexdigest(),
            "checkpoint_sha256": record["checkpoint_sha256"],
            "reason": "Identical economic parameters, count transitions, and marginal belief closure"})
        write_json(destination.with_suffix(".json"), record)
        with np.load(destination.with_suffix(".npz"), allow_pickle=False) as saved:
            return saved["policy"].copy(), record
    return None


def save_result(experiment, result, sources, extra_sources, elapsed_seconds, *, seed=None, attempts=None):
    if not result.get("summary", {}).get("verified"):
        raise ArithmeticError("An unverified seller-count equilibrium cannot be saved")
    if numerical_sources() != sources or extension_sources() != extra_sources:
        raise ArithmeticError("Numerical sources changed during seller-count search")
    summary = result["summary"]
    stem = case_key(summary["capacity"], summary["p0"], summary["sellers"], summary["closure"])
    target = _checkpoint(stem)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {name: value for name, value in result.items() if isinstance(value, np.ndarray)}
    arrays["states"] = experiment.kernel.states
    temporary = target.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(target.with_suffix(".npz"))
    record = {
        "campaign": "seller_count_extension", "summary": summary,
        "configuration": result.get("configuration", experiment.configuration()),
        "diagnostics": result["diagnostics"], "search_history": result.get("history", []),
        "search_calls": result.get("search_calls", experiment.calls),
        "search_attempts": attempts or [], "seed": seed,
        "sources": sources, "extension_sources": extra_sources,
        "elapsed_seconds": elapsed_seconds, "completed_utc": now(),
        "checkpoint_sha256": digest(target.with_suffix(".npz"))}
    write_json(target.with_suffix(".json"), record)
    failure_path = CAMPAIGN / "data/failures" / (stem + ".json")
    if failure_path.exists():
        failure = json.loads(failure_path.read_text())
        failure.update(resolved=True, resolved_utc=now(),
                       resolved_checkpoint_sha256=record["checkpoint_sha256"])
        write_json(failure_path, failure)
    return stem


def direct_warm_search(experiment, initial_policy, max_evaluations=100):
    """Use a nearby policy directly, without imposing a policy-band shape."""
    initial = np.asarray(initial_policy, dtype=float).copy()
    if initial.shape != (experiment.n,) or np.any(~np.isfinite(initial)) or np.any((initial < 0) | (initial > 1)):
        raise ValueError("Warm policy must be a probability at every count state")
    before = experiment.calls
    first = experiment.calculate(initial)
    if first["regret"] <= experiment.tolerance:
        result = experiment.certify(initial)
        if result["verified"]:
            return dict(result, history=[dict(stage="Certified previous-seller-count policy",
                        regret=first["regret"], calls=experiment.calls-before)], search_calls=experiment.calls)

    def equations(raw):
        data = experiment.calculate(raw)
        return raw-np.clip(raw+data["advantage"], 0, 1)

    def jacobian(raw):
        data = experiment.calculate(raw)
        derivative = data["jacobian"]*((raw >= 0)&(raw <= 1))[None, :]
        interior = (raw+data["advantage"] > 0)&(raw+data["advantage"] < 1)
        return np.eye(experiment.n)-interior[:, None]*(np.eye(experiment.n)+derivative)

    solved = root(equations, initial, jac=jacobian, method="hybr",
                  options=dict(xtol=1e-10, maxfev=max_evaluations, factor=.3))
    policy = np.clip(solved.x, 0, 1)
    policy[policy < 1e-9] = 0
    policy[policy > 1-1e-9] = 1
    data = experiment.calculate(policy)
    if data["regret"] > experiment.tolerance:
        raise ArithmeticError(f"Seller-count warm search did not certify: regret={data['regret']:g}")
    result = experiment.certify(policy)
    if not result["verified"]:
        raise ArithmeticError("Seller-count warm candidate failed its full numerical certificate")
    return dict(result, history=[dict(stage="Direct natural-map search from a previous seller count",
        regret=data["regret"], calls=experiment.calls-before,
        root_success=bool(solved.success), root_message=str(solved.message))], search_calls=experiment.calls)


def solve_sequence(capacity, p0, counts, sources, extra_sources, *, max_states=250_000,
                   max_evaluations=500, force=False):
    """Solve ascending N at one C,p0, retaining an exact-state policy warm start."""
    from .seller_count_model import ExtendedCountExperiment

    completed, failed, skipped = [], [], []
    previous = None
    first_count = min(counts)
    for lower_count in range(first_count-1, 0, -1):
        lower_stem = case_key(capacity, p0, lower_count, "monopoly" if lower_count == 1 else "marginal")
        cached_seed = load_cached(lower_stem, sources, extra_sources)
        if cached_seed is not None:
            lower_policy, lower_record = cached_seed
            previous = lower_policy, lower_stem, lower_record["checkpoint_sha256"]
            break
    for count in sorted(counts):
        closure = "monopoly" if count == 1 else "marginal"
        stem = case_key(capacity, p0, count, closure)
        started = time.monotonic()
        try:
            cached = None if force else (load_cached(stem, sources, extra_sources) or copy_historical(stem, sources))
            if cached is not None:
                policy, record = cached
                previous = policy, stem, record["checkpoint_sha256"]
                completed.append(stem)
                print(now(), "CACHED", stem, flush=True)
                continue
            size = comb(capacity+count, capacity)
            if size > max_states:
                skipped_case = dict(case=stem, reason="Exact joint-state budget exceeded",
                                    joint_states=size, max_states=max_states)
                skipped.append(skipped_case)
                print(now(), "SKIPPED", stem, f"states={size}, budget={max_states}", flush=True)
                continue
            experiment = ExtendedCountExperiment(capacity=capacity, sellers=count, closure=closure, p0=p0,
                primitives=Primitives(p0=p0, **CONFIG["economic_parameters"]),
                order=CONFIG["quadrature_order"], max_states=max_states,
                direct_state_limit=CONFIG["direct_state_limit"])
            seed = (dict(case=previous[1], checkpoint_sha256=previous[2],
                         method="Same-capacity verified policy from the previous seller count")
                    if previous is not None else None)
            attempts, result = [], None
            print(now(), "START", stem, f"states={size}", flush=True)
            if previous is not None:
                try:
                    result = direct_warm_search(experiment, previous[0], min(100, max_evaluations))
                    attempts.append(dict(method="Direct natural-map warm search", verified=True))
                except (ArithmeticError, RuntimeError, ValueError) as error:
                    attempts.append(dict(method="Direct natural-map warm search", verified=False, error=repr(error)))
            if result is None:
                result = experiment.solve(initial_policy=None if previous is None else previous[0],
                                          max_evaluations=max_evaluations)
                attempts.append(dict(method="Ordinary exact-count equilibrium solver", verified=bool(result["verified"])))
            save_result(experiment, result, sources, extra_sources, time.monotonic()-started,
                        seed=seed, attempts=attempts)
            record = json.loads(_checkpoint(stem).with_suffix(".json").read_text())
            previous = result["policy"].copy(), stem, record["checkpoint_sha256"]
            completed.append(stem)
            print(now(), "DONE", stem, result["summary"]["product2_conditional"], flush=True)
            del experiment, result
        except Exception as error:
            failure = dict(case=stem, error=repr(error), elapsed_seconds=time.monotonic()-started,
                           failed_utc=now(), resolved=False)
            failed.append(failure)
            write_json(CAMPAIGN / "data/failures" / (stem + ".json"), failure)
            print(now(), "FAILED", stem, repr(error), flush=True)
    return dict(completed=completed, failed=failed, skipped=skipped)


def collect(sources=None, extra_sources=None):
    sources = numerical_sources() if sources is None else sources
    extra_sources = extension_sources() if extra_sources is None else extra_sources
    rows, policies, excluded = [], [], []
    for path in sorted((CAMPAIGN / "data/cases").glob("*.json")):
        record = json.loads(path.read_text())
        if not path.with_suffix(".npz").exists() or not _record_matches(record, path.with_suffix(""), sources, extra_sources):
            excluded.append(dict(case=path.stem, reason="Stale sources, unverified result, or checksum mismatch"))
            continue
        summary = record["summary"]
        rows.append(dict(summary, case=path.stem))
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as saved:
            for count, alpha in enumerate(saved["policy"]):
                policies.append(dict(case=path.stem, capacity=summary["capacity"], p0=summary["p0"],
                    sellers=summary["sellers"], closure=summary["closure"], S=count,
                    retained_success_fraction=count/summary["capacity"], product2_probability=float(alpha),
                    belief_demand=float(saved["demand"][count]), advantage=float(saved["advantage"][count])))
    write_csv(CAMPAIGN / "data/summary.csv", rows)
    write_csv(CAMPAIGN / "data/policies.csv", policies)
    write_json(CAMPAIGN / "data/collection_manifest.json", dict(
        created_utc=now(), accepted_cases=len(rows), excluded_cases=excluded,
        note="Excluded or uncomputed cases are absent; no missing outcome is represented as zero."))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("solve", "collect"), default="solve")
    parser.add_argument("--capacities", type=int, nargs="+", default=CONFIG["capacities"])
    parser.add_argument("--p0s", type=float, nargs="+", default=CONFIG["outside_options"])
    parser.add_argument("--sellers", type=int, nargs="+", help="Override the configured N values at every selected C")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-states", type=int, default=CONFIG["maximum_joint_states"])
    parser.add_argument("--max-evaluations", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-illustrative", action="store_true", help="Skip the explicitly selected cutoff example")
    args = parser.parse_args()
    if (not 1 <= args.workers <= 3 or args.max_states < 2 or args.max_evaluations < 1
            or any(c not in CONFIG["capacities"] for c in args.capacities)
            or len(set(args.capacities)) != len(args.capacities)
            or any(p0 not in CONFIG["outside_options"] for p0 in args.p0s)
            or len(set(args.p0s)) != len(args.p0s)
            or (args.sellers is not None and (min(args.sellers) < 1 or len(set(args.sellers)) != len(args.sellers)))):
        parser.error("Use distinct configured capacities/outside options, positive seller counts, and one to three workers")
    sources, extra_sources = numerical_sources(), extension_sources()
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    write_json(CAMPAIGN / "config.json", CONFIG)
    if args.stage == "solve":
        requested = expected_cases(args.capacities, args.p0s, args.sellers,
                                   include_illustrative=not args.no_illustrative)
        manifest = dict(created_utc=now(), status="running", configuration=CONFIG,
            requested_capacities=args.capacities, requested_outside_options=args.p0s,
            requested_sellers={str(c): seller_counts(c, args.sellers) for c in args.capacities},
            expected_case_count=len(requested), max_states=args.max_states,
            sources=sources, extension_sources=extra_sources, completed=[], failed=[], skipped=[],
            software_versions={name: version(name) for name in ("numpy", "scipy", "matplotlib")})
        write_json(CAMPAIGN / "data/manifest.json", manifest)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            jobs = [executor.submit(solve_sequence, c, p0, seller_counts(c, args.sellers), sources, extra_sources,
                    max_states=args.max_states, max_evaluations=args.max_evaluations, force=args.force)
                    for c in args.capacities for p0 in args.p0s]
            example = CONFIG.get("illustrative_example")
            if not args.no_illustrative and example and example["capacity"] in args.capacities:
                counts = example["seller_counts"] if args.sellers is None else sorted(set(args.sellers)&set(example["seller_counts"]))
                if counts:
                    jobs.append(executor.submit(solve_sequence, example["capacity"], example["p0"], counts,
                        sources, extra_sources, max_states=args.max_states,
                        max_evaluations=args.max_evaluations, force=args.force))
            for future in as_completed(jobs):
                result = future.result()
                for field in ("completed", "failed", "skipped"):
                    manifest[field].extend(result[field])
                write_json(CAMPAIGN / "data/manifest.json", manifest)
        status = "complete" if not manifest["failed"] and not manifest["skipped"] else "incomplete"
        manifest.update(status=status, completed_utc=now())
        write_json(CAMPAIGN / "data/manifest.json", manifest)
    rows = collect(sources, extra_sources)
    write_json(CAMPAIGN / "data/source_manifest.json", dict(created_utc=now(), numerical_sources=sources,
        extension_sources=extra_sources, campaign_entrypoint_sha256=digest(__file__)))
    print(f"Collected {len(rows)} certified seller-count cases.", flush=True)


if __name__ == "__main__":
    main()
