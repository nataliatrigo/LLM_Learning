"""Retry OE distribution at rho=.99, n=101 without changing the original campaign.

Run from the project root. Completed attempts survive interruption; use
--resume to continue. Add --simulate to create the comparison's grid simulation
after finding a certified equilibrium. The original campaign remains read-only.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


# Import this before numerical libraries: it limits numerical worker threads.
RUN = module("oe_retry_original_refinement", HERE / "run_refinement.py")

import numpy as np


def path_key(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(RUN.ROOT))
    except ValueError:
        return str(path)


def retry_sources():
    sources = RUN.source_hashes()
    for path in (Path(__file__).resolve(), HERE / "oe_active_set.py"):
        sources[path_key(path)] = RUN.digest(path)
    return sources


@contextmanager
def output_lock(directory):
    with (directory / ".retry.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another retry is running in this output folder") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def seed_policies(case, seed_case=None, n=101):
    """Recover search seeds, including rejected candidates; claim no validity."""
    proposals = []
    if seed_case is not None:
        candidate = seed_case.get("selected", seed_case)
        proposals.append(("explicit_seed", candidate))
    proposals.append(("previous_selected", case.get("selected")))
    proposals.extend(("previous_attempt_" + str(i), attempt.get("candidate"))
                     for i, attempt in enumerate(case.get("attempts", [])))
    proposals.extend(("previous_candidate_" + str(i), candidate)
                     for i, candidate in enumerate(case.get("candidates", [])))
    seeds, seen = [], set()
    for label, candidate in proposals:
        if not isinstance(candidate, dict):
            continue
        grid = np.asarray(candidate.get("grid", []), dtype=float)
        policy = np.asarray(candidate.get("policy", []), dtype=float)
        if (grid.ndim != 1 or grid.size < 3 or policy.shape != grid.shape
                or np.any(~np.isfinite(grid)) or np.any(np.diff(grid) <= 0)
                or grid[0] != 0 or grid[-1] != 1
                or np.any(~np.isfinite(policy)) or np.any((policy < 0) | (policy > 1))):
            raise ValueError(f"Invalid initial policy: {label}")
        rule = np.interp(np.linspace(0, 1, n), grid, policy)
        digest = sha256(rule.tobytes()).hexdigest()
        if digest not in seen:
            seen.add(digest)
            seeds.append(dict(name=label, policy=rule, policy_sha256=digest))
    if not seeds:
        raise ValueError("The source case contains no usable initial policy")
    return seeds


def read_inputs(source_directory, seed_base):
    manifest_path = source_directory / "manifest.json"
    original = json.loads(manifest_path.read_text())
    if RUN.source_hashes() != original["source_sha256"]:
        raise ValueError("Original campaign sources changed; review before retrying")
    base = RUN.case_dir(source_directory, "oe", .99, 101) / "solution"
    case = RUN.load_checked(base)
    if case is None:
        raise ValueError("Missing checked OE distribution .99/101 source checkpoint")
    config = original["configuration"]
    if (case.get("campaign_fingerprint") != original["fingerprint"]
            or case.get("n") != 101 or case.get("rho") != .99
            or case.get("configuration", {}).get("parameters")
            != dict(config["parameters"], rho=.99)
            or case["configuration"].get("order") != config["order"]):
        raise ValueError("The source checkpoint has incompatible provenance or economics")
    if not case.get("demand_audit", {}).get("verified"):
        raise ValueError("The original numerical demand matrix was not verified")
    files = [manifest_path, base.with_suffix(".json"), base.with_suffix(".npz"),
             base.with_name(base.name + "_sha256.json")]
    seed_case = None
    if seed_base is not None:
        from oblivious.oe_p0.code.run_experiment import load_bundle
        seed_case = load_bundle(seed_base)
        files.extend(seed_base.with_suffix(suffix) for suffix in (".json", ".npz"))
    inputs = {path_key(path): RUN.digest(path) for path in files}
    return original, case, seed_policies(case, seed_case), inputs


def run_simulation(directory, manifest, solution, experiment):
    """Use precisely the original campaign's grid Monte Carlo configuration."""
    base = directory / "simulation_grid"
    policy = solution["selected"]["policy"]
    policy_hash = sha256(policy.tobytes()).hexdigest()
    existing = RUN.load_checked(base)
    if existing is not None:
        if (existing.get("campaign_fingerprint") != manifest["fingerprint"]
                or existing.get("policy_sha256") != policy_hash):
            raise ValueError("The saved retry simulation has incompatible provenance")
        return
    from oblivious.equilibrium_comparison.code.simulation import simulate_pair
    from oblivious.equilibrium_comparison.code.cross_rho import summarize_profile
    config = manifest["simulation_configuration"]
    print("Simulating certified equilibrium on the 101-point grid", flush=True)
    pair = dict(grid=experiment.grid, params=experiment.params,
                profiles={"oe": dict(grid=experiment.grid, policy=policy)})
    simulation = simulate_pair(pair, **config, mode="grid", short_periods=0,
        initial=(.5, .5), assignments={"oe": ("oe", "oe")},
        progress=lambda done, total: print(f"  {done:,}/{total:,} periods", flush=True))
    summary = summarize_profile(experiment.grid, simulation["profiles"]["oe"],
                                start=config["burn_in"], length=config["periods"])
    summary.update(metadata=simulation["metadata"], predicted=solution["predicted"],
        policy_sha256=policy_hash, campaign_fingerprint=manifest["fingerprint"],
        source_campaign_fingerprint=manifest["source_campaign_fingerprint"],
        interpretation="MC intervals exclude initialization and discretization error; grid mode validates the finite kernel")
    RUN.save_checked(base, summary)


def execute_retry(source_directory, directory, *, max_evaluations=150, resume=False,
                  seed_base=None, simulate=False):
    source_directory, directory = Path(source_directory).resolve(), Path(directory).resolve()
    if directory == source_directory or source_directory in directory.parents:
        raise ValueError("Retry output must be outside the original campaign")
    if max_evaluations < 1:
        raise ValueError("A positive search budget is required")
    seed_base = Path(seed_base).resolve() if seed_base is not None else None
    original, source_case, seeds, inputs = read_inputs(source_directory, seed_base)
    old_config = original["configuration"]
    configuration = dict(model="oe", rho=.99, n=101, parameters=old_config["parameters"],
        order=old_config["order"], tol=old_config["tol"], dp_tol=old_config["dp_tol"],
        max_evaluations=max_evaluations, source_campaign=str(source_directory),
        seed_base=str(seed_base) if seed_base is not None else None,
        input_sha256=inputs)
    sources = retry_sources()
    fingerprint = RUN.fingerprint(configuration, sources)
    manifest_path = directory / "manifest.json"
    if directory.exists() and any(directory.iterdir()) and not resume:
        raise ValueError("Retry output already exists; use --resume or another --output-name")
    if resume and not manifest_path.exists():
        raise ValueError("Cannot resume without a retry manifest")
    directory.mkdir(parents=True, exist_ok=True)
    with output_lock(directory):
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("fingerprint") != fingerprint:
                raise ValueError("Retry sources, inputs or configuration changed; use another output name")
        else:
            manifest = dict(configuration=configuration, fingerprint=fingerprint,
                source_sha256=sources, input_sha256=inputs,
                source_campaign_fingerprint=original["fingerprint"],
                simulation_configuration={key: old_config[key] for key in
                    ("periods", "burn_in", "replications", "seed")},
                started_utc=RUN.stamp(), status="running")
        manifest.update(status="running", resumed_utc=RUN.stamp() if resume else None)
        RUN.atomic_json(manifest_path, manifest)
        try:
            from oblivious.oe_p0.code.primitives import Parameters
            experiment = RUN.experiment_for("oe", 101,
                Parameters(**configuration["parameters"], rho=.99), configuration)
            solution = RUN.load_checked(directory / "solution")
            if solution is not None and solution.get("campaign_fingerprint") != fingerprint:
                raise ValueError("Retry solution has incompatible provenance")
            if solution is None or not RUN.verified(solution.get("selected")):
                algorithm = module("oe_retry_active_set", HERE / "oe_active_set.py")
                attempted, candidates, elapsed = [], [], time.monotonic()
                for index, seed in enumerate(seeds):
                    base = directory / "attempts" / f"{index:03d}_{seed['name']}"
                    attempt = RUN.load_checked(base)
                    if attempt is not None:
                        if (attempt.get("campaign_fingerprint") != fingerprint
                                or attempt.get("seed_sha256") != seed["policy_sha256"]):
                            raise ValueError("Saved retry attempt has incompatible provenance")
                        print(f"Reusing completed attempt {index + 1}: {seed['name']}", flush=True)
                    else:
                        print(f"Attempt {index + 1}/{len(seeds)}: {seed['name']}", flush=True)
                        candidate = algorithm.solve_active_set(experiment, seed["policy"],
                            max_evaluations=max_evaluations,
                            progress=lambda message: print("  " + str(message), flush=True))
                        attempt = dict(candidate=candidate, start=seed["name"],
                            seed_sha256=seed["policy_sha256"], campaign_fingerprint=fingerprint)
                        RUN.save_checked(base, attempt)
                    candidate = attempt["candidate"]
                    attempted.append(dict(start=seed["name"], base=str(base),
                                          verified=RUN.verified(candidate)))
                    candidates.append(candidate)
                    selected = min(candidates, key=lambda item: (not RUN.verified(item),
                        item.get("dp_diagnostics", {}).get("conditional_regret", float("inf"))))
                    solution = dict(rho=.99, n=101, selected=selected,
                        candidates=[item for item in candidates if RUN.verified(item)],
                        attempts=attempted, configuration=experiment.configuration(),
                        demand_audit=source_case["demand_audit"],
                        campaign_fingerprint=fingerprint,
                        source_campaign_fingerprint=original["fingerprint"],
                        provenance=dict(kind="isolated_active_set_retry",
                            source_campaign=str(source_directory), input_sha256=inputs),
                        elapsed_seconds=time.monotonic() - elapsed,
                        search_scope="Bounded mixed-strategy search; no uniqueness or continuous-grid convergence claim")
                    if RUN.verified(selected):
                        solution["predicted"] = RUN.stationary_predictions(experiment, selected)
                    RUN.save_checked(directory / "solution", solution)
                    if RUN.verified(selected):
                        break
            verified = RUN.verified(solution.get("selected"))
            if verified and simulate:
                run_simulation(directory, manifest, solution, experiment)
            if retry_sources() != sources:
                raise ValueError("Retry sources changed during execution; review saved results")
            manifest.update(status="complete" if verified else "unverified",
                verified=verified, finished_utc=RUN.stamp(),
                simulation_grid_saved=RUN.load_checked(directory / "simulation_grid") is not None)
            RUN.atomic_json(manifest_path, manifest)
            print(f"{'Certified equilibrium saved' if verified else 'No certified equilibrium found'}: {directory}", flush=True)
            return 0 if verified else 2
        except BaseException as exc:
            manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                finished_utc=RUN.stamp(), error_type=type(exc).__name__, message=str(exc))
            RUN.atomic_json(manifest_path, manifest)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-campaign", default="refinement_101_201")
    parser.add_argument("--output-name", default="oe099_retry_n101")
    parser.add_argument("--max-evaluations", type=int, default=150)
    parser.add_argument("--seed-base", type=Path,
                        help="Optional .json/.npz bundle base; used only as a policy seed")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--simulate", action="store_true",
                        help="Also run the original campaign's grid simulation after certification")
    args = parser.parse_args(argv)
    for name in (args.source_campaign, args.output_name):
        if Path(name).name != name or name in ("", ".", ".."):
            parser.error("Campaign and output names must be simple folder names")
    if args.max_evaluations < 1:
        parser.error("--max-evaluations must be positive")
    try:
        return execute_retry(RUN.OUTPUT / args.source_campaign, RUN.OUTPUT / args.output_name,
            max_evaluations=args.max_evaluations, resume=args.resume,
            seed_base=args.seed_base, simulate=args.simulate)
    except (ArithmeticError, RuntimeError, ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
