"""Independent finite-grid refinement workers, seeded by certified 101 cases.

The original 101 campaign is read-only. Each worker owns one output case and
keeps the original equilibrium certificates and Monte Carlo configuration.
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


# Import before numerical packages: retain one BLAS thread per worker process.
RUN = module("refine_201_original", HERE / "run_refinement.py")
from oblivious.layout import resolve_saved_path


def project_path(value):
    return resolve_saved_path(value, project_root=RUN.ROOT)


def worker_sources():
    sources = RUN.source_hashes()
    for path in (Path(__file__).resolve(), HERE / "oe_active_set.py"):
        sources[str(path.relative_to(RUN.ROOT))] = RUN.digest(path)
    return sources


def validate_manifest(manifest):
    """Check all controller-recorded sources and inputs, including this worker."""
    if manifest["configuration"].get("input_sha256") != manifest.get("input_sha256"):
        raise ValueError("Campaign input hashes are not bound to its scientific configuration")
    expected = manifest["source_sha256"]
    if any(expected.get(path) != value for path, value in worker_sources().items()):
        raise ValueError("Numerical worker sources changed; use a new campaign name")
    for kind in ("source_sha256", "input_sha256"):
        for name, value in manifest.get(kind, {}).items():
            path = project_path(name)
            if not path.is_file() or RUN.digest(path) != value:
                raise ValueError(f"Campaign {kind} changed: {path}")
    if RUN.fingerprint(manifest["configuration"], expected) != manifest["fingerprint"]:
        raise ValueError("Campaign configuration or fingerprint changed")


def scalar_tolerances(config):
    return dict(dp_tol=config["dp_tol"], stationary_tol=config.get("stationary_tol", 1e-11),
                stability_tol=config.get("stability_tol", 1e-7),
                near_tie_tol=config.get("near_tie_tol", 1e-7))


def experiment_for(model, n, params, config):
    if model in ("p0", "monopoly"):
        from oblivious.oe_p0.code.core import ScalarExperiment
        cls, extra = ScalarExperiment, {}
        if model == "monopoly":
            from oblivious.oe_following_monopoly.code.model import QualityExperiment
            cls, extra = QualityExperiment, {"weighting": "purchase"}
        return cls(n=n, params=params, order=config["order"],
                   **scalar_tolerances(config), **extra)
    return RUN.experiment_for(model, n, params, config)


def previous_case(manifest, model, rho):
    """Read the certified seed; the failed original OE .99 is never a seed."""
    config = manifest["configuration"]
    source = project_path(config["source_campaign"])
    old = json.loads((source / "manifest.json").read_text())
    if RUN.source_hashes() != old["source_sha256"]:
        raise ValueError("The 101 campaign's numerical sources changed")
    if old["fingerprint"] != manifest.get("source_campaign_fingerprint", old["fingerprint"]):
        raise ValueError("The 101 campaign's scientific identity changed")
    n = config.get("seed_grid", 101)
    if model == "oe" and rho == .99:
        retry_directory = config.get("oe_retry")
        if not retry_directory:
            raise ValueError("OE .99 requires the certified isolated retry as its 101 seed")
        retry = project_path(retry_directory)
        seed_manifest = json.loads((retry / "manifest.json").read_text())
        if seed_manifest.get("source_campaign_fingerprint") != old["fingerprint"]:
            raise ValueError("OE retry belongs to another source campaign")
        base, seed_fingerprint = retry / "solution", seed_manifest["fingerprint"]
    else:
        base = RUN.case_dir(source, model, rho, n) / "solution"
        seed_fingerprint = old["fingerprint"]
    case = RUN.load_checked(base)
    settings = case.get("configuration", {}) if case else {}
    if (not case or not RUN.verified(case.get("selected"))
            or case.get("campaign_fingerprint") != seed_fingerprint
            or case.get("n") != n or case.get("rho") != rho
            or settings.get("parameters") != dict(config["parameters"], rho=rho)
            or settings.get("order") != config["order"]
            or not case.get("demand_audit", {}).get("verified")):
        raise ValueError(f"Missing or incompatible certified {model} rho={rho:g} n={n} seed")
    return case, dict(kind="certified_101_grid_continuation", base=str(base),
        source_fingerprint=seed_fingerprint,
        files={suffix: RUN.digest(base.with_suffix(suffix)) for suffix in (".json", ".npz")})


def policy_seeds(previous, grid):
    import numpy as np
    seeds = []
    for candidate in [previous.get("selected"), *previous.get("candidates", [])]:
        if not RUN.verified(candidate):
            continue
        old_grid = np.asarray(candidate["grid"], dtype=float)
        alpha = np.asarray(candidate["policy"], dtype=float)
        if (old_grid.ndim != 1 or alpha.shape != old_grid.shape
                or not np.isfinite(old_grid).all() or np.any(np.diff(old_grid) <= 0)
                or old_grid[0] != 0 or old_grid[-1] != 1
                or not np.isfinite(alpha).all() or np.any((alpha < 0) | (alpha > 1))):
            raise ValueError("Malformed certified seed policy")
        policy = np.interp(grid, old_grid, alpha)
        if not any(np.array_equal(policy, old) for _, old in seeds):
            seeds.append((f"previous_grid_branch_{len(seeds)}", policy))
    if not seeds:
        raise ValueError("No certified policy branch in the 101 seed")
    return seeds


def solve_distribution(experiment, previous, config, folder, fingerprint):
    """Try movable active faces first, then the original solver if needed."""
    import numpy as np
    active = module("refine_201_active_set", HERE / "oe_active_set.py")
    seeds = policy_seeds(previous, experiment.grid)
    candidates, attempts = [], []

    def attempt(method, name, seed):
        seed_hash = sha256(np.ascontiguousarray(seed).tobytes()).hexdigest()
        base = Path(folder) / "search_checkpoints" / f"{method}_{name}"
        saved = RUN.load_checked(base)
        if saved is not None:
            if (saved.get("campaign_fingerprint") != fingerprint
                    or saved.get("seed_sha256") != seed_hash):
                raise ValueError("OE search attempt has incompatible provenance")
            print(f"  Reusing {method}: {name}", flush=True)
        else:
            print(f"  {method}: {name}", flush=True)
            try:
                progress = lambda text: print("    " + str(text), flush=True)
                if method == "active_set":
                    result = active.solve_active_set(experiment, seed,
                        max_evaluations=config.get("active_max_evaluations", 150), progress=progress)
                else:
                    result = experiment.solve(initial_policy=seed, prefer_local=True,
                        max_evaluations=config["max_evaluations"], progress=progress)
                saved = dict(candidate=result)
            except (ArithmeticError, RuntimeError, ValueError) as exc:
                saved = dict(error_type=type(exc).__name__, error=str(exc))
                print(f"    Numerical failure: {exc}", flush=True)
            saved.update(campaign_fingerprint=fingerprint, seed_sha256=seed_hash,
                         start=name, method=method)
            RUN.save_checked(base, saved)
        attempts.append(saved)
        candidate = saved.get("candidate")
        if RUN.verified(candidate) and not any(np.max(abs(candidate["policy"] - old["policy"])) <= 1e-6
                                               for old in candidates):
            candidates.append(candidate)

    # Preserve every distinct certified 101 branch as a search proposal.
    for name, seed in seeds:
        attempt("active_set", name, seed)
    if config["multistart"] or not candidates:
        fallback = seeds + [("product_1", np.zeros(experiment.n)),
                            ("half", np.full(experiment.n, .5)),
                            ("product_2", np.ones(experiment.n))]
        for name, seed in fallback:
            attempt("original_solver", name, seed)
    evaluated = [item["candidate"] for item in attempts if "candidate" in item]
    if not evaluated:
        raise ArithmeticError("No evaluable OE candidate; inspect worker.log")
    selected = candidates[0] if candidates else min(evaluated,
        key=lambda item: item.get("dp_diagnostics", {}).get("conditional_regret", float("inf")))
    return dict(rho=experiment.params.rho, n=experiment.n, selected=selected,
        candidates=candidates, attempts=attempts, configuration=experiment.configuration(),
        selection_rule="First certified continuation candidate; original solver fallback if none certified",
        search_scope="Continue saved branches with movable pure/mixed decisions; no uniqueness or exhaustive-root claim")


def simulate(folder, model, experiment, case, manifest):
    from oblivious.equilibrium_comparison.code.simulation import simulate_pair
    from oblivious.equilibrium_comparison.code.cross_rho import summarize_profile
    config = manifest["configuration"]
    selected = case["selected"]
    policy_hash = sha256(selected["policy"].tobytes()).hexdigest()
    for mode in config["simulation_modes"]:
        base = Path(folder) / ("simulation_" + mode)
        existing = RUN.load_checked(base)
        if existing is not None:
            if (existing.get("campaign_fingerprint") != manifest["fingerprint"]
                    or existing.get("policy_sha256") != policy_hash):
                raise ValueError("Saved simulation has incompatible policy or provenance")
            continue
        print(f"Simulation ({mode}): {config['replications']} independent paths", flush=True)
        pair = dict(grid=experiment.grid, params=experiment.params,
                    profiles={model: dict(grid=experiment.grid, policy=selected["policy"])})
        simulation = simulate_pair(pair, periods=config["periods"], burn_in=config["burn_in"],
            replications=config["replications"], seed=config["seed"], mode=mode,
            short_periods=0, initial=(.5, .5), assignments={model: (model, model)},
            progress=lambda done, total: print(f"  {done:,}/{total:,} periods", flush=True))
        summary = summarize_profile(experiment.grid, simulation["profiles"][model],
                                    start=config["burn_in"], length=config["periods"])
        summary.update(metadata=simulation["metadata"], predicted=case["predicted"],
            policy_sha256=policy_hash, campaign_fingerprint=manifest["fingerprint"],
            interpretation="MC intervals exclude initialization and discretization error; grid mode validates the finite kernel")
        RUN.save_checked(base, summary)


@contextmanager
def case_lock(folder):
    """An orphaned live worker keeps its case exclusive after controller exit."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".worker.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another worker is active in {folder}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def worker(directory, model, rho, n=201):
    folder = RUN.case_dir(directory, model, rho, n)
    with case_lock(folder):
        return _worker(directory, model, rho, n)


def _worker(directory, model, rho, n):
    from oblivious.oe_p0.code.core import ScalarExperiment
    from oblivious.oe_p0.code.primitives import Parameters
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    validate_manifest(manifest)
    config = manifest["configuration"]
    if model not in config["models"] or rho not in config["rhos"] or n not in config["grids"]:
        raise ValueError("Worker case not included in campaign configuration")
    params = Parameters(**config["parameters"], rho=rho)
    folder = RUN.case_dir(directory, model, rho, n)
    folder.mkdir(parents=True, exist_ok=True)
    status = dict(model=model, rho=rho, n=n, status="running", started_utc=RUN.stamp())
    RUN.atomic_json(folder / "status.json", status)
    try:
        started = time.monotonic()
        experiment = experiment_for(model, n, params, config)
        case = RUN.load_checked(folder / "solution")
        if case is not None and (case.get("campaign_fingerprint") != manifest["fingerprint"]
                or case.get("n") != n or case.get("rho") != rho
                or case.get("configuration") != experiment.configuration()):
            raise ValueError("Saved solution has incompatible provenance or case configuration")
        if case is None:
            previous, provenance = previous_case(manifest, model, rho)
            print(f"Refining {model}, rho={rho:g}, n={previous['n']} -> {n}", flush=True)
            if model in ("p0", "monopoly"):
                from oblivious.oe_p0.code.refinement import solve_refinement
                kwargs = dict(n=n, params=params, previous=previous,
                    checkpoint_dir=folder / "search_checkpoints", order=config["order"],
                    p0_points=config["p0_points"], consistency_tol=config["tol"],
                    progress=lambda text: print("  " + str(text), flush=True),
                    **scalar_tolerances(config))
                if model == "monopoly":
                    from oblivious.oe_following_monopoly.code.model import QualityExperiment
                    kwargs.update(experiment_class=QualityExperiment,
                                  experiment_kwargs={"weighting": "purchase"})
                case = solve_refinement(**kwargs)
            elif model == "oe":
                case = solve_distribution(experiment, previous, config, folder, manifest["fingerprint"])
            else:
                case = RUN.solve_own(model, experiment, previous, config)
            audit = case.get("demand_audit") or ScalarExperiment(
                n=n, params=params, order=config["order"]).demand_audit()
            case.update(demand_audit=audit, provenance=provenance,
                        campaign_fingerprint=manifest["fingerprint"],
                        source_campaign_fingerprint=manifest.get("source_campaign_fingerprint"),
                        elapsed_seconds=time.monotonic() - started)
            if RUN.verified(case.get("selected")):
                case["predicted"] = RUN.stationary_predictions(experiment, case["selected"])
            RUN.save_checked(folder / "solution", case)
        if not RUN.verified(case.get("selected")):
            status.update(status="unverified", finished_utc=RUN.stamp(),
                          message="No certified equilibrium found; not simulated")
            RUN.atomic_json(folder / "status.json", status)
            return 2
        simulate(folder, model, experiment, case, manifest)
        validate_manifest(manifest)
        status.update(status="complete", finished_utc=RUN.stamp(),
                      elapsed_seconds=time.monotonic() - started)
        RUN.atomic_json(folder / "status.json", status)
        return 0
    except BaseException as exc:
        status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            finished_utc=RUN.stamp(), error_type=type(exc).__name__, message=str(exc))
        RUN.atomic_json(folder / "status.json", status)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--model", choices=RUN.MODELS, required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--n", type=int, default=201)
    args = parser.parse_args(argv)
    return worker(args.directory, args.model, args.rho, args.n)


if __name__ == "__main__":
    raise SystemExit(main())
