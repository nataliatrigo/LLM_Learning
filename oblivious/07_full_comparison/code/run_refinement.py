"""Refine the four comparison models, one model at a time and several rho in parallel.

Each subprocess owns one model/rho/grid directory. Existing experiments are
read-only inputs: imported candidates are freshly certified before reuse.
Interrupting the controller stops its workers; completed solutions and scalar
search checkpoints can be reused with --resume.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from hashlib import sha256
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

# Set these before importing numerical libraries, including in worker processes.
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/full-refinement-mpl")

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / "oblivious/07_full_comparison/data"
MODELS = ("p0", "oe", "conditional", "monopoly")
LABELS = {"p0": "OE p0 (endogenous)", "oe": "OE distribution",
          "conditional": "Conditional OE", "monopoly": "Following monopoly / OE quality"}
RHOS = (.6, .7, .8, .9, .95, .97, .99)
SOURCES = {
    "p0": ("oblivious/02_oe_p0/data/oe_p0_n101", "oblivious/02_oe_p0/data/rho_forgetting_mixed"),
    "oe": ("oblivious/01_oe_distribution/data/oe_distribution_n101", "oblivious/01_oe_distribution/data/rho_forgetting"),
    "conditional": ("oblivious/04_oe_conditional_distribution/data/rho_forgetting",),
    "monopoly": ("oblivious/03_oe_following_monopoly/data/quality_sales_n51",),
}


def stamp():
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_hashes():
    paths = {Path(__file__).resolve()}
    for package in ("oblivious/02_oe_p0", "oblivious/01_oe_distribution", "oblivious/04_oe_conditional_distribution",
                    "oblivious/03_oe_following_monopoly", "oblivious/06_equilibrium_comparison"):
        for path in (ROOT / package / "code").glob("*"):
            if path.suffix in (".py", ".cpp", ".h") and not path.name.startswith("test_"):
                paths.add(path)
    paths.add(ROOT / "forgetting/Code/model.py")
    return {str(p.relative_to(ROOT)): digest(p) for p in sorted(paths)}


def fingerprint(configuration, sources):
    return sha256(json.dumps(dict(configuration=configuration, sources=sources),
                             sort_keys=True).encode()).hexdigest()


def case_dir(directory, model, rho, n):
    return Path(directory) / model / ("rho_" + str(rho).replace(".", "p")) / f"n{n:03d}"


def save_checked(base, bundle):
    """Commit arrays, metadata, then hashes; an interrupted write is not reusable."""
    from oblivious.oe_p0.code.run_experiment import save_bundle
    base = Path(base)
    base.parent.mkdir(parents=True, exist_ok=True)
    temporary = base.with_name("pending_" + base.name)
    save_bundle(temporary, bundle)
    for suffix in (".npz", ".json"):
        temporary.with_suffix(suffix).replace(base.with_suffix(suffix))
    atomic_json(base.with_name(base.name + "_sha256.json"),
                {suffix: digest(base.with_suffix(suffix)) for suffix in (".json", ".npz")})


def load_checked(base):
    from oblivious.oe_p0.code.run_experiment import load_bundle
    base = Path(base)
    hashes = base.with_name(base.name + "_sha256.json")
    if not hashes.exists():
        return None
    expected = json.loads(hashes.read_text())
    for suffix in (".json", ".npz"):
        path = base.with_suffix(suffix)
        if not path.exists() or expected.get(suffix) != digest(path):
            raise ValueError(f"Saved checkpoint changed: {path}")
    return load_bundle(base)


def verified(candidate):
    if not candidate or not candidate.get("verified") or not candidate.get("candidate_verified"):
        return False
    for key in ("dp_diagnostics", "stationary_diagnostics"):
        if not candidate.get(key, {}).get("verified"):
            return False
    return all(candidate[key].get("verified") for key in
               ("conditional_diagnostics", "monopoly_certificate") if key in candidate)


def experiment_for(model, n, params, config):
    kwargs = dict(n=n, params=params, order=config["order"])
    if model == "p0":
        from oblivious.oe_p0.code.core import ScalarExperiment
        return ScalarExperiment(**kwargs)
    if model == "monopoly":
        from oblivious.oe_following_monopoly.code.model import QualityExperiment
        return QualityExperiment(**kwargs, weighting="purchase")
    if model == "oe":
        from oblivious.oe_distribution.code.model import JointMarginalExperiment
        return JointMarginalExperiment(**kwargs, tol=config["tol"], dp_tol=config["dp_tol"])
    from oblivious.oe_conditional_distribution.code.model import ConditionalExperiment
    return ConditionalExperiment(**kwargs, tol=config["tol"], dp_tol=config["dp_tol"])


def historical_case(model, rho, n, params, config):
    """Only use completed case files with matching economics and quadrature."""
    from oblivious.oe_p0.code.run_experiment import case_tag, load_bundle
    tag = case_tag(rho, n)
    for folder in SOURCES[model]:
        directory = ROOT / folder
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text())
        if tag not in manifest.get("completed_cases", []):
            continue
        base = directory / "arrays" / tag
        hashes = manifest.get("checkpoint_sha256", {})
        for suffix in (".json", ".npz"):
            path = base.with_suffix(suffix)
            relative = str(path.relative_to(directory))
            if relative in hashes and digest(path) != hashes[relative]:
                raise ValueError(f"Historical checkpoint changed: {path}")
        case = load_bundle(base)
        settings = case.get("configuration", {})
        if (case.get("rho") != rho or case.get("n") != n
                or settings.get("parameters") != asdict(params)
                or settings.get("order") != config["order"]):
            continue
        provenance = dict(kind="historical_read_only", base=str(base.relative_to(ROOT)),
                          files={suffix: digest(base.with_suffix(suffix)) for suffix in (".json", ".npz")},
                          manifest_sha256=digest(manifest_path))
        return case, provenance
    return None, None


def recertify(case, model, experiment, config):
    import numpy as np
    from oblivious.oe_p0.code.run_experiment import select_candidate
    old = case.get("selected")
    proposals = ([old] if old is not None else []) + case.get("candidates", [])
    candidates, seen, failures = [], [], []
    for candidate in proposals:
        if not verified(candidate):
            continue
        alpha = np.asarray(candidate["policy"])
        p0 = candidate.get("p0")
        if any(np.array_equal(alpha, a) and p0 == p for a, p in seen):
            continue
        seen.append((alpha, p0))
        try:
            if model in ("p0", "monopoly"):
                fresh = experiment.evaluate_mixed(p0, alpha, config["tol"])
            else:
                fresh = experiment.evaluate(alpha)
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            failures.append(dict(p0=p0,error=str(exc)))
            print(f"  Historical candidate could not be certified: {exc}",flush=True)
            continue
        if verified(fresh):
            candidates.append(fresh)
    if not candidates:
        return None
    selected = select_candidate(candidates) if model in ("p0", "monopoly") else candidates[0]
    return dict(rho=experiment.params.rho, n=experiment.n, candidates=candidates,
                selected=selected, configuration=experiment.configuration(),
                historical_recertification_failures=failures,
                selection_rule="Historical selection order, freshly certified; scalar models prefer interior then lowest p0")


def previous_case(directory, model, rho, n, params, config):
    choices = sorted({51, *config["grids"]}, reverse=True)
    for size in choices:
        if size >= n:
            continue
        saved = load_checked(case_dir(directory, model, rho, size) / "solution")
        if saved and verified(saved.get("selected")):
            return saved, dict(kind="campaign_previous_grid", n=size,
                               base=str(case_dir(directory, model, rho, size) / "solution"))
        historical, provenance = historical_case(model, rho, size, params, config)
        if historical and verified(historical.get("selected")):
            return historical, provenance
    return None, None


def solve_own(model, experiment, previous, config):
    import numpy as np
    candidates, attempts, seeds = [], [], []
    if previous:
        for candidate in [previous.get("selected"), *previous.get("candidates", [])]:
            if verified(candidate):
                seed = np.interp(experiment.grid, candidate["grid"], candidate["policy"])
                if not any(np.array_equal(seed, old) for _, old in seeds):
                    seeds.append(("previous_grid", seed))
    if config["multistart"] or not seeds:
        seeds += [("product_1", np.zeros(experiment.n)), ("half", np.full(experiment.n, .5)),
                  ("product_2", np.ones(experiment.n))]
    def attempt(name,seed):
        print(f"  Search from {name}", flush=True)
        try:
            result = experiment.solve(initial_policy=seed, prefer_local=True,
                max_evaluations=config["max_evaluations"],
                progress=lambda message: print("    " + str(message), flush=True))
        except (ArithmeticError, RuntimeError, ValueError) as exc:
            attempts.append(dict(start=name,error=str(exc)))
            print(f"    Numerical failure: {exc}",flush=True)
            return
        attempts.append(dict(start=name, candidate=result))
        if verified(result) and not any(np.max(abs(result["policy"]-old["policy"])) <= 1e-6
                                        for old in candidates):
            candidates.append(result)
    for name,seed in seeds:
        attempt(name,seed)
    if not candidates and not config["multistart"] and any(name=="previous_grid" for name,_ in seeds):
        for name,seed in (("product_1",np.zeros(experiment.n)),("half",np.full(experiment.n,.5)),
                          ("product_2",np.ones(experiment.n))):
            attempt(name,seed)
    evaluated = [a["candidate"] for a in attempts if "candidate" in a]
    if not evaluated:
        raise ArithmeticError("No evaluable candidate after all search starts; see worker.log")
    selected = candidates[0] if candidates else min(
        evaluated,
        key=lambda c: c.get("dp_diagnostics", {}).get("conditional_regret", float("inf")))
    return dict(rho=experiment.params.rho, n=experiment.n, selected=selected,
        candidates=candidates, attempts=attempts, configuration=experiment.configuration(),
        selection_rule="Continue each saved branch; first verified continuation candidate",
        search_scope="Additional generic starts enabled" if config["multistart"] else
                     "Grid continuation; does not establish uniqueness or exhaustive root discovery")


def stationary_predictions(experiment, candidate):
    import numpy as np
    p, nu, alpha = experiment.params, candidate["nu"], candidate["policy"]
    usage = float(np.sum(nu*(experiment.game.D*alpha[:, None] + experiment.game.D.T*alpha[None, :])))
    return dict(product2_market=usage, seller_profit=.5*(p.R-p.c1-(p.c2-p.c1)*usage),
                success_rate=p.p1+(p.p2-p.p1)*usage,
                mean_reputation=float(nu.sum(axis=0) @ experiment.grid))


def worker(directory, model, rho, n):
    from oblivious.oe_p0.code.core import ScalarExperiment
    from oblivious.oe_p0.code.primitives import Parameters
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["configuration"]
    if source_hashes() != manifest["source_sha256"]:
        raise ValueError("Code changed during this campaign; use a new run name")
    params = Parameters(**config["parameters"], rho=rho)
    folder = case_dir(directory, model, rho, n)
    folder.mkdir(parents=True, exist_ok=True)
    status = dict(model=model, rho=rho, n=n, status="running", started_utc=stamp())
    atomic_json(folder / "status.json", status)
    try:
        started = time.monotonic()
        experiment = experiment_for(model, n, params, config)
        case = load_checked(folder / "solution")
        if case is None:
            historical, provenance = historical_case(model, rho, n, params, config)
            if historical:
                print("Re-certifying saved candidates", flush=True)
                case = recertify(historical, model, experiment, config)
            previous, seed_source = previous_case(directory, model, rho, n, params, config)
            if case is None:
                if model in ("p0", "monopoly"):
                    from oblivious.oe_p0.code.refinement import solve_refinement
                    kwargs = dict(n=n, params=params, previous=previous,
                        checkpoint_dir=folder / "search_checkpoints", order=config["order"],
                        p0_points=config["p0_points"], consistency_tol=config["tol"],
                        progress=lambda message: print("  " + str(message), flush=True))
                    if model == "monopoly":
                        from oblivious.oe_following_monopoly.code.model import QualityExperiment
                        kwargs["experiment_class"] = QualityExperiment
                    case = solve_refinement(**kwargs)
                else:
                    case = solve_own(model, experiment, previous, config)
                provenance = dict(kind="new_search", previous_grid=seed_source)
            audit = case.get("demand_audit") or ScalarExperiment(
                n=n, params=params, order=config["order"]).demand_audit()
            case.update(demand_audit=audit, provenance=provenance,
                        campaign_fingerprint=manifest["fingerprint"],
                        elapsed_seconds=time.monotonic()-started)
            if verified(case.get("selected")):
                case["predicted"] = stationary_predictions(experiment, case["selected"])
            save_checked(folder / "solution", case)
        if case.get("campaign_fingerprint") != manifest["fingerprint"]:
            raise ValueError("Checkpoint belongs to another configuration or code version")
        if not verified(case.get("selected")):
            status.update(status="unverified", finished_utc=stamp(),
                          message="Search completed without a certified equilibrium; not simulated")
            atomic_json(folder / "status.json", status)
            return 2
        for mode in config["simulation_modes"]:
            base = folder / ("simulation_" + mode)
            existing = load_checked(base)
            if existing is not None:
                if (existing.get("campaign_fingerprint") != manifest["fingerprint"]
                        or existing.get("policy_sha256") != sha256(case["selected"]["policy"].tobytes()).hexdigest()):
                    raise ValueError("Saved simulation has incompatible provenance or policy")
                continue
            from oblivious.equilibrium_comparison.code.simulation import simulate_pair
            from oblivious.equilibrium_comparison.code.cross_rho import summarize_profile
            selected = case["selected"]
            pair = dict(grid=experiment.grid, params=params,
                        profiles={model: dict(grid=experiment.grid, policy=selected["policy"])})
            print(f"Simulation ({mode}): {config['replications']} independent paths", flush=True)
            simulation = simulate_pair(pair, periods=config["periods"], burn_in=config["burn_in"],
                replications=config["replications"], seed=config["seed"], mode=mode,
                short_periods=0, initial=(.5, .5), assignments={model:(model, model)},
                progress=lambda done,total: print(f"  {done:,}/{total:,} periods", flush=True))
            summary = summarize_profile(experiment.grid, simulation["profiles"][model],
                start=config["burn_in"], length=config["periods"])
            summary.update(metadata=simulation["metadata"], predicted=case["predicted"],
                policy_sha256=sha256(selected["policy"].tobytes()).hexdigest(),
                campaign_fingerprint=manifest["fingerprint"],
                interpretation="MC intervals exclude initialization and discretization error; grid mode validates the finite kernel")
            save_checked(base, summary)
            del simulation, summary
        if source_hashes() != manifest["source_sha256"]:
            raise ValueError("Code changed during this job; results must be reviewed")
        status.update(status="complete", finished_utc=stamp(),
                      elapsed_seconds=time.monotonic()-started)
        atomic_json(folder / "status.json", status)
        return 0
    except BaseException as exc:
        status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                      finished_utc=stamp(), error_type=type(exc).__name__, message=str(exc))
        atomic_json(folder / "status.json", status)
        raise


def export_campaign(directory):
    """A single controller writes aggregate tables; workers only own their folders."""
    from oblivious.oe_p0.code.run_experiment import write_csv
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["configuration"]
    rows, refinement = [], []
    for model in config["models"]:
        for rho in config["rhos"]:
            previous = None
            for n in sorted(config["grids"]):
                folder = case_dir(directory, model, rho, n)
                status_path = folder / "status.json"
                status = json.loads(status_path.read_text()) if status_path.exists() else {}
                row = dict(model=model, label=LABELS[model], rho=rho, n=n,
                           status=status.get("status", "pending"))
                case = load_checked(folder / "solution")
                if case:
                    row.update(verified=verified(case.get("selected")),
                               candidates=len(case.get("candidates", [])),
                               elapsed_seconds=case.get("elapsed_seconds"),
                               **case.get("predicted", {}))
                    if row["verified"]:
                        if previous:
                            import numpy as np
                            from scipy.stats import wasserstein_distance
                            a, b = previous["selected"], case["selected"]
                            refinement.append(dict(model=model, rho=rho, base_grid=previous["n"],
                                refined_grid=n,
                                product2_change=case["predicted"]["product2_market"]-previous["predicted"]["product2_market"],
                                mean_reputation_change=case["predicted"]["mean_reputation"]-previous["predicted"]["mean_reputation"],
                                marginal_wasserstein=wasserstein_distance(a["grid"],b["grid"],
                                    u_weights=a["nu"].sum(axis=0),v_weights=b["nu"].sum(axis=0)),
                                policy_max_interpolated_difference=float(np.max(abs(
                                    b["policy"]-np.interp(b["grid"],a["grid"],a["policy"]))))))
                        previous = case
                complete_simulations = True
                for mode in config["simulation_modes"]:
                    simulation = load_checked(folder / ("simulation_" + mode))
                    if simulation:
                        for metric in ("product2_market", "seller_profit", "success_rate"):
                            row.update({f"{mode}_{metric}_{k}":v
                                        for k,v in simulation["summary"][metric].items()})
                    else:
                        complete_simulations = False
                if row["status"] == "complete" and (not row.get("verified") or not complete_simulations):
                    row["status"] = "incomplete_checkpoint"
                rows.append(row)
    write_csv(directory / "summary.csv", rows)
    write_csv(directory / "grid_refinement.csv", refinement)
    return rows


def execute_campaign(directory, workers_101=3, workers_201=1):
    manifest = json.loads((directory / "manifest.json").read_text())
    config = manifest["configuration"]
    children, lock, stop = {}, threading.Lock(), threading.Event()

    def run_one(model, rho, n):
        if stop.is_set():
            return None
        folder = case_dir(directory, model, rho, n)
        status_path = folder / "status.json"
        if status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
            saved = load_checked(folder / "solution")
            simulations = [load_checked(folder / ("simulation_"+mode))
                           for mode in config["simulation_modes"]]
            if (saved is not None and verified(saved.get("selected"))
                    and saved.get("campaign_fingerprint") == manifest["fingerprint"]
                    and all(s is not None and s.get("campaign_fingerprint") == manifest["fingerprint"]
                            and s.get("policy_sha256") == sha256(saved["selected"]["policy"].tobytes()).hexdigest()
                            for s in simulations)):
                return 0
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / "worker.log").open("a") as log:
            with lock:
                if stop.is_set():
                    return None
                process = subprocess.Popen([sys.executable, "-u", "-B", str(Path(__file__).resolve()),
                    "--worker", model, str(rho), str(n), "--directory", str(directory)],
                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                children[process.pid] = process
            try:
                return process.wait()
            finally:
                with lock:
                    children.pop(process.pid, None)

    manifest.update(status="running", updated_utc=stamp())
    atomic_json(directory / "manifest.json", manifest)
    failures = []
    try:
        for model in config["models"]:
            print(f"\n{LABELS[model]}: starting model stage", flush=True)
            for n in sorted(config["grids"]):
                workers = workers_101 if n <= 101 else workers_201
                print(f"  n={n}: up to {workers} concurrent rho; logs in {directory/model}", flush=True)
                executor = ThreadPoolExecutor(max_workers=workers)
                futures = {executor.submit(run_one,model,rho,n):rho for rho in
                           sorted(config["rhos"], key=lambda r:(r not in (.95,.97,.99),r))}
                try:
                    for future in as_completed(futures):
                        rho = futures[future]
                        code = future.result()
                        print(f"  {model}, rho={rho:g}, n={n}: " +
                              ("complete" if code == 0 else f"needs review (exit {code})"), flush=True)
                        if code != 0:
                            failures.append(dict(model=model,rho=rho,n=n,exit_code=code))
                        export_campaign(directory)
                except BaseException:
                    stop.set()
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
                else:
                    executor.shutdown(wait=True)
        rows = export_campaign(directory)
        manifest.update(status="complete" if all(r["status"]=="complete" for r in rows)
                        else "needs_review", finished_utc=stamp(), failures=failures)
        atomic_json(directory / "manifest.json", manifest)
        print(f"\nResults: {directory/'summary.csv'}\nRefinement: {directory/'grid_refinement.csv'}",flush=True)
        return 0 if manifest["status"] == "complete" else 2
    except BaseException as exc:
        stop.set()
        with lock:
            processes = list(children.values())
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic()+10
        for process in processes:
            try:
                process.wait(timeout=max(.1,deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except ProcessLookupError:
                    pass
        manifest.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",
                        updated_utc=stamp(),last_error=dict(type=type(exc).__name__,message=str(exc)))
        atomic_json(directory / "manifest.json", manifest)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="refinement_101_201")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--rhos", nargs="+", type=float, default=list(RHOS))
    parser.add_argument("--grids", nargs="+", type=int, default=[101,201])
    parser.add_argument("--workers-101", type=int, default=3)
    parser.add_argument("--workers-201", type=int, default=1)
    parser.add_argument("--multistart", action="store_true",
                        help="Also search generic starts in Distribution and Conditional")
    parser.add_argument("--simulation-modes", nargs="*", choices=("grid","nearest","linear"), default=["grid"])
    parser.add_argument("--periods", type=int, default=100000)
    parser.add_argument("--burn-in", type=int, default=10000)
    parser.add_argument("--replications", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261017)
    parser.add_argument("--max-evaluations", type=int, default=1500)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker", nargs=3, metavar=("MODEL","RHO","GRID"), help=argparse.SUPPRESS)
    parser.add_argument("--directory", type=Path, help=argparse.SUPPRESS)
    tokens = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(tokens)
    if args.worker:
        model,rho,n = args.worker
        return worker(args.directory,model,float(rho),int(n))
    if (Path(args.name).name != args.name or args.name in ("", ".", "..")
            or len(set(args.models)) != len(args.models) or len(set(args.rhos)) != len(args.rhos)
            or len(set(args.grids)) != len(args.grids) or min(args.grids) < 3
            or any(n % 2 != 1 for n in args.grids) or any(not .5 < r <= .99 for r in args.rhos)
            or min(args.workers_101,args.workers_201,args.max_evaluations) < 1
            or args.periods < 2 or args.burn_in < 0 or args.replications < 2
            or len(set(args.simulation_modes)) != len(args.simulation_modes)):
        parser.error("Require distinct models/rhos/odd grids, rho<=.99, positive workers and valid simulation settings")
    directory = OUTPUT / args.name
    manifest_path = directory / "manifest.json"
    if args.status:
        if not manifest_path.exists():
            parser.error("No campaign with this name")
        rows = export_campaign(directory)
        for row in rows:
            print(f"{row['model']:11s} rho={row['rho']:g} n={row['n']:3d}: {row['status']}")
        return 0
    if args.dry_run:
        print(f"Output: {directory}\nModels sequentially: {', '.join(args.models)}")
        print(f"Rho per model: {args.rhos}; grids sequentially: {sorted(args.grids)}")
        print(f"Concurrency: <=101: {args.workers_101}; >101: {args.workers_201}; one BLAS thread per process")
        print("Historical results remain unchanged and are freshly certified before reuse.")
        return 0
    if args.resume:
        allowed = {"--resume", "--name", "--workers-101", "--workers-201", "--status", "--dry-run"}
        if any(token.split("=")[0] not in allowed for token in tokens if token.startswith("--")):
            parser.error("Resume restores saved settings; only name and worker counts may be supplied")
        if not manifest_path.exists():
            parser.error("No campaign to resume")
        manifest = json.loads(manifest_path.read_text())
        if source_hashes() != manifest["source_sha256"]:
            parser.error("Code changed since this campaign; use a new --name to preserve provenance")
    else:
        if directory.exists():
            parser.error("Existing campaign protected; use --resume or another --name")
        base = json.loads((OUTPUT / "comparison.json").read_text())["parameters"]
        config = dict(models=args.models,rhos=args.rhos,grids=sorted(args.grids),parameters=base,
            order=128,tol=1e-7,dp_tol=1e-10,p0_points=41,multistart=args.multistart,
            max_evaluations=args.max_evaluations,simulation_modes=args.simulation_modes,
            periods=args.periods,burn_in=args.burn_in,replications=args.replications,seed=args.seed)
        sources = source_hashes()
        manifest = dict(configuration=config,source_sha256=sources,
            fingerprint=fingerprint(config,sources),status="prepared",created_utc=stamp(),
            scope="Finite-grid equilibrium refinement; neither uniqueness nor continuous-state convergence is assumed")
        directory.mkdir(parents=True)
        atomic_json(manifest_path,manifest)
    return execute_campaign(directory,args.workers_101,args.workers_201)


if __name__ == "__main__":
    raise SystemExit(main())
