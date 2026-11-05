"""Run the four 201-point models locally, reusing certified 101 inputs.

The new campaign owns its results. The old campaign and the repaired OE .99
case are read-only seeds. Scalar checkpoints from interrupted 201 jobs are
validated and copied once; numerical tolerances and the simulation design stay
the same. Use --dry-run to inspect the plan without starting calculations.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import fcntl
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time


HERE=Path(__file__).resolve().parent


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec)
    sys.modules[name]=result
    spec.loader.exec_module(result)
    return result


RUN=module("campaign_201_original",HERE/"run_refinement.py")
WORKER=module("campaign_201_worker",HERE/"refine_201_worker.py")
FINISH=module("campaign_201_interrupt",HERE/"finish_101.py")
from oblivious.layout import resolve_saved_path


def key(path):
    path=Path(path).resolve()
    return str(path.relative_to(RUN.ROOT)) if path.is_relative_to(RUN.ROOT) else str(path)


def sources():
    result=WORKER.worker_sources()
    for path in (Path(__file__).resolve(),HERE/"import_201_checkpoints.py",HERE/"finish_101.py"):
        result[key(path)]=RUN.digest(path)
    return result


def capture(base,inputs):
    for path in (base.with_suffix(".json"),base.with_suffix(".npz"),
                 base.with_name(base.name+"_sha256.json")):
        inputs[key(path)]=RUN.digest(path)


def retry_seed(retry_directory,source_manifest,source_directory):
    if retry_directory is None:
        raise ValueError("OE .99/101 needs the certified retry; run retry_oe099.py first")
    directory=Path(retry_directory).resolve()
    manifest=json.loads((directory/"manifest.json").read_text())
    config=manifest["configuration"]
    original=source_manifest["configuration"]
    if (manifest.get("source_campaign_fingerprint")!=source_manifest["fingerprint"]
            or config.get("model")!="oe" or config.get("n")!=101 or config.get("rho")!=.99
            or config.get("parameters")!=original["parameters"]
            or any(config.get(k)!=original[k] for k in ("order","tol","dp_tol"))
            or resolve_saved_path(config["source_campaign"], project_root=RUN.ROOT).resolve()!=source_directory
            or config.get("input_sha256")!=manifest.get("input_sha256")
            or manifest.get("fingerprint")!=RUN.fingerprint(config,manifest["source_sha256"])):
        raise ValueError("The OE retry has incompatible scientific provenance")
    if any(manifest["source_sha256"].get(name)!=value
           for name,value in source_manifest["source_sha256"].items()):
        raise ValueError("The OE retry used different original numerical sources")
    for collection in (manifest["source_sha256"],manifest["input_sha256"]):
        for name,expected in collection.items():
            path=resolve_saved_path(name, project_root=RUN.ROOT)
            if RUN.digest(path)!=expected:
                raise ValueError(f"OE retry input or source changed: {path}")
    return directory/"solution",manifest["fingerprint"]


def prepare_campaign(directory,source_directory,retry_directory,*,models=RUN.MODELS,
                     rhos=RUN.RHOS,simulate=True):
    directory,source_directory=Path(directory).resolve(),Path(source_directory).resolve()
    if directory==source_directory or source_directory in directory.parents:
        raise ValueError("201 output must be outside the source campaign")
    if directory.exists():
        raise ValueError("Existing campaign protected; use --resume or another --name")
    original=json.loads((source_directory/"manifest.json").read_text())
    if RUN.source_hashes()!=original["source_sha256"]:
        raise ValueError("Original numerical sources changed; cannot combine these grids")
    old=original["configuration"]
    if (not models or len(set(models))!=len(models) or not set(models)<=set(old["models"])
            or not rhos or len(set(rhos))!=len(rhos) or not set(rhos)<=set(old["rhos"])
            or any(not .5<float(r)<=.99 for r in rhos) or 101 not in old["grids"]):
        raise ValueError("Require distinct source models and rho values, with rho<=.99 and n101 inputs")
    active=FINISH.active_campaign_processes(source_directory)
    if active:
        raise RuntimeError("Stop the source campaign controller before starting 201")
    inputs,seeds={},[]
    for model in models:
        for rho in rhos:
            base=RUN.case_dir(source_directory,model,rho,101)/"solution"
            expected=original["fingerprint"]
            case=RUN.load_checked(base)
            if model=="oe" and rho==.99:
                base,expected=retry_seed(retry_directory,original,source_directory)
                case=RUN.load_checked(base)
                if case is not None and case.get("source_campaign_fingerprint")!=original["fingerprint"]:
                    raise ValueError("The repaired OE solution identifies a different source campaign")
            if (case is None or not RUN.verified(case.get("selected"))
                    or not case.get("demand_audit",{}).get("verified")
                    or case.get("campaign_fingerprint")!=expected
                    or case.get("n")!=101 or case.get("rho")!=rho
                    or case.get("configuration",{}).get("parameters")!=dict(old["parameters"],rho=rho)
                    or case["configuration"].get("order")!=old["order"]):
                raise ValueError(f"Missing or incompatible certified 101 input: {model}, rho={rho}")
            capture(base,inputs)
            seeds.append(dict(model=model,rho=rho,n=101,base=str(base),campaign_fingerprint=expected))
    configuration=dict(old,models=list(models),rhos=list(rhos),grids=[201],
        simulation_modes=["grid"] if simulate else [],source_campaign=str(source_directory),
        oe_retry=str(Path(retry_directory).resolve()) if retry_directory is not None else None,
        seed_grid=101,active_max_evaluations=150,input_sha256=inputs)
    code=sources()
    manifest=dict(configuration=configuration,source_sha256=code,input_sha256=inputs,
        source_campaign_fingerprint=original["fingerprint"],seed_sources=seeds,
        fingerprint=RUN.fingerprint(configuration,code),status="prepared",created_utc=RUN.stamp(),
        scope="Finite-grid refinement; candidates need fresh 201 certification; branch completeness and continuous-state convergence are not established")
    directory.mkdir(parents=True)
    importer=module("campaign_201_checkpoint_import",HERE/"import_201_checkpoints.py")
    imported=[]
    try:
        for model in models:
            for rho in rhos:
                entry=importer.import_checkpoints(source_directory,directory,model,rho,configuration)
                if entry is not None:
                    imported.append(entry)
        manifest["imported_checkpoints"]=imported
        RUN.atomic_json(directory/"manifest.json",manifest)
    except BaseException as exc:
        manifest.update(status="preparation_failed",error_type=type(exc).__name__,message=str(exc),
                        imported_checkpoints=imported)
        RUN.atomic_json(directory/"manifest.json",manifest)
        raise
    return manifest


@contextmanager
def campaign_lock(directory):
    with (Path(directory)/".run_201.lock").open("a+") as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another 201 controller is active for this campaign") from exc
        try:
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


def active_processes(directory,proc_root=Path("/proc")):
    """Reject orphaned workers left by an abruptly killed controller."""
    directory=Path(directory).resolve()
    scripts={Path(__file__).resolve(),HERE/"refine_201_worker.py"}
    active=[]
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():
            continue
        try:
            if entry.stat().st_uid!=os.getuid():
                continue
            args=[v.decode(errors="replace") for v in (entry/"cmdline").read_bytes().split(b"\0") if v]
            if not any(Path(v).name in {p.name for p in scripts} for v in args):
                continue
            cwd=(entry/"cwd").resolve(strict=True)
            index=next((i for i,v in enumerate(args) if (cwd/v).resolve() in scripts),None)
            if index is None:
                continue
            args=args[index+1:]
            def option(name,default):
                for i,v in enumerate(args):
                    if v==name:
                        return args[i+1] if i+1<len(args) else default
                    if v.startswith(name+"="):
                        return v.split("=",1)[1]
                return default
            target=option("--directory",None)
            target=(cwd/target).resolve() if target else (RUN.OUTPUT/option("--name","refinement_201")).resolve()
            if target==directory:
                active.append(int(entry.name))
        except (FileNotFoundError,ProcessLookupError):
            continue
    return active


def complete_case(directory,manifest,model,rho):
    folder=RUN.case_dir(directory,model,rho,201)
    case=RUN.load_checked(folder/"solution")
    if (case is None or case.get("campaign_fingerprint")!=manifest["fingerprint"]
            or case.get("n")!=201 or case.get("rho")!=rho
            or case.get("configuration",{}).get("parameters")!=dict(manifest["configuration"]["parameters"],rho=rho)
            or not RUN.verified(case.get("selected")) or not case.get("demand_audit",{}).get("verified")):
        return False
    policy_hash=sha256(case["selected"]["policy"].tobytes()).hexdigest()
    for mode in manifest["configuration"]["simulation_modes"]:
        simulation=RUN.load_checked(folder/("simulation_"+mode))
        if (simulation is None or simulation.get("campaign_fingerprint")!=manifest["fingerprint"]
                or simulation.get("policy_sha256")!=policy_hash):
            return False
    return True


def export_comparison(directory,manifest):
    rows=RUN.export_campaign(directory)
    from oblivious.oe_p0.code.run_experiment import write_csv
    import numpy as np
    from scipy.stats import wasserstein_distance
    comparison=[]
    for seed in manifest["seed_sources"]:
        case=RUN.load_checked(RUN.case_dir(directory,seed["model"],seed["rho"],201)/"solution")
        if case is None or not RUN.verified(case.get("selected")):
            continue
        previous=RUN.load_checked(resolve_saved_path(seed["base"], project_root=RUN.ROOT))
        row=dict(model=seed["model"],rho=seed["rho"],base_grid=101,refined_grid=201,
                 base_candidates=len(previous.get("candidates",[])),
                 refined_candidates=len(case.get("candidates",[])),branch_tracking_established=False)
        for metric in ("product2_market","seller_profit","mean_reputation"):
            a,b=previous["predicted"][metric],case["predicted"][metric]
            row.update({metric+"_101":a,metric+"_201":b,metric+"_change":b-a})
        row["product2_change_pp"]=100*row["product2_market_change"]
        a,b=previous["selected"],case["selected"]
        row["marginal_wasserstein"]=float(wasserstein_distance(a["grid"],b["grid"],
            u_weights=a["nu"].sum(axis=0),v_weights=b["nu"].sum(axis=0)))
        row["policy_max_interpolated_difference"]=float(np.max(np.abs(
            b["policy"]-np.interp(b["grid"],a["grid"],a["policy"]))))
        comparison.append(row)
    write_csv(Path(directory)/"comparison_101_201.csv",comparison)
    return rows


def last_log_line(path):
    with Path(path).open("rb") as stream:
        stream.seek(0,2)
        stream.seek(max(0,stream.tell()-2048))
        lines=stream.read().decode(errors="replace").strip().splitlines()
    return lines[-1].strip() if lines else "starting numerical calculation"


def execute_201(directory,workers=1):
    directory=Path(directory).resolve()
    if workers<1:
        raise ValueError("workers must be positive")
    with campaign_lock(directory):
        active=active_processes(directory)
        if active:
            raise RuntimeError(f"201 processes are still active (PID {active}); stop their original terminal first")
        manifest=json.loads((directory/"manifest.json").read_text())
        if sources()!=manifest["source_sha256"]:
            raise ValueError("201 code changed; use another --name")
        WORKER.validate_manifest(manifest)
        config=manifest["configuration"]
        if manifest.get("status")=="preparation_failed":
            raise ValueError("Preparation failed; review its error before starting workers")
        children,lock,stop={},threading.Lock(),threading.Event()
        failures=[]
        manifest.update(status="running",updated_utc=RUN.stamp(),workers=workers)
        RUN.atomic_json(directory/"manifest.json",manifest)

        def run_one(model,rho):
            if stop.is_set():
                return None
            if complete_case(directory,manifest,model,rho):
                return 0
            folder=RUN.case_dir(directory,model,rho,201)
            folder.mkdir(parents=True,exist_ok=True)
            log_path=folder/"worker.log"
            with log_path.open("a") as log:
                with lock:
                    if stop.is_set():
                        return None
                    print(f"  Starting {model}, rho={rho:g}; log: {log_path}",flush=True)
                    process=subprocess.Popen([sys.executable,"-u","-B",str(HERE/"refine_201_worker.py"),
                        "--directory",str(directory),"--model",model,"--rho",str(rho),"--n","201"],
                        cwd=RUN.ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    children[process.pid]=process
                started=time.monotonic()
                try:
                    while True:
                        try:
                            return process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            with lock:
                                print(f"  {model}, rho={rho:g}, {(time.monotonic()-started)/60:.1f} min: "
                                      +last_log_line(log_path),flush=True)
                finally:
                    with lock:
                        children.pop(process.pid,None)
        try:
            for model in config["models"]:
                print(f"\n{RUN.LABELS[model]}: n=201, up to {workers} concurrent rho",flush=True)
                executor=ThreadPoolExecutor(max_workers=workers)
                futures={executor.submit(run_one,model,rho):rho for rho in
                    sorted(config["rhos"],key=lambda r:(r not in (.95,.97,.99),r))}
                try:
                    for future in as_completed(futures):
                        rho=futures[future]
                        code=future.result()
                        print(f"  {model}, rho={rho:g}, n=201: "+("complete" if code==0 else f"needs review (exit {code})"),flush=True)
                        if code!=0:
                            failures.append(dict(model=model,rho=rho,exit_code=code))
                        export_comparison(directory,manifest)
                except BaseException:
                    stop.set()
                    executor.shutdown(wait=False,cancel_futures=True)
                    raise
                else:
                    executor.shutdown(wait=True)
            finished=all(complete_case(directory,manifest,model,rho)
                         for model in config["models"] for rho in config["rhos"])
            manifest.update(status="complete" if finished else "needs_review",finished_utc=RUN.stamp(),failures=failures)
            RUN.atomic_json(directory/"manifest.json",manifest)
            print(f"\nSaved results: {directory}\n101→201: {directory/'comparison_101_201.csv'}",flush=True)
            return 0 if finished else 2
        except BaseException as exc:
            stop.set()
            with lock:
                processes=list(children.values())
            FINISH.stop_children(processes)
            manifest.update(status="interrupted" if isinstance(exc,KeyboardInterrupt) else "failed",
                updated_utc=RUN.stamp(),last_error=dict(type=type(exc).__name__,message=str(exc)))
            RUN.atomic_json(directory/"manifest.json",manifest)
            raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",default="refinement_201")
    parser.add_argument("--source-campaign",default="refinement_101_201")
    parser.add_argument("--oe-retry",default="oe099_retry_n101")
    parser.add_argument("--models",nargs="+",choices=RUN.MODELS,default=list(RUN.MODELS))
    parser.add_argument("--rhos",nargs="+",type=float,default=list(RUN.RHOS))
    parser.add_argument("--workers",type=int,default=1)
    parser.add_argument("--solve-only",action="store_true")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--status",action="store_true")
    parser.add_argument("--dry-run",action="store_true")
    tokens=sys.argv[1:] if argv is None else argv
    args=parser.parse_args(tokens)
    for name in (args.name,args.source_campaign,args.oe_retry):
        if Path(name).name!=name or name in ("", ".", ".."):
            parser.error("Campaign and retry names must be folder names under data/")
    if (args.workers<1 or len(set(args.models))!=len(args.models)
            or len(set(args.rhos))!=len(args.rhos) or any(r not in RUN.RHOS for r in args.rhos)):
        parser.error("Require positive workers, distinct models and the original rho values through .99")
    directory=RUN.OUTPUT/args.name
    try:
        if args.resume:
            allowed={"--name","--workers","--resume","--status","--dry-run"}
            if any(t.split("=")[0] not in allowed for t in tokens if t.startswith("--")):
                parser.error("Resume restores saved settings; only --name and --workers may change")
            manifest=json.loads((directory/"manifest.json").read_text())
            planned=manifest["configuration"]
        else:
            planned=dict(models=args.models,rhos=args.rhos)
        if args.status:
            manifest=json.loads((directory/"manifest.json").read_text())
            print(f"Campaign: {manifest['status']}")
            for model in manifest["configuration"]["models"]:
                for rho in manifest["configuration"]["rhos"]:
                    path=RUN.case_dir(directory,model,rho,201)/"status.json"
                    state=json.loads(path.read_text()).get("status") if path.exists() else "pending"
                    print(f"{model:11s} rho={rho:g} n=201: {state}")
            return 0
        if args.dry_run:
            print(f"Output: {directory}\nModels sequentially: {', '.join(planned['models'])}")
            print(f"Grid: 201; rho per model: {planned['rhos']}; up to {args.workers} concurrent rho")
            print("Certified 101 seeds; repaired OE .99; compatible interrupted 201 checkpoints reused.")
            print("No calculations started or files written.")
            return 0
        if not args.resume:
            prepare_campaign(directory,RUN.OUTPUT/args.source_campaign,RUN.OUTPUT/args.oe_retry,
                             models=args.models,rhos=args.rhos,simulate=not args.solve_only)
        return execute_201(directory,args.workers)
    except (ArithmeticError,RuntimeError,ValueError,OSError) as exc:
        parser.error(str(exc))


if __name__=="__main__":
    raise SystemExit(main())
