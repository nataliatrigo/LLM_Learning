"""Add homogeneous Conditional OE outcomes without solving any equilibrium.

Run from the repository root, automatically resuming compatible checkpoints:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/full-comparison-mpl \
    .venv/bin/python -B 'oblivious/07_full_comparison/code/prepare_conditional_outcomes.py'

Only conditional_outcomes.csv, conditional_replications.csv,
conditional_outcomes_manifest.json and conditional_outcomes/ are written.
The original three-model snapshots and Conditional OE sources stay read-only.
"""

import csv
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/full-comparison-mpl")
ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parents[1]/"data"
SOURCE = ROOT/"oblivious/04_oe_conditional_distribution/data/rho_forgetting"
CHECKPOINTS = DATA/"conditional_outcomes"
MANIFEST = DATA/"conditional_outcomes_manifest.json"
sys.path.insert(0,str(ROOT))

import numpy as np
from oblivious.oe_p0.code.game import Game
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import load_bundle, save_bundle, case_tag, write_json, write_csv
from oblivious.equilibrium_comparison.code.analysis import interval
from oblivious.equilibrium_comparison.code.cross_rho import summarize_profile
from oblivious.equilibrium_comparison.code.simulation import simulate_pair

METRICS = ("product2_market", "seller_profit")
EXPECTED_SIMULATION = dict(periods=100000,burn_in=10000,replications=20,seed=20261017,mode="grid")
INITIAL = [.5,.5]
ASSIGNMENTS = {"conditional": ("conditional","conditional")}


def require(condition,message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return sha256(path.read_bytes()).hexdigest()


def verify_hashes(base, hashes):
    for name, expected in hashes.items():
        path = base/name
        require(path.is_file() and digest(path) == expected,f"Input changed or missing: {path}")


def code_hashes():
    paths = [Path(__file__).resolve()]
    paths += [ROOT/"oblivious/06_equilibrium_comparison/code"/name for name in
              ("simulation.py","cross_rho.py","analysis.py")]
    paths += [ROOT/"oblivious/02_oe_p0/code"/name for name in
              ("run_experiment.py","primitives.py","game.py","policy.py","diagnostics.py")]
    return {str(p.relative_to(ROOT)):digest(p) for p in paths}


def metadata_expected(params, simulation):
    return dict(params=asdict(params),**simulation,short_periods=0,initial=INITIAL,
                total_periods=simulation["periods"]+simulation["burn_in"],
                primary_window=[simulation["burn_in"],simulation["burn_in"]+simulation["periods"]],
                assignments={key:list(value) for key,value in ASSIGNMENTS.items()})


def check_summary(result, params, simulation):
    rows = result["replications"]
    require(len(rows) == simulation["replications"],"Incorrect Conditional OE replication count")
    require(sorted(row["replication"] for row in rows) == list(range(len(rows))),"Invalid replication identifiers")
    for row in rows:
        expected = .5*(params.R-params.c1-(params.c2-params.c1)*row["product2_market"])
        require(abs(row["seller_profit"]-expected) < 1e-12,"Simulation profit accounting failed")
        require(0 <= row["product2_market"] <= 1 and 0 <= row["success_rate"] <= 1,
                "Invalid simulation purchase share")
    for metric in (*METRICS,"success_rate"):
        estimated = interval([row[metric] for row in rows])
        require(all(np.isclose(estimated[key],result["summary"][metric][key],rtol=0,atol=1e-12)
                    for key in estimated),f"Summary mismatch: {metric}")


def main():
    comparison_manifest = json.loads((DATA/"manifest.json").read_text())
    policy_manifest = json.loads((DATA/"conditional_policy_manifest.json").read_text())
    source_manifest = json.loads((SOURCE/"manifest.json").read_text())
    config = json.loads((DATA/"comparison.json").read_text())
    require(all(m["status"] == "complete" for m in (comparison_manifest,policy_manifest,source_manifest)),
            "All input manifests must be complete")
    verify_hashes(DATA,comparison_manifest["data_files_sha256"])
    verify_hashes(DATA,policy_manifest["data_files_sha256"])
    verify_hashes(ROOT,policy_manifest["source_files_sha256"])
    comparison_hash = digest(DATA/"manifest.json")
    policy_hash = digest(DATA/"conditional_policy_manifest.json")
    require(policy_manifest["comparison_manifest_sha256"] == comparison_hash,"Policy/base comparison provenance mismatch")
    rhos,n,simulation = config["rhos"],config["grid"],config["simulation"]
    require(n == 51 and simulation == EXPECTED_SIMULATION and config["initial"] == INITIAL,
            "Expected original common n51, 20-path grid comparison design")
    require(policy_manifest["rhos"] == rhos and policy_manifest["grid"] == n
            and policy_manifest["parameters"] == config["parameters"],"Conditional policy settings differ")
    model = policy_manifest["model"]
    require(model["key"] == "conditional","Wrong model in Conditional OE policy manifest")
    with (DATA/"conditional_policies.csv").open(newline="") as stream:
        policy_rows = list(csv.DictReader(stream))
    require(len(policy_rows) == len(rhos)*n,"Conditional policy CSV has incorrect row count")
    codes = code_hashes()
    source_files = dict(policy_manifest["source_files_sha256"])
    source_files.update({str((DATA/name).relative_to(ROOT)):digest(DATA/name) for name in
                        ("comparison.json","stationary_inputs.npz","conditional_policies.csv",
                         "manifest.json","conditional_policy_manifest.json")})
    contract = dict(rhos=rhos,grid=n,parameters=config["parameters"],simulation=simulation,
        initial=INITIAL,model=model,short_periods=0,assignments={key:list(v) for key,v in ASSIGNMENTS.items()},
        comparison_manifest_sha256=comparison_hash,conditional_policy_manifest_sha256=policy_hash,
        source_files_sha256=source_files,code_files_sha256=codes)
    fingerprint = sha256(json.dumps(contract,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    CHECKPOINTS.mkdir(parents=True,exist_ok=True)
    previous = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else None
    if previous is not None:
        require(previous.get("run_fingerprint") == fingerprint,
                "Existing Conditional OE outcome run has incompatible inputs, settings or code; checkpoints were not overwritten")
        verify_hashes(DATA,previous.get("data_files_sha256",{}))
    manifest = dict(**contract,status="incomplete",run_fingerprint=fingerprint,
        started_utc=(previous or {}).get("started_utc",datetime.now(timezone.utc).isoformat()),
        completed_rhos=[],data_files_sha256={},no_equilibrium_solved=True,
        new_simulations="Actual two-seller Beta draws, finite grid stochastic projection; no previous continuous paths reused",
        checkpoint_content="Policy, stationary joint law, demand, predicted outcomes, replication summaries, empirical distributions and full simulation metadata; event-level arrays not retained",
        definitions=dict(product2_market="Product 2 share of all market purchases; one purchase per calendar period",
            seller_profit="Mean profit across sellers A and B per calendar period, including unselected zero-profit periods",
            uncertainty="Pointwise 95% Student-t intervals across 20 independent path means; initialization bias excluded"))
    write_json(MANIFEST,manifest)
    outcomes,replications = [],[]
    with np.load(DATA/"stationary_inputs.npz",allow_pickle=False) as common:
        for rho in rhos:
            started = time.monotonic()
            tag = case_tag(rho,n)
            base = SOURCE/"arrays"/tag
            require(tag in source_manifest["completed_cases"],f"Source case incomplete: {tag}")
            for suffix in (".json",".npz"):
                path = base.with_suffix(suffix)
                require(digest(path) == source_manifest["checkpoint_sha256"][str(path.relative_to(SOURCE))],
                        f"Source checkpoint hash changed: {tag}")
            case = load_bundle(base)
            selected = case["selected"]
            params = Parameters(**case["configuration"]["parameters"])
            require(case["rho"] == rho and case["n"] == n and params.rho == rho,"Source case mismatch")
            require({k:v for k,v in asdict(params).items() if k != "rho"} == config["parameters"],
                    "Conditional OE primitives differ from original comparison")
            require(all(selected[key] for key in ("verified","candidate_verified")) and
                    all(selected[key]["verified"] for key in ("dp_diagnostics","stationary_diagnostics","conditional_diagnostics"))
                    and case["demand_audit"]["verified"],f"Unverified selected Conditional OE candidate: {tag}")
            grid,alpha,nu = selected["grid"],selected["policy"],selected["nu"]
            rows = sorted((row for row in policy_rows if float(row["rho"]) == rho),key=lambda row:float(row["reputation"]))
            require(len(rows) == n and all(int(row["n"]) == n and row["model"] == "conditional" for row in rows),
                    "Policy CSV case mismatch")
            require(np.array_equal(grid,np.array([float(row["reputation"]) for row in rows])) and
                    np.array_equal(alpha,np.array([float(row["product2_probability"]) for row in rows])),
                    "Simulation policy differs from exact displayed Conditional OE policy")
            require(np.array_equal(grid,common[f"{tag}/grid"]),"Conditional OE grid differs from original comparison")
            require(nu.shape == (n,n) and np.isfinite(nu).all() and np.all(nu >= 0) and abs(nu.sum()-1)<1e-12,
                    "Invalid saved Conditional OE joint law")
            game = Game(grid,params,order=case["configuration"]["order"])
            demand = common[f"{tag}/demand_A"]
            demand_error = float(np.max(abs(game.D-demand)))
            require(demand_error < 1e-13,"Conditional OE actual demand differs from common comparison kernel")
            sigma = np.broadcast_to(alpha[:,None],(n,n))
            transition = game.transition(sigma,sigma.T)
            stationary_residual = float(np.max(abs(nu.ravel()@transition-nu.ravel())))
            require(stationary_residual < 1e-10,"Conditional OE joint law fails actual-game stationarity")
            usage = float(np.sum(nu*(demand*sigma+demand.T*sigma.T)))
            profit = float(np.sum(game.rewards(sigma,sigma.T)*nu[None,:,:])/2)
            require(abs(profit-.5*(params.R-params.c1-(params.c2-params.c1)*usage)) < 1e-12,
                    "Stationary profit accounting failed")
            predicted = dict(product2_market=usage,seller_profit=profit,
                             success_rate=params.p1+(params.p2-params.p1)*usage)
            checkpoint = CHECKPOINTS/tag
            if checkpoint.with_suffix(".json").exists() or checkpoint.with_suffix(".npz").exists():
                require(checkpoint.with_suffix(".json").exists() and checkpoint.with_suffix(".npz").exists(),
                        "Partial checkpoint; interrupted write must be removed before resuming")
                result = load_bundle(checkpoint)
                require(result.get("run_fingerprint") == fingerprint and result.get("verified") is True,
                        f"Incompatible/unverified saved outcome checkpoint: {tag}")
                require(np.array_equal(result["policy"],alpha) and np.array_equal(result["stationary_joint"],nu)
                        and np.array_equal(result["demand_A"],demand),"Cached outcome uses different inputs")
                require(all(abs(result["predicted"][k]-v)<1e-12 for k,v in predicted.items()),
                        "Cached stationary outcomes differ")
                print(f"rho={rho:g}: reusing compatible Conditional OE grid simulations",flush=True)
            else:
                pair = dict(params=params,grid=grid,profiles={"conditional":dict(grid=grid,policy=alpha)})
                print(f"rho={rho:g}: simulating 20 Conditional OE grid paths",flush=True)
                simulated = simulate_pair(pair,**simulation,short_periods=0,initial=INITIAL,
                    assignments=ASSIGNMENTS,progress=lambda done,total:
                        print(f"  {done:,}/{total:,} periods",flush=True))
                result = summarize_profile(grid,simulated["profiles"]["conditional"],
                    start=simulation["burn_in"],length=simulation["periods"])
                result.update(rho=rho,n=n,verified=True,run_fingerprint=fingerprint,
                    parameters=asdict(params),grid=grid.copy(),policy=alpha.copy(),stationary_joint=nu.copy(),
                    demand_A=demand.copy(),predicted=predicted,
                    simulation_provenance=dict(kind="existing_simulate_pair",metadata=simulated["metadata"],
                                               replication_seeds=simulated["replication_seeds"]),
                    audit=dict(source_candidate_verified=True,displayed_policy_exact_match=True,
                        stationary_residual_max=stationary_residual,common_demand_max_difference=demand_error,
                        source_case_json_sha256=digest(base.with_suffix(".json")),
                        source_case_npz_sha256=digest(base.with_suffix(".npz")),
                        saved_dp_diagnostics=selected["dp_diagnostics"],
                        saved_conditional_diagnostics=selected["conditional_diagnostics"]),
                    elapsed_seconds=time.monotonic()-started)
                del simulated
                check_summary(result,params,simulation)
                save_bundle(checkpoint,result)
            expected_metadata = metadata_expected(params,simulation)
            require(all(result["simulation_provenance"]["metadata"].get(k) == v for k,v in expected_metadata.items()),
                    "Conditional outcome simulation metadata mismatch")
            check_summary(result,params,simulation)
            for metric in METRICS:
                outcomes.append(dict(rho=rho,n=n,model="conditional",label=model["label"],metric=metric,
                    stationary=predicted[metric],**{f"simulation_{k}":v for k,v in result["summary"][metric].items()}))
            replications.extend(dict(rho=rho,model="conditional",replication=row["replication"],
                **{metric:row[metric] for metric in METRICS}) for row in result["replications"])
            manifest["completed_rhos"].append(rho)
            for suffix in (".json",".npz"):
                path=checkpoint.with_suffix(suffix)
                manifest["data_files_sha256"][str(path.relative_to(DATA))]=digest(path)
            write_json(MANIFEST,manifest)
            print(f"  verified: usage MC={result['summary']['product2_market']['mean']:.6f}, "
                  f"stationary={usage:.6f}; profit MC={result['summary']['seller_profit']['mean']:.6f}",flush=True)
    require(len(outcomes)==16 and len(replications)==160,"Unexpected final outcome/replication row count")
    write_csv(DATA/"conditional_outcomes.csv",outcomes)
    write_csv(DATA/"conditional_replications.csv",replications)
    for name in ("conditional_outcomes.csv","conditional_replications.csv"):
        manifest["data_files_sha256"][name]=digest(DATA/name)
    require(code_hashes()==codes,"Numerical preparation code changed during run")
    verify_hashes(ROOT,source_files)
    manifest.update(status="complete",finished_utc=datetime.now(timezone.utc).isoformat(),
                    all_eight_cases_verified=True,outcome_rows=len(outcomes),replication_rows=len(replications))
    write_json(MANIFEST,manifest)
    print("Completed homogeneous Conditional OE outcomes for all eight rho values",flush=True)


if __name__ == "__main__":
    main()
