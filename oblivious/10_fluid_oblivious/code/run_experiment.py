"""Solve, refine, compare, and plot the separate fluid point-conjecture study.

Run from the repository root:
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -B \
    -m oblivious.fluid_oblivious.code.run_experiment
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import gzip
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluid-oblivious-mpl")

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

PROJECT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT))
HOME = Path(__file__).resolve().parents[1]

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.fluid_oblivious.code.model import find_equilibria, solve_response, audit_response
from oblivious.fluid_oblivious.code.trajectories import simulate_pair
from oblivious.fluid_oblivious.code.results import write_results


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def fingerprint(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def case_tag(p0, rho):
    return f"p0_{p0:g}_rho_{rho:g}".replace(".", "p")


def numerical_sources():
    files = [HOME / "code/model.py", HOME / "code/trajectories.py",
             PROJECT / "forgetting/Code/fluid_convergence/numerics.py",
             PROJECT / "forgetting/Code/model.py",
             PROJECT / "oblivious/02_oe_p0/code/primitives.py",
             PROJECT / "oblivious/08_oe_distribution_fixed_p0/code/model.py"]
    return {str(p.relative_to(PROJECT)): fingerprint(p) for p in files}


def solve_case(config, p0, rho, output, source_hashes, recompute=False):
    """A self-contained, resumable task; no shared mutable numerical objects."""
    start = time.monotonic()
    requested = dict(parameters=config["parameters"], p0=p0, rho=rho,
                     grids=config["grids"], order=config["quadrature_order"],
                     search_points=config["search_points"],
                     search_refinement_points=config.get("search_refinement_points",config["search_points"]),
                     sources=source_hashes)
    cache = Path(output) / "checkpoints" / (case_tag(p0, rho) + ".json.gz")
    if cache.exists() and not recompute:
        with gzip.open(cache, "rt") as handle:
            saved = json.load(handle)
        if saved.get("inputs") == requested and saved.get("completed"):
            saved["cached"] = True
            return saved
    params = Parameters(**config["parameters"], rho=rho)
    grids = []
    for nodes in config["grids"]:
        try:
            result = find_equilibria(params, p0, n=nodes,
                                    order=config["quadrature_order"],
                                    scan_points=config.get("search_refinement_points",config["search_points"])
                                    if nodes == config["grids"][-1] else config["search_points"])
            if nodes == config["grids"][-1]:
                result["demand_audits"] = []
                if result["equilibria"]:
                    for eq in result["equilibria"]:
                        audit = audit_response(params,p0,eq["response"],order=config["quadrature_order"])
                        eq["demand_audit"] = audit
                        eq["diagnostics"]["demand_audit_passed"] = bool(audit["passed"])
                        eq["verified"] = bool(eq["verified"] and audit["passed"])
                        result["demand_audits"].append(dict(conjecture=eq["x_star"],audit=audit))
                else:
                    # Before reporting no candidate, audit the demand at the
                    # rejected pure and mixed conjectures used by that search.
                    conjectures = sorted(set([params.p1,params.p2] +
                        [item["x_star"] for item in result["search"]["rejected_candidates"]]))
                    for conjecture in conjectures:
                        response = solve_response(params,p0,conjecture,n=nodes,order=config["quadrature_order"])
                        audit = audit_response(params,p0,response,order=config["quadrature_order"])
                        result["demand_audits"].append(dict(conjecture=conjecture,audit=audit))
                if not all(item["audit"]["passed"] for item in result["demand_audits"]):
                    raise ArithmeticError("Final demand quadrature/adaptive audit failed")
            grids.append(dict(n=nodes, result=result, error=None))
        except Exception as exc:
            grids.append(dict(n=nodes, result=None, error=f"{type(exc).__name__}: {exc}",
                              traceback=traceback.format_exc()))
    saved = dict(inputs=requested, p0=p0, rho=rho, grids=grids,
                 completed=all(g["error"] is None for g in grids), cached=False,
                 elapsed_seconds=time.monotonic() - start)
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp.gz")
    with gzip.open(temporary, "wt") as handle:
        json.dump(clean(saved), handle, separators=(",", ":"), allow_nan=False)
    temporary.replace(cache)
    return clean(saved)


def match_equilibria(fine, coarse):
    if not fine or not coarse:
        return {}
    distances = np.array([[abs(a["x_star"] - b["x_star"])
                           for b in coarse] for a in fine])
    rows, columns = linear_sum_assignment(distances)
    return dict(zip(rows.tolist(), columns.tolist()))


def refinement_checks(case, limits):
    records = []
    for coarse, fine in zip(case["grids"], case["grids"][1:]):
        if coarse["result"] is None or fine["result"] is None:
            records.append(dict(p0=case["p0"], rho=case["rho"], coarse_n=coarse["n"],
                                fine_n=fine["n"], equilibrium_id=-1, passed=False,
                                reason="solver_error"))
            continue
        ce, fe = coarse["result"]["equilibria"], fine["result"]["equilibria"]
        matching = match_equilibria(fe, ce)
        if not fe:
            records.append(dict(p0=case["p0"], rho=case["rho"], coarse_n=coarse["n"],
                                fine_n=fine["n"], equilibrium_id=-1,
                                coarse_count=len(ce), fine_count=0,
                                passed=len(ce) == 0, reason="no_candidate_on_fine_grid"))
        for i, eq in enumerate(fe):
            row = dict(p0=case["p0"], rho=case["rho"], coarse_n=coarse["n"],
                       fine_n=fine["n"], equilibrium_id=i,
                       coarse_count=len(ce), fine_count=len(fe), passed=False)
            if i not in matching:
                row["reason"] = "unmatched_candidate"
                records.append(row)
                continue
            previous = ce[matching[i]]
            response, old_response = eq["response"], previous["response"]
            common = np.linspace(0, 1, 8001)
            fine_actions = np.interp(common, response["x"], response["gap"]) > 0
            coarse_actions = np.interp(common, old_response["x"], old_response["gap"]) > 0
            disagreement = float(np.trapezoid((fine_actions != coarse_actions).astype(float), common))
            # Sale probability is stored by the solver when available; otherwise
            # evaluate its response demand at the consistent own reputation.
            share = float(np.interp(eq["x_star"], response["x"], response["demand"]))
            old_share = float(np.interp(previous["x_star"], old_response["x"], old_response["demand"]))
            row.update(reputation_change=abs(eq["x_star"] - previous["x_star"]),
                       alpha_change=abs(eq["alpha"] - previous["alpha"]),
                       market_share_change=abs(share - old_share),
                       policy_disagreement_measure=disagreement)
            row["passed"] = bool(
                len(ce) == len(fe) and eq["verified"] and previous["verified"]
                and row["reputation_change"] <= limits["equilibrium_reputation_absolute"]
                and row["alpha_change"] <= limits["equilibrium_fraction_absolute"]
                and row["market_share_change"] <= limits["market_share_absolute"]
                and disagreement <= limits["policy_disagreement_measure"]
            )
            row["reason"] = "within_configured_sensitivity" if row["passed"] else "sensitivity_exceeded"
            records.append(row)
    return records


def tables(cases, config):
    summary, policy_rows, refinements, search_rows = [], [], [], []
    for case in cases:
        comparisons = refinement_checks(case, config["refinement"])
        refinements.extend(comparisons)
        for grid in case["grids"]:
            search_rows.append(dict(p0=case["p0"], rho=case["rho"], n=grid["n"],
                                    solver_error=grid["error"],
                                    search=None if grid["result"] is None else grid["result"]["search"]))
        last = case["grids"][-1]
        base = dict(p0=case["p0"], rho=case["rho"], n=last["n"],
                    model="fluid", seller_count=2)
        if last["result"] is None or not last["result"]["equilibria"]:
            summary.append(dict(base, equilibrium_id=-1, verified=False,
                                status="solver_error" if last["result"] is None else "no_equilibrium_found",
                                equilibrium_count=0))
            continue
        equilibria = last["result"]["equilibria"]
        for i, eq in enumerate(equilibria):
            matched = [r for r in comparisons if r["fine_n"] == last["n"] and r["equilibrium_id"] == i]
            refined = len(matched) == 1 and bool(matched[0]["passed"])
            valid = bool(eq["verified"] and refined)
            response = eq["response"]
            z, alpha = float(eq["x_star"]), float(eq["alpha"])
            # At symmetry, the exact TS identity avoids cancellation in tiny
            # purchase probabilities and decouples accounting from grid error.
            from scipy.special import betainc, betaincc
            a, b = 1 + z/(1-case["rho"]), 1 + (1-z)/(1-case["rho"])
            cdf, survival = float(betainc(a,b,case["p0"])), float(betaincc(a,b,case["p0"]))
            share = survival * (1+cdf) / 2
            outside = cdf*cdf
            reward = config["parameters"]["R"] - config["parameters"]["c1"] - (config["parameters"]["c2"]-config["parameters"]["c1"])*alpha
            row = dict(base, equilibrium_id=i, equilibrium_count=len(equilibria),
                       status="verified" if valid else "refinement_failed",
                       verified=valid, core_verified=bool(eq["verified"]), refined=refined,
                       x_star=z, alpha=alpha, equilibrium_kind=eq["kind"],
                       product2_conditional=alpha, product2_calendar=share*alpha,
                       customer_product2_calendar=2*share*alpha, sale_rate=share,
                       outside_rate=outside, profit_calendar=share*reward,
                       value_stationary=share*reward/(-math.log(config["parameters"]["gamma"])),
                       perceived_value_at_equilibrium=float(np.interp(z,response["x"],response["value"])),
                       hjb_residual=response["residual"])
            row.update({"diagnostic_"+key: value for key,value in eq.get("diagnostics",{}).items()
                        if value is None or isinstance(value,(int,float,bool,str))})
            summary.append(row)
            for x, action in zip(response["x"], response["policy"]):
                policy_rows.append(dict(base, equilibrium_id=i, verified=valid, reputation=x,
                                        product2_probability=action, holding=False))
            for point in eq["policy"].get("holding_points", []):
                policy_rows.append(dict(base,equilibrium_id=i,verified=valid,reputation=point["x"],
                                        product2_probability=point["alpha"],holding=True))
    return pd.DataFrame(summary), pd.DataFrame(policy_rows), pd.DataFrame(refinements), search_rows


def trajectories(cases, summary, config, output):
    frames, bins, diagnostics = [], [], []
    selected = summary[summary.verified]
    settings = config["trajectories"]
    times = np.linspace(0, settings["horizon"], settings["samples"])
    for row in selected.itertuples():
        case = next(c for c in cases if c["p0"] == row.p0 and c["rho"] == row.rho)
        eq = case["grids"][-1]["result"]["equilibria"][row.equilibrium_id]
        params = Parameters(**config["parameters"], rho=row.rho)
        result = simulate_pair(params,row.p0,eq["policy"],times=times,
                               initial=tuple(settings["initial"]),bin_width=settings["bin_width"],
                               rtol=settings["rtol"],atol=settings["atol"],max_step=settings["max_step"],
                               demand_order=config["quadrature_order"])
        identity = dict(p0=row.p0,rho=row.rho,equilibrium_id=row.equilibrium_id)
        for key,destination in (("by_time",frames),("binned",bins)):
            frame = result[key].copy()
            for column,value in identity.items():
                frame[column] = value
            destination.append(frame)
        diagnostics.append(dict(identity,diagnostics=result["diagnostics"],events=result["events"]))
        index = selected.index[(selected.p0 == row.p0)&(selected.rho == row.rho)&(selected.equilibrium_id == row.equilibrium_id)][0]
        if "value_initial_a" in result["diagnostics"]:
            summary.loc[index,"value_initial"] = result["diagnostics"]["value_initial_a"]
        if len(frames[-1]):
            last = frames[-1].iloc[-1]
            summary.loc[index,"trajectory_terminal_x"] = last.x_a
            summary.loc[index,"trajectory_distance_to_equilibrium"] = abs(last.x_a-row.x_star)
    by_time = pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    binned = pd.concat(bins,ignore_index=True) if bins else pd.DataFrame()
    by_time.to_csv(output/"trajectories.csv",index=False)
    binned.to_csv(output/"trajectory_bins.csv",index=False)
    write_json(output/"trajectory_diagnostics.json",diagnostics)
    return by_time,binned


def run(config_path, output, *, workers=None, recompute=False, figures_only=False):
    config = json.loads(Path(config_path).read_text())
    output = Path(output).resolve()
    data, figures = output/"data", output/"figures"
    data.mkdir(parents=True,exist_ok=True)
    figures.mkdir(parents=True,exist_ok=True)
    from oblivious.fluid_oblivious.code.reporting import make_figures
    from oblivious.fluid_oblivious.code.monopoly import build_monopoly_benchmark
    if figures_only:
        recorded = json.loads((data/"manifest.json").read_text())
        if recorded["configuration"] != config:
            raise ValueError("Plot configuration differs from the saved numerical campaign; rerun the numerical study first")
        summary = pd.read_csv(data/"fluid_summary.csv",float_precision="round_trip")
        policies = pd.read_csv(data/"fluid_policies.csv",float_precision="round_trip")
        paths = pd.read_csv(data/"trajectories.csv",float_precision="round_trip")
        bins = pd.read_csv(data/"trajectory_bins.csv",float_precision="round_trip")
        monopoly_summary = pd.read_csv(data/"monopoly_summary.csv",float_precision="round_trip")
        monopoly_policies = pd.read_csv(data/"monopoly_policies.csv",float_precision="round_trip")
        monopoly_paths = pd.read_csv(data/"monopoly_trajectories.csv",float_precision="round_trip")
        monopoly_bins = pd.read_csv(data/"monopoly_trajectory_bins.csv",float_precision="round_trip")
    else:
        hashes = numerical_sources()
        cases = []
        tasks = [(p0,rho) for p0 in config["p0s"] for rho in config["rhos"]]
        count = config.get("workers",2) if workers is None else workers
        if count == 1:
            for p0,rho in tasks:
                case = solve_case(config,p0,rho,data,hashes,recompute)
                cases.append(case)
                print(f"{len(cases)}/{len(tasks)} p0={p0:g} rho={rho:g}: "
                      f"{['error' if g['result'] is None else len(g['result']['equilibria']) for g in case['grids']]} candidates",flush=True)
        else:
            with ProcessPoolExecutor(max_workers=count) as executor:
                futures = [executor.submit(solve_case,config,p0,rho,data,hashes,recompute) for p0,rho in tasks]
                for future in as_completed(futures):
                    case = future.result()
                    cases.append(case)
                    print(f"{len(cases)}/{len(tasks)} p0={case['p0']:g} rho={case['rho']:g}: "
                          f"{['error' if g['result'] is None else len(g['result']['equilibria']) for g in case['grids']]} candidates",flush=True)
        cases.sort(key=lambda c:(c["p0"],c["rho"]))
        summary,policies,refinements,search = tables(cases,config)
        refinements.to_csv(data/"grid_refinement.csv",index=False)
        write_json(data/"equilibrium_search.json",search)
        paths,bins = trajectories(cases,summary,config,data)
        summary.to_csv(data/"fluid_summary.csv",index=False)
        policies.to_csv(data/"fluid_policies.csv",index=False)
        monopoly_summary,monopoly_policies,monopoly_paths,monopoly_bins,monopoly_diagnostics = build_monopoly_benchmark(config)
        monopoly_summary.to_csv(data/"monopoly_summary.csv",index=False)
        monopoly_policies.to_csv(data/"monopoly_policies.csv",index=False)
        monopoly_paths.to_csv(data/"monopoly_trajectories.csv",index=False)
        monopoly_bins.to_csv(data/"monopoly_trajectory_bins.csv",index=False)
        write_json(data/"monopoly_diagnostics.json",monopoly_diagnostics)
        write_json(data/"manifest.json",dict(
            created_utc=datetime.now(timezone.utc).isoformat(),configuration=config,
            configuration_file=str(config_path),numerical_sources=hashes,
            experiment_source_sha256=fingerprint(__file__),
            case_count=len(tasks),candidate_rows=len(summary),
            verified_equilibria=int(summary.verified.sum()),
            monopoly_verified_stationary_states=int(monopoly_summary.verified.sum()),
            comparison_models=config["comparison_models"],
            monopoly_source_sha256=fingerprint(HOME/"code/monopoly.py"),
            status_counts=summary.status.value_counts().to_dict(),
            method="Fixed-rival point conjecture, fluid HJB, pure/mixed stationary search; not stochastic distribution OE.",
            no_candidate_scope="No equilibrium found by the declared numerical search; not a nonexistence proof.",
            cases=[dict(p0=c["p0"],rho=c["rho"],completed=c["completed"],cached=c["cached"],
                        elapsed_seconds=c["elapsed_seconds"]) for c in cases],
        ))
    competition_summary = summary.assign(model="fluid_competition")
    combined_summary = pd.concat([monopoly_summary,competition_summary],ignore_index=True)
    combined_policies = pd.concat([monopoly_policies,policies.assign(model="fluid_competition")],ignore_index=True)
    combined_paths = pd.concat([monopoly_paths,paths.assign(model="fluid_competition")],ignore_index=True)
    combined_bins = pd.concat([monopoly_bins,bins.assign(model="fluid_competition")],ignore_index=True)
    generated = make_figures(combined_summary,combined_policies,figures,p0s=config["p0s"],rhos=config["rhos"],
                             trajectories=combined_paths,trajectory_bins=combined_bins)
    write_results(combined_summary,data,output,config)
    print(f"Verified equilibria: {int(summary.verified.sum())}; status: {summary.status.value_counts().to_dict()}",flush=True)
    print(f"Outputs: {output}",flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=HOME/"config.json")
    parser.add_argument("--output-dir",type=Path,default=HOME)
    parser.add_argument("--workers",type=int)
    parser.add_argument("--recompute",action="store_true")
    parser.add_argument("--figures-only",action="store_true")
    args = parser.parse_args()
    run(args.config,args.output_dir,workers=args.workers,recompute=args.recompute,figures_only=args.figures_only)


if __name__ == "__main__":
    main()
