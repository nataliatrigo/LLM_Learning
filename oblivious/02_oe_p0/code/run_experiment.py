"""Run the endogenous scalar-benchmark experiment.

Examples (from the project root):
    .venv/bin/python -B -m oblivious.oe_p0.code.run_experiment
    .venv/bin/python -B -m oblivious.oe_p0.code.run_experiment --rhos .95 --grids 21
    .venv/bin/python -B -m oblivious.oe_p0.code.run_experiment --resume
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from oblivious.layout import resolve_saved_path
from oblivious.oe_p0.code.core import ScalarExperiment
from oblivious.oe_p0.code.diagnostics import (
    simulate_ts, demand_diagnostics, deviation_diagnostics, compare_existing,
)
from oblivious.oe_p0.code.primitives import Parameters

RHOS = [.6, .7, .8, .9, .95, .97, .99]


def now():
    return datetime.now(timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def write_csv(path, rows):
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(clean(v)) if isinstance(v, (dict, list, tuple))
                             else clean(v) for k, v in row.items()})


def source_hashes():
    files = list((ROOT / "oblivious/02_oe_p0/code").glob("*.py"))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(files)}


def save_bundle(base, bundle):
    """Store nested metadata separately from numeric arrays, without pickle."""
    arrays = {}

    def pack(value, path="root"):
        if isinstance(value, np.ndarray):
            arrays[path] = value
            return {"__array__": path}
        if isinstance(value, dict):
            return {str(k): pack(v, path + "/" + str(k)) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(v, path + "/" + str(i)) for i, v in enumerate(value)]
        return clean(value)

    meta = pack(bundle)
    np.savez_compressed(base.with_suffix(".npz"), **arrays)
    write_json(base.with_suffix(".json"), meta)


def load_bundle(base):
    meta = json.loads(base.with_suffix(".json").read_text())
    with np.load(base.with_suffix(".npz"), allow_pickle=False) as stored:
        def unpack(value):
            if isinstance(value, dict):
                if set(value) == {"__array__"}:
                    return stored[value["__array__"]].copy()
                return {k: unpack(v) for k, v in value.items()}
            if isinstance(value, list):
                return [unpack(v) for v in value]
            return value
        return unpack(meta)


def case_tag(rho, n):
    return "rho_" + str(rho).replace(".", "p") + f"_n{n:03d}"


def evaluation_row(evaluation):
    row = {}
    for key, value in evaluation.items():
        if isinstance(value, (str, int, float, bool, np.generic)) or value is None:
            row[key] = clean(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (str, int, float, bool, np.generic)) or v is None:
                    row[key + "_" + k] = clean(v)
    return row


def mixed_count(evaluation):
    if "mixed_nodes" in evaluation:
        return int(evaluation["mixed_nodes"])
    alpha=np.asarray(evaluation.get("policy",[]))
    return int(np.sum((alpha>1e-8)&(alpha<1-1e-8)))


def select_candidate(candidates):
    # Show an interior candidate when one exists. Every certified candidate,
    # including pure endpoints, is still saved and simulated on the chosen grid.
    interior=[e for e in candidates if mixed_count(e)>0]
    return min(interior or candidates,key=lambda e:e["p0"])


def load_search_cache(folder,tag):
    if folder is None:
        return None,None
    folder=Path(folder)
    manifest_path=folder/"manifest.json"
    recorded=json.loads((folder/"checkpoint_sha256.json").read_text())
    originals=json.loads(manifest_path.read_text())
    hashes={}
    for suffix in (".json",".npz"):
        path=folder/(tag+suffix)
        actual=hashlib.sha256(path.read_bytes()).hexdigest()
        if recorded.get(path.name)!=actual:
            raise ValueError(f"Search checkpoint hash mismatch: {path.name}")
        hashes[path.name]=actual
    # Imported numerical primitives must be the same. Core independently
    # verifies parameters and re-certifies every proposed mixed/pure result.
    numerical_sources = {ROOT / "oblivious/02_oe_p0/code" / name for name in
                         ("game.py", "primitives.py", "policy.py")}
    for name,expected in originals["source_sha256"].items():
        source = resolve_saved_path(name, project_root=ROOT).resolve()
        if not source.is_relative_to(ROOT) or not source.is_file():
            raise ValueError("Search cache must reference existing active project sources")
        if source in numerical_sources:
            if hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
                raise ValueError(f"Search cache used different numerical primitives: {name}")
    return load_bundle(folder/tag),dict(checkpoint_sha256=hashes,
        source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        source_code_sha256=originals["source_sha256"],
        interpretation="Search proposals only; every final candidate is independently reverified with mixed-strategy conditions")


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="rho_forgetting_mixed")
    parser.add_argument("--output", type=Path, default=ROOT / "oblivious/02_oe_p0/data")
    parser.add_argument("--figures-output", type=Path, default=ROOT / "oblivious/02_oe_p0/figures")
    parser.add_argument("--search-cache", type=Path, default=None,
                        help="Optional checked search proposals; final candidates are reverified")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--rhos", nargs="+", type=float, default=RHOS)
    parser.add_argument("--grids", nargs="+", type=int, default=[21, 31, 51])
    parser.add_argument("--p0-points", type=int, default=41)
    parser.add_argument("--order", type=int, default=128)
    parser.add_argument("--consistency-tol", type=float, default=1e-7)
    parser.add_argument("--periods", type=int, default=100000)
    parser.add_argument("--burn-in", type=int, default=10000)
    parser.add_argument("--trajectories", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261017)
    for key, val in asdict(Parameters()).items():
        if key != "rho":
            parser.add_argument("--" + key, type=float, default=val)
    args = parser.parse_args()
    if not args.name or Path(args.name).name != args.name:
        parser.error("Use a single directory name for --name")
    if args.resume:
        allowed = {"--resume", "--name", "--output"}
        if any(a.split("=")[0] not in allowed for a in sys.argv[1:] if a.startswith("--")):
            parser.error("Resume uses saved parameters; only --resume, --name, --output are allowed")
    elif (len(set(args.rhos)) != len(args.rhos) or len(set(args.grids)) != len(args.grids)
          or min(args.grids) < 3 or args.p0_points < 3 or args.order < 8
          or args.periods < 2 or args.trajectories < 2 or args.burn_in < 0
          or not 0 < args.consistency_tol < .01):
        parser.error("Invalid or repeated grids/rhos, or invalid numerical/simulation settings")
    return args


def panel_grid(rhos, scale=(5.5, 3.7)):
    cols = min(2, len(rhos))
    rows = (len(rhos) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols*scale[0], rows*scale[1]),
                             squeeze=False, layout="constrained")
    for ax in axes.flat[len(rhos):]:
        ax.set_visible(False)
    return fig, list(axes.flat)


def save_figure(fig, directory, name):
    for suffix in ("png", "pdf"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)


def selected_case(cases,rho):
    group=[c for c in cases if c["rho"]==rho]
    verified=[c for c in group if c["candidates"]]
    return max(verified or group,key=lambda c:c["n"])


def product_choice_figures(chosen, directory):
    """Draw separate policy and advantage figures from saved selected results."""
    rhos = sorted(chosen)
    advantages = [np.asarray(case["selected"]["advantage"]) for case in chosen.values()
                  if case["selected"] is not None]
    lower = min([0.] + [float(a.min()) for a in advantages])
    upper = max([0.] + [float(a.max()) for a in advantages])
    padding = .05 * max(upper - lower, 1.)
    with plt.rc_context({
        "font.family": "DejaVu Sans", "mathtext.fontset": "dejavusans",
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
        "legend.fontsize": 8, "axes.linewidth": .8,
        "axes.spines.top": True, "axes.spines.right": True,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    }):
        for kind, name, title in (
            ("policy", "02_policy", "Product choice: OE p0"),
            ("advantage", "03_advantage", "Product 2 advantage: OE p0"),
        ):
            fig, axes = panel_grid(rhos)
            for rho, ax in zip(rhos, axes):
                case = chosen[rho]
                e = case["selected"]
                ax.set(xlabel="Own reputation", xlim=(-.04, 1.04),
                       xticks=np.linspace(0, 1, 6))
                if kind == "policy":
                    ax.set(ylabel="Probability of product 2", ylim=(-.04, 1.04),
                           yticks=np.linspace(0, 1, 6))
                else:
                    ax.set(ylabel="Product 2 advantage\nwhen selected",
                           ylim=(lower - padding, upper + padding))
                    ax.axhline(0, color="black", ls="--", lw=.8, label="Indifference")
                ax.grid(alpha=.25)
                if e is None:
                    ax.set_title(f"ρ={rho:g}, n={case['n']} | UNAVAILABLE")
                    ax.text(.5, .5, "No valid stationary evaluation", ha="center",
                            transform=ax.transAxes)
                    continue
                verified = bool(case["candidates"] and e.get("verified")
                                and e.get("candidate_verified"))
                status = "verified on grid" if verified else "UNVERIFIED diagnostic"
                ax.set_title(f"ρ={rho:g}, n={case['n']}, p₀={e['p0']:.6f}\n{status}")
                grid = np.asarray(e["grid"])
                policy = np.asarray(e["policy"])
                values = policy if kind == "policy" else np.asarray(e["advantage"])
                label = "OE p0" if kind == "policy" else "Product 2 − product 1"
                if not verified:
                    label += " (unverified)"
                ax.plot(grid, values, color="#D55E00", marker=".", ms=3, lw=1.3,
                        ls="-" if verified else ":", label=label, zorder=3)
                mixed = (policy > 1e-9) & (policy < 1 - 1e-9)
                if mixed.any():
                    ax.scatter(grid[mixed], values[mixed], marker="D", color="#D55E00",
                               s=24, label="Mixing at a grid point", zorder=4)
                ax.legend(loc="best")
            fig.suptitle(title, fontsize=13)
            save_figure(fig, directory, name)


def make_figures(cases, diagnostics, directory):
    rhos = sorted({c["rho"] for c in cases})
    chosen = {rho: selected_case(cases,rho) for rho in rhos}
    title = "Generated mean TS draw"
    fig, axes = panel_grid(rhos)
    for rho, ax in zip(rhos, axes):
        group = sorted((c for c in cases if c["rho"] == rho), key=lambda c:c["n"])
        for case in group:
            valid = [r for r in case["map_rows"] if r.get("F") is not None
                     and str(r.get("strategy_mode","pure")).startswith("pure")]
            ax.plot([r["p0"] for r in valid], [r["F"] for r in valid], ".-",
                    lw=.8, ms=2, label=f"Pure selector, n={case['n']}",
                    color={21:"#0072B2",31:"#D55E00",51:"#009E73"}.get(case["n"]))
            for branch in case.get("mixing_branches",[]):
                trials=[t for t in branch.get("trials",[]) if t.get("F") is not None]
                if trials and min(t["G"] for t in trials)<=0<=max(t["G"] for t in trials):
                    ax.vlines(branch["p0"],min(t["F"] for t in trials),max(t["F"] for t in trials),
                              color="#666666",alpha=.5,lw=2)
            for root in case["candidates"]:
                interior=mixed_count(root)>0
                ax.plot(root["p0"], root["F"], "D" if interior else "o",
                        mfc="#F0E442" if interior else "none",mec="black",ms=7)
        lo, hi = (1-rho)/(3-2*rho), (2-rho)/(3-2*rho)
        ax.plot([lo,hi], [lo,hi], "k--", lw=1, label="F(p₀)=p₀")
        ax.set(title=f"ρ={rho:g}", xlabel="Assumed p₀", ylabel=title)
        ax.legend(fontsize=8, ncols=2)
        ax.grid(alpha=.18)
    fig.suptitle("OE p0: " + title + " | verified: diamonds=mixed, circles=pure endpoints\nAt action ties, generated means depend on the mixing probabilities", fontsize=12)
    save_figure(fig, directory, "01_fixed_point_map")

    product_choice_figures(chosen, directory)

    fig, axes = panel_grid(rhos)
    for rho, ax in zip(rhos, axes):
        e=chosen[rho]["selected"]
        if e is not None:
            im=ax.pcolormesh(e["grid"],e["grid"],e["nu"].T,shading="auto",cmap="viridis")
            fig.colorbar(im,ax=ax,label="Probability mass")
        ax.set(title=f"ρ={rho:g}, n={chosen[rho]['n']}",xlabel="Reputation A",ylabel="Reputation B")
    fig.suptitle("OE p0: actual TS game, selected symmetric joint invariant distribution",fontsize=12)
    save_figure(fig,directory,"04_joint_stationary")

    # Diagnostic figures are filled from saved structured results.
    diagnostic_figures(chosen, diagnostics, directory)


def diagnostic_figures(chosen, diagnostics, directory):
    rhos=sorted(chosen)
    reports={}
    for rho,case in chosen.items():
        matches=[report for (r,n,index),report in diagnostics.items()
                 if r==rho and n==case["n"]]
        if matches:
            reports[rho]=min(matches,key=lambda r:abs(r["p0"]-case["selected"]["p0"]))

    fig,axes=panel_grid(rhos)
    for rho,ax in zip(rhos,axes):
        if rho not in reports:
            continue
        report=reports[rho];a=report["demand"]["arrays"]
        for key,label,style,color in [
                ("d_fixed","Used for product choice: fixed p₀","-","#0072B2"),
                ("d_marginal","Real game: unconditional rival distribution","--","#D55E00"),
                ("d_conditional","Real game: conditional on own reputation",":","#009E73")]:
            ax.plot(a["grid"],a[key],style,lw=1.5,label=label,color=color)
        ax.set(title=f"ρ={rho:g}, n={report['n']}",xlabel="Own reputation",ylabel="Probability seller A is selected",ylim=(-.03,1.03))
        ax.legend(fontsize=8);ax.grid(alpha=.18)
    fig.suptitle("Demand comparison: OE p0",fontsize=12)
    save_figure(fig,directory,"07_demand_comparison")

    fig,axes=panel_grid(rhos)
    for rho,ax in zip(rhos,axes):
        if rho not in reports:
            continue
        report=reports[rho];a=report["comparisons"]["arrays"]
        ax.step(a["grid"],a["scalar_policy"],where="mid",color="#0072B2",lw=1.8,label="OE p0")
        for row in report["comparisons"]["summary"]["references"]:
            if row.get("model") != "OE_REAL_MARGINAL":
                continue
            prefix=row["key"]
            curve=a[prefix+"_sigma_a"][:,0]
            label=f"OE distribution (n={row['source_grid']})"
            ax.plot(a["grid"],curve,"--",lw=1.5,label=label,color="#009E73")
        ax.set(title=f"ρ={rho:g}, comparison n={report['n']}",xlabel="Own reputation",ylabel="Probability of product 2",ylim=(-.04,1.04))
        ax.legend(fontsize=7);ax.grid(alpha=.18)
    fig.suptitle("Product choice: OE p0 and OE distribution",fontsize=12)
    save_figure(fig,directory,"08_reference_policies")


def main():
    args=arguments()
    directory=args.output/args.name
    manifest_path=directory/"manifest.json"
    if args.resume:
        if not manifest_path.exists():
            raise SystemExit("No saved manifest to resume")
        manifest=json.loads(manifest_path.read_text())
        if manifest["status"] == "complete":
            raise SystemExit("Run already complete; use a new name")
        saved=manifest["configuration"]
        args=argparse.Namespace(**saved, resume=True)
        args.output=resolve_saved_path(args.output, project_root=ROOT)
        args.figures_output=resolve_saved_path(args.figures_output, project_root=ROOT)
        args.search_cache=resolve_saved_path(args.search_cache, project_root=ROOT) if args.search_cache else None
        manifest.setdefault("resume_history",[]).append(dict(time=now(),source_sha256=source_hashes()))
    else:
        if directory.exists() or (args.figures_output/args.name).exists():
            raise SystemExit(f"Existing results protected: {directory}. Use --resume or a new --name.")
        directory.mkdir(parents=True)
        configuration={k:clean(v) for k,v in vars(args).items() if k!="resume"}
        manifest=dict(status="incomplete",started_utc=now(),configuration=configuration,
                      source_sha256=source_hashes(),completed_cases=[],completed_diagnostics=[],
                      strategy_space="Mixed probabilities alpha(x) in [0,1]; pure actions remain admissible endpoints",
                      interpretation="Scalar consistency plus conditional mixed optimality in a finite approximation is not MPE. Numerical search does not prove completeness or uniqueness.")
    (directory/"arrays").mkdir(exist_ok=True)
    figures=args.figures_output/args.name
    figures.mkdir(parents=True,exist_ok=args.resume)
    write_json(manifest_path,manifest)
    params=Parameters(**{k:getattr(args,k) for k in asdict(Parameters()) if k!="rho"})
    cases=[];diagnostics={}
    ordered=([.95] if .95 in args.rhos else [])+[r for r in sorted(args.rhos) if r!=.95]
    try:
        for rho in ordered:
            for n in sorted(args.grids):
                tag=case_tag(rho,n);base=directory/"arrays"/tag
                if tag in manifest["completed_cases"]:
                    case=load_bundle(base)
                    print(f"Reusing search ρ={rho:g}, n={n}",flush=True)
                else:
                    started=time.monotonic()
                    print(f"Searching scalar map ρ={rho:g}, n={n}",flush=True)
                    experiment=ScalarExperiment(n=n,params=replace(params,rho=rho),order=args.order)
                    pure_seed,seed_provenance=load_search_cache(args.search_cache,tag)
                    search=experiment.search(p0_points=args.p0_points,consistency_tol=args.consistency_tol,
                                             pure_seed=pure_seed,progress=lambda message:print("  "+message,flush=True))
                    selected=(select_candidate(search["candidates"])
                              if search["candidates"] else search["best_diagnostic"])
                    case=dict(rho=rho,n=n,selected=selected,candidates=search["candidates"],
                              map_rows=[evaluation_row(e) for e in search["evaluations"]],
                              search_log=search["search_log"],failures=search["failures"],
                              brackets=search["brackets"],search_summary=search["summary"],
                              mixing_branches=search.get("mixing_branches",[]),
                              search_seed_provenance=seed_provenance,
                              configuration=experiment.configuration(),
                              demand_audit=experiment.demand_audit(),
                              elapsed_seconds=time.monotonic()-started)
                    save_bundle(base,case)
                    manifest["completed_cases"].append(tag)
                    write_json(manifest_path,manifest)
                    print(f"  {len(case['candidates'])} verified fixed points ({sum(mixed_count(e)>0 for e in case['candidates'])} interior); selected |F-p₀|="+
                          (f"{abs(selected['G']):.3g}" if selected is not None else "unavailable"),flush=True)
                cases.append(case)
                export_tables(cases,directory)
                if rho==.95 and case["selected"] is None:
                    raise RuntimeError("The rho=.95 joint-stationary validation failed; further rho runs are blocked until it is resolved")
            finest=max((c for c in cases if c["rho"]==rho),key=lambda c:c["n"])
            preferred=selected_case(cases,rho)
            diagnostic_cases=[preferred]+([finest] if finest["n"]!=preferred["n"] else [])
            entries=[(case,idx,evaluation) for case in diagnostic_cases for idx,evaluation in enumerate(
                case["candidates"] or ([case["selected"]] if case["selected"] is not None else []))]
            # Diagnose every root on the finest verified grid and, if different,
            # the finest requested grid's best unverified diagnostic as well.
            for diagnostic_case,idx,evaluation in entries:
                n=diagnostic_case["n"]
                tag=case_tag(rho,n)+f"_diagnostics_{idx}"
                base=directory/"arrays"/tag
                if tag in manifest["completed_diagnostics"]:
                    report=load_bundle(base)
                    print(f"Reusing diagnostics ρ={rho:g}, candidate={idx}",flush=True)
                else:
                    experiment=ScalarExperiment(n=n,params=replace(params,rho=rho),order=args.order)
                    print(f"Simulation and actual-game diagnostics ρ={rho:g}, n={n}, candidate={idx}",flush=True)
                    report=dict(rho=rho,n=n,p0=evaluation["p0"],
                                mixed_nodes=mixed_count(evaluation),
                                fixed_point_verified=bool(diagnostic_case["candidates"]),
                                simulation=simulate_ts(experiment.game,evaluation,periods=args.periods,
                                    burn_in=args.burn_in,trajectories=args.trajectories,
                                    seed=args.seed+RHOS.index(rho) if rho in RHOS else args.seed+idx),
                                demand=demand_diagnostics(experiment.game,evaluation),
                                deviations=deviation_diagnostics(experiment.game,evaluation),
                                comparisons=compare_existing(experiment.game,evaluation,ROOT))
                    save_bundle(base,report)
                    manifest["completed_diagnostics"].append(tag)
                    write_json(manifest_path,manifest)
                diagnostics[(rho,n,idx)]=report
        make_figures(cases,diagnostics,figures)
        export_tables(cases,directory)
        export_diagnostics(diagnostics,directory)
        write_findings(cases,diagnostics,directory)
        manifest.update(status="complete",finished_utc=now(),final_source_sha256=source_hashes())
        write_json(manifest_path,manifest)
        print(f"Complete. Data: {directory}\nFigures: {figures}",flush=True)
    except BaseException as exc:
        manifest["last_error"]={"type":type(exc).__name__,"message":str(exc),"time":now()}
        write_json(manifest_path,manifest)
        raise


def export_tables(cases,directory):
    rows=[];roots=[];mapping=[];refinement=[]
    for case in cases:
        base=dict(rho=case["rho"],grid_size=case["n"])
        e=case["selected"]
        rows.append(dict(**base,fixed_points_found=len(case["candidates"]),
                         interior_fixed_points_found=sum(mixed_count(e)>0 for e in case["candidates"]),
                         fixed_point_verified=bool(case["candidates"]),
                         **(evaluation_row(e) if e is not None else {}),
                         elapsed_seconds=case["elapsed_seconds"]))
        for i,root in enumerate(case["candidates"]):
            roots.append(dict(**base,candidate=i,**evaluation_row(root),
                              mixed_reputations=np.asarray(root["grid"])[(root["policy"]>0)&(root["policy"]<1)].tolist(),
                              mixing_probabilities=np.asarray(root["policy"])[(root["policy"]>0)&(root["policy"]<1)].tolist()))
        mapping.extend(dict(**base,**row) for row in case["map_rows"])
    for rho in sorted({c["rho"] for c in cases}):
        group=sorted((c for c in cases if c["rho"]==rho),key=lambda c:c["n"])
        for low,high in zip(group[:-1],group[1:]):
            a,b=low["selected"],high["selected"]
            row=dict(rho=rho,base_grid=low["n"],refined_grid=high["n"],
                     both_fixed_points_verified=bool(low["candidates"] and high["candidates"]))
            if a is not None and b is not None:
                row.update(p0_change=b["p0"]-a["p0"],F_change=b["F"]-a["F"],
                           base_consistency_residual=abs(a["G"]),refined_consistency_residual=abs(b["G"]),
                           mean_reputation_change=float(b["mu_b"]@b["grid"]-a["mu_b"]@a["grid"]))
            refinement.append(row)
    write_csv(directory/"summary.csv",rows)
    write_csv(directory/"fixed_points.csv",roots)
    write_csv(directory/"map_evaluations.csv",mapping)
    write_csv(directory/"grid_refinement.csv",refinement)


def export_diagnostics(diagnostics,directory):
    summaries={};sim_rows=[];demand_rows=[];comparison_rows=[];deviation_rows=[]
    for (rho,n,index),report in diagnostics.items():
        summaries[f"rho={rho:g},n={n},candidate={index}"]={
            k:report[k]["summary"] for k in ("simulation","demand","deviations","comparisons")}
        base=dict(rho=rho,grid_size=n,candidate=index,p0=report["p0"],
                  fixed_point_verified=report["fixed_point_verified"])
        sim=report["simulation"]["summary"]
        row={**base,"stationary_F":sim["stationary_F"]}
        for metric,estimate in sim["estimates"].items():
            for stat,value in estimate.items():
                row[metric+"_"+stat]=value
        row.update({key:sim[key] for key in ("periods","burn_in","trajectories","seed",
                    "ts_minus_posterior_B_ci_contains_zero","ts_B_ci_contains_stationary_F",
                    "posterior_B_ci_contains_stationary_F")})
        sim_rows.append(row)
        demand_rows.extend({**base,"comparison":name,**error} for name,error
                           in report["demand"]["summary"]["errors"].items())
        comp=report["comparisons"]["summary"]
        comparison_rows.append({**base,"model":"scalar","status":"evaluated",**comp["scalar_statistics"]})
        for ref in comp["references"]:
            comparison_rows.append({**base,**{k:v for k,v in ref.items() if k not in
                    ("native_summary","stationary_diagnostics","statistics")},**ref.get("statistics",{})})
        deviation_rows.append({**base,**report["deviations"]["summary"]})
    write_json(directory/"diagnostic_summaries.json",summaries)
    write_csv(directory/"simulation_summary.csv",sim_rows)
    write_csv(directory/"demand_errors.csv",demand_rows)
    write_csv(directory/"model_comparisons.csv",comparison_rows)
    write_csv(directory/"deviation_summary.csv",deviation_rows)


def write_findings(cases,diagnostics,directory):
    lines=["EXPERIMENTO OE p0 CON ESTRATEGIAS MIXTAS", "",
        "Cada política asigna una probabilidad al producto 2. La mezcla interior exige indiferencia.",
        "Los extremos 0 y 1 siguen siendo admisibles dentro del espacio de estrategias mixtas.",
        "Se verifican consistencia escalar, optimalidad mixta y estacionariedad conjunta; esto no certifica MPE.",
        "La búsqueda finita no demuestra unicidad ni que se hayan encontrado todos los puntos fijos.",
        "", "Resultados por rho y tamaño de grilla:",
        "rho | puntos | candidatos verificados | candidatos con mezcla interior | |F-p0| seleccionado | p0 seleccionado | estados con mezcla"]
    for case in sorted(cases,key=lambda c:(c["rho"],c["n"])):
        e=case["selected"]
        lines.append(f"{case['rho']:g} | {case['n']} | {len(case['candidates'])} | {sum(mixed_count(c)>0 for c in case['candidates'])} | "+
                     (f"{abs(e['G']):.9g} | {e['p0']:.9f} | {mixed_count(e)}" if e else "no evaluable | no evaluable | no evaluable"))
    lines += ["", "Simulaciones y diagnóstico de demanda:",
        "Los intervalos son del 95%, calculados entre trayectorias independientes.",
        "La igualdad del promedio TS y el promedio posterior no basta para verificar el punto fijo."]
    for (rho,n,index),report in sorted(diagnostics.items()):
        sim=report["simulation"]["summary"]
        estimates=sim["estimates"]
        error=report["demand"]["summary"]["errors"]["fixed_minus_conditional"]
        label="punto fijo verificado" if report["fixed_point_verified"] else "diagnóstico sin punto fijo verificado"
        lines += [f"rho={rho:g}, n={n}, candidato={index}: {label}",
            f"  p0={report['p0']:.9f}; F={sim['stationary_F']:.9f}",
            f"  TS medio B={estimates['mean_ts']['mean'][1]:.9f}; media posterior B={estimates['mean_posterior']['mean'][1]:.9f}",
            f"  IC95 de TS menos media posterior B={estimates['ts_minus_posterior']['ci95'][1]}",
            f"  Error absoluto de demanda fija vs condicional, ponderado por reputación={error['stationary_weighted_absolute']:.9g}",
            f"  Ganancia media descontada por desviación (A,B)={report['deviations']['summary']['weighted_regret']}"]
    lines += ["", "Para rho altos, el promedio de una simulación puede depender fuertemente de su trayectoria.",
              "La invariancia numérica no demuestra rapidez de convergencia ni precisión en el modelo continuo.",
              "Los fallos de búsqueda y saltos de política están guardados en arrays/*.json."]
    (directory/"RESULTADOS.txt").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    main()
