"""Compare saved, certified refinement results without new solves/simulations."""

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/refinement-comparison-plots")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]


def module(name,filename):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(filename))
    result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


RUN=module("comparison_saved_results","run_refinement.py")
from oblivious.layout import resolve_saved_path

PLOT=module("comparison_original_layout","plot_comparison.py")
ORDER=("oe","p0","monopoly","conditional")
METRICS=("product2_market","seller_profit")


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def record(base,inputs):
    for suffix in (".json",".npz"):
        path=base.with_suffix(suffix)
        inputs[str(path.relative_to(ROOT))]=digest(path)


def load_oe_retry(directory, campaign, manifest, inputs):
    """Accept a separately certified retry only for the failed OE .99/101 case."""
    directory=Path(directory)
    retry_path=directory/"manifest.json"
    retry=json.loads(retry_path.read_text())
    settings=retry["configuration"]
    config=manifest["configuration"]
    if (settings.get("model")!="oe" or settings.get("rho")!=.99
            or settings.get("n")!=101
            or settings.get("parameters")!=config["parameters"]
            or any(settings.get(key)!=config[key] for key in ("order","tol","dp_tol"))
            or resolve_saved_path(settings["source_campaign"], project_root=ROOT).resolve()!=campaign.resolve()
            or retry.get("source_campaign_fingerprint")!=manifest["fingerprint"]
            or settings.get("input_sha256")!=retry.get("input_sha256")
            or retry.get("fingerprint")!=RUN.fingerprint(settings,retry["source_sha256"])):
        raise ValueError("OE retry belongs to another experiment or source campaign")
    for collection in (retry["input_sha256"],retry["source_sha256"]):
        for name,expected in collection.items():
            path=resolve_saved_path(name, project_root=ROOT)
            if digest(path)!=expected:
                raise ValueError(f"OE retry input or code changed: {path}")
            inputs[str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)]=expected
    inputs[str(retry_path.relative_to(ROOT))]=digest(retry_path)
    case=RUN.load_checked(directory/"solution")
    if (case is None or case.get("campaign_fingerprint")!=retry["fingerprint"]
            or case.get("source_campaign_fingerprint")!=manifest["fingerprint"]
            or not RUN.verified(case.get("selected"))
            or not case.get("demand_audit",{}).get("verified")):
        raise ValueError("OE retry has no compatible certified equilibrium")
    return retry,case


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",default="refinement_101_201")
    parser.add_argument("--grid",type=int,default=101)
    parser.add_argument("--oe-retry",help="Isolated OE .99 retry name under data/, after its grid simulation")
    args=parser.parse_args(argv)
    if Path(args.name).name!=args.name or args.name in ("", ".", "..") or args.grid<3:
        parser.error("Require a campaign name and grid >=3")
    campaign=RUN.OUTPUT/args.name
    manifest=json.loads((campaign/"manifest.json").read_text())
    config=manifest["configuration"]
    if set(config["models"])!=set(ORDER):
        raise ValueError("The comparison requires all four models in this campaign")
    rhos=sorted(config["rhos"])
    original_config=RUN.OUTPUT/"comparison.json"
    design=json.loads(original_config.read_text())
    if config["parameters"]!=design["parameters"]:
        raise ValueError("Campaign and original comparison primitives differ")
    models={key:dict(design["models"][key],linestyle="-") for key in ORDER if key!="conditional"}
    models["conditional"]=dict(label="Conditional OE",color="#0072B2",linestyle="-")
    simulation_config={key:config[key] for key in ("periods","burn_in","replications","seed")}
    plot_config=dict(rhos=rhos,grid=args.grid,models=models,simulation=simulation_config)
    inputs={str(original_config.relative_to(ROOT)):digest(original_config)}
    retry_folder=retry_manifest=retry_case=None
    if args.oe_retry:
        if args.grid!=101 or Path(args.oe_retry).name!=args.oe_retry or args.oe_retry in ("", ".", ".."):
            parser.error("--oe-retry requires a data folder name and --grid 101")
        retry_folder=RUN.OUTPUT/args.oe_retry
        retry_manifest,retry_case=load_oe_retry(retry_folder,campaign,manifest,inputs)
    policies,outcomes,missing=[],[],[]
    for model in ORDER:
        for rho in rhos:
            folder=RUN.case_dir(campaign,model,rho,args.grid)
            is_retry=retry_case is not None and model=="oe" and rho==.99
            expected_fingerprint=manifest["fingerprint"]
            if is_retry:
                folder=retry_folder
                case=retry_case
                expected_fingerprint=retry_manifest["fingerprint"]
            else:
                case=RUN.load_checked(folder/"solution")
            if case is None:
                raise ValueError(f"Missing completed search: {model}, rho={rho}, n={args.grid}")
            record(folder/"solution",inputs)
            if (case.get("campaign_fingerprint")!=expected_fingerprint
                    or case.get("n")!=args.grid or case.get("rho")!=rho
                    or case["configuration"]["parameters"]!=dict(config["parameters"],rho=rho)):
                raise ValueError("A saved case has incompatible provenance or economics")
            selected=case.get("selected")
            valid=RUN.verified(selected) and case.get("demand_audit",{}).get("verified")
            if not valid:
                missing.append(dict(model=model,rho=rho,n=args.grid,reason="No certified equilibrium"))
                grid=np.linspace(0,1,args.grid)
                alpha=np.full(args.grid,np.nan)
            else:
                grid,alpha=np.asarray(selected["grid"]),np.asarray(selected["policy"])
                if not np.array_equal(grid,np.linspace(0,1,args.grid)) or alpha.shape!=grid.shape:
                    raise ValueError("Policy grid differs from comparison grid")
                simulation=RUN.load_checked(folder/"simulation_grid")
                if simulation is None:
                    raise ValueError(f"Missing saved simulation: {model}, rho={rho}")
                record(folder/"simulation_grid",inputs)
                meta=simulation["metadata"]
                if (simulation.get("campaign_fingerprint")!=expected_fingerprint
                        or (is_retry and simulation.get("source_campaign_fingerprint")!=manifest["fingerprint"])
                        or simulation.get("policy_sha256")!=sha256(alpha.tobytes()).hexdigest()
                        or any(meta.get(k)!=v for k,v in simulation_config.items())
                        or meta.get("mode")!="grid" or meta.get("initial")!=[.5,.5]
                        or meta.get("params")!=case["configuration"]["parameters"]):
                    raise ValueError("Simulation settings/policy differ from the common comparison")
                for metric in METRICS:
                    predicted=case["predicted"][metric]
                    if abs(predicted-simulation["predicted"][metric])>1e-12:
                        raise ValueError("Solution and simulation predictions disagree")
                    estimate=simulation["summary"][metric]
                    outcomes.append(dict(rho=rho,n=args.grid,model=model,metric=metric,
                        stationary=predicted,simulation_mean=estimate["mean"],
                        simulation_ci_low=estimate["ci_low"],simulation_ci_high=estimate["ci_high"],
                        simulation_n_valid=estimate["n_valid"],verified=True))
            if not valid:
                for metric in METRICS:
                    outcomes.append(dict(rho=rho,n=args.grid,model=model,metric=metric,
                        stationary=np.nan,simulation_mean=np.nan,simulation_ci_low=np.nan,
                        simulation_ci_high=np.nan,simulation_n_valid=0,verified=False))
            policies.extend(dict(rho=rho,n=args.grid,model=model,reputation=float(x),
                                 product2_probability=float(a),verified=valid)
                            for x,a in zip(grid,alpha))
    directory=ROOT/"oblivious/07_full_comparison/Figures"/args.name/f"n{args.grid:03d}"
    directory.mkdir(parents=True,exist_ok=True)
    policies,outcomes=pd.DataFrame(policies),pd.DataFrame(outcomes)
    figures=[]
    def save(fig,name):
        if name=="01_optimal_policies_across_rho":
            for rho,ax in zip(rhos,fig.axes):
                absent=[models[row["model"]]["label"] for row in missing if row["rho"]==rho]
                if absent:
                    ax.set_title(rf"$\rho={rho:g}$, n={args.grid} | {len(models)-len(absent)} verified models")
                    ax.text(.98,.88,"Unverified; not plotted:\n"+", ".join(absent),
                        transform=ax.transAxes,ha="right",va="top",fontsize=8,
                        color=".35",bbox=dict(facecolor="white",edgecolor="none",alpha=.9))
        else:
            for ax in fig.axes:
                for line in ax.lines:
                    if line.get_linestyle() in ("--",":","-."):
                        line.set_linestyle("-")
            for legend in fig.legends:
                for line in legend.get_lines():
                    if line.get_linestyle() in ("--",":","-."):
                        line.set_linestyle("-")
            if missing:
                labels="; ".join(f"{models[row['model']]['label']}, ρ={row['rho']:g}" for row in missing)
                fig.text(.53,.006,"Unavailable: "+labels+" (no certified equilibrium).",
                         ha="center",fontsize=8,color=".35")
        for extension in ("png","pdf"):
            path=directory/f"{name}.{extension}"
            fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white")
            figures.append(path)
        plt.close(fig)
    previous_save=PLOT.save
    try:
        PLOT.save=save
        with plt.rc_context(PLOT.STYLE):
            PLOT.policy_figure(plot_config,policies)
            PLOT.outcome_figure(plot_config,outcomes,"product2_market",f"Product 2 usage (n={args.grid})",
                "Product 2 share of all purchases","02_product2_usage_across_rho")
            PLOT.outcome_figure(plot_config,outcomes,"seller_profit",f"Seller profit (n={args.grid})",
                "Mean seller profit per calendar period","03_profit_across_rho")
    finally:
        PLOT.save=previous_save
    policies.to_csv(directory/"plotted_policies.csv",index=False)
    outcomes.to_csv(directory/"plotted_outcomes.csv",index=False)
    RUN.atomic_json(directory/"figure_manifest.json",dict(campaign=args.name,grid=args.grid,
        oe_retry=args.oe_retry,
        rhos=rhos,models=models,missing=missing,verified_cases=len(ORDER)*len(rhos)-len(missing),
        input_sha256=inputs,plot_code_sha256=digest(Path(__file__)),
        reused_layout_sha256=digest(Path(PLOT.__file__)),
        figures={p.name:digest(p) for p in figures},
        scope="Only saved certified equilibria and saved grid simulations; MC intervals exclude initialization and grid error"))
    print(f"Saved three comparisons, PNG and PDF, in {directory}")
    print(f"Certified cases plotted: {len(ORDER)*len(rhos)-len(missing)}; unavailable: {missing}")


if __name__=="__main__":
    main()
