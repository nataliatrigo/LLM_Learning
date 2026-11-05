"""Render saved OE p0 refinement results without solving or simulating.

The original numerical and runner sources remain unchanged. Figures are kept
inside oblivious/02_oe_p0/figures/<campaign>/n<grid>, separate from the n51 figures.
"""

import argparse
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/p0-refinement-plots-mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from oblivious.layout import resolve_saved_path
from oblivious.oe_p0.code import run_experiment as original
from oblivious.oe_p0.code.core import ScalarExperiment
from oblivious.oe_p0.code.diagnostics import demand_diagnostics
from oblivious.oe_p0.code.primitives import Parameters

SPEC = importlib.util.spec_from_file_location("saved_refinement_inputs",
                                             Path(__file__).with_name("run_refinement.py"))
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def record(base, inputs):
    for suffix in (".json", ".npz"):
        path=base.with_suffix(suffix)
        inputs[str(path.relative_to(ROOT))]=digest(path)


def restore_map(case, inputs):
    """Restore search geometry only; keep the current, recertified candidates."""
    result=dict(case)
    if "map_rows" in result:
        return result
    provenance=case.get("provenance",{})
    if provenance.get("kind") != "historical_read_only":
        raise ValueError("No saved scalar map or historical map provenance")
    base=resolve_saved_path(provenance["base"], project_root=ROOT).resolve()
    if not base.is_relative_to(ROOT):
        raise ValueError("Historical map must be inside the project")
    for suffix,expected in provenance["files"].items():
        if digest(base.with_suffix(suffix)) != expected:
            raise ValueError(f"Historical map changed: {base}")
    saved=original.load_bundle(base)
    if saved["rho"] != case["rho"] or saved["n"] != case["n"]:
        raise ValueError("Historical map grid/rho differ from the displayed solution")
    for field in ("map_rows","mixing_branches"):
        result[field]=saved.get(field,[])
    record(base,inputs)
    return result


def distribution_reference(campaign,rho,n,params,config,inputs,campaign_fingerprint):
    """Use a certified same-grid reference only; no transferred stationary law."""
    base=RUN.case_dir(campaign,"oe",rho,n)/"solution"
    reference=RUN.load_checked(base)
    if reference and RUN.verified(reference.get("selected")):
        if (reference.get("campaign_fingerprint")!=campaign_fingerprint
                or reference.get("configuration",{}).get("parameters")!=dict(config["parameters"],rho=rho)):
            raise ValueError("OE distribution reference belongs to another campaign or parameter set")
        record(base,inputs)
        return reference["selected"]
    reference,provenance=RUN.historical_case("oe",rho,n,params,config)
    if reference and RUN.verified(reference.get("selected")):
        record(resolve_saved_path(provenance["base"], project_root=ROOT),inputs)
        return reference["selected"]
    return None


def cdf_figure(cases,simulations,save):
    rhos=sorted(cases)
    fig,axes=original.panel_grid(rhos)
    for rho,ax in zip(rhos,axes):
        case,simulation=cases[rho],simulations[rho]
        selected=case["selected"]
        grid=selected["grid"]
        stationary=np.cumsum((selected["nu"].sum(axis=0)+selected["nu"].sum(axis=1))/2)
        empirical=simulation["distribution"]
        ax.step(grid,stationary,where="post",color="#D55E00",lw=1.7,label="Stationary grid law")
        ax.plot(empirical["grid"],empirical["cdf"],color="#0072B2",lw=1.4,label="Simulation")
        ax.fill_between(empirical["grid"],empirical["cdf_low"],empirical["cdf_high"],
                        color="#0072B2",alpha=.15,label="95% pointwise MC CI")
        ax.set(title=f"ρ={rho:g}, n={case['n']}",xlabel="Reputation",ylabel="CDF, pooled sellers",
               xlim=(0,1),ylim=(-.02,1.02))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8,loc="best")
    fig.suptitle("OE p0: stationary and simulated reputation distributions",fontsize=13)
    save(fig,"05_reputation_cdf")


def outcomes_figure(cases,simulations,save):
    rhos=sorted(cases)
    fig,axes=plt.subplots(3,1,figsize=(10,9.5),layout="constrained")
    metrics=(("product2_market","Product 2 usage","Share of purchases"),
             ("seller_profit","Seller profit","Profit per seller and calendar period"),
             ("success_rate","Customer success","Success share of purchases"))
    for ax,(metric,title,ylabel) in zip(axes,metrics):
        x=np.arange(len(rhos))
        predicted=np.array([cases[rho]["predicted"][metric] for rho in rhos])
        estimates=[simulations[rho]["summary"][metric] for rho in rhos]
        mean=np.array([e["mean"] for e in estimates])
        low=np.array([e["ci_low"] for e in estimates])
        high=np.array([e["ci_high"] for e in estimates])
        ax.plot(x,predicted,color="#D55E00",lw=1.6,marker="D",mfc="white",ms=5,
                label="Stationary prediction")
        ax.errorbar(x,mean,yerr=[mean-low,high-mean],color="#0072B2",lw=1.4,
                    marker="o",ms=4,capsize=3,label="Simulation (95% MC CI)")
        ax.set_xticks(x,[f"{rho:g}" for rho in rhos])
        ax.set(title=title,xlabel="Forgetting factor ρ (discrete values)",ylabel=ylabel)
        if metric != "seller_profit":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y",alpha=.22)
        ax.legend(fontsize=9)
    meta=next(iter(simulations.values()))["metadata"]
    fig.suptitle(f"OE p0, n={next(iter(cases.values()))['n']}: actual two-seller TS\n"
                 f"{meta['replications']} paths; {meta['periods']:,} retained periods; "
                 f"{meta['burn_in']:,} burn-in; initial (0.5, 0.5)",fontsize=13)
    save(fig,"06_simulated_outcomes")


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",default="refinement_101_201")
    parser.add_argument("--grid",type=int,default=101)
    args=parser.parse_args(argv)
    if Path(args.name).name != args.name or args.name in ("", ".", "..") or args.grid<3:
        parser.error("Require a campaign name and a valid grid")
    campaign=RUN.OUTPUT/args.name
    manifest=json.loads((campaign/"manifest.json").read_text())
    config=manifest["configuration"]
    directory=ROOT/"oblivious/02_oe_p0/figures"/args.name/f"n{args.grid:03d}"
    cases,simulations,reports,inputs,missing_references={},{},{},{},[]
    for rho in sorted(config["rhos"]):
        folder=RUN.case_dir(campaign,"p0",rho,args.grid)
        case=RUN.load_checked(folder/"solution")
        simulation=RUN.load_checked(folder/"simulation_grid")
        if case is None or simulation is None or not RUN.verified(case.get("selected")):
            raise ValueError(f"Need a verified solution and saved grid simulation: rho={rho}, n={args.grid}")
        selected=case["selected"]
        if (case.get("campaign_fingerprint")!=manifest["fingerprint"]
                or simulation.get("campaign_fingerprint")!=manifest["fingerprint"]
                or simulation.get("policy_sha256")!=sha256(selected["policy"].tobytes()).hexdigest()):
            raise ValueError("Saved solution/simulation provenance mismatch")
        record(folder/"solution",inputs)
        record(folder/"simulation_grid",inputs)
        cases[rho]=restore_map(case,inputs)
        cases[rho]["map_rows"]=sorted(cases[rho].get("map_rows",[]),
                                      key=lambda row:float(row["p0"]))
        simulations[rho]=simulation
        params=Parameters(**case["configuration"]["parameters"])
        experiment=ScalarExperiment(n=args.grid,params=params,order=config["order"])
        demand=demand_diagnostics(experiment.game,selected)
        reference=distribution_reference(campaign,rho,args.grid,params,config,inputs,manifest["fingerprint"])
        arrays=dict(grid=selected["grid"],scalar_policy=selected["policy"])
        references=[]
        if reference is not None:
            if not np.array_equal(reference["grid"],selected["grid"]):
                raise ValueError("Reference policy must use exactly the displayed grid")
            arrays["oe_sigma_a"]=np.broadcast_to(reference["policy"][:,None],(args.grid,args.grid))
            references=[dict(model="OE_REAL_MARGINAL",key="oe",source_grid=args.grid)]
        else:
            missing_references.append(rho)
        reports[(rho,args.grid,0)]=dict(rho=rho,n=args.grid,p0=selected["p0"],demand=demand,
            comparisons=dict(arrays=arrays,summary=dict(references=references)))
    directory.mkdir(parents=True,exist_ok=True)
    figures=[]
    def save(fig,location_or_name,name=None):
        name=location_or_name if name is None else name
        if name=="08_reference_policies" and missing_references:
            for rho,ax in zip(sorted(cases),fig.axes):
                if rho in missing_references:
                    ax.text(.98,.05,f"No verified OE distribution at n={args.grid}",
                            transform=ax.transAxes,ha="right",fontsize=7,color=".4")
        if name in ("07_demand_comparison","08_reference_policies"):
            for ax in fig.axes:
                for line in ax.lines:
                    line.set_linestyle("-")
        for extension in ("png","pdf"):
            path=directory/f"{name}.{extension}"
            fig.savefig(path,dpi=200,bbox_inches="tight",facecolor="white")
            figures.append(path)
        plt.close(fig)
    previous_save=original.save_figure
    try:
        original.save_figure=save
        with plt.rc_context({"pdf.fonttype":42,"ps.fonttype":42}):
            original.make_figures(list(cases.values()),reports,directory)
            cdf_figure(cases,simulations,save)
            outcomes_figure(cases,simulations,save)
    finally:
        original.save_figure=previous_save
    original.write_json(directory/"figure_manifest.json",dict(
        campaign=args.name,grid=args.grid,rhos=sorted(cases),model="OE p0",
        input_sha256=inputs,plot_code_sha256=digest(Path(__file__)),
        reused_plot_code_sha256=digest(Path(original.__file__)),
        figures={p.name:digest(p) for p in figures},
        scope="Saved certified equilibria and saved grid simulations; no new solves or simulations",
        missing_verified_same_grid_distribution_reference=missing_references))
    print(f"Saved {len(figures)//2} figures, PNG and PDF, in {directory}")


if __name__=="__main__":
    main()
