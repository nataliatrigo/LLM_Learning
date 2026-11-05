"""Compare saved joint stationary laws of Distribution and Conditional OE.

Uses the same certified grid for both models. Does not solve or simulate.
Each row shares a probability color scale between its two model panels;
the third panel shows the signed difference of the actual probability masses.
"""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/joint-comparison-mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.ticker import FuncFormatter, LogLocator, NullLocator
import matplotlib.pyplot as plt
import numpy as np


ROOT=Path(__file__).resolve().parents[3]
SPEC=importlib.util.spec_from_file_location("joint_comparison_saved_sources",
    Path(__file__).with_name("plot_refinement_comparison.py"))
COMMON=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMMON)
RUN=COMMON.RUN
COLORS={"oe":"#009E73","conditional":"#0072B2"}
STYLE={"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":11,
       "axes.labelsize":10,"pdf.fonttype":42,"ps.fonttype":42}


def record(base,inputs):
    COMMON.record(base,inputs)
    path=base.with_name(base.name+"_sha256.json")
    inputs[str(path.relative_to(ROOT))]=COMMON.digest(path)


def moments(grid,nu):
    ma,mb=nu.sum(axis=1),nu.sum(axis=0)
    a,b=float(grid@ma),float(grid@mb)
    va,vb=float((grid-a)**2@ma),float((grid-b)**2@mb)
    covariance=float(grid@nu@grid-a*b)
    correlation=covariance/np.sqrt(va*vb) if va*vb>0 else None
    peak=np.unravel_index(np.argmax(nu),nu.shape)
    return dict(mean_a=a,mean_b=b,correlation=correlation,
        peak_a=float(grid[peak[0]]),peak_b=float(grid[peak[1]]),
        peak_mass=float(nu[peak]),both_ge_half=float(nu[np.ix_(grid>=.5,grid>=.5)].sum()))


def collect(campaign,n,rhos,retry_name):
    path=campaign/"manifest.json"
    manifest=json.loads(path.read_text())
    config=manifest["configuration"]
    if not set(("oe","conditional"))<=set(config["models"]):
        raise ValueError("The campaign must contain both OE distribution and Conditional OE")
    if n not in config["grids"] or not set(rhos)<=set(config["rhos"]):
        raise ValueError("Requested grid or rho is absent from this campaign")
    inputs={str(path.relative_to(ROOT)):COMMON.digest(path)}
    retry_manifest=retry_case=None
    if n==101 and .99 in rhos:
        if not retry_name:
            raise ValueError("rho=.99/n101 needs the certified isolated OE retry")
        retry_folder=RUN.OUTPUT/retry_name
        retry_manifest,retry_case=COMMON.load_oe_retry(retry_folder,campaign,manifest,inputs)
    pairs,metrics=[],[]
    for rho in rhos:
        cases={}
        for model in ("oe","conditional"):
            base=RUN.case_dir(campaign,model,rho,n)/"solution"
            fingerprint=manifest["fingerprint"]
            if retry_case is not None and model=="oe" and rho==.99:
                base=RUN.OUTPUT/retry_name/"solution"
                case=retry_case
                fingerprint=retry_manifest["fingerprint"]
            else:
                case=RUN.load_checked(base)
            if (case is None or not RUN.verified(case.get("selected"))
                    or not case.get("demand_audit",{}).get("verified")
                    or case.get("campaign_fingerprint")!=fingerprint
                    or case.get("n")!=n or case.get("rho")!=rho
                    or case["configuration"]["parameters"]!=dict(config["parameters"],rho=rho)
                    or case["configuration"]["order"]!=config["order"]):
                raise ValueError(f"Missing or incompatible certified result: {model}, rho={rho}, n={n}")
            candidate=case["selected"]
            grid,nu=np.asarray(candidate["grid"]),np.asarray(candidate["nu"])
            if (not np.array_equal(grid,np.linspace(0,1,n)) or nu.shape!=(n,n)
                    or not np.isfinite(nu).all() or np.any(nu<0)
                    or abs(float(nu.sum())-1)>1e-10
                    or np.max(np.abs(nu-nu.T))>1e-9):
                raise ValueError("Expected a normalized symmetric joint law on the common grid")
            record(base,inputs)
            cases[model]=dict(grid=grid,nu=nu,moments=moments(grid,nu),source=str(base.relative_to(ROOT)))
        a,b=cases["oe"],cases["conditional"]
        delta=b["nu"]-a["nu"]
        if abs(float(delta.sum()))>2e-10:
            raise ValueError("Signed probability differences must sum to zero")
        marginal_a,marginal_b=a["nu"].sum(axis=0),b["nu"].sum(axis=0)
        distance=float(np.sum(np.abs(np.cumsum(marginal_a-marginal_b)[:-1]))/(n-1))
        row=dict(rho=rho,n=n,joint_total_variation=float(.5*np.abs(delta).sum()),
                 marginal_wasserstein=distance)
        for model in ("oe","conditional"):
            row.update({model+"_"+name:value for name,value in cases[model]["moments"].items()})
        pairs.append(dict(rho=rho,grid=a["grid"],oe=a,conditional=b,delta=delta,metrics=row))
        metrics.append(row)
    return pairs,metrics,inputs


def zoom_bounds(pair,tail=.0005):
    """Common square view; its exact retained joint masses are reported."""
    grid=pair["grid"]
    lower,upper=[],[]
    for model in ("oe","conditional"):
        for marginal in (pair[model]["nu"].sum(axis=0),pair[model]["nu"].sum(axis=1)):
            cdf=np.cumsum(marginal)
            lower.append(grid[min(np.searchsorted(cdf,tail),len(grid)-1)])
            upper.append(grid[min(np.searchsorted(cdf,1-tail),len(grid)-1)])
    pad=1.5/(len(grid)-1)
    return max(0.,float(min(lower)-pad)),min(1.,float(max(upper)+pad))


def percent(value,_):
    return f"{100*value:.3g}"


def make_figure(pairs,n,directory,name,zoom):
    rows=len(pairs)
    fig=plt.figure(figsize=(13.8,3.65*rows+1.1))
    gs=fig.add_gridspec(rows,5,width_ratios=[1,1,.045,1,.045],
        left=.065,right=.955,bottom=.075 if rows<=3 else .04,
        top=.92 if rows<=3 else .965,wspace=.32,hspace=.42)
    probability_cmap=plt.get_cmap("viridis").copy()
    probability_cmap.set_bad("white")
    probability_cmap.set_under("white")
    difference_cmap=LinearSegmentedColormap.from_list("oe_conditional_difference",
        [COLORS["oe"],"#ffffff",COLORS["conditional"]])
    view=[]
    for r,pair in enumerate(pairs):
        axes=[fig.add_subplot(gs[r,c]) for c in (0,1,3)]
        g=pair["grid"];h=g[1]-g[0]
        extent=(g[0]-h/2,g[-1]+h/2,g[0]-h/2,g[-1]+h/2)
        lo,hi=zoom_bounds(pair) if zoom else (0.,1.)
        norm=LogNorm(vmin=1e-7,vmax=max(pair[k]["nu"].max() for k in ("oe","conditional")))
        for ax,model,label in zip(axes[:2],("oe","conditional"),("OE distribution","Conditional OE")):
            # Stored rows index A and columns B; transpose for x=A, y=B.
            im=ax.imshow(np.ma.masked_less(pair[model]["nu"].T,norm.vmin),origin="lower",
                extent=extent,cmap=probability_cmap,norm=norm,interpolation="nearest")
            ax.set_title(rf"{label} | $\rho={pair['rho']:g}$",color=COLORS[model],pad=9)
            ax.text(.025,.975,f"Media: {pair[model]['moments']['mean_a']:.3f}",
                transform=ax.transAxes,ha="left",va="top",fontsize=9,
                bbox=dict(facecolor="white",edgecolor="none",alpha=.93,pad=2.5))
        cb=fig.colorbar(im,cax=fig.add_subplot(gs[r,2]),extend="min")
        cb.ax.yaxis.set_major_locator(LogLocator(base=10,numticks=5))
        cb.ax.yaxis.set_major_formatter(FuncFormatter(percent))
        cb.ax.yaxis.set_minor_locator(NullLocator())
        cb.set_label("Masa por celda (%) · log",fontsize=9)
        maximum=float(np.max(np.abs(pair["delta"])))
        diff=axes[2].imshow(pair["delta"].T,origin="lower",extent=extent,
            cmap=difference_cmap,norm=TwoSlopeNorm(0,vmin=-max(maximum,1e-15),vmax=max(maximum,1e-15)),
            interpolation="nearest")
        axes[2].set_title(rf"Conditional $-$ Distribution | $\rho={pair['rho']:g}$",pad=9)
        cb=fig.colorbar(diff,cax=fig.add_subplot(gs[r,4]))
        cb.ax.yaxis.set_major_formatter(FuncFormatter(percent))
        cb.set_label("Diferencia por celda (pp)",fontsize=9)
        retained={}
        for model in ("oe","conditional"):
            mask=(g>=lo)&(g<=hi)
            retained[model]=float(pair[model]["nu"][np.ix_(mask,mask)].sum())
        if zoom and min(retained.values())<.998-1e-12:
            raise ValueError("The shared zoom must retain at least 99.8% of each joint law")
        for k,ax in enumerate(axes):
            ax.plot([lo,hi],[lo,hi],color=".55",lw=.65,alpha=.65)
            ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel="Reputación del vendedor A")
            ax.set_aspect("equal")
            if k==0:
                ax.set_ylabel("Reputación del vendedor B")
            ax.tick_params(labelsize=9)
            if k==2:
                ax.tick_params(labelleft=False)
        view.append(dict(rho=pair["rho"],limits=[lo,hi],retained_joint_probability=retained,
                         mass_vmin=norm.vmin,mass_vmax=norm.vmax,difference_limit=maximum))
    fig.suptitle(f"Distribuciones conjuntas: Conditional OE y OE distribution | {n} nodos",
                 fontsize=15,y=.985 if rows<=3 else .992)
    bottom=.032 if rows<=3 else .017
    fig.text(.5,bottom,
        "Misma escala de masa entre modelos dentro de cada fila. Verde: más masa en Distribution; azul: más masa en Conditional.",
        ha="center",fontsize=9)
    fig.text(.5,bottom-.019 if rows<=3 else bottom-.009,
        ("Zoom común: conserva al menos 99.8% de cada conjunta. " if zoom else "Dominio completo de reputaciones [0,1]. ")+
        "Escala log con límite visual 10⁻⁷; sin suavizado. Diagonal gris: reputaciones iguales.",ha="center",fontsize=8)
    paths=[]
    for extension in ("png","pdf"):
        path=directory/f"{name}.{extension}"
        fig.savefig(path,dpi=300,bbox_inches="tight",facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths,view


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name",default="refinement_101_201")
    parser.add_argument("--grid",type=int,default=101)
    parser.add_argument("--oe-retry",default="oe099_retry_n101")
    parser.add_argument("--rhos",nargs="+",type=float)
    args=parser.parse_args(argv)
    for name in (args.name,args.oe_retry):
        if Path(name).name!=name or name in ("", ".", ".."):
            parser.error("Require folder names under oblivious/07_full_comparison/data")
    campaign=RUN.OUTPUT/args.name
    manifest=json.loads((campaign/"manifest.json").read_text())
    rhos=sorted(args.rhos if args.rhos is not None else manifest["configuration"]["rhos"])
    if not rhos or len(rhos)!=len(set(rhos)):
        parser.error("Require distinct rho values")
    pairs,metrics,inputs=collect(campaign,args.grid,rhos,args.oe_retry)
    directory=ROOT/"oblivious/07_full_comparison/Figures"/args.name/f"n{args.grid:03d}"/"joint_comparison"
    directory.mkdir(parents=True,exist_ok=True)
    paths,views=[],{}
    with plt.rc_context(STYLE):
        focused=[p for p in pairs if p["rho"]>=.95]
        if focused:
            outputs,view=make_figure(focused,args.grid,directory,"01_joint_distributions_high_rho",True)
            paths.extend(outputs);views["high_rho"]=view
        outputs,view=make_figure(pairs,args.grid,directory,"02_joint_distributions_all_rho",False)
        paths.extend(outputs);views["all_rho"]=view
    with (directory/"joint_comparison_metrics.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    arrays={}
    for pair in pairs:
        tag="rho_"+str(pair["rho"]).replace(".","p")
        arrays[tag+"_grid"]=pair["grid"]
        for model in ("oe","conditional"):
            arrays[tag+"_"+model]=pair[model]["nu"]
        arrays[tag+"_difference"]=pair["delta"]
    np.savez_compressed(directory/"plotted_joint_distributions.npz",**arrays)
    RUN.atomic_json(directory/"figure_manifest.json",dict(campaign=args.name,n=args.grid,rhos=rhos,
        input_sha256=inputs,plot_code_sha256=COMMON.digest(Path(__file__)),
        validation_code_sha256={str(Path(m.__file__).relative_to(ROOT)):COMMON.digest(Path(m.__file__))
                                for m in (COMMON,RUN)},
        figures={p.name:COMMON.digest(p) for p in paths},views=views,
        data_sha256={name:COMMON.digest(directory/name) for name in
            ("joint_comparison_metrics.csv","plotted_joint_distributions.npz")},
        scope="Saved certified calendar-time joint laws of each equilibrium; no solves, simulations, smoothing or density conversion",
        orientation="Stored rows=A, columns=B; plotted horizontal=A, vertical=B, with transposed matrices",
        difference="Conditional minus Distribution; raw cell-probability differences; all cells retained in metrics",
        interpretation="Comparing different endogenous equilibria; these are joint laws, not rowwise conditional distributions; grid convergence is not established"))
    print(f"Saved {len(paths)//2} comparisons (PNG and PDF): {directory}")
    for row in metrics:
        print(f"rho={row['rho']:g}: mean reputation {row['oe_mean_a']:.4f} -> {row['conditional_mean_a']:.4f}; joint TV={row['joint_total_variation']:.4f}")


if __name__=="__main__":
    main()
