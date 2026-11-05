"""Compare the fluid monopoly and symmetric fluid point-conjecture competition.

Every plotted state is deterministic. The monopoly has one seller and the
competition model has two; customer-level and own-seller flows retain their
original, different denominators. All verified stationary branches are shown.
"""
from __future__ import annotations

from hashlib import sha256
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/fluid-oblivious-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

# Reuse the established plot style and equally spaced rho axis, without loading
# stochastic reference data or stochastic equilibrium results.
from oblivious.oe_distribution_fixed_p0.code.reporting import _style, _rho_axis

SERIES = (
    ("fluid_monopoly", "Fluid monopoly", "#0072B2", "-", "o"),
    ("fluid_competition", "Fluid competition", "#D55E00", "--", "D"),
)
MODEL_ALIASES = {"fluid": "fluid_competition", "monopoly": "fluid_monopoly"}
DEFAULT_PLOT_DIRECTORY = "overview"
PLOT_DIRECTORIES = {
    "11_": "11_policies",
    "13_": "13_reputation_trajectories",
    "14_": "14_transient_metrics",
}


def _digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _frame(data):
    return data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)


def _models(data):
    data = _frame(data)
    if "model" not in data:
        data["model"] = "fluid_competition"
    data["model"] = data.model.replace(MODEL_ALIASES)
    if not data.model.isin([entry[0] for entry in SERIES]).all():
        raise ValueError("Fluid comparison received an unsupported model")
    return data


def _normalise_fluid(summary):
    data = _models(summary)
    if data.empty:
        raise ValueError("Fluid report requires case rows, including failed searches")
    if "verified" not in data:
        data["verified"] = data.status.eq("verified")
    data["verified"] = data.verified.fillna(False).astype(bool)
    if "equilibrium_id" not in data:
        data["equilibrium_id"] = 0
    inferred = data.model.map({"fluid_monopoly": 1, "fluid_competition": 2})
    if "seller_count" in data:
        supplied = data.seller_count.dropna()
        if not (supplied.astype(int) == inferred.loc[supplied.index]).all():
            raise ValueError("Seller count differs from the fluid model")
    data["seller_count"] = inferred
    if "product2_conditional" not in data:
        data["product2_conditional"] = data.get("alpha", np.nan)
    if "x_star" not in data:
        data["x_star"] = np.nan
    # Compatibility for generic tables; both are deterministic stationary states.
    data["mean_reputation"] = data.x_star
    if "sale_rate" not in data:
        data["sale_rate"] = np.nan
    if "product2_calendar" not in data:
        data["product2_calendar"] = data.sale_rate*data.product2_conditional
    if "outside_rate" not in data:
        data["outside_rate"] = 1-data.seller_count*data.sale_rate
    expected_customer = data.seller_count*data.product2_calendar
    if "customer_product2_calendar" not in data:
        data["customer_product2_calendar"] = expected_customer
    accepted = data[data.verified]
    if len(accepted):
        errors = [
            abs(accepted.product2_calendar-accepted.sale_rate*accepted.product2_conditional),
            abs(accepted.outside_rate+accepted.seller_count*accepted.sale_rate-1),
            abs(accepted.customer_product2_calendar-accepted.seller_count*accepted.product2_calendar),
        ]
        if any(error.max() > 2e-8 for error in errors):
            raise ValueError("Fluid stationary market/product accounting failed")
        for name in ("product2_conditional", "product2_calendar", "customer_product2_calendar", "sale_rate", "outside_rate"):
            if not accepted[name].between(-1e-10, 1+1e-10).all():
                raise ValueError("Invalid fluid probability: " + name)
    return data


def _raw_policy(policies, candidate):
    chosen = policies[(policies.model == candidate["model"]) &
                      np.isclose(policies.p0, candidate["p0"]) &
                      np.isclose(policies.rho, candidate["rho"]) &
                      policies.equilibrium_id.eq(candidate["equilibrium_id"])]
    if "holding" in chosen:
        chosen = chosen[~chosen.holding.fillna(False).astype(bool)]
    return chosen.sort_values("reputation").drop_duplicates("reputation")


def _policy_distances(a, b):
    if a.empty or b.empty:
        return {}
    xa, xb = a.reputation.to_numpy(), b.reputation.to_numpy()
    edges = np.unique(np.r_[0., (xa[:-1]+xa[1:])/2, (xb[:-1]+xb[1:])/2, 1.])
    mid = (edges[:-1]+edges[1:])/2
    ia = np.searchsorted((xa[:-1]+xa[1:])/2, mid)
    ib = np.searchsorted((xb[:-1]+xb[1:])/2, mid)
    gap = abs(a.product2_probability.to_numpy()[ia]-b.product2_probability.to_numpy()[ib])
    return dict(policy_l1_length=float(gap@np.diff(edges)),
                policy_mismatch_fraction_of_state_interval=float((gap > 1e-6)@np.diff(edges)))


def quantitative_comparisons(summary, policies):
    """Compare every competition/monopoly stationary branch pair explicitly.

    No branch is selected arbitrarily. Policy differences are Lebesgue lengths
    on [0,1], not visitation-weighted or welfare comparisons. Isolated holding
    mixtures have zero mass in these interval metrics.
    """
    data, policies = _normalise_fluid(summary), _models(policies)
    rows = []
    metrics = ("x_star", "sale_rate", "outside_rate", "product2_conditional",
               "product2_calendar", "customer_product2_calendar", "profit_calendar", "value_initial")
    for (p0, rho), cases in data.groupby(["p0", "rho"]):
        competition = cases[(cases.model == "fluid_competition") & cases.verified]
        monopoly = cases[(cases.model == "fluid_monopoly") & cases.verified]
        if competition.empty or monopoly.empty:
            rows.append(dict(p0=p0, rho=rho, comparison_status="missing_verified_stationary_branch",
                             competition_branch_count=len(competition), monopoly_branch_count=len(monopoly)))
            continue
        for c in competition.to_dict("records"):
            for m in monopoly.to_dict("records"):
                row = dict(p0=p0, rho=rho, comparison_status="verified_pair",
                           competition_equilibrium_id=c["equilibrium_id"],
                           monopoly_equilibrium_id=m["equilibrium_id"],
                           competition_branch_count=len(competition), monopoly_branch_count=len(monopoly))
                for name in metrics:
                    if name in c and name in m:
                        row["competition_"+name] = c[name]
                        row["monopoly_"+name] = m[name]
                        row["competition_minus_monopoly_"+name] = c[name]-m[name]
                row.update(_policy_distances(_raw_policy(policies,c), _raw_policy(policies,m)))
                rows.append(row)
    return pd.DataFrame(rows)


def _subplots(count, *, sharey=True, height=3.05):
    columns = min(3, max(1,count))
    rows = math.ceil(max(1,count)/columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.35*columns,height*rows+1.05),
                             sharey=sharey,squeeze=False,layout="constrained")
    active = list(axes.flat)[:count]
    for ax in list(axes.flat)[count:]:
        ax.set_visible(False)
    return fig, active


def _legend(fig, axes):
    handles = {}
    for ax in axes:
        artists, labels = ax.get_legend_handles_labels()
        for artist,label in zip(artists,labels,strict=True):
            handles.setdefault(label,artist)
    if handles:
        fig.legend(handles.values(),handles.keys(),loc="outside lower center",ncol=2,frameon=False,fontsize=9)


def _curves(ax, data, *, horizontal, positions, metric):
    for model,label,color,linestyle,marker in SERIES:
        selected = data[(data.model == model)&data.verified]
        grouped = {float(v):group for v,group in selected.groupby(horizontal)}
        values = [float(grouped[v].iloc[0][metric]) if v in grouped and len(grouped[v]) == 1 else np.nan
                  for v in positions]
        xx = [positions[v] for v in positions]
        ax.plot(xx,values,color=color,ls=linestyle,lw=1.6,marker=marker,ms=5,
                markerfacecolor="none",label=label)
        for value,group in grouped.items():
            if len(group)>1 and value in positions:
                ax.scatter([positions[value]]*len(group),group[metric],edgecolors=color,
                           marker=marker,facecolors="none",s=32,zorder=4)


def _metric_figure(data,p0s,rhos,*,horizontal,metric,title,ylabel,fixed_limits=True):
    facets = rhos if horizontal == "p0" else p0s
    fig,axes = _subplots(len(facets),sharey=fixed_limits)
    for ax,value in zip(axes,facets,strict=True):
        fixed = "rho" if horizontal == "p0" else "p0"
        chosen = data[np.isclose(data[fixed],value)]
        values = p0s if horizontal == "p0" else rhos
        positions = {float(v):float(v) if horizontal == "p0" else i for i,v in enumerate(values)}
        _curves(ax,chosen,horizontal=horizontal,positions=positions,metric=metric)
        ax.set_title(rf"$\rho={value:g}$" if fixed == "rho" else rf"$p_0={value:g}$",loc="left")
        ax.set_ylabel(ylabel)
        if horizontal == "p0":
            ax.set_xlabel(r"Outside option $p_0$")
            ax.set_xticks(p0s,[f"{v:g}" for v in p0s])
        else:
            _rho_axis(ax,rhos)
            for tick in ax.get_xticklabels():
                tick.set_rotation(30); tick.set_horizontalalignment("right")
        if metric != "x_star":
            ax.yaxis.set_major_formatter(PercentFormatter(1))
        if fixed_limits:
            ax.set_ylim(-.025,1.025)
            ax.set_yticks(np.arange(0,1.01,.2))
        else:
            ax.set_ylim(bottom=-.03*max(ax.get_ylim()[1],.001))
        _style(ax)
    _legend(fig,axes)
    fig.suptitle(title+"\nBoth models are fluid; every verified stationary branch is retained.",fontsize=12)
    return fig


def _status_figure(data,p0s,rhos):
    fig,axes = plt.subplots(1,2,figsize=(12.8,max(4.,len(p0s)*.55+1.6)),layout="constrained")
    colors = {"pure":"#D7E6F2","mixed":"#DCCBE5","multiple":"#A88CC2",
              "no_equilibrium_found":"#E6E6E6","error":"#F0D4CF"}
    for ax,(model,label,*_) in zip(axes,SERIES,strict=True):
        for row,p0 in enumerate(p0s):
            for column,rho in enumerate(rhos):
                part = data[(data.model == model)&np.isclose(data.p0,p0)&np.isclose(data.rho,rho)]
                accepted = part[part.verified]
                if len(accepted)>1:
                    kind,text="multiple",f"{len(accepted)} states"
                elif len(accepted)==1:
                    alpha=float(accepted.iloc[0].product2_conditional)
                    kind="mixed" if 1e-8<alpha<1-1e-8 else "pure"
                    text="mixed" if kind=="mixed" else f"pure {2 if alpha>.5 else 1}"
                else:
                    kind="no_equilibrium_found" if len(part) and part.status.eq("no_equilibrium_found").all() else "error"
                    text="none found" if kind=="no_equilibrium_found" else "unverified"
                ax.add_patch(plt.Rectangle((column-.46,row-.4),.92,.8,facecolor=colors[kind],edgecolor="none"))
                ax.text(column,row,text,ha="center",va="center",fontsize=9)
        ax.set_xticks(range(len(rhos)),[f"{v:g}" for v in rhos])
        ax.set_yticks(range(len(p0s)),[f"{v:g}" for v in p0s])
        ax.set_xlim(-.6,len(rhos)-.4); ax.set_ylim(len(p0s)-.5,-.6)
        ax.set(xlabel=r"Persistence $\rho$",ylabel=r"Outside option $p_0$",title=label)
        ax.spines[:].set_visible(False); ax.tick_params(length=0)
    fig.suptitle("Verified optimal stationary states\nMonopoly optima and symmetric competition equilibria; numerical search outcomes.",fontsize=12)
    return fig


def _policy_figure(summary,policies,*,p0,rhos):
    fig,axes = _subplots(len(rhos))
    styles=("-","--",":","-.")
    for ax,rho in zip(axes,rhos,strict=True):
        for model,label,color,base,_ in SERIES:
            accepted = summary[(summary.model == model)&summary.verified&np.isclose(summary.p0,p0)&np.isclose(summary.rho,rho)]
            states=accepted.to_dict("records")
            # Monopoly solves one HJB/policy per parameter case. Several rest
            # points of that policy are markers, not separate strategy curves.
            strategies=states[:1] if model=="fluid_monopoly" else states
            for j,row in enumerate(strategies):
                samples=_raw_policy(policies,row)
                if samples.empty:
                    continue
                single_policy=model=="fluid_monopoly" or len(states)==1
                description=label if single_policy else rf"{label}, $x^*={row['x_star']:.3f}$"
                linestyle=base if single_policy else styles[j%len(styles)]
                ax.step(samples.reputation,samples.product2_probability,where="mid",color=color,
                        ls=linestyle,lw=1.5,alpha=.85,label=description)
                holding = policies[(policies.model == model)&np.isclose(policies.p0,p0)&np.isclose(policies.rho,rho)&policies.equilibrium_id.eq(row["equilibrium_id"])]
                if "holding" in holding:
                    holding=holding[holding.holding.fillna(False).astype(bool)]
                    ax.scatter(holding.reputation,holding.product2_probability,color=color,s=13,zorder=4)
            for row in states:
                ax.scatter([row["x_star"]],[row["product2_conditional"]],marker="D",s=30,
                           facecolors="none",edgecolors=color,zorder=5)
        ax.set(title=rf"$\rho={rho:g}$",xlabel="Own normalized reputation x",ylabel=r"$\sigma(x)$: product 2 probability",
               xlim=(0,1),ylim=(-.03,1.04))
        ax.yaxis.set_major_formatter(PercentFormatter(1)); _style(ax)
    _legend(fig,axes)
    fig.suptitle(rf"Fluid optimal policies | outside option $p_0={p0:g}$"+
                 "\nDiamonds: stationary product mix; small filled dots: isolated holding mixtures.",fontsize=12)
    return fig


def _trajectory_figures(trajectories,bins,p0s,rhos):
    figures=[]
    traces=_models(trajectories) if trajectories is not None else pd.DataFrame()
    block=_models(bins) if bins is not None else pd.DataFrame()
    styles=("-","--",":","-.")
    metrics=(
        ("own_sale_rate","market_share","Own-seller demand over time","Selected seller (% of calendar time)"),
        ("product2_given_sale","primary_product2_given_sale","Primary: product 2 among own sales","Product 2 (% of own sales)"),
        ("own_product2_rate","seller_product2_per_period","Own seller selected and uses product 2","One seller & product 2 (% of calendar time)"),
        ("customer_product2_rate","secondary_customer_product2","Secondary: customer receives product 2","Any seller & product 2 (% of calendar time)"),
    )
    for rho in rhos:
        tag=str(rho).replace(".","p")
        if not traces.empty:
            fig,axes=_subplots(len(p0s))
            for ax,p0 in zip(axes,p0s,strict=True):
                for model,label,color,base,_ in SERIES:
                    part=traces[(traces.model==model)&np.isclose(traces.p0,p0)&np.isclose(traces.rho,rho)]
                    for j,(identity,group) in enumerate(part.groupby("equilibrium_id")):
                        group=group.sort_values("time")
                        description=label if part.equilibrium_id.nunique()==1 else f"{label}, branch {identity}"
                        ax.plot(group.time,group.x_a,color=color,ls=base if j==0 else styles[(j+1)%len(styles)],lw=1.6,label=description)
                        if model=="fluid_competition" and "x_b" in group and not np.allclose(group.x_a,group.x_b,atol=1e-10):
                            ax.plot(group.time,group.x_b,color=color,ls=":",lw=1,label=description+", seller B")
                ax.set(title=rf"$p_0={p0:g}$",xlabel="Continuous calendar time t",ylabel="Reputation x(t)",ylim=(-.02,1.02))
                _style(ax)
            _legend(fig,axes)
            fig.suptitle(rf"Fluid reputation trajectories | $\rho={rho:g}$"+
                         "\nCommon mature start x(0)=0.5; competition sellers coincide by symmetry.",fontsize=12)
            figures.append(("13_deterministic_reputations_rho_"+tag,fig))
        if block.empty:
            continue
        for metric,stem,title,ylabel in metrics:
            fig,axes=_subplots(len(p0s))
            for ax,p0 in zip(axes,p0s,strict=True):
                for model,label,color,base,_ in SERIES:
                    part=block[(block.model==model)&np.isclose(block.p0,p0)&np.isclose(block.rho,rho)]
                    for j,(identity,group) in enumerate(part.groupby("equilibrium_id")):
                        group=group.sort_values("bin_start")
                        center=group.bin_center if "bin_center" in group else (group.bin_start+group.bin_stop)/2
                        description=label if part.equilibrium_id.nunique()==1 else f"{label}, branch {identity}"
                        ax.plot(center,group[metric],color=color,ls=base if j==0 else styles[(j+1)%len(styles)],lw=1.5,label=description)
                ax.set(title=rf"$p_0={p0:g}$",xlabel="Continuous calendar time t",ylabel=ylabel,ylim=(-.025,1.025))
                ax.yaxis.set_major_formatter(PercentFormatter(1)); _style(ax)
            _legend(fig,axes)
            fig.suptitle(title+rf" | $\rho={rho:g}$"+
                         "\nBoth models fluid; common mature start x=0.5; 25-unit blocks; conditional use is sales weighted.",fontsize=12)
            figures.append(("14_transient_"+stem+"_rho_"+tag,fig))
    return figures


def make_figures(summary,policies,output_dir,*,p0s,rhos,trajectories=None,trajectory_bins=None):
    """Render a fluid-versus-fluid study from supplied tables only."""
    p0s,rhos=list(map(float,p0s)),list(map(float,rhos))
    data,policies=_normalise_fluid(summary),_models(policies)
    expected={(model,p0,rho) for model,*_ in SERIES for p0 in p0s for rho in rhos}
    actual=set(data[["model","p0","rho"]].itertuples(index=False,name=None))
    if actual!=expected:
        raise ValueError("Comparison tables do not cover exactly the requested fluid model grid")
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=True)
    tables=root.parent/"data";tables.mkdir(parents=True,exist_ok=True)
    table_paths=[tables/"comparison_summary.csv",tables/"comparison_policies.csv",tables/"fluid_vs_fluid_differences.csv"]
    data.to_csv(table_paths[0],index=False)
    policies.to_csv(table_paths[1],index=False)
    quantitative_comparisons(data,policies).to_csv(table_paths[2],index=False)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9.5,"figure.max_open_warning":0,
                         "axes.titlesize":11,"legend.frameon":False})
    figures=[]
    for number,(metric,stem,title,ylabel,limits) in enumerate((
        ("product2_conditional","primary_product2_given_sale","Primary: product 2 among own sales","Product 2 (% of own sales)",True),
        ("customer_product2_calendar","secondary_customer_product2","Secondary: product 2 received by the customer","Any seller & product 2\n(% of calendar time)",False),
        ("product2_calendar","seller_product2_per_period","Own seller selected and uses product 2","One seller & product 2\n(% of calendar time)",False),
    )):
        for offset,horizontal in enumerate(("p0","rho"),1):
            if metric == "product2_calendar" and horizontal == "p0":
                continue
            figures.append((f"{2*number+offset:02d}_{stem}_by_{horizontal}",
                _metric_figure(data,p0s,rhos,horizontal=horizontal,metric=metric,title=title,ylabel=ylabel,fixed_limits=limits)))
    for metric,stem,title,ylabel in (
        ("x_star","07_equilibrium_reputations","Fluid stationary reputations","Normalized stationary reputation x*"),
        ("sale_rate","08_market_shares","Own-seller market share","One seller selected\n(% of calendar time)"),
        ("outside_rate","09_outside_option_share","Outside-option market share","Outside selected\n(% of calendar time)"),
    ):
        figures.append((stem,_metric_figure(data,p0s,rhos,horizontal="rho",metric=metric,title=title,ylabel=ylabel)))
    figures.append(("10_equilibrium_status",_status_figure(data,p0s,rhos)))
    for p0 in p0s:
        figures.append(("11_policies_p0_"+str(p0).replace(".","p"),_policy_figure(data,policies,p0=p0,rhos=rhos)))
    figures.extend(_trajectory_figures(trajectories,trajectory_bins,p0s,rhos))
    overview=root/DEFAULT_PLOT_DIRECTORY;overview.mkdir(parents=True,exist_ok=True)
    paths=[];pdf_path=overview/"fluid_monopoly_vs_competition.pdf"
    with PdfPages(pdf_path) as pdf:
        for name,fig in figures:
            folder = next((directory for prefix,directory in PLOT_DIRECTORIES.items()
                           if name.startswith(prefix)), DEFAULT_PLOT_DIRECTORY)
            destination = root/folder
            destination.mkdir(parents=True,exist_ok=True)
            for extension in ("png","pdf"):
                path=destination/(name+"."+extension)
                fig.savefig(path,dpi=175,bbox_inches="tight");paths.append(path)
            pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
    paths.append(pdf_path)
    report=dict(status="complete",figure_count=len(figures),p0s=p0s,rhos=rhos,
                plot_directories=PLOT_DIRECTORIES,default_plot_directory=DEFAULT_PLOT_DIRECTORY,
                excluded_plots=["05_seller_product2_per_period_by_p0"],
                comparison="Fluid monopoly versus symmetric fluid point-conjecture competition",
                models=[dict(model=model,label=label,seller_count=1 if model=="fluid_monopoly" else 2) for model,label,*_ in SERIES],
                primary="P(product 2 | own sale) = alpha at each fluid stationary state",
                seller_calendar="Own seller selected AND product 2: D * alpha",
                secondary="Customer receives product 2: N * D * alpha, with N=1 monopoly and N=2 competition",
                reputation="Both models have deterministic stationary reputations; no stochastic invariant law or Monte Carlo band is used",
                multiple_states="All verified stationary states shown; quantitative comparisons enumerate every branch pair",
                policy_difference="Lebesgue length on [0,1], excluding isolated holding mixtures; not visitation weighted or welfare",
                trajectory="Common mature start .5; competition uses actual joint demand and monopoly uses its own Thompson demand",
                reporting_source_sha256=_digest(Path(__file__)),
                generated_data_sha256={str(path.relative_to(root.parent)):_digest(path) for path in table_paths},
                generated_file_sha256={str(path.relative_to(root)):_digest(path) for path in paths})
    (root/"report_manifest.json").write_text(json.dumps(report,indent=2)+"\n")
    return paths,report
