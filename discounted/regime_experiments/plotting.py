from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/regime_experiments_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


def style() -> None:
    plt.rcParams.update({"figure.facecolor":"white", "axes.facecolor":"white",
                         "axes.spines.top":False, "axes.spines.right":False,
                         "axes.grid":True, "grid.alpha":.25, "legend.frameon":False})


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def ordered_regime_representatives(params: pd.DataFrame) -> pd.DataFrame:
    order = ["I: p0<p1<p2", "II: p1<p0<p2", "III: p1<p2<p0"]
    preferred = params[params.representative].copy()
    rows = []
    for regime in order:
        candidates = preferred[preferred.regime == regime]
        if candidates.empty:
            candidates = params[params.regime == regime]
        if not candidates.empty:
            rows.append(candidates.sort_values(["p0", "parameter_id"]).iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def representative_figures(states: pd.DataFrame, diag: pd.DataFrame, params: pd.DataFrame, out: Path) -> None:
    style(); reps = ordered_regime_representatives(params)
    fig, axes = plt.subplots(1, len(reps), figsize=(5*len(reps), 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, reps.itertuples()):
        s = states[(states.parameter_id == p.parameter_id) & states.reliable_interior & (states.optimal_product == 2)]
        ax.scatter(s.n, s.m, s=8, c="#0f766e", alpha=.7,
                   label="states where product 2 is optimal", rasterized=True)
        for y, c, lab in [(p.p0,"#111827","p0"),(p.p1,"#2563eb","p1"),(p.p2,"#dc6b19","p2")]:
            ax.axhline(y, c=c, ls="--", lw=1, label=lab)
        ax.set(title=f"{p.regime}\np0={p.p0:.2f}, p1={p.p1:.2f}, p2={p.p2:.2f}", xlabel="history length n", ylabel="posterior mean m")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(.5, 1.04))
    save(fig, out/"figure1_policy_by_regime.png")

    fig, axes = plt.subplots(1, len(reps), figsize=(5*len(reps), 4.8), sharey=True)
    for ax, p in zip(np.atleast_1d(axes), reps.itertuples()):
        d = diag[(diag.parameter_id == p.parameter_id) & diag.active & diag.interval_property]
        ax.plot(d.n, d.lower_m_boundary, c="#2563eb", label="lower")
        ax.plot(d.n, d.upper_m_boundary, c="#dc6b19", label="upper")
        for y, c in [(p.p0,"#111827"),(p.p1,"#2563eb"),(p.p2,"#dc6b19")]: ax.axhline(y,c=c,ls="--",lw=.8)
        ax.set(title=f"{p.regime}\np0={p.p0:.2f}, p1={p.p1:.2f}, p2={p.p2:.2f}", xlabel="history length n", ylabel="m boundary")
    axes[-1].legend(); save(fig, out/"figure2_empirical_boundaries.png")

    fig, axes = plt.subplots(1, len(reps), figsize=(5*len(reps), 4.8), sharey=True)
    for ax, p in zip(np.atleast_1d(axes), reps.itertuples()):
        s = states[(states.parameter_id == p.parameter_id) & states.reliable_interior & (states.optimal_product == 2)]
        ax.scatter(s.n, s.m, s=5+300*s.embedded_reach_probability, c="#0f766e", alpha=.35, rasterized=True)
        ax.axhline(p.p0,c="#111827",ls="--"); ax.set(title=f"{p.regime}\np0={p.p0:.2f}, p1={p.p1:.2f}, p2={p.p2:.2f}",xlabel="n",ylabel="m")
    save(fig, out/"figure4_reachability_weighted_policy.png")


def comparative_figures(summary: pd.DataFrame, states: pd.DataFrame, out: Path) -> None:
    style(); a = summary[summary.experiment == "A_dense_p0"].sort_values("p0")
    metrics = [("incremental_value_product2","incremental value"),
               ("discounted_product2_uses","discounted product-2 uses H2(0,0)"),
               ("discounted_product2_share","discounted product-2 share"),
               ("last_active_diagonal","last active diagonal")]
    fig, axes = plt.subplots(2,2,figsize=(12,8.5))
    for ax,(col,label) in zip(axes.flat,metrics):
        ax.axvspan(a.p0.min(), .35, color="#dbeafe", alpha=.45, label="Regime I: p0<p1")
        ax.axvspan(.35, .80, color="#dcfce7", alpha=.40, label="Regime II: p1≤p0<p2")
        ax.axvspan(.80, a.p0.max(), color="#ffedd5", alpha=.45, label="Regime III: p2≤p0")
        if col == "last_active_diagonal" and "last_active_censored" in a:
            uncensored = ~a.last_active_censored
            ax.plot(a.loc[uncensored,"p0"],a.loc[uncensored,col],marker="o",ms=3,color="#1f2937",lw=1.8,label="confirmed last active diagonal")
            ax.scatter(a.loc[~uncensored,"p0"],a.loc[~uncensored,col],marker="^",s=38,facecolors="none",edgecolors="#b91c1c",label="right-censored at grid limit",zorder=4)
        else:
            ax.plot(a.p0,a[col],marker="o",ms=3,color="#1f2937",lw=1.8,label="baseline p0 sweep")
        ax.axvline(.35,ls="--",c="#2563eb",lw=1.4,label="p1=0.35")
        ax.axvline(.8,ls="--",c="#dc6b19",lw=1.4,label="p2=0.80")
        title = f"{label} as p0 varies"
        ax.set(xlabel="outside-option quality p0",ylabel=label,title=title)
    handles, labels = axes[0,0].get_legend_handles_labels()
    handles.append(Line2D([0],[0],marker="^",linestyle="None",markersize=7,
                          markerfacecolor="none",markeredgecolor="#b91c1c"))
    labels.append("last active diagonal is right-censored")
    fig.legend(handles,labels,loc="upper center",ncol=3,bbox_to_anchor=(.5,1.03))
    save(fig,out/"figure3_p0_comparative_statics.png")

    available = a[a.parameter_id.isin(states.parameter_id.unique())].drop_duplicates("parameter_id")
    fig,axes=plt.subplots(1,len(available),figsize=(5*len(available),4.8),sharex=True,sharey=True)
    for ax,p in zip(np.atleast_1d(axes),available.itertuples()):
        s=states[(states.parameter_id==p.parameter_id)&states.reliable_interior&(states.optimal_product==2)]
        bounds=s.groupby("n").z.agg(["min","max"]).reset_index()
        ax.fill_between(bounds.n,bounds["min"],bounds["max"],color="#0f766e",alpha=.22,label="product-2 z range")
        ax.plot(bounds.n,bounds["min"],color="#0f766e",lw=1.1,label="lower boundary")
        ax.plot(bounds.n,bounds["max"],color="#dc6b19",lw=1.1,label="upper boundary")
        for z in [-2,-1,0,1,2]:ax.axhline(z,c="#94a3b8",lw=.7,ls="--")
        ax.set(title=f"p0={p.p0:.2f} ({p.regime.split(':')[0]})",xlabel="history length n",ylabel="standardized distance z",ylim=(-3,3))
    axes[-1].legend(loc="upper right",fontsize=8)
    save(fig,out/"figure5_standardized_frontier_distance.png")

    fig,ax=plt.subplots(figsize=(9,5.2))
    ax.axvspan(a.p0.min(),.35,color="#dbeafe",alpha=.45,label="Regime I")
    ax.axvspan(.35,.80,color="#dcfce7",alpha=.40,label="Regime II")
    ax.axvspan(.80,a.p0.max(),color="#ffedd5",alpha=.45,label="Regime III")
    ax.plot(a.p0,a.reach_weighted_investment_center_z,color="#1f2937",marker="o",ms=4,label="reach-weighted investment center")
    ax.axhline(0,color="#b91c1c",ls="--",lw=1.2,label="outside-option frontier z=0")
    ax.axvline(.35,ls="--",c="#2563eb",lw=1);ax.axvline(.8,ls="--",c="#dc6b19",lw=1)
    ax.set(xlabel="outside-option quality p0",ylabel="reach-weighted investment center z̄_inv",
           title="Where reached product-2 states lie relative to p0")
    ax.legend(ncol=2)
    save(fig,out/"figure7_reach_weighted_investment_center.png")


def heatmaps(states: pd.DataFrame, params: pd.DataFrame, out: Path) -> None:
    reps=ordered_regime_representatives(params)
    fig,axes=plt.subplots(1,len(reps),figsize=(5*len(reps),4.5))
    for ax,p in zip(np.atleast_1d(axes),reps.itertuples()):
        s=states[(states.parameter_id==p.parameter_id)&states.reliable_interior]
        ax.scatter(s.S,s.F,c=s.optimal_product,cmap="viridis",s=5,rasterized=True)
        ax.set(title=f"{p.regime}\np0={p.p0:.2f}, p1={p.p1:.2f}, p2={p.p2:.2f}",xlabel="S",ylabel="F",aspect="equal")
    save(fig,out/"figure6_state_heatmaps.png")
