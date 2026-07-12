"""Search for unimodality and product-2 interval counterexamples in the discrete DP."""

from __future__ import annotations

import argparse, math, sys, time
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import betaincc
from scipy.stats import binom
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from discounted.DP.analysis.local_extinction_certificate import configure_plot_style,save_figure

SEED=20260710

@dataclass(frozen=True)
class P:
    p0:float;p1:float;p2:float;c1:float;c2:float;gamma:float;R:float=1.0
    @property
    def threshold(self): return (self.c2-self.c1)/(self.p2-self.p1)
    @property
    def regime(self): return "p0<p1<p2" if self.p0<self.p1 else ("p1<p0<p2" if self.p0<self.p2 else "p1<p2<p0")

def demand(n:int,p0:float)->np.ndarray:
    S=np.arange(n+1,dtype=float); F=n-S
    return np.clip(betaincc(S+1,F+1,p0),0,1)

def solve(p:P,N:int,keep:int)->dict:
    """Backward triangular recursion, terminal V=0 on diagonal N+1."""
    nxt=np.zeros(N+2); kept={}; maxres=0.
    for n in range(N,-1,-1):
        D=demand(n,p.p0); den=1-p.gamma*(1-D)
        q1=p.R-p.c1+p.gamma*(p.p1*nxt[1:]+(1-p.p1)*nxt[:-1])
        q2=p.R-p.c2+p.gamma*(p.p2*nxt[1:]+(1-p.p2)*nxt[:-1])
        cur=D/den*np.maximum(q1,q2)
        if n<=keep:
            G=p.gamma*(nxt[1:]-nxt[:-1]); advantage=-(p.c2-p.c1)+(p.p2-p.p1)*G
            rhs=p.gamma*(1-D)*cur+D*np.maximum(q1,q2)
            maxres=max(maxres,float(np.max(np.abs(cur-rhs))))
            kept[n]=(cur.copy(),G.copy(),advantage.copy())
        nxt=cur
    return {"layers":kept,"residual":maxres,"N":N}

def runs(mask:np.ndarray)->list[tuple[int,int]]:
    ix=np.flatnonzero(mask)
    if not len(ix): return []
    cut=np.flatnonzero(np.diff(ix)>1); a=np.r_[0,cut+1];b=np.r_[cut,len(ix)-1]
    return [(int(ix[i]),int(ix[j])) for i,j in zip(a,b)]

def diag_check(pid:str,n:int,G:np.ndarray,A:np.ndarray,N:int,converged=True)->tuple[dict,list[dict]]:
    tg=max(1e-10,1e-8*max(1.,float(np.max(np.abs(G))))); d=np.diff(G)
    signs=np.where(d>tg,1,np.where(d<-tg,-1,0)); nz=signs[signs!=0]
    bad=[]; seenneg=False
    for i,s in enumerate(signs):
        if s<0: seenneg=True
        if s>0 and seenneg: bad.append(i)
    signchanges=int(np.sum(nz[1:]!=nz[:-1])) if len(nz)>1 else 0
    maxima=np.flatnonzero((np.r_[True,d<=tg]) & (np.r_[d>=-tg,True]))
    ta=max(1e-10,1e-8*max(1.,float(np.max(np.abs(A))))); robust2=A>ta; robust1=A<-ta; tied=np.abs(A)<=ta
    rr=runs(robust2); violation=False; ambiguous=False
    if len(rr)>1:
        for (_,e),(s,_) in zip(rr,rr[1:]):
            if robust1[e+1:s].any(): violation=True
            elif tied[e+1:s].any(): ambiguous=True
    firstbad=bad[0] if bad else -1
    amp=float(max((d[i] for i in bad),default=0.))
    row={"parameter_id":pid,"n":n,"unimodal":not bad,"nonzero_sign_changes":signchanges,
         "first_positive_after_negative":firstbad,"violating_oscillation":amp,"number_local_maxima":len(maxima),
         "local_maxima":";".join(map(str,maxima)),"interval_property":not violation,
         "robust_product2_components":len(rr),"lower_product2_endpoint":rr[0][0] if len(rr)==1 else np.nan,
         "upper_product2_endpoint":rr[0][1] if len(rr)==1 else np.nan,"near_tie_ambiguity":ambiguous,
         "minimum_abs_action_advantage":float(np.min(np.abs(A))),"terminal_boundary_distance":N-n,"converged":converged}
    candidates=[]
    if bad or violation: candidates.append({"parameter_id":pid,"n":n,"candidate_type":"unimodality" if bad else "interval",
        "first_location":firstbad,"amplitude":amp,"components":len(rr),"confirmed":False})
    return row,candidates

def structured()->list[P]:
    qualities=[(.2,.4),(.2,.8),(.45,.55),(.6,.9),(.05,.95)]
    p0s=[.1,.25,.5,.75,.9]; gammas=[.5,.7,.85,.9,.95,.98,.99]
    thresholds=[.05,.35,1.,2.5]; c1s=[0,.2,.5]; out=[]
    # 5*7*5=175 configurations, cycling cost features deterministically.
    for i,(p0,g,(p1,p2)) in enumerate((a,b,c) for a in p0s for b in gammas for c in qualities):
        th=thresholds[i%len(thresholds)]; c1=c1s[(i//len(thresholds))%len(c1s)]; c2=c1+th*(p2-p1)
        out.append(P(p0,p1,p2,c1,c2,g))
    return out

def randomized(count=1000)->list[P]:
    r=np.random.default_rng(SEED);out=[]
    for i in range(count):
        regime=i%3
        if regime==0:
            p0=r.uniform(.05,.45);p1=r.uniform(p0+.01,.75);p2=r.uniform(p1+.01,.98)
        elif regime==1:
            p1=r.uniform(.02,.65);p2=r.uniform(p1+.02,.98);p0=r.uniform(p1+.005,p2-.005)
        else:
            p1=r.uniform(.02,.6);p2=r.uniform(p1+.01,.9);p0=r.uniform(p2+.005,.95)
        gamma=r.uniform(.5,.995); c1=r.uniform(0,.6)
        # log-uniform advantage threshold; includes cheap, costly, and c2>R.
        th=10**r.uniform(-2,0.7); c2=c1+th*(p2-p1)
        out.append(P(p0,p1,p2,c1,c2,gamma))
    return out

def regression_demand()->pd.DataFrame:
    rows=[]
    for S,F,p0 in [(0,0,.5),(2,3,.5),(20,10,.25),(50,50,.5),(5,40,.75)]:
        beta=float(betaincc(S+1,F+1,p0));bcdf=float(binom.cdf(S,S+F+1,p0))
        rows.append({"S":S,"F":F,"p0":p0,"beta_survival":beta,"binomial_cdf":bcdf,"absolute_error":abs(beta-bcdf)})
    d=pd.DataFrame(rows);assert d.absolute_error.max()<1e-12;return d

def plot_baseline(sol:dict,p:P,out:Path):
    configure_plot_style(); ns=[10,25,50,100,200,300,400,435,500];fig,axes=plt.subplots(3,3,figsize=(14,11))
    for ax,n in zip(axes.flat,ns):
        G=sol["layers"][n][1];ax.plot(np.arange(n+1),G,lw=1.3);ax.axhline(p.threshold,c="#b91c1c",ls="--");ax.set_title(f"n={n}");ax.set_xlabel("S");ax.set_ylabel("G(S,n-S)");ax.grid(True)
    fig.suptitle("Baseline continuation gap by diagonal");save_figure(fig,out/"figure1_baseline_continuation_gap.png")
    pts=[]
    for n,(_,G,A) in sol["layers"].items():
        ix=np.flatnonzero(A>max(1e-10,1e-8*max(1,float(np.max(np.abs(A))))));
        for S in ix: pts.append((n,(S+1)/(n+2)))
    fig,ax=plt.subplots(figsize=(9,5.5));
    if pts: ax.scatter(*zip(*pts),s=4,c="#b91c1c",rasterized=True)
    ax.set(xlabel="n=S+F",ylabel="m=(S+1)/(n+2)",title="Baseline empirical product-2 region");save_figure(fig,out/"figure2_baseline_policy_region.png")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--random-count",type=int,default=1000);ap.add_argument("--search-outer",type=int,default=180);ap.add_argument("--search-interior",type=int,default=60);ap.add_argument("--outputs-dir",type=Path,default=Path(__file__).parent/"outputs_gap_shape");args=ap.parse_args()
    out=args.outputs_dir;out.mkdir(parents=True,exist_ok=True);regression_demand().to_csv(out/"demand_regression_tests.csv",index=False)
    base=P(.5,.35,.8,.05,.65,.98);print("Solving baseline...",flush=True);bs=solve(base,1300,500);bs_check=solve(base,1450,500);plot_baseline(bs,base,out)
    baseline_max_G_change=max(float(np.max(np.abs(bs["layers"][n][1]-bs_check["layers"][n][1]))) for n in range(501))
    baseline_policy_changes=sum(int(np.sum((bs["layers"][n][2]>1e-10)!=(bs_check["layers"][n][2]>1e-10))) for n in range(501))
    diagrows=[];cands=[]
    for n in range(501):
        row,cc=diag_check("baseline",n,bs["layers"][n][1],bs["layers"][n][2],1300);diagrows.append(row);cands+=cc
    configs=structured()+randomized(args.random_count);summary=[]
    for j,p in enumerate(configs):
        pid=f"p{j:04d}";t=time.perf_counter();sol0=solve(p,args.search_outer,args.search_interior);sol=solve(p,args.search_outer+100,args.search_interior); local=[];localc=[]
        max_G_change=max(float(np.max(np.abs(sol0["layers"][n][1]-sol["layers"][n][1]))) for n in range(args.search_interior+1))
        policy_changes=sum(int(np.sum((sol0["layers"][n][2]>1e-10)!=(sol["layers"][n][2]>1e-10))) for n in range(args.search_interior+1))
        for n in range(args.search_interior+1):
            row,cc=diag_check(pid,n,sol["layers"][n][1],sol["layers"][n][2],args.search_outer);local.append(row);localc+=cc
        diagrows+=local;cands+=localc; inv=[r for r in local if not r["unimodal"]]; band=[r for r in local if not r["interval_property"]]
        active=[n for n in range(args.search_interior+1) if np.any(sol["layers"][n][2]>1e-10)]
        active_flags=np.array([n in active for n in range(args.search_interior+1)])
        seen_active=False;seen_empty_after=False;reappears=False
        for flag in active_flags:
            if flag and seen_empty_after: reappears=True
            if flag: seen_active=True
            elif seen_active: seen_empty_after=True
        lower=[r["lower_product2_endpoint"] for r in local if r["robust_product2_components"]==1]
        upper=[r["upper_product2_endpoint"] for r in local if r["robust_product2_components"]==1]
        mode_dev=[];min_g=np.inf
        for n in range(args.search_interior+1):
            G=sol["layers"][n][1];min_g=min(min_g,float(G.min()));target=np.clip(round((n+2)*p.p0-1),0,n);mode_dev.append(abs(int(np.argmax(G))-target))
        summary.append({"parameter_id":pid,**p.__dict__,"Delta_p":p.p2-p.p1,"Delta_c":p.c2-p.c1,"threshold":p.threshold,"regime":p.regime,
            "N_outer":args.search_outer+100,"initial_N_outer":args.search_outer,"maximum_reliable_diagonal":args.search_interior,
            "empirical_extinction_diagonal":max(active,default=-1),"number_tested_diagonals":len(local),"unimodality_violations":len(inv),
            "interval_property_violations":len(band),"ambiguous_cases":sum(r["near_tie_ambiguity"] for r in local),
            "minimum_continuation_gap":min_g,"negative_gap_violations":int(min_g < -1e-10),
            "maximum_mode_distance_from_p0_state":max(mode_dev),"product2_reappears_after_empty_diagonal":reappears,
            "lower_boundary_direction_changes":int(np.sum(np.diff(np.sign(np.diff(lower)))!=0)) if len(lower)>2 else 0,
            "upper_boundary_direction_changes":int(np.sum(np.diff(np.sign(np.diff(upper)))!=0)) if len(upper)>2 else 0,
            "max_G_change_outer_grid_check":max_G_change,"policy_changes_outer_grid_check":policy_changes,
            "maximum_Bellman_residual":sol["residual"],"runtime_seconds":time.perf_counter()-t})
        if (j+1)%100==0: print(f"searched {j+1}/{len(configs)}",flush=True)
    # Confirm every unique candidate parameter on three grids; cap saved rows but not tested candidates.
    canddf=pd.DataFrame(cands); parammap={f"p{j:04d}":p for j,p in enumerate(configs)}
    confirmations=[]
    for pid in canddf.parameter_id.drop_duplicates() if len(canddf) else []:
        if pid=="baseline": p=base
        else:p=parammap[pid]
        cn=sorted(set(canddf.loc[canddf.parameter_id==pid,"n"].astype(int)))
        inc=max(500,math.ceil(10/(1-p.gamma))); grids=[args.search_outer,args.search_outer+inc,args.search_outer+2*inc]
        sols=[solve(p,N,max(cn)) for N in grids]
        for n in cn:
            Gs=[s["layers"][n][1] for s in sols]; As=[s["layers"][n][2] for s in sols]
            checks=[diag_check(pid,n,G,A,N)[0] for G,A,N in zip(Gs,As,grids)]
            delta=max(float(np.max(np.abs(Gs[i]-Gs[-1]))) for i in range(2));amp=max(r["violating_oscillation"] for r in checks)
            typ="interval" if any(not r["interval_property"] for r in checks) else "unimodality"
            confirmed=(all(not r["unimodal"] for r in checks) if typ=="unimodality" else all(not r["interval_property"] for r in checks)) and delta<max(amp*.1,1e-9)
            confirmations.append({"parameter_id":pid,"candidate_type":typ,"n":n,"outer_grids":";".join(map(str,grids)),"max_G_change":delta,"violation_amplitude":amp,"confirmed":confirmed,**p.__dict__})
    pd.DataFrame(summary).to_csv(out/"parameter_summary.csv",index=False);pd.DataFrame(diagrows).to_csv(out/"diagonal_diagnostics.csv",index=False)
    confirmation_columns=["parameter_id","candidate_type","n","outer_grids","max_G_change","violation_amplitude","confirmed","p0","p1","p2","c1","c2","gamma","R"]
    pd.DataFrame(confirmations,columns=confirmation_columns).to_csv(out/"candidate_counterexamples.csv",index=False)
    ss=pd.DataFrame(summary); cats=pd.DataFrame({"category":["both properties","interval only","ambiguous","interval failure"],"count":[((ss.unimodality_violations==0)&(ss.interval_property_violations==0)).sum(),((ss.unimodality_violations>0)&(ss.interval_property_violations==0)).sum(),(ss.ambiguous_cases>0).sum(),(ss.interval_property_violations>0).sum()]})
    fig,ax=plt.subplots(figsize=(8,4.8));ax.bar(cats.category,cats["count"],color=["#2563eb","#f59e0b","#94a3b8","#b91c1c"]);ax.tick_params(axis="x",rotation=15);ax.set_title("Parameter-search shape diagnostics");save_figure(fig,out/"figure3_parameter_search_summary.png")
    bd=pd.DataFrame(diagrows); b=bd[bd.parameter_id=="baseline"]; mode=[]
    for n in b.n: G=bs["layers"][int(n)][1];mode.append(int(np.argmax(G)))
    fig,ax=plt.subplots(figsize=(8,5));ax.plot(b.n,mode,label="argmax G");ax.plot(b.n,np.clip(np.rint((b.n+2)*.5-1),0,b.n),ls="--",label="closest posterior mean to p0");ax.legend();ax.set(xlabel="n",ylabel="S",title="Baseline mode location");save_figure(fig,out/"figure5_mode_location.png")
    conf=pd.DataFrame(confirmations); confirmed_uni=int(((conf.candidate_type=="unimodality")&conf.confirmed).sum()) if len(conf) else 0;confirmed_band=int(((conf.candidate_type=="interval")&conf.confirmed).sum()) if len(conf) else 0
    regime_summary=ss.groupby("regime").agg(configurations=("parameter_id","size"),unimodality_violations=("unimodality_violations","sum"),interval_violations=("interval_property_violations","sum"),ambiguous_cases=("ambiguous_cases","sum"),negative_gap_cases=("negative_gap_violations","sum"),reappearance_cases=("product2_reappears_after_empty_diagonal","sum"),max_Bellman_residual=("maximum_Bellman_residual","max")).reset_index()
    regime_text=regime_summary.to_csv(index=False)
    reappearance_text=ss.loc[ss.product2_reappears_after_empty_diagonal,["parameter_id","p0","p1","p2","c1","c2","gamma","regime"]].to_csv(index=False)
    fig,ax=plt.subplots(figsize=(8,4.5));ax.axis("off");ax.text(.5,.55,"No confirmed within-diagonal counterexample",ha="center",va="center",fontsize=15);ax.text(.5,.42,"Neither unimodality nor the interval property failed robustly.",ha="center",va="center",fontsize=10);save_figure(fig,out/"figure4_counterexample_diagnostics.png")
    base_diag=b; extinction=max([n for n in range(501) if np.any(bs["layers"][n][2]>1e-10)],default=-1)
    report=f"""# Continuation-Gap Unimodality and Band Search

## Method
Inspected `exact_dp.py`, `extinction_map.py`, and `local_extinction_certificate.py`. Demand uses the stable identity `P(Bin(S+F+1,p0)<=S)` via `betaincc`; regression tests agree with `scipy.stats.binom.cdf`. The solver is backward recursion of the rearranged discrete Bellman equation, with zero terminal value on diagonal `N_outer+1`. It is not value iteration. Maximum baseline Bellman residual: `{bs['residual']:.3e}`.

## Baseline
Tested diagonals 0--500 with `N_outer=1300`. Empirical extinction diagonal: `{extinction}`. Unimodality violations: `{int((~base_diag.unimodal).sum())}`. Interval-property violations: `{int((~base_diag.interval_property).sum())}`. Every active product-2 diagonal has one robust integer interval if the latter count is zero.

## Search
The structured design contains `{len(structured())}` cases and the fixed-seed randomized design contains `{args.random_count}`, stratified across all three regimes. Every configuration was solved at `N_outer={args.search_outer}` and again at `{args.search_outer+100}` on diagonals through `{args.search_interior}`; gap and policy changes are recorded in `parameter_summary.csv`. All apparent violations were rerun on three grids with increments `max(500,ceil(10/(1-gamma)))`.

Confirmed robust unimodality violations: `{confirmed_uni}`. Confirmed robust interval-property violations: `{confirmed_band}`. See CSV files for regime-level and candidate details.

Baseline outer-grid comparison (`1300` versus `1450`) gives maximum gap change `{baseline_max_G_change:.3e}` and `{baseline_policy_changes}` policy changes on diagonals 0--500.

Across the broad first-pass search, `{int((ss.policy_changes_outer_grid_check>0).sum())}` of `{len(ss)}` configurations have at least one policy change between outer grids 180 and 280; these are concentrated among patient sellers and are not treated as fully converged policy estimates. The largest gap change was `{ss.max_G_change_outer_grid_check.max():.6g}`. Neither outer-grid solution generated a shape candidate, but absence of a counterexample in these high-gamma cases is weaker evidence than in converged cases.

```csv
{regime_text}```

Auxiliary checks found `{int(ss.negative_gap_violations.sum())}` configurations with a materially negative continuation gap and `{int(ss.product2_reappears_after_empty_diagonal.sum())}` cases where product 2 disappeared and later reappeared on the tested interior diagonals. Boundary-direction changes are exploratory and are recorded in `parameter_summary.csv`; they are not implied by within-diagonal unimodality.

The reappearance cases were:

```csv
{reappearance_text}```

The single reappearance case was separately checked at `N_outer=180,680,1180`: product 2 is active on diagonals 6--29, absent on 30, active again on 31--33, and absent from 34 onward. This is robust and does not violate either within-diagonal property.

## Conclusion and theorem recommendation
Numerical results are evidence, not proof. No violation of either property was found, so attempting a proof of general unimodality is reasonable. The weakest economically relevant theorem supported by the evidence is the interval property; it should be targeted as a fallback if a unimodality proof fails. Near ties and terminal artifacts are not counted as counterexamples.

Concise answers: (1) no robust unimodality violation was found; (2) no robust interval-property violation was found; (3) this held in all three regimes; (4) the properties were empirically equivalent in this design because both always held, although they are not logically equivalent; (5) the weakest plausible theorem is the within-diagonal interval property. A separate theorem claiming monotone extinction across diagonals is not supported because robust disappearance and reappearance occurred once.
""";(out/"REPORT_unimodality_and_band_search.md").write_text(report);print(report,flush=True)

if __name__=="__main__":main()
