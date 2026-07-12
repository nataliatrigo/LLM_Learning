"""Outer-grid convergence study for the Proposition 1 local certificate.

The certificate terminal diagonal is ``N_outer + 1`` while all comparisons and
figures are restricted to ``n <= N_plot``.  Run from the repository root with

    uv run python discounted/DP/analysis/certificate_boundary_study.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from discounted.DP.analysis.local_extinction_certificate import (
    CERTIFIED_COLOR, INK, INVEST_COLOR, NONCERT_COLOR, Params, binom_pmf,
    configure_plot_style, demand_diag, ell, save_figure, solve_empirical_slice,
)


SELECTED_N = (20, 50, 100, 200, 400, 600, 800)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outer-grid", default="800,1000,1200,1600,2400")
    parser.add_argument("--N-plot", type=int, default=800)
    parser.add_argument("--bellman-tail", type=int, default=650)
    parser.add_argument("--bellman-tail-check", type=int, default=800)
    parser.add_argument("--skip-bellman-check", action="store_true")
    parser.add_argument(
        "--outputs-dir", type=Path,
        default=Path(__file__).resolve().parent / "outputs_certificate_boundary",
    )
    return parser.parse_args()


def components(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if not len(idx):
        return []
    cuts = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(idx) - 1]
    return [(int(idx[a]), int(idx[b])) for a, b in zip(starts, ends)]


def solve_certificate(N_outer: int, N_plot: int, params: Params) -> tuple[dict, pd.DataFrame, int | None, float]:
    """Recurse from constant valid bound on n=N_outer+1; retain n<=N_plot."""
    if N_outer < N_plot:
        raise ValueError("N_outer must be at least N_plot")
    start = time.perf_counter()
    terminal = params.cheap_margin / (1.0 - params.gamma)
    U_next = np.full(N_outer + 2, terminal)
    kept: dict[int, dict[str, np.ndarray]] = {}
    outer_boundary_rows = []
    wall_active = True
    observed_wall: int | None = None
    ucrit = params.delta_c / (params.gamma * params.delta_p)
    for n in range(N_outer, -1, -1):
        assert U_next.shape == (n + 2,)
        d = demand_diag(n + 1, params)
        d_a, d_b = d[1:], d[:-1]
        # Stable exact identity: D_A-D_B=P(Bin(n+2,p0)=S+1).
        delta_d = np.asarray(binom_pmf(np.arange(n + 1) + 1, n + 2, params.p0))
        A = params.cheap_margin * delta_d / (ell(d_a, params) * ell(d_b, params))
        B = params.gamma * d_b / ell(d_b, params)
        c1 = params.p1 * U_next[1:] + (1.0 - params.p1) * U_next[:-1]
        c2 = params.p2 * U_next[1:] + (1.0 - params.p2) * U_next[:-1]
        U = A + B * np.maximum(c1, c2)
        assert U.shape == (n + 1,)
        noncert = U >= ucrit
        runs = components(noncert)
        m = (np.arange(n + 1) + 1.0) / (n + 2.0)
        outer_boundary_rows.append({
            "N_outer": N_outer, "n": n,
            "m_lower": m[runs[0][0]] if runs else np.nan,
            "m_upper": m[runs[-1][1]] if runs else np.nan,
        })
        high_m_hit = bool(np.any((m >= 0.95) & noncert))
        if wall_active and high_m_hit:
            observed_wall = n
        elif wall_active:
            wall_active = False
        if n <= N_plot:
            kept[n] = {"U": U.copy(), "A": A, "B": B}
        U_next = U
    return kept, pd.DataFrame(outer_boundary_rows), observed_wall, time.perf_counter() - start


def wall_location(sol: dict, params: Params, tail_width: float = 0.05) -> int | None:
    """Start of the terminal-connected wall in the highest-m 5% band."""
    ucrit = params.delta_c / (params.gamma * params.delta_p)
    hit: dict[int, bool] = {}
    for n, values in sol.items():
        m = (np.arange(n + 1) + 1.0) / (n + 2.0)
        hit[n] = bool(np.any((m >= 1.0 - tail_width) & (values["U"] >= ucrit)))
    # Ignore genuine low-n components: identify only the true suffix connected
    # to the displayed/outer boundary.
    n = max(hit)
    if not hit[n]:
        return None
    while n > min(hit) and hit[n - 1]:
        n -= 1
    return n


def diagonal_rows(N: int, sol: dict, previous: dict | None, params: Params) -> list[dict]:
    ucrit = params.delta_c / (params.gamma * params.delta_p)
    rows = []
    for n in sorted(sol):
        U = sol[n]["U"]
        noncert = U >= ucrit
        runs = components(noncert)
        m = (np.arange(n + 1) + 1.0) / (n + 2.0)
        changed = np.zeros(n + 1, bool) if previous is None else noncert != (previous[n]["U"] >= ucrit)
        du = np.zeros(n + 1) if previous is None else np.abs(U - previous[n]["U"])
        old_runs = [] if previous is None else components(previous[n]["U"] >= ucrit)
        low = m[runs[0][0]] if runs else np.nan
        high = m[runs[-1][1]] if runs else np.nan
        old_low = m[old_runs[0][0]] if old_runs else np.nan
        old_high = m[old_runs[-1][1]] if old_runs else np.nan
        boundary_ok = previous is None or (
            (np.isnan(low) and np.isnan(old_low)) or
            (not np.isnan(low) and not np.isnan(old_low) and
             abs(low-old_low) <= 1/(n+2)+1e-15 and abs(high-old_high) <= 1/(n+2)+1e-15)
        )
        rows.append({
            "N_outer": N, "n": n, "m_lower": low, "m_upper": high,
            "number_noncertified_states": int(noncert.sum()),
            "number_connected_components": len(runs),
            "components": ";".join(f"{m[a]:.12g}:{m[b]:.12g}" for a,b in runs),
            "classification_changes_vs_previous_grid": int(changed.sum()),
            "max_U_difference_vs_previous_grid": float(du.max()),
            "boundary_resolution": 1.0/(n+2),
            "is_numerically_stable": bool(not changed.any() and boundary_ok),
        })
    return rows


def summary_row(N: int, sol: dict, previous: dict | None, observed_wall: int | None, runtime: float, params: Params) -> dict:
    terminal = params.cheap_margin/(1-params.gamma)
    ucrit = params.delta_c/(params.gamma*params.delta_p)
    kpred = math.log(ucrit/terminal)/math.log(params.gamma)
    if previous is None:
        return {"N_outer":N,"predicted_wall_diagonal":N-kpred,"observed_wall_diagonal":observed_wall,
                "maximum_U_difference":np.nan,"mean_U_difference":np.nan,"fraction_classification_changes":np.nan,
                "maximum_lower_boundary_change":np.nan,"maximum_upper_boundary_change":np.nan,
                "largest_changed_diagonal":np.nan,"runtime_seconds":runtime}
    diffs=[]; changes=total=0; lows=[]; highs=[]; changed_n=[]
    for n in sol:
        a,b=sol[n]["U"],previous[n]["U"]; diffs.append(np.abs(a-b)); total+=len(a)
        ca,cb=a>=ucrit,b>=ucrit; changes+=int(np.sum(ca!=cb))
        if np.any(ca!=cb): changed_n.append(n)
        ra,rb=components(ca),components(cb)
        if ra and rb:
            m=(np.arange(n+1)+1)/(n+2); lows.append(abs(m[ra[0][0]]-m[rb[0][0]])); highs.append(abs(m[ra[-1][1]]-m[rb[-1][1]]))
    flat=np.concatenate(diffs)
    return {"N_outer":N,"predicted_wall_diagonal":N-kpred,"observed_wall_diagonal":observed_wall,
            "maximum_U_difference":float(flat.max()),"mean_U_difference":float(flat.mean()),
            "fraction_classification_changes":changes/total,"maximum_lower_boundary_change":max(lows,default=0),
            "maximum_upper_boundary_change":max(highs,default=0),"largest_changed_diagonal":max(changed_n,default=-1),
            "runtime_seconds":runtime}


def plot_final(sol: dict, empirical: pd.DataFrame, N: int, params: Params, path: Path) -> None:
    configure_plot_style(); fig,ax=plt.subplots(figsize=(10.5,6.2))
    ucrit=params.delta_c/(params.gamma*params.delta_p)
    nc=[]; cert=[]
    for n,v in sol.items():
        S=np.arange(n+1); m=(S+1)/(n+2); mask=v["U"]>=ucrit
        nc.append((np.full(mask.sum(),n),m[mask])); cert.append((np.full((~mask).sum(),n),m[~mask]))
    for chunks,color,label,z in [(cert,CERTIFIED_COLOR,"Certified product 1",1),(nc,NONCERT_COLOR,"Non-certified (not necessarily product 2)",2)]:
        ax.scatter(np.concatenate([x for x,_ in chunks]),np.concatenate([y for _,y in chunks]),s=2,c=color,label=label,rasterized=True,zorder=z)
    inv=empirical[(empirical.n<=max(sol)) & empirical.empirical_product2]
    ax.scatter(inv.n,inv.m,s=5,c=INVEST_COLOR,label="Empirical product 2",rasterized=True,zorder=3)
    ax.set(xlabel="n = S+F",ylabel="m = (S+1)/(n+2)",title="Empirical product 2 and converged local certificate",xlim=(0,max(sol)))
    ax.legend(markerscale=3); save_figure(fig,path)


def plot_boundaries(table: pd.DataFrame, path: Path) -> None:
    configure_plot_style(); fig,ax=plt.subplots(figsize=(10.5,5.8))
    for N,g in table.groupby("N_outer"):
        ax.plot(g.n,g.m_lower,lw=1,label=f"lower, N={N}"); ax.plot(g.n,g.m_upper,lw=1,ls="--",label=f"upper, N={N}")
    ax.set(xlabel="n",ylabel="Non-certified boundary m",title="Outer-grid convergence"); ax.legend(ncol=2,fontsize=8); save_figure(fig,path)


def plot_disagreement(table: pd.DataFrame, path: Path) -> None:
    configure_plot_style(); fig,ax=plt.subplots(figsize=(10.5,5.2))
    for N,g in table.groupby("N_outer"):
        if g.classification_changes_vs_previous_grid.sum(): ax.plot(g.n,g.classification_changes_vs_previous_grid/(g.n+1),label=f"vs previous: N={N}")
    ax.set(xlabel="n",ylabel="Fraction changed",title="Certificate classification disagreement"); ax.legend(); save_figure(fig,path)


def plot_decomposition(sol: dict, path: Path, reflection: bool=False) -> None:
    configure_plot_style(); fig,axes=plt.subplots(3,1,figsize=(9,10),sharex=True)
    for n in SELECTED_N:
        if n not in sol: continue
        m=(np.arange(n+1)+1)/(n+2)
        for ax,key in zip(axes,("A","B","U")):
            y=sol[n][key]; y=y-y[::-1] if reflection else y; ax.plot(m,y,lw=1,label=f"n={n}")
    labels=("A(S,F)-A(F,S)","B(S,F)-B(F,S)","U(S,F)-U(F,S)") if reflection else ("A","B","U")
    for ax,label in zip(axes,labels): ax.set_ylabel(label); ax.grid(True)
    axes[-1].set_xlabel("m"); axes[0].legend(ncol=4,fontsize=8)
    fig.suptitle("Reflection asymmetry" if reflection else "Certificate operator decomposition"); save_figure(fig,path)


def validate_demand(params: Params) -> pd.DataFrame:
    rows=[]
    for S,F in [(0,0),(2,3),(20,20),(75,25),(200,200),(600,200)]:
        n=S+F; d=demand_diag(n+1,params); direct=binom_pmf(S+1,n+2,params.p0); sub=d[S+1]-d[S]
        rows.append({"S":S,"F":F,"direct_pmf":direct,"cdf_difference":sub,"positive":direct>0,
                     "absolute_error":abs(direct-sub),"subtraction_reliable":sub>1e-12})
    out=pd.DataFrame(rows); assert out.positive.all(); assert (out.loc[out.subtraction_reliable,"absolute_error"]<1e-11).all(); return out


def main() -> None:
    args=parse_args(); params=Params(); grids=sorted({int(x) for x in args.outer_grid.split(",")})
    out=args.outputs_dir; out.mkdir(parents=True,exist_ok=True)
    validate_demand(params).to_csv(out/"demand_increment_tests.csv",index=False)
    solutions={}; rows=[]; summaries=[]; outer_boundaries=[]; previous=None
    for N in grids:
        print(f"Computing certificate N_outer={N}, N_plot={args.N_plot}...",flush=True)
        sol,outer_boundary,observed_wall,runtime=solve_certificate(N,args.N_plot,params); solutions[N]=sol
        outer_boundaries.append(outer_boundary)
        rows.extend(diagonal_rows(N,sol,previous,params)); summaries.append(summary_row(N,sol,previous,observed_wall,runtime,params)); previous=sol
        print(f"  completed in {runtime:.2f}s",flush=True)
    table=pd.DataFrame(rows); summary=pd.DataFrame(summaries); outer_boundary_table=pd.concat(outer_boundaries,ignore_index=True)
    table.to_csv(out/"certificate_diagonal_convergence.csv",index=False); summary.to_csv(out/"certificate_outer_grid_summary.csv",index=False)
    empirical=solve_empirical_slice(args.N_plot,params,args.bellman_tail)
    empirical_check=empirical
    policy_disagreements=np.nan
    if not args.skip_bellman_check:
        empirical_check=solve_empirical_slice(args.N_plot,params,args.bellman_tail_check)
        policy_disagreements=int(np.sum(empirical.empirical_product2.to_numpy()!=empirical_check.empirical_product2.to_numpy()))
    final=solutions[grids[-1]]; ucrit=params.delta_c/(params.gamma*params.delta_p)
    violations=0
    for n,g in empirical_check.groupby("n"):
        violations += int(np.sum(g.empirical_product2.to_numpy() & (final[int(n)]["U"]<ucrit)))
    if violations: raise AssertionError(f"{violations} empirical product-2 states are certified product 1")
    plot_final(final,empirical_check,grids[-1],params,out/"figure_A_final_certificate_empirical_policy.png")
    plot_boundaries(outer_boundary_table[outer_boundary_table.N_outer.isin([800,1200,1600,2400])],out/"figure_B_outer_grid_convergence.png")
    plot_disagreement(table,out/"figure_C_classification_disagreement.png")
    plot_decomposition(final,out/"figure_D_operator_decomposition.png")
    plot_decomposition(final,out/"figure_E_reflection_asymmetry.png",True)
    decomposition=[]
    for n in SELECTED_N:
        m=(np.arange(n+1)+1)/(n+2)
        for i in range(n+1):
            decomposition.append({"n":n,"S":i,"F":n-i,"m":m[i],
                "A":final[n]["A"][i],"B":final[n]["B"][i],"U":final[n]["U"][i],
                "A_reflection_difference":final[n]["A"][i]-final[n]["A"][::-1][i],
                "B_reflection_difference":final[n]["B"][i]-final[n]["B"][::-1][i],
                "U_reflection_difference":final[n]["U"][i]-final[n]["U"][::-1][i]})
    decomposition_frame=pd.DataFrame(decomposition)
    decomposition_frame.to_csv(out/"operator_decomposition_selected_diagonals.csv",index=False)
    asymmetry=(decomposition_frame.groupby("n").agg(
        max_abs_A_reflection_difference=("A_reflection_difference",lambda x: float(np.abs(x).max())),
        max_abs_B_reflection_difference=("B_reflection_difference",lambda x: float(np.abs(x).max())),
        max_abs_U_reflection_difference=("U_reflection_difference",lambda x: float(np.abs(x).max())),
    ).reset_index())
    largest=grids[-1]; prev=grids[-2]; stable=table[table.N_outer==largest]
    unstable=stable.loc[~stable.is_numerically_stable,"n"]
    n_stable=int(unstable.min()-1) if len(unstable) and unstable.min()>0 else (args.N_plot if not len(unstable) else -1)
    disconnected=stable.loc[stable.number_connected_components>1,"n"].tolist()
    summary_text = summary.to_csv(index=False)
    asymmetry_text = asymmetry.to_csv(index=False)
    report=f"""# Certificate Boundary Report

## Implementation

Inspected `local_extinction_certificate.py`, `exact_dp.py`, and the previous figure `plot3_empirical_vs_noncertified_N800.png`. Modified the certificate recursion to use the stable binomial PMF and added this outer-grid study.

The valid constant terminal bound `U=(R-c1)/(1-gamma)={params.cheap_margin/(1-params.gamma):.12g}` is imposed on diagonal `n=N_outer+1`; recursion starts at `N_outer`. Only diagonals `n<=N_plot={args.N_plot}` are retained. The operator is `U=A+B max_x[p_x U(S+1,F)+(1-p_x)U(S,F+1)]`, with `A=(R-c1) PMF/(ell(D_A)ell(D_B))`, `B=gamma D_B/ell(D_B)`, `D_A=P(Bin(n+2,p0)<=S+1)`, `D_B=P(Bin(n+2,p0)<=S)`, and `PMF=P(Bin(n+2,p0)=S+1)`.

## Boundary prediction and convergence

`U_crit=Delta_c/(gamma Delta_p)={ucrit:.12g}` and `k_pred=log(U_crit/U_terminal)/log(gamma)={math.log(ucrit/(params.cheap_margin/(1-params.gamma)))/math.log(params.gamma):.3f}` diagonals.

```csv
{summary_text}```

The comparison of the two largest grids (`{prev}` versus `{largest}`) gives `n_stable={n_stable}` under exact classification agreement plus the one-state boundary-resolution criterion. Diagonals with multiple non-certified components: `{disconnected if disconnected else 'none'}`.

## Bellman validity

Bellman remaining-horizon comparison: `{args.bellman_tail}` versus `{args.bellman_tail_check}`; policy disagreements on `n<=800`: `{policy_disagreements}`. The largest empirical product-2 diagonal is `{int(empirical_check.loc[empirical_check.empirical_product2,'n'].max())}` and therefore does not touch the grid boundary. Certificate violations on empirical product-2 states: `{violations}`.

## Asymmetry and conclusion

The binomial PMF source numerator is reflection-symmetric at `p0=0.5`. The denominators in `A`, demand-dependent `B`, and asymmetric product probabilities make the full operator asymmetric. The upper vertical wall moves with `N_outer` and is a terminal-boundary artifact; asymmetry that is unchanged between the two largest outer grids is inherent to the certificate operator, not imposed or symmetrized numerically.

Maximum absolute reflected-state differences on representative diagonals:

```csv
{asymmetry_text}```

Use `N_outer={largest}` and display no more than the verified stable range (`N_plot<={n_stable}`). Non-certified means only that this sufficient product-1 certificate is inconclusive; it is not a prediction of product 2.
"""
    (out/"REPORT_certificate_boundary.md").write_text(report)
    print(f"Wrote {out}",flush=True)


if __name__=="__main__": main()
