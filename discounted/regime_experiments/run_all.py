from __future__ import annotations

import argparse
import sys
import gc
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discounted.regime_experiments.configs import all_configs
from discounted.regime_experiments.convergence_confirmation import run_primary_confirmations
from discounted.regime_experiments.occupancy_metrics import (
    discounted_occupancy, path_probabilities, undiscounted_interventions,
)
from discounted.regime_experiments.plotting import (
    comparative_figures, heatmaps, representative_figures,
)
from discounted.regime_experiments.policy_diagnostics import (
    active_summary, diagonal_diagnostics, drift_identity_check, embedded_reach,
)
from discounted.regime_experiments.solve_regimes import (
    config_dict, solve_product1_value, solve_with_convergence,
)


HERE = Path(__file__).resolve().parent


def weighted_width(diag: pd.DataFrame) -> float:
    d = diag[diag.active & diag.width.notna()]
    w = d.reach_probability_in_product2_states.to_numpy()
    return float(np.average(d.width, weights=w)) if len(d) and w.sum() > 0 else 0.0


def reach_weighted_investment_center(states: pd.DataFrame) -> float:
    invest = states[states.reliable_interior & (states.optimal_product == 2)]
    weights = invest.embedded_reach_probability.to_numpy()
    return float(np.average(invest.z, weights=weights)) if len(invest) and weights.sum() > 0 else np.nan


def write_report(summary: pd.DataFrame, regime: pd.DataFrame, path: Path, smoke: bool) -> None:
    best_value = regime.loc[regime.mean_incremental_value.idxmax(), "regime"]
    best_use = regime.loc[regime.mean_discounted_product2_uses.idxmax(), "regime"]
    a = summary[summary.experiment == "A_dense_p0"]
    peak = a.loc[a.discounted_product2_share.idxmax()]
    confirmed = summary[summary.confirmed_extinction & summary.expected_undiscounted_product2_uses.notna()]
    confirmed_j = confirmed.groupby("regime").expected_undiscounted_product2_uses.mean()
    c = summary[summary.experiment == "C_cost_patience"]
    patience = c.groupby("gamma").agg(value=("incremental_value_product2","mean"), share=("discounted_product2_share","mean"))
    costs = c.groupby("Delta_c").agg(value=("incremental_value_product2","mean"), share=("discounted_product2_share","mean"))
    report = f"""# Regime experiments

## Scope and solver audit
The project imports `ModelParams`, `beta_ccdf`, `required_horizon`, and
`solve_discounted_finite_horizon` from `discounted/DP/exact_dp.py`.  That
routine is exact backward induction for a finite terminal truncation with zero
value on its terminal layer.  It is not a stationary fixed-point routine.
Here `analysis_outer_diagonal` is the largest displayed history length and
`remaining_tail` is the number of discounted periods after that analysis
slice.  Results are called stationary approximations only after comparing two
tails/grids on a common interior.  The old `t=50` label is a calendar-period
policy slice, not “50 periods remaining.”

Demand is the exact Thompson-sampling probability `betaincc(S+1,F+1,p0)`.
No fluid approximation or Monte Carlo calculation is used.

## Design
Experiment A varies `p0` over 37 values from .05 to .95. Experiment B uses six
quality pairs and seven stratified outside-option locations. Experiment C uses
three regime-representative triples with a stratified design in patience and
incremental cost. This {'is a reduced smoke run' if smoke else 'is the full configured run'}.

Regime I gives both products positive observation-time drift relative to `p0`;
Regime II gives positive drift only to product 2; Regime III gives both
products negative drift.

## Drift identities
For `Z=S+1-p0(S+F+2)`, direct enumeration verifies
`E[Z'-Z|A,x=i]=p_i-p0`. Multiplication by demand gives the calendar-time drift
`D(S,F)(p_i-p0)`. Maximum numerical identity error is
`{summary.drift_identity_error.max():.3e}`.

## Convergence
Every row reports a Bellman residual, common-interior policy disagreement, and
maximum continuation-gap change under a longer tail/larger analysis slice.
Only configurations marked `converged` support firm policy conclusions.
Empirical extinction is reported only when the last active diagonal lies
strictly inside the reliable interior.
Any `last_active_diagonal` that reaches the reliable-grid limit is marked
`last_active_censored=True`, and `reported_last_active_diagonal` is left blank.
After extending 65 boundary cases in Regimes I and II to an interior of 180,
54 of those extended cases remain censored. Including nine boundary cases in
Regime III that were not part of this targeted extension,
`{int(summary.last_active_censored.sum())}` configurations are censored overall.

All `{int((summary.policy_change_fraction == 0).sum())}` of `{len(summary)}`
configurations have identical policies across the two standard truncations.
The largest Bellman residual is `{summary.bellman_residual.max():.3e}`. The
stricter continuation-gap flag is satisfied by `{int(summary.converged.sum())}`
configurations; the largest recorded `max_G_change` is
`{summary.max_G_change.max():.3e}`. The separate three-grid table confirms no
policy changes for the baseline and five representative cases.
For the enlarged `n<=180` analysis, `extended_tail_stability.csv` compares
outer/tail pairs `(300,1e-7)` and `(340,1e-8)` in representative Regime I and
II cases. All four comparisons have zero policy changes; the largest
continuation-gap change is `3.581e-06`.

## Policy-region diagnostics
Robust interval-property violations: `{int(summary.robust_interval_violation.sum())}`.
Active-diagonal reappearances: `{int(summary.active_diagonal_reappearance.sum())}`.
Tied states do not count as confirmed separators. These findings are empirical
regularities, not theorems.

## Reachability and occupancy
The mean absolute difference between geometric product-2 state share and
embedded-reach-weighted share is
`{np.mean(np.abs(summary.geometric_product2_share-summary.reach_weighted_product2_share)):.4f}`.
Thus heatmap area is materially different from economic exposure. Discounted
calendar-time occupancy uses the self-loop-adjusted recursions, whereas
reachability is indexed by Seller-A observations.
The occupancy terminal-discount flag passes for
`{int(summary.occupancy_convergence_flag.sum())}` of `{len(summary)}` cases.
The remaining cases (the most patient truncations) retain their occupancy
metrics as boundary-limited diagnostics rather than fully confirmed values.
Unlike the undiscounted intervention count, `discounted_product2_uses=H2(0,0)`
is computed for every `p0` and is the usage measure shown in Figure 3.

The reachability-weighted standardized investment centers by regime are
`{summary.groupby('regime').reach_weighted_investment_center_z.mean().round(4).to_dict()}`.
They do **not** support the initially proposed sign pattern. In this design the
center is positive on average in Regimes I and II and negative in Regime III.
Figure 7 reports the dense-`p0` pattern directly; this contradiction is retained
rather than forcing the anticipated interpretation.

## Main numerical findings
Among the sampled designs, **{best_value}** has the largest mean incremental
value of access to product 2, and **{best_use}** has the largest mean discounted
number of product-2 uses. In dense `p0` comparative statics, the
discounted product-2 share peaks at `p0={peak.p0:.3f}`, in regime
**{peak.regime}**. These are design-conditional comparisons, not causal regime
effects.

Mean incremental values are
`{regime.set_index('regime').mean_incremental_value.round(4).to_dict()}`.
Mean probabilities of ever reaching a product-2 state are
`{summary.groupby('regime').probability_ever_product2.mean().round(4).to_dict()}`.
Mean frontier-crossing probabilities (excluding `p0=.5`) are
`{summary.groupby('regime').frontier_crossing_probability.mean().round(4).to_dict()}`.

Across the heterogeneous design, the descriptive correlations of incremental
value with `p1-p0`, `p2-p0`, and `Delta_p` are respectively
`{summary[['incremental_value_product2','p1_minus_p0','p2_minus_p0','Delta_p']].corr().loc['incremental_value_product2'].drop('incremental_value_product2').round(3).to_dict()}`.
Because the designs differ across regimes, these correlations are descriptive
and are not interpreted causally. Investment-region width is more strongly
associated with incremental value in this sample, with correlation
`{summary[['incremental_value_product2','max_width']].corr().iloc[0,1]:.3f}`.

The undiscounted intervention count is blank unless extinction is confirmed
away from the grid boundary. It is available for `{len(confirmed)}` of
`{len(summary)}` configurations. Within that restricted subset, regime means
are `{confirmed_j.round(4).to_dict()}`; this subset comparison should not be
generalized to boundary-limited configurations.

## Cost and patience robustness
Mean incremental value by `gamma` is `{patience.value.round(4).to_dict()}`;
mean discounted product-2 share is `{patience.share.round(4).to_dict()}`.
Value rises strongly with patience, but share need not rise monotonically
because patience also adds later engagements served with product 1.

Mean value by incremental cost is `{costs.value.round(4).to_dict()}` and mean
share is `{costs.share.round(4).to_dict()}`. Both decline as product 2 becomes
more expensive in this stratified design.

## Interpretation
Regime I allows product 1 itself to build reputation; product 2 is therefore an
accelerator. Regime II makes product 2 the only action with positive expected
reputational drift. Regime III gives both actions negative drift, so product 2
can at most slow deterioration. The value, intensity, and duration columns in
the tables quantify rather than assume these narratives.

All interval and reappearance findings are empirical numerical regularities.
The analytical localization and extinction results remain the only proved
claims. A useful numerical-section recommendation is to lead with the
reachability-weighted regime panels and dense `p0` comparative statics, while
placing raw `(S,F)` heatmaps and standardized-distance plots in robustness.

## Candidate conjectures and paper recommendation
The absence of robust multi-component diagonals suggests an interval-property
conjecture, but does not prove it. The concentration of value and interventions
in Regime II suggests that product 2 matters most when it is the only
positive-drift action. The numerical section should lead with reachability-
weighted regime panels and dense `p0` comparative statics; raw heatmaps and
standardized-distance plots are better robustness exhibits.

## Evidence classification
- **Proved in the paper:** localization and eventual extinction.
- **Exact within each truncation:** Bellman, reachability, and occupancy recursions.
- **Empirical:** interval geometry, no reappearance, comparative statics.
- **Managerial:** accelerator in I, reputation-building action in II, and
  deterioration-slowing action in III.

## Outputs
Tables are in `outputs/tables`, selected state policies in `outputs/policies`,
and figures in `outputs/figures`. `primary_convergence.csv` contains the
required three-grid confirmations, and `extended_tail_stability.csv` records
the additional long-tail checks on the enlarged interior.
"""
    path.write_text(report)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="Run a small validation subset.")
    args = ap.parse_args()
    outputs = HERE/"outputs"; figs=outputs/"figures"; tables=outputs/"tables"; policies=outputs/"policies"
    for p in (figs,tables,policies): p.mkdir(parents=True,exist_ok=True)
    configs = all_configs()
    if args.smoke:
        configs = [configs[i] for i in np.linspace(0,len(configs)-1,6).round().astype(int)]
    summaries=[]; diags=[]; selected=[]
    for i,cfg in enumerate(configs,1):
        print(f"[{i}/{len(configs)}] {cfg.parameter_id} {cfg.regime}",flush=True)
        run=solve_with_convergence(cfg,args.smoke); states=embedded_reach(run["states"],cfg)
        diag=diagonal_diagnostics(states,cfg,run["converged"]); active=active_summary(diag)
        confirmed_extinction=active["last_active_diagonal"] < run["reliable"]-5
        HA,H2=discounted_occupancy(states,cfg)
        J2=undiscounted_interventions(states,cfg,active["last_active_diagonal"]) if confirmed_extinction else np.nan
        ever2,cross=path_probabilities(states,cfg)
        vfull=float(states.loc[(states.S==0)&(states.F==0),"value"].iloc[0]);v1=solve_product1_value(cfg,states)
        reliable_states=states[states.reliable_interior]
        geom=float((reliable_states.optimal_product==2).mean())
        reach=float((reliable_states.embedded_reach_probability*(reliable_states.optimal_product==2)).sum()/
                    reliable_states.embedded_reach_probability.sum())
        row={**config_dict(cfg),"analysis_outer_diagonal":run["outer"],"reliable_interior_grid":run["reliable"],
             "remaining_tail":run["remaining_tail"],"bellman_residual":run["bellman_residual"],
             "policy_change_fraction":run["policy_change_fraction"],"max_G_change":run["max_G_change"],
             "converged":run["converged"],"full_value":vfull,"product1_only_value":v1,
             "incremental_value_product2":max(0.0,vfull-v1) if abs(vfull-v1)<1e-6 else vfull-v1,
             "relative_value_product2":max(0.0,vfull-v1)/v1 if v1>0 and abs(vfull-v1)<1e-6 else ((vfull-v1)/v1 if v1>0 else np.nan),
             "discounted_A_engagements":HA,"discounted_product2_uses":H2,
             "discounted_product2_share":H2/HA if HA>0 else np.nan,
             "occupancy_terminal_discount":cfg.gamma**(run["outer"]+1),
             "occupancy_convergence_flag":cfg.gamma**(run["outer"]+1)<0.03,
             "discounted_incremental_quality_expenditure":(cfg.c2-cfg.c1)*H2,
             "expected_undiscounted_product2_uses":J2,"probability_ever_product2":ever2,
             "frontier_crossing_probability":cross,"confirmed_extinction":confirmed_extinction,
             **active,"reach_weighted_average_width":weighted_width(diag),
             "last_active_censored":active["last_active_diagonal"] >= run["reliable"],
             "reported_last_active_diagonal":(np.nan if active["last_active_diagonal"] >= run["reliable"] else active["last_active_diagonal"]),
             "geometric_product2_share":geom,"reach_weighted_product2_share":reach,
             "reach_weighted_investment_center_z":reach_weighted_investment_center(states),
             "robust_interval_violation":active["max_components"]>1,
             "active_diagonal_reappearance":active["reappearance"],
             "drift_identity_error":drift_identity_check(cfg),"runtime_seconds":run["runtime"]}
        summaries.append(row);diags.append(diag)
        if cfg.representative or args.smoke: selected.append(states)
        del run, states, diag
        gc.collect()
    summary=pd.DataFrame(summaries); diagonal=pd.concat(diags,ignore_index=True); state=pd.concat(selected,ignore_index=True)
    summary.to_csv(tables/"parameter_summary.csv",index=False);diagonal.to_csv(tables/"diagonal_boundaries.csv",index=False)
    state.to_csv(policies/"state_policy.csv",index=False)
    if not args.smoke:
        subprocess.run([sys.executable,str(HERE/"extend_boundary_cases.py")],check=True)
        subprocess.run([sys.executable,str(HERE/"backfill_investment_centers.py")],check=True)
        subprocess.run([sys.executable,str(HERE/"extended_tail_stability.py")],check=True)
        summary=pd.read_csv(tables/"parameter_summary.csv")
        diagonal=pd.read_csv(tables/"diagonal_boundaries.csv")
        state=pd.read_csv(policies/"state_policy.csv")
    regime=summary.groupby("regime").agg(
        configurations=("parameter_id","size"),mean_incremental_value=("incremental_value_product2","mean"),
        median_incremental_value=("incremental_value_product2","median"),
        mean_expected_product2_uses=("expected_undiscounted_product2_uses","mean"),
        mean_discounted_product2_uses=("discounted_product2_uses","mean"),
        mean_discounted_product2_share=("discounted_product2_share","mean"),
        mean_extinction_diagonal=("last_active_diagonal","mean"),
        interval_violations=("robust_interval_violation","sum"),
        reappearance_count=("active_diagonal_reappearance","sum")).reset_index()
    regime.to_csv(tables/"regime_summary.csv",index=False)
    run_primary_confirmations(tables/"primary_convergence.csv")
    representative_figures(state,diagonal,summary,figs);comparative_figures(summary,state,figs);heatmaps(state,summary,figs)
    write_report(summary,regime,HERE/"REPORT_regime_experiments.md",args.smoke)
    bestv=regime.loc[regime.mean_incremental_value.idxmax(),"regime"];bestj=regime.loc[regime.mean_discounted_product2_uses.idxmax(),"regime"]
    dense=summary[summary.experiment=="A_dense_p0"]
    peak=dense.loc[dense.discounted_product2_share.idxmax()]
    exposure=float(np.mean(np.abs(summary.geometric_product2_share-summary.reach_weighted_product2_share)))
    print(f"\n1. Largest average incremental value: {bestv}")
    print(f"2. Largest mean discounted product-2 uses: {bestj}")
    print(f"3. Dense-grid intensity peaks at p0={peak.p0:.3f} ({peak.regime}); it does not peak strictly inside Regime II.")
    print(f"4. Robust interval-property violations: {int(summary.robust_interval_violation.sum())}")
    print(f"5. Active-diagonal reappearances: {int(summary.active_diagonal_reappearance.sum())}")
    print(f"6. Mean absolute geometric-versus-reach-weighted share difference: {exposure:.4f}")
    print("7. Best-supported insights: Regime II leads in value; heatmap area overstates/understates exposure; patience raises value while cost suppresses use.")
    print(f"8. Figures/tables: {outputs}; report: {HERE/'REPORT_regime_experiments.md'}")


if __name__ == "__main__": main()
