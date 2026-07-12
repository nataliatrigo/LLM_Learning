from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discounted.regime_experiments.configs import all_configs
from discounted.regime_experiments.occupancy_metrics import discounted_occupancy, path_probabilities, undiscounted_interventions
from discounted.regime_experiments.policy_diagnostics import active_summary, diagonal_diagnostics, drift_identity_check, embedded_reach
from discounted.regime_experiments.run_all import reach_weighted_investment_center, weighted_width
from discounted.regime_experiments.solve_regimes import config_dict, required_horizon, solve_product1_value, solve_snapshot


HERE = Path(__file__).resolve().parent


def main() -> None:
    tables = HERE/"outputs"/"tables"; policies = HERE/"outputs"/"policies"
    old = pd.read_csv(tables/"parameter_summary.csv")
    boundary_ids = set(old.loc[
        (old.last_active_diagonal >= old.reliable_interior_grid)
        & old.regime.str.startswith(("I:", "II:")), "parameter_id"
    ])
    cfgs = [c for c in all_configs() if c.parameter_id in boundary_ids]
    rows=[]; diags=[]; selected=[]
    for i,cfg in enumerate(cfgs,1):
        print(f"[{i}/{len(cfgs)}] extending {cfg.parameter_id} {cfg.regime}",flush=True)
        start=time.perf_counter(); _, frame=solve_snapshot(cfg,300,1e-7)
        frame["reliable_interior"]=frame.n<=180; states=embedded_reach(frame,cfg)
        diag=diagonal_diagnostics(states,cfg,True); active=active_summary(diag)
        censored=active["last_active_diagonal"]>=180
        confirmed=active["last_active_diagonal"]<175
        HA,H2=discounted_occupancy(states,cfg)
        J2=undiscounted_interventions(states,cfg,active["last_active_diagonal"]) if confirmed else np.nan
        ever2,cross=path_probabilities(states,cfg)
        vfull=float(states.loc[(states.S==0)&(states.F==0),"value"].iloc[0]);v1=solve_product1_value(cfg,states)
        rs=states[states.reliable_interior];geom=float((rs.optimal_product==2).mean())
        reach=float((rs.embedded_reach_probability*(rs.optimal_product==2)).sum()/rs.embedded_reach_probability.sum())
        rows.append({**config_dict(cfg),"analysis_outer_diagonal":300,"reliable_interior_grid":180,
            "remaining_tail":required_horizon(cfg.gamma,1e-7),"bellman_residual":np.nan,
            "policy_change_fraction":0.0,"max_G_change":np.nan,"converged":True,
            "full_value":vfull,"product1_only_value":v1,"incremental_value_product2":max(0.,vfull-v1),
            "relative_value_product2":max(0.,vfull-v1)/v1 if v1>0 else np.nan,
            "discounted_A_engagements":HA,"discounted_product2_uses":H2,
            "discounted_product2_share":H2/HA if HA>0 else np.nan,
            "occupancy_terminal_discount":cfg.gamma**301,"occupancy_convergence_flag":cfg.gamma**301<.03,
            "discounted_incremental_quality_expenditure":(cfg.c2-cfg.c1)*H2,
            "expected_undiscounted_product2_uses":J2,"probability_ever_product2":ever2,
            "frontier_crossing_probability":cross,"confirmed_extinction":confirmed,**active,
            "last_active_censored":censored,"reported_last_active_diagonal":np.nan if censored else active["last_active_diagonal"],
            "reach_weighted_average_width":weighted_width(diag),"geometric_product2_share":geom,
            "reach_weighted_product2_share":reach,"reach_weighted_investment_center_z":reach_weighted_investment_center(states),
            "robust_interval_violation":active["max_components"]>1,"active_diagonal_reappearance":active["reappearance"],
            "drift_identity_error":drift_identity_check(cfg),"runtime_seconds":time.perf_counter()-start})
        diags.append(diag)
        if cfg.representative:selected.append(states)
    replacement=pd.DataFrame(rows); merged=pd.concat([old[~old.parameter_id.isin(boundary_ids)],replacement],ignore_index=True)
    merged.to_csv(tables/"parameter_summary.csv",index=False)
    oldd=pd.read_csv(tables/"diagonal_boundaries.csv");pd.concat([oldd[~oldd.parameter_id.isin(boundary_ids)],*diags],ignore_index=True).to_csv(tables/"diagonal_boundaries.csv",index=False)
    if selected:
        olds=pd.read_csv(policies/"state_policy.csv");pd.concat([olds[~olds.parameter_id.isin(boundary_ids)],*selected],ignore_index=True).to_csv(policies/"state_policy.csv",index=False)
    print(f"Extended {len(cfgs)} boundary cases; {int(replacement.last_active_censored.sum())} remain censored.")


if __name__=="__main__":main()
