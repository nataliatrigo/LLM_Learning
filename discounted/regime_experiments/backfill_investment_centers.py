from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from discounted.regime_experiments.configs import all_configs
from discounted.regime_experiments.policy_diagnostics import embedded_reach
from discounted.regime_experiments.run_all import reach_weighted_investment_center
from discounted.regime_experiments.solve_regimes import solve_snapshot


def main()->None:
    path=Path(__file__).resolve().parent/"outputs"/"tables"/"parameter_summary.csv"
    p=pd.read_csv(path);missing=set(p.loc[p.reach_weighted_investment_center_z.isna(),"parameter_id"])
    cfgs=[c for c in all_configs() if c.parameter_id in missing]
    for i,cfg in enumerate(cfgs,1):
        row=p[p.parameter_id==cfg.parameter_id].iloc[0];outer=int(row.analysis_outer_diagonal)
        print(f"[{i}/{len(cfgs)}] center {cfg.parameter_id}",flush=True)
        _,states=solve_snapshot(cfg,outer,1e-7);states["reliable_interior"]=states.n<=int(row.reliable_interior_grid)
        states=embedded_reach(states,cfg)
        p.loc[p.parameter_id==cfg.parameter_id,"reach_weighted_investment_center_z"]=reach_weighted_investment_center(states)
    p["last_active_censored"]=(p.last_active_diagonal>=p.reliable_interior_grid)
    p["reported_last_active_diagonal"]=p.last_active_diagonal.mask(p.last_active_censored)
    p.to_csv(path,index=False)
    print(f"Backfilled {len(cfgs)} centers.")


if __name__=="__main__":main()
