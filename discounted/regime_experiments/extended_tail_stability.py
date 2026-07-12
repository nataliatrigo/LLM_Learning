from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from discounted.regime_experiments.configs import all_configs
from discounted.regime_experiments.solve_regimes import solve_snapshot


def main()->None:
    ids={"A_006","A_018","B_03_00","B_03_03"};rows=[]
    for cfg in [c for c in all_configs() if c.parameter_id in ids]:
        _,base=solve_snapshot(cfg,300,1e-7);_,long=solve_snapshot(cfg,340,1e-8)
        a=base[base.n<=180].reset_index(drop=True);b=long[long.n<=180].reset_index(drop=True)
        rows.append({"parameter_id":cfg.parameter_id,"regime":cfg.regime,"p0":cfg.p0,
            "base_outer":300,"base_tail_tolerance":1e-7,"long_outer":340,"long_tail_tolerance":1e-8,
            "common_interior":180,"policy_changes":int(np.sum(a.optimal_product.to_numpy()!=b.optimal_product.to_numpy())),
            "policy_change_fraction":float(np.mean(a.optimal_product.to_numpy()!=b.optimal_product.to_numpy())),
            "max_G_change":float(np.max(np.abs(a.G.to_numpy()-b.G.to_numpy())))})
    out=pd.DataFrame(rows);path=Path(__file__).resolve().parent/"outputs"/"tables"/"extended_tail_stability.csv"
    out.to_csv(path,index=False);print(out.to_string(index=False))


if __name__=="__main__":main()
