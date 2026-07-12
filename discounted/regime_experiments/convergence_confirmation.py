from __future__ import annotations

import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from discounted.regime_experiments.configs import all_configs
from discounted.regime_experiments.solve_regimes import solve_snapshot


def run_primary_confirmations(path: Path) -> pd.DataFrame:
    configs = [c for c in all_configs() if c.representative]
    rows = []
    for cfg in configs:
        grids = [(120, 1e-5), (160, 1e-6), (200, 1e-7)]
        frames = []
        for outer, tol in grids:
            _, frame = solve_snapshot(cfg, outer, tol)
            frames.append(frame[frame.n <= 80].reset_index(drop=True))
            gc.collect()
        base = frames[-1]
        for (outer, tol), frame in zip(grids, frames):
            rows.append({
                "parameter_id": cfg.parameter_id, "p0": cfg.p0, "p1": cfg.p1,
                "p2": cfg.p2, "gamma": cfg.gamma, "regime": cfg.regime,
                "analysis_outer_diagonal": outer, "tail_tolerance": tol,
                "common_interior": 80,
                "policy_changes_vs_largest": int(np.sum(
                    frame.optimal_product.to_numpy() != base.optimal_product.to_numpy())),
                "max_G_change_vs_largest": float(np.max(np.abs(
                    frame.G.to_numpy()-base.G.to_numpy()))),
            })
    result = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True); result.to_csv(path,index=False)
    return result


if __name__ == "__main__":
    out = Path(__file__).resolve().parent/"outputs"/"tables"/"primary_convergence.csv"
    print(run_primary_confirmations(out).to_string(index=False))
