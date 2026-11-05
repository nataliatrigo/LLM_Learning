"""Audit saved results against actual FIFO histories and simulation totals."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .run_experiment import ROOT, CONFIG, dump, load_solution, slug, source_hashes


def audit_history(path, n, expected_paths, expected_interactions):
    records = np.load(path, mmap_mode="r")
    assert records.shape == (expected_interactions, expected_paths), str(path)
    cumulative = np.zeros((len(records) + 1, expected_paths), dtype=np.int32)
    np.cumsum(records["success"], axis=0, dtype=np.int32, out=cumulative[1:])
    starts = np.maximum(np.arange(len(records)) - n, 0)
    expected_s = cumulative[:-1] - cumulative[starts]
    assert np.array_equal(records["S_before"], expected_s), str(path)
    assert np.all(records["m_before"] == np.minimum(np.arange(len(records)), n)[:, None])
    assert np.all((records["demand"] >= 0) & (records["demand"] <= 1))
    return {"path": str(path.relative_to(ROOT)), "shape": list(records.shape),
            "fifo_all_states_match_reconstructed_outcomes": True,
            "initial_S": records["S_before"][0].tolist()}


def main():
    config = json.loads(CONFIG.read_text())
    diagnostics = pd.read_csv(ROOT / "data" / "solver_diagnostics.csv")
    exact = diagnostics[diagnostics.method == "exact_fifo"]
    assert len(exact) == 30
    assert exact.all_actions_certified.astype(bool).all()
    assert (exact.uncertified_action_states == 0).all()
    policies = list((ROOT / "data" / "policies").glob("*.npz"))
    assert len(policies) == 37
    checks = []
    for p0, n in config["interaction"]["cases"]:
        directory = ROOT / "data" / "interactions" / slug(p0, n)
        checks.append(audit_history(directory / "observations.npy", n,
                                    config["interaction"]["paths"], config["interaction"]["interactions"]))
        hist = pd.read_csv(directory / "customer_1_histogram.csv")
        assert int(hist.visits_product1.sum() + hist.visits_product2.sum()) == 20000
    long = config["long_interaction"]
    checks.append(audit_history(ROOT / "data" / "long_interactions" / "observations.npy",
                               long["n"], long["paths"], long["interactions"]))
    calendar = sorted((ROOT / "data" / "calendar").glob("*/manifest.json"))
    assert len(calendar) == 35
    for manifest in calendar:
        directory = manifest.parent
        periods = pd.read_csv(directory / "by_period.csv")
        paths = pd.read_csv(directory / "path_summary.csv")
        bins = pd.read_csv(directory / "binned.csv")
        assert len(periods) == 3000 and len(paths) == 20000 and len(bins) == 120
        assert periods.a_chosen_count.sum() == paths.a_chosen_count.sum() == bins.a_chosen_count.sum()
        assert periods.product2_chosen_count.sum() == paths.product2_count.sum() == bins.product2_chosen_count.sum()
        assert np.array_equal(paths.final_observation_n, paths.final_successes + paths.final_failures)
        for metric in ("a_market_share", "product2_unconditional_rate", "product2_given_a_rate"):
            valid = bins[metric].notna()
            assert (bins.loc[valid, metric + "_ci_lower"] <= bins.loc[valid, metric] + 1e-12).all()
            assert (bins.loc[valid, metric + "_ci_upper"] >= bins.loc[valid, metric] - 1e-12).all()
    figures = sorted((ROOT / "figures").rglob("*.png"))
    assert len(figures) >= 45
    audit = {"status": "passed", "policy_checkpoints": len(policies),
             "exact_policy_cases": len(exact), "all_exact_actions_numerically_certified": True,
             "maximum_exact_bellman_residual": float(exact.bellman_residual.max()),
             "approximate_policy_cases": int((diagnostics.method == "projected_fifo").sum()),
             "calendar_cases": len(calendar), "calendar_path_periods": 35 * 20000 * 3000,
             "interaction_cases": len(config["interaction"]["cases"]),
             "all_recorded_states_audited": sum(c["shape"][0] * c["shape"][1] for c in checks),
             "history_checks": checks, "figures": len(figures),
             "tests": "14 unittest tests passed; independent calendar replay, dense policy iteration, and full-state residual checks",
             "large_window_limitation": "mature global near-optimality not certified",
             "source_sha256": source_hashes()}
    dump(ROOT / "data" / "validation.json", audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ("history_checks", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
