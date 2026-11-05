"""Verify saved outputs against the opposite-replacement transition and policy."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .run_experiment import ROOT, CONFIG, MODEL, dump, load_solution, slug, source_hashes, sha256


def audit_history(path, solution, expected_shape):
    records = np.load(path,mmap_mode="r")
    assert records.shape == expected_shape, str(path)
    counts = np.zeros(records.shape[1],dtype=np.int64)
    for t in range(len(records)):
        m = min(t,solution.n)
        row = records[t]
        if not np.array_equal(row["S_before"],counts) or not np.all(row["m_before"] == m):
            raise AssertionError(f"incorrect pre-choice state: {path}, interaction {t+1}")
        prescribed = solution.advantages[m][counts] > solution.action_tolerance
        if not np.array_equal(row["product2"],prescribed):
            raise AssertionError(f"incorrect policy choice: {path}, interaction {t+1}")
        if not np.allclose(row["demand"],solution.demands[m][counts],atol=3e-8,rtol=1e-7):
            raise AssertionError(f"incorrect demand: {path}, interaction {t+1}")
        y = row["success"].astype(np.int64)
        if m < solution.n:
            counts += y
        else:
            counts = np.clip(counts+2*y-1,0,solution.n)
    return {"path": str(path.relative_to(ROOT)), "shape": list(records.shape),
            "all_count_transitions_match": True, "all_policy_choices_match": True,
            "all_demands_match": True}


def main():
    config = json.loads(CONFIG.read_text())
    assert config["model"] == MODEL
    checkpoints = sorted((ROOT / "data" / "policies").glob("*.npz"))
    assert len(checkpoints) == 37
    diagnostics = pd.read_csv(ROOT / "data" / "solver_diagnostics.csv")
    assert len(diagnostics) == 37
    assert (diagnostics.method == "exact_opposite_replacement").all()
    assert diagnostics.all_actions_certified.astype(bool).all()
    assert (diagnostics.uncertified_action_states == 0).all()
    for path in checkpoints:
        solution = load_solution(path)
        assert solution.values[-1].shape == (solution.n+1,)
        p, s = solution.primitives, np.arange(solution.n+1)
        v, d = solution.values[-1], solution.demands[-1]
        vs, vf = v[np.minimum(s+1,solution.n)], v[np.maximum(s-1,0)]
        q1 = p.net_rewards[0]+p.gamma*(p.p1*vs+(1-p.p1)*vf)
        q2 = p.net_rewards[1]+p.gamma*(p.p2*vs+(1-p.p2)*vf)
        residual = np.max(np.abs(d/(1-p.gamma*(1-d))*np.maximum(q1,q2)-v))
        assert residual <= 1.1e-10
        gap = q2-q1
        np.testing.assert_allclose(gap,solution.advantages[-1],atol=1e-12)
    histories = []
    for p0,n in config["interaction"]["cases"]:
        label = slug(p0,n)
        directory = ROOT / "data" / "interactions" / label
        solution = load_solution(ROOT / "data" / "policies" / f"{label}.npz")
        histories.append(audit_history(directory / "observations.npy",solution,
            (config["interaction"]["interactions"],config["interaction"]["paths"])))
        hist = pd.read_csv(directory / "customer_1_histogram.csv")
        assert int(hist.visits_product1.sum()+hist.visits_product2.sum()) == config["interaction"]["interactions"]
    long = config["long_interaction"]
    solution = load_solution(ROOT / "data" / "policies" / f"{slug(long['p0'],long['n'])}.npz")
    histories.append(audit_history(ROOT / "data" / "long_interactions" / "observations.npy",solution,
                                 (long["interactions"],long["paths"])))
    calendar = sorted((ROOT / "data" / "calendar").glob("*/manifest.json"))
    assert len(calendar) == 35
    for manifest_path in calendar:
        manifest = json.loads(manifest_path.read_text())
        assert manifest["signature"]["model"] == MODEL
        directory = manifest_path.parent
        periods = pd.read_csv(directory / "by_period.csv")
        paths = pd.read_csv(directory / "path_summary.csv")
        bins = pd.read_csv(directory / "binned.csv")
        assert len(periods) == 3000 and len(paths) == 20000 and len(bins) == 120
        assert periods.a_chosen_count.sum() == paths.a_chosen_count.sum() == bins.a_chosen_count.sum()
        assert periods.product2_chosen_count.sum() == paths.product2_count.sum() == bins.product2_chosen_count.sum()
        assert np.array_equal(paths.final_observation_n,paths.final_successes+paths.final_failures)
        for frame in (periods,paths,bins):
            assert (frame.method == "exact_opposite_replacement").all()
        for metric in ("a_market_share","product2_unconditional_rate","product2_given_a_rate"):
            valid = bins[metric].notna()
            assert (bins.loc[valid,metric+"_ci_lower"] <= bins.loc[valid,metric]+1e-12).all()
            assert (bins.loc[valid,metric+"_ci_upper"] >= bins.loc[valid,metric]-1e-12).all()
    figures = sorted((ROOT / "figures").rglob("*.png"))
    assert len(figures) == 47
    assert not list((ROOT / "figures").glob("*all_histories*"))
    old = ROOT / "archive" / "fifo_previous" / "figures" / "individual_states" / "individual_states_p0_0.5_n_20.png"
    new = ROOT / "figures" / "individual_states" / old.name
    assert sha256(old) != sha256(new)
    audit = {"status": "passed", "model": MODEL, "exact_policy_cases": len(checkpoints),
        "all_actions_numerically_certified": True,
        "maximum_bellman_residual": float(diagnostics.bellman_residual.max()),
        "calendar_cases": len(calendar), "calendar_path_periods": 35*20000*3000,
        "interaction_cases": len(config["interaction"]["cases"]),
        "all_recorded_states_and_choices_audited": sum(h["shape"][0]*h["shape"][1] for h in histories),
        "history_checks": histories, "figures": len(figures),
        "old_and_new_main_plot_differ": True, "previous_model": "archive/fifo_previous",
        "source_sha256": source_hashes()}
    dump(ROOT / "data" / "validation.json",audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ("source_sha256","history_checks")},indent=2),flush=True)


if __name__ == "__main__":
    main()
