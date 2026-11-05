"""Audit saved integer counts, random removals, choices and calendar totals."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from .run_experiment import ROOT, CONFIG, MODEL, dump, load_solution, slug, source_hashes, initial_state


def audit_history(path,solution,expected_shape,initial):
    records = np.load(path,mmap_mode="r")
    assert records.shape == expected_shape,str(path)
    assert np.all(records["S_before"][0] == initial["S"])
    error_sum,variance_sum,removals = 0.0,0.0,0
    for start in range(0,len(records),10000):
        stop = min(start+10000,len(records))
        sample = records[start:stop]
        times = np.arange(start,stop)
        m = np.minimum(initial["m"]+times,solution.n)[:,None]
        assert np.all(sample["m_before"] == m)
        s = sample["S_before"].astype(np.int64)
        assert np.all((s >= 0)&(s <= m))
        actions,demand = solution.evaluate(m,s)
        assert np.array_equal(sample["product2"],actions)
        np.testing.assert_allclose(sample["demand"],demand,atol=3e-8,rtol=1e-7)
        removed = sample["removed_success"].astype(np.int64)
        full = np.broadcast_to(m == solution.n,s.shape)
        assert np.all(removed[~full] == -1)
        assert np.all((removed[full] == 0)|(removed[full] == 1))
        assert np.all(removed[full&(s == 0)] == 0)
        assert np.all(removed[full&(s == solution.n)] == 1)
        transitions = min(stop,len(records)-1)-start
        next_s = s[:transitions]+sample["success"][:transitions].astype(np.int64)-np.maximum(removed[:transitions],0)
        assert np.array_equal(next_s,records["S_before"][start+1:start+transitions+1])
        probabilities = s[full]/solution.n
        error_sum += float(np.sum(removed[full]-probabilities))
        variance_sum += float(np.sum(probabilities*(1-probabilities)))
        removals += int(full.sum())
    return {"path":str(path.relative_to(ROOT)),"shape":list(records.shape),
            "all_count_transitions_and_choices_match":True,"recorded_removals":removals,
            "removal_martingale_z":error_sum/np.sqrt(variance_sum) if variance_sum else 0.0}


def main():
    config = json.loads(CONFIG.read_text())
    diagnostics = pd.read_csv(ROOT / "data" / "solver_diagnostics.csv")
    assert len(diagnostics) == 37
    assert (diagnostics.method == "exact_random_replacement").all()
    assert diagnostics.all_actions_certified.astype(bool).all()
    assert (diagnostics.uncertified_action_states == 0).all()
    for path in sorted((ROOT / "data" / "policies").glob("*.npz")):
        sol = load_solution(path)
        s,p = np.arange(sol.n+1),sol.primitives
        v,d = sol.values[-1],sol.demands[-1]
        r = s/sol.n
        vs = (1-r)*v[np.minimum(s+1,sol.n)]+r*v
        vf = r*v[np.maximum(s-1,0)]+(1-r)*v
        q1 = p.net_rewards[0]+p.gamma*(p.p1*vs+(1-p.p1)*vf)
        q2 = p.net_rewards[1]+p.gamma*(p.p2*vs+(1-p.p2)*vf)
        assert np.max(np.abs(d/(1-p.gamma*(1-d))*np.maximum(q1,q2)-v)) <= 1.1e-10
        np.testing.assert_allclose(q2-q1,sol.advantages[-1],atol=1e-12)
    histories = []
    for p0,n in config["interaction"]["cases"]:
        directory = ROOT / "data" / "interactions" / slug(p0,n)
        solution = load_solution(ROOT / "data" / "policies" / f"{slug(p0,n)}.npz")
        histories.append(audit_history(directory / "observations.npy",solution,
            (config["interaction"]["interactions"],config["interaction"]["paths"]),initial_state(config,n)))
        histogram = pd.read_csv(directory / "customer_1_histogram.csv")
        assert histogram.visits_product1.sum()+histogram.visits_product2.sum() == config["interaction"]["interactions"]
    settings = config["long_interaction"]
    sol = load_solution(ROOT / "data" / "policies" / f"{slug(settings['p0'],settings['n'])}.npz")
    histories.append(audit_history(ROOT / "data" / "long_interactions" / "observations.npy",sol,
        (settings["interactions"],settings["paths"]),initial_state(config,settings["n"])))
    calendar = list((ROOT / "data" / "calendar").glob("*/manifest.json"))
    assert len(calendar) == 35
    for path in calendar:
        manifest = json.loads(path.read_text())
        assert manifest["signature"]["model"] == MODEL
        directory = path.parent
        periods,paths,bins = [pd.read_csv(directory/name) for name in
                             ("by_period.csv","path_summary.csv","binned.csv")]
        assert len(periods) == 3000 and len(paths) == 20000 and len(bins) == 120
        assert periods.a_chosen_count.sum() == paths.a_chosen_count.sum() == bins.a_chosen_count.sum()
        assert periods.product2_chosen_count.sum() == paths.product2_count.sum() == bins.product2_chosen_count.sum()
        assert np.array_equal(paths.final_observation_n,paths.final_successes+paths.final_failures)
        assert np.array_equal(paths.final_remembered_successes,
            paths.initial_successes+paths.final_successes-paths.removed_success_count)
        assert np.array_equal(paths.final_window_length,
            paths.initial_window_length+paths.final_observation_n-paths.replacement_count)
        assert (paths.discounted_profit >= 0).all()
        for frame in (periods,paths,bins):
            assert (frame.method == "exact_random_replacement").all()
    transitions = pd.read_csv(ROOT / "data" / "conditional_transitions.csv")
    np.testing.assert_allclose(transitions.groupby(["n","S","new_success"]).probability.sum(),1.0)
    example = transitions[(transitions.n == 20)&(transitions.S == 18)&(transitions.new_success == 0)]
    actual = {int(k):float(v) for k,v in zip(example.next_S,example.probability)}
    np.testing.assert_allclose([actual[17],actual[18]],[.9,.1])
    profit_checks = pd.read_csv(ROOT / "data" / "calendar_summary.csv")
    maximum_profit_z = float(profit_checks.profit_z_score.abs().max())
    assert maximum_profit_z < 6.0, "Monte Carlo profits disagree with Bellman values"
    figures = list((ROOT / "figures").rglob("*.png"))
    assert len(figures) == 48
    audit = {"status":"passed","model":MODEL,"initialization":config["initialization"],
        "exact_policy_cases":len(diagnostics),"all_actions_numerically_certified":True,
        "maximum_bellman_residual":float(diagnostics.bellman_residual.max()),
        "calendar_cases":len(calendar),"calendar_path_periods":35*20000*3000,
        "recorded_states_and_choices_checked":sum(h["shape"][0]*h["shape"][1] for h in histories),
        "history_checks":histories,"figures":len(figures),"user_example_checked":actual,
        "maximum_absolute_profit_z_score":maximum_profit_z,
        "source_sha256":source_hashes()}
    # JSON object keys for the concrete example are integer counts.
    dump(ROOT / "data" / "validation.json",audit)
    print(json.dumps({k:v for k,v in audit.items() if k not in ("source_sha256","history_checks")},indent=2),flush=True)


if __name__ == "__main__":
    main()
