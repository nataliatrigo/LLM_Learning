"""Audit Monte Carlo uncertainty against exact finite-window reward moments.

This separate read-only audit does not replace the requested 20 simulation paths.
It reveals when empirical path SEs miss rare events. For a per-sale reward h(x),
R_ij = E[reward * 1{next state=j} | state=i] = d_i*h_i*K_ij.
Conditional first and second cumulative moments obey v'=Pv+r and
w'=Pw+2Rv+r2. Binary block powering evaluates 100000 periods without simulation.
"""

from hashlib import sha256
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/monopoly-vs-competition-mpl")
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
import numpy as np
from oblivious.oe_p0.code.run_experiment import load_bundle, case_tag, write_json, write_csv
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.policy import transition_matrix


def moments(P, K, demand, reward_per_sale, at_start, periods):
    n = len(demand)
    h = np.asarray(reward_per_sale)
    weighted = (demand*h)[:, None]*K
    block = np.zeros((2*n+1, 2*n+1))
    block[:n, :n] = P
    block[:n, n:2*n] = 2*weighted
    block[:n, -1] = demand*h*h
    block[n:2*n, n:2*n] = P
    block[n:2*n, -1] = demand*h
    block[-1, -1] = 1
    powered = np.linalg.matrix_power(block, periods)
    mean = float(at_start @ powered[n:2*n, -1])
    second = float(at_start @ powered[:n, -1])
    variance = second-mean*mean
    if variance < -1e-6*max(second, 1):
        raise ArithmeticError("Invalid Markov reward variance")
    return mean, max(variance, 0.)


def main():
    # An independent binomial case checks the reward-moment recurrence.
    mean, variance = moments(np.ones((1,1)), np.ones((1,1)), np.array([.3]),
                             np.ones(1), np.ones(1), 1000)
    assert abs(mean-300) < 1e-10 and abs(variance-210) < 1e-7
    data = Path(__file__).resolve().parents[1]/"data"
    config = json.loads((data/"comparison.json").read_text())
    import csv
    with (data/"outcomes.csv").open() as stream:
        summaries = list(csv.DictReader(stream))
    rows = []
    for rho in config["rhos"]:
        b = load_bundle(data/"arrays"/case_tag(rho, config["grid"]))
        p = Parameters(**b["parameters"])
        g, a, d = b["grid"], b["policy"], b["monopoly"]["demand"]
        K = transition_matrix(g, np.ones(len(g)), a, p).toarray()
        P = transition_matrix(g, d, a, p).toarray()
        f = b["monopoly"]["finite_window"]
        initial = f["at_window_start"]
        T = config["simulation"]["periods"]
        count = config["simulation"]["replications"]
        q = f["outcomes"]["product2_conditional"]
        for metric, h in (("sale_rate", np.ones(len(g))), ("product2_calendar", a),
                          ("seller_profit", p.R-p.c1-(p.c2-p.c1)*a),
                          ("product2_conditional", a-q)):
            mean, variance = moments(P, K, d, h, initial, T)
            denominator = T if metric != "product2_conditional" else T*f["outcomes"]["sale_rate"]
            exact_se = float(np.sqrt(variance/count)/denominator)
            expected = f["outcomes"][metric]
            if metric != "product2_conditional":
                assert abs(mean/T-expected) < 1e-8
            saved = next(row for row in summaries if float(row["rho"]) == rho
                         and row["model"] == "monopoly" and row["metric"] == metric)
            empirical_se = float(saved["simulation_se"])
            estimate = float(saved["simulation_mean"])
            low, high = float(saved["simulation_ci_low"]), float(saved["simulation_ci_high"])
            rows.append(dict(rho=rho, metric=metric, finite_window_expected=expected,
                simulation_mean=estimate, empirical_se=empirical_se, exact_ensemble_se=exact_se,
                standardized_error_exact_se=(estimate-expected)/exact_se if exact_se else 0.,
                empirical_ci_contains_finite_expectation=low <= expected <= high,
                exact_se_over_empirical_se=exact_se/empirical_se if empirical_se else None,
                exact_se_interpretation="Delta-method SE of pooled ratio" if metric == "product2_conditional"
                    else "Exact standard deviation of 20-path mean under finite grid kernel"))
    write_csv(data/"finite_window_variance_audit.csv", rows)
    write_json(data/"finite_window_variance_audit.json", dict(verified=True,
        method="Block-matrix powers of first and second Markov additive reward moments",
        independent_unit_check="1000 Bernoulli(.3) rewards: mean 300, variance 210",
        caveat="Neither empirical Student-t intervals nor exact-variance normal approximations guarantee 95% coverage for skewed rare-event path distributions. Exact ensemble SE is a diagnostic, not a new confidence interval.",
        missing_empirical_intervals=[{k:row[k] for k in ("rho", "metric", "finite_window_expected", "simulation_mean", "exact_se_over_empirical_se")}
                                    for row in rows if not row["empirical_ci_contains_finite_expectation"]],
        maximum_absolute_error_in_exact_se_units=max(abs(r["standardized_error_exact_se"]) for r in rows),
        code_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
        numerical_manifest_sha256=sha256((data/"manifest.json").read_bytes()).hexdigest()))
    print("Finite variance audit complete; maximum error in true ensemble-SE units:",
          max(abs(r["standardized_error_exact_se"]) for r in rows))


if __name__ == "__main__":
    main()
