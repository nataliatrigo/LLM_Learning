"""Reconstruct and audit every saved exact-count oblivious experiment.

Checksums and frozen numerical sources bind each checkpoint to its stored
dual-order positive-GTH certificate.  This audit independently reconstructs
flows, every positive own-count belief row, Bellman incentives, discounted
payoffs, and exported economic metrics.  It never drops or floors rare states.
"""
from __future__ import annotations

from hashlib import sha256
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from oblivious.oe_p0.code.positive_stationary import positive_invariant
from .kernel import CountKernel
from .run_experiment import CONFIG, HOME, case_key, now, numerical_sources, write_json


def require(condition, message):
    if not condition:
        raise ArithmeticError(message)


def expected_cases():
    """Unique count models: repeated reference rhos share one exact solution."""
    result = [(capacity, p0, sellers, closure)
              for capacity in CONFIG["capacities"]
              for p0 in CONFIG["outside_options"]
              for sellers, closure in ((1, "monopoly"), (2, "marginal"), (2, "conditional"))]
    extension = CONFIG["seller_count_plot"]
    result.extend((extension["capacity"], p0, sellers, "marginal")
                  for p0 in extension["outside_options"]
                  for sellers in extension["sellers"] if sellers > 2)
    require(len(result) == len(set(result)), "Configuration repeats a unique count model")
    return result


def _maximum(values):
    return float(np.max(np.abs(values), initial=0.))


def _relative_error(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    scale = np.maximum(abs(actual), abs(expected))
    return _maximum(np.divide(abs(actual-expected), scale,
                              out=np.zeros_like(scale, dtype=float), where=scale > 0))


def _initial_mean(values, states, capacity, sellers):
    low, high = capacity//2, (capacity+1)//2
    lookup = {tuple(state): index for index, state in enumerate(states)}
    return sum(float(values[lookup[tuple(sorted(choice))]])
               for choice in product((low, high), repeat=sellers))/(2**sellers)


def _stored_certificate(record, size):
    diagnostics = record["diagnostics"]
    stationary = diagnostics["stationary"]
    require(stationary.get("verified") is True, "Stored stationary certificate is unverified")
    methods = stationary.get("methods", [])
    require(len(methods) == 2 and all("positive GTH" in method for method in methods)
            and "reversed" in methods[1], "Missing both positive-GTH elimination orders")
    require(stationary["solved_states"] == size, "Stored stationary state count differs")
    require(stationary["minimum_probability"] > 0 and stationary["minimum_exit_rate"] > 0,
            "Stored certificate lost a positive count state")
    limits = {"stability_l1": 1e-7, "conditional_demand_stability": 1e-7,
              "rate_weighted_residual": 1e-11, "residual": 1e-11, "row_sum_error": 1e-12}
    for metric, limit in limits.items():
        require(np.isfinite(stationary[metric]) and 0 <= stationary[metric] <= limit,
                f"Stored stationary certificate exceeds {metric} tolerance")
    limits = {"conditional_regret": 1e-7, "bellman_residual": 1e-10,
              "optimal_value_difference": 1e-5, "actual_value_residual": 1e-10,
              "sale_law_l1": 1e-7, "market_partition_error": 1e-10}
    for metric, limit in limits.items():
        require(np.isfinite(diagnostics[metric]) and 0 <= diagnostics[metric] <= limit,
                f"Stored equilibrium certificate exceeds {metric} tolerance")
    demand = diagnostics["demand"]
    require(demand["positive_tail_underflow_entries"] == 0, "Demand certificate has underflowed tails")
    for metric, limit in (("raw_relative_selected_mass_error", 2e-7),
                          ("order_doubling_max_abs", 2e-9),
                          ("order_doubling_max_relative_tail", 2e-6),
                          ("partition_error", 1e-11)):
        require(np.isfinite(demand[metric]) and 0 <= demand[metric] <= limit,
                f"Stored demand certificate exceeds {metric} tolerance")


def audit_case(capacity, p0, sellers, closure, sources=None, *, home=None,
               certificate_checker=None, kernel_factory=None, initial_evaluator=None):
    """Audit one immutable checkpoint without searching for an equilibrium."""
    sources = numerical_sources() if sources is None else sources
    stem = case_key(capacity, p0, sellers, closure)
    target = (HOME if home is None else Path(home))/"data/cases"/stem
    record = json.loads(target.with_suffix(".json").read_text())
    require(record.get("sources") == sources, "Checkpoint numerical sources differ from current sources")
    checkpoint_hash = sha256(target.with_suffix(".npz").read_bytes()).hexdigest()
    require(record.get("checkpoint_sha256") == checkpoint_hash, "Checkpoint checksum mismatch")
    summary = record["summary"]
    require(summary.get("verified") is True, "Checkpoint equilibrium is unverified")
    for name, value in (("capacity", capacity), ("p0", p0), ("sellers", sellers), ("closure", closure)):
        require(summary[name] == value, f"Checkpoint identity differs at {name}")
    with np.load(target.with_suffix(".npz"), allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    game = (CountKernel if kernel_factory is None else kernel_factory)(
        capacity, sellers, p0, order=CONFIG["quadrature_order"])
    size, n = game.size, capacity+1
    (_stored_certificate if certificate_checker is None else certificate_checker)(record, size)
    require(summary["count_states"] == n and summary["joint_states"] == size,
            "Summary state counts differ from exact count model")
    require(np.array_equal(arrays["states"], game.states), "Saved quotient state ordering differs")
    for name, shape in (("policy", (n,)), ("value", (n,)), ("advantage", (n,)),
                        ("demand", (n,)), ("joint", (size,)), ("actual_value", (size,)),
                        ("sale_stationary", (n,))):
        require(arrays[name].shape == shape and np.all(np.isfinite(arrays[name])),
                f"Invalid saved {name} array")
    policy, law, value = arrays["policy"], arrays["joint"], arrays["value"]
    require(np.all((policy >= 0)&(policy <= 1)), "Policy probabilities leave [0,1]")
    require(np.all(law > 0), "A positive joint count state is absent")
    require(abs(law.sum()-1) <= 1e-12, "Joint probabilities do not sum to one")
    parameters = CONFIG["economic_parameters"]
    p1, p2, gamma = (parameters[name] for name in ("p1", "p2", "gamma"))
    reward1, cost_gap = parameters["revenue"]-parameters["c1"], parameters["c2"]-parameters["c1"]
    probability = p1+(p2-p1)*policy
    off = game.off_diagonal(policy, p1, p2)
    rates = np.asarray(off.sum(axis=1)).ravel()
    require(np.all(rates > 0), "A joint state has no representable positive exit rate")
    jump = off.copy()
    jump.data /= np.repeat(rates, np.diff(jump.indptr))
    log_jump = np.log(law)+np.log(rates)
    jump_law = np.exp(log_jump-logsumexp(log_jump))
    require(np.all(jump_law > 0), "A positive jump count state is absent")
    jump_residual = float(abs(jump.T@jump_law-jump_law).sum())
    calendar_residual = float(abs(off.T@law-law*rates).sum())
    require(jump_residual <= 1e-11 and calendar_residual <= 1e-11,
            "Reconstructed stationary flow does not balance")
    own, sold = np.zeros(n), np.zeros(n)
    # Sum actual labelled-slot multiplicities independently of the stored
    # sparse own-marginal operators. Every rare positive row is retained.
    for slot in range(sellers):
        np.add.at(own, game.states[:, slot], law/sellers)
        np.add.at(sold, game.states[:, slot], law*game.selection[:, slot]/sellers)
    require(np.all(own > 0) and np.all(sold > 0), "A positive own-count or sale-count row is absent")
    conditional_l1, independent_jump_residual, independent_law_l1 = 0., 0., 0.
    if closure == "conditional":
        # Certification saves the law from reversed GTH elimination. Recompute
        # the original-order positive solve and compare the *entire* rival
        # distribution in every positive own-count row, not just its demand.
        other_jump = positive_invariant(jump, np.ones(size))
        require(np.all(other_jump > 0), "Independent conditional GTH lost a positive jump state")
        log_other = np.log(other_jump)-np.log(rates)
        other = np.exp(log_other-logsumexp(log_other))
        require(np.all(other > 0), "Independent conditional GTH lost a positive calendar state")
        independent_jump_residual = float(abs(jump.T@other_jump-other_jump).sum())
        independent_law_l1 = max(float(abs(other-law).sum()),
                                 float(abs(other_jump-jump_law).sum()))
        require(independent_jump_residual <= 1e-11 and independent_law_l1 <= 1e-7,
                "Fresh conditional GTH flow or global law disagrees with saved reverse-order law")
        other_own = np.zeros(n)
        for slot in range(sellers):
            np.add.at(other_own, game.states[:, slot], other/sellers)
        require(np.all(other_own > 0), "Independent conditional law lost an own-count row")
        coordinates = game.own_marginal.tocoo()
        row_differences = coordinates.data*abs(
            other[coordinates.col]/other_own[coordinates.row]
            - law[coordinates.col]/own[coordinates.row])
        row_l1 = np.bincount(coordinates.row, weights=row_differences, minlength=n)
        conditional_l1 = _maximum(row_l1)
        require(conditional_l1 <= 1e-7, "Independent full conditional rival rows disagree")
    if sellers == 1:
        demand = game.selection[:, 0].copy()
    elif closure == "conditional":
        demand = sold/own
    else:
        lookup = {tuple(state): index for index, state in enumerate(game.rivals)}
        demand = np.zeros(n)
        for slot in range(sellers):
            rival_ids = np.array([lookup[tuple(state)]
                                  for state in np.delete(game.states, slot, axis=1)])
            demand += (game.focal_demand[:, rival_ids]@law)/sellers
    require(np.all(demand > 0), "Reconstructed belief lost a positive count row")
    demand_relative_error = _relative_error(arrays["demand"], demand)
    require(demand_relative_error <= 2e-10, "Saved beliefs differ from reconstructed count rows")
    counts = np.arange(n)
    fraction = counts/capacity
    failure = fraction*value[np.maximum(counts-1, 0)]+(1-fraction)*value
    success = (1-fraction)*value[np.minimum(counts+1, capacity)]+fraction*value
    advantage = -cost_gap+gamma*(p2-p1)*(success-failure)
    updated = demand*(reward1-cost_gap*policy+gamma*(probability*success+(1-probability)*failure))
    updated += gamma*(1-demand)*value
    bellman_residual = _maximum(updated-value)
    conditional_regret = float(np.max(np.maximum(advantage, 0)-policy*advantage))
    advantage_error = _maximum(advantage-arrays["advantage"])
    require(bellman_residual <= 1e-10 and conditional_regret <= 1e-7 and advantage_error <= 1e-10,
            "Reconstructed Bellman incentives do not certify the saved policy")
    # Reconstruct the birth-death purchase-time invariant independently from
    # its adjacent detailed-balance ratios. Calendar selection cancels here.
    log_sale = np.zeros(n)
    for count in range(capacity):
        log_sale[count+1] = (log_sale[count]+np.log(probability[count])+np.log(capacity-count)
                            -np.log1p(-probability[count+1])-np.log(count+1))
    sale_law = np.exp(log_sale-logsumexp(log_sale))
    sale_rate = float(sold.sum())
    sale_error = float(abs(sold/sale_rate-sale_law).sum())
    require(sale_error <= 1e-7 and _relative_error(arrays["sale_stationary"], sale_law) <= 1e-10,
            "Saved sale-time distribution differs from independent birth-death law")
    mean_reward = np.mean(game.selection*(reward1-cost_gap*policy[game.states]), axis=1)
    actual_value = arrays["actual_value"]
    # Off-diagonal form preserves tiny state-change probabilities that can be
    # lost in a diagonal written as 1-exit_rate.
    actual_updated = mean_reward+gamma*(actual_value+off@actual_value-rates*actual_value)
    actual_residual = _maximum(actual_updated-actual_value)
    require(actual_residual <= 1e-10, "Actual joint-game discounted value has a large residual")
    product2_calendar = float(sold@policy)
    metrics = dict(sale_rate=sale_rate, outside_rate=float(law@game.outside),
                   product2_calendar=product2_calendar,
                   product2_conditional=product2_calendar/sale_rate,
                   profit_calendar=float(sold@(reward1-cost_gap*policy)),
                   value_initial=(_initial_mean if initial_evaluator is None else initial_evaluator)(
                       actual_value, game.states, capacity, sellers),
                   perceived_value_initial=float((value[capacity//2]+value[(capacity+1)//2])/2),
                   effective_rho=1-1/capacity)
    metric_errors = {name: _relative_error(summary[name], expected) for name, expected in metrics.items()}
    require(max(metric_errors.values()) <= 2e-8, "Exported economic metrics differ from checkpoint arrays")
    market_error = abs(metrics["outside_rate"]+sellers*sale_rate-1)
    require(market_error <= 1e-10, "Outside and seller sales do not partition the market")
    return dict(case=stem, passed=True, capacity=capacity, p0=p0, sellers=sellers, closure=closure,
                checkpoint_sha256=checkpoint_hash,
                record_sha256=sha256(target.with_suffix(".json").read_bytes()).hexdigest(),
                joint_states=size, minimum_joint_probability=float(law.min()),
                minimum_own_probability=float(own.min()),
                errors=dict(jump_stationary_l1=jump_residual, calendar_flow_l1=calendar_residual,
                            belief_relative=demand_relative_error, bellman=bellman_residual,
                            conditional_regret=conditional_regret, advantage=advantage_error,
                            sale_law_l1=sale_error, actual_value=actual_residual,
                            metric_relative=max(metric_errors.values()), market_partition=market_error,
                            full_conditional_rival_row_l1=conditional_l1,
                            fresh_conditional_jump_l1=independent_jump_residual,
                            fresh_conditional_law_l1=independent_law_l1),
                stored_dual_order_stability_l1=record["diagnostics"]["stationary"]["stability_l1"],
                stored_conditional_demand_stability=record["diagnostics"]["stationary"].get("conditional_demand_stability", 0.))


def audit_campaign():
    """Write a complete coverage report, raising before publication on failure."""
    sources = numerical_sources()
    cases = expected_cases()
    stems = {case_key(*case) for case in cases}
    report = dict(created_utc=now(), status="running", verified=False, sources=sources,
                  audit_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  expected_case_count=len(cases), passed_case_count=0, cases=[], failures=[],
                  scope="Exact mature-count finite models; all positive joint and own-count rows retained. "
                        "Stored dual-order positive-GTH certificates are bound by numerical source and checkpoint checksums; "
                        "flow, beliefs, Bellman incentives, sale laws, actual payoffs and metrics are reconstructed afresh. "
                        "Conditional models additionally recompute original-order positive GTH and compare the full rival "
                        "distribution in every own-count row against the saved reversed-order law.")
    actual = {path.stem for path in (HOME/"data/cases").glob("*.json")}
    for stem in sorted(actual-stems):
        report["failures"].append(dict(case=stem, error="Unexpected case outside configured unique coverage"))
    for case in cases:
        try:
            report["cases"].append(audit_case(*case, sources=sources))
        except Exception as error:
            report["failures"].append(dict(case=case_key(*case), error=repr(error)))
    report["passed_case_count"] = len(report["cases"])
    names = {name for case in report["cases"] for name in case["errors"]}
    report["worst_errors"] = {name: max(case["errors"][name] for case in report["cases"])
                              for name in sorted(names)}
    if numerical_sources() != sources:
        report["failures"].append(dict(case="campaign", error="Numerical sources changed during audit"))
    report["verified"] = not report["failures"] and len(report["cases"]) == len(cases)
    report["status"] = "complete_verified" if report["verified"] else "failed"
    report["completed_utc"] = now()
    write_json(HOME/"data/validation.json", report)
    require(report["verified"], f"Count campaign audit failed: {len(report['failures'])} failure(s); "
            f"see {HOME/'data/validation.json'}")
    return report


if __name__ == "__main__":
    audit_campaign()
