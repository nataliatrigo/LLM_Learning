"""Audit the separate integer-capacity campaign with the original flow checks."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .audit import audit_case, _stored_certificate, require
from .run_experiment import ROOT, case_key, now, numerical_sources, write_json


def certificate(record, size):
    stationary = record["diagnostics"]["stationary"]
    if len(stationary.get("methods", [])) == 2 and "positive GTH" in stationary["methods"][0]:
        return _stored_certificate(record, size)
    require(stationary.get("verified") is True, "Iterative stationary certificate is unverified")
    require(stationary["solved_states"] == size, "Iterative stationary size differs from exact quotient")
    require(stationary["minimum_probability"] > 0 and stationary["minimum_exit_rate"] > 0,
            "Iterative stationary law lost a positive state")
    methods = stationary.get("methods", [])
    require(len(methods) >= 3 and any("reversed" in method.lower() for method in methods)
            and any("positive" in method.lower() for method in methods),
            "Missing independent iterative orders and positive starts")
    for name, limit in (("stability_l1", 1e-7), ("residual", 1e-11),
                        ("rate_weighted_residual", 1e-11), ("row_sum_error", 1e-12)):
        value = stationary[name]
        require(np.isfinite(value) and 0 <= value <= limit, f"Iterative certificate failed {name}")
    positive = stationary["positive_iteration"]
    require(positive["discrepancy"] <= 1e-7, "Positive independent starts disagree with candidate law")
    for name, limit in (("conditional_regret", 1e-7), ("bellman_residual", 1e-10),
                        ("optimal_value_difference", 1e-5), ("actual_value_residual", 1e-10),
                        ("sale_law_l1", 1e-7), ("market_partition_error", 1e-10)):
        value = record["diagnostics"][name]
        require(np.isfinite(value) and 0 <= value <= limit, f"Equilibrium certificate failed {name}")
    demand = record["diagnostics"]["demand"]
    require(demand["positive_tail_underflow_entries"] == 0, "Demand certificate has missing tails")
    for name, limit in (("raw_relative_selected_mass_error", 2e-7),
                        ("order_doubling_max_abs", 2e-9),
                        ("order_doubling_max_relative_tail", 2e-6), ("partition_error", 1e-11)):
        require(demand[name] <= limit, f"Demand certificate failed {name}")


def audit_campaign(max_sellers=5):
    from .capacity_sweep import CAMPAIGN, expected_cases

    sources = numerical_sources()
    cases = expected_cases(max_sellers=max_sellers)
    report = dict(created_utc=now(), status="running", verified=False,
                  sources=sources, expected_case_count=len(cases), passed_case_count=0,
                  capacities=list(range(1, 31)), max_sellers=max_sellers,
                  audit_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  shared_audit_source_sha256=sha256(Path(__file__).with_name("audit.py").read_bytes()).hexdigest(),
                  cases=[], failures=[], scope="Exact integer capacities; original independent flow, "
                  "belief, Bellman, payoff and metric reconstruction; dual-order positive GTH for small chains "
                  "or independently normalized GMRES plus two positive starts for large chains.")
    for capacity, p0, sellers, closure in cases:
        stem = case_key(capacity, p0, sellers, closure)
        try:
            record = json.loads((CAMPAIGN/"data/cases"/(stem+".json")).read_text())
            for filename, checksum in record.get("solver_sources", {}).items():
                path = Path(filename)
                path = path if path.is_absolute() else ROOT/path
                require(sha256(path.read_bytes()).hexdigest() == checksum,
                        f"Scalable solver source changed: {filename}")
            result = audit_case(capacity, p0, sellers, closure, sources=sources,
                                home=CAMPAIGN, certificate_checker=certificate)
            report["cases"].append(result)
        except Exception as error:
            report["failures"].append(dict(case=stem, error=repr(error)))
    report["passed_case_count"] = len(report["cases"])
    names = {name for case in report["cases"] for name in case["errors"]}
    report["worst_errors"] = {name: max(case["errors"][name] for case in report["cases"])
                              for name in sorted(names)}
    if sources != numerical_sources():
        report["failures"].append(dict(case="campaign", error="Numerical sources changed during audit"))
    report["verified"] = not report["failures"] and len(report["cases"]) == len(cases)
    report.update(status="complete_verified" if report["verified"] else "failed", completed_utc=now())
    write_json(CAMPAIGN/"data/validation.json", report)
    require(report["verified"], f"Capacity campaign audit failed with {len(report['failures'])} failure(s)")
    return report


if __name__ == "__main__":
    audit_campaign()
