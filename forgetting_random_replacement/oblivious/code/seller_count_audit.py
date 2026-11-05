"""Independently reconstruct the saved large-N seller-count extension."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln, logsumexp

from .audit import audit_case, require
from .capacity_audit import certificate
from .run_experiment import ROOT, now, numerical_sources, write_json
from .seller_count_model import ExtendedCountKernel
from .seller_count_sweep import CAMPAIGN, CONFIG, expected_cases, extension_sources, case_key, digest


def initial_mean(values, states, capacity, sellers):
    """Sum binomial permutation-orbit weights instead of 2**N labeled starts."""
    low, high = capacity//2, (capacity+1)//2
    if low == high:
        index = np.flatnonzero(np.all(states == low, axis=1))
        require(len(index) == 1, "Initial midpoint state missing from the quotient")
        return float(values[index[0]])
    retained_high = np.arange(sellers+1)
    log_weights = (gammaln(sellers+1)-gammaln(retained_high+1)
                   -gammaln(sellers-retained_high+1)-sellers*np.log(2))
    weights = np.exp(log_weights-logsumexp(log_weights))
    selected = np.all((states == low)|(states == high), axis=1)
    selected_states = states[selected]
    high_counts = np.sum(selected_states == high, axis=1)
    require(len(high_counts) == sellers+1 and len(set(high_counts)) == sellers+1,
            "Initial binomial orbit states are incomplete")
    return float(weights[high_counts]@values[selected])


def audit_campaign():
    sources, extra = numerical_sources(), extension_sources()
    requested = expected_cases()
    expected = {case_key(*case): case for case in requested}
    report = dict(created_utc=now(), verified=False, status="running",
                  sources=sources, extension_sources=extra, cases=[], failures=[],
                  requested_case_count=len(expected), passed_case_count=0,
                  configuration=CONFIG, audit_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest(),
                  shared_audit_source_sha256=sha256(Path(__file__).with_name("audit.py").read_bytes()).hexdigest(),
                  scope="Fresh count kernel, stationary flows, beliefs, Bellman incentives, "
                        "purchase-time law, actual values and every metric; full rival correlations retained.")
    audit_inputs = dict(sources=sources, extension_sources=extra,
        audit=report["audit_source_sha256"], shared_audit=report["shared_audit_source_sha256"],
        certificate=sha256(Path(__file__).with_name("capacity_audit.py").read_bytes()).hexdigest())
    signature = sha256(json.dumps(audit_inputs, sort_keys=True).encode()).hexdigest()
    report["reused_audit_cases"] = 0
    present = {path.stem: path for path in (CAMPAIGN/"data/cases").glob("*.json")}
    for stem, path in sorted(present.items()):
        try:
            require(stem in expected, "Case lies outside the configured seller-count coverage")
            record = json.loads(path.read_text())
            require(record.get("sources") == sources, "Count model sources changed")
            if record.get("extension_sources"):
                require(record["extension_sources"] == extra, "Large-N extension sources changed")
            for filename, checksum in record.get("solver_sources", {}).items():
                require(sha256((ROOT/filename).read_bytes()).hexdigest() == checksum,
                        f"Scalable solver source changed: {filename}")
            record_hash, checkpoint_hash = digest(path), digest(path.with_suffix(".npz"))
            cache_path = CAMPAIGN/"data/audit_cache"/(stem+".json")
            cached = json.loads(cache_path.read_text()) if cache_path.exists() else None
            if (cached and cached.get("audit_signature") == signature
                    and cached["result"]["record_sha256"] == record_hash
                    and cached["result"]["checkpoint_sha256"] == checkpoint_hash):
                result = cached["result"]
                report["reused_audit_cases"] += 1
            else:
                result = audit_case(*expected[stem], sources=sources, home=CAMPAIGN,
                    certificate_checker=certificate, kernel_factory=ExtendedCountKernel,
                    initial_evaluator=initial_mean)
                write_json(cache_path, dict(audit_signature=signature, audit_inputs=audit_inputs, result=result))
            report["cases"].append(result)
        except Exception as error:
            report["failures"].append(dict(case=stem, error=repr(error)))
    accepted = {case["case"] for case in report["cases"]}
    missing = sorted(set(expected)-accepted)
    report["passed_case_count"] = len(accepted)
    report["coverage"] = dict(complete=not missing, requested=len(expected),
                              verified=len(accepted), missing_cases=missing,
                              note="Missing/failed cases are absent, never represented by zero values.")
    report["uncomputed"] = []
    for stem in missing:
        failure_path = CAMPAIGN/"data/failures"/(stem+".json")
        failure = json.loads(failure_path.read_text()) if failure_path.exists() else dict(case=stem, error="Not computed")
        report["uncomputed"].append(failure)
    names = {name for case in report["cases"] for name in case["errors"]}
    report["worst_errors"] = {name: max(case["errors"][name] for case in report["cases"])
                              for name in sorted(names)}
    require(sources == numerical_sources() and extra == extension_sources(),
            "Numerical sources changed during the audit")
    report["verified"] = bool(accepted) and not report["failures"]
    report.update(status=("complete_verified" if not missing else "verified_partial_coverage")
                  if report["verified"] else "failed", completed_utc=now())
    write_json(CAMPAIGN/"data/validation.json", report)
    require(report["verified"], "Saved seller-count outputs failed independent reconstruction")
    return report


if __name__ == "__main__":
    audit_campaign()
