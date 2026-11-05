"""Read saved mixed-policy search evaluations without solving the model again."""

import numpy as np


def _certified(selected):
    return bool(selected and selected.get("candidate_verified")
                and selected.get("verified")
                and selected.get("dp_diagnostics", {}).get("verified")
                and selected.get("stationary_diagnostics", {}).get("verified")
                and selected.get("monopoly_certificate", {}).get("verified"))


def selected_solver_trace(case):
    """Return the actual saved evaluations on the selected candidate's branch.

    The first five evaluations scan an optimal-action tie face. Subsequent
    evaluations refine its mixing weight at an already-localized p0. Their
    order and residuals are preserved, including increases and exact zeros.
    A separate last row records the full two-seller stationary certificate.
    These are function evaluations, not a trajectory of repeated p0 updates.
    """
    selected = case.get("selected")
    result = dict(status="unavailable", rows=[], scan_evaluations=0,
                  tolerance=float(case.get("search_summary", {}).get(
                      "consistency_tolerance", 1e-7)),
                  verified=_certified(selected),
                  p0=None if selected is None else float(selected["p0"]))
    if selected is None:
        return result
    if not result["verified"]:
        result["status"] = "unverified"
        return result

    policy = np.asarray(selected["policy"], dtype=float)
    mixed = bool(np.any((policy > 0) & (policy < 1)))
    result["status"] = "missing mixed trace" if mixed else "pure endpoint"
    weight = selected.get("mixture_search_weight")
    if mixed and weight is not None:
        for branch in case.get("mixing_branches", []):
            if abs(float(branch["p0"]) - selected["p0"]) > 1e-10:
                continue
            matches = any(candidate.get("verified")
                          and abs(float(candidate["weight"]) - weight) <= 1e-9
                          for candidate in branch.get("candidates", []))
            indices = np.asarray(branch["tied_indices"], dtype=int)
            start = np.asarray(branch["start_probabilities"], dtype=float)
            end = np.asarray(branch["end_probabilities"], dtype=float)
            probabilities = (1 - weight) * start + weight * end
            if (not matches or probabilities.shape != policy[indices].shape
                    or not np.allclose(probabilities, policy[indices], rtol=0, atol=1e-8)):
                continue
            trials = branch.get("trials", [])
            if (not trials or not all(np.isfinite(float(trial["G"])) for trial in trials)):
                continue
            # The saved search starts every branch with these five scan points.
            # Refuse an incomplete/different history instead of inventing steps.
            if (len(trials) < 5 or not np.allclose(
                    [trial["weight"] for trial in trials[:5]], np.linspace(0, 1, 5),
                    rtol=0, atol=1e-14)):
                continue
            result["status"] = "mixed branch"
            result["scan_evaluations"] = 5
            for index, trial in enumerate(trials):
                result["rows"].append(dict(
                    evaluation=index + 1,
                    stage="face scan" if index < 5 else "mixing refinement",
                    p0=float(trial["p0"]), F=float(trial["F"]), G=float(trial["G"]),
                    residual=abs(float(trial["G"])), weight=float(trial["weight"]),
                    exploratory=True))
            break

    result["rows"].append(dict(
        evaluation=len(result["rows"]) + 1, stage="full-game certificate",
        p0=float(selected["p0"]), F=float(selected["F"]), G=float(selected["G"]),
        residual=abs(float(selected["G"])),
        weight=None if weight is None else float(weight), exploratory=False))
    return result
