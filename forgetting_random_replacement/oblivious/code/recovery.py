"""Outside-option continuation for difficult marginal-OE searches.

This changes the search initialization only. Every returned solution passes
the original experiment's unchanged independent numerical certificate.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from scipy.optimize import root

from .model import CountExperiment


def solve_marginal_continuation(experiment, seed_policy, start_p0=.3, step=.01):
    """Continue a certified two-seller marginal policy down to target ``p0``.

    ``seed_policy`` must solve the same-capacity model at ``start_p0``; the
    function independently certifies it before continuation. Each warm search
    solves the natural-map complementarity equations directly with their
    analytic Jacobian, preserving pure-policy coordinates automatically.
    Failed steps are halved. The final result is certified by ``experiment``
    and includes search history, call count, and recovery provenance.
    """
    if experiment.sellers != 2 or experiment.closure != "marginal":
        raise ValueError("Outside-option continuation requires two-seller marginal OE")
    start_p0, step = float(start_p0), float(step)
    if (not np.isfinite(start_p0) or not experiment.p0 < start_p0 < 1
            or not np.isfinite(step) or step <= 0):
        raise ValueError("Require 0<=target p0<start_p0<1 and a positive finite step")
    alpha = np.asarray(seed_policy, dtype=float).copy()
    if (alpha.shape != (experiment.n,) or np.any(~np.isfinite(alpha))
            or np.any((alpha < 0) | (alpha > 1))):
        raise ValueError("Continuation seed must be a probability at every exact count state")

    def intermediate(p0):
        return CountExperiment(
            capacity=experiment.capacity, sellers=2, closure="marginal", p0=p0,
            primitives=replace(experiment.primitives, p0=p0),
            order=experiment.game.order, tolerance=experiment.tolerance)

    start = intermediate(start_p0)
    seed_certificate = start.certify(alpha)
    if not seed_certificate["verified"]:
        raise ArithmeticError("Outside-option continuation seed did not certify at start_p0")
    history = [{"stage": "outside-option continuation seed", "p0": start_p0,
                "verified": True,
                "conditional_regret": seed_certificate["diagnostics"]["conditional_regret"]}]
    current_p0, calls, attempted_steps = start_p0, 0, 0
    minimum_step = min(step, .00005)
    while current_p0 > experiment.p0 + 1e-12:
        target = max(experiment.p0, round(current_p0 - step, 12))
        if target >= current_p0:
            raise ArithmeticError("Outside-option continuation step lost floating-point resolution")
        stage = experiment if target == experiment.p0 else intermediate(target)
        best_regret = float("inf")

        def equations(raw):
            nonlocal best_regret
            data = stage.calculate(raw)
            best_regret = min(best_regret, data["regret"])
            return raw - np.clip(raw + data["advantage"], 0, 1)

        def jacobian(raw):
            data = stage.calculate(raw)
            derivative = data["jacobian"] * ((raw >= 0) & (raw <= 1))[None, :]
            inside = (raw + data["advantage"] > 0) & (raw + data["advantage"] < 1)
            return np.eye(stage.n) - inside[:, None] * (np.eye(stage.n) + derivative)

        previous_calls = stage.calls
        solved = root(equations, alpha, jac=jacobian, method="hybr",
                      options={"xtol": 1e-10, "maxfev": 100, "factor": .3})
        candidate = np.clip(solved.x, 0, 1)
        candidate[candidate < 1e-9] = 0
        candidate[candidate > 1 - 1e-9] = 1
        data = stage.calculate(candidate)
        stage_calls = stage.calls - previous_calls
        calls += stage_calls
        attempted_steps += 1
        passed = bool(data["regret"] <= experiment.tolerance)
        history.append({"stage": "outside-option continuation", "p0": target,
                        "from_p0": current_p0, "step": step,
                        "regret": data["regret"], "best_regret": min(best_regret, data["regret"]),
                        "calls": stage_calls, "accepted": passed,
                        "root_success": bool(solved.success), "root_message": str(solved.message)})
        if not passed:
            step /= 2
            if step < minimum_step or attempted_steps >= 1000:
                raise ArithmeticError(
                    f"Outside-option continuation failed at p0={target:g}; "
                    f"best regret={min(best_regret, data['regret']):g}")
            continue
        alpha, current_p0 = candidate, target
        if attempted_steps >= 1000:
            raise ArithmeticError("Outside-option continuation exceeded its step budget")

    final = experiment.certify(alpha)
    if not final["verified"]:
        raise ArithmeticError("Outside-option continuation final solution did not certify")
    return dict(final, history=history, search_calls=calls,
                recovery={"method": "Certified-seed outside-option continuation with analytic natural-map Jacobian",
                          "start_p0": start_p0, "target_p0": experiment.p0,
                          "requested_step": float(history[1]["step"]), "attempted_steps": attempted_steps,
                          "seed_certified": True, "final_certified": True})


__all__ = ["solve_marginal_continuation"]
