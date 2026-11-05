"""Additional OE search with movable pure and mixed decisions.

This module is deliberately separate from the historical campaign sources.
It solves the same mixed complementarity problem and uses the experiment's
unchanged full certificate.  An endpoint decision may enter the mixed set,
and an unsuccessful mixed face may be abandoned in either direction.
Neither a reputation threshold nor a pure-strategy restriction is imposed.
"""

import numpy as np

from oblivious.oe_distribution.code.model import _JointResponse


class _SearchBudgetExhausted(RuntimeError):
    pass


def _natural_map(policy, advantage):
    return policy - np.clip(policy + advantage, 0., 1.)


def _natural_jacobian(policy, advantage, derivative):
    target = policy + advantage
    interior = (target > 0.) & (target < 1.)
    identity = np.eye(len(policy))
    return identity - interior[:, None] * (identity + derivative)


def _score(policy, advantage):
    residual = _natural_map(policy, advantage)
    return float(np.max(np.abs(residual))), float(residual @ residual)


def _key(policy):
    return np.ascontiguousarray(policy, dtype=float).tobytes()


def _face_proposals(policy, advantage, tolerance):
    """Change a failed face without treating its old endpoints as constraints.

    Every mixed coordinate may become either endpoint.  Every currently
    suboptimal endpoint may switch or become mixed.  Coordinates need not be
    adjacent in reputation and no monotonicity is assumed.
    """
    mixed = np.flatnonzero((policy > 0.) & (policy < 1.))
    violating = np.flatnonzero(((policy == 0.) & (advantage > tolerance))
                              | ((policy == 1.) & (advantage < -tolerance)))
    # Explore the existing mixed face first.  A mixed root can be a locally
    # unhelpful branch even when the only failed inequality is at a neighbour.
    for index in mixed:
        for endpoint in (0., 1.):
            trial = policy.copy()
            trial[index] = endpoint
            yield trial, dict(kind="mixed to endpoint", index=int(index), value=endpoint)
    for index in violating:
        for value in (1. - policy[index], .5):
            trial = policy.copy()
            trial[index] = value
            yield trial, dict(kind="release violating endpoint", index=int(index), value=float(value))
    # A simultaneous revision is another candidate, never a policy-form
    # restriction.  Preserve exact indifferences, including their mixtures.
    trial = policy.copy()
    trial[advantage > tolerance] = 1.
    trial[advantage < -tolerance] = 0.
    yield trial, dict(kind="simultaneous action revision")


def solve_active_set(experiment, initial_policy, max_evaluations=150, progress=None):
    """Search exact mixed complementarity and return a fresh full certificate.

    The returned dictionary has the same candidate fields as
    ``experiment.evaluate``, plus ``history`` and ``search_metadata``.  Only
    that unchanged evaluator can mark the result verified.  The exploratory
    budget is a hard bound on distinct analytic-response calls; a final full
    certificate and at most six certificate-based Newton corrections are
    additional.  No simulations or files are written here.

    The method uses projected analytic Newton steps and a bounded search over
    alternative active faces.  All endpoint inequalities and all mixed-node
    indifferences are rechecked whenever a face changes.  Failure to find a
    certificate is reported as an unverified candidate, never nonexistence.
    """
    initial = np.asarray(initial_policy, dtype=float)
    if (initial.shape != experiment.grid.shape or not np.isfinite(initial).all()
            or np.any((initial < 0.) | (initial > 1.))):
        raise ValueError("Initial policy must be a finite probability on every grid node")
    if (not isinstance(max_evaluations, (int, np.integer)) or max_evaluations < 1):
        raise ValueError("Positive integer exploratory evaluation budget required")

    engine = _JointResponse(experiment)
    history, cache, certificates = [], {}, {}
    visited, pending = set(), {}
    alpha = initial.copy()
    best_alpha, best_score = initial.copy(), (np.inf, np.inf)
    candidate = None
    stop_reason = "exploratory budget exhausted"
    target = experiment.dp_tol * .1

    def report(message):
        if progress is not None:
            progress(message)

    def calculate(rule):
        nonlocal best_alpha, best_score
        key = _key(rule)
        if key not in cache:
            if engine.calls >= max_evaluations:
                raise _SearchBudgetExhausted()
            data = engine.calculate(rule)
            if (not np.isfinite(data["advantage"]).all()
                    or not np.isfinite(data["jacobian"]).all()):
                raise ArithmeticError("Nonfinite analytic response during active-set search")
            cache[key] = data
        result = cache[key]
        score = _score(rule, result["advantage"])
        if score < best_score:
            best_alpha, best_score = rule.copy(), score
        return result

    def certify(rule):
        key = _key(rule)
        if key not in certificates:
            certificates[key] = experiment.evaluate(rule)
        return certificates[key]

    try:
        for iteration in range(max_evaluations):
            data = calculate(alpha)
            advantage = data["advantage"]
            score = _score(alpha, advantage)
            visited.add(_key(alpha))
            # Enforce bounds selected by the *current exact natural map*,
            # rather than a probability cutoff.  This removes roundoff-size
            # mixtures only when their active equation is an endpoint.
            snapped = alpha.copy()
            snapped[alpha + advantage <= 0.] = 0.
            snapped[alpha + advantage >= 1.] = 1.
            if not np.array_equal(snapped, alpha):
                proposal = calculate(snapped)
                if _score(snapped, proposal["advantage"]) <= score:
                    alpha, data = snapped, proposal
                    advantage = data["advantage"]
                    score = _score(alpha, advantage)
                    visited.add(_key(alpha))

            free = np.flatnonzero(((alpha > 0.) & (alpha < 1.))
                                 | ((alpha == 0.) & (advantage > experiment.dp_tol))
                                 | ((alpha == 1.) & (advantage < -experiment.dp_tol)))
            history.append(dict(stage="review all complementarity conditions", iteration=iteration,
                                equation_error=score[0], free_indices=free.tolist(),
                                evaluations=engine.calls))
            report(f"active set {iteration+1}, residual={score[0]:.3g}, free nodes={len(free)}")
            if score[0] <= target:
                candidate = certify(alpha)
                if candidate["verified"]:
                    stop_reason = "fresh full certificate verified"
                else:
                    stop_reason = "exploratory root requires full-certificate correction"
                break

            residual = _natural_map(alpha, advantage)
            jacobian = _natural_jacobian(alpha, advantage, data["jacobian"])
            # SVD least squares remains defined when a proposed mixed face is
            # singular.  Projection permits an interior node to reach a bound.
            direction = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            projected = np.clip(alpha + direction, 0., 1.) - alpha
            descent = float(residual @ (jacobian @ projected))
            accepted = False
            if np.any(projected) and descent < 0.:
                for damping in (1., .5, .25, .125, .0625, .03125):
                    trial = np.clip(alpha + damping * direction, 0., 1.)
                    if np.array_equal(trial, alpha):
                        continue
                    probe = calculate(trial)
                    trial_score = _score(trial, probe["advantage"])
                    # A decrease of the squared natural-map residual is the
                    # line-search condition.  Maximum regret is certified later.
                    if trial_score[1] < score[1] * (1. - 1e-4 * damping):
                        history.append(dict(stage="projected mixed Newton", iteration=iteration,
                                            damping=damping, equation_error=trial_score[0]))
                        alpha, accepted = trial, True
                        break
            if accepted:
                continue

            # Newton can point out of the feasible cube on a false mixed
            # branch.  In that event actually change faces; do not freeze the
            # pure coordinates and repeat the same indifference solve.
            options = []
            for trial, description in _face_proposals(alpha, advantage, experiment.dp_tol):
                key = _key(trial)
                if key in visited or np.array_equal(trial, alpha):
                    continue
                probe = calculate(trial)
                trial_score = _score(trial, probe["advantage"])
                pending[key] = (trial_score, trial.copy(), description)
                options.append((trial_score, trial, description))
                if trial_score[0] <= target:
                    break
            if options:
                selected = min(options, key=lambda item: item[0])
                if selected[0] < score:
                    alpha = selected[1].copy()
                    history.append(dict(stage="active face pivot", iteration=iteration,
                                        equation_error=selected[0][0], **selected[2]))
                    continue
            available = [item for key, item in pending.items() if key not in visited]
            if not available:
                stop_reason = "no unexplored active-face proposal"
                break
            selected = min(available, key=lambda item: item[0])
            alpha = selected[1].copy()
            history.append(dict(stage="nonmonotone active face restart", iteration=iteration,
                                equation_error=selected[0][0], **selected[2]))
    except _SearchBudgetExhausted:
        pass
    except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
        history.append(dict(stage="active-set numerical failure", message=str(exc)))
        stop_reason = "numerical failure in exploratory response"

    if candidate is None or not candidate["verified"]:
        candidate = certify(best_alpha)

    # A sparse exploratory stationary law can differ by roundoff from the
    # independent positive invariant certificate.  Correct against the full
    # certified advantage while retaining the analytic Jacobian.  Every such
    # step can change endpoint status and receives another complete check.
    strong_polish = 0
    for _ in range(6):
        if candidate["verified"]:
            stop_reason = "fresh full certificate verified"
            break
        alpha = candidate["policy"].copy()
        advantage = candidate["advantage"]
        residual = _natural_map(alpha, advantage)
        if np.max(np.abs(residual)) > experiment.dp_tol * 100.:
            break
        try:
            data = calculate(alpha)
            jacobian = _natural_jacobian(alpha, advantage, data["jacobian"])
            direction = np.linalg.lstsq(jacobian, -residual, rcond=None)[0]
            trial = np.clip(alpha + direction, 0., 1.)
            trial[alpha + advantage <= 0.] = 0.
            trial[alpha + advantage >= 1.] = 1.
            if np.array_equal(trial, alpha):
                break
            fresh = certify(trial)
            strong_polish += 1
            history.append(dict(stage="full-certificate Newton correction",
                                equation_error=fresh["dp_diagnostics"]["complementarity_residual"],
                                verified=bool(fresh["verified"])))
            if (fresh["verified"] or fresh["dp_diagnostics"]["conditional_regret"]
                    < candidate["dp_diagnostics"]["conditional_regret"]):
                candidate = fresh
            else:
                break
        except _SearchBudgetExhausted:
            break

    if candidate["verified"]:
        stop_reason = "fresh full certificate verified"
    candidate = dict(candidate)
    candidate.update(history=history, search_metadata=dict(
        method="Adaptive active faces with analytic projected mixed complementarity Newton",
        evaluations=engine.calls + len(certificates), exploratory_evaluations=engine.calls,
        full_certifications=len(certificates), strong_polish_evaluations=strong_polish,
        max_evaluations=int(max_evaluations), stop_reason=stop_reason,
        all_nodes_rechecked=True, fixed_policy_threshold=False,
        pure_strategy_restriction=False, exhaustive=False, unique=False,
        budget_interpretation="Hard exploratory-call budget; full final certificates are additional",
        scope="Finite grid and numerical demand matrix; no continuous-state or MPE claim"))
    return candidate
