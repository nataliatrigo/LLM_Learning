"""Recovery-only policy proposals using the projected complementarity map.

The nonlinear residual z-clip(z+advantage,0,1) permits both strict pure actions
and interior indifference. It never demands zero advantage at a correct strict
bound. Only globally violating nodes enter an adaptive face. This routine does
not certify anything: original full positive/reversed-law checks remain required.
No active or historical model, driver, helper or checkpoint is modified.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import time

import numpy as np
from scipy.optimize import root

from oblivious.oe_distribution_fixed_p0.code import sweep201_quick_prescreen as quick
from oblivious.oe_distribution_fixed_p0.code.conditional_model import FixedOutsideConditionalExperiment
from oblivious.oe_distribution_fixed_p0.code.sweep201_active_face_proposal import score,SearchBudgetExceeded


def helper_reference():
    from oblivious.oe_distribution_fixed_p0.code.run_experiment import ROOT
    path = Path(__file__).resolve()
    return {str(path.relative_to(ROOT)):sha256(path.read_bytes()).hexdigest()}


def active_face_proposal(experiment,initial,*,maximum_calls=80,maximum_face=16,
                         target_score=2e-12,progress=lambda message:None):
    response = quick.QuickConditionalResponse if isinstance(
        experiment,FixedOutsideConditionalExperiment) else quick.QuickMarginalResponse
    engine = response(experiment)
    alpha = np.clip(np.asarray(initial,dtype=float),0,1).copy()
    if alpha.shape != experiment.grid.shape or not np.isfinite(alpha).all():
        raise ValueError('Initial proposal must be a finite policy on the target grid')
    started = time.monotonic()
    history,rounds = [],[]
    best = None

    def evaluate(rule):
        nonlocal best
        if engine.calls >= maximum_calls:
            raise SearchBudgetExceeded('Exploratory response budget exhausted')
        calculated = engine.calculate(rule)
        metrics = score(rule,calculated)
        objective = max(metrics.values())
        history.append(dict(call=engine.calls,**metrics,
                            quick_method=calculated.get('quick_prescreen',{})))
        if best is None or objective < best[0]:
            best = (objective,rule.copy(),calculated,metrics)
        return calculated,metrics

    status = 'unresolved_search'
    try:
        data,metrics = evaluate(alpha)
        pure_seen = {alpha.tobytes()}
        best_pure = ((max(metrics.values()),alpha.copy(),data,metrics)
                     if np.all((alpha == 0)|(alpha == 1)) else None)
        for _ in range(3):
            if max(metrics.values()) <= target_score:
                break
            pure = (data['advantage'] >= 0).astype(float)
            if pure.tobytes() in pure_seen:
                break
            pure_seen.add(pure.tobytes())
            data,metrics = evaluate(pure)
            alpha = pure
            pure_objective = max(metrics.values())
            if best_pure is None or pure_objective < best_pure[0]:
                best_pure = (pure_objective,pure.copy(),data,metrics)
        if best_pure is not None and max(metrics.values()) > target_score:
            _,alpha,data,metrics = best_pure
        for iteration in range(12):
            objective = max(metrics.values())
            progress(f'round={iteration} natural_map_quick_score={objective:.3g} calls={engine.calls}')
            if objective <= target_score:
                status = 'quick_global_regret_passed_but_uncertified'
                break
            natural = alpha-np.clip(alpha+data['advantage'],0,1)
            face = np.flatnonzero(abs(natural) > target_score)
            if not len(face) or len(face) > maximum_face:
                status = 'no_small_adaptive_face'
                rounds.append(dict(iteration=iteration,face=face.tolist(),reason=status))
                break
            fixed = alpha.copy()

            def residual(z):
                trial = fixed.copy()
                trial[face] = np.clip(z,0,1)
                calculated,_ = evaluate(trial)
                # Raw z outside the box must be driven back onto the boundary.
                # The model itself sees only the valid clipped trial policy.
                return z-np.clip(z+calculated['advantage'][face],0,1)

            initial_face = alpha[face].copy()
            initial_face[(initial_face == 0)|(initial_face == 1)] = .5
            result = root(residual,initial_face,method='hybr',
                options=dict(xtol=1e-10,eps=1e-10,maxfev=max(1,maximum_calls-engine.calls)))
            alpha = fixed.copy()
            alpha[face] = np.clip(result.x,0,1)
            data,metrics = evaluate(alpha)
            rounds.append(dict(iteration=iteration,face=face.tolist(),success=bool(result.success),
                               message=result.message,nfev=int(result.nfev),**metrics))
    except SearchBudgetExceeded:
        status = 'exploratory_response_budget_exhausted'
    if best is None:
        raise ArithmeticError('No finite exploratory policy could be evaluated')
    objective,alpha,data,metrics = best
    if objective <= target_score:
        status = 'quick_global_regret_passed_but_uncertified'
    return dict(grid=experiment.grid.copy(),policy=alpha,verified=False,candidate_verified=False,
                proposal_status=status,quick_metrics=metrics,
                search_diagnostics=dict(method='Adaptive globally violating face with projected natural-map root',
                    global_regret_checked_on_all_nodes=True,imposed_threshold_structure=False,
                    strict_bound_advantages_need_not_be_zero=True,
                    pure_improvement_sweeps=3,pure_policy_cycle_detection=True,
                    maximum_calls=maximum_calls,maximum_face=maximum_face,target_score=target_score,
                    engine_calls=engine.calls,original_response_fallback_calls=engine.fallback_calls,
                    elapsed_seconds=time.monotonic()-started,history=history,rounds=rounds,
                    full_certificate_performed=False,
                    final_original_positive_and_reversed_certificates_required=True))


def install():
    """Opt in only through a new entrypoint that records helper_reference()."""
    from oblivious.oe_distribution_fixed_p0.code import sweep201_active_face_proposal as old
    old.active_face_proposal = active_face_proposal
