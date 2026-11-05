"""Optional adaptive-face proposals gated by original full certification.

This layer changes candidate search only. Low exploratory regret is never a
certificate. Every accepted policy must pass experiment.evaluate(), including
its original positive stationary and reversed-order conditional checks. A failed
fresh certificate falls back to the frozen full local solver.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np

from oblivious.oe_distribution_fixed_p0.code import multi_p0_run as runner
from oblivious.oe_distribution_fixed_p0.code import sweep201_quick_prescreen as quick
from oblivious.oe_distribution_fixed_p0.code import sweep201_active_face_proposal as proposer
from oblivious.oe_distribution_fixed_p0.code.conditional_model import FixedOutsideConditionalExperiment


def helper_reference():
    paths = (Path(__file__).resolve(),Path(proposer.__file__).resolve())
    return {str(path.relative_to(runner.ROOT)):sha256(path.read_bytes()).hexdigest() for path in paths}


def fully_verified(experiment,result):
    verified = bool(result.get('verified',False)
                    and result.get('dp_diagnostics',{}).get('verified',False)
                    and result.get('stationary_diagnostics',{}).get('verified',False))
    if isinstance(experiment,FixedOutsideConditionalExperiment):
        diagnostics = result.get('conditional_diagnostics',{})
        verified = verified and bool(diagnostics.get('verified',False)
                                     and diagnostics.get('independent_order_verified',False))
    return verified


def solve_with_face(experiment,proposals,max_evaluations,log,regular_solve):
    conditional = isinstance(experiment,FixedOutsideConditionalExperiment)
    response = quick.QuickConditionalResponse if conditional else quick.QuickMarginalResponse
    engine = response(experiment)
    attempts,candidates,seen = [],[],set()
    priority = {'fresh_same_p0_same_grid_marginal':0,'fresh_monopoly':1,
                'same_p0_previous_grid':2,'historical_p0_0p5_proposal':3}
    ordered = sorted(enumerate(proposals),key=lambda item:(priority.get(item[1][0],4),item[0]))
    try:
        for _,(name,old) in ordered:
            alpha = np.interp(experiment.grid,old['grid'],old['policy'])
            key = alpha.tobytes()
            if key in seen:
                continue
            seen.add(key)
            data = engine.calculate(alpha)
            metrics = proposer.score(alpha,data)
            objective = max(metrics.values())
            attempts.append(dict(initialization=name,stage='face_proposal_prescreen',verified=False,
                                 **metrics,full_certificate_performed=False))
            candidates.append((objective,len(candidates),name,alpha))
            if objective <= 1e-10:
                break
        if candidates:
            objective,_,name,alpha = min(candidates,key=lambda item:(item[0],item[1]))
            proposal = None
            if objective > 1e-10:
                log('  adaptive-face proposal from '+name)
                proposal = proposer.active_face_proposal(experiment,alpha,
                    maximum_calls=min(35,max_evaluations),target_score=2e-12,
                    progress=lambda message:log('    '+message))
                alpha = proposal['policy']
                objective = max(proposal['quick_metrics'].values())
                attempts.append(dict(initialization=name,stage='adaptive_face_search',verified=False,
                    **proposal['quick_metrics'],search=proposal['search_diagnostics'],
                    full_certificate_performed=False))
            if objective <= 1e-10:
                log('  fresh original full certificate of adaptive-face policy (all laws/values discarded)')
                fresh = experiment.evaluate(alpha)
                verified = fully_verified(experiment,fresh)
                attempts.append(dict(initialization=name,stage='fresh_original_full_certificate',
                    verified=verified,dp=fresh.get('dp_diagnostics',{}),
                    conditional=fresh.get('conditional_diagnostics',{})))
                if verified:
                    fresh['attempts'] = attempts
                    fresh.setdefault('search_metadata',{}).update(
                        method=('Adaptive small-face proposal followed by unchanged original full certificate'
                                if proposal is not None else
                                'Checked policy proposal followed by unchanged original full certificate'),
                        adaptive_face_polishing_performed=proposal is not None,
                        seed_is_proposal_only=True,imported_stationary_or_value_reuse=False,
                        fresh_full_certificate=True,face_search_helper_sha256=helper_reference(),
                        exploratory_regret_is_not_a_certificate=True,
                        global_regret_checked_on_all_nodes=True,imposed_threshold_structure=False,
                        unique=False,exhaustive=False)
                    return fresh
                log('    fresh full certificate failed; use original local search')
    except (ArithmeticError,RuntimeError,ValueError,np.linalg.LinAlgError) as error:
        attempts.append(dict(stage='adaptive_face_search_or_certificate',verified=False,error=str(error)))
        log('    adaptive-face proposal failed; original local search: '+str(error))
    result = regular_solve(experiment,proposals,max_evaluations,log)
    result['attempts'] = attempts+result.get('attempts',[])
    result.setdefault('search_metadata',{})['adaptive_face_search_attempted'] = True
    return result


def install():
    if getattr(runner,'_adaptive_face_proposal_installed',False):
        return
    original_seed,regular_solve = runner.baseline_seed,runner.solve_from_proposals

    def seed_with_face_source(rho,n,closure):
        seed,inputs = original_seed(rho,n,closure)
        return seed,dict(inputs,**helper_reference())

    runner.baseline_seed = seed_with_face_source
    runner.solve_from_proposals = lambda experiment,proposals,max_evaluations,log: solve_with_face(
        experiment,proposals,max_evaluations,log,regular_solve)
    runner._adaptive_face_proposal_installed = True
