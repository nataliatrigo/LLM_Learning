"""Search-only small-face policy proposals, never equilibrium checkpoints.

Start from a checked historical policy, polish mixed and violating pure nodes,
and scan regret on every grid node after each round. The face is adaptive and
does not impose a policy threshold. Returned artifacts remain uncertified even
when exploratory regret is tiny; acceptance requires original evaluate(), its
positive stationary and reversed conditional checks, and actual values.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code import sweep201_run as worker
from oblivious.oe_distribution_fixed_p0.code import sweep201_quick_prescreen as quick
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment
from oblivious.oe_distribution_fixed_p0.code.multi_p0_endpoint_recovery import EndpointSafeConditionalExperiment


class SearchBudgetExceeded(RuntimeError):
    pass


def score(alpha,data):
    advantage = np.asarray(data['advantage'])
    return dict(conditional_regret=float(np.max(np.maximum(advantage,0)-alpha*advantage)),
                complementarity_residual=float(np.max(abs(alpha-np.clip(alpha+advantage,0,1)))))


def active_face_proposal(experiment,initial,*,maximum_calls=80,maximum_face=16,
                         target_score=2e-12,progress=lambda message:None):
    """Generate an uncertified alpha without calling any final certificate."""
    engine_type = quick.QuickConditionalResponse if isinstance(
        experiment,EndpointSafeConditionalExperiment) else quick.QuickMarginalResponse
    engine = engine_type(experiment)
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
        data = engine.calculate(rule)
        metrics = score(rule,data)
        objective = max(metrics.values())
        history.append(dict(call=engine.calls,**metrics,
                            quick_method=data.get('quick_prescreen',{})))
        if best is None or objective < best[0]:
            best = (objective,rule.copy(),data,metrics)
        return data,metrics

    status = 'unresolved_search'
    try:
        data,metrics = evaluate(alpha)
        for iteration in range(12):
            objective = max(metrics.values())
            progress(f'round={iteration} quick_score={objective:.3g} calls={engine.calls}')
            if objective <= target_score:
                status = 'quick_global_regret_passed_but_uncertified'
                break
            advantage = data['advantage']
            mixed = (alpha > 0)&(alpha < 1)
            violating = ((alpha == 0)&(advantage > target_score)
                         | (alpha == 1)&(advantage < -target_score))
            face = np.flatnonzero(mixed|violating)
            if not len(face) or len(face) > maximum_face:
                status = 'no_small_adaptive_face'
                rounds.append(dict(iteration=iteration,face=face.tolist(),reason=status))
                break
            # A new interpolated node often simply belongs to a pure region.
            # Check that possibility globally before numerical face polishing.
            pure = alpha.copy()
            pure[face] = (advantage[face] >= 0).astype(float)
            pure_data,pure_metrics = evaluate(pure)
            if max(pure_metrics.values()) <= target_score:
                alpha,data,metrics = pure,pure_data,pure_metrics
                rounds.append(dict(iteration=iteration,face=face.tolist(),
                                   method='Pure policy proposal with full-grid exploratory regret scan',
                                   **metrics))
                status = 'quick_global_regret_passed_but_uncertified'
                break
            fixed = alpha.copy()

            def residual(values):
                candidate = fixed.copy()
                candidate[face] = values
                calculated,_ = evaluate(candidate)
                return calculated['advantage'][face]

            initial_face = alpha[face].copy()
            initial_face[(initial_face == 0)|(initial_face == 1)] = .5
            result = least_squares(residual,np.clip(initial_face,1e-6,1-1e-6),
                bounds=(np.zeros(len(face)),np.ones(len(face))),jac='3-point',
                diff_step=1e-4,ftol=1e-13,xtol=1e-13,gtol=1e-13,
                max_nfev=max(1,(maximum_calls-engine.calls)//(2*len(face)+1)))
            alpha = fixed.copy()
            alpha[face] = result.x
            alpha[alpha < 1e-10] = 0
            alpha[alpha > 1-1e-10] = 1
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
                search_diagnostics=dict(method='Adaptive small face with bounded numerical Jacobian',
                    global_regret_checked_on_all_nodes=True,imposed_threshold_structure=False,
                    maximum_calls=maximum_calls,maximum_face=maximum_face,
                    target_score=target_score,engine_calls=engine.calls,
                    original_response_fallback_calls=engine.fallback_calls,
                    elapsed_seconds=time.monotonic()-started,history=history,rounds=rounds,
                    full_certificate_performed=False,
                    final_original_positive_and_reversed_certificates_required=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--p0',type=float,choices=worker.P0S,required=True)
    parser.add_argument('--rho',type=float,choices=worker.RHOS,default=.9)
    parser.add_argument('--n',type=int,choices=(51,101,201,251),required=True)
    parser.add_argument('--closure',choices=('marginal','conditional'),required=True)
    parser.add_argument('--maximum-calls',type=int,default=80)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    # Only historical campaign files are read. Active 201/251 cases are untouched.
    args.order,args.tol,args.dp_tol = 256,1e-7,1e-10
    print('Checking immutable warm proposal',flush=True)
    worker.runner._sweep201_original_baseline_seed = worker.runner.baseline_seed
    seed,inputs = worker.checked_same_p0_seed(args,201,args.closure)
    print('Constructing fresh target demand and game',flush=True)
    cls = EndpointSafeConditionalExperiment if args.closure == 'conditional' else FixedOutsideMarginalExperiment
    experiment = cls(n=args.n,params=Parameters(rho=args.rho),p0=args.p0,order=args.order,
                     tol=args.tol,dp_tol=args.dp_tol)
    initial = np.interp(experiment.grid,seed['grid'],seed['policy'])
    print('Searching adaptive face; full certification disabled',flush=True)
    result = active_face_proposal(experiment,initial,maximum_calls=args.maximum_calls,
                                 progress=lambda message:print(message,flush=True))
    scripts = [Path(__file__).resolve(),Path(quick.__file__).resolve(),Path(worker.__file__).resolve()]
    scripts.append(worker.runner.HOME/'code/multi_p0_endpoint_recovery.py')
    bundle = dict(configuration=experiment.configuration(),competition=result,
        proposal_script_sha256={str(p.relative_to(worker.runner.ROOT)):sha256(p.read_bytes()).hexdigest()
                                for p in scripts},
        source_fingerprints=worker.runner.source_fingerprints(),reference_input_sha256=inputs,
        explanation='UNCERTIFIED SEARCH PROPOSAL ONLY: original target evaluate and actual values must be freshly computed')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.with_suffix('.json').exists() or args.output.with_suffix('.npz').exists():
        parser.error('Existing proposal artifacts are protected; choose another output prefix')
    worker.runner.save_bundle(args.output,bundle)
    print(dict(p0=args.p0,rho=args.rho,n=args.n,closure=args.closure,status=result['proposal_status'],
               **result['quick_metrics'],calls=result['search_diagnostics']['engine_calls'],
               elapsed_seconds=result['search_diagnostics']['elapsed_seconds'],
               full_certificate_performed=False,output=str(args.output)),flush=True)


if __name__ == '__main__':
    main()
