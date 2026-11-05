"""Avoid redundant full certificates of distant fixed-p0 policy proposals.

The immutable multi_p0_run driver, models, solver and final certificates remain
unchanged. This optional search wrapper records its own SHA as a proposal input.
An exploratory complementarity screen only changes search order: every policy
accepted as an equilibrium still passes experiment.evaluate, including positive
stationarity and the independent reversed-order conditional law check.

Run with the same arguments as multi_p0_run, in an exclusively owned campaign.
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from oblivious.oe_distribution_fixed_p0.code import multi_p0_run as runner
from oblivious.oe_distribution_fixed_p0.code.model import _OutsideResponse
from oblivious.oe_distribution_fixed_p0.code.conditional_model import (
    FixedOutsideConditionalExperiment, _ConditionalResponse,
)

PRESCREEN_LIMIT = 1e-5


def helper_reference():
    path = Path(__file__).resolve()
    return {str(path.relative_to(runner.ROOT)): sha256(path.read_bytes()).hexdigest()}


def fast_solve_from_proposals(experiment, proposals, max_evaluations, log):
    """Screen proposals cheaply, preserving every exact final acceptance check."""
    inputs = helper_reference()
    conditional = isinstance(experiment, FixedOutsideConditionalExperiment)
    engine = (_ConditionalResponse if conditional else _OutsideResponse)(experiment)
    priority = {'fresh_same_p0_same_grid_marginal':0, 'fresh_monopoly':1,
                'same_p0_previous_grid':2, 'historical_p0_0p5_proposal':3}
    ordered = sorted(enumerate(proposals), key=lambda pair:(priority.get(pair[1][0], 4), pair[0]))
    unique, seen = [], set()
    for _, (name, old) in ordered:
        alpha = np.interp(experiment.grid, old['grid'], old['policy'])
        signature = alpha.tobytes()
        if signature not in seen:
            seen.add(signature)
            unique.append((name, alpha))
    attempts, screened = [], []

    def finish(candidate):
        # Full evaluate() or the established solve() supplied the certificate.
        # The exploratory response is never stored as the equilibrium law.
        candidate['attempts'] = attempts
        candidate.setdefault('search_metadata', {}).update(
            proposal_search_method='Exploratory complementarity prescreen followed by unchanged full certification/local solver',
            proposal_prescreen_limit=PRESCREEN_LIMIT,
            proposal_prescreen_full_certificate_required=True,
            proposal_prescreen_inputs=inputs,
            proposal_prescreen_records=[dict(item) for item in attempts if item['stage'] == 'exploratory_prescreen'],
            seed_is_proposal_only=True,
        )
        return candidate

    for name, alpha in unique:
        log('  exploratory proposal screen: '+name)
        score = float('inf')
        try:
            data = engine.calculate(alpha)
            advantage = np.asarray(data['advantage'])
            regret = float(np.max(np.maximum(advantage, 0)-alpha*advantage))
            complementarity = float(np.max(abs(alpha-np.clip(alpha+advantage, 0, 1))))
            score = max(regret, complementarity)
            require_full = bool(np.isfinite(score) and score <= PRESCREEN_LIMIT)
            attempts.append(dict(initialization=name, stage='exploratory_prescreen',
                verified=False, conditional_regret=regret, complementarity_residual=complementarity,
                full_certificate_attempted=require_full,
                explanation='Search diagnostic only; every accepted policy needs unchanged full certification'))
            log(f'    search regret={regret:.3g}, complementarity={complementarity:.3g}, full_certificate={require_full}')
            if require_full:
                candidate = experiment.evaluate(alpha)
                attempts.append(dict(initialization=name, stage='fresh_certificate',
                    verified=bool(candidate['verified']), dp=candidate['dp_diagnostics'],
                    conditional=candidate.get('conditional_diagnostics', {})))
                if candidate['verified']:
                    candidate['search_metadata'] = dict(method='Fresh full certificate after exploratory proposal prescreen',
                                                         unique=False, exhaustive=False)
                    return finish(candidate)
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            attempts.append(dict(initialization=name, stage='exploratory_prescreen', verified=False,
                                 error=str(error), full_certificate_attempted=False))
            log('    exploratory screen failed; retain proposal for unchanged local solver: '+str(error))
        screened.append((score, len(screened), name, alpha))
    best = None
    for _, _, name, alpha in sorted(screened, key=lambda item:(item[0], item[1])):
        log('  unchanged exact local solve: '+name)
        try:
            candidate = experiment.solve(initial_policy=alpha, prefer_local=True,
                max_evaluations=max_evaluations, progress=lambda message:log('    '+message))
            attempts.append(dict(initialization=name, stage='local_solve',
                verified=bool(candidate['verified']), dp=candidate['dp_diagnostics'],
                conditional=candidate.get('conditional_diagnostics', {}),
                search=dict(candidate.get('search_metadata', {}))))
            if best is None or candidate['dp_diagnostics']['conditional_regret'] < best['dp_diagnostics']['conditional_regret']:
                best = candidate
            if candidate['verified']:
                return finish(candidate)
        except (ArithmeticError, RuntimeError, ValueError, np.linalg.LinAlgError) as error:
            attempts.append(dict(initialization=name, stage='local_solve', verified=False, error=str(error)))
            log('    failed: '+str(error))
    error = ArithmeticError('No certified solution after exploratory proposal prescreen; attempts='+json.dumps(attempts))
    error.attempts = attempts
    error.best_candidate = finish(best) if best is not None else None
    raise error


def install():
    """Record this search implementation through the driver's checked input map."""
    if getattr(runner, '_fast_proposal_helper_installed', False):
        return
    original_baseline_seed = runner.baseline_seed

    def baseline_seed_with_search_provenance(rho, n, closure):
        seed, inputs = original_baseline_seed(rho, n, closure)
        inputs = dict(inputs, **helper_reference())
        return seed, inputs

    runner.baseline_seed = baseline_seed_with_search_provenance
    runner.solve_from_proposals = fast_solve_from_proposals
    runner._fast_proposal_helper_installed = True


def main():
    install()
    runner.main()


if __name__ == '__main__':
    main()
