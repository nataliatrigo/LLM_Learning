"""Optional search-only advantage screen without unused policy Jacobians.

The symmetric joint kernel and own-state policy evaluation are exact model
objects. A sparse pinned-coordinate jump-law solve avoids a dense normalization
row. Unsafe exploratory laws fall back to the original response calculation.
Final equilibrium acceptance remains the original independent full certificate.
This module is not installed by the frozen 201/251 worker.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import splu

from oblivious.oe_distribution_fixed_p0.code.model import _OutsideResponse
from oblivious.oe_distribution_fixed_p0.code.conditional_model import _ConditionalResponse
from oblivious.oe_p0.code.policy import evaluate_policy


def helper_reference():
    from oblivious.oe_distribution_fixed_p0.code.run_experiment import ROOT
    path = Path(__file__).resolve()
    return {str(path.relative_to(ROOT)):sha256(path.read_bytes()).hexdigest()}


class _QuickResponse:
    conditional = False
    original_response = _OutsideResponse

    def __init__(self,experiment):
        self.experiment = experiment
        n = experiment.n
        self.i,self.j = np.triu_indices(n)
        self.representatives = self.i*n+self.j
        groups = np.empty((n,n),dtype=int)
        groups[self.i,self.j] = np.arange(len(self.i))
        groups[self.j,self.i] = np.arange(len(self.i))
        self.membership = csr_matrix((np.ones(n*n),(np.arange(n*n),groups.ravel())),
                                     shape=(n*n,len(self.i)))
        columns = np.arange(len(self.i))
        self.marginal = csr_matrix((np.full(2*len(self.i),.5),
            (np.concatenate((self.i,self.j)),np.tile(columns,2))),shape=(n,len(self.i)))
        self.sale_marginal = csr_matrix((np.concatenate((
            .5*experiment.game.D[self.i,self.j],.5*experiment.game.D[self.j,self.i])),
            (np.concatenate((self.i,self.j)),np.tile(columns,2))),shape=self.marginal.shape)
        self.last_alpha = self.last_result = self._original = None
        self.calls = self.fallback_calls = 0

    def fallback(self,alpha,reason):
        self.fallback_calls += 1
        if self._original is None:
            self._original = self.original_response(self.experiment)
        result = dict(self._original.calculate(alpha))
        result['quick_prescreen'] = dict(method='Original response fallback',reason=reason,
                                        original_full_final_certificate_still_required=True)
        return result

    def calculate(self,alpha):
        alpha = np.clip(np.asarray(alpha,dtype=float),0,1)
        if self.last_alpha is not None and np.array_equal(alpha,self.last_alpha):
            return self.last_result
        self.calls += 1
        e,n,p = self.experiment,self.experiment.n,self.experiment.params
        full = e.game.transition(np.broadcast_to(alpha[:,None],(n,n)),
                                 np.broadcast_to(alpha[None,:],(n,n)))
        reduced = (full[self.representatives]@self.membership).tocsr()
        off = reduced.copy()
        off.setdiag(0)
        off.eliminate_zeros()
        rates = np.asarray(off.sum(axis=1)).ravel()
        if np.any(rates <= 0) or not np.isfinite(rates).all():
            return self.fallback(alpha,'Nonpositive/nonfinite jump exit rate')
        jump = (diags(1/rates)@off).tocsr()
        anchor = len(rates)//2
        keep = np.arange(len(rates)) != anchor
        linear = (eye(len(rates),format='csc')-jump.T).tocsc()
        try:
            lu = splu(linear[keep,:][:,keep].tocsc())
            w = np.ones(len(rates))
            w[keep] = lu.solve(-linear[keep,anchor].toarray().ravel())
            w /= w.sum()
            calendar = w/rates
            calendar /= calendar.sum()
        except (RuntimeError,ValueError,ArithmeticError) as error:
            return self.fallback(alpha,'Pinned invariant solve failed: '+str(error))
        residual = float(np.sum(abs(reduced.T@calendar-calendar)))
        jump_residual = float(np.sum(abs(jump.T@w-w)))
        mu = self.marginal@calendar
        numerator = self.sale_marginal@calendar
        unsafe = (not np.isfinite(calendar).all() or not np.isfinite(w).all()
                  or calendar.min() < -1e-10 or residual > 1e-10 or jump_residual > 1e-10)
        if self.conditional:
            unsafe = (unsafe or np.any(mu <= 0) or np.any(numerator < 0)
                      or np.any(numerator > mu*(1+1e-13)))
        if unsafe:
            return self.fallback(alpha,'Unsafe pinned law or conditional selected-mass identity')
        if not self.conditional:
            # Same search-only roundoff convention as the original response.
            calendar = np.maximum(calendar,0)
            calendar /= calendar.sum()
            mu = self.marginal@calendar
        demand = e.game.D@mu
        if self.conditional:
            demand = np.clip(numerator/mu,0,1)
        evaluated = evaluate_policy(e.grid,demand,alpha,p)
        result = dict(advantage=evaluated['advantage'],value=evaluated['value'],mu=mu,
                      demand=demand,policy=alpha,stationary_residual=residual,
                      quick_prescreen=dict(method='Pinned-coordinate sparse joint jump-law solve; no Jacobian',
                          anchor=anchor,jump_invariant_l1=jump_residual,
                          calendar_invariant_l1=residual,minimum_calendar_mass=float(calendar.min()),
                          original_full_final_certificate_still_required=True))
        self.last_alpha,self.last_result = alpha.copy(),result
        return result


class QuickMarginalResponse(_QuickResponse):
    pass


class QuickConditionalResponse(_QuickResponse):
    conditional = True
    original_response = _ConditionalResponse


def install():
    """Opt in only in a new archived worker phase, preserving final gates."""
    from oblivious.oe_distribution_fixed_p0.code import multi_p0_fast_run as fast
    from oblivious.oe_distribution_fixed_p0.code import multi_p0_run as runner
    if getattr(runner,'_quick_prescreen_installed',False):
        return
    old = runner.baseline_seed

    def checked_seed_with_helper(rho,n,closure):
        proposal,inputs = old(rho,n,closure)
        return proposal,dict(inputs,**helper_reference())

    runner.baseline_seed = checked_seed_with_helper
    fast._OutsideResponse = QuickMarginalResponse
    fast._ConditionalResponse = QuickConditionalResponse
    runner._quick_prescreen_installed = True
