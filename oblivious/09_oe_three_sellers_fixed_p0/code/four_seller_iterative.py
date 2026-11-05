"""Matrix-free normalized GMRES for the exact four-seller equations.

The rank-one normalization solves (I-J.T+u 1.T)w=u. Positive u sums to one,
so this is exactly the normalized jump-invariant system. No large LU is used.
Exploratory derivatives use its fundamental operator and an exact pinned gauge.
Final certification retains independent coordinate orders and normalization
references, positive two-start iteration and the original numerical tolerances.
``install`` is process-local and returns a restoration callback.
"""

import time

import numpy as np
from scipy.sparse import eye
from scipy.sparse.linalg import LinearOperator, gmres, splu as direct_splu

from . import four_seller_model as model
from .stationary import _normalized_positive


SETTINGS = dict(
    method='Matrix-free normalized GMRES; exact rank-one normalization, no large LU',
    normalization='(I-J.T+u 1.T)w=u; u=.99*uniform+.01*reference-state mass',
    gmres_relative_tolerance=1e-12, gmres_absolute_tolerance=1e-14,
    gmres_restart=100, gmres_max_restart_cycles=1000,
    true_residual_safety_factor=2., true_backward_error_tolerance=1e-12,
    positive_initial_iterations=1000,
    derivative_gauge='Pinned zero derivative at largest positive jump-mass state',
    failure_policy='Explicit failure; no clipping, tolerance relaxation or residual-only stationary acceptance',
)
_RECORDS = []


class IterativeFactor:
    """Pinned-solve interface using the exact normalized fundamental operator."""

    def __init__(self, jump, reference):
        self.transpose = jump.T.tocsr()
        self.size = jump.shape[0]
        self.source = np.full(self.size, .99/self.size)
        self.source[reference] += .01
        self.operator = LinearOperator(
            (self.size, self.size),
            matvec=lambda value: value-self.transpose@value+self.source*value.sum(),
            dtype=float)
        self.record = dict(size=self.size, matrix_nnz=jump.nnz,
                           normalization_reference=int(reference), solves=0,
                           total_inner_iterations=0, maximum_relative_residual=0.,
                           maximum_absolute_residual=0., maximum_backward_error=0., solve_seconds=0.)
        _RECORDS.append(self.record)
        initial = self.source.copy()
        for _ in range(SETTINGS['positive_initial_iterations']):
            initial = .5*(initial+self.transpose@initial)
        self.law = _normalized_positive(self._normalized_solve(self.source, initial),
                                        'Normalized GMRES jump invariant')
        self.anchor = int(np.argmax(self.law))
        self.keep = np.arange(self.size) != self.anchor

    def _normalized_solve(self, rhs, initial=None):
        rhs = np.asarray(rhs, dtype=float)
        rhs_norm = float(np.linalg.norm(rhs))
        if rhs_norm == 0:
            return np.zeros_like(rhs)
        started = time.monotonic()
        iterations = 0

        def callback(_):
            nonlocal iterations
            iterations += 1

        solution, info = gmres(
            self.operator, rhs, x0=initial,
            rtol=SETTINGS['gmres_relative_tolerance'],
            atol=SETTINGS['gmres_absolute_tolerance'],
            restart=SETTINGS['gmres_restart'], maxiter=SETTINGS['gmres_max_restart_cycles'],
            callback=callback, callback_type='pr_norm')
        residual = float(np.linalg.norm(rhs-self.operator@solution))
        target = max(SETTINGS['gmres_absolute_tolerance'], SETTINGS['gmres_relative_tolerance']*rhs_norm)
        if (info != 0 or np.any(~np.isfinite(solution))
                or residual > SETTINGS['true_residual_safety_factor']*target):
            raise ArithmeticError('Normalized GMRES failed true residual check: '
                                  f'info={info}, ||r||={residual:g}, target={target:g}, iterations={iterations}')
        self.record['solves'] += 1
        self.record['total_inner_iterations'] += iterations
        self.record['maximum_relative_residual'] = max(self.record['maximum_relative_residual'], residual/rhs_norm)
        self.record['maximum_absolute_residual'] = max(self.record['maximum_absolute_residual'], residual)
        self.record['solve_seconds'] += time.monotonic()-started
        return solution

    def solve(self, rhs):
        rhs = np.asarray(rhs, dtype=float)
        if rhs.ndim == 2:
            return np.column_stack([self.solve(rhs[:, k]) for k in range(rhs.shape[1])])
        if rhs.shape != (self.size-1,) or np.any(~np.isfinite(rhs)):
            raise ValueError('Finite vector/matrix RHS must match the pinned linear system')
        complete = np.zeros(self.size)
        complete[self.keep] = rhs
        complete[self.anchor] = -rhs.sum()
        solution = self._normalized_solve(complete)
        # All solutions of (I-J.T)x=complete differ by a multiple of law.
        # The largest-mass gauge avoids division by a rare probability.
        solution -= self.law*(solution[self.anchor]/self.law[self.anchor])
        solution[self.anchor] = 0.
        actual = solution-self.transpose@solution
        residual = float(np.linalg.norm(rhs-actual[self.keep]))
        norm_bound = 1+float(np.max(np.asarray(self.transpose.sum(axis=1))))
        backward = residual/(norm_bound*np.linalg.norm(solution)+np.linalg.norm(rhs)) if np.any(rhs) else 0.
        if not np.isfinite(backward) or backward > SETTINGS['true_backward_error_tolerance']:
            raise ArithmeticError(f'Pinned fundamental solve failed true backward error: {backward:g}')
        self.record['maximum_backward_error'] = max(self.record['maximum_backward_error'], backward)
        return solution[self.keep]


def iterative_pinned_jump(jump, anchor):
    factor = IterativeFactor(jump, anchor)
    keep = factor.keep
    linear = (eye(factor.size, format='csr')-factor.transpose)[keep, :][:, keep].tocsc()
    linear._four_seller_factor = factor
    return factor.law, factor, linear, keep


def _factorization(linear, *args, **kwargs):
    factor = getattr(linear, '_four_seller_factor', None)
    return factor if factor is not None else direct_splu(linear, *args, **kwargs)


def install():
    """Install this declared solver phase and return an exception-safe restore."""
    if model._pinned_jump is iterative_pinned_jump:
        raise RuntimeError('Four-seller iterative adapter is already installed in this process')
    previous_pinned, previous_factor = model._pinned_jump, model.splu
    previous_checker, previous_configuration = model.checked_quotient_stationary, model.CERTIFICATION

    def checked(jump, rates):
        first = len(_RECORDS)
        result = previous_checker(jump, rates)
        result['summary']['methods'] = [
            'Normalized matrix-free GMRES jump solve, original order and middle reference',
            'Normalized matrix-free GMRES jump solve, reversed order and edge reference',
            'positive lazy jump iteration, uniform and concentrated starts']
        result['summary']['iterative_settings'] = dict(SETTINGS)
        result['summary']['linear_solves'] = [dict(record) for record in _RECORDS[first:]]
        return result

    model._pinned_jump = iterative_pinned_jump
    model.splu = _factorization
    model.checked_quotient_stationary = checked
    model.CERTIFICATION = dict(previous_configuration, sparse_linear_solver=dict(SETTINGS))

    def restore():
        model._pinned_jump, model.splu = previous_pinned, previous_factor
        model.checked_quotient_stationary, model.CERTIFICATION = previous_checker, previous_configuration

    return restore


def diagnostics():
    return [dict(record) for record in _RECORDS]


__all__ = ['SETTINGS', 'IterativeFactor', 'iterative_pinned_jump', 'install', 'diagnostics']
