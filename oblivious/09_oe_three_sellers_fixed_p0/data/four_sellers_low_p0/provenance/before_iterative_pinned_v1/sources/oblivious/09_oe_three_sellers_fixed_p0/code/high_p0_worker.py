"""High-p0 worker using the original large-chain stationary checks.

The frozen expanded worker is reused without changing an equation, numerical
tolerance, or acceptance check.  Both independent sparse solutions use MMD
factorization ordering while retaining the original two coordinate orders and
anchors.  A declared selection made before solving uses original large-chain
checks for rho<0.8, and two positive GTH orders for rho>=0.8, where independent
positive iteration can mix slowly.  Disagreement remains an explicit failure.
"""

from hashlib import sha256
from math import isfinite
from numbers import Real
from pathlib import Path

from . import expanded_run, expanded_stationary, stationary


# Capture before an entrypoint redirects expanded_run.work_case to this worker.
# This also leaves the submitted module-level function picklable under spawn.
_FROZEN_WORK_CASE = expanded_run.work_case
LARGE_CHAIN_REFERENCE_MAX_STATES = 6000
DENSE_REFERENCE_MAX_STATES = 25000

CERTIFICATION = dict(
    reference_max_states_by_rho={
        'rho_below_0p8': LARGE_CHAIN_REFERENCE_MAX_STATES,
        'rho_at_least_0p8': DENSE_REFERENCE_MAX_STATES,
    },
    sparse_factorization_ordering='MMD_AT_PLUS_A',
    independent_sparse_systems='Original two coordinate orders and middle/edge anchors retained',
    positive_max_iterations=1_000_000,
    positive_time_limit_seconds=240,
    invariant_tolerance=1e-11,
    independent_stability_tolerance=1e-7,
    large_chain_methods=[
        'two sparse jump-chain solves with different orders and anchors',
        'positive lazy jump iteration from uniform and concentrated starts',
    ],
    failure_policy='Preserve explicit failure; no dense retry or tolerance relaxation',
)


def reference_budget(rho):
    """Select the declared reference method before any candidate is evaluated."""
    if isinstance(rho, bool) or not isinstance(rho, Real) or not isfinite(rho) or not 0 <= rho < 1:
        raise ValueError('rho must be a finite numeric persistence in [0,1)')
    return DENSE_REFERENCE_MAX_STATES if rho >= .8 else LARGE_CHAIN_REFERENCE_MAX_STATES


def source_reference():
    """Fingerprint this adapter in addition to all frozen campaign sources."""
    path = Path(__file__).resolve()
    return {str(path.relative_to(expanded_run.ROOT)): sha256(path.read_bytes()).hexdigest()}


def work_case(task):
    """Run one frozen worker with process-local ordering and reference budget.

    The original installer retains the original 1e-11 invariant and 1e-7
    independent stability tolerances and provides 1,000,000 iterations/240s.
    Its return callback restores the numerical backend inside the frozen
    worker.  This finally block also restores its installer and sparse solver,
    including on interruption.  Only sparse fill ordering changes; the two
    matrices, anchors, pivoting, positive iteration starts and checks survive.
    """
    budget = reference_budget(task['rho'])
    original_install = expanded_stationary.install
    original_sparse_solve = stationary.spsolve

    def install_budgeted_reference():
        return original_install(reference_max_states=budget, threads=2)

    def solve_with_mmd_ordering(*args, **kwargs):
        # The frozen caller uses a named permc_spec.  Handle its positional form
        # too, preserving every other argument including numerical pivot options.
        if len(args) >= 3:
            args = (*args[:2], 'MMD_AT_PLUS_A', *args[3:])
        else:
            kwargs['permc_spec'] = 'MMD_AT_PLUS_A'
        return original_sparse_solve(*args, **kwargs)

    expanded_stationary.install = install_budgeted_reference
    stationary.spsolve = solve_with_mmd_ordering
    try:
        return _FROZEN_WORK_CASE(task)
    finally:
        expanded_stationary.install = original_install
        stationary.spsolve = original_sparse_solve


__all__ = ['CERTIFICATION', 'reference_budget', 'source_reference', 'work_case']
