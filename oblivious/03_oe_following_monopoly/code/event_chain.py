"""Exact sale-time scalar closure for an own-reputation product policy.

The timing of a seller's sales depends on its competitor, but the reputation
transition at each of its own sales does not. In any stationary actual game,
the seller's sale-weighted reputation law is therefore invariant for this
one-dimensional embedded chain. This helper accelerates search only; final
equilibrium certificates also check the actual two-seller joint invariant law.
"""

import numpy as np

from oblivious.oe_p0.code.core import _jump_system, _positive_invariant
from oblivious.oe_p0.code.policy import check_grid, transition_matrix
from oblivious.oe_p0.code.primitives import probability_array


def purchase_chain_statistics(grid, policy, params, *, stationary_tol=1e-11,
                              stability_tol=1e-9):
    """Return true success probability conditional on this seller making a sale.

    Two subtraction-free invariant solves, with reversed state order, check
    numerical stability. The chain uses the unchanged reputation transitions
    and mixture-dependent true success probability from the existing model.
    It introduces neither an assumed TS-demand approximation nor a modified
    transition kernel.
    """
    grid = check_grid(grid)
    alpha = np.asarray(policy, dtype=float)
    if alpha.shape != grid.shape:
        raise ValueError("Policy must have one product-2 probability per grid node.")
    alpha = probability_array(alpha)
    if (not np.isfinite(stationary_tol) or stationary_tol <= 0
            or not np.isfinite(stability_tol) or stability_tol <= 0):
        raise ValueError("Stationarity and stability tolerances must be finite and positive.")

    kernel = transition_matrix(grid, np.ones(len(grid)), alpha, params)
    row_error = float(np.max(np.abs(np.asarray(kernel.sum(axis=1)).ravel()-1)))
    if row_error > 1e-12:
        raise ArithmeticError(f"Sale-time reputation kernel is not stochastic: {row_error:g}.")
    jump, rates = _jump_system(kernel)
    law = _positive_invariant(jump, rates)
    reverse = _positive_invariant(jump, rates, np.arange(len(grid))[::-1])
    discrepancy = float(np.sum(np.abs(law-reverse)))
    residual = float(np.sum(np.abs(kernel.T @ law-law)))
    if (discrepancy > stability_tol or residual > stationary_tol
            or np.any(~np.isfinite(law)) or np.any(law < 0)
            or abs(law.sum()-1) > stationary_tol):
        raise ArithmeticError(
            f"Sale-time stationary verification failed: residual={residual:g}, "
            f"independent-order L1={discrepancy:g}.")

    quality = params.p1+(params.p2-params.p1)*alpha
    generated = float(law @ quality)
    reputation = float(law @ grid)
    return dict(
        sale_stationary=law,
        generated_quality=generated,
        mean_reputation_at_sale=reputation,
        quality_reputation_difference=generated-reputation,
        stationary_residual=residual,
        stationary_verified=True,
        stability_l1=discrepancy,
        row_sum_error=row_error,
        minimum_exit_rate=float(rates.min()),
        method="Sale-time embedded chain; independent positive GTH elimination in both state orders",
        scope="Finite numerical kernel for own-reputation policies; final actual-game joint law is checked separately",
    )
