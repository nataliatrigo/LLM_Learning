"""Model parameters, Beta demand, and linear interpolation for the MPE game.

These primitives preserve the previous model; this module is self-contained.
"""

from dataclasses import dataclass, asdict

import numpy as np
from scipy.integrate import quad
from scipy.special import betainc, betaln, roots_jacobi
from scipy.sparse import csr_matrix


@dataclass(frozen=True)
class Parameters:
    R: float = 1.0
    c1: float = 0.05
    c2: float = 0.65
    p1: float = 0.35
    p2: float = 0.80
    gamma: float = 0.98
    rho: float = 0.95

    def __post_init__(self):
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Parameters must be finite.")
        if not (0 <= self.c1 < self.c2 < self.R):
            raise ValueError("Require 0 <= c1 < c2 < R.")
        if not (0 < self.p1 < self.p2 < 1):
            raise ValueError("Require 0 < p1 < p2 < 1.")
        if not (0 <= self.gamma < 1 and 0.5 < self.rho < 1):
            raise ValueError("Require 0 <= gamma < 1 and 0.5 < rho < 1.")


def probability_array(values):
    """Validate probabilities, correcting only floating-point boundary noise.

    Bilinear interpolation of a policy can produce 1 + 2e-16. This is not an
    infeasible strategy. Larger violations and nonfinite inputs remain errors.
    Never use this helper for values, advantages, or Newton trial iterates.
    """
    values = np.asarray(values, dtype=float)
    roundoff = 32 * np.finfo(float).eps
    if (not np.isfinite(values).all()
            or np.any(values < -roundoff) or np.any(values > 1 + roundoff)):
        raise ValueError("Policies must be finite probabilities in [0,1].")
    return np.clip(values, 0, 1)


def check_grid(grid):
    grid = np.asarray(grid, dtype=float)
    if (grid.ndim != 1 or len(grid) < 3 or grid[0] != 0 or grid[-1] != 1
            or not np.all(np.diff(grid) > 0)):
        raise ValueError("Grid must increase strictly from 0 to 1, with >=3 nodes.")
    return grid


def posterior(x, params):
    x = np.asarray(x)
    if np.any(~np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("Reputations must lie in [0,1].")
    return 1 + x / (1 - params.rho), 1 + (1 - x) / (1 - params.rho)


def demand_probability(x, y, params):
    """Independent adaptive integration, used to audit the faster matrix."""
    a, b = posterior(float(x), params)
    c, d = posterior(float(y), params)

    def integrand(t):
        if t == 0 or t == 1:
            return 0.0  # Endpoint values do not affect the integral.
        pdf = np.exp((a - 1) * np.log(t) + (b - 1) * np.log1p(-t) - betaln(a, b))
        return pdf * betainc(c, d, t)

    result, error = quad(integrand, 0, 1, epsabs=2e-12, epsrel=2e-12, limit=300)
    if error > 1e-9 or not -1e-12 <= result <= 1 + 1e-12:
        raise ArithmeticError(f"Demand quadrature failed: error={error:g}.")
    return float(np.clip(result, 0, 1))


def demand_matrix(grid, params, order=64):
    """D[x,y] = integral Beta_pdf(t|x) Beta_cdf(t|y) dt.

    Gauss-Jacobi integrates with respect to each row's Beta density, avoiding
    endpoint singularities. Raw errors are recorded BEFORE enforcing symmetry.
    Accuracy must also be checked by order doubling and independent quad.
    """
    grid = check_grid(grid)
    if order < 8:
        raise ValueError("Use quadrature order >=8.")
    a, b = posterior(grid, params)
    raw = np.empty((len(grid), len(grid)))
    for i in range(len(grid)):
        nodes, weights = roots_jacobi(order, b[i] - 1, a[i] - 1)
        raw[i] = betainc(a[:, None], b[:, None], (nodes[None, :] + 1) / 2) @ (weights / weights.sum())
    diagnostics = {
        "order": order,
        "raw_diagonal_error": float(np.max(np.abs(raw.diagonal() - .5))),
        "raw_symmetry_error": float(np.max(np.abs(raw + raw.T - 1))),
        "raw_min": float(raw.min()), "raw_max": float(raw.max()),
    }
    # Integrate the SMALL tail directly. Averaging D with 1-D.T cancels rare
    # selection probabilities, which can matter greatly for calendar-time mass.
    result = np.zeros_like(raw)
    upper = np.triu_indices(len(grid), 1)
    result[upper] = np.clip(raw[upper], 0, 1)
    result[(upper[1], upper[0])] = 1 - result[upper]
    np.fill_diagonal(result, .5)
    return result, diagnostics


def interpolation_matrix(grid, targets):
    """Rows allocate off-grid targets between their adjacent grid nodes."""
    grid = check_grid(grid)
    targets = np.asarray(targets, dtype=float).ravel()
    if np.any(~np.isfinite(targets)) or np.any((targets < -1e-14) | (targets > 1+1e-14)):
        raise ValueError("Interpolation targets must lie in [0,1].")
    targets = np.clip(targets, 0, 1)
    lo = np.clip(np.searchsorted(grid, targets, side="right") - 1, 0, len(grid) - 2)
    w = (targets - grid[lo]) / (grid[lo + 1] - grid[lo])
    rows = np.repeat(np.arange(len(targets)), 2)
    return csr_matrix((np.column_stack((1-w, w)).ravel(),
                       (rows, np.column_stack((lo, lo+1)).ravel())),
                      shape=(len(targets), len(grid)))

