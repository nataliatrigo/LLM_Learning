"""Expected finite-window occupation, without an invariant-law assumption.

For a row-stochastic kernel P and initial law q_0 this module computes
T^{-1} sum_{t=B}^{B+T-1} q_0 P^t.  States are measured before selection.
No simulation, laziness, stationary solve, or assumption of irreducibility is
used.  The optional early exit has a telescoping truncation bound; this bound
does not certify floating-point, interpolation, or demand-quadrature errors.
"""

import numpy as np
from scipy.sparse import csr_matrix

from oblivious.oe_p0.code.primitives import interpolation_matrix, probability_array


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or int(value) != value or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _kernel(matrix):
    matrix = csr_matrix(matrix, dtype=float)
    matrix.sum_duplicates()
    if (matrix.shape[0] < 1 or matrix.shape[0] != matrix.shape[1]
            or np.any(~np.isfinite(matrix.data)) or matrix.data.min(initial=0) < 0):
        raise ValueError("Kernel must be finite, square, and nonnegative.")
    row_error = float(np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel() - 1)))
    if row_error > 1e-12:
        raise ValueError(f"Kernel must be row-stochastic; error={row_error:g}.")
    return matrix, row_error


def _normalized(law):
    law = np.asarray(law, dtype=float)
    mass = float(law.sum())
    if np.any(~np.isfinite(law)) or law.min(initial=0) < 0 or not mass > 0:
        raise ArithmeticError("Propagation produced invalid probability mass.")
    normalized = law / mass
    return normalized, mass, float(np.sum(np.abs(normalized - law)))


def finite_window_occupation(matrix, initial_law, periods=100000, burn_in=10000,
                             tolerance=1e-10, *, early_stop=True):
    """Propagate any finite Markov kernel and average pre-transition laws.

    ``initial_law`` is normalized on entry.  ``law`` is the expected average
    law for t=burn_in,...,burn_in+periods-1. ``final_law`` is at t=burn_in+
    periods, one transition after the final observation. A zero tolerance is
    allowed; ``early_stop=False`` always evaluates the full finite horizon.

    If q_t is held constant for the uncomputed tail and
    delta=||q_t P-q_t||_1, Markov contraction gives
    ||q_(t+k)-q_t||_1 <= k*delta. Thus the average-law truncation error is at
    most delta/periods times the sum of the remaining observation offsets k.
    This remains valid for reducible and periodic kernels. It is a real-
    arithmetic bound for the specified stochastic kernel, not a stationarity
    or mixing certificate. Floating-point error is reported separately and
    is not included in that bound.
    """
    periods = _integer(periods, "periods", 1)
    burn_in = _integer(burn_in, "burn_in", 0)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative.")
    matrix, row_error = _kernel(matrix)
    initial = np.asarray(initial_law, dtype=float)
    if (initial.ndim != 1 or initial.size != matrix.shape[0]
            or np.any(~np.isfinite(initial)) or np.any(initial < 0) or initial.sum() <= 0):
        raise ValueError("Initial law must be a finite nonnegative vector with positive mass.")
    initial, _, _ = _normalized(initial)
    law = initial.copy()
    transpose = matrix.T.tocsr()
    total = np.zeros_like(law)
    end = burn_in + periods
    burn_law = None
    tail_bound = burn_bound = final_bound = 0.
    stopped = False
    tail_start = None
    delta = 0.
    propagated = 0
    for t in range(end):
        if t == burn_in:
            burn_law = law.copy()
        moved = transpose @ law
        propagated += 1
        delta = float(np.sum(np.abs(moved - law)))
        first_offset = max(burn_in - t, 0)
        last_offset = end - 1 - t
        observations_left = end - max(t, burn_in)
        offset_sum = observations_left * (first_offset + last_offset) / 2
        bound = delta * offset_sum / periods
        if early_stop and t < end - 1 and bound <= tolerance:
            # Earlier observations have already been accumulated; q_t itself
            # has not yet been added. Include it only when it is in-window.
            total += observations_left * law
            if burn_law is None:
                burn_law = law.copy()
                burn_bound = delta * (burn_in - t)
            final_bound = delta * (end - t)
            tail_bound = bound
            stopped = True
            tail_start = t
            break
        if t >= burn_in:
            total += law
        law = moved
    occupation, raw_mass, correction = _normalized(total / periods)
    final, final_mass, final_correction = _normalized(law)
    burn, burn_mass, burn_correction = _normalized(burn_law)
    summary = dict(
        verified=bool(tail_bound <= tolerance), periods=periods, burn_in=burn_in,
        first_observation=burn_in, last_observation=end - 1, final_state_time=end,
        propagated_steps=propagated, requested_transitions=end,
        early_stopped=stopped, tail_replacement_time=tail_start,
        occupation_truncation_l1_bound=float(tail_bound),
        burn_in_law_truncation_l1_bound=float(burn_bound),
        final_law_truncation_l1_bound=float(final_bound), tolerance=float(tolerance),
        last_step_l1=delta, row_sum_error=row_error,
        probability_sum=float(occupation.sum()), minimum_probability=float(occupation.min()),
        occupation_mass_before_normalization=raw_mass,
        occupation_normalization_l1_correction=correction,
        burn_in_mass_before_normalization=burn_mass,
        burn_in_normalization_l1_correction=burn_correction,
        final_mass_before_normalization=final_mass,
        final_normalization_l1_correction=final_correction,
        arithmetic_roundoff_certified=False, arithmetic_roundoff_error_bound=None,
        method="positive sparse Markov propagation with certified optional tail replacement",
        selection="expected finite-window occupation law",
        observation_timing="pre-selection states, t=B,...,B+T-1; initial state is t=0",
        scope="Finite numerical Markov kernel; truncation certificate excludes arithmetic roundoff, grid interpolation, and demand quadrature; no stationarity assertion",
    )
    return dict(law=occupation, initial_law=initial, burn_in_law=burn,
                final_law=final, summary=summary)


def symmetric_pair_kernel(matrix, n):
    """Exactly lump an exchange-symmetric n-by-n joint state space.

    Pair (i,j) and (j,i) share one state. The returned membership matrix maps
    full laws to pair laws; lifting divides off-diagonal mass equally. The
    full exchange check is needed in addition to row lumpability to preserve
    symmetric laws after lifting. Observed floating-point discrepancies are
    reported; mathematical symmetry comes from the identical seller policy.
    """
    n = _integer(n, "n", 1)
    matrix, row_error = _kernel(matrix)
    if matrix.shape != (n * n, n * n):
        raise ValueError("Joint kernel shape must equal (n*n,n*n).")
    i, j = np.triu_indices(n)
    representatives, swapped = i * n + j, j * n + i
    groups = np.empty((n, n), dtype=int)
    groups[i, j] = np.arange(len(i))
    groups[j, i] = np.arange(len(i))
    membership = csr_matrix((np.ones(n * n), (np.arange(n * n), groups.ravel())),
                            shape=(n * n, len(i)))
    reduced = (matrix[representatives] @ membership).tocsr()
    difference = reduced - matrix[swapped] @ membership
    lump_error = float(np.max(np.abs(difference.data), initial=0))
    swap = np.arange(n * n).reshape(n, n).T.ravel()
    exchange = matrix - matrix[swap][:, swap]
    exchange_error = float(np.max(np.abs(exchange.data), initial=0))
    if max(lump_error, exchange_error) > 1e-13:
        raise ValueError("Joint kernel is not exchange symmetric; symmetric lifting is invalid.")
    summary = dict(symmetric_lumping=True, lumping_error=lump_error,
                   exchange_symmetry_error=exchange_error, full_states=n * n,
                   solved_states=len(i), full_kernel_row_sum_error=row_error)
    return reduced, membership, (i, j), summary


def lift_symmetric_pair_law(law, n, pairs=None):
    """Lift unordered-pair masses to an exchange-symmetric joint law."""
    i, j = np.triu_indices(n) if pairs is None else pairs
    law = np.asarray(law, dtype=float)
    if law.shape != (len(i),):
        raise ValueError("Pair-law length does not match joint grid size.")
    joint = np.zeros((n, n))
    shares = law / np.where(i == j, 1, 2)
    joint[i, j] = shares
    joint[j, i] = shares
    return joint


def finite_window_law(game, policy, periods=100000, burn_in=10000,
                      initial=.5, tolerance=1e-10):
    """Expected joint temporal law when both sellers use one own-state policy.

    ``policy[i]`` is product-2 probability at own reputation grid[i].
    ``initial`` is a common reputation, or an equal pair such as (.5,.5).
    Off-grid initial reputation uses the game's linear grid interpolation;
    joint initialization is the product of those two interpolation laws.
    """
    policy = np.asarray(policy, dtype=float)
    if policy.ndim == 0:
        policy = np.full(game.n, float(policy))
    if policy.shape != (game.n,):
        raise ValueError("policy must be a scalar or a vector with one entry per own-state grid node.")
    policy = probability_array(policy)
    initial = np.asarray(initial, dtype=float)
    if initial.ndim == 0:
        initial = np.repeat(initial, 2)
    if (initial.shape != (2,) or np.any(~np.isfinite(initial))
            or np.any(initial < 0) or np.any(initial > 1) or initial[0] != initial[1]):
        raise ValueError("Symmetric pair propagation requires equal finite initial reputations in [0,1].")
    marginal = interpolation_matrix(game.grid, initial[:1]).toarray()[0]
    full_initial = np.outer(marginal, marginal)
    sigma_a = np.broadcast_to(policy[:, None], (game.n, game.n))
    sigma_b = np.broadcast_to(policy[None, :], (game.n, game.n))
    matrix = game.transition(sigma_a, sigma_b)
    reduced, membership, pairs, lump_summary = symmetric_pair_kernel(matrix, game.n)
    pair_initial = membership.T @ full_initial.ravel()
    result = finite_window_occupation(reduced, pair_initial, periods=periods,
                                     burn_in=burn_in, tolerance=tolerance)
    nu = lift_symmetric_pair_law(result["law"], game.n, pairs)
    initial_joint = lift_symmetric_pair_law(result["initial_law"], game.n, pairs)
    burn_joint = lift_symmetric_pair_law(result["burn_in_law"], game.n, pairs)
    final_joint = lift_symmetric_pair_law(result["final_law"], game.n, pairs)
    summary = {**result["summary"], **lump_summary,
               "initial_reputations": initial.tolist(),
               "initial_interpolation": "independent linear interpolation on each reputation axis",
               "marginal_symmetry_l1": float(np.sum(np.abs(nu.sum(axis=1) - nu.sum(axis=0))))}
    return dict(nu=nu, mu_a=nu.sum(axis=1), mu_b=nu.sum(axis=0), summary=summary,
                initial_law=initial_joint, burn_in_law=burn_joint, final_law=final_joint)
