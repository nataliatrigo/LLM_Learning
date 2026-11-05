"""Sparse stationary laws for the finite multi-seller calendar-time chain.

Exit rates are supplied separately from its jump chain: subtracting a diagonal
entry from one would erase rare but representable transitions.  Every accepted
law has independent numerical checks beyond its invariant residual.  These
checks concern the finite kernel, not a mixing-time bound or grid convergence.
"""

import time
import warnings

import numpy as np
from scipy.sparse import csr_matrix, eye
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import MatrixRankWarning, spsolve

from oblivious.oe_p0.code.positive_stationary import positive_invariant


def _validated_system(jump, rates):
    matrix = csr_matrix(jump, dtype=float)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    rates = np.asarray(rates, dtype=float)
    size = matrix.shape[0]
    if (matrix.shape != (size, size) or rates.shape != (size,) or size < 2
            or np.any(~np.isfinite(matrix.data)) or np.any(matrix.data < 0)
            or np.any(~np.isfinite(rates)) or np.any(rates <= 0)
            or np.any(rates > 1 + 1e-12)):
        raise ValueError("A finite nonnegative square jump chain and exit rates in (0,1] are required.")
    row_error = float(np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel() - 1)))
    if row_error > 1e-12:
        raise ValueError(f"Jump chain is not row-stochastic: {row_error:g}.")
    components, _ = connected_components(matrix, directed=True, connection="strong")
    if components != 1:
        raise ArithmeticError(f"Jump chain has {components} communicating components; stationary selection is unresolved.")
    return matrix, rates, row_error


def jump_system(kernel):
    """Extract off-diagonal positive flows without ever computing 1-P[ii]."""
    matrix = csr_matrix(kernel, dtype=float)
    size = matrix.shape[0]
    if (matrix.shape != (size, size) or np.any(~np.isfinite(matrix.data))
            or np.any(matrix.data < 0)):
        raise ValueError("Calendar kernel must be a finite nonnegative square matrix.")
    if np.max(np.abs(np.asarray(matrix.sum(axis=1)).ravel() - 1)) > 1e-12:
        raise ValueError("Calendar kernel is not row-stochastic.")
    off = matrix.copy()
    off.setdiag(0)
    off.eliminate_zeros()
    rates = np.asarray(off.sum(axis=1)).ravel()
    if np.any(rates <= 0):
        raise ArithmeticError("Zero exit rate: no unique full-support law can be certified.")
    off.data /= np.repeat(rates, np.diff(off.indptr))
    return off, rates


def _normalized_positive(weights, label):
    weights = np.asarray(weights, dtype=float)
    # Clipping even tiny negative values can silently erase a rare-flow class.
    if np.any(~np.isfinite(weights)) or np.any(weights <= 0):
        raise ArithmeticError(f"{label} did not produce strictly positive finite mass.")
    scaled = weights / weights.max()
    normalized = scaled / scaled.sum()
    if np.any(normalized <= 0):
        raise ArithmeticError(f"{label} lost positive mass during normalization.")
    return normalized


def _calendar(jump_law, rates):
    """Stable normalization of w/r, including subnormal calendar exit rates."""
    jump_law = _normalized_positive(jump_law, "Jump invariant")
    log_mass = np.log(jump_law) - np.log(rates)
    mass = np.exp(log_mass - log_mass.max())
    return _normalized_positive(mass, "Calendar invariant")


def _jump_mass(calendar_law, rates):
    log_mass = np.log(calendar_law) + np.log(rates)
    return _normalized_positive(np.exp(log_mass - log_mass.max()), "Rate-weighted invariant")


def _direct(jump, anchor):
    size = jump.shape[0]
    # Delete the normalization column as well as its row.  Fixing w[anchor]=1
    # avoids the dense normalization row used in the older two-seller code.
    keep = np.arange(size) != anchor
    transpose = jump.T.tocsr()
    linear = (eye(size, format="csr") - transpose)[keep, :][:, keep].tocsc()
    rhs = np.asarray(transpose[keep, anchor].toarray()).ravel()
    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        interior = spsolve(linear, rhs, permc_spec="COLAMD")
    law = np.empty(size)
    law[anchor] = 1
    law[keep] = interior
    return _normalized_positive(law, "Sparse jump-chain solve")


def _flow_checks(jump, rates, law):
    weighted = _jump_mass(law, rates)
    incoming = np.asarray(jump.T @ weighted).ravel()
    jump_residual = float(np.sum(np.abs(incoming - weighted)))
    # The absolute calendar residual alone can be vacuous when every exit
    # rate is tiny.  Normalize flow before checking invariance as well.
    calendar_flow = law * rates
    calendar_residual = float(np.sum(np.abs(jump.T @ calendar_flow - calendar_flow)))
    return jump_residual, calendar_residual, weighted


def _positive_iterates(jump, rates, target, *, tolerance, max_iterations, time_limit):
    """Independent positive iteration, with widely separated initial laws.

    Lazification avoids periodicity.  Both starts must approach the candidate;
    a small one-step residual never suffices.  Budget exhaustion is an error.
    This is a numerical stability check, not a certified total-variation bound.
    """
    size = jump.shape[0]
    target_jump = _jump_mass(target, rates)
    mass = np.full((size, 2), 1 / size)
    mass[:, 1] *= 0.01
    mass[-1, 1] += 0.99
    started = time.monotonic()
    discrepancy = float("inf")
    for iteration in range(1, max_iterations + 1):
        mass = 0.5 * (mass + jump.T @ mass)
        if iteration % 25 == 0 or iteration == max_iterations:
            mass /= mass.sum(axis=0)
            laws = [_calendar(mass[:, index], rates) for index in range(2)]
            calendar_discrepancy = max(float(np.sum(np.abs(law - target))) for law in laws)
            jump_discrepancy = max(float(np.sum(np.abs(mass[:, index] - target_jump)))
                                   for index in range(2))
            discrepancy = max(calendar_discrepancy, jump_discrepancy)
            if discrepancy <= tolerance:
                return dict(iterations=iteration, discrepancy=discrepancy,
                            calendar_discrepancy=calendar_discrepancy,
                            jump_discrepancy=jump_discrepancy,
                            elapsed_seconds=time.monotonic() - started)
            if time_limit is not None and time.monotonic() - started > time_limit:
                break
    raise ArithmeticError("Positive independent iteration did not validate the candidate "
                          f"within its budget: iterations={iteration}, L1={discrepancy:g}.")


def checked_stationary(jump, rates, *, tolerance=1e-11, stability_tolerance=1e-7,
                       reference_max_states=2000, positive_max_iterations=100000,
                       positive_time_limit=120):
    """Return ``law``, ``jump_law`` and an explicit finite-kernel audit.

    Small chains use subtraction-free GTH in both elimination orders.  Larger
    chains require two independent sparse solves and positive iterations from
    two starts.  Large rare-flow chains may fail these checks; they are never
    silently accepted or retried with a weaker residual-only criterion.
    """
    if tolerance <= 0 or stability_tolerance <= 0 or reference_max_states < 0:
        raise ValueError("Positive tolerances and a nonnegative reference limit are required.")
    matrix, rates, row_error = _validated_system(jump, rates)
    size = len(rates)
    started = time.monotonic()
    methods, failures, candidates = [], [], []
    for reverse in (False, True):
        try:
            permutation = np.arange(size)[::-1] if reverse else np.arange(size)
            anchor = 0 if reverse else size // 2
            ordered = _direct(matrix[permutation, :][:, permutation], anchor)
            mass = np.empty(size)
            mass[permutation] = ordered
            candidates.append(_calendar(mass, rates))
            methods.append("sparse jump solve, reversed order and edge anchor" if reverse
                           else "sparse jump solve, original order and middle anchor")
        except (ArithmeticError, RuntimeError, MatrixRankWarning) as exc:
            failures.append(str(exc))
    if size <= reference_max_states:
        for permutation in (None, np.arange(size)[::-1]):
            positive_jump = positive_invariant(matrix, np.ones(size), permutation)
            candidates.append(_calendar(positive_jump, rates))
            methods.append("positive GTH, reversed order" if permutation is not None else "positive GTH")
    elif len(candidates) != 2:
        raise ArithmeticError(f"Large-chain sparse solves failed: {failures}")
    if not candidates:
        raise ArithmeticError("No independent invariant-law candidates were produced.")
    # Prefer the subtraction-free candidate when available.
    law = candidates[-1]
    calendar_discrepancy = max(float(np.sum(np.abs(candidate - law))) for candidate in candidates)
    target_jump = _jump_mass(law, rates)
    jump_discrepancy = max(float(np.sum(np.abs(_jump_mass(candidate, rates) - target_jump)))
                           for candidate in candidates)
    discrepancy = max(calendar_discrepancy, jump_discrepancy)
    if discrepancy > stability_tolerance:
        raise ArithmeticError(f"Independent stationary laws disagree: L1={discrepancy:g}.")
    positive_check = None
    if size > reference_max_states:
        positive_check = _positive_iterates(
            matrix, rates, law, tolerance=stability_tolerance,
            max_iterations=positive_max_iterations, time_limit=positive_time_limit)
        discrepancy = max(discrepancy, positive_check["discrepancy"])
        methods.append("positive lazy jump iteration, uniform and concentrated starts")
    jump_residual, calendar_residual, jump_law = _flow_checks(matrix, rates, law)
    if jump_residual > tolerance or calendar_residual > tolerance:
        raise ArithmeticError(f"Invariant flow verification failed: jump={jump_residual:g}, calendar={calendar_residual:g}.")
    log_dynamic_range = float(np.log(rates.max()) - np.log(rates.min()))
    dynamic_range = float(np.exp(log_dynamic_range)) if log_dynamic_range < np.log(np.finfo(float).max) else "overflow"
    summary = dict(
        verified=True, probability_sum=float(law.sum()), minimum_probability=float(law.min()),
        minimum_exit_rate=float(rates.min()), maximum_exit_rate=float(rates.max()),
        exit_rate_dynamic_range=dynamic_range,
        residual=calendar_residual, rate_weighted_residual=jump_residual,
        row_sum_error=row_error, stability_l1=discrepancy,
        independent_calendar_l1=calendar_discrepancy, independent_jump_l1=jump_discrepancy,
        tolerance=tolerance, stability_tolerance=stability_tolerance,
        methods=methods, preliminary_failures=failures, positive_iteration=positive_check,
        solved_states=size, communicating_components=1,
        elapsed_seconds=time.monotonic() - started,
        scope="Finite numerical kernel and independent numerical stability checks; no mixing-time or continuous-state guarantee.")
    return dict(law=law, jump_law=jump_law, summary=summary)


def symmetric_lumping(jump, rates, shape, *, symmetry_tolerance=1e-12):
    """Exactly validate the quotient under permutations of identical sellers.

    Every full row is checked against its orbit representative.  Sorting the
    coordinate tuple identifies each orbit; no n^(2N) dense object is formed.
    """
    matrix, rates, _ = _validated_system(jump, rates)
    shape = tuple(shape)
    if len(shape) < 1 or len(set(shape)) != 1 or int(np.prod(shape)) != len(rates):
        raise ValueError("Exchangeable seller state shape must be (n,)*N and match the kernel.")
    coordinates = np.indices(shape).reshape(len(shape), -1).T
    canonical = np.sort(coordinates, axis=1)
    canonical_ids = np.ravel_multi_index(canonical.T, shape)
    representatives, inverse, counts = np.unique(canonical_ids, return_inverse=True, return_counts=True)
    membership = csr_matrix((np.ones(len(rates)), (np.arange(len(rates)), inverse)),
                            shape=(len(rates), len(representatives)))
    all_rows = (matrix @ membership).tocsr()
    reduced = all_rows[representatives].tocsr()
    difference = all_rows - membership @ reduced
    transition_error = float(np.max(np.abs(difference.data), initial=0))
    reduced_rates = rates[representatives]
    rate_relative_error = float(np.max(np.abs(rates / reduced_rates[inverse] - 1)))
    if transition_error > symmetry_tolerance or rate_relative_error > symmetry_tolerance:
        raise ArithmeticError("Seller permutation lumping failed its full-row check: "
                              f"jump={transition_error:g}, relative_rates={rate_relative_error:g}.")
    return dict(jump=reduced, rates=reduced_rates, inverse=inverse, counts=counts,
                representatives=representatives,
                summary=dict(transition_error=transition_error,
                             exit_rate_relative_error=rate_relative_error,
                             full_states=len(rates), solved_states=len(representatives)))


def checked_symmetric_stationary(jump, rates, shape, **kwargs):
    """Solve the exact seller-permutation quotient and audit the full law."""
    matrix, rates, _ = _validated_system(jump, rates)
    lumping = symmetric_lumping(matrix, rates, shape)
    result = checked_stationary(lumping["jump"], lumping["rates"], **kwargs)
    law = result["law"][lumping["inverse"]] / lumping["counts"][lumping["inverse"]]
    jump_residual, calendar_residual, jump_law = _flow_checks(matrix, rates, law)
    tolerance = kwargs.get("tolerance", 1e-11)
    if jump_residual > tolerance or calendar_residual > tolerance:
        raise ArithmeticError("Expanded exchangeable law failed full-kernel flow verification.")
    joint = law.reshape(shape)
    marginals = [joint.sum(axis=tuple(axis for axis in range(len(shape)) if axis != seller))
                 for seller in range(len(shape))]
    marginal_symmetry = max(float(np.sum(np.abs(marginal - marginals[0]))) for marginal in marginals)
    summary = dict(result["summary"], symmetric_lumping=True,
                   lumping=lumping["summary"], full_states=len(law),
                   minimum_probability=float(law.min()), probability_sum=float(law.sum()),
                   full_residual=calendar_residual, full_rate_weighted_residual=jump_residual,
                   marginal_symmetry_l1=marginal_symmetry)
    return dict(law=law, jump_law=jump_law, marginals=np.asarray(marginals),
                reduced_law=result["law"], summary=summary)
