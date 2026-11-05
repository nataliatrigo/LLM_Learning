"""Actual-game diagnostics for own-state conditional-demand optimization.

The conjecture is a row-stochastic rival distribution conditional on the
seller's own reputation. Its demand is sum_j D[i,j] mu_cond[i,j]. Marginal
demand remains a comparison only. Conditional laws at zero-mass own states
are off-path conventions, never observations from the stationary joint law.
"""

import numpy as np

from oblivious.oe_p0.code.diagnostics import _checked_joint
from oblivious.oe_p0.code.primitives import probability_array
from oblivious.oe_distribution.code.diagnostics import (
    _weighted_errors,
    demand_diagnostics as _marginal_demand_diagnostics,
    deviation_diagnostics as _distribution_deviation_diagnostics,
    simulate_ts as _distribution_simulate_ts,
)


def demand_diagnostics(game, candidate, *, mass_floor=0.):
    """Check the conditional conjecture against the actual joint law.

    Every strictly positive row participates in the consistency checks,
    including rows below a requested display ``mass_floor``. Exact zero rows
    have no observed conditional law. The saved conjecture is retained there
    and reported separately from the actual conditional distribution.

    Generic demand/error keys retain the original diagnostics interface.
    In those keys, "conjectured" now means the conditional conjecture; the
    explicitly named ``d_conjectured_marginal`` is an unconditional reference.
    """
    if not np.isfinite(mass_floor) or mass_floor < 0:
        raise ValueError("Require a finite nonnegative conditional display mass floor.")
    nu = _checked_joint(game, candidate)
    mu_a, mu_b = nu.sum(axis=1), nu.sum(axis=0)
    optimized = probability_array(candidate["demand"])
    conjecture = probability_array(candidate["conditional_mu"])
    if optimized.shape != (game.n,) or conjecture.shape != (game.n, game.n):
        raise ValueError("Require one demand per own state and an n-by-n conditional conjecture.")
    row_sum_error = float(np.max(np.abs(conjecture.sum(axis=1) - 1)))
    if row_sum_error > 1e-8:
        raise ValueError("Every conditional conjecture row, including off-path rows, must sum to one.")

    # Reuse established dependence statistics and the original plotting schema.
    # Its unconditional conjecture is supplied only for the reference curve;
    # all actual conditional-consistency quantities are computed below.
    reference = dict(candidate, mu=np.asarray(candidate.get("mu", mu_b)),
                     demand=optimized)
    result = _marginal_demand_diagnostics(game, reference, mass_floor=mass_floor)
    arrays, summary = result["arrays"], result["summary"]
    positive = mu_a > 0
    zero = ~positive
    defined = mu_a > mass_floor
    actual_conditional = np.full_like(nu, np.nan)
    actual_conditional[positive] = nu[positive] / mu_a[positive, None]
    actual_demand = np.full(game.n, np.nan)
    actual_demand[positive] = np.sum(actual_conditional[positive] * game.D[positive], axis=1)
    displayed_demand = actual_demand.copy()
    displayed_demand[~defined] = np.nan
    conjectured_demand = probability_array(np.sum(game.D * conjecture, axis=1))

    # Complete zero rows using the explicit solver default for transparency.
    # This completed matrix is not described as an observed conditional law.
    completed_default = actual_conditional.copy()
    completed_default[zero] = mu_b
    row_l1 = np.full(game.n, np.nan)
    row_l1[positive] = np.abs(conjecture[positive] - actual_conditional[positive]).sum(axis=1)
    exact_demand_error = optimized - actual_demand
    max_row_l1 = float(np.max(row_l1[positive]))
    conditional_error = float(np.max(np.abs(exact_demand_error[positive])))
    errors = dict(
        optimized_minus_conjectured=optimized - conjectured_demand,
        conjectured_minus_real_marginal=conjectured_demand - arrays["d_real_marginal"],
        conjectured_minus_conditional=conjectured_demand - displayed_demand,
        optimized_minus_conditional=optimized - displayed_demand,
        real_marginal_minus_conditional=arrays["d_real_marginal"] - displayed_demand,
    )
    arrays.update(
        mu_cond=conjecture.copy(), conditional_mu=conjecture.copy(),
        mu_cond_real=actual_conditional,
        mu_cond_real_with_default_off_path=completed_default,
        conditional_row_l1=row_l1,
        conditioning_positive_rows=positive,
        conditioning_zero_rows=zero,
        d_optimized=optimized.copy(),
        d_conjectured_conditional=conjectured_demand,
        d_conditional=displayed_demand,
        d_conditional_positive_mass=actual_demand,
        conditional_defined=defined,
        error_optimized_minus_actual_conditional_positive_mass=exact_demand_error,
        **{"error_" + name: value for name, value in errors.items()},
    )
    if "conditional_demand" in candidate:
        saved_conditional_demand = probability_array(candidate["conditional_demand"])
        if saved_conditional_demand.shape != (game.n,):
            raise ValueError("Saved conditional demand must have one value per own state.")
        arrays["d_saved_conditional"] = saved_conditional_demand.copy()
        summary["saved_conditional_minus_optimized_max"] = float(
            np.max(np.abs(saved_conditional_demand - optimized)))
    summary.update(
        conditional_mass_floor=mass_floor,
        conditional_undefined_nodes=int((~defined).sum()),
        conditional_omitted_own_mass=float(mu_a[~defined].sum()),
        conditional_exact_zero_nodes=int(zero.sum()),
        conditional_exact_zero_indices=np.flatnonzero(zero).tolist(),
        conditional_conjecture_row_sum_error=row_sum_error,
        consistency_conditional_max_row_l1=max_row_l1,
        consistency_conditional_stationary_weighted_l1=float(mu_a[positive] @ row_l1[positive]),
        consistency_conditional_demand_max=conditional_error,
        conditional_residual=max_row_l1,
        off_path_conjecture_minus_default_max_row_l1=(
            float(np.max(np.abs(conjecture[zero] - mu_b).sum(axis=1))) if zero.any() else 0.),
        stationary_weighted_conjectured_demand=float(mu_a @ conjectured_demand),
        errors={name: _weighted_errors(error, mu_a) for name, error in errors.items()},
        conditional_conjecture_definition="Rows are P(rival reputation | own reputation); demand is the rowwise sum of D times the conjecture.",
        zero_row_interpretation="Exact zero-mass own-state rows have no actual conditional law. mu_cond retains the saved off-path conjecture, mu_cond_real is NaN there, and mu_cond_real_with_default_off_path uses the unconditional rival marginal only as the explicit default completion.",
        marginal_reference_interpretation="mu, d_conjectured_marginal and consistency_marginal_l1 are unconditional reference diagnostics; they do not define ConditionalOE demand or its equilibrium closure.",
        interpretation="Consistency matches the entire conditional rival distribution on every strictly positive own-state row. Unconditional marginal demand is a comparison only. This own-state conditional-demand problem is not a full-state Markov-perfect equilibrium certificate.",
    )
    return result


def simulate_ts(game, candidate, *, periods=100000, burn_in=10000,
                trajectories=4, seed=20261017):
    """Reuse actual TS paths, joint-stationary starts and continuous updates.

    The simulation uses the saved own-state mixed policy and the actual two
    posterior draws. Conditional beliefs enter the original policy problem,
    not simulated seller selection, state updates or realized products.
    """
    result = _distribution_simulate_ts(
        game, candidate, periods=periods, burn_in=burn_in,
        trajectories=trajectories, seed=seed)
    result["summary"]["model"] = "ConditionalOE"
    result["summary"]["decision_problem"] = (
        "Saved own-state mixed policy optimized with conditional rival demand; "
        "both sellers follow that policy in the actual TS game.")
    return result


def deviation_diagnostics(game, candidate, *, tol=1e-10):
    """Retain full-state deviations as a distinct actual-game diagnostic."""
    result = _distribution_deviation_diagnostics(game, candidate, tol=tol)
    result["summary"]["interpretation"] = (
        "Unrestricted two-state unilateral discounted deviations against the "
        "candidate own-state mixed policy, weighted by its actual joint "
        "stationary law. ConditionalOE optimality conditions use conditional "
        "demand and an own-state decision problem; they do not assert optimality "
        "under unrestricted full-state information.")
    return result


def run_diagnostics(game, candidate, *, periods=100000, burn_in=10000,
                    trajectories=4, seed=20261017):
    """Original runner-compatible report schema, with conditional closure."""
    return dict(
        simulation=simulate_ts(game, candidate, periods=periods, burn_in=burn_in,
                               trajectories=trajectories, seed=seed),
        demand=demand_diagnostics(game, candidate),
        deviations=deviation_diagnostics(game, candidate),
    )
