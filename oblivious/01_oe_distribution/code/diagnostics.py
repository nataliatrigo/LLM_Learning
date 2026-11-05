"""Diagnostics for mixed OE distribution decisions closed by the actual joint marginal.

The decision problem uses unconditional rival demand. Actual-game conditional
demand, continuous simulations, and two-state deviations are diagnostics; they
are not substituted into the equilibrium conditions.
"""

import numpy as np

from oblivious.oe_p0.code.diagnostics import (
    _checked_joint, _policy, posterior_mean, simulate_ts as _simulate_ts,
)
from oblivious.oe_p0.code.primitives import probability_array


def _weighted_errors(error, weights):
    error, weights = np.asarray(error), np.asarray(weights)
    valid = np.isfinite(error)
    mass = float(weights[valid].sum())
    return dict(
        max_absolute=float(np.max(np.abs(error[valid]))) if valid.any() else None,
        stationary_weighted_absolute=float(weights[valid] @ np.abs(error[valid]) / mass) if mass else None,
        stationary_weighted_rmse=float(np.sqrt(weights[valid] @ error[valid] ** 2 / mass)) if mass else None,
        stationary_weighted_signed=float(weights[valid] @ error[valid] / mass) if mass else None,
        included_stationary_mass=mass,
    )


def demand_diagnostics(game, candidate, *, mass_floor=1e-15):
    """Separate consistency error from the remaining conditional-demand error.

    The first is d_mu - D @ actual_mu_B, which vanishes at a solution. The
    second is d_mu - E[D(x,X_B)|X_A=x], which need not vanish: even the real
    marginal does not reveal the conditional distribution of the rival.
    """
    nu = _checked_joint(game, candidate)
    mu_a, mu_b = nu.sum(axis=1), nu.sum(axis=0)
    mu = np.asarray(candidate["mu"], dtype=float)
    # A normalized marginal can sum to 1 plus one ULP, so D@mu can inherit
    # boundary roundoff. Use the shared 32-eps validator, not a model tolerance.
    optimized = probability_array(candidate["demand"])
    if (not np.isfinite(mass_floor) or mass_floor < 0 or mu.shape != (game.n,)
            or not np.isfinite(mu).all() or np.any(mu < 0)
            or abs(mu.sum() - 1) > 1e-8 or optimized.shape != (game.n,)):
        raise ValueError("Valid conjectured marginal, demand and nonnegative mass floor required.")
    conjectured, marginal = game.D @ mu, game.D @ mu_b
    defined = mu_a > mass_floor
    conditional = np.full(game.n, np.nan)
    conditional[defined] = np.sum(nu * game.D, axis=1)[defined] / mu_a[defined]
    errors = dict(
        optimized_minus_conjectured=optimized - conjectured,
        conjectured_minus_real_marginal=conjectured - marginal,
        conjectured_minus_conditional=conjectured - conditional,
        real_marginal_minus_conditional=marginal - conditional,
    )
    independent = np.outer(mu_a, mu_b)
    dependence = nu - independent
    means = np.array([mu_a @ game.grid, mu_b @ game.grid])
    centered_a, centered_b = game.grid - means[0], game.grid - means[1]
    covariance = float(centered_a @ nu @ centered_b)
    variances = np.array([mu_a @ centered_a ** 2, mu_b @ centered_b ** 2])
    scale = float(np.sqrt(variances.prod()))
    positive = nu > 0
    # Positive nu implies positive marginals. Work in logs to avoid underflow
    # in the outer product for very small probabilities at high rho.
    ia, ib = np.nonzero(positive)
    mutual_information = float(np.sum(nu[positive] * (
        np.log(nu[positive]) - np.log(mu_a[ia]) - np.log(mu_b[ib]))))
    summary = dict(
        conditional_mass_floor=mass_floor,
        conditional_undefined_nodes=int((~defined).sum()),
        conditional_omitted_own_mass=float(mu_a[~defined].sum()),
        consistency_marginal_l1=float(np.abs(mu - mu_b).sum()),
        marginal_symmetry_l1=float(np.abs(mu_a - mu_b).sum()),
        actual_stationary_market_share_A=float(np.sum(nu * game.D)),
        independent_real_marginal_market_share_A=float(mu_a @ marginal),
        stationary_weighted_conjectured_demand=float(mu_a @ conjectured),
        errors={name: _weighted_errors(error, mu_a) for name, error in errors.items()},
        joint_dependence=dict(
            total_variation_from_product_marginals=float(.5 * np.abs(dependence).sum()),
            maximum_absolute_from_product_marginals=float(np.max(np.abs(dependence))),
            reputation_covariance=covariance,
            reputation_correlation=covariance / scale if scale else None,
            mutual_information_nats=mutual_information,
        ),
        interpretation="Consistency uses the entire actual marginal. Conditional-demand differences diagnose the remaining oblivious approximation, not failure of the marginal consistency condition.",
    )
    arrays = dict(
        grid=game.grid.copy(), mu=mu.copy(), mu_a=mu_a, mu_b=mu_b,
        d_optimized=optimized.copy(), d_conjectured_marginal=conjectured,
        d_real_marginal=marginal, d_conditional=conditional,
        conditional_defined=defined, joint_minus_independent=dependence,
        **{"error_" + name: value for name, value in errors.items()},
    )
    return dict(arrays=arrays, summary=summary)


def simulate_ts(game, candidate, *, periods=100000, burn_in=10000,
                trajectories=4, seed=20261017):
    """Use the shared actual-TS simulator, with distribution-based reporting.

    Its internal scalar benchmark fields are solely a reporting adapter. No
    scalar benchmark enters seller selection, policy choice, or state updates;
    all scalar-p0 labels are removed from this experiment's returned output.
    """
    nu = _checked_joint(game, candidate)
    stationary_means = np.array([nu.sum(axis=1), nu.sum(axis=0)]) @ posterior_mean(game.grid, game.params)
    adapted = dict(candidate, F=float(stationary_means[1]), p0=float(stationary_means[1]))
    result = _simulate_ts(game, adapted, periods=periods, burn_in=burn_in,
                          trajectories=trajectories, seed=seed)
    summary = result["summary"]
    summary.pop("p0")
    summary.pop("stationary_F")
    summary["stationary_posterior_means"] = stationary_means.tolist()
    for source, destination in (
        ("ts_B_ci_contains_stationary_F", "ts_B_ci_contains_stationary_posterior_mean"),
        ("posterior_B_ci_contains_stationary_F", "posterior_B_ci_contains_stationary_posterior_mean"),
    ):
        summary[destination] = summary.pop(source)
    for rows in (summary["estimates"], *result["path_rows"]):
        for key in list(rows):
            if "stationary_F" in key:
                rows[key.replace("stationary_F", "stationary_posterior_mean")] = rows.pop(key)
    mass = result["arrays"]["empirical_mass"]
    stationary_marginals = np.array([nu.sum(axis=1), nu.sum(axis=0)])
    cdf_error = np.cumsum(mass - stationary_marginals, axis=1)
    result["arrays"]["stationary_marginals"] = stationary_marginals
    result["arrays"]["empirical_minus_stationary_cdf"] = cdf_error
    summary["marginal_comparison"] = dict(
        deposition="Linear mass deposition of continuous reputation states onto the numerical grid",
        l1=np.abs(mass - stationary_marginals).sum(axis=1).tolist(),
        maximum_cdf_gap=np.abs(cdf_error).max(axis=1).tolist(),
        wasserstein_grid=(np.abs(cdf_error[:, :-1]) @ np.diff(game.grid)).tolist(),
        interpretation="Pooled finite-path diagnostics against the grid law, without an IID-time confidence claim",
    )
    summary["interpretation"] = "Finite continuous-state paths, initialized independently from the joint grid stationary law. Grid marginals and continuous simulations may differ due to discretization, initialization and slow mixing; this is not a continuous-state equilibrium certificate."
    return result


def deviation_diagnostics(game, candidate, *, tol=1e-10):
    """Best responses see both reputations; the competing own-state policy stays fixed."""
    nu, policy = _checked_joint(game, candidate), _policy(candidate)
    a = np.broadcast_to(policy[:, None], nu.shape).copy()
    b = np.broadcast_to(policy[None, :], nu.shape).copy()
    result = game.verify(a, b, tol=tol, weights=nu)
    return dict(
        arrays={key: value for key, value in result.items() if key != "summary"},
        summary={**result["summary"],
                 "interpretation": "Unrestricted two-state unilateral discounted deviations against the candidate own-state mixed policy, weighted by its actual joint stationary law. Positive gains do not invalidate this experiment's OE distribution optimality condition, but distinguish it from MPE."},
    )


def run_diagnostics(game, candidate, *, periods=100000, burn_in=10000,
                    trajectories=4, seed=20261017):
    """Run actual-game diagnostics without changing or re-solving the candidate."""
    return dict(
        simulation=simulate_ts(game, candidate, periods=periods, burn_in=burn_in,
                               trajectories=trajectories, seed=seed),
        demand=demand_diagnostics(game, candidate),
        deviations=deviation_diagnostics(game, candidate),
    )
