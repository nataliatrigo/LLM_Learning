"""Actual-game diagnostics for the scalar p0 approximation.

Saved verified policies from OE distribution are loaded read-only;
any transfer to a different numerical game is explicitly identified.
"""

from dataclasses import asdict
from pathlib import Path
import hashlib
import json

import numpy as np
from scipy.stats import t as student_t

from oblivious.oe_p0.code.reference_paths import read_saved_bundle, reference_source
from oblivious.oe_p0.code.primitives import probability_array


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def posterior_mean(x, params):
    """Mean of the mature Beta posterior, distinct from reputation x."""
    return (1 - params.rho + np.asarray(x)) / (3 - 2 * params.rho)


def _policy(evaluation):
    for key in ("policy0", "policy"):
        if key in evaluation:
            return np.asarray(evaluation[key], dtype=float)
    raise KeyError("The evaluation must contain policy0 or policy (probability of product 2).")


def _checked_joint(game, evaluation):
    nu = np.asarray(evaluation["nu"], dtype=float)
    if (nu.shape != (game.n, game.n) or not np.isfinite(nu).all()
            or np.any(nu < 0) or abs(nu.sum() - 1) > 1e-9):
        raise ValueError("A valid normalized joint stationary distribution is required.")
    return nu / nu.sum()


def _interval(values):
    """A t interval over independent path means; no IID-time assumption."""
    values = np.asarray(values, dtype=float)
    mean = values.mean(axis=0)
    if len(values) < 2:
        return {"mean": _plain(mean), "se": None, "ci95": None}
    se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    half = student_t.ppf(.975, len(values) - 1) * se
    return {"mean": _plain(mean), "se": _plain(se),
            "ci95": _plain(np.stack((mean - half, mean + half), axis=-1))}


def _mass_from_states(grid, states):
    """Linear deposition of empirical continuous states onto the saved grid."""
    states = np.asarray(states).ravel()
    low = np.clip(np.searchsorted(grid, states, side="right") - 1, 0, len(grid) - 2)
    weight = (states - grid[low]) / (grid[low + 1] - grid[low])
    return (np.bincount(low, weights=1 - weight, minlength=len(grid))
            + np.bincount(low + 1, weights=weight, minlength=len(grid))) / len(states)


def simulate_ts(game, evaluation, *, periods=100000, burn_in=10000,
                trajectories=4, seed=20261017):
    """Simulate actual two-seller TS with independent mixed product choices.

    Both posterior draws and both posterior means are recorded BEFORE selection
    at every retained calendar period, including the unselected seller.  Each
    independent trajectory starts from one JOINT draw from nu. The selected
    seller draws product 2 with probability alpha at its own reputation, using
    linear interpolation between grid nodes. Separate random streams generate
    seller selection, product choice and success conditional on that product.
    Off-grid interpolation is a numerical extension, not an optimality claim.

    """
    if periods < 1 or burn_in < 0 or trajectories < 1:
        raise ValueError("Positive periods/trajectories and nonnegative burn-in required.")
    grid, p = game.grid, game.params
    policy = _policy(evaluation)
    if (policy.shape != (game.n,) or not np.isfinite(policy).all()
            or np.any(policy < -1e-14) or np.any(policy > 1 + 1e-14)):
        raise ValueError("The product-2 policy must be a finite probability on the grid.")
    policy = np.clip(policy, 0, 1)
    nu = _checked_joint(game, evaluation)
    shape = (trajectories, periods)
    states, theta, means, profits = [np.empty(shape + (2,)) for _ in range(4)]
    profits.fill(0)
    selected = np.empty(shape, dtype=np.int8)
    products = np.empty(shape, dtype=np.int8)
    successes = np.empty(shape, dtype=bool)
    chosen_probability = np.empty(shape)
    product_uniform, success_uniform = np.empty(shape), np.empty(shape)
    initial, final = np.empty((trajectories, 2)), np.empty((trajectories, 2))
    for path, child_seed in enumerate(np.random.SeedSequence(seed).spawn(trajectories)):
        rng_initial, rng_a, rng_b, rng_product, rng_success = [
            np.random.default_rng(s) for s in child_seed.spawn(5)]
        flat = rng_initial.choice(game.n ** 2, p=nu.ravel())
        x = grid[np.array(np.unravel_index(flat, nu.shape))].copy()
        initial[path] = x
        product_draws = rng_product.random(periods + burn_in)
        outcome_draws = rng_success.random(periods + burn_in)
        for time in range(periods + burn_in):
            draw = np.array([rng_a.beta(1 + x[0] / (1 - p.rho), 1 + (1 - x[0]) / (1 - p.rho)),
                             rng_b.beta(1 + x[1] / (1 - p.rho), 1 + (1 - x[1]) / (1 - p.rho))])
            seller = int(draw[1] > draw[0])  # Exact TS ties choose A (probability zero).
            alpha = float(np.interp(x[seller], grid, policy))
            chosen_2 = bool(product_draws[time] < alpha)
            outcome = outcome_draws[time] < p.p1 + (p.p2 - p.p1) * chosen_2
            if time >= burn_in:
                k = time - burn_in
                states[path, k], theta[path, k] = x, draw
                means[path, k] = posterior_mean(x, p)
                selected[path, k], products[path, k] = seller, 1 + chosen_2
                successes[path, k] = outcome
                chosen_probability[path, k] = alpha
                product_uniform[path, k], success_uniform[path, k] = product_draws[time], outcome_draws[time]
                profits[path, k, seller] = p.R - (p.c2 if chosen_2 else p.c1)
            x[seller] = p.rho * x[seller] + (1 - p.rho) * outcome
        final[path] = x

    path_rows = []
    for path in range(trajectories):
        row = {
            "mean_ts": theta[path].mean(axis=0),
            "mean_posterior": means[path].mean(axis=0),
            "ts_minus_posterior": (theta[path] - means[path]).mean(axis=0),
            "ts_b_minus_stationary_F": theta[path, :, 1].mean() - evaluation["F"],
            "posterior_b_minus_stationary_F": means[path, :, 1].mean() - evaluation["F"],
            "market_shares": np.array([(selected[path] == j).mean() for j in (0, 1)]),
            "product2_purchase_fraction": (products[path] == 2).mean(),
            "mean_chosen_product2_probability": chosen_probability[path].mean(),
            "product2_per_calendar_period": np.array([
                ((selected[path] == j) & (products[path] == 2)).mean() for j in (0, 1)]),
            "mean_profit_per_seller": profits[path].mean(axis=0),
            "mean_reputation": states[path].mean(axis=0),
            "realized_success_fraction": successes[path].mean(),
        }
        path_rows.append(_plain(row))
    estimates = {key: _interval([row[key] for row in path_rows]) for key in path_rows[0]}
    running_periods = np.unique(np.concatenate((
        [1, periods], np.geomspace(1, periods, min(500, periods)).astype(int))))
    take = running_periods - 1
    running_ts = np.cumsum(theta[:, :, 1], axis=1)[:, take] / running_periods
    running_posterior = np.cumsum(means[:, :, 1], axis=1)[:, take] / running_periods
    empirical = np.stack([_mass_from_states(grid, states[:, :, j]) for j in (0, 1)])
    summary = dict(
        periods=periods, burn_in=burn_in, trajectories=trajectories, seed=seed,
        estimates=estimates, p0=float(evaluation["p0"]), stationary_F=float(evaluation["F"]),
        initial_states=initial.tolist(),
        initialization="Independent draws from the joint stationary nu, not product marginals",
        recorded_time="Predecision states, both Beta draws and both posterior means every retained calendar period",
        policy_mode="mixed",
        policy_extension="Linear interpolation of alpha=P(product 2 | own reputation, selected); independent Bernoulli choice. This extension does not certify off-grid indifference or optimality.",
        product_choice="Independent product uniform compared to own-state alpha; success uses the realized selected product",
        random_streams="Five independent streams per path: joint initialization, TS A, TS B, product choice, success",
        customer_selection="Independent actual Beta draws for A and B; exactly one seller selected",
        uncertainty="95% Student-t intervals over independent path means, with paths-1 degrees of freedom; time observations are not treated as IID. Few independent paths give imprecise uncertainty estimates, especially with slow mixing.",
        interpretation="Finite-path evidence, not a proof of ergodicity or off-grid optimality. Grid stationary F and continuous simulation may differ because of discretization, initialization or slow mixing.",
    )
    difference_ci = estimates["ts_minus_posterior"]["ci95"]
    f_ci = estimates["mean_ts"]["ci95"]
    posterior_f_ci = estimates["mean_posterior"]["ci95"]
    summary["ts_minus_posterior_B_ci_contains_zero"] = (
        bool(difference_ci[1][0] <= 0 <= difference_ci[1][1]) if difference_ci else None)
    summary["ts_B_ci_contains_stationary_F"] = (
        bool(f_ci[1][0] <= evaluation["F"] <= f_ci[1][1]) if f_ci else None)
    summary["posterior_B_ci_contains_stationary_F"] = (
        bool(posterior_f_ci[1][0] <= evaluation["F"] <= posterior_f_ci[1][1]) if posterior_f_ci else None)
    return dict(arrays=dict(states=states, theta=theta, posterior_mean=means,
                           selected=selected, product=products, success=successes, profit=profits,
                           chosen_probability=chosen_probability, product_uniform=product_uniform,
                           success_uniform=success_uniform,
                           initial_state=initial, final_state=final, empirical_mass=empirical,
                           running_periods=running_periods, running_ts_b=running_ts,
                           running_posterior_b=running_posterior),
                summary=summary, path_rows=path_rows)


def demand_diagnostics(game, evaluation, *, mass_floor=1e-15):
    """Compare fixed, unconditional-marginal and own-state conditional demand.

    Conditional demand is undefined on negligible own-state mass; those points
    remain NaN, are omitted from the supported max error and have their omitted
    mass reported. The DP always uses evaluation['d_fixed'], never these tables.
    """
    nu = _checked_joint(game, evaluation)
    mu_a, mu_b = nu.sum(axis=1), nu.sum(axis=0)
    fixed = np.asarray(evaluation["d_fixed"], dtype=float)
    if (not np.isfinite(mass_floor) or mass_floor < 0 or fixed.shape != (game.n,)
            or not np.isfinite(fixed).all() or np.any(fixed < 0) or np.any(fixed > 1)):
        raise ValueError("Demand must be a probability on the grid; mass_floor must be nonnegative.")
    marginal = game.D @ mu_b
    defined = mu_a > mass_floor
    conditional = np.full(game.n, np.nan)
    conditional[defined] = np.sum(game.D * nu, axis=1)[defined] / mu_a[defined]
    errors = {"fixed_minus_marginal": fixed - marginal,
              "fixed_minus_conditional": fixed - conditional,
              "marginal_minus_conditional": marginal - conditional}
    summary = {"conditional_mass_floor": mass_floor,
               "conditional_undefined_nodes": int((~defined).sum()),
               "conditional_omitted_own_mass": float(mu_a[~defined].sum()),
               "actual_stationary_market_share_A": float(np.sum(game.D * nu)),
               "independent_marginal_market_share_A": float(mu_a @ marginal),
               "weighted_fixed_demand": float(mu_a @ fixed), "errors": {}}
    for name, error in errors.items():
        valid = np.isfinite(error)
        mass = mu_a[valid].sum()
        summary["errors"][name] = dict(
            max_absolute=float(np.max(np.abs(error[valid]))) if valid.any() else None,
            stationary_weighted_absolute=float(mu_a[valid] @ np.abs(error[valid]) / mass) if mass else None,
            stationary_weighted_rmse=float(np.sqrt(mu_a[valid] @ error[valid] ** 2 / mass)) if mass else None,
            stationary_weighted_signed=float(mu_a[valid] @ error[valid] / mass) if mass else None,
            included_stationary_mass=float(mass))
    return dict(arrays=dict(grid=game.grid.copy(), d_fixed=fixed, d_marginal=marginal,
                           d_conditional=conditional, conditional_defined=defined,
                           mu_a=mu_a, mu_b=mu_b,
                           **{"error_" + name: error for name, error in errors.items()}), summary=summary)


def deviation_diagnostics(game, evaluation, tol=1e-10):
    """Unilateral best responses may depend on BOTH states; rival stays fixed."""
    nu = _checked_joint(game, evaluation)
    policy = _policy(evaluation)
    sigma_a = np.broadcast_to(policy[:, None], nu.shape).copy()
    sigma_b = np.broadcast_to(policy[None, :], nu.shape).copy()
    result = game.verify(sigma_a, sigma_b, tol=tol, weights=nu)
    return dict(arrays={key: value for key, value in result.items() if key != "summary"},
                summary={**result["summary"],
                         "interpretation": "OE p0 self-consistency does not establish optimality with full state information. These are unrestricted two-state unilateral discounted gains against the OE p0 policy, weighted by its joint stationary law."})


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _matching(saved, params, *, skip_rho=False):
    try:
        return all(key in saved and np.isclose(saved[key], value, rtol=0, atol=1e-12)
                   for key, value in asdict(params).items()
                   if not (skip_rho and key == "rho"))
    except (TypeError, ValueError):
        return False


def load_existing_references(params, project_root, n=None, order=None):
    """Read one verified, matching policy from the active real-marginal model.

    Prefer the requested grid, then the finest verified grid. When several
    runs have that grid, prefer matching demand quadrature and then source path.
    Certification always belongs to the source numerical game.
    """
    root = Path(project_root).resolve()
    folder = root / "oblivious/01_oe_distribution" / "data"
    references, provenance, excluded, catalog, candidates = [], {}, [], [], []

    def record(path):
        name = reference_source(path, root)
        provenance[name] = _hash(path)
        return name

    for manifest_path in sorted(folder.glob("*/manifest.json")):
        source = reference_source(manifest_path, root)
        try:
            manifest = json.loads(manifest_path.read_text())
            config = manifest.get("configuration", {})
            economic_params = config.get("parameters", config)
            if (manifest.get("status") != "complete"
                    or not _matching(economic_params, params, skip_rho=True)):
                excluded.append(dict(source=source, reason="Run incomplete or economic parameters do not match"))
                continue
            record(manifest_path)
            for case_path in sorted((manifest_path.parent / "arrays").glob("rho_*_n*.json")):
                if "_diagnostics_" in case_path.stem:
                    continue
                case_source = reference_source(case_path, root)
                try:
                    metadata = json.loads(case_path.read_text())
                    config = metadata.get("configuration", {})
                    if not _matching(config.get("parameters", {}), params):
                        excluded.append(dict(source=case_source, reason="Case parameters or rho do not match"))
                        continue
                    selected = metadata.get("selected")
                    if not (selected and selected.get("verified")
                            and selected.get("candidate_verified")
                            and selected.get("dp_diagnostics", {}).get("verified")
                            and selected.get("stationary_diagnostics", {}).get("verified")):
                        excluded.append(dict(source=case_source, reason="Selected OE distribution candidate is not verified"))
                        continue
                    case = read_saved_bundle(case_path.with_suffix(""))
                    selected = case["selected"]
                    arrays = {key: np.asarray(selected[key], dtype=float)
                              for key in ("grid", "policy", "mu", "demand", "nu", "mu_a", "mu_b")}
                    grid, policy, mu, nu = (arrays[key] for key in ("grid", "policy", "mu", "nu"))
                    # Permit machine roundoff at probability boundaries without
                    # rewriting the native arrays retained in the report.
                    probability_array(policy)
                    probability_array(arrays["demand"])
                    valid = (grid.ndim == 1 and len(grid) >= 3 and grid[0] == 0 and grid[-1] == 1
                             and np.isfinite(grid).all() and np.all(np.diff(grid) > 0)
                             and case.get("n") == len(grid) and config.get("n") == len(grid)
                             and abs(float(case.get("rho", np.nan)) - params.rho) <= 1e-12
                             and policy.shape == mu.shape == arrays["demand"].shape == grid.shape
                             and arrays["mu_a"].shape == arrays["mu_b"].shape == grid.shape
                             and nu.shape == (len(grid), len(grid))
                             and all(np.isfinite(value).all() for value in arrays.values())
                             and np.all(mu >= 0) and np.all(nu >= 0)
                             and abs(mu.sum() - 1) <= 1e-8 and abs(nu.sum() - 1) <= 1e-8
                             and np.allclose(arrays["mu_a"], nu.sum(axis=1), rtol=0, atol=1e-8)
                             and np.allclose(arrays["mu_b"], nu.sum(axis=0), rtol=0, atol=1e-8)
                             and np.max(np.abs(mu - arrays["mu_b"])) <= config.get("tolerance", 1e-7))
                    if not valid:
                        raise ValueError("Malformed or inconsistent saved probability arrays")
                    record(case_path)
                    arrays_path = case_path.with_suffix(".npz")
                    candidates.append(dict(
                        kind="OE_REAL_MARGINAL", label="OE distribution",
                        grid=grid, policy=policy, mu=mu, demand=arrays["demand"],
                        native_nu=nu, native_mu_a=arrays["mu_a"], native_mu_b=arrays["mu_b"],
                        original_summary=selected.get("summary", {}), verified=True,
                        symmetry_error=float(np.max(np.abs(nu - nu.T))),
                        source=record(arrays_path), source_tag=case_path.stem,
                        source_order=config.get("order")))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    excluded.append(dict(source=case_source, reason=f"Invalid saved case: {error}"))
        except (OSError, ValueError, TypeError) as error:
            excluded.append(dict(source=source, reason=f"Invalid run manifest: {error}"))

    if candidates:
        chosen = min(candidates, key=lambda item: (
            n is not None and len(item["grid"]) != n,
            -len(item["grid"]),
            order is not None and item["source_order"] != order,
            item["source"]))
        references.append(chosen)
        catalog = [dict(source=item["source"], grid_points=len(item["grid"]),
                        source_order=item["source_order"], included=item is chosen)
                   for item in candidates]
    else:
        excluded.append(dict(source=reference_source(folder, root),
                             reason="No verified OE distribution case with matching parameters and rho"))
    return dict(references=references, provenance=provenance, excluded=excluded,
                catalog=catalog,
                selection_rule="Active OE distribution only: matching economic parameters and rho; prefer the comparison grid, otherwise the finest verified grid; prefer matching demand quadrature among ties. Native certification concerns the saved grid and quadrature.")


def _regions(grid, policy, cutoff=.5):
    """Contiguous sampled-node blocks, not inferred off-grid switch points."""
    active = np.asarray(policy) >= cutoff
    starts = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    ends = np.flatnonzero(active & ~np.r_[active[1:], False])
    return [[float(grid[a]), float(grid[b])] for a, b in zip(starts, ends)]


def probability_regions(grid, policy, tolerance=1e-8):
    """Describe mixed-policy support without treating alpha>=.5 as optimality."""
    alpha = np.asarray(policy, dtype=float)
    return dict(
        definition="Endpoints bound contiguous sampled grid nodes only; full alpha arrays are saved. Alpha>=0.5 describes probability, not strict optimality or the whole support of product 2.",
        boundary_tolerance=tolerance,
        probability_at_least_half=_regions(grid, alpha, cutoff=.5),
        positive_product2_support=_regions(grid, (alpha > tolerance).astype(float)),
        strictly_mixed_support=_regions(grid, ((alpha > tolerance) & (alpha < 1 - tolerance)).astype(float)),
        product2_probability_one=_regions(grid, (alpha >= 1 - tolerance).astype(float)))


def _profile_statistics(game, a, b, nu):
    shares = np.array([np.sum(nu * game.selection[j]) for j in (0, 1)])
    use = np.array([np.sum(nu * game.selection[j] * sigma) for j, sigma in enumerate((a, b))])
    return dict(market_shares=shares.tolist(), product2_purchase_fraction=float(use.sum()),
                product2_per_calendar_period=use.tolist(),
                product2_conditional_on_selection=np.divide(use, shares, out=np.zeros(2), where=shares > 0).tolist(),
                mean_profit_per_seller=np.sum(game.rewards(a, b) * nu, axis=(1, 2)).tolist(),
                mean_reputation=[float(nu.sum(axis=1) @ game.grid), float(nu.sum(axis=0) @ game.grid)],
                mean_posterior=[float(nu.sum(axis=1) @ posterior_mean(game.grid, game.params)),
                                float(nu.sum(axis=0) @ posterior_mean(game.grid, game.params))])


def compare_existing(game, evaluation, project_root):
    """Evaluate OE distribution in the OE p0 comparison game.

    This recomputes stationary performance without solving an equilibrium.
    Source certification is retained only for its native numerical game.
    """
    from .core import stationary_joint

    loaded = load_existing_references(game.params, project_root, n=game.n, order=game.order)
    nu = _checked_joint(game, evaluation)
    policy = _policy(evaluation)
    scalar_a = np.broadcast_to(policy[:, None], nu.shape)
    scalar_b = np.broadcast_to(policy[None, :], nu.shape)
    arrays = {"grid": game.grid.copy(), "scalar_policy": policy,
              "scalar_nu": nu, "scalar_mu_a": nu.sum(axis=1), "scalar_mu_b": nu.sum(axis=0)}
    rows = []
    for index, ref in enumerate(loaded["references"]):
        key = f"reference_{index:03d}"
        arrays.update({key + "_native_" + name: ref[name]
                       for name in ("grid", "policy", "mu", "demand")})
        arrays.update({key + "_" + name: ref[name]
                       for name in ("native_nu", "native_mu_a", "native_mu_b")})
        alpha = probability_array(np.interp(game.grid, ref["grid"], ref["policy"]))
        a, b = np.broadcast_to(alpha[:, None], nu.shape), np.broadcast_to(alpha[None, :], nu.shape)
        arrays[key + "_sigma_a"], arrays[key + "_sigma_b"] = a, b
        same_grid = np.array_equal(ref["grid"], game.grid)
        same_game = same_grid and ref.get("source_order") == game.order
        row = dict(key=key, label=ref["label"], model=ref["kind"],
                   source=ref["source"], source_grid=len(ref["grid"]), evaluation_grid=game.n,
                   source_quadrature_order=ref.get("source_order"), evaluation_quadrature_order=game.order,
                   native_certificate_verified=True, symmetry_error=0.,
                   evaluation_profile=("Native probability policy on the same grid and demand quadrature"
                       if same_game else "Native probability policy with changed demand quadrature; equilibrium certificate not transferred"
                       if same_grid else "Linearly interpolated probabilities on the comparison grid; equilibrium certificate not transferred"),
                   policy_difference_max=[float(np.max(np.abs(a - scalar_a))), float(np.max(np.abs(b - scalar_b)))],
                   policy_difference_weighted_by_scalar_nu=[float(np.sum(nu * np.abs(a - scalar_a))), float(np.sum(nu * np.abs(b - scalar_b)))],
                   product2_regions=probability_regions(ref["grid"], ref["policy"]),
                   native_summary=ref["original_summary"])
        try:
            stationary = stationary_joint(game, game.transition(a, b))
            reference_nu = np.asarray(stationary["nu"])
            if not stationary.get("summary", {}).get("verified", False):
                raise RuntimeError("Reference-profile stationarity was not verified")
            stats = _profile_statistics(game, a, b, reference_nu)
            reference_demand = demand_diagnostics(game, {"nu": reference_nu, "d_fixed": evaluation["d_fixed"]})
            row.update(status="evaluated", statistics=stats,
                       stationary_diagnostics={k: _plain(v) for k, v in stationary.items() if k not in ("nu", "mu", "mu_a", "mu_b")},
                       demand_errors_against_scalar_fixed=reference_demand["summary"]["errors"],
                       marginal_l1_to_scalar=[float(np.abs(reference_nu.sum(axis=1) - nu.sum(axis=1)).sum()),
                                              float(np.abs(reference_nu.sum(axis=0) - nu.sum(axis=0)).sum())])
            arrays.update({key + "_nu": reference_nu,
                           key + "_mu_a": reference_nu.sum(axis=1), key + "_mu_b": reference_nu.sum(axis=0),
                           key + "_d_marginal": reference_demand["arrays"]["d_marginal"],
                           key + "_d_conditional": reference_demand["arrays"]["d_conditional"]})
        except (ArithmeticError, ValueError, RuntimeError) as error:
            row.update(status="stationary_evaluation_failed", error=f"{type(error).__name__}: {error}")
        rows.append(row)
    return dict(arrays=arrays, summary=dict(
        scalar_statistics=_profile_statistics(game, scalar_a, scalar_b, nu),
        scalar_product2_regions=probability_regions(game.grid, policy), references=rows,
        excluded=loaded["excluded"], catalog=loaded["catalog"], selection_rule=loaded["selection_rule"],
        interpretation="Same economic parameters and numerical game for performance. OE distribution retains verification for its native grid and demand quadrature. Stationary comparison does not rerun equilibrium solvers."),
        provenance=loaded["provenance"])
