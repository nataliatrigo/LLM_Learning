"""Read and independently check the two retained equilibrium experiments.

No equilibrium is solved here.  In particular, an unavailable or unverified
real-game-marginal OE distribution is an error, never a reason to substitute another model.
All source equilibrium files are opened read-only.
"""

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from oblivious.oe_p0.code.core import posterior_mean
from oblivious.oe_p0.code.game import Game
from oblivious.oe_p0.code.primitives import (
    Parameters, check_grid, demand_matrix, demand_probability, probability_array,
)
from oblivious.oe_p0.code.run_experiment import case_tag, load_bundle
from model_names import MODEL_LABELS


LABELS = MODEL_LABELS
STATIONARITY_TOLERANCE = 1e-10
CONSISTENCY_TOLERANCE = 1e-7
DEMAND_TOLERANCE = 1e-6


def _hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def _probability_law(values, shape, name):
    values = np.asarray(values, dtype=float)
    if (values.shape != shape or not np.isfinite(values).all()
            or np.any(values < 0) or abs(float(values.sum()) - 1) > 1e-10):
        raise ValueError(f"{name} must be a normalized nonnegative probability law of shape {shape}.")
    return values.copy()


def _source_case(project_root, key, run, rho, n, candidate_index):
    experiment = "oblivious/01_oe_distribution" if key == "oe" else "oblivious/02_oe_p0"
    if not isinstance(run, str) or not run or Path(run).name != run or run in (".", ".."):
        raise ValueError("Use a run folder name inside the requested experiment's data directory.")
    data_root = (project_root / experiment / "data").resolve()
    directory = (data_root / run).resolve()
    if directory.parent != data_root:
        raise ValueError(f"Run must belong to {experiment}/data.")
    manifest_path = directory / "manifest.json"
    tag = case_tag(float(rho), int(n))
    base = directory / "arrays" / tag
    required = [manifest_path, base.with_suffix(".json"), base.with_suffix(".npz")]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {LABELS[key]} equilibrium inputs: {', '.join(missing)}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"{LABELS[key]} run is not complete: {manifest_path}")
    if tag not in manifest.get("completed_cases", []):
        raise ValueError(f"{LABELS[key]} manifest does not mark {tag} complete.")
    if key == "oe":
        description = manifest.get("model", "").lower()
        if "actual joint" not in description or "marginal" not in description:
            raise ValueError("OE distribution manifest does not identify actual-joint marginal consistency.")
    case = load_bundle(base)
    config = case.get("configuration", {})
    if case.get("n") != n or config.get("n") != n or case.get("rho") != rho:
        raise ValueError(f"{LABELS[key]} saved case does not match requested rho and grid size.")
    try:
        parameters = config["parameters"]
        if set(parameters) != set(Parameters.__dataclass_fields__):
            raise ValueError("Missing or unknown model primitives.")
        params = Parameters(**parameters)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid saved primitives for {LABELS[key]}: {exc}") from exc
    if params.rho != rho:
        raise ValueError(f"{LABELS[key]} saved rho differs from the requested rho.")
    manifest_config = manifest.get("configuration", {})
    for name, value in asdict(params).items():
        if name == "rho":
            if value not in manifest_config.get("rhos", []):
                raise ValueError(f"{LABELS[key]} rho is absent from the manifest.")
        elif manifest_config.get(name) != value:
            raise ValueError(f"{LABELS[key]} {name} differs between its manifest and case.")
    order = config.get("order")
    if (not isinstance(order, int) or isinstance(order, bool) or order < 8
            or order != manifest_config.get("order") or n not in manifest_config.get("grids", [])):
        raise ValueError(f"{LABELS[key]} has incompatible grid or quadrature metadata.")
    candidates = case.get("candidates", [])
    if candidate_index is None:
        selected = case.get("selected")
        selection = "saved_selected"
    else:
        if (not isinstance(candidate_index, (int, np.integer)) or isinstance(candidate_index, bool)
                or candidate_index < 0 or candidate_index >= len(candidates)):
            raise ValueError(f"{LABELS[key]} candidate index is out of range (zero-based).")
        selected = candidates[int(candidate_index)]
        selection = int(candidate_index)
    if not isinstance(selected, dict):
        raise ValueError(f"{LABELS[key]} has no selected equilibrium candidate.")
    statuses = {
        "verified": selected.get("verified"),
        "candidate_verified": selected.get("candidate_verified"),
        "dp_verified": selected.get("dp_diagnostics", {}).get("verified"),
        "stationary_verified": selected.get("stationary_diagnostics", {}).get("verified"),
        "demand_verified": case.get("demand_audit", {}).get("verified"),
    }
    if not all(value is True for value in statuses.values()):
        raise ValueError(f"UNVERIFIED {LABELS[key]} candidate ({selection}): {statuses}")
    grid = check_grid(selected.get("grid")).copy()
    if grid.size != n:
        raise ValueError(f"{LABELS[key]} candidate grid has an incompatible size.")
    policy = probability_array(selected.get("policy"))
    if policy.shape != (n,):
        raise ValueError(f"{LABELS[key]} policy must depend only on own mature reputation.")
    nu = _probability_law(selected.get("nu"), (n, n), f"{LABELS[key]} nu")
    mu = _probability_law(selected.get("mu" if key == "oe" else "mu_b"),
                          (n,), f"{LABELS[key]} mu")
    p0 = None if key == "oe" else selected.get("p0")
    if key == "p0" and (not isinstance(p0, (int, float)) or not np.isfinite(p0) or not 0 <= p0 <= 1):
        raise ValueError("OE p0 requires a finite saved benchmark in [0,1].")
    selected_index = candidate_index
    if selected_index is None:
        for index, candidate in enumerate(candidates):
            if (np.array_equal(candidate.get("policy"), policy)
                    and np.array_equal(candidate.get("nu"), nu)
                    and candidate.get("p0") == p0):
                selected_index = index
                break
    source = dict(
        experiment=experiment, run=run, case=tag, directory=str(directory),
        candidate_selection=selection,
        candidate_index=None if selected_index is None else int(selected_index),
        candidate_count=len(candidates),
        verified_candidate_count=sum(candidate.get("verified") is True for candidate in candidates),
        manifest=str(manifest_path), case_json=str(required[1]), case_npz=str(required[2]),
        manifest_sha256=_hash(manifest_path), case_json_sha256=_hash(required[1]),
        case_npz_sha256=_hash(required[2]),
        saved_solver_source_sha256=manifest.get("source_sha256", {}),
    )
    profile = dict(label=LABELS[key], grid=grid, policy=policy.copy(), nu=nu, mu=mu,
                   p0=None if p0 is None else float(p0), case=case, selected=selected, source=source)
    return params, order, profile, statuses


def _demand_audit(game):
    doubled, raw = demand_matrix(game.grid, game.params, 2 * game.order)
    discrepancy = np.abs(doubled - game.D)
    worst = tuple(int(index) for index in np.unravel_index(np.argmax(discrepancy), discrepancy.shape))
    n = game.n
    pairs = sorted({worst, (0, n - 1), (n // 2, n // 2), (n // 4, 3 * n // 4),
                    (max(n // 2 - 1, 0), min(n // 2 + 1, n - 1))})
    checks = []
    for i, j in pairs:
        direct = demand_probability(game.grid[i], game.grid[j], game.params)
        checks.append(dict(i=i, j=j, x=float(game.grid[i]), y=float(game.grid[j]),
                           matrix_probability=float(game.D[i, j]), direct_probability=direct,
                           absolute_error=float(abs(game.D[i, j] - direct))))
    maximum = max(float(discrepancy.max()), max(row["absolute_error"] for row in checks))
    if maximum > DEMAND_TOLERANCE:
        raise ValueError(f"Actual TS demand failed numerical audit: {maximum:g}.")
    return dict(verified=True, order=game.order, doubled_order=2 * game.order,
                doubled_order_max_difference=float(discrepancy.max()),
                independent_checks=checks, maximum_observed_error=maximum,
                tolerance=DEMAND_TOLERANCE, original_raw=game.demand_diagnostics,
                doubled_raw=raw,
                interpretation="Order doubling over the whole grid and independent adaptive quadrature at selected nodes; observed errors are diagnostics, not rigorous global error bounds. Direct Beta-draw selection matches the numerical kernel up to demand integration error.")


def predictions(pair, profile_key):
    """Stationary calendar-time economics under the stored joint law.

    Conditional product use divides by actual selection probability. No product
    probability is rounded, and no independent-marginal approximation is used.
    """
    profile = pair["profiles"][profile_key]
    grid, nu, policy = profile["grid"], profile["nu"], profile["policy"]
    params, game = pair["params"], pair["game"]
    marginals = (nu.sum(axis=1), nu.sum(axis=0))
    result = {}
    policies = (np.broadcast_to(policy[:, None], nu.shape), np.broadcast_to(policy[None, :], nu.shape))
    for index, seller in enumerate(("A", "B")):
        marginal, sigma, demand = marginals[index], policies[index], game.selection[index]
        mean = float(marginal @ grid)
        share = float(np.sum(nu * demand))
        product2 = float(np.sum(nu * demand * sigma))
        result.update({
            f"mean_reputation_{seller}": mean,
            f"variance_reputation_{seller}": float(marginal @ (grid - mean) ** 2),
            f"mean_posterior_{seller}": float(marginal @ posterior_mean(grid, params)),
            f"market_share_{seller}": share,
            f"product2_conditional_{seller}": product2 / share,
            f"product2_calendar_{seller}": product2,
            f"profit_{seller}": float(np.sum(nu * demand * (params.R - params.c1 - (params.c2 - params.c1) * sigma))),
        })
    covariance = float(np.sum(nu * (grid[:, None] - result["mean_reputation_A"])
                              * (grid[None, :] - result["mean_reputation_B"])))
    scale = np.sqrt(result["variance_reputation_A"] * result["variance_reputation_B"])
    result["correlation"] = covariance / float(scale) if scale > 0 else None
    result["product2_market"] = result["product2_calendar_A"] + result["product2_calendar_B"]
    result["industry_profit"] = result["profit_A"] + result["profit_B"]
    result["success_rate"] = params.p1 + (params.p2 - params.p1) * result["product2_market"]
    result["failure_rate"] = 1 - result["success_rate"]
    result["expected_success_probability"] = result["success_rate"]
    return result


def load_pair(project_root: Path, rho=.95, n=51, oe_run="rho_forgetting",
              p0_run="rho_forgetting_mixed", oe_candidate=None, p0_candidate=None):
    """Load compatible, verified saved candidates and audit the actual game.

    Candidate indices are zero-based; None means the explicitly saved selected
    candidate, not an automatic search for another verified candidate.
    """
    if not isinstance(n, (int, np.integer)) or isinstance(n, bool) or n < 3:
        raise ValueError("Grid size must be an integer of at least three.")
    project_root = Path(project_root).resolve()
    first = _source_case(project_root, "oe", oe_run, rho, n, oe_candidate)
    second = _source_case(project_root, "p0", p0_run, rho, n, p0_candidate)
    params, order, oe, oe_status = first
    p0_params, p0_order, p0, p0_status = second
    if params != p0_params:
        raise ValueError("The two saved equilibria use different primitives; comparison aborted.")
    if not np.array_equal(oe["grid"], p0["grid"]):
        raise ValueError("The two policies use different mature reputation grids.")
    if order != p0_order:
        raise ValueError("The two equilibria use different demand quadrature orders.")
    game = Game(oe["grid"], params, order=order)
    pair = dict(params=params, grid=oe["grid"].copy(), order=order, game=game,
                profiles={"oe": oe, "p0": p0})
    audit = dict(verified=True, parameters=asdict(params), n=int(n), order=order,
                 state_definition="Mature reputation x in [0,1]; Beta(1+x/(1-rho),1+(1-x)/(1-rho)); only the selected seller updates.",
                 demand=_demand_audit(game), profiles={})
    for key, profile, statuses in (("oe", oe, oe_status), ("p0", p0, p0_status)):
        policy, nu = profile["policy"], profile["nu"]
        sigma_a = np.broadcast_to(policy[:, None], nu.shape)
        sigma_b = np.broadcast_to(policy[None, :], nu.shape)
        transition = game.transition(sigma_a, sigma_b)
        profile["transition"] = transition
        residual = np.asarray(nu.ravel() @ transition).ravel() - nu.ravel()
        residual_max = float(np.abs(residual).max())
        row_error = float(np.max(np.abs(np.asarray(transition.sum(axis=1)).ravel() - 1)))
        if residual_max > STATIONARITY_TOLERANCE or row_error > STATIONARITY_TOLERANCE:
            raise ValueError(f"{profile['label']} saved nu is not stationary in the actual joint game (residual {residual_max:g}).")
        mu_a, mu_b = nu.sum(axis=1), nu.sum(axis=0)
        saved_marginal_errors = {}
        for name, expected in (("mu_a", mu_a), ("mu_b", mu_b)):
            if name in profile["selected"]:
                saved = _probability_law(profile["selected"][name], (n,), f"{profile['label']} {name}")
                saved_marginal_errors[name] = float(np.abs(saved - expected).sum())
        if max(saved_marginal_errors.values(), default=0) > CONSISTENCY_TOLERANCE:
            raise ValueError(f"{profile['label']} saved marginals disagree with its joint law.")
        if key == "oe":
            consistency = float(np.abs(profile["mu"] - mu_b).sum())
            saved_consistency = profile["selected"]["marginal_residual"]
            condition = "L1(mu_star - sum_A nu)"
        else:
            consistency = float(abs(profile["p0"] - posterior_mean(pair["grid"], params) @ mu_b))
            saved_consistency = abs(profile["selected"]["G"])
            condition = "abs(p0_star - E_nu[m(X_B)])"
        if consistency > CONSISTENCY_TOLERANCE:
            raise ValueError(f"{profile['label']} consistency failed: {consistency:g}.")
        profile["predicted"] = predictions(pair, key)
        audit["profiles"][key] = dict(
            label=profile["label"], statuses=statuses, source=profile["source"],
            stationary_residual_max=residual_max, stationary_residual_l1=float(np.abs(residual).sum()),
            transition_row_sum_error=row_error, stationary_tolerance=STATIONARITY_TOLERANCE,
            consistency_condition=condition, consistency_residual=consistency,
            saved_consistency_residual=float(saved_consistency), consistency_tolerance=CONSISTENCY_TOLERANCE,
            saved_marginal_l1_errors=saved_marginal_errors,
            marginal_symmetry_l1=float(np.abs(mu_a - mu_b).sum()),
            predicted=profile["predicted"],
        )
    runtime_root = Path(__file__).resolve().parents[3]
    source_paths = ("oblivious/02_oe_p0/code/primitives.py", "oblivious/02_oe_p0/code/game.py",
                    "oblivious/02_oe_p0/code/policy.py", "oblivious/02_oe_p0/code/core.py",
                    "oblivious/02_oe_p0/code/run_experiment.py", "oblivious/01_oe_distribution/code/model.py",
                    "oblivious/06_equilibrium_comparison/code/inputs.py")
    audit["current_numerical_source_sha256"] = {path: _hash(runtime_root / path) for path in source_paths}
    audit["provenance_note"] = "Current files are hashed separately from the historical solver hashes saved in each source manifest. Loaded policies and stationary laws are checked under the current actual-game kernel; no equilibrium solver was run."
    pair["audit"] = audit
    return pair
