"""Calendar-time simulations of saved policies in the actual two-seller game.

All modes draw both actual Beta posteriors and select their larger draw.  The
``grid`` mode then uses the solver's stochastic interpolation rule for the
selected seller's transition.  Its selection law agrees with the numerical
solver kernel up to the separately audited demand-quadrature error; it does
not replace Thompson sampling with either approximation's optimization demand.

``nearest`` and ``linear`` retain continuous reputations.  Nearest-node policy
extension (ties go to the lower node) preserves stored pure and mixed choices.
Linear extension is an explicitly additional policy approximation: it can mix
between otherwise pure grid choices.  Both extensions are exact at grid nodes.
"""

from collections.abc import Callable, Mapping
from dataclasses import asdict

import numpy as np
from scipy.special import betaincinv

from oblivious.oe_p0.code.primitives import (
    check_grid, interpolation_matrix, posterior, probability_array,
)


UNIFORM_CHANNELS = ("theta_A", "theta_B", "product", "success", "projection")


def _nearest_indices(grid, states):
    right = np.clip(np.searchsorted(grid, states, side="left"), 0, len(grid) - 1)
    left = np.maximum(right - 1, 0)
    # Explicitly choose the lower node on exact midpoint ties.
    return np.where(states - grid[left] <= grid[right] - states, left, right)


def policy_probability(grid, policy, states, mode="nearest"):
    """Evaluate saved own-state product-2 probabilities without thresholding.

    ``grid`` requires its inputs to be grid nodes.  At non-grid states, use
    ``nearest`` or explicitly request the artificial mixtures of ``linear``.
    The shape of ``states`` is preserved, including scalar inputs.
    """
    grid = check_grid(grid)
    policy = probability_array(policy)
    states = np.asarray(states, dtype=float)
    if policy.shape != grid.shape:
        raise ValueError("Own-state policy must have one probability per grid node.")
    if np.any(~np.isfinite(states)) or np.any((states < 0) | (states > 1)):
        raise ValueError("Reputations must be finite and lie in [0,1].")
    if mode == "linear":
        return probability_array(np.interp(states, grid, policy))
    if mode not in ("grid", "nearest"):
        raise ValueError("Policy-extension mode must be grid, nearest, or linear.")
    indices = _nearest_indices(grid, states)
    if mode == "grid" and np.any(np.abs(states - grid[indices]) > 8 * np.finfo(float).eps):
        raise ValueError("Grid policy evaluation requires states on the stored grid.")
    return policy[indices]


def _projection_rows(grid, targets):
    """Cache adjacent-node probabilities directly from the solver primitive."""
    matrix = interpolation_matrix(grid, targets)
    first, last = matrix.indptr[:-1], matrix.indptr[1:] - 1
    return matrix.indices[first], matrix.indices[last], matrix.data[last]


def simulate_pair(
    pair,
    *,
    periods=100_000,
    burn_in=10_000,
    replications=20,
    seed=20261017,
    mode="grid",
    short_periods=1000,
    initial=(.5, .5),
    total_periods=None,
    assignments=None,
    progress: Callable[[int, int], None] | None = None,
):
    """Simulate profiles with independent replications and paired random draws.

    ``periods`` is the number of observations *after* ``burn_in``.  The raw
    arrays also contain burn-in and the initial short trajectory, so discounted
    profits can be computed from the common initial state rather than from
    stationarity.  ``total_periods`` can extend the same paths for stability
    checks; it must contain the entire primary and short-trajectory windows.

    ``assignments`` optionally maps output names to two input profile keys,
    e.g. ``{'cross': ('oe', 'p0')}``.  The default is both symmetric profiles.
    Cross assignments are performance simulations, not equilibrium claims.

    Arrays under ``profiles`` have shape (replication, calendar time[, seller]).
    Sellers are A=0/B=1 and products are 1/2.  Every record is before/after the
    indicated opportunity, and only the selected seller receives profit.
    ``uniforms`` has five channels listed in ``metadata['uniform_channels']``.
    Each replication uses its own spawned SeedSequence; its time prefix does
    not depend on the horizon, modes, or the number of profiles simulated.
    """
    integers = {"periods": periods, "burn_in": burn_in,
                "replications": replications, "short_periods": short_periods}
    for name, value in integers.items():
        if isinstance(value, (bool, np.bool_)) or int(value) != value:
            raise ValueError(f"{name} must be an integer.")
    periods, burn_in = int(periods), int(burn_in)
    replications, short_periods = int(replications), int(short_periods)
    if periods < 1 or burn_in < 0 or replications < 1 or short_periods < 0:
        raise ValueError("Require positive periods/replications and nonnegative burn-in/short periods.")
    minimum_total = max(periods + burn_in, short_periods)
    if total_periods is None:
        total_periods = minimum_total
    if (isinstance(total_periods, (bool, np.bool_))
            or int(total_periods) != total_periods or total_periods < minimum_total):
        raise ValueError("total_periods must be an integer containing both requested windows.")
    total_periods = int(total_periods)
    if mode not in ("grid", "nearest", "linear"):
        raise ValueError("Simulation mode must be grid, nearest, or linear.")
    params, grid = pair["params"], check_grid(pair["grid"])
    initial = np.asarray(initial, dtype=float)
    if initial.shape != (2,):
        raise ValueError("The common initial state must contain A and B reputations.")
    posterior(initial, params)  # Reuse the common state-domain validation.
    initial_indices = _nearest_indices(grid, initial)
    if mode == "grid" and np.any(np.abs(initial - grid[initial_indices]) > 8 * np.finfo(float).eps):
        raise ValueError("Grid simulation requires the common initial state on the grid.")
    assignments = {"oe": ("oe", "oe"), "p0": ("p0", "p0")} if assignments is None else assignments
    if not isinstance(assignments, Mapping) or not assignments:
        raise ValueError("assignments must map at least one name to two policy keys.")
    names = list(assignments)
    policy_tables = []
    for name in names:
        if len(assignments[name]) != 2:
            raise ValueError("Each profile must assign exactly two seller policies.")
        policies = []
        for key in assignments[name]:
            source = pair["profiles"][key]
            if not np.array_equal(source.get("grid", grid), grid):
                raise ValueError("All profiles must use the shared grid.")
            policy = probability_array(source["policy"])
            if policy.shape != grid.shape:
                raise ValueError("Own-state policies must have one probability per node.")
            policies.append(policy)
        policy_tables.append(policies)
    policy_tables = np.asarray(policy_tables)
    profile_count = len(names)

    seed_sequence = np.random.SeedSequence(seed)
    streams = seed_sequence.spawn(replications)
    uniforms = np.empty((replications, total_periods, len(UNIFORM_CHANNELS)))
    for replication, stream in enumerate(streams):
        uniforms[replication] = np.random.default_rng(stream).random(
            (total_periods, len(UNIFORM_CHANNELS)))

    shape = (profile_count, replications, total_periods)
    arrays = {key: np.empty((*shape, 2)) for key in ("before", "after", "theta", "profit")}
    arrays.update({key: np.empty(shape, dtype=np.uint8) for key in ("selected", "product")})
    arrays["success"] = np.empty(shape, dtype=bool)
    arrays["product_probability"] = np.empty(shape)
    states = np.broadcast_to(initial, (profile_count, replications, 2)).copy()
    indices = np.broadcast_to(initial_indices, states.shape).copy()
    profile_index = np.arange(profile_count)[:, None]
    replication_index = np.arange(replications)[None, :]
    if mode == "grid":
        grid_alpha, grid_beta = posterior(grid, params)
        transition_rows = [
            _projection_rows(grid, params.rho * grid),
            _projection_rows(grid, params.rho * grid + 1 - params.rho),
        ]
        lower, upper, upper_probability = [np.stack(rows) for rows in zip(*transition_rows)]

    for t in range(total_periods):
        arrays["before"][:, :, t] = states
        draws = uniforms[:, t]
        if mode == "grid":
            alpha, beta = grid_alpha[indices], grid_beta[indices]
        else:
            alpha, beta = posterior(states, params)
        theta = betaincinv(alpha, beta, draws[None, :, :2])
        selected = (theta[:, :, 0] <= theta[:, :, 1]).astype(np.uint8)
        own_state = states[profile_index, replication_index, selected]
        if mode == "grid":
            own_index = indices[profile_index, replication_index, selected]
            probability = policy_tables[profile_index, selected, own_index]
        elif mode == "nearest":
            own_index = _nearest_indices(grid, own_state)
            probability = policy_tables[profile_index, selected, own_index]
        else:
            # Linear interpolation is confined to this named sensitivity mode.
            right = np.clip(np.searchsorted(grid, own_state, side="right"), 1, len(grid) - 1)
            left = right - 1
            weight = (own_state - grid[left]) / (grid[right] - grid[left])
            probability = (1 - weight) * policy_tables[profile_index, selected, left]
            probability += weight * policy_tables[profile_index, selected, right]
        product2 = draws[None, :, 2] < probability
        success_probability = np.where(product2, params.p2, params.p1)
        success = draws[None, :, 3] < success_probability
        if mode == "grid":
            outcome = success.astype(np.uint8)
            use_upper = draws[None, :, 4] < upper_probability[outcome, own_index]
            new_index = np.where(use_upper, upper[outcome, own_index], lower[outcome, own_index])
            indices[profile_index, replication_index, selected] = new_index
            states[profile_index, replication_index, selected] = grid[new_index]
        else:
            states[profile_index, replication_index, selected] = (
                params.rho * own_state + (1 - params.rho) * success)
        profit = np.zeros((profile_count, replications, 2))
        profit[profile_index, replication_index, selected] = (
            params.R - np.where(product2, params.c2, params.c1))
        arrays["theta"][:, :, t] = theta
        arrays["selected"][:, :, t] = selected
        arrays["product"][:, :, t] = product2.astype(np.uint8) + 1
        arrays["success"][:, :, t] = success
        arrays["profit"][:, :, t] = profit
        arrays["product_probability"][:, :, t] = probability
        arrays["after"][:, :, t] = states
        if progress is not None and ((t + 1) % 25_000 == 0 or t + 1 == total_periods):
            progress(t + 1, total_periods)

    return {
        "profiles": {name: {key: value[i] for key, value in arrays.items()}
                     for i, name in enumerate(names)},
        "uniforms": uniforms,
        "time": np.arange(total_periods, dtype=np.int64),
        "replication_seeds": [{"entropy": stream.entropy, "spawn_key": list(stream.spawn_key)}
                              for stream in streams],
        "metadata": {
            "params": asdict(params), "mode": mode, "seed": seed,
            "periods": periods, "burn_in": burn_in, "replications": replications,
            "total_periods": total_periods, "short_periods": short_periods,
            "initial": initial.tolist(), "primary_window": [burn_in, burn_in + periods],
            "assignments": {key: list(value) for key, value in assignments.items()},
            "uniform_channels": list(UNIFORM_CHANNELS),
            "random_generator": "numpy.random.PCG64; independent SeedSequence child per replication",
            "common_random_numbers": "Same five uniforms per replication/time in every profile and mode",
            "selection": "Independent actual Beta posterior draws; A iff theta_A > theta_B, ties B",
            "transition": ("Solver stochastic interpolation; selection matches finite kernel up to audited demand quadrature error"
                           if mode == "grid" else "Exact continuous reputation update for the selected seller only"),
            "policy_extension": {
                "grid": "Stored probabilities at grid nodes; mixed choices sampled without thresholding",
                "nearest": "Nearest grid node, midpoint ties to lower node; preserves stored mixtures",
                "linear": "Linear interpolation; additional artificial mixing between pure grid choices",
            }[mode],
        },
    }
