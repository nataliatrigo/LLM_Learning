"""
Average-per-stage study for the Bernoulli/Beta dynamic reputation model.

This script is separate from the discounted-cost project in the repository
root. It writes data and plots under average_cost/outputs by default.

The solver supports finite average-reward approximations and exact finite-horizon
dynamic programming. The default average-reward approximation is the original
rolling-window method. A projection-to-boundary approximation and finite-horizon
robustness experiments are available from the same command-line entry point.
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.special import betaincc

os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "llm_learning_matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm
from matplotlib.ticker import PercentFormatter


AVERAGE_REWARD_METHODS = ("rolling_window", "projection", "stochastic_projection")
FINITE_HORIZON_METHOD = "finite_horizon"
DEFAULT_ROBUSTNESS_GRID = "25,50,100,200"
DEFAULT_ROBUSTNESS_P0_GRID = "0.1,0.3,0.5,0.7,0.9"
POSTERIOR_BIN_COUNT = 20
STATIONARY_MAPPING_RULES = ("closest_posterior_mean", "cap_total_count_preserve_share")


PATH_DIAGNOSTIC_COLUMNS = [
    "p0",
    "rep",
    "t",
    "S",
    "F",
    "observations",
    "projected",
    "projected_S",
    "projected_F",
    "posterior_mean",
    "demand_probability",
    "relative_bias",
    "bias_gap_success_minus_failure",
    "product2_bias_gap_threshold",
    "bias_gap_minus_threshold",
    "q_gap_product2_minus_product1",
    "best_response_product",
    "chosen_A",
    "product_used",
    "success",
    "profit",
]


@dataclass(frozen=True)
class ModelParams:
    """Numerical calibration and approximation controls."""

    p1: float = 0.35
    p2: float = 0.80
    c1: float = 0.05
    c2: float = 0.65
    revenue: float = 1.0
    max_observations: int = 25
    max_iter: int = 10_000
    tol: float = 1e-8
    demand_floor: float = 1e-300


@dataclass(frozen=True)
class StateSpace:
    """Triangular grid of feasible sufficient statistics S + F <= N."""

    method: str
    max_observations: int
    S: np.ndarray
    F: np.ndarray
    total: np.ndarray
    state_index: np.ndarray
    success_index: np.ndarray
    success_probability: np.ndarray
    success_stay_probability: np.ndarray
    failure_index: np.ndarray
    failure_probability: np.ndarray
    failure_stay_probability: np.ndarray


def make_state_space(
    max_observations: int,
    method: str = "rolling_window",
) -> StateSpace:
    if method not in AVERAGE_REWARD_METHODS:
        raise ValueError(f"Unknown average-reward approximation method: {method}")

    states: list[tuple[int, int]] = []
    state_index = -np.ones((max_observations + 1, max_observations + 1), dtype=int)

    for total in range(max_observations + 1):
        for successes in range(total + 1):
            failures = total - successes
            state_index[successes, failures] = len(states)
            states.append((successes, failures))

    S = np.array([state[0] for state in states], dtype=int)
    F = np.array([state[1] for state in states], dtype=int)
    total = S + F
    success_index = np.empty(len(states), dtype=int)
    success_probability = np.empty(len(states), dtype=float)
    success_stay_probability = np.empty(len(states), dtype=float)
    failure_index = np.empty(len(states), dtype=int)
    failure_probability = np.empty(len(states), dtype=float)
    failure_stay_probability = np.empty(len(states), dtype=float)

    for idx, (successes, failures) in enumerate(states):
        if successes + failures < max_observations:
            success_index[idx] = state_index[successes + 1, failures]
            success_probability[idx] = 1.0
            success_stay_probability[idx] = 0.0
            failure_index[idx] = state_index[successes, failures + 1]
            failure_probability[idx] = 1.0
            failure_stay_probability[idx] = 0.0
            continue

        if method == "rolling_window":
            success_successes = min(successes + 1, max_observations)
            success_failures = max_observations - success_successes
            failure_successes = max(successes - 1, 0)
            failure_failures = max_observations - failure_successes
            success_index[idx] = state_index[success_successes, success_failures]
            success_probability[idx] = failures / max_observations
            success_stay_probability[idx] = successes / max_observations
            failure_index[idx] = state_index[failure_successes, failure_failures]
            failure_probability[idx] = successes / max_observations
            failure_stay_probability[idx] = failures / max_observations
            continue

        if method == "projection":
            success_share = (successes + 1.0) / (max_observations + 1.0)
            success_successes = int(round(max_observations * success_share))
            success_successes = int(np.clip(success_successes, 0, max_observations))
            success_failures = max_observations - success_successes
            failure_share = successes / (max_observations + 1.0)
            failure_successes = int(round(max_observations * failure_share))
            failure_successes = int(np.clip(failure_successes, 0, max_observations))
            failure_failures = max_observations - failure_successes
            success_index[idx] = state_index[success_successes, success_failures]
            success_probability[idx] = 1.0
            success_stay_probability[idx] = 0.0
            failure_index[idx] = state_index[failure_successes, failure_failures]
            failure_probability[idx] = 1.0
            failure_stay_probability[idx] = 0.0
            continue

        success_move_probability = failures / (max_observations + 1.0)
        failure_move_probability = successes / (max_observations + 1.0)
        success_successes = min(successes + 1, max_observations)
        success_failures = max_observations - success_successes
        failure_successes = max(successes - 1, 0)
        failure_failures = max_observations - failure_successes
        success_index[idx] = state_index[success_successes, success_failures]
        success_probability[idx] = success_move_probability
        success_stay_probability[idx] = 1.0 - success_move_probability
        failure_index[idx] = state_index[failure_successes, failure_failures]
        failure_probability[idx] = failure_move_probability
        failure_stay_probability[idx] = 1.0 - failure_move_probability

    return StateSpace(
        method=method,
        max_observations=max_observations,
        S=S,
        F=F,
        total=total,
        state_index=state_index,
        success_index=success_index,
        success_probability=success_probability,
        success_stay_probability=success_stay_probability,
        failure_index=failure_index,
        failure_probability=failure_probability,
        failure_stay_probability=failure_stay_probability,
    )


def beta_ccdf(p0: float, S: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Pr(Beta(1 + S, 1 + F) >= p0)."""
    return np.clip(betaincc(S + 1.0, F + 1.0, p0), 0.0, 1.0)


def beta_ccdf_scalar(p0: float, successes: int, failures: int) -> float:
    return float(beta_ccdf(p0, np.array([successes]), np.array([failures]))[0])


def solver_demand_probability(
    p0: float,
    S: np.ndarray,
    F: np.ndarray,
    params: ModelParams,
) -> np.ndarray:
    """Demand probabilities used inside the finite average-reward solver."""
    return np.clip(beta_ccdf(p0, S, F), params.demand_floor, 1.0)


def posterior_mean(S: np.ndarray, F: np.ndarray) -> np.ndarray:
    return (S + 1.0) / (S + F + 2.0)


def posterior_std(S: np.ndarray, F: np.ndarray) -> np.ndarray:
    alpha = S + 1.0
    beta = F + 1.0
    posterior_precision = alpha + beta
    variance = alpha * beta / (
        posterior_precision**2 * (posterior_precision + 1.0)
    )
    return np.sqrt(variance)


def posterior_mean_scalar(successes: int, failures: int) -> float:
    return (successes + 1.0) / (successes + failures + 2.0)


def closest_counts_for_posterior_mean(
    target_mean: float,
    max_observations: int,
) -> tuple[int, int]:
    if not 0.0 < target_mean < 1.0:
        raise ValueError("initial-belief-mean must be strictly between 0 and 1.")

    best: tuple[float, int, int, int] | None = None
    for total in range(max_observations + 1):
        for successes in range(total + 1):
            failures = total - successes
            mean = posterior_mean_scalar(successes, failures)
            candidate = (abs(mean - target_mean), total, successes, failures)
            if best is None or candidate < best:
                best = candidate

    if best is None:
        raise ValueError("Could not construct an initial belief state.")
    _, _, successes, failures = best
    return successes, failures


def product2_bias_gap_threshold(params: ModelParams) -> float:
    """Product 2 is optimal when expected success-vs-failure bias exceeds this."""
    return (params.c2 - params.c1) / (params.p2 - params.p1)


def aggregate_transition_probabilities(
    transitions: list[tuple[int, float]],
) -> list[tuple[int, float]]:
    """Combine transition mass when multiple branches lead to the same state."""
    mass_by_state: defaultdict[int, float] = defaultdict(float)
    for state_idx, probability in transitions:
        if probability:
            mass_by_state[int(state_idx)] += float(probability)
    return sorted(mass_by_state.items())


def state_action_transition_probabilities(
    state_idx: int,
    product: int,
    rho: float,
    params: ModelParams,
    state_space: StateSpace,
) -> tuple[list[tuple[int, float]], float]:
    """Return full one-period transition probabilities and expected reward.

    The transition includes the event that the user chooses Seller B, in which
    case calendar time advances and the reputation state is unchanged.
    """
    if product not in {1, 2}:
        raise ValueError("product must be 1 or 2.")
    success_probability = params.p2 if product == 2 else params.p1
    cost = params.c2 if product == 2 else params.c1
    state_idx = int(state_idx)
    rho = float(rho)

    transitions = [
        (state_idx, 1.0 - rho),
        (
            int(state_space.success_index[state_idx]),
            rho * success_probability * state_space.success_probability[state_idx],
        ),
        (
            state_idx,
            rho * success_probability * state_space.success_stay_probability[state_idx],
        ),
        (
            int(state_space.failure_index[state_idx]),
            rho * (1.0 - success_probability) * state_space.failure_probability[state_idx],
        ),
        (
            state_idx,
            rho * (1.0 - success_probability) * state_space.failure_stay_probability[state_idx],
        ),
    ]
    expected_reward = rho * (params.revenue - cost)
    return aggregate_transition_probabilities(transitions), expected_reward


def state_action_transitions(
    state_idx: int,
    product: int,
    rho: float,
    params: ModelParams,
    state_space: StateSpace,
) -> list[tuple[tuple[int, int], float, float]]:
    """Human-readable transition interface used by diagnostics and inspection."""
    transitions, expected_reward = state_action_transition_probabilities(
        state_idx=state_idx,
        product=product,
        rho=rho,
        params=params,
        state_space=state_space,
    )
    return [
        (
            (int(state_space.S[next_idx]), int(state_space.F[next_idx])),
            probability,
            expected_reward,
        )
        for next_idx, probability in transitions
    ]


def projected_state_index(
    successes: int,
    failures: int,
    state_space: StateSpace,
) -> int:
    """Map actual counts to the finite grid used by the solver."""
    max_observations = state_space.max_observations
    total = successes + failures
    if total <= max_observations:
        return int(state_space.state_index[successes, failures])

    projected_successes = int(round(max_observations * successes / total))
    projected_successes = int(np.clip(projected_successes, 0, max_observations))
    projected_failures = max_observations - projected_successes
    return int(state_space.state_index[projected_successes, projected_failures])


def build_posterior_mean_mapper(
    state_space: StateSpace,
) -> tuple[np.ndarray, np.ndarray]:
    means = posterior_mean(state_space.S, state_space.F)
    order = np.lexsort((-state_space.total, means))
    return means[order], order


def finite_state_index_for_true_counts(
    successes: int,
    failures: int,
    state_space: StateSpace,
    mapping_rule: str,
    sorted_means: np.ndarray | None = None,
    sorted_indices: np.ndarray | None = None,
) -> int:
    """Map true Bayesian counts to a finite stationary policy state."""
    if mapping_rule not in STATIONARY_MAPPING_RULES:
        raise ValueError(f"Unknown state mapping rule: {mapping_rule}")
    max_observations = state_space.max_observations
    total = successes + failures
    if total <= max_observations:
        return int(state_space.state_index[successes, failures])
    if mapping_rule == "cap_total_count_preserve_share":
        return projected_state_index(successes, failures, state_space)

    if sorted_means is None or sorted_indices is None:
        sorted_means, sorted_indices = build_posterior_mean_mapper(state_space)
    target_mean = posterior_mean_scalar(successes, failures)
    insert_at = int(np.searchsorted(sorted_means, target_mean, side="left"))
    candidates = []
    if insert_at < len(sorted_means):
        candidates.append(int(sorted_indices[insert_at]))
    if insert_at > 0:
        candidates.append(int(sorted_indices[insert_at - 1]))
    if not candidates:
        return int(state_space.state_index[0, 0])
    return min(
        candidates,
        key=lambda idx: (
            abs(float(posterior_mean_scalar(int(state_space.S[idx]), int(state_space.F[idx]))) - target_mean),
            -int(state_space.total[idx]),
        ),
    )


def validate_average_reward_transitions(
    p0: float,
    params: ModelParams,
    state_space: StateSpace,
) -> pd.DataFrame:
    """Check probability mass and finite-state support for a method/p0 pair."""
    reported_rho = beta_ccdf(p0, state_space.S, state_space.F)
    solver_rho = solver_demand_probability(p0, state_space.S, state_space.F, params)
    max_abs_probability_error = 0.0
    out_of_space_count = 0
    negative_probability_count = 0

    for state_idx in range(len(state_space.S)):
        for product in (1, 2):
            transitions, _ = state_action_transition_probabilities(
                state_idx=state_idx,
                product=product,
                rho=float(solver_rho[state_idx]),
                params=params,
                state_space=state_space,
            )
            probability_sum = sum(probability for _, probability in transitions)
            max_abs_probability_error = max(
                max_abs_probability_error,
                abs(probability_sum - 1.0),
            )
            for next_idx, probability in transitions:
                if next_idx < 0 or next_idx >= len(state_space.S):
                    out_of_space_count += 1
                if probability < -1e-12:
                    negative_probability_count += 1

    return pd.DataFrame(
        [
            {
                "method": state_space.method,
                "p0": p0,
                "N": state_space.max_observations,
                "max_abs_probability_error": max_abs_probability_error,
                "out_of_space_transition_count": out_of_space_count,
                "negative_probability_count": negative_probability_count,
                "reported_min_rho": float(np.min(reported_rho)),
                "solver_min_rho": float(np.min(solver_rho)),
                "demand_floor": params.demand_floor,
                "rho_floor_binding_state_count": int(np.sum(solver_rho > reported_rho)),
                "uses_discount_factor": False,
            }
        ]
    )


def expected_outcome_bias(
    bias: np.ndarray,
    state_space: StateSpace,
) -> tuple[np.ndarray, np.ndarray]:
    """Expected next bias after a success or failure outcome."""
    current_bias = bias
    success_bias = (
        state_space.success_probability * bias[state_space.success_index]
        + state_space.success_stay_probability * current_bias
    )
    failure_bias = (
        state_space.failure_probability * bias[state_space.failure_index]
        + state_space.failure_stay_probability * current_bias
    )
    return success_bias, failure_bias


def compute_action_values(
    bias: np.ndarray,
    p0: float,
    params: ModelParams,
    state_space: StateSpace,
) -> tuple[np.ndarray, np.ndarray]:
    bias_success, bias_failure = expected_outcome_bias(bias, state_space)
    rho = solver_demand_probability(p0, state_space.S, state_space.F, params)
    current_bias = bias

    q1 = (
        rho * (params.revenue - params.c1)
        + (1.0 - rho) * current_bias
        + rho * (params.p1 * bias_success + (1.0 - params.p1) * bias_failure)
    )
    q2 = (
        rho * (params.revenue - params.c2)
        + (1.0 - rho) * current_bias
        + rho * (params.p2 * bias_success + (1.0 - params.p2) * bias_failure)
    )

    return q1, q2


def build_policy_transition_reward(
    policy_product: np.ndarray,
    p0: float,
    params: ModelParams,
    state_space: StateSpace,
) -> tuple[coo_matrix, np.ndarray, np.ndarray]:
    n_states = len(state_space.S)
    rho = solver_demand_probability(p0, state_space.S, state_space.F, params)
    success_probability = np.where(policy_product == 2, params.p2, params.p1)
    cost = np.where(policy_product == 2, params.c2, params.c1)
    expected_reward = rho * (params.revenue - cost)

    state_ids = np.arange(n_states)
    rows = np.concatenate([state_ids, state_ids, state_ids, state_ids, state_ids])
    cols = np.concatenate(
        [
            state_ids,
            state_space.success_index,
            state_ids,
            state_space.failure_index,
            state_ids,
        ]
    )
    probs = np.concatenate(
        [
            1.0 - rho,
            rho * success_probability * state_space.success_probability,
            rho * success_probability * state_space.success_stay_probability,
            rho * (1.0 - success_probability) * state_space.failure_probability,
            rho * (1.0 - success_probability) * state_space.failure_stay_probability,
        ]
    )
    transition = coo_matrix((probs, (rows, cols)), shape=(n_states, n_states)).tocsr()
    return transition, expected_reward, rho


def build_conditional_policy_transition_reward(
    policy_product: np.ndarray,
    p0: float,
    params: ModelParams,
    state_space: StateSpace,
) -> tuple[coo_matrix, np.ndarray, np.ndarray]:
    """Build transitions conditional on Seller A being chosen.

    The full average-reward transition includes a large self-loop when the user
    chooses Seller B. Policy evaluation collects that self-loop algebraically,
    which is equivalent and substantially better conditioned when rho is tiny.
    """
    n_states = len(state_space.S)
    rho = solver_demand_probability(p0, state_space.S, state_space.F, params)
    success_probability = np.where(policy_product == 2, params.p2, params.p1)
    cost = np.where(policy_product == 2, params.c2, params.c1)
    reward_when_chosen = params.revenue - cost

    state_ids = np.arange(n_states)
    rows = np.concatenate([state_ids, state_ids, state_ids, state_ids])
    cols = np.concatenate(
        [
            state_space.success_index,
            state_ids,
            state_space.failure_index,
            state_ids,
        ]
    )
    probs = np.concatenate(
        [
            success_probability * state_space.success_probability,
            success_probability * state_space.success_stay_probability,
            (1.0 - success_probability) * state_space.failure_probability,
            (1.0 - success_probability) * state_space.failure_stay_probability,
        ]
    )
    transition = coo_matrix((probs, (rows, cols)), shape=(n_states, n_states)).tocsr()
    return transition, reward_when_chosen, rho


def evaluate_stationary_policy(
    policy_product: np.ndarray,
    p0: float,
    params: ModelParams,
    state_space: StateSpace,
) -> tuple[float, np.ndarray]:
    """Evaluate a stationary policy after collecting the Seller-B self-loop."""
    n_states = len(state_space.S)
    reference_idx = int(state_space.state_index[0, 0])
    transition, reward_when_chosen, rho = build_conditional_policy_transition_reward(
        policy_product,
        p0,
        params,
        state_space,
    )
    transition = transition.tocoo()

    rows = []
    cols = []
    data = []
    rhs = np.zeros(n_states + 1, dtype=float)
    rhs[:n_states] = rho * reward_when_chosen

    state_ids = np.arange(n_states)
    rows.extend(state_ids.tolist())
    cols.extend(state_ids.tolist())
    data.extend(rho.tolist())

    rows.extend(transition.row.tolist())
    cols.extend(transition.col.tolist())
    data.extend((-(rho[transition.row] * transition.data)).tolist())

    rows.extend(state_ids.tolist())
    cols.extend(np.full(n_states, n_states, dtype=int).tolist())
    data.extend(np.ones(n_states).tolist())

    rows.append(n_states)
    cols.append(reference_idx)
    data.append(1.0)

    system = coo_matrix(
        (data, (rows, cols)),
        shape=(n_states + 1, n_states + 1),
    ).tocsr()
    solution = spsolve(system, rhs)
    if not np.all(np.isfinite(solution)):
        raise RuntimeError("Average-reward policy evaluation did not return finite values.")
    bias = np.asarray(solution[:n_states], dtype=float)
    average_profit = float(solution[n_states])
    return average_profit, bias


def solve_average_reward_policy(
    p0: float,
    params: ModelParams,
    state_space: StateSpace,
    fixed_product: int | None = None,
) -> dict:
    """Solve the finite-state average-reward problem by policy iteration."""
    if fixed_product not in {None, 1, 2}:
        raise ValueError("fixed_product must be None, 1, or 2.")

    rho = solver_demand_probability(p0, state_space.S, state_space.F, params)
    reported_rho = beta_ccdf(p0, state_space.S, state_space.F)
    policy_product = np.ones(len(state_space.S), dtype=int)
    if fixed_product is not None:
        policy_product = np.full(len(state_space.S), fixed_product, dtype=int)
    convergence_records = []

    for iteration in range(1, params.max_iter + 1):
        average_profit, bias = evaluate_stationary_policy(
            policy_product,
            p0,
            params,
            state_space,
        )
        q1, q2 = compute_action_values(bias, p0, params, state_space)
        best_q = np.maximum(q1, q2)
        bellman_error = best_q - bias
        span_residual = float(np.max(bellman_error) - np.min(bellman_error))

        if fixed_product is None:
            improved_policy = policy_product.copy()
            improved_policy[q2 > q1 + params.tol] = 2
            improved_policy[q1 > q2 + params.tol] = 1
        else:
            improved_policy = policy_product

        policy_changes = int(np.sum(improved_policy != policy_product))
        convergence_records.append(
            {
                    "p0": p0,
                    "method": state_space.method,
                    "N": state_space.max_observations,
                    "fixed_product": fixed_product
                    if fixed_product is not None
                    else "optimal",
                "iteration": iteration,
                "average_profit_gain": average_profit,
                "span_residual": span_residual,
                "policy_changes": policy_changes,
            }
        )

        policy_product = improved_policy
        if fixed_product is not None or policy_changes == 0:
            break

    average_profit, bias = evaluate_stationary_policy(
        policy_product,
        p0,
        params,
        state_space,
    )
    q1, q2 = compute_action_values(bias, p0, params, state_space)
    best_q = np.maximum(q1, q2)
    bellman_error = best_q - bias
    span_residual = float(np.max(bellman_error) - np.min(bellman_error))
    bias_success, bias_failure = expected_outcome_bias(bias, state_space)
    bias_gap = bias_success - bias_failure

    if fixed_product is None:
        policy_product = np.where(q2 > q1, 2, 1)
    else:
        policy_product = np.full(len(state_space.S), fixed_product, dtype=int)

    return {
        "p0": p0,
        "method": state_space.method,
        "N": state_space.max_observations,
        "value": bias,
        "bias": bias,
        "average_profit_gain": average_profit,
        "rho": reported_rho,
        "solver_rho": rho,
        "q1": q1,
        "q2": q2,
        "q_gap": q2 - q1,
        "bias_gap": bias_gap,
        "policy_product": policy_product,
        "fixed_product": fixed_product,
        "iterations": iteration,
        "span_residual": span_residual,
        "convergence": pd.DataFrame.from_records(convergence_records),
    }


def simulate_solution(
    solution: dict,
    params: ModelParams,
    state_space: StateSpace,
    n_rep: int,
    horizon: int,
    seed: int,
    initial_successes: int = 0,
    initial_failures: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate Thompson-sampling demand under Seller A's stationary policy."""
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if n_rep <= 0:
        raise ValueError("n-rep must be positive unless simulation is skipped.")
    if initial_successes < 0 or initial_failures < 0:
        raise ValueError("Initial successes and failures must be nonnegative.")

    rng = np.random.default_rng(seed)
    p0 = float(solution["p0"])
    policy_product = solution["policy_product"]
    bias = solution["bias"]
    bias_gap = solution["bias_gap"]
    q_gap = solution["q_gap"]
    threshold = product2_bias_gap_threshold(params)
    max_observations = state_space.state_index.shape[0] - 1
    tail_start = int(np.floor(0.5 * horizon))
    tail_denominator = max(1, horizon - tail_start)
    rep_records = []
    path_records = []

    chosen_by_t = np.zeros(horizon, dtype=float)
    product2_by_t = np.zeros(horizon, dtype=float)
    product2_den_by_t = np.zeros(horizon, dtype=float)
    profit_by_t = np.zeros(horizon, dtype=float)
    posterior_mean_by_t = np.zeros(horizon, dtype=float)
    posterior_std_by_t = np.zeros(horizon, dtype=float)
    demand_prob_by_t = np.zeros(horizon, dtype=float)
    observations_by_t = np.zeros(horizon, dtype=float)
    successes_by_t = np.zeros(horizon, dtype=float)
    failures_by_t = np.zeros(horizon, dtype=float)

    for rep in range(n_rep):
        successes = initial_successes
        failures = initial_failures
        chosen_count = 0
        product2_count = 0
        success_count = 0
        profit_sum = 0.0
        tail_profit_sum = 0.0

        for t in range(horizon):
            current_successes = successes
            current_failures = failures
            demand_prob = beta_ccdf_scalar(p0, current_successes, current_failures)
            observations = current_successes + current_failures
            projected = observations > max_observations
            state_idx = projected_state_index(
                current_successes,
                current_failures,
                state_space,
            )
            projected_successes = int(state_space.S[state_idx])
            projected_failures = int(state_space.F[state_idx])
            alpha = current_successes + 1.0
            beta = current_failures + 1.0
            posterior_precision = alpha + beta
            posterior_mean_t = alpha / posterior_precision
            posterior_std_t = np.sqrt(
                alpha
                * beta
                / (posterior_precision**2 * (posterior_precision + 1.0))
            )
            current_bias = float(bias[state_idx])
            current_bias_gap = float(bias_gap[state_idx])
            gap_minus_threshold = current_bias_gap - threshold
            current_q_gap = float(q_gap[state_idx])
            best_response_product = int(policy_product[state_idx])

            posterior_mean_by_t[t] += posterior_mean_t
            posterior_std_by_t[t] += posterior_std_t
            demand_prob_by_t[t] += demand_prob
            observations_by_t[t] += observations
            successes_by_t[t] += current_successes
            failures_by_t[t] += current_failures
            chosen_A = rng.random() < demand_prob
            product = np.nan
            success = np.nan
            profit = 0.0

            if chosen_A:
                product = best_response_product
                success_probability = params.p2 if product == 2 else params.p1
                success = rng.random() < success_probability

                if success:
                    successes += 1
                    success_count += 1
                else:
                    failures += 1

                chosen_count += 1
                product2_count += int(product == 2)
                profit = params.revenue - (params.c2 if product == 2 else params.c1)
                product2_by_t[t] += int(product == 2)
                product2_den_by_t[t] += 1.0

            path_records.append(
                (
                    p0,
                    rep,
                    t + 1,
                    current_successes,
                    current_failures,
                    observations,
                    int(projected),
                    projected_successes,
                    projected_failures,
                    posterior_mean_t,
                    demand_prob,
                    current_bias,
                    current_bias_gap,
                    threshold,
                    gap_minus_threshold,
                    current_q_gap,
                    best_response_product,
                    int(chosen_A),
                    product,
                    float(success) if chosen_A else np.nan,
                    profit,
                )
            )
            chosen_by_t[t] += float(chosen_A)
            profit_by_t[t] += profit
            profit_sum += profit
            if t >= tail_start:
                tail_profit_sum += profit

        total_A_observations = successes + failures
        rep_records.append(
            {
                "p0": p0,
                "rep": rep,
                "A_market_share": chosen_count / horizon,
                "product2_rate_when_A_chosen": (
                    product2_count / chosen_count if chosen_count else np.nan
                ),
                "A_success_rate_when_chosen": (
                    success_count / chosen_count if chosen_count else np.nan
                ),
                "avg_profit_per_period": profit_sum / horizon,
                "tail_avg_profit_per_period": tail_profit_sum / tail_denominator,
                "final_successes": successes,
                "final_failures": failures,
                "final_posterior_mean": (successes + 1.0)
                / (total_A_observations + 2.0),
            }
        )

    time_records = []
    for t in range(horizon):
        time_records.append(
            {
                "p0": p0,
                "t": t + 1,
                "A_market_share": chosen_by_t[t] / n_rep,
                "product2_rate_when_A_chosen": (
                    product2_by_t[t] / product2_den_by_t[t]
                    if product2_den_by_t[t] > 0
                    else np.nan
                ),
                "avg_profit_per_period": profit_by_t[t] / n_rep,
                "user_posterior_mean_A": posterior_mean_by_t[t] / n_rep,
                "user_posterior_std_A": posterior_std_by_t[t] / n_rep,
                "user_demand_probability_A": demand_prob_by_t[t] / n_rep,
                "user_A_observations": observations_by_t[t] / n_rep,
                "user_A_successes": successes_by_t[t] / n_rep,
                "user_A_failures": failures_by_t[t] / n_rep,
            }
        )

    return (
        pd.DataFrame.from_records(rep_records),
        pd.DataFrame.from_records(time_records),
        pd.DataFrame.from_records(path_records, columns=PATH_DIAGNOSTIC_COLUMNS),
    )


def summarize_solution(
    solution: dict,
    fixed_product1: dict,
    fixed_product2: dict,
    params: ModelParams,
    state_space: StateSpace,
    simulation_reps: pd.DataFrame,
    initial_successes: int = 0,
    initial_failures: int = 0,
) -> dict:
    product2 = solution["policy_product"] == 2
    rho = solution["rho"]
    demand_weight_sum = float(np.sum(rho))
    initial_idx = projected_state_index(
        initial_successes,
        initial_failures,
        state_space,
    )
    product2_reps = simulation_reps["product2_rate_when_A_chosen"].dropna()

    return {
        "method": solution["method"],
        "p0": solution["p0"],
        "N": state_space.max_observations,
        "average_profit_gain": solution["average_profit_gain"],
        "eta": solution["average_profit_gain"],
        "fixed_product1_average_profit": fixed_product1["average_profit_gain"],
        "fixed_product2_average_profit": fixed_product2["average_profit_gain"],
        "initial_demand_probability": rho[initial_idx],
        "initial_best_response_product": int(solution["policy_product"][initial_idx]),
        "initial_uses_product2": float(solution["policy_product"][initial_idx] == 2),
        "initial_q_gap_product2_minus_product1": solution["q_gap"][initial_idx],
        "initial_bias": solution["bias"][initial_idx],
        "share_states_product2": float(np.mean(product2)),
        "share_reachable_states_product2": float(np.mean(product2)),
        "demand_weighted_share_states_product2": (
            float(np.sum(product2 * rho) / demand_weight_sum)
            if demand_weight_sum > 0
            else np.nan
        ),
        "mean_A_market_share_sim": simulation_reps["A_market_share"].mean(),
        "mean_product2_rate_when_A_chosen_sim": product2_reps.mean(),
        "mean_A_success_rate_when_chosen_sim": simulation_reps[
            "A_success_rate_when_chosen"
        ].mean(),
        "mean_profit_per_period_sim": simulation_reps["avg_profit_per_period"].mean(),
        "mean_tail_profit_per_period_sim": simulation_reps[
            "tail_avg_profit_per_period"
        ].mean(),
        "mean_final_posterior_mean_sim": simulation_reps["final_posterior_mean"].mean(),
        "policy_iteration_iterations": solution["iterations"],
        "num_policy_iterations": solution["iterations"],
        "policy_iteration_span_residual": solution["span_residual"],
        "residual_span": solution["span_residual"],
        "p1": params.p1,
        "p2": params.p2,
        "c1": params.c1,
        "c2": params.c2,
        "revenue": params.revenue,
        "max_observations": params.max_observations,
    }


def build_policy_state_table(
    solutions: list[dict],
    state_space: StateSpace,
    params: ModelParams,
) -> pd.DataFrame:
    frames = []
    threshold = product2_bias_gap_threshold(params)
    for solution in solutions:
        frames.append(
            pd.DataFrame(
                {
                    "method": solution["method"],
                    "p0": solution["p0"],
                    "N": state_space.max_observations,
                    "S": state_space.S,
                    "F": state_space.F,
                    "observations": state_space.total,
                    "total_count": state_space.total,
                    "posterior_mean_A": posterior_mean(state_space.S, state_space.F),
                    "posterior_mean": posterior_mean(state_space.S, state_space.F),
                    "posterior_std_A": posterior_std(state_space.S, state_space.F),
                    "demand_probability": solution["rho"],
                    "rho": solution["rho"],
                    "best_response_product": solution["policy_product"],
                    "action": solution["policy_product"],
                    "uses_product2": (solution["policy_product"] == 2).astype(int),
                    "use_product_2": (solution["policy_product"] == 2).astype(int),
                    "q_gap_product2_minus_product1": solution["q_gap"],
                    "bias_gap_success_minus_failure": solution["bias_gap"],
                    "product2_bias_gap_threshold": threshold,
                    "threshold_rhs": threshold,
                    "bias_gap_minus_product2_threshold": (
                        solution["bias_gap"] - threshold
                    ),
                    "relative_bias": solution["bias"],
                    "average_profit_gain": solution["average_profit_gain"],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#111827",
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.color": "#4b5563",
            "ytick.color": "#4b5563",
            "grid.color": "#e5e7eb",
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "font.size": 10,
            "savefig.facecolor": "white",
        }
    )


def prettify_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cbd5e1")
    ax.spines["bottom"].set_color("#cbd5e1")
    ax.grid(True, axis=grid_axis)


def save_figure(fig: plt.Figure, path: Path) -> None:
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def p0_filename_fragment(p0: float) -> str:
    return f"{p0:.2f}".replace(".", "_")


def plot_p0_summary(summary: pd.DataFrame, outputs_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    ax_gain, ax_q, ax_p2, ax_profit = axes.ravel()

    ax_gain.plot(
        summary["p0"],
        summary["average_profit_gain"],
        color="#0f766e",
        marker="o",
        linewidth=2.0,
        label="Optimal",
    )
    ax_gain.plot(
        summary["p0"],
        summary["fixed_product1_average_profit"],
        color="#64748b",
        marker="o",
        linewidth=1.7,
        label="Always product 1",
    )
    ax_gain.plot(
        summary["p0"],
        summary["fixed_product2_average_profit"],
        color="#dc2626",
        marker="o",
        linewidth=1.7,
        label="Always product 2",
    )
    ax_gain.set_title("Long-run average profit", loc="left")
    ax_gain.set_ylabel("Average profit per period")
    prettify_axes(ax_gain)
    ax_gain.legend()

    ax_q.axhline(0.0, color="#6b7280", linewidth=1.1, linestyle=":")
    ax_q.plot(
        summary["p0"],
        summary["initial_q_gap_product2_minus_product1"],
        color="#2563eb",
        marker="o",
        linewidth=2.0,
    )
    ax_q.set_title("Initial-state incentive for product 2", loc="left")
    ax_q.set_ylabel("Q2 - Q1 at (S,F) = initial")
    prettify_axes(ax_q)

    ax_p2.plot(
        summary["p0"],
        summary["demand_weighted_share_states_product2"],
        color="#475569",
        marker="o",
        linewidth=2.0,
        label="Demand-weighted states",
    )
    ax_p2.plot(
        summary["p0"],
        summary["mean_product2_rate_when_A_chosen_sim"],
        color="#dc2626",
        marker="o",
        linewidth=2.0,
        label="Simulated when A is chosen",
    )
    ax_p2.step(
        summary["p0"],
        summary["initial_uses_product2"],
        where="mid",
        color="#111827",
        linewidth=1.4,
        label="Initial state uses product 2",
    )
    ax_p2.set_title("High-quality product use", loc="left")
    ax_p2.set_xlabel("Known competitor success probability p_0")
    ax_p2.set_ylabel("Share using product 2")
    ax_p2.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_p2.set_ylim(-0.03, 1.03)
    prettify_axes(ax_p2)
    ax_p2.legend()

    ax_profit.plot(
        summary["p0"],
        summary["mean_profit_per_period_sim"],
        color="#7c3aed",
        marker="o",
        linewidth=2.0,
        label="Full horizon",
    )
    ax_profit.plot(
        summary["p0"],
        summary["mean_tail_profit_per_period_sim"],
        color="#0f766e",
        marker="o",
        linewidth=2.0,
        label="Second half",
    )
    ax_profit.set_title("Simulated average profit", loc="left")
    ax_profit.set_xlabel("Known competitor success probability p_0")
    ax_profit.set_ylabel("Average profit per period")
    prettify_axes(ax_profit)
    ax_profit.legend()

    save_figure(fig, outputs_dir / "average_reward_by_p0.png")


def plot_policy_heatmaps(
    solutions: list[dict],
    state_space: StateSpace,
    outputs_dir: Path,
) -> None:
    selected_count = min(6, len(solutions))
    selected_indices = sorted(
        set(np.linspace(0, len(solutions) - 1, selected_count).round().astype(int))
    )
    selected_solutions = [solutions[idx] for idx in selected_indices]

    cmap = ListedColormap(["#f2c94c", "#0f766e"])
    cmap.set_bad("#e5e7eb")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    n_panels = len(selected_solutions)
    ncols = 3 if n_panels > 3 else n_panels
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.9 * ncols, 4.5 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    max_obs = state_space.state_index.shape[0] - 1
    image = None
    for ax, solution in zip(axes, selected_solutions, strict=False):
        matrix = np.full((max_obs + 1, max_obs + 1), np.nan)
        uses_product2 = (solution["policy_product"] == 2).astype(float)
        matrix[state_space.F, state_space.S] = uses_product2

        image = ax.imshow(
            matrix,
            origin="lower",
            extent=[-0.5, max_obs + 0.5, -0.5, max_obs + 0.5],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(f"p_0 = {solution['p0']:.2f}", loc="left")
        ax.set_xlabel("Successes S")
        ax.set_ylabel("Failures F")
        ax.plot([0], [0], marker="o", color="#111827", markersize=3)
        prettify_axes(ax, grid_axis="both")

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    cbar = fig.colorbar(image, ax=axes[:n_panels], ticks=[0, 1], shrink=0.82)
    cbar.ax.set_yticklabels(["Product 1: cheap", "Product 2: quality"])
    fig.suptitle("Average-reward best-response policy", x=0.01, ha="left")
    save_figure(fig, outputs_dir / "average_reward_policy_heatmaps.png")


def plot_policy_posterior_state_space(
    solutions: list[dict],
    state_space: StateSpace,
    outputs_dir: Path,
) -> None:
    selected_count = min(6, len(solutions))
    selected_indices = sorted(
        set(np.linspace(0, len(solutions) - 1, selected_count).round().astype(int))
    )
    selected_solutions = [solutions[idx] for idx in selected_indices]

    cmap = ListedColormap(["#f2c94c", "#0f766e"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    posterior_m = posterior_mean(state_space.S, state_space.F)
    posterior_s = posterior_std(state_space.S, state_space.F)

    n_panels = len(selected_solutions)
    ncols = 3 if n_panels > 3 else n_panels
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.9 * ncols, 4.5 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    image = None
    for ax, solution in zip(axes, selected_solutions, strict=False):
        uses_product2 = (solution["policy_product"] == 2).astype(float)
        image = ax.scatter(
            posterior_m,
            posterior_s,
            c=uses_product2,
            cmap=cmap,
            norm=norm,
            s=9,
            marker="s",
            linewidths=0.0,
            alpha=0.9,
            rasterized=True,
        )
        ax.axvline(
            solution["p0"],
            color="#111827",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.plot(
            [0.5],
            [np.sqrt(1.0 / 12.0)],
            marker="o",
            color="#111827",
            markersize=3.2,
        )
        ax.set_title(f"p_0 = {solution['p0']:.2f}", loc="left")
        ax.set_xlabel("Posterior mean E[theta | S,F]")
        ax.set_ylabel("Posterior standard deviation")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.005, np.sqrt(1.0 / 12.0) * 1.04)
        prettify_axes(ax, grid_axis="both")

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    cbar = fig.colorbar(image, ax=axes[:n_panels], ticks=[0, 1], shrink=0.82)
    cbar.ax.set_yticklabels(["Product 1: cheap", "Product 2: quality"])
    fig.suptitle(
        "Average-reward policy over posterior state space",
        x=0.01,
        ha="left",
    )
    save_figure(fig, outputs_dir / "average_reward_policy_posterior_state_space.png")


def plot_bias_gap_posterior_state_space(
    solutions: list[dict],
    state_space: StateSpace,
    params: ModelParams,
    outputs_dir: Path,
) -> None:
    selected_count = min(6, len(solutions))
    selected_indices = sorted(
        set(np.linspace(0, len(solutions) - 1, selected_count).round().astype(int))
    )
    selected_solutions = [solutions[idx] for idx in selected_indices]

    posterior_m = posterior_mean(state_space.S, state_space.F)
    posterior_s = posterior_std(state_space.S, state_space.F)
    threshold = product2_bias_gap_threshold(params)
    centered_values = [
        solution["bias_gap"] - threshold for solution in selected_solutions
    ]
    all_centered_values = np.concatenate(centered_values)
    vlim = float(np.nanpercentile(np.abs(all_centered_values), 98))
    if not np.isfinite(vlim) or vlim <= 0.0:
        vlim = 1.0
    norm = TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    n_panels = len(selected_solutions)
    ncols = 3 if n_panels > 3 else n_panels
    nrows = int(np.ceil(n_panels / ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.9 * ncols, 4.5 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    image = None
    for ax, solution, centered in zip(
        axes,
        selected_solutions,
        centered_values,
        strict=False,
    ):
        image = ax.scatter(
            posterior_m,
            posterior_s,
            c=centered,
            cmap="RdBu_r",
            norm=norm,
            s=9,
            marker="s",
            linewidths=0.0,
            alpha=0.9,
            rasterized=True,
        )
        ax.axvline(
            solution["p0"],
            color="#111827",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
        )
        ax.plot(
            [0.5],
            [np.sqrt(1.0 / 12.0)],
            marker="o",
            color="#111827",
            markersize=3.2,
        )
        ax.set_title(f"p_0 = {solution['p0']:.2f}", loc="left")
        ax.set_xlabel("Posterior mean E[theta | S,F]")
        ax.set_ylabel("Posterior standard deviation")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.005, np.sqrt(1.0 / 12.0) * 1.04)
        prettify_axes(ax, grid_axis="both")

    for ax in axes[n_panels:]:
        ax.set_visible(False)

    cbar = fig.colorbar(image, ax=axes[:n_panels], shrink=0.82)
    cbar.set_label("h(S+1,F) - h(S,F+1) - threshold")
    fig.suptitle("Relative-bias incentive for product 2", x=0.01, ha="left")
    save_figure(fig, outputs_dir / "average_reward_bias_gap_posterior_state_space.png")


def plot_simulation_timeseries(
    time_series: pd.DataFrame,
    outputs_dir: Path,
    initial_belief_mean: float | None = None,
    filename_suffix: str = "",
) -> None:
    selected_count = min(5, time_series["p0"].nunique())
    p0_values = np.array(sorted(time_series["p0"].unique()))
    selected_indices = sorted(
        set(np.linspace(0, len(p0_values) - 1, selected_count).round().astype(int))
    )
    selected_p0 = p0_values[selected_indices]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), sharex=True)
    ax_share, ax_product, ax_profit, ax_mean = axes.ravel()
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(selected_p0)))

    for p0, color in zip(selected_p0, colors, strict=True):
        data = time_series[np.isclose(time_series["p0"], p0)]
        label = f"p_0={p0:.2f}"
        ax_share.plot(
            data["t"],
            data["A_market_share"],
            color=color,
            linewidth=2.0,
            label=label,
        )
        ax_product.plot(
            data["t"],
            data["product2_rate_when_A_chosen"],
            color=color,
            linewidth=2.0,
            label=label,
        )
        ax_profit.plot(
            data["t"],
            data["avg_profit_per_period"],
            color=color,
            linewidth=2.0,
            label=label,
        )
        ax_mean.plot(
            data["t"],
            data["user_posterior_mean_A"],
            color=color,
            linewidth=2.0,
            label=label,
        )

    share_title = "Simulated demand path"
    mean_title = "User posterior mean for Seller A"
    if initial_belief_mean is not None:
        share_title += f" (initial mean={initial_belief_mean:.2f})"
        mean_title += f" (initial mean={initial_belief_mean:.2f})"

    ax_share.set_title(share_title, loc="left")
    ax_share.set_ylabel("A market share")
    ax_share.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_share.set_ylim(-0.03, 1.03)
    prettify_axes(ax_share)
    ax_share.legend(ncols=min(3, len(selected_p0)))

    ax_product.set_title("Product 2 use when A is chosen", loc="left")
    ax_product.set_ylabel("Product 2 rate")
    ax_product.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_product.set_ylim(-0.03, 1.03)
    prettify_axes(ax_product)

    ax_profit.set_title("Per-period profit", loc="left")
    ax_profit.set_xlabel("Period")
    ax_profit.set_ylabel("Average profit")
    prettify_axes(ax_profit)

    ax_mean.set_title(mean_title, loc="left")
    ax_mean.set_xlabel("Period")
    ax_mean.set_ylabel("E[theta | S,F]")
    ax_mean.set_ylim(-0.03, 1.03)
    prettify_axes(ax_mean)

    suffix = f"_{filename_suffix}" if filename_suffix else ""
    save_figure(fig, outputs_dir / f"simulation_paths_by_p0{suffix}.png")


def plot_sample_simulation_paths(
    path_diagnostics: pd.DataFrame,
    outputs_dir: Path,
    target_p0: float,
    n_paths: int = 5,
) -> None:
    if n_paths <= 0:
        return

    target = path_diagnostics[np.isclose(path_diagnostics["p0"], target_p0)].copy()
    if target.empty:
        return

    rep_ids = np.array(sorted(target["rep"].unique()))
    if rep_ids.size > n_paths:
        selected_indices = np.linspace(0, rep_ids.size - 1, n_paths).round().astype(int)
        rep_ids = rep_ids[selected_indices]

    paths = target[target["rep"].isin(rep_ids)].sort_values(["rep", "t"]).copy()
    paths["product2_when_A_chosen"] = np.where(
        paths["chosen_A"] == 1,
        (paths["product_used"] == 2).astype(float),
        0.0,
    )
    chosen_cum = paths.groupby("rep", sort=False)["chosen_A"].cumsum()
    product2_cum = paths.groupby("rep", sort=False)["product2_when_A_chosen"].cumsum()
    paths["realized_A_market_share"] = chosen_cum / paths["t"]
    paths["realized_product2_rate_when_A_chosen"] = np.where(
        chosen_cum > 0,
        product2_cum / chosen_cum,
        np.nan,
    )

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.8), sharex=True)
    ax_mean, ax_demand, ax_share, ax_events = axes.ravel()
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(rep_ids)))

    for path_num, (rep_id, color) in enumerate(
        zip(rep_ids, colors, strict=True),
        start=1,
    ):
        data = paths[paths["rep"] == rep_id]
        label = f"path {path_num}"
        ax_mean.plot(
            data["t"],
            data["posterior_mean"],
            color=color,
            linewidth=1.7,
            label=label,
        )
        ax_demand.plot(
            data["t"],
            data["demand_probability"],
            color=color,
            linewidth=1.7,
            label=label,
        )
        ax_share.plot(
            data["t"],
            data["realized_A_market_share"],
            color=color,
            linewidth=1.7,
            label=label,
        )

        chosen = data[data["chosen_A"] == 1]
        product1 = chosen[chosen["product_used"] == 1]
        product2 = chosen[chosen["product_used"] == 2]
        ax_events.scatter(
            product1["t"],
            np.full(len(product1), path_num),
            color="#9ca3af",
            s=7,
            alpha=0.22,
            linewidths=0.0,
        )
        ax_events.scatter(
            product2["t"],
            np.full(len(product2), path_num),
            color=color,
            s=16,
            alpha=0.92,
            marker="s",
            linewidths=0.0,
        )

    ax_mean.axhline(
        target_p0,
        color="#111827",
        linewidth=1.1,
        linestyle="--",
        label=f"p_0={target_p0:.2f}",
    )
    ax_mean.set_title("User posterior mean for Seller A", loc="left")
    ax_mean.set_ylabel("E[theta | S,F]")
    ax_mean.set_ylim(-0.03, 1.03)
    prettify_axes(ax_mean)
    ax_mean.legend(ncols=min(3, len(rep_ids) + 1))

    ax_demand.set_title("Demand probability for Seller A", loc="left")
    ax_demand.set_ylabel("rho(S,F)")
    ax_demand.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_demand.set_ylim(-0.03, 1.03)
    prettify_axes(ax_demand)

    ax_share.set_title("Realized cumulative A market share", loc="left")
    ax_share.set_xlabel("Period")
    ax_share.set_ylabel("A market share")
    ax_share.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_share.set_ylim(-0.03, 1.03)
    prettify_axes(ax_share)

    ax_events.set_title("Product used when A is chosen", loc="left")
    ax_events.set_xlabel("Period")
    ax_events.set_ylabel("Sample path")
    ax_events.set_yticks(range(1, len(rep_ids) + 1))
    ax_events.set_ylim(0.5, len(rep_ids) + 0.5)
    prettify_axes(ax_events)
    ax_events.scatter([], [], color="#9ca3af", s=18, alpha=0.45, label="Product 1")
    ax_events.scatter([], [], color="#111827", s=28, marker="s", label="Product 2")
    ax_events.legend(loc="upper right")

    fig.suptitle(
        f"Sample simulation paths at p_0={target_p0:.2f}",
        x=0.01,
        ha="left",
    )
    filename = f"sample_paths_p0_{p0_filename_fragment(target_p0)}.png"
    save_figure(fig, outputs_dir / filename)


def plot_convergence(convergence: pd.DataFrame, outputs_dir: Path) -> None:
    optimal = convergence[convergence["fixed_product"] == "optimal"].copy()
    if optimal.empty:
        return

    selected_count = min(6, optimal["p0"].nunique())
    p0_values = np.array(sorted(optimal["p0"].unique()))
    selected_indices = sorted(
        set(np.linspace(0, len(p0_values) - 1, selected_count).round().astype(int))
    )
    selected_p0 = p0_values[selected_indices]

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(selected_p0)))
    for p0, color in zip(selected_p0, colors, strict=True):
        data = optimal[np.isclose(optimal["p0"], p0)]
        ax.semilogy(
            data["iteration"],
            data["span_residual"],
            color=color,
            linewidth=2.0,
            label=f"p_0={p0:.2f}",
        )

    ax.set_title("Average-reward policy iteration convergence", loc="left")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Span residual")
    prettify_axes(ax)
    ax.legend(ncols=min(3, len(selected_p0)))
    save_figure(fig, outputs_dir / "average_reward_policy_iteration.png")


def add_posterior_bins(
    frame: pd.DataFrame,
    value_column: str = "use_product_2",
    bin_count: int = POSTERIOR_BIN_COUNT,
) -> pd.DataFrame:
    binned = frame.copy()
    binned["posterior_mean_bin"] = pd.cut(
        binned["posterior_mean"],
        bins=np.linspace(0.0, 1.0, bin_count + 1),
        include_lowest=True,
    )
    grouped = (
        binned.groupby(["p0", "posterior_mean_bin"], observed=True)
        .agg(
            posterior_mean=("posterior_mean", "mean"),
            value=(value_column, "mean"),
            state_count=(value_column, "size"),
        )
        .reset_index()
    )
    return grouped


def plot_policy_by_posterior_mean(
    policy_states: pd.DataFrame,
    outputs_dir: Path,
    filename: str = "policy_by_posterior_mean.png",
) -> None:
    selected_count = min(5, policy_states["p0"].nunique())
    p0_values = np.array(sorted(policy_states["p0"].unique()))
    selected_indices = sorted(
        set(np.linspace(0, len(p0_values) - 1, selected_count).round().astype(int))
    )
    selected_p0 = p0_values[selected_indices]

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(selected_p0)))
    for p0, color in zip(selected_p0, colors, strict=True):
        data = policy_states[np.isclose(policy_states["p0"], p0)]
        binned = add_posterior_bins(data)
        ax.plot(
            binned["posterior_mean"],
            binned["value"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=f"p_0={p0:.2f}",
        )

    ax.set_title("Product 2 policy by posterior mean", loc="left")
    ax.set_xlabel("Posterior mean E[theta | S,F]")
    ax.set_ylabel("Share of states using product 2")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, 1.03)
    prettify_axes(ax)
    ax.legend(ncols=min(3, len(selected_p0)))
    save_figure(fig, outputs_dir / filename)


def plot_rho_by_posterior_mean(
    policy_states: pd.DataFrame,
    outputs_dir: Path,
    filename: str = "rho_by_posterior_mean.png",
) -> None:
    selected_count = min(5, policy_states["p0"].nunique())
    p0_values = np.array(sorted(policy_states["p0"].unique()))
    selected_indices = sorted(
        set(np.linspace(0, len(p0_values) - 1, selected_count).round().astype(int))
    )
    selected_p0 = p0_values[selected_indices]

    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    colors = plt.cm.plasma(np.linspace(0.08, 0.88, len(selected_p0)))
    for p0, color in zip(selected_p0, colors, strict=True):
        data = policy_states[np.isclose(policy_states["p0"], p0)]
        binned = add_posterior_bins(data, value_column="rho")
        ax.plot(
            binned["posterior_mean"],
            binned["value"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.5,
            label=f"p_0={p0:.2f}",
        )

    ax.set_title("Demand probability by posterior mean", loc="left")
    ax.set_xlabel("Posterior mean E[theta | S,F]")
    ax.set_ylabel("rho(S,F)")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.03, 1.03)
    prettify_axes(ax)
    ax.legend(ncols=min(3, len(selected_p0)))
    save_figure(fig, outputs_dir / filename)


def solve_finite_horizon_policy(
    p0: float,
    params: ModelParams,
    horizon: int,
) -> dict:
    """Solve the exact finite-horizon Bayesian DP by backward induction."""
    if horizon <= 0:
        raise ValueError("finite horizon T must be positive.")

    state_space = make_state_space(horizon, method="rolling_window")
    n_states = len(state_space.S)
    value_by_time = np.full((horizon + 2, n_states), np.nan, dtype=float)
    policy_by_time = np.zeros((horizon + 1, n_states), dtype=np.int8)
    value_by_time[horizon + 1, :] = 0.0
    finite_horizon_violations = 0

    for t in range(horizon, 0, -1):
        reachable_idx = np.flatnonzero(state_space.total <= t - 1)
        current_values = np.full(n_states, np.nan, dtype=float)
        current_policy = np.zeros(n_states, dtype=np.int8)
        successes = state_space.S[reachable_idx]
        failures = state_space.F[reachable_idx]
        rho = beta_ccdf(p0, successes, failures)
        successor_success_idx = state_space.state_index[successes + 1, failures]
        successor_failure_idx = state_space.state_index[successes, failures + 1]

        if np.any(state_space.total[successor_success_idx] > t):
            finite_horizon_violations += 1
        if np.any(state_space.total[successor_failure_idx] > t):
            finite_horizon_violations += 1

        continuation_same = value_by_time[t + 1, reachable_idx]
        continuation_success = value_by_time[t + 1, successor_success_idx]
        continuation_failure = value_by_time[t + 1, successor_failure_idx]

        q1 = (
            rho * (params.revenue - params.c1)
            + (1.0 - rho) * continuation_same
            + rho
            * (
                params.p1 * continuation_success
                + (1.0 - params.p1) * continuation_failure
            )
        )
        q2 = (
            rho * (params.revenue - params.c2)
            + (1.0 - rho) * continuation_same
            + rho
            * (
                params.p2 * continuation_success
                + (1.0 - params.p2) * continuation_failure
            )
        )
        use_product2 = q2 > q1 + params.tol
        current_policy[reachable_idx] = np.where(use_product2, 2, 1)
        current_values[reachable_idx] = np.maximum(q1, q2)
        policy_by_time[t, :] = current_policy
        value_by_time[t, :] = current_values

    initial_idx = int(state_space.state_index[0, 0])
    avg_value = float(value_by_time[1, initial_idx] / horizon)
    return {
        "method": FINITE_HORIZON_METHOD,
        "p0": p0,
        "T": horizon,
        "state_space": state_space,
        "value_by_time": value_by_time,
        "policy_by_time": policy_by_time,
        "avg_value_T": avg_value,
        "initial_action": int(policy_by_time[1, initial_idx]),
        "finite_horizon_transition_violations": finite_horizon_violations,
    }


def build_finite_horizon_policy_table(solution: dict) -> pd.DataFrame:
    state_space: StateSpace = solution["state_space"]
    horizon = int(solution["T"])
    frames = []
    for t in range(1, horizon + 1):
        idx = np.flatnonzero(state_space.total <= t - 1)
        frames.append(
            pd.DataFrame(
                {
                    "method": FINITE_HORIZON_METHOD,
                    "p0": solution["p0"],
                    "T": horizon,
                    "t": t,
                    "time_remaining": horizon - t + 1,
                    "S": state_space.S[idx],
                    "F": state_space.F[idx],
                    "total_count": state_space.total[idx],
                    "posterior_mean": posterior_mean(state_space.S[idx], state_space.F[idx]),
                    "rho": beta_ccdf(solution["p0"], state_space.S[idx], state_space.F[idx]),
                    "action": solution["policy_by_time"][t, idx].astype(int),
                    "use_product_2": (
                        solution["policy_by_time"][t, idx].astype(int) == 2
                    ).astype(int),
                    "V_t": solution["value_by_time"][t, idx],
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def summarize_finite_horizon_solution(
    solution: dict,
    policy_table: pd.DataFrame,
) -> dict:
    return {
        "method": FINITE_HORIZON_METHOD,
        "p0": solution["p0"],
        "T": solution["T"],
        "N_or_T": solution["T"],
        "avg_value_T": solution["avg_value_T"],
        "eta_or_avg_value": solution["avg_value_T"],
        "initial_action": solution["initial_action"],
        "fraction_states_product_2": policy_table["use_product_2"].mean(),
        "fraction_reachable_states_product_2": policy_table["use_product_2"].mean(),
        "product_2_share": policy_table["use_product_2"].mean(),
        "finite_horizon_transition_violations": solution[
            "finite_horizon_transition_violations"
        ],
    }


def build_finite_horizon_usage_tables(
    policy_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    usage_by_time = (
        policy_table.groupby(["p0", "T", "time_remaining"], observed=True)
        .agg(
            product_2_share=("use_product_2", "mean"),
            state_count=("use_product_2", "size"),
        )
        .reset_index()
    )
    binned = add_posterior_bins(policy_table)
    binned = binned.rename(columns={"value": "product_2_share"})
    binned["T"] = policy_table["T"].iloc[0]
    return usage_by_time, binned


def plot_finite_horizon_policy_regions(
    solution: dict,
    outputs_dir: Path,
) -> None:
    state_space: StateSpace = solution["state_space"]
    horizon = int(solution["T"])
    selected_remaining = sorted(
        set(
            [
                horizon,
                max(1, int(round(0.75 * horizon))),
                max(1, int(round(0.50 * horizon))),
                max(1, int(round(0.25 * horizon))),
                1,
            ]
        ),
        reverse=True,
    )
    cmap = ListedColormap(["#f2c94c", "#0f766e"])
    cmap.set_bad("#e5e7eb")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    n_panels = len(selected_remaining)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 4.3 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    image = None
    for ax, time_remaining in zip(axes, selected_remaining, strict=False):
        t = horizon - time_remaining + 1
        matrix = np.full((horizon + 1, horizon + 1), np.nan)
        idx = np.flatnonzero(state_space.total <= t - 1)
        matrix[state_space.F[idx], state_space.S[idx]] = (
            solution["policy_by_time"][t, idx] == 2
        ).astype(float)
        image = ax.imshow(
            matrix,
            origin="lower",
            extent=[-0.5, horizon + 0.5, -0.5, horizon + 0.5],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(f"time remaining = {time_remaining}", loc="left")
        ax.set_xlabel("Successes S")
        ax.set_ylabel("Failures F")
        prettify_axes(ax, grid_axis="both")
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    cbar = fig.colorbar(image, ax=axes[:n_panels], ticks=[0, 1], shrink=0.82)
    cbar.ax.set_yticklabels(["Product 1", "Product 2"])
    fig.suptitle(
        f"Finite-horizon policy regions, p_0={solution['p0']:.2f}, T={horizon}",
        x=0.01,
        ha="left",
    )
    save_figure(fig, outputs_dir / "finite_horizon_policy_regions.png")


def plot_finite_horizon_usage(
    usage_by_time: pd.DataFrame,
    usage_by_mean: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    p0 = usage_by_time["p0"].iloc[0]
    horizon = usage_by_time["T"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    ax_time, ax_mean = axes
    ax_time.plot(
        usage_by_time["time_remaining"],
        usage_by_time["product_2_share"],
        color="#0f766e",
        linewidth=2.0,
        marker="o",
        markersize=3.0,
    )
    ax_time.set_title("Product 2 use by time remaining", loc="left")
    ax_time.set_xlabel("Time remaining")
    ax_time.set_ylabel("Share of states using product 2")
    ax_time.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_time.set_ylim(-0.03, 1.03)
    prettify_axes(ax_time)

    ax_mean.plot(
        usage_by_mean["posterior_mean"],
        usage_by_mean["product_2_share"],
        color="#2563eb",
        linewidth=2.0,
        marker="o",
        markersize=3.0,
    )
    ax_mean.set_title("Product 2 use by posterior mean", loc="left")
    ax_mean.set_xlabel("Posterior mean E[theta | S,F]")
    ax_mean.set_ylabel("Share of states using product 2")
    ax_mean.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_mean.set_xlim(-0.02, 1.02)
    ax_mean.set_ylim(-0.03, 1.03)
    prettify_axes(ax_mean)

    fig.suptitle(f"Finite-horizon summaries, p_0={p0:.2f}, T={horizon}", x=0.01, ha="left")
    save_figure(fig, outputs_dir / "finite_horizon_product2_usage.png")


def plot_finite_horizon_mid_policy_regions_by_p0(
    solutions: list[dict],
    outputs_dir: Path,
) -> None:
    if not solutions:
        return
    horizon = int(solutions[0]["T"])
    time_remaining = max(1, int(round(0.5 * horizon)))
    t = horizon - time_remaining + 1
    n_panels = len(solutions)
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    cmap = ListedColormap(["#f2c94c", "#0f766e"])
    cmap.set_bad("#e5e7eb")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.8 * ncols, 4.3 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()
    image = None
    for ax, solution in zip(axes, solutions, strict=False):
        state_space: StateSpace = solution["state_space"]
        matrix = np.full((horizon + 1, horizon + 1), np.nan)
        idx = np.flatnonzero(state_space.total <= t - 1)
        matrix[state_space.F[idx], state_space.S[idx]] = (
            solution["policy_by_time"][t, idx] == 2
        ).astype(float)
        image = ax.imshow(
            matrix,
            origin="lower",
            extent=[-0.5, horizon + 0.5, -0.5, horizon + 0.5],
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(f"p_0={solution['p0']:.2f}", loc="left")
        ax.set_xlabel("Successes S")
        ax.set_ylabel("Failures F")
        prettify_axes(ax, grid_axis="both")
    for ax in axes[n_panels:]:
        ax.set_visible(False)
    cbar = fig.colorbar(image, ax=axes[:n_panels], ticks=[0, 1], shrink=0.82)
    cbar.ax.set_yticklabels(["Product 1", "Product 2"])
    fig.suptitle(
        "Finite-horizon policy regions across p_0 "
        f"(time remaining={time_remaining}, T={horizon})",
        x=0.01,
        ha="left",
    )
    save_figure(fig, outputs_dir / "finite_horizon_mid_policy_regions_by_p0.png")


def plot_finite_horizon_usage_all_p0(
    usage_by_time: pd.DataFrame,
    usage_by_mean: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    if usage_by_time.empty or usage_by_mean.empty:
        return
    horizon = int(usage_by_time["T"].iloc[0])
    p0_values = np.array(sorted(usage_by_time["p0"].unique()))
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(p0_values)))

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    ax_time, ax_mean = axes
    for p0, color in zip(p0_values, colors, strict=True):
        time_data = usage_by_time[np.isclose(usage_by_time["p0"], p0)].sort_values(
            "time_remaining"
        )
        mean_data = usage_by_mean[np.isclose(usage_by_mean["p0"], p0)].sort_values(
            "posterior_mean"
        )
        label = f"p_0={p0:.2f}"
        ax_time.plot(
            time_data["time_remaining"],
            time_data["product_2_share"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.0,
            label=label,
        )
        ax_mean.plot(
            mean_data["posterior_mean"],
            mean_data["product_2_share"],
            color=color,
            linewidth=2.0,
            marker="o",
            markersize=3.0,
            label=label,
        )

    ax_time.set_title("Product 2 use by time remaining", loc="left")
    ax_time.set_xlabel("Time remaining")
    ax_time.set_ylabel("Share of states using product 2")
    ax_time.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_time.set_ylim(-0.03, 1.03)
    prettify_axes(ax_time)
    ax_time.legend(ncols=min(3, len(p0_values)))

    ax_mean.set_title("Product 2 use by posterior mean", loc="left")
    ax_mean.set_xlabel("Posterior mean E[theta | S,F]")
    ax_mean.set_ylabel("Share of states using product 2")
    ax_mean.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_mean.set_xlim(-0.02, 1.02)
    ax_mean.set_ylim(-0.03, 1.03)
    prettify_axes(ax_mean)

    fig.suptitle(
        f"Finite-horizon product 2 summaries across p_0, T={horizon}",
        x=0.01,
        ha="left",
    )
    save_figure(fig, outputs_dir / "finite_horizon_product2_usage_all_p0.png")


def plot_finite_horizon_time_remaining_extended(
    by_time: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    if by_time.empty:
        return
    horizon = int(by_time["T"].iloc[0])
    p0_values = np.array(sorted(by_time["p0"].unique()))
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(p0_values)))
    metrics = [
        (
            "simulated_product2_given_A_share",
            "Product 2 use conditional on A",
            "Share",
            True,
        ),
        ("simulated_A_choice_share", "A-choice share", "Share", True),
        ("simulated_avg_profit", "Average profit", "Profit", False),
    ]
    fig, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(10.8, 3.35 * len(metrics)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    for ax, (column, title, ylabel, as_percent) in zip(axes, metrics, strict=True):
        for p0, color in zip(p0_values, colors, strict=True):
            data = by_time[np.isclose(by_time["p0"], p0)].sort_values("tau")
            if column not in data:
                continue
            ax.plot(
                data["tau"],
                data[column],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3.0,
                label=f"p_0={p0:.2f}",
            )
        ax.set_title(title, loc="left")
        ax.set_ylabel(ylabel)
        if as_percent:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            ax.set_ylim(-0.03, 1.03)
        prettify_axes(ax)
    axes[-1].set_xlabel("Time remaining tau")
    axes[0].legend(ncols=min(5, len(p0_values)))
    fig.suptitle(
        f"Finite-horizon simulated diagnostics by time remaining, T={horizon}",
        x=0.01,
        ha="left",
    )
    save_figure(fig, outputs_dir / "finite_horizon_time_remaining_extended.png")


def plot_finite_horizon_policy_heatmaps_extended(
    solutions: list[dict],
    outputs_dir: Path,
) -> None:
    if not solutions:
        return
    horizon = int(solutions[0]["T"])
    selected_tau = sorted(
        {
            horizon,
            max(1, int(round(0.75 * horizon))),
            max(1, int(round(0.5 * horizon))),
            max(1, int(round(0.25 * horizon))),
            min(10, horizon),
            1,
        },
        reverse=True,
    )
    p0_values = [float(solution["p0"]) for solution in solutions]
    nrows = len(selected_tau)
    ncols = len(solutions)
    cmap = ListedColormap(["#f2c94c", "#0f766e"])
    cmap.set_bad("#e5e7eb")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(3.25 * ncols, 2.85 * nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = np.atleast_2d(axes)
    image = None
    for col_idx, solution in enumerate(solutions):
        state_space: StateSpace = solution["state_space"]
        for row_idx, tau in enumerate(selected_tau):
            t = horizon - tau + 1
            ax = axes[row_idx, col_idx]
            matrix = np.full((horizon + 1, horizon + 1), np.nan)
            idx = np.flatnonzero(state_space.total <= t - 1)
            matrix[state_space.F[idx], state_space.S[idx]] = (
                solution["policy_by_time"][t, idx] == 2
            ).astype(float)
            image = ax.imshow(
                matrix,
                origin="lower",
                extent=[-0.5, horizon + 0.5, -0.5, horizon + 0.5],
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )
            if row_idx == 0:
                ax.set_title(f"p_0={p0_values[col_idx]:.2f}", loc="left")
            if col_idx == 0:
                ax.set_ylabel(f"tau={tau}\nFailures F")
            if row_idx == nrows - 1:
                ax.set_xlabel("Successes S")
            prettify_axes(ax, grid_axis="both")
    cbar = fig.colorbar(image, ax=axes.ravel(), ticks=[0, 1], shrink=0.82)
    cbar.ax.set_yticklabels(["Product 1", "Product 2"])
    fig.suptitle(
        f"Finite-horizon policy heatmaps by p_0 and time remaining, T={horizon}",
        x=0.01,
        ha="left",
    )
    save_figure(fig, outputs_dir / "finite_horizon_policy_heatmaps_extended.png")


def plot_finite_horizon_value_curve(
    comparison: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    finite = comparison[comparison["method"] == FINITE_HORIZON_METHOD].copy()
    if finite.empty:
        return
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    p0_values = np.array(sorted(finite["p0"].unique()))
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(p0_values)))
    for p0, color in zip(p0_values, colors, strict=True):
        data = finite[np.isclose(finite["p0"], p0)].sort_values("N_or_T")
        ax.plot(
            data["N_or_T"],
            data["eta_or_avg_value"],
            color=color,
            linewidth=2.0,
            marker="o",
            label=f"p_0={p0:.2f}",
        )
    ax.set_title("Finite-horizon average value by horizon", loc="left")
    ax.set_xlabel("T")
    ax.set_ylabel("V_1(0,0) / T")
    prettify_axes(ax)
    ax.legend(ncols=min(3, len(p0_values)))
    save_figure(fig, outputs_dir / "finite_horizon_avg_value_by_T.png")


def plot_method_comparison_dashboard(
    comparison: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    if comparison.empty:
        return
    methods = ["rolling_window", "projection", FINITE_HORIZON_METHOD]
    present_methods = [method for method in methods if method in set(comparison["method"])]
    if not present_methods:
        return

    fig, axes = plt.subplots(
        len(present_methods),
        2,
        figsize=(12.2, 3.8 * len(present_methods)),
        sharex=False,
        constrained_layout=True,
        squeeze=False,
    )
    axes = np.atleast_2d(axes)
    p0_values = np.array(sorted(comparison["p0"].unique()))
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(p0_values)))
    method_labels = {
        "rolling_window": "Rolling window",
        "projection": "Projection",
        FINITE_HORIZON_METHOD: "Finite horizon",
    }

    for row_idx, method in enumerate(present_methods):
        method_data = comparison[comparison["method"] == method]
        ax_value, ax_product = axes[row_idx]
        for p0, color in zip(p0_values, colors, strict=True):
            data = method_data[np.isclose(method_data["p0"], p0)].sort_values("N_or_T")
            if data.empty:
                continue
            label = f"p_0={p0:.2f}"
            ax_value.plot(
                data["N_or_T"],
                data["eta_or_avg_value"],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3.8,
                label=label,
            )
            ax_product.plot(
                data["N_or_T"],
                data["product_2_share"],
                color=color,
                linewidth=2.0,
                marker="o",
                markersize=3.8,
                label=label,
            )

        x_label = "T" if method == FINITE_HORIZON_METHOD else "N"
        ax_value.set_title(f"{method_labels[method]} value", loc="left")
        ax_value.set_xlabel(x_label)
        ax_value.set_ylabel("eta or V_1(0,0)/T")
        prettify_axes(ax_value)
        ax_value.legend(ncols=min(3, len(p0_values)))

        ax_product.set_title(f"{method_labels[method]} product 2 share", loc="left")
        ax_product.set_xlabel(x_label)
        ax_product.set_ylabel("Share of states using product 2")
        ax_product.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax_product.set_ylim(-0.03, 1.03)
        prettify_axes(ax_product)

    fig.suptitle("Robustness comparison across methods and p_0", x=0.01, ha="left")
    save_figure(fig, outputs_dir / "method_comparison_dashboard.png")


def plot_method_comparison_dashboard_extended(
    comparison: pd.DataFrame,
    path_weighted: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    if comparison.empty or path_weighted.empty:
        return
    method_order = [
        "rolling_window",
        "projection",
        "stochastic_projection",
        FINITE_HORIZON_METHOD,
    ]
    method_labels = {
        "rolling_window": "Rolling",
        "projection": "Deterministic projection",
        "stochastic_projection": "Stochastic projection",
        FINITE_HORIZON_METHOD: "Finite horizon",
    }
    colors = {
        "rolling_window": "#0f766e",
        "projection": "#dc2626",
        "stochastic_projection": "#2563eb",
        FINITE_HORIZON_METHOD: "#7c3aed",
    }
    p0_values = np.array(sorted(comparison["p0"].unique()))
    metrics = [
        ("eta_or_avg_value", "Value", comparison, "eta or V_1(0,0)/T"),
        ("product_2_share", "Unweighted product 2 share", comparison, "Share"),
        (
            "product2_given_A_share",
            "Path-weighted product 2 | A",
            path_weighted,
            "Share",
        ),
        ("A_choice_share", "Demand share", path_weighted, "Share"),
        ("avg_realized_profit", "Realized average profit", path_weighted, "Profit"),
    ]

    fig, axes = plt.subplots(
        len(metrics),
        len(p0_values),
        figsize=(4.25 * len(p0_values), 3.1 * len(metrics)),
        sharex=False,
        constrained_layout=True,
        squeeze=False,
    )
    axes = np.atleast_2d(axes)

    for col_idx, p0 in enumerate(p0_values):
        for row_idx, (column, title, source, ylabel) in enumerate(metrics):
            ax = axes[row_idx, col_idx]
            for method in method_order:
                data = source[
                    (source["method"] == method)
                    & np.isclose(source["p0"].astype(float), p0)
                ].sort_values("N_or_T")
                if data.empty:
                    continue
                ax.plot(
                    data["N_or_T"],
                    data[column],
                    color=colors[method],
                    marker="o",
                    markersize=3.2,
                    linewidth=1.8,
                    label=method_labels[method],
                )
            if row_idx == 0:
                ax.set_title(f"p_0={p0:.2f}", loc="left")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            if column in {
                "product_2_share",
                "product2_given_A_share",
                "A_choice_share",
            }:
                ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
                ax.set_ylim(-0.03, 1.03)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("N or T")
            prettify_axes(ax)
            if col_idx == len(p0_values) - 1:
                ax.text(
                    1.02,
                    0.5,
                    title,
                    transform=ax.transAxes,
                    rotation=270,
                    va="center",
                    ha="left",
                    fontsize=10,
                    fontweight="bold",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=len(method_order), frameon=False)
    fig.suptitle(
        "Extended method comparison with path-weighted diagnostics",
        x=0.01,
        ha="left",
    )
    save_figure(fig, outputs_dir / "method_comparison_dashboard_extended.png")


def plot_simulation_comparison(
    paths: pd.DataFrame,
    outputs_dir: Path,
    filename: str = "simulation_comparison.png",
) -> None:
    if paths.empty:
        return
    data = paths.copy()
    data["product_2_when_A_chosen"] = np.where(
        data["user_chose_A"] == 1,
        (data["action_if_A"] == 2).astype(float),
        np.nan,
    )
    time_summary = (
        data.groupby("t", observed=True)
        .agg(
            average_profit_to_date=("average_profit_to_date", "mean"),
            posterior_mean=("posterior_mean", "mean"),
            rho=("rho", "mean"),
            product_2_frequency=("product_2_when_A_chosen", "mean"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), sharex=True)
    ax_profit, ax_mean, ax_rho, ax_product = axes.ravel()
    ax_profit.plot(
        time_summary["t"],
        time_summary["average_profit_to_date"],
        color="#0f766e",
        linewidth=2.0,
    )
    ax_profit.set_title("Average profit to date", loc="left")
    ax_profit.set_ylabel("Average profit")
    prettify_axes(ax_profit)

    ax_mean.plot(
        time_summary["t"],
        time_summary["posterior_mean"],
        color="#2563eb",
        linewidth=2.0,
    )
    ax_mean.set_title("Posterior mean", loc="left")
    ax_mean.set_ylabel("E[theta | S,F]")
    ax_mean.set_ylim(-0.03, 1.03)
    prettify_axes(ax_mean)

    ax_rho.plot(
        time_summary["t"],
        time_summary["rho"],
        color="#7c3aed",
        linewidth=2.0,
    )
    ax_rho.set_title("Demand probability", loc="left")
    ax_rho.set_xlabel("Period")
    ax_rho.set_ylabel("rho(S,F)")
    ax_rho.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_rho.set_ylim(-0.03, 1.03)
    prettify_axes(ax_rho)

    ax_product.plot(
        time_summary["t"],
        time_summary["product_2_frequency"],
        color="#dc2626",
        linewidth=2.0,
    )
    ax_product.set_title("Product 2 frequency when A is chosen", loc="left")
    ax_product.set_xlabel("Period")
    ax_product.set_ylabel("Product 2 share")
    ax_product.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    ax_product.set_ylim(-0.03, 1.03)
    prettify_axes(ax_product)
    save_figure(fig, outputs_dir / filename)


def simulate_policy_paths(
    p0: float,
    params: ModelParams,
    n_rep: int,
    horizon: int,
    seed: int,
    method: str,
    n_or_t: int,
    stationary_solution: dict | None = None,
    state_space: StateSpace | None = None,
    finite_solution: dict | None = None,
    state_mapping: str = "posterior_mean",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate true Bayesian count paths under a stationary or finite policy."""
    if n_rep <= 0:
        return pd.DataFrame(), pd.DataFrame()
    if method != FINITE_HORIZON_METHOD and (stationary_solution is None or state_space is None):
        raise ValueError("stationary_solution and state_space are required.")
    if method == FINITE_HORIZON_METHOD and finite_solution is None:
        raise ValueError("finite_solution is required.")
    if state_mapping not in STATIONARY_MAPPING_RULES:
        raise ValueError(
            "state_mapping must be one of: "
            + ", ".join(STATIONARY_MAPPING_RULES)
        )

    rng = np.random.default_rng(seed)
    records = []
    sorted_means: np.ndarray | None = None
    sorted_indices: np.ndarray | None = None
    if state_space is not None and state_mapping == "closest_posterior_mean":
        sorted_means, sorted_indices = build_posterior_mean_mapper(state_space)
    for rep in range(n_rep):
        successes = 0
        failures = 0
        cumulative_profit = 0.0
        for t in range(1, horizon + 1):
            current_successes = successes
            current_failures = failures
            demand_prob = beta_ccdf_scalar(p0, successes, failures)
            posterior_mean_t = posterior_mean_scalar(successes, failures)

            if method == FINITE_HORIZON_METHOD:
                fh_state_space: StateSpace = finite_solution["state_space"]
                state_idx = int(fh_state_space.state_index[successes, failures])
                action = int(finite_solution["policy_by_time"][t, state_idx])
            else:
                state_idx = finite_state_index_for_true_counts(
                    successes,
                    failures,
                    state_space,
                    state_mapping,
                    sorted_means,
                    sorted_indices,
                )
                action = int(stationary_solution["policy_product"][state_idx])

            user_chose_A = rng.random() < demand_prob
            success_value = np.nan
            profit = 0.0
            if user_chose_A:
                success_probability = params.p2 if action == 2 else params.p1
                success_value = rng.random() < success_probability
                if success_value:
                    successes += 1
                else:
                    failures += 1
                profit = params.revenue - (params.c2 if action == 2 else params.c1)
                cumulative_profit += profit

            records.append(
                {
                    "method": method,
                    "p0": p0,
                    "N_or_T": n_or_t,
                    "rep": rep,
                    "t": t,
                    "true_S": current_successes,
                    "true_F": current_failures,
                    "posterior_mean": posterior_mean_t,
                    "rho": demand_prob,
                    "user_chose_A": int(user_chose_A),
                    "action_if_A": action,
                    "success": float(success_value) if user_chose_A else np.nan,
                    "profit": profit,
                    "cumulative_profit": cumulative_profit,
                    "average_profit_to_date": cumulative_profit / t,
                    "state_mapping": state_mapping
                    if method != FINITE_HORIZON_METHOD
                    else "exact",
                }
            )

    paths = pd.DataFrame.from_records(records)
    paths["product_2_when_A_chosen"] = np.where(
        paths["user_chose_A"] == 1,
        (paths["action_if_A"] == 2).astype(float),
        np.nan,
    )
    time_summary = (
        paths.groupby(["method", "p0", "N_or_T", "t"], observed=True)
        .agg(
            average_profit_to_date=("average_profit_to_date", "mean"),
            posterior_mean=("posterior_mean", "mean"),
            rho=("rho", "mean"),
            A_market_share=("user_chose_A", "mean"),
            product_2_frequency=("product_2_when_A_chosen", "mean"),
        )
        .reset_index()
    )
    return paths, time_summary


def summarize_replication_metrics(
    records: list[dict],
    method: str,
    p0: float,
    n_or_t: int,
    mapping_rule: str,
) -> dict:
    frame = pd.DataFrame.from_records(records)
    return {
        "method": method,
        "p0": p0,
        "N_or_T": n_or_t,
        "replication_count": len(frame),
        "mapping_rule": mapping_rule,
        "avg_realized_profit": frame["avg_realized_profit"].mean(),
        "std_realized_profit": frame["avg_realized_profit"].std(ddof=1),
        "A_choice_share": frame["A_choice_share"].mean(),
        "product2_calendar_share": frame["product2_calendar_share"].mean(),
        "product2_given_A_share": frame["product2_given_A_share"].mean(),
        "avg_posterior_mean": frame["avg_posterior_mean"].mean(),
        "avg_rho": frame["avg_rho"].mean(),
    }


def simulate_stationary_path_weighted_metrics(
    solution: dict,
    params: ModelParams,
    state_space: StateSpace,
    n_rep: int,
    horizon: int,
    seed: int,
    mapping_rule: str,
) -> tuple[dict, np.ndarray]:
    rng = np.random.default_rng(seed)
    p0 = float(solution["p0"])
    policy_product = solution["policy_product"]
    sorted_means: np.ndarray | None = None
    sorted_indices: np.ndarray | None = None
    if mapping_rule == "closest_posterior_mean":
        sorted_means, sorted_indices = build_posterior_mean_mapper(state_space)
    visit_counts = np.zeros(len(state_space.S), dtype=float)
    rep_records = []

    for rep in range(n_rep):
        successes = 0
        failures = 0
        profit_sum = 0.0
        chosen_count = 0
        product2_count = 0
        posterior_sum = 0.0
        rho_sum = 0.0

        for _ in range(horizon):
            posterior_mean_t = posterior_mean_scalar(successes, failures)
            demand_prob = beta_ccdf_scalar(p0, successes, failures)
            state_idx = finite_state_index_for_true_counts(
                successes,
                failures,
                state_space,
                mapping_rule,
                sorted_means,
                sorted_indices,
            )
            visit_counts[state_idx] += 1.0
            action = int(policy_product[state_idx])
            posterior_sum += posterior_mean_t
            rho_sum += demand_prob

            if rng.random() < demand_prob:
                chosen_count += 1
                product2_count += int(action == 2)
                success_probability = params.p2 if action == 2 else params.p1
                if rng.random() < success_probability:
                    successes += 1
                else:
                    failures += 1
                profit_sum += params.revenue - (params.c2 if action == 2 else params.c1)

        rep_records.append(
            {
                "rep": rep,
                "avg_realized_profit": profit_sum / horizon,
                "A_choice_share": chosen_count / horizon,
                "product2_calendar_share": product2_count / horizon,
                "product2_given_A_share": (
                    product2_count / chosen_count if chosen_count else np.nan
                ),
                "avg_posterior_mean": posterior_sum / horizon,
                "avg_rho": rho_sum / horizon,
            }
        )

    return (
        summarize_replication_metrics(
            rep_records,
            method=solution["method"],
            p0=p0,
            n_or_t=state_space.max_observations,
            mapping_rule=mapping_rule,
        ),
        visit_counts,
    )


def simulate_finite_horizon_path_weighted_metrics(
    solution: dict,
    params: ModelParams,
    n_rep: int,
    seed: int,
) -> tuple[dict, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    p0 = float(solution["p0"])
    horizon = int(solution["T"])
    state_space: StateSpace = solution["state_space"]
    policy_by_time = solution["policy_by_time"]
    rep_records = []
    by_tau = {
        tau: {
            "chosen": 0.0,
            "product2": 0.0,
            "profit": 0.0,
            "periods": 0.0,
        }
        for tau in range(1, horizon + 1)
    }

    for rep in range(n_rep):
        successes = 0
        failures = 0
        profit_sum = 0.0
        chosen_count = 0
        product2_count = 0
        posterior_sum = 0.0
        rho_sum = 0.0

        for t in range(1, horizon + 1):
            tau = horizon - t + 1
            posterior_mean_t = posterior_mean_scalar(successes, failures)
            demand_prob = beta_ccdf_scalar(p0, successes, failures)
            state_idx = int(state_space.state_index[successes, failures])
            action = int(policy_by_time[t, state_idx])
            posterior_sum += posterior_mean_t
            rho_sum += demand_prob
            by_tau[tau]["periods"] += 1.0

            profit = 0.0
            if rng.random() < demand_prob:
                chosen_count += 1
                product2_count += int(action == 2)
                by_tau[tau]["chosen"] += 1.0
                by_tau[tau]["product2"] += int(action == 2)
                success_probability = params.p2 if action == 2 else params.p1
                if rng.random() < success_probability:
                    successes += 1
                else:
                    failures += 1
                profit = params.revenue - (params.c2 if action == 2 else params.c1)
                profit_sum += profit
            by_tau[tau]["profit"] += profit

        rep_records.append(
            {
                "rep": rep,
                "avg_realized_profit": profit_sum / horizon,
                "A_choice_share": chosen_count / horizon,
                "product2_calendar_share": product2_count / horizon,
                "product2_given_A_share": (
                    product2_count / chosen_count if chosen_count else np.nan
                ),
                "avg_posterior_mean": posterior_sum / horizon,
                "avg_rho": rho_sum / horizon,
            }
        )

    by_tau_rows = []
    for tau, values in by_tau.items():
        periods = values["periods"]
        chosen = values["chosen"]
        by_tau_rows.append(
            {
                "p0": p0,
                "T": horizon,
                "tau": tau,
                "simulated_product2_calendar_share": values["product2"] / periods,
                "simulated_product2_given_A_share": (
                    values["product2"] / chosen if chosen else np.nan
                ),
                "simulated_A_choice_share": chosen / periods,
                "simulated_avg_profit": values["profit"] / periods,
            }
        )

    return (
        summarize_replication_metrics(
            rep_records,
            method=FINITE_HORIZON_METHOD,
            p0=p0,
            n_or_t=horizon,
            mapping_rule="exact",
        ),
        pd.DataFrame.from_records(by_tau_rows),
    )


def average_method_folder(method: str, n_value: int) -> str:
    return f"{method}_N={n_value}"


def finite_horizon_folder(horizon: int) -> str:
    return f"{FINITE_HORIZON_METHOD}_T={horizon}"


def run_average_reward_method_outputs(
    p0_grid: np.ndarray,
    params: ModelParams,
    method: str,
    outputs_dir: Path,
    args: argparse.Namespace,
    run_simulation: bool,
    write_legacy_names: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame, pd.DataFrame]:
    data_dir = outputs_dir / "data"
    plots_dir = outputs_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    state_space = make_state_space(params.max_observations, method=method)
    optimal_solutions = []
    convergence_frames = []
    diagnostics_frames = []
    simulation_path_frames = []
    simulation_time_frames = []
    summary_records = []

    for p0_idx, p0 in enumerate(p0_grid):
        p0 = float(p0)
        diagnostics_frames.append(validate_average_reward_transitions(p0, params, state_space))
        optimal_solution = solve_average_reward_policy(
            p0,
            params,
            state_space,
            fixed_product=None,
        )
        fixed_product1 = solve_average_reward_policy(
            p0,
            params,
            state_space,
            fixed_product=1,
        )
        fixed_product2 = solve_average_reward_policy(
            p0,
            params,
            state_space,
            fixed_product=2,
        )
        optimal_solutions.append(optimal_solution)
        convergence_frames.extend(
            [
                optimal_solution["convergence"],
                fixed_product1["convergence"],
                fixed_product2["convergence"],
            ]
        )

        if run_simulation:
            paths, time_summary = simulate_policy_paths(
                p0=p0,
                params=params,
                n_rep=args.n_rep,
                horizon=args.horizon,
                seed=args.seed + 10_000 * p0_idx + params.max_observations,
                method=method,
                n_or_t=params.max_observations,
                stationary_solution=optimal_solution,
                state_space=state_space,
                state_mapping=args.simulation_state_mapping,
            )
            simulation_path_frames.append(paths)
            simulation_time_frames.append(time_summary)
            simulation_reps = (
                paths.groupby("rep", observed=True)
                .agg(
                    A_market_share=("user_chose_A", "mean"),
                    product2_rate_when_A_chosen=("product_2_when_A_chosen", "mean"),
                    A_success_rate_when_chosen=("success", "mean"),
                    avg_profit_per_period=("profit", "mean"),
                    tail_avg_profit_per_period=("profit", "mean"),
                    final_posterior_mean=("posterior_mean", "last"),
                )
                .reset_index()
            )
        else:
            simulation_reps = pd.DataFrame(
                {
                    "rep": [0],
                    "A_market_share": [np.nan],
                    "product2_rate_when_A_chosen": [np.nan],
                    "A_success_rate_when_chosen": [np.nan],
                    "avg_profit_per_period": [np.nan],
                    "tail_avg_profit_per_period": [np.nan],
                    "final_posterior_mean": [np.nan],
                }
            )

        summary_records.append(
            summarize_solution(
                optimal_solution,
                fixed_product1,
                fixed_product2,
                params,
                state_space,
                simulation_reps,
            )
        )

    summary = pd.DataFrame.from_records(summary_records)
    policy_states = build_policy_state_table(optimal_solutions, state_space, params)
    convergence = pd.concat(convergence_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostics_frames, ignore_index=True)

    if write_legacy_names:
        summary.to_csv(data_dir / "average_reward_summary.csv", index=False)
        policy_states.to_csv(data_dir / "average_reward_policy_by_state.csv", index=False)
        convergence.to_csv(data_dir / "average_reward_policy_iteration.csv", index=False)
    else:
        summary.to_csv(data_dir / "summary.csv", index=False)
        policy_states.to_csv(data_dir / "policy_by_state.csv", index=False)
        convergence.to_csv(data_dir / "policy_iteration.csv", index=False)
    diagnostics.to_csv(data_dir / "diagnostics.csv", index=False)

    if simulation_path_frames:
        simulation_paths = pd.concat(simulation_path_frames, ignore_index=True)
        simulation_times = pd.concat(simulation_time_frames, ignore_index=True)
        simulation_paths.to_csv(data_dir / "simulation_paths.csv", index=False)
        simulation_times.to_csv(data_dir / "simulation_timeseries.csv", index=False)
        plot_simulation_comparison(simulation_paths, plots_dir)

    plot_p0_summary(summary, plots_dir)
    plot_policy_heatmaps(optimal_solutions, state_space, plots_dir)
    plot_policy_posterior_state_space(optimal_solutions, state_space, plots_dir)
    plot_bias_gap_posterior_state_space(optimal_solutions, state_space, params, plots_dir)
    plot_policy_by_posterior_mean(policy_states, plots_dir)
    plot_rho_by_posterior_mean(policy_states, plots_dir)
    plot_convergence(convergence, plots_dir)
    return summary, policy_states, optimal_solutions, convergence, diagnostics


def run_finite_horizon_outputs(
    p0_grid: np.ndarray,
    params: ModelParams,
    horizon: int,
    outputs_dir: Path,
    args: argparse.Namespace,
    run_simulation: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict], pd.DataFrame, pd.DataFrame]:
    data_dir = outputs_dir / "data"
    plots_dir = outputs_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    for pattern in (
        "p0_*/finite_horizon_policy_regions.png",
        "p0_*/finite_horizon_product2_usage.png",
    ):
        for old_path in plots_dir.glob(pattern):
            old_path.unlink()

    policy_frames = []
    summary_records = []
    usage_time_frames = []
    usage_mean_frames = []
    simulation_path_frames = []
    simulation_time_frames = []
    solutions = []

    for p0_idx, p0 in enumerate(p0_grid):
        solution = solve_finite_horizon_policy(float(p0), params, horizon)
        solutions.append(solution)
        policy_table = build_finite_horizon_policy_table(solution)
        policy_frames.append(policy_table)
        summary_records.append(summarize_finite_horizon_solution(solution, policy_table))
        usage_by_time, usage_by_mean = build_finite_horizon_usage_tables(policy_table)
        usage_time_frames.append(usage_by_time)
        usage_mean_frames.append(usage_by_mean)

        if run_simulation:
            p0_plots_dir = plots_dir / f"p0_{p0_filename_fragment(float(p0))}"
            p0_plots_dir.mkdir(parents=True, exist_ok=True)
            paths, time_summary = simulate_policy_paths(
                p0=float(p0),
                params=params,
                n_rep=args.n_rep,
                horizon=horizon,
                seed=args.seed + 100_000 + 10_000 * p0_idx + horizon,
                method=FINITE_HORIZON_METHOD,
                n_or_t=horizon,
                finite_solution=solution,
            )
            simulation_path_frames.append(paths)
            simulation_time_frames.append(time_summary)
            plot_simulation_comparison(
                paths,
                p0_plots_dir,
                filename="finite_horizon_simulation_comparison.png",
            )

    policy = pd.concat(policy_frames, ignore_index=True)
    summary = pd.DataFrame.from_records(summary_records)
    usage_by_time_all = pd.concat(usage_time_frames, ignore_index=True)
    usage_by_mean_all = pd.concat(usage_mean_frames, ignore_index=True)
    policy.to_csv(data_dir / "finite_horizon_policy.csv", index=False)
    summary.to_csv(data_dir / "finite_horizon_summary.csv", index=False)
    usage_by_time_all.to_csv(data_dir / "finite_horizon_product2_by_time.csv", index=False)
    usage_by_mean_all.to_csv(
        data_dir / "finite_horizon_product2_by_posterior_mean.csv",
        index=False,
    )
    plot_finite_horizon_mid_policy_regions_by_p0(solutions, plots_dir)
    plot_finite_horizon_policy_heatmaps_extended(solutions, plots_dir)
    plot_finite_horizon_usage_all_p0(usage_by_time_all, usage_by_mean_all, plots_dir)
    if simulation_path_frames:
        simulation_paths = pd.concat(simulation_path_frames, ignore_index=True)
        simulation_times = pd.concat(simulation_time_frames, ignore_index=True)
        simulation_paths.to_csv(data_dir / "simulation_paths.csv", index=False)
        simulation_times.to_csv(data_dir / "simulation_timeseries.csv", index=False)
    return summary, policy, solutions, usage_by_time_all, usage_by_mean_all


def build_method_comparison_rows(
    average_summaries: list[pd.DataFrame],
    finite_summaries: list[pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for summary in average_summaries:
        for _, row in summary.iterrows():
            rows.append(
                {
                    "method": row["method"],
                    "p0": row["p0"],
                    "N_or_T": int(row["N"]),
                    "eta_or_avg_value": row["eta"],
                    "initial_action": int(row["initial_best_response_product"]),
                    "product_2_share": row["share_states_product2"],
                    "notes": "finite average-reward approximation",
                }
            )
    for summary in finite_summaries:
        for _, row in summary.iterrows():
            rows.append(
                {
                    "method": FINITE_HORIZON_METHOD,
                    "p0": row["p0"],
                    "N_or_T": int(row["T"]),
                    "eta_or_avg_value": row["avg_value_T"],
                    "initial_action": int(row["initial_action"]),
                    "product_2_share": row["fraction_states_product_2"],
                    "notes": "exact Bayesian finite-horizon DP",
                }
            )
    return pd.DataFrame.from_records(rows)


def build_policy_agreement_summary(
    policy_frames: list[pd.DataFrame],
    visit_count_lookup: dict[tuple[str, float, int], np.ndarray],
) -> pd.DataFrame:
    combined = pd.concat(policy_frames, ignore_index=True)
    rows = []
    methods = [method for method in AVERAGE_REWARD_METHODS if method in set(combined["method"])]
    for (p0, n_value), group in combined.groupby(["p0", "N"], observed=True):
        for method_a, method_b in combinations(methods, 2):
            left = group[group["method"] == method_a][
                ["S", "F", "use_product_2"]
            ].rename(columns={"use_product_2": "use_a"})
            right = group[group["method"] == method_b][
                ["S", "F", "use_product_2"]
            ].rename(columns={"use_product_2": "use_b"})
            merged = left.merge(right, on=["S", "F"], how="inner")
            if merged.empty:
                continue
            agrees = (merged["use_a"].to_numpy() == merged["use_b"].to_numpy()).astype(float)
            key_a = (method_a, float(p0), int(n_value))
            key_b = (method_b, float(p0), int(n_value))
            weights = None
            if key_a in visit_count_lookup and key_b in visit_count_lookup:
                weights = visit_count_lookup[key_a] + visit_count_lookup[key_b]
                weights = weights[: len(agrees)]
            weighted_agreement = (
                float(np.average(agrees, weights=weights))
                if weights is not None and np.sum(weights) > 0.0
                else np.nan
            )
            rows.append(
                {
                    "p0": p0,
                    "N": int(n_value),
                    "method_a": method_a,
                    "method_b": method_b,
                    "agreement_rate_all_states": float(np.mean(agrees)),
                    "agreement_rate_reachable_states": float(np.mean(agrees)),
                    "agreement_rate_weighted_by_simulated_visits": weighted_agreement,
                }
            )
    return pd.DataFrame.from_records(rows)


def build_bias_threshold_diagnostic(
    policy_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    combined = pd.concat(policy_frames, ignore_index=True)
    rows = []
    for _, row in combined.iterrows():
        predicted_action = 2 if row["bias_gap_success_minus_failure"] >= row["threshold_rhs"] else 1
        rows.append(
            {
                "method": row["method"],
                "p0": row["p0"],
                "N": int(row["N"]),
                "S": int(row["S"]),
                "F": int(row["F"]),
                "rho": row["rho"],
                "posterior_mean": row["posterior_mean"],
                "Delta_h": row["bias_gap_success_minus_failure"],
                "threshold_rhs": row["threshold_rhs"],
                "action": int(row["action"]),
                "predicted_action_from_threshold": predicted_action,
                "threshold_matches_policy": int(predicted_action == int(row["action"])),
                "boundary_state": int(row["S"] + row["F"] == row["N"]),
            }
        )
    return pd.DataFrame.from_records(rows)


def expected_share_after_outcome(
    state_idx: int,
    outcome: str,
    state_space: StateSpace,
) -> float:
    if state_space.max_observations <= 0:
        return np.nan
    current_share = state_space.S[state_idx] / state_space.max_observations
    if outcome == "success":
        return float(
            state_space.success_probability[state_idx]
            * state_space.S[state_space.success_index[state_idx]]
            / state_space.max_observations
            + state_space.success_stay_probability[state_idx] * current_share
        )
    if outcome == "failure":
        return float(
            state_space.failure_probability[state_idx]
            * state_space.S[state_space.failure_index[state_idx]]
            / state_space.max_observations
            + state_space.failure_stay_probability[state_idx] * current_share
        )
    raise ValueError("outcome must be 'success' or 'failure'.")


def build_boundary_drift_diagnostic(
    p0_grid: np.ndarray,
    params_by_method_n: dict[tuple[str, int], ModelParams],
) -> pd.DataFrame:
    rows = []
    for method in AVERAGE_REWARD_METHODS:
        n_values = sorted(n for candidate_method, n in params_by_method_n if candidate_method == method)
        for n_value in n_values:
            params = params_by_method_n[(method, n_value)]
            state_space = make_state_space(n_value, method=method)
            boundary_idx = np.flatnonzero(state_space.total == n_value)
            for p0 in p0_grid:
                rho = beta_ccdf(float(p0), state_space.S, state_space.F)
                for state_idx in boundary_idx:
                    current_share = state_space.S[state_idx] / n_value
                    share_success = expected_share_after_outcome(state_idx, "success", state_space)
                    share_failure = expected_share_after_outcome(state_idx, "failure", state_space)
                    for action in (1, 2):
                        success_probability = params.p2 if action == 2 else params.p1
                        expected_next_share = (
                            (1.0 - rho[state_idx]) * current_share
                            + rho[state_idx]
                            * (
                                success_probability * share_success
                                + (1.0 - success_probability) * share_failure
                            )
                        )
                        rows.append(
                            {
                                "method": method,
                                "p0": float(p0),
                                "N": n_value,
                                "S": int(state_space.S[state_idx]),
                                "F": int(state_space.F[state_idx]),
                                "action": action,
                                "current_share": current_share,
                                "expected_next_share": expected_next_share,
                                "drift": expected_next_share - current_share,
                                "expected_next_share_if_success": share_success,
                                "drift_if_success": share_success - current_share,
                                "expected_next_share_if_failure": share_failure,
                                "drift_if_failure": share_failure - current_share,
                            }
                        )
    return pd.DataFrame.from_records(rows)


def plot_average_method_region_comparison(
    policy_frames: list[pd.DataFrame],
    outputs_dir: Path,
) -> None:
    if len(policy_frames) < 2:
        return

    combined = pd.concat(policy_frames, ignore_index=True)
    methods = sorted(combined["method"].unique())
    if not set(AVERAGE_REWARD_METHODS).issubset(methods):
        return
    cmap = ListedColormap(
        [
            "#f2c94c",
            "#94a3b8",
            "#60a5fa",
            "#818cf8",
            "#f472b6",
            "#0f766e",
            "#f97316",
            "#111827",
        ]
    )
    norm = BoundaryNorm(np.arange(-0.5, 8.5, 1.0), cmap.N)

    for n_value in sorted(combined["N"].unique()):
        n_data = combined[combined["N"] == n_value]
        p0_values = np.array(sorted(n_data["p0"].unique()))
        n_panels = len(p0_values)
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        max_obs = int(n_value)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.8 * ncols, 4.3 * nrows),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        axes = np.atleast_1d(axes).ravel()
        image = None

        for ax, p0 in zip(axes, p0_values, strict=False):
            group = n_data[np.isclose(n_data["p0"], p0)]
            pivot = group.pivot_table(
                index=["S", "F"],
                columns="method",
                values="use_product_2",
                aggfunc="first",
            ).reset_index()
            if not set(AVERAGE_REWARD_METHODS).issubset(pivot.columns):
                ax.set_visible(False)
                continue
            matrix = np.full((max_obs + 1, max_obs + 1), np.nan)
            code = (
                pivot["rolling_window"].astype(int)
                + 2 * pivot["projection"].astype(int)
                + 4 * pivot["stochastic_projection"].astype(int)
            )
            matrix[pivot["F"].astype(int), pivot["S"].astype(int)] = code
            image = ax.imshow(
                matrix,
                origin="lower",
                extent=[-0.5, max_obs + 0.5, -0.5, max_obs + 0.5],
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                aspect="equal",
            )
            ax.set_title(f"p_0={p0:.2f}", loc="left")
            ax.set_xlabel("Successes S")
            ax.set_ylabel("Failures F")
            prettify_axes(ax, grid_axis="both")

        for ax in axes[n_panels:]:
            ax.set_visible(False)
        cbar = fig.colorbar(
            image,
            ax=axes[:n_panels],
            ticks=list(range(8)),
            shrink=0.82,
        )
        cbar.ax.set_yticklabels(
            [
                "neither",
                "rolling",
                "deterministic",
                "rolling+deterministic",
                "stochastic",
                "rolling+stochastic",
                "deterministic+stochastic",
                "all three",
            ]
        )
        fig.suptitle(
            f"Product 2 regions across p_0, N={max_obs}",
            x=0.01,
            ha="left",
        )
        save_figure(
            fig,
            outputs_dir / f"product2_region_comparison_extended_N={max_obs}.png",
        )


def plot_bias_threshold_diagnostics(
    diagnostic: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    if diagnostic.empty:
        return
    max_n = int(diagnostic["N"].max())
    data = diagnostic[diagnostic["N"] == max_n].copy()
    method_order = [method for method in AVERAGE_REWARD_METHODS if method in set(data["method"])]
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, data["p0"].nunique()))
    p0_values = np.array(sorted(data["p0"].unique()))
    color_by_p0 = {p0: color for p0, color in zip(p0_values, colors, strict=True)}

    fig, axes = plt.subplots(
        len(method_order),
        2,
        figsize=(12.2, 3.6 * len(method_order)),
        sharex=False,
        constrained_layout=True,
        squeeze=False,
    )
    axes = np.atleast_2d(axes)
    for row_idx, method in enumerate(method_order):
        method_data = data[data["method"] == method]
        for p0 in p0_values:
            subset = method_data[np.isclose(method_data["p0"], p0)]
            if subset.empty:
                continue
            color = color_by_p0[p0]
            label = f"p_0={p0:.2f}"
            axes[row_idx, 0].scatter(
                subset["posterior_mean"],
                subset["Delta_h"],
                color=color,
                s=5,
                alpha=0.45,
                linewidths=0.0,
                rasterized=True,
                label=label,
            )
            axes[row_idx, 1].scatter(
                subset["rho"],
                subset["Delta_h"],
                color=color,
                s=5,
                alpha=0.45,
                linewidths=0.0,
                rasterized=True,
                label=label,
            )
        threshold = method_data["threshold_rhs"].iloc[0]
        for ax in axes[row_idx]:
            ax.axhline(threshold, color="#111827", linewidth=1.0, linestyle="--")
            prettify_axes(ax)
        axes[row_idx, 0].set_title(f"{method}: Delta_h vs posterior mean", loc="left")
        axes[row_idx, 0].set_xlabel("Posterior mean")
        axes[row_idx, 0].set_ylabel("Delta_h")
        axes[row_idx, 1].set_title(f"{method}: Delta_h vs rho", loc="left")
        axes[row_idx, 1].set_xlabel("rho(S,F)")
        axes[row_idx, 1].set_ylabel("Delta_h")
        axes[row_idx, 1].set_xlim(-0.03, 1.03)
    axes[0, 0].legend(ncols=min(5, len(p0_values)))
    fig.suptitle(
        f"Bias-threshold diagnostic at N={max_n}",
        x=0.01,
        ha="left",
    )
    save_figure(fig, outputs_dir / "bias_threshold_diagnostic_extended.png")

    for n_value in sorted(diagnostic["N"].unique()):
        n_data = diagnostic[diagnostic["N"] == n_value]
        p0_values = np.array(sorted(n_data["p0"].unique()))
        nrows = len(method_order)
        ncols = len(p0_values)
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(3.2 * ncols, 3.0 * nrows),
            sharex=True,
            sharey=True,
            constrained_layout=True,
            squeeze=False,
        )
        axes = np.atleast_2d(axes)
        cmap = ListedColormap(["#f2c94c", "#0f766e", "#dc2626"])
        cmap.set_bad("#e5e7eb")
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
        image = None
        for row_idx, method in enumerate(method_order):
            for col_idx, p0 in enumerate(p0_values):
                subset = n_data[
                    (n_data["method"] == method)
                    & np.isclose(n_data["p0"], p0)
                ]
                ax = axes[row_idx, col_idx]
                matrix = np.full((int(n_value) + 1, int(n_value) + 1), np.nan)
                code = np.where(
                    subset["threshold_matches_policy"].to_numpy() == 1,
                    subset["predicted_action_from_threshold"].to_numpy() == 2,
                    2,
                ).astype(float)
                matrix[subset["F"].astype(int), subset["S"].astype(int)] = code
                image = ax.imshow(
                    matrix,
                    origin="lower",
                    extent=[-0.5, n_value + 0.5, -0.5, n_value + 0.5],
                    cmap=cmap,
                    norm=norm,
                    interpolation="nearest",
                    aspect="equal",
                )
                if row_idx == 0:
                    ax.set_title(f"p_0={p0:.2f}", loc="left")
                if col_idx == 0:
                    ax.set_ylabel(f"{method}\nFailures F")
                if row_idx == nrows - 1:
                    ax.set_xlabel("Successes S")
                prettify_axes(ax, grid_axis="both")
        cbar = fig.colorbar(image, ax=axes.ravel(), ticks=[0, 1, 2], shrink=0.82)
        cbar.ax.set_yticklabels(["Product 1", "Product 2", "mismatch"])
        fig.suptitle(
            f"Bias-threshold regions and mismatches, N={int(n_value)}",
            x=0.01,
            ha="left",
        )
        save_figure(
            fig,
            outputs_dir / f"bias_threshold_regions_extended_N={int(n_value)}.png",
        )


def plot_boundary_drift_diagnostics(
    diagnostic: pd.DataFrame,
    outputs_dir: Path,
) -> None:
    if diagnostic.empty:
        return
    method_order = [method for method in AVERAGE_REWARD_METHODS if method in set(diagnostic["method"])]
    metrics = [
        ("drift", "Total expected drift"),
        ("drift_if_success", "Drift after success event"),
        ("drift_if_failure", "Drift after failure event"),
    ]
    p0_values = np.array(sorted(diagnostic["p0"].unique()))
    colors = plt.cm.viridis(np.linspace(0.08, 0.88, len(p0_values)))
    color_by_p0 = {p0: color for p0, color in zip(p0_values, colors, strict=True)}

    for n_value in sorted(diagnostic["N"].unique()):
        n_data = diagnostic[diagnostic["N"] == n_value]
        fig, axes = plt.subplots(
            len(method_order),
            len(metrics),
            figsize=(4.5 * len(metrics), 3.2 * len(method_order)),
            sharex=True,
            constrained_layout=True,
            squeeze=False,
        )
        axes = np.atleast_2d(axes)
        for row_idx, method in enumerate(method_order):
            method_data = n_data[n_data["method"] == method]
            for col_idx, (column, title) in enumerate(metrics):
                ax = axes[row_idx, col_idx]
                plot_data = method_data
                if column != "drift":
                    plot_data = plot_data[plot_data["action"] == 1]
                for p0 in p0_values:
                    for action, linestyle in ((1, "-"), (2, "--")):
                        subset = plot_data[np.isclose(plot_data["p0"], p0)]
                        if column == "drift":
                            subset = subset[subset["action"] == action]
                        elif action == 2:
                            continue
                        if subset.empty:
                            continue
                        subset = subset.sort_values("current_share")
                        label = f"p_0={p0:.2f}" if action == 1 else None
                        ax.plot(
                            subset["current_share"],
                            subset[column],
                            color=color_by_p0[p0],
                            linestyle=linestyle,
                            linewidth=1.7,
                            alpha=0.9,
                            label=label,
                        )
                ax.axhline(0.0, color="#111827", linewidth=0.9, linestyle=":")
                if row_idx == 0:
                    ax.set_title(title, loc="left")
                if col_idx == 0:
                    ax.set_ylabel(f"{method}\nDrift")
                if row_idx == len(method_order) - 1:
                    ax.set_xlabel("Current success share S/N")
                prettify_axes(ax)
        axes[0, 0].legend(ncols=min(5, len(p0_values)))
        fig.suptitle(
            f"Boundary drift diagnostic, N={int(n_value)}",
            x=0.01,
            ha="left",
        )
        save_figure(
            fig,
            outputs_dir / f"boundary_drift_diagnostic_extended_N={int(n_value)}.png",
        )


def build_finite_horizon_policy_time_summary(solution: dict) -> pd.DataFrame:
    state_space: StateSpace = solution["state_space"]
    horizon = int(solution["T"])
    rows = []
    for t in range(1, horizon + 1):
        tau = horizon - t + 1
        idx = np.flatnonzero(state_space.total <= t - 1)
        use_product2 = solution["policy_by_time"][t, idx] == 2
        rows.append(
            {
                "p0": float(solution["p0"]),
                "T": horizon,
                "tau": tau,
                "share_states_product2_unweighted": float(np.mean(use_product2)),
                "share_reachable_states_product2": float(np.mean(use_product2)),
            }
        )
    return pd.DataFrame.from_records(rows)


def run_path_weighted_diagnostics(
    average_solution_records: list[dict],
    finite_solution_records: list[dict],
    params: ModelParams,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[str, float, int], np.ndarray]]:
    if args.path_diagnostic_rep <= 0:
        raise ValueError("path-diagnostic-rep must be positive.")
    path_rows = []
    finite_time_frames = []
    visit_count_lookup: dict[tuple[str, float, int], np.ndarray] = {}
    seed_counter = 0

    for record in average_solution_records:
        method = record["method"]
        n_value = int(record["N"])
        method_params: ModelParams = record["params"]
        state_space = make_state_space(n_value, method=method)
        for solution in record["solutions"]:
            summary, visit_counts = simulate_stationary_path_weighted_metrics(
                solution=solution,
                params=method_params,
                state_space=state_space,
                n_rep=args.path_diagnostic_rep,
                horizon=args.horizon,
                seed=args.seed + 1_000_000 + seed_counter,
                mapping_rule=args.simulation_state_mapping,
            )
            seed_counter += 1
            path_rows.append(summary)
            visit_count_lookup[(method, float(solution["p0"]), n_value)] = visit_counts

    for solution in finite_solution_records:
        summary, simulated_by_tau = simulate_finite_horizon_path_weighted_metrics(
            solution=solution,
            params=params,
            n_rep=args.path_diagnostic_rep,
            seed=args.seed + 2_000_000 + seed_counter,
        )
        seed_counter += 1
        path_rows.append(summary)
        policy_by_tau = build_finite_horizon_policy_time_summary(solution)
        finite_time_frames.append(
            policy_by_tau.merge(
                simulated_by_tau,
                on=["p0", "T", "tau"],
                how="left",
            )
        )

    path_weighted = pd.DataFrame.from_records(path_rows)
    finite_by_time = (
        pd.concat(finite_time_frames, ignore_index=True)
        if finite_time_frames
        else pd.DataFrame()
    )
    return path_weighted, finite_by_time, visit_count_lookup


def write_diagnostic_report(
    output_path: Path,
    comparison: pd.DataFrame,
    path_weighted: pd.DataFrame,
    policy_agreement: pd.DataFrame,
    bias_threshold: pd.DataFrame,
    boundary_drift: pd.DataFrame,
) -> None:
    lines = [
        "# Average-Cost Robustness Diagnostics",
        "",
        "This report summarizes the extended diagnostics generated by `average_cost/main.py --run-robustness`.",
        "",
        "## Main Outputs",
        "",
        "- `data/path_weighted_comparison.csv`",
        "- `data/finite_horizon_by_time_remaining.csv`",
        "- `data/policy_agreement_summary.csv`",
        "- `data/bias_threshold_diagnostic.csv`",
        "- `data/boundary_drift_diagnostic.csv`",
        "- `plots/method_comparison_dashboard_extended.png`",
        "- `plots/product2_region_comparison_extended_N=25.png` and analogous N values",
        "- `plots/bias_threshold_diagnostic_extended.png`",
        "- `plots/boundary_drift_diagnostic_extended_N=25.png` and analogous N values",
        "",
        "## Short Answers",
        "",
    ]

    finite = comparison[comparison["method"] == FINITE_HORIZON_METHOD]
    rolling = comparison[comparison["method"] == "rolling_window"]
    projection = comparison[comparison["method"] == "projection"]
    stochastic = comparison[comparison["method"] == "stochastic_projection"]
    if not finite.empty and not rolling.empty:
        merged_roll = finite.merge(
            rolling,
            on=["p0", "N_or_T"],
            suffixes=("_finite", "_rolling"),
        )
        merged_proj = finite.merge(
            projection,
            on=["p0", "N_or_T"],
            suffixes=("_finite", "_projection"),
        )
        roll_gap = (
            (merged_roll["eta_or_avg_value_finite"] - merged_roll["eta_or_avg_value_rolling"])
            .abs()
            .mean()
            if not merged_roll.empty
            else np.nan
        )
        proj_gap = (
            (merged_proj["eta_or_avg_value_finite"] - merged_proj["eta_or_avg_value_projection"])
            .abs()
            .mean()
            if not merged_proj.empty
            else np.nan
        )
        closer = "rolling_window" if roll_gap <= proj_gap else "deterministic_projection"
        lines.append(
            f"1. Finite horizon is closer in value to `{closer}` on average "
            f"(mean absolute gaps: rolling={roll_gap:.4f}, projection={proj_gap:.4f})."
        )
    if not rolling.empty and not projection.empty and not stochastic.empty:
        rp = rolling.merge(projection, on=["p0", "N_or_T"], suffixes=("_rolling", "_projection"))
        rs = rolling.merge(stochastic, on=["p0", "N_or_T"], suffixes=("_rolling", "_stochastic"))
        proj_gap = (rp["eta_or_avg_value_rolling"] - rp["eta_or_avg_value_projection"]).abs().mean()
        stoch_gap = (rs["eta_or_avg_value_rolling"] - rs["eta_or_avg_value_stochastic"]).abs().mean()
        lines.append(
            f"2. Stochastic projection is {'closer' if stoch_gap <= proj_gap else 'not closer'} "
            f"to rolling_window in value than deterministic projection "
            f"(mean gaps: stochastic={stoch_gap:.4f}, deterministic={proj_gap:.4f})."
        )
    if not path_weighted.empty:
        unweighted = comparison.merge(
            path_weighted,
            on=["method", "p0", "N_or_T"],
            how="inner",
            suffixes=("_state", "_path"),
        )
        gap = (unweighted["product_2_share"] - unweighted["product2_given_A_share"]).abs().mean()
        lines.append(
            f"3. Raw unweighted product-2 state shares differ from path-weighted usage "
            f"by {gap:.4f} on average, so path weighting is economically important."
        )
        lines.append(
            f"4. Mean path-weighted product-2 use conditional on A is "
            f"{path_weighted['product2_given_A_share'].mean():.4f}; "
            f"mean calendar-time product-2 use is {path_weighted['product2_calendar_share'].mean():.4f}."
        )
    if not bias_threshold.empty:
        mismatch_rate = 1.0 - bias_threshold["threshold_matches_policy"].mean()
        lines.append(
            f"5. The bias-threshold rule matches the computed average-reward policy "
            f"with mismatch rate {mismatch_rate:.6f}."
        )
    if not boundary_drift.empty:
        det = boundary_drift[boundary_drift["method"] == "projection"]
        stoch = boundary_drift[boundary_drift["method"] == "stochastic_projection"]
        high_det = det[det["current_share"] >= 0.8]["drift"].mean()
        high_stoch = stoch[stoch["current_share"] >= 0.8]["drift"].mean()
        lines.append(
            "6. Boundary drift at high success shares can be inspected in "
            "`boundary_drift_diagnostic.csv`; mean high-share drift is "
            f"{high_det:.6f} for deterministic projection and {high_stoch:.6f} "
            "for stochastic projection."
        )
    if not policy_agreement.empty:
        weighted = policy_agreement.dropna(subset=["agreement_rate_weighted_by_simulated_visits"])
        if not weighted.empty:
            best = weighted.sort_values(
                "agreement_rate_weighted_by_simulated_visits",
                ascending=False,
            ).iloc[0]
            lines.append(
                "7. The highest simulated-visit-weighted policy agreement is "
                f"{best['agreement_rate_weighted_by_simulated_visits']:.4f} for "
                f"{best['method_a']} vs {best['method_b']} at p0={best['p0']}, N={int(best['N'])}."
            )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `projection` denotes the deterministic projection method.",
            "- `stochastic_projection` preserves projected success share in expectation at the boundary.",
            "- Stationary-policy path simulations use true growing Bayesian counts and map them back to the finite grid using the configured mapping rule.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_robustness_experiment(args: argparse.Namespace, params: ModelParams) -> None:
    project_dir = Path(__file__).resolve().parent
    base_outputs_dir = Path(args.outputs_dir) if args.outputs_dir else project_dir / "outputs"
    outputs_dir = (
        base_outputs_dir
        if base_outputs_dir.name == "robustness"
        else base_outputs_dir / "robustness"
    )
    data_root = outputs_dir / "data"
    plots_root = outputs_dir / "plots"
    data_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    p0_grid = parse_float_grid(args.robustness_p0_grid)
    if p0_grid.size == 0:
        p0_grid = np.array([args.diagnostic_p0], dtype=float)
    n_grid = parse_int_grid(args.robustness_n_grid)
    t_grid = parse_int_grid(args.finite_horizon_t_grid)
    if args.robustness_demand_floor <= 0.0:
        raise ValueError("robustness-demand-floor must be positive.")

    average_summaries = []
    average_policy_frames = []
    average_solution_records = []
    params_by_method_n: dict[tuple[str, int], ModelParams] = {}
    finite_summaries = []
    finite_solution_records = []
    run_simulation = not args.skip_simulation

    for method in AVERAGE_REWARD_METHODS:
        for n_value in n_grid:
            method_params = ModelParams(
                p1=params.p1,
                p2=params.p2,
                c1=params.c1,
                c2=params.c2,
                revenue=params.revenue,
                max_observations=int(n_value),
                max_iter=params.max_iter,
                tol=params.tol,
                demand_floor=args.robustness_demand_floor,
            )
            params_by_method_n[(method, int(n_value))] = method_params
            label = average_method_folder(method, int(n_value))
            print(f"Solving {label}...")
            summary, policy_states, solutions, _, _ = run_average_reward_method_outputs(
                p0_grid=p0_grid,
                params=method_params,
                method=method,
                outputs_dir=outputs_dir / label,
                args=args,
                run_simulation=run_simulation,
                write_legacy_names=False,
            )
            average_summaries.append(summary)
            average_policy_frames.append(policy_states)
            average_solution_records.append(
                {
                    "method": method,
                    "N": int(n_value),
                    "params": method_params,
                    "solutions": solutions,
                }
            )

    for horizon in t_grid:
        label = finite_horizon_folder(int(horizon))
        print(f"Solving {label}...")
        summary, _, solutions, _, _ = run_finite_horizon_outputs(
            p0_grid=p0_grid,
            params=params,
            horizon=int(horizon),
            outputs_dir=outputs_dir / label,
            args=args,
            run_simulation=run_simulation,
        )
        finite_summaries.append(summary)
        finite_solution_records.extend(solutions)

    comparison = build_method_comparison_rows(average_summaries, finite_summaries)
    comparison.to_csv(data_root / "method_comparison.csv", index=False)
    print("Running path-weighted diagnostics...")
    path_weighted, finite_by_time, visit_count_lookup = run_path_weighted_diagnostics(
        average_solution_records=average_solution_records,
        finite_solution_records=finite_solution_records,
        params=params,
        args=args,
    )
    path_weighted.to_csv(data_root / "path_weighted_comparison.csv", index=False)
    finite_by_time.to_csv(
        data_root / "finite_horizon_by_time_remaining.csv",
        index=False,
    )
    policy_agreement = build_policy_agreement_summary(
        average_policy_frames,
        visit_count_lookup,
    )
    policy_agreement.to_csv(data_root / "policy_agreement_summary.csv", index=False)
    bias_threshold = build_bias_threshold_diagnostic(average_policy_frames)
    bias_threshold.to_csv(data_root / "bias_threshold_diagnostic.csv", index=False)
    boundary_drift = build_boundary_drift_diagnostic(
        p0_grid=p0_grid,
        params_by_method_n=params_by_method_n,
    )
    boundary_drift.to_csv(data_root / "boundary_drift_diagnostic.csv", index=False)

    plot_average_method_region_comparison(average_policy_frames, plots_root)
    plot_method_comparison_dashboard_extended(comparison, path_weighted, plots_root)
    plot_finite_horizon_value_curve(comparison, plots_root)
    for horizon in sorted(finite_by_time["T"].unique()) if not finite_by_time.empty else []:
        target_dir = outputs_dir / finite_horizon_folder(int(horizon)) / "plots"
        target_dir.mkdir(parents=True, exist_ok=True)
        plot_finite_horizon_time_remaining_extended(
            finite_by_time[finite_by_time["T"] == horizon],
            target_dir,
        )
    plot_bias_threshold_diagnostics(bias_threshold, plots_root)
    plot_boundary_drift_diagnostics(boundary_drift, plots_root)
    write_diagnostic_report(
        output_path=outputs_dir / "diagnostic_report.md",
        comparison=comparison,
        path_weighted=path_weighted,
        policy_agreement=policy_agreement,
        bias_threshold=bias_threshold,
        boundary_drift=boundary_drift,
    )

    print("\nMethod comparison")
    print(
        comparison[
            [
                "method",
                "p0",
                "N_or_T",
                "eta_or_avg_value",
                "initial_action",
                "product_2_share",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )
    value_summary = comparison.pivot_table(
        index=["p0", "N_or_T"],
        columns="method",
        values="eta_or_avg_value",
        aggfunc="first",
    ).reset_index()
    closer_stochastic = np.nan
    if {"rolling_window", "projection", "stochastic_projection"}.issubset(
        value_summary.columns
    ):
        det_gap = (
            value_summary["rolling_window"] - value_summary["projection"]
        ).abs().mean()
        stoch_gap = (
            value_summary["rolling_window"] - value_summary["stochastic_projection"]
        ).abs().mean()
        closer_stochastic = stoch_gap <= det_gap
    finite_closer = "not computed"
    if {"rolling_window", "projection", FINITE_HORIZON_METHOD}.issubset(
        value_summary.columns
    ):
        roll_gap = (
            value_summary[FINITE_HORIZON_METHOD] - value_summary["rolling_window"]
        ).abs().mean()
        proj_gap = (
            value_summary[FINITE_HORIZON_METHOD] - value_summary["projection"]
        ).abs().mean()
        finite_closer = "rolling_window" if roll_gap <= proj_gap else "projection"

    print("\nDiagnostics completed")
    print(f"  methods: {', '.join([*AVERAGE_REWARD_METHODS, FINITE_HORIZON_METHOD])}")
    print(f"  p0 values: {', '.join(f'{value:.2f}' for value in p0_grid)}")
    print(f"  N values: {', '.join(str(value) for value in n_grid)}")
    print(f"  T values: {', '.join(str(value) for value in t_grid)}")
    if not np.isnan(closer_stochastic):
        print(
            "  stochastic_projection closer to rolling_window than deterministic projection: "
            f"{bool(closer_stochastic)}"
        )
    print(f"  finite_horizon value closer to: {finite_closer}")
    print(f"\nSaved method comparison to: {data_root / 'method_comparison.csv'}")
    print(f"Saved path-weighted diagnostics to: {data_root / 'path_weighted_comparison.csv'}")
    print(f"Saved diagnostic report to: {outputs_dir / 'diagnostic_report.md'}")


def parse_p0_grid(args: argparse.Namespace) -> np.ndarray:
    if args.p0_grid:
        values = np.array([float(item.strip()) for item in args.p0_grid.split(",")])
    else:
        values = np.linspace(args.p0_min, args.p0_max, args.p0_count)

    values = np.append(values, args.diagnostic_p0)
    values = np.append(values, parse_float_grid(args.sample_path_p0_grid))
    values = np.unique(np.round(values, 10))
    if np.any(values <= 0.0) or np.any(values >= 1.0):
        raise ValueError("All p_0 values must be strictly between 0 and 1.")
    return values


def parse_float_grid(value: str) -> np.ndarray:
    if not value.strip():
        return np.array([], dtype=float)
    return np.array([float(item.strip()) for item in value.split(",") if item.strip()])


def parse_int_grid(value: str) -> list[int]:
    if not value.strip():
        return []
    grid = [int(item.strip()) for item in value.split(",") if item.strip()]
    if any(item <= 0 for item in grid):
        raise ValueError("Integer grids must contain only positive values.")
    return grid


def parse_args() -> argparse.Namespace:
    defaults = ModelParams()
    parser = argparse.ArgumentParser(
        description=(
            "Solve and simulate Seller A's average-per-stage best response "
            "in the Bernoulli/Beta reputation model."
        )
    )
    parser.add_argument("--p0-grid", default=None, help="Comma-separated p_0 values.")
    parser.add_argument("--p0-min", type=float, default=0.10)
    parser.add_argument("--p0-max", type=float, default=0.90)
    parser.add_argument("--p0-count", type=int, default=9)
    parser.add_argument("--p1", type=float, default=defaults.p1)
    parser.add_argument("--p2", type=float, default=defaults.p2)
    parser.add_argument("--c1", type=float, default=defaults.c1)
    parser.add_argument("--c2", type=float, default=defaults.c2)
    parser.add_argument("--revenue", "-R", type=float, default=defaults.revenue)
    parser.add_argument("--max-observations", type=int, default=defaults.max_observations)
    parser.add_argument("--max-iter", type=int, default=defaults.max_iter)
    parser.add_argument("--tol", type=float, default=defaults.tol)
    parser.add_argument("--demand-floor", type=float, default=defaults.demand_floor)
    parser.add_argument(
        "--method",
        choices=AVERAGE_REWARD_METHODS,
        default="rolling_window",
        help="Finite average-reward approximation used in the default study.",
    )
    parser.add_argument(
        "--run-robustness",
        action="store_true",
        help="Run rolling-window, projection, and finite-horizon robustness experiments.",
    )
    parser.add_argument(
        "--robustness-n-grid",
        default=DEFAULT_ROBUSTNESS_GRID,
        help="Comma-separated N values for rolling_window and projection.",
    )
    parser.add_argument(
        "--finite-horizon-t-grid",
        default=DEFAULT_ROBUSTNESS_GRID,
        help="Comma-separated T values for finite_horizon.",
    )
    parser.add_argument(
        "--robustness-p0-grid",
        default=DEFAULT_ROBUSTNESS_P0_GRID,
        help="Comma-separated p_0 values for robustness experiments.",
    )
    parser.add_argument(
        "--robustness-demand-floor",
        type=float,
        default=1e-8,
        help=(
            "Solver-only rho floor for robustness average-reward runs. Reported "
            "rho values and simulations still use the unfloored Beta tail."
        ),
    )
    parser.add_argument(
        "--simulation-state-mapping",
        choices=STATIONARY_MAPPING_RULES,
        default="closest_posterior_mean",
        help="Mapping from true Bayesian counts to a finite stationary policy.",
    )
    parser.add_argument(
        "--path-diagnostic-rep",
        type=int,
        default=1000,
        help="Replication count for path-weighted robustness diagnostics.",
    )
    parser.add_argument("--T", "--horizon", dest="horizon", type=int, default=250)
    parser.add_argument("--n-rep", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260617)
    parser.add_argument(
        "--sample-path-p0-grid",
        default="0.5,0.7",
        help="Comma-separated p_0 values for individual sample-path plots.",
    )
    parser.add_argument(
        "--sample-path-count",
        type=int,
        default=5,
        help="Number of individual paths to show in each sample-path plot.",
    )
    parser.add_argument(
        "--outputs-dir",
        default=None,
        help="Defaults to average_cost/outputs next to this script.",
    )
    parser.add_argument("--skip-simulation", action="store_true")
    parser.add_argument(
        "--initial-belief-mean",
        type=float,
        default=None,
        help=(
            "Optional initial posterior mean for Seller A. The script uses the "
            "closest integer Beta(1+S,1+F) state."
        ),
    )
    parser.add_argument(
        "--diagnostic-p0",
        type=float,
        default=0.50,
        help="p_0 value automatically included in the p_0 grid.",
    )
    return parser.parse_args()


def validate_params(params: ModelParams) -> None:
    if not 0.0 < params.p1 < params.p2 <= 1.0:
        raise ValueError("Require 0 < p1 < p2 <= 1.")
    if not params.c1 < params.c2:
        raise ValueError("Require c1 < c2.")
    if params.max_observations < 2:
        raise ValueError("max-observations must be at least 2.")
    if params.max_iter < 1:
        raise ValueError("max-iter must be positive.")
    if params.tol <= 0.0:
        raise ValueError("tol must be positive.")
    if params.demand_floor <= 0.0:
        raise ValueError("demand-floor must be positive.")


def main() -> None:
    args = parse_args()
    if args.p0_count < 1:
        raise ValueError("p0-count must be positive.")
    if args.sample_path_count < 0:
        raise ValueError("sample-path-count must be nonnegative.")
    sample_path_p0_values = parse_float_grid(args.sample_path_p0_grid)
    p0_grid = parse_p0_grid(args)
    params = ModelParams(
        p1=args.p1,
        p2=args.p2,
        c1=args.c1,
        c2=args.c2,
        revenue=args.revenue,
        max_observations=args.max_observations,
        max_iter=args.max_iter,
        tol=args.tol,
        demand_floor=args.demand_floor,
    )
    validate_params(params)
    configure_plot_style()
    if args.run_robustness:
        run_robustness_experiment(args, params)
        return

    initial_successes = 0
    initial_failures = 0
    if args.initial_belief_mean is not None:
        initial_successes, initial_failures = closest_counts_for_posterior_mean(
            args.initial_belief_mean,
            params.max_observations,
        )
    initial_belief_mean = posterior_mean_scalar(initial_successes, initial_failures)

    project_dir = Path(__file__).resolve().parent
    outputs_dir = Path(args.outputs_dir) if args.outputs_dir else project_dir / "outputs"
    data_dir = outputs_dir / "data"
    plots_dir = outputs_dir / "plots"
    data_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    state_space = make_state_space(params.max_observations, method=args.method)
    path_diagnostics_path = data_dir / "path_diagnostics.csv"
    path_diagnostics_needs_header = True

    optimal_solutions = []
    convergence_frames = []
    simulation_rep_frames = []
    simulation_time_frames = []
    sample_path_frames = []
    summary_records = []
    diagnostics_frames = []

    print(
        "Solving average-per-stage best responses across p_0 values "
        f"with method={args.method}..."
    )
    if initial_successes or initial_failures:
        print(
            "Starting simulations from "
            f"S={initial_successes}, F={initial_failures}, "
            f"posterior mean={initial_belief_mean:.4f}"
        )

    for p0_idx, p0 in enumerate(p0_grid):
        p0 = float(p0)
        diagnostics_frames.append(validate_average_reward_transitions(p0, params, state_space))
        optimal_solution = solve_average_reward_policy(
            p0,
            params,
            state_space,
            fixed_product=None,
        )
        fixed_product1 = solve_average_reward_policy(
            p0,
            params,
            state_space,
            fixed_product=1,
        )
        fixed_product2 = solve_average_reward_policy(
            p0,
            params,
            state_space,
            fixed_product=2,
        )
        optimal_solutions.append(optimal_solution)
        convergence_frames.extend(
            [
                optimal_solution["convergence"],
                fixed_product1["convergence"],
                fixed_product2["convergence"],
            ]
        )

        if args.skip_simulation:
            simulation_reps = pd.DataFrame(
                {
                    "p0": [p0],
                    "rep": [0],
                    "A_market_share": [np.nan],
                    "product2_rate_when_A_chosen": [np.nan],
                    "A_success_rate_when_chosen": [np.nan],
                    "avg_profit_per_period": [np.nan],
                    "tail_avg_profit_per_period": [np.nan],
                    "final_posterior_mean": [np.nan],
                }
            )
        else:
            simulation_reps, simulation_time, path_diagnostics = simulate_solution(
                solution=optimal_solution,
                params=params,
                state_space=state_space,
                n_rep=args.n_rep,
                horizon=args.horizon,
                seed=args.seed + 10_000 * p0_idx,
                initial_successes=initial_successes,
                initial_failures=initial_failures,
            )
            path_diagnostics.to_csv(
                path_diagnostics_path,
                index=False,
                mode="w" if path_diagnostics_needs_header else "a",
                header=path_diagnostics_needs_header,
            )
            path_diagnostics_needs_header = False
            if np.any(np.isclose(sample_path_p0_values, p0)):
                sample_path_frames.append(path_diagnostics)
            simulation_rep_frames.append(simulation_reps)
            simulation_time_frames.append(simulation_time)

        summary_records.append(
            summarize_solution(
                optimal_solution,
                fixed_product1,
                fixed_product2,
                params,
                state_space,
                simulation_reps,
                initial_successes=initial_successes,
                initial_failures=initial_failures,
            )
        )
        print(
            f"  p_0={p0:.3f}: gain={optimal_solution['average_profit_gain']:.4f}, "
            f"initial product {int(optimal_solution['policy_product'][0])}, "
            f"initial Q2-Q1={optimal_solution['q_gap'][0]:.4f}, "
            f"iterations={optimal_solution['iterations']}, "
            f"span residual={optimal_solution['span_residual']:.2e}"
        )

    summary = pd.DataFrame.from_records(summary_records)
    policy_states = build_policy_state_table(
        optimal_solutions,
        state_space,
        params,
    )
    convergence = pd.concat(convergence_frames, ignore_index=True)
    diagnostics = pd.concat(diagnostics_frames, ignore_index=True)

    summary.to_csv(data_dir / "average_reward_summary.csv", index=False)
    policy_states.to_csv(data_dir / "average_reward_policy_by_state.csv", index=False)
    convergence.to_csv(
        data_dir / "average_reward_policy_iteration.csv",
        index=False,
    )
    diagnostics.to_csv(data_dir / "average_reward_diagnostics.csv", index=False)

    if simulation_rep_frames:
        simulation_reps = pd.concat(simulation_rep_frames, ignore_index=True)
        simulation_times = pd.concat(simulation_time_frames, ignore_index=True)
        simulation_reps.to_csv(data_dir / "simulation_replications.csv", index=False)
        simulation_times.to_csv(data_dir / "simulation_timeseries.csv", index=False)
        initial_plot_mean = None
        initial_filename_suffix = ""
        if initial_successes or initial_failures:
            initial_plot_mean = initial_belief_mean
            initial_filename_suffix = (
                "initmean_"
                f"{p0_filename_fragment(initial_belief_mean)}"
            )
        plot_simulation_timeseries(
            simulation_times,
            plots_dir,
            initial_belief_mean=initial_plot_mean,
            filename_suffix=initial_filename_suffix,
        )
        if sample_path_frames:
            sample_paths = pd.concat(sample_path_frames, ignore_index=True)
            for sample_p0 in sample_path_p0_values:
                plot_sample_simulation_paths(
                    sample_paths,
                    plots_dir,
                    target_p0=float(sample_p0),
                    n_paths=args.sample_path_count,
                )

    plot_p0_summary(summary, plots_dir)
    plot_policy_heatmaps(optimal_solutions, state_space, plots_dir)
    plot_policy_posterior_state_space(optimal_solutions, state_space, plots_dir)
    plot_bias_gap_posterior_state_space(
        optimal_solutions,
        state_space,
        params,
        plots_dir,
    )
    plot_convergence(convergence, plots_dir)

    display_columns = [
        "p0",
        "average_profit_gain",
        "initial_best_response_product",
        "initial_q_gap_product2_minus_product1",
        "mean_A_market_share_sim",
        "mean_product2_rate_when_A_chosen_sim",
        "mean_tail_profit_per_period_sim",
    ]
    print("\nSummary")
    print(summary[display_columns].round(4).to_string(index=False))
    print(f"\nSaved CSVs to: {data_dir}")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()
