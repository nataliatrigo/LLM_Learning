"""Calendar-time Monte Carlo paths for the observation-forgetting model."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from forgetting.layers import solve_observation_layers
from forgetting.model import Primitives


def _mean_interval(
    values_by_path: np.ndarray,
    confidence_level: float,
) -> tuple[float, float, float, float]:
    """Mean and normal interval treating each simulated path as one cluster."""

    values = np.asarray(values_by_path, dtype=float)
    estimate = float(np.mean(values))
    if len(values) < 2:
        return estimate, np.nan, np.nan, np.nan
    standard_error = float(np.std(values, ddof=1) / np.sqrt(len(values)))
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return (
        estimate,
        standard_error,
        max(0.0, estimate - z_value * standard_error),
        min(1.0, estimate + z_value * standard_error),
    )


def _ratio_interval(
    numerator_by_path: np.ndarray,
    denominator_by_path: np.ndarray,
    confidence_level: float,
) -> tuple[float, float, float, float]:
    """Ratio and path-clustered delta-method confidence interval."""

    numerator = np.asarray(numerator_by_path, dtype=float)
    denominator = np.asarray(denominator_by_path, dtype=float)
    if numerator.shape != denominator.shape or numerator.ndim != 1:
        raise ValueError("ratio inputs must be one-dimensional and conformable")
    denominator_total = float(np.sum(denominator))
    if denominator_total <= 0.0:
        return np.nan, np.nan, np.nan, np.nan
    estimate = float(np.sum(numerator) / denominator_total)
    if len(numerator) < 2:
        return estimate, np.nan, np.nan, np.nan
    residual = numerator - estimate * denominator
    standard_error = float(
        np.std(residual, ddof=1)
        / (np.sqrt(len(numerator)) * float(np.mean(denominator)))
    )
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return (
        estimate,
        standard_error,
        max(0.0, estimate - z_value * standard_error),
        min(1.0, estimate + z_value * standard_error),
    )


def _policy_and_demand(
    observation_count: np.ndarray,
    success_mass: np.ndarray,
    *,
    rho: float,
    policy: dict[str, object],
    action_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate the correct transient layer for each path state."""

    coordinate = np.asarray(policy["coordinate"], dtype=float)
    advantage_table = np.asarray(policy["advantage_table"], dtype=float)
    demand_table = np.asarray(policy["demand_table"], dtype=float)
    terminal_observation = int(policy["terminal_observation"])
    expected_shape = (terminal_observation + 1, len(coordinate))
    if advantage_table.shape != expected_shape or demand_table.shape != expected_shape:
        raise ValueError("policy interpolation tables are malformed")
    if not np.allclose(
        coordinate,
        np.linspace(0.0, 1.0, len(coordinate)),
        rtol=0.0,
        atol=4.0 * np.finfo(float).eps,
    ):
        raise ValueError("calendar simulation requires a uniform policy grid")

    total_mass = 1.0 - np.power(rho, observation_count)
    success_share = np.full(len(observation_count), 0.5, dtype=float)
    np.divide(success_mass, total_mass, out=success_share, where=total_mass > 0.0)
    success_share = np.clip(success_share, 0.0, 1.0)
    grid_position = success_share * (len(coordinate) - 1)
    left = np.minimum(grid_position.astype(np.intp), len(coordinate) - 2)
    weight = grid_position - left
    layer = np.minimum(observation_count, terminal_observation)
    advantage = (
        (1.0 - weight) * advantage_table[layer, left]
        + weight * advantage_table[layer, left + 1]
    )
    demand = (
        (1.0 - weight) * demand_table[layer, left]
        + weight * demand_table[layer, left + 1]
    )
    return advantage > action_tolerance, np.clip(demand, 0.0, 1.0)


def simulate_calendar_paths(
    primitives: Primitives,
    rho: float,
    *,
    n_paths: int,
    seed: int,
    max_periods: int,
    bin_width: int = 25,
    confidence_level: float = 0.95,
    policy_nodes: int = 2_001,
    terminal_gap: float = 1e-5,
    solver_tolerance: float = 1e-10,
    action_tolerance: float = 1e-6,
    max_iterations: int = 100_000,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate calendar-time demand and product choice under forgetting.

    Seller-A evidence is exponentially discounted only when Seller A is
    selected. Choosing the outside option leaves both the observation layer
    and normalized evidence state unchanged. Policies before the configured
    terminal layer use backward induction on the numerical grid; later
    observations use the mature-boundary policy whose remaining normalized
    mass is at most ``terminal_gap``.
    """

    if n_paths <= 0 or max_periods <= 0 or bin_width <= 0:
        raise ValueError("n_paths, max_periods, and bin_width must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0,1)")
    if not np.isfinite(action_tolerance) or action_tolerance < 0.0:
        raise ValueError("action_tolerance must be finite and nonnegative")

    _, policy = solve_observation_layers(
        primitives,
        rho,
        nodes=policy_nodes,
        terminal_gap=terminal_gap,
        tolerance=solver_tolerance,
        action_tolerance=action_tolerance,
        max_iterations=max_iterations,
        return_policy=True,
    )
    rng = np.random.default_rng(seed)
    innovation = 1.0 - rho
    bin_count = int(np.ceil(max_periods / bin_width))

    observation_count = np.zeros(n_paths, dtype=np.int32)
    success_count = np.zeros(n_paths, dtype=np.int32)
    failure_count = np.zeros(n_paths, dtype=np.int32)
    success_mass = np.zeros(n_paths, dtype=float)
    path_a_counts = np.zeros(n_paths, dtype=np.int32)
    path_product2_counts = np.zeros(n_paths, dtype=np.int32)
    last_product2_period = np.full(n_paths, np.nan)
    last_product2_pre_action_n = np.full(n_paths, np.nan)
    path_bin_a_counts = np.zeros((n_paths, bin_count), dtype=np.int32)
    path_bin_product2_counts = np.zeros((n_paths, bin_count), dtype=np.int32)
    bin_observation_sums = np.zeros(bin_count, dtype=np.int64)
    bin_prescribed_counts = np.zeros(bin_count, dtype=np.int64)
    period_rows: list[dict[str, float | int]] = []

    for period in range(max_periods):
        product2_prescribed, state_demand = _policy_and_demand(
            observation_count,
            success_mass,
            rho=rho,
            policy=policy,
            action_tolerance=action_tolerance,
        )
        seller_a_chosen = rng.random(n_paths) < state_demand
        product2_chosen = seller_a_chosen & product2_prescribed
        successful = rng.random(n_paths) < np.where(
            product2_prescribed, primitives.p2, primitives.p1
        )

        a_rate, a_se, a_lower, a_upper = _mean_interval(
            seller_a_chosen, confidence_level
        )
        p2_rate, p2_se, p2_lower, p2_upper = _ratio_interval(
            product2_chosen, seller_a_chosen, confidence_level
        )
        up2_rate, up2_se, up2_lower, up2_upper = _mean_interval(
            product2_chosen, confidence_level
        )
        quantiles = np.quantile(observation_count, (0.10, 0.50, 0.90))
        period_rows.append(
            {
                "period": period,
                "mean_observation_n": float(np.mean(observation_count)),
                "observation_n_q10": float(quantiles[0]),
                "observation_n_q50": float(quantiles[1]),
                "observation_n_q90": float(quantiles[2]),
                "product2_prescribed_share": float(np.mean(product2_prescribed)),
                "a_market_share": a_rate,
                "a_market_share_se": a_se,
                "a_market_share_ci_lower": a_lower,
                "a_market_share_ci_upper": a_upper,
                "a_chosen_count": int(np.sum(seller_a_chosen)),
                "paths_with_a_count": int(np.sum(seller_a_chosen)),
                "product2_chosen_count": int(np.sum(product2_chosen)),
                "product2_unconditional_rate": up2_rate,
                "product2_unconditional_cluster_se": up2_se,
                "product2_unconditional_ci_lower": up2_lower,
                "product2_unconditional_ci_upper": up2_upper,
                "product2_given_a_rate": p2_rate,
                "product2_given_a_cluster_se": p2_se,
                "product2_given_a_ci_lower": p2_lower,
                "product2_given_a_ci_upper": p2_upper,
            }
        )

        bin_index = period // bin_width
        path_bin_a_counts[:, bin_index] += seller_a_chosen
        path_bin_product2_counts[:, bin_index] += product2_chosen
        bin_observation_sums[bin_index] += int(np.sum(observation_count))
        bin_prescribed_counts[bin_index] += int(np.sum(product2_prescribed))
        path_a_counts += seller_a_chosen
        path_product2_counts += product2_chosen
        last_product2_period[product2_chosen] = float(period)
        last_product2_pre_action_n[product2_chosen] = observation_count[
            product2_chosen
        ]

        chosen_success = seller_a_chosen & successful
        chosen_failure = seller_a_chosen & ~successful
        success_mass[seller_a_chosen] *= rho
        success_mass[chosen_success] += innovation
        observation_count += seller_a_chosen
        success_count += chosen_success
        failure_count += chosen_failure

    binned_rows: list[dict[str, float | int]] = []
    for bin_index in range(bin_count):
        start = bin_index * bin_width
        stop = min(start + bin_width, max_periods)
        periods_in_bin = stop - start
        a_counts = path_bin_a_counts[:, bin_index]
        p2_counts = path_bin_product2_counts[:, bin_index]
        a_rate, a_se, a_lower, a_upper = _mean_interval(
            a_counts / periods_in_bin, confidence_level
        )
        p2_rate, p2_se, p2_lower, p2_upper = _ratio_interval(
            p2_counts, a_counts, confidence_level
        )
        up2_rate, up2_se, up2_lower, up2_upper = _mean_interval(
            p2_counts / periods_in_bin, confidence_level
        )
        binned_rows.append(
            {
                "bin_start": start,
                "bin_stop_exclusive": stop,
                "bin_center": 0.5 * (start + stop - 1),
                "periods_in_bin": periods_in_bin,
                "mean_observation_n": float(
                    bin_observation_sums[bin_index]
                    / (n_paths * periods_in_bin)
                ),
                "product2_prescribed_share": float(
                    bin_prescribed_counts[bin_index]
                    / (n_paths * periods_in_bin)
                ),
                "a_market_share": a_rate,
                "a_market_share_cluster_se": a_se,
                "a_market_share_ci_lower": a_lower,
                "a_market_share_ci_upper": a_upper,
                "a_chosen_count": int(np.sum(a_counts)),
                "paths_with_a_count": int(np.count_nonzero(a_counts)),
                "product2_chosen_count": int(np.sum(p2_counts)),
                "product2_unconditional_rate": up2_rate,
                "product2_unconditional_cluster_se": up2_se,
                "product2_unconditional_ci_lower": up2_lower,
                "product2_unconditional_ci_upper": up2_upper,
                "product2_given_a_rate": p2_rate,
                "product2_given_a_cluster_se": p2_se,
                "product2_given_a_ci_lower": p2_lower,
                "product2_given_a_ci_upper": p2_upper,
            }
        )

    path_summary = pd.DataFrame(
        {
            "path": np.arange(n_paths, dtype=int),
            "a_chosen_count": path_a_counts,
            "product2_count": path_product2_counts,
            "last_product2_period_within_window": last_product2_period,
            "last_product2_pre_action_n_within_window": (
                last_product2_pre_action_n
            ),
            "final_successes": success_count,
            "final_failures": failure_count,
            "final_observation_n": observation_count,
            "final_normalized_success_mass": success_mass,
            "final_normalized_failure_mass": (
                1.0 - np.power(rho, observation_count) - success_mass
            ),
        }
    )
    return pd.DataFrame(period_rows), pd.DataFrame(binned_rows), path_summary
