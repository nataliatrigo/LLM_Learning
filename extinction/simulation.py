"""Monte Carlo illustrations for the computed stationary policy."""

from __future__ import annotations

from statistics import NormalDist

import numpy as np
import pandas as pd

from extinction.model import (
    POLICY_PRODUCT2,
    Parameters,
    action_tolerance,
    demand,
)




def _cluster_mean_interval(
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


def _cluster_ratio_interval(
    numerator_by_path: np.ndarray,
    denominator_by_path: np.ndarray,
    confidence_level: float,
) -> tuple[float, float, float, float]:
    """Ratio estimate with a path-clustered delta-method interval.

    The estimate is ``sum(numerator) / sum(denominator)``.  Its standard
    error is based on the path-level influence residual
    ``numerator_i - estimate * denominator_i``.  This retains paths with a
    zero denominator as clusters and therefore respects the random number of
    Seller-A choices in a calendar-time period or bin.
    """

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
    mean_denominator = float(np.mean(denominator))
    standard_error = float(
        np.std(residual, ddof=1)
        / (np.sqrt(len(numerator)) * mean_denominator)
    )
    z_value = NormalDist().inv_cdf(0.5 + confidence_level / 2.0)
    return (
        estimate,
        standard_error,
        max(0.0, estimate - z_value * standard_error),
        min(1.0, estimate + z_value * standard_error),
    )




def simulate_calendar_paths(
    solution: dict,
    *,
    n_paths: int,
    seed: int,
    max_periods: int,
    bin_width: int = 25,
    confidence_level: float = 0.95,
    absolute_action_tolerance: float = 1e-10,
    relative_action_tolerance: float = 1e-8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate the stationary policy in calendar time.

    At the start of each calendar period, Seller A is selected with the
    Thompson-sampling demand stored in the solution.  The count state
    ``(S,F)`` advances only after such a selection; choosing the outside
    option leaves it unchanged.  Product-2 rates are consequently reported
    conditional on Seller A being selected.

    Confidence intervals for conditional product-2 rates use independent
    paths as clusters.  In particular, the binned interval does not treat
    the repeated calendar periods within a path as independent observations.
    If Seller A is never selected in a period or bin, the corresponding
    conditional rate and interval are missing rather than recorded as zero.
    """

    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if max_periods <= 0:
        raise ValueError("max_periods must be positive")
    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0,1)")
    if max_periods - 1 > int(solution["report_diagonal"]):
        raise ValueError("simulation would leave the reported policy region")

    params: Parameters = solution["parameters"]
    rng = np.random.default_rng(seed)
    bin_count = int(np.ceil(max_periods / bin_width))

    # A dense lookup makes the 20,000-path baseline inexpensive while retaining
    # the exact diagonal-specific action tolerance used elsewhere.
    policy_lookup = np.zeros((max_periods, max_periods), dtype=bool)
    demand_lookup = np.zeros((max_periods, max_periods), dtype=float)
    if "actions" in solution:
        for n in range(max_periods):
            actions = np.asarray(solution["actions"][n], dtype=np.int8)
            policy_lookup[n, : n + 1] = actions == POLICY_PRODUCT2
            demand_lookup[n, : n + 1] = demand(n, params.p0)
    else:
        for n in range(max_periods):
            layer = solution["layers"][n]
            advantage = np.asarray(layer["advantage"], dtype=float)
            tolerance = action_tolerance(
                advantage,
                absolute=absolute_action_tolerance,
                relative=relative_action_tolerance,
            )
            policy_lookup[n, : n + 1] = advantage > tolerance
            demand_lookup[n, : n + 1] = np.asarray(
                layer["demand"],
                dtype=float,
            )

    successes = np.zeros(n_paths, dtype=np.int32)
    failures = np.zeros(n_paths, dtype=np.int32)
    path_a_counts = np.zeros(n_paths, dtype=np.int32)
    path_product2_counts = np.zeros(n_paths, dtype=np.int32)
    last_product2_period = np.full(n_paths, np.nan, dtype=float)
    last_product2_pre_action_n = np.full(n_paths, np.nan, dtype=float)

    path_bin_a_counts = np.zeros((n_paths, bin_count), dtype=np.int32)
    path_bin_product2_counts = np.zeros((n_paths, bin_count), dtype=np.int32)
    bin_n_sums = np.zeros(bin_count, dtype=np.int64)
    bin_prescribed_counts = np.zeros(bin_count, dtype=np.int64)
    period_rows: list[dict[str, float | int]] = []

    for period in range(max_periods):
        observation_n = successes + failures
        product2_prescribed = policy_lookup[observation_n, successes]
        state_demand = demand_lookup[observation_n, successes]
        seller_a_chosen = rng.random(n_paths) < state_demand
        product2_chosen = seller_a_chosen & product2_prescribed

        a_rate, a_se, a_lower, a_upper = _cluster_mean_interval(
            seller_a_chosen.astype(float),
            confidence_level,
        )
        p2_rate, p2_se, p2_lower, p2_upper = _cluster_ratio_interval(
            product2_chosen.astype(float),
            seller_a_chosen.astype(float),
            confidence_level,
        )
        (
            unconditional_p2_rate,
            unconditional_p2_se,
            unconditional_p2_lower,
            unconditional_p2_upper,
        ) = _cluster_mean_interval(
            product2_chosen.astype(float),
            confidence_level,
        )
        quantiles = np.quantile(observation_n, (0.10, 0.50, 0.90))
        a_count = int(np.sum(seller_a_chosen))
        p2_count = int(np.sum(product2_chosen))
        period_rows.append(
            {
                "period": period,
                "mean_observation_n": float(np.mean(observation_n)),
                "observation_n_q10": float(quantiles[0]),
                "observation_n_q50": float(quantiles[1]),
                "observation_n_q90": float(quantiles[2]),
                "product2_prescribed_share": float(
                    np.mean(product2_prescribed)
                ),
                "a_market_share": a_rate,
                "a_market_share_se": a_se,
                "a_market_share_ci_lower": a_lower,
                "a_market_share_ci_upper": a_upper,
                "a_chosen_count": a_count,
                "paths_with_a_count": a_count,
                "product2_chosen_count": p2_count,
                "product2_unconditional_rate": unconditional_p2_rate,
                "product2_unconditional_cluster_se": unconditional_p2_se,
                "product2_unconditional_ci_lower": unconditional_p2_lower,
                "product2_unconditional_ci_upper": unconditional_p2_upper,
                "product2_given_a_rate": p2_rate,
                "product2_given_a_cluster_se": p2_se,
                "product2_given_a_ci_lower": p2_lower,
                "product2_given_a_ci_upper": p2_upper,
            }
        )

        bin_index = period // bin_width
        path_bin_a_counts[:, bin_index] += seller_a_chosen
        path_bin_product2_counts[:, bin_index] += product2_chosen
        bin_n_sums[bin_index] += int(np.sum(observation_n))
        bin_prescribed_counts[bin_index] += int(np.sum(product2_prescribed))
        path_a_counts += seller_a_chosen
        path_product2_counts += product2_chosen
        last_product2_period[product2_chosen] = float(period)
        last_product2_pre_action_n[product2_chosen] = observation_n[
            product2_chosen
        ]

        success_probability = np.where(product2_prescribed, params.p2, params.p1)
        successful_outcome = rng.random(n_paths) < success_probability
        successes += seller_a_chosen & successful_outcome
        failures += seller_a_chosen & ~successful_outcome

    binned_rows: list[dict[str, float | int]] = []
    for bin_index in range(bin_count):
        start = bin_index * bin_width
        stop = min(start + bin_width, max_periods)
        periods_in_bin = stop - start
        a_counts = path_bin_a_counts[:, bin_index]
        p2_counts = path_bin_product2_counts[:, bin_index]
        a_rate, a_se, a_lower, a_upper = _cluster_mean_interval(
            a_counts / periods_in_bin,
            confidence_level,
        )
        p2_rate, p2_se, p2_lower, p2_upper = _cluster_ratio_interval(
            p2_counts,
            a_counts,
            confidence_level,
        )
        (
            unconditional_p2_rate,
            unconditional_p2_se,
            unconditional_p2_lower,
            unconditional_p2_upper,
        ) = _cluster_mean_interval(
            p2_counts / periods_in_bin,
            confidence_level,
        )
        binned_rows.append(
            {
                "bin_start": start,
                "bin_stop_exclusive": stop,
                "bin_center": 0.5 * (start + stop - 1),
                "periods_in_bin": periods_in_bin,
                "mean_observation_n": float(
                    bin_n_sums[bin_index] / (n_paths * periods_in_bin)
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
                "product2_unconditional_rate": unconditional_p2_rate,
                "product2_unconditional_cluster_se": unconditional_p2_se,
                "product2_unconditional_ci_lower": unconditional_p2_lower,
                "product2_unconditional_ci_upper": unconditional_p2_upper,
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
            "final_successes": successes,
            "final_failures": failures,
            "final_observation_n": successes + failures,
        }
    )
    return (
        pd.DataFrame(period_rows),
        pd.DataFrame(binned_rows),
        path_summary,
    )
