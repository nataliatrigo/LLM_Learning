"""Streaming simulations with common random numbers across learning rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .discrete_solver import DiscreteSolution
from .primitives import SellerPrimitives


def model_fields(model: Any) -> dict[str, Any]:
    metadata = getattr(model, "metadata", {})
    if callable(metadata):
        metadata = metadata()
    method = str(metadata.get("method", model.__class__.__name__))
    parameter_name = metadata.get("parameter_name")
    parameter_value = metadata.get("parameter_value")
    if parameter_name is None or parameter_value is None:
        key = method
        label = str(metadata.get("name", method))
    else:
        compact = f"{float(parameter_value):g}".replace("-", "m").replace(".", "p")
        key = f"{method}__{parameter_name}_{compact}"
        label = f"{metadata.get('name', method)} ({parameter_name}={float(parameter_value):g})"
    return {
        "method": method,
        "parameter_name": parameter_name,
        "parameter_value": parameter_value,
        "method_key": key,
        "method_label": label,
    }


def _validate_model_primitives(model: Any, primitives: SellerPrimitives) -> None:
    model_primitives = getattr(model, "primitives", None)
    if model_primitives is not None and model_primitives != primitives:
        raise ValueError(
            "model.primitives and the explicitly supplied SellerPrimitives differ"
        )


class CountPolicy(Protocol):
    def action_for_counts(
        self, successes: np.ndarray, failures: np.ndarray
    ) -> np.ndarray: ...


@dataclass
class LayerPolicy:
    """Policy lookup backed by a reported triangular DP interior.

    A product-1 tail is allowed only when the caller explicitly certifies that
    the reported solution has a sufficiently long inactive tail. The lookup
    otherwise raises instead of silently extrapolating.
    """

    solution: DiscreteSolution
    allow_product1_tail: bool = False

    def action_for_counts(
        self, successes: np.ndarray, failures: np.ndarray
    ) -> np.ndarray:
        successes = np.asarray(successes, dtype=np.int64)
        failures = np.asarray(failures, dtype=np.int64)
        totals = successes + failures
        result = np.zeros(successes.shape, dtype=bool)
        outside = totals > self.solution.report_diagonal
        if np.any(outside) and not self.allow_product1_tail:
            maximum = int(np.max(totals))
            raise RuntimeError(
                "simulation left the reported policy region "
                f"(n={maximum}>{self.solution.report_diagonal})"
            )
        for n in np.unique(totals[~outside]):
            selected = totals == n
            result[selected] = self.solution.layers[int(n)]["action2"][successes[selected]]
        return result


@dataclass(frozen=True)
class BalancePolicy:
    """One-dimensional stationary epsilon-greedy solution at p0=1/2."""

    epsilon: float
    half_width: int
    k: np.ndarray
    value: np.ndarray
    advantage: np.ndarray
    action2: np.ndarray
    iterations: int
    iteration_residual: float
    bellman_residual: float
    active_min_k: int | None
    active_max_k: int | None
    boundary_safe: bool

    def action_for_counts(
        self, successes: np.ndarray, failures: np.ndarray
    ) -> np.ndarray:
        balance = np.asarray(successes, dtype=np.int64) - np.asarray(
            failures, dtype=np.int64
        )
        inside = np.abs(balance) <= self.half_width
        result = np.zeros(balance.shape, dtype=bool)
        result[inside] = self.action2[balance[inside] + self.half_width]
        if np.any(~inside) and not self.boundary_safe:
            raise RuntimeError("epsilon-greedy balance left an uncertified truncation")
        return result


def _uniform_epsilon_demand(balance: np.ndarray, epsilon: float) -> np.ndarray:
    return np.where(
        balance > 0,
        1.0 - epsilon / 2.0,
        np.where(balance < 0, epsilon / 2.0, 0.5),
    )


def solve_epsilon_balance_policy(
    epsilon: float,
    primitives: SellerPrimitives,
    half_width: int = 1200,
    tolerance: float = 1e-11,
    max_iterations: int = 100_000,
    boundary_margin: int = 50,
) -> BalancePolicy:
    """Solve the exact balance reduction available at the symmetric threshold."""
    if not np.isclose(primitives.p0, 0.5, rtol=0.0, atol=1e-14):
        raise ValueError("the balance reduction requires p0=0.5")
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("epsilon must lie in [0,1]")
    if half_width <= boundary_margin:
        raise ValueError("half_width must exceed boundary_margin")
    k = np.arange(-half_width, half_width + 1, dtype=np.int64)
    demand = _uniform_epsilon_demand(k, epsilon)
    prefactor = demand / (1.0 - primitives.gamma * (1.0 - demand))
    value = np.zeros(k.size, dtype=float)
    difference = np.inf

    def update(candidate: np.ndarray) -> np.ndarray:
        up = np.r_[candidate[1:], candidate[-1]]
        down = np.r_[candidate[0], candidate[:-1]]
        q1 = primitives.revenue - primitives.c1 + primitives.gamma * (
            primitives.p1 * up + (1.0 - primitives.p1) * down
        )
        q2 = primitives.revenue - primitives.c2 + primitives.gamma * (
            primitives.p2 * up + (1.0 - primitives.p2) * down
        )
        return prefactor * np.maximum(q1, q2)

    for iteration in range(1, max_iterations + 1):
        updated = update(value)
        difference = float(np.max(np.abs(updated - value)))
        value = updated
        if difference <= tolerance:
            break
    else:
        raise RuntimeError("epsilon-greedy balance value iteration did not converge")

    up = np.r_[value[1:], value[-1]]
    down = np.r_[value[0], value[:-1]]
    advantage = -(
        primitives.c2 - primitives.c1
    ) + primitives.gamma * (primitives.p2 - primitives.p1) * (up - down)
    action2 = advantage > 0.0
    active = k[action2]
    active_min = int(active.min()) if active.size else None
    active_max = int(active.max()) if active.size else None
    boundary_safe = bool(
        active.size == 0
        or (
            active_min is not None
            and active_max is not None
            and active_min > -half_width + boundary_margin
            and active_max < half_width - boundary_margin
        )
    )
    return BalancePolicy(
        epsilon=float(epsilon),
        half_width=int(half_width),
        k=k,
        value=value,
        advantage=advantage,
        action2=action2,
        iterations=iteration,
        iteration_residual=difference,
        bellman_residual=float(np.max(np.abs(update(value) - value))),
        active_min_k=active_min,
        active_max_k=active_max,
        boundary_safe=boundary_safe,
    )


@dataclass
class SimulationResult:
    summary: pd.DataFrame
    path_summary: pd.DataFrame
    calendar_series: pd.DataFrame
    observation_series: pd.DataFrame
    representative_trajectories: pd.DataFrame
    late_use: pd.DataFrame


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    result = np.full(np.broadcast_shapes(np.shape(numerator), np.shape(denominator)), np.nan, dtype=float)
    return np.divide(numerator, denominator, out=result, where=np.asarray(denominator) > 0)


def _pooled_conditional_share(numerator: np.ndarray, denominator: np.ndarray) -> float:
    denominator_total = int(np.sum(denominator))
    if denominator_total == 0:
        return float("nan")
    return float(np.sum(numerator) / denominator_total)


def _finalize_simulation(
    *,
    model: Any,
    primitives: SellerPrimitives,
    paths: int,
    periods: int,
    seed: int,
    late_thresholds: tuple[int, ...],
    late_window_start: int,
    discounted_profit: np.ndarray,
    cumulative_profit: np.ndarray,
    demand_sum: np.ndarray,
    prescribed2_count: np.ndarray,
    chosen_count: np.ndarray,
    product2_used_count: np.ndarray,
    late_chosen_count: np.ndarray,
    late_used2_count: np.ndarray,
    last_product2_use: np.ndarray,
    any_use_after: dict[int, np.ndarray],
    maximum_observations: np.ndarray,
    calendar_rows: list[dict[str, Any]],
    observation_accumulators: dict[str, np.ndarray],
    representative_rows: list[dict[str, Any]],
) -> SimulationResult:
    fields = model_fields(model)
    product2_conditional = _safe_ratio(product2_used_count, chosen_count)
    late_conditional = _safe_ratio(late_used2_count, late_chosen_count)
    ever_used = last_product2_use >= 0
    observed_last = np.where(ever_used, last_product2_use, np.nan)
    used_in_late_window = last_product2_use >= late_window_start
    per_path = pd.DataFrame(
        {
            **{key: value for key, value in fields.items() if key != "method_label"},
            "path": np.arange(paths),
            "discounted_profit": discounted_profit,
            "calendar_profit": cumulative_profit / periods,
            "mean_demand": demand_sum / periods,
            "product2_share_calendar": prescribed2_count / periods,
            "product2_prescribed_share_calendar": prescribed2_count / periods,
            "product2_used_share_calendar": product2_used_count / periods,
            "realized_A_share": chosen_count / periods,
            "product2_share_conditional_on_A": product2_conditional,
            "late_product2_share_conditional_on_A": late_conditional,
            "last_product2_use": observed_last,
            "ever_used_product2": ever_used,
            "product2_used_in_late_window": used_in_late_window,
            "maximum_observation_count": maximum_observations,
        }
    )
    for threshold in late_thresholds:
        per_path[f"used_product2_after_{threshold}"] = any_use_after[threshold]

    summary_row: dict[str, Any] = {
        **fields,
        "p0": primitives.p0,
        "p1": primitives.p1,
        "p2": primitives.p2,
        "c1": primitives.c1,
        "c2": primitives.c2,
        "R": primitives.revenue,
        "gamma": primitives.gamma,
        "paths": paths,
        "periods": periods,
        "seed": seed,
        "mean_discounted_profit": float(np.mean(discounted_profit)),
        "standard_error_discounted_profit": float(np.std(discounted_profit, ddof=1) / np.sqrt(paths)) if paths > 1 else 0.0,
        "mean_calendar_profit": float(np.mean(cumulative_profit) / periods),
        "mean_demand": float(np.mean(demand_sum) / periods),
        "realized_A_share": float(np.mean(chosen_count) / periods),
        "product2_share_calendar": float(np.mean(prescribed2_count) / periods),
        "product2_prescribed_share_calendar": float(
            np.mean(prescribed2_count) / periods
        ),
        "product2_used_share_calendar": float(
            np.mean(product2_used_count) / periods
        ),
        "product2_share_conditional_on_A": _pooled_conditional_share(
            product2_used_count, chosen_count
        ),
        "late_product2_share_conditional_on_A": _pooled_conditional_share(
            late_used2_count, late_chosen_count
        ),
        "probability_ever_product2_used": float(np.mean(ever_used)),
        "probability_product2_used_in_late_window": float(
            np.mean(used_in_late_window)
        ),
        "probability_no_product2_use_in_late_window": float(
            np.mean(~used_in_late_window)
        ),
        "mean_observed_last_product2_use": float(np.nanmean(observed_last)) if np.any(np.isfinite(observed_last)) else np.nan,
        "maximum_observation_count": int(np.max(maximum_observations)),
        "late_window_start": late_window_start,
    }
    late_rows = []
    for threshold in late_thresholds:
        probability = float(np.mean(any_use_after[threshold]))
        summary_row[
            f"probability_product2_used_at_or_after_{threshold}"
        ] = probability
        late_rows.append(
            {
                **fields,
                "threshold": threshold,
                "threshold_is_inclusive": True,
                "probability_product2_used_at_or_after_threshold": probability,
            }
        )

    obs = observation_accumulators
    observed = obs["count"] > 0
    observation_frame = pd.DataFrame(
        {
            **{key: value for key, value in fields.items() if key != "method_label"},
            "observation": np.flatnonzero(observed),
            "paths_observed": obs["count"][observed],
            "mean_demand": _safe_ratio(obs["demand"][observed], obs["count"][observed]),
            "product2_share": _safe_ratio(obs["action2"][observed], obs["count"][observed]),
            "mean_profit": _safe_ratio(obs["profit"][observed], obs["count"][observed]),
            "mean_posterior_mean": _safe_ratio(obs["posterior_mean"][observed], obs["count"][observed]),
            "mean_effective_sample_size": _safe_ratio(obs["effective_sample_size"][observed], obs["count"][observed]),
            "mean_effective_concentration": _safe_ratio(obs["effective_concentration"][observed], obs["count"][observed]),
        }
    )
    return SimulationResult(
        summary=pd.DataFrame([summary_row]),
        path_summary=per_path,
        calendar_series=pd.DataFrame(calendar_rows),
        observation_series=observation_frame,
        representative_trajectories=pd.DataFrame(representative_rows),
        late_use=pd.DataFrame(late_rows),
    )


def _observation_accumulators(periods: int) -> dict[str, np.ndarray]:
    return {
        key: np.zeros(periods + 1, dtype=float)
        for key in [
            "count",
            "demand",
            "action2",
            "profit",
            "posterior_mean",
            "effective_sample_size",
            "effective_concentration",
        ]
    }


def _add_observation_events(
    accumulators: dict[str, np.ndarray],
    observation_count: np.ndarray,
    chosen: np.ndarray,
    *,
    demand: np.ndarray,
    action2: np.ndarray,
    profit: np.ndarray,
    posterior_mean: np.ndarray,
    effective_sample_size: np.ndarray,
    effective_concentration: np.ndarray,
) -> None:
    indices = observation_count[chosen]
    length = len(accumulators["count"])
    accumulators["count"] += np.bincount(indices, minlength=length)[:length]
    for key, values in [
        ("demand", demand),
        ("action2", action2.astype(float)),
        ("profit", profit),
        ("posterior_mean", posterior_mean),
        ("effective_sample_size", effective_sample_size),
        ("effective_concentration", effective_concentration),
    ]:
        accumulators[key] += np.bincount(
            indices, weights=values[chosen], minlength=length
        )[:length]


def simulate_discrete(
    model: Any,
    policy: CountPolicy,
    primitives: SellerPrimitives,
    *,
    paths: int,
    periods: int,
    seed: int,
    representative_paths: int = 5,
    late_thresholds: tuple[int, ...] = (1000, 2500, 4000),
    late_window_start: int = 4000,
) -> SimulationResult:
    """Simulate an integer-count policy in calendar and observation time."""
    if paths <= 0 or periods <= 0:
        raise ValueError("paths and periods must be positive")
    _validate_model_primitives(model, primitives)
    if isinstance(policy, LayerPolicy) and policy.solution.primitives != primitives:
        raise ValueError("policy solution and simulation primitives differ")
    fields = model_fields(model)
    rng = np.random.default_rng(seed)
    successes = np.zeros(paths, dtype=np.int64)
    failures = np.zeros(paths, dtype=np.int64)
    discounted_profit = np.zeros(paths)
    cumulative_profit = np.zeros(paths)
    demand_sum = np.zeros(paths)
    prescribed2_count = np.zeros(paths, dtype=np.int64)
    chosen_count = np.zeros(paths, dtype=np.int64)
    product2_used_count = np.zeros(paths, dtype=np.int64)
    late_chosen_count = np.zeros(paths, dtype=np.int64)
    late_used2_count = np.zeros(paths, dtype=np.int64)
    last_use = np.full(paths, -1, dtype=np.int64)
    any_after = {threshold: np.zeros(paths, dtype=bool) for threshold in late_thresholds}
    maximum_observations = np.zeros(paths, dtype=np.int64)
    calendar_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    representative_ids = np.arange(min(paths, representative_paths), dtype=int)
    observation = _observation_accumulators(periods)
    discount = 1.0

    for t in range(periods):
        counts = successes + failures
        demand = np.asarray(model.demand_counts(successes, failures), dtype=float)
        action2 = np.asarray(policy.action_for_counts(successes, failures), dtype=bool)
        posterior_mean = np.asarray(
            model.posterior_mean_counts(successes, failures), dtype=float
        )
        effective_sample_size = np.asarray(
            model.effective_sample_size_counts(successes, failures), dtype=float
        )
        effective_concentration = np.asarray(
            model.effective_concentration_counts(successes, failures), dtype=float
        )
        chosen = rng.random(paths) < demand
        success_probability = np.where(action2, primitives.p2, primitives.p1)
        outcome_uniform = rng.random(paths)
        success = chosen & (outcome_uniform < success_probability)
        failure = chosen & ~success
        product2_used = chosen & action2
        profit = chosen * np.where(
            action2, primitives.revenue - primitives.c2, primitives.revenue - primitives.c1
        )

        discounted_profit += discount * profit
        cumulative_profit += profit
        demand_sum += demand
        prescribed2_count += action2
        chosen_count += chosen
        product2_used_count += product2_used
        period = t + 1
        if period >= late_window_start:
            late_chosen_count += chosen
            late_used2_count += product2_used
        last_use[product2_used] = period
        for threshold in late_thresholds:
            if period >= threshold:
                any_after[threshold] |= product2_used

        _add_observation_events(
            observation,
            counts,
            chosen,
            demand=demand,
            action2=action2,
            profit=profit,
            posterior_mean=posterior_mean,
            effective_sample_size=effective_sample_size,
            effective_concentration=effective_concentration,
        )
        calendar_rows.append(
            {
                **fields,
                "period": period,
                "mean_demand": float(np.mean(demand)),
                "realized_A_share": float(np.mean(chosen)),
                "product2_prescribed_share": float(np.mean(action2)),
                "product2_used_share": float(np.mean(product2_used)),
                "product2_share_conditional_on_A": _pooled_conditional_share(
                    product2_used, chosen
                ),
                "mean_profit": float(np.mean(profit)),
                "mean_posterior_mean": float(np.mean(posterior_mean)),
                "mean_effective_sample_size": float(np.mean(effective_sample_size)),
                "mean_effective_concentration": float(
                    np.mean(effective_concentration)
                ),
                "mean_observation_count": float(np.mean(counts)),
            }
        )
        for path_id in representative_ids:
            representative_rows.append(
                {
                    **fields,
                    "representative_path": int(path_id),
                    "period": period,
                    "S": int(successes[path_id]),
                    "F": int(failures[path_id]),
                    "x": np.nan,
                    "y": np.nan,
                    "posterior_mean": float(posterior_mean[path_id]),
                    "effective_sample_size": float(effective_sample_size[path_id]),
                    "effective_concentration": float(
                        effective_concentration[path_id]
                    ),
                    "observation_count": int(counts[path_id]),
                    "demand": float(demand[path_id]),
                    "product2_prescribed": bool(action2[path_id]),
                    "chosen_A": bool(chosen[path_id]),
                    "product2_used": bool(product2_used[path_id]),
                    "success": bool(success[path_id]),
                    "profit": float(profit[path_id]),
                }
            )

        successes += success
        failures += failure
        maximum_observations = np.maximum(maximum_observations, successes + failures)
        discount *= primitives.gamma

    return _finalize_simulation(
        model=model,
        primitives=primitives,
        paths=paths,
        periods=periods,
        seed=seed,
        late_thresholds=late_thresholds,
        late_window_start=late_window_start,
        discounted_profit=discounted_profit,
        cumulative_profit=cumulative_profit,
        demand_sum=demand_sum,
        prescribed2_count=prescribed2_count,
        chosen_count=chosen_count,
        product2_used_count=product2_used_count,
        late_chosen_count=late_chosen_count,
        late_used2_count=late_used2_count,
        last_product2_use=last_use,
        any_use_after=any_after,
        maximum_observations=maximum_observations,
        calendar_rows=calendar_rows,
        observation_accumulators=observation,
        representative_rows=representative_rows,
    )


def simulate_continuous(
    model: Any,
    solution: Any,
    primitives: SellerPrimitives,
    *,
    paths: int,
    periods: int,
    seed: int,
    representative_paths: int = 5,
    late_thresholds: tuple[int, ...] = (1000, 2500, 4000),
    late_window_start: int = 4000,
) -> SimulationResult:
    """Simulate a forgetting policy, evaluating actions from interpolated value."""
    if paths <= 0 or periods <= 0:
        raise ValueError("paths and periods must be positive")
    _validate_model_primitives(model, primitives)
    if getattr(solution, "primitives", primitives) != primitives:
        raise ValueError("continuous solution and simulation primitives differ")
    fields = model_fields(model)
    rng = np.random.default_rng(seed)
    x = np.zeros(paths)
    y = np.zeros(paths)
    observation_count = np.zeros(paths, dtype=np.int64)
    discounted_profit = np.zeros(paths)
    cumulative_profit = np.zeros(paths)
    demand_sum = np.zeros(paths)
    prescribed2_count = np.zeros(paths, dtype=np.int64)
    chosen_count = np.zeros(paths, dtype=np.int64)
    product2_used_count = np.zeros(paths, dtype=np.int64)
    late_chosen_count = np.zeros(paths, dtype=np.int64)
    late_used2_count = np.zeros(paths, dtype=np.int64)
    last_use = np.full(paths, -1, dtype=np.int64)
    any_after = {threshold: np.zeros(paths, dtype=bool) for threshold in late_thresholds}
    maximum_observations = np.zeros(paths, dtype=np.int64)
    calendar_rows: list[dict[str, Any]] = []
    representative_rows: list[dict[str, Any]] = []
    representative_ids = np.arange(min(paths, representative_paths), dtype=int)
    observation = _observation_accumulators(periods)
    discount = 1.0
    rho = float(model.rho)
    innovation = 1.0 - rho

    for t in range(periods):
        demand = np.asarray(model.demand_xy(x, y), dtype=float)
        success_x, success_y = rho * x + innovation, rho * y
        failure_x, failure_y = rho * x, rho * y + innovation
        value_success = np.asarray(solution.grid.interpolate(solution.value, success_x, success_y), dtype=float)
        value_failure = np.asarray(solution.grid.interpolate(solution.value, failure_x, failure_y), dtype=float)
        gap = primitives.gamma * (value_success - value_failure)
        advantage = -(primitives.c2 - primitives.c1) + (
            primitives.p2 - primitives.p1
        ) * gap
        action2 = advantage > 0.0

        a = 1.0 + x / innovation
        b = 1.0 + y / innovation
        posterior_mean = a / (a + b)
        effective_sample_size = (x + y) / innovation
        effective_concentration = a + b
        chosen = rng.random(paths) < demand
        success_probability = np.where(action2, primitives.p2, primitives.p1)
        outcome_uniform = rng.random(paths)
        success = chosen & (outcome_uniform < success_probability)
        failure = chosen & ~success
        product2_used = chosen & action2
        profit = chosen * np.where(
            action2, primitives.revenue - primitives.c2, primitives.revenue - primitives.c1
        )

        discounted_profit += discount * profit
        cumulative_profit += profit
        demand_sum += demand
        prescribed2_count += action2
        chosen_count += chosen
        product2_used_count += product2_used
        period = t + 1
        if period >= late_window_start:
            late_chosen_count += chosen
            late_used2_count += product2_used
        last_use[product2_used] = period
        for threshold in late_thresholds:
            if period >= threshold:
                any_after[threshold] |= product2_used

        _add_observation_events(
            observation,
            observation_count,
            chosen,
            demand=demand,
            action2=action2,
            profit=profit,
            posterior_mean=posterior_mean,
            effective_sample_size=effective_sample_size,
            effective_concentration=effective_concentration,
        )
        calendar_rows.append(
            {
                **fields,
                "period": period,
                "mean_demand": float(np.mean(demand)),
                "realized_A_share": float(np.mean(chosen)),
                "product2_prescribed_share": float(np.mean(action2)),
                "product2_used_share": float(np.mean(product2_used)),
                "product2_share_conditional_on_A": _pooled_conditional_share(
                    product2_used, chosen
                ),
                "mean_profit": float(np.mean(profit)),
                "mean_posterior_mean": float(np.mean(posterior_mean)),
                "mean_effective_sample_size": float(np.mean(effective_sample_size)),
                "mean_effective_concentration": float(
                    np.mean(effective_concentration)
                ),
                "mean_observation_count": float(np.mean(observation_count)),
            }
        )
        for path_id in representative_ids:
            representative_rows.append(
                {
                    **fields,
                    "representative_path": int(path_id),
                    "period": period,
                    "S": np.nan,
                    "F": np.nan,
                    "x": float(x[path_id]),
                    "y": float(y[path_id]),
                    "posterior_mean": float(posterior_mean[path_id]),
                    "effective_sample_size": float(effective_sample_size[path_id]),
                    "effective_concentration": float(
                        effective_concentration[path_id]
                    ),
                    "observation_count": int(observation_count[path_id]),
                    "demand": float(demand[path_id]),
                    "product2_prescribed": bool(action2[path_id]),
                    "chosen_A": bool(chosen[path_id]),
                    "product2_used": bool(product2_used[path_id]),
                    "success": bool(success[path_id]),
                    "profit": float(profit[path_id]),
                }
            )

        if bool(model.has_exact_idle_self_loop):
            idle_x, idle_y = x, y
        else:
            idle_x, idle_y = rho * x, rho * y
        x = np.where(success, success_x, np.where(failure, failure_x, idle_x))
        y = np.where(success, success_y, np.where(failure, failure_y, idle_y))
        observation_count += chosen
        maximum_observations = np.maximum(maximum_observations, observation_count)
        discount *= primitives.gamma

    return _finalize_simulation(
        model=model,
        primitives=primitives,
        paths=paths,
        periods=periods,
        seed=seed,
        late_thresholds=late_thresholds,
        late_window_start=late_window_start,
        discounted_profit=discounted_profit,
        cumulative_profit=cumulative_profit,
        demand_sum=demand_sum,
        prescribed2_count=prescribed2_count,
        chosen_count=chosen_count,
        product2_used_count=product2_used_count,
        late_chosen_count=late_chosen_count,
        late_used2_count=late_used2_count,
        last_product2_use=last_use,
        any_use_after=any_after,
        maximum_observations=maximum_observations,
        calendar_rows=calendar_rows,
        observation_accumulators=observation,
        representative_rows=representative_rows,
    )


def concatenate_simulations(results: list[SimulationResult]) -> dict[str, pd.DataFrame]:
    """Concatenate homogeneous output tables from several configurations."""
    attributes = [
        "summary",
        "path_summary",
        "calendar_series",
        "observation_series",
        "representative_trajectories",
        "late_use",
    ]
    return {
        name: pd.concat([getattr(result, name) for result in results], ignore_index=True)
        if results
        else pd.DataFrame()
        for name in attributes
    }
