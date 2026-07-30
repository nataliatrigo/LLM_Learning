"""Monte Carlo dynamic panels modeled on the repository's original DP figure.

The legacy figure called its curves "paths", but every plotted point was a
contemporaneous cross-sectional statistic over independent simulations.  This
module preserves those useful definitions, adds confidence intervals and
common random numbers, and uses the modular study's stationary discounted
policies.  No rolling or cumulative average is used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/modular_dp_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy.stats import norm, t as student_t

from .continuous_solver import (
    ContinuousSolverConfig,
    compare_nested_solutions,
    solve_continuous,
)
from .demand_models import (
    CalendarForgettingTS,
    EpsilonGreedy,
    MeanPreservingTemperature,
    ObservationForgettingTS,
    ScaledUpdates,
    StandardThompsonSampling,
)
from .diagnostics import compare_discrete_solutions
from .discrete_solver import DiscreteSolverConfig, solve_discrete
from .outputs import ExperimentPaths, configuration_hash, write_frame, write_json
from .primitives import SellerPrimitives
from .simulation import BalancePolicy, LayerPolicy, solve_epsilon_balance_policy


FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


DYNAMIC_METHOD_COLORS = {
    "standard_ts": "#4e79a7",  # blue
    "scaled_eta_0p5": "#f28e2b",  # orange
    "scaled_eta_2": "#e15759",  # red
    "temperature_0p5": "#76b7b2",  # teal
    "temperature_2": "#59a14f",  # green
    "epsilon_0p05": "#b89b00",  # dark yellow
    "epsilon_0p1": "#b07aa1",  # purple
    "epsilon_0p2": "#6b7280",  # gray
    "observation_forgetting_0p95": "#ff7f91",  # pink
    "calendar_forgetting_0p95": "#9c755f",  # brown
}


@dataclass(frozen=True)
class CommonRandomNumbers:
    """Uniform draws shared across every curve in a comparison."""

    choice: FloatArray
    outcome: FloatArray

    @property
    def periods(self) -> int:
        return int(self.choice.shape[0])

    @property
    def paths(self) -> int:
        return int(self.choice.shape[1])


@dataclass(frozen=True)
class DynamicMethodSpecification:
    """One representative demand rule to compare across outside options."""

    key: str
    label: str
    model_kind: str
    solver_kind: str
    parameter_value: float | None = None


def _compact_number(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _concise_method_label(label: str) -> str:
    """Shorten method labels so comparisons fit in a shared legend."""

    replacements = {
        "Standard Thompson sampling": "Standard TS",
        "Mean-preserving temperature": "Temperature",
        "Observation-time forgetting": "Observation forgetting",
        "Calendar-time forgetting": "Calendar forgetting",
        "epsilon": "ε",
        "eta": "η",
        "rho": "ρ",
    }
    for source, target in replacements.items():
        label = label.replace(source, target)
    return label


def _method_specifications(
    dynamic: dict[str, Any],
) -> list[DynamicMethodSpecification]:
    representative = dynamic.get("representative_methods", {})
    scaled = [float(value) for value in representative.get("scaled_eta", [0.5, 2.0])]
    temperature = [
        float(value)
        for value in representative.get("temperature", [0.5, 2.0])
    ]
    epsilon = [float(value) for value in representative.get("epsilon", [0.1])]
    rho = [float(value) for value in representative.get("rho", [0.95])]
    specifications = [
        DynamicMethodSpecification(
            key="standard_ts",
            label="Standard Thompson sampling",
            model_kind="standard_ts",
            solver_kind="discrete",
        )
    ]
    specifications.extend(
        DynamicMethodSpecification(
            key=f"scaled_eta_{_compact_number(value)}",
            label=f"Scaled updates, eta={value:g}",
            model_kind="scaled_updates",
            solver_kind="discrete",
            parameter_value=value,
        )
        for value in scaled
    )
    specifications.extend(
        DynamicMethodSpecification(
            key=f"temperature_{_compact_number(value)}",
            label=f"Mean-preserving temperature, T={value:g}",
            model_kind="temperature",
            solver_kind="discrete",
            parameter_value=value,
        )
        for value in temperature
    )
    specifications.extend(
        DynamicMethodSpecification(
            key=f"epsilon_{_compact_number(value)}",
            label=f"Epsilon-greedy, epsilon={value:g}",
            model_kind="epsilon_greedy",
            solver_kind="discrete",
            parameter_value=value,
        )
        for value in epsilon
    )
    for value in rho:
        compact = _compact_number(value)
        specifications.extend(
            [
                DynamicMethodSpecification(
                    key=f"observation_forgetting_{compact}",
                    label=f"Observation-time forgetting, rho={value:g}",
                    model_kind="observation_forgetting",
                    solver_kind="continuous",
                    parameter_value=value,
                ),
                DynamicMethodSpecification(
                    key=f"calendar_forgetting_{compact}",
                    label=f"Calendar-time forgetting, rho={value:g}",
                    model_kind="calendar_forgetting",
                    solver_kind="continuous",
                    parameter_value=value,
                ),
            ]
        )
    if len({specification.key for specification in specifications}) != len(
        specifications
    ):
        raise ValueError("dynamic representative method keys must be unique")
    return specifications


def _model_for_specification(
    specification: DynamicMethodSpecification,
    primitives: SellerPrimitives,
) -> Any:
    value = specification.parameter_value
    if specification.model_kind == "standard_ts":
        return StandardThompsonSampling(primitives)
    if specification.model_kind == "scaled_updates":
        return ScaledUpdates(float(value), primitives)
    if specification.model_kind == "temperature":
        return MeanPreservingTemperature(float(value), primitives)
    if specification.model_kind == "epsilon_greedy":
        return EpsilonGreedy(float(value), primitives)
    if specification.model_kind == "observation_forgetting":
        return ObservationForgettingTS(float(value), primitives)
    if specification.model_kind == "calendar_forgetting":
        return CalendarForgettingTS(float(value), primitives)
    raise ValueError(f"unknown dynamic model kind {specification.model_kind!r}")


def make_common_random_numbers(
    *, paths: int, periods: int, seed: int
) -> CommonRandomNumbers:
    if paths <= 0 or periods <= 0:
        raise ValueError("paths and periods must be positive")
    rng = np.random.default_rng(seed)
    return CommonRandomNumbers(
        choice=rng.random((periods, paths)),
        outcome=rng.random((periods, paths)),
    )


def _random_number_digest(random_numbers: CommonRandomNumbers) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(random_numbers.choice).tobytes())
    digest.update(np.ascontiguousarray(random_numbers.outcome).tobytes())
    return digest.hexdigest()


def _dynamic_reuse_configuration_hash(
    config: dict[str, Any], dynamic: dict[str, Any]
) -> str:
    """Hash settings that must match when adding method specifications."""

    reusable_dynamic = dict(dynamic)
    reusable_dynamic.pop("representative_methods", None)
    return configuration_hash(
        {"seller": config.get("seller", {}), "dynamic_panels": reusable_dynamic}
    )


def _confidence_probability(confidence_level: float) -> float:
    confidence_level = float(confidence_level)
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0,1)")
    return 0.5 + confidence_level / 2.0


def _mean_interval(
    values: NDArray[np.generic], confidence_level: float
) -> tuple[float, float, float, float, int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    count = int(finite.size)
    if count == 0:
        return (float("nan"),) * 4 + (0,)
    mean = float(np.mean(finite))
    if count < 2:
        return mean, float("nan"), float("nan"), float("nan"), count
    standard_error = float(np.std(finite, ddof=1) / math.sqrt(count))
    critical = float(
        student_t.ppf(_confidence_probability(confidence_level), count - 1)
    )
    return (
        mean,
        standard_error,
        mean - critical * standard_error,
        mean + critical * standard_error,
        count,
    )


def _wilson_interval(
    successes: int, trials: int, confidence_level: float
) -> tuple[float, float, float, float, int]:
    if trials <= 0:
        return (float("nan"),) * 4 + (0,)
    successes = int(successes)
    trials = int(trials)
    proportion = successes / trials
    standard_error = math.sqrt(proportion * (1.0 - proportion) / trials)
    z = float(norm.ppf(_confidence_probability(confidence_level)))
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return (
        float(proportion),
        float(standard_error),
        float(max(0.0, center - half_width)),
        float(min(1.0, center + half_width)),
        trials,
    )


def _metric_columns(
    prefix: str,
    result: tuple[float, float, float, float, int],
) -> dict[str, float | int]:
    mean, standard_error, lower, upper, count = result
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_standard_error": standard_error,
        f"{prefix}_ci_lower": lower,
        f"{prefix}_ci_upper": upper,
        f"{prefix}_valid_count": count,
    }


def period_monte_carlo_statistics(
    *,
    demand_probability: FloatArray,
    chosen_a: BoolArray,
    action2: BoolArray,
    profit: FloatArray,
    posterior_mean: FloatArray,
    effective_sample_size: FloatArray,
    effective_concentration: FloatArray,
    confidence_level: float,
) -> dict[str, float | int]:
    """Contemporaneous ensemble statistics for one calendar period.

    Product-2 conditional use is computed only on paths where A is selected;
    an unselected path is not silently recoded as product 1.
    """

    chosen_a = np.asarray(chosen_a, dtype=bool)
    action2 = np.asarray(action2, dtype=bool)
    product2_used = chosen_a & action2
    conditional_actions = action2[chosen_a]
    result: dict[str, float | int] = {}
    result.update(
        _metric_columns(
            "realized_A_share",
            _wilson_interval(int(np.sum(chosen_a)), chosen_a.size, confidence_level),
        )
    )
    result.update(
        _metric_columns(
            "mean_demand_probability",
            _mean_interval(demand_probability, confidence_level),
        )
    )
    result.update(
        _metric_columns(
            "product2_share_conditional_on_A",
            _wilson_interval(
                int(np.sum(conditional_actions)),
                int(conditional_actions.size),
                confidence_level,
            ),
        )
    )
    result.update(
        _metric_columns(
            "product2_share_calendar",
            _wilson_interval(
                int(np.sum(product2_used)), product2_used.size, confidence_level
            ),
        )
    )
    result.update(
        _metric_columns(
            "product2_prescribed_share",
            _wilson_interval(int(np.sum(action2)), action2.size, confidence_level),
        )
    )
    result.update(
        _metric_columns(
            "calendar_profit", _mean_interval(profit, confidence_level)
        )
    )
    result.update(
        _metric_columns(
            "posterior_mean", _mean_interval(posterior_mean, confidence_level)
        )
    )
    result.update(
        _metric_columns(
            "effective_sample_size",
            _mean_interval(effective_sample_size, confidence_level),
        )
    )
    result.update(
        _metric_columns(
            "effective_concentration",
            _mean_interval(effective_concentration, confidence_level),
        )
    )
    result["A_selected_count"] = int(np.sum(chosen_a))
    result["product2_used_count"] = int(np.sum(product2_used))
    return result


def _series_metadata(
    *,
    panel_group: str,
    method_spec_key: str | None,
    method_spec_label: str | None,
    series_key: str,
    series_label: str,
    model: Any,
    primitives: SellerPrimitives,
    policy_solver: str,
    random_numbers: CommonRandomNumbers,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    metadata = model.metadata
    return {
        "panel_group": panel_group,
        "method_spec_key": method_spec_key or panel_group,
        "method_spec_label": method_spec_label or str(metadata["name"]),
        "series_key": series_key,
        "series_label": series_label,
        "method": metadata["method"],
        "parameter_name": metadata.get("parameter_name"),
        "parameter_value": metadata.get("parameter_value"),
        "p0": primitives.p0,
        "p1": primitives.p1,
        "p2": primitives.p2,
        "c1": primitives.c1,
        "c2": primitives.c2,
        "R": primitives.revenue,
        "gamma": primitives.gamma,
        "policy_solver": policy_solver,
        "paths": random_numbers.paths,
        "periods": random_numbers.periods,
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "aggregation": "contemporaneous cross-path Monte Carlo",
        "rolling_window": 0,
        "common_random_numbers": True,
    }


def simulate_discrete_dynamic_panel(
    *,
    model: Any,
    policy: LayerPolicy | BalancePolicy,
    primitives: SellerPrimitives,
    random_numbers: CommonRandomNumbers,
    seed: int,
    confidence_level: float,
    panel_group: str,
    series_key: str,
    series_label: str,
    policy_solver: str,
    method_spec_key: str | None = None,
    method_spec_label: str | None = None,
) -> pd.DataFrame:
    if model.primitives != primitives:
        raise ValueError("model and simulation primitives differ")
    if isinstance(policy, LayerPolicy):
        if policy.solution.primitives != primitives:
            raise ValueError("policy and simulation primitives differ")
        if policy.solution.report_diagonal < random_numbers.periods:
            raise ValueError("reported discrete policy does not cover the panel horizon")

    successes = np.zeros(random_numbers.paths, dtype=np.int64)
    failures = np.zeros(random_numbers.paths, dtype=np.int64)
    rows: list[dict[str, Any]] = []
    metadata = _series_metadata(
        panel_group=panel_group,
        method_spec_key=method_spec_key,
        method_spec_label=method_spec_label,
        series_key=series_key,
        series_label=series_label,
        model=model,
        primitives=primitives,
        policy_solver=policy_solver,
        random_numbers=random_numbers,
        seed=seed,
        confidence_level=confidence_level,
    )

    for offset in range(random_numbers.periods):
        demand = np.asarray(model.demand_counts(successes, failures), dtype=float)
        action2 = np.asarray(
            policy.action_for_counts(successes, failures), dtype=bool
        )
        posterior = np.asarray(
            model.posterior_mean_counts(successes, failures), dtype=float
        )
        effective_sample = np.asarray(
            model.effective_sample_size_counts(successes, failures), dtype=float
        )
        concentration = np.asarray(
            model.effective_concentration_counts(successes, failures), dtype=float
        )
        chosen = random_numbers.choice[offset] < demand
        success_probability = np.where(action2, primitives.p2, primitives.p1)
        success = chosen & (
            random_numbers.outcome[offset] < success_probability
        )
        failure = chosen & ~success
        profit = chosen * np.where(
            action2,
            primitives.revenue - primitives.c2,
            primitives.revenue - primitives.c1,
        )
        rows.append(
            {
                **metadata,
                "period": offset + 1,
                **period_monte_carlo_statistics(
                    demand_probability=demand,
                    chosen_a=chosen,
                    action2=action2,
                    profit=np.asarray(profit, dtype=float),
                    posterior_mean=posterior,
                    effective_sample_size=effective_sample,
                    effective_concentration=concentration,
                    confidence_level=confidence_level,
                ),
            }
        )
        successes += success
        failures += failure
    return pd.DataFrame(rows)


def simulate_continuous_dynamic_panel(
    *,
    model: Any,
    solution: Any,
    primitives: SellerPrimitives,
    random_numbers: CommonRandomNumbers,
    seed: int,
    confidence_level: float,
    panel_group: str,
    series_key: str,
    series_label: str,
    policy_solver: str,
    method_spec_key: str | None = None,
    method_spec_label: str | None = None,
    comparison_solution: Any | None = None,
) -> pd.DataFrame:
    if model.primitives != primitives or solution.primitives != primitives:
        raise ValueError("model, solution, and simulation primitives must agree")
    if (
        comparison_solution is not None
        and comparison_solution.primitives != primitives
    ):
        raise ValueError("comparison solution and simulation primitives differ")
    x = np.zeros(random_numbers.paths)
    y = np.zeros(random_numbers.paths)
    rho = float(model.rho)
    innovation = 1.0 - rho
    rows: list[dict[str, Any]] = []
    metadata = _series_metadata(
        panel_group=panel_group,
        method_spec_key=method_spec_key,
        method_spec_label=method_spec_label,
        series_key=series_key,
        series_label=series_label,
        model=model,
        primitives=primitives,
        policy_solver=policy_solver,
        random_numbers=random_numbers,
        seed=seed,
        confidence_level=confidence_level,
    )

    for offset in range(random_numbers.periods):
        demand = np.asarray(model.demand_xy(x, y), dtype=float)
        success_x, success_y = rho * x + innovation, rho * y
        failure_x, failure_y = rho * x, rho * y + innovation
        value_success = np.asarray(
            solution.grid.interpolate(solution.value, success_x, success_y),
            dtype=float,
        )
        value_failure = np.asarray(
            solution.grid.interpolate(solution.value, failure_x, failure_y),
            dtype=float,
        )
        advantage = -(primitives.c2 - primitives.c1) + (
            primitives.p2 - primitives.p1
        ) * primitives.gamma * (value_success - value_failure)
        action2 = advantage > 0.0
        comparison_statistics: dict[str, int | float] = {}
        if comparison_solution is not None:
            comparison_value_success = np.asarray(
                comparison_solution.grid.interpolate(
                    comparison_solution.value, success_x, success_y
                ),
                dtype=float,
            )
            comparison_value_failure = np.asarray(
                comparison_solution.grid.interpolate(
                    comparison_solution.value, failure_x, failure_y
                ),
                dtype=float,
            )
            comparison_advantage = -(primitives.c2 - primitives.c1) + (
                primitives.p2 - primitives.p1
            ) * primitives.gamma * (
                comparison_value_success - comparison_value_failure
            )
            comparison_action2 = comparison_advantage > 0.0
            fine_robust_action2 = advantage > solution.action_tolerance
            comparison_robust_action2 = (
                comparison_advantage > comparison_solution.action_tolerance
            )
            comparison_statistics = {
                "policy_grid_raw_action_disagreement_count": int(
                    np.count_nonzero(action2 != comparison_action2)
                ),
                "policy_grid_robust_action_disagreement_count": int(
                    np.count_nonzero(
                        fine_robust_action2 != comparison_robust_action2
                    )
                ),
                "policy_grid_action_comparison_count": int(action2.size),
            }
        a = 1.0 + x / innovation
        b = 1.0 + y / innovation
        posterior = a / (a + b)
        effective_sample = (x + y) / innovation
        concentration = a + b
        chosen = random_numbers.choice[offset] < demand
        success_probability = np.where(action2, primitives.p2, primitives.p1)
        success = chosen & (
            random_numbers.outcome[offset] < success_probability
        )
        failure = chosen & ~success
        profit = chosen * np.where(
            action2,
            primitives.revenue - primitives.c2,
            primitives.revenue - primitives.c1,
        )
        rows.append(
            {
                **metadata,
                "period": offset + 1,
                **comparison_statistics,
                **period_monte_carlo_statistics(
                    demand_probability=demand,
                    chosen_a=chosen,
                    action2=action2,
                    profit=np.asarray(profit, dtype=float),
                    posterior_mean=posterior,
                    effective_sample_size=effective_sample,
                    effective_concentration=concentration,
                    confidence_level=confidence_level,
                ),
            }
        )

        if bool(model.has_exact_idle_self_loop):
            idle_x, idle_y = x, y
        else:
            idle_x, idle_y = rho * x, rho * y
        x = np.where(success, success_x, np.where(failure, failure_x, idle_x))
        y = np.where(success, success_y, np.where(failure, failure_y, idle_y))
    return pd.DataFrame(rows)


def _discrete_policy_and_check(
    *,
    model: Any,
    primitives: SellerPrimitives,
    outer_grids: list[int],
    report_diagonal: int,
) -> tuple[LayerPolicy, dict[str, Any]]:
    if len(outer_grids) < 2:
        raise ValueError("dynamic panels require at least two discrete outer grids")
    solutions = [
        solve_discrete(
            model,
            primitives,
            DiscreteSolverConfig(
                outer_diagonal=outer,
                report_diagonal=report_diagonal,
            ),
        )
        for outer in outer_grids
    ]
    comparison = compare_discrete_solutions(
        solutions[-2], solutions[-1], report_diagonal
    )
    check = {
        "method": model.method,
        "parameter_name": model.parameter_name,
        "parameter_value": model.parameter_value,
        "p0": primitives.p0,
        "solver_type": "stationary_discrete_triangular",
        **asdict(comparison),
        "bellman_residual": solutions[-1].maximum_bellman_residual,
        "converged": bool(
            comparison.action_changes == 0
            and solutions[-1].maximum_bellman_residual <= 1e-9
        ),
    }
    return LayerPolicy(solutions[-1], allow_product1_tail=False), check


def _continuous_solution_and_check(
    *,
    model: Any,
    primitives: SellerPrimitives,
    resolutions: list[int],
    tolerance: float,
    max_iterations: int,
) -> tuple[Any, Any, dict[str, Any]]:
    if len(resolutions) < 2:
        raise ValueError("dynamic panels require at least two continuous grids")
    solutions = [
        solve_continuous(
            model,
            primitives,
            ContinuousSolverConfig(
                grid_size=resolution,
                tolerance=tolerance,
                max_iterations=max_iterations,
            ),
        )
        for resolution in resolutions
    ]
    comparison = compare_nested_solutions(solutions[-2], solutions[-1])
    final = solutions[-1]
    check = {
        "method": model.method,
        "parameter_name": model.parameter_name,
        "parameter_value": model.parameter_value,
        "p0": primitives.p0,
        "solver_type": "stationary_continuous_triangular_interpolation",
        **comparison.as_dict(),
        "bellman_residual": final.maximum_bellman_residual,
        "converged": bool(
            final.converged
            and final.maximum_bellman_residual <= max(5e-8, 20 * tolerance)
            and comparison.action_changes == 0
        ),
        "policy_grid_stable_for_panels": bool(
            final.converged
            and final.maximum_bellman_residual <= max(5e-8, 20 * tolerance)
            and comparison.action_change_share <= 1e-4
        ),
    }
    return final, solutions[-2], check


def quality_maintenance_mix(p0: float, p1: float, p2: float) -> float:
    """Return the feasible stationary mix that delivers expected quality ``p0``.

    ``nan`` denotes the economically infeasible case ``p0 > p2``. The seller
    primitives require ``p2 > p1``, but checking here keeps the plotting and
    reporting helper safe when it is used independently.
    """

    if p2 <= p1:
        raise ValueError("quality maintenance requires p2 > p1")
    if p0 > p2:
        return float("nan")
    return float(np.clip((p0 - p1) / (p2 - p1), 0.0, 1.0))


def summarize_dynamic_series(
    time_series: pd.DataFrame, late_window_start: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metadata_columns = [
        "panel_group",
        "method_spec_key",
        "method_spec_label",
        "series_key",
        "series_label",
        "method",
        "parameter_name",
        "parameter_value",
        "p0",
        "p1",
        "p2",
        "c1",
        "c2",
        "R",
        "gamma",
        "policy_solver",
        "paths",
        "periods",
        "seed",
        "confidence_level",
        "aggregation",
        "rolling_window",
        "common_random_numbers",
    ]
    mean_metrics = [
        "realized_A_share",
        "mean_demand_probability",
        "product2_share_calendar",
        "product2_prescribed_share",
        "calendar_profit",
        "posterior_mean",
        "effective_sample_size",
        "effective_concentration",
    ]
    for _, data in time_series.groupby("series_key", sort=False):
        data = data.sort_values("period")
        late = data[data.period >= late_window_start]
        row = {column: data[column].iloc[0] for column in metadata_columns}
        maintenance_mix = quality_maintenance_mix(
            float(row["p0"]), float(row["p1"]), float(row["p2"])
        )
        row["quality_maintenance_mix_feasible"] = bool(
            np.isfinite(maintenance_mix)
        )
        row["quality_maintenance_mix"] = maintenance_mix
        row["late_window_start"] = int(late_window_start)
        for metric in mean_metrics:
            row[f"overall_{metric}"] = float(data[f"{metric}_mean"].mean())
            row[f"late_{metric}"] = float(late[f"{metric}_mean"].mean())
        for prefix, subset in [("overall", data), ("late", late)]:
            selected = int(subset["A_selected_count"].sum())
            used2 = int(subset["product2_used_count"].sum())
            row[f"{prefix}_A_selected_count"] = selected
            row[f"{prefix}_product2_used_count"] = used2
            row[f"{prefix}_product2_share_conditional_on_A"] = (
                used2 / selected if selected else float("nan")
            )
            row[f"{prefix}_product2_gap_to_quality_maintenance_mix"] = (
                row[f"{prefix}_product2_share_conditional_on_A"]
                - maintenance_mix
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.edgecolor": "#cbd5e1",
            "axes.grid": True,
            "grid.color": "#e2e8f0",
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _plot_four_panel(
    frame: pd.DataFrame,
    *,
    series_order: list[str],
    colors: dict[str, Any],
    title: str,
    stem: Path,
    preliminary: bool,
    show_confidence_bands: bool,
) -> None:
    _configure_style()
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.6), sharex=True)
    specifications = [
        ("realized_A_share", "Realized Seller A market share", True),
        (
            "product2_share_conditional_on_A",
            "Realized product-2 use conditional on A",
            True,
        ),
        ("calendar_profit", "Seller A profit per calendar period", False),
        ("posterior_mean", "Mean belief used by the demand rule", False),
    ]
    handles: list[Any] = []
    labels: list[str] = []
    for series_key in series_order:
        data = frame[frame.series_key == series_key].sort_values("period")
        if data.empty:
            continue
        color = colors[series_key]
        label = str(data.series_label.iloc[0])
        for panel, (metric, panel_title, percent) in enumerate(specifications):
            ax = axes.ravel()[panel]
            line = ax.plot(
                data.period,
                data[f"{metric}_mean"],
                color=color,
                linewidth=1.25,
                label=label,
            )[0]
            if panel == 0:
                handles.append(line)
                labels.append(label)
            if show_confidence_bands:
                ax.fill_between(
                    data.period,
                    data[f"{metric}_ci_lower"],
                    data[f"{metric}_ci_upper"],
                    color=color,
                    alpha=0.08,
                    linewidth=0.0,
                )
            ax.set_title(panel_title, loc="left")
            if percent:
                ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    first = frame.iloc[0]
    p0 = float(first.p0)
    p1 = float(first.p1)
    p2 = float(first.p2)
    maintenance_mix = quality_maintenance_mix(p0, p1, p2)
    if np.isfinite(maintenance_mix):
        reference = axes[0, 1].axhline(
            maintenance_mix,
            color="#111827",
            linewidth=1.1,
            linestyle="--",
            alpha=0.8,
            label="Quality-maintenance mix",
        )
        handles.append(reference)
        labels.append("Quality-maintenance mix")
    else:
        axes[0, 1].text(
            0.98,
            0.96,
            "No feasible maintenance mix (p0 > p2)",
            transform=axes[0, 1].transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="#475569",
        )

    axes[0, 0].set_ylabel("Market share")
    axes[0, 1].set_ylabel("Conditional rate")
    axes[1, 0].set_ylabel("Profit")
    axes[1, 1].set_ylabel("Mean belief")
    axes[1, 0].set_xlabel("Calendar period")
    axes[1, 1].set_xlabel("Calendar period")
    for ax in (axes[0, 0], axes[0, 1], axes[1, 1]):
        ax.set_ylim(-0.03, 1.03)
    axes[1, 0].set_ylim(-0.03, 1.0)

    band_note = (
        "95% confidence bands shown"
        if show_confidence_bands
        else "95% CIs saved in CSV"
    )
    fig.suptitle(title, x=0.01, ha="left", y=0.99, fontsize=13)
    legend_y = 0.91 if preliminary else 0.945
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, legend_y),
        ncol=min(5, len(labels)),
        fontsize=8,
    )
    fig.text(
        0.5,
        0.015,
        (
            f"Contemporaneous Monte Carlo means; N={int(first.paths):,}, "
            f"horizon={int(first.periods)}, seed={int(first.seed)}; "
            f"no rolling or cumulative averaging; {band_note}."
        ),
        ha="center",
        fontsize=8,
        color="#475569",
    )
    if preliminary:
        fig.text(
            0.5,
            0.95,
            "PRELIMINARY — reduced simulation",
            ha="center",
            va="top",
            fontsize=9,
            color="#be123c",
            fontweight="bold",
        )
    top = 0.86 if preliminary else 0.89
    fig.tight_layout(rect=(0, 0.05, 1, top))
    _save_figure(fig, stem)


def _dynamic_figure_stems(
    methods_p0: pd.DataFrame,
    *,
    start: int = 13,
    suffix: str = "across_methods",
) -> list[str]:
    p0_values = sorted(float(value) for value in methods_p0.p0.unique())
    return [
        f"{number:02d}_dynamic_p0_{_compact_number(p0)}_{suffix}"
        for number, p0 in enumerate(p0_values, start=start)
    ]


def _method_color_map(method_keys: list[str]) -> dict[str, Any]:
    """Return stable colors, including a deterministic fallback for new specs."""

    fallback_keys = [key for key in method_keys if key not in DYNAMIC_METHOD_COLORS]
    fallback = plt.cm.tab20(np.linspace(0.0, 1.0, len(fallback_keys)))
    fallback_colors = dict(zip(fallback_keys, fallback, strict=True))
    return {
        key: DYNAMIC_METHOD_COLORS.get(key, fallback_colors.get(key))
        for key in method_keys
    }


def _make_p0_comparison_figures(
    methods_p0: pd.DataFrame,
    figures: Path,
    *,
    method_order: list[str],
    colors_by_method: dict[str, Any],
    stems: list[str],
    title_label: str,
    preliminary: bool,
) -> None:
    for stem_name, (p0, frame) in zip(
        stems,
        methods_p0.groupby("p0", sort=True),
        strict=True,
    ):
        ordered = (
            frame[["series_key", "method_spec_key"]]
            .drop_duplicates()
            .assign(
                method_order=lambda data: data.method_spec_key.map(
                    {key: index for index, key in enumerate(method_order)}
                )
            )
            .sort_values("method_order")
        )
        series_order = list(ordered.series_key)
        colors = {
            str(row.series_key): colors_by_method[str(row.method_spec_key)]
            for row in ordered.itertuples(index=False)
        }
        frame = frame.copy()
        frame["series_label"] = frame.method_spec_label.map(_concise_method_label)
        _plot_four_panel(
            frame,
            series_order=series_order,
            colors=colors,
            title=(
                "Average simulated dynamics under the optimal policy — "
                f"{title_label} at p0={float(p0):g}"
            ),
            stem=figures / stem_name,
            preliminary=preliminary,
            show_confidence_bands=False,
        )


def make_dynamic_panel_figures(
    methods_p0: pd.DataFrame,
    figures: Path,
    *,
    preliminary: bool,
) -> None:
    """Save the complete comparison and a focused, more readable subset."""

    required = {
        "method",
        "method_spec_key",
        "method_spec_label",
        "p0",
        "series_key",
    }
    missing = required.difference(methods_p0.columns)
    if missing:
        raise ValueError(f"dynamic panel table is missing columns: {sorted(missing)}")
    method_keys = list(methods_p0.method_spec_key.drop_duplicates())
    colors_by_method = _method_color_map(method_keys)

    # Preserve the original eight-specification comparison: additional epsilon
    # values belong only in the focused figures.
    complete = methods_p0[
        ~(
            (methods_p0.method == "epsilon_greedy")
            & (methods_p0.method_spec_key != "epsilon_0p1")
        )
    ]
    complete_order = [key for key in method_keys if key in set(complete.method_spec_key)]
    complete_stems = _dynamic_figure_stems(complete)
    _make_p0_comparison_figures(
        complete,
        figures / "dynamic",
        method_order=complete_order,
        colors_by_method=colors_by_method,
        stems=complete_stems,
        title_label="complete method comparison",
        preliminary=preliminary,
    )

    focused_methods = {
        "standard_ts",
        "scaled_updates",
        "observation_forgetting_ts",
    }
    focused = methods_p0[
        methods_p0.method.isin(focused_methods)
        | (methods_p0.method_spec_key == "epsilon_0p1")
    ]
    focused_order = [key for key in method_keys if key in set(focused.method_spec_key)]
    focused_stems = _dynamic_figure_stems(
        focused,
        start=18,
        suffix="focused_methods",
    )
    _make_p0_comparison_figures(
        focused,
        figures / "dynamic" / "focused",
        method_order=focused_order,
        colors_by_method=colors_by_method,
        stems=focused_stems,
        title_label="focused learning rules",
        preliminary=preliminary,
    )


def run_dynamic_panel_experiment(
    config: dict[str, Any],
    paths: ExperimentPaths,
    *,
    reuse_existing: bool = False,
) -> dict[str, pd.DataFrame]:
    """Solve selected policies, optionally reusing compatible saved series."""

    dynamic = config.get("dynamic_panels", {})
    if not bool(dynamic.get("enabled", False)):
        return {}
    paths.create()
    primitives = SellerPrimitives(**config.get("seller", {}))
    simulation_paths = int(dynamic["paths"])
    periods = int(dynamic["periods"])
    seed = int(dynamic["seed"])
    confidence_level = float(dynamic.get("confidence_level", 0.95))
    late_start = int(dynamic.get("late_window_start", max(1, periods - 49)))
    if not 1 <= late_start <= periods:
        raise ValueError("dynamic late_window_start must lie in the panel horizon")
    report_diagonal = max(periods, int(dynamic.get("report_diagonal", periods)))
    outer_grids = [int(value) for value in dynamic["discrete_outer_grids"]]
    if any(outer <= report_diagonal for outer in outer_grids):
        raise ValueError("dynamic outer grids must exceed report_diagonal")
    random_numbers = make_common_random_numbers(
        paths=simulation_paths, periods=periods, seed=seed
    )

    check_rows: list[dict[str, Any]] = []
    resolutions = [int(value) for value in dynamic["continuous_resolutions"]]
    tolerance = float(
        dynamic.get(
            "continuous_tolerance",
            config.get("continuous", {}).get("residual_tolerance", 1e-9),
        )
    )
    max_iterations = int(
        dynamic.get(
            "continuous_max_iterations",
            config.get("continuous", {}).get("max_iterations", 5000),
        )
    )
    p0_values = [float(value) for value in dynamic["p0_values"]]
    if not p0_values or len(set(p0_values)) != len(p0_values):
        raise ValueError("dynamic p0_values must be nonempty and unique")
    specifications = _method_specifications(dynamic)
    expected_series = {
        f"{specification.key}__p0_{_compact_number(p0)}"
        for specification in specifications
        for p0 in p0_values
    }
    existing_methods = pd.DataFrame()
    existing_checks = pd.DataFrame()
    existing_series: set[str] = set()
    if reuse_existing:
        timeseries_path = paths.results / "dynamic_methods_p0_timeseries.csv"
        checks_path = paths.results / "dynamic_solver_checks.csv"
        manifest_path = paths.results / "dynamic_panels_manifest.json"
        missing_paths = [
            path
            for path in (timeseries_path, checks_path, manifest_path)
            if not path.exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                "cannot reuse dynamic panels; missing "
                + ", ".join(str(path) for path in missing_paths)
            )
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        reuse_hash = _dynamic_reuse_configuration_hash(config, dynamic)
        if previous_manifest.get("reuse_configuration_hash") != reuse_hash:
            raise ValueError(
                "saved dynamic panels are incompatible with the current solver, "
                "simulation, p0, or seller settings"
            )
        if previous_manifest.get(
            "common_random_number_sha256"
        ) != _random_number_digest(random_numbers):
            raise ValueError("saved dynamic panels use different random numbers")
        existing_methods = pd.read_csv(timeseries_path)
        existing_checks = pd.read_csv(checks_path)
        metadata = {
            "paths": simulation_paths,
            "periods": periods,
            "seed": seed,
            "confidence_level": confidence_level,
        }
        for column, expected in metadata.items():
            values = pd.to_numeric(existing_methods[column], errors="coerce")
            if values.isna().any() or not np.allclose(values, expected):
                raise ValueError(
                    f"saved dynamic column {column!r} is incompatible with "
                    "the current config"
                )
        if existing_methods.duplicated(["series_key", "period"]).any():
            raise ValueError("saved dynamic panels contain duplicate series periods")
        expected_periods = np.arange(1, periods + 1)
        for series_key, frame in existing_methods.groupby("series_key"):
            actual_periods = np.sort(
                pd.to_numeric(frame.period, errors="coerce").to_numpy()
            )
            if not np.array_equal(actual_periods, expected_periods):
                raise ValueError(
                    f"saved dynamic series {series_key!r} is partial or malformed"
                )
        existing_series = set(existing_methods.series_key.astype(str).unique())
        unexpected = existing_series.difference(expected_series)
        if unexpected:
            raise ValueError(
                "saved dynamic panels contain specifications absent from the "
                f"current config: {sorted(unexpected)}"
            )
        check_keys = existing_checks.apply(
            lambda row: (
                f"{row['method_spec_key']}__p0_"
                f"{_compact_number(float(row['p0']))}"
            ),
            axis=1,
        )
        if check_keys.duplicated().any() or set(check_keys) != existing_series:
            raise ValueError(
                "saved dynamic solver checks do not match the reusable series"
            )
    frames: list[pd.DataFrame] = []
    for specification in specifications:
        for p0 in p0_values:
            series_key = f"{specification.key}__p0_{_compact_number(p0)}"
            if series_key in existing_series:
                continue
            seller = replace(primitives, p0=p0)
            model = _model_for_specification(specification, seller)
            if specification.solver_kind == "discrete":
                policy, check = _discrete_policy_and_check(
                    model=model,
                    primitives=seller,
                    outer_grids=outer_grids,
                    report_diagonal=report_diagonal,
                )
                policy_solver = "stationary discrete triangular DP"
                if (
                    specification.model_kind == "epsilon_greedy"
                    and math.isclose(p0, 0.5, rel_tol=0.0, abs_tol=1e-14)
                ):
                    balance = solve_epsilon_balance_policy(
                        float(specification.parameter_value),
                        seller,
                        half_width=max(1200, periods + 100),
                    )
                    disagreements = 0
                    for n in range(periods):
                        successes = np.arange(n + 1, dtype=np.int64)
                        failures = n - successes
                        disagreements += int(
                            np.count_nonzero(
                                balance.action_for_counts(successes, failures)
                                != policy.action_for_counts(successes, failures)
                            )
                        )
                    if disagreements:
                        raise RuntimeError(
                            "epsilon balance and triangular policies disagree on "
                            f"{disagreements} simulated-horizon states"
                        )
                    check["balance_policy_disagreements"] = disagreements
                    check["balance_bellman_residual"] = balance.bellman_residual
                simulation = simulate_discrete_dynamic_panel(
                    model=model,
                    policy=policy,
                    primitives=seller,
                    random_numbers=random_numbers,
                    seed=seed,
                    confidence_level=confidence_level,
                    panel_group="methods_across_p0",
                    method_spec_key=specification.key,
                    method_spec_label=specification.label,
                    series_key=series_key,
                    series_label=f"p0={p0:.2f}",
                    policy_solver=policy_solver,
                )
            elif specification.solver_kind == "continuous":
                solution, comparison_solution, check = _continuous_solution_and_check(
                    model=model,
                    primitives=seller,
                    resolutions=resolutions,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
                simulation = simulate_continuous_dynamic_panel(
                    model=model,
                    solution=solution,
                    primitives=seller,
                    random_numbers=random_numbers,
                    seed=seed,
                    confidence_level=confidence_level,
                    panel_group="methods_across_p0",
                    method_spec_key=specification.key,
                    method_spec_label=specification.label,
                    series_key=series_key,
                    series_label=f"p0={p0:.2f}",
                    policy_solver="stationary continuous interpolated DP",
                    comparison_solution=comparison_solution,
                )
                comparisons = int(
                    simulation["policy_grid_action_comparison_count"].sum()
                )
                raw_disagreements = int(
                    simulation[
                        "policy_grid_raw_action_disagreement_count"
                    ].sum()
                )
                robust_disagreements = int(
                    simulation[
                        "policy_grid_robust_action_disagreement_count"
                    ].sum()
                )
                check["visited_state_action_comparisons"] = comparisons
                check["visited_state_raw_action_disagreements"] = raw_disagreements
                check["visited_state_robust_action_disagreements"] = (
                    robust_disagreements
                )
                check["visited_state_raw_action_disagreement_share"] = (
                    raw_disagreements / comparisons
                )
                check["visited_state_robust_action_disagreement_share"] = (
                    robust_disagreements / comparisons
                )
                check["policy_stable_on_simulated_states"] = bool(
                    robust_disagreements == 0
                )
            else:
                raise ValueError(
                    f"unknown dynamic solver kind {specification.solver_kind!r}"
                )
            check["method_spec_key"] = specification.key
            check["method_spec_label"] = specification.label
            check_rows.append(check)
            frames.append(simulation)

    if existing_methods.empty and not frames:
        raise RuntimeError("dynamic panel run produced no time-series data")
    methods_p0 = pd.concat(
        [frame for frame in [existing_methods, *frames] if not frame.empty],
        ignore_index=True,
    )
    method_order = {
        specification.key: index
        for index, specification in enumerate(specifications)
    }
    methods_p0["_method_order"] = methods_p0.method_spec_key.map(method_order)
    methods_p0 = (
        methods_p0.sort_values(["_method_order", "p0", "period"])
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )
    panel_summary = summarize_dynamic_series(methods_p0, late_start)
    solver_checks = pd.concat(
        [
            frame
            for frame in [existing_checks, pd.DataFrame(check_rows)]
            if not frame.empty
        ],
        ignore_index=True,
        sort=False,
    )
    solver_checks["_method_order"] = solver_checks.method_spec_key.map(method_order)
    solver_checks = (
        solver_checks.sort_values(["_method_order", "p0"])
        .drop(columns="_method_order")
        .reset_index(drop=True)
    )

    tables = {
        "dynamic_methods_p0_timeseries": methods_p0,
        "dynamic_panel_summary": panel_summary,
        "dynamic_solver_checks": solver_checks,
    }
    for name, frame in tables.items():
        write_frame(paths.results / f"{name}.csv", frame)
    make_dynamic_panel_figures(
        methods_p0,
        paths.figures,
        preliminary=str(config["profile"]) != "full",
    )
    complete_figure_data = methods_p0[
        ~(
            (methods_p0.method == "epsilon_greedy")
            & (methods_p0.method_spec_key != "epsilon_0p1")
        )
    ]
    focused_figure_data = methods_p0[
        methods_p0.method.isin(
            {
                "standard_ts",
                "scaled_updates",
                "observation_forgetting_ts",
            }
        )
        | (methods_p0.method_spec_key == "epsilon_0p1")
    ]
    figure_groups = {
        "complete": _dynamic_figure_stems(complete_figure_data),
        "focused": _dynamic_figure_stems(
            focused_figure_data,
            start=18,
            suffix="focused_methods",
        ),
    }
    manifest = {
        "profile": config["profile"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_hash": configuration_hash(
            {"seller": config.get("seller", {}), "dynamic_panels": dynamic}
        ),
        "reuse_configuration_hash": _dynamic_reuse_configuration_hash(
            config, dynamic
        ),
        "paths": simulation_paths,
        "periods": periods,
        "seed": seed,
        "confidence_level": confidence_level,
        "common_random_numbers": True,
        "common_random_number_scope": "all method-p0 series",
        "common_random_number_sha256": _random_number_digest(random_numbers),
        "rolling_window": 0,
        "p0_configurations": len(p0_values),
        "method_specifications": len(specifications),
        "figure_grouping": "one figure per p0 in complete and focused sets",
        "color_encoding": "method specification, fixed across p0 figures",
        "method_specification_details": [
            asdict(specification) for specification in specifications
        ],
        "update_mode": "incremental_reuse" if reuse_existing else "full_recompute",
        "reused_method_p0_configurations": len(existing_series),
        "computed_method_p0_configurations": len(frames),
        "total_method_p0_configurations": int(methods_p0.series_key.nunique()),
        "tables": {name: len(frame) for name, frame in tables.items()},
        "figure_groups": figure_groups,
        "focused_method_specifications": list(
            focused_figure_data.method_spec_key.drop_duplicates()
        ),
        "figures": figure_groups["complete"] + figure_groups["focused"],
        "metric_definitions": {
            "realized_A_share": (
                "fraction of Monte Carlo paths selecting Seller A in the calendar period"
            ),
            "product2_share_conditional_on_A": (
                "product-2 uses divided by Seller-A selections in the calendar period; "
                "undefined when no path selects A"
            ),
            "product2_share_calendar": (
                "fraction of all paths that both select Seller A and use product 2"
            ),
            "calendar_profit": (
                "mean realized Seller-A profit across all paths, including zero when "
                "the outside option is selected"
            ),
            "posterior_mean": (
                "cross-path mean of the belief mean used by the demand rule at the "
                "start of the period"
            ),
        },
        "confidence_intervals": {
            "binary_proportions": "Wilson score, two-sided 95%",
            "cross_path_means": "Student-t, two-sided 95%",
        },
        "reference_figure_audit": {
            "legacy_title": "Exact discounted simulated paths",
            "legacy_curves_are": "contemporaneous cross-path Monte Carlo statistics",
            "legacy_rolling_window": 0,
            "legacy_cumulative_average": False,
            "legacy_default_paths": 400,
            "legacy_simulation_periods": 250,
            "legacy_finite_dp_horizon": 700,
            "new_policy_family": "modular stationary discounted policies",
        },
    }
    write_json(paths.results / "dynamic_panels_manifest.json", manifest)
    return tables


__all__ = [
    "CommonRandomNumbers",
    "make_common_random_numbers",
    "make_dynamic_panel_figures",
    "period_monte_carlo_statistics",
    "quality_maintenance_mix",
    "run_dynamic_panel_experiment",
    "simulate_continuous_dynamic_panel",
    "simulate_discrete_dynamic_panel",
    "summarize_dynamic_series",
]
