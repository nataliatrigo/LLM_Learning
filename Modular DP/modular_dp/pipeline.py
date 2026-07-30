"""End-to-end experiment pipeline for all specified learning rules."""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from .continuous_solver import (
    ContinuousDPSolution,
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
from .diagnostics import (
    classify_product2_policy,
    compare_discrete_solutions,
    compute_discrete_sensitivity,
)
from .discrete_solver import DiscreteSolution, DiscreteSolverConfig, solve_discrete
from .outputs import (
    ExperimentPaths,
    configuration_hash,
    write_frame,
    write_json,
)
from .plotting import make_all_figures
from .primitives import SellerPrimitives
from .simulation import (
    LayerPolicy,
    SimulationResult,
    concatenate_simulations,
    model_fields,
    simulate_continuous,
    simulate_discrete,
    solve_epsilon_balance_policy,
)


def _values(config: dict, name: str, defaults: list[float]) -> list[float]:
    return [float(value) for value in config.get("methods", {}).get(name, defaults)]


def build_discrete_models(config: dict, primitives: SellerPrimitives) -> list[Any]:
    """All displayed discrete configurations, including verified T=1/eta=1 aliases."""
    return [
        StandardThompsonSampling(primitives),
        *[
            ScaledUpdates(value, primitives)
            for value in _values(config, "scaled_eta", [0.25, 0.5, 1.0, 2.0, 4.0])
        ],
        *[
            MeanPreservingTemperature(value, primitives)
            for value in _values(config, "temperature", [0.25, 0.5, 1.0, 2.0, 4.0])
        ],
        *[
            EpsilonGreedy(value, primitives)
            for value in _values(config, "epsilon", [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
        ],
    ]


def build_forgetting_models(config: dict, primitives: SellerPrimitives) -> list[Any]:
    rho_values = _values(config, "rho", config.get("continuous", {}).get("rho_values", [0.9]))
    return [
        *[ObservationForgettingTS(rho, primitives) for rho in rho_values],
        *[CalendarForgettingTS(rho, primitives) for rho in rho_values],
    ]


def _base_summary(model: Any, primitives: SellerPrimitives) -> dict[str, Any]:
    return {
        **model_fields(model),
        "p0": primitives.p0,
        "p1": primitives.p1,
        "p2": primitives.p2,
        "c1": primitives.c1,
        "c2": primitives.c2,
        "R": primitives.revenue,
        "gamma": primitives.gamma,
    }


def _discrete_policy_records(
    solution: DiscreteSolution,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = _base_summary(solution.model, solution.primitives)
    diagonal_rows: list[dict[str, Any]] = []
    policy_rows: list[dict[str, Any]] = []
    localization = classify_product2_policy(solution)
    for row in localization.diagonals:
        diagonal_rows.append(
            {
                **fields,
                "outer_diagonal": solution.outer_diagonal,
                "report_diagonal": solution.report_diagonal,
                "n": row.n,
                "active": row.active,
                "product2_states": row.product2_states,
                "components": len(row.components),
                "lower_S": row.lower_s,
                "upper_S": row.upper_s,
                "lower_posterior_mean": row.lower_posterior_mean,
                "upper_posterior_mean": row.upper_posterior_mean,
                "maximum_continuation_gap": row.maximum_gap,
                "maximum_advantage": row.maximum_advantage,
                "action_tolerance": row.action_tolerance,
            }
        )
        layer = solution.layers[row.n]
        active = np.flatnonzero(layer["action2"])
        for successes in active:
            policy_rows.append(
                {
                    **fields,
                    "n": row.n,
                    "S": int(successes),
                    "F": int(row.n - successes),
                    "posterior_mean": solution.model.posterior_mean(
                        (int(successes), int(row.n - successes))
                    ),
                    "effective_sample_size": solution.model.effective_sample_size(
                        (int(successes), int(row.n - successes))
                    ),
                    "effective_concentration": solution.model.effective_concentration(
                        (int(successes), int(row.n - successes))
                    ),
                    "raw_action2": True,
                    "robust_action2": bool(
                        layer.get("robust_action2", layer["action2"])[successes]
                    ),
                    "numerical_tie": bool(
                        layer.get("tie", np.zeros_like(layer["action2"]))[
                            successes
                        ]
                    ),
                    "advantage": float(layer["advantage"][successes]),
                }
            )
    return diagonal_rows, policy_rows


def _sensitivity_records(
    model: Any,
    max_n: int,
    fit_start: int,
    fit_end: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = compute_discrete_sensitivity(
        model,
        max_n=max_n,
        fit_start=fit_start,
        fit_end=fit_end,
    )
    fields = model_fields(model)
    rows = [
        {
            **fields,
            "n": int(n),
            "g_n": float(g),
            "argmax_S": int(s),
            "argmax_posterior_mean": float(mean),
        }
        for n, g, s, mean in zip(
            result.n,
            result.g_n,
            result.argmax_s,
            result.argmax_posterior_mean,
            strict=True,
        )
    ]
    regression = {
        **fields,
        "estimated_sensitivity_slope": result.estimated_slope,
        "max_demand_sensitivity": result.max_demand_sensitivity,
        "fit_n_min": result.fit_start,
        "fit_n_max": result.fit_end,
        "fit_observations": result.fit_observations,
    }
    return rows, regression


def _tail_is_numerically_clear(solution: DiscreteSolution, clearance: int) -> bool:
    raw_active = [
        n for n, layer in solution.layers.items() if np.any(layer["action2"])
    ]
    if not raw_active:
        return True
    return max(raw_active) <= solution.report_diagonal - clearance


def _simulate_discrete_configuration(
    model: Any,
    solution: DiscreteSolution,
    grid_stable: bool,
    primitives: SellerPrimitives,
    simulation_config: dict,
    outer_buffer: int,
) -> tuple[SimulationResult, dict[str, Any]]:
    periods = int(simulation_config["periods"])
    paths = int(simulation_config["paths"])
    seed = int(simulation_config["seed"])
    representative = int(simulation_config["representative_paths"])
    thresholds = tuple(int(value) for value in simulation_config["late_thresholds"])
    late_start = int(simulation_config["late_window_start"])
    notes: dict[str, Any] = {}

    if isinstance(model, EpsilonGreedy):
        balance = solve_epsilon_balance_policy(
            model.epsilon,
            primitives,
            half_width=max(1200, min(4000, periods)),
        )
        disagreements = 0
        compared = 0
        for n in range(min(solution.report_diagonal, 150) + 1):
            successes = np.arange(n + 1, dtype=np.int64)
            failures = n - successes
            exact = balance.action_for_counts(successes, failures)
            disagreements += int(np.count_nonzero(exact != solution.layers[n]["action2"]))
            compared += n + 1
        if disagreements:
            raise RuntimeError(
                f"epsilon balance reduction disagrees in {disagreements}/{compared} common states"
            )
        policy = balance
        notes.update(
            {
                "simulation_policy_source": "exact_balance_reduction_p0_0p5",
                "balance_half_width": balance.half_width,
                "balance_bellman_residual": balance.bellman_residual,
                "balance_active_min_k": balance.active_min_k,
                "balance_active_max_k": balance.active_max_k,
                "balance_policy_disagreements": disagreements,
            }
        )
    else:
        clearance = min(100, max(20, solution.report_diagonal // 5))
        clear_tail = grid_stable and _tail_is_numerically_clear(solution, clearance)
        if clear_tail:
            policy = LayerPolicy(solution, allow_product1_tail=True)
            reported_localization = classify_product2_policy(solution)
            notes.update(
                {
                    "simulation_policy_source": "reported_grid_plus_numerically_inactive_product1_tail",
                    "tail_clearance": clearance,
                    "tail_extrapolation": True,
                    "simulation_policy_last_active_diagonal": reported_localization.last_active_diagonal,
                    "simulation_policy_last_active_censored": False,
                }
            )
        else:
            simulation_outer = max(solution.outer_diagonal, periods + outer_buffer)
            simulation_solution = solve_discrete(
                model,
                primitives,
                DiscreteSolverConfig(
                    outer_diagonal=simulation_outer,
                    report_diagonal=periods,
                ),
            )
            policy = LayerPolicy(simulation_solution, allow_product1_tail=False)
            simulation_localization = classify_product2_policy(simulation_solution)
            notes.update(
                {
                    "simulation_policy_source": "extended_triangular_grid",
                    "simulation_outer_diagonal": simulation_outer,
                    "tail_extrapolation": False,
                    "simulation_policy_last_active_diagonal": simulation_localization.last_active_diagonal,
                    "simulation_policy_last_active_censored": bool(
                        simulation_localization.last_active_diagonal == periods
                    ),
                }
            )

    result = simulate_discrete(
        model,
        policy,
        primitives,
        paths=paths,
        periods=periods,
        seed=seed,
        representative_paths=representative,
        late_thresholds=thresholds,
        late_window_start=late_start,
    )
    for key, value in notes.items():
        result.summary[key] = value
    return result, notes


def _continuous_policy_records(solution: ContinuousDPSolution) -> list[dict[str, Any]]:
    fields = _base_summary(solution.model, solution.primitives)
    active = np.flatnonzero(solution.action2)
    rho = float(solution.model.rho)
    innovation = 1.0 - rho
    x = solution.grid.x[active]
    y = solution.grid.y[active]
    a = 1.0 + x / innovation
    b = 1.0 + y / innovation
    mean = a / (a + b)
    effective = (x + y) / innovation
    return [
        {
            **fields,
            "grid_size": solution.grid.node_count,
            "grid_node_count": solution.grid.node_count,
            "grid_resolution": solution.grid_size,
            "x": float(x_value),
            "y": float(y_value),
            "posterior_mean": float(mean_value),
            "effective_sample_size": float(effective_value),
            "effective_concentration": float(effective_value + 2.0),
            "raw_action2": True,
            "robust_action2": bool(solution.robust_action2[index]),
            "numerical_tie": bool(solution.tie[index]),
            "advantage": float(solution.advantage[index]),
        }
        for index, x_value, y_value, mean_value, effective_value in zip(
            active, x, y, mean, effective, strict=True
        )
    ]


def _continuous_sensitivity(solution: ContinuousDPSolution) -> dict[str, Any]:
    model = solution.model
    rho = float(model.rho)
    innovation = 1.0 - rho
    x, y = solution.grid.x, solution.grid.y
    sensitivity = model.demand_xy(rho * x + innovation, rho * y) - model.demand_xy(
        rho * x, rho * y + innovation
    )
    index = int(np.argmax(sensitivity))
    state = (float(x[index]), float(y[index]))
    return {
        **model_fields(model),
        "grid_size": solution.grid.node_count,
        "grid_node_count": solution.grid.node_count,
        "grid_resolution": solution.grid_size,
        "max_demand_sensitivity": float(sensitivity[index]),
        "argmax_x": state[0],
        "argmax_y": state[1],
        "argmax_posterior_mean": float(model.posterior_mean(state)),
        "argmax_effective_sample_size": float(model.effective_sample_size(state)),
        "argmax_effective_concentration": float(
            model.effective_concentration(state)
        ),
    }


def _off_grid_residual(solution: ContinuousDPSolution, points: int = 160) -> float:
    # Deterministic low-discrepancy-like lattice, reflected into the simplex.
    index = np.arange(1, points + 1, dtype=float)
    x = np.mod(index * 0.6180339887498949, 1.0)
    y = np.mod(index * 0.4142135623730950, 1.0)
    reflected = x + y > 1.0
    x[reflected] = 1.0 - x[reflected]
    y[reflected] = 1.0 - y[reflected]
    return max(
        abs(solution.evaluate_state((float(xi), float(yi))).bellman_residual)
        for xi, yi in zip(x, y, strict=True)
    )


def _continuous_strong_changes(
    coarse: ContinuousDPSolution, fine: ContinuousDPSolution
) -> int:
    indices = coarse.grid.indices_in(fine.grid)
    coarse_advantage = coarse.advantage
    fine_advantage = fine.advantage[indices]
    tolerance = max(coarse.action_tolerance, fine.action_tolerance)
    opposite = ((coarse_advantage > tolerance) & (fine_advantage < -tolerance)) | (
        (coarse_advantage < -tolerance) & (fine_advantage > tolerance)
    )
    return int(np.count_nonzero(opposite))


def run_experiment(
    config: dict,
    paths: ExperimentPaths,
    *,
    validation_manifest: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run diagnostics, dynamic programs, simulations, outputs, and figures."""
    paths.create()
    primitives = SellerPrimitives(**config.get("seller", {}))
    discrete_cfg = config["discrete"]
    continuous_cfg = config["continuous"]
    simulation_cfg = {**config["simulation"], "seed": int(config["seed"])}
    outer_grids = [int(value) for value in discrete_cfg["outer_grids"]]
    report_diagonal = int(discrete_cfg["report_diagonal"])
    max_n = int(discrete_cfg["sensitivity_n_max"])
    fit_start = int(discrete_cfg["sensitivity_fit_min"])
    fit_end = int(discrete_cfg["sensitivity_fit_max"])

    summary_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    regression_rows: list[dict[str, Any]] = []
    diagonal_rows: list[dict[str, Any]] = []
    discrete_convergence_rows: list[dict[str, Any]] = []
    discrete_policy_rows: list[dict[str, Any]] = []
    continuous_convergence_rows: list[dict[str, Any]] = []
    continuous_policy_rows: list[dict[str, Any]] = []
    continuous_sensitivity_rows: list[dict[str, Any]] = []
    simulation_results: list[SimulationResult] = []
    start_time = time.perf_counter()

    discrete_models = build_discrete_models(config, primitives)
    for position, model in enumerate(discrete_models, start=1):
        fields = _base_summary(model, primitives)
        print(
            f"[discrete {position}/{len(discrete_models)}] {fields['method_label']}",
            flush=True,
        )
        model_start = time.perf_counter()
        sens_rows, regression = _sensitivity_records(
            model, max_n, fit_start, fit_end
        )
        sensitivity_rows.extend(sens_rows)
        regression_rows.append(regression)
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
        convergence = [
            compare_discrete_solutions(coarse, fine, report_diagonal)
            for coarse, fine in zip(solutions[:-1], solutions[1:], strict=True)
        ]
        for comparison in convergence:
            discrete_convergence_rows.append(
                {**fields, "comparison_scope": "base", **asdict(comparison)}
            )
        largest = solutions[-1]
        localization = classify_product2_policy(largest)
        comparison_scope = "base"
        last_comparison = convergence[-1]

        # A boundary censored by the common report interior is not used as
        # extinction evidence.  For configured full runs, solve and compare
        # two larger outer grids, then export that enlarged common interior.
        extended_report = discrete_cfg.get("extended_report_diagonal")
        extended_outers = [
            int(value) for value in discrete_cfg.get("extended_outer_grids", [])
        ]
        needs_extended_boundary = bool(
            not isinstance(model, EpsilonGreedy)
            and localization.last_active_diagonal == report_diagonal
            and extended_report is not None
            and len(extended_outers) >= 2
        )
        if needs_extended_boundary:
            extended_report = int(extended_report)
            if extended_report <= report_diagonal:
                raise ValueError(
                    "extended_report_diagonal must exceed report_diagonal"
                )
            if any(outer <= extended_report for outer in extended_outers):
                raise ValueError(
                    "each extended outer diagonal must exceed the extended report"
                )
            extended_solutions = [
                solve_discrete(
                    model,
                    primitives,
                    DiscreteSolverConfig(
                        outer_diagonal=outer,
                        report_diagonal=extended_report,
                    ),
                )
                for outer in extended_outers
            ]
            extended_convergence = [
                compare_discrete_solutions(coarse, fine, extended_report)
                for coarse, fine in zip(
                    extended_solutions[:-1], extended_solutions[1:], strict=True
                )
            ]
            for comparison in extended_convergence:
                discrete_convergence_rows.append(
                    {
                        **fields,
                        "comparison_scope": "extended_boundary",
                        **asdict(comparison),
                    }
                )
            largest = extended_solutions[-1]
            localization = classify_product2_policy(largest)
            last_comparison = extended_convergence[-1]
            comparison_scope = "extended_boundary"

        new_diagonals, new_policy = _discrete_policy_records(largest)
        diagonal_rows.extend(new_diagonals)
        discrete_policy_rows.extend(new_policy)
        last_active = localization.last_active_diagonal
        censored = bool(
            last_active is not None and last_active == largest.report_diagonal
        )
        converged = bool(
            largest.maximum_bellman_residual <= 1e-9
            and last_comparison.action_changes == 0
        )
        alias = None
        if isinstance(model, ScaledUpdates) and model.eta == 1.0:
            alias = "standard_ts"
        if isinstance(model, MeanPreservingTemperature) and model.temperature == 1.0:
            alias = "standard_ts"
        summary_rows.append(
            {
                **fields,
                "aliased_to": alias,
                "solver_type": "discrete_triangular_truncation",
                "grid_size": sum(range(largest.report_diagonal + 2)),
                "grid_node_count": sum(range(largest.report_diagonal + 2)),
                "grid_resolution": np.nan,
                "grid_size_unit": "reported triangular states",
                "outer_diagonal": largest.outer_diagonal,
                "report_diagonal": largest.report_diagonal,
                "comparison_scope": comparison_scope,
                "bellman_residual": largest.maximum_bellman_residual,
                "action_changes": last_comparison.action_changes,
                "raw_action_changes": last_comparison.raw_action_changes,
                "last_active_diagonal": last_active,
                "last_active_censored": censored,
                "distance_from_last_reported_active_region_to_outer_boundary": localization.distance_to_outer_boundary,
                "estimated_sensitivity_slope": regression["estimated_sensitivity_slope"],
                "max_demand_sensitivity": regression["max_demand_sensitivity"],
                "converged": converged,
                "notes": "; ".join(
                    note
                    for note in [
                        "alias verified" if alias else "",
                        (
                            "extended boundary compared and exported"
                            if comparison_scope == "extended_boundary"
                            else ""
                        ),
                    ]
                    if note
                ),
                "runtime_seconds": time.perf_counter() - model_start,
            }
        )
        simulation, _ = _simulate_discrete_configuration(
            model,
            largest,
            grid_stable=last_comparison.action_changes == 0,
            primitives=primitives,
            simulation_config=simulation_cfg,
            outer_buffer=max(
                800, largest.outer_diagonal - largest.report_diagonal
            ),
        )
        simulation_results.append(simulation)

    forgetting_models = build_forgetting_models(config, primitives)
    base_resolutions = [int(value) for value in continuous_cfg["grid_resolutions"]]
    selected_high = continuous_cfg.get("selected_convergence_resolution")
    selected_rho = {float(value) for value in continuous_cfg.get("selected_convergence_rho", [])}
    selected_maps = {
        float(value)
        for value in continuous_cfg.get(
            "selected_map_rho", _values(config, "rho", [0.9])
        )
    }
    tolerance = float(continuous_cfg.get("residual_tolerance", 1e-9))
    max_iterations = int(continuous_cfg.get("max_iterations", 5000))

    for position, model in enumerate(forgetting_models, start=1):
        fields = _base_summary(model, primitives)
        print(
            f"[continuous {position}/{len(forgetting_models)}] {fields['method_label']}",
            flush=True,
        )
        model_start = time.perf_counter()
        resolutions = list(base_resolutions)
        if selected_high is not None and model.rho in selected_rho:
            resolutions.append(int(selected_high))
        resolutions = sorted(set(resolutions))
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
        comparisons = []
        for coarse, fine in zip(solutions[:-1], solutions[1:], strict=True):
            comparison = compare_nested_solutions(coarse, fine)
            strong_changes = _continuous_strong_changes(coarse, fine)
            row = {
                **fields,
                **comparison.as_dict(),
                "strong_opposite_action_changes": strong_changes,
                "cells_per_transition_coarse": (1.0 - model.rho) * (coarse.grid_size - 1),
                "cells_per_transition_fine": (1.0 - model.rho) * (fine.grid_size - 1),
            }
            continuous_convergence_rows.append(row)
            comparisons.append((comparison, strong_changes))
        primary = solutions[-1]
        sensitivity = _continuous_sensitivity(primary)
        continuous_sensitivity_rows.append(sensitivity)
        off_grid_residual = _off_grid_residual(primary)
        if model.rho in selected_maps:
            continuous_policy_rows.extend(_continuous_policy_records(primary))
        last_comparison, strong_changes = comparisons[-1]
        converged = bool(
            primary.converged
            and primary.maximum_bellman_residual <= max(5e-8, 20 * tolerance)
            and last_comparison.action_changes == 0
        )
        summary_rows.append(
            {
                **fields,
                "aliased_to": None,
                "solver_type": "continuous_triangular_interpolation",
                "grid_size": primary.grid.node_count,
                "grid_node_count": primary.grid.node_count,
                "grid_resolution": primary.grid_size,
                "grid_size_unit": "triangular interpolation nodes",
                "outer_diagonal": np.nan,
                "report_diagonal": np.nan,
                "comparison_scope": "continuous_nested_grid",
                "bellman_residual": primary.maximum_bellman_residual,
                "off_grid_bellman_residual": off_grid_residual,
                "action_changes": last_comparison.action_changes,
                "raw_action_changes": last_comparison.raw_action_changes,
                "strong_opposite_action_changes": strong_changes,
                "last_active_diagonal": np.nan,
                "last_active_censored": np.nan,
                "distance_from_last_reported_active_region_to_outer_boundary": np.nan,
                "estimated_sensitivity_slope": np.nan,
                "max_demand_sensitivity": sensitivity["max_demand_sensitivity"],
                "converged": converged,
                "notes": f"{primary.bellman_form}; barycentric interpolation",
                "runtime_seconds": time.perf_counter() - model_start,
            }
        )
        simulation_results.append(
            simulate_continuous(
                model,
                primary,
                primitives,
                paths=int(simulation_cfg["paths"]),
                periods=int(simulation_cfg["periods"]),
                seed=int(simulation_cfg["seed"]),
                representative_paths=int(simulation_cfg["representative_paths"]),
                late_thresholds=tuple(int(value) for value in simulation_cfg["late_thresholds"]),
                late_window_start=int(simulation_cfg["late_window_start"]),
            )
        )

    simulation_tables = concatenate_simulations(simulation_results)
    summary = pd.DataFrame(summary_rows)
    simulation_summary = simulation_tables["summary"]
    simulation_metrics = [
        column
        for column in simulation_summary.columns
        if column not in summary.columns or column == "method_key"
    ]
    summary = summary.merge(
        simulation_summary[simulation_metrics], on="method_key", how="left"
    )
    tables = {
        "summary": summary,
        "demand_sensitivity": pd.DataFrame(sensitivity_rows),
        "sensitivity_regressions": pd.DataFrame(regression_rows),
        "discrete_diagonals": pd.DataFrame(diagonal_rows),
        "discrete_convergence": pd.DataFrame(discrete_convergence_rows),
        "discrete_policy_points": pd.DataFrame(discrete_policy_rows),
        "continuous_convergence": pd.DataFrame(continuous_convergence_rows),
        "continuous_policy_points": pd.DataFrame(continuous_policy_rows),
        "continuous_sensitivity": pd.DataFrame(continuous_sensitivity_rows),
        "simulation_summary": simulation_summary,
        "simulation_paths": simulation_tables["path_summary"],
        "simulation_calendar": simulation_tables["calendar_series"],
        "simulation_observation": simulation_tables["observation_series"],
        "representative_trajectories": simulation_tables["representative_trajectories"],
        "simulation_late_use": simulation_tables["late_use"],
    }

    compressed = {"discrete_policy_points", "continuous_policy_points", "representative_trajectories"}
    for name, frame in tables.items():
        suffix = ".csv.gz" if name in compressed else ".csv"
        write_frame(paths.results / f"{name}{suffix}", frame)
    make_all_figures(tables, paths.figures)
    manifest = {
        "profile": config["profile"],
        "configuration_hash": configuration_hash(config),
        "validation_configuration_hash": (
            validation_manifest.get("configuration_hash")
            if validation_manifest is not None
            else None
        ),
        "validation_passed": (
            bool(validation_manifest.get("passed"))
            if validation_manifest is not None
            else None
        ),
        "validation_unit_tests": (
            validation_manifest.get("unit_tests")
            if validation_manifest is not None
            else None
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "seller": primitives.as_dict(),
        "discrete_configurations": len(discrete_models),
        "forgetting_configurations": len(forgetting_models),
        "economically_distinct_configurations": len(discrete_models) + len(forgetting_models) - 2,
        "runtime_seconds": time.perf_counter() - start_time,
        "tables": {name: len(frame) for name, frame in tables.items()},
    }
    write_json(paths.results / "run_manifest.json", manifest)
    return tables


__all__ = [
    "build_discrete_models",
    "build_forgetting_models",
    "run_experiment",
]
