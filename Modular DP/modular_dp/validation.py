"""Validation gate executed before parameter sweeps."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import platform
import sys
import unittest

import numpy as np
import pandas as pd

from .continuous_solver import (
    ContinuousSolverConfig,
    compare_nested_solutions,
    solve_continuous,
)
from .demand_models import (
    CalendarForgettingTS,
    MeanPreservingTemperature,
    ObservationForgettingTS,
    ScaledUpdates,
    StandardThompsonSampling,
)
from .diagnostics import compare_discrete_solutions
from .discrete_solver import DiscreteSolverConfig, solve_discrete
from .outputs import ExperimentPaths, configuration_hash, write_frame, write_json
from .primitives import SellerPrimitives


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parent


def primitives_from_config(config: dict) -> SellerPrimitives:
    return SellerPrimitives(**config.get("seller", {}))


def run_unit_tests(verbosity: int = 1) -> tuple[bool, int, int, int]:
    """Run the stdlib suite without requiring an additional test dependency."""
    loader = unittest.TestLoader()
    suite = loader.discover(str(EXPERIMENT_ROOT / "tests"))
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return result.wasSuccessful(), result.testsRun, len(result.failures), len(result.errors)


def _check(
    rows: list[dict],
    name: str,
    metric: float,
    tolerance: float,
    *,
    relation: str = "<=",
    notes: str = "",
) -> None:
    if relation == "<=":
        passed = bool(np.isfinite(metric) and metric <= tolerance)
    elif relation == "==":
        passed = bool(metric == tolerance)
    else:
        raise ValueError(f"unknown validation relation {relation!r}")
    rows.append(
        {
            "check": name,
            "passed": passed,
            "metric": float(metric),
            "relation": relation,
            "tolerance": float(tolerance),
            "notes": notes,
        }
    )


def run_numerical_validation(config: dict, output_paths: ExperimentPaths) -> dict:
    """Validate the legacy baseline and one instance of each solver family."""
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from discounted.paper_numerics.stationary_solver import (  # noqa: PLC0415
        Parameters as LegacyParameters,
        solve_stationary_truncation as solve_legacy,
    )

    primitives = primitives_from_config(config)
    discrete_config = config["discrete"]
    outer_grids = tuple(int(value) for value in discrete_config["outer_grids"])
    report = int(discrete_config["report_diagonal"])
    rows: list[dict] = []

    standard_model = StandardThompsonSampling(primitives)
    standard_solutions = [
        solve_discrete(
            standard_model,
            primitives,
            DiscreteSolverConfig(outer_diagonal=outer, report_diagonal=report),
        )
        for outer in outer_grids
    ]
    largest = standard_solutions[-1]
    legacy = solve_legacy(
        LegacyParameters(
            p0=primitives.p0,
            p1=primitives.p1,
            p2=primitives.p2,
            c1=primitives.c1,
            c2=primitives.c2,
            revenue=primitives.revenue,
            gamma=primitives.gamma,
        ),
        largest.outer_diagonal,
        report,
    )
    for field in ["value", "demand", "gap", "advantage"]:
        difference = max(
            float(np.max(np.abs(largest.layers[n][field] - legacy["layers"][n][field])))
            for n in range(report + 1)
        )
        _check(rows, f"legacy_baseline_{field}", difference, 5e-13)
    _check(
        rows,
        "legacy_baseline_bellman_residual",
        largest.maximum_bellman_residual,
        1e-11,
    )
    discrete_comparison = compare_discrete_solutions(
        standard_solutions[-2], standard_solutions[-1], report
    )
    _check(
        rows,
        "baseline_policy_stability_outer_grids",
        discrete_comparison.action_changes,
        0,
        relation="==",
        notes=f"common interior n<={report}",
    )

    standard = standard_solutions[-1]
    for alias_name, alias_model in [
        ("eta_one", ScaledUpdates(1.0, primitives)),
        ("temperature_one", MeanPreservingTemperature(1.0, primitives)),
    ]:
        alias = solve_discrete(alias_model, primitives, largest.config)
        maximum = max(
            float(np.max(np.abs(alias.layers[n]["value"] - standard.layers[n]["value"])))
            for n in range(report + 1)
        )
        actions = sum(
            int(np.count_nonzero(alias.layers[n]["action2"] != standard.layers[n]["action2"]))
            for n in range(report + 1)
        )
        _check(rows, f"{alias_name}_value", maximum, 5e-13)
        _check(rows, f"{alias_name}_actions", actions, 0, relation="==")

    continuous_config = config["continuous"]
    rho = float(continuous_config.get("rho_values", [0.9])[0])
    resolutions = [int(value) for value in continuous_config["grid_resolutions"]]
    tolerance = float(continuous_config.get("residual_tolerance", 1e-9))
    max_iterations = int(continuous_config.get("max_iterations", 5000))
    continuous_records = []
    for model in [
        ObservationForgettingTS(rho, primitives),
        CalendarForgettingTS(rho, primitives),
    ]:
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
        for solution in solutions:
            _check(
                rows,
                f"{model.method}_grid_{solution.grid_size}_bellman_residual",
                solution.maximum_bellman_residual,
                max(5e-8, 20 * tolerance),
            )
            expected_form = (
                "exact_idle_self_loop_rearranged"
                if model.has_exact_idle_self_loop
                else "generic_idle_transition"
            )
            _check(
                rows,
                f"{model.method}_bellman_dispatch",
                0.0 if solution.bellman_form == expected_form else 1.0,
                0.0,
                relation="==",
                notes=solution.bellman_form,
            )
        comparison = compare_nested_solutions(solutions[-2], solutions[-1])
        # Coarse continuous policies need not be identical; stability is
        # assessed by the robust disagreement share and reported explicitly.
        _check(
            rows,
            f"{model.method}_robust_policy_grid_stability",
            comparison.action_change_share,
            0.02,
            notes=f"grid {comparison.coarse_grid_size}->{comparison.fine_grid_size}",
        )
        continuous_records.append(
            {
                "method": model.method,
                "rho": rho,
                **comparison.as_dict(),
            }
        )

    frame = pd.DataFrame(rows)
    passed = bool(frame.passed.all())
    output_paths.create()
    write_frame(output_paths.results / "validation_summary.csv", frame)
    write_frame(
        output_paths.results / "continuous_validation_convergence.csv",
        pd.DataFrame(continuous_records),
    )
    manifest = {
        "passed": passed,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "configuration_hash": configuration_hash(config),
        "seller": primitives.as_dict(),
        "python": platform.python_version(),
        "checks": int(len(frame)),
        "failed_checks": frame.loc[~frame.passed, "check"].tolist(),
    }
    write_json(output_paths.results / "validation_manifest.json", manifest)
    markdown = [
        "# Validation",
        "",
        f"Overall result: **{'PASS' if passed else 'FAIL'}**.",
        "",
        "| Check | Pass | Metric | Requirement | Notes |",
        "|---|---:|---:|---:|---|",
    ]
    for row in frame.itertuples(index=False):
        markdown.append(
            f"| {row.check} | {row.passed} | {row.metric:.6g} | "
            f"{row.relation} {row.tolerance:.6g} | {row.notes} |"
        )
    (output_paths.results / "VALIDATION.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )
    if not passed:
        failed = ", ".join(manifest["failed_checks"])
        raise RuntimeError(f"numerical validation failed: {failed}")
    return manifest


def validate_all(config: dict, output_paths: ExperimentPaths, verbosity: int = 1) -> dict:
    unit_passed, tests, failures, errors = run_unit_tests(verbosity=verbosity)
    if not unit_passed:
        raise RuntimeError(
            f"unit tests failed: {failures} failures, {errors} errors in {tests} tests"
        )
    manifest = run_numerical_validation(config, output_paths)
    manifest.update(
        {
            "unit_tests": tests,
            "unit_test_failures": failures,
            "unit_test_errors": errors,
        }
    )
    write_json(output_paths.results / "validation_manifest.json", manifest)
    return manifest
