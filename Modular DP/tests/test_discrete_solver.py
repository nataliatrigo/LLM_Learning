"""Regression and Bellman tests for the modular triangular solver."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = EXPERIMENT_ROOT.parent
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp.demand_models import (  # noqa: E402
    CalendarForgettingTS,
    StandardThompsonSampling,
)
from modular_dp.diagnostics import (  # noqa: E402
    compare_discrete_solutions,
)
from modular_dp.discrete_solver import (  # noqa: E402
    DiscreteSolverConfig,
    solve_discrete,
)
from modular_dp.primitives import SellerPrimitives  # noqa: E402


def _load_legacy_solver():
    path = REPOSITORY_ROOT / "discounted" / "paper_numerics" / "stationary_solver.py"
    spec = importlib.util.spec_from_file_location("paper_stationary_solver", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"could not import legacy solver at {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclass resolves annotations through sys.modules during execution.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class BaselineRegressionTests(unittest.TestCase):
    def test_full_reported_arrays_match_legacy_solver(self) -> None:
        legacy_module = _load_legacy_solver()
        primitives = SellerPrimitives()
        model = StandardThompsonSampling(primitives)
        config = DiscreteSolverConfig(outer_diagonal=80, report_diagonal=30)

        result = solve_discrete(model, primitives, config)
        legacy_parameters = legacy_module.Parameters(
            p0=primitives.p0,
            p1=primitives.p1,
            p2=primitives.p2,
            c1=primitives.c1,
            c2=primitives.c2,
            revenue=primitives.revenue,
            gamma=primitives.gamma,
        )
        legacy = legacy_module.solve_stationary_truncation(
            legacy_parameters,
            outer_diagonal=config.outer_diagonal,
            report_diagonal=config.report_diagonal,
        )

        for n in range(config.report_diagonal + 1):
            for key in ("value", "demand", "gap", "advantage"):
                np.testing.assert_allclose(
                    result.layers[n][key],
                    legacy["layers"][n][key],
                    rtol=0.0,
                    atol=2e-14,
                    err_msg=f"legacy mismatch for {key} on diagonal {n}",
                )
        self.assertLess(result.maximum_bellman_residual, 2e-14)

    def test_bellman_equation_and_advantage_identity(self) -> None:
        primitives = SellerPrimitives()
        result = solve_discrete(
            StandardThompsonSampling(primitives),
            primitives,
            DiscreteSolverConfig(outer_diagonal=70, report_diagonal=25),
        )
        for layer in result.layers.values():
            value = layer["value"]
            demand = layer["demand"]
            rhs = (
                primitives.gamma * (1.0 - demand) * value
                + demand * np.maximum(layer["q1"], layer["q2"])
            )
            np.testing.assert_allclose(value, rhs, rtol=0.0, atol=2e-14)
            np.testing.assert_allclose(
                layer["q2"] - layer["q1"],
                layer["advantage"],
                rtol=0.0,
                atol=2e-14,
            )

    def test_solver_rejects_non_self_loop_model(self) -> None:
        primitives = SellerPrimitives()
        with self.assertRaisesRegex(ValueError, "self_loop"):
            solve_discrete(
                CalendarForgettingTS(0.9, primitives),
                primitives,
                DiscreteSolverConfig(outer_diagonal=20, report_diagonal=5),
            )

    def test_solver_rejects_mismatched_primitives(self) -> None:
        model_primitives = SellerPrimitives(p0=0.4)
        solver_primitives = SellerPrimitives(p0=0.6)
        with self.assertRaisesRegex(ValueError, "primitives"):
            solve_discrete(
                StandardThompsonSampling(model_primitives),
                solver_primitives,
                DiscreteSolverConfig(outer_diagonal=20, report_diagonal=5),
            )


class DiscreteConvergenceTests(unittest.TestCase):
    def test_baseline_actions_stable_on_common_interior(self) -> None:
        primitives = SellerPrimitives()
        model = StandardThompsonSampling(primitives)
        coarse = solve_discrete(
            model,
            primitives,
            DiscreteSolverConfig(outer_diagonal=300, report_diagonal=20),
        )
        fine = solve_discrete(
            model,
            primitives,
            DiscreteSolverConfig(outer_diagonal=400, report_diagonal=20),
        )
        comparison = compare_discrete_solutions(coarse, fine)
        self.assertTrue(comparison.action_stable)
        self.assertEqual(comparison.maximum_demand_difference, 0.0)
        self.assertLess(comparison.maximum_value_difference, 2e-2)
        if comparison.last_active_diagonal is not None:
            self.assertGreater(comparison.distance_to_outer_boundary, 100)


if __name__ == "__main__":
    unittest.main()
