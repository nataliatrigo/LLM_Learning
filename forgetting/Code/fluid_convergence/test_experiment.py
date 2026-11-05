"""Small integration checks for schedules, checkpoints, and study comparisons."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from oblivious.oe_p0.code.policy import evaluate_policy as original_evaluate_policy
from oblivious.oe_p0.code.primitives import Parameters
from forgetting.Code.fluid_convergence import numerics as N
from forgetting.Code.fluid_convergence import run_experiment as RUN
from forgetting.Code.model import BoundaryGrid, Primitives, boundary_demand, solve_boundary


def small_config():
    config = copy.deepcopy(json.loads(RUN.DEFAULT_CONFIG.read_text()))
    config.update(
        p0_values=[.5], rhos=[.9], grid_intervals=[20, 40, 80],
        report_nodes=21, workers=1,
    )
    RUN.validate_config(config)
    return config


class FluidConvergenceExperimentTests(unittest.TestCase):
    def setUp(self):
        RUN.CANCEL.clear()

    def test_configuration_requires_nested_original_grids(self):
        config = small_config()
        self.assertEqual(config["grid_intervals"], [20, 40, 80])
        for intervals in ([20, 40], [20, 30, 80], [21, 42, 84], [20, 40, 40]):
            with self.subTest(intervals=intervals):
                invalid = copy.deepcopy(config)
                invalid["grid_intervals"] = intervals
                with self.assertRaises(ValueError):
                    RUN.validate_config(invalid)
        invalid = copy.deepcopy(config)
        invalid["rhos"] = [.5]
        with self.assertRaises(ValueError):
            RUN.validate_config(invalid)

    def test_checkpoints_reuse_compatible_data_and_reject_changed_inputs_or_arrays(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            store = RUN.Checkpoints(folder, {"numerics": "version-one"})
            settings = {"kind": "optimal", "n": 5}
            calls = []

            def calculate():
                calls.append(True)
                return {"value": np.arange(5, dtype=float), "residual": 0.0}

            original = store.get("solution", settings, calculate)
            reused = store.get("solution", settings, calculate)
            self.assertEqual(len(calls), 1)
            np.testing.assert_array_equal(reused["value"], original["value"])
            with self.assertRaisesRegex(ValueError, "Changed numerical inputs"):
                store.get("solution", dict(settings, n=7), calculate)
            changed_code = RUN.Checkpoints(folder, {"numerics": "version-two"})
            with self.assertRaisesRegex(ValueError, "Changed numerical inputs"):
                changed_code.get("solution", settings, calculate)
            with (folder / "solution.npz").open("wb") as stream:
                np.savez_compressed(stream, value=np.full(5, -1.0))
            with self.assertRaisesRegex(ValueError, "Changed checkpoint array"):
                store.get("solution", settings, calculate)
            self.assertEqual(len(calls), 1)

    def test_original_case_reports_finite_certified_losses_and_matches_old_solver(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            tables = RUN.case_worker(config, Path(temporary), .5, .9, {"test": "small"}, False)
            original = tables["original_summary"][0]
            self.assertEqual(set(tables), {
                "original_summary", "original_curves", "grid_refinement",
                "fluid_refinement", "solver_diagnostics",
            })
            self.assertEqual(len(tables["original_summary"]), 1)
            self.assertEqual(len(tables["original_curves"]), 21)
            self.assertEqual(len(tables["grid_refinement"]), 2)
            self.assertEqual(len(tables["fluid_refinement"]), 2)
            self.assertEqual(original["n_stochastic"], 81)
            self.assertEqual(original["n_fluid"], 81)
            for rows in tables.values():
                for row in rows:
                    for value in row.values():
                        if isinstance(value, (float, np.floating)):
                            self.assertTrue(math.isfinite(value))
            self.assertGreaterEqual(original["loss_min"], -original["solver_value_error_bound"] - 1e-10)
            self.assertLessEqual(
                original["solver_value_error_bound"],
                2 * config["solver"]["maximum_value_error_bound"],
            )
            for row in tables["original_curves"]:
                self.assertAlmostEqual(row["loss"], row["value_star"] - row["value_policy"], delta=1e-12)
            # Compare with the older, independent original value iteration.
            n = original["n_stochastic"]
            grid = np.linspace(0., 1., n)
            old = solve_boundary(
                Primitives(p0=.5, **config["parameters"]), .9,
                grid=BoundaryGrid(grid, n, 0, 9.), tolerance=1e-13,
            )
            self.assertAlmostEqual(original["value_star_x0"], float(old.value[n // 2]), delta=2e-9)

    def test_fluid_refinement_uses_one_original_kernel_and_one_final_candidate(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            tables = RUN.case_worker(config, output, .5, .9, {"test": "fixed-kernel"}, False)
            summary = tables["original_summary"][0]
            n_original, n_fluid = summary["n_stochastic"], summary["n_fluid"]
            final_candidate = [row for row in tables["solver_diagnostics"]
                               if row["kind"] == "evaluation" and row["n_fluid"] == n_fluid]
            self.assertEqual({row["n_stochastic"] for row in final_candidate}, {21, 41, 81})
            grid = np.linspace(0., 1., n_original)
            primitives = Primitives(p0=.5, **config["parameters"])
            params = Parameters(
                p1=primitives.p1, p2=primitives.p2, c1=primitives.c1,
                c2=primitives.c2, R=primitives.revenue, gamma=primitives.gamma, rho=.9,
            )
            demand = boundary_demand(grid, .9, primitives.p0)
            folder = output / "checkpoints" / "p0_0p5" / "rho_0p9"
            changes = []
            for row in tables["fluid_refinement"]:
                values = []
                for n in (row["n_coarse"], row["n_fine"]):
                    with np.load(folder / f"fluid_n{n}.npz", allow_pickle=False) as arrays:
                        fluid = {"grid": arrays["grid"], "gap": arrays["gap"]}
                        policy = N.transfer_policy(fluid, grid, config["policy_action_tolerance"])
                    values.append(original_evaluate_policy(grid, demand, policy, params)["value"])
                independent_change = float(np.max(np.abs(values[0] - values[1])))
                self.assertAlmostEqual(row["policy_value_change_sup"], independent_change, delta=2e-9)
                changes.append(independent_change)
            self.assertGreater(max(changes), 1e-4)


if __name__ == "__main__":
    unittest.main()
