"""Checks for threshold sweeps, cache isolation, and structural diagnostics.

Every file created here is confined to a temporary directory.  The cache
tests use solved, saved triangular grids rather than mocked metadata alone.
"""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from diagnostics import load_solution, save_solution
from model import Parameters, choose_cutoff, solve
import run_p0_comparison as comparison


def small_config():
    return {
        "primitives": asdict(Parameters(gamma=.8)),
        "solver": {
            "mesh_steps": [1., .5], "report_n": 2.,
            "truncation_tolerance": .5, "action_tolerance": 1e-8,
        },
        "simulation": {
            "horizon": 1., "dt": .1, "initial_states": [[0., 0.]],
            "check_time_refinement": True,
        },
        "p0_comparison": {"values": [.1, .3, .5, .7, .9]},
    }


class ThresholdConfigurationTests(unittest.TestCase):
    def test_valid_thresholds_keep_requested_order_without_mutation(self):
        config = small_config()
        config["p0_comparison"]["values"] = [.9, .1, .5]
        before = deepcopy(config)
        self.assertEqual(comparison.validate_config(config), [.9, .1, .5])
        self.assertEqual(config, before)

    def test_thresholds_require_nonempty_list(self):
        for raw in (None, [], (), ".1,.9", {}):
            config = small_config()
            config["p0_comparison"]["values"] = raw
            with self.subTest(values=raw), self.assertRaises(ValueError):
                comparison.validate_config(config)
        config = small_config()
        del config["p0_comparison"]
        with self.assertRaises(ValueError):
            comparison.validate_config(config)

    def test_thresholds_reject_nonfinite_outside_domain_and_booleans(self):
        for value in (float("nan"), float("inf"), -float("inf"),
                      0., 1., -.1, 1.1, True, False):
            config = small_config()
            config["p0_comparison"]["values"] = [.5, value]
            with self.subTest(value=value), self.assertRaises(ValueError):
                comparison.validate_config(config)

    def test_duplicate_values_and_colliding_directory_names_are_rejected(self):
        for values in ([.1, .1], [.1, "0.1"], [.12345671, .12345672]):
            config = small_config()
            config["p0_comparison"]["values"] = values
            with self.subTest(values=values), self.assertRaises(ValueError):
                comparison.validate_config(config)
        # Distinct floating-point inputs can share the default :g format.
        self.assertEqual(comparison.case_name(.12345671),
                         comparison.case_name(.12345672))

    def test_figure_requirements_reject_too_many_cases_or_missing_origin(self):
        config = small_config()
        config["p0_comparison"]["values"] = [.1, .2, .3, .4, .5, .6]
        with self.assertRaises(ValueError):
            comparison.validate_config(config)
        for states in ([], [[1., 1.]], [[0., 1.], [1., 0.]]):
            config = small_config()
            config["simulation"]["initial_states"] = states
            with self.subTest(states=states), self.assertRaises(ValueError):
                comparison.validate_config(config)

    def test_case_configuration_changes_only_p0_and_shares_no_nested_lists(self):
        config = small_config()
        before = deepcopy(config)
        case = comparison.build_case_config(config, .7)
        expected = deepcopy(before)
        expected["primitives"]["p0"] = .7
        self.assertEqual(case, expected)
        self.assertEqual(config, before)

        case["solver"]["mesh_steps"].append(.25)
        case["simulation"]["initial_states"][0][0] = 9.
        case["p0_comparison"]["values"].append(.8)
        case["primitives"]["p1"] = .2
        self.assertEqual(config, before)


class BaseGridCacheTests(unittest.TestCase):
    @staticmethod
    def save_base_grids(root, config):
        params = Parameters(**config["primitives"])
        settings = config["solver"]
        source_dir = root / "outputs" / "data"
        source_dir.mkdir(parents=True)
        steps = settings["mesh_steps"]
        for h in steps:
            cutoff = choose_cutoff(params, h, settings["report_n"],
                                   settings["truncation_tolerance"])
            solution = solve(params, h, cutoff, settings["report_n"],
                             action_tolerance=settings["action_tolerance"])
            name = "solution.npz" if h == steps[-1] else f"mesh_h_{h:g}.npz"
            save_solution(solution, source_dir / name)

    def test_identical_economics_and_solver_copy_both_real_saved_grids(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_base_grids(root, config)
            target = root / "comparison" / "p0_0.5"
            with patch.object(comparison, "ROOT", root):
                copied = comparison.seed_matching_base_grids(config, target)
            self.assertEqual(copied, ["mesh_h_1.npz", "solution.npz"])
            for name in copied:
                source = root / "outputs" / "data" / name
                destination = target / "data" / name
                self.assertEqual(destination.read_bytes(), source.read_bytes())
                restored = load_solution(destination)
                self.assertEqual(restored.params, Parameters(**config["primitives"]))
                self.assertEqual(restored.report_n, config["solver"]["report_n"])

    def test_different_threshold_economics_or_solver_do_not_reuse_base(self):
        base = small_config()
        mismatches = []
        for group, key, value in (("primitives", "p0", .7),
                                  ("primitives", "gamma", .75),
                                  ("solver", "report_n", 1.),
                                  ("solver", "truncation_tolerance", .01),
                                  ("solver", "action_tolerance", 1e-5)):
            case = deepcopy(base)
            case[group][key] = value
            mismatches.append((group, key, case))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_base_grids(root, base)
            with patch.object(comparison, "ROOT", root):
                for group, key, case in mismatches:
                    target = root / f"mismatch_{group}_{key}"
                    with self.subTest(group=group, key=key):
                        self.assertEqual(comparison.seed_matching_base_grids(case, target), [])
                        self.assertFalse((target / "data").exists())

    def test_existing_scenario_grid_is_preserved(self):
        config = small_config()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.save_base_grids(root, config)
            target = root / "comparison" / "p0_0.5"
            with patch.object(comparison, "ROOT", root):
                comparison.seed_matching_base_grids(config, target)
                paths = list((target / "data").glob("*.npz"))
                originals = {path.name: path.read_bytes() for path in paths}
                self.assertEqual(comparison.seed_matching_base_grids(config, target), [])
            self.assertEqual({path.name: path.read_bytes() for path in paths}, originals)


class ValueMonotonicityTests(unittest.TestCase):
    def test_actual_small_solves_have_strictly_decreasing_origin_values(self):
        origin_values = []
        for p0 in (.1, .3, .5, .7, .9):
            solution = solve(Parameters(p0=p0, gamma=.8),
                             h=1., cutoff_n=5., report_n=2.)
            origin_values.append(solution.values[0][0])
            self.assertTrue(np.isfinite(np.concatenate(solution.values)).all())
        self.assertTrue(np.all(np.diff(origin_values) < 0.))

    @staticmethod
    def save_values(directory, p0, values, *, h=1., p1=.35):
        case = directory / comparison.case_name(p0)
        (case / "data").mkdir(parents=True)
        metadata = {
            "parameters": asdict(Parameters(p0=p0, p1=p1)),
            "h": h, "report_n": 2., "cutoff_n": 5.,
        }
        np.savez_compressed(case / "data" / "solution.npz",
                            metadata=json.dumps(metadata),
                            values=np.asarray(values, dtype=float),
                            row_offsets=np.array([0, 1, 3, 6]))
        return case

    def test_monotone_values_are_checked_in_p0_order_on_every_node(self):
        lower_values = np.array([3., 2., 2.5, 1., 1.5, 2.])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lower = self.save_values(root, .1, lower_values)
            higher = self.save_values(root, .9, lower_values - .1)
            checks = comparison.check_value_monotonicity([higher, lower])
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0]["lower_p0"], .1)
        self.assertEqual(checks[0]["higher_p0"], .9)
        self.assertEqual(checks[0]["retained_states"], 6)
        self.assertEqual(checks[0]["violations_above_tolerance"], 0)
        self.assertEqual(checks[0]["maximum_positive_value_increase"], 0.)

    def test_increase_away_from_origin_is_detected(self):
        lower_values = np.array([3., 2., 2.5, 1., 1.5, 2.])
        higher_values = lower_values - .1
        higher_values[4] = lower_values[4] + .25
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lower = self.save_values(root, .1, lower_values)
            higher = self.save_values(root, .9, higher_values)
            checks = comparison.check_value_monotonicity([lower, higher])
        self.assertEqual(checks[0]["violations_above_tolerance"], 1)
        self.assertEqual(checks[0]["maximum_positive_value_increase"], .25)

    def test_increase_below_requested_tolerance_is_not_a_violation(self):
        values = np.ones(6)
        higher_values = values.copy()
        higher_values[5] += 2.5e-7
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lower = self.save_values(root, .1, values)
            higher = self.save_values(root, .9, higher_values)
            checks = comparison.check_value_monotonicity([lower, higher], tolerance=1e-6)
        self.assertEqual(checks[0]["violations_above_tolerance"], 0)
        self.assertGreater(checks[0]["maximum_positive_value_increase"], 0.)

    def test_incompatible_grids_or_other_primitives_are_rejected(self):
        for changes in ({"h": .5}, {"p1": .3}):
            with self.subTest(changes=changes), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                lower = self.save_values(root, .1, np.ones(6))
                higher = self.save_values(root, .9, np.zeros(6), **changes)
                with self.assertRaises(ValueError):
                    comparison.check_value_monotonicity([lower, higher])


if __name__ == "__main__":
    unittest.main()
