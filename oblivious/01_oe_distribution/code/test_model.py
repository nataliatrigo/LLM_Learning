"""Independent checks for OE decisions closed with the real joint marginal.

Run from the project root:
    .venv/bin/python -B -m unittest oblivious.oe_distribution.code.test_model -v

The small dense reference chain below is assembled from observed game events,
without calling the production transition or stationary-distribution helpers.
These checks concern the finite grid, not an exact continuous-state equilibrium.
"""

import csv
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution.code.model import JointMarginalExperiment
from oblivious.oe_distribution.code.diagnostics import demand_diagnostics, simulate_ts


def reference_joint_matrix(grid, demand, alpha, params):
    """Enumerate seller selection, Bernoulli outcome, and grid interpolation."""
    n = len(grid)
    matrix = np.zeros((n*n, n*n))

    def add_target(row, own, rival, seller, target, weight):
        left = int(np.clip(np.searchsorted(grid, target, side="right")-1, 0, n-2))
        upper_weight = (target-grid[left])/(grid[left+1]-grid[left])
        for new_own, interpolation_weight in ((left, 1-upper_weight),
                                               (left+1, upper_weight)):
            col = new_own*n+rival if seller == 0 else rival*n+new_own
            matrix[row, col] += weight*interpolation_weight

    for i in range(n):
        for j in range(n):
            row = i*n+j
            for seller, own, rival, selected in ((0, i, j, demand[i, j]),
                                                (1, j, i, demand[j, i])):
                for product, product_probability in ((0, 1-alpha[own]),
                                                      (1, alpha[own])):
                    success_probability = (params.p1, params.p2)[product]
                    for outcome, outcome_probability in ((0, 1-success_probability),
                                                           (1, success_probability)):
                        target = params.rho*grid[own]+(1-params.rho)*outcome
                        add_target(row, own, rival, seller, target,
                                   selected*product_probability*outcome_probability)
    return matrix


def reference_stationary(matrix):
    """Small well-conditioned dense linear system with normalization."""
    linear = matrix.T-np.eye(len(matrix))
    linear[-1] = 1.
    rhs = np.zeros(len(matrix))
    rhs[-1] = 1.
    mass = np.linalg.solve(linear, rhs)
    if mass.min() < -1e-12:
        raise AssertionError("Dense reference returned negative stationary mass")
    return mass


class JointMarginalEconomics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = JointMarginalExperiment(n=7, params=Parameters(rho=.6))
        cls.alpha = np.array([.15, .7, .2, .85, .1, .6, .3])
        cls.result = cls.experiment.evaluate(cls.alpha)
        cls.myopic = JointMarginalExperiment(n=7, params=Parameters(rho=.6, gamma=0.))
        cls.myopic_result = cls.myopic.evaluate(np.zeros(7))

    def test_real_joint_law_matches_independently_enumerated_calendar_chain(self):
        exp, result = self.experiment, self.result
        matrix = reference_joint_matrix(exp.grid, exp.game.D, self.alpha, exp.params)
        assert_allclose(matrix.sum(axis=1), 1., atol=3e-15)
        reference = reference_stationary(matrix).reshape(7, 7)
        assert_allclose(result["nu"], reference, atol=2e-11, rtol=0)
        assert_allclose(result["mu_a"], reference.sum(axis=1), atol=2e-11, rtol=0)
        assert_allclose(result["mu_b"], reference.sum(axis=0), atol=2e-11, rtol=0)
        assert_allclose(result["mu"], result["mu_b"], atol=2e-11, rtol=0)
        assert_allclose(result["mu_a"], result["mu_b"], atol=2e-11, rtol=0)
        self.assertLess(np.abs(result["nu"].ravel()@matrix-result["nu"].ravel()).sum(), 1e-11)
        # Both sellers cannot move in one calendar period.
        rows, cols = np.nonzero(matrix > 0)
        ai, bi, af, bf = rows//7, rows % 7, cols//7, cols % 7
        self.assertTrue(np.all((ai == af) | (bi == bf)))

    def test_real_joint_law_has_dependence_between_sellers(self):
        result = self.result
        self.assertGreater(np.abs(result["nu"]-np.outer(result["mu_a"], result["mu_b"])).sum(), .02)

    def test_optimization_uses_unconditional_marginal_not_conditional_demand(self):
        exp, result = self.experiment, self.result
        expected = exp.game.D@result["mu_b"]
        conditional = (result["nu"]*exp.game.D).sum(axis=1)/result["mu_a"]
        assert_allclose(result["demand"], expected, atol=2e-13, rtol=0)
        self.assertGreater(np.max(np.abs(expected-conditional)), .01)

    def test_calendar_marginal_is_not_selected_seller_distribution(self):
        exp, result = self.experiment, self.result
        selected = (result["nu"]*exp.game.D).sum(axis=1)
        self.assertAlmostEqual(selected.sum(), .5, places=12)
        selected /= selected.sum()
        self.assertGreater(np.abs(result["mu_a"]-selected).sum(), .2)
        # Symmetry alone does not mean the joint law factors into marginals.
        assert_allclose(result["nu"], result["nu"].T, atol=2e-11, rtol=0)
        # With product 1 everywhere, stationarity of the reputation update
        # implies E[X_A | A selected] = p1. The calendar mean is different.
        myopic = self.myopic_result
        flow = (myopic["nu"]*self.myopic.game.D).sum(axis=1)
        self.assertAlmostEqual(flow@self.myopic.grid/flow.sum(), self.myopic.params.p1, places=11)
        self.assertGreater(abs(myopic["mu_a"]@self.myopic.grid-self.myopic.params.p1), .05)

    def test_fractional_policy_values_use_expected_cost_and_unselected_periods(self):
        exp, result = self.experiment, self.result
        p, value, grid = exp.params, result["value"], exp.grid
        success_value = np.interp(p.rho*grid+1-p.rho, grid, value)
        failure_value = np.interp(p.rho*grid, grid, value)
        q1 = p.R-p.c1+p.gamma*(p.p1*success_value+(1-p.p1)*failure_value)
        q2 = p.R-p.c2+p.gamma*(p.p2*success_value+(1-p.p2)*failure_value)
        demand = result["demand"]
        updated = (p.gamma*(1-demand)*value
                   + demand*((1-self.alpha)*q1+self.alpha*q2))
        assert_allclose(value, updated, atol=2e-11, rtol=0)
        assert_allclose(result["advantage"], q2-q1, atol=2e-12, rtol=0)
        myopic = self.myopic.evaluate(self.alpha)
        assert_allclose(myopic["value"], myopic["demand"]*(
            p.R-p.c1-self.alpha*(p.c2-p.c1)), atol=2e-13, rtol=0)

    def test_myopic_equilibrium_has_analytical_best_response(self):
        exp, result = self.myopic, self.myopic_result
        self.assertTrue(result["verified"])
        assert_allclose(result["policy"], 0., atol=0, rtol=0)
        assert_allclose(result["value"], result["demand"]*(exp.params.R-exp.params.c1), atol=1e-12)
        assert_allclose(result["advantage"], -(exp.params.c2-exp.params.c1), atol=1e-13)

    def test_wrong_full_distribution_rejected_even_when_mean_is_identical(self):
        exp, result = self.myopic, self.myopic_result
        wrong = result["mu_b"].copy()
        shift = .1*min(wrong[1], wrong[2]/2, wrong[3])
        wrong[1:4] += shift*np.array([1., -2., 1.])
        assert_allclose(wrong.sum(), 1., atol=2e-15)
        assert_allclose(wrong@exp.grid, result["mu_b"]@exp.grid, atol=2e-15)
        self.assertGreater(np.abs(wrong-result["mu_b"]).sum(), 1e-3)
        rejected = exp.evaluate(np.zeros(7), conjectured_mu=wrong)
        self.assertFalse(rejected["verified"])
        assert_allclose(rejected["nu"], result["nu"], atol=2e-11, rtol=0)
        assert_allclose(rejected["demand"], exp.game.D@wrong, atol=2e-13, rtol=0)
        self.assertGreater(np.max(np.abs(rejected["marginal_residual"])), 1e-4)

    def test_consistent_but_suboptimal_mixing_is_not_equilibrium(self):
        # Closing the marginal is automatic in evaluate(alpha). It must not
        # conceal a strict incentive to deviate from a fractional action.
        result = self.myopic.evaluate(np.full(7, .4))
        self.assertFalse(result["verified"])
        self.assertLess(np.max(np.abs(result["marginal_residual"])), 1e-10)
        assert_allclose(result["advantage"], -.6, atol=2e-13, rtol=0)

    def test_invalid_policies_and_conjectures_are_rejected(self):
        for bad in (-.01, 1.01, np.nan, np.inf):
            with self.subTest(policy=bad), self.assertRaises(ValueError):
                self.myopic.evaluate(np.full(7, bad))
        for wrong in (np.ones(7), np.zeros(7), np.full(7, np.nan),
                      np.array([-.1, .1, .2, .2, .2, .2, .2])):
            with self.subTest(mu=wrong), self.assertRaises(ValueError):
                self.myopic.evaluate(np.zeros(7), conjectured_mu=wrong)


class JointMarginalDiagnostics(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.experiment = JointMarginalExperiment(n=7, params=Parameters(rho=.6))
        cls.result = cls.experiment.evaluate(np.linspace(.15, .85, 7))
        cls.simulation = simulate_ts(cls.experiment.game, cls.result, periods=500,
                                     burn_in=20, trajectories=2, seed=77312)

    def test_simulation_uses_both_draws_and_randomizes_actual_product(self):
        exp, a = self.experiment, self.simulation["arrays"]
        self.assertEqual(a["theta"].shape, (2, 500, 2))
        self.assertTrue(np.all((a["theta"] > 0) & (a["theta"] < 1)))
        assert_array_equal(a["selected"], a["theta"][:, :, 1] > a["theta"][:, :, 0])
        paths, times = np.indices(a["selected"].shape)
        sellers = a["selected"].astype(int)
        own = a["states"][paths, times, sellers]
        probability = np.interp(own, exp.grid, self.result["policy"])
        assert_allclose(a["chosen_probability"], probability, atol=1e-14, rtol=0)
        two = a["product_uniform"] < probability
        assert_array_equal(a["product"], 1+two)
        success_probability = exp.params.p1+(exp.params.p2-exp.params.p1)*two
        assert_array_equal(a["success"], a["success_uniform"] < success_probability)
        self.assertTrue(two.any() and (~two).any())
        expected = a["states"].copy()
        expected[paths, times, sellers] = exp.params.rho*own+(1-exp.params.rho)*a["success"]
        observed = np.concatenate((a["states"][:, 1:], a["final_state"][:, None]), axis=1)
        assert_allclose(observed, expected, atol=2e-14, rtol=0)
        expected_posterior_mean = (1-exp.params.rho+a["states"])/(3-2*exp.params.rho)
        assert_allclose(a["posterior_mean"], expected_posterior_mean, atol=2e-14, rtol=0)

    def test_simulation_compares_full_marginals_and_uses_distribution_labels(self):
        result, simulation = self.result, self.simulation
        a, summary = simulation["arrays"], simulation["summary"]
        marginals = np.array([result["nu"].sum(axis=1), result["nu"].sum(axis=0)])
        assert_allclose(a["stationary_marginals"], marginals, atol=2e-14)
        cdf_error = np.cumsum(a["empirical_mass"]-marginals, axis=1)
        assert_allclose(a["empirical_minus_stationary_cdf"], cdf_error, atol=2e-14)
        assert_allclose(summary["marginal_comparison"]["l1"],
                        np.abs(a["empirical_mass"]-marginals).sum(axis=1), atol=2e-14)
        self.assertNotIn("p0", summary)
        self.assertNotIn("stationary_F", summary)
        self.assertIn("stationary_posterior_means", summary)

    def test_conditional_demand_uses_joint_dependence(self):
        exp = self.experiment
        marginal = np.array([.05, .1, .15, .4, .15, .1, .05])
        correlated = dict(self.result, nu=np.diag(marginal), mu=marginal,
                          demand=exp.game.D@marginal)
        diagnostic = demand_diagnostics(exp.game, correlated)
        arrays, summary = diagnostic["arrays"], diagnostic["summary"]
        assert_allclose(arrays["d_conditional"], .5, atol=1e-13)
        assert_allclose(arrays["d_real_marginal"], exp.game.D@marginal, atol=1e-13)
        assert_allclose(arrays["error_conjectured_minus_real_marginal"], 0., atol=1e-13)
        self.assertGreater(np.max(np.abs(arrays["error_conjectured_minus_conditional"])), .1)
        self.assertEqual(summary["consistency_marginal_l1"], 0.)
        self.assertGreater(summary["joint_dependence"]["total_variation_from_product_marginals"], .5)

    def test_demand_reporting_accepts_boundary_roundoff_but_rejects_real_errors(self):
        # The actual rho=.999,n51 result has max(D@mu)=1+one ulp because
        # sum(mu)=1+one ulp. Reporting must retain this valid solver result.
        demand = self.result["demand"].copy()
        demand[0], demand[-1] = np.nextafter(0., -np.inf), np.nextafter(1., np.inf)
        candidate = dict(self.result, demand=demand)
        diagnostic = demand_diagnostics(self.experiment.game, candidate)
        expected = demand.copy()
        expected[0], expected[-1] = 0., 1.
        assert_allclose(diagnostic["arrays"]["d_optimized"], expected, atol=0, rtol=0)
        for bad in (-1e-8, 1+1e-8, np.nan):
            incorrect = demand.copy()
            incorrect[3] = bad
            with self.subTest(demand=bad), self.assertRaises(ValueError):
                demand_diagnostics(self.experiment.game, dict(self.result, demand=incorrect))


class JointMarginalSolver(unittest.TestCase):
    def test_myopic_solver_finds_unique_product_rule_from_fractional_start(self):
        exp = JointMarginalExperiment(n=7, params=Parameters(rho=.6, gamma=0.))
        result = exp.solve(initial_policy=np.full(7, .5), max_evaluations=300)
        self.assertTrue(result["verified"])
        assert_allclose(result["policy"], 0., atol=0, rtol=0)
        self.assertLess(result["marginal_residual"], 1e-10)

    def test_genuine_interior_equilibrium_has_indifference_and_real_marginal_closure(self):
        exp = JointMarginalExperiment(n=21, params=Parameters(rho=.97))
        result = exp.solve(max_evaluations=1500)
        self.assertTrue(result["verified"], result["summary"])
        alpha, value, p = result["policy"], result["value"], exp.params
        mixed = (alpha > 0) & (alpha < 1)
        self.assertTrue(mixed.any(), "This pilot has a genuine interior mixed solution")
        success_value = np.interp(p.rho*exp.grid+1-p.rho, exp.grid, value)
        failure_value = np.interp(p.rho*exp.grid, exp.grid, value)
        q1 = p.R-p.c1+p.gamma*(p.p1*success_value+(1-p.p1)*failure_value)
        q2 = p.R-p.c2+p.gamma*(p.p2*success_value+(1-p.p2)*failure_value)
        assert_allclose((q2-q1)[mixed], 0., atol=1e-10, rtol=0)
        gain = np.maximum(q1, q2)-((1-alpha)*q1+alpha*q2)
        self.assertLess(np.max(gain), 1e-10)
        bellman = (p.gamma*(1-result["demand"])*value
                   + result["demand"]*np.maximum(q1, q2))
        assert_allclose(value, bellman, atol=1e-10, rtol=0)
        matrix = reference_joint_matrix(exp.grid, exp.game.D, alpha, p)
        nu = result["nu"]
        self.assertLess(np.abs(nu.ravel()@matrix-nu.ravel()).sum(), 1e-11)
        assert_allclose(result["mu"], nu.sum(axis=0), atol=1e-11, rtol=0)
        assert_allclose(result["demand"], exp.game.D@nu.sum(axis=0), atol=1e-11, rtol=0)
        self.assertTrue(result["dp_diagnostics"]["verified"])
        self.assertTrue(result["stationary_diagnostics"]["verified"])


class JointMarginalWorkflow(unittest.TestCase):
    @staticmethod
    def resume_fixture(directory):
        original = {
            "oblivious/01_oe_distribution/code/model.py": "unchanged_numerical_source",
            "oblivious/01_oe_distribution/code/diagnostics.py": "previous_reporting_source",
            "oblivious/01_oe_distribution/code/run_experiment.py": "previous_runner_source",
            "oblivious/01_oe_distribution/code/test_model.py": "previous_test_source",
        }
        current = dict(original)
        for name in ("diagnostics.py", "run_experiment.py", "test_model.py"):
            current["oblivious/01_oe_distribution/code/"+name] = "updated_"+name
        case, diagnostic = "rho_0p6_n003", "rho_0p6_n003_diagnostics_0"
        (directory/"arrays").mkdir()
        for tag in (case, diagnostic):
            for suffix in ("json", "npz"):
                (directory/"arrays"/(tag+"."+suffix)).write_bytes((tag+suffix).encode())
        manifest = dict(source_sha256=original, configuration=dict(rhos=[.6], grids=[3]),
                        completed_cases=[case], completed_diagnostics=[diagnostic])
        return manifest, current

    def test_report_refresh_preserves_numerical_sources_cases_and_prior_report_hashes(self):
        from oblivious.oe_distribution.code import run_experiment as runner
        with tempfile.TemporaryDirectory(prefix="real_marginal_resume_") as temporary:
            directory = Path(temporary)
            manifest, current = self.resume_fixture(directory)
            original = deepcopy(manifest["source_sha256"])
            files = {str(p.relative_to(directory)): p.read_bytes() for p in (directory/"arrays").iterdir()}
            with patch.object(runner, "source_hashes", return_value=current):
                runner.prepare_resume(manifest, directory, refresh_diagnostics=True)
                self.assertEqual(manifest["source_sha256"], original)
                self.assertEqual(manifest["diagnostic_source_sha256"], current)
                self.assertEqual(manifest["completed_diagnostics"], [])
                self.assertEqual(manifest["completed_cases"], ["rho_0p6_n003"])
                history = manifest["resume_history"][-1]
                self.assertEqual(history["previous_source_sha256"], original)
                self.assertEqual(history["new_source_sha256"], current)
                for name, contents in files.items():
                    self.assertEqual((directory/name).read_bytes(), contents)
                    digest = hashlib.sha256(contents).hexdigest()
                    hashes = (history["superseded_report_sha256"] if "diagnostics" in name
                              else manifest["case_bundle_sha256"])
                    self.assertEqual(hashes[name], digest)
                # Interrupted report generation can resume normally using its
                # recorded reporting revision, while keeping solver provenance.
                manifest["completed_diagnostics"] = ["rho_0p6_n003_diagnostics_0"]
                runner.prepare_resume(manifest, directory)
                self.assertEqual(manifest["completed_diagnostics"], ["rho_0p6_n003_diagnostics_0"])
                self.assertEqual(manifest["source_sha256"], original)
                self.assertFalse(manifest["resume_history"][-1]["refresh_diagnostics"])

    def test_report_refresh_rejects_numerical_changes_and_unfinished_cases(self):
        from oblivious.oe_distribution.code import run_experiment as runner
        with tempfile.TemporaryDirectory(prefix="real_marginal_resume_") as temporary:
            directory = Path(temporary)
            original, current = self.resume_fixture(directory)
            changed_solver = dict(current)
            changed_solver["oblivious/01_oe_distribution/code/model.py"] = "changed_equilibrium_definition"
            for source, finished, message in ((changed_solver, True, "Numerical solver sources changed"),
                                               (current, False, "Finish every equilibrium case")):
                manifest = deepcopy(original)
                if not finished:
                    manifest["completed_cases"] = []
                before = deepcopy(manifest)
                with self.subTest(finished=finished), patch.object(runner, "source_hashes", return_value=source):
                    with self.assertRaisesRegex(SystemExit, message):
                        runner.prepare_resume(manifest, directory, refresh_diagnostics=True)
                self.assertEqual(manifest, before)
            manifest = deepcopy(original)
            with patch.object(runner, "source_hashes", return_value=current):
                with self.assertRaisesRegex(SystemExit, "Code changed since this run"):
                    runner.prepare_resume(manifest, directory)
            self.assertEqual(manifest, original)

    def test_report_refresh_detects_changed_saved_equilibrium_cases(self):
        from oblivious.oe_distribution.code import run_experiment as runner
        with tempfile.TemporaryDirectory(prefix="real_marginal_resume_") as temporary:
            directory = Path(temporary)
            manifest, current = self.resume_fixture(directory)
            with patch.object(runner, "source_hashes", return_value=current):
                runner.prepare_resume(manifest, directory, refresh_diagnostics=True)
                (directory/"arrays"/"rho_0p6_n003.npz").write_bytes(b"modified_candidate_arrays")
                before = deepcopy(manifest)
                with self.assertRaisesRegex(SystemExit, "Saved equilibrium cases changed"):
                    runner.prepare_resume(manifest, directory, refresh_diagnostics=True)
                self.assertEqual(manifest, before)

    def test_cli_keeps_code_data_figures_separate_and_protects_existing_run(self):
        root = Path(__file__).resolve().parents[3]
        with tempfile.TemporaryDirectory(prefix="real_marginal_cli_") as temporary:
            temporary = Path(temporary)
            environment = dict(os.environ, MPLBACKEND="Agg",
                               MPLCONFIGDIR=str(temporary/"mpl"),
                               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
            command = [sys.executable, "-B", "-m", "oblivious.oe_distribution.code.run_experiment",
                       "--output", str(temporary/"data"),
                       "--figures-output", str(temporary/"figures"), "--name", "check",
                       "--rhos", ".6", "--grids", "3", "--gamma", "0",
                       "--periods", "20", "--burn-in", "0", "--trajectories", "2"]
            completed = subprocess.run(command, cwd=root, env=environment,
                                       capture_output=True, text=True, timeout=60)
            self.assertEqual(completed.returncode, 0, completed.stdout+completed.stderr)
            data, figures = temporary/"data"/"check", temporary/"figures"/"check"
            original_manifest = (data/"manifest.json").read_bytes()
            manifest = json.loads(original_manifest)
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["verified_cases"], 1)
            self.assertEqual(manifest["total_cases"], 1)
            self.assertEqual(len(manifest["completed_cases"]), 1)
            self.assertEqual(len(manifest["completed_diagnostics"]), 1)
            report_path = data/"arrays"/(manifest["completed_diagnostics"][0]+".json")
            report = json.loads(report_path.read_text())
            self.assertEqual(set(report), {"rho", "n", "candidate", "verified",
                                          "simulation", "demand", "deviations"})
            self.assertEqual({path.name for path in data.glob("*_summary.json")},
                             {"simulation_summary.json", "demand_summary.json", "deviations_summary.json"})
            self.assertIn("full marginal", manifest["model"])
            self.assertIn("actual joint", manifest["model"])
            self.assertFalse((data/"figures").exists())
            self.assertFalse(list(data.rglob("*.png")))
            self.assertFalse(list(data.rglob("*.py")))
            self.assertEqual(len(list(figures.glob("*.png"))), 6)
            self.assertEqual(len(list(figures.glob("*.pdf"))), 6)
            self.assertFalse(list(figures.rglob("*.npz")))
            for filename in ("summary.csv", "candidates.csv"):
                with (data/filename).open() as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["verified"], "True")
                self.assertLess(float(rows[0]["marginal_residual"]), 1e-10)
                self.assertEqual(int(rows[0]["mixed_nodes"]), 0)
            with (data/"initializations.csv").open() as handle:
                starts = list(csv.DictReader(handle))
            self.assertEqual({row["start"] for row in starts}, {"product_1", "half", "product_2"})
            self.assertTrue(all(row["verified"] == "True" for row in starts))
            with np.load(data/"arrays"/"rho_0p6_n003.npz", allow_pickle=False) as stored:
                alpha = stored["root/selected/policy"]
                mu = stored["root/selected/mu"]
                nu = stored["root/selected/nu"]
                assert_array_equal(alpha, np.zeros(3))
                assert_allclose(mu, nu.sum(axis=0), atol=1e-12)
            repeated = subprocess.run(command, cwd=root, env=environment,
                                      capture_output=True, text=True, timeout=30)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertIn("Existing results protected", repeated.stdout+repeated.stderr)
            self.assertEqual((data/"manifest.json").read_bytes(), original_manifest)


if __name__ == "__main__":
    unittest.main()
