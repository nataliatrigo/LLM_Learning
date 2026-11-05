"""Small tests for the isolated retry's provenance, interruption and reuse."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np


SPEC = importlib.util.spec_from_file_location("test_isolated_oe_retry",
    Path(__file__).with_name("retry_oe099.py"))
RETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RETRY)
RUN = RETRY.RUN
ORIGINAL_SOURCES = {"original.py": "original-v1"}
RETRY_SOURCES = dict(ORIGINAL_SOURCES, retry="retry-v1")


def candidate(verified, action=0.):
    return dict(grid=np.linspace(0, 1, 101), policy=np.full(101, action),
        nu=np.full((101, 101), 1 / 101 ** 2),
        verified=verified, candidate_verified=verified,
        dp_diagnostics=dict(verified=verified, conditional_regret=0. if verified else .01),
        stationary_diagnostics=dict(verified=True))


def prepare(source):
    config = dict(parameters=dict(R=1., c1=.05, c2=.65, p1=.35, p2=.8, gamma=.98),
        order=128, tol=1e-7, dp_tol=1e-10, periods=100, burn_in=10,
        replications=2, seed=1234)
    manifest = dict(configuration=config, source_sha256=ORIGINAL_SOURCES,
                    fingerprint=RUN.fingerprint(config, ORIGINAL_SOURCES))
    RUN.atomic_json(source / "manifest.json", manifest)
    checkpoint = dict(n=101, rho=.99, selected=candidate(False),
        attempts=[dict(start="other", candidate=candidate(False, 1.))],
        configuration=dict(n=101, parameters=dict(config["parameters"], rho=.99), order=128),
        demand_audit=dict(verified=True), campaign_fingerprint=manifest["fingerprint"])
    RUN.save_checked(RUN.case_dir(source, "oe", .99, 101) / "solution", checkpoint)
    return manifest


class RetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.source = Path(self.temp.name) / "original"
        self.output = Path(self.temp.name) / "retry"
        self.original = prepare(self.source)
        self.before = {path.relative_to(self.source): path.read_bytes()
                       for path in self.source.rglob("*") if path.is_file()}
        self.patches = [patch.object(RUN, "source_hashes", return_value=ORIGINAL_SOURCES),
            patch.object(RETRY, "retry_sources", return_value=RETRY_SOURCES),
            patch.object(RUN, "experiment_for", side_effect=self.experiment)]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    @staticmethod
    def experiment(model, n, params, config):
        return SimpleNamespace(grid=np.linspace(0, 1, n), params=params,
            game=SimpleNamespace(D=np.full((n, n), .5)),
            configuration=lambda: dict(n=n, parameters=dict(config["parameters"], rho=.99),
                order=config["order"]))

    def execute(self, solver, **kwargs):
        with patch.object(RETRY, "module", return_value=SimpleNamespace(solve_active_set=solver)), \
                redirect_stdout(io.StringIO()):
            return RETRY.execute_retry(self.source, self.output, **kwargs)

    def test_stops_at_certificate_preserves_original_and_records_own_provenance(self):
        calls = []

        def solve(experiment, policy, **kwargs):
            calls.append((policy.copy(), kwargs))
            return candidate(True, .5)

        self.assertEqual(self.execute(solve), 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1]["max_evaluations"], 150)
        after = {path.relative_to(self.source): path.read_bytes()
                 for path in self.source.rglob("*") if path.is_file()}
        self.assertEqual(after, self.before)
        manifest = json.loads((self.output / "manifest.json").read_text())
        saved = RUN.load_checked(self.output / "solution")
        self.assertEqual(manifest["status"], "complete")
        self.assertNotEqual(saved["campaign_fingerprint"], self.original["fingerprint"])
        self.assertEqual(saved["campaign_fingerprint"], manifest["fingerprint"])
        self.assertEqual(saved["source_campaign_fingerprint"], self.original["fingerprint"])
        self.assertTrue(saved["demand_audit"]["verified"])
        self.assertAlmostEqual(saved["predicted"]["product2_market"], .5)
        self.assertIsNone(RUN.load_checked(self.output / "simulation_grid"))

    def test_resume_certified_result_skips_search(self):
        self.assertEqual(self.execute(lambda *args, **kwargs: candidate(True)), 0)
        self.assertEqual(self.execute(lambda *args, **kwargs: self.fail("repeated search"),
                                      resume=True), 0)

    def test_resume_after_interruption_reuses_completed_attempt(self):
        calls = []

        def interrupted(experiment, policy, **kwargs):
            calls.append(policy[0])
            if len(calls) == 2:
                raise KeyboardInterrupt()
            return candidate(False)

        with self.assertRaises(KeyboardInterrupt):
            self.execute(interrupted)
        self.assertEqual(calls, [0., 1.])
        self.assertEqual(json.loads((self.output / "manifest.json").read_text())["status"],
                         "interrupted")
        resumed = []

        def finish(experiment, policy, **kwargs):
            resumed.append(policy[0])
            return candidate(True, 1.)

        self.assertEqual(self.execute(finish, resume=True), 0)
        self.assertEqual(resumed, [1.])

    def test_no_certificate_is_not_success_or_simulated(self):
        with patch.object(RETRY, "run_simulation", side_effect=AssertionError("uncertified")):
            self.assertEqual(self.execute(lambda *args, **kwargs: candidate(False),
                                          simulate=True), 2)
        manifest = json.loads((self.output / "manifest.json").read_text())
        self.assertEqual(manifest["status"], "unverified")
        self.assertNotIn("predicted", RUN.load_checked(self.output / "solution"))

    def test_no_overwrite_and_original_sources_checked(self):
        self.execute(lambda *args, **kwargs: candidate(True))
        with self.assertRaisesRegex(ValueError, "already exists"):
            self.execute(lambda *args, **kwargs: self.fail("search should not run"))
        with patch.object(RUN, "source_hashes", return_value={}), \
                self.assertRaisesRegex(ValueError, "Original campaign sources changed"):
            self.execute(lambda *args, **kwargs: self.fail("search should not run"), resume=True)
        with self.assertRaisesRegex(ValueError, "outside the original"):
            RETRY.execute_retry(self.source, self.source / "nested")

    def test_resume_rejects_changed_retry_sources(self):
        self.execute(lambda *args, **kwargs: candidate(True))
        with patch.object(RETRY, "retry_sources", return_value=dict(RETRY_SOURCES, retry="v2")), \
                self.assertRaisesRegex(ValueError, "sources, inputs or configuration changed"):
            self.execute(lambda *args, **kwargs: self.fail("search should not run"), resume=True)

    def test_seeds_include_rejected_candidates_and_remove_duplicates(self):
        first, other = candidate(False), candidate(False, 1.)
        seeds = RETRY.seed_policies(dict(selected=first,
            attempts=[dict(candidate=first), dict(candidate=other)]))
        self.assertEqual([item["name"] for item in seeds],
                         ["previous_selected", "previous_attempt_1"])
        self.assertFalse(first["verified"])
        with self.assertRaisesRegex(ValueError, "Invalid initial"):
            RETRY.seed_policies(dict(selected=dict(grid=np.array([0., .5, 1.]),
                policy=np.array([0., 2., 1.]))))

    def test_stale_retry_simulation_rejected(self):
        self.execute(lambda *args, **kwargs: candidate(True))
        manifest = json.loads((self.output / "manifest.json").read_text())
        solution = RUN.load_checked(self.output / "solution")
        RUN.save_checked(self.output / "simulation_grid", dict(
            campaign_fingerprint=manifest["fingerprint"], policy_sha256="wrong"))
        with self.assertRaisesRegex(ValueError, "simulation has incompatible"):
            RETRY.run_simulation(self.output, manifest, solution, None)

    def test_lock_excludes_another_retry(self):
        self.output.mkdir()
        with RETRY.output_lock(self.output):
            with self.assertRaisesRegex(RuntimeError, "Another retry"):
                with RETRY.output_lock(self.output):
                    self.fail("second retry acquired lock")


if __name__ == "__main__":
    unittest.main()
