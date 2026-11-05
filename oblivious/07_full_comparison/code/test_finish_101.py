"""Bounded tests for the operational 101-only recovery controller."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np


SCRIPT = Path(__file__).with_name("finish_101.py")
SPEC = importlib.util.spec_from_file_location("finish_comparison_101", SCRIPT)
FINISH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FINISH)
RUN = FINISH.RUN


def prepare(directory, status="interrupted"):
    config = dict(models=["p0", "oe"], rhos=[.6, .95], grids=[101, 201],
                  simulation_modes=["grid"])
    sources = RUN.source_hashes()
    manifest = dict(configuration=config, source_sha256=sources,
                    fingerprint=RUN.fingerprint(config, sources), status=status)
    RUN.atomic_json(directory / "manifest.json", manifest)
    return manifest


def save_complete(directory, manifest, model, rho):
    folder = RUN.case_dir(directory, model, rho, 101)
    policy = np.array([0., 1.])
    candidate = dict(verified=True, candidate_verified=True, policy=policy,
                     dp_diagnostics=dict(verified=True),
                     stationary_diagnostics=dict(verified=True))
    RUN.save_checked(folder / "solution",
                     dict(n=101, selected=candidate, candidates=[candidate],
                          predicted=dict(product2_market=.5, seller_profit=.3,
                                         success_rate=.6, mean_reputation=.5),
                          campaign_fingerprint=manifest["fingerprint"]))
    RUN.save_checked(folder / "simulation_grid",
                     dict(campaign_fingerprint=manifest["fingerprint"],
                          policy_sha256=FINISH.sha256(policy.tobytes()).hexdigest(),
                          summary={key: dict(mean=.5) for key in
                                   ("product2_market", "seller_profit", "success_rate")}))
    RUN.atomic_json(folder / "status.json", dict(status="complete"))


class Finish101Tests(unittest.TestCase):
    def test_only_101_missing_cases_run_and_resume_preserves_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original = prepare(directory)
            save_complete(directory, original, "p0", .6)
            existing = RUN.case_dir(directory, "p0", .6, 101) / "solution.npz"
            existing_bytes = existing.read_bytes()
            partial = RUN.case_dir(directory, "p0", .95, 201) / "search_checkpoints/evaluations.json"
            RUN.atomic_json(partial, dict(entries={"finished": {"value": 1}}))
            partial_bytes = partial.read_bytes()
            calls, call_lock = [], threading.Lock()

            class FakeProcess:
                def __init__(self, command, **kwargs):
                    model, rho, n = command[5:8]
                    self.model, self.rho = model, float(rho)
                    self.returncode = None
                    with call_lock:
                        self.pid = 100000 + len(calls)
                        calls.append((model, float(rho), int(n), kwargs))

                def wait(self, timeout=None):
                    save_complete(directory, original, self.model, self.rho)
                    self.returncode = 0
                    return 0

                def poll(self):
                    return self.returncode

            with patch.object(FINISH.subprocess, "Popen", FakeProcess), redirect_stdout(io.StringIO()):
                self.assertEqual(FINISH.execute_101(directory, workers=2), 0)
            self.assertEqual(calls[0][:3], ("p0", .95, 101))
            self.assertCountEqual([(m, r, n) for m, r, n, _ in calls[1:]],
                                  [("oe", .95, 101), ("oe", .6, 101)])
            self.assertTrue(all(kwargs["start_new_session"] for _, _, _, kwargs in calls))
            updated = json.loads((directory / "manifest.json").read_text())
            for key in ("configuration", "source_sha256", "fingerprint"):
                self.assertEqual(updated[key], original[key])
            self.assertEqual(updated["status"], "awaiting_finer_grids")
            self.assertEqual(existing.read_bytes(), existing_bytes)
            self.assertEqual(partial.read_bytes(), partial_bytes)
            with patch.object(FINISH.subprocess, "Popen", side_effect=AssertionError("repeated case")), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(FINISH.execute_101(directory), 0)
            self.assertEqual(len(json.loads((directory / "finish_101.json").read_text())["runs"]), 2)

    def test_rejects_active_campaign_or_changed_original_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prepare(directory, status="running")
            original = (directory / "manifest.json").read_bytes()
            with self.assertRaisesRegex(RuntimeError, "Ctrl.C"):
                FINISH.execute_101(directory)
            self.assertEqual((directory / "manifest.json").read_bytes(), original)
            prepare(directory)
            with patch.object(RUN, "source_hashes", return_value={}), \
                    self.assertRaisesRegex(ValueError, "sources changed"):
                FINISH.execute_101(directory)

    def test_lock_excludes_second_launcher(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with FINISH.campaign_lock(directory):
                with self.assertRaisesRegex(RuntimeError, "Another finish_101"):
                    with FINISH.campaign_lock(directory):
                        self.fail("second lock acquired")

    def test_explicit_recovery_blocks_live_campaign_processes(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            prepare(directory, status="running")
            original = (directory / "manifest.json").read_bytes()
            with patch.object(FINISH, "active_campaign_processes",
                              return_value=[dict(pid=12345, command="--resume")]), \
                    patch.object(FINISH.subprocess, "Popen",
                                 side_effect=AssertionError("must not launch")), \
                    self.assertRaisesRegex(RuntimeError, "PID 12345"):
                FINISH.execute_101(directory, recover_interrupted=True)
            self.assertEqual((directory / "manifest.json").read_bytes(), original)
            self.assertFalse((directory / "finish_101.json").exists())

    def test_explicit_recovery_preserves_scientific_provenance_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            original = prepare(directory, status="running")
            for model in original["configuration"]["models"]:
                for rho in original["configuration"]["rhos"]:
                    save_complete(directory, original, model, rho)
            checkpoint = RUN.case_dir(directory, "p0", .95, 201) / "search_checkpoints/value.json"
            RUN.atomic_json(checkpoint, dict(value=12))
            checkpoint_bytes = checkpoint.read_bytes()
            with patch.object(FINISH, "active_campaign_processes", return_value=[]) as detector, \
                    patch.object(FINISH.subprocess, "Popen",
                                 side_effect=AssertionError("must reuse completed cases")), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(FINISH.execute_101(directory, recover_interrupted=True), 0)
            detector.assert_called_once_with(directory)
            updated = json.loads((directory / "manifest.json").read_text())
            for key in ("configuration", "source_sha256", "fingerprint"):
                self.assertEqual(updated[key], original[key])
            self.assertEqual(checkpoint.read_bytes(), checkpoint_bytes)
            operational = json.loads((directory / "finish_101.json").read_text())["runs"][-1]
            self.assertEqual(operational["recovery"]["previous_status"], "running")
            self.assertEqual(operational["recovery"]["local_process_check"], "clear")

    def test_process_detector_matches_project_campaign_and_workers(self):
        with tempfile.TemporaryDirectory() as temporary:
            proc_root = Path(temporary)
            campaign = RUN.OUTPUT / "test_recovery_campaign"

            def process(pid, arguments):
                entry = proc_root / str(pid)
                entry.mkdir()
                (entry / "cmdline").write_bytes(b"\0".join(str(arg).encode() for arg in arguments))
                (entry / "cwd").symlink_to(RUN.ROOT, target_is_directory=True)

            process(100001, ["python", FINISH.SCRIPT, "--name", campaign.name])
            process(100002, ["python", FINISH.SCRIPT, "--worker", "oe", ".95", "201",
                             "--directory", campaign])
            process(100003, ["python", FINISH.SCRIPT, "--name", "another_campaign"])
            process(100004, ["python", "/tmp/another_project/run_refinement.py",
                             "--name", campaign.name])
            process(FINISH.os.getpid(), ["python", FINISH.__file__, "--name", campaign.name])
            self.assertEqual({item["pid"] for item in
                              FINISH.active_campaign_processes(campaign, proc_root)},
                             {100001, 100002})

    def test_stale_simulation_policy_is_not_accepted_as_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            manifest = prepare(directory)
            save_complete(directory, manifest, "p0", .6)
            self.assertTrue(FINISH.complete_case(directory, manifest, "p0", .6))
            base = RUN.case_dir(directory, "p0", .6, 101) / "simulation_grid"
            simulation = RUN.load_checked(base)
            simulation["policy_sha256"] = "different_policy"
            RUN.save_checked(base, simulation)
            self.assertFalse(FINISH.complete_case(directory, manifest, "p0", .6))


if __name__ == "__main__":
    unittest.main()
