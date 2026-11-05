"""Bounded checks for the 201-only controller; no numerical searches run."""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np


SPEC = importlib.util.spec_from_file_location(
    "test_comparison_201_controller", Path(__file__).with_name("run_201.py"))
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)
RUN = CONTROL.RUN
WORKER = CONTROL.WORKER
ORIGINAL_SOURCES = {"original.py": "v1"}
CONTROLLER_SOURCES = dict(ORIGINAL_SOURCES, controller="v1", worker="v1")


def candidate(n=101, valid=True):
    return dict(grid=np.linspace(0, 1, n), policy=np.zeros(n),
        nu=np.full((n, n), 1 / n ** 2), verified=valid,
        candidate_verified=valid, dp_diagnostics=dict(verified=valid),
        stationary_diagnostics=dict(verified=valid))


def prepare_source(directory, models=("p0", "oe"), rhos=(.95, .99)):
    config = dict(models=list(models), rhos=list(rhos), grids=[101, 201],
        parameters=dict(R=1., c1=.05, c2=.65, p1=.35, p2=.8, gamma=.98),
        order=128, tol=1e-7, dp_tol=1e-10, p0_points=41,
        multistart=False, max_evaluations=1500, simulation_modes=["grid"],
        periods=100, burn_in=10, replications=2, seed=1234)
    manifest = dict(configuration=config, source_sha256=ORIGINAL_SOURCES,
        fingerprint=RUN.fingerprint(config, ORIGINAL_SOURCES), status="partial_complete")
    RUN.atomic_json(directory / "manifest.json", manifest)
    for model in models:
        for rho in rhos:
            case = dict(n=101, rho=rho, selected=candidate(), candidates=[candidate()],
                configuration=dict(n=101, parameters=dict(config["parameters"], rho=rho),
                    order=128), demand_audit=dict(verified=True),
                predicted=dict(product2_market=0., seller_profit=.475,
                    success_rate=.35, mean_reputation=.5),
                campaign_fingerprint=manifest["fingerprint"])
            RUN.save_checked(RUN.case_dir(directory, model, rho, 101) / "solution", case)
    return manifest


def snapshot(directory):
    return {str(path.relative_to(directory)): path.read_bytes()
            for path in directory.rglob("*") if path.is_file()}


def save_complete(directory, manifest, model, rho):
    folder = RUN.case_dir(directory, model, rho, 201)
    case = dict(n=201, rho=rho, selected=candidate(201), candidates=[candidate(201)],
        configuration=dict(n=201, parameters=dict(manifest["configuration"]["parameters"], rho=rho),
            order=128), demand_audit=dict(verified=True),
        predicted=dict(product2_market=0., seller_profit=.475,
            success_rate=.35, mean_reputation=.5),
        campaign_fingerprint=manifest["fingerprint"])
    RUN.save_checked(folder / "solution", case)
    for mode in manifest["configuration"]["simulation_modes"]:
        RUN.save_checked(folder / ("simulation_" + mode), dict(
            campaign_fingerprint=manifest["fingerprint"],
            policy_sha256=CONTROL.sha256(case["selected"]["policy"].tobytes()).hexdigest(),
            summary={metric: dict(mean=value) for metric, value in
                (("product2_market", 0.), ("seller_profit", .475), ("success_rate", .35))}))
    RUN.atomic_json(folder / "status.json", dict(status="complete"))


class Run201Tests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "original"
        self.output = self.root / "201"
        self.retry = self.root / "retry"
        self.original = prepare_source(self.source)
        self.original_bytes = snapshot(self.source)
        for item in (patch.object(RUN, "source_hashes", return_value=ORIGINAL_SOURCES),
                     patch.object(CONTROL, "sources", return_value=CONTROLLER_SOURCES)):
            item.start()
            self.addCleanup(item.stop)

    def test_prepare_isolated_201_and_preserves_all_original_bytes(self):
        CONTROL.prepare_campaign(self.output, self.source, None,
            models=["p0"], rhos=[.95])
        manifest = json.loads((self.output / "manifest.json").read_text())
        config = manifest["configuration"]
        self.assertEqual(config["grids"], [201])
        self.assertEqual(config["models"], ["p0"])
        self.assertEqual(config["rhos"], [.95])
        self.assertEqual(config["simulation_modes"], ["grid"])
        for key in ("parameters", "order", "tol", "dp_tol", "p0_points",
                    "periods", "burn_in", "replications", "seed"):
            self.assertEqual(config[key], self.original["configuration"][key])
        self.assertEqual(manifest["source_campaign_fingerprint"], self.original["fingerprint"])
        self.assertEqual(manifest["source_sha256"], CONTROLLER_SOURCES)
        self.assertEqual(config["input_sha256"], manifest["input_sha256"])
        self.assertEqual(manifest["fingerprint"], RUN.fingerprint(config, CONTROLLER_SOURCES))
        self.assertTrue(manifest["seed_sources"])
        self.assertEqual(snapshot(self.source), self.original_bytes)
        self.assertIsNone(RUN.load_checked(RUN.case_dir(self.output, "p0", .95, 201) / "solution"))

    def test_solve_only_disables_simulation_without_changing_numerical_tolerances(self):
        CONTROL.prepare_campaign(self.output, self.source, None,
            models=["p0"], rhos=[.95], simulate=False)
        config = json.loads((self.output / "manifest.json").read_text())["configuration"]
        self.assertEqual(config["simulation_modes"], [])
        self.assertEqual(config["tol"], 1e-7)
        self.assertEqual(config["dp_tol"], 1e-10)

    def test_existing_output_and_nesting_are_protected(self):
        CONTROL.prepare_campaign(self.output, self.source, None,
            models=["p0"], rhos=[.95])
        before = snapshot(self.output)
        with self.assertRaises((ValueError, FileExistsError)):
            CONTROL.prepare_campaign(self.output, self.source, None,
                models=["p0"], rhos=[.95])
        self.assertEqual(snapshot(self.output), before)
        for directory in (self.source, self.source / "nested"):
            with self.assertRaises(ValueError):
                CONTROL.prepare_campaign(directory, self.source, None,
                    models=["p0"], rhos=[.95])
        self.assertEqual(snapshot(self.source), self.original_bytes)

    def test_source_code_and_checked_input_provenance_are_enforced(self):
        with patch.object(RUN, "source_hashes", return_value={}):
            with self.assertRaises(ValueError):
                CONTROL.prepare_campaign(self.output, self.source, None,
                    models=["p0"], rhos=[.95])
        base = RUN.case_dir(self.source, "p0", .95, 101) / "solution"
        case = RUN.load_checked(base)
        case["campaign_fingerprint"] = "another campaign"
        RUN.save_checked(base, case)
        with self.assertRaises(ValueError):
            CONTROL.prepare_campaign(self.output, self.source, None,
                models=["p0"], rhos=[.95])

    def test_uncertified_101_seed_is_rejected(self):
        base = RUN.case_dir(self.source, "p0", .95, 101) / "solution"
        case = RUN.load_checked(base)
        case["selected"] = candidate(valid=False)
        RUN.save_checked(base, case)
        with self.assertRaises(ValueError):
            CONTROL.prepare_campaign(self.output, self.source, None,
                models=["p0"], rhos=[.95])

    def test_dry_run_has_no_output_or_numerical_subprocess(self):
        with patch.object(RUN, "OUTPUT", self.root), \
                patch.object(CONTROL, "prepare_campaign", side_effect=AssertionError("mutation")), \
                patch.object(CONTROL, "execute_201", side_effect=AssertionError("execution")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(CONTROL.main(["--name", "new_201", "--dry-run"]), 0)
        self.assertFalse((self.root / "new_201").exists())

    def test_resume_rejects_scientific_changes_and_invalid_workers(self):
        CONTROL.prepare_campaign(self.output, self.source, None,
            models=["p0"], rhos=[.95])
        before = snapshot(self.output)
        with patch.object(RUN, "OUTPUT", self.root), \
                patch.object(CONTROL, "execute_201", side_effect=AssertionError("execution")), \
                redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            for extra in (["--resume", "--rhos", ".95"],
                          ["--resume", "--models", "p0"],
                          ["--workers", "0"], ["--rhos", ".999"]):
                with self.assertRaises(SystemExit):
                    CONTROL.main(["--name", self.output.name, *extra])
        self.assertEqual(snapshot(self.output), before)

    def test_cli_resume_restores_saved_settings_and_changes_only_worker_count(self):
        CONTROL.prepare_campaign(self.output, self.source, None,
            models=["p0"], rhos=[.95], simulate=False)
        before = snapshot(self.output)
        with patch.object(RUN, "OUTPUT", self.root), \
                patch.object(CONTROL, "prepare_campaign", side_effect=AssertionError("reprepare")), \
                patch.object(CONTROL, "execute_201", return_value=0) as execute, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(CONTROL.main(
                ["--name", self.output.name, "--resume", "--workers", "2"]), 0)
        execute.assert_called_once_with(self.output, 2)
        self.assertEqual(snapshot(self.output), before)

    def test_only_201_runs_models_sequentially_respects_workers_and_skips_complete(self):
        # Two nonproblematic rho values exercise parallel jobs in each stage.
        other_source = self.root / "parallel_source"
        prepare_source(other_source, rhos=(.9, .95))
        parallel = self.root / "parallel_201"
        manifest = CONTROL.prepare_campaign(parallel, other_source, None,
            models=["p0", "oe"], rhos=[.9, .95])
        before = snapshot(other_source)
        save_complete(parallel, manifest, "p0", .9)
        calls, active, maximum = [], 0, 0
        lock = threading.Lock()

        class FakeProcess:
            def __init__(process, command, **kwargs):
                nonlocal active, maximum
                process.model = command[command.index("--model") + 1]
                process.rho = float(command[command.index("--rho") + 1])
                process.n = int(command[command.index("--n") + 1])
                process.returncode = None
                with lock:
                    process.pid = 100000 + len(calls)
                    calls.append((process.model, process.rho, process.n, kwargs))
                    active += 1
                    maximum = max(maximum, active)

            def wait(process, timeout=None):
                nonlocal active
                if process.returncode is None:
                    time.sleep(.01)
                    save_complete(parallel, manifest, process.model, process.rho)
                    with lock:
                        active -= 1
                    process.returncode = 0
                return process.returncode

            def poll(process):
                return process.returncode

        with patch.object(CONTROL, "active_processes", return_value=[]), \
                patch.object(WORKER, "validate_manifest"), \
                patch.object(CONTROL.subprocess, "Popen", FakeProcess), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(CONTROL.execute_201(parallel, workers=2), 0)
        self.assertEqual(calls[0][:3], ("p0", .95, 201))
        self.assertCountEqual([(model, rho, n) for model, rho, n, _ in calls[1:]],
            [("oe", .95, 201), ("oe", .9, 201)])
        self.assertLessEqual(maximum, 2)
        self.assertTrue(all(kwargs["start_new_session"] for _, _, _, kwargs in calls))
        self.assertEqual(snapshot(other_source), before)
        updated = json.loads((parallel / "manifest.json").read_text())
        self.assertEqual(updated["status"], "complete")
        for name in ("configuration", "fingerprint", "input_sha256", "source_sha256"):
            self.assertEqual(updated[name], manifest[name])
        self.assertTrue((parallel / "comparison_101_201.csv").is_file())
        with patch.object(CONTROL, "active_processes", return_value=[]), \
                patch.object(WORKER, "validate_manifest"), \
                patch.object(CONTROL.subprocess, "Popen", side_effect=AssertionError("repeat")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(CONTROL.execute_201(parallel, workers=1), 0)
        self.assertEqual(snapshot(other_source), before)

    def test_orphan_worker_and_second_controller_are_rejected(self):
        CONTROL.prepare_campaign(self.output, self.source, None,
            models=["p0"], rhos=[.95])
        manifest_bytes = (self.output / "manifest.json").read_bytes()
        with patch.object(CONTROL, "active_processes", return_value=[12345]), \
                patch.object(CONTROL.subprocess, "Popen", side_effect=AssertionError("execution")), \
                self.assertRaisesRegex(RuntimeError, "12345"):
            CONTROL.execute_201(self.output)
        self.assertEqual((self.output / "manifest.json").read_bytes(), manifest_bytes)
        with CONTROL.campaign_lock(self.output):
            with self.assertRaisesRegex(RuntimeError, "Another 201"):
                with CONTROL.campaign_lock(self.output):
                    self.fail("second controller acquired lock")


if __name__ == "__main__":
    unittest.main()
