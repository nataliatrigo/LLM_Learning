"""Bounded checks for isolated 201 continuation, provenance and safe restart."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


SPEC = importlib.util.spec_from_file_location("comparison_201_worker",
    Path(__file__).with_name("refine_201_worker.py"))
WORKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKER)
RUN = WORKER.RUN

import numpy as np
from oblivious.oe_p0.code.primitives import Parameters


def flags(policy, good=True):
    return dict(policy=np.asarray(policy, dtype=float), grid=np.linspace(0, 1, len(policy)),
        verified=good, candidate_verified=good, dp_diagnostics={"verified": good},
        stationary_diagnostics={"verified": good})


def prepare(parent, *, rhos=(.6,), models=RUN.MODELS):
    source, target = Path(parent) / "source", Path(parent) / "target"
    parameters = dict(R=1., c1=.05, c2=.65, p1=.35, p2=.8, gamma=0.)
    common = dict(models=list(models), rhos=list(rhos), parameters=parameters,
        order=32, tol=1e-7, dp_tol=1e-10, p0_points=3, multistart=False,
        max_evaluations=100, active_max_evaluations=30, simulation_modes=["grid"],
        periods=20, burn_in=3, replications=2, seed=21)
    old_config = dict(common, grids=[3])
    old_sources = RUN.source_hashes()
    old = dict(configuration=old_config, source_sha256=old_sources,
        fingerprint=RUN.fingerprint(old_config, old_sources))
    RUN.atomic_json(source / "manifest.json", old)
    inputs = {}
    for model in models:
        for rho in rhos:
            params = Parameters(**parameters, rho=rho)
            experiment = RUN.experiment_for(model, 3, params, common)
            alpha = np.zeros(3)
            candidate = (experiment.evaluate_mixed(.35, alpha, common["tol"])
                         if model in ("p0", "monopoly") else experiment.evaluate(alpha))
            if model in ("p0", "monopoly"):
                candidate = experiment.evaluate_mixed(candidate["F"], alpha, common["tol"])
            assert RUN.verified(candidate)
            from oblivious.oe_p0.code.core import ScalarExperiment
            case = dict(rho=rho, n=3, selected=candidate, candidates=[candidate],
                configuration=experiment.configuration(), campaign_fingerprint=old["fingerprint"],
                demand_audit=ScalarExperiment(n=3, params=params, order=32).demand_audit())
            base = RUN.case_dir(source, model, rho, 3) / "solution"
            RUN.save_checked(base, case)
            for path in (base.with_suffix(".json"), base.with_suffix(".npz"),
                         base.with_name("solution_sha256.json")):
                inputs[str(path)] = RUN.digest(path)
    config = dict(common, grids=[5], seed_grid=3, source_campaign=str(source), oe_retry=None,
                  input_sha256=inputs)
    sources = WORKER.worker_sources()
    manifest = dict(configuration=config, source_sha256=sources, input_sha256=inputs,
        source_campaign_fingerprint=old["fingerprint"], fingerprint=RUN.fingerprint(config, sources))
    RUN.atomic_json(target / "manifest.json", manifest)
    return source, target, manifest


class WorkerTests(unittest.TestCase):
    def test_four_models_continue_certify_simulate_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target, manifest = prepare(temporary)
            source_hashes = {p: RUN.digest(p) for p in source.rglob("*") if p.is_file()}
            for model in RUN.MODELS:
                with self.subTest(model=model), redirect_stdout(io.StringIO()):
                    self.assertEqual(WORKER.worker(target, model, .6, 5), 0)
                    folder = RUN.case_dir(target, model, .6, 5)
                    case = RUN.load_checked(folder / "solution")
                    self.assertTrue(RUN.verified(case["selected"]))
                    self.assertEqual(case["predicted"]["product2_market"], 0.)
                    self.assertEqual(case["provenance"]["source_fingerprint"],
                                     manifest["source_campaign_fingerprint"])
                    simulation = RUN.load_checked(folder / "simulation_grid")
                    self.assertEqual(simulation["summary"]["product2_market"]["mean"], 0.)
                    with patch.object(WORKER, "previous_case", side_effect=AssertionError("repeated search")):
                        self.assertEqual(WORKER.worker(target, model, .6, 5), 0)
            self.assertEqual(source_hashes,
                {p: RUN.digest(p) for p in source.rglob("*") if p.is_file()})

    def test_input_change_and_configuration_tampering_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target, manifest = prepare(temporary, models=("p0",))
            changed = dict(manifest, configuration=dict(manifest["configuration"], periods=21))
            with self.assertRaisesRegex(ValueError, "fingerprint"):
                WORKER.validate_manifest(changed)
            with self.assertRaisesRegex(ValueError, "bound"):
                WORKER.validate_manifest(dict(manifest, input_sha256={}))
            base = RUN.case_dir(source, "p0", .6, 3) / "solution.json"
            base.write_text(base.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "input_sha256"):
                WORKER.worker(target, "p0", .6, 5)

    def test_oe099_requires_certified_isolated_retry(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target, manifest = prepare(temporary, rhos=(.99,), models=("oe",))
            with self.assertRaisesRegex(ValueError, "isolated retry"):
                WORKER.previous_case(manifest, "oe", .99)
            retry = Path(temporary) / "retry"
            original = RUN.load_checked(RUN.case_dir(source, "oe", .99, 3) / "solution")
            original["campaign_fingerprint"] = "retry-id"
            RUN.save_checked(retry / "solution", original)
            RUN.atomic_json(retry / "manifest.json", dict(fingerprint="retry-id",
                source_campaign_fingerprint=manifest["source_campaign_fingerprint"]))
            manifest["configuration"]["oe_retry"] = str(retry)
            case, provenance = WORKER.previous_case(manifest, "oe", .99)
            self.assertEqual(case["campaign_fingerprint"], "retry-id")
            self.assertEqual(provenance["base"], str(retry / "solution"))
            case["selected"]["verified"] = False
            RUN.save_checked(retry / "solution", case)
            with self.assertRaisesRegex(ValueError, "certified"):
                WORKER.previous_case(manifest, "oe", .99)

    def test_branches_preserve_mixed_decisions_and_deduplicate(self):
        first = flags([0., .3, 1.])
        second = flags([1., .7, 0.])
        seeds = WORKER.policy_seeds(dict(selected=first, candidates=[first, second]),
                                    np.linspace(0, 1, 5))
        self.assertEqual(len(seeds), 2)
        np.testing.assert_allclose(seeds[0][1], [0., .15, .3, .65, 1.])
        self.assertTrue(np.any((seeds[1][1] > 0) & (seeds[1][1] < 1)))

    def test_case_lock_blocks_live_competing_worker_and_releases(self):
        with tempfile.TemporaryDirectory() as temporary:
            with WORKER.case_lock(temporary):
                with self.assertRaisesRegex(RuntimeError, "Another worker"):
                    with WORKER.case_lock(temporary):
                        self.fail("The live worker's case was not exclusive")
            with WORKER.case_lock(temporary):
                pass

    def test_active_attempt_reused_and_original_solver_is_fallback(self):
        candidate = flags([0., 0., 0.])
        experiment = SimpleNamespace(grid=np.linspace(0, 1, 3), n=3,
            params=SimpleNamespace(rho=.6), configuration=lambda: {}, solve=Mock(return_value=candidate))
        active = SimpleNamespace(solve_active_set=Mock(return_value=candidate))
        previous = dict(selected=candidate, candidates=[candidate])
        config = dict(multistart=False, max_evaluations=10)
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(WORKER, "module", return_value=active), redirect_stdout(io.StringIO()):
            first = WORKER.solve_distribution(experiment, previous, config, temporary, "test-id")
            self.assertTrue(RUN.verified(first["selected"]))
            self.assertEqual(active.solve_active_set.call_count, 1)
            experiment.solve.assert_not_called()
            WORKER.solve_distribution(experiment, previous, config, temporary, "test-id")
            self.assertEqual(active.solve_active_set.call_count, 1)
        active.solve_active_set = Mock(return_value=flags([0., 0., 0.], good=False))
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(WORKER, "module", return_value=active), redirect_stdout(io.StringIO()):
            result = WORKER.solve_distribution(experiment, previous, config, temporary, "test-id")
            self.assertTrue(RUN.verified(result["selected"]))
            self.assertEqual(experiment.solve.call_count, 4)

    def test_unverified_case_is_saved_without_simulation(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target, manifest = prepare(temporary, models=("oe",))
            rejected = dict(selected=flags([0.] * 5, good=False), candidates=[],
                rho=.6, n=5, demand_audit={"verified": True})
            with patch.object(WORKER, "solve_distribution", return_value=rejected), \
                    patch.object(WORKER, "simulate", side_effect=AssertionError("simulated invalid policy")), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(WORKER.worker(target, "oe", .6, 5), 2)
            folder = RUN.case_dir(target, "oe", .6, 5)
            self.assertEqual(json.loads((folder / "status.json").read_text())["status"], "unverified")
            self.assertFalse((folder / "simulation_grid.json").exists())

    def test_scalar_tolerances_reach_both_experiment_and_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target, manifest = prepare(temporary, models=("p0",))
            config = manifest["configuration"]
            config.update(dp_tol=1e-9, stationary_tol=2e-11, stability_tol=2e-7, near_tie_tol=3e-7)
            manifest["fingerprint"] = RUN.fingerprint(config, manifest["source_sha256"])
            RUN.atomic_json(target / "manifest.json", manifest)
            params = Parameters(**config["parameters"], rho=.6)
            experiment = WORKER.experiment_for("p0", 5, params, config)
            rejected = dict(selected=flags([0.] * 5, good=False), candidates=[],
                rho=.6, n=5, demand_audit={"verified": True}, configuration=experiment.configuration())
            with patch("oblivious.oe_p0.code.refinement.solve_refinement", return_value=rejected) as search, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(WORKER.worker(target, "p0", .6, 5), 2)
            for key in ("dp_tol", "stationary_tol", "stability_tol", "near_tie_tol"):
                self.assertEqual(getattr(experiment, key), config[key])
                self.assertEqual(search.call_args.kwargs[key], config[key])

    def test_saved_solution_cannot_be_reused_for_another_case(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target, manifest = prepare(temporary, models=("p0",))
            wrong = RUN.load_checked(RUN.case_dir(source, "p0", .6, 3) / "solution")
            wrong["campaign_fingerprint"] = manifest["fingerprint"]
            RUN.save_checked(RUN.case_dir(target, "p0", .6, 5) / "solution", wrong)
            with self.assertRaisesRegex(ValueError, "case configuration"), redirect_stdout(io.StringIO()):
                WORKER.worker(target, "p0", .6, 5)


if __name__ == "__main__":
    unittest.main()
