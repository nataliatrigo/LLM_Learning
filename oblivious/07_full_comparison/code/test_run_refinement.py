"""Bounded integration tests: four closures, safe reuse and independent workers."""

import importlib.util
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/full-refinement-test-mpl")
SCRIPT = Path(__file__).with_name("run_refinement.py")
SPEC = importlib.util.spec_from_file_location("comparison_refinement", SCRIPT)
RUN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN)

import numpy as np
from oblivious.oe_p0.code.primitives import Parameters


def prepare(directory, models, rhos=(.6,), grids=(5,)):
    parameters = dict(R=1., c1=.05, c2=.65, p1=.35, p2=.8, gamma=0.)
    configuration = dict(models=list(models), rhos=list(rhos), grids=list(grids),
        parameters=parameters, order=32, tol=1e-7, dp_tol=1e-10, p0_points=3,
        multistart=False, max_evaluations=100, simulation_modes=["grid"],
        periods=20, burn_in=3, replications=2, seed=21)
    sources = RUN.source_hashes()
    manifest = dict(configuration=configuration, source_sha256=sources,
        fingerprint=RUN.fingerprint(configuration,sources),status="prepared")
    RUN.atomic_json(Path(directory)/"manifest.json",manifest)
    return configuration


class RefinementIntegration(unittest.TestCase):
    def test_atomic_checkpoint_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)/"solution"
            RUN.save_checked(base,dict(policy=np.array([0.,1.])))
            np.testing.assert_array_equal(RUN.load_checked(base)["policy"],[0.,1.])
            base.with_suffix(".npz").write_bytes(b"interrupted overwrite")
            with self.assertRaisesRegex(ValueError,"changed"):
                RUN.load_checked(base)

    def test_all_four_models_certify_and_simulate_then_resume_without_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            config = prepare(folder,RUN.MODELS)
            for model in RUN.MODELS:
                with self.subTest(model=model):
                    with redirect_stdout(io.StringIO()):
                        self.assertEqual(RUN.worker(folder,model,.6,5),0)
                    base = RUN.case_dir(folder,model,.6,5)
                    case = RUN.load_checked(base/"solution")
                    self.assertTrue(RUN.verified(case["selected"]))
                    self.assertEqual(case["predicted"]["product2_market"],0.)
                    self.assertAlmostEqual(case["predicted"]["seller_profit"],.475)
                    simulation = RUN.load_checked(base/"simulation_grid")
                    self.assertEqual(len(simulation["replications"]),config["replications"])
                    self.assertEqual(simulation["summary"]["product2_market"]["mean"],0.)
                    if model == "monopoly":
                        self.assertTrue(case["selected"]["monopoly_certificate"]["verified"])
                    with patch.object(RUN,"solve_own",side_effect=AssertionError("repeated search")), \
                         patch("oblivious.oe_p0.code.refinement.solve_refinement",side_effect=AssertionError("repeated search")):
                        self.assertEqual(RUN.worker(folder,model,.6,5),0)
            rows = RUN.export_campaign(folder)
            self.assertEqual(len(rows),4)
            self.assertTrue(all(row["status"]=="complete" for row in rows))
            # Losing a simulation is not silently accepted as a complete job.
            base = RUN.case_dir(folder,"p0",.6,5)/"simulation_grid"
            base.with_name(base.name+"_sha256.json").unlink()
            rows = RUN.export_campaign(folder)
            self.assertEqual(next(r for r in rows if r["model"]=="p0")["status"],"incomplete_checkpoint")
            with redirect_stdout(io.StringIO()):
                self.assertEqual(RUN.worker(folder,"p0",.6,5),0)
            self.assertEqual(RUN.export_campaign(folder)[0]["status"],"complete")

    def test_old_flags_do_not_replace_a_fresh_certificate(self):
        config = dict(order=32,tol=1e-7,dp_tol=1e-10)
        experiment = RUN.experiment_for("conditional",5,Parameters(gamma=0,rho=.6),config)
        good = experiment.evaluate(np.zeros(5))
        # A historical flag cannot turn a strictly suboptimal policy into an equilibrium.
        forged = dict(good,policy=np.full(5,.5))
        self.assertIsNone(RUN.recertify(dict(selected=forged,candidates=[]),
                                      "conditional",experiment,config))

    def test_process_controller_keeps_rho_outputs_separate_and_grid_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            prepare(folder,["p0"],rhos=(.6,.7),grids=(5,7))
            self.assertEqual(RUN.execute_campaign(folder,workers_101=2,workers_201=1),0)
            manifest = json.loads((folder/"manifest.json").read_text())
            self.assertEqual(manifest["status"],"complete")
            for rho in (.6,.7):
                for n in (5,7):
                    job = RUN.case_dir(folder,"p0",rho,n)
                    self.assertTrue((job/"worker.log").exists())
                    self.assertTrue(RUN.verified(RUN.load_checked(job/"solution")["selected"]))
            # The continuation table requires two certified adjacent grids.
            self.assertEqual(len((folder/"grid_refinement.csv").read_text().splitlines()),3)

    def test_cli_excludes_point999_and_rejects_changing_resume_economics(self):
        with self.assertRaises(SystemExit) as error, redirect_stderr(io.StringIO()):
            RUN.main(["--dry-run","--rhos",".999"])
        self.assertEqual(error.exception.code,2)
        with self.assertRaises(SystemExit) as error, redirect_stderr(io.StringIO()):
            RUN.main(["--resume","--grids","101"])
        self.assertEqual(error.exception.code,2)


if __name__ == "__main__":
    unittest.main()
