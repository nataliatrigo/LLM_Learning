"""No-solve checks for isolated campaign checkpoint imports."""

import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).with_name("import_201_checkpoints.py")
SPEC = importlib.util.spec_from_file_location("import_comparison_201_checkpoints", SCRIPT)
IMPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(IMPORT)

from oblivious.oe_p0.code import core, refinement
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_following_monopoly.code.model import QualityExperiment


CONFIG = dict(parameters=dict(R=1., c1=.05, c2=.65, p1=.35, p2=.8, gamma=.98),
              tol=1e-7, order=128, dp_tol=1e-10)


def source_checkpoint(campaign, model="p0", rho=.95):
    directory = IMPORT._checkpoint_directory(campaign, model, rho)
    identity = IMPORT.expected_checkpoint_identity(model, rho, CONFIG)
    store = refinement.EvaluationCheckpoint(directory, identity)
    store.put("pure", float(.5).hex(), dict(p0=.5, value=[1., 2.]))
    return directory


class CheckpointImportTests(unittest.TestCase):
    def test_identity_matches_actual_adapter_without_constructing_game(self):
        for model, cls in (("p0", core.ScalarExperiment), ("monopoly", QualityExperiment)):
            with self.subTest(model=model), tempfile.TemporaryDirectory() as temporary:
                with patch.object(core, "Game", return_value=SimpleNamespace(order=128)):
                    experiment = cls(n=201, params=Parameters(rho=.95), order=128)
                identity = IMPORT.expected_checkpoint_identity(model, .95, CONFIG)
                # Exercise the original adapter's identity creation, rather
                # than reproducing its expected dictionary in this test.
                refinement.CheckpointedScalarMixin.configure_checkpoint(
                    experiment, temporary, consistency_tol=1e-7, experiment_class=cls)
                self.assertEqual(experiment.evaluation_checkpoint.identity, identity)

    def test_import_validated_generations_and_reuse_identical_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "old", Path(temporary) / "new"
            original = source_checkpoint(source)
            (original / "unreferenced_partial.json").write_text("unfinished")
            original_bytes = {p.name: p.read_bytes() for p in original.iterdir()}
            with patch.object(core, "Game", side_effect=AssertionError("No game should be allocated")):
                provenance = IMPORT.import_checkpoints(source, target, "p0", .95, CONFIG)
            copied = IMPORT._checkpoint_directory(target, "p0", .95)
            self.assertEqual(provenance["entries"], 1)
            self.assertEqual(provenance["files"], 2)
            self.assertFalse(provenance["existing_identical_target"])
            self.assertNotIn("unreferenced_partial.json", {p.name for p in copied.iterdir()})
            self.assertEqual({p.name: p.read_bytes() for p in original.iterdir()}, original_bytes)
            repeated = IMPORT.import_checkpoints(source, target, "p0", .95, CONFIG)
            self.assertTrue(repeated["existing_identical_target"])

    def test_incompatible_identity_does_not_create_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "old", Path(temporary) / "new"
            source_checkpoint(source)
            for changed in (dict(CONFIG, order=256), dict(CONFIG, dp_tol=1e-9),
                            dict(CONFIG, tol=1e-6)):
                with self.assertRaisesRegex(ValueError, "configuration or numerical sources"):
                    IMPORT.import_checkpoints(source, target, "p0", .95, changed)
            self.assertFalse(target.exists())

    def test_corrupted_generation_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "old", Path(temporary) / "new"
            original = source_checkpoint(source)
            next(original.glob("*.npz")).write_bytes(b"broken")
            with self.assertRaisesRegex(ValueError, "missing or corrupted"):
                IMPORT.import_checkpoints(source, target, "p0", .95, CONFIG)
            self.assertFalse(target.exists())

    def test_existing_different_target_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "old", Path(temporary) / "new"
            source_checkpoint(source)
            copied = source_checkpoint(target)
            index_path = copied / "evaluations.json"
            index = json.loads(index_path.read_text())
            index["entries"] = {}
            index_path.write_text(json.dumps(index))
            before = {p.name: p.read_bytes() for p in copied.iterdir()}
            with self.assertRaisesRegex(ValueError, "different checkpoint evaluations"):
                IMPORT.import_checkpoints(source, target, "p0", .95, CONFIG)
            self.assertEqual({p.name: p.read_bytes() for p in copied.iterdir()}, before)

    def test_missing_and_nonscalar_sources_are_noops(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, target = Path(temporary) / "old", Path(temporary) / "new"
            self.assertIsNone(IMPORT.import_checkpoints(source, target, "p0", .95, CONFIG))
            self.assertIsNone(IMPORT.import_checkpoints(source, target, "oe", .95, CONFIG))
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
