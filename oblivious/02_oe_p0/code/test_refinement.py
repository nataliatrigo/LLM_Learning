"""Small numerical checks for restart and grid/closure isolation."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose

from oblivious.oe_p0.code import core
from oblivious.oe_p0.code import refinement
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.refinement import (
    EvaluationCheckpoint, is_verified_case, solve_refinement,
)
from oblivious.oe_following_monopoly.code.model import QualityExperiment


class ScalarRefinementChecks(unittest.TestCase):
    params = Parameters(gamma=0, rho=.6)

    def solve(self, **kwargs):
        return solve_refinement(5, self.params, p0_points=3, order=32, **kwargs)

    def test_restart_keeps_completed_evaluation_and_stationary_policy_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            def interrupt(message):
                if message.startswith("Checkpoint: pure"):
                    raise KeyboardInterrupt("simulated shutdown")

            with self.assertRaises(KeyboardInterrupt):
                self.solve(checkpoint_dir=directory, progress=interrupt)
            with patch.object(core, "checked_joint_stationary",
                              wraps=core.checked_joint_stationary) as checked:
                resumed = self.solve(checkpoint_dir=directory)
            baseline = self.solve()
            self.assertTrue(is_verified_case(resumed, 5, self.params, order=32))
            self.assertGreater(resumed["refinement"]["checkpoint"]["reused_evaluations"], 0)
            # The interrupted pure-policy law is restored, so new pure scan
            # points do not solve it again. Only the final mixed/pure endpoint
            # independently strong certification remains.
            self.assertEqual(checked.call_count, 1)
            assert_allclose(resumed["selected"]["nu"], baseline["selected"]["nu"], atol=1e-14)
            assert_allclose(resumed["selected"]["policy"], baseline["selected"]["policy"], atol=0)
            self.assertAlmostEqual(resumed["selected"]["p0"], baseline["selected"]["p0"], places=14)

    def test_completed_same_grid_case_can_be_reused_without_search(self):
        previous = self.solve()
        with patch.object(core.ScalarExperiment, "search", side_effect=AssertionError("unexpected recomputation")):
            result = self.solve(previous=previous)
        self.assertTrue(result["refinement"]["reused_verified_case"])
        inconsistent = dict(previous, selected=dict(previous["selected"], G=.01))
        self.assertFalse(is_verified_case(inconsistent, 5, self.params, order=32))

    def test_finer_grid_never_relabels_previous_scalar_map_as_compatible(self):
        previous = self.solve()
        with patch.object(core.ScalarExperiment, "search", autospec=True,
                          wraps=None) as search:
            search.side_effect = RuntimeError("stop before numerical search")
            with self.assertRaisesRegex(RuntimeError, "stop before"):
                solve_refinement(7, self.params, previous=previous, p0_points=3, order=32)
        self.assertIsNone(search.call_args.kwargs["pure_seed"])

    def test_checkpoint_detects_changed_configuration_and_corrupted_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            store = EvaluationCheckpoint(directory, dict(grid=5))
            store.put("example", "one", dict(value=np.array([1., 2.])))
            assert_allclose(store.get("example", "one")["value"], [1., 2.])
            with self.assertRaisesRegex(ValueError, "configuration"):
                EvaluationCheckpoint(directory, dict(grid=7))
            entry = next(iter(store.index["entries"].values()))
            (Path(directory) / (entry["stem"] + ".npz")).write_bytes(b"interrupted write")
            with self.assertRaisesRegex(ValueError, "corrupted"):
                store.get("example", "one")

    def test_checkpoint_fingerprint_includes_compiled_stationary_sources(self):
        original = refinement._source_fingerprint(core.ScalarExperiment)
        sources = {Path(path).name for path in original}
        self.assertTrue({"positive_stationary.py", "positive_stationary.cpp"} <= sources)
        digest = refinement._digest

        def changed_cpp(path):
            return "modified-source" if Path(path).name == "positive_stationary.cpp" else digest(path)

        with patch.object(refinement, "_digest", side_effect=changed_cpp):
            modified = refinement._source_fingerprint(core.ScalarExperiment)
        self.assertNotEqual(original, modified)
        with tempfile.TemporaryDirectory() as directory:
            EvaluationCheckpoint(directory, dict(numerical_sources=original))
            with self.assertRaisesRegex(ValueError, "numerical sources"):
                EvaluationCheckpoint(directory, dict(numerical_sources=modified))

    def test_interrupted_replacement_preserves_last_indexed_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            identity = dict(grid=5)
            store = EvaluationCheckpoint(directory, identity)
            store.put("example", "one", dict(value=np.array([1.])))
            # Entry data are complete, but shutdown precedes publishing their
            # new index. Reopening must still load the prior completed value.
            with patch.object(refinement, "write_json", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    store.put("example", "one", dict(value=np.array([2.])))
            resumed = EvaluationCheckpoint(directory, identity)
            assert_allclose(resumed.get("example", "one")["value"], [1.])

    def test_quality_closure_uses_same_checkpoint_adapter_and_own_certificate(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.solve(checkpoint_dir=directory,
                                experiment_class=QualityExperiment,
                                experiment_kwargs=dict(weighting="purchase"))
            self.assertTrue(is_verified_case(result, 5, self.params, order=32))
            self.assertEqual(result["configuration"]["quality_weighting"], "purchase")
            self.assertTrue(result["selected"]["monopoly_certificate"]["verified"])
            self.assertAlmostEqual(result["selected"]["F"], self.params.p1, places=14)
            # Equal grids and primitive parameters do not make different
            # closure equations interchangeable as a completed result.
            scalar = self.solve(previous=result)
            self.assertFalse(scalar["refinement"]["reused_verified_case"])
            self.assertNotAlmostEqual(scalar["selected"]["F"], result["selected"]["F"], places=5)
            with self.assertRaisesRegex(ValueError, "configuration"):
                self.solve(checkpoint_dir=directory)


if __name__ == "__main__":
    unittest.main()
