"""Reject incompatible inputs and independently check economic accounting."""

import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from oblivious.oe_p0.code.game import Game
from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_p0.code.run_experiment import load_bundle, save_bundle
from oblivious.equilibrium_comparison.code.inputs import load_pair, predictions


ROOT = Path(__file__).resolve().parents[3]
RUNS = {"oblivious/01_oe_distribution": "rho_forgetting", "oblivious/02_oe_p0": "rho_forgetting_mixed"}


class SavedInputValidation(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bases = {}
        self.manifests = {}
        for experiment, run in RUNS.items():
            source = ROOT / experiment / "data" / run
            destination = self.root / experiment / "data" / run
            (destination / "arrays").mkdir(parents=True)
            self.manifests[experiment] = destination / "manifest.json"
            shutil.copy2(source / "manifest.json", self.manifests[experiment])
            self.bases[experiment] = destination / "arrays" / "rho_0p95_n021"
            for suffix in (".json", ".npz"):
                shutil.copy2(source / "arrays" / ("rho_0p95_n021" + suffix),
                             self.bases[experiment].with_suffix(suffix))

    def rewrite_case(self, experiment, change):
        case = load_bundle(self.bases[experiment])
        change(case)
        save_bundle(self.bases[experiment], case)

    def rewrite_manifest(self, experiment, change):
        path = self.manifests[experiment]
        manifest = json.loads(path.read_text())
        change(manifest)
        path.write_text(json.dumps(manifest))

    def test_complete_verified_results_are_read_only_and_audit_is_json_serializable(self):
        paths = list(self.root.rglob("*.json")) + list(self.root.rglob("*.npz"))
        before = {path: path.read_bytes() for path in paths}
        pair = load_pair(self.root, n=21)
        json.dumps(pair["audit"], allow_nan=False)
        for path, content in before.items():
            self.assertEqual(content, path.read_bytes())
        for key in ("oe", "p0"):
            profile, audit = pair["profiles"][key], pair["audit"]["profiles"][key]
            self.assertLess(audit["stationary_residual_max"], 1e-10)
            self.assertLess(audit["consistency_residual"], 1e-7)
            self.assertEqual(profile["source"]["candidate_selection"], "saved_selected")
            self.assertEqual(profile["transition"].shape, (21 ** 2, 21 ** 2))
            assert_array_equal(profile["policy"], profile["case"]["selected"]["policy"])
        self.assertIn("oblivious/02_oe_p0/code/game.py", pair["audit"]["current_numerical_source_sha256"])

    def test_missing_correct_oe_stops_instead_of_substitution(self):
        self.manifests["oblivious/01_oe_distribution"].unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing OE distribution"):
            load_pair(self.root, n=21)

    def test_incomplete_run_is_rejected_even_with_saved_case(self):
        self.rewrite_manifest("oblivious/02_oe_p0", lambda m: m.update(status="incomplete"))
        with self.assertRaisesRegex(ValueError, "not complete"):
            load_pair(self.root, n=21)

    def test_oe_must_explicitly_identify_real_joint_marginal_consistency(self):
        self.rewrite_manifest("oblivious/01_oe_distribution", lambda m: m.update(model="one-dimensional demand approximation"))
        with self.assertRaisesRegex(ValueError, "actual-joint"):
            load_pair(self.root, n=21)

    def test_unverified_default_cannot_silently_fall_back_to_another_candidate(self):
        self.rewrite_case("oblivious/01_oe_distribution", lambda c: c["selected"].update(verified=False))
        with self.assertRaisesRegex(ValueError, "UNVERIFIED"):
            load_pair(self.root, n=21)
        explicit = load_pair(self.root, n=21, oe_candidate=0)
        self.assertEqual(explicit["profiles"]["oe"]["source"]["candidate_selection"], 0)

    def test_matching_manifest_and_case_with_different_primitives_are_rejected(self):
        self.rewrite_manifest("oblivious/02_oe_p0", lambda m: m["configuration"].update(c1=.06))
        self.rewrite_case("oblivious/02_oe_p0", lambda c: c["configuration"]["parameters"].update(c1=.06))
        with self.assertRaisesRegex(ValueError, "different primitives"):
            load_pair(self.root, n=21)

    def test_nonmatching_mature_grids_are_rejected(self):
        def change(case):
            case["selected"]["grid"][4] += .001
        self.rewrite_case("oblivious/02_oe_p0", change)
        with self.assertRaisesRegex(ValueError, "different mature reputation grids"):
            load_pair(self.root, n=21)

    def test_recomputed_stationarity_detects_changed_policy_despite_saved_flags(self):
        self.rewrite_case("oblivious/01_oe_distribution", lambda c: c["selected"].update(policy=np.zeros(21)))
        with self.assertRaisesRegex(ValueError, "not stationary"):
            load_pair(self.root, n=21)

    def test_joint_marginal_consistency_is_recomputed(self):
        def change(case):
            mu = case["selected"]["mu"].copy()
            mu[:] = 1 / len(mu)
            case["selected"]["mu"] = mu
        self.rewrite_case("oblivious/01_oe_distribution", change)
        with self.assertRaisesRegex(ValueError, "consistency failed"):
            load_pair(self.root, n=21)

    def test_scalar_consistency_is_recomputed_from_posterior_not_reputation(self):
        self.rewrite_case("oblivious/02_oe_p0", lambda c: c["selected"].update(p0=c["selected"]["p0"] + .01))
        with self.assertRaisesRegex(ValueError, "consistency failed"):
            load_pair(self.root, n=21)


class StationaryEconomics(unittest.TestCase):
    def test_joint_law_and_actual_selection_both_determine_product_frequency(self):
        # Three occupied reputation pairs have product-2 chances .34, .34,
        # and .75. Their masses .4, .1, .5 imply .545 overall, rather than
        # the .45 obtained by selecting each seller with probability one half.
        grid = np.array([0., .5, 1.])
        nu = np.zeros((3, 3))
        nu[0, 1], nu[1, 0], nu[2, 1] = .4, .1, .5
        demand = np.array([[.5, .2, .1], [.8, .5, .3], [.9, .7, .5]])
        profile = dict(grid=grid, policy=np.array([.1, .4, .9]), nu=nu)
        pair = dict(params=Parameters(), game=SimpleNamespace(selection=(demand, 1-demand)),
                    profiles={"synthetic": profile})
        result = predictions(pair, "synthetic")
        self.assertAlmostEqual(result["market_share_A"], .51)
        self.assertAlmostEqual(result["product2_calendar_A"], .355)
        self.assertAlmostEqual(result["product2_calendar_B"], .19)
        self.assertAlmostEqual(result["product2_market"], .545)
        self.assertAlmostEqual(result["product2_conditional_A"], .355 / .51)
        self.assertAlmostEqual(result["product2_conditional_B"], .19 / .49)

    def test_mixed_products_have_distinct_conditional_and_calendar_frequencies(self):
        # At equal reputations each seller is selected with probability 1/2.
        # Each independently uses product 2 with probability 1/4 when selected.
        params = Parameters()
        grid = np.array([0., .5, 1.])
        nu = np.zeros((3, 3))
        nu[1, 1] = 1.
        profile = dict(grid=grid, policy=np.full(3, .25), nu=nu)
        pair = dict(params=params, game=Game(grid, params), profiles={"synthetic": profile})
        metrics = predictions(pair, "synthetic")
        for seller in ("A", "B"):
            self.assertAlmostEqual(metrics[f"market_share_{seller}"], .5)
            self.assertAlmostEqual(metrics[f"product2_conditional_{seller}"], .25)
            self.assertAlmostEqual(metrics[f"product2_calendar_{seller}"], .125)
            self.assertAlmostEqual(metrics[f"profit_{seller}"], .4)
            self.assertAlmostEqual(metrics[f"mean_posterior_{seller}"], .5)
        self.assertAlmostEqual(metrics["product2_market"], .25)
        self.assertAlmostEqual(metrics["industry_profit"], .8)
        self.assertAlmostEqual(metrics["success_rate"], .4625)
        self.assertAlmostEqual(metrics["failure_rate"], .5375)
        self.assertIsNone(metrics["correlation"])

    def test_existing_mixed_policy_and_probabilities_are_not_rounded(self):
        pair = load_pair(ROOT, n=51)
        profile = pair["profiles"]["p0"]
        mixed = (profile["policy"] > 0) & (profile["policy"] < 1)
        self.assertTrue(mixed.any())
        assert_array_equal(profile["policy"], profile["case"]["selected"]["policy"])
        metrics = profile["predicted"]
        assert_allclose(metrics["product2_conditional_A"], 2 * metrics["product2_calendar_A"], atol=1e-12)
        assert_allclose(metrics["mean_posterior_B"], profile["p0"], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
