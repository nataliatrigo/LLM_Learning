"""Ensure convergence figures preserve saved search evidence and certification."""

import copy
import unittest

import numpy as np

from following_monopoly.code.solver_trace import selected_solver_trace


def mixed_case():
    """Include a nonmonotone refinement and an exact zero deliberately."""
    point, weight = .62, .4
    selected = dict(
        p0=point, F=point + 3e-14, G=3e-14,
        policy=np.array([0., weight, 1.]),
        mixed_nodes=1, mixed_indices=[1], mixed_probabilities=[weight],
        mixture_search_weight=weight,
        candidate_verified=True, verified=True,
        dp_diagnostics=dict(verified=True),
        stationary_diagnostics=dict(verified=True),
        monopoly_certificate=dict(verified=True),
    )
    weights = [0., .25, .5, .75, 1., .38, .41, .4, .4000001]
    residuals = [-.2, -.1, .1, .3, .4, .003, -.002, 0., 2e-9]
    trials = [
        dict(weight=w, p0=point, F=point + g, G=g,
             mixing_probabilities=[w], exploratory=True)
        for w, g in zip(weights, residuals)
    ]
    branch = dict(
        p0=point, tied_indices=[1], tied_reputations=[.5],
        start_probabilities=[0.], end_probabilities=[1.], trials=trials,
        candidates=[dict(weight=weight, verified=True, G=selected["G"],
                         mixed_probabilities=[weight])],
    )
    return dict(selected=selected, mixing_branches=[branch],
                search_summary=dict(consistency_tolerance=1e-7))


class SelectedSolverTraceChecks(unittest.TestCase):
    def test_preserves_chronological_signed_residuals_and_exact_zero(self):
        case = mixed_case()
        before = copy.deepcopy(case)

        trace = selected_solver_trace(case)

        self.assertEqual(trace["status"], "mixed branch")
        self.assertTrue(trace["verified"])
        self.assertEqual(trace["p0"], case["selected"]["p0"])
        self.assertEqual(trace["tolerance"], 1e-7)
        self.assertEqual(trace["scan_evaluations"], 5)
        trials = case["mixing_branches"][0]["trials"]
        rows = trace["rows"]
        self.assertEqual(len(rows), len(trials) + 1)
        self.assertEqual([row["G"] for row in rows[:-1]],
                         [trial["G"] for trial in trials])
        self.assertEqual([row["weight"] for row in rows[:-1]],
                         [trial["weight"] for trial in trials])
        self.assertEqual([row["stage"] for row in rows],
                         ["face scan"] * 5 + ["mixing refinement"] * 4
                         + ["full-game certificate"])
        evaluations = [row["evaluation"] for row in rows]
        self.assertTrue(all(isinstance(value, int) for value in evaluations))
        self.assertEqual(evaluations,
                         list(range(evaluations[0], evaluations[0] + len(rows))))
        for row, trial in zip(rows, trials):
            self.assertEqual(row["p0"], trial["p0"])
            self.assertEqual(row["F"], trial["F"])
            self.assertEqual(row["residual"], abs(trial["G"]))
            self.assertTrue(row["exploratory"])
        self.assertEqual(rows[7]["residual"], 0.)
        self.assertGreater(rows[8]["residual"], rows[7]["residual"])
        self.assertEqual(rows[-1]["G"], case["selected"]["G"])
        self.assertEqual(rows[-1]["F"], case["selected"]["F"])
        self.assertEqual(rows[-1]["residual"], abs(case["selected"]["G"]))
        self.assertFalse(rows[-1]["exploratory"])
        # Extracting plot evidence must not alter saved numerical evidence.
        self.assertEqual(case["mixing_branches"], before["mixing_branches"])
        np.testing.assert_array_equal(case["selected"]["policy"],
                                      before["selected"]["policy"])

    def test_rejects_branch_with_wrong_p0(self):
        case = mixed_case()
        case["mixing_branches"][0]["p0"] += 2e-10
        self.assert_missing_mixed_trace(case)

    def test_rejects_branch_with_wrong_certified_weight(self):
        case = mixed_case()
        case["mixing_branches"][0]["candidates"][0]["weight"] += 2e-9
        self.assert_missing_mixed_trace(case)

    def test_rejects_branch_without_verified_candidate(self):
        case = mixed_case()
        case["mixing_branches"][0]["candidates"][0]["verified"] = False
        self.assert_missing_mixed_trace(case)

    def test_rejects_interpolated_policy_mismatch(self):
        case = mixed_case()
        # The scalar root and its weight match, but this is a different face.
        case["mixing_branches"][0]["start_probabilities"] = [.01]
        self.assert_missing_mixed_trace(case)

    def test_accepts_matching_branch_within_declared_tolerances(self):
        case = mixed_case()
        branch = case["mixing_branches"][0]
        branch["p0"] += 5e-11
        branch["candidates"][0]["weight"] += 5e-10

        trace = selected_solver_trace(case)

        self.assertEqual(trace["status"], "mixed branch")
        self.assertGreater(len(trace["rows"]), 1)

    def test_skips_unrelated_branch_before_correct_match(self):
        case = mixed_case()
        unrelated = copy.deepcopy(case["mixing_branches"][0])
        unrelated["tied_indices"] = [0]
        case["mixing_branches"].insert(0, unrelated)

        trace = selected_solver_trace(case)

        self.assertEqual(trace["status"], "mixed branch")
        self.assertEqual(trace["rows"][0]["G"], -.2)

    def test_missing_mixed_branch_only_reports_saved_certificate(self):
        case = mixed_case()
        case["mixing_branches"] = []
        self.assert_missing_mixed_trace(case)

    def test_pure_endpoint_has_no_invented_iteration_path(self):
        case = mixed_case()
        selected = case["selected"]
        selected.update(p0=.35, F=.35, G=-5.55e-17,
                        policy=np.zeros(3), mixed_nodes=0,
                        mixed_indices=[], mixed_probabilities=[])
        selected.pop("mixture_search_weight")
        case["mixing_branches"] = []

        trace = selected_solver_trace(case)

        self.assertEqual(trace["status"], "pure endpoint")
        self.assertTrue(trace["verified"])
        self.assertEqual(trace["scan_evaluations"], 0)
        self.assertEqual(len(trace["rows"]), 1)
        self.assertEqual(trace["rows"][0]["stage"], "full-game certificate")
        self.assertEqual(trace["rows"][0]["G"], selected["G"])
        self.assertFalse(trace["rows"][0]["exploratory"])

    def test_each_failed_certificate_omits_all_equilibrium_rows(self):
        for path in (("candidate_verified",), ("verified",),
                     ("dp_diagnostics", "verified"),
                     ("stationary_diagnostics", "verified"),
                     ("monopoly_certificate", "verified")):
            with self.subTest(certificate=path):
                case = mixed_case()
                target = case["selected"]
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = False

                trace = selected_solver_trace(case)

                self.assertEqual(trace["status"], "unverified")
                self.assertFalse(trace["verified"])
                self.assertEqual(trace["rows"], [])

    def test_missing_certificate_is_unverified(self):
        case = mixed_case()
        del case["selected"]["stationary_diagnostics"]
        trace = selected_solver_trace(case)
        self.assertEqual(trace["status"], "unverified")
        self.assertFalse(trace["verified"])
        self.assertEqual(trace["rows"], [])

    def test_absent_selected_result_is_unavailable(self):
        for case in ({}, {"selected": None}):
            with self.subTest(case=case):
                trace = selected_solver_trace(case)
                self.assertEqual(trace["status"], "unavailable")
                self.assertFalse(trace["verified"])
                self.assertIsNone(trace["p0"])
                self.assertEqual(trace["rows"], [])

    def assert_missing_mixed_trace(self, case):
        trace = selected_solver_trace(case)
        self.assertEqual(trace["status"], "missing mixed trace")
        self.assertTrue(trace["verified"])
        self.assertEqual(trace["scan_evaluations"], 0)
        self.assertEqual(len(trace["rows"]), 1)
        self.assertEqual(trace["rows"][0]["stage"], "full-game certificate")
        self.assertEqual(trace["rows"][0]["G"], case["selected"]["G"])


if __name__ == "__main__":
    unittest.main()
