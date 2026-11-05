"""Check equilibrium reporting independently of the auxiliary pure update."""

import copy
import unittest

from oblivious.oe_following_monopoly.code.run_experiment import select_mixed_equilibrium


def certified_candidate(p0, mixed_nodes=0):
    """A small reporting fixture with all economically relevant certificates."""
    return dict(
        p0=p0,
        mixed_nodes=mixed_nodes,
        candidate_verified=True,
        verified=True,
        dp_diagnostics=dict(verified=True),
        stationary_diagnostics=dict(verified=True),
        monopoly_certificate=dict(verified=True),
    )


class MixedEquilibriumSelectionChecks(unittest.TestCase):
    def test_pure_iteration_limit_does_not_outrank_certified_mixing(self):
        pure = certified_candidate(.60)
        mixed = certified_candidate(.62, mixed_nodes=1)
        case = dict(
            candidates=[pure, mixed],
            selected=pure,
            iteration_status="converged",
            iteration_history=[dict(p0=pure["p0"])],
        )

        selected, reason = select_mixed_equilibrium(case)

        self.assertIs(selected, mixed)
        self.assertIsInstance(reason, str)

    def test_lowest_p0_is_chosen_within_the_interior_candidates(self):
        pure = certified_candidate(.35)
        mixed_low = certified_candidate(.61, mixed_nodes=1)
        mixed_high = certified_candidate(.72, mixed_nodes=2)

        selected, _ = select_mixed_equilibrium(
            dict(candidates=[mixed_high, pure, mixed_low])
        )

        self.assertIs(selected, mixed_low)

    def test_each_failed_certificate_excludes_a_mixed_candidate(self):
        certificate_paths = (
            ("candidate_verified",),
            ("verified",),
            ("dp_diagnostics", "verified"),
            ("stationary_diagnostics", "verified"),
            ("monopoly_certificate", "verified"),
        )
        pure = certified_candidate(.60)
        for path in certificate_paths:
            with self.subTest(certificate=path):
                invalid_mixed = copy.deepcopy(certified_candidate(.62, mixed_nodes=1))
                target = invalid_mixed
                for key in path[:-1]:
                    target = target[key]
                target[path[-1]] = False

                selected, _ = select_mixed_equilibrium(
                    dict(candidates=[invalid_mixed, pure])
                )

                self.assertIs(selected, pure)

    def test_missing_certificate_does_not_count_as_verified(self):
        pure = certified_candidate(.60)
        invalid_mixed = certified_candidate(.62, mixed_nodes=1)
        del invalid_mixed["monopoly_certificate"]

        selected, _ = select_mixed_equilibrium(
            dict(candidates=[invalid_mixed, pure])
        )

        self.assertIs(selected, pure)

    def test_a_pure_endpoint_remains_valid_when_no_interior_exists(self):
        pure_low = certified_candidate(.35)
        pure_high = certified_candidate(.70)

        selected, _ = select_mixed_equilibrium(
            dict(candidates=[pure_high, pure_low])
        )

        self.assertIs(selected, pure_low)

    def test_without_verified_candidates_the_saved_result_is_only_diagnostic(self):
        diagnostic = certified_candidate(.62, mixed_nodes=1)
        diagnostic["candidate_verified"] = False

        selected, reason = select_mixed_equilibrium(
            dict(candidates=[diagnostic], selected=diagnostic)
        )

        self.assertIs(selected, diagnostic)
        self.assertFalse(selected["candidate_verified"])
        self.assertIn("diagnostic", reason.lower())

    def test_saved_verified_flag_cannot_promote_an_uncertified_diagnostic(self):
        diagnostic = certified_candidate(.62, mixed_nodes=1)
        diagnostic["stationary_diagnostics"]["verified"] = False

        selected, reason = select_mixed_equilibrium(
            dict(candidates=[diagnostic], selected=diagnostic)
        )

        self.assertFalse(selected["candidate_verified"])
        self.assertIn("diagnostic", reason.lower())
        # Selection must not rewrite the saved exploratory candidate in place.
        self.assertTrue(diagnostic["candidate_verified"])


if __name__ == "__main__":
    unittest.main()
