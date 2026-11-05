"""Proposal screens can save work but cannot replace an equilibrium certificate."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from oblivious.oe_distribution_fixed_p0.code import multi_p0_fast_run as fast
from oblivious.oe_distribution_fixed_p0.code.conditional_model import FixedOutsideConditionalExperiment
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment
from oblivious.oe_p0.code.primitives import Parameters


class FastProposalTests(unittest.TestCase):
    def test_distant_proposal_is_skipped_and_myopic_solution_gets_full_certificate(self):
        for cls in (FixedOutsideMarginalExperiment, FixedOutsideConditionalExperiment):
            with self.subTest(model=cls.__name__):
                e = cls(n=7, params=Parameters(rho=.6, gamma=0.), p0=.3)
                proposals = [('historical_p0_0p5_proposal', dict(grid=e.grid, policy=np.ones(7))),
                             ('same_p0_previous_grid', dict(grid=e.grid, policy=np.zeros(7)))]
                with patch.object(e, 'evaluate', wraps=e.evaluate) as full:
                    result = fast.fast_solve_from_proposals(e, proposals, 100, lambda _:None)
                self.assertTrue(result['verified'])
                np.testing.assert_array_equal(result['policy'], np.zeros(7))
                self.assertEqual(full.call_count, 1)
                self.assertTrue(result['search_metadata']['proposal_prescreen_full_certificate_required'])
                self.assertEqual(result['search_metadata']['proposal_prescreen_inputs'], fast.helper_reference())

    def test_zero_exploratory_residual_cannot_publish_a_failed_final_certificate(self):
        rejected = dict(verified=False, dp_diagnostics=dict(conditional_regret=1e-3))
        e = SimpleNamespace(grid=np.linspace(0,1,3),
                            evaluate=Mock(return_value=dict(rejected)),
                            solve=Mock(return_value=dict(rejected)))
        engine = SimpleNamespace(calculate=lambda _:dict(advantage=np.zeros(3)))
        proposal = [('proposal', dict(grid=e.grid, policy=np.zeros(3)))]
        with patch.object(fast, '_OutsideResponse', return_value=engine):
            with self.assertRaisesRegex(ArithmeticError, 'No certified solution'):
                fast.fast_solve_from_proposals(e, proposal, 10, lambda _:None)
        self.assertEqual(e.evaluate.call_count, 1)
        self.assertEqual(e.solve.call_count, 1)


if __name__ == '__main__':
    unittest.main()
