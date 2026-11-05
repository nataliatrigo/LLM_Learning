"""Regression: strict pure bounds must not be forced to indifference."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from oblivious.oe_distribution_fixed_p0.code import sweep201_complementarity_proposal as proposer


def response_type(advantage):
    class Response:
        def __init__(self,experiment):
            self.calls = self.fallback_calls = 0

        def calculate(self,alpha):
            self.calls += 1
            return dict(advantage=advantage(alpha),quick_prescreen=dict(method='synthetic'))
    return Response


class ComplementarityProposalTests(unittest.TestCase):
    def solve(self,advantage,initial):
        experiment = SimpleNamespace(grid=np.linspace(0,1,len(initial)),
                                     evaluate=lambda alpha:self.fail('A proposal cannot certify'))
        with patch.object(proposer.quick,'QuickMarginalResponse',response_type(advantage)):
            result = proposer.active_face_proposal(experiment,np.array(initial,dtype=float),maximum_calls=60)
        self.assertFalse(result['verified'])
        self.assertFalse(result['search_diagnostics']['full_certificate_performed'])
        self.assertLess(max(result['quick_metrics'].values()),2e-12)
        return result

    def test_one_interior_and_one_strict_bound_without_forcing_both_advantages_zero(self):
        result = self.solve(lambda alpha:np.array([.3-alpha[0],-.5]),[.5,.5])
        np.testing.assert_allclose(result['policy'],[.3,0],atol=1e-11,rtol=0)
        self.assertEqual(result['search_diagnostics']['rounds'][0]['face'],[0])

    def test_two_interior_nodes_are_solved_jointly(self):
        result = self.solve(lambda alpha:np.array([.2,.8])-alpha,[.6,.4])
        np.testing.assert_allclose(result['policy'],[.2,.8],atol=1e-11,rtol=0)
        self.assertEqual(result['search_diagnostics']['rounds'][0]['face'],[0,1])

    def test_pure_solution_needs_no_indifference_root(self):
        result = self.solve(lambda alpha:np.array([-.2,.5]),[.5,.5])
        np.testing.assert_array_equal(result['policy'],[0,1])
        self.assertEqual(result['search_diagnostics']['engine_calls'],2)
        self.assertEqual(result['search_diagnostics']['rounds'],[])

    def test_second_pure_improvement_recovers_strict_policy_in_three_calls(self):
        result = self.solve(lambda alpha:np.array([.1+.1*alpha[0]-alpha[1],-.5]),[.5,.5])
        np.testing.assert_array_equal(result['policy'],[1,0])
        self.assertEqual(result['search_diagnostics']['engine_calls'],3)
        self.assertEqual(result['search_diagnostics']['rounds'],[])


if __name__ == '__main__':
    unittest.main()
