"""Small-face search may propose policies but cannot certify them."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from oblivious.oe_distribution_fixed_p0.code import sweep201_active_face_proposal as proposer


class FakeResponse:
    def __init__(self,experiment):
        self.calls = self.fallback_calls = 0
        self.target = np.array([.2,0,1,.7])

    def calculate(self,alpha):
        self.calls += 1
        return dict(advantage=self.target-alpha,quick_prescreen=dict(method='synthetic'))


class ActiveFaceTests(unittest.TestCase):
    def test_violating_pure_nodes_enter_face_and_every_state_is_checked(self):
        experiment = SimpleNamespace(grid=np.linspace(0,1,4),
                                     evaluate=lambda alpha:self.fail('Search cannot certify'))
        with patch.object(proposer.quick,'QuickMarginalResponse',FakeResponse):
            result = proposer.active_face_proposal(experiment,np.array([0,0,1,0]),maximum_calls=100)
        self.assertFalse(result['verified'])
        self.assertFalse(result['search_diagnostics']['full_certificate_performed'])
        self.assertLess(max(result['quick_metrics'].values()),2e-12)
        np.testing.assert_allclose(result['policy'],[.2,0,1,.7],atol=1e-10,rtol=0)
        self.assertEqual(result['search_diagnostics']['rounds'][0]['face'],[0,3])

    def test_search_budget_does_not_turn_an_unresolved_candidate_into_certificate(self):
        experiment = SimpleNamespace(grid=np.linspace(0,1,4))
        with patch.object(proposer.quick,'QuickMarginalResponse',FakeResponse):
            result = proposer.active_face_proposal(experiment,np.zeros(4),maximum_calls=1)
        self.assertFalse(result['verified'])
        self.assertEqual(result['proposal_status'],'exploratory_response_budget_exhausted')
        self.assertGreater(max(result['quick_metrics'].values()),.5)


if __name__ == '__main__':
    unittest.main()
