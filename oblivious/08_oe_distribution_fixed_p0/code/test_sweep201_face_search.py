"""Only original full certificates can accept an adaptive-face proposal."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import numpy as np

from oblivious.oe_distribution_fixed_p0.code import sweep201_face_search as face


class FakeQuick:
    def __init__(self,experiment):
        pass

    def calculate(self,alpha):
        return dict(advantage=np.zeros(len(alpha)))


class FaceCertificateGateTests(unittest.TestCase):
    def test_exact_priority_proposal_skips_distant_other_prescreens(self):
        grid = np.linspace(0,1,3)
        fresh = dict(verified=True,dp_diagnostics=dict(verified=True),
                     stationary_diagnostics=dict(verified=True))
        experiment = SimpleNamespace(grid=grid,evaluate=Mock(return_value=fresh))
        calculate = Mock(return_value=dict(advantage=np.zeros(3)))
        engine = SimpleNamespace(calculate=calculate)
        fallback = Mock(side_effect=AssertionError('A fully certified proposal is sufficient'))
        proposals = [('historical_p0_0p5_proposal',dict(grid=grid,policy=np.ones(3))),
                     ('fresh_monopoly',dict(grid=grid,policy=np.zeros(3)))]
        with patch.object(face.quick,'QuickMarginalResponse',return_value=engine):
            result = face.solve_with_face(experiment,proposals,35,lambda message:None,fallback)
        calculate.assert_called_once()
        np.testing.assert_array_equal(calculate.call_args.args[0],np.zeros(3))
        experiment.evaluate.assert_called_once()
        self.assertTrue(result['verified'])
        self.assertFalse(result['search_metadata']['adaptive_face_polishing_performed'])

    def test_zero_quick_regret_with_failed_full_certificate_falls_back(self):
        experiment = SimpleNamespace(grid=np.linspace(0,1,3),evaluate=Mock(return_value=dict(
            verified=False,dp_diagnostics=dict(verified=False),stationary_diagnostics=dict(verified=True))))
        fallback = Mock(return_value=dict(verified=True,attempts=[]))
        proposals = [('warm',dict(grid=experiment.grid,policy=np.zeros(3)))]
        with patch.object(face.quick,'QuickMarginalResponse',FakeQuick):
            result = face.solve_with_face(experiment,proposals,35,lambda message:None,fallback)
        experiment.evaluate.assert_called_once()
        fallback.assert_called_once()
        self.assertFalse(result['attempts'][-1]['verified'])

    def test_quick_regret_outside_small_face_never_reaches_full_acceptance(self):
        experiment = SimpleNamespace(grid=np.linspace(0,1,3),evaluate=Mock())
        class ViolatingQuick(FakeQuick):
            def calculate(self,alpha):
                return dict(advantage=np.array([0,0,1.]))
        unresolved = dict(policy=np.zeros(3),quick_metrics=dict(conditional_regret=1.,complementarity_residual=1.),
                          search_diagnostics=dict(global_regret_checked_on_all_nodes=True))
        fallback = Mock(return_value=dict(verified=True,attempts=[]))
        proposals = [('warm',dict(grid=experiment.grid,policy=np.zeros(3)))]
        with patch.object(face.quick,'QuickMarginalResponse',ViolatingQuick), \
                patch.object(face.proposer,'active_face_proposal',return_value=unresolved):
            face.solve_with_face(experiment,proposals,35,lambda message:None,fallback)
        experiment.evaluate.assert_not_called()
        fallback.assert_called_once()

    def test_incomplete_reported_verified_flag_is_insufficient(self):
        self.assertFalse(face.fully_verified(SimpleNamespace(),dict(verified=True)))

    def test_conditional_acceptance_requires_reversed_order_certificate(self):
        experiment = object.__new__(face.FixedOutsideConditionalExperiment)
        result = dict(verified=True,dp_diagnostics=dict(verified=True),
                      stationary_diagnostics=dict(verified=True),
                      conditional_diagnostics=dict(verified=True,independent_order_verified=False))
        self.assertFalse(face.fully_verified(experiment,result))


if __name__ == '__main__':
    unittest.main()
