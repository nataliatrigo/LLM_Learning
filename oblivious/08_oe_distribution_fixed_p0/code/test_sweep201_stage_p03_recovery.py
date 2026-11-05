"""Imported candidate metadata cannot replace the fresh target certificate."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from oblivious.oe_distribution_fixed_p0.code.sweep201_stage_p03_recovery import certify_policy_only


class StagingCertificateTests(unittest.TestCase):
    def proposal(self):
        return dict(configuration=dict(n=201,p0=.3),policy=np.array([.2,.8]),
                    input_sha256={},archived_proposal='diagnostic-only')

    def test_fresh_target_result_supplies_values_and_laws(self):
        proposal = self.proposal()
        fresh = dict(verified=True,dp_diagnostics=dict(verified=True),
                     stationary_diagnostics=dict(verified=True),value=np.array([7.,8.]))
        experiment = SimpleNamespace(configuration=lambda:proposal['configuration'],
                                     evaluate=Mock(return_value=fresh))
        result = certify_policy_only(experiment,proposal)
        self.assertIs(result,fresh)
        np.testing.assert_array_equal(result['value'],[7,8])
        np.testing.assert_array_equal(experiment.evaluate.call_args.args[0],proposal['policy'])
        self.assertFalse(result['search_metadata']['imported_stationary_or_value_reuse'])

    def test_false_fresh_certificate_cannot_be_committed(self):
        proposal = self.proposal()
        experiment = SimpleNamespace(configuration=lambda:proposal['configuration'],
            evaluate=Mock(return_value=dict(verified=False,dp_diagnostics=dict(verified=False),
                                            stationary_diagnostics=dict(verified=True))))
        with self.assertRaisesRegex(ArithmeticError,'did not verify'):
            certify_policy_only(experiment,proposal)


if __name__ == '__main__':
    unittest.main()
