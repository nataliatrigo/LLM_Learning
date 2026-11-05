"""External policies must pass fresh target certification before publication."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from oblivious.oe_distribution_fixed_p0.code import multi_p0_proposal_resume as proposal_resume


class ExternalPolicyProposalTests(unittest.TestCase):
    def proposal(self):
        return dict(configuration=dict(p0=.1,n=3), policy=np.array([0.,1.,0.]),
                    input_sha256={}, preserved_prefix='preserved/proposal',
                    proposer_numerical_source_sha256={}, proposer_script_sha256='closed-script-hash')

    def test_a_new_certificate_supplies_the_result_instead_of_imported_values(self):
        proposal = self.proposal()
        fresh = dict(verified=True, dp_diagnostics=dict(verified=True),
                     stationary_diagnostics=dict(verified=True), value=np.array([3.,4.,5.]))
        experiment = SimpleNamespace(configuration=lambda:proposal['configuration'],
                                     evaluate=Mock(return_value=fresh))
        fallback = Mock()
        runner = proposal_resume.runner
        with patch.object(proposal_resume.endpoint, 'install'), \
                patch.object(runner, 'baseline_seed', return_value=({},{})), \
                patch.object(runner, 'solve_from_proposals', fallback):
            proposal_resume.install(proposal)
            result = runner.solve_from_proposals(experiment, [], 20, lambda _:None)
        self.assertIs(result, fresh)
        np.testing.assert_array_equal(experiment.evaluate.call_args.args[0], proposal['policy'])
        self.assertFalse(result['search_metadata']['imported_stationary_or_value_reuse'])
        fallback.assert_not_called()

    def test_a_failed_fresh_certificate_falls_back_even_if_proposal_was_verified(self):
        proposal = self.proposal()
        proposal['original_proposal_verified'] = True
        failed = dict(verified=False, dp_diagnostics=dict(verified=False),
                      stationary_diagnostics=dict(verified=True))
        experiment = SimpleNamespace(configuration=lambda:proposal['configuration'],
                                     evaluate=Mock(return_value=failed))
        fallback_result = dict(verified=False)
        fallback = Mock(return_value=fallback_result)
        runner = proposal_resume.runner
        with patch.object(proposal_resume.endpoint, 'install'), \
                patch.object(runner, 'baseline_seed', return_value=({},{})), \
                patch.object(runner, 'solve_from_proposals', fallback):
            proposal_resume.install(proposal)
            result = runner.solve_from_proposals(experiment, [], 20, lambda _:None)
        self.assertIs(result, fallback_result)
        self.assertFalse(result['verified'])
        fallback.assert_called_once()


if __name__ == '__main__':
    unittest.main()
