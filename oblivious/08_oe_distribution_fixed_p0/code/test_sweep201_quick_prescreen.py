"""Check equivalent search advantages, fallback and unchanged dispatch scope."""

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from oblivious.oe_p0.code.primitives import Parameters
from oblivious.oe_distribution_fixed_p0.code.model import FixedOutsideMarginalExperiment,_OutsideResponse
from oblivious.oe_distribution_fixed_p0.code.multi_p0_endpoint_recovery import EndpointSafeConditionalExperiment
from oblivious.oe_distribution_fixed_p0.code.conditional_model import _ConditionalResponse
from oblivious.oe_distribution_fixed_p0.code import sweep201_quick_prescreen as quick
from oblivious.oe_distribution_fixed_p0.code import sweep201_quick_entry as entry


class QuickPrescreenTests(unittest.TestCase):
    def test_pinned_no_jacobian_response_matches_original_for_both_closures(self):
        for cls,old,new in ((FixedOutsideMarginalExperiment,_OutsideResponse,quick.QuickMarginalResponse),
                            (EndpointSafeConditionalExperiment,_ConditionalResponse,quick.QuickConditionalResponse)):
            with self.subTest(closure=cls.__name__):
                experiment = cls(n=11,params=Parameters(rho=.6),p0=.5,order=64)
                alpha = np.linspace(.2,.8,experiment.n)
                expected = old(experiment).calculate(alpha)
                engine = new(experiment)
                with patch.object(experiment,'evaluate',side_effect=AssertionError('Search must not certify')):
                    actual = engine.calculate(alpha)
                np.testing.assert_allclose(actual['advantage'],expected['advantage'],atol=1e-12,rtol=0)
                np.testing.assert_allclose(actual['demand'],expected['demand'],atol=1e-12,rtol=0)
                self.assertNotIn('jacobian',actual)
                self.assertNotIn('verified',actual)
                self.assertEqual(engine.fallback_calls,0)
                self.assertLess(actual['quick_prescreen']['jump_invariant_l1'],1e-12)

    def test_unsafe_pinned_law_uses_original_response(self):
        experiment = FixedOutsideMarginalExperiment(n=7,params=Parameters(rho=.6),p0=.5,order=64)
        engine = quick.QuickMarginalResponse(experiment)
        sentinel = dict(advantage=np.full(7,.123),jacobian=np.eye(7))
        fake = SimpleNamespace(calculate=lambda alpha:sentinel)
        engine._original = fake
        with patch.object(quick,'splu',return_value=SimpleNamespace(solve=lambda rhs:np.full(27,np.nan))):
            result = engine.calculate(np.zeros(7))
        self.assertEqual(engine.fallback_calls,1)
        np.testing.assert_array_equal(result['advantage'],sentinel['advantage'])
        self.assertEqual(result['quick_prescreen']['method'],'Original response fallback')

    def test_expanded_concurrency_preserves_scope_and_explicit_child_entrypoint(self):
        args = entry.batch_arguments(['--workers','6','--resume'])
        self.assertEqual(args.workers,6)
        self.assertTrue(args.resume)
        self.assertEqual(args.p0s,[.1,.3,.5,.7,.8,.9])
        self.assertEqual(args.rhos,[.6,.7,.8,.9])
        self.assertEqual(args.grids,[201,251])
        command = ['python','-m',entry.WORKER_MODULE,'--p0','.8','--rho','.9']
        self.assertEqual(entry.child_command(command,True)[2],entry.ENTRY_MODULE)
        self.assertIn('--no-quick-prescreen',entry.child_command(command,False))
        self.assertEqual(command[2],entry.WORKER_MODULE)


if __name__ == '__main__':
    unittest.main()
