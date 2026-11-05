"""The expanded reference changes storage/work budgets, not numerical checks."""

from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np

from oblivious.oe_p0.code import positive_stationary
from oblivious.oe_p0.code.primitives import Parameters
from . import expanded_stationary as entry
from . import model, stationary
from .kernel import FixedOutsideKernel


class ExpandedStationaryChecks(unittest.TestCase):
    def test_real_three_seller_reference_is_bitwise_unchanged(self):
        grid = np.linspace(0,1,5)
        game = FixedOutsideKernel(grid,Parameters(rho=.7),p0=.5,sellers=3,order=64)
        jump,rates = game.jump(np.array([0,.2,1,.7,0]))
        before = stationary.checked_symmetric_stationary(jump,rates,game.shape)
        full_before = positive_stationary.positive_invariant(jump,rates)
        backend_before = positive_stationary._kernel
        restore = entry.install(reference_max_states=6000)
        try:
            after = model.checked_symmetric_stationary(jump,rates,game.shape)
            full_after = positive_stationary.positive_invariant(jump,rates)
        finally:
            restore()
        self.assertIs(positive_stationary._kernel,backend_before)
        np.testing.assert_array_equal(before['law'],after['law'])
        np.testing.assert_array_equal(full_before,full_after)
        np.testing.assert_allclose(after['law'],full_after,rtol=2e-11,atol=2e-13)
        self.assertEqual(after['summary']['methods'],before['summary']['methods'])
        self.assertIsNone(after['summary']['positive_iteration'])

    def test_only_work_budgets_change_and_original_tolerances_survive(self):
        check = Mock(return_value={'law':'sentinel'})
        stationary_before = stationary.checked_symmetric_stationary
        with patch.object(model,'checked_symmetric_stationary',check):
            restore = entry.install(reference_max_states=25000)
            try:
                result = model.checked_symmetric_stationary('jump','rates',(31,)*3,
                    tolerance=3e-12,stability_tolerance=7e-8)
            finally:
                restore()
        self.assertEqual(result,{'law':'sentinel'})
        self.assertIs(stationary.checked_symmetric_stationary,stationary_before)
        kwargs = check.call_args.kwargs
        self.assertEqual(kwargs['reference_max_states'],25000)
        self.assertEqual(kwargs['positive_max_iterations'],1_000_000)
        self.assertEqual(kwargs['positive_time_limit'],240)
        self.assertEqual(kwargs['tolerance'],3e-12)
        self.assertEqual(kwargs['stability_tolerance'],7e-8)

    def test_source_reference_covers_backend_and_original_positive_sources(self):
        references = entry.source_reference()
        self.assertEqual(len(references),6)
        self.assertTrue(any(name.endswith('sweep201_memory_gate.py') for name in references))
        self.assertTrue(any(name.endswith('positive_stationary.cpp') for name in references))
        for name,expected in references.items():
            self.assertEqual(sha256((entry.ROOT/name).read_bytes()).hexdigest(),expected)

    def test_unsupported_reference_or_thread_budget_is_rejected(self):
        for reference,threads in [(5999,2),(6000,1),(True,2),(6000.5,2)]:
            with self.subTest(reference=reference,threads=threads):
                with self.assertRaises(ValueError):
                    entry.install(reference,threads)


if __name__ == '__main__':
    unittest.main()
