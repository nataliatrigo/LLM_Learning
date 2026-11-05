"""Endpoint roundoff repair preserves every valid probability, however small."""

import unittest

import numpy as np

from oblivious.oe_distribution_fixed_p0.code.multi_p0_endpoint_recovery import (
    EndpointSafeConditionalExperiment, project_demand_endpoints,
)
from oblivious.oe_p0.code.primitives import Parameters


class DemandEndpointRecoveryTests(unittest.TestCase):
    def test_only_invalid_endpoint_roundoff_changes_and_tiny_probabilities_survive(self):
        tiny = np.nextafter(0., 1.)
        raw = np.array([[1+2*np.finfo(float).eps, tiny], [.2, .8]])
        projected, metadata = project_demand_endpoints(raw)
        self.assertEqual(projected[0,0], 1.)
        self.assertEqual(projected[0,1], tiny)
        valid = (raw >= 0)&(raw <= 1)
        np.testing.assert_array_equal(projected[valid], raw[valid])
        self.assertEqual(metadata['changed_entry_count'], 1)
        self.assertEqual(metadata['positive_probability_floor'], 0.)
        self.assertTrue(metadata['valid_entries_unchanged'])
        np.testing.assert_array_equal(raw, [[1+2*np.finfo(float).eps,tiny],[.2,.8]])

    def test_meaningful_probability_errors_are_rejected(self):
        for bad in (-1e-8, 1+1e-8, np.nan, np.inf):
            with self.subTest(bad=bad):
                with self.assertRaisesRegex(ArithmeticError, 'exceeds explicit'):
                    project_demand_endpoints([[.2,bad],[.4,.6]])

    def test_actual_failed_calibration_has_machine_precision_projection_only(self):
        e = EndpointSafeConditionalExperiment(n=51, params=Parameters(rho=.97), p0=.1, order=256)
        metadata = e.configuration()['demand_endpoint_roundoff_projection']
        self.assertGreater(metadata['changed_entry_count'], 0)
        self.assertLessEqual(metadata['maximum_absolute_change'], 5e-16)
        self.assertGreater(e.game.D.min(), 0)
        self.assertLessEqual(e.game.D.max(), 1)
        np.testing.assert_array_equal(e.game.selection, np.stack((e.game.D,e.game.D.T)))
        self.assertLess(np.max(abs(e.game.D+e.game.D.T+e.game.outside-1)), 2e-15)


if __name__ == '__main__':
    unittest.main()
