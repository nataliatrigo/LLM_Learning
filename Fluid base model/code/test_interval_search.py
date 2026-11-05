"""Search detectors must allow disconnected sets and exclude mere ties."""

import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from audit_saved_intervals import three_point_witness
from search_intervals import search_group
from search_intervals_independent import strict_witness


class IntervalDetectionTests(unittest.TestCase):
    def test_detects_strict_ordered_pattern_in_both_detectors(self):
        gap = np.array([-.1, .2, .3, -.4, -.3, .5, .2, -.2])
        first = three_point_witness(gap)
        second = strict_witness(gap, ds=.5, tolerance=0, min_width=0)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        i, j, k = first["indices"]
        self.assertLess(i, j)
        self.assertLess(j, k)
        self.assertTrue(gap[i] > 0 and gap[j] < 0 and gap[k] > 0)

    def test_single_interval_and_tied_hole_are_not_strict_separation(self):
        for gap in [np.array([-.1, .2, .3, -.2]), np.array([.2, 0., .3]),
                    np.zeros(4), -np.ones(4), np.ones(4)]:
            self.assertIsNone(three_point_witness(gap))
            self.assertIsNone(strict_witness(gap, ds=.5, tolerance=0, min_width=0))

    def test_uncertain_middle_is_not_a_boundary_bracketed_witness(self):
        lower = np.array([.1, -.02, .1])
        upper = np.array([.2, .02, .2])
        self.assertIsNotNone(three_point_witness(lower))
        self.assertIsNone(three_point_witness(lower, upper))

    def test_zero_margin_search_does_not_hide_tiny_strict_signs(self):
        gap = np.array([1e-10, -1e-10, 1e-10])
        self.assertIsNotNone(three_point_witness(gap))
        self.assertIsNotNone(strict_witness(gap, ds=.5, tolerance=0, min_width=0))
        self.assertIsNone(three_point_witness(gap, margin=1e-6))

    def test_full_engine_can_detect_two_bands_with_artificial_demand(self):
        # Positive control ONLY. The empirical search itself retains beta
        # demand; this deliberately changed function is not a counterexample.
        def two_steps(alpha, beta, threshold):
            S = np.asarray(alpha)-1
            return .02 + .4*expit(4*(S-6)) + .4*expit(4*(S-18))
        with patch("search_intervals.betaincc", side_effect=two_steps):
            result = search_group([(0., 1., .05)], .5, .3, .5, 25., 1e-7, 1e-4)
        self.assertGreater(result["strict_witness_count"], 0)


if __name__ == "__main__":
    unittest.main()
