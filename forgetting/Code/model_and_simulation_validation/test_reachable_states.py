"""Geometry checks for the reachable-set visualization."""
import unittest
from types import SimpleNamespace

import numpy as np

from forgetting.Code.experiments.states import (
    intersect_intervals, policy_intervals, reachable_cover,
)


class ReachableSetPlotTests(unittest.TestCase):
    def test_table_advantage_excludes_cantor_gaps(self):
        from forgetting.Code.experiments.policy_summary import advantage_summary

        x = np.array([0., .3, .5, .7, 1.])
        advantage = np.array([-1., -1., 1., -1., -1.])
        cantor = advantage_summary(.3, x, advantage)
        interval = advantage_summary(.5, x, advantage)
        self.assertAlmostEqual(cantor['A_rho'], -1.)
        self.assertAlmostEqual(cantor['A_rho_cover_radius_x'], .3**16)
        self.assertEqual(interval['A_rho'], 1.)
        self.assertEqual(interval['A_rho_cover_radius_x'], 0.)

    def test_cylinder_gaps_are_nested_and_have_the_expected_length(self):
        rho = .45
        np.testing.assert_allclose(reachable_cover(rho, 1), [[0,.45],[.55,1]])
        for depth in (2,5,10):
            cover = reachable_cover(rho, depth)
            parent = reachable_cover(rho, depth-1)
            self.assertEqual(len(cover), 2**depth)
            np.testing.assert_allclose(cover[:,1]-cover[:,0], rho**depth, atol=1e-15)
            self.assertTrue(np.all((cover[:,1] <= rho) | (cover[:,0] >= 1-rho)))
            for lower, upper in cover:
                self.assertTrue(np.any((parent[:,0] <= lower+1e-15) & (parent[:,1] >= upper-1e-15)))

    def test_overlapping_maps_cover_the_full_interval(self):
        for rho in (.5,.7,.99):
            for depth in (0,1,10):
                np.testing.assert_array_equal(reachable_cover(rho, depth), [[0,1]])

    def test_positive_policy_in_a_geometric_gap_is_not_shown_as_reachable(self):
        solution = SimpleNamespace(coordinate=np.array([0,.45,.5,.55,1]),
                                   advantage=np.array([-1,0,1,0,-1]))
        intervals = policy_intervals(solution, 0)
        positive = [(a,b) for a,b,k in intervals if k=='product2']
        self.assertEqual(positive, [(.45,.55)])
        self.assertEqual(len(intersect_intervals(reachable_cover(.45, 4), *positive[0])), 0)
        self.assertEqual(len(intersect_intervals(reachable_cover(.5, 4), *positive[0])), 1)
