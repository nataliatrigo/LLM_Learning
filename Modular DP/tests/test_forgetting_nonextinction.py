"""Checks for the observation-forgetting non-extinction certificate."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp.primitives import SellerPrimitives  # noqa: E402
from scripts.verify_observation_forgetting_nonextinction import (  # noqa: E402
    boundary_demand,
    solve_boundary_enclosure,
    synchronizing_word,
    word_probabilities,
)


class ObservationForgettingNonExtinctionTests(unittest.TestCase):
    def test_boundary_demand_is_monotone_and_symmetric_at_half(self) -> None:
        x = np.linspace(0.0, 1.0, 101)
        demand = boundary_demand(x, rho=0.95, p0=0.5)
        self.assertTrue(np.all(np.diff(demand) >= 0.0))
        self.assertAlmostEqual(float(demand[50]), 0.5, places=14)
        np.testing.assert_allclose(demand + demand[::-1], 1.0, atol=2e-14)

    def test_synchronizing_word_lies_strictly_inside_action_interval(self) -> None:
        word = synchronizing_word(
            rho=0.95, failures=19, successes=6, repetitions=2
        )
        self.assertEqual(word["notation"], "(0^19 1^6)^2")
        self.assertEqual(word["length"], 50)
        self.assertGreater(float(word["image_lower"]), 0.33)
        self.assertLess(float(word["image_upper"]), 0.42)
        self.assertAlmostEqual(
            float(word["image_width"]), 0.95**50, places=14
        )

    def test_small_cell_enclosure_certifies_an_interior_subinterval(self) -> None:
        certificate = solve_boundary_enclosure(
            SellerPrimitives(),
            rho=0.95,
            cells=1_000,
            certified_interval=(0.30, 0.40),
            tolerance=1e-11,
            max_iterations=5_000,
            demand_padding=5e-13,
        )
        self.assertGreater(certificate.minimum_certified_advantage, 0.0)
        self.assertTrue(
            np.all(certificate.lower_value <= certificate.upper_value)
        )
        self.assertAlmostEqual(certificate.demand_floor, 2.0**-21, places=18)

    def test_certificate_rejects_unsupported_parameter_edges(self) -> None:
        with self.assertRaisesRegex(ValueError, "nonnegative net rewards"):
            solve_boundary_enclosure(
                SellerPrimitives(revenue=0.10),
                rho=0.95,
                cells=100,
                certified_interval=(0.30, 0.40),
                tolerance=1e-8,
                max_iterations=10,
                demand_padding=5e-13,
            )
        word = synchronizing_word(
            rho=0.95, failures=19, successes=6, repetitions=2
        )
        with self.assertRaisesRegex(ValueError, "both outcomes"):
            word_probabilities(SellerPrimitives(p1=0.0), word)


if __name__ == "__main__":
    unittest.main()
