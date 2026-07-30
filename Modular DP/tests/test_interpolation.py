"""Fast validation of triangular-grid barycentric interpolation."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np


EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))

from modular_dp.interpolation import TriangularGrid  # noqa: E402


class TriangularGridTests(unittest.TestCase):
    def test_node_count_and_reachable_coordinates(self) -> None:
        grid = TriangularGrid(9)
        self.assertEqual(grid.node_count, 45)
        self.assertTrue(np.all(grid.x >= 0.0))
        self.assertTrue(np.all(grid.y >= 0.0))
        self.assertTrue(np.all(grid.x + grid.y <= 1.0 + 1e-15))

    def test_stencil_reproduces_every_grid_node_exactly(self) -> None:
        grid = TriangularGrid(17)
        values = np.sin(np.arange(grid.node_count, dtype=float))
        stencil = grid.stencil(grid.x, grid.y)
        self.assertEqual(stencil.indices.shape, (grid.node_count, 3))
        self.assertEqual(stencil.weights.shape, (grid.node_count, 3))
        np.testing.assert_array_equal(stencil.apply(values), values)

    def test_piecewise_barycentric_interpolation_is_affine_exact(self) -> None:
        grid = TriangularGrid(11)
        values = 2.5 - 1.75 * grid.x + 3.25 * grid.y
        query_x = np.array([0.03, 0.21, 0.49, 0.73, 0.125])
        query_y = np.array([0.04, 0.52, 0.11, 0.20, 0.875])
        expected = 2.5 - 1.75 * query_x + 3.25 * query_y
        actual = grid.interpolate(values, query_x, query_y)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2e-14)

    def test_nested_grid_indices_are_exact(self) -> None:
        coarse = TriangularGrid(5)
        fine = TriangularGrid(17)
        self.assertTrue(coarse.is_nested_in(fine))
        fine_indices = coarse.indices_in(fine)
        np.testing.assert_array_equal(fine.x[fine_indices], coarse.x)
        np.testing.assert_array_equal(fine.y[fine_indices], coarse.y)
        self.assertFalse(TriangularGrid(6).is_nested_in(fine))

    def test_points_outside_simplex_are_rejected(self) -> None:
        grid = TriangularGrid(5)
        with self.assertRaises(ValueError):
            grid.stencil(np.array([0.6]), np.array([0.41]))
        with self.assertRaises(ValueError):
            grid.stencil(np.array([-0.01]), np.array([0.2]))


if __name__ == "__main__":
    unittest.main()
