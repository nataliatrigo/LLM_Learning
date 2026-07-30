"""Regular triangular grids and reusable barycentric interpolation stencils.

The normalized forgetting state space is the closed simplex

``x >= 0, y >= 0, x + y <= 1``.

``TriangularGrid(n)`` places ``n`` equally spaced nodes on each coordinate
axis.  Successor states generally fall between nodes, so the dynamic-program
solver builds a stencil once and reuses it at every value-iteration step.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class InterpolationStencil:
    """Three-node barycentric stencil for one or more query points."""

    indices: IntArray
    weights: FloatArray
    output_shape: tuple[int, ...]

    def __post_init__(self) -> None:
        indices = np.asarray(self.indices, dtype=np.int64)
        weights = np.asarray(self.weights, dtype=float)
        if indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError("stencil indices must have shape (m, 3)")
        if weights.shape != indices.shape:
            raise ValueError("stencil weights must match the index shape")
        if int(np.prod(self.output_shape, dtype=np.int64)) != indices.shape[0]:
            raise ValueError("output_shape is inconsistent with stencil size")
        object.__setattr__(self, "indices", indices)
        object.__setattr__(self, "weights", weights)

    @property
    def point_count(self) -> int:
        return self.indices.shape[0]

    def apply(self, node_values: ArrayLike) -> FloatArray:
        """Interpolate a scalar value stored at every grid node."""
        values = np.asarray(node_values, dtype=float)
        if values.ndim != 1:
            raise ValueError("node_values must be one-dimensional")
        if self.indices.size and int(self.indices.max()) >= values.size:
            raise ValueError("node_values does not cover all stencil indices")
        interpolated = np.sum(values[self.indices] * self.weights, axis=1)
        return interpolated.reshape(self.output_shape)


@dataclass(frozen=True, slots=True)
class TriangularGrid:
    """Uniform triangular grid on the normalized reachable state space.

    Rows are ordered by increasing ``y`` and, within a row, increasing ``x``.
    The three vertices of every cell use a fixed diagonal, which makes the
    interpolation affine-exact and deterministic on shared cell boundaries.
    """

    n: int
    x: FloatArray = field(init=False, repr=False)
    y: FloatArray = field(init=False, repr=False)
    lattice_i: IntArray = field(init=False, repr=False)
    lattice_j: IntArray = field(init=False, repr=False)
    _index_map: IntArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.n, bool) or int(self.n) != self.n or self.n < 2:
            raise ValueError("n must be an integer of at least 2")
        n = int(self.n)
        h = 1.0 / (n - 1)
        lattice_j = np.repeat(np.arange(n, dtype=np.int64), n - np.arange(n))
        lattice_i = np.concatenate(
            [np.arange(n - j, dtype=np.int64) for j in range(n)]
        )
        index_map = np.full((n, n), -1, dtype=np.int64)
        index_map[lattice_j, lattice_i] = np.arange(lattice_i.size)

        object.__setattr__(self, "n", n)
        object.__setattr__(self, "lattice_i", lattice_i)
        object.__setattr__(self, "lattice_j", lattice_j)
        object.__setattr__(self, "x", lattice_i.astype(float) * h)
        object.__setattr__(self, "y", lattice_j.astype(float) * h)
        object.__setattr__(self, "_index_map", index_map)

    @property
    def spacing(self) -> float:
        return 1.0 / (self.n - 1)

    @property
    def node_count(self) -> int:
        return self.n * (self.n + 1) // 2

    @property
    def coordinates(self) -> FloatArray:
        return np.column_stack((self.x, self.y))

    def node_index(self, i: ArrayLike, j: ArrayLike) -> IntArray:
        """Return flat indices for valid integer lattice coordinates."""
        ii, jj = np.broadcast_arrays(
            np.asarray(i, dtype=np.int64), np.asarray(j, dtype=np.int64)
        )
        valid = (ii >= 0) & (jj >= 0) & (ii + jj < self.n)
        if not np.all(valid):
            raise ValueError("invalid triangular-grid lattice coordinate")
        return self._index_map[jj, ii]

    def identity_stencil(self) -> InterpolationStencil:
        """Return an exact stencil mapping each node to itself."""
        indices = np.arange(self.node_count, dtype=np.int64)
        repeated = np.repeat(indices[:, None], 3, axis=1)
        weights = np.zeros((self.node_count, 3), dtype=float)
        weights[:, 0] = 1.0
        return InterpolationStencil(repeated, weights, (self.node_count,))

    def stencil(self, x: ArrayLike, y: ArrayLike) -> InterpolationStencil:
        """Build barycentric interpolation weights for points in the simplex.

        Queries that coincide with grid nodes receive an explicit weight-one
        stencil.  Consequently, applying a stencil to all grid coordinates
        reproduces the original node array exactly, without interpolation
        roundoff.
        """
        x_query, y_query = np.broadcast_arrays(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        )
        output_shape = x_query.shape
        qx = x_query.ravel().copy()
        qy = y_query.ravel().copy()
        if not np.all(np.isfinite(qx)) or not np.all(np.isfinite(qy)):
            raise ValueError("interpolation coordinates must be finite")

        coordinate_tolerance = 128.0 * np.finfo(float).eps * self.n
        if np.any(qx < -coordinate_tolerance) or np.any(
            qy < -coordinate_tolerance
        ):
            raise ValueError("interpolation point lies outside the simplex")
        total = qx + qy
        if np.any(total > 1.0 + coordinate_tolerance):
            raise ValueError("interpolation point lies outside the simplex")

        qx = np.maximum(qx, 0.0)
        qy = np.maximum(qy, 0.0)
        # Project only floating-point overshoots of the hypotenuse.
        total = qx + qy
        overshoot = total > 1.0
        qx[overshoot] /= total[overshoot]
        qy[overshoot] /= total[overshoot]

        u = qx * (self.n - 1)
        v = qy * (self.n - 1)
        lattice_tolerance = 128.0 * np.finfo(float).eps * self.n
        rounded_u = np.rint(u)
        rounded_v = np.rint(v)
        u = np.where(np.abs(u - rounded_u) <= lattice_tolerance, rounded_u, u)
        v = np.where(np.abs(v - rounded_v) <= lattice_tolerance, rounded_v, v)

        is_node = (
            (np.abs(u - np.rint(u)) <= lattice_tolerance)
            & (np.abs(v - np.rint(v)) <= lattice_tolerance)
            & (np.rint(u) + np.rint(v) <= self.n - 1)
        )

        point_count = qx.size
        indices = np.empty((point_count, 3), dtype=np.int64)
        weights = np.zeros((point_count, 3), dtype=float)

        if np.any(is_node):
            node_i = np.rint(u[is_node]).astype(np.int64)
            node_j = np.rint(v[is_node]).astype(np.int64)
            node_indices = self._index_map[node_j, node_i]
            indices[is_node, :] = node_indices[:, None]
            weights[is_node, 0] = 1.0

        off_node = ~is_node
        if np.any(off_node):
            off_positions = np.flatnonzero(off_node)
            uu = u[off_node]
            vv = v[off_node]
            cell_i = np.floor(uu).astype(np.int64)
            cell_j = np.floor(vv).astype(np.int64)
            cell_i = np.clip(cell_i, 0, self.n - 2)
            cell_j = np.clip(cell_j, 0, self.n - 2)
            frac_x = uu - cell_i
            frac_y = vv - cell_j

            # The upper half only exists for squares strictly inside the
            # simplex.  Boundary cells consist of their lower triangle.
            upper = (frac_x + frac_y > 1.0 + lattice_tolerance) & (
                cell_i + cell_j <= self.n - 3
            )
            lower = ~upper

            if np.any(lower):
                positions = off_positions[lower]
                i_lower = cell_i[lower]
                j_lower = cell_j[lower]
                r = frac_x[lower]
                s = frac_y[lower]
                lower_weights = np.column_stack((1.0 - r - s, r, s))
                lower_weights[np.abs(lower_weights) <= lattice_tolerance] = 0.0
                lower_weights = np.maximum(lower_weights, 0.0)
                lower_weights /= lower_weights.sum(axis=1, keepdims=True)
                indices[positions] = np.column_stack(
                    (
                        self._index_map[j_lower, i_lower],
                        self._index_map[j_lower, i_lower + 1],
                        self._index_map[j_lower + 1, i_lower],
                    )
                )
                weights[positions] = lower_weights

            if np.any(upper):
                positions = off_positions[upper]
                i_upper = cell_i[upper]
                j_upper = cell_j[upper]
                r = frac_x[upper]
                s = frac_y[upper]
                indices[positions] = np.column_stack(
                    (
                        self._index_map[j_upper, i_upper + 1],
                        self._index_map[j_upper + 1, i_upper],
                        self._index_map[j_upper + 1, i_upper + 1],
                    )
                )
                weights[positions] = np.column_stack(
                    (1.0 - s, 1.0 - r, r + s - 1.0)
                )

        if np.any(indices < 0):
            raise RuntimeError("internal error while constructing triangular stencil")
        return InterpolationStencil(indices, weights, output_shape)

    barycentric_stencil = stencil

    def interpolate(
        self, node_values: ArrayLike, x: ArrayLike, y: ArrayLike
    ) -> FloatArray:
        """Convenience wrapper that builds and applies a stencil."""
        return self.stencil(x, y).apply(node_values)

    def is_nested_in(self, finer: "TriangularGrid") -> bool:
        """Whether every node of this grid is also a node of ``finer``."""
        return (
            finer.n >= self.n
            and (finer.n - 1) % (self.n - 1) == 0
        )

    def indices_in(self, finer: "TriangularGrid") -> IntArray:
        """Map this grid's nodes to the corresponding nodes of a nested grid."""
        if not self.is_nested_in(finer):
            raise ValueError(
                f"grid n={self.n} is not nested in grid n={finer.n}"
            )
        stride = (finer.n - 1) // (self.n - 1)
        return finer.node_index(self.lattice_i * stride, self.lattice_j * stride)
