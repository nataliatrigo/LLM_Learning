"""Independent checks of the fluid and original stochastic equations.

The dense reference below constructs the calendar kernel from interpolation
of coordinate basis functions.  It does not use the new solver's kernel,
policy-evaluation, or Bellman helpers.
"""

from __future__ import annotations

from dataclasses import replace
import itertools
import math
import unittest

import numpy as np
from scipy.special import betaincc

from forgetting.Code.fluid_convergence.numerics import (
    evaluate_stochastic_policy,
    solve_fluid,
    solve_stochastic,
    transfer_policy,
)
from forgetting.Code.model import BoundaryGrid, Primitives, solve_boundary


def dense_calendar_system(primitives, rho, grid, policy):
    """Build the original calendar kernel independently on a small grid."""
    grid = np.asarray(grid, dtype=float)
    policy = np.asarray(policy, dtype=float)
    discount = primitives.gamma
    retention = rho
    innovation = 1.0 - retention
    demand = betaincc(
        1.0 + grid / (1.0 - rho),
        1.0 + (1.0 - grid) / (1.0 - rho),
        primitives.p0,
    )
    basis = np.eye(len(grid))
    success = np.column_stack([
        np.interp(retention * grid + innovation, grid, column)
        for column in basis.T
    ])
    failure = np.column_stack([
        np.interp(retention * grid, grid, column) for column in basis.T
    ])
    probability = primitives.p1 + (primitives.p2 - primitives.p1) * policy
    transition = (
        np.diag(1.0 - demand)
        + demand[:, None]
        * (probability[:, None] * success + (1.0 - probability[:, None]) * failure)
    )
    reward = demand * (
        primitives.revenue - primitives.c1
        - (primitives.c2 - primitives.c1) * policy
    )
    return np.eye(len(grid)) - discount * transition, reward


class FluidConvergenceNumericsTests(unittest.TestCase):
    def test_constant_demand_has_exact_values_and_uses_product_one(self):
        # Different primitives catch accidental reuse of the paper calibration.
        primitives = Primitives(
            p0=.61, p1=.2, p2=.72, c1=.1, c2=.8, revenue=1.4, gamma=.93
        )
        demand = .37
        constant = lambda grid: np.full_like(grid, demand)
        expected_fluid = demand * (primitives.revenue - primitives.c1) / -math.log(primitives.gamma)
        fluid = solve_fluid(primitives, .76, 51, tol=1e-11, demand_function=constant)
        np.testing.assert_allclose(fluid["value"], expected_fluid, rtol=0, atol=2e-10)
        self.assertFalse(np.any(fluid["policy"]))
        result = solve_stochastic(
            primitives, .76, 51, tol=1e-11, demand_function=constant,
        )
        expected = demand * (primitives.revenue - primitives.c1) / (1.0 - primitives.gamma)
        np.testing.assert_allclose(result["value"], expected, rtol=0, atol=2e-10)
        self.assertFalse(np.any(result["policy"]))

    def test_stochastic_optimum_matches_the_original_boundary_solver(self):
        primitives, rho, n = Primitives(), .9, 101
        grid = np.linspace(0.0, 1.0, n)
        original = solve_boundary(
            primitives, rho, grid=BoundaryGrid(grid, n, 0, 9.0), tolerance=1e-13
        )
        result = solve_stochastic(primitives, rho, n, tol=1e-11)
        np.testing.assert_allclose(result["grid"], grid, rtol=0, atol=0)
        np.testing.assert_allclose(result["value"], original.value, rtol=0, atol=2e-10)
        np.testing.assert_allclose(result["advantage"], original.advantage, rtol=0, atol=2e-10)
        strict = np.abs(original.advantage) > 1e-8
        np.testing.assert_array_equal(
            result["policy"][strict], (original.advantage[strict] > 0).astype(int)
        )

    def test_mixed_policy_matches_an_independent_dense_calendar_solve(self):
        primitives = Primitives(
            p0=.42, p1=.23, p2=.78, c1=.08, c2=.57, revenue=1.3, gamma=.91
        )
        grid = np.array([0., .03, .11, .25, .45, .64, .8, .92, 1.])
        policy = .1 + .7 * grid
        rho = .73
        matrix, reward = dense_calendar_system(primitives, rho, grid, policy)
        expected = np.linalg.solve(matrix, reward)
        result = evaluate_stochastic_policy(primitives, rho, grid, policy)
        np.testing.assert_allclose(result["value"], expected, rtol=0, atol=2e-11)
        independent_residual = float(np.max(np.abs(matrix @ result["value"] - reward)))
        self.assertLess(independent_residual, 1e-12)
        self.assertLessEqual(
            float(np.max(np.abs(result["value"] - expected))),
            result["value_error_bound"] + 2e-11,
        )

    def test_original_beta_demand_and_no_sale_states_are_preserved(self):
        primitives, rho, n = Primitives(), .8, 101
        result = solve_stochastic(primitives, rho, n, tol=1e-11)
        grid = result["grid"]
        expected = betaincc(
            1.0 + grid / (1.0 - rho),
            1.0 + (1.0 - grid) / (1.0 - rho), primitives.p0,
        )
        np.testing.assert_allclose(result["demand"], expected, rtol=0, atol=2e-14)
        # A state with no sales remains unchanged forever, rather than
        # acquiring or forgetting evidence during calendar periods without sales.
        grid = np.linspace(0.0, 1.0, 11)
        demand = lambda x: np.where(x < .4, 0.0, .7)
        fixed = evaluate_stochastic_policy(
            primitives, rho, grid, np.zeros(len(grid)), demand_function=demand,
        )
        np.testing.assert_allclose(fixed["value"][grid < .4], 0.0, rtol=0, atol=1e-13)
        self.assertGreater(float(fixed["value"][-1]), 1.0)

    def test_optimality_certificate_and_loss_against_enumerated_policies(self):
        primitives = replace(Primitives(), gamma=.91)
        rho, n = .9, 9
        result = solve_stochastic(primitives, rho, n, tol=1e-11)
        grid = result["grid"]
        # All 512 deterministic stationary policies are affordable here.
        values = []
        for actions in itertools.product((0.0, 1.0), repeat=n):
            matrix, reward = dense_calendar_system(primitives, rho, grid, actions)
            values.append(np.linalg.solve(matrix, reward))
        exact_optimum = np.max(values, axis=0)
        self.assertLessEqual(
            float(np.max(np.abs(result["value"] - exact_optimum))),
            result["value_error_bound"] + 2e-11,
        )
        self.assertAlmostEqual(
            result["value_error_bound"],
            result["residual"] / (1.0 - primitives.gamma), places=12,
        )
        self.assertLess(result["value_error_bound"], 1e-8)
        fluid = solve_fluid(primitives, rho, 201, tol=1e-11)
        candidate = transfer_policy(fluid, grid)
        evaluated = evaluate_stochastic_policy(primitives, rho, grid, candidate)
        loss = result["value"] - evaluated["value"]
        error_budget = result["value_error_bound"] + evaluated["value_error_bound"] + 2e-11
        self.assertGreaterEqual(float(loss.min()), -error_budget)
        self.assertGreater(float(loss.max()), 1e-4)

    def test_transfer_uses_gap_and_creates_no_artificial_mixed_band(self):
        fluid = {
            "grid": np.array([0.0, .5, 1.0]),
            "gap": np.array([-1.0, 1.0, -1.0]),
            "policy": np.array([0, 1, 0]),
        }
        grid = np.linspace(0.0, 1.0, 9)
        candidate = transfer_policy(fluid, grid, action_tolerance=0.0)
        np.testing.assert_array_equal(candidate, [0, 0, 0, 1, 1, 1, 0, 0, 0])
        self.assertTrue(set(candidate) <= {0, 1})
        # Strict comparison assigns both exact tolerance ties to product 1.
        np.testing.assert_array_equal(
            transfer_policy(fluid, grid, action_tolerance=.5),
            [0, 0, 0, 0, 1, 0, 0, 0, 0],
        )


if __name__ == "__main__":
    unittest.main()
