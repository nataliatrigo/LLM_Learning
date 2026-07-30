"""Diagnostics shared by discrete and continuous learning rules.

No smoothing is applied here.  Every reported endpoint, maximum, and slope is
computed from the raw demand or dynamic-program arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np

from .discrete_solver import DiscreteSolution


def contiguous_components(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    """Return inclusive endpoints of all true runs in a one-dimensional mask."""

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("component mask must be one-dimensional")
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return ()
    cuts = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, indices.size - 1]
    return tuple((int(indices[i]), int(indices[j])) for i, j in zip(starts, ends))


@dataclass(frozen=True)
class DiagonalPolicySummary:
    n: int
    active: bool
    product2_states: int
    components: tuple[tuple[int, int], ...]
    lower_s: int | None
    upper_s: int | None
    lower_posterior_mean: float | None
    upper_posterior_mean: float | None
    maximum_gap: float
    maximum_advantage: float
    action_tolerance: float


@dataclass(frozen=True)
class PolicyLocalization:
    diagonals: tuple[DiagonalPolicySummary, ...]
    last_active_diagonal: int | None
    distance_to_outer_boundary: int | None
    last_active_censored: bool

    @property
    def by_diagonal(self) -> dict[int, DiagonalPolicySummary]:
        return {row.n: row for row in self.diagonals}


def classify_product2_policy(solution: DiscreteSolution) -> PolicyLocalization:
    """Summarize raw product-2 regions on each reported diagonal."""

    rows: list[DiagonalPolicySummary] = []
    active_diagonals: list[int] = []
    for n in sorted(solution.layers):
        layer = solution.layers[n]
        action2 = np.asarray(layer.get("robust_action2", layer["action2"]), dtype=bool)
        components = contiguous_components(action2)
        indices = np.flatnonzero(action2)
        lower = int(indices[0]) if indices.size else None
        upper = int(indices[-1]) if indices.size else None
        if indices.size:
            active_diagonals.append(n)
        tolerance_values = np.asarray(layer.get("action_tolerance", [0.0]))
        rows.append(
            DiagonalPolicySummary(
                n=n,
                active=bool(indices.size),
                product2_states=int(indices.size),
                components=components,
                lower_s=lower,
                upper_s=upper,
                lower_posterior_mean=solution.model.posterior_mean(
                    (lower, n - lower)
                )
                if lower is not None
                else None,
                upper_posterior_mean=solution.model.posterior_mean(
                    (upper, n - upper)
                )
                if upper is not None
                else None,
                maximum_gap=float(np.max(layer["gap"])),
                maximum_advantage=float(np.max(layer["advantage"])),
                action_tolerance=float(np.max(tolerance_values)),
            )
        )

    last_active = max(active_diagonals) if active_diagonals else None
    boundary_distance = (
        solution.outer_diagonal - last_active if last_active is not None else None
    )
    return PolicyLocalization(
        tuple(rows),
        last_active,
        boundary_distance,
        last_active is not None and last_active == solution.report_diagonal,
    )

@dataclass(frozen=True)
class DiscreteGridComparison:
    coarse_outer_diagonal: int
    fine_outer_diagonal: int
    common_report_diagonal: int
    states_compared: int
    action_changes: int
    raw_action_changes: int
    action_change_fraction: float
    maximum_value_difference: float
    maximum_demand_difference: float
    maximum_gap_difference: float
    maximum_advantage_difference: float
    maximum_lower_endpoint_change: float
    maximum_upper_endpoint_change: float
    last_active_diagonal: int | None
    distance_to_outer_boundary: int | None

    @property
    def action_stable(self) -> bool:
        """Whether the robust policy classification is unchanged."""

        return self.action_changes == 0


def compare_discrete_solutions(
    coarse: DiscreteSolution,
    fine: DiscreteSolution,
    common_report_diagonal: int | None = None,
) -> DiscreteGridComparison:
    """Compare two outer truncations on an identical reported interior."""

    if common_report_diagonal is None:
        common_report_diagonal = min(
            coarse.report_diagonal, fine.report_diagonal
        )
    if common_report_diagonal < 0:
        raise ValueError("common_report_diagonal must be nonnegative")
    if common_report_diagonal > min(
        coarse.report_diagonal, fine.report_diagonal
    ):
        raise ValueError("requested common interior was not stored in both solutions")

    maxima = {"value": 0.0, "demand": 0.0, "gap": 0.0, "advantage": 0.0}
    changes = 0
    raw_changes = 0
    states = 0
    lower_endpoint_changes: list[float] = []
    upper_endpoint_changes: list[float] = []
    for n in range(common_report_diagonal + 1):
        coarse_layer = coarse.layers[n]
        fine_layer = fine.layers[n]
        for key in maxima:
            maxima[key] = max(
                maxima[key],
                float(np.max(np.abs(coarse_layer[key] - fine_layer[key]))),
            )
        coarse_robust = coarse_layer.get("robust_action2", coarse_layer["action2"])
        fine_robust = fine_layer.get("robust_action2", fine_layer["action2"])
        changes += int(np.count_nonzero(coarse_robust != fine_robust))
        raw_changes += int(
            np.count_nonzero(coarse_layer["action2"] != fine_layer["action2"])
        )
        coarse_indices = np.flatnonzero(coarse_robust)
        fine_indices = np.flatnonzero(fine_robust)
        if coarse_indices.size and fine_indices.size:
            lower_endpoint_changes.append(
                abs(float(coarse_indices[0]) - float(fine_indices[0]))
            )
            upper_endpoint_changes.append(
                abs(float(coarse_indices[-1]) - float(fine_indices[-1]))
            )
        states += n + 1

    active = [
        n
        for n in range(common_report_diagonal + 1)
        if np.any(
            fine.layers[n].get("robust_action2", fine.layers[n]["action2"])
        )
    ]
    last_active = max(active) if active else None
    distance = fine.outer_diagonal - last_active if last_active is not None else None
    return DiscreteGridComparison(
        coarse_outer_diagonal=coarse.outer_diagonal,
        fine_outer_diagonal=fine.outer_diagonal,
        common_report_diagonal=common_report_diagonal,
        states_compared=states,
        action_changes=changes,
        raw_action_changes=raw_changes,
        action_change_fraction=changes / states,
        maximum_value_difference=maxima["value"],
        maximum_demand_difference=maxima["demand"],
        maximum_gap_difference=maxima["gap"],
        maximum_advantage_difference=maxima["advantage"],
        maximum_lower_endpoint_change=max(lower_endpoint_changes, default=float("nan")),
        maximum_upper_endpoint_change=max(upper_endpoint_changes, default=float("nan")),
        last_active_diagonal=last_active,
        distance_to_outer_boundary=distance,
    )


@dataclass(frozen=True)
class DiscreteSensitivity:
    """Raw diagonal maxima of q(S,F)=D(S+1,F)-D(S,F+1).

    ``estimated_slope`` is OLS on ``log(g_n)`` against ``log(n)`` over the
    inclusive range ``fit_start <= n <= fit_end``, after omitting nonpositive
    or non-finite observations.  The default range is the last half of the
    requested positive diagonals, with at least n=10.
    """

    n: np.ndarray
    g_n: np.ndarray
    argmax_s: np.ndarray
    argmax_posterior_mean: np.ndarray
    estimated_slope: float
    fit_start: int
    fit_end: int
    fit_observations: int

    @property
    def max_demand_sensitivity(self) -> float:
        return float(np.max(self.g_n))


def compute_discrete_sensitivity(
    model: Any,
    max_n: int,
    *,
    min_n: int = 0,
    fit_start: int | None = None,
    fit_end: int | None = None,
) -> DiscreteSensitivity:
    """Compute sensitivity maxima and a documented large-n log-log slope."""

    if max_n < min_n or min_n < 0:
        raise ValueError("require 0 <= min_n <= max_n")
    diagonals = np.arange(min_n, max_n + 1, dtype=int)
    maxima = np.empty(diagonals.size, dtype=float)
    argmax = np.empty(diagonals.size, dtype=int)

    for row, n in enumerate(diagonals):
        successes = np.arange(n + 1, dtype=float)
        failures = n - successes
        if hasattr(model, "demand_array"):
            after_success = np.asarray(
                model.demand_array(successes + 1.0, failures), dtype=float
            )
            after_failure = np.asarray(
                model.demand_array(successes, failures + 1.0), dtype=float
            )
        else:
            after_success = np.fromiter(
                (
                    model.demand(model.success_state((int(s), int(n - s))))
                    for s in range(n + 1)
                ),
                dtype=float,
                count=n + 1,
            )
            after_failure = np.fromiter(
                (
                    model.demand(model.failure_state((int(s), int(n - s))))
                    for s in range(n + 1)
                ),
                dtype=float,
                count=n + 1,
            )
        q = after_success - after_failure
        location = int(np.argmax(q))
        maxima[row] = float(q[location])
        argmax[row] = location

    if hasattr(model, "posterior_mean_counts"):
        means = np.asarray(
            model.posterior_mean_counts(argmax, diagonals - argmax),
            dtype=float,
        )
    else:
        means = np.fromiter(
            (
                model.posterior_mean((int(successes), int(n - successes)))
                for successes, n in zip(argmax, diagonals, strict=True)
            ),
            dtype=float,
            count=diagonals.size,
        )
    if fit_end is None:
        fit_end = max_n
    if fit_start is None:
        fit_start = max(10, int(np.ceil(max_n / 2)))
    fit_mask = (
        (diagonals >= fit_start)
        & (diagonals <= fit_end)
        & np.isfinite(maxima)
        & (maxima > 0.0)
        & (diagonals > 0)
    )
    fit_count = int(np.count_nonzero(fit_mask))
    if fit_count >= 2:
        slope = float(
            np.polyfit(np.log(diagonals[fit_mask]), np.log(maxima[fit_mask]), 1)[0]
        )
    else:
        slope = float("nan")

    return DiscreteSensitivity(
        n=diagonals,
        g_n=maxima,
        argmax_s=argmax,
        argmax_posterior_mean=means,
        estimated_slope=slope,
        fit_start=fit_start,
        fit_end=fit_end,
        fit_observations=fit_count,
    )


@dataclass(frozen=True)
class ContinuousSensitivity:
    x: np.ndarray
    y: np.ndarray
    q: np.ndarray
    reachable_mask: np.ndarray
    maximum: float
    argmax_state: tuple[float, float]
    argmax_coordinates: dict[str, float]


def continuous_local_sensitivity(
    model: Any,
    x: Iterable[float],
    y: Iterable[float],
    reachable_mask: np.ndarray | None = None,
) -> ContinuousSensitivity:
    """Evaluate D(success(state))-D(failure(state)) on a normalized grid.

    The first array axis corresponds to ``x`` and the second to ``y``.  If no
    mask is supplied, the normalized reachable triangle x+y<=1 is used.
    """

    x_values = np.asarray(tuple(x), dtype=float)
    y_values = np.asarray(tuple(y), dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    xx, yy = np.meshgrid(x_values, y_values, indexing="ij")
    if reachable_mask is None:
        mask = xx + yy <= 1.0 + 1e-12
    else:
        mask = np.asarray(reachable_mask, dtype=bool)
        if mask.shape != xx.shape:
            raise ValueError(f"reachable_mask shape must be {xx.shape}")

    q = np.full(xx.shape, np.nan, dtype=float)
    try:
        reachable_x = xx[mask]
        reachable_y = yy[mask]
        # Normalized forgetting transitions are affine.  Using their public
        # rho parameter keeps an 801-grid diagnostic vectorized even when the
        # scalar transition methods intentionally validate individual states.
        if hasattr(model, "rho"):
            rho = float(model.rho)
            success_x = rho * reachable_x + (1.0 - rho)
            success_y = rho * reachable_y
            failure_x = rho * reachable_x
            failure_y = rho * reachable_y + (1.0 - rho)
        else:
            success_x, success_y = model.success_state((reachable_x, reachable_y))
            failure_x, failure_y = model.failure_state((reachable_x, reachable_y))
        q[mask] = np.asarray(
            model.demand_array(success_x, success_y)
            - model.demand_array(failure_x, failure_y),
            dtype=float,
        )
    except (TypeError, ValueError):
        for index in zip(*np.nonzero(mask)):
            state = (float(xx[index]), float(yy[index]))
            q[index] = float(
                model.demand(model.success_state(state))
                - model.demand(model.failure_state(state))
            )
    flat_index = int(np.nanargmax(q))
    index = np.unravel_index(flat_index, q.shape)
    state = (float(xx[index]), float(yy[index]))
    raw_coordinates = model.state_coordinates(state)
    if isinstance(raw_coordinates, Mapping):
        coordinates = {key: float(value) for key, value in raw_coordinates.items()}
    else:
        coordinates = {
            "x": float(raw_coordinates[0]),
            "y": float(raw_coordinates[1]),
        }
    if hasattr(model, "posterior_mean"):
        coordinates["posterior_mean"] = float(model.posterior_mean(state))
    if hasattr(model, "effective_concentration"):
        coordinates["effective_concentration"] = float(
            model.effective_concentration(state)
        )
    return ContinuousSensitivity(
        x=x_values,
        y=y_values,
        q=q,
        reachable_mask=mask,
        maximum=float(q[index]),
        argmax_state=state,
        argmax_coordinates=coordinates,
    )
