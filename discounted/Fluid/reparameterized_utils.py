"""Shared coordinate-transform helpers for discounted fluid outputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from fluid_model import FluidParams


@dataclass(frozen=True)
class InputSpec:
    p0: float
    grid_step: float
    path: Path

    @property
    def tag(self) -> str:
        p0_tag = f"{round(100 * self.p0):03d}"
        h_tag = f"{self.grid_step:g}".replace(".", "p")
        return f"p0_{p0_tag}_h_{h_tag}"


def triangular_array(
    frame: pd.DataFrame,
    column: str,
    grid_step: float,
    max_count: float,
) -> np.ndarray:
    steps = int(round(max_count / grid_step))
    array = np.full((steps + 1, steps + 1), np.nan)
    s = frame["s"].to_numpy(dtype=float)
    f = frame["f"].to_numpy(dtype=float)
    layer = np.rint((s + f) / grid_step).astype(int)
    success_index = np.rint(s / grid_step).astype(int)
    array[layer, success_index] = frame[column].to_numpy(dtype=float)
    return array


def transform_solution(
    frame: pd.DataFrame,
    spec: InputSpec,
    params: FluidParams,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    """Add (n, m) derivatives, switching margins, and HJB residuals."""
    result = frame.copy()
    h = spec.grid_step
    n = result["s"].to_numpy(dtype=float) + result["f"].to_numpy(dtype=float)
    m = (result["s"].to_numpy(dtype=float) + 1.0) / (n + 2.0)
    max_count = float(n.max())
    steps = int(round(max_count / h))
    layer = np.rint(n / h).astype(int)
    success_index = np.rint(
        result["s"].to_numpy(dtype=float) / h
    ).astype(int)
    valid = layer < steps

    values = triangular_array(result, "value", h, max_count)
    current = result["value"].to_numpy(dtype=float)
    next_failure = np.full(len(result), np.nan)
    next_success = np.full(len(result), np.nan)
    next_failure[valid] = values[
        layer[valid] + 1,
        success_index[valid],
    ]
    next_success[valid] = values[
        layer[valid] + 1,
        success_index[valid] + 1,
    ]

    v_s = (next_success - current) / h
    v_f = (next_failure - current) / h
    phi = (next_success - next_failure) / h
    w_n = m * v_s + (1.0 - m) * v_f
    w_m = (n + 2.0) * phi
    continuous_threshold = (
        (params.c2 - params.c1) / (params.p2 - params.p1)
    )
    continuous_margin = phi - continuous_threshold

    demand = result["demand"].to_numpy(dtype=float)
    r = params.discount_rate
    exponent = np.full(len(result), np.inf)
    positive_demand = demand > 0.0
    exponent[positive_demand] = r * h / demand[positive_demand]
    discount = np.exp(-exponent)
    reward_factor = np.zeros(len(result))
    reward_factor[positive_demand] = (
        demand[positive_demand]
        * (-np.expm1(-exponent[positive_demand]))
        / r
    )
    delta_value = next_success - next_failure
    discrete_action_gap = (
        -(params.c2 - params.c1) * reward_factor
        + discount * (params.p2 - params.p1) * delta_value
    )
    finite_step_threshold = np.full(len(result), np.inf)
    usable_discount = valid & (discount > 0.0)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        finite_step_threshold[usable_discount] = (
            continuous_threshold
            * reward_factor[usable_discount]
            / (discount[usable_discount] * h)
        )

    ham_1 = (
        params.revenue
        - params.c1
        + w_n
        + (params.p1 - m) * phi
    )
    ham_2 = (
        params.revenue
        - params.c2
        + w_n
        + (params.p2 - m) * phi
    )
    hjb_rhs = demand * np.maximum(ham_1, ham_2)
    hjb_residual = r * current - hjb_rhs

    source_policy = result["product"].to_numpy(dtype=np.int8)
    discrete_policy = np.where(discrete_action_gap > 0.0, 2, 1).astype(np.int8)
    continuous_policy = np.where(
        continuous_margin > 0.0,
        2,
        1,
    ).astype(np.int8)
    if not np.array_equal(source_policy[valid], discrete_policy[valid]):
        mismatch = int(
            np.sum(source_policy[valid] != discrete_policy[valid])
        )
        raise AssertionError(
            f"Discrete action-gap reconstruction failed at {mismatch} states."
        )

    result.insert(0, "n", n)
    result.insert(1, "m", m)
    result["v_s_forward"] = v_s
    result["v_f_forward"] = v_f
    result["w_n_forward"] = w_n
    result["w_m_forward"] = w_m
    result["phi"] = phi
    result["continuous_switch_threshold"] = continuous_threshold
    result["continuous_switch_margin"] = continuous_margin
    result["finite_step_switch_threshold"] = finite_step_threshold
    result["discrete_action_value_gap"] = discrete_action_gap
    result["continuous_limit_policy"] = continuous_policy
    result["hjb_rhs_forward"] = hjb_rhs
    result["hjb_residual_forward"] = hjb_residual

    finite_residual = hjb_residual[valid & np.isfinite(hjb_residual)]
    continuous_mismatch = (
        source_policy[valid] != continuous_policy[valid]
    )
    summary: dict[str, float | int] = {
        "p0": spec.p0,
        "grid_step": h,
        "max_count": max_count,
        "states": len(result),
        "derivative_states": int(valid.sum()),
        "continuous_threshold": continuous_threshold,
        "continuous_policy_mismatches": int(continuous_mismatch.sum()),
        "continuous_policy_mismatch_share": float(continuous_mismatch.mean()),
        "hjb_residual_max_abs": float(np.max(np.abs(finite_residual))),
        "hjb_residual_mean_abs": float(np.mean(np.abs(finite_residual))),
        "hjb_residual_rmse": float(
            np.sqrt(np.mean(np.square(finite_residual)))
        ),
        "hjb_residual_p95_abs": float(
            np.quantile(np.abs(finite_residual), 0.95)
        ),
    }
    return result, summary


def extract_boundaries(
    transformed: pd.DataFrame,
    spec: InputSpec,
) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for n, layer_frame in transformed.groupby("n", sort=True):
        ordered = layer_frame.sort_values("m")
        product2 = ordered["product"].eq(2).to_numpy()
        starts = product2 & ~np.r_[False, product2[:-1]]
        run_count = int(starts.sum())
        region = ordered.loc[product2]
        natural_lower = 1.0 / (float(n) + 2.0)
        natural_upper = (float(n) + 1.0) / (float(n) + 2.0)
        rows.append(
            {
                "p0": spec.p0,
                "grid_step": spec.grid_step,
                "n": float(n),
                "product2_present": int(len(region) > 0),
                "product2_runs": run_count,
                "m_lower": (
                    float(region["m"].min()) if len(region) else np.nan
                ),
                "m_upper": (
                    float(region["m"].max()) if len(region) else np.nan
                ),
                "m_width": (
                    float(region["m"].max() - region["m"].min())
                    if len(region)
                    else 0.0
                ),
                "natural_m_lower": natural_lower,
                "natural_m_upper": natural_upper,
                "touches_s_zero": (
                    int(np.isclose(region["s"].min(), 0.0))
                    if len(region)
                    else 0
                ),
                "touches_f_zero": (
                    int(np.isclose(region["f"].min(), 0.0))
                    if len(region)
                    else 0
                ),
                "product2_nodes": len(region),
            }
        )
    return pd.DataFrame(rows)


def regular_nm_field(
    transformed: pd.DataFrame,
    grid_step: float,
    column: str,
    n_step: float = 0.25,
    m_points: int = 501,
) -> tuple[np.ndarray, np.ndarray, np.ma.MaskedArray]:
    max_count = float(transformed["n"].max())
    n_grid = np.arange(0.0, max_count + n_step / 2.0, n_step)
    m_grid = np.linspace(0.0, 1.0, m_points)
    source = triangular_array(
        transformed,
        column,
        grid_step,
        max_count,
    )
    field = np.full((len(m_grid), len(n_grid)), np.nan)
    for column_index, n_value in enumerate(n_grid):
        layer = int(np.clip(np.rint(n_value / grid_step), 0, source.shape[0] - 1))
        snapped_n = layer * grid_step
        valid = (
            (m_grid >= 1.0 / (snapped_n + 2.0))
            & (m_grid <= (snapped_n + 1.0) / (snapped_n + 2.0))
        )
        success = (snapped_n + 2.0) * m_grid[valid] - 1.0
        success_index = np.rint(success / grid_step).astype(int)
        success_index = np.clip(success_index, 0, layer)
        field[valid, column_index] = source[layer, success_index]
    return n_grid, m_grid, np.ma.masked_invalid(field)
