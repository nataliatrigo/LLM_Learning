"""Export numerical evidence without conflating three approximation errors."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd

from model import Solution


def solution_summary(solution: Solution) -> dict:
    counts = {str(action): sum(int(np.count_nonzero(row == action)) for row in solution.actions)
              for action in (0, 1, 2)}
    return {
        "h": solution.h, "cutoff_n": solution.cutoff_n, "report_n": solution.report_n,
        "action_tolerance": solution.action_tolerance,
        "retained_states": sum(len(row) for row in solution.values),
        "value_at_origin_lower": float(solution.values[0][0]),
        "value_at_origin_upper": float(solution.upper_values[0][0]),
        "advantage_at_origin": float(solution.advantages[0][0]),
        "maximum_hjb_residual": solution.residual,
        "maximum_truncation_interval_width": solution.value_error_bound,
        "theoretical_truncation_bound": solution.theoretical_error_bound,
        "actions": counts,
        "action_codes": {"0": "tie or unresolved truncation uncertainty", "1": "product 1", "2": "product 2"},
    }


def compare_meshes(coarse: Solution, fine: Solution) -> dict:
    ratio = round(coarse.h / fine.h)
    if ratio < 1 or not np.isclose(ratio * fine.h, coarse.h) or coarse.report_n > fine.report_n:
        raise ValueError("Mesh comparison requires nested grids on a common retained interior")
    value_change = advantage_change = 0.0
    nominal_disagreements = certified_disagreements = count = 0
    largest_disagreement_advantage = 0.0
    for level, values in enumerate(coarse.values):
        other_value = fine.values[level * ratio][::ratio]
        a = coarse.advantages[level]
        b = fine.advantages[level * ratio][::ratio]
        value_change = max(value_change, float(np.max(np.abs(values - other_value))))
        advantage_change = max(advantage_change, float(np.max(np.abs(a - b))))
        disagreements = (a > coarse.action_tolerance) != (b > fine.action_tolerance)
        nominal_disagreements += int(disagreements.sum())
        ca, cb = coarse.actions[level], fine.actions[level * ratio][::ratio]
        certified_disagreements += int(np.count_nonzero(disagreements & (ca != 0) & (cb != 0)))
        if disagreements.any():
            largest_disagreement_advantage = max(
                largest_disagreement_advantage,
                float(np.maximum(np.abs(a), np.abs(b))[disagreements].max()))
        count += len(values)
    return {
        "coarse_h": coarse.h, "fine_h": fine.h, "common_states": count,
        "maximum_value_change": value_change, "maximum_advantage_change": advantage_change,
        "nominal_policy_disagreements": nominal_disagreements,
        "policy_disagreement_fraction": nominal_disagreements / count,
        "disagreements_with_both_truncation_certificates_strict": certified_disagreements,
        "maximum_advantage_magnitude_at_policy_disagreement": largest_disagreement_advantage,
        "interpretation": "Mesh differences are numerical evidence; truncation certificates do not certify mesh accuracy.",
    }


def save_solution(solution: Solution, path: Path):
    """Flat triangular arrays plus row offsets avoid padding the triangle."""
    offsets = np.r_[0, np.cumsum([len(row) for row in solution.values])]
    arrays = {name: np.concatenate(getattr(solution, name)) for name in (
        "values", "upper_values", "advantages", "advantage_lower", "advantage_upper", "actions", "demand_values")}
    metadata = {"parameters": asdict(solution.params), "delta": solution.params.delta,
                **solution_summary(solution)}
    np.savez_compressed(path, row_offsets=offsets, metadata=json.dumps(metadata), **arrays)


def load_solution(path: Path) -> Solution:
    from model import Parameters
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
        offsets = data["row_offsets"]
        fields = {}
        for name in ("values", "upper_values", "advantages", "advantage_lower", "advantage_upper", "actions", "demand_values"):
            # NpzFile reads/decompresses on each __getitem__; read once per
            # field, then split its in-memory vector into diagonal arrays.
            flat = data[name]
            fields[name] = [flat[a:b].copy() for a, b in zip(offsets[:-1], offsets[1:])]
    return Solution(params=Parameters(**meta["parameters"]), h=meta["h"], cutoff_n=meta["cutoff_n"],
                    report_n=meta["report_n"], **fields, residual=meta["maximum_hjb_residual"],
                    value_error_bound=meta["maximum_truncation_interval_width"],
                    theoretical_error_bound=meta["theoretical_truncation_bound"],
                    action_tolerance=meta.get("action_tolerance", 1e-8))


def switching_table(solution: Solution) -> pd.DataFrame:
    rows = []
    for level, advantage in enumerate(solution.advantages):
        n = level * solution.h
        mask = advantage > solution.action_tolerance
        starts = np.flatnonzero(mask & ~np.r_[False, mask[:-1]])
        ends = np.flatnonzero(mask & ~np.r_[mask[1:], False])
        if not len(starts):
            rows.append(dict(n=n, component=0, S_min=np.nan, S_max=np.nan,
                             x_min=np.nan, x_max=np.nan, product2_states=0))
        for index, (left, right) in enumerate(zip(starts, ends), 1):
            rows.append(dict(n=n, component=index, S_min=left * solution.h, S_max=right * solution.h,
                             x_min=(left * solution.h + 1) / (n + 2),
                             x_max=(right * solution.h + 1) / (n + 2),
                             product2_states=int(right - left + 1)))
    return pd.DataFrame(rows)


def trajectory_diagnostics(paths: pd.DataFrame, solution: Solution) -> pd.DataFrame:
    """Check mass identity and bound discounted omitted tail, without declaring
    a finite ODE integral equal to the infinite-horizon value function.
    """
    rows = []
    for (state_index, path_id, policy), group in paths.groupby(["state_index", "path_id", "policy"], sort=False):
        first, last = group.iloc[0], group.iloc[-1]
        t = float(last.t)
        tail_bound = solution.params.max_reward / solution.params.delta * np.exp(-solution.params.delta * t)
        initial_value = solution.interpolate_field(float(first.S), float(first.F), "values")
        terminal_value = solution.interpolate_field(float(last.S), float(last.F), "values")
        estimate = float(last.discounted_profit) + np.exp(-solution.params.delta * t) * terminal_value
        rows.append({"state_index": int(state_index), "path_id": path_id, "policy": policy, "terminal_t": t,
                     "terminal_S": float(last.S), "terminal_F": float(last.F),
                     "terminal_x": float(last.x), "terminal_demand": float(last.demand),
                     "finite_horizon_discounted_profit": float(last.discounted_profit),
                     "maximum_remaining_profit_bound": float(tail_bound),
                     "finite_profit_plus_optimal_grid_continuation": float(estimate),
                     "initial_optimal_grid_value": initial_value,
                     "continuation_identity_gap": float(estimate - initial_value),
                     "maximum_mass_identity_error": float(np.max(np.abs(group.n - group.S - group.F))),
                     "termination_reason": str(last.get("termination_reason", "horizon")),
                     "note": "Optimal-grid continuation is a numerical proxy, not a certified continuous-model tail bound or the fixed-policy infinite return."})
    return pd.DataFrame(rows)
