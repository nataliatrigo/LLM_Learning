"""Independent continuous-state Monte Carlo check of the mature sale mix.

Each step is a sale, so the invariant distribution sampled here is the jump
chain's distribution. Product-2 frequency estimates P(product 2 | own sale),
not the calendar-time fraction of periods in which product 2 is sold.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

from forgetting.Code.model import Primitives, solve_boundary


def interval(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    estimate = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    z = NormalDist().inv_cdf(0.975)
    return {"estimate": estimate, "cluster_se": se,
            "ci_lower": estimate-z*se, "ci_upper": estimate+z*se}


def switching_points(coordinate: np.ndarray, advantage: np.ndarray,
                     tolerance: float) -> list[float]:
    shifted = advantage-tolerance
    crossing = np.flatnonzero((shifted[:-1] > 0) != (shifted[1:] > 0))
    return [float(coordinate[j]-shifted[j]*(coordinate[j+1]-coordinate[j])
                  / (shifted[j+1]-shifted[j])) for j in crossing]


def run_case(primitives: Primitives, rho: float, *, nodes: int,
             paths: int, burn_in: int, retained: int, seed: int,
             output: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    solved = solve_boundary(primitives, rho, central_nodes=nodes, outer_nodes=2,
                            half_widths=1e6, tolerance=1e-10)
    x = np.full(paths, .5)
    rng = np.random.default_rng(seed)
    counts = np.zeros(paths, dtype=np.int64)
    first_counts = counts.copy()
    second_counts = counts.copy()
    success_counts = counts.copy()
    x_sum = np.zeros(paths)
    retained_start = np.zeros(paths)
    half = retained//2
    action_tolerance = 1e-6
    for step in range(burn_in+retained):
        product2 = np.interp(x, solved.coordinate, solved.advantage) > action_tolerance
        successful = rng.random(paths) < np.where(product2, primitives.p2, primitives.p1)
        if step >= burn_in:
            offset = step-burn_in
            if offset == 0:
                retained_start = x.copy()
            counts += product2
            success_counts += successful
            x_sum += x
            if offset < half:
                first_counts += product2
            else:
                second_counts += product2
        x = rho*x + (1-rho)*successful
    conditional = counts/retained
    first = first_counts/half
    second = second_counts/(retained-half)
    mean_x = x_sum/retained
    expected_success = primitives.p1+primitives.delta_p*conditional
    realized_success = success_counts/retained
    identity_residual = mean_x-realized_success+(x-retained_start)/((1-rho)*retained)
    if np.max(np.abs(identity_residual)) > 1e-11:
        raise AssertionError("continuous-state updating identity failed")
    if not np.all((0 <= x) & (x <= 1)):
        raise AssertionError("reputation left [0,1]")
    frame = pd.DataFrame({"p0": primitives.p0, "rho": rho, "nodes": nodes,
        "path": np.arange(paths), "product2_count": counts,
        "retained_sales": retained, "product2_given_sale": conditional,
        "first_half_product2_given_sale": first,
        "second_half_product2_given_sale": second,
        "mean_pre_sale_x": mean_x, "success_rate": realized_success,
        "expected_success_rate": expected_success,
        "retained_start_x": retained_start, "final_x": x,
        "updating_identity_residual": identity_residual})
    half_check = interval(second-first)
    mean_identity_check = interval(mean_x-expected_success)
    result = {"p0": primitives.p0, "rho": rho, "nodes": nodes,
        "paths": paths, "burn_in_sales": burn_in, "retained_sales_per_path": retained,
        "product2_given_sale": interval(conditional),
        "first_half_product2_given_sale": interval(first),
        "second_half_product2_given_sale": interval(second),
        "second_minus_first_half": half_check,
        "half_window_difference_within_3se": abs(half_check["estimate"]) <= 3*half_check["cluster_se"],
        "mean_pre_sale_x": interval(mean_x),
        "p1_plus_delta_p_times_product2": float(expected_success.mean()),
        "mean_x_minus_expected_success": mean_identity_check,
        "mean_identity_within_3se": abs(mean_identity_check["estimate"]) <= 3*mean_identity_check["cluster_se"],
        "max_exact_update_identity_residual": float(np.max(np.abs(identity_residual))),
        "bellman_residual": solved.bellman_residual,
        "iterations": solved.iterations,
        "switching_points_action_tol": switching_points(solved.coordinate, solved.advantage, action_tolerance)}
    tag = f"p0_{str(primitives.p0).replace('.', 'p')}_n{nodes}"
    frame.to_csv(output/f"paths_{tag}.csv", index=False)
    np.savez_compressed(output/f"policy_{tag}.npz", coordinate=solved.coordinate,
                        demand=solved.demand, value=solved.value, advantage=solved.advantage)
    print(json.dumps(result), flush=True)
    return frame, result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paths", type=int, default=1000)
    parser.add_argument("--burn-in", type=int, default=1000)
    parser.add_argument("--retained", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    if args.paths < 2 or args.burn_in < 0 or args.retained < 2:
        parser.error("require >=2 paths, >=0 burn-in sales, >=2 retained sales")
    args.output.mkdir(parents=True, exist_ok=True)
    base = Primitives()
    p0_values = [.8, .81039, .85, .9]
    nodes_values = [2001, 4001]
    rho = .7
    manifest = {"parameters": base.as_dict(), "p0_values": p0_values,
        "rho": rho, "nodes": nodes_values, "paths": args.paths,
        "burn_in_sales": args.burn_in, "retained_sales_per_path": args.retained,
        "seed": args.seed, "initial_mature_x": .5,
        "solver_tolerance": 1e-10, "action_tolerance": 1e-6,
        "common_random_numbers": "Same random seed across p0 and policy grids.",
        "distribution": "Mature continuous-state sale-time jump chain; not calendar-time stationary mass.",
        "estimated_measure": "Stationary P(product 2 | own sale), subject to burn-in diagnostics.",
        "uncertainty": "Independent paths are clusters; 95% normal intervals. Equal retained sales per path.",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.output/"manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    results = []
    frames: dict[tuple[float, int], pd.DataFrame] = {}
    for nodes in nodes_values:
        for p0 in p0_values:
            frame, summary = run_case(replace(base, p0=p0), rho, nodes=nodes,
                paths=args.paths, burn_in=args.burn_in, retained=args.retained,
                seed=args.seed, output=args.output)
            frames[(p0, nodes)] = frame
            results.append(summary)
    paired = []
    for nodes in nodes_values:
        reference = frames[(.8, nodes)].product2_given_sale.to_numpy()
        for p0 in p0_values[1:]:
            difference = frames[(p0, nodes)].product2_given_sale.to_numpy()-reference
            paired.append({"nodes": nodes, "p0": p0, "reference_p0": .8,
                           "product2_difference": interval(difference)})
    refinement = []
    for p0 in p0_values:
        difference = (frames[(p0, 4001)].product2_given_sale.to_numpy()
                      - frames[(p0, 2001)].product2_given_sale.to_numpy())
        refinement.append({"p0": p0, "fine_nodes": 4001, "coarse_nodes": 2001,
                           "product2_difference": interval(difference)})
    output = {"cases": results, "paired_p0_differences": paired,
              "paired_grid_refinement": refinement}
    (args.output/"summary.json").write_text(json.dumps(output, indent=2)+"\n")
    pd.json_normalize(results, sep="_").to_csv(args.output/"summary.csv", index=False)
    pd.json_normalize(paired, sep="_").to_csv(args.output/"paired_p0_differences.csv", index=False)
    pd.json_normalize(refinement, sep="_").to_csv(args.output/"paired_grid_refinement.csv", index=False)


if __name__ == "__main__":
    main()
